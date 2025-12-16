#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B for DX7 patch generation (caption -> patch JSON),
optionally mixing in intermediate reasoning stored in the CSV `cot` column.

Key behavior
- If `cot` is present and non-empty, we wrap it between Qwen-style thinking tokens:
    <think>
    ...
    </think>
  and then append the final DX7 patch JSON (wrapped as a fenced ```json code block).
- If `cot` is empty / missing, we output only the final DX7 patch JSON code block (non-reasoning).

Notes
- This script mirrors the older training logic in `GCT634_final/fine_tuning.py`, but:
  - uses JSON (not Python `specs = {...}` formatting),
  - supports `cot` -> `<think>...</think>` injection,
  - keeps the actual training call commented out by default (per request).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from datasets import Dataset

from prompt import zeroshot_prompt, zeroshot_schema_prompt


# Canonical DX7 JSON key order (matches `LLMforFM/prompt.py` schema).
KEY_ORDER: list[str] = [
    "modmatrix",
    "outmatrix",
    "feedback",
    "fixed_freq",
    "coarse",
    "fine",
    "detune",
    "transpose",
    "ol",
    "eg_rate",
    "eg_level",
    "sensitivity",
]


def _is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _parse_patch_data(patch_data: Any, *, row_id: Any) -> dict:
    """
    Parse `patch_data` which may be:
    - a dict,
    - a JSON object string,
    - (legacy) a Python literal dict string.
    """
    if isinstance(patch_data, dict):
        return patch_data
    if patch_data is None:
        raise ValueError(f"Row {row_id}: patch_data is missing")
    if not isinstance(patch_data, str):
        raise TypeError(f"Row {row_id}: patch_data has unsupported type {type(patch_data)}")

    candidate = patch_data.strip()
    if candidate == "":
        raise ValueError(f"Row {row_id}: patch_data is empty")

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Row {row_id}: patch_data JSON must be an object, got {type(parsed)}")
    except json.JSONDecodeError:
        # Legacy: Python literal dict (single quotes, True/False)
        try:
            parsed = ast.literal_eval(candidate)
        except Exception as exc:
            raise ValueError(f"Row {row_id}: failed to parse patch_data as JSON or Python literal: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Row {row_id}: patch_data literal must be a dict, got {type(parsed)}")
        return parsed


def _canonicalize_specs(specs: dict, *, row_id: Any, keys_to_remove: list[str]) -> dict:
    """
    Remove non-tool fields and emit a canonical dict with stable key order.
    """
    if not isinstance(specs, dict):
        raise TypeError(f"Row {row_id}: specs must be dict, got {type(specs)}")

    for k in keys_to_remove:
        specs.pop(k, None)

    missing = [k for k in KEY_ORDER if k not in specs]
    if missing:
        raise ValueError(f"Row {row_id}: patch_data is missing required keys: {missing}")

    # Keep only the schema keys (avoid leaking extra fields into training targets).
    return {k: specs[k] for k in KEY_ORDER}


def _is_2d_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(row, list) for row in value)


def _format_2d_list_rows(value: list[list], *, indent: str) -> list[str]:
    """
    Format a list-of-lists as:
      [
        [...],
        [...],
      ]
    keeping each row compact on a single line.
    """
    lines: list[str] = ["["]
    for i, row in enumerate(value):
        row_str = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        comma = "," if i < len(value) - 1 else ""
        lines.append(f"{indent}{row_str}{comma}")
    lines.append("]")
    return lines


def _format_json_compact_top_level(obj: dict, *, key_order: list[str]) -> str:
    """
    Format a JSON object so that:
    - top-level keys appear one per line (stable ordering),
    - nested matrices (list-of-lists) are rendered with one row per line,
    - other nested structures stay compact.

    This matches the readability requirement while keeping sequence length bounded.
    """
    if not isinstance(obj, dict):
        raise TypeError(f"obj must be a dict, got {type(obj)}")

    missing = [k for k in key_order if k not in obj]
    if missing:
        raise ValueError(f"Object is missing required keys: {missing}")

    lines: list[str] = ["{"]
    for i, k in enumerate(key_order):
        is_last = i == len(key_order) - 1
        v = obj[k]

        block: list[str]
        if _is_2d_list(v):
            matrix_lines = _format_2d_list_rows(v, indent="    ")
            block = [f'  "{k}": {matrix_lines[0]}'] + [f"  {ln}" for ln in matrix_lines[1:]]
        else:
            v_str = json.dumps(v, ensure_ascii=False, separators=(",", ": "))
            block = [f'  "{k}": {v_str}']

        if not is_last:
            block[-1] = block[-1] + ","

        lines.extend(block)
    lines.append("}")
    return "\n".join(lines)


def _wrap_json_fence(json_text: str) -> str:
    return f"```json\n{json_text}\n```"


def _describe_lengths(
    name: str,
    lengths: list[int],
    *,
    top_frac: float,
    max_seq_length: int | None,
) -> str:
    if not lengths:
        return f"- {name}: (no samples)\n"
    if not (0.0 < float(top_frac) < 1.0):
        raise ValueError(f"top_frac must be in (0,1), got {top_frac}")

    arr = np.asarray(lengths, dtype=np.int64)
    n = int(arr.size)

    def q(p: float) -> int:
        return int(np.quantile(arr, p, method="linear"))

    p50 = q(0.50)
    p90 = q(0.90)
    p95 = q(0.95)
    p99 = q(0.99)
    mx = int(arr.max())
    mean = float(arr.mean())

    # Top-k% tail summary
    cutoff_p = 1.0 - float(top_frac)
    cutoff = int(np.quantile(arr, cutoff_p, method="linear"))
    tail = arr[arr >= cutoff]
    tail_n = int(tail.size)
    tail_min = int(tail.min()) if tail_n else cutoff
    tail_mean = float(tail.mean()) if tail_n else float("nan")
    tail_max = int(tail.max()) if tail_n else mx

    over_s = ""
    if max_seq_length is not None and max_seq_length > 0:
        over = int((arr > int(max_seq_length)).sum())
        over_pct = 100.0 * over / n
        over_s = f", >{max_seq_length}: {over} ({over_pct:.2f}%)"

    return (
        f"- {name}: n={n}, mean={mean:.1f}, p50={p50}, p90={p90}, p95={p95}, p99={p99}, max={mx}{over_s}\n"
        f"  - top {int(top_frac*100)}% (>=~p{int(cutoff_p*100)} cutoff={cutoff}): n={tail_n}, range=[{tail_min}, {tail_max}], mean={tail_mean:.1f}\n"
    )


def _compute_token_lengths_by_group(
    convs: list[list[dict[str, str]]],
    *,
    tokenizer_name_or_path: str,
    batch_size: int,
    enable_thinking: bool,
    max_seq_length: int | None,
) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    reasoning: list[list[dict[str, str]]] = []
    non_reasoning: list[list[dict[str, str]]] = []
    for conv in convs:
        assistant = conv[1]["content"]
        if isinstance(assistant, str) and assistant.lstrip().startswith("<think>"):
            reasoning.append(conv)
        else:
            non_reasoning.append(conv)

    def lengths_for(group_convs: list[list[dict[str, str]]]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(group_convs), batch_size):
            chunk = group_convs[i : i + batch_size]
            texts = tok.apply_chat_template(
                chunk,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=bool(enable_thinking),
            )
            enc = tok(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            out.extend(len(ids) for ids in enc["input_ids"])
        return out

    # Note: Qwen3 chat templates typically include an (empty) <think></think> wrapper even for
    # non-reasoning assistant messages; reasoning messages carry non-empty content inside <think>.
    reason_lens = lengths_for(reasoning)
    non_lens = lengths_for(non_reasoning)
    all_lens = reason_lens + non_lens

    # "Top 5%" by default (user request)
    top_frac = 0.05

    print("\nToken length stats (train, after apply_chat_template):")
    print(f"- tokenizer: {tokenizer_name_or_path}")
    print(f"- enable_thinking passed to template: {bool(enable_thinking)}")
    if max_seq_length is not None:
        print(f"- reference max_seq_length: {max_seq_length}")

    print(_describe_lengths("ALL", all_lens, top_frac=top_frac, max_seq_length=max_seq_length), end="")
    print(_describe_lengths("REASONING (cot used)", reason_lens, top_frac=top_frac, max_seq_length=max_seq_length), end="")
    print(_describe_lengths("NON-REASONING (no cot)", non_lens, top_frac=top_frac, max_seq_length=max_seq_length), end="")


def _is_reasoning_cot(cot_text: Any) -> bool:
    return isinstance(cot_text, str) and cot_text.strip() != ""


def _build_assistant_target(*, cot_text: Any, patch_block: str) -> str:
    """
    Always keep ALL reasoning data:
    - If cot is non-empty: emit <think>cot</think> + final patch block.
    - Else: emit only the final patch block (no explicit think content).
    """
    if _is_reasoning_cot(cot_text):
        cot = str(cot_text).strip()
        return f"<think>\n{cot}\n</think>\n{patch_block}"
    return patch_block


def _downsample_non_reasoning_for_target_ratio(
    train_df: pd.DataFrame,
    *,
    reasoning_prob: float | None,
    seed: int,
) -> pd.DataFrame:
    """
    Interpret `reasoning_prob` as the desired fraction of reasoning examples in the FINAL training set:

      p = (#reasoning) / (#reasoning + #non_reasoning_kept)

    Constraints (as requested):
    - All reasoning rows (cot non-empty) are ALWAYS kept.
    - Only non-reasoning rows may be downsampled.
    - If `reasoning_prob` is None: keep all data (natural ratio).
    - You cannot request a ratio lower than the natural ratio, since we never drop reasoning rows.
    """
    if reasoning_prob is None:
        return train_df

    p = float(reasoning_prob)
    if not (0.0 < p <= 1.0):
        raise ValueError(f"--reasoning_prob must be in (0,1], got {reasoning_prob}")

    if "cot" not in train_df.columns:
        # No reasoning info available; nothing to downsample towards p.
        return train_df

    is_reasoning = train_df["cot"].apply(_is_reasoning_cot)
    r_df = train_df[is_reasoning].copy()
    n_df = train_df[~is_reasoning].copy()

    r = int(len(r_df))
    n = int(len(n_df))
    total = r + n
    if total == 0:
        return train_df
    if r == 0:
        raise ValueError("No reasoning (non-empty cot) rows found, but --reasoning_prob was provided.")

    natural = r / total
    if p < natural:
        raise ValueError(
            f"Requested --reasoning_prob={p:.4f} is lower than the natural ratio {natural:.4f} "
            f"(reasoning={r}, non_reasoning={n}). We never drop reasoning rows, so this is impossible."
        )

    # Desired non-reasoning count to achieve ratio p:
    #   p = r / (r + n_keep)  =>  n_keep = r*(1-p)/p
    if p >= 1.0:
        n_keep = 0
    else:
        n_keep = int(math.floor(r * (1.0 - p) / p))

    if n_keep >= n:
        # Already at or above the requested ratio; keep all data.
        return train_df

    n_keep_df = n_df.sample(n=n_keep, random_state=int(seed)) if n_keep > 0 else n_df.head(0)
    return pd.concat([r_df, n_keep_df], ignore_index=True)


def _load_csv(path: Path, *, index_col: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, index_col=index_col)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Data inputs (defaults match the current LLMforFM layout).
    parser.add_argument("--yamaha_train_csv", type=str, default="data/DX7_YAMAHA_train.csv")
    parser.add_argument("--alltheweb_train_csv", type=str, default="data/DX7_AllTheWeb_train.csv")
    parser.add_argument("--yamaha_test_csv", type=str, default="data/DX7_YAMAHA_test.csv")
    parser.add_argument("--yamaha_train_captions_csv", type=str, default="data/DX7_YAMAHA_train_captions.csv")
    parser.add_argument("--alltheweb_train_captions_csv", type=str, default="data/DX7_AllTheWeb_train_captions.csv")
    parser.add_argument("--yamaha_test_captions_csv", type=str, default="data/DX7_YAMAHA_test_captions.csv")

    # Prompt / output formatting.
    parser.add_argument("--use_schema_prompt", action="store_true", help="Use the longer JSON schema prompt template.")
    parser.add_argument(
        "--keys_to_remove",
        type=str,
        default='["name","has_fixed_freq"]',
        help="JSON list of patch keys to drop from patch_data before serializing to the final JSON.",
    )
    parser.add_argument(
        "--append_think_tags",
        action="store_true",
        help=(
            "Append a Qwen3-style soft switch tag to the *caption* inside the user prompt: "
            "rows with non-empty `cot` get '/think', rows without get '/no_think'. "
            "This provides an explicit controllable token you can mirror at inference time. "
            "Note: this is independent of the tokenizer template's enable_thinking flag."
        ),
    )

    # Reasoning mixing.
    parser.add_argument(
        "--reasoning_prob",
        type=float,
        default=None,
        help=(
            "Target fraction of reasoning examples in the FINAL training set. "
            "All reasoning rows (non-empty `cot`) are always kept; only non-reasoning rows are downsampled. "
            "If omitted, uses ALL data (natural ratio)."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)

    # Debug output.
    parser.add_argument("--print_examples", action="store_true", help="Print a few formatted examples and exit (default behavior).")
    parser.add_argument("--num_examples", type=int, default=3, help="How many examples to print from train set.")
    parser.add_argument(
        "--analyze_lengths",
        action="store_true",
        help="Compute token-length distribution for reasoning vs non-reasoning (train set), including a top-5% tail summary.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="unsloth/Qwen3-8B",
        help="Tokenizer identifier for length analysis (HF repo id or local path).",
    )
    parser.add_argument("--length_batch_size", type=int, default=64, help="Batch size for token-length analysis.")
    parser.add_argument(
        "--template_enable_thinking",
        action="store_true",
        help="Pass enable_thinking=True to the tokenizer chat template during length analysis (default: False).",
    )

    # Training (kept for parity; training call is commented out by default).
    parser.add_argument("--load_model", action="store_true", help="Load model + build Trainer (does NOT start training unless you uncomment).")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-8B")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-8B-fp8-cot-mix")
    parser.add_argument("--precision", type=str, default="fp8", choices=["fp8", "bf16"])
    parser.add_argument("--responses_only", action="store_true")
    parser.add_argument("--filter_train", action="store_true")
    parser.add_argument("--filter_test", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # W&B logging (optional; mirrors the previous script's behavior but avoids hardcoding secrets).
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help="Enable experiment logging. Use 'wandb' to match previous fine_tuning.py style.",
    )
    parser.add_argument("--wandb_project", type=str, default="LLMforFM")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="Optional. If provided, sets WANDB_API_KEY and calls wandb.login(key=...). "
        "Otherwise wandb will use existing env/credentials.",
    )

    args = parser.parse_args()

    keys_to_remove = json.loads(args.keys_to_remove)
    if not isinstance(keys_to_remove, list) or not all(isinstance(x, str) for x in keys_to_remove):
        raise ValueError("--keys_to_remove must be a JSON list of strings")

    project_root = Path(__file__).resolve().parent
    yamaha_train_csv = project_root / args.yamaha_train_csv
    alltheweb_train_csv = project_root / args.alltheweb_train_csv
    yamaha_test_csv = project_root / args.yamaha_test_csv
    yamaha_train_caps_csv = project_root / args.yamaha_train_captions_csv
    alltheweb_train_caps_csv = project_root / args.alltheweb_train_captions_csv
    yamaha_test_caps_csv = project_root / args.yamaha_test_captions_csv

    train_caption = _load_csv(yamaha_train_caps_csv)
    train_caption_add = _load_csv(alltheweb_train_caps_csv)
    test_caption = _load_csv(yamaha_test_caps_csv)
    train_data = _load_csv(yamaha_train_csv)
    train_data_add = _load_csv(alltheweb_train_csv)
    test_data = _load_csv(yamaha_test_csv)

    train_df = pd.merge(train_data, train_caption[["id", "caption"]], on="id", how="left")
    train_df_add = pd.merge(train_data_add, train_caption_add[["id", "caption"]], on="id", how="left")
    train_df = pd.concat([train_df, train_df_add], ignore_index=True)
    test_df = pd.merge(test_data, test_caption[["id", "caption"]], on="id", how="left")

    # Same filtering logic as the previous script (with a bit more NaN safety).
    train_filter = (train_df["inaudible"] == False) & (~train_df["name"].astype(str).str.contains("NULL", na=False))
    test_filter = (test_df["inaudible"] == False) & (~test_df["name"].astype(str).str.contains("NULL", na=False))
    if args.filter_train:
        train_df = train_df[train_filter].copy()
    if args.filter_test:
        test_df = test_df[test_filter].copy()

    if "cot" not in train_df.columns:
        # Training can still proceed, but all examples will be non-reasoning.
        train_df["cot"] = None

    # Apply dataset-level reasoning ratio (keeps ALL reasoning rows; downsamples non-reasoning only).
    train_df = _downsample_non_reasoning_for_target_ratio(
        train_df,
        reasoning_prob=args.reasoning_prob,
        seed=int(args.seed),
    )

    prompt_template = zeroshot_schema_prompt if args.use_schema_prompt else zeroshot_prompt

    # Keep only what we need.
    train_keep = ["id", "caption", "patch_data", "cot"]
    test_keep = ["id", "caption", "patch_data"]
    train_dataset = Dataset.from_pandas(train_df[train_keep])
    test_dataset = Dataset.from_pandas(test_df[test_keep])

    def preprocess(examples: dict) -> dict:
        conversations: list[list[dict[str, str]]] = []
        ids = examples["id"]
        captions = examples["caption"]
        patches = examples["patch_data"]
        cots = examples.get("cot", [None] * len(ids))

        for sample_id, caption, patch_data, cot_text in zip(ids, captions, patches, cots):
            if not _is_non_empty_str(caption):
                raise ValueError(f"Row {sample_id}: missing caption")

            caption_for_prompt = str(caption)
            if args.append_think_tags:
                caption_for_prompt = caption_for_prompt.rstrip() + (" /think" if _is_reasoning_cot(cot_text) else " /no_think")

            user_text = prompt_template.format(prompt=caption_for_prompt)
            specs_raw = _parse_patch_data(patch_data, row_id=sample_id)
            specs = _canonicalize_specs(specs_raw, row_id=sample_id, keys_to_remove=keys_to_remove)

            # Readable JSON (top-level key per line) wrapped as a fenced code block.
            json_patch_pretty = _format_json_compact_top_level(specs, key_order=KEY_ORDER)
            patch_block = _wrap_json_fence(json_patch_pretty)
            assistant_text = _build_assistant_target(
                cot_text=cot_text,
                patch_block=patch_block,
            )

            conversations.append(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )

        return {"conversations": conversations}

    # NOTE:
    # `datasets.Dataset.__getitem__` returns a `datasets.arrow_dataset.Column`, which is list-like but
    # NOT a real Python `list`. Some Transformers versions fail to detect "batched conversations"
    # for non-list sequences and will interpret the outer list as a single conversation, causing:
    #   jinja2.exceptions.UndefinedError: 'list object' has no attribute 'content'
    # To make `tokenizer.apply_chat_template` reliably batch-render, force a concrete `list`.
    train_conv = list(train_dataset.map(preprocess, batched=True)["conversations"])
    test_conv = list(test_dataset.map(preprocess, batched=True)["conversations"])

    if args.analyze_lengths:
        if args.length_batch_size <= 0:
            raise ValueError("--length_batch_size must be > 0")
        _compute_token_lengths_by_group(
            train_conv,
            tokenizer_name_or_path=str(args.tokenizer_name_or_path),
            batch_size=int(args.length_batch_size),
            enable_thinking=bool(args.template_enable_thinking),
            max_seq_length=int(args.max_seq_length) if args.max_seq_length else None,
        )
        return

    # -------------------------------
    # Optional: target reasoning ratio
    # -------------------------------
    # Example: to target ~75% reasoning examples overall (keep all reasoning; downsample non-reasoning):
    #
    #   python fine_tuning.py --reasoning_prob 0.75

    if args.print_examples or not args.load_model:
        # Basic stats for sanity.
        cot_nonempty = int(train_df["cot"].apply(_is_reasoning_cot).sum())
        total = int(len(train_df))
        non = int(total - cot_nonempty)
        frac = (cot_nonempty / total) if total else 0.0
        print(f"Train rows: {total} (reasoning={cot_nonempty}, non_reasoning={non}, frac={frac:.4f})")
        print(f"Test rows:  {len(test_df)}")
        print(f"reasoning_prob (target overall ratio): {args.reasoning_prob}")

        def _print_with_head_tail(label: str, *, user_text: str, assistant_text: str) -> None:
            """
            Print a compact view that confirms:
            - <think> ... </think> wrapping (if present)
            - final JSON patch is appended after thinking (or is the whole output)
            without dumping huge CoTs to the terminal.
            """
            head_u = user_text[:500]
            head_a = assistant_text[:800]
            tail_a = assistant_text[-800:] if len(assistant_text) > 800 else assistant_text

            print(f"\n--- {label} ---")
            print("USER (head):\n", head_u, sep="")
            print("\nASSISTANT (head):\n", head_a, sep="")
            if tail_a != head_a:
                print("\nASSISTANT (tail):\n", tail_a, sep="")

        # Print a few examples: try to show at least one with <think> if available.
        shown = 0
        want = max(1, int(args.num_examples))

        # First pass: prefer examples containing <think>
        for conv in train_conv:
            if shown >= want:
                break
            assistant = conv[1]["content"]
            if "<think>" in assistant and "</think>" in assistant:
                _print_with_head_tail(
                    "TRAIN EXAMPLE (with <think>)",
                    user_text=conv[0]["content"],
                    assistant_text=assistant,
                )
                shown += 1

        # Second pass: fill remaining with non-reasoning examples.
        for conv in train_conv:
            if shown >= want:
                break
            assistant = conv[1]["content"]
            if "<think>" not in assistant:
                _print_with_head_tail(
                    "TRAIN EXAMPLE (no <think>)",
                    user_text=conv[0]["content"],
                    assistant_text=assistant,
                )
                shown += 1

        if not args.load_model:
            return

    # -------------------------------
    # Model + Trainer setup (parity)
    # -------------------------------
    # Heavy imports live down here so `--print_examples` can run without a full training stack.
    if args.report_to == "wandb":
        # Mirror previous script's pattern via env vars, but do NOT hardcode secrets.
        os.environ["WANDB_PROJECT"] = str(args.wandb_project)
        if isinstance(args.wandb_entity, str) and args.wandb_entity.strip():
            os.environ["WANDB_ENTITY"] = args.wandb_entity.strip()
        if isinstance(args.wandb_api_key, str) and args.wandb_api_key.strip():
            os.environ["WANDB_API_KEY"] = args.wandb_api_key.strip()
        import wandb

        # If key is missing, wandb will try env/default credentials.
        wandb.login(key=os.environ.get("WANDB_API_KEY") or None)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer, SFTConfig
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=int(args.max_seq_length),
        load_in_4bit=False,
        load_in_8bit=args.precision == "fp8",
        full_finetuning=True,
    )

    # IMPORTANT: Explicitly disable any template-driven "auto thinking" injection.
    # We control thinking strictly via the CSV `cot` column.
    train_texts = tokenizer.apply_chat_template(train_conv, tokenize=False, enable_thinking=False)
    test_texts = tokenizer.apply_chat_template(test_conv, tokenize=False, enable_thinking=False)

    train_text_ds = Dataset.from_pandas(pd.DataFrame({"text": pd.Series(train_texts, name="text")})).shuffle(seed=int(args.seed))
    test_text_ds = Dataset.from_pandas(pd.DataFrame({"text": pd.Series(test_texts, name="text")}))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text_ds,
        eval_dataset=test_text_ds,
        args=SFTConfig(
            eval_strategy="steps",
            eval_steps=100,
            dataset_text_field="text",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            num_train_epochs=1,
            learning_rate=1e-4,
            logging_steps=1,
            optim="adamw_8bit" if args.precision == "fp8" else "adamw_torch_fused",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=int(args.seed),
            report_to=("wandb" if args.report_to == "wandb" else "none"),
            run_name=(args.wandb_run_name if isinstance(args.wandb_run_name, str) and args.wandb_run_name.strip() else None),
            bf16=args.precision == "bf16",
        ),
    )

    if args.responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    # -------------------------------
    # Training start (intentionally commented out)
    # -------------------------------
    trainer.train()
    trainer.save_model(args.model_path)


if __name__ == "__main__":
    main()


