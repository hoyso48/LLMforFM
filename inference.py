#!/usr/bin/env python3
"""
Run inference with a fine-tuned Qwen3 model for DX7 patch generation.

This script:
- reads a caption CSV (id, caption, optionally name/wav_path),
- builds the same prompt style used for training (`prompt.py`),
- generates a response with optional "thinking" enabled/disabled via chat template,
- extracts the final DX7 patch JSON from the response,
- validates the patch against DX7 constraints,
- optionally renders WAVs,
- writes per-example results to a CSV.

The output CSV is designed to be compatible with the evaluation utilities in this repo
(e.g., `evaluate.py` expects at least `id` and `wav_path` when scoring audio).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Allow running this script from anywhere (not only from the repo root).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt import zeroshot_prompt, zeroshot_schema_prompt
from dx7.utils import parse_last_specs, validate_specs, render_from_specs


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


def _json_compact(obj: dict) -> str:
    # One-line JSON is easier to store and re-load from CSVs.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _canonicalize_specs(specs: dict, *, row_id: Any, keys_to_remove: list[str]) -> dict:
    if not isinstance(specs, dict):
        raise TypeError(f"Row {row_id}: specs must be a dict, got {type(specs)}")

    for k in keys_to_remove:
        specs.pop(k, None)

    missing = [k for k in KEY_ORDER if k not in specs]
    if missing:
        raise ValueError(f"Row {row_id}: predicted patch is missing required keys: {missing}")

    # Keep only schema keys to avoid leaking extra fields into rendering/eval.
    return {k: specs[k] for k in KEY_ORDER}


def _extract_think_text_from_string(text: str) -> str | None:
    """
    Extract the first <think>...</think> block from a model response, if present.
    """
    if not isinstance(text, str):
        return None
    start = text.find("<think>")
    if start < 0:
        return None
    end = text.find("</think>", start)
    if end < 0:
        return None
    inner = text[start + len("<think>") : end]
    inner = inner.strip("\n").strip()
    return inner


def _get_single_token_id(tokenizer: Any, token: str) -> int | None:
    """
    Best-effort helper to get the token id for a single special token like </think>.
    Returns None if it cannot be resolved to exactly one id.
    """
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int):
            unk = getattr(tokenizer, "unk_token_id", None)
            if unk is None or tid != unk:
                return tid
    except Exception:
        pass

    try:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
            return ids[0]
    except Exception:
        pass

    return None


def _split_qwen3_thinking(tokenizer: Any, gen_token_ids: list[int]) -> tuple[str | None, str, bool]:
    """
    Split a generated completion into:
    - thinking content (inside the <think>...</think> block),
    - final content (after </think>),
    using the official Qwen3 approach: locate the </think> token id and split on it.

    Important:
    - Qwen3 uses special tokens for <think> / </think>, so decoding with
      skip_special_tokens=True will typically *hide* the literal tags.
    - This function returns decoded strings without special tokens, matching Qwen3 docs.
    """
    if not gen_token_ids:
        return None, "", False

    end_think_id = _get_single_token_id(tokenizer, "</think>")
    if end_think_id is None:
        # Fallback: the official Qwen3 docs use 151668 for </think>.
        # If the tokenizer cannot resolve it, try the known id to avoid "no thinking" false alarms.
        end_think_id = 151668

    try:
        # index points to the position *after* the last </think> token.
        index = len(gen_token_ids) - gen_token_ids[::-1].index(int(end_think_id))
    except ValueError:
        # No </think> token present; treat everything as final content.
        final_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True).strip("\n")
        # As a secondary fallback, try literal <think> tags if they survived decoding.
        think_from_str = _extract_think_text_from_string(final_text)
        if think_from_str is not None:
            return think_from_str, final_text, True
        return None, final_text, False

    think_ids = gen_token_ids[:index]
    final_ids = gen_token_ids[index:]
    think_text = tokenizer.decode(think_ids, skip_special_tokens=True).strip("\n")
    final_text = tokenizer.decode(final_ids, skip_special_tokens=True).strip("\n")

    # If tags survived as plain text (e.g., due to manual training injection), strip them.
    think_text = think_text.replace("<think>", "").replace("</think>", "").strip()
    final_text = final_text.replace("<think>", "").replace("</think>", "").strip()

    return think_text, final_text, True


def _load_inputs(
    *,
    caption_csv_path: Path,
    data_csv_path: Path | None,
    id_column: str,
    caption_column: str,
    limit: int | None,
    seed: int,
    shuffle: bool,
    filter_data: bool,
) -> pd.DataFrame:
    caps = pd.read_csv(caption_csv_path)
    if id_column not in caps.columns:
        raise ValueError(f"caption_csv_path is missing required column '{id_column}': {caption_csv_path}")
    if caption_column not in caps.columns:
        raise ValueError(f"caption_csv_path is missing required column '{caption_column}': {caption_csv_path}")

    df = caps.copy()

    if data_csv_path is not None:
        data = pd.read_csv(data_csv_path, index_col=0)
        if id_column not in data.columns:
            raise ValueError(f"data_csv_path is missing required column '{id_column}': {data_csv_path}")

        df = pd.merge(data, df, on=id_column, how="inner", suffixes=("_data", ""))

        if filter_data:
            if "inaudible" in df.columns:
                df = df[df["inaudible"] == False].copy()
            if "name" in df.columns:
                df = df[~df["name"].astype(str).str.contains("NULL", na=False)].copy()

    if shuffle:
        df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        df = df.head(int(limit)).copy()

    return df


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str, *, max_len: int = 80) -> str:
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".", " ") else "_" for c in name)
    safe = safe.strip().replace(" ", "_")
    if not safe:
        safe = "sample"
    return safe[:max_len]


def _default_render_specs() -> dict:
    """
    A safe DX7 patch used to repair/fallback rendering.
    We keep it minimal and compatible with dx7_synth() requirements.
    """
    from dx7.pydx7 import get_modmatrix, get_outmatrix, DEFAULT_SPEC_PARAMS

    alg = int(DEFAULT_SPEC_PARAMS.get("algorithm", 31))
    return {
        "modmatrix": get_modmatrix(alg).tolist(),
        "outmatrix": get_outmatrix(alg).tolist(),
        "feedback": int(DEFAULT_SPEC_PARAMS.get("feedback", 0)),
        "fixed_freq": list(DEFAULT_SPEC_PARAMS.get("fixed_freq", [0] * 6)),
        "coarse": list(DEFAULT_SPEC_PARAMS.get("coarse", [1] * 6)),
        "fine": list(DEFAULT_SPEC_PARAMS.get("fine", [0] * 6)),
        "detune": list(DEFAULT_SPEC_PARAMS.get("detune", [0] * 6)),
        "transpose": int(DEFAULT_SPEC_PARAMS.get("transpose", 0)),
        "ol": list(DEFAULT_SPEC_PARAMS.get("ol", [99, 0, 0, 0, 0, 0])),
        "eg_rate": [list(r) for r in DEFAULT_SPEC_PARAMS.get("eg_rate", [[95] * 6] * 4)],
        "eg_level": [list(r) for r in DEFAULT_SPEC_PARAMS.get("eg_level", [[99] * 6] * 3 + [[0] * 6])],
        "sensitivity": list(DEFAULT_SPEC_PARAMS.get("sensitivity", [0] * 6)),
    }


def _repair_specs_for_render(specs: dict) -> tuple[dict, list[str]]:
    """
    Best-effort repair so the DX7 renderer doesn't crash.
    - Truncate/pad 1D arrays to length 6
    - Truncate/pad 2D EG arrays to shape (4,6)
    - Ensure modmatrix is (6,6) and outmatrix is (6,)
    Returns (repaired_specs, notes).
    """
    defaults = _default_render_specs()
    notes: list[str] = []

    def fix_list(name: str, value: Any, *, n: int, default: list, clamp01: bool = False) -> list:
        if not isinstance(value, list):
            notes.append(f"{name}: non-list -> default")
            return list(default)
        out = list(value[:n])
        if len(out) < n:
            out.extend(list(default[len(out) : n]))
            notes.append(f"{name}: padded {len(value)}->{n}")
        elif len(value) > n:
            notes.append(f"{name}: truncated {len(value)}->{n}")
        if clamp01:
            out = [0 if int(v) <= 0 else 1 for v in out]
        return out

    def fix_matrix(name: str, value: Any, *, rows: int, cols: int, default: list[list]) -> list[list]:
        if not isinstance(value, list) or not value or not all(isinstance(r, list) for r in value):
            notes.append(f"{name}: non-2d-list -> default")
            return [list(r) for r in default]
        out_rows: list[list] = []
        for r in range(rows):
            if r < len(value) and isinstance(value[r], list):
                row = list(value[r][:cols])
                if len(row) < cols:
                    row.extend(list(default[r][len(row) : cols]))
                out_rows.append(row)
            else:
                out_rows.append(list(default[r]))
        if len(value) != rows or any((r < len(value) and isinstance(value[r], list) and len(value[r]) != cols) for r in range(min(len(value), rows))):
            notes.append(f"{name}: reshaped to ({rows},{cols})")
        return out_rows

    repaired = dict(specs) if isinstance(specs, dict) else {}

    repaired["modmatrix"] = fix_matrix("modmatrix", repaired.get("modmatrix"), rows=6, cols=6, default=defaults["modmatrix"])
    repaired["outmatrix"] = fix_list("outmatrix", repaired.get("outmatrix"), n=6, default=defaults["outmatrix"], clamp01=True)
    repaired["fixed_freq"] = fix_list("fixed_freq", repaired.get("fixed_freq"), n=6, default=defaults["fixed_freq"], clamp01=True)
    repaired["coarse"] = fix_list("coarse", repaired.get("coarse"), n=6, default=defaults["coarse"])
    repaired["fine"] = fix_list("fine", repaired.get("fine"), n=6, default=defaults["fine"])
    repaired["detune"] = fix_list("detune", repaired.get("detune"), n=6, default=defaults["detune"])
    repaired["ol"] = fix_list("ol", repaired.get("ol"), n=6, default=defaults["ol"])
    repaired["sensitivity"] = fix_list("sensitivity", repaired.get("sensitivity"), n=6, default=defaults["sensitivity"])
    repaired["eg_rate"] = fix_matrix("eg_rate", repaired.get("eg_rate"), rows=4, cols=6, default=defaults["eg_rate"])
    repaired["eg_level"] = fix_matrix("eg_level", repaired.get("eg_level"), rows=4, cols=6, default=defaults["eg_level"])

    # Scalars
    try:
        repaired["feedback"] = int(repaired.get("feedback", defaults["feedback"]))
    except Exception:
        notes.append("feedback: non-int -> default")
        repaired["feedback"] = int(defaults["feedback"])

    try:
        repaired["transpose"] = int(repaired.get("transpose", defaults["transpose"]))
    except Exception:
        notes.append("transpose: non-int -> default")
        repaired["transpose"] = int(defaults["transpose"])

    return repaired, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DX7 patches from captions with a fine-tuned Qwen3 model.")

    # Inputs
    parser.add_argument("--caption_csv_path", type=str, default="data/DX7_YAMAHA_test_captions.csv")
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default=None,
        help="Optional. If provided, merges dataset CSV (for filtering / metadata) with the caption CSV on id.",
    )
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--filter_data", action="store_true", help="Apply basic dataset filters (inaudible/NULL name) if data_csv_path is provided.")

    # Prompt
    parser.add_argument("--use_schema_prompt", action="store_true")
    parser.add_argument(
        "--keys_to_remove",
        type=str,
        default='["name","has_fixed_freq"]',
        help="JSON list of patch keys to drop from the extracted patch before validating/rendering.",
    )
    parser.add_argument(
        "--think_tag",
        type=str,
        default="auto",
        choices=["none", "auto", "think", "no_think"],
        help=(
            "Optionally append a Qwen3 soft-switch tag to the *caption* before formatting the prompt. "
            "'think' appends '/think', 'no_think' appends '/no_think'. "
            "'auto' appends '/think' when --enable_thinking is set, else appends '/no_think'. "
            "This matches the intended hybrid-thinking control behavior described in Qwen3 docs/blogs."
        ),
    )

    # Model / generation
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-8B", help="Base model id for fallback loading (if needed).")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--precision", type=str, default="fp8", choices=["fp8", "bf16"])
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g. cuda, cpu). Default: auto.")

    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to the tokenizer chat template.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="If set, stream generated text to stdout during generation (useful for debugging).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--do_sample", action="store_true", help="Use sampling; if unset, uses greedy decoding.")

    # Outputs
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--save_every", type=int, default=0, help="If >0, periodically saves intermediate CSV every N rows.")
    parser.add_argument("--print_every", type=int, default=0, help="If >0, prints a short sample output every N rows.")

    # Optional WAV rendering
    parser.add_argument("--render_wav", action="store_true")
    parser.add_argument("--wav_dir", type=str, default=None, help="Directory to write WAVs. Required if --render_wav is set.")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--v", type=int, default=100)
    parser.add_argument("--out_scale", type=float, default=1.0)
    parser.add_argument(
        "--render_fallback_default",
        action="store_true",
        help="If rendering the predicted patch fails, render a default patch instead (still records render_error).",
    )

    args = parser.parse_args()

    keys_to_remove = json.loads(args.keys_to_remove)
    if not isinstance(keys_to_remove, list) or not all(isinstance(x, str) for x in keys_to_remove):
        raise ValueError("--keys_to_remove must be a JSON list of strings")

    caption_csv_path = PROJECT_ROOT / args.caption_csv_path
    data_csv_path = (PROJECT_ROOT / args.data_csv_path) if isinstance(args.data_csv_path, str) and args.data_csv_path else None
    output_csv_path = Path(args.output_csv_path)

    if args.render_wav:
        if not isinstance(args.wav_dir, str) or not args.wav_dir.strip():
            raise ValueError("--wav_dir is required when --render_wav is set")
        wav_dir = Path(args.wav_dir)
        wav_dir.mkdir(parents=True, exist_ok=True)
    else:
        wav_dir = None

    df = _load_inputs(
        caption_csv_path=caption_csv_path,
        data_csv_path=data_csv_path,
        id_column=str(args.id_column),
        caption_column=str(args.caption_column),
        limit=args.limit,
        seed=int(args.seed),
        shuffle=bool(args.shuffle),
        filter_data=bool(args.filter_data),
    )

    # Lazy heavy imports so `--help` and data loading stay snappy.
    import torch
    from unsloth import FastLanguageModel

    device = args.device
    if not isinstance(device, str) or not device.strip():
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = str(args.model_path)

    # Prefer loading the fine-tuned model directory directly.
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=int(args.max_seq_length),
            load_in_4bit=False,
            load_in_8bit=(args.precision == "fp8"),
            full_finetuning=True,
        )
    except Exception as exc:
        # Fallback: load base model, then load safetensors from model_path.
        from safetensors import safe_open

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.base_model),
            max_seq_length=int(args.max_seq_length),
            load_in_4bit=False,
            load_in_8bit=(args.precision == "fp8"),
            full_finetuning=True,
        )

        all_safetensors = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        if not all_safetensors:
            raise RuntimeError(
                f"Failed to load model from '{model_path}' directly ({exc}). "
                f"Fallback loading also failed because no .safetensors files were found in the directory."
            ) from exc

        state_dict = {}
        for filename in all_safetensors:
            filepath = os.path.join(model_path, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    prompt_template = zeroshot_schema_prompt if args.use_schema_prompt else zeroshot_prompt

    out_rows: list[dict[str, Any]] = []

    def _save_partial() -> None:
        if not out_rows:
            return
        _ensure_parent_dir(output_csv_path)
        pd.DataFrame(out_rows).to_csv(output_csv_path, index=False)

    # Keep tqdm on stderr so stdout can be used for piping/streaming.
    pbar = tqdm(df.iterrows(), total=len(df), desc="Inference", dynamic_ncols=True, file=sys.stderr)
    n_parse_ok = 0
    n_valid_ok = 0
    for _, row in pbar:
        sample_id = row.get(args.id_column)
        caption = row.get(args.caption_column)

        if not _is_non_empty_str(caption):
            out_rows.append(
                {
                    "id": sample_id,
                    "caption": caption,
                    "prompt": None,
                    "response_text": None,
                    "think_text": None,
                    "final_text": None,
                    "think_found": False,
                    "patch_data": None,
                    "parse_ok": False,
                    "parse_error": "missing caption",
                    "validation_ok": False,
                    "wav_path": None,
                }
            )
            continue

        caption_for_prompt = str(caption)
        suffix = None
        if args.think_tag == "think":
            suffix = "/think"
        elif args.think_tag == "no_think":
            suffix = "/no_think"
        elif args.think_tag == "auto":
            suffix = "/think" if bool(args.enable_thinking) else "/no_think"

        if suffix is not None:
            caption_for_prompt = caption_for_prompt.rstrip() + " " + suffix

        user_text = prompt_template.format(prompt=caption_for_prompt)
        messages = [{"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(args.enable_thinking),
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = int(inputs["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(args.max_new_tokens),
        }
        if args.do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        streamer = None
        if args.stream:
            # Lazy import to keep non-streaming runs lightweight.
            from transformers import TextStreamer

            # Print only the generated continuation.
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            print(f"\n\n--- id={sample_id} ---\n", flush=True)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs, streamer=streamer)

        # Decode only the generated continuation (exclude prompt).
        gen_only = output_ids[0][input_len:]
        gen_only_ids = gen_only.tolist()
        response_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        think_text, final_text, think_found = _split_qwen3_thinking(tokenizer, gen_only_ids)

        parse_ok = False
        parse_error = None
        validation_ok = False
        validation_error = None
        patch_data_json = None
        wav_path = row.get("wav_path") if "wav_path" in row else None

        try:
            # Prefer parsing from the post-thinking content if available.
            parse_text = final_text if _is_non_empty_str(final_text) else response_text
            raw_specs = parse_last_specs(parse_text)
            cleaned_specs = _canonicalize_specs(raw_specs, row_id=sample_id, keys_to_remove=keys_to_remove)
            patch_data_json = _json_compact(cleaned_specs)
            parse_ok = True
        except Exception as exc:
            parse_error = str(exc)

        if parse_ok:
            try:
                # Keep validation silent by default to avoid spam; store only the boolean.
                validation_ok = bool(validate_specs(cleaned_specs, syx_file="", patch_number=-1, verbose=False))
            except Exception as exc:
                validation_ok = False
                validation_error = str(exc)

        # Optional audio rendering
        rendered_wav_rel = None
        render_ok = False
        render_error = None
        render_used = "none"
        patch_data_rendered = None

        if args.render_wav:
            # If the caption CSV already has a wav_path, preserve directory structure under wav_dir.
            if isinstance(wav_path, str) and wav_path.strip():
                rendered_wav_rel = wav_path
            else:
                name = row.get("name") if "name" in row else None
                stem = _sanitize_filename(str(name) if _is_non_empty_str(name) else f"id_{sample_id}")
                rendered_wav_rel = f"{stem}.wav"

            full_wav_path = wav_dir / str(rendered_wav_rel)
            full_wav_path.parent.mkdir(parents=True, exist_ok=True)

            # Import here so running without --render_wav does not require scipy.
            from scipy.io.wavfile import write

            def _try_render(specs_to_render: dict, *, label: str) -> bool:
                nonlocal render_error, render_used, patch_data_rendered
                audio = render_from_specs(
                    specs_to_render,
                    sr=int(args.sr),
                    n=int(args.n),
                    v=int(args.v),
                    out_scale=float(args.out_scale),
                )
                write(str(full_wav_path), int(args.sr), audio)
                render_used = label
                patch_data_rendered = _json_compact(specs_to_render)
                return True

            try:
                if parse_ok and validation_ok:
                    # Try rendering the predicted patch as-is first.
                    render_ok = _try_render(cleaned_specs, label="predicted")
                elif parse_ok:
                    # Invalid patch: try a repaired version for rendering (truncate/pad to expected shapes).
                    repaired, notes = _repair_specs_for_render(cleaned_specs)
                    render_ok = _try_render(repaired, label="repaired")
                    if notes:
                        render_error = f"repaired_for_render: {', '.join(notes)}"
                else:
                    render_ok = False
            except Exception as exc:
                render_ok = False
                render_error = str(exc)

                # Retry with repaired specs if we have a parsed patch.
                if parse_ok:
                    try:
                        repaired, notes = _repair_specs_for_render(cleaned_specs)
                        render_ok = _try_render(repaired, label="repaired_after_error")
                        note_s = ", ".join(notes) if notes else "no_changes"
                        render_error = f"{render_error} | repaired_after_error: {note_s}"
                    except Exception as exc2:
                        render_ok = False
                        render_error = f"{render_error} | repaired_failed: {exc2}"

                # Optional final fallback: default patch
                if (not render_ok) and bool(args.render_fallback_default):
                    try:
                        default_specs = _default_render_specs()
                        render_ok = _try_render(default_specs, label="default_fallback")
                        render_error = (render_error + " | " if render_error else "") + "used default fallback"
                    except Exception as exc3:
                        render_ok = False
                        render_error = f"{render_error} | default_fallback_failed: {exc3}"

        out_rows.append(
            {
                "id": sample_id,
                "caption": caption,
                "prompt": user_text,
                "enable_thinking": bool(args.enable_thinking),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "do_sample": bool(args.do_sample),
                "max_new_tokens": int(args.max_new_tokens),
                "think_tag": str(args.think_tag),
                "response_text": response_text,
                "think_text": think_text,
                "final_text": final_text,
                "think_found": bool(think_found),
                "patch_data": patch_data_json,
                "patch_data_rendered": patch_data_rendered,
                "parse_ok": bool(parse_ok),
                "parse_error": parse_error,
                "validation_ok": bool(validation_ok),
                "validation_error": validation_error,
                "render_ok": bool(render_ok),
                "render_error": render_error,
                "render_used": render_used,
                "wav_path": rendered_wav_rel if args.render_wav else wav_path,
            }
        )

        if parse_ok:
            n_parse_ok += 1
        if validation_ok:
            n_valid_ok += 1
        pbar.set_postfix(
            parse_ok=f"{n_parse_ok}/{len(out_rows)}",
            valid_ok=f"{n_valid_ok}/{len(out_rows)}",
        )

        if args.print_every and int(args.print_every) > 0 and ((len(out_rows) % int(args.print_every)) == 0):
            head = response_text[:400].replace("\n", "\\n")
            print(f"[{len(out_rows)}] id={sample_id} parse_ok={parse_ok} validation_ok={validation_ok} head='{head}'")

        if args.save_every and int(args.save_every) > 0 and ((len(out_rows) % int(args.save_every)) == 0):
            _save_partial()

    _save_partial()
    print(f"Saved {len(out_rows)} rows to {output_csv_path}")


if __name__ == "__main__":
    main()


