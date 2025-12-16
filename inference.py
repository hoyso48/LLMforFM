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


def _extract_think_text(text: str) -> str | None:
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

    # Model / generation
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-8B", help="Base model id for fallback loading (if needed).")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--precision", type=str, default="fp8", choices=["fp8", "bf16"])
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g. cuda, cpu). Default: auto.")

    parser.add_argument("--enable_thinking", action="store_true", help="Pass enable_thinking=True to the tokenizer chat template.")
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

    for i, row in df.iterrows():
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
                    "patch_data": None,
                    "parse_ok": False,
                    "parse_error": "missing caption",
                    "validation_ok": False,
                    "wav_path": None,
                }
            )
            continue

        user_text = prompt_template.format(prompt=str(caption))
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

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the generated continuation (exclude prompt).
        gen_only = output_ids[0][input_len:]
        response_text = tokenizer.decode(gen_only, skip_special_tokens=True)

        think_text = _extract_think_text(response_text)

        parse_ok = False
        parse_error = None
        validation_ok = False
        validation_error = None
        patch_data_json = None
        wav_path = row.get("wav_path") if "wav_path" in row else None

        try:
            raw_specs = parse_last_specs(response_text)
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
        if args.render_wav and parse_ok and validation_ok:
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

            audio = render_from_specs(
                cleaned_specs,
                sr=int(args.sr),
                n=int(args.n),
                v=int(args.v),
                out_scale=float(args.out_scale),
            )
            write(str(full_wav_path), int(args.sr), audio)

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
                "response_text": response_text,
                "think_text": think_text,
                "patch_data": patch_data_json,
                "parse_ok": bool(parse_ok),
                "parse_error": parse_error,
                "validation_ok": bool(validation_ok),
                "validation_error": validation_error,
                "wav_path": rendered_wav_rel if args.render_wav else wav_path,
            }
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


