import argparse
import ast
import asyncio
import base64
import io
import importlib.util
import json
import mimetypes
import os
import sys
import wave
from pathlib import Path

import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from google import genaif


KEY_ORDER = [
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


def _load_llmforfm_prompts():
    """
    Load prompt templates from LLMforFM/prompt.py without import-name collisions
    with GCT634_final/prompt.py.
    """
    projects_root = Path(__file__).resolve().parents[1]
    prompt_path = projects_root / "LLMforFM" / "prompt.py"
    if not prompt_path.exists():
        raise FileNotFoundError(f"LLMforFM prompt.py not found: {prompt_path}")

    spec = importlib.util.spec_from_file_location("llmforfm_prompt", prompt_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    return {
        "off": mod.cot_generation_prompt,
        "on": getattr(mod, "cot_generation_prompt_opk_on"),
    }


def _load_concat_renderer():
    """
    Load `LLMforFM/scripts/render_concat_ablation_audio.py` as a module so we can render
    concatenated ablation audio on-the-fly (without pre-saving WAVs for every sample).
    """
    projects_root = Path(__file__).resolve().parents[1]
    renderer_path = projects_root / "LLMforFM" / "scripts" / "render_concat_ablation_audio.py"
    if not renderer_path.exists():
        raise FileNotFoundError(f"Concat renderer not found: {renderer_path}")

    spec = importlib.util.spec_from_file_location("llmforfm_concat_renderer", renderer_path)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses can break if the module isn't in sys.modules during exec_module
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _get_api_key() -> str:
    raise RuntimeError("Internal error: use _resolve_api_key(api_key_arg=...) instead.")


def _resolve_api_key(*, api_key_arg: str | None) -> str:
    """
    Resolve Gemini API key.
    Priority:
    1) --api_key argument
    2) env GEMINI_API_KEY
    3) env GOOGLE_API_KEY
    """
    if isinstance(api_key_arg, str) and api_key_arg.strip():
        return api_key_arg.strip()

    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise ValueError(
        "No Gemini API key found. Set environment variable GEMINI_API_KEY (recommended) "
        "or GOOGLE_API_KEY, or pass --api_key."
    )


def _guess_mime_type(audio_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(audio_path))
    if mime_type:
        return mime_type
    ext = audio_path.suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".aac":
        return "audio/aac"
    raise ValueError(f"Could not determine MIME type for: {audio_path}")


def _encode_audio_file_inline_part(audio_path: Path) -> dict:
    mime_type = _guess_mime_type(audio_path)
    audio_bytes = audio_path.read_bytes()
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return {"inline_data": {"data": encoded, "mime_type": mime_type}}


def _encode_wav_int16_inline_part(audio, *, sr: int) -> dict:
    """
    Encode a mono int16 numpy array as WAV bytes and return a Gemini inline_data part.
    """
    import numpy as np

    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be a numpy array, got {type(audio)}")
    if audio.dtype != np.int16:
        raise TypeError(f"audio must be int16, got dtype={audio.dtype}")
    if audio.ndim != 1:
        raise ValueError(f"audio must be mono (1D), got shape={audio.shape}")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sr))
        wf.writeframes(audio.tobytes())
    wav_bytes = buf.getvalue()

    encoded = base64.b64encode(wav_bytes).decode("utf-8")
    return {"inline_data": {"data": encoded, "mime_type": "audio/wav"}}


def _parse_patch_data_value(patch_data_value, *, row_id=None) -> dict:
    """
    Parse patch_data which may be:
    - a dict,
    - a JSON string,
    - a Python literal dict string.
    """
    if isinstance(patch_data_value, dict):
        return patch_data_value
    if patch_data_value is None:
        raise ValueError(f"Row {row_id}: missing patch_data")
    if not isinstance(patch_data_value, str):
        raise TypeError(f"Row {row_id}: unsupported patch_data type {type(patch_data_value)}")

    candidate = patch_data_value.strip()
    if candidate == "":
        raise ValueError(f"Row {row_id}: empty patch_data string")

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    parsed = ast.literal_eval(candidate)
    if not isinstance(parsed, dict):
        raise ValueError(f"Row {row_id}: patch_data literal must be a dict, got {type(parsed)}")
    return parsed


def _extract_dx7_specs_json_from_patch_data(patch_data) -> dict:
    """
    Convert patch_data into the canonical JSON object used by the CoT teacher prompt.
    """
    specs = _parse_patch_data_value(patch_data)

    missing = [k for k in KEY_ORDER if k not in specs]
    if missing:
        raise ValueError(f"patch_data is missing required keys for CoT prompt: {missing}")

    out = {k: specs[k] for k in KEY_ORDER}
    # Stable order (helps readability; model shouldn't rely on it).
    out = {k: out[k] for k in KEY_ORDER}
    return out


def _is_2d_list(value) -> bool:
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
    - nested arrays/objects are rendered compactly (minimal newlines).

    This reduces prompt length drastically compared to indent=2 on deeply nested arrays.
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

        # Keep formatting compact: do not add extra blank lines between top-level entries.

    lines.append("}")
    return "\n".join(lines)


def _slim_op_ablation_list_for_prompt(op_ablation_list: list[dict], *, row_id=None) -> list[dict]:
    """
    Reduce concat metadata to only what the teacher prompt needs:
    - label (e.g., OP2_OFF / OP3_ON)
    - start_sec / end_sec
    """
    if not isinstance(op_ablation_list, list):
        raise TypeError(f"Row {row_id}: op_ablation_list must be a list, got {type(op_ablation_list)}")

    slim: list[dict] = []
    for idx, item in enumerate(op_ablation_list):
        if not isinstance(item, dict):
            raise TypeError(f"Row {row_id}: op_ablation_list[{idx}] must be a dict, got {type(item)}")

        label = item.get("label")
        start_sec = item.get("start_sec")
        end_sec = item.get("end_sec")

        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Row {row_id}: op_ablation_list[{idx}] missing non-empty 'label'")
        if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)):
            raise ValueError(f"Row {row_id}: op_ablation_list[{idx}] missing numeric start_sec/end_sec")

        slim.append(
            {
                "label": label,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
        )
    return slim


def _format_op_ablation_list_for_prompt(op_ablation_list_slim: list[dict]) -> str:
    """
    Format a list of small dicts as JSON with one item per line (readable but compact).
    """
    if not isinstance(op_ablation_list_slim, list):
        raise TypeError(f"op_ablation_list_slim must be a list, got {type(op_ablation_list_slim)}")

    lines: list[str] = ["["]
    for i, item in enumerate(op_ablation_list_slim):
        if not isinstance(item, dict):
            raise TypeError(f"op_ablation_list_slim[{i}] must be a dict, got {type(item)}")
        s = json.dumps(item, ensure_ascii=False, separators=(",", ": "))
        comma = "," if i < len(op_ablation_list_slim) - 1 else ""
        lines.append(f"  {s}{comma}")
    lines.append("]")
    return "\n".join(lines)


def _build_prompt_text(
    *,
    template: str,
    prompt_text: str,
    dx7_specs_json_obj: dict,
    op_ablation_list_obj: list[dict],
) -> str:
    dx7_json_str = _format_json_compact_top_level(dx7_specs_json_obj, key_order=KEY_ORDER)
    op_list_str = _format_op_ablation_list_for_prompt(op_ablation_list_obj)
    return template.format(
        PROMPT_TEXT=prompt_text,
        DX7_SPECS_JSON=dx7_json_str,
        OP_ABLATION_LIST=op_list_str,
    )


def _load_and_merge_captions(data_df: pd.DataFrame, caption_csv_path: Path | None) -> pd.DataFrame:
    if "caption" in data_df.columns:
        return data_df
    if caption_csv_path is None:
        raise ValueError("caption column missing in data CSV, so --caption_csv_path is required.")
    cap = pd.read_csv(caption_csv_path)
    if "id" not in cap.columns or "caption" not in cap.columns:
        raise ValueError("Caption CSV must contain columns: id, caption")
    cap = cap[["id", "caption"]].copy()
    return pd.merge(data_df, cap, on="id", how="left")


def _parse_cot_response(text: str) -> dict:
    """
    Best-effort parsing:
    - Extract cot text inside \\cot{...}
    - Extract trailing JSON object as raw string
    """
    marker = "\\cot{"
    start = text.find(marker)
    if start == -1:
        return {"cot": None, "final_json_str": None}

    i = start + len(marker)
    depth = 1
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return {"cot": None, "final_json_str": None}

    cot_text = text[start + len(marker) : i - 1].strip()
    tail = text[i:].strip()
    # Find the first '{' in the tail and treat it as the start of the final JSON.
    brace = tail.find("{")
    final_json_str = tail[brace:].strip() if brace != -1 else None
    return {"cot": cot_text, "final_json_str": final_json_str}


def _load_existing_done_keys(jsonl_path: Path) -> set[tuple[str, str]]:
    """
    Load already-generated items from an existing JSONL file.

    Key choice: (ablation_variant, wav_path)
    - This allows keeping separate runs per ablation_variant in the same file if desired.
    """
    success: set[tuple[str, str]] = set()
    error_keys: set[tuple[str, str]] = set()
    if not jsonl_path.exists():
        return set()

    bad = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                bad += 1
                print(f"[resume] WARNING: could not parse JSON on line {line_no} in {jsonl_path}: {exc}")
                continue
            wav_path = obj.get("wav_path")
            variant = obj.get("ablation_variant") or obj.get("variant")
            error = obj.get("error")
            if isinstance(wav_path, str) and isinstance(variant, str):
                key = (variant, wav_path)
                if isinstance(error, str) and error.strip():
                    error_keys.add(key)
                else:
                    success.add(key)
    if bad:
        print(f"[resume] NOTE: ignored {bad} malformed lines in {jsonl_path}")

    if error_keys:
        error_only = error_keys - success
        print(
            f"[resume] NOTE: found {len(error_keys)} key(s) with existing 'error' fields. "
            f"{len(error_only)} key(s) have only errors and will be retried."
        )
    return success


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
async def _generate_one(client, *, model: str, prompt_text: str, audio_part: dict) -> str:
    contents = [prompt_text, audio_part]
    resp = await client.aio.models.generate_content(model=model, contents=contents)
    return resp.text


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CoT reasoning traces with Gemini (audio + metadata). "
        "By default, audio is rendered on-the-fly (no bulk WAV storage)."
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument(
        "--api_key",
        type=str,
        help="Gemini API key. Overrides GEMINI_API_KEY/GOOGLE_API_KEY env vars. (Recommended: set env var.)",
    )
    parser.add_argument("--data_csv_path", type=str, default="data/DX7_YAMAHA_test.csv")
    parser.add_argument("--caption_csv_path", type=str, default="data/DX7_YAMAHA_test_captions.csv")
    parser.add_argument("--out_jsonl_path", type=str, default="outputs/cot_reasoning_gemini.jsonl")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="0-based start row index (inclusive) in the input CSV. Default: 0.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="0-based end row index (exclusive) in the input CSV. -1 means end. Default: -1.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="How many rows to process concurrently per batch (asyncio.gather).",
    )
    parser.add_argument(
        "--batch_sleep_seconds",
        type=float,
        default=0.0,
        help="Optional sleep between batches to reduce rate-limit pressure.",
    )

    # Ablation / isolation variant used for the concatenated audio segments.
    parser.add_argument("--ablation_variant", choices=["auto", "off", "on"], default="off")
    parser.add_argument("--ablate_mode", choices=["active", "carriers", "modulators", "all"], default="active")
    parser.add_argument("--segment_seconds", type=float, default=10.0)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--note", type=int, default=60)
    parser.add_argument("--velocity", type=int, default=100)
    parser.add_argument("--out_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=str, default="0")
    parser.add_argument("--max_ablations", type=int, help="Optional cap on number of OP segments per sample.")

    # Optional disk mode: use pre-rendered concatenated WAV+meta.
    parser.add_argument("--concat_wav_dir", type=str)
    parser.add_argument("--concat_meta_dir", type=str)

    # Save only a few examples (for debugging/inspection), not the entire dataset.
    parser.add_argument("--save_examples_dir", type=str)
    parser.add_argument("--save_examples_limit", type=int, default=0)
    parser.add_argument(
        "--overwrite_examples",
        action="store_true",
        help="If set, overwrite example wav/meta/prompt/response files if they already exist.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, append to --out_jsonl_path and skip rows already present in that JSONL (by ablation_variant + wav_path).",
    )

    parser.add_argument("--dry_run", action="store_true", help="Do not call Gemini; still builds prompts/meta.")
    args = parser.parse_args()

    if args.save_examples_limit < 0:
        raise ValueError("--save_examples_limit must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.batch_sleep_seconds < 0:
        raise ValueError("--batch_sleep_seconds must be >= 0")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")
    if args.end_index != -1 and args.end_index < 0:
        raise ValueError("--end_index must be -1 or >= 0")
    if args.end_index != -1 and args.end_index <= args.start_index:
        raise ValueError("--end_index must be > --start_index (or -1 for end)")

    prompts = _load_llmforfm_prompts()

    use_disk_mode = bool(args.concat_wav_dir) and bool(args.concat_meta_dir)
    if (args.concat_wav_dir is None) != (args.concat_meta_dir is None):
        raise ValueError("Provide both --concat_wav_dir and --concat_meta_dir, or neither (on-the-fly mode).")

    concat_renderer = None if use_disk_mode else _load_concat_renderer()

    save_dir = Path(args.save_examples_dir) if args.save_examples_dir else None
    saved = 0
    saved_lock = asyncio.Lock()

    data_df = pd.read_csv(args.data_csv_path)
    original_len = len(data_df)
    end_idx = None if args.end_index == -1 else int(args.end_index)
    data_df = data_df.iloc[int(args.start_index) : end_idx].copy()
    if args.limit:
        data_df = data_df.head(args.limit)
    data_df = _load_and_merge_captions(data_df, Path(args.caption_csv_path) if args.caption_csv_path else None)
    print(
        f"Loaded {original_len} row(s) from {args.data_csv_path}. "
        f"Using slice [{args.start_index}:{args.end_index}] -> {len(data_df)} row(s) before resume-skip."
    )

    out_path = Path(args.out_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str]] = set()
    if args.resume:
        done_keys = _load_existing_done_keys(out_path)
        if done_keys:
            print(
                f"[resume] Found {len(done_keys)} existing successful key(s) in {out_path}. "
                f"Will skip them and append new ones."
            )

    async def _maybe_save_example(
        *,
        wav_rel: Path,
        variant: str,
        meta: dict,
        filled_prompt: str,
        response_text: str | None,
        audio_part: dict | None,
        source_concat_wav_path: Path | None,
    ) -> None:
        nonlocal saved
        if save_dir is None or args.save_examples_limit <= 0:
            return

        # Lock so we don't exceed the global examples limit under concurrency.
        async with saved_lock:
            if saved >= args.save_examples_limit:
                return

        save_wav_path = save_dir / "wav" / variant / wav_rel
        save_meta_path = (save_dir / "meta" / variant / wav_rel).with_suffix(".json")
        save_prompt_path = (save_dir / "prompt" / variant / wav_rel).with_suffix(".txt")
        save_resp_path = (save_dir / "response" / variant / wav_rel).with_suffix(".txt")

        save_wav_path.parent.mkdir(parents=True, exist_ok=True)
        save_meta_path.parent.mkdir(parents=True, exist_ok=True)
        save_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        save_resp_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.overwrite_examples and (save_meta_path.exists() or save_wav_path.exists() or save_prompt_path.exists()):
            return

        # Count only if we are actually going to write.
        saved += 1

        # Audio (optional in dry_run; only save if available)
        if source_concat_wav_path is not None:
            save_wav_path.write_bytes(source_concat_wav_path.read_bytes())
        elif audio_part is not None:
            save_wav_path.write_bytes(base64.b64decode(audio_part["inline_data"]["data"]))

        save_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        save_prompt_path.write_text(filled_prompt, encoding="utf-8")
        if response_text is not None:
            save_resp_path.write_text(response_text, encoding="utf-8")

        saved += 1

    client = None
    if not args.dry_run:
        api_key = _resolve_api_key(api_key_arg=args.api_key)
        client = genai.Client(api_key=api_key)

    rows = data_df.to_dict(orient="records")

    # Apply resume-skip BEFORE doing any heavy work (rendering audio / encoding / API calls).
    rows_to_process: list[dict] = []
    for row in rows:
        wav_rel = Path(str(row["wav_path"]))
        wav_path_str = str(row["wav_path"])

        if args.resume:
            # Resolve the effective variant that will be stored in the JSONL ("off" / "on"),
            # so resume keys match the output rows.
            if use_disk_mode and args.ablation_variant == "auto":
                concat_meta_path = (Path(args.concat_meta_dir) / wav_rel).with_suffix(".json")
                meta_tmp = json.loads(concat_meta_path.read_text(encoding="utf-8"))
                variant_key = str(meta_tmp.get("ablation_variant", "off"))
            else:
                variant_key = args.ablation_variant if args.ablation_variant != "auto" else "off"

            if (variant_key, wav_path_str) in done_keys:
                continue

        rows_to_process.append(row)

    if not rows_to_process:
        print(f"No rows to process. Output remains: {out_path}")
        return

    total = len(rows_to_process)
    print(f"Total rows to process: {total} (batch_size={args.batch_size}, dry_run={args.dry_run})")

    async def _process_one(row: dict) -> dict:
        """
        Process one row end-to-end. Never raises (returns a record with 'error' on failure)
        so one bad item doesn't crash the whole run.
        """
        row_id = row.get("id")
        wav_path_str = str(row.get("wav_path", ""))
        wav_rel = Path(wav_path_str) if wav_path_str else Path("UNKNOWN.wav")
        prompt_txt = str(row.get("caption", "")).strip()
        variant_guess = args.ablation_variant if args.ablation_variant != "auto" else "off"

        try:
            if not wav_path_str:
                raise ValueError("Missing wav_path")
            if not prompt_txt:
                raise ValueError("Missing caption")

            specs_dict = _parse_patch_data_value(row.get("patch_data"), row_id=row_id)

            if use_disk_mode:
                concat_wav_path = Path(args.concat_wav_dir) / wav_rel
                concat_meta_path = (Path(args.concat_meta_dir) / wav_rel).with_suffix(".json")
                meta = json.loads(concat_meta_path.read_text(encoding="utf-8"))

                variant = args.ablation_variant
                if variant == "auto":
                    variant = str(meta.get("ablation_variant", "off"))
                if variant not in ("off", "on"):
                    raise ValueError(f"Unexpected ablation_variant resolved: {variant}")

                op_ablation_list = meta["op_ablation_list"]
                op_ablation_list_for_prompt = _slim_op_ablation_list_for_prompt(op_ablation_list, row_id=row_id)
                audio_part = None if args.dry_run else _encode_audio_file_inline_part(concat_wav_path)
                source_concat_wav_path = concat_wav_path
            else:
                variant = variant_guess
                per_item_seed = f"{args.seed}-{row_id}"
                audio_arr, meta = concat_renderer._render_concat_for_one_specs(
                    specs_dict,
                    sr=args.sr,
                    segment_seconds=args.segment_seconds,
                    note=args.note,
                    velocity=args.velocity,
                    out_scale=args.out_scale,
                    ablate_mode=args.ablate_mode,
                    ablation_variant=variant,
                    seed=per_item_seed,
                    max_ablations=args.max_ablations,
                )
                op_ablation_list = meta["op_ablation_list"]
                op_ablation_list_for_prompt = _slim_op_ablation_list_for_prompt(op_ablation_list, row_id=row_id)

                need_audio_part = (not args.dry_run) or (save_dir is not None and args.save_examples_limit > 0)
                audio_part = _encode_wav_int16_inline_part(audio_arr, sr=args.sr) if need_audio_part else None
                source_concat_wav_path = None
                meta["row_id"] = int(row_id) if row_id is not None else None
                meta["wav_path"] = str(wav_rel)

            template = prompts[variant]
            dx7_obj = _extract_dx7_specs_json_from_patch_data(specs_dict)
            filled_prompt = _build_prompt_text(
                template=template,
                prompt_text=prompt_txt,
                dx7_specs_json_obj=dx7_obj,
                op_ablation_list_obj=op_ablation_list_for_prompt,
            )

            response_text: str | None = None
            parsed: dict = {"cot": None, "final_json_str": None}
            error: str | None = None

            if not args.dry_run:
                assert client is not None
                try:
                    response_text = await _generate_one(
                        client,
                        model=args.model,
                        prompt_text=filled_prompt,
                        audio_part=audio_part,  # required in non-dry_run
                    )
                    parsed = _parse_cot_response(response_text)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

            await _maybe_save_example(
                wav_rel=wav_rel,
                variant=variant,
                meta=meta,
                filled_prompt=filled_prompt,
                response_text=response_text,
                audio_part=audio_part,
                source_concat_wav_path=source_concat_wav_path,
            )

            if args.dry_run:
                record = {
                    "id": int(row["id"]),
                    "wav_path": str(row["wav_path"]),
                    "ablation_variant": variant,
                    "caption": prompt_txt,
                    "op_ablation_list": op_ablation_list,
                    "op_ablation_list_prompt": op_ablation_list_for_prompt,
                    "dx7_specs_json": dx7_obj,
                    "filled_prompt": filled_prompt,
                    "render_meta": meta,
                }
            else:
                record = {
                    "id": int(row["id"]),
                    "wav_path": str(row["wav_path"]),
                    "ablation_variant": variant,
                    "caption": prompt_txt,
                    "op_ablation_list": op_ablation_list,
                    "op_ablation_list_prompt": op_ablation_list_for_prompt,
                    "dx7_specs_json": dx7_obj,
                    "model": args.model,
                    "response_text": response_text,
                    "cot": parsed["cot"],
                    "final_json_str": parsed["final_json_str"],
                    "render_meta": meta,
                }
                if error is not None:
                    record["error"] = error

            return record
        except Exception as exc:
            return {
                "id": int(row_id) if row_id is not None else None,
                "wav_path": wav_path_str,
                "ablation_variant": variant_guess,
                "caption": prompt_txt,
                "error": f"{type(exc).__name__}: {exc}",
            }

    out_mode = "a" if args.resume else "w"
    with out_path.open(out_mode, encoding="utf-8") as f:
        batches = (total + args.batch_size - 1) // args.batch_size
        done_count = 0
        ok_count = 0
        err_count = 0
        for b in range(batches):
            start = b * args.batch_size
            end = min(total, (b + 1) * args.batch_size)
            batch_rows = rows_to_process[start:end]

            print(f"Starting batch {b + 1}/{batches}: dispatching {len(batch_rows)} request(s) ({start}:{end})")

            tasks = [asyncio.create_task(_process_one(row)) for row in batch_rows]
            batch_ok = 0
            batch_err = 0
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_count += 1
                if isinstance(rec.get("error"), str) and rec.get("error"):
                    err_count += 1
                    batch_err += 1
                    print(f"[error] {done_count}/{total} wav_path={rec.get('wav_path')} err={rec.get('error')}")
                else:
                    ok_count += 1
                    batch_ok += 1
                    # Light progress signal every 100 completions.
                    if done_count % 100 == 0 or done_count == total:
                        print(f"[ok] progress {done_count}/{total} (ok={ok_count}, error={err_count})")

            f.flush()

            print(f"Completed batch {b + 1}/{batches}: ok={batch_ok} error={batch_err} (total {done_count}/{total})")
            if args.batch_sleep_seconds > 0:
                await asyncio.sleep(float(args.batch_sleep_seconds))

    if args.dry_run:
        print(f"[dry_run] Wrote request package(s) to: {out_path}")
    else:
        print(f"Wrote responses to: {out_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


