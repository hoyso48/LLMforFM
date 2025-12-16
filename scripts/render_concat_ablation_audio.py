#!/usr/bin/env python3
"""
Render one concatenated audio clip per DX7 patch:

- Segment 0: FULL patch audio
- Segments 1..K: OPk_OFF or OPk_ON variants in a RANDOMIZED order (important!)

Each segment is forced to an exact fixed duration (default: 10.0s). The current
DX7 renderer outputs ~9.6s, so we pad trailing silence to reach exactly 10.0s.

We also save a metadata JSON sidecar that maps each operator-variant segment to the
time span (in seconds) inside the concatenated audio. This metadata is meant
to be used to fill the {OP_ABLATION_LIST} placeholder in `cot_generation_prompt`.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io.wavfile import write

# Allow running this script from anywhere (not only via `python -m`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dx7.utils import render_from_specs


def _parse_patch_data(raw_value: Any, *, row_id: Any | None = None) -> dict:
    if isinstance(raw_value, dict):
        return raw_value
    if raw_value is None:
        raise ValueError(f"Row {row_id}: missing patch_data")
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if candidate == "":
            raise ValueError(f"Row {row_id}: empty patch_data string")
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError(f"Row {row_id}: patch_data JSON must be an object, got {type(parsed)}")
            return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception as exc:
                raise ValueError(f"Row {row_id}: failed to parse patch_data ({exc})") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Row {row_id}: patch_data literal must be a dict, got {type(parsed)}")
            return parsed
    raise TypeError(f"Row {row_id}: unsupported patch_data type {type(raw_value)}")


def _as_int_matrix(value: Any, *, name: str) -> np.ndarray:
    arr = np.array(value, dtype=int)
    if arr.shape != (6, 6):
        raise ValueError(f"Expected {name} with shape (6,6), got {arr.shape}")
    return arr


def _as_int_vector(value: Any, *, name: str) -> np.ndarray:
    arr = np.array(value, dtype=int)
    if arr.shape != (6,):
        raise ValueError(f"Expected {name} with shape (6,), got {arr.shape}")
    return arr


def _op_index_to_param_index(op_idx: int) -> int:
    """
    Map operator index (OP1=0 .. OP6=5) to the index used by per-operator parameter arrays
    in this repo's DX7 specs.

    Important:
    - In the dataset/specs used by this project, `modmatrix` / `outmatrix` are in OP1..OP6 order.
    - Many per-operator parameter arrays (e.g., `ol`, `sensitivity`, `coarse`, `fine`, `detune`,
      `fixed_freq`, and columns of `eg_rate`/`eg_level`) are stored in the reversed order.
    - The renderer compensates by reversing/using (5-i) indexing internally.

    Therefore, when we want to mute/keep parameters for a logical operator OPk, we must
    apply changes at index (5-k) in those per-operator arrays.
    """
    if not (0 <= op_idx < 6):
        raise ValueError(f"op_idx must be in [0,5], got {op_idx}")
    return 5 - op_idx


def _get_ablation_candidates(specs: dict, *, mode: str) -> list[int]:
    """
    Returns 0-based operator indices eligible for ablation.

    mode:
    - carriers  : outmatrix[k] == 1
    - modulators: outmatrix[k] == 0 and operator k modulates someone (column sum > 0)
    - active    : carriers OR modulators
    - all       : [0..5]
    """
    mod = _as_int_matrix(specs["modmatrix"], name="modmatrix")
    out = _as_int_vector(specs["outmatrix"], name="outmatrix")

    outgoing_counts = mod.sum(axis=0)  # column sums: how many operators are modulated by OP(k)

    if mode == "carriers":
        return [k for k in range(6) if int(out[k]) == 1]
    if mode == "modulators":
        return [k for k in range(6) if int(out[k]) == 0 and int(outgoing_counts[k]) > 0]
    if mode == "active":
        return [k for k in range(6) if int(out[k]) == 1 or int(outgoing_counts[k]) > 0]
    if mode == "all":
        return list(range(6))

    raise ValueError(f"Unknown ablation mode: {mode}")


def _ablate_operator(specs: dict, *, op_idx: int) -> dict:
    """
    Apply the ablation semantics from research_progress.md:
    - mute operator k as a carrier,
    - remove operator k's modulation outputs,
    - do not create any new connections.

    Implementation details:
    - Set outmatrix[k] = 0
    - Zero the modmatrix column k (and explicitly clear modmatrix[k][k] feedback)
    - Set ol[k] = 0 and sensitivity[k] = 0 for a hard mute in this renderer
    """
    if not (0 <= op_idx < 6):
        raise ValueError(f"op_idx must be in [0,5], got {op_idx}")

    out_specs = json.loads(json.dumps(specs))  # deep copy, JSON-safe

    mod = _as_int_matrix(out_specs["modmatrix"], name="modmatrix")
    out = _as_int_vector(out_specs["outmatrix"], name="outmatrix")

    # Remove modulation outputs of OPk (including self-feedback).
    mod[:, op_idx] = 0
    mod[op_idx, op_idx] = 0
    out[op_idx] = 0

    out_specs["modmatrix"] = mod.tolist()
    out_specs["outmatrix"] = out.tolist()

    if "ol" in out_specs:
        pidx = _op_index_to_param_index(op_idx)
        ol = list(out_specs["ol"])
        if len(ol) != 6:
            raise ValueError(f"Expected ol with length 6, got {len(ol)}")
        ol[pidx] = 0
        out_specs["ol"] = ol

    if "sensitivity" in out_specs:
        pidx = _op_index_to_param_index(op_idx)
        sens = list(out_specs["sensitivity"])
        if len(sens) != 6:
            raise ValueError(f"Expected sensitivity with length 6, got {len(sens)}")
        sens[pidx] = 0
        out_specs["sensitivity"] = sens

    return out_specs


def _get_downstream_closure(specs: dict, *, op_idx: int) -> set[int]:
    """
    Compute the transitive closure of operators *downstream* of OPk.

    Conventions:
    - modmatrix[i][j] = 1 means OP(j+1) modulates OP(i+1)
    - This corresponds to a directed edge OP(j+1) -> OP(i+1).
    - Downstream of OPk means all operators reachable by repeatedly following edges
      originating from OPk.
    """
    if not (0 <= op_idx < 6):
        raise ValueError(f"op_idx must be in [0,5], got {op_idx}")
    mod = _as_int_matrix(specs["modmatrix"], name="modmatrix")

    keep: set[int] = {op_idx}
    stack: list[int] = [op_idx]
    while stack:
        j = stack.pop()
        # Operators downstream of j are all i where mod[i, j] == 1 (edge j -> i)
        for i in range(6):
            if int(mod[i, j]) == 1 and i not in keep:
                keep.add(i)
                stack.append(i)
    return keep


def _isolate_operator_on(specs: dict, *, op_idx: int) -> tuple[dict, list[int], bool]:
    """
    OPk_ON semantics:
    Keep only operator k and all operators that are downstream of k (transitively),
    and disable everything else (no bypass connections are created).
    """
    keep = _get_downstream_closure(specs, op_idx=op_idx)
    keep_sorted = sorted(keep)

    out_specs = json.loads(json.dumps(specs))  # deep copy, JSON-safe

    mod = _as_int_matrix(out_specs["modmatrix"], name="modmatrix")
    out = _as_int_vector(out_specs["outmatrix"], name="outmatrix")

    # Remove any connection involving a disabled operator.
    for i in range(6):
        for j in range(6):
            if i not in keep or j not in keep:
                mod[i, j] = 0

    # Disable carrier outputs for operators outside the closure.
    for i in range(6):
        if i not in keep:
            out[i] = 0

    # Output routing:
    # - Preserve original carriers within the kept subgraph (outmatrix entries remain as-is).
    # - If the kept subgraph contains no carriers, the resulting OPk_ON audio is silent by design.
    #   (We optionally SKIP such OPk_ON segments at the concatenation stage.)
    forced_output = False

    out_specs["modmatrix"] = mod.tolist()
    out_specs["outmatrix"] = out.tolist()

    if "ol" in out_specs:
        ol = list(out_specs["ol"])
        if len(ol) != 6:
            raise ValueError(f"Expected ol with length 6, got {len(ol)}")
        keep_param_idxs = {_op_index_to_param_index(i) for i in keep}
        for param_idx in range(6):
            if param_idx not in keep_param_idxs:
                ol[param_idx] = 0
        out_specs["ol"] = ol

    if "sensitivity" in out_specs:
        sens = list(out_specs["sensitivity"])
        if len(sens) != 6:
            raise ValueError(f"Expected sensitivity with length 6, got {len(sens)}")
        keep_param_idxs = {_op_index_to_param_index(i) for i in keep}
        for param_idx in range(6):
            if param_idx not in keep_param_idxs:
                sens[param_idx] = 0
        out_specs["sensitivity"] = sens

    return out_specs, keep_sorted, forced_output


def _pad_or_trim_to_length(audio: np.ndarray, *, target_len: int) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio shape (N,), got {audio.shape}")
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        return audio[:target_len]
    pad = np.zeros(target_len - len(audio), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=0)


@dataclass(frozen=True)
class SegmentInfo:
    segment_index: int
    label: str
    start_sec: float
    end_sec: float
    op_index: int | None = None  # 0-based, only for OPk_{OFF,ON}
    variant: str | None = None  # "off" | "on" for OPk segments; None for FULL
    kept_ops: list[int] | None = None  # only for OPk_ON (0-based)
    forced_output: bool | None = None  # only meaningful for OPk_ON (True iff OPk was force-routed to output)


def _segments_to_metadata_dict(segments: list[SegmentInfo]) -> tuple[list[dict], list[dict]]:
    seg_dicts: list[dict] = []
    op_segments: list[dict] = []
    for s in segments:
        d = {
            "segment_index": s.segment_index,
            "label": s.label,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
        }
        if s.op_index is not None:
            d["op_index"] = int(s.op_index)  # 0-based
            d["op_number"] = int(s.op_index) + 1  # 1-based
        if s.variant is not None:
            d["variant"] = s.variant
        if s.kept_ops is not None:
            d["kept_ops"] = [int(x) for x in s.kept_ops]
            d["kept_ops_1based"] = [int(x) + 1 for x in s.kept_ops]
        if s.forced_output is not None:
            d["forced_output"] = bool(s.forced_output)
        seg_dicts.append(d)
        if s.op_index is not None:
            op_segments.append(
                {
                    "op_index": int(s.op_index),
                    "op_number": int(s.op_index) + 1,
                    "label": s.label,
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "variant": s.variant,
                    "kept_ops": [int(x) for x in (s.kept_ops or [])],
                    "forced_output": bool(s.forced_output) if s.forced_output is not None else False,
                }
            )
    return seg_dicts, op_segments


def _render_concat_for_one_specs(
    specs: dict,
    *,
    sr: int,
    segment_seconds: float,
    note: int,
    velocity: int,
    out_scale: float,
    ablate_mode: str,
    ablation_variant: str,
    seed: str,
    max_ablations: int | None,
) -> tuple[np.ndarray, dict]:
    if segment_seconds <= 0:
        raise ValueError(f"segment_seconds must be > 0, got {segment_seconds}")
    if ablation_variant not in {"off", "on"}:
        raise ValueError(f"ablation_variant must be 'off' or 'on', got {ablation_variant}")

    target_len = int(round(sr * segment_seconds))
    rng = random.Random(seed)

    candidates = _get_ablation_candidates(specs, mode=ablate_mode)
    if not candidates:
        raise ValueError(f"No ablation candidates found (mode={ablate_mode}).")

    rng.shuffle(candidates)  # CRITICAL: randomized OPk_{OFF,ON} order
    if max_ablations is not None:
        if max_ablations <= 0:
            raise ValueError(f"max_ablations must be > 0, got {max_ablations}")
        candidates = candidates[:max_ablations]

    rendered_segments: list[np.ndarray] = []
    segment_infos: list[SegmentInfo] = []
    skipped_no_carrier_ops: list[int] = []

    # FULL segment (always first)
    full_audio = render_from_specs(specs, sr=sr, n=note, v=velocity, out_scale=out_scale)
    full_audio = _pad_or_trim_to_length(full_audio, target_len=target_len)
    rendered_segments.append(full_audio)
    segment_infos.append(
        SegmentInfo(
            segment_index=0,
            label="FULL",
            start_sec=0.0,
            end_sec=float(segment_seconds),
            op_index=None,
            variant=None,
            kept_ops=None,
            forced_output=None,
        )
    )

    # OPk_OFF segments (randomized order)
    for seg_idx, op_idx in enumerate(candidates, start=1):
        kept_ops: list[int] | None = None
        forced_output: bool | None = None
        if ablation_variant == "off":
            variant_specs = _ablate_operator(specs, op_idx=op_idx)
            label = f"OP{op_idx + 1}_OFF"
        else:
            # If OPk_ON downstream subgraph contains no carriers, it does not contribute to the final sound.
            # We skip generating this segment to avoid emitting 10s of silence and to keep semantics strict.
            keep_set = _get_downstream_closure(specs, op_idx=op_idx)
            out_full = _as_int_vector(specs["outmatrix"], name="outmatrix")
            has_carrier = any(int(out_full[i]) == 1 for i in keep_set)
            if not has_carrier:
                skipped_no_carrier_ops.append(int(op_idx))
                continue
            variant_specs, kept_ops, forced_output = _isolate_operator_on(specs, op_idx=op_idx)
            label = f"OP{op_idx + 1}_ON"

        audio = render_from_specs(variant_specs, sr=sr, n=note, v=velocity, out_scale=out_scale)
        audio = _pad_or_trim_to_length(audio, target_len=target_len)
        rendered_segments.append(audio)

        start = float(seg_idx) * float(segment_seconds)
        end = float(seg_idx + 1) * float(segment_seconds)
        segment_infos.append(
            SegmentInfo(
                segment_index=seg_idx,
                label=label,
                start_sec=start,
                end_sec=end,
                op_index=op_idx,
                variant=ablation_variant,
                kept_ops=kept_ops,
                forced_output=forced_output,
            )
        )

    combined = np.concatenate(rendered_segments, axis=0)

    segments_list, op_segment_list = _segments_to_metadata_dict(segment_infos)
    meta = {
        "sr": int(sr),
        "segment_seconds": float(segment_seconds),
        "segment_count": int(len(segment_infos)),
        "ablate_mode": str(ablate_mode),
        "ablation_variant": str(ablation_variant),
        "opk_on_output_policy": (
            "preserve_original_outmatrix_within_kept_set; skip OPk_ON if no carriers are reachable downstream"
            if ablation_variant == "on"
            else None
        ),
        "seed": str(seed),
        "op_candidates_random_order": [int(k) for k in candidates],  # 0-based, in audio order
        "skipped_no_carrier_ops": skipped_no_carrier_ops,  # 0-based op indices skipped for OPk_ON
        "segments": segments_list,  # FULL + OPk_OFF in chronological order
        # This list is intended to fill {OP_ABLATION_LIST} in the CoT teacher prompt(s).
        # It is NOT guaranteed to be sorted by op_number.
        "op_ablation_list": op_segment_list,
    }
    return combined, meta


def _load_records(input_path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path, lines=True)
    if limit:
        df = df.head(limit)
    records = df.to_dict(orient="records")
    if not records:
        raise ValueError(f"No rows found in {input_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Render concatenated FULL+OPk_OFF audio clips + metadata.")
    parser.add_argument(
        "--input_path",
        type=str,
        help="CSV or JSONL file containing at least patch_data. If provided, renders one output per row.",
    )
    parser.add_argument(
        "--patch_json",
        type=str,
        help="Path to a single patch JSON file (alternative to --input_path).",
    )
    parser.add_argument(
        "--out_wav_dir",
        type=str,
        default="data/wav_concat",
        help="Base directory to write concatenated WAVs.",
    )
    parser.add_argument(
        "--out_meta_dir",
        type=str,
        default="data/wav_concat_meta",
        help="Base directory to write metadata JSON sidecars.",
    )
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--segment_seconds", type=float, default=10.0)
    parser.add_argument("--note", type=int, default=60)
    parser.add_argument("--velocity", type=int, default=100)
    parser.add_argument("--out_scale", type=float, default=1.0)
    parser.add_argument(
        "--ablation_variant",
        choices=["off", "on"],
        default="off",
        help="Which operator variant to render as segments: OPk_OFF (off) or OPk_ON (on).",
    )
    parser.add_argument(
        "--ablate_mode",
        choices=["active", "carriers", "modulators", "all"],
        default="active",
        help="Which operators are eligible for ablation.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="0",
        help="Base seed. The per-item seed becomes f'{seed}-{row_id}' when using --input_path.",
    )
    parser.add_argument("--max_ablations", type=int, help="Optional cap on number of OPk_OFF segments per sample.")
    parser.add_argument("--limit", type=int, help="Optional limit on number of rows when using --input_path.")
    args = parser.parse_args()

    out_wav_dir = Path(args.out_wav_dir)
    out_meta_dir = Path(args.out_meta_dir)
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.input_path) == bool(args.patch_json):
        raise ValueError("Provide exactly one of --input_path or --patch_json.")

    if args.patch_json:
        patch_path = Path(args.patch_json)
        specs = json.loads(patch_path.read_text(encoding="utf-8"))
        combined, meta = _render_concat_for_one_specs(
            specs,
            sr=args.sr,
            segment_seconds=args.segment_seconds,
            note=args.note,
            velocity=args.velocity,
            out_scale=args.out_scale,
            ablate_mode=args.ablate_mode,
            ablation_variant=args.ablation_variant,
            seed=str(args.seed),
            max_ablations=args.max_ablations,
        )

        out_wav_path = out_wav_dir / (patch_path.stem + "__concat.wav")
        out_meta_path = out_meta_dir / (patch_path.stem + "__concat.json")

        write(out_wav_path, args.sr, combined.astype(np.int16))
        meta["out_wav_path"] = str(out_wav_path)
        meta["source_patch_json"] = str(patch_path)
        out_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote: {out_wav_path}")
        print(f"Wrote: {out_meta_path}")
        return

    # Dataset mode
    input_path = Path(args.input_path)
    records = _load_records(input_path, limit=args.limit)

    for row in records:
        row_id = row.get("id")
        specs = _parse_patch_data(row.get("patch_data"), row_id=row_id)

        # Preserve directory structure when possible (wav_path is usually present in this repo).
        wav_path_rel = row.get("wav_path")
        if isinstance(wav_path_rel, str) and wav_path_rel.strip():
            wav_rel = Path(wav_path_rel)
            out_wav_path = out_wav_dir / wav_rel
            out_meta_path = (out_meta_dir / wav_rel).with_suffix(".json")
        else:
            out_wav_path = out_wav_dir / f"{row_id}__concat.wav"
            out_meta_path = out_meta_dir / f"{row_id}__concat.json"

        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        out_meta_path.parent.mkdir(parents=True, exist_ok=True)

        per_item_seed = f"{args.seed}-{row_id}"
        combined, meta = _render_concat_for_one_specs(
            specs,
            sr=args.sr,
            segment_seconds=args.segment_seconds,
            note=args.note,
            velocity=args.velocity,
            out_scale=args.out_scale,
            ablate_mode=args.ablate_mode,
            ablation_variant=args.ablation_variant,
            seed=per_item_seed,
            max_ablations=args.max_ablations,
        )

        write(out_wav_path, args.sr, combined.astype(np.int16))

        # Row metadata
        meta["row_id"] = row_id
        if "name" in row:
            meta["name"] = row["name"]
        if "wav_path" in row:
            meta["source_wav_path"] = row["wav_path"]
        meta["out_wav_path"] = str(out_wav_path)
        meta["source_dataset_path"] = str(input_path)

        out_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Done. Wrote concatenated WAVs under: {out_wav_dir}")
    print(f"Done. Wrote metadata JSONs under: {out_meta_dir}")


if __name__ == "__main__":
    main()


