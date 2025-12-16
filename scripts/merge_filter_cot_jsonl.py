#!/usr/bin/env python3
"""
Merge + filter CoT JSONL shards into a single clean JSONL.

Typical use in this repo:
- Remove records that contain a non-empty "error" field
- Optionally require that "cot" and "final_json_str" exist and that final_json_str is valid JSON
- Deduplicate by (ablation_variant, wav_path)
- Sort for deterministic output

This script intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Stats:
    total_lines: int = 0
    parsed_json_objects: int = 0
    skipped_blank: int = 0
    skipped_malformed_json: int = 0
    skipped_non_object: int = 0
    skipped_error_field: int = 0
    skipped_wrong_dataset: int = 0
    skipped_missing_wav_path: int = 0
    skipped_missing_variant: int = 0
    skipped_missing_cot: int = 0
    skipped_missing_final_json_str: int = 0
    skipped_bad_final_json_str: int = 0
    dedup_dropped: int = 0
    kept: int = 0


def _is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _parse_json_line(line: str) -> dict[str, Any] | None:
    obj = json.loads(line)
    if not isinstance(obj, dict):
        return None
    return obj


def _final_json_str_is_valid_json_object(final_json_str: str) -> bool:
    """
    Validate that final_json_str starts with a JSON object and that the first JSON value is a dict.
    Uses raw_decode so it tolerates trailing whitespace; it does NOT tolerate trailing non-whitespace.
    """
    s = final_json_str.lstrip()
    if not s.startswith("{"):
        return False
    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(s)
    if not isinstance(obj, dict):
        return False
    tail = s[end:].strip()
    return tail == ""


def _dedup_key(rec: dict[str, Any]) -> tuple[str, str]:
    return (str(rec.get("ablation_variant", "")), str(rec.get("wav_path", "")))


def _sort_key(rec: dict[str, Any]) -> tuple[int, str]:
    """
    Stable deterministic ordering:
    - primary: numeric id if available, else large sentinel
    - secondary: wav_path
    """
    wav_path = str(rec.get("wav_path", ""))
    rid = rec.get("id")
    if isinstance(rid, int):
        return (rid, wav_path)
    if isinstance(rid, str) and rid.isdigit():
        return (int(rid), wav_path)
    return (2**31 - 1, wav_path)


def merge_filter_jsonl(
    *,
    input_paths: list[Path],
    out_path: Path,
    dataset_prefix: str | None,
    strict: bool,
) -> Stats:
    stats = Stats()

    # Keep the "best" record per key. For now, first wins; duplicates are counted and dropped.
    kept_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for p in sorted(input_paths, key=lambda x: str(x)):
        if not p.exists():
            raise FileNotFoundError(f"Input JSONL not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                stats.total_lines += 1
                s = line.strip()
                if not s:
                    stats.skipped_blank += 1
                    continue

                try:
                    obj = _parse_json_line(s)
                except json.JSONDecodeError:
                    stats.skipped_malformed_json += 1
                    continue

                if obj is None:
                    stats.skipped_non_object += 1
                    continue

                stats.parsed_json_objects += 1

                err = obj.get("error")
                if _is_non_empty_str(err):
                    stats.skipped_error_field += 1
                    continue

                wav_path = obj.get("wav_path")
                if not _is_non_empty_str(wav_path):
                    stats.skipped_missing_wav_path += 1
                    continue

                if dataset_prefix is not None and not str(wav_path).startswith(dataset_prefix):
                    stats.skipped_wrong_dataset += 1
                    continue

                variant = obj.get("ablation_variant") or obj.get("variant")
                if not _is_non_empty_str(variant):
                    stats.skipped_missing_variant += 1
                    continue
                obj["ablation_variant"] = str(variant)

                if strict:
                    cot = obj.get("cot")
                    if not _is_non_empty_str(cot):
                        stats.skipped_missing_cot += 1
                        continue
                    final_json_str = obj.get("final_json_str")
                    if not _is_non_empty_str(final_json_str):
                        stats.skipped_missing_final_json_str += 1
                        continue
                    if not _final_json_str_is_valid_json_object(str(final_json_str)):
                        stats.skipped_bad_final_json_str += 1
                        continue

                key = _dedup_key(obj)
                if key in kept_by_key:
                    stats.dedup_dropped += 1
                    continue
                kept_by_key[key] = obj

    records = list(kept_by_key.values())
    records.sort(key=_sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats.kept = len(records)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge + filter CoT JSONL shards.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL file paths (you can pass multiple). Shell globs are supported by your shell.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default=None,
        help="If set, only keep records whose wav_path starts with this prefix (e.g., DX7_YAMAHA/).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, also require non-empty cot + valid final_json_str JSON object (no trailing junk).",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    out_path = Path(args.out)
    stats = merge_filter_jsonl(
        input_paths=input_paths,
        out_path=out_path,
        dataset_prefix=args.dataset_prefix,
        strict=bool(args.strict),
    )

    print("Done.")
    print(f"  out: {out_path}")
    print(f"  total_lines: {stats.total_lines}")
    print(f"  parsed_json_objects: {stats.parsed_json_objects}")
    print(f"  kept: {stats.kept}")
    print(f"  skipped_error_field: {stats.skipped_error_field}")
    if args.dataset_prefix is not None:
        print(f"  skipped_wrong_dataset: {stats.skipped_wrong_dataset}")
    if args.strict:
        print(f"  skipped_missing_cot: {stats.skipped_missing_cot}")
        print(f"  skipped_missing_final_json_str: {stats.skipped_missing_final_json_str}")
        print(f"  skipped_bad_final_json_str: {stats.skipped_bad_final_json_str}")
    print(f"  dedup_dropped: {stats.dedup_dropped}")
    if stats.skipped_malformed_json:
        print(f"  skipped_malformed_json: {stats.skipped_malformed_json}")
    if stats.skipped_non_object:
        print(f"  skipped_non_object: {stats.skipped_non_object}")
    if stats.skipped_blank:
        print(f"  skipped_blank: {stats.skipped_blank}")


if __name__ == "__main__":
    main()


