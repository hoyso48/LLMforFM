#!/usr/bin/env python3
"""
Add a 'cot' column to a CSV by parsing \\cot{...} from JSONL 'response_text'.

Expected JSONL format (per line): a JSON object containing at least:
- id (int)
- response_text (str) OR cot (str)

We extract the inner text of the first \\cot{...} block and write it into the CSV row
matched by 'id'. Rows without a match are left empty.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class JsonlStats:
    total_lines: int = 0
    parsed_objects: int = 0
    skipped_blank: int = 0
    skipped_malformed_json: int = 0
    skipped_non_object: int = 0
    skipped_error_field: int = 0
    skipped_missing_id: int = 0
    skipped_missing_cot: int = 0
    duplicate_id_dropped: int = 0
    kept: int = 0


def _is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def extract_cot_from_response_text(text: str) -> str | None:
    """
    Extract the inner text of the first \\cot{...} block.
    Uses a simple brace-depth scan (mirrors the logic used in generate_cot_reasoning_gemini.py).
    """
    marker = "\\cot{"
    start = text.find(marker)
    if start == -1:
        return None

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
        return None

    return text[start + len(marker) : i - 1].strip()


def load_cot_map_from_jsonl(jsonl_path: Path) -> tuple[dict[int, str], JsonlStats]:
    stats = JsonlStats()
    mapping: dict[int, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            stats.total_lines += 1
            s = line.strip()
            if not s:
                stats.skipped_blank += 1
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                stats.skipped_malformed_json += 1
                continue

            if not isinstance(obj, dict):
                stats.skipped_non_object += 1
                continue

            stats.parsed_objects += 1

            err = obj.get("error")
            if _is_non_empty_str(err):
                stats.skipped_error_field += 1
                continue

            rid = obj.get("id")
            if isinstance(rid, bool) or rid is None:
                stats.skipped_missing_id += 1
                continue
            if isinstance(rid, str):
                if not rid.isdigit():
                    stats.skipped_missing_id += 1
                    continue
                rid_int = int(rid)
            elif isinstance(rid, int):
                rid_int = rid
            else:
                stats.skipped_missing_id += 1
                continue

            cot_text: str | None = None

            resp = obj.get("response_text")
            if _is_non_empty_str(resp):
                cot_text = extract_cot_from_response_text(str(resp))

            if cot_text is None and _is_non_empty_str(obj.get("cot")):
                cot_text = str(obj["cot"]).strip()

            if not _is_non_empty_str(cot_text):
                stats.skipped_missing_cot += 1
                continue

            if rid_int in mapping:
                stats.duplicate_id_dropped += 1
                continue

            mapping[rid_int] = cot_text
            stats.kept += 1

    return mapping, stats


def add_cot_column_to_csv(
    *,
    csv_path: Path,
    jsonl_path: Path,
    backup: bool,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    cot_map, st = load_cot_map_from_jsonl(jsonl_path)

    # Preserve the "blank header index column" formatting by reading the first column as index.
    df = pd.read_csv(csv_path, index_col=0)
    if "id" not in df.columns:
        raise ValueError(f"CSV is missing required column 'id': {csv_path}")

    # Convert IDs to numeric where possible for mapping.
    ids = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["cot"] = ids.map(lambda x: cot_map.get(int(x)) if pd.notna(x) else None)

    filled = int(df["cot"].notna().sum())
    total = int(len(df))

    if backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)

    df.to_csv(csv_path, index=True, index_label="")

    print("Updated CSV:")
    print(f"  csv: {csv_path}")
    print(f"  jsonl: {jsonl_path}")
    print(f"  rows: {total}")
    print(f"  rows_with_cot: {filled}")
    print("JSONL parse stats:")
    print(f"  total_lines: {st.total_lines}")
    print(f"  parsed_objects: {st.parsed_objects}")
    print(f"  kept_cot_records: {st.kept}")
    if st.skipped_error_field:
        print(f"  skipped_error_field: {st.skipped_error_field}")
    if st.skipped_missing_cot:
        print(f"  skipped_missing_cot: {st.skipped_missing_cot}")
    if st.duplicate_id_dropped:
        print(f"  duplicate_id_dropped: {st.duplicate_id_dropped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a 'cot' column to a CSV from CoT JSONL.")
    parser.add_argument("--csv_path", required=True, help="Target CSV to update in-place.")
    parser.add_argument("--jsonl_path", required=True, help="Source JSONL containing response_text with \\cot{...}.")
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Disable creation of a .bak copy next to the CSV (default: create backup if missing).",
    )
    args = parser.parse_args()

    add_cot_column_to_csv(
        csv_path=Path(args.csv_path),
        jsonl_path=Path(args.jsonl_path),
        backup=not bool(args.no_backup),
    )


if __name__ == "__main__":
    main()


