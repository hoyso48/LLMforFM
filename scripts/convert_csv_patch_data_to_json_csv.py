#!/usr/bin/env python3
"""
Convert DX7 dataset CSV files while keeping CSV format, but normalizing `patch_data`
from Python literal strings to JSON strings.

This avoids `eval` and makes the dataset "JSON-friendly" while remaining CSV.

Example:
  python scripts/convert_csv_patch_data_to_json_csv.py \
    --csv data/DX7_YAMAHA_train.csv \
    --out data/DX7_YAMAHA_train_json.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_patch_data(value: Any, row_id: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Row {row_id}: patch_data has unsupported type {type(value)}")

    candidate = value.strip()
    if candidate == "":
        return None

    # Prefer JSON if already normalized
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Row {row_id}: patch_data JSON must be an object, got {type(parsed)}")
    except json.JSONDecodeError:
        pass

    # Legacy format: Python literal dict (single quotes, True/False)
    try:
        parsed = ast.literal_eval(candidate)
    except Exception as exc:
        raise ValueError(f"Row {row_id}: failed to parse patch_data as Python literal: {exc}") from exc
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise ValueError(f"Row {row_id}: patch_data Python literal must be a dict, got {type(parsed)}")
    return parsed


def convert_csv(csv_path: Path, out_path: Path, *, ensure_ascii: bool = False) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Allow very large fields (patch_data can be huge and multiline)
    csv.field_size_limit(1024 * 1024 * 1024)

    with csv_path.open("r", newline="", encoding="utf-8") as infile, out_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for i, row in enumerate(reader):
            row_id = row.get("id", i)
            if "patch_data" in row:
                parsed = _parse_patch_data(row.get("patch_data"), row_id=row_id)
                if parsed is None:
                    row["patch_data"] = ""
                else:
                    # Compact JSON string to keep file size reasonable
                    row["patch_data"] = json.dumps(parsed, ensure_ascii=ensure_ascii, separators=(",", ":"))
            writer.writerow(row)

    print(f"Wrote CSV with JSON patch_data: {out_path} (source: {csv_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV patch_data to JSON strings (keep CSV).")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--ensure_ascii", action="store_true", help="Escape non-ASCII characters in JSON output")
    args = parser.parse_args()

    convert_csv(Path(args.csv), Path(args.out), ensure_ascii=args.ensure_ascii)


if __name__ == "__main__":
    main()


