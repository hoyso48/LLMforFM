#!/usr/bin/env python3
"""
Normalize various result CSV schemas into the evaluation format expected by:
- dx7/patch_to_wav.py (needs: id, wav_path, patch_data)
- kadtk/evaluate.py   (needs: id, wav_path)
- evaluate_timbral.py (needs: id, wav_path)
- evaluate.py (CLAP)  (needs: id, wav_path; captions provided separately)

Supported inputs (auto-detected):
- Inference outputs: columns include wav_path, patch_data (and optionally patch_data_rendered)
- gemma2_results.csv: columns include id, wav_path_x, wav_path_y, generated_patch_data

By default, rows with patch_data == 'FAILED_RENDER' are dropped (no patchwork fallbacks).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


FAILED_SENTINELS = {"FAILED_RENDER", "FAILED", "ERROR", "None", "nan", "NaN", ""}


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        return df.drop(columns=unnamed)
    return df


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _is_missing_patch(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s in FAILED_SENTINELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument(
        "--drop_failed_render",
        action="store_true",
        help="Drop rows whose patch_data is missing/FAILED_RENDER (recommended).",
    )
    ap.add_argument(
        "--prefer_rendered_patch",
        action="store_true",
        help="If patch_data_rendered exists, use it instead of patch_data.",
    )
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    df = _drop_unnamed(df)

    if "id" not in df.columns:
        raise ValueError("Input CSV must contain column 'id'")

    # Choose patch_data source
    patch_col = None
    if bool(args.prefer_rendered_patch):
        patch_col = _first_existing(df, ["patch_data_rendered", "patch_data", "generated_patch_data"])
    else:
        patch_col = _first_existing(df, ["patch_data", "patch_data_rendered", "generated_patch_data"])

    # Choose wav_path source (prefer *_x which is the GT-relative path for Yamaha rows)
    wav_col = _first_existing(df, ["wav_path", "wav_path_x", "wav_path_y"])
    if wav_col is None:
        raise ValueError("Could not find a wav path column (expected one of: wav_path, wav_path_x, wav_path_y)")

    if patch_col is None:
        raise ValueError("Could not find a patch data column (expected one of: patch_data, patch_data_rendered, generated_patch_data)")

    out = pd.DataFrame(
        {
            "id": df["id"],
            "wav_path": df[wav_col].astype(str),
            "patch_data": df[patch_col],
        }
    )

    n_total = len(out)
    if bool(args.drop_failed_render):
        mask_ok = ~out["patch_data"].apply(_is_missing_patch)
        out = out.loc[mask_ok].copy()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n_kept = len(out)
    n_dropped = n_total - n_kept
    print(f"[normalize_eval_csv] input={in_path} rows={n_total} kept={n_kept} dropped={n_dropped}")
    print(f"[normalize_eval_csv] wav_col={wav_col} patch_col={patch_col} output={out_path}")


if __name__ == "__main__":
    main()


