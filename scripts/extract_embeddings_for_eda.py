#!/usr/bin/env python3
"""
Extract PaSST + CLAP embeddings (and CLAP similarity scores) for EDA and filtering.
This script is intentionally "EDA-first": it produces compact, analysis-friendly artifacts
without depending on training code.

Typical workflow
1) Ensure WAVs exist for the split you want to analyze.
   - For DX7_YAMAHA, this repo does NOT ship the full rendered WAV set by default.
     You can render them with:
       python dx7/patch_to_wav.py --input_path data/DX7_YAMAHA_train.csv --wav_dir data/wav
       python dx7/patch_to_wav.py --input_path data/DX7_YAMAHA_test.csv  --wav_dir data/wav

2) Extract embeddings:
   - AllTheWeb train:
       python scripts/extract_embeddings_for_eda.py \
         --data_csv_path data/DX7_AllTheWeb_train.csv \
         --caption_csv_path data/DX7_AllTheWeb_train_captions.csv \
         --wav_dir data/wav \
         --output_dir artifacts/eda_embeddings/alltheweb_train \
         --split_name train

   - Yamaha test:
       python scripts/extract_embeddings_for_eda.py \
         --data_csv_path data/DX7_YAMAHA_test.csv \
         --caption_csv_path data/DX7_YAMAHA_test_captions.csv \
         --wav_dir data/wav \
         --output_dir artifacts/eda_embeddings/yamaha_test \
         --split_name test

Outputs (in --output_dir)
- meta.csv                  : row-aligned metadata
- clap_audio_emb.npy        : [N, 512] float16 (if CLAP enabled)
- clap_text_emb.npy         : [N, 512] float16 (if CLAP enabled)
- clap_score.npy            : [N] float32 cosine similarity (if CLAP enabled)
- passt_emb.npy             : [N, D] float16 (if PaSST enabled; D depends on model variant)
- config.json               : the run configuration for reproducibility
- missing_audio.csv         : rows whose wav files were not found (if any)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Allow running this script from anywhere (not only via `python -m`).
# Without this, `python scripts/extract_embeddings_for_eda.py` may fail to import `kadtk`
# because Python puts the script directory on sys.path instead of the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kadtk.emb_loader import EmbeddingLoader
from kadtk.model_loader import get_all_models
from clap_score import load_clap_model, compute_clap_embeddings


def _set_csv_field_size_limit() -> None:
    # Patch data / captions can be very large and even multiline.
    csv.field_size_limit(1024 * 1024 * 1024)


def _read_csv_selected(csv_path: Path, keep_columns: list[str]) -> pd.DataFrame:
    """
    Read a potentially large/multiline CSV while keeping only a subset of columns.
    Uses Python's csv module for robustness.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    _set_csv_field_size_limit()

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")
        missing = [c for c in keep_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV {csv_path} is missing required columns: {missing}")

        for row in reader:
            rows.append({c: row.get(c) for c in keep_columns})

    return pd.DataFrame(rows)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64").astype(bool)

    norm = series.astype(str).str.strip().str.lower()
    mapped = norm.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
            "t": True,
            "f": False,
        }
    )
    return mapped


def _apply_base_filters(
    df: pd.DataFrame,
    *,
    filter_inaudible: bool,
    filter_null_name: bool,
    exclude_fixed_freqs: bool,
) -> pd.DataFrame:
    """
    Base filtering used in fine-tuning:
    - inaudible == False
    - name does not contain 'NULL'
    """
    out = df.copy()

    if filter_inaudible:
        if "inaudible" not in out.columns:
            raise ValueError("Requested inaudible filtering, but column 'inaudible' was not found.")
        inaudible = _coerce_bool(out["inaudible"]).fillna(False)
        out = out[~inaudible]

    if filter_null_name:
        if "name" not in out.columns:
            raise ValueError("Requested NULL-name filtering, but column 'name' was not found.")
        out = out[~out["name"].astype(str).str.contains("NULL", na=False)]

    if exclude_fixed_freqs:
        if "has_fixed_freqs" not in out.columns:
            raise ValueError("Requested fixed-freq filtering, but column 'has_fixed_freqs' was not found.")
        fixed = _coerce_bool(out["has_fixed_freqs"]).fillna(False)
        out = out[~fixed]

    return out.reset_index(drop=True)


def _reduce_to_vector(emb: np.ndarray) -> np.ndarray:
    """
    Convert an embedding array into a single fixed-size vector.
    If the model outputs multiple frames, take mean pooling over time.
    """
    if emb.ndim == 1:
        return emb
    if emb.ndim != 2:
        raise ValueError(f"Expected embedding with ndim=1 or 2, got shape={emb.shape}")
    return emb.mean(axis=0)


def _ensure_audio_exists(
    df: pd.DataFrame,
    *,
    wav_dir: Path,
    missing_audio_policy: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    wav_paths = df["wav_path"].astype(str).tolist()
    abs_paths = [str(wav_dir / p) for p in wav_paths]
    exists = np.array([Path(p).exists() for p in abs_paths], dtype=bool)

    missing_count = int((~exists).sum())
    if missing_count > 0:
        missing_df = df.loc[~exists, [c for c in ["id", "wav_path", "name"] if c in df.columns]].copy()
        missing_df.to_csv(output_dir / "missing_audio.csv", index=False)

        msg = (
            f"Found {missing_count} missing wav files out of {len(df)} rows. "
            f"Wrote the list to {output_dir / 'missing_audio.csv'}."
        )
        if missing_audio_policy == "error":
            raise FileNotFoundError(msg + " Use --missing_audio_policy skip to ignore missing files.")
        if missing_audio_policy == "skip":
            df = df.loc[exists].reset_index(drop=True)
            abs_paths = [p for p, ok in zip(abs_paths, exists) if ok]
        else:
            raise ValueError(f"Unexpected missing_audio_policy: {missing_audio_policy}")

    return df, abs_paths


def _cleanup_torch() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_numpy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _compute_clap(
    clap_model_name: str,
    *,
    captions: list[str],
    audio_paths: list[str],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = load_clap_model(clap_model_name, device=device)
    audio_emb_t, text_emb_t = compute_clap_embeddings(
        model,
        text_list=captions,
        audio_path_list=audio_paths,
        batch_size=batch_size,
        device=device,
    )

    sim_t = torch.nn.functional.cosine_similarity(audio_emb_t, text_emb_t, dim=1, eps=1e-8)
    return (
        audio_emb_t.numpy(),
        text_emb_t.numpy(),
        sim_t.numpy(),
    )


def _compute_passt(
    passt_ml_name: str,
    models_by_name: dict[str, Any],
    *,
    audio_paths: list[str],
    use_cache: bool,
) -> np.ndarray:
    if passt_ml_name not in models_by_name:
        raise ValueError(f"Unknown PaSST model '{passt_ml_name}'. Available: {sorted(models_by_name.keys())}")

    passt_ml = models_by_name[passt_ml_name]
    loader = EmbeddingLoader(model=passt_ml, load_model=True)

    vecs: list[np.ndarray] = []
    for p in tqdm(audio_paths, desc=f"Computing PaSST embeddings ({passt_ml_name})"):
        emb = loader.get_or_compute_embedding(Path(p), use_cache=use_cache)
        vecs.append(_reduce_to_vector(emb))

    return np.stack(vecs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PaSST + CLAP embeddings for EDA.")
    parser.add_argument("--data_csv_path", required=True, help="Dataset CSV containing at least id, wav_path.")
    parser.add_argument("--caption_csv_path", help="Caption CSV containing at least id, caption.")
    parser.add_argument("--wav_dir", default="data/wav", help="Base directory containing wav files.")
    parser.add_argument("--output_dir", required=True, help="Directory to write EDA artifacts into.")
    parser.add_argument("--split_name", help="If provided, sets/overrides meta['split'] to this value.")
    parser.add_argument("--limit", type=int, help="Optional limit on rows after filtering (for quick iteration).")

    # Filtering (defaults follow fine_tuning.py)
    parser.add_argument("--no_filter_inaudible", action="store_true", help="Disable filtering inaudible==True.")
    parser.add_argument("--no_filter_null_name", action="store_true", help="Disable filtering name containing 'NULL'.")
    parser.add_argument("--exclude_fixed_freqs", action="store_true", help="Filter out rows with has_fixed_freqs==True.")
    parser.add_argument(
        "--missing_audio_policy",
        choices=["error", "skip"],
        default="error",
        help="How to handle rows whose wav files are missing.",
    )

    # Embeddings
    parser.add_argument("--skip_clap", action="store_true", help="Skip CLAP embedding/score extraction.")
    parser.add_argument("--skip_passt", action="store_true", help="Skip PaSST embedding extraction.")
    parser.add_argument(
        "--clap_model_name",
        default="music_audioset_epoch_15_esc_90.14.pt",
        help="LAION-CLAP checkpoint name (default: music).",
    )
    parser.add_argument("--passt_model", default="passt-base-10s", help="PaSST model name to use.")
    parser.add_argument("--clap_batch_size", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true", help="Cache per-audio embeddings under wav folders (kadtk convention).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.wav_dir)

    # --- Load metadata ---
    data_keep = ["id", "wav_path", "name", "inaudible", "has_fixed_freqs", "split"]
    header = pd.read_csv(args.data_csv_path, nrows=0).columns.tolist()
    data_keep_present = [c for c in data_keep if c in header]
    df = _read_csv_selected(Path(args.data_csv_path), data_keep_present)

    if "id" not in df.columns or "wav_path" not in df.columns:
        raise ValueError("Dataset CSV must contain 'id' and 'wav_path' columns.")

    # Normalize types
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    df["wav_path"] = df["wav_path"].astype(str)
    if "name" in df.columns:
        df["name"] = df["name"].astype(str)

    if args.split_name:
        df["split"] = args.split_name

    df = _apply_base_filters(
        df,
        filter_inaudible=not args.no_filter_inaudible,
        filter_null_name=not args.no_filter_null_name,
        exclude_fixed_freqs=args.exclude_fixed_freqs,
    )

    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    # Captions (optional but required for CLAP score)
    if not args.skip_clap:
        if not args.caption_csv_path:
            raise ValueError("--caption_csv_path is required unless --skip_clap is set.")
        cap_header = pd.read_csv(args.caption_csv_path, nrows=0).columns.tolist()
        if "id" not in cap_header or "caption" not in cap_header:
            raise ValueError("Caption CSV must contain 'id' and 'caption' columns.")
        cap_df = _read_csv_selected(Path(args.caption_csv_path), ["id", "caption"])
        cap_df["id"] = pd.to_numeric(cap_df["id"], errors="raise").astype(int)
        cap_df["caption"] = cap_df["caption"].astype(str)
        df = pd.merge(df, cap_df, on="id", how="inner")
        if df.empty:
            raise ValueError("After merging with captions, no rows remained. Check that ids match.")

    # Audio existence
    df, abs_audio_paths = _ensure_audio_exists(
        df,
        wav_dir=wav_dir,
        missing_audio_policy=args.missing_audio_policy,
        output_dir=output_dir,
    )

    # Save meta first (row order is the canonical order for .npy arrays)
    meta_out = output_dir / "meta.csv"
    df.to_csv(meta_out, index=False)
    print(f"Wrote metadata: {meta_out} ({len(df)} rows)")

    # --- Model registry ---
    models_by_name = {m.name: m for m in get_all_models(audio_len=None)}

    # --- CLAP embeddings + score ---
    if not args.skip_clap:
        print(f"Computing CLAP embeddings with '{args.clap_model_name}' on device='{args.device}' ...")
        audio_emb, text_emb, sim = _compute_clap(
            args.clap_model_name,
            captions=df["caption"].tolist(),
            audio_paths=abs_audio_paths,
            device=args.device,
            batch_size=args.clap_batch_size,
        )

        _save_numpy(output_dir / "clap_audio_emb.npy", audio_emb.astype(np.float16))
        _save_numpy(output_dir / "clap_text_emb.npy", text_emb.astype(np.float16))
        _save_numpy(output_dir / "clap_score.npy", sim.astype(np.float32))
        print(f"Wrote CLAP artifacts to: {output_dir}")

        del audio_emb, text_emb, sim
        _cleanup_torch()

    # --- PaSST embeddings ---
    if not args.skip_passt:
        print(f"Computing PaSST embeddings with '{args.passt_model}' on device='{args.device}' ...")
        passt = _compute_passt(
            args.passt_model,
            models_by_name,
            audio_paths=abs_audio_paths,
            use_cache=args.use_cache,
        )
        _save_numpy(output_dir / "passt_emb.npy", passt.astype(np.float16))
        print(f"Wrote PaSST artifacts to: {output_dir}")

        del passt
        _cleanup_torch()

    # --- Config ---
    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Wrote config: {config_path}")


if __name__ == "__main__":
    main()


