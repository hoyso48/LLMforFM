"""
Audio evaluation utilities for DX7 dataset generations.

This script computes:
- CLAP score (LAION-CLAP, default: music)
- Optional: audiobox aesthetics (if installed and requested)

Input CSV requirements:
- --data_csv_path   : columns ['id', 'wav_path']
- --caption_csv_path: columns ['id', 'caption']
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

from clap_score import compute_clap_score, load_clap_model


def _read_csv_drop_unnamed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def _require_columns(df: pd.DataFrame, cols: Iterable[str], *, path: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def evaluate(
    *,
    df_audio: pd.DataFrame,
    df_captions: pd.DataFrame,
    wav_dir: str,
    metrics: list[str],
    batch_size: int,
    device: str,
    clap_model_name: str,
) -> pd.DataFrame:
    df = pd.merge(df_audio, df_captions, on="id", how="inner")
    wav_paths = [os.path.join(wav_dir, wav_path) for wav_path in df["wav_path"].tolist()]

    out = pd.DataFrame(
        {
            "id": df["id"].tolist(),
            "wav_path": df["wav_path"].tolist(),
            "caption": df["caption"].tolist(),
        }
    )

    if "clap" in metrics:
        clap_model = load_clap_model(clap_model_name=clap_model_name, device=device)
        out["clap_score"] = compute_clap_score(
            clap_model,
            df["caption"].tolist(),
            wav_paths,
            batch_size=batch_size,
            device=device,
        )
        out["clap_score_synthesized"] = compute_clap_score(
            clap_model,
            ["Synthesized " + c for c in df["caption"].tolist()],
            wav_paths,
            batch_size=batch_size,
            device=device,
        )

        del clap_model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "audiobox" in metrics:
        try:
            from audiobox_aesthetics.infer import initialize_predictor
        except ImportError as exc:
            raise ImportError(
                "audiobox_aesthetics is not installed but audiobox metric was requested. "
                "Install it or run with --metrics clap."
            ) from exc

        audiobox_model = initialize_predictor()
        audiobox_scores = []
        for i in tqdm(range(0, len(wav_paths), batch_size), desc="Audiobox scoring", dynamic_ncols=True):
            audiobox_scores.extend(audiobox_model.forward([{"path": p} for p in wav_paths[i : i + batch_size]]))

        out["audiobox_ce"] = [score["CE"] for score in audiobox_scores]
        out["audiobox_cu"] = [score["CU"] for score in audiobox_scores]
        out["audiobox_pc"] = [score["PC"] for score in audiobox_scores]
        out["audiobox_pq"] = [score["PQ"] for score in audiobox_scores]

        del audiobox_model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv_path", type=str, required=True)
    parser.add_argument("--caption_csv_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, default="data/wav")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        choices=["clap", "audiobox"],
        help="Metrics to compute. Default: clap (and audiobox if installed).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for CLAP scoring (cuda/cpu).",
    )
    parser.add_argument(
        "--clap_model_name",
        type=str,
        default="music_audioset_epoch_15_esc_90.14.pt",
        help="LAION-CLAP checkpoint name (see clap_score.py).",
    )

    args = parser.parse_args()

    df = _read_csv_drop_unnamed(args.data_csv_path)
    df_captions = _read_csv_drop_unnamed(args.caption_csv_path)

    _require_columns(df, ["id", "wav_path"], path=args.data_csv_path)
    _require_columns(df_captions, ["id", "caption"], path=args.caption_csv_path)

    df = df[["id", "wav_path"]]
    df_captions = df_captions[["id", "caption"]]

    if args.metrics is None:
        metrics = ["clap"]
        try:
            import audiobox_aesthetics  # noqa: F401
        except Exception:
            pass
        else:
            metrics.append("audiobox")
    else:
        metrics = list(args.metrics)

    result = evaluate(
        df_audio=df,
        df_captions=df_captions,
        wav_dir=str(args.wav_dir),
        metrics=metrics,
        batch_size=int(args.batch_size),
        device=str(args.device),
        clap_model_name=str(args.clap_model_name),
    )

    numeric_cols = [c for c in result.columns if c not in ["id", "wav_path", "caption"]]
    if numeric_cols:
        print(result[numeric_cols].describe())
    result.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    main()
