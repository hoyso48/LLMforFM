from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script from anywhere (not only from the repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dx7.utils import render_from_specs
from scipy.io.wavfile import write
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import ast
import json
import pandas as pd


def _parse_patch_data(raw_value, row_id=None):
    if isinstance(raw_value, dict):
        return raw_value
    if raw_value is None:
        raise ValueError(f"Row {row_id}: missing patch_data")
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if candidate == "":
            raise ValueError(f"Row {row_id}: empty patch_data string")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except Exception as exc:
                raise ValueError(f"Row {row_id}: failed to parse patch_data ({exc})") from exc
    raise TypeError(f"Row {row_id}: unsupported patch_data type {type(raw_value)}")


def _load_records(input_path: Path, limit: int | None = None):
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


def process_row(row_data, sr=48000, n=60, v=100, out_scale=1.0, wav_dir="data/wav"):
    row_id = row_data.get("id")
    patch_data_processed = _parse_patch_data(row_data.get("patch_data"), row_id=row_id)

    if "wav_path" not in row_data or not row_data["wav_path"]:
        raise ValueError(f"Row {row_id}: wav_path is missing or empty")

    wav_path = Path(wav_dir) / str(row_data["wav_path"])
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    audio = render_from_specs(patch_data_processed, sr=sr, n=n, v=v, out_scale=out_scale)
    write(wav_path, sr, audio)
    return f"Processed {wav_path}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render DX7 patches to WAV from CSV/JSONL datasets.")
    parser.add_argument(
        "--input_path",
        "--csv_path",
        dest="input_path",
        type=str,
        default="data/DX7_YAMAHA_train.csv",
        help="CSV or JSONL file containing patch_data and wav_path columns.",
    )
    parser.add_argument("--wav_dir", type=str, default="data/wav", help="Base directory for rendered WAV files.")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--v", type=int, default=100)
    parser.add_argument("--velocity", dest="v", type=int, help="Alias for --v (MIDI velocity).")
    parser.add_argument("--out_scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="Optional limit on number of rows to render.")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel workers (joblib).")
    args = parser.parse_args()

    records = _load_records(Path(args.input_path), args.limit)
    print(f"Starting parallel processing for {len(records)} items from {args.input_path}...")

    results = Parallel(n_jobs=args.jobs)(
        delayed(process_row)(
            row_data,
            sr=args.sr,
            n=args.n,
            v=args.v,
            out_scale=args.out_scale,
            wav_dir=args.wav_dir,
        )
        for row_data in tqdm(records, total=len(records), desc="Generating audio")
    )

    print(f"\nFinished processing {len(results)} items.")