import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import librosa
import timbral_models
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed


TIMBRAL_KEYS: List[str] = [
	'hardness',
	'depth',
	'brightness',
	'roughness',
	'warmth',
	'sharpness',
	'boominess',
	'reverb',
]

def _ensure_timbral_models_librosa_compat() -> None:
	"""
	timbral_models (AudioCommons) is unmaintained and uses older librosa call patterns.

	Example failure on newer librosa:
	  TypeError: onset_detect() takes 0 positional arguments but 2 positional arguments ...

	This shim patches a small set of librosa functions that timbral_models calls
	positionally (e.g., onset_detect(y, sr), onset_strength(y, sr), resample(y, orig_sr, target_sr)).
	Newer librosa makes many of these keyword-only, which breaks timbral_models.
	"""
	try:
		import inspect

		def _wrap_kw_only(func, *, positional_kw_names: List[str], tag: str):
			# Avoid double-patching
			if getattr(func, "_llmforfm_patched", False):
				return func

			sig = inspect.signature(func)
			has_kw_only = any(p.kind == inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())
			if not has_kw_only:
				return func

			_orig = func

			def _compat(*args, **kwargs):
				if len(args) > len(positional_kw_names):
					raise TypeError(f"{tag} legacy wrapper received too many positional args: {len(args)}")
				for i, name in enumerate(positional_kw_names):
					if i < len(args) and name not in kwargs:
						kwargs[name] = args[i]
				return _orig(**kwargs)

			_compat._llmforfm_patched = True  # type: ignore[attr-defined]
			return _compat

		# Patch only what timbral_models uses positionally (see timbral_models/timbral_util.py).
		librosa.onset.onset_detect = _wrap_kw_only(librosa.onset.onset_detect, positional_kw_names=["y", "sr"], tag="onset_detect")  # type: ignore[assignment]
		librosa.onset.onset_strength = _wrap_kw_only(librosa.onset.onset_strength, positional_kw_names=["y", "sr"], tag="onset_strength")  # type: ignore[assignment]
		librosa.core.resample = _wrap_kw_only(librosa.core.resample, positional_kw_names=["y", "orig_sr", "target_sr"], tag="core.resample")  # type: ignore[assignment]
		librosa.core.stft = _wrap_kw_only(librosa.core.stft, positional_kw_names=["y"], tag="core.stft")  # type: ignore[assignment]
		librosa.decompose.hpss = _wrap_kw_only(librosa.decompose.hpss, positional_kw_names=["S"], tag="decompose.hpss")  # type: ignore[assignment]
		librosa.core.istft = _wrap_kw_only(librosa.core.istft, positional_kw_names=["stft_matrix"], tag="core.istft")  # type: ignore[assignment]
	except Exception:
		# If patching fails, keep original behavior; downstream will raise and we will surface warnings.
		return


def resolve_path(root_dir: str, path: str) -> str:
	if os.path.isabs(path) or root_dir is None or len(root_dir) == 0:
		return path
	return os.path.join(root_dir, path)


def safe_cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
	x_norm = np.linalg.norm(x)
	y_norm = np.linalg.norm(y)
	denom = (x_norm * y_norm) + eps
	return float(np.dot(x, y) / denom)


def _run_timbral_extractor(path: str) -> Dict[str, float] | None:
	_ensure_timbral_models_librosa_compat()
	try:
		return timbral_models.timbral_extractor(path, clip_output=True)
	except TypeError:
		# Older versions may not support clip_output
		return timbral_models.timbral_extractor(path)
	except Exception:
		return None


def extract_timbral_vector(path: str) -> np.ndarray:
	try:
		if not os.path.isfile(path):
			print(f"WARNING: audio file not found for timbral extraction: {path}")
			return np.asarray([np.nan] * len(TIMBRAL_KEYS), dtype=float)
		features = _run_timbral_extractor(path)
		if features is None or not isinstance(features, dict) or len(features) == 0:
			print(f"WARNING: timbral extractor returned no features for: {path}")
			return np.asarray([np.nan] * len(TIMBRAL_KEYS), dtype=float)
		# Case-insensitive lookup
		feat_lc = {str(k).lower(): v for k, v in features.items()}
		vector_list: List[float] = []
		for key in TIMBRAL_KEYS:
			val = feat_lc.get(key.lower(), np.nan)
			try:
				vector_list.append(float(val))
			except Exception:
				vector_list.append(np.nan)
		vec = np.asarray(vector_list, dtype=float)
		if np.isnan(vec).all():
			print(f"WARNING: all-NaN timbral features for: {path}")
		return vec
	except Exception:
		return np.asarray([np.nan] * len(TIMBRAL_KEYS), dtype=float)


def extract_rms_series(path: str, frame_length: int, hop_length: int) -> np.ndarray:
	try:
		y, sr = librosa.load(path, sr=44100, mono=True)
		rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
		return rms.flatten().astype(float)
	except Exception:
		return np.asarray([], dtype=float)


def extract_features_for_path(path: str, frame_length: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
	timbral_vec = extract_timbral_vector(path)
	rms_series = extract_rms_series(path, frame_length=frame_length, hop_length=hop_length)
	return timbral_vec, rms_series


def ensure_timbral_cache(df: pd.DataFrame, csv_path: str, wav_dir: str, n_jobs: int) -> pd.DataFrame:
	# Ensure columns exist
	for key in TIMBRAL_KEYS:
		if key not in df.columns:
			df[key] = np.nan

	# Coerce to numeric to handle string placeholders like '' or 'nan'
	try:
		df[TIMBRAL_KEYS] = df[TIMBRAL_KEYS].apply(pd.to_numeric, errors='coerce')
	except Exception:
		pass

	# Determine which rows need computation (only rows with ALL timbral keys NaN)
	need_idx: List[int] = []
	need_paths: List[str] = []
	for i, row in df.iterrows():
		if all(pd.isna(row.get(k)) for k in TIMBRAL_KEYS):
			need_idx.append(i)
			need_paths.append(resolve_path(wav_dir, str(row['wav_path'])))

	# If cache columns exist but all rows are NaN, force recompute everything
	all_nan_cache = False
	try:
		all_nan_cache = df[TIMBRAL_KEYS].isna().to_numpy().all()
	except Exception:
		all_nan_cache = False
	if all_nan_cache and len(df) > 0:
		print(f"NOTE: timbral cache columns exist in {csv_path} but all values are NaN. Forcing full recompute.")
		need_idx = list(range(len(df)))
		need_paths = [resolve_path(wav_dir, str(p)) for p in df['wav_path'].astype(str).tolist()]

	# Compute missing feature rows (threads to avoid multiprocess import issues)
	if len(need_paths) > 0:
		results = Parallel(n_jobs=n_jobs, prefer='threads')(
			delayed(extract_timbral_vector)(p) for p in tqdm(need_paths, desc='Caching timbral features', dynamic_ncols=True, mininterval=0.2)
		)
		for idx, vec in zip(need_idx, results):
			for key, val in zip(TIMBRAL_KEYS, vec.tolist()):
				df.at[idx, key] = float(val) if val is not None else np.nan

		# Write back to the CSV to persist cache
		Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(csv_path, index=False)
		print(f"Cached timbral features for {len(need_idx)} rows -> {csv_path}")

	return df


def evaluate_pairs(
	df_pairs: pd.DataFrame,
	wav_dir_a: str = None,
	wav_dir_b: str = None,
	n_jobs: int = 1,
	frame_length: int = 2048,
	hop_length: int = 512,
	path_to_timbral_raw: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
	required_columns = ['id', 'wav_path_a', 'wav_path_b']
	for col in required_columns:
		if col not in df_pairs.columns:
			raise ValueError(f"Required columns {required_columns} not found in the input CSV")

	df_pairs = df_pairs.copy()
	df_pairs['wav_path_a'] = df_pairs['wav_path_a'].astype(str)
	df_pairs['wav_path_b'] = df_pairs['wav_path_b'].astype(str)

	paths_a: List[str] = [resolve_path(wav_dir_a, p) for p in df_pairs['wav_path_a'].tolist()]
	paths_b: List[str] = [resolve_path(wav_dir_b, p) for p in df_pairs['wav_path_b'].tolist()]

	# Collect unique paths to avoid redundant computation
	unique_paths: List[str] = sorted(set(paths_a + paths_b))

	# Prepare timbral (RAW) features mapping. If not provided, compute here.
	if path_to_timbral_raw is None:
		raw_results = Parallel(n_jobs=n_jobs, prefer='threads')(
			delayed(extract_timbral_vector)(p) for p in tqdm(unique_paths, desc='Extracting timbral (raw)', dynamic_ncols=True, mininterval=0.2)
		)
		path_to_timbral_raw = {p: v for p, v in zip(unique_paths, raw_results)}

	# RMS feature extraction is not cached (lighter-weight)
	rms_results = Parallel(n_jobs=n_jobs, prefer='threads')(
		delayed(extract_rms_series)(p, frame_length, hop_length) for p in tqdm(unique_paths, desc='Extracting RMS', dynamic_ncols=True, mininterval=0.2)
	)
	path_to_rms: Dict[str, np.ndarray] = {p: r for p, r in zip(unique_paths, rms_results)}

	# Compute metrics per pair
	timbral_mses: List[float] = []
	timbral_l1s: List[float] = []
	timbral_cosines: List[float] = []
	rms_cosines: List[float] = []

	# Collect raw features (pre-normalized) to save in output
	raw_a_cols: Dict[str, List[float]] = {f"timbral_raw_a_{k}": [] for k in TIMBRAL_KEYS}
	raw_b_cols: Dict[str, List[float]] = {f"timbral_raw_b_{k}": [] for k in TIMBRAL_KEYS}

	for a_path, b_path in tqdm(list(zip(paths_a, paths_b)), total=len(paths_a), desc='Computing metrics', dynamic_ncols=True, mininterval=0.2):
		a_timbral_raw = path_to_timbral_raw.get(a_path, np.asarray([np.nan] * len(TIMBRAL_KEYS), dtype=float))
		b_timbral_raw = path_to_timbral_raw.get(b_path, np.asarray([np.nan] * len(TIMBRAL_KEYS), dtype=float))

		# Record RAW (pre-normalized) values per key
		for key_index, key in enumerate(TIMBRAL_KEYS):
			raw_a_cols[f"timbral_raw_a_{key}"].append(float(a_timbral_raw[key_index]) if key_index < len(a_timbral_raw) else np.nan)
			raw_b_cols[f"timbral_raw_b_{key}"].append(float(b_timbral_raw[key_index]) if key_index < len(b_timbral_raw) else np.nan)

		# Normalize: divide by 100 for all keys except 'reverb' (which is already 0/1)
		a_timbral = np.asarray([
			(a_timbral_raw[i] if k == 'reverb' else a_timbral_raw[i] / 100.0)
			for i, k in enumerate(TIMBRAL_KEYS)
		], dtype=float)
		b_timbral = np.asarray([
			(b_timbral_raw[i] if k == 'reverb' else b_timbral_raw[i] / 100.0)
			for i, k in enumerate(TIMBRAL_KEYS)
		], dtype=float)

		# Replace NaNs with 0 in normalized vectors
		a_timbral = np.nan_to_num(a_timbral, nan=0.0)
		b_timbral = np.nan_to_num(b_timbral, nan=0.0)

		# Timbral vector distances/similarities
		diff = a_timbral - b_timbral
		timbral_mses.append(float(np.mean(diff ** 2)))
		timbral_l1s.append(float(np.mean(np.abs(diff))))
		timbral_cosines.append(safe_cosine_similarity(a_timbral, b_timbral))

		# RMS cosine similarity
		a_rms = path_to_rms.get(a_path, np.asarray([], dtype=float))
		b_rms = path_to_rms.get(b_path, np.asarray([], dtype=float))
		L = int(min(len(a_rms), len(b_rms)))
		if L <= 0:
			rms_cosines.append(np.nan)
		else:
			rms_cosines.append(safe_cosine_similarity(a_rms[:L], b_rms[:L]))

	result = pd.DataFrame(
		{
			'id': df_pairs['id'].tolist(),
			'wav_path_a': paths_a,
			'wav_path_b': paths_b,
			'timbral_mse': timbral_mses,
			'timbral_l1': timbral_l1s,
			'timbral_cosine': timbral_cosines,
			'rms_cosine': rms_cosines,
		}
	)

	# Attach RAW timbral columns
	for col_name, values in raw_a_cols.items():
		result[col_name] = values
	for col_name, values in raw_b_cols.items():
		result[col_name] = values

	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Evaluate pairs of audio files using timbral features and RMS energy similarity (compatible with kadtk/evaluate.py CSVs).')
	# Follow kadtk/evaluate.py-style args
	parser.add_argument('--eval_csv_path', type=str, required=True, help="CSV with 'id' and 'wav_path' for evaluation files.")
	parser.add_argument('--eval_wav_dir', type=str, required=True, help='Directory with evaluation audio files.')
	parser.add_argument('--baseline_csv_path', type=str, required=True, help="CSV with 'id' and 'wav_path' for baseline files.")
	parser.add_argument('--baseline_wav_dir', type=str, required=True, help='Directory with baseline audio files.')
	parser.add_argument('--output_csv_path', type=str, required=True, help='Where to save the results CSV')
	parser.add_argument('--n_jobs', type=int, default=1, help='Parallel jobs for feature extraction using joblib')
	parser.add_argument('--frame_length', type=int, default=2048, help='Frame length for RMS computation')
	parser.add_argument('--hop_length', type=int, default=512, help='Hop length for RMS computation')

	args = parser.parse_args()

	# Load CSVs (must contain id,wav_path)
	df_eval = pd.read_csv(args.eval_csv_path)
	df_baseline = pd.read_csv(args.baseline_csv_path)
	for df_name, df_obj in [('eval', df_eval), ('baseline', df_baseline)]:
		if not all(col in df_obj.columns for col in ['id', 'wav_path']):
			raise ValueError(f"{df_name} CSV must contain ['id', 'wav_path'] columns.")
		df_obj[['id', 'wav_path']]

	# Ensure timbral feature cache in the input CSVs (raw values)
	df_eval = ensure_timbral_cache(df_eval, args.eval_csv_path, args.eval_wav_dir, args.n_jobs)
	df_baseline = ensure_timbral_cache(df_baseline, args.baseline_csv_path, args.baseline_wav_dir, args.n_jobs)

	# Warn if baseline has any NaNs in cached timbral features
	if any(k in df_baseline.columns for k in TIMBRAL_KEYS):
		mask = df_baseline[TIMBRAL_KEYS].isna().any(axis=1)
		n_nan = int(mask.sum())
		if n_nan > 0:
			ids_with_nan = df_baseline.loc[mask, 'id'].tolist()[:10]
			print(f"WARNING: baseline CSV has {n_nan} rows with NaN timbral features. Example IDs: {ids_with_nan}")

	# Merge on id to make pairs
	df_merged = pd.merge(df_eval[['id', 'wav_path']], df_baseline[['id', 'wav_path']], on='id', how='inner', suffixes=('_eval', '_baseline'))
	if len(df_merged) == 0:
		raise ValueError('No overlapping ids between eval and baseline CSVs.')

	df_pairs = pd.DataFrame(
		{
			'id': df_merged['id'].tolist(),
			'wav_path_a': df_merged['wav_path_eval'].tolist(),
			'wav_path_b': df_merged['wav_path_baseline'].tolist(),
		}
	)

	# Build RAW timbral mapping from cached columns
	path_to_timbral_raw: Dict[str, np.ndarray] = {}
	for src_df, root_dir in [(df_eval, args.eval_wav_dir), (df_baseline, args.baseline_wav_dir)]:
		for _, row in src_df.iterrows():
			p = resolve_path(root_dir, str(row['wav_path']))
			vec = [float(row.get(k)) if k in row and not pd.isna(row.get(k)) else np.nan for k in TIMBRAL_KEYS]
			path_to_timbral_raw[p] = np.asarray(vec, dtype=float)

	results_df = evaluate_pairs(
		df_pairs,
		wav_dir_a=args.eval_wav_dir,
		wav_dir_b=args.baseline_wav_dir,
		n_jobs=args.n_jobs,
		frame_length=args.frame_length,
		hop_length=args.hop_length,
		path_to_timbral_raw=path_to_timbral_raw,
	)

	# Save results
	output_path = Path(args.output_csv_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_csv(output_path, index=False)

	# Quick summary
	print(results_df[['timbral_mse', 'timbral_l1', 'timbral_cosine', 'rms_cosine']].describe())
	print(f"Saved timbral results to {output_path}") 