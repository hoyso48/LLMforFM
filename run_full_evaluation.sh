#!/usr/bin/env bash
set -eo pipefail

# One-shot end-to-end evaluation runner for DX7_YAMAHA_test (100 samples):
# - Render predicted wavs for every artifacts/*.csv (into outputs/pred_wav/<name>/...)
# - Run KADTK metrics with vggish and passt-base-10s (KL/FAD/KAD)
# - Run timbral evaluation (AudioCommons timbral_models + RMS)
# - Run CLAP score evaluation (LAION-CLAP fusion-best) via evaluate.py
#
# This script is designed to be run non-interactively and without relying on `conda activate`.
# It uses `conda run -n <ENV>` for every python invocation.
#
# Usage:
#   bash LLMforFM/run_full_evaluation.sh
#   # or from repo root:
#   bash run_full_evaluation.sh
#
# Optional env vars:
#   ENV_NAME=llmforfm_eval
#   N_JOBS=16
#   BATCH_SIZE=32

ENV_NAME="${ENV_NAME:-llmforfm_eval}"

N_JOBS="${N_JOBS:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"
BASELINE_CSV="$PROJECT_ROOT/data/DX7_YAMAHA_test.csv"
BASELINE_WAV_DIR="$PROJECT_ROOT/data/wav"
CAPTION_CSV="$PROJECT_ROOT/data/DX7_YAMAHA_test_captions.csv"

PRED_WAV_ROOT="$PROJECT_ROOT/outputs/pred_wav"
RESULTS_ROOT="$PROJECT_ROOT/outputs/evaluation"
LOG_DIR="$RESULTS_ROOT/logs"

mkdir -p "$PRED_WAV_ROOT" "$RESULTS_ROOT" "$LOG_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[FATAL] conda not found on PATH. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

activate_env() {
  local conda_base conda_sh
  conda_base="$(conda info --base)"
  conda_sh="$conda_base/etc/profile.d/conda.sh"
  if [[ ! -f "$conda_sh" ]]; then
    echo "[FATAL] Could not find conda.sh at: $conda_sh" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$conda_sh"

  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[FATAL] conda env '$ENV_NAME' not found." >&2
    echo "Run: bash setup_eval_env.sh" >&2
    exit 1
  fi

  conda activate "$ENV_NAME"
}

check_imports() {
  python -c "import torch, torchaudio, timbral_models, laion_clap; print('imports_ok')"
}

detect_device() {
  if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "cuda"
  else
    echo "cpu"
  fi
}

activate_env
check_imports

# Ensure in-repo imports (e.g., `import kadtk`) work regardless of where this script is invoked from.
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

DEVICE="$(detect_device)"
echo "[INFO] Using device: $DEVICE"

if [[ ! -d "$ARTIFACTS_DIR" ]]; then
  echo "[FATAL] artifacts dir not found: $ARTIFACTS_DIR" >&2
  exit 1
fi

shopt -s nullglob
ARTIFACT_CSVS=("$ARTIFACTS_DIR"/*.csv)
if [[ "${#ARTIFACT_CSVS[@]}" -eq 0 ]]; then
  echo "[FATAL] No CSVs found in $ARTIFACTS_DIR" >&2
  exit 1
fi

if [[ ! -f "$BASELINE_CSV" ]]; then
  echo "[FATAL] Baseline CSV not found: $BASELINE_CSV" >&2
  exit 1
fi
if [[ ! -d "$BASELINE_WAV_DIR" ]]; then
  echo "[FATAL] Baseline wav dir not found: $BASELINE_WAV_DIR" >&2
  exit 1
fi
if [[ ! -f "$CAPTION_CSV" ]]; then
  echo "[FATAL] Caption CSV not found: $CAPTION_CSV" >&2
  exit 1
fi

echo "[INFO] Found ${#ARTIFACT_CSVS[@]} artifact CSV(s)."

for CSV in "${ARTIFACT_CSVS[@]}"; do
  NAME="$(basename "$CSV" .csv)"
  OUT_DIR="$RESULTS_ROOT/$NAME"
  PRED_WAV_DIR="$PRED_WAV_ROOT/$NAME"
  mkdir -p "$OUT_DIR" "$PRED_WAV_DIR"

  echo
  echo "==================== $NAME ===================="
  echo "[INFO] CSV: $CSV"
  echo "[INFO] Pred wav dir: $PRED_WAV_DIR"
  echo "[INFO] Out dir: $OUT_DIR"

  # 1) Render predicted wavs (use the repo's canonical renderer)
  echo "[STEP] Rendering predicted wavs (dx7/patch_to_wav.py) ..."
  python "$PROJECT_ROOT/dx7/patch_to_wav.py" \
    --input_path "$CSV" \
    --wav_dir "$PRED_WAV_DIR" \
    --sr 48000 --n 60 --velocity 100 --out_scale 1.0 \
    --jobs -1

  # 2) KADTK vggish
  echo "[STEP] KADTK (vggish): KL/FAD/KAD ..."
  python -m kadtk.evaluate \
    --eval_csv_path "$CSV" \
    --eval_wav_dir "$PRED_WAV_DIR" \
    --output_csv_path "$OUT_DIR/${NAME}_kadtk_vggish.csv" \
    --baseline_csv_path "$BASELINE_CSV" \
    --baseline_wav_dir "$BASELINE_WAV_DIR" \
    --metrics kl fad kad \
    --kadtk_model vggish \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/${NAME}_kadtk_vggish.log"

  # 3) KADTK passt
  echo "[STEP] KADTK (passt-base-10s): KL/FAD/KAD ..."
  python -m kadtk.evaluate \
    --eval_csv_path "$CSV" \
    --eval_wav_dir "$PRED_WAV_DIR" \
    --output_csv_path "$OUT_DIR/${NAME}_kadtk_passt.csv" \
    --baseline_csv_path "$BASELINE_CSV" \
    --baseline_wav_dir "$BASELINE_WAV_DIR" \
    --metrics kl fad kad \
    --kadtk_model passt-base-10s \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/${NAME}_kadtk_passt.log"

  # 4) Timbral evaluation (avoid mutating original CSVs by using copies)
  echo "[STEP] Timbral evaluation (timbral_models + RMS) ..."
  EVAL_CSV_TIMBRAL="$OUT_DIR/${NAME}_eval_for_timbral.csv"
  BASELINE_CSV_TIMBRAL="$OUT_DIR/baseline_for_timbral.csv"
  cp -f "$CSV" "$EVAL_CSV_TIMBRAL"
  cp -f "$BASELINE_CSV" "$BASELINE_CSV_TIMBRAL"
  python "$PROJECT_ROOT/evaluate_timbral.py" \
    --eval_csv_path "$EVAL_CSV_TIMBRAL" \
    --eval_wav_dir "$PRED_WAV_DIR" \
    --baseline_csv_path "$BASELINE_CSV_TIMBRAL" \
    --baseline_wav_dir "$BASELINE_WAV_DIR" \
    --output_csv_path "$OUT_DIR/${NAME}_timbral.csv" \
    --n_jobs "$N_JOBS" \
    2>&1 | tee "$LOG_DIR/${NAME}_timbral.log"

  # 5) CLAP score evaluation (fusion-best) via evaluate.py
  echo "[STEP] CLAP evaluation (fusion-best) ..."
  python "$PROJECT_ROOT/evaluate.py" \
    --data_csv_path "$CSV" \
    --caption_csv_path "$CAPTION_CSV" \
    --output_csv_path "$OUT_DIR/${NAME}_clap.csv" \
    --wav_dir "$PRED_WAV_DIR" \
    --batch_size "$BATCH_SIZE" \
    --metrics clap \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/${NAME}_clap.log"

  echo "[OK] Done: $NAME"
done

echo
echo "[DONE] All evaluations complete."
echo "[INFO] Results root: $RESULTS_ROOT"


