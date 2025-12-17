#!/usr/bin/env bash
set -eo pipefail

# Create/repair the evaluation conda environment for LLMforFM.
#
# This script ONLY manages the conda env. It does NOT run evaluation.
#
# Usage:
#   bash setup_eval_env.sh
#
# Optional env vars:
#   ENV_NAME=llmforfm_eval
#   PYTHON_VERSION=3.11
#   TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu126
#   TORCH_VERSION=2.7.0+cu126
#   TORCHAUDIO_VERSION=2.7.0+cu126
#   FORCE_RECREATE=0|1

ENV_NAME="${ENV_NAME:-llmforfm_eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0+cu126}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0+cu126}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[FATAL] conda not found on PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
PKGS_DIR="$CONDA_BASE/pkgs"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

remove_stale_partials() {
  if ls "$PKGS_DIR"/*.conda.partial "$PKGS_DIR"/*.tar.bz2.partial >/dev/null 2>&1; then
    echo "[WARN] Removing stale partial downloads in $PKGS_DIR ..."
    rm -f "$PKGS_DIR"/*.conda.partial "$PKGS_DIR"/*.tar.bz2.partial || true
  fi
}

env_exists_registered() {
  conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

env_core_healthy() {
  # Core: if this fails, the env is genuinely broken/incomplete.
  conda run -n "$ENV_NAME" python -c "import torch, torchaudio; print('core_ok')" >/dev/null 2>&1
}

env_extras_healthy() {
  # Extras needed by the evaluation stack (vggish via torch.hub needs resampy).
  conda run -n "$ENV_NAME" python -c "import timbral_models, laion_clap, librosa, pyloudnorm, sklearn, resampy; print('extras_ok')" >/dev/null 2>&1
}

remove_env_anyway() {
  echo "[INFO] Removing env '$ENV_NAME' (if present) ..."
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
  conda env remove -p "$ENV_PREFIX" -y >/dev/null 2>&1 || true
  rm -rf "$ENV_PREFIX" || true
}

remove_stale_partials

if env_exists_registered; then
  if [[ "$FORCE_RECREATE" == "1" ]]; then
    echo "[INFO] FORCE_RECREATE=1 set."
    remove_env_anyway
  else
    if env_core_healthy; then
      if env_extras_healthy; then
        echo "[OK] conda env '$ENV_NAME' already exists and is healthy."
        exit 0
      fi
      echo "[WARN] conda env '$ENV_NAME' exists, but some evaluation deps are missing. Installing missing deps..."
      # Fall through to dependency install below (no full recreate).
    else
      echo "[WARN] conda env '$ENV_NAME' exists but is broken/incomplete (core imports failed). Recreating it..."
      remove_env_anyway
    fi
  fi
else
  # Not registered, but prefix dir might still exist after an interrupted create.
  if [[ -d "$ENV_PREFIX" ]]; then
    echo "[WARN] Found stale env prefix dir (not registered): $ENV_PREFIX"
    remove_env_anyway
  fi
fi

remove_stale_partials

if ! env_exists_registered; then
  echo "[INFO] Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION) ..."
  conda create -n "$ENV_NAME" -y "python=$PYTHON_VERSION"
fi

echo "[INFO] Upgrading pip ..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip

echo "[INFO] Installing torch + torchaudio (must match ABI) ..."
if ! env_core_healthy; then
  conda run -n "$ENV_NAME" python -m pip install \
    --index-url "$TORCH_CUDA_INDEX_URL" \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
fi

echo "[INFO] Installing evaluation dependencies ..."
# Some third-party deps still incorrectly depend on the deprecated `sklearn` shim package.
# Newer pip versions block it unless this env var is set. We install scikit-learn explicitly
# and allow the shim only if it is pulled as a transitive dependency.
conda run -n "$ENV_NAME" env SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
  python -m pip install \
  numpy pandas scipy \
  librosa soundfile joblib tqdm packaging requests \
  numba einops \
  scikit-learn \
  resampy \
  pyloudnorm \
  timbral_models \
  laion-clap \
  hear21passt \
  hypy-utils

echo "[INFO] Verifying imports ..."
conda run -n "$ENV_NAME" python -c "import torch, torchaudio, timbral_models, laion_clap, librosa, pyloudnorm, sklearn, resampy; print('imports_ok')"

echo
echo "[OK] Environment ready: $ENV_NAME"
echo "Next:"
echo "  cd /home/hoyso/projects/LLMforFM"
echo "  ./run_full_evaluation.sh"


