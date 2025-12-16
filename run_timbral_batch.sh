#!/usr/bin/env bash

set -u -o pipefail

export CUDA_VISIBLE_DEVICES=1

BASE="/home/hoyeol/projects/GCT634_final"
PY_TIMBRAL="$BASE/evaluate_timbral.py"
BASELINE_CSV="$BASE/data/DX7_YAMAHA_test.csv"
BASELINE_WAV_DIR="$BASE/data/wav"

RESULTS_DIR="$BASE/results_timbral"
LOG_DIR="$BASE/logs"
LOG_FILE="$LOG_DIR/timbral_summaries.txt"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
# Reset log for this batch run
rm -f "$LOG_FILE" && touch "$LOG_FILE"

# name|eval_csv|eval_wav_dir
DATASETS=(
	"audiogen|data/audiogen_outputs.csv|data/wav/audiogen"
	# "gemini20_flash|data/gemini20_flash_results.csv|data/wav/gemini20_flash"
	# "gemini25_flash2|data/gemini25_flash_results2.csv|data/wav/gemini25_flash2"
	# "gemini25_flash_fewshot|data/gemini25_flash_results_fewshot.csv|data/wav/gemini25_flash_fewshot"
	# "gemini25_flash_cot|data/gemini25_flash_results_cot.csv|data/wav/gemini25_flash_cot"
	# "gemini25_pro|data/gemini25_pro_results.csv|data/wav/gemini25_pro"
	# "llama3|data/llama3_results.csv|data/wav/llama3"
	# "gemma2|data/gemma2_results.csv|data/wav/gemma2"
	# "qwen|data/qwen_results.csv|data/wav/qwen"
	# "audioldm|data/audioldm_outputs.csv|data/wav/audioldm"
	# "qwen3_8b_finetuned|data/qwen3_8b_finetuned.csv|data/wav/qwen3_8b_finetuned/Qwen3_8B-fp8-filtered_full-tune_temp1.0_topP0.95_topK20"
)

echo "==== TIMBRAL BATCH START $(date) ====" | tee -a "$LOG_FILE"
for ITEM in "${DATASETS[@]}"; do
	IFS='|' read -r NAME EVAL_CSV REL_EVAL_WAV_DIR <<< "$ITEM"
	EVAL_CSV_ABS="$BASE/$EVAL_CSV"
	EVAL_WAV_DIR_ABS="$BASE/$REL_EVAL_WAV_DIR"
	OUT_CSV="$RESULTS_DIR/${NAME}_timbral.csv"

	HEADER="\n---- [$NAME] $(date) ----"
	echo -e "$HEADER" | tee -a "$LOG_FILE"
	CMD="python \"$PY_TIMBRAL\" \\
	  --eval_csv_path \"$EVAL_CSV_ABS\" \\
	  --eval_wav_dir \"$EVAL_WAV_DIR_ABS\" \\
	  --baseline_csv_path \"$BASELINE_CSV\" \\
	  --baseline_wav_dir \"$BASELINE_WAV_DIR\" \\
	  --output_csv_path \"$OUT_CSV\" \\
	  --n_jobs 16"
	# Do not echo full command to log; only append tail of output below

	TMP_OUT="$(mktemp)"
	# Use a pseudo-TTY so tqdm renders; capture full output to TMP_OUT
	if ! script -q -c "$CMD" "$TMP_OUT"; then
		echo "[ERROR] $NAME failed" | tee -a "$LOG_FILE"
	fi
	# Append only the last 20 lines (df.describe and tail of run) to the batch log
	tail -n 20 "$TMP_OUT" >> "$LOG_FILE" || true
	rm -f "$TMP_OUT"
done

echo -e "==== TIMBRAL BATCH END $(date) ====\n" | tee -a "$LOG_FILE" 