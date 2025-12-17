## LLMforFM

Research code for learning DX7/FM-synth patch representations with LLMs, plus utilities for embedding extraction and evaluation.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Evaluation environment (recommended)

The full evaluation pipeline depends on heavier audio stacks (CUDA `torch`/`torchaudio`, `timbral_models`, etc.). For reproducibility, this repo provides a dedicated conda env setup script:

```bash
cd /home/hoyso/projects/LLMforFM
bash setup_eval_env.sh
```

### Data + Git LFS

- `data/*.csv` is **tracked via Git LFS** (so the repo stays manageable as datasets grow).
- Large/raw assets are intentionally not versioned in git (e.g. `data/wav/`, `data/dx7_patches/`, `artifacts/`, `outputs/`).

After cloning:

```bash
git lfs install
git lfs pull
```

### Repository layout (what each folder is for)

- **`data/`**: DX7 datasets and (optionally) rendered audio.
  - `DX7_*_train.csv`, `DX7_*_test.csv`: rows contain `id`, `wav_path`, `patch_data` (+ metadata like `name`, `inaudible`, `has_fixed_freqs`).
  - `DX7_*_captions.csv`: `id` → `caption` (optionally `cot` for reasoning supervision).
  - `dx7_patches/`: raw `.syx` patch dumps (sources).
  - `wav/`: rendered audio for each patch (`wav_path` is *relative* to this directory).
- **`dx7/`**: the DX7 “tool” (patch → audio) and shared parsing/validation utilities.
- **`kadtk/`**: metric toolkit (embeddings + KL/FAD/KAD + CLAP). This repo typically calls it via `python -m kadtk.evaluate ...`.
- **`scripts/`**: dataset/evaluation utilities (CSV normalization, CoT helpers, embedding extraction).
- **`load/`**: cached model checkpoints used by evaluation (e.g., LAION-CLAP `.pt`).
- **`artifacts/`**: human/model artifacts used as *inputs* to evaluation (e.g., model output CSVs, EDA embedding dumps).
- **`outputs/`**: generated outputs (predicted WAVs, evaluation CSVs/logs, CoT examples, CoT JSONL shards, etc.).
- **`references/`**: papers / notebooks used during research.

### Entry points

- **`fine_tuning.py`**: SFT for caption → DX7 patch **JSON** (optionally supervised with `cot` reasoning).
- **`grpo.py`**: RL (GRPO) for DX7 tool learning:
  - `reward_mode=toolrl_exact`: ToolRL-faithful exact matching reward.
  - `reward_mode=dx7_dist`: dense weighted parameter-distance reward (recommended).
- **`inference.py`**: run a fine-tuned Qwen3 checkpoint on caption CSVs, parse/validate the final patch JSON, optionally render WAVs, and write an evaluation-ready CSV.
- **`dx7/patch_to_wav.py`**: render WAVs from `patch_data` (CSV or JSONL) using the repo’s DX7 renderer.
- **`evaluate.py`**: compute CLAP score (and optionally audiobox aesthetics) given captions + rendered audio.
- **`evaluate_timbral.py`**: timbral attribute distances + RMS envelope similarity (includes a librosa-compat shim for `timbral_models`).
- **`run_full_evaluation.sh`**: one-shot end-to-end evaluation runner (normalize → render → KADTK → timbral → CLAP).

### Data contracts (CSV + DX7 JSON schema)

- **DX7 patch representation**
  - Stored as a JSON object (often embedded as a JSON string inside CSV `patch_data`).
  - Canonical top-level keys (shared across SFT/RL/inference/eval):  
    `modmatrix`, `outmatrix`, `feedback`, `fixed_freq`, `coarse`, `fine`, `detune`, `transpose`, `ol`, `eg_rate`, `eg_level`, `sensitivity`
  - Parsing + validation is centralized in `dx7/utils.py`:
    - `parse_last_specs(text)`: extracts the **last** JSON object from a model response (prefers fenced ```json blocks).
    - `validate_specs(specs)`: checks shapes/ranges and DX7 constraints (e.g., at most one feedback diagonal in `modmatrix`).

- **Evaluation CSV schema (what evaluation scripts expect)**
  - `id`: identifier used to join with caption CSVs.
  - `wav_path`: relative path under a wav root (e.g. `data/wav` or `outputs/pred_wav/<run>/`).
  - `patch_data`: patch JSON (string or dict), used by `dx7/patch_to_wav.py`.
  - Use `scripts/normalize_eval_csv.py` to convert “model output CSVs” into this canonical schema.

### Typical workflows

#### 1) Render dataset audio (ground truth)

```bash
python dx7/patch_to_wav.py --input_path data/DX7_YAMAHA_test.csv --wav_dir data/wav --jobs -1
```

#### 2) SFT (fine-tuning)

`fine_tuning.py` builds Qwen3-style supervised targets:
- user prompt ends with `/think` if `cot` is present, else `/no_think`
- assistant output always starts with `<think>...</think>` and then a fenced DX7 patch JSON

Useful sanity checks (no training):

```bash
python fine_tuning.py --print_examples --num_examples 3
python fine_tuning.py --analyze_lengths --tokenizer_name_or_path unsloth/Qwen3-8B
```

#### 3) Inference (caption → patch JSON → optional audio)

```bash
python inference.py \
  --model_path /path/to/your_checkpoint_dir \
  --caption_csv_path data/DX7_YAMAHA_test_captions.csv \
  --output_csv_path artifacts/my_run_outputs.csv
```

To also render predicted audio (writes WAVs under `--wav_dir` and stores relative `wav_path`):

```bash
python inference.py \
  --model_path /path/to/your_checkpoint_dir \
  --caption_csv_path data/DX7_YAMAHA_test_captions.csv \
  --output_csv_path artifacts/my_run_outputs.csv \
  --render_wav --wav_dir outputs/pred_wav/my_run
```

#### 4) End-to-end evaluation (recommended)

This runs:
1) `scripts/normalize_eval_csv.py` (schema normalization)
2) `dx7/patch_to_wav.py` (render predicted wavs)
3) `python -m kadtk.evaluate` (KL/FAD/KAD with vggish + PaSST)
4) `evaluate_timbral.py` (AudioCommons timbral metrics + RMS)
5) `evaluate.py` (LAION-CLAP score)

```bash
cd /home/hoyso/projects/LLMforFM
bash setup_eval_env.sh
./run_full_evaluation.sh
```

You can also evaluate a single CSV:

```bash
./run_full_evaluation.sh /home/hoyso/projects/LLMforFM/artifacts/my_run_outputs.csv
```

### CoT (teacher) data generation (Gemini)

This pipeline supports generating teacher outputs that include:
- a diagnostic analysis section (teacher-only),
- a pseudo reasoning trace inside `\cot{ ... }`,
- and a final DX7 patch JSON that must exactly match ground truth.

Main components:
- **`scripts/render_concat_ablation_audio.py`**: renders a single concatenated audio clip per patch: FULL + OPk_OFF/OPk_ON segments (randomized order), plus a segment map JSON.
- **`generate_cot_reasoning_gemini.py`**: calls Gemini (via `google-genai`) and writes JSONL shards containing CoT + final JSON.
- **`scripts/merge_filter_cot_jsonl.py`**: merges shards, filters errors, optionally enforces “strict” validity and deduplicates.
- **`scripts/add_cot_to_csv_from_jsonl.py`**: writes the extracted CoT into a dataset CSV as a `cot` column.

Gemini auth: set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) or pass `--api_key`.

### Notes / gotchas

- **`evaluate_timbral.py` patches librosa calls at runtime** to keep the unmaintained `timbral_models` compatible with newer librosa versions.
- **`run_timbral_batch.sh` is legacy** (hardcoded paths to an older project layout). Prefer `run_full_evaluation.sh`.

### RL (GRPO) training (recommended one-shot settings)

Activate your env first:

```bash
conda activate hoyso_ml
```

ToolRL-faithful baseline (no annealing):

```bash
python grpo.py \
  --model_name_or_path /path/to/your/sft_checkpoint \
  --reward_mode toolrl_exact \
  --output_dir outputs/grpo_toolrl_exact \
  --filter_train \
  --prompt_pool_size 4096 \
  --prompt_length_quantile 0.90 \
  --append_think_control no_think \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 1 \
  --num_generations 4 \
  --beta 0.0
```

Our dense reward extension (recommended):

```bash
python grpo.py \
  --model_name_or_path /path/to/your/sft_checkpoint \
  --reward_mode dx7_dist \
  --output_dir outputs/grpo_dx7_dist \
  --filter_train \
  --prompt_pool_size 4096 \
  --prompt_length_quantile 0.90 \
  --append_think_control no_think \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 1 \
  --num_generations 4 \
  --beta 0.0
```

LoRA (optional; default is full fine-tuning):

```bash
python grpo.py \
  --model_name_or_path /path/to/your/sft_checkpoint \
  --reward_mode dx7_dist \
  --output_dir outputs/grpo_dx7_dist_lora \
  --filter_train \
  --use_lora \
  --lora_rank 32 \
  --lora_alpha 64
```

W&B logging (optional):

```bash
python grpo.py \
  --model_name_or_path /path/to/your/sft_checkpoint \
  --reward_mode dx7_dist \
  --output_dir outputs/grpo_dx7_dist_wandb \
  --filter_train \
  --report_to wandb \
  --wandb_project LLMforFM \
  --wandb_entity YOUR_ENTITY \
  --wandb_api_key YOUR_WANDB_KEY
```

Notes:
- By default we keep only the **last 2** checkpoints (`--save_total_limit 2`).
- To resume a crashed run: add `--resume_from_checkpoint latest`.

