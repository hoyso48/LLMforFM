## LLMforFM

Research code for learning DX7/FM-synth patch representations with LLMs, plus utilities for embedding extraction and evaluation.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data + Git LFS

- `data/*.csv` is **tracked via Git LFS** (so the repo stays manageable as datasets grow).
- Large/raw assets are intentionally not versioned in git (e.g. `data/wav/`, `data/dx7_patches/`, `artifacts/`, `outputs/`).

After cloning:

```bash
git lfs install
git lfs pull
```

### Entry points

- `fine_tuning.py`: training
- `grpo.py`: RL (GRPO) training for DX7 tool learning (ToolRL-style exact reward + our dense distance reward)
- `evaluate.py`: evaluation
- `scripts/`: data/embedding helper scripts

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

