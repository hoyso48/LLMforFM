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
- `evaluate.py`: evaluation
- `scripts/`: data/embedding helper scripts

