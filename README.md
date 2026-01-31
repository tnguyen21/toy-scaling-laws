# baby-scaling-laws

Minimal transformer for exploring scaling laws.

## Setup

```bash
uv sync
```

## Quick Start

```bash
# 1. Prepare data (downloads FineWeb documents, char-level encoding)
python data.py

# 2. Train with a FLOP budget
python train.py --flop_budget 1e15

# Or train for fixed iterations
python train.py --max_iters 5000
```

Data is saved to `data/fineweb_char/` by default. Customize with:

```bash
python data.py --num_docs 25000 --out_dir data/fineweb_25k # ~100M training tokens
python train.py --data_dir data/fineweb_25k --flop_budget 1e15
```

## FLOP-Targeted Training

The key idea: instead of fixing iterations, fix a compute budget. This makes experiments comparable across model sizes.

```bash
# Same compute, different model sizes
python train.py --n_embd 64  --n_layer 2 --flop_budget 1e15
python train.py --n_embd 128 --n_layer 4 --flop_budget 1e15
python train.py --n_embd 256 --n_layer 6 --flop_budget 1e15
```

## Sweeps

Run a scaling law sweep across model sizes and compute budgets:

```bash
# Default sweep
python sweep.py

# Custom sweep
python sweep.py --flop_budgets 1e12 3e12 1e13 --n_embds 32 64 128 256
```

Results are saved to `sweep_results/` with plots and a CSV.

## Files

- `model.py` — minimal GPT (attention, MLP, embeddings)
- `data.py` — data prep and loading (char-level from FineWeb)
- `train.py` — training loop with FLOP targeting
- `sweep.py` — scaling law experiments across model sizes
