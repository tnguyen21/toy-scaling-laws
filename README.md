# baby-scaling-laws

Minimal transformer for exploring scaling laws at character level.

## Setup

```bash
pip install torch numpy
```

## Quick Start

```bash
# 1. Prepare data (downloads tiny shakespeare)
python data.py --download_shakespeare --out_dir data/shakespeare

# 2. Train with a FLOP budget
python train.py --data_dir data/shakespeare --flop_budget 1e15

# Or train for fixed iterations
python train.py --data_dir data/shakespeare --max_iters 5000
```

## FLOP-Targeted Training

The key idea: instead of fixing iterations, fix a compute budget. This makes experiments comparable across model sizes.

```bash
# Same compute, different model sizes
python train.py --n_embd 64  --n_layer 2 --flop_budget 1e15
python train.py --n_embd 128 --n_layer 4 --flop_budget 1e15
python train.py --n_embd 256 --n_layer 6 --flop_budget 1e15
```

## Files

- `model.py` — minimal GPT (attention, MLP, embeddings)
- `data.py` — character-level data prep and loading
- `train.py` — training loop with FLOP targeting
