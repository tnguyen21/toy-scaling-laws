# baby-scaling-laws

Minimal transformer for exploring scaling laws.

## Setup

```bash
uv sync
```

## Quick Start

```bash
# 1. Prepare data (downloads FineWeb documents, GPT-2 tokenization)
python data.py

# 2. Train with a FLOP budget
python train.py --flop_budget 1e12

# Or train for fixed iterations
python train.py --max_iters 5000
```

Data is saved to `data/fineweb_gpt2/` by default. Customize with:

```bash
python data.py --num_docs 50000 --out_dir data/fineweb_50k
python train.py --data_dir data/fineweb_50k --flop_budget 1e12
```

## FLOP-Targeted Training

The key idea: instead of fixing iterations, fix a compute budget. This makes experiments comparable across model sizes.

```bash
# Same compute, different model sizes
python train.py --n_embd 64  --n_layer 2 --flop_budget 1e14
python train.py --n_embd 96  --n_layer 4 --flop_budget 1e14
python train.py --n_embd 128 --n_layer 4 --flop_budget 1e14
```

## Sweeps

Run a scaling law sweep across model sizes and compute budgets:

```bash
# Default sweep (1e13-1e14 range shows clear U-shaped isoFLOP curves)
python sweep.py

# Custom sweep with --depths (n_embd = depth * 32, n_layer = depth)
python sweep.py --flop_budgets 1e14 3e14 1e15 --depths 1 2 3 4 5 6 8
```

**Note:** FLOP budgets below ~1e13 don't show meaningful scaling behavior for character-level models.

Results are saved to `sweep_results/` with plots and a CSV.

## Multi-GPU Training (DDP)

Both `train.py` and `sweep.py` support distributed data parallel training via `torchrun`:

```bash
# Train on 8 GPUs
torchrun --nproc_per_node=8 train.py --flop_budget 1e15

# Small sweep (depths 1-6)
torchrun --nproc_per_node=8 sweep.py --flop_budgets 1e13 3e13 1e14 --depths 1 2 3 4 5 6 --batch_size 64 --block_size 256

# Medium sweep (depths 2-10)
torchrun --nproc_per_node=8 sweep.py --flop_budgets 1e14 3e14 1e15 --depths 2 3 4 5 6 8 10 --batch_size 64 --block_size 256

# Large sweep (wider models with dim_mult=64)
torchrun --nproc_per_node=8 sweep.py --flop_budgets 1e15 3e15 1e16 --depths 4 6 8 10 12 --dim_mult 64 --batch_size 64 --block_size 256
```

With DDP:

- Batch size scales linearly with GPU count (8 GPUs = 8x batch)
- Learning rate scales by sqrt(world_size)
- Each GPU gets different random batches (per-rank seeding)
- Only rank 0 prints, saves checkpoints, and writes results

## Files

- `model.py` — minimal GPT (attention, MLP, embeddings)
- `data.py` — data prep and loading (GPT-2 tokenization from FineWeb)
- `train.py` — training loop with FLOP targeting
- `sweep.py` — scaling law experiments across model sizes
