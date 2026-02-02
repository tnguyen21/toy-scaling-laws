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

# Custom sweep
python sweep.py --flop_budgets 1e15 3e15 1e1 --n_embds 64 96 128 192 256
```

**Note:** FLOP budgets below ~1e13 don't show meaningful scaling behavior for character-level models.

Results are saved to `sweep_results/` with plots and a CSV.

## Multi-GPU Training (DDP)

Both `train.py` and `sweep.py` support distributed data parallel training via `torchrun`:

```bash
# Train on 8 GPUs
torchrun --nproc_per_node=8 train.py --flop_budget 1e15

# Small sweep (shows U-curves at lower compute)
torchrun --nproc_per_node=8 sweep.py --flop_budgets 1e13 3e13 1e14 --n_embds 32 48 64 96 128 --n_layers 2 2 2 4 4 --batch_size 64 --block_size 128

# Medium sweep (clearer scaling, higher compute)
torchrun --nproc_per_node=8 sweep.py --flop_budgets 3e13 6e13 1e14 --n_embds 8 16 32 32 64 96 128  --n_layers 1 1 1 2 2 4 4 --batch_size 128 --block_size 128

# Large sweep (production-scale)
torchrun --nproc_per_node=8 sweep.py --flop_budgets 1e15 3e15 1e16 --n_embds 128 192 256 320 384 --n_layers 4 4 6 6 6 --batch_size 64 --block_size 256
```

With DDP:

- Batch size scales linearly with GPU count (8 GPUs = 8x batch)
- Learning rate scales by sqrt(world_size)
- Each GPU gets different random batches (per-rank seeding)
- Only rank 0 prints, saves checkpoints, and writes results

## Validating Scaling Laws

After running a sweep, test if your fitted scaling laws actually predict:

```bash
# Show predictions for a new FLOP budget (dry run)
python validate.py --results sweep_results/results.csv --flop_budget 2e14 --dry_run

# Train and measure prediction error
python validate.py --results sweep_results/results.csv --flop_budget 2e14
```

The script will:
1. Fit power laws from your sweep results (N_opt ∝ C^α, L_opt ∝ C^β)
2. Predict optimal model size and expected loss for the new budget
3. Train at that budget and report prediction error

A good fit should have <10% prediction error.

## Files

- `model.py` — minimal GPT (attention, MLP, embeddings)
- `data.py` — data prep and loading (GPT-2 tokenization from FineWeb)
- `train.py` — training loop with FLOP targeting
- `sweep.py` — scaling law experiments across model sizes
- `validate.py` — test scaling law predictions at new FLOP budgets
