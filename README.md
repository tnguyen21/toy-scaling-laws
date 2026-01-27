# baby-scaling-laws

Minimal transformer for exploring scaling laws.

## Setup

```bash
uv sync
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

## Encoding Options

Data can be prepared with character-level or BPE tokenization:

```bash
# Character-level (default) - each character is a token
python data.py --download_shakespeare --encoding char --out_dir data/char

# GPT-2 BPE tokenizer (50,257 vocab)
python data.py --download_shakespeare --encoding gpt2 --out_dir data/gpt2

# Other tiktoken encodings
python data.py --input mytext.txt --encoding cl100k_base --out_dir data/cl100k
```

The training script auto-detects the encoding from `meta.json`:

```bash
python train.py --data_dir data/gpt2 --flop_budget 1e15
```

| Encoding | Vocab Size | Use Case |
|----------|------------|----------|
| `char` | ~65 (text-dependent) | Fast experiments, small models |
| `gpt2` | 50,257 | Standard BPE, comparable to GPT-2 |
| `r50k_base` | 50,257 | Codex models |
| `cl100k_base` | 100,277 | GPT-4, ChatGPT models |

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
- `data.py` — data prep and loading (char-level or BPE via tiktoken)
- `train.py` — training loop with FLOP targeting
