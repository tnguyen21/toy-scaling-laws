"""Character-level data loading for text files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CharDataset:
    """Memory-mapped character dataset."""
    data: np.memmap
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]
    vocab_size: int

    def __len__(self) -> int:
        return len(self.data)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in ids)


def prepare_char_data(
    input_path: str,
    out_dir: str,
    val_fraction: float = 0.1,
) -> tuple[str, str, dict]:
    """
    Prepare character-level train/val splits from a text file.

    Returns paths to train.bin, val.bin, and metadata dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Build vocabulary from all unique characters
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    # Encode full text
    ids = np.array([char_to_idx[c] for c in text], dtype=np.uint16)

    # Split into train/val
    n_val = int(len(ids) * val_fraction)
    n_train = len(ids) - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    # Write binary files
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "vocab_size": vocab_size,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }

    return str(train_path), str(val_path), meta


def load_char_data(data_dir: str) -> tuple[CharDataset, CharDataset]:
    """Load prepared character data from a directory."""
    import json

    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    char_to_idx = meta["char_to_idx"]
    idx_to_char = {int(k): v for k, v in meta["idx_to_char"].items()}
    vocab_size = meta["vocab_size"]

    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    train_ds = CharDataset(train_data, char_to_idx, idx_to_char, vocab_size)
    val_ds = CharDataset(val_data, char_to_idx, idx_to_char, vocab_size)

    return train_ds, val_ds


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of sequences."""
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# --- CLI for preparing data ---

if __name__ == "__main__":
    import argparse
    import json
    import urllib.request

    ap = argparse.ArgumentParser(description="Prepare character-level data")
    ap.add_argument("--input", type=str, help="Path to input text file")
    ap.add_argument("--out_dir", type=str, default="data/char", help="Output directory")
    ap.add_argument("--val_fraction", type=float, default=0.1, help="Fraction for validation")
    ap.add_argument("--download_shakespeare", action="store_true", help="Download tiny shakespeare")
    args = ap.parse_args()

    if args.download_shakespeare:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        input_path = Path(args.out_dir) / "input.txt"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading tiny shakespeare to {input_path}...")
        urllib.request.urlretrieve(url, input_path)
        args.input = str(input_path)

    if not args.input:
        raise ValueError("Must provide --input or --download_shakespeare")

    train_path, val_path, meta = prepare_char_data(
        args.input,
        args.out_dir,
        val_fraction=args.val_fraction,
    )

    meta_path = Path(args.out_dir) / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {train_path} ({meta['train_tokens']:,} tokens)")
    print(f"Wrote {val_path} ({meta['val_tokens']:,} tokens)")
    print(f"Wrote {meta_path} (vocab_size={meta['vocab_size']})")
