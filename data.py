"""Text data loading with character-level encoding from FineWeb."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


@dataclass
class CharDataset:
    """Memory-mapped character-level text dataset."""

    data: np.memmap
    vocab_size: int
    char_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_char: dict[int, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in ids)


def load_data(data_dir: str) -> tuple[CharDataset, CharDataset]:
    """Load prepared character-level data from a directory."""
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    char_to_idx = meta["char_to_idx"]
    idx_to_char = {int(k): v for k, v in meta["idx_to_char"].items()}

    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    train_ds = CharDataset(
        data=train_data,
        vocab_size=vocab_size,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
    )
    val_ds = CharDataset(
        data=val_data,
        vocab_size=vocab_size,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
    )

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
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# --- CLI for preparing data from FineWeb ---

if __name__ == "__main__":
    import argparse

    from tqdm import tqdm

    ap = argparse.ArgumentParser(description="Prepare FineWeb data for training (character-level)")
    ap.add_argument("--out_dir", type=str, default="data/fineweb_char", help="Output directory")
    ap.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb", help="HuggingFace dataset")
    ap.add_argument("--name", type=str, default="sample-10BT", help="Dataset config name")
    ap.add_argument("--split", type=str, default="train", help="Dataset split")
    ap.add_argument("--text_field", type=str, default="text", help="Field containing text")
    ap.add_argument("--num_docs", type=int, default=5_000, help="Number of documents to process")
    ap.add_argument("--val_fraction", type=float, default=0.1, help="Fraction for validation")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    meta_path = out_dir / "meta.json"

    for p in (train_path, val_path, meta_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {p} (pass --overwrite)")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}/{args.name} (streaming)...")
    ds = load_dataset(args.dataset, args.name, split=args.split, streaming=True)

    # First pass: collect all text to build vocabulary
    print("Pass 1: Building vocabulary...")
    all_chars: set[str] = set()
    texts: list[str] = []

    for i, ex in enumerate(tqdm(ds, total=args.num_docs, desc="scanning")):
        if i >= args.num_docs:
            break
        text = ex.get(args.text_field, "")
        if not isinstance(text, str) or not text:
            continue
        texts.append(text)
        all_chars.update(text)

    chars = sorted(all_chars)
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    print(f"Vocabulary size: {vocab_size}")

    # Second pass: encode all text
    print("Pass 2: Encoding...")
    all_ids: list[int] = []
    for text in tqdm(texts, desc="encoding"):
        all_ids.extend(char_to_idx[c] for c in text)

    # Split into train/val
    ids = np.array(all_ids, dtype=np.uint16)
    n_val = int(len(ids) * args.val_fraction)
    n_train = len(ids) - n_val

    train_arr = ids[:n_train]
    val_arr = ids[n_train:]

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    meta = {
        "dataset": args.dataset,
        "name": args.name,
        "num_docs": len(texts),
        "vocab_size": vocab_size,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        "train_tokens": len(train_arr),
        "val_tokens": len(val_arr),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {train_path} ({len(train_arr):,} chars)")
    print(f"Wrote {val_path} ({len(val_arr):,} chars)")
    print(f"Wrote {meta_path} (vocab_size={vocab_size})")
