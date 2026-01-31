"""Text data loading with GPT-2 BPE encoding from FineWeb."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch
from datasets import load_dataset

# GPT-2 encoding (50257 tokens)
GPT2_VOCAB_SIZE = 50257


@dataclass
class TokenDataset:
    """Memory-mapped token dataset with tiktoken GPT-2 encoding."""

    data: np.memmap
    vocab_size: int
    encoding: tiktoken.Encoding

    def __len__(self) -> int:
        return len(self.data)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode(ids)


def load_data(data_dir: str) -> tuple[TokenDataset, TokenDataset]:
    """Load prepared token data from a directory."""
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    enc = tiktoken.get_encoding("gpt2")

    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    train_ds = TokenDataset(data=train_data, vocab_size=vocab_size, encoding=enc)
    val_ds = TokenDataset(data=val_data, vocab_size=vocab_size, encoding=enc)

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

    ap = argparse.ArgumentParser(description="Prepare FineWeb data for training (GPT-2 BPE)")
    ap.add_argument("--out_dir", type=str, default="data/fineweb_gpt2", help="Output directory")
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

    # Initialize GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print(f"Using GPT-2 BPE encoding (vocab_size={GPT2_VOCAB_SIZE})")

    # Single pass: encode all documents
    print("Encoding documents...")
    all_ids: list[int] = []

    for i, ex in enumerate(tqdm(ds, total=args.num_docs, desc="encoding")):
        if i >= args.num_docs:
            break
        text = ex.get(args.text_field, "")
        if not isinstance(text, str) or not text:
            continue
        all_ids.extend(enc.encode(text))

    # Split into train/val
    ids = np.array(all_ids, dtype=np.uint16)
    n_val = int(len(ids) * args.val_fraction)
    n_train = len(ids) - n_val

    train_arr = ids[:n_train]
    val_arr = ids[n_train:]

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    meta = {
        "encoding": "gpt2",
        "dataset": args.dataset,
        "name": args.name,
        "num_docs": args.num_docs,
        "vocab_size": GPT2_VOCAB_SIZE,
        "train_tokens": len(train_arr),
        "val_tokens": len(val_arr),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {train_path} ({len(train_arr):,} tokens)")
    print(f"Wrote {val_path} ({len(val_arr):,} tokens)")
    print(f"Wrote {meta_path} (vocab_size={GPT2_VOCAB_SIZE})")
