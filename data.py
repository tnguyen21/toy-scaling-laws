"""Text data loading with character-level and BPE encoding support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class TextDataset:
    """Memory-mapped text dataset with encoding-aware encode/decode."""

    data: np.memmap
    vocab_size: int
    encoding: str
    # For char encoding
    char_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_char: dict[int, str] = field(default_factory=dict)
    # For BPE encoding (lazy-loaded)
    _tiktoken_enc: Any = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.data)

    def _get_tiktoken_enc(self):
        """Lazy-load tiktoken encoder."""
        if self._tiktoken_enc is None:
            import tiktoken

            self._tiktoken_enc = tiktoken.get_encoding(self.encoding)
        return self._tiktoken_enc

    def encode(self, text: str) -> list[int]:
        if self.encoding == "char":
            return [self.char_to_idx.get(c, 0) for c in text]
        else:
            return self._get_tiktoken_enc().encode(text)

    def decode(self, ids: list[int]) -> str:
        if self.encoding == "char":
            return "".join(self.idx_to_char.get(i, "?") for i in ids)
        else:
            return self._get_tiktoken_enc().decode(ids)


def prepare_data(
    input_path: str,
    out_dir: str,
    encoding: str = "char",
    val_fraction: float = 0.1,
) -> tuple[str, str, dict]:
    """
    Prepare train/val splits from a text file.

    Args:
        input_path: Path to input text file
        out_dir: Output directory for train.bin, val.bin, meta.json
        encoding: "char" for character-level, or tiktoken encoding name
                  (e.g., "gpt2", "r50k_base", "cl100k_base")
        val_fraction: Fraction of data for validation

    Returns:
        Tuple of (train_path, val_path, metadata dict)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if encoding == "char":
        # Character-level encoding
        chars = sorted(set(text))
        vocab_size = len(chars)
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for i, c in enumerate(chars)}

        ids = np.array([char_to_idx[c] for c in text], dtype=np.uint16)

        meta = {
            "encoding": "char",
            "vocab_size": vocab_size,
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
        }
    else:
        # BPE encoding via tiktoken
        import tiktoken

        enc = tiktoken.get_encoding(encoding)
        vocab_size = enc.n_vocab

        token_ids = enc.encode(text)
        # Use uint32 for BPE since vocab can be > 65535
        ids = np.array(token_ids, dtype=np.uint32)

        meta = {
            "encoding": encoding,
            "vocab_size": vocab_size,
        }

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

    meta["train_tokens"] = len(train_ids)
    meta["val_tokens"] = len(val_ids)

    return str(train_path), str(val_path), meta


def load_data(data_dir: str) -> tuple[TextDataset, TextDataset]:
    """Load prepared data from a directory. Auto-detects encoding from meta.json."""
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    encoding = meta.get("encoding", "char")
    vocab_size = meta["vocab_size"]

    # Determine dtype based on encoding
    if encoding == "char":
        dtype = np.uint16
        char_to_idx = meta["char_to_idx"]
        idx_to_char = {int(k): v for k, v in meta["idx_to_char"].items()}
    else:
        dtype = np.uint32
        char_to_idx = {}
        idx_to_char = {}

    train_data = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=dtype, mode="r")

    train_ds = TextDataset(
        data=train_data,
        vocab_size=vocab_size,
        encoding=encoding,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
    )
    val_ds = TextDataset(
        data=val_data,
        vocab_size=vocab_size,
        encoding=encoding,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
    )

    return train_ds, val_ds


# Backwards compatibility aliases
CharDataset = TextDataset
prepare_char_data = prepare_data
load_char_data = load_data


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


# --- CLI for preparing data ---

if __name__ == "__main__":
    import argparse
    import urllib.request

    ap = argparse.ArgumentParser(description="Prepare text data for training")
    ap.add_argument("--input", type=str, help="Path to input text file")
    ap.add_argument("--out_dir", type=str, default="data/char", help="Output directory")
    ap.add_argument("--val_fraction", type=float, default=0.1, help="Fraction for validation")
    ap.add_argument(
        "--encoding",
        type=str,
        default="char",
        help="Encoding: 'char' for character-level, or tiktoken name (gpt2, r50k_base, cl100k_base, etc.)",
    )
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

    train_path, val_path, meta = prepare_data(
        args.input,
        args.out_dir,
        encoding=args.encoding,
        val_fraction=args.val_fraction,
    )

    meta_path = Path(args.out_dir) / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {train_path} ({meta['train_tokens']:,} tokens)")
    print(f"Wrote {val_path} ({meta['val_tokens']:,} tokens)")
    print(f"Wrote {meta_path} (encoding={meta['encoding']}, vocab_size={meta['vocab_size']})")
