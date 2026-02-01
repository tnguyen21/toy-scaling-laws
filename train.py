#!/usr/bin/env python3
"""
Training script with FLOP-budget targeting.

The key idea: instead of fixing iterations, we fix a compute budget (in FLOPs)
and let the training run until that budget is exhausted. This makes experiments
comparable across different model sizes.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, ModelConfig
from data import load_data, get_batch


def setup_ddp():
    """Initialize DDP if launched via torchrun, otherwise single-GPU/CPU fallback."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return rank, world_size, device


def cleanup_ddp(world_size: int):
    """Clean up DDP process group."""
    if world_size > 1:
        dist.destroy_process_group()


@torch.no_grad()
def estimate_loss(model: GPT, train_data, val_data, batch_size: int, block_size: int, device: str, eval_iters: int = 50):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data.data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def plot_training_progress(history: dict, out_dir: str):
    """Plot training curves and save to files."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss vs iteration
    ax = axes[0, 0]
    ax.plot(history["iter"], history["train_loss"], label="train")
    ax.plot(history["iter"], history["val_loss"], label="val")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss vs FLOPs (scaling law plot)
    ax = axes[0, 1]
    ax.plot(history["flops"], history["train_loss"], label="train")
    ax.plot(history["flops"], history["val_loss"], label="val")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Compute (Scaling Law)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    # Learning rate schedule
    ax = axes[1, 0]
    ax.plot(history["iter"], history["lr"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    # Log-scale loss vs FLOPs
    ax = axes[1, 1]
    # Filter out zero-flop points for log scale
    flops_filtered = [f for f in history["flops"] if f > 0]
    loss_filtered = [loss for f, loss in zip(history["flops"], history["val_loss"]) if f > 0]
    if flops_filtered:
        ax.loglog(flops_filtered, loss_filtered, "o-", label="val_loss")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Compute (Log-Log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_progress.png"), dpi=150)
    plt.close()
    print(f"Saved training plots to {out_dir}/training_progress.png")


def main():
    # Setup DDP first (before parsing args that depend on device)
    rank, world_size, device = setup_ddp()
    is_main = rank == 0

    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_dir", type=str, default="data/fineweb_gpt2", help="Directory with train.bin, val.bin, meta.json")

    # Model (small defaults for GPT-2 tokenizer with 50K vocab)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--n_embd", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Training
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_frac", type=float, default=0.1, help="Fraction of training for warmup")

    # FLOP budget (the key parameter!)
    ap.add_argument("--flop_budget", type=float, default=1e12, help="Total training FLOPs budget")
    ap.add_argument("--max_iters", type=int, default=None, help="Override: max iterations (ignores flop_budget)")

    # Eval & logging
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="out")

    # Device (ignored when using DDP, kept for backward compatibility)
    ap.add_argument("--device", type=str, default=None, help="Device (auto-detected with DDP)")
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    # Per-rank seed for different batches across GPUs
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Scale LR with world size: sqrt scaling
    effective_lr = args.learning_rate * math.sqrt(world_size)
    effective_min_lr = args.min_lr * math.sqrt(world_size)

    # Load data
    train_ds, val_ds = load_data(args.data_dir)
    if is_main:
        print(f"Loaded data: {len(train_ds):,} train tokens, {len(val_ds):,} val tokens, vocab_size={train_ds.vocab_size}")

    # Build model
    cfg = ModelConfig(
        vocab_size=train_ds.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    if is_main:
        print(f"Model: {n_params:,} parameters ({n_params / 1e6:.2f}M)")

    # Wrap model in DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Compute training schedule from FLOP budget
    # Get the raw model for FLOP estimation (unwrap DDP if needed)
    raw_model = model.module if world_size > 1 else model
    tokens_per_iter = args.batch_size * args.block_size * world_size  # Scale with world_size
    flops_per_iter = raw_model.estimate_flops_per_token() * tokens_per_iter

    if args.max_iters is not None:
        max_iters = args.max_iters
        total_flops = flops_per_iter * max_iters
    else:
        max_iters = int(args.flop_budget / flops_per_iter)
        total_flops = args.flop_budget

    total_tokens = max_iters * tokens_per_iter
    warmup_iters = int(args.warmup_frac * max_iters)

    if is_main:
        print("\n=== Training Plan ===")
        print(f"FLOP budget: {total_flops:.2e}")
        print(f"FLOPs per iter: {flops_per_iter:.2e}")
        print(f"Max iterations: {max_iters:,}")
        print(f"Total tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M)")
        print(f"Warmup iters: {warmup_iters:,}")
        if world_size > 1:
            print(f"World size: {world_size} GPUs")
            print(f"Effective batch size: {args.batch_size * world_size}")
            print(f"Effective LR: {effective_lr:.2e}")
        print("======================\n")

    # Optimizer (use raw_model for configure_optim, use scaled LR)
    optimizer = raw_model.configure_optim(
        weight_decay=args.weight_decay,
        learning_rate=effective_lr,
    )

    # Output directory (only rank 0 creates)
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)

    # Metrics history for plotting
    history = {
        "iter": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "flops": [],
    }

    # Training loop
    t0 = time.time()
    best_val_loss = float("inf")
    flops_used = 0

    for it in range(1, max_iters + 1):
        # Learning rate schedule (use effective LRs)
        lr = get_lr(it - 1, warmup_iters, max_iters, effective_lr, effective_min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Eval (only rank 0 prints and saves)
        if it == 1 or it % args.eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, train_ds, val_ds, args.batch_size, args.block_size, device, args.eval_iters)
            if is_main:
                print(f"[iter {it:5d}] train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} lr={lr:.2e} flops={flops_used:.2e}")

                # Record metrics for plotting
                history["iter"].append(it)
                history["train_loss"].append(losses["train"])
                history["val_loss"].append(losses["val"])
                history["lr"].append(lr)
                history["flops"].append(flops_used)

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    ckpt = {
                        "model": raw_model.state_dict(),
                        "config": cfg.__dict__,
                        "iter": it,
                        "best_val_loss": best_val_loss,
                        "flops_used": flops_used,
                    }
                    torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

        # Training step
        x, y = get_batch(train_ds.data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        flops_used += flops_per_iter

        # Logging (only rank 0)
        if it % args.log_interval == 0 and is_main:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = tokens_per_iter * args.log_interval / dt
            print(f"  iter {it:5d} | loss {loss.item():.4f} | {tokens_per_sec:.0f} tok/s | {dt * 1000 / args.log_interval:.1f} ms/iter")

    # Only rank 0 handles plotting, generation sample, and final checkpoint
    if is_main:
        # Plot training progress
        if len(history["iter"]) > 1:
            plot_training_progress(history, args.out_dir)

        # Final generation sample
        print("\n=== Sample Generation ===")
        raw_model.eval()
        prompt = "ROMEO:"
        ids = train_ds.encode(prompt)
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = raw_model.generate(x, max_new_tokens=200, temperature=0.8)
        print(train_ds.decode(out[0].tolist()))

        # Save final checkpoint
        final_ckpt = {
            "model": raw_model.state_dict(),
            "config": cfg.__dict__,
            "iter": max_iters,
            "final_val_loss": losses["val"],
            "flops_used": flops_used,
        }
        torch.save(final_ckpt, os.path.join(args.out_dir, "final.pt"))
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Cleanup DDP
    cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
