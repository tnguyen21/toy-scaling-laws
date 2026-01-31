#!/usr/bin/env python3
"""
Scaling law sweep: train multiple model sizes at fixed compute budgets.

Produces a plot like Karpathy's miniseries showing loss vs compute for different model sizes,
revealing the "compute-optimal frontier" where larger models reach lower loss but need more FLOPs.

Usage:
    python sweep.py
    python sweep.py --flop_budgets 1e12 3e12 1e13 --n_embds 32 64 128 256

Example configs:

    # Tiny: for CPU testing, ~10K-100K params, runs in minutes
    # flop_budgets: [1e10, 3e10, 1e11, 3e11]
    # n_embds: [16, 32, 48, 64]
    # n_layers: [1, 2, 3, 4]
    # batch_size: 32, block_size: 64, eval_iters: 20

    # Medium: ~1M-10M params, 1 OOM param sweep
    # Model sizes: d=128 (~1M), d=192 (~2M), d=256 (~5M), d=320 (~7M), d=384 (~10M)
    # FLOP budgets span 2 OOM to see scaling curves cross
    # flop_budgets: [1e14, 3e14, 1e15, 3e15, 1e16]
    # n_embds: [128, 192, 256, 320, 384]
    # n_layers: [4, 4, 6, 6, 6]
    # batch_size: 64, block_size: 256, eval_iters: 50

    # Large: ~60M-240M params
    # FLOP budgets span 1 order of magnitude for visible scaling curves
    # Model sizes: d=640 (~59M), d=768 (~85M), d=896 (~116M), d=1024 (~151M), d=1280 (~236M)
    # flop_budgets: [3e16, 1e17, 3e17]
    # n_embds: [640, 768, 896, 1024, 1280]
    # n_layers: [6, 6, 6, 6, 6]
    # batch_size: 64, block_size: 1024, eval_iters: 100
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_data, get_batch
from model import GPT, ModelConfig


@dataclass
class SweepResult:
    """Result from a single training run."""

    n_embd: int
    n_layer: int
    n_head: int
    n_params: int
    flop_budget: float
    flops_used: float
    tokens_trained: int
    num_iters: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    train_time_sec: float

    @property
    def tokens_per_param(self) -> float:
        """Tokens per parameter ratio. Chinchilla-optimal is ~10-20."""
        return self.tokens_trained / self.n_params if self.n_params > 0 else 0.0


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters):
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


def train_model(
    n_embd: int,
    n_layer: int,
    flop_budget: float,
    train_ds,
    val_ds,
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int = 50,
    learning_rate: float = 1e-3,
    min_lr: float = 1e-4,
    weight_decay: float = 0.1,
    warmup_frac: float = 0.1,
    seed: int = 1337,
    verbose: bool = True,
) -> tuple[SweepResult, list[dict]]:
    """
    Train a single model configuration to a fixed FLOP budget.

    Returns:
        SweepResult with final metrics, and list of intermediate checkpoints
        for plotting loss curves.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Ensure n_head divides n_embd (use n_head = n_embd // 16, minimum 1)
    n_head = max(1, n_embd // 16)
    while n_embd % n_head != 0 and n_head > 1:
        n_head -= 1

    cfg = ModelConfig(
        vocab_size=train_ds.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()

    tokens_per_iter = batch_size * block_size
    flops_per_iter = model.estimate_flops_per_token() * tokens_per_iter
    max_iters = max(1, int(flop_budget / flops_per_iter))
    warmup_iters = int(warmup_frac * max_iters)

    tokens_trained = max_iters * tokens_per_iter
    tokens_per_param = tokens_trained / n_params if n_params > 0 else 0.0

    if verbose:
        print(f"  Model: d={n_embd}, L={n_layer}, h={n_head}, params={n_params:,}, iters={max_iters}, tok/param={tokens_per_param:.1f}")

    optimizer = model.configure_optim(weight_decay=weight_decay, learning_rate=learning_rate)

    # Training loop
    t0 = time.time()
    best_val_loss = float("inf")
    flops_used = 0
    checkpoints = []  # For plotting loss curves

    eval_interval = max(1, max_iters // 10)

    for it in range(1, max_iters + 1):
        lr = get_lr(it - 1, warmup_iters, max_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Eval
        if it == 1 or it % eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, train_ds, val_ds, batch_size, block_size, device, eval_iters)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

            checkpoints.append(
                {
                    "iter": it,
                    "flops": flops_used,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                }
            )

        # Training step
        x, y = get_batch(train_ds.data, batch_size, block_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        flops_used += flops_per_iter

    train_time = time.time() - t0
    final_losses = estimate_loss(model, train_ds, val_ds, batch_size, block_size, device, eval_iters)

    result = SweepResult(
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_params=n_params,
        flop_budget=flop_budget,
        flops_used=flops_used,
        tokens_trained=max_iters * tokens_per_iter,
        num_iters=max_iters,
        final_train_loss=final_losses["train"],
        final_val_loss=final_losses["val"],
        best_val_loss=best_val_loss,
        train_time_sec=train_time,
    )

    return result, checkpoints


def plot_scaling_curves(
    all_results: list[SweepResult],
    all_checkpoints: dict[tuple[int, int], list[dict]],
    out_dir: str,
):
    """
    Plot scaling law curves similar to Karpathy's miniseries plot.

    Main plot: Loss vs FLOPs, with each model size as a separate curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get unique model configs (by n_params) and assign colors
    unique_configs = sorted(set((r.n_embd, r.n_layer, r.n_params) for r in all_results))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_configs)))
    config_to_color = {cfg: colors[i] for i, cfg in enumerate(unique_configs)}

    # Plot 1: Loss curves over training (loss vs FLOPs)
    ax = axes[0]
    for (n_embd, n_layer), checkpoints in all_checkpoints.items():
        if not checkpoints:
            continue
        n_params = next(r.n_params for r in all_results if r.n_embd == n_embd and r.n_layer == n_layer)
        cfg = (n_embd, n_layer, n_params)
        color = config_to_color.get(cfg, "gray")

        # Filter out zero-flop points (first checkpoint is at iter=1 before training step)
        filtered = [(c["flops"], c["val_loss"]) for c in checkpoints if c["flops"] > 0]
        if not filtered:
            continue
        flops, val_loss = zip(*filtered)

        if flops:
            label = f"d{n_embd}_L{n_layer} ({n_params // 1000}K)"
            ax.plot(flops, val_loss, "-", color=color, label=label, alpha=0.8)
            ax.scatter([flops[-1]], [val_loss[-1]], color=color, s=50, zorder=5)

    ax.set_xlabel("Training FLOPs")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Loss vs Compute (Training Curves)")
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    # Plot 2: Final loss vs FLOPs (the scaling law plot)
    ax = axes[1]
    for n_embd, n_layer, n_params in unique_configs:
        results = [r for r in all_results if r.n_embd == n_embd and r.n_layer == n_layer]
        results.sort(key=lambda r: r.flops_used)

        flops = [r.flops_used for r in results]
        losses = [r.best_val_loss for r in results]

        color = config_to_color[(n_embd, n_layer, n_params)]
        label = f"d{n_embd}_L{n_layer} ({n_params // 1000}K)"
        ax.plot(flops, losses, "o-", color=color, label=label, markersize=8)

    ax.set_xlabel("Training FLOPs")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Scaling Law: Loss vs Compute")
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scaling_curves.png"), dpi=150)
    plt.close()
    print(f"Saved scaling curves to {out_dir}/scaling_curves.png")

    # Additional plot: log-log to see power law
    fig, ax = plt.subplots(figsize=(8, 6))
    for n_embd, n_layer, n_params in unique_configs:
        results = [r for r in all_results if r.n_embd == n_embd and r.n_layer == n_layer]
        results.sort(key=lambda r: r.flops_used)

        flops = [r.flops_used for r in results]
        losses = [r.best_val_loss for r in results]

        color = config_to_color[(n_embd, n_layer, n_params)]
        label = f"d{n_embd}_L{n_layer} ({n_params // 1000}K)"
        ax.loglog(flops, losses, "o-", color=color, label=label, markersize=8)

    ax.set_xlabel("Training FLOPs")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Scaling Law (Log-Log)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scaling_curves_loglog.png"), dpi=150)
    plt.close()
    print(f"Saved log-log plot to {out_dir}/scaling_curves_loglog.png")


def main():
    ap = argparse.ArgumentParser(description="Scaling law sweep")
    ap.add_argument("--data_dir", type=str, default="data/fineweb_char")
    ap.add_argument("--out_dir", type=str, default="sweep_results")
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--flop_budgets", type=float, nargs="+", default=[1e12, 3e12, 1e13], help="FLOP budgets to sweep")
    ap.add_argument("--n_embds", type=int, nargs="+", default=[64, 128, 256], help="Embedding dimensions to sweep")
    ap.add_argument("--n_layers", type=int, nargs="+", default=[2, 4, 6], help="Layer counts to sweep")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--eval_iters", type=int, default=50)

    args = ap.parse_args()

    flop_budgets = args.flop_budgets
    n_embds = args.n_embds
    n_layers = args.n_layers
    batch_size = args.batch_size
    block_size = args.block_size
    eval_iters = args.eval_iters

    print(f"Device: {args.device}")
    print(f"FLOP budgets: [{', '.join(f'{b:.0e}' for b in flop_budgets)}]")
    print(f"n_embds: {n_embds}")
    print(f"n_layers: {n_layers}")

    train_ds, val_ds = load_data(args.data_dir)
    print(f"Loaded data: {len(train_ds):,} train tokens, vocab_size={train_ds.vocab_size}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_results: list[SweepResult] = []
    all_checkpoints: dict[tuple[int, int], list[dict]] = {}

    csv_path = os.path.join(args.out_dir, "results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "flop_budget",
            "n_embd",
            "n_layer",
            "n_head",
            "n_params",
            "flops_used",
            "tokens_trained",
            "num_iters",
            "tokens_per_param",
            "final_train_loss",
            "final_val_loss",
            "best_val_loss",
            "train_time_sec",
        ]
    )

    total_runs = len(flop_budgets) * len(n_embds)
    run_idx = 0

    for flop_budget in flop_budgets:
        print(f"\n{'=' * 60}")
        print(f"FLOP Budget: {flop_budget:.2e}")
        print(f"{'=' * 60}")

        for n_embd, n_layer in zip(n_embds, n_layers):
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Training d={n_embd}, L={n_layer} @ {flop_budget:.0e} FLOPs")

            result, checkpoints = train_model(
                n_embd=n_embd,
                n_layer=n_layer,
                flop_budget=flop_budget,
                train_ds=train_ds,
                val_ds=val_ds,
                batch_size=batch_size,
                block_size=block_size,
                device=args.device,
                eval_iters=eval_iters,
                seed=args.seed,
            )

            all_results.append(result)

            # Keep only checkpoints from highest FLOP budget run (for clean curves)
            key = (n_embd, n_layer)
            all_checkpoints[key] = checkpoints

            csv_writer.writerow(
                [
                    result.flop_budget,
                    result.n_embd,
                    result.n_layer,
                    result.n_head,
                    result.n_params,
                    result.flops_used,
                    result.tokens_trained,
                    result.num_iters,
                    f"{result.tokens_per_param:.1f}",
                    result.final_train_loss,
                    result.final_val_loss,
                    result.best_val_loss,
                    result.train_time_sec,
                ]
            )
            csv_file.flush()

            print(f"  -> val_loss={result.best_val_loss:.4f}, tok/param={result.tokens_per_param:.1f}, time={result.train_time_sec:.1f}s")

    csv_file.close()
    print(f"\nResults saved to {csv_path}")

    plot_scaling_curves(all_results, all_checkpoints, args.out_dir)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Config':<20} {'Params':>10} {'FLOPs':>12} {'Tok/Param':>10} {'Val Loss':>10}")
    print("-" * 100)
    for r in sorted(all_results, key=lambda x: (x.n_params, x.flop_budget)):
        cfg = f"d{r.n_embd}_L{r.n_layer}"
        print(f"{cfg:<20} {r.n_params:>10,} {r.flops_used:>12.2e} {r.tokens_per_param:>10.1f} {r.best_val_loss:>10.4f}")


if __name__ == "__main__":
    main()
