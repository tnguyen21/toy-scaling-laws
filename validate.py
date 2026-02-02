#!/usr/bin/env python3
"""
Validate scaling law predictions.

Given sweep results, fit power laws and test predictions at a new FLOP budget.
This is the "does the scaling law actually predict" sanity check.

Usage:
    # Predict and train at 2e14 FLOPs
    python validate.py --results sweep_results/results.csv --flop_budget 2e14

    # Just show predictions without training
    python validate.py --results sweep_results/results.csv --flop_budget 2e14 --dry_run

    # Multi-GPU validation
    torchrun --nproc_per_node=4 validate.py --results sweep_results/results.csv --flop_budget 2e14
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def fit_power_laws(df: pd.DataFrame) -> dict:
    """
    Fit power laws from sweep results.

    Returns dict with:
        - N_opt(C) = a_N * C^alpha_N
        - L_opt(C) = a_L * C^alpha_L
    """
    # Group by FLOP budget, find optimal N (min loss) at each
    optima = []
    for flop_budget, grp in df.groupby("flop_budget"):
        best = grp.loc[grp["final_val_loss"].idxmin()]
        optima.append(
            {
                "C": flop_budget,
                "N_opt": best["n_params"],
                "L_opt": best["final_val_loss"],
                "n_embd": best["n_embd"],
                "n_layer": best["n_layer"],
            }
        )
    opt_df = pd.DataFrame(optima).sort_values("C")

    if len(opt_df) < 2:
        raise ValueError(f"Need at least 2 FLOP budgets for fitting, got {len(opt_df)}")

    # Fit in log-log space
    log_C = np.log(opt_df["C"].values)
    log_N = np.log(opt_df["N_opt"].values)
    log_L = np.log(opt_df["L_opt"].values)

    # N_opt = a_N * C^alpha_N
    alpha_N, log_a_N = np.polyfit(log_C, log_N, 1)
    a_N = np.exp(log_a_N)

    # L_opt = a_L * C^alpha_L
    alpha_L, log_a_L = np.polyfit(log_C, log_L, 1)
    a_L = np.exp(log_a_L)

    return {
        "a_N": a_N,
        "alpha_N": alpha_N,
        "a_L": a_L,
        "alpha_L": alpha_L,
        "optima": opt_df,
    }


def predict_optimal(fits: dict, C: float) -> dict:
    """Predict optimal model size and loss for a given FLOP budget."""
    N_pred = fits["a_N"] * (C ** fits["alpha_N"])
    L_pred = fits["a_L"] * (C ** fits["alpha_L"])
    D_pred = C / (6 * N_pred)  # C ≈ 6 * N * D

    return {
        "N_pred": N_pred,
        "L_pred": L_pred,
        "D_pred": D_pred,
    }


def find_model_config(N_target: float, dim_mult: int = 32) -> tuple[int, int, int]:
    """
    Find model config (n_embd, n_layer, n_head) closest to target param count.

    Uses depth-based scaling: n_embd = depth * dim_mult, n_layer = depth.
    """
    best_depth = 1
    best_diff = float("inf")

    for depth in range(1, 20):
        n_embd = depth * dim_mult
        n_layer = depth
        # Rough param estimate: ~12 * n_layer * n_embd^2 (from transformer architecture)
        n_params_est = 12 * n_layer * n_embd**2
        diff = abs(n_params_est - N_target)
        if diff < best_diff:
            best_diff = diff
            best_depth = depth

    n_embd = best_depth * dim_mult
    n_layer = best_depth
    n_head = max(1, n_embd // 32)  # ~32 dims per head

    return n_embd, n_layer, n_head


def run_training(
    flop_budget: float,
    n_embd: int,
    n_layer: int,
    n_head: int,
    data_dir: str,
    batch_size: int,
    block_size: int,
    out_dir: str,
) -> float:
    """Run training and return final validation loss."""
    cmd = [
        sys.executable,
        "train.py",
        "--flop_budget",
        str(flop_budget),
        "--n_embd",
        str(n_embd),
        "--n_layer",
        str(n_layer),
        "--n_head",
        str(n_head),
        "--data_dir",
        data_dir,
        "--batch_size",
        str(batch_size),
        "--block_size",
        str(block_size),
        "--out_dir",
        out_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    # Load final checkpoint to get loss
    import torch

    ckpt_path = os.path.join(out_dir, "final.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["final_val_loss"]


def main():
    ap = argparse.ArgumentParser(description="Validate scaling law predictions")
    ap.add_argument("--results", type=str, required=True, help="Path to sweep results.csv")
    ap.add_argument("--flop_budget", type=float, required=True, help="FLOP budget to validate at")
    ap.add_argument("--dim_mult", type=int, default=32, help="Multiplier for depth -> n_embd")
    ap.add_argument("--data_dir", type=str, default="data/fineweb_gpt2", help="Data directory")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size")
    ap.add_argument("--block_size", type=int, default=128, help="Context length")
    ap.add_argument("--out_dir", type=str, default="validate_out", help="Output directory")
    ap.add_argument("--dry_run", action="store_true", help="Show predictions without training")
    args = ap.parse_args()

    # Load and fit
    print(f"Loading results from {args.results}")
    df = pd.read_csv(args.results)
    print(f"Found {len(df)} runs across {df['flop_budget'].nunique()} FLOP budgets")

    fits = fit_power_laws(df)

    print("\n" + "=" * 60)
    print("FITTED SCALING LAWS")
    print("=" * 60)
    print(f"N_opt(C) = {fits['a_N']:.2e} × C^{fits['alpha_N']:.3f}")
    print(f"L_opt(C) = {fits['a_L']:.2e} × C^{fits['alpha_L']:.3f}")
    print("\nChinchilla expects: α_N ≈ 0.5, α_L ≈ -0.05 to -0.1")
    print(f"Your fits:          α_N = {fits['alpha_N']:.3f}, α_L = {fits['alpha_L']:.3f}")

    print("\n" + "-" * 60)
    print("DATA USED FOR FIT")
    print("-" * 60)
    print(fits["optima"].to_string(index=False))

    # Predict for new budget
    C_new = args.flop_budget
    preds = predict_optimal(fits, C_new)

    print("\n" + "=" * 60)
    print(f"PREDICTIONS FOR C = {C_new:.2e}")
    print("=" * 60)
    print(f"Predicted optimal N: {preds['N_pred']:,.0f} params")
    print(f"Predicted optimal D: {preds['D_pred']:,.0f} tokens")
    print(f"Predicted loss:      {preds['L_pred']:.4f}")

    # Find closest model config
    n_embd, n_layer, n_head = find_model_config(preds["N_pred"], args.dim_mult)

    # Estimate actual params for this config
    # More accurate: vocab_size * n_embd + n_layer * (12 * n_embd^2) + ...
    # Rough: 12 * n_layer * n_embd^2
    n_params_approx = 12 * n_layer * n_embd**2

    print("\nClosest model config:")
    print(f"  n_embd:  {n_embd}")
    print(f"  n_layer: {n_layer}")
    print(f"  n_head:  {n_head}")
    print(f"  ~params: {n_params_approx:,}")

    if args.dry_run:
        print("\n[Dry run - skipping training]")
        return

    # Run training
    print("\n" + "=" * 60)
    print("RUNNING VALIDATION TRAINING")
    print("=" * 60)

    L_actual = run_training(
        flop_budget=C_new,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        out_dir=args.out_dir,
    )

    # Report results
    error = L_actual - preds["L_pred"]
    error_pct = abs(error) / preds["L_pred"] * 100

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Predicted loss: {preds['L_pred']:.4f}")
    print(f"Actual loss:    {L_actual:.4f}")
    print(f"Error:          {error:+.4f} ({error_pct:.1f}%)")
    print()

    if error_pct < 5:
        print("✓ Excellent! Scaling law is highly predictive (<5% error)")
    elif error_pct < 10:
        print("✓ Good! Scaling law is reasonably predictive (<10% error)")
    elif error_pct < 20:
        print("~ Fair. Scaling law gives rough predictions (10-20% error)")
    else:
        print("✗ Poor. Scaling law may need more data points or better fits")


if __name__ == "__main__":
    main()
