"""
simulation_main.py — Main comparison: Baseline vs. V1 vs. V2
=============================================================
Runs all three conditions and produces:
  - Regret curves
  - Spatial overlap
  - Belief error
  - Coverage heatmaps
  - ROI discovery time bar chart

Usage:
  python simulation_main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from core import (
    run_multiple, run_simulation,
    region_to_xy, GRID_SIZE, NO_FLY, ROI_CENTERS, T_FINAL, N_RUNS
)

ALPHA  = 0.2
OUTDIR = "./outputs"
os.makedirs(OUTDIR, exist_ok=True)

COLORS = {
    "baseline": "#4C72B0",
    "v1":       "#DD8452",
    "v2":       "#55A868",
}
LABELS = {
    "baseline": "Baseline (α=0)",
    "v1":       f"Coord V1 (α={ALPHA})",
    "v2":       f"Coord V2 (α={ALPHA})",
}


def smooth(x, w=30):
    return np.convolve(x, np.ones(w)/w, mode='same')


def plot_regret(results):
    fig, ax = plt.subplots(figsize=(8, 4))
    timesteps = np.arange(1, T_FINAL + 1)
    for mode in ["baseline", "v1", "v2"]:
        mu, std = results[mode]["regret"]
        ax.plot(timesteps, mu, color=COLORS[mode], lw=2, label=LABELS[mode])
        ax.fill_between(timesteps, mu-std, mu+std, alpha=0.15, color=COLORS[mode])
    ax.set_xlabel("Timestep k", fontsize=12)
    ax.set_ylabel("Time-Averaged Regret (Eq. 5)", fontsize=12)
    ax.set_title("Regret: Baseline vs. Coordinated Ergodic Coverage", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_regret.png", dpi=150)
    plt.close()
    print("Saved: fig_regret.png")


def plot_overlap_belief(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    timesteps = np.arange(1, T_FINAL + 1)

    # Belief error
    ax = axes[0]
    for mode in ["baseline", "v1", "v2"]:
        mu, std = results[mode]["belief_err"]
        ax.plot(timesteps, mu, color=COLORS[mode], lw=2, label=LABELS[mode])
        ax.fill_between(timesteps, mu-std, mu+std, alpha=0.15, color=COLORS[mode])
    ax.set_xlabel("Timestep k", fontsize=12)
    ax.set_ylabel("Belief Error  ||ρ̄ - ρ*||₁  (Eq. 7)", fontsize=12)
    ax.set_title("Belief Error over Time", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)

    # Spatial overlap
    ax = axes[1]
    for mode in ["baseline", "v1", "v2"]:
        mu, std = results[mode]["overlap"]
        ax.plot(timesteps, smooth(mu), color=COLORS[mode], lw=2, label=LABELS[mode])
        ax.fill_between(timesteps, smooth(mu-std), smooth(mu+std), alpha=0.15, color=COLORS[mode])
    ax.set_xlabel("Timestep k", fontsize=12)
    ax.set_ylabel("Spatial Overlap (fraction of UAVs co-located)", fontsize=12)
    ax.set_title("Spatial Overlap: Do UAVs Spread Out?", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 0.5)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_belief_overlap.png", dpi=150)
    plt.close()
    print("Saved: fig_belief_overlap.png")


def plot_heatmaps(results):
    """Show true target vs. final empirical coverage for all three conditions."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    def to_grid(dist_vec, accessible):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for i, r in enumerate(accessible):
            x, y = region_to_xy(r)
            grid[x, y] = dist_vec[i]
        return grid

    accessible = results["baseline"]["last_res"]["accessible"]
    rho_star   = results["baseline"]["last_res"]["rho_star"]

    grids = [
        (to_grid(rho_star, accessible), "True Target ρ*"),
        (to_grid(np.mean(results["baseline"]["last_res"]["rho_hat"], axis=0), accessible), "Baseline ρ̂"),
        (to_grid(np.mean(results["v1"]["last_res"]["rho_hat"], axis=0), accessible), f"V1 ρ̂ (α={ALPHA})"),
        (to_grid(np.mean(results["v2"]["last_res"]["rho_hat"], axis=0), accessible), f"V2 ρ̂ (α={ALPHA})"),
    ]

    for ax, (grid, title) in zip(axes, grids):
        im = ax.imshow(grid, cmap="YlOrRd", vmin=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Column"); ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, fraction=0.046)
        for (nx_, ny_) in NO_FLY:
            ax.add_patch(plt.Rectangle((ny_-0.5, nx_-0.5), 1, 1,
                         fill=True, color='black', alpha=0.7))
        for (rx, ry) in ROI_CENTERS:
            ax.plot(ry, rx, 'b+', markersize=10, markeredgewidth=2)

    plt.suptitle("Coverage Heatmaps: Target vs. Empirical Visitation", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_heatmaps.png")


def plot_roi_discovery(results):
    """
    Bar chart: mean time to discover each ROI for baseline, v1, v2.
    Also prints time to discover the LAST ROI (worst-case coverage time).
    """
    roi_labels = [f"ROI {i+1}\n({ROI_CENTERS[i]})" for i in range(len(ROI_CENTERS))]
    x = np.arange(len(ROI_CENTERS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, mode in enumerate(["baseline", "v1", "v2"]):
        roi_times = results[mode]["roi_times"]
        means = [roi_times[k][0] for k in sorted(roi_times.keys())]
        stds  = [roi_times[k][1] for k in sorted(roi_times.keys())]
        ax.bar(x + (i-1)*width, means, width, yerr=stds, capsize=4,
               color=COLORS[mode], label=LABELS[mode], alpha=0.85)

    ax.set_xlabel("Region of Interest", fontsize=12)
    ax.set_ylabel("Timestep of First Discovery", fontsize=12)
    ax.set_title("ROI Discovery Time: Baseline vs. V1 vs. V2", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_roi_discovery.png", dpi=150)
    plt.close()
    print("Saved: fig_roi_discovery.png")

    # Print summary
    print("\n── ROI Discovery Summary ──")
    for mode in ["baseline", "v1", "v2"]:
        times = results[mode]["roi_times"]
        last_roi = max(times[k][0] for k in times)
        mean_all = np.mean([times[k][0] for k in times])
        print(f"  {LABELS[mode]:25s}  mean: {mean_all:.1f}  last ROI: {last_roi:.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running all three conditions...")
    print("=" * 60)

    results = {}
    for mode in ["baseline", "v1", "v2"]:
        alpha_val = ALPHA if mode != "baseline" else 0.0
        print(f"\n  [{mode.upper()}]  alpha={alpha_val}")
        results[mode] = run_multiple(mode=mode, alpha=alpha_val, n_runs=N_RUNS)

    print("\nGenerating figures...")
    plot_regret(results)
    plot_overlap_belief(results)
    plot_heatmaps(results)
    plot_roi_discovery(results)

    print(f"\nAll figures saved to {OUTDIR}/")
