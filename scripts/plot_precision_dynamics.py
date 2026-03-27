"""Plot Phase 1 precision dynamics from a W&B training run.

Generates two publication-quality figures:
  figures/precision_dynamics.{pdf,png}  — 3-panel stacked (error / mean / std)
  figures/precision_overlay.pdf         — dual-y overlay (error vs precision mean)

Usage:
    python scripts/plot_precision_dynamics.py --run-id xlxm6d3x
    python scripts/plot_precision_dynamics.py --run-id xlxm6d3x --smoothing 0.95
    python scripts/plot_precision_dynamics.py --run-id xlxm6d3x --project entity/My-Project
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

import wandb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROJECT = "aktuator-ai/Sudoku-extreme-1k-aug-1000 CORAL-v3"
FIGURE_DIR = "figures"

METRICS = [
    "train/prediction_error",
    "train/precision_mean",
    "train/precision_std",
]

# Colorblind-safe palette (Wong 2011)
C_PRED  = "#0072B2"   # blue   — prediction error
C_PMEAN = "#D55E00"   # vermilion — precision mean
C_PSTD  = "#009E73"   # green  — precision std
C_BAND  = "#E69F00"   # amber  — phase-transition band

# NeurIPS single-column formatting
# (single column ≈ 3.5", double ≈ 7.16")
_RC = {
    # Fonts
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
    "mathtext.fontset":    "cm",
    "font.size":           9,
    "axes.titlesize":      9,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    # Lines / axes
    "lines.linewidth":     1.2,
    "axes.linewidth":      0.7,
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
    "xtick.minor.width":   0.5,
    "ytick.minor.width":   0.5,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    # Clean look
    "axes.grid":           False,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "legend.frameon":      False,
    "legend.handlelength": 1.4,
    # Output
    "figure.dpi":          150,   # screen preview
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "pdf.fonttype":        42,    # embed fonts (avoids Type-3 in submission systems)
    "ps.fonttype":         42,
}


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------

def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average.

    Args:
        values: 1-D array of raw metric values.
        alpha:  Smoothing weight in [0, 1).
                0   → identity (no smoothing).
                0.9 → moderate.
                0.99 → very heavy.

    Returns:
        Smoothed array of the same length.
    """
    if alpha <= 0.0 or len(values) == 0:
        return values.copy()
    out = np.empty_like(values, dtype=float)
    out[0] = float(values[0])
    decay = 1.0 - alpha
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + decay * float(values[i])
    return out


# ---------------------------------------------------------------------------
# Phase-transition detection
# ---------------------------------------------------------------------------

def detect_phase_transition(
    steps: np.ndarray,
    smoothed_error: np.ndarray,
) -> tuple:
    """Find the sharpest sustained drop in prediction error.

    Returns:
        (center_step, half_width)  — both as floats in step units.
        Returns (None, None) if the signal is too short.
    """
    if len(smoothed_error) < 10:
        return None, None

    grad = np.gradient(smoothed_error, steps)
    idx_min = int(np.argmin(grad))
    center = float(steps[idx_min])

    # Extend left/right while gradient stays below 10 % of peak steepness
    threshold = 0.10 * grad[idx_min]   # negative value
    left = idx_min
    while left > 0 and grad[left] < threshold:
        left -= 1
    right = idx_min
    while right < len(grad) - 1 and grad[right] < threshold:
        right += 1

    span = float(steps[right]) - float(steps[left])
    # Floor: at least 2 % of total training span so the band is visible
    floor = (float(steps[-1]) - float(steps[0])) * 0.02
    half_width = max(span / 2.0, floor)
    return center, half_width


def _add_band(ax, center, half_width, *, label=None):
    """Shade the phase-transition region on *ax*."""
    if center is None:
        return
    ax.axvspan(
        center - half_width, center + half_width,
        color=C_BAND, alpha=0.22, zorder=0, label=label,
    )


# ---------------------------------------------------------------------------
# W&B data fetching
# ---------------------------------------------------------------------------

def fetch_metrics(run_path: str, keys: list) -> dict:
    """Pull metric histories from W&B.

    Returns:
        {metric_key: (steps_np, values_np)}
        Missing/empty metrics get (empty_array, empty_array) with a warning.
    """
    api = wandb.Api()
    print(f"[wandb] connecting to {run_path!r} …")
    run = api.run(run_path)
    print(f"[wandb] run name: {run.name!r}  state: {run.state}")

    # scan_history streams rows without page-size limits
    rows = list(run.scan_history(keys=["_step"] + keys))
    print(f"[wandb] fetched {len(rows):,} history rows")

    buckets: dict = {k: ([], []) for k in keys}
    for row in rows:
        step = row.get("_step")
        if step is None:
            continue
        for k in keys:
            v = row.get(k)
            if v is not None:
                buckets[k][0].append(int(step))
                buckets[k][1].append(float(v))

    result = {}
    for k in keys:
        raw_steps, raw_vals = buckets[k]
        if not raw_steps:
            warnings.warn(f"No data found for '{k}' — check that Phase 1 was enabled for this run.")
            result[k] = (np.array([], dtype=float), np.array([], dtype=float))
            continue
        idx = np.argsort(raw_steps)
        result[k] = (
            np.array(raw_steps, dtype=float)[idx],
            np.array(raw_vals, dtype=float)[idx],
        )
        print(f"  {k}: {len(raw_steps):,} points  "
              f"[{result[k][0][0]:.0f} … {result[k][0][-1]:.0f} steps]")

    return result


# ---------------------------------------------------------------------------
# X-axis formatter
# ---------------------------------------------------------------------------

def _step_label(ax, steps: np.ndarray) -> str:
    """Apply k-unit formatter when total steps ≥ 5000; return axis label string."""
    if len(steps) == 0:
        return "Training steps"
    if steps[-1] >= 5_000:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
        )
        return "Training steps (×10³)"
    return "Training steps"


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, directory: str, stem: str, formats=("pdf", "png")):
    os.makedirs(directory, exist_ok=True)
    for fmt in formats:
        path = os.path.join(directory, f"{stem}.{fmt}")
        fig.savefig(path, format=fmt)
        print(f"  → {path}")


# ---------------------------------------------------------------------------
# Figure 1: 3-panel stacked
# ---------------------------------------------------------------------------

def plot_3panel(data: dict, smoothing: float, out_dir: str) -> None:
    """Three vertically stacked panels sharing the x-axis."""
    steps_e, raw_e = data["train/prediction_error"]
    steps_m, raw_m = data["train/precision_mean"]
    steps_s, raw_s = data["train/precision_std"]

    if len(steps_e) == 0:
        print("[3-panel] train/prediction_error is empty — skipping.")
        return

    sm_e = ema_smooth(raw_e, smoothing)
    sm_m = ema_smooth(raw_m, smoothing)
    sm_s = ema_smooth(raw_s, smoothing)

    center, half_width = detect_phase_transition(steps_e, sm_e)
    if center is not None:
        print(f"[3-panel] phase transition detected at step ~{center:.0f} "
              f"(±{half_width:.0f})")

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(3.5, 5.5))
    gs = GridSpec(
        3, 1, figure=fig,
        hspace=0.10,          # minimal gap — shared axis ticks hidden
        top=0.96, bottom=0.10, left=0.18, right=0.97,
    )
    ax_e = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1], sharex=ax_e)
    ax_s = fig.add_subplot(gs[2], sharex=ax_e)

    # ── Panel A: prediction error ───────────────────────────────────────────
    ax_e.plot(steps_e, raw_e, color=C_PRED, alpha=0.18, lw=0.7, zorder=1)
    ax_e.plot(steps_e, sm_e,  color=C_PRED, lw=1.5,   zorder=2,
              label="prediction error")
    _add_band(ax_e, center, half_width, label="phase transition")
    ax_e.set_ylabel("Prediction\nerror")
    ax_e.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))
    leg = ax_e.legend(loc="upper right", ncol=1)

    # ── Panel B: precision mean ─────────────────────────────────────────────
    if len(steps_m):
        ax_m.plot(steps_m, raw_m, color=C_PMEAN, alpha=0.18, lw=0.7, zorder=1)
        ax_m.plot(steps_m, sm_m,  color=C_PMEAN, lw=1.5,   zorder=2)
    _add_band(ax_m, center, half_width)
    ax_m.set_ylabel("Precision\nmean")
    ax_m.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

    # ── Panel C: precision std ──────────────────────────────────────────────
    if len(steps_s):
        ax_s.plot(steps_s, raw_s, color=C_PSTD, alpha=0.18, lw=0.7, zorder=1)
        ax_s.plot(steps_s, sm_s,  color=C_PSTD, lw=1.5,   zorder=2)
    _add_band(ax_s, center, half_width)
    ax_s.set_ylabel("Precision\nstd")
    ax_s.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

    # ── Shared x-axis ───────────────────────────────────────────────────────
    plt.setp(ax_e.get_xticklabels(), visible=False)
    plt.setp(ax_m.get_xticklabels(), visible=False)
    xlabel = _step_label(ax_s, steps_e)
    ax_s.set_xlabel(xlabel)

    x0 = float(steps_e[0])
    x1 = float(steps_e[-1])
    for ax in (ax_e, ax_m, ax_s):
        ax.set_xlim(x0, x1)
        ax.tick_params(axis="x", which="both", top=False)

    _save(fig, out_dir, "precision_dynamics", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: overlay (dual y-axis)
# ---------------------------------------------------------------------------

def plot_overlay(data: dict, smoothing: float, out_dir: str) -> None:
    """Single panel: prediction error (left) vs precision mean (right)."""
    steps_e, raw_e = data["train/prediction_error"]
    steps_m, raw_m = data["train/precision_mean"]

    if len(steps_e) == 0 or len(steps_m) == 0:
        print("[overlay] missing data — skipping.")
        return

    sm_e = ema_smooth(raw_e, smoothing)
    sm_m = ema_smooth(raw_m, smoothing)

    center, half_width = detect_phase_transition(steps_e, sm_e)

    fig, ax_l = plt.subplots(figsize=(3.5, 2.5))
    # Override spines for dual-axis: keep right spine for the second axis
    ax_l.spines["right"].set_visible(True)
    ax_r = ax_l.twinx()

    # ── Left y-axis: prediction error ──────────────────────────────────────
    ax_l.plot(steps_e, raw_e, color=C_PRED, alpha=0.18, lw=0.7, zorder=1)
    ln1, = ax_l.plot(steps_e, sm_e, color=C_PRED, lw=1.5, zorder=2,
                     label="Prediction error")
    ax_l.set_ylabel("Prediction error", color=C_PRED)
    ax_l.tick_params(axis="y", labelcolor=C_PRED, direction="in")
    ax_l.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

    # ── Right y-axis: precision mean ────────────────────────────────────────
    ax_r.plot(steps_m, raw_m, color=C_PMEAN, alpha=0.18, lw=0.7, zorder=1)
    ln2, = ax_r.plot(steps_m, sm_m, color=C_PMEAN, lw=1.5, zorder=2,
                     label="Precision mean")
    ax_r.set_ylabel("Precision mean", color=C_PMEAN)
    ax_r.tick_params(axis="y", labelcolor=C_PMEAN, direction="in")
    ax_r.spines["top"].set_visible(False)
    ax_r.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

    # ── Phase-transition band ───────────────────────────────────────────────
    _add_band(ax_l, center, half_width)
    ax_l.spines["top"].set_visible(False)

    # ── Legend ──────────────────────────────────────────────────────────────
    band_patch = mpatches.Patch(color=C_BAND, alpha=0.5, label="Phase transition")
    ax_l.legend(
        handles=[ln1, ln2, band_patch],
        loc="upper center",
        ncol=3,
        fontsize=7,
        handlelength=1.0,
        columnspacing=0.6,
        bbox_to_anchor=(0.5, 1.02),
    )

    # ── X-axis ──────────────────────────────────────────────────────────────
    xlabel = _step_label(ax_l, steps_e)
    ax_l.set_xlabel(xlabel)
    ax_l.set_xlim(float(steps_e[0]), float(steps_e[-1]))
    ax_l.tick_params(axis="x", direction="in")

    fig.tight_layout()
    _save(fig, out_dir, "precision_overlay", formats=("pdf",))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Phase 1 precision dynamics from a W&B run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id", required=True, metavar="ID",
        help="W&B run ID (e.g. xlxm6d3x)",
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT, metavar="ENTITY/PROJECT",
        help="W&B entity/project path",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.95, metavar="ALPHA",
        help="EMA smoothing factor in [0, 1).  0=raw, 0.95=heavy smoothing.",
    )
    parser.add_argument(
        "--figures-dir", default=FIGURE_DIR, metavar="DIR",
        help="Output directory for figures",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if not 0.0 <= args.smoothing < 1.0:
        sys.exit(f"--smoothing must be in [0, 1), got {args.smoothing}")

    plt.rcParams.update(_RC)

    run_path = f"{args.project}/{args.run_id}"
    data = fetch_metrics(run_path, METRICS)

    print(f"\nGenerating figures  (smoothing α={args.smoothing})")
    plot_3panel(data, args.smoothing, args.figures_dir)
    plot_overlay(data, args.smoothing, args.figures_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
