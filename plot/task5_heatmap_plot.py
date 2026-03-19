"""
task5_heatmap_plot.py
Reads ./log/task5_heatmap/summary.csv and produces a 2x3 grid of heatmaps
(GPU Frequency on X-axis, Batch Size on Y-axis) for five metrics:
  1. Throughput      (tok/s)          — higher = better (green)
  2. E2E Latency     (s)              — lower  = better (green when low)
  3. TBT / TPOT      (ms/tok)         — lower  = better
  4. Avg Power       (W)              — informational (yellow→red)
  5. Energy Eff      (tok/J, i.e TPJ) — higher = better (green)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Metric definitions
#   (column, display_title, colormap, annotation_fmt_string)
#   Use calmer, publication-style colormaps:
#     - "YlGnBu" / "YlGnBu_r" for performance metrics
#     - "mako" for informational power map
# ---------------------------------------------------------------------------
METRICS = [
    ("throughput_tps", "Throughput (tok/s)",      "YlGnBu",   "{:.1f}"),
    ("duration_s",     "E2E Latency (s)",         "YlGnBu_r", "{:.2f}"),
    ("tpot_ms",        "TBT / TPOT (ms/tok)",     "YlGnBu_r", "{:.1f}"),
    ("avg_power_w",    "Avg Power (W)",           "mako",     "{:.0f}"),
    ("tpj",            "Energy Eff (tok/J)",      "YlGnBu",   "{:.2f}"),
]

NCOLS = 3
NROWS = 2   # ceil(5 / 3) = 2, last subplot hidden


def load_summary(log_dir: Path) -> pd.DataFrame:
    csv_path = log_dir / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found at: {csv_path}\n"
            "Run task5_heatmap_run.sh first to generate the data."
        )
    df = pd.read_csv(csv_path)
    required = ["freq_mhz", "batch_size", "duration_s", "ttft_s", "tpot_s",
                "avg_power_w", "total_energy_j", "throughput_tps", "tpj"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv is missing columns: {missing}")

    # Derived column: TPOT in milliseconds
    df["tpot_ms"] = df["tpot_s"] * 1000.0
    return df


def build_annot_matrix(pivot: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Return a same-shape DataFrame of formatted annotation strings."""
    # Use Series.map via DataFrame.apply for compatibility across pandas versions.
    return pivot.apply(lambda col: col.map(lambda v: fmt.format(v) if not pd.isna(v) else "N/A"))


def plot_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * 6.2, NROWS * 5.0),
        dpi=150,
    )
    axes_flat = axes.flatten()

    for idx, (col, title, cmap, fmt) in enumerate(METRICS):
        ax = axes_flat[idx]

        # Pivot after transpose:
        #   rows = batch_size (ascending top->bottom)
        #   cols = freq (ascending left->right)
        pivot = df.pivot_table(
            index="batch_size",
            columns="freq_mhz",
            values=col,
            aggfunc="mean",
        )
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        annot = build_annot_matrix(pivot, fmt)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt="",
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"shrink": 0.85, "pad": 0.02},
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("GPU Freq (MHz)", fontsize=10, labelpad=6)
        ax.set_ylabel("Batch Size", fontsize=10, labelpad=6)
        ax.tick_params(axis="both", labelsize=9)

        # Rotate x-axis labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Hide the unused 6th subplot
    for idx in range(len(METRICS), NROWS * NCOLS):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "GPU Frequency x Batch Size Performance Heatmap  (Mistral-7B)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[plot] Heatmap saved: {out_path}")


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a quick text summary of the pivot for each metric."""
    print("\n=== Summary Statistics ===")
    for col, title, _, fmt in METRICS:
        pivot = df.pivot_table(index="batch_size", columns="freq_mhz", values=col, aggfunc="mean")
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        print(f"\n{title}:")
        print(pivot.to_string(float_format=lambda v: fmt.format(v)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GPU Frequency x Batch Size heatmaps from task5 sweep results."
    )
    parser.add_argument(
        "--log-dir", type=str, default="./log/task5_heatmap",
        help="Directory containing summary.csv (default: ./log/task5_heatmap)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output PNG path (default: <log-dir>/task5_heatmap.png)"
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else log_dir / "task5_heatmap.png"

    df = load_summary(log_dir)
    print(f"[plot] Loaded {len(df)} rows from {log_dir / 'summary.csv'}")
    print(f"[plot] Frequencies : {sorted(df['freq_mhz'].unique())}")
    print(f"[plot] Batch sizes : {sorted(df['batch_size'].unique())}")

    print_summary_table(df)
    plot_heatmaps(df, out_path)


if __name__ == "__main__":
    main()
