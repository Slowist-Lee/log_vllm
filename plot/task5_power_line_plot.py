"""
task5_power_line_plot.py
Plot Avg Power vs GPU Frequency from ./log/task5_heatmap/summary.csv.
Each batch size is shown as a different colored line.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns


def load_summary(log_dir: Path) -> pd.DataFrame:
    csv_path = log_dir / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found at: {csv_path}\n"
            "Run task5_heatmap_run.sh first to generate the data."
        )

    df = pd.read_csv(csv_path)
    required = ["freq_mhz", "batch_size", "avg_power_w"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv is missing columns: {missing}")

    return df


def prepare_power_table(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate repeated measurements by mean power.
    grouped = (
        df.groupby(["freq_mhz", "batch_size"], as_index=False)["avg_power_w"]
        .mean()
    )

    pivot = grouped.pivot(index="freq_mhz", columns="batch_size", values="avg_power_w")
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def plot_power_lines(pivot: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for batch_size in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[batch_size],
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=f"Batch {batch_size}",
        )

    ax.set_title("Avg Power vs GPU Frequency", fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("GPU Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Avg Power (W)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.legend(title="Batch Size", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[plot] Line chart saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Avg Power vs GPU Frequency with one line per batch size."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./log/task5_heatmap",
        help="Directory containing summary.csv (default: ./log/task5_heatmap)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <log-dir>/task5_power_vs_freq.png)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else log_dir / "task5_power_vs_freq.png"

    df = load_summary(log_dir)
    print(f"[plot] Loaded {len(df)} rows from {log_dir / 'summary.csv'}")

    pivot = prepare_power_table(df)
    print(f"[plot] Frequencies: {list(pivot.index)}")
    print(f"[plot] Batch sizes: {list(pivot.columns)}")

    plot_power_lines(pivot, out_path)


if __name__ == "__main__":
    main()
