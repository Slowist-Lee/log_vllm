"""
plot_task4_bs16_power.py
Plot avg power vs frequency for bs=16 from a CSV file.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = ["frequency_mhz", "batch_size", "avg_power_w"]


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df


def prepare_bs16(df: pd.DataFrame) -> pd.DataFrame:
    bs16 = df[df["batch_size"] == 16].copy()
    if bs16.empty:
        raise ValueError("No rows found for batch_size == 16")

    bs16 = bs16.sort_values("frequency_mhz", ascending=True)
    return bs16


def plot_line(df_bs16: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

    ax.plot(
        df_bs16["frequency_mhz"],
        df_bs16["avg_power_w"],
        marker="o",
        linewidth=2.2,
        markersize=6,
        color="#1f77b4",
        label="BS=16",
    )

    for x, y in zip(df_bs16["frequency_mhz"], df_bs16["avg_power_w"]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax.set_title("Avg Power vs GPU Frequency (BS=16)", fontsize=14, fontweight="bold")
    ax.set_xlabel("GPU Frequency (MHz)", fontsize=11)
    ax.set_ylabel("Average Power (W)", fontsize=11)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[plot] Saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bs=16 avg power line chart from CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="./log/task4_bs16_avg_power_vs_freq.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./log/task4_bs16_avg_power_vs_freq.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    df_bs16 = prepare_bs16(df)
    plot_line(df_bs16, out_path)


if __name__ == "__main__":
    main()
