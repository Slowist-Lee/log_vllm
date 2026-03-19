#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "log" / "log_03_16_2" / "idle_power_log.csv"
DEFAULT_FIGURE_DIR = ROOT / "log" / "idle_power_plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot idle power log data",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to idle_power_log.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory to save output figures",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["timestamp", "time_offset", "power_w", "energy_j", "gpu_clock_mhz", "mem_clock_mhz"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    for col in ["power_w", "energy_j", "gpu_clock_mhz", "mem_clock_mhz", "time_offset", "util_gpu_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["power_w", "time_offset"])
    return df


def plot_idle_power(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    avg_power = df["power_w"].mean()
    max_power = df["power_w"].max()
    min_power = df["power_w"].min()
    std_power = df["power_w"].std()
    
    total_energy = df["energy_j"].sum()
    total_duration = df["time_offset"].max()

    print(f"\n{'='*60}")
    print(f"Power Statistics:")
    print(f"{'='*60}")
    print(f"Average Power: {avg_power:.2f} W")
    print(f"Max Power: {max_power:.2f} W")
    print(f"Min Power: {min_power:.2f} W")
    print(f"Std Dev: {std_power:.4f} W")
    print(f"Total Duration: {total_duration:.2f} s")
    print(f"Total Energy: {total_energy:.2f} J")
    print(f"{'='*60}\n")

    # Figure 1: Power over time
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    ax.plot(
        df["time_offset"],
        df["power_w"],
        linewidth=1.5,
        color="#4C72B0",
        alpha=0.8,
        label="Instantaneous Power",
    )
    ax.axhline(y=avg_power, color="#C44E52", linestyle="--", linewidth=2, label=f"Average Power ({avg_power:.2f} W)")
    ax.fill_between(
        df["time_offset"],
        avg_power - std_power,
        avg_power + std_power,
        alpha=0.2,
        color="#C44E52",
        label=f"±1σ ({std_power:.4f} W)",
    )
    
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.set_title("Idle Power Over Time", fontsize=14, fontweight="bold")
    ax.grid(axis="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    
    fig.tight_layout()
    power_fig = output_dir / "idle_power_time.png"
    fig.savefig(power_fig)
    plt.close(fig)
    print(f"Figure saved: {power_fig}")

    # Figure 2: Energy accumulation
    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    cumulative_energy = df["energy_j"].cumsum()
    ax.plot(
        df["time_offset"],
        cumulative_energy,
        linewidth=2,
        color="#55A868",
        label="Cumulative Energy",
    )
    ax.fill_between(
        df["time_offset"],
        0,
        cumulative_energy,
        alpha=0.2,
        color="#55A868",
    )
    
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Cumulative Energy (J)", fontsize=12)
    ax.set_title("Energy Accumulation Over Time", fontsize=14, fontweight="bold")
    ax.grid(axis="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    
    fig.tight_layout()
    energy_fig = output_dir / "idle_energy_accumulation.png"
    fig.savefig(energy_fig)
    plt.close(fig)
    print(f"Figure saved: {energy_fig}")

    # Figure 3: Power distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    
    # Histogram
    axes[0].hist(df["power_w"], bins=50, color="#4C72B0", alpha=0.7, edgecolor="black")
    axes[0].axvline(x=avg_power, color="#C44E52", linestyle="--", linewidth=2, label=f"Mean: {avg_power:.2f} W")
    axes[0].axvline(x=df["power_w"].median(), color="#F28E2B", linestyle="--", linewidth=2, label=f"Median: {df['power_w'].median():.2f} W")
    axes[0].set_xlabel("Power (W)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Power Distribution", fontsize=12, fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.3)
    
    # Box plot
    box_data = [df["power_w"]]
    bp = axes[1].boxplot(box_data, vert=True, patch_artist=True, labels=["Power (W)"])
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][0].set_alpha(0.7)
    axes[1].set_ylabel("Power (W)", fontsize=11)
    axes[1].set_title("Power Statistics", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    dist_fig = output_dir / "idle_power_distribution.png"
    fig.savefig(dist_fig)
    plt.close(fig)
    print(f"Figure saved: {dist_fig}")

    # Figure 4: GPU/Memory Clock and Power correlation
    if "gpu_clock_mhz" in df.columns and "mem_clock_mhz" in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=180)
        
        # Top: GPU/Memory Clock
        ax1 = axes[0]
        ax1.plot(df["time_offset"], df["gpu_clock_mhz"], label="GPU Clock", linewidth=1.5, color="#4C72B0")
        ax1.plot(df["time_offset"], df["mem_clock_mhz"], label="Memory Clock", linewidth=1.5, color="#55A868")
        ax1.set_ylabel("Clock Frequency (MHz)", fontsize=11)
        ax1.set_title("GPU and Memory Clock Frequencies", fontsize=12, fontweight="bold")
        ax1.grid(axis="both", alpha=0.3)
        ax1.legend(frameon=False)
        
        # Bottom: Power
        ax2 = axes[1]
        ax2.plot(df["time_offset"], df["power_w"], linewidth=1.5, color="#C44E52", label="Power")
        ax2.fill_between(df["time_offset"], df["power_w"], alpha=0.3, color="#C44E52")
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.set_ylabel("Power (W)", fontsize=11)
        ax2.set_title("Power Consumption", fontsize=12, fontweight="bold")
        ax2.grid(axis="both", alpha=0.3)
        ax2.legend(frameon=False)
        
        fig.tight_layout()
        clock_fig = output_dir / "idle_clock_power.png"
        fig.savefig(clock_fig)
        plt.close(fig)
        print(f"Figure saved: {clock_fig}")

    # Figure 5: Time interval vs Power (to check for patterns)
    if "time_interval" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
        
        # Scatter plot
        axes[0].scatter(df["time_interval"], df["power_w"], alpha=0.5, s=20, color="#4C72B0")
        axes[0].set_xlabel("Time Interval (s)", fontsize=11)
        axes[0].set_ylabel("Power (W)", fontsize=11)
        axes[0].set_title("Power vs Time Interval", fontsize=12, fontweight="bold")
        axes[0].grid(alpha=0.3)
        
        # Moving average (if enough data)
        if len(df) > 20:
            window = max(1, len(df) // 50)
            moving_avg = df["power_w"].rolling(window=window, center=True).mean()
            axes[1].plot(df["time_offset"], moving_avg, linewidth=2, color="#C44E52", label=f"Moving Avg (window={window})")
            axes[1].plot(df["time_offset"], df["power_w"], linewidth=1, alpha=0.3, color="#4C72B0", label="Raw")
            axes[1].set_xlabel("Time (s)", fontsize=11)
            axes[1].set_ylabel("Power (W)", fontsize=11)
            axes[1].set_title("Power with Moving Average", fontsize=12, fontweight="bold")
            axes[1].grid(alpha=0.3)
            axes[1].legend(frameon=False)
        
        fig.tight_layout()
        interval_fig = output_dir / "idle_power_pattern.png"
        fig.savefig(interval_fig)
        plt.close(fig)
        print(f"Figure saved: {interval_fig}")

    # Save summary statistics
    summary_stats = {
        "Metric": [
            "Average Power (W)",
            "Max Power (W)",
            "Min Power (W)",
            "Std Dev (W)",
            "Median Power (W)",
            "P95 Power (W)",
            "P99 Power (W)",
            "Total Duration (s)",
            "Total Energy (J)",
        ],
        "Value": [
            f"{avg_power:.4f}",
            f"{max_power:.4f}",
            f"{min_power:.4f}",
            f"{std_power:.4f}",
            f"{df['power_w'].median():.4f}",
            f"{df['power_w'].quantile(0.95):.4f}",
            f"{df['power_w'].quantile(0.99):.4f}",
            f"{total_duration:.4f}",
            f"{total_energy:.4f}",
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_csv = output_dir / "idle_power_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary statistics saved to: {summary_csv}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    plot_idle_power(df=df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
