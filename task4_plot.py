from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log"
TASK4A_CSV = LOG_DIR / "task4a_results.csv"
TASK4B_CSV = LOG_DIR / "task4b_results.csv"
OUTPUT_FIG = ROOT / "task4_bar_charts.png"


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, name: str, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}，当前列: {list(df.columns)}")


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=columns)


def plot_grouped_bars(ax, categories, series_a, series_b, labels, colors, title, ylabel):
    width = 0.36
    positions = range(len(categories))
    left_positions = [position - width / 2 for position in positions]
    right_positions = [position + width / 2 for position in positions]

    bars_a = ax.bar(left_positions, series_a, width=width, label=labels[0], color=colors[0])
    bars_b = ax.bar(right_positions, series_b, width=width, label=labels[1], color=colors[1])

    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(category) for category in categories])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    ax.bar_label(bars_a, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars_b, fmt="%.2f", padding=3, fontsize=8)


def plot_single_bars(ax, categories, values, color, title, ylabel):
    bars = ax.bar([str(category) for category in categories], values, color=color, width=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)


def main() -> None:
    df_task4a = read_csv_required(TASK4A_CSV)
    df_task4b = read_csv_required(TASK4B_CSV)

    require_columns(
        df_task4a,
        "task4a_results.csv",
        ["phase", "frequency_mhz", "throughput_tps", "j_per_token"],
    )
    require_columns(
        df_task4b,
        "task4b_results.csv",
        ["batch_size", "throughput_tps", "j_per_token"],
    )

    df_task4a = coerce_numeric(df_task4a, ["frequency_mhz", "throughput_tps", "j_per_token"])
    df_task4b = coerce_numeric(df_task4b, ["batch_size", "throughput_tps", "j_per_token"])

    prefill_df = df_task4a[df_task4a["phase"].astype(str).str.lower() == "prefill"].sort_values("frequency_mhz")
    decode_df = df_task4a[df_task4a["phase"].astype(str).str.lower() == "decode"].sort_values("frequency_mhz")

    common_freqs = sorted(set(prefill_df["frequency_mhz"]).intersection(set(decode_df["frequency_mhz"])))
    if not common_freqs:
        raise ValueError("task4a_results.csv 中找不到同时包含 Prefill 和 Decode 的频率数据。")

    prefill_df = prefill_df[prefill_df["frequency_mhz"].isin(common_freqs)].sort_values("frequency_mhz")
    decode_df = decode_df[decode_df["frequency_mhz"].isin(common_freqs)].sort_values("frequency_mhz")
    df_task4b = df_task4b.sort_values("batch_size")

    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.linewidth"] = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)

    plot_grouped_bars(
        axes[0, 0],
        common_freqs,
        prefill_df["throughput_tps"],
        decode_df["throughput_tps"],
        labels=("Prefill", "Decode"),
        colors=("#C44E52", "#4C72B0"),
        title="Task 4a: Throughput vs GPU Frequency",
        ylabel="Throughput (tokens/s)",
    )
    axes[0, 0].set_xlabel("GPU Frequency (MHz)")

    plot_grouped_bars(
        axes[0, 1],
        common_freqs,
        prefill_df["j_per_token"],
        decode_df["j_per_token"],
        labels=("Prefill", "Decode"),
        colors=("#DD8452", "#55A868"),
        title="Task 4a: Energy per Token vs GPU Frequency",
        ylabel="Energy per Token (J/token)",
    )
    axes[0, 1].set_xlabel("GPU Frequency (MHz)")

    plot_single_bars(
        axes[1, 0],
        df_task4b["batch_size"],
        df_task4b["throughput_tps"],
        color="#8172B3",
        title="Task 4b: Throughput vs Batch Size",
        ylabel="Throughput (tokens/s)",
    )
    axes[1, 0].set_xlabel("Batch Size")

    plot_single_bars(
        axes[1, 1],
        df_task4b["batch_size"],
        df_task4b["j_per_token"],
        color="#64B5CD",
        title="Task 4b: Energy per Token vs Batch Size",
        ylabel="Energy per Token (J/token)",
    )
    axes[1, 1].set_xlabel("Batch Size")

    fig.suptitle("Task 4 Results Overview", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_FIG)
    print(f"柱状图已保存到: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()