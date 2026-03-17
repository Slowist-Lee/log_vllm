from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "log/log_03_16/"
FALLBACK_LOG_DIR = ROOT / "log"
TASK4A_CSV = LOG_DIR / "task4a_results.csv"
TASK4B_CSV = LOG_DIR / "task4b_results.csv"
OUTPUT_FIG = ROOT / "task4_bar_charts.png"
OUTPUT_BATCH_TREND_FIG = ROOT / "task4_batch_power_latency.png"
OUTPUT_SWEET_SPOT_FIG = ROOT / "task4_sweet_spot.png"


def resolve_existing_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Task 4 figures with sweet spot annotations.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        help="Directory containing task4a_results.csv and task4b_results.csv",
    )
    parser.add_argument(
        "--throughput-threshold-ratio",
        type=float,
        default=0.95,
        help="Throughput threshold ratio used for constrained sweet spot (default: 0.95)",
    )
    return parser.parse_args()


def find_constrained_sweet_spot(df: pd.DataFrame, threshold_ratio: float) -> tuple[pd.Series, float]:
    max_tps = float(df["throughput_tps"].max())
    threshold_tps = max_tps * threshold_ratio
    candidates = df[df["throughput_tps"] >= threshold_tps]
    if candidates.empty:
        candidates = df
    best_idx = candidates["j_per_token"].idxmin()
    return df.loc[best_idx], threshold_tps


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


def plot_batch_power_latency(df_task4b: pd.DataFrame, output_path: Path) -> None:
    batch_sizes = df_task4b["batch_size"]
    latency_s = df_task4b["duration_s"]

    # 优先使用峰值功率列；若数据未提供峰值，则回退到平均功率。
    if "peak_power_w" in df_task4b.columns:
        power_col = "peak_power_w"
        power_label = "Peak Power"
    elif "max_power_w" in df_task4b.columns:
        power_col = "max_power_w"
        power_label = "Peak Power"
    else:
        power_col = "avg_power_w"
        power_label = "Average Power"

    power_w = df_task4b[power_col]

    fig, ax_left = plt.subplots(figsize=(10, 6), dpi=160)
    ax_right = ax_left.twinx()

    line_power = ax_left.plot(
        batch_sizes,
        power_w,
        marker="o",
        linewidth=2.2,
        color="#C44E52",
        label=f"{power_label} (W)",
    )
    line_latency = ax_right.plot(
        batch_sizes,
        latency_s,
        marker="s",
        linewidth=2.2,
        color="#4C72B0",
        label="Latency (s)",
    )

    ax_left.set_xlabel("Batch Size")
    ax_left.set_ylabel(f"{power_label} (W)", color="#C44E52")
    ax_right.set_ylabel("Latency (s)", color="#4C72B0")
    ax_left.tick_params(axis="y", labelcolor="#C44E52")
    ax_right.tick_params(axis="y", labelcolor="#4C72B0")
    ax_left.grid(axis="both", alpha=0.25)
    ax_left.set_xticks(batch_sizes)

    title_suffix = "(peak power)" if power_label == "Peak Power" else "(avg power used; peak not provided)"
    ax_left.set_title(f"Task 4b: Power and Latency vs Batch Size {title_suffix}", fontsize=13, fontweight="bold")

    handles = line_power + line_latency
    labels = [line.get_label() for line in handles]
    ax_left.legend(handles, labels, loc="upper left", frameon=False)

    for x, y in zip(batch_sizes, power_w):
        ax_left.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    for x, y in zip(batch_sizes, latency_s):
        ax_right.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)


def plot_task4a_sweet_spot(
    prefill_df: pd.DataFrame,
    decode_df: pd.DataFrame,
    output_path: Path,
    threshold_ratio: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)

    for ax, phase_name, phase_df, color in [
        (axes[0], "Prefill", prefill_df, "#C44E52"),
        (axes[1], "Decode", decode_df, "#4C72B0"),
    ]:
        phase_df = phase_df.sort_values("frequency_mhz").reset_index(drop=True)
        max_tps = float(phase_df["throughput_tps"].max())

        # 主曲线：J/token 与吞吐，用双坐标把 trade-off 画在同一张图。
        ax2 = ax.twinx()
        line_j = ax.plot(
            phase_df["frequency_mhz"],
            phase_df["j_per_token"],
            marker="o",
            linewidth=2.2,
            color=color,
            label="J/token",
        )
        line_t = ax2.plot(
            phase_df["frequency_mhz"],
            phase_df["throughput_tps"],
            marker="s",
            linewidth=2.0,
            color="#55A868",
            label="Throughput",
        )

        abs_best = phase_df.loc[phase_df["j_per_token"].idxmin()]
        constrained_best, threshold_tps = find_constrained_sweet_spot(phase_df, threshold_ratio)

        ax.scatter(
            [abs_best["frequency_mhz"]],
            [abs_best["j_per_token"]],
            marker="*",
            s=220,
            color="#F28E2B",
            zorder=6,
            label="Best J/token",
        )
        ax.annotate(
            f"Best J/token\n{int(abs_best['frequency_mhz'])} MHz\n{abs_best['j_per_token']:.4f}",
            (abs_best["frequency_mhz"], abs_best["j_per_token"]),
            textcoords="offset points",
            xytext=(6, -36),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
        )

        ax.scatter(
            [constrained_best["frequency_mhz"]],
            [constrained_best["j_per_token"]],
            marker="D",
            s=70,
            color="#8172B3",
            zorder=6,
            label=f"Best J/token @TPS>={threshold_ratio:.0%} max",
        )
        ax.annotate(
            f"Constrained\n{int(constrained_best['frequency_mhz'])} MHz",
            (constrained_best["frequency_mhz"], constrained_best["j_per_token"]),
            textcoords="offset points",
            xytext=(8, 12),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
        )

        ax2.axhline(threshold_tps, linestyle="--", linewidth=1.0, color="#55A868", alpha=0.6)

        ax.set_title(f"Task 4a {phase_name}: Sweet Spot", fontsize=12, fontweight="bold")
        ax.set_xlabel("GPU Frequency (MHz)")
        ax.set_ylabel("Energy per Token (J/token)", color=color)
        ax2.set_ylabel("Throughput (tokens/s)", color="#55A868")
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor="#55A868")
        ax.grid(axis="both", alpha=0.25)
        ax.set_xticks(phase_df["frequency_mhz"])

        handles = line_j + line_t
        labels = [line.get_label() for line in handles]
        ax.legend(handles, labels, loc="upper left", frameon=False)
        ax.text(
            0.02,
            0.03,
            f"Max TPS={max_tps:.2f}, threshold={threshold_tps:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.8,
        )

    fig.suptitle("Task 4a Sweet Spot Visualization", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir
    task4a_path = resolve_existing_path(log_dir / "task4a_results.csv", FALLBACK_LOG_DIR / "task4a_results.csv")
    task4b_path = resolve_existing_path(log_dir / "task4b_results.csv", FALLBACK_LOG_DIR / "task4b_results.csv")

    df_task4a = read_csv_required(task4a_path)
    df_task4b = read_csv_required(task4b_path)

    require_columns(
        df_task4a,
        "task4a_results.csv",
        ["phase", "frequency_mhz", "throughput_tps", "j_per_token"],
    )
    require_columns(
        df_task4b,
        "task4b_results.csv",
        ["batch_size", "duration_s", "throughput_tps", "j_per_token"],
    )

    df_task4a = coerce_numeric(df_task4a, ["frequency_mhz", "throughput_tps", "j_per_token"])
    numeric_cols_task4b = ["batch_size", "duration_s", "throughput_tps", "j_per_token"]
    if "peak_power_w" in df_task4b.columns:
        numeric_cols_task4b.append("peak_power_w")
    elif "max_power_w" in df_task4b.columns:
        numeric_cols_task4b.append("max_power_w")
    elif "avg_power_w" in df_task4b.columns:
        numeric_cols_task4b.append("avg_power_w")
    else:
        raise ValueError("task4b_results.csv 缺少功率列：需要 peak_power_w / max_power_w / avg_power_w 之一。")
    df_task4b = coerce_numeric(df_task4b, numeric_cols_task4b)

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

    plot_batch_power_latency(df_task4b, OUTPUT_BATCH_TREND_FIG)
    print(f"批大小-功率-延迟趋势图已保存到: {OUTPUT_BATCH_TREND_FIG}")

    plot_task4a_sweet_spot(prefill_df, decode_df, OUTPUT_SWEET_SPOT_FIG, args.throughput_threshold_ratio)
    print(f"sweet spot 图已保存到: {OUTPUT_SWEET_SPOT_FIG}")


if __name__ == "__main__":
    main()