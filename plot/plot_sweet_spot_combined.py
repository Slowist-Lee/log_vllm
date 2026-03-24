
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PREFILL_INPUT = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_prefill_summary.csv"
DEFAULT_DECODE_INPUT = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_decode_summary.csv"
DEFAULT_FIGURE = PROJECT_ROOT / "log" / "sweet_spot_combined_new.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "log" / "sweet_spot_combined_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sweet spot by phase from Prefill and Decode summary CSVs",
    )
    parser.add_argument(
        "--prefill-input",
        type=Path,
        default=DEFAULT_PREFILL_INPUT,
        help="Path to prefill summary CSV",
    )
    parser.add_argument(
        "--decode-input",
        type=Path,
        default=DEFAULT_DECODE_INPUT,
        help="Path to decode summary CSV",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help="Path to output figure",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to output sweet spot summary csv",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tpj",
        choices=["tpj", "throughput_tps", "avg_power_w", "total_energy_j", "j_per_token"],
        help="Metric to optimize for (find maximum)",
    )
    parser.add_argument(
        "--min-throughput-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional throughput constraint in [0,1]. "
            "If > 0, candidates must satisfy throughput >= ratio * max_throughput_of_phase"
        ),
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Current columns: {list(df.columns)}")


def load_data(prefill_path: Path, decode_path: Path) -> pd.DataFrame:
    inputs = [
        ("Prefill", prefill_path),
        ("Decode", decode_path),
    ]

    dfs: list[pd.DataFrame] = []
    missing_files: list[Path] = []

    for phase_name, csv_path in inputs:
        if not csv_path.exists():
            missing_files.append(csv_path)
            continue

        phase_df = pd.read_csv(csv_path)
        require_columns(
            phase_df,
            ["frequency_mhz", "tpj", "throughput_tps", "duration_s", "avg_power_w"],
        )

        phase_df = phase_df.copy()
        # Force source phase label so each input contributes exactly one phase.
        phase_df["phase"] = phase_name
        dfs.append(phase_df)

    if missing_files:
        missing_text = "\n".join(str(p) for p in missing_files)
        raise FileNotFoundError(
            "Missing required sweet spot input CSVs:\n"
            f"{missing_text}"
        )

    df = pd.concat(dfs, ignore_index=True)
    
    df = df.copy()
    df["phase"] = df["phase"].astype(str)
    df["frequency_mhz"] = pd.to_numeric(df["frequency_mhz"], errors="coerce")
    df["tpj"] = pd.to_numeric(df["tpj"], errors="coerce")
    df["throughput_tps"] = pd.to_numeric(df["throughput_tps"], errors="coerce")
    df["duration_s"] = pd.to_numeric(df.get("duration_s"), errors="coerce")
    df["avg_power_w"] = pd.to_numeric(df.get("avg_power_w"), errors="coerce")
    df["total_energy_j"] = pd.to_numeric(df.get("total_energy_j"), errors="coerce")
    df["j_per_token"] = pd.to_numeric(df.get("j_per_token"), errors="coerce")
    
    df = df.dropna(subset=["phase", "frequency_mhz", "tpj", "throughput_tps"])
    return df


def pick_sweet_spot(
    phase_df: pd.DataFrame, 
    metric: str, 
    min_throughput_ratio: float
) -> tuple[pd.Series, float]:
    max_tps = float(phase_df["throughput_tps"].max())
    threshold_tps = max_tps * min_throughput_ratio

    if min_throughput_ratio > 0:
        candidates = phase_df[phase_df["throughput_tps"] >= threshold_tps]
        if candidates.empty:
            candidates = phase_df
    else:
        candidates = phase_df

    # For metrics like power and energy, we want minimum; for tpj and throughput, maximum
    if metric in ["avg_power_w", "total_energy_j", "j_per_token"]:
        best_idx = candidates[metric].idxmin()
    else:
        best_idx = candidates[metric].idxmax()
    
    return phase_df.loc[best_idx], threshold_tps


def plot_sweet_spot(
    df: pd.DataFrame,
    figure_path: Path,
    summary_path: Path,
    metric: str,
    min_throughput_ratio: float,
) -> None:
    colors = {
        "Prefill": "#C44E52",
        "Decode": "#4C72B0",
    }
    
    metric_labels = {
        "tpj": "TPJ (tokens per joule)",
        "throughput_tps": "Throughput (tokens/s)",
        "avg_power_w": "Average Power (W)",
        "total_energy_j": "Total Energy (J)",
        "j_per_token": "Energy per Token (J)",
    }
    
    # Determine if we're maximizing or minimizing
    if metric in ["avg_power_w", "total_energy_j", "j_per_token"]:
        optimization_label = "Minimum"
    else:
        optimization_label = "Maximum"

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=180)
    ax_left = axes[0]
    ax_right = axes[1]
    ax_right_twin = ax_right.twinx()

    summary_rows: list[dict] = []
    prefill_df = df[df["phase"] == "Prefill"].sort_values("frequency_mhz").reset_index(drop=True)
    decode_df = df[df["phase"] == "Decode"].sort_values("frequency_mhz").reset_index(drop=True)

    if prefill_df.empty or decode_df.empty:
        raise ValueError("Both Prefill and Decode rows are required for combined plotting.")

    if metric not in prefill_df.columns or metric not in decode_df.columns:
        raise ValueError(f"Metric '{metric}' not found in both Prefill/Decode data.")

    merged = pd.merge(
        prefill_df[["frequency_mhz", "duration_s", "avg_power_w"]],
        decode_df[["frequency_mhz", "duration_s", "avg_power_w"]],
        on="frequency_mhz",
        how="inner",
        suffixes=("_prefill", "_decode"),
    ).sort_values("frequency_mhz")

    if merged.empty:
        raise ValueError("No frequency intersection between Prefill and Decode rows.")

    # 左图：时延堆叠 + 功耗曲线
    ax_left.bar(
        merged["frequency_mhz"],
        merged["duration_s_prefill"],
        width=34,
        color=colors["Prefill"],
        label="Prefill duration",
    )
    ax_left.bar(
        merged["frequency_mhz"],
        merged["duration_s_decode"],
        width=34,
        bottom=merged["duration_s_prefill"],
        color=colors["Decode"],
        label="Decode duration",
    )

    ax_left_twin = ax_left.twinx()
    ax_left_twin.plot(
        merged["frequency_mhz"],
        merged["avg_power_w_prefill"],
        marker="o",
        linewidth=1.9,
        color="#8C1C13",
        label="Prefill avg power",
    )
    ax_left_twin.plot(
        merged["frequency_mhz"],
        merged["avg_power_w_decode"],
        marker="o",
        linestyle="--",
        linewidth=1.9,
        color="#1D4E89",
        label="Decode avg power",
    )

    ax_left.set_title("Time Split + Power Curves")
    ax_left.set_xlabel("GPU Frequency (MHz)", labelpad=6)
    ax_left.set_ylabel("Duration (s)")
    ax_left_twin.set_ylabel("Average Power (W)")
    ax_left.set_xticks(merged["frequency_mhz"])
    ax_left.grid(axis="y", alpha=0.25)

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_left_twin.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")

    # 右图：效率双轴（Prefill 左轴，Decode 右轴）
    ax_right.plot(
        prefill_df["frequency_mhz"],
        prefill_df[metric],
        marker="o",
        linewidth=2.2,
        color=colors["Prefill"],
        label=f"Prefill {metric_labels.get(metric, metric)}",
    )
    ax_right_twin.plot(
        decode_df["frequency_mhz"],
        decode_df[metric],
        marker="o",
        linewidth=2.2,
        color=colors["Decode"],
        label=f"Decode {metric_labels.get(metric, metric)}",
    )

    # 右图双轴等跨度，仅上下平移
    p_min = float(prefill_df[metric].min())
    p_max = float(prefill_df[metric].max())
    d_min = float(decode_df[metric].min())
    d_max = float(decode_df[metric].max())
    p_span = p_max - p_min
    d_span = d_max - d_min
    common_span = max(p_span, d_span, 1e-6)
    padded_span = common_span * 1.12
    p_mid = (p_max + p_min) / 2.0
    d_mid = (d_max + d_min) / 2.0
    ax_right.set_ylim(p_mid - padded_span / 2.0, p_mid + padded_span / 2.0)
    ax_right_twin.set_ylim(d_mid - padded_span / 2.0, d_mid + padded_span / 2.0)

    for phase, phase_df, axis in [
        ("Prefill", prefill_df, ax_right),
        ("Decode", decode_df, ax_right_twin),
    ]:
        best_row, threshold_tps = pick_sweet_spot(phase_df, metric, min_throughput_ratio)

        axis.scatter(
            [best_row["frequency_mhz"]],
            [best_row[metric]],
            color="#F28E2B",
            marker="*",
            s=240,
            zorder=6,
            label=f"{phase} Sweet Spot",
        )

        note = (
            f"{phase} Sweet Spot\n"
            f"{int(best_row['frequency_mhz'])} MHz\n"
            f"{metric}={best_row[metric]:.4f}"
        )
        if min_throughput_ratio > 0:
            note += f"\nTPS>={threshold_tps:.2f}"

        axis.annotate(
            note,
            (best_row["frequency_mhz"], best_row[metric]),
            textcoords="offset points",
            xytext=(8, 10),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.9},
        )

        summary_rows.append(
            {
                "phase": phase,
                "sweet_spot_frequency_mhz": int(best_row["frequency_mhz"]),
                f"sweet_spot_{metric}": float(best_row[metric]),
                "sweet_spot_throughput_tps": float(best_row["throughput_tps"]),
                "sweet_spot_avg_power_w": float(best_row["avg_power_w"]),
                "min_throughput_ratio": float(min_throughput_ratio),
                "throughput_threshold_tps": float(threshold_tps),
            }
        )

    ax_right.set_title("Phase Efficiency (Dual Axis, Matched Scale Span)")
    ax_right.set_xlabel("GPU Frequency (MHz)", labelpad=6)
    ax_right.set_ylabel(f"Prefill {metric_labels.get(metric, metric)}", color=colors["Prefill"])
    ax_right_twin.set_ylabel(f"Decode {metric_labels.get(metric, metric)}", color=colors["Decode"])
    ax_right.set_xticks(sorted(set(prefill_df["frequency_mhz"]).intersection(set(decode_df["frequency_mhz"]))))
    ax_right.grid(axis="both", alpha=0.25)

    r1, rl1 = ax_right.get_legend_handles_labels()
    r2, rl2 = ax_right_twin.get_legend_handles_labels()
    ax_right.legend(r1 + r2, rl1 + rl2, frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"{optimization_label} {metric_labels.get(metric, metric)} by Phase", 
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(figure_path)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{metric_labels.get(metric, metric)} sweet spot summary:")
    print(summary_df.to_string(index=False))
    print(f"\nFigure saved to: {figure_path}")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_throughput_ratio <= 1.0:
        raise ValueError("--min-throughput-ratio must be within [0, 1].")

    df = load_data(args.prefill_input, args.decode_input)
    plot_sweet_spot(
        df=df,
        figure_path=args.output_figure,
        summary_path=args.output_summary,
        metric=args.metric,
        min_throughput_ratio=args.min_throughput_ratio,
    )


if __name__ == "__main__":
    main()
