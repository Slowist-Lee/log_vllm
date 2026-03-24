from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_unified_summary.csv"
DEFAULT_FIGURE = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_ttft_split.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_ttft_split_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TTFT-split sweet spot from unified summary CSV",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to sweet_spot_unified_summary.csv",
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
        help="Path to output summary CSV",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate major points on chart",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Current columns: {list(df.columns)}")


def load_unified_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    require_columns(
        df,
        [
            "phase",
            "frequency_mhz",
            "duration_s",
            "ttft_s",
            "avg_power_w",
            "throughput_tps",
            "j_per_token",
            "tpj",
            "total_energy_j",
        ],
    )

    df = df.copy()
    df["phase"] = df["phase"].astype(str)
    numeric_cols = [
        "frequency_mhz",
        "duration_s",
        "ttft_s",
        "avg_power_w",
        "throughput_tps",
        "j_per_token",
        "tpj",
        "total_energy_j",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["phase"].isin(["Prefill", "Decode"])].dropna(
        subset=["phase", "frequency_mhz", "duration_s", "ttft_s", "tpj"]
    )

    if df.empty:
        raise ValueError("No valid rows found for Prefill/Decode in unified summary.")

    grouped = (
        df.groupby(["phase", "frequency_mhz"], as_index=False)
        .agg(
            duration_s=("duration_s", "mean"),
            ttft_s=("ttft_s", "mean"),
            avg_power_w=("avg_power_w", "mean"),
            throughput_tps=("throughput_tps", "mean"),
            j_per_token=("j_per_token", "mean"),
            tpj=("tpj", "mean"),
            total_energy_j=("total_energy_j", "mean"),
            runs=("phase", "count"),
        )
        .sort_values(["phase", "frequency_mhz"])
        .reset_index(drop=True)
    )
    return grouped


def find_best_rows(df: pd.DataFrame) -> dict[str, pd.Series]:
    best: dict[str, pd.Series] = {}
    for phase in ["Prefill", "Decode"]:
        phase_df = df[df["phase"] == phase]
        if phase_df.empty:
            continue
        best_idx = phase_df["tpj"].idxmax()
        best[phase] = phase_df.loc[best_idx]
    return best


def plot_ttft_split(df: pd.DataFrame, figure_path: Path, annotate: bool) -> None:
    prefill = df[df["phase"] == "Prefill"].sort_values("frequency_mhz")
    decode = df[df["phase"] == "Decode"].sort_values("frequency_mhz")

    merged = pd.merge(
        prefill[["frequency_mhz", "duration_s", "tpj", "avg_power_w"]],
        decode[["frequency_mhz", "duration_s", "tpj", "avg_power_w"]],
        on="frequency_mhz",
        how="inner",
        suffixes=("_prefill", "_decode"),
    ).sort_values("frequency_mhz")

    if merged.empty:
        raise ValueError("No frequency intersection between Prefill and Decode rows.")

    best_rows = find_best_rows(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)

    ax0 = axes[0]
    ax0.bar(
        merged["frequency_mhz"],
        merged["duration_s_prefill"],
        width=34,
        color="#C44E52",
        label="Prefill duration (TTFT split)",
    )
    ax0.bar(
        merged["frequency_mhz"],
        merged["duration_s_decode"],
        width=34,
        bottom=merged["duration_s_prefill"],
        color="#4C72B0",
        label="Decode duration (after TTFT)",
    )

    ax0.set_xlabel("GPU Frequency (MHz)")
    ax0.set_ylabel("Duration (s)")
    ax0.grid(axis="y", alpha=0.25)

    # 左图叠加功耗线（副轴）
    ax0_twin = ax0.twinx()
    ax0_twin.plot(
        merged["frequency_mhz"],
        merged["avg_power_w_prefill"],
        marker="o",
        linewidth=1.8,
        color="#8C1C13",
        label="Prefill avg power",
    )
    ax0_twin.plot(
        merged["frequency_mhz"],
        merged["avg_power_w_decode"],
        marker="o",
        linestyle="--",
        linewidth=1.8,
        color="#1D4E89",
        label="Decode avg power",
    )
    ax0_twin.set_ylabel("Average Power (W)")

    h0, l0 = ax0.get_legend_handles_labels()
    h0t, l0t = ax0_twin.get_legend_handles_labels()
    ax0.legend(h0 + h0t, l0 + l0t, frameon=False, fontsize=8, loc="upper right")

    ax1 = axes[1]
    ax1_twin = ax1.twinx()

    prefill_df = df[df["phase"] == "Prefill"].sort_values("frequency_mhz")
    decode_df = df[df["phase"] == "Decode"].sort_values("frequency_mhz")

    if not prefill_df.empty:
        ax1.plot(
            prefill_df["frequency_mhz"],
            prefill_df["tpj"],
            marker="o",
            linewidth=2.1,
            color="#C44E52",
            label="Prefill TPJ",
        )
    if not decode_df.empty:
        ax1_twin.plot(
            decode_df["frequency_mhz"],
            decode_df["tpj"],
            marker="o",
            linewidth=2.1,
            color="#4C72B0",
            label="Decode TPJ",
        )

    # 两个 y 轴保持相同缩放跨度（max-min 一致），仅做上下平移
    # 同时加少量留白，避免曲线贴边。
    if not prefill_df.empty and not decode_df.empty:
        p_min = float(prefill_df["tpj"].min())
        p_max = float(prefill_df["tpj"].max())
        d_min = float(decode_df["tpj"].min())
        d_max = float(decode_df["tpj"].max())

        p_span = p_max - p_min
        d_span = d_max - d_min
        common_span = max(p_span, d_span, 1e-6)
        padded_span = common_span * 1.12

        p_mid = (p_max + p_min) / 2.0
        d_mid = (d_max + d_min) / 2.0

        ax1.set_ylim(p_mid - padded_span / 2.0, p_mid + padded_span / 2.0)
        ax1_twin.set_ylim(d_mid - padded_span / 2.0, d_mid + padded_span / 2.0)

    for phase, axis, color in [
        ("Prefill", ax1, "#C44E52"),
        ("Decode", ax1_twin, "#4C72B0"),
    ]:
        best = best_rows.get(phase)
        if best is None:
            continue

        axis.scatter(
            [best["frequency_mhz"]],
            [best["tpj"]],
            marker="*",
            s=220,
            color="#F28E2B",
            zorder=6,
        )
        axis.annotate(
            f"{phase} sweet spot\n{int(best['frequency_mhz'])} MHz\nTPJ={best['tpj']:.3f}",
            (best["frequency_mhz"], best["tpj"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.9},
        )

    ax1.set_xlabel("GPU Frequency (MHz)")
    ax1.set_ylabel("Prefill TPJ", color="#C44E52")
    ax1_twin.set_ylabel("Decode TPJ", color="#4C72B0")
    ax1.grid(axis="both", alpha=0.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="best")

    # 子图小标题放到底部
    ax0.text(
        0.5,
        -0.20,
        "TTFT-based Time Split",
        transform=ax0.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="semibold",
    )
    ax1.text(
        0.5,
        -0.20,
        "Energy Per Token by Phase",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="semibold",
    )

    fig.suptitle("Unified Sweet Spot: TTFT-split Prefill/Decode", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def save_summary(df: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    best_rows = find_best_rows(df)

    power_min = float(df["avg_power_w"].min())
    power_max = float(df["avg_power_w"].max())
    power_range = power_max - power_min

    rows: list[dict] = []
    for phase in ["Prefill", "Decode"]:
        best = best_rows.get(phase)
        if best is None:
            continue

        rows.append(
            {
                "phase": phase,
                "sweet_spot_frequency_mhz": int(best["frequency_mhz"]),
                "sweet_spot_tpj": float(best["tpj"]),
                "sweet_spot_throughput_tps": float(best["throughput_tps"]),
                "sweet_spot_j_per_token": float(best["j_per_token"]),
                "sweet_spot_avg_power_w": float(best["avg_power_w"]),
                "sweet_spot_avg_power_norm": (
                    float((best["avg_power_w"] - power_min) / power_range) if power_range > 1e-12 else 0.0
                ),
                "sweet_spot_duration_s": float(best["duration_s"]),
                "runs_aggregated": int(best["runs"]),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def main() -> None:
    args = parse_args()
    df = load_unified_summary(args.input)

    plot_ttft_split(df=df, figure_path=args.output_figure, annotate=args.annotate)
    summary_df = save_summary(df=df, summary_path=args.output_summary)

    print("\nTTFT-split sweet spot summary:")
    print(summary_df.to_string(index=False))
    print(f"\nFigure saved to: {args.output_figure}")
    print(f"Summary saved to: {args.output_summary}")


if __name__ == "__main__":
    main()
