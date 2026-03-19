
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PREFILL_INPUT = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_prefill_summary.csv"
DEFAULT_DECODE_INPUT = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_decode_summary.csv"
DEFAULT_E2E_INPUT = PROJECT_ROOT / "log" / "sweet_spot" / "sweet_spot_e2e_batch16_summary.csv"
DEFAULT_FIGURE = PROJECT_ROOT / "log" / "sweet_spot_combined_new.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "log" / "sweet_spot_combined_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sweet spot by phase from Prefill and E2E summary CSVs",
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
        "--e2e-input",
        type=Path,
        default=DEFAULT_E2E_INPUT,
        help="Path to e2e summary CSV",
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


def load_data(prefill_path: Path, decode_path: Path, e2e_path: Path) -> pd.DataFrame:
    inputs = [
        ("Prefill", prefill_path),
        ("Decode", decode_path),
        ("E2E", e2e_path),
    ]

    dfs: list[pd.DataFrame] = []
    missing_files: list[Path] = []

    for phase_name, csv_path in inputs:
        if not csv_path.exists():
            missing_files.append(csv_path)
            continue

        phase_df = pd.read_csv(csv_path)
        require_columns(phase_df, ["frequency_mhz", "tpj", "throughput_tps"])

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
    phases = ["Prefill", "Decode", "E2E"]

    colors = {
        "Prefill": "#C44E52",
        "Decode": "#4C72B0",
        "E2E": "#55A868",
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

    n = len(phases)
    fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 5), dpi=180)
    if n == 1:
        axes = [axes]

    summary_rows: list[dict] = []

    for ax, phase in zip(axes, phases):
        phase_df = df[df["phase"] == phase].sort_values("frequency_mhz").reset_index(drop=True)
        if phase_df.empty:
            raise ValueError(f"No rows found for phase: {phase}")
        
        if metric not in phase_df.columns:
            print(f"Warning: {metric} not found in {phase} data")
            continue
        
        best_row, threshold_tps = pick_sweet_spot(phase_df, metric, min_throughput_ratio)

        color = colors.get(phase, "#333333")
        ax.plot(
            phase_df["frequency_mhz"],
            phase_df[metric],
            marker="o",
            linewidth=2.2,
            color=color,
            label=metric_labels.get(metric, metric),
        )

        ax.scatter(
            [best_row["frequency_mhz"]],
            [best_row[metric]],
            color="#F28E2B",
            marker="*",
            s=240,
            zorder=6,
            label="Sweet Spot",
        )

        note = (
            f"Sweet Spot\n"
            f"{int(best_row['frequency_mhz'])} MHz\n"
            f"{metric}={best_row[metric]:.4f}"
        )
        if min_throughput_ratio > 0:
            note += f"\nTPS>={threshold_tps:.2f}"

        ax.annotate(
            note,
            (best_row["frequency_mhz"], best_row[metric]),
            textcoords="offset points",
            xytext=(8, 10),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.9},
        )

        ax.set_xlabel("GPU Frequency (MHz)", labelpad=6)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.grid(axis="both", alpha=0.25)
        ax.set_xticks(phase_df["frequency_mhz"])
        ax.legend(frameon=False)
        ax.text(
            0.5,
            -0.30,
            f"{phase} {optimization_label} {metric_labels.get(metric, metric)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

        summary_rows.append(
            {
                "phase": phase,
                "sweet_spot_frequency_mhz": int(best_row["frequency_mhz"]),
                f"sweet_spot_{metric}": float(best_row[metric]),
                "sweet_spot_throughput_tps": float(best_row["throughput_tps"]),
                "min_throughput_ratio": float(min_throughput_ratio),
                "throughput_threshold_tps": float(threshold_tps),
            }
        )

    fig.suptitle(f"{optimization_label} {metric_labels.get(metric, metric)} by Phase", 
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))

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

    df = load_data(args.prefill_input, args.decode_input, args.e2e_input)
    plot_sweet_spot(
        df=df,
        figure_path=args.output_figure,
        summary_path=args.output_summary,
        metric=args.metric,
        min_throughput_ratio=args.min_throughput_ratio,
    )


if __name__ == "__main__":
    main()
