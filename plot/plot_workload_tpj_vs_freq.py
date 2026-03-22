import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


WORKLOAD_ORDER = ["SS", "SL", "LS", "LL"]


def load_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = [
        "workload",
        "frequency_mhz",
        "tpj",
        "duration_s",
        "slo_s",
        "slo_met_rate",
        "input_tokens_target",
        "output_tokens_target",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_single_workload(df: pd.DataFrame, workload: str, out_dir: Path) -> dict:
    wdf = df[df["workload"] == workload].copy().sort_values("frequency_mhz")
    if wdf.empty:
        return {}

    best = wdf.loc[wdf["tpj"].idxmax()]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)

    ax.plot(
        wdf["frequency_mhz"],
        wdf["tpj"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="TPJ",
    )

    ax.scatter(
        [best["frequency_mhz"]],
        [best["tpj"]],
        marker="*",
        s=220,
        color="#e4572e",
        zorder=5,
        label=f"Sweet Spot: {int(best['frequency_mhz'])} MHz",
    )

    in_len = int(best["input_tokens_target"])
    out_len = int(best["output_tokens_target"])
    ax.set_title(f"{workload} Workload (Input={in_len}, Output={out_len})", fontweight="bold")
    ax.set_xlabel("GPU Frequency (MHz)")
    ax.set_ylabel("TPJ (tokens / joule)")
    ax.legend(frameon=True)

    # 用红叉标记 SLO 不满足点，方便看到低频超时风险
    fails = wdf[wdf["slo_met_rate"] < 1.0]
    if not fails.empty:
        ax.scatter(
            fails["frequency_mhz"],
            fails["tpj"],
            marker="x",
            s=90,
            color="#d62728",
            label="SLO not fully met",
        )
        ax.legend(frameon=True)

    out_file = out_dir / f"workload_{workload.lower()}_freq_vs_tpj.png"
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_file}")

    return {
        "workload": workload,
        "input_tokens": in_len,
        "output_tokens": out_len,
        "sweet_spot_freq_mhz": int(best["frequency_mhz"]),
        "sweet_spot_tpj": float(best["tpj"]),
        "sweet_spot_duration_s": float(best["duration_s"]),
        "slo_s": float(best["slo_s"]),
        "slo_met_all_freq": int((wdf["slo_met_rate"] >= 1.0).all()),
        "min_freq_slo_met": float(wdf[wdf["slo_met_rate"] >= 1.0]["frequency_mhz"].min())
        if (wdf["slo_met_rate"] >= 1.0).any()
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frequency vs TPJ for SS/SL/LS/LL workloads")
    parser.add_argument(
        "--csv",
        type=str,
        default="./log/workload_tpj_freq_summary.csv",
        help="Input summary CSV from workload_tpj_freq_scan.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./log/workload_tpj_plots",
        help="Output directory for per-workload figures",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="./log/workload_tpj_sweet_spot_summary.csv",
        help="Output CSV for sweet-spot and SLO summary",
    )
    args = parser.parse_args()

    df = load_summary(Path(args.csv))
    out_dir = Path(args.out_dir)
    ensure_output_dir(out_dir)

    rows = []
    for wl in WORKLOAD_ORDER:
        item = plot_single_workload(df, wl, out_dir)
        if item:
            rows.append(item)

    if not rows:
        raise ValueError("No workload data found for plotting.")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.summary_out, index=False)

    print("\n[sweet spot summary]")
    print(summary_df.to_string(index=False))
    print(f"\n[summary] Saved: {args.summary_out}")


if __name__ == "__main__":
    main()
