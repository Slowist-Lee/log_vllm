"""
plot_e2e_energy_per_token.py
Reads e2e_ttft_profile.csv and plots instantaneous energy-per-token over time,
with prefill / decode phases shaded (style mirrors the power+freq reference figure).

Instantaneous J/token = power_w(t) / token_throughput(phase)
  - Prefill throughput = prompt_tokens / ttft_s          (parallel prefill)
  - Decode  throughput = total_output_tokens / decode_s  (autoregressive decode)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required_cols = ["time_offset", "power_w", "ttft_s", "total_duration_s",
                     "prompt_tokens", "energy_j"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Backward/forward compatibility: some profiles use output_tokens,
    # while others use total_output_tokens.
    if "total_output_tokens" not in df.columns and "output_tokens" in df.columns:
        df["total_output_tokens"] = df["output_tokens"]

    if "total_output_tokens" not in df.columns:
        raise ValueError("Missing column: total_output_tokens (or alias output_tokens)")

    return df


def compute_j_per_token(df: pd.DataFrame) -> pd.DataFrame:
    ttft_s          = float(df["ttft_s"].iloc[0])
    total_dur_s     = float(df["total_duration_s"].iloc[0])
    prompt_tokens   = int(df["prompt_tokens"].iloc[0])
    output_tokens   = int(df["total_output_tokens"].iloc[0])

    prefill_tps = prompt_tokens  / ttft_s                     # tok/s during prefill
    decode_dur  = max(total_dur_s - ttft_s, 1e-6)
    decode_tps  = output_tokens  / decode_dur                 # tok/s during decode

    is_prefill = df["time_offset"] < ttft_s
    tps = np.where(is_prefill, prefill_tps, decode_tps)

    df = df.copy()
    df["j_per_token_inst"]   = df["power_w"] / tps
    df["j_per_token_smooth"] = (
        df["j_per_token_inst"]
        .rolling(window=7, center=True, min_periods=1)
        .mean()
    )
    return df, ttft_s, total_dur_s, prompt_tokens, output_tokens, prefill_tps, decode_tps


def plot(df: pd.DataFrame, ttft_s: float, total_dur_s: float,
         prompt_tokens: int, output_tokens: int,
         prefill_tps: float, decode_tps: float,
         out_path: Path) -> None:

    t       = df["time_offset"].values
    j_inst  = df["j_per_token_inst"].values
    j_smooth= df["j_per_token_smooth"].values

    t_end = t[-1] + 0.3

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fb")

    # ── phase backgrounds ─────────────────────────────────────────────────────
    ax.axvspan(0,      ttft_s, alpha=0.18, color="#f4a0a0", zorder=0)
    ax.axvspan(ttft_s, t_end,  alpha=0.10, color="#a0c4f4", zorder=0)

    # ── raw trace (faint) ─────────────────────────────────────────────────────
    ax.plot(t, j_inst,   color="#cc3333", alpha=0.22, linewidth=0.9, zorder=2)

    # ── smoothed trace ────────────────────────────────────────────────────────
    ax.plot(t, j_smooth, color="#cc3333", linewidth=2.0, zorder=3,
            label="J/token (smoothed, w=7)")

    # ── TTFT dashed line ──────────────────────────────────────────────────────
    ax.axvline(x=ttft_s, color="black", linestyle="--", linewidth=1.6, zorder=4,
               label=f"TTFT = {ttft_s:.3f} s")

    # ── phase labels (positioned via axes-fraction transform) ─────────────────
    y_label = 0.91
    ax.text(ttft_s * 0.47, y_label, "Prefill Phase",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=12,
            color="#cc3333", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.75, edgecolor="none"))
    ax.text((ttft_s + t_end) * 0.5, y_label, "Decode Phase",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=12,
            color="#2266aa", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.75, edgecolor="none"))

    # ── mean-value annotations ────────────────────────────────────────────────
    prefill_mask = df["time_offset"] < ttft_s
    decode_mask  = ~prefill_mask

    mean_pre = df.loc[prefill_mask, "j_per_token_inst"].mean()
    mean_dec = df.loc[decode_mask,  "j_per_token_inst"].mean()

    ax.axhline(mean_pre, xmin=0, xmax=ttft_s / t_end,
               color="#cc3333", linestyle=":", linewidth=1.3, alpha=0.8, zorder=2)
    ax.axhline(mean_dec, xmin=ttft_s / t_end, xmax=1.0,
               color="#cc3333", linestyle=":", linewidth=1.3, alpha=0.8, zorder=2)

    ax.annotate(f"avg {mean_pre:.3f} J/tok\n({prompt_tokens} input tok, {prefill_tps:.0f} tok/s)",
                xy=(ttft_s * 0.05, mean_pre),
                xytext=(ttft_s * 0.05, mean_pre + (mean_dec - mean_pre) * 0.08),
                fontsize=8.5, color="#993333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                          edgecolor="#cc3333", linewidth=0.8))

    ax.annotate(f"avg {mean_dec:.2f} J/tok\n({output_tokens} output tok, {decode_tps:.1f} tok/s)",
                xy=(ttft_s + 0.3, mean_dec),
                xytext=(ttft_s + 0.3, mean_dec * 0.88),
                fontsize=8.5, color="#993333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                          edgecolor="#cc3333", linewidth=0.8))

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0, t_end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (s)", fontsize=12, labelpad=6)
    ax.set_ylabel("Energy per Token  (J / token)", fontsize=12, labelpad=6)
    ax.set_title("End-to-End Inference: Energy per Token  (Prefill vs Decode)",
                 fontsize=14, fontweight="bold", pad=10)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.4, alpha=0.30)

    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot instantaneous J/token vs time from e2e_ttft_profile.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="./log/log_03_16_2/e2e_ttft_profile.csv",
        help="Path to e2e_ttft_profile.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: same dir as CSV)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "e2e_energy_per_token.png"

    df = load_csv(csv_path)
    df, ttft_s, total_dur_s, prompt_tokens, output_tokens, prefill_tps, decode_tps = \
        compute_j_per_token(df)

    print(f"TTFT            : {ttft_s:.4f} s")
    print(f"Prefill throughput: {prefill_tps:.1f} tok/s  ({prompt_tokens} tokens)")
    print(f"Decode  throughput: {decode_tps:.2f} tok/s  ({output_tokens} tokens)")

    prefill_avg = df.loc[df["time_offset"] < ttft_s, "j_per_token_inst"].mean()
    decode_avg  = df.loc[df["time_offset"] >= ttft_s, "j_per_token_inst"].mean()
    print(f"Avg Prefill J/tok : {prefill_avg:.4f}")
    print(f"Avg Decode  J/tok : {decode_avg:.4f}")
    print(f"Decode/Prefill ratio: {decode_avg/prefill_avg:.1f}×")

    plot(df, ttft_s, total_dur_s, prompt_tokens, output_tokens,
         prefill_tps, decode_tps, out_path)


if __name__ == "__main__":
    main()
