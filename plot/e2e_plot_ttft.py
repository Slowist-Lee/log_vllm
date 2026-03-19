import argparse

import matplotlib.pyplot as plt
import pandas as pd


def load_profile_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = ["time_offset", "power_w", "gpu_clock_mhz"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required_cols).sort_values("time_offset").reset_index(drop=True)

    if df.empty:
        raise ValueError("CSV 没有可用的数值数据。")
    return df


def infer_ttft(df: pd.DataFrame) -> float:
    if "event" in df.columns:
        ttft_rows = df[df["event"].astype(str).str.upper() == "TTFT"]
        if not ttft_rows.empty:
            return float(ttft_rows.iloc[0]["time_offset"])

    if "ttft_s" in df.columns:
        ttft_series = pd.to_numeric(df["ttft_s"], errors="coerce").dropna()
        if not ttft_series.empty:
            return float(ttft_series.iloc[0])

    raise ValueError("无法从 CSV 推断 TTFT。请确保包含 event=TTFT 或 ttft_s 列。")


def compute_phase_stats(df: pd.DataFrame, ttft_s: float) -> dict:
    work = df.copy()
    # 用采样点差分近似积分面积（能量）。
    work["dt"] = work["time_offset"].diff().fillna(0.0)
    work["energy_j"] = work["power_w"] * work["dt"]
    work["phase"] = work["time_offset"].apply(lambda t: "prefill" if t < ttft_s else "decode")

    pre = work[work["phase"] == "prefill"]
    dec = work[work["phase"] == "decode"]

    return {
        "prefill_avg_power_w": float(pre["power_w"].mean()) if not pre.empty else 0.0,
        "decode_avg_power_w": float(dec["power_w"].mean()) if not dec.empty else 0.0,
        "prefill_peak_power_w": float(pre["power_w"].max()) if not pre.empty else 0.0,
        "decode_peak_power_w": float(dec["power_w"].max()) if not dec.empty else 0.0,
        "prefill_energy_j": float(pre["energy_j"].sum()) if not pre.empty else 0.0,
        "decode_energy_j": float(dec["energy_j"].sum()) if not dec.empty else 0.0,
        "prefill_avg_clock_mhz": float(pre["gpu_clock_mhz"].mean()) if not pre.empty else 0.0,
        "decode_avg_clock_mhz": float(dec["gpu_clock_mhz"].mean()) if not dec.empty else 0.0,
    }


def plot_ttft_profile(df: pd.DataFrame, ttft_s: float, output_png: str, title: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    fig, ax_power = plt.subplots(figsize=(12, 6), dpi=160)
    ax_clock = ax_power.twinx()

    # 背景分相：左侧 Prefill，右侧 Decode。
    t_min = float(df["time_offset"].min())
    t_max = float(df["time_offset"].max())
    ax_power.axvspan(t_min, ttft_s, color="#f7d9d9", alpha=0.45, label="Prefill Phase")
    ax_power.axvspan(ttft_s, t_max, color="#dceaf8", alpha=0.35, label="Decode Phase")

    # 主曲线：功耗 + 频率。
    line_power, = ax_power.plot(
        df["time_offset"],
        df["power_w"],
        color="#d62828",
        linewidth=2.0,
        label="GPU Power (W)",
    )
    line_clock, = ax_clock.plot(
        df["time_offset"],
        df["gpu_clock_mhz"],
        color="#1d4e89",
        linewidth=1.8,
        alpha=0.9,
        label="GPU Clock (MHz)",
    )

    # TTFT 分界线。
    ttft_line = ax_power.axvline(
        ttft_s,
        color="#111111",
        linestyle="--",
        linewidth=1.8,
        label=f"TTFT = {ttft_s:.3f}s",
    )

    ax_power.text(
        x=(t_min + ttft_s) / 2,
        y=ax_power.get_ylim()[1] * 0.95,
        s="Prefill Phase",
        ha="center",
        va="top",
        fontsize=10,
        color="#7a1c1c",
        fontweight="bold",
    )
    ax_power.text(
        x=(ttft_s + t_max) / 2,
        y=ax_power.get_ylim()[1] * 0.95,
        s="Decode Phase",
        ha="center",
        va="top",
        fontsize=10,
        color="#163a63",
        fontweight="bold",
    )

    ax_power.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax_power.set_xlabel("Time (s)")
    ax_power.set_ylabel("Power (W)", color="#d62828")
    ax_clock.set_ylabel("GPU Clock (MHz)", color="#1d4e89")

    # 合并图例
    handles = [line_power, line_clock, ttft_line]
    labels = [h.get_label() for h in handles]
    ax_power.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_png)
    print(f"图像已保存: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制带 TTFT 分界线的功耗/频率时序图")
    parser.add_argument("--input-csv", type=str, default="./log/log_03_16_2/e2e_ttft_profile.csv")
    parser.add_argument("--output-png", type=str, default="./e2e_ttft_profile.png")
    parser.add_argument("--title", type=str, default="End-to-End Inference: Power & Frequency with TTFT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_profile_csv(args.input_csv)
    ttft_s = infer_ttft(df)

    stats = compute_phase_stats(df, ttft_s)
    plot_ttft_profile(df, ttft_s, args.output_png, args.title)

    print("\n========== Phase Statistics ==========")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    print("======================================\n")


if __name__ == "__main__":
    main()
