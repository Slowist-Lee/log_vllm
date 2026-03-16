import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = ["batch_size", "decoding_only_ms", "decoding_with_prefill_ms", "prefill_slowdown_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=required).sort_values("batch_size")


def plot_one(df: pd.DataFrame, prefill_length: int, out_png: Path) -> None:
    plt.figure(figsize=(8, 6), dpi=160)

    plt.plot(
        df["batch_size"],
        df["decoding_with_prefill_ms"],
        marker="o",
        label="decoding-with-one-prefill",
        color="#1f77b4",
    )
    plt.plot(
        df["batch_size"],
        df["decoding_only_ms"],
        marker="o",
        label="decoding-only",
        color="#ff7f0e",
    )

    baseline_prefill_time = df.loc[
        df["batch_size"] == df["batch_size"].min(), "prefill_slowdown_ms"
    ].values[0]
    plt.axhline(y=baseline_prefill_time, color="#1f77b4", linestyle="--", alpha=0.6)

    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title(f"Batch Execution Time (Input Length ~ {prefill_length})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved plot: {out_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local plotting for interference experiment")
    parser.add_argument("--log-dir", type=str, default="./log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)

    csv_128 = log_dir / "interference_data_len128.csv"
    csv_1024 = log_dir / "interference_data_len1024.csv"

    df_128 = read_required_csv(csv_128)
    df_1024 = read_required_csv(csv_1024)

    plot_one(df_128, prefill_length=128, out_png=log_dir / "interference_len128.png")
    plot_one(df_1024, prefill_length=1024, out_png=log_dir / "interference_len1024.png")


if __name__ == "__main__":
    main()
