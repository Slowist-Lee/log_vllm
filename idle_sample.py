import argparse
import os
import time

import pandas as pd
import pynvml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU 空闲功耗采样脚本")
    parser.add_argument("--duration", type=float, default=180.0, help="采样总时长（秒）")
    parser.add_argument("--interval", type=float, default=0.05, help="采样间隔（秒），建议 0.01~0.1")
    parser.add_argument("--gpu-index", type=int, default=0, help="采样哪张 GPU")
    parser.add_argument("--output", type=str, default="./log/idle_power_log.csv", help="输出 CSV 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.duration <= 0:
        raise ValueError("--duration 必须大于 0")
    if args.interval <= 0:
        raise ValueError("--interval 必须大于 0")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_index)

    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")

    driver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes):
        driver = driver.decode("utf-8")

    print("=== Idle Sampling Start ===")
    print(f"GPU: {gpu_name}")
    print(f"Driver: {driver}")
    print(f"Duration: {args.duration:.1f}s | Interval: {args.interval:.3f}s")
    print("请确保采样期间不要运行其它推理/训练任务。")

    records = []
    start_wall = time.time()
    end_wall = start_wall + args.duration

    while True:
        now = time.time()
        if now >= end_wall:
            break

        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_util_pct = (mem_info.used / mem_info.total) * 100.0 if mem_info.total > 0 else 0.0

        records.append(
            {
                "timestamp": now,
                "time_offset": now - start_wall,
                "power_w": float(power_w),
                "gpu_clock_mhz": float(gpu_clock),
                "mem_clock_mhz": float(mem_clock),
                "util_gpu_pct": float(util.gpu),
                "util_mem_pct": float(mem_util_pct),
            }
        )

        time.sleep(args.interval)

    if not records:
        raise RuntimeError("采样结果为空，请检查参数或 NVML 状态")

    df = pd.DataFrame(records)
    df["time_interval"] = df["timestamp"].diff().fillna(args.interval)
    df["energy_j"] = df["power_w"] * df["time_interval"]

    avg_power = float(df["power_w"].mean())
    p95_power = float(df["power_w"].quantile(0.95))
    total_energy = float(df["energy_j"].sum())

    # 把关键摘要写回每一行，后续画图/汇总时不用再拼接额外文件。
    df["idle_avg_power_w"] = avg_power
    df["idle_p95_power_w"] = p95_power
    df["idle_total_energy_j"] = total_energy
    df["sampling_duration_s"] = args.duration
    df["sampling_interval_s"] = args.interval

    df.to_csv(args.output, index=False)

    print("\n=== Idle Sampling Summary ===")
    print(f"Samples: {len(df)}")
    print(f"Average Power: {avg_power:.3f} W")
    print(f"P95 Power: {p95_power:.3f} W")
    print(f"Total Energy: {total_energy:.3f} J")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
