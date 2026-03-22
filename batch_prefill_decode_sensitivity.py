import argparse
import os
import time
from typing import Dict, List

from vllm import SamplingParams

from gpu_utils import GPUMonitor, load_long_prompt, save_system_info
from inference_core import build_engine


def parse_batch_sizes(raw: str) -> List[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--batch-sizes cannot be empty")
    if any(v <= 0 for v in values):
        raise ValueError("batch size must be positive")
    return sorted(set(values))


def run_batch_phase(engine, prompts: List[str], max_tokens: int) -> Dict[str, float]:
    req_ids = []
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)

    for i, prompt in enumerate(prompts):
        req_id = f"req-{i}-{time.time_ns()}"
        engine.add_request(request_id=req_id, prompt=prompt, params=sampling_params)
        req_ids.append(req_id)

    t0 = time.perf_counter()
    ttft_dict = {rid: None for rid in req_ids}
    finished_dict = {rid: False for rid in req_ids}
    total_output_tokens = 0

    while not all(finished_dict.values()):
        step_outputs = engine.step()
        now = time.perf_counter()

        for out in step_outputs:
            rid = out.request_id
            if rid not in finished_dict:
                continue

            generated_len = len(out.outputs[0].token_ids) if out.outputs else 0
            if ttft_dict[rid] is None and generated_len > 0:
                ttft_dict[rid] = now - t0

            if out.finished and not finished_dict[rid]:
                finished_dict[rid] = True
                total_output_tokens += generated_len

    total_duration_s = time.perf_counter() - t0
    valid_ttfts = [t for t in ttft_dict.values() if t is not None]
    mean_ttft_s = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else total_duration_s
    mean_tpot_s = max(total_duration_s - mean_ttft_s, 0.0) / max(total_output_tokens / len(prompts), 1)

    return {
        "duration_s": round(total_duration_s, 4),
        "mean_ttft_s": round(mean_ttft_s, 4),
        "mean_tpot_s": round(mean_tpot_s, 6),
        "total_output_tokens": int(total_output_tokens),
    }


def append_row(path: str, row: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure batch-size sensitivity for prefill vs decode."
    )
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--decode-max-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="./log/prefill_decode_bs_sensitivity_raw.csv",
        help="Raw run-level result CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="./log/prefill_decode_bs_sensitivity_summary.csv",
        help="Aggregated mean result CSV by phase and batch size.",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be >= 1")

    batch_sizes = parse_batch_sizes(args.batch_sizes)

    os.makedirs("./log", exist_ok=True)

    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    save_system_info(model_path, script_name="batch_prefill_decode_sensitivity")

    prefill_prompt = load_long_prompt()
    decode_prompt = "Hello."

    engine = build_engine(
        model_path=model_path,
        max_num_seqs=max(args.max_num_seqs, max(batch_sizes)),
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    warmup_params = SamplingParams(temperature=0.0, max_tokens=8)
    engine.add_request(request_id="warmup", prompt="hello", params=warmup_params)
    while True:
        outs = engine.step()
        if any(o.request_id == "warmup" and o.finished for o in outs):
            break
    time.sleep(2)

    raw_header = (
        "phase,batch_size,repeat_idx,duration_s,mean_ttft_s,mean_tpot_s,"
        "avg_power_w,peak_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens"
    )
    summary_header = (
        "phase,batch_size,duration_s,mean_ttft_s,mean_tpot_s,"
        "avg_power_w,peak_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens"
    )

    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(raw_header + "\n")

    rows = []
    for bs in batch_sizes:
        for repeat_idx in range(1, args.repeats + 1):
            print(f"[prefill] bs={bs}, repeat={repeat_idx}/{args.repeats}")
            prefill_prompts = [prefill_prompt] * bs
            with GPUMonitor(interval=0.02) as monitor:
                result = run_batch_phase(engine, prefill_prompts, max_tokens=1)
            metrics = monitor.get_metrics(result["total_output_tokens"])
            row = (
                f"prefill,{bs},{repeat_idx},{result['duration_s']},{result['mean_ttft_s']},"
                f"{result['mean_tpot_s']},{metrics['avg_power_w']},{metrics['peak_power_w']},"
                f"{metrics['total_energy_j']},{metrics['throughput_tps']},{metrics['j_per_token']},"
                f"{result['total_output_tokens']}"
            )
            append_row(args.out_csv, row)
            rows.append(("prefill", bs, result, metrics))
            time.sleep(2)

            print(f"[decode ] bs={bs}, repeat={repeat_idx}/{args.repeats}")
            decode_prompts = [decode_prompt] * bs
            with GPUMonitor(interval=0.02) as monitor:
                result = run_batch_phase(engine, decode_prompts, max_tokens=args.decode_max_tokens)
            metrics = monitor.get_metrics(result["total_output_tokens"])
            row = (
                f"decode,{bs},{repeat_idx},{result['duration_s']},{result['mean_ttft_s']},"
                f"{result['mean_tpot_s']},{metrics['avg_power_w']},{metrics['peak_power_w']},"
                f"{metrics['total_energy_j']},{metrics['throughput_tps']},{metrics['j_per_token']},"
                f"{result['total_output_tokens']}"
            )
            append_row(args.out_csv, row)
            rows.append(("decode", bs, result, metrics))
            time.sleep(2)

    # 聚合（按 phase + bs）
    grouped: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    for phase, bs, result, metrics in rows:
        grouped.setdefault(phase, {})
        grouped[phase].setdefault(
            bs,
            {
                "duration_s": [],
                "mean_ttft_s": [],
                "mean_tpot_s": [],
                "avg_power_w": [],
                "peak_power_w": [],
                "total_energy_j": [],
                "throughput_tps": [],
                "j_per_token": [],
                "total_output_tokens": [],
            },
        )
        g = grouped[phase][bs]
        g["duration_s"].append(float(result["duration_s"]))
        g["mean_ttft_s"].append(float(result["mean_ttft_s"]))
        g["mean_tpot_s"].append(float(result["mean_tpot_s"]))
        g["avg_power_w"].append(float(metrics["avg_power_w"]))
        g["peak_power_w"].append(float(metrics["peak_power_w"]))
        g["total_energy_j"].append(float(metrics["total_energy_j"]))
        g["throughput_tps"].append(float(metrics["throughput_tps"]))
        g["j_per_token"].append(float(metrics["j_per_token"]))
        g["total_output_tokens"].append(float(result["total_output_tokens"]))

    with open(args.summary_csv, "w", encoding="utf-8") as f:
        f.write(summary_header + "\n")
        for phase in ["prefill", "decode"]:
            for bs in sorted(grouped.get(phase, {}).keys()):
                g = grouped[phase][bs]
                mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
                f.write(
                    f"{phase},{bs},{mean(g['duration_s']):.4f},{mean(g['mean_ttft_s']):.4f},"
                    f"{mean(g['mean_tpot_s']):.6f},{mean(g['avg_power_w']):.2f},{mean(g['peak_power_w']):.2f},"
                    f"{mean(g['total_energy_j']):.4f},{mean(g['throughput_tps']):.2f},{mean(g['j_per_token']):.4f},"
                    f"{mean(g['total_output_tokens']):.2f}\n"
                )

    print("==========================================================")
    print(f"Raw result saved to: {args.out_csv}")
    print(f"Summary saved to:   {args.summary_csv}")
    print("Use plot/plot_prefill_decode_bs_sensitivity.py to visualize.")
    print("==========================================================")


if __name__ == "__main__":
    main()
