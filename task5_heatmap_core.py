import inspect
import os
import time
import argparse
import pandas as pd
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from gpu_utils import load_long_prompt, GPUMonitor


def build_engine(model_path: str) -> LLMEngine:
    engine_kwargs = {
        "model": model_path,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
    }
    arg_names = inspect.signature(EngineArgs).parameters
    if "disable_log_requests" in arg_names:
        engine_kwargs["disable_log_requests"] = True
    if "max_model_len" in arg_names:
        engine_kwargs["max_model_len"] = 8192
    return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))


def run_batch_e2e_requests(engine: LLMEngine, prompts: list, max_tokens: int = 128) -> dict:
    """Batch inference with precise per-request TTFT tracking via step() loop."""
    req_ids = []
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)

    for i, prompt in enumerate(prompts):
        req_id = f"batch-req-{i}"
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
            if rid not in ttft_dict:
                continue
            generated_len = len(out.outputs[0].token_ids) if out.outputs else 0
            if ttft_dict[rid] is None and generated_len > 0:
                ttft_dict[rid] = now - t0
            if out.finished:
                finished_dict[rid] = True
                total_output_tokens += generated_len

    total_duration_s = time.perf_counter() - t0
    valid_ttfts = [t for t in ttft_dict.values() if t is not None]
    mean_ttft_s = sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else total_duration_s
    mean_tpot_s = max(total_duration_s - mean_ttft_s, 0.0) / max(total_output_tokens / len(prompts), 1)

    return {
        "mean_ttft_s": round(mean_ttft_s, 4),
        "total_duration_s": round(total_duration_s, 4),
        "mean_tpot_s": round(mean_tpot_s, 6),
        "total_output_tokens": total_output_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Task5 heatmap inference: single (freq, bs) experiment")
    parser.add_argument("--freq", type=int, required=True, help="GPU frequency in MHz (already locked externally)")
    parser.add_argument("--bs",   type=int, required=True, help="Batch size (number of concurrent requests)")
    parser.add_argument("--log-dir", type=str, default="./log/task5_heatmap")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max decode tokens per request")
    args = parser.parse_args()

    log_dir = args.log_dir
    power_log_dir = os.path.join(log_dir, "power_logs")
    os.makedirs(power_log_dir, exist_ok=True)

    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    prefill_prompt = load_long_prompt()
    prompts = [prefill_prompt] * args.bs

    # Build engine
    print(f"\n[task5] freq={args.freq} MHz | bs={args.bs} | Building engine...")
    engine = build_engine(model_path)

    # Warmup
    warmup_params = SamplingParams(temperature=0.0, max_tokens=8)
    engine.add_request(request_id="warmup", prompt="hello", params=warmup_params)
    while True:
        outs = engine.step()
        if any(o.request_id == "warmup" and o.finished for o in outs):
            break
    time.sleep(2)

    # Run batch inference with GPU monitoring (clock monitoring enabled for detailed per-timestep logs)
    print(f"[task5] Running batch inference (bs={args.bs}, max_tokens={args.max_tokens})...")
    with GPUMonitor(interval=0.05, monitor_clock=True) as monitor:
        batch_result = run_batch_e2e_requests(engine, prompts, max_tokens=args.max_tokens)

    # --- Save per-timestep power log ---
    power_log_path = os.path.join(power_log_dir, f"freq{args.freq}_bs{args.bs}_power_log.csv")
    df_power = pd.DataFrame(monitor.data)
    df_power.to_csv(power_log_path, index=False)
    print(f"[task5] Per-timestep power log saved: {power_log_path}")

    # --- Compute summary metrics ---
    m = monitor.get_metrics(batch_result["total_output_tokens"])
    tpj = (batch_result["total_output_tokens"] / m["total_energy_j"]
           if m["total_energy_j"] > 0 else 0.0)

    # --- Append summary row to CSV ---
    summary_path = os.path.join(log_dir, "summary.csv")
    row = (
        f"{args.freq},{args.bs},"
        f"{m['duration_s']},{batch_result['mean_ttft_s']},{batch_result['mean_tpot_s']},"
        f"{m['avg_power_w']},{m['peak_power_w']},{m['total_energy_j']},"
        f"{m['throughput_tps']},{m['j_per_token']},{round(tpj, 4)},{batch_result['total_output_tokens']}\n"
    )
    with open(summary_path, "a") as f:
        f.write(row)

    print(
        f"[task5] DONE | freq={args.freq} MHz | bs={args.bs} | "
        f"throughput={m['throughput_tps']:.2f} tok/s | "
        f"e2e={batch_result['total_duration_s']:.3f} s | "
        f"ttft={batch_result['mean_ttft_s']:.4f} s | "
        f"tpot={batch_result['mean_tpot_s']*1000:.2f} ms/tok | "
        f"power={m['avg_power_w']:.1f} W | "
        f"tpj={tpj:.2f} tok/J"
    )


if __name__ == "__main__":
    main()
