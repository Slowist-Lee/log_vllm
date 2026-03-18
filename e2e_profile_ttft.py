import argparse
import inspect
import os
import threading
import time
import uuid

import pandas as pd
import pynvml
import vllm
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine


def load_long_prompt(prompt_path: str) -> str:
    """优先读取真实长提示词文件，避免使用无意义重复文本。"""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return text
    raise FileNotFoundError(f"Prompt 文件不存在或为空: {prompt_path}")


class GPUSampler:
    """后台异步采样 GPU 指标。"""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.05) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.running = False
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start_ts = 0.0

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def start(self) -> None:
        self.records = []
        self.running = True
        self._start_ts = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self.running:
            now = time.time()
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

                # 显存利用率按占用比例记录，更直观。
                mem_util_pct = (mem_info.used / mem_info.total) * 100.0 if mem_info.total > 0 else 0.0

                row = {
                    "timestamp": now,
                    "time_offset": now - self._start_ts,
                    "power_w": power_w,
                    "gpu_clock_mhz": float(gpu_clock),
                    "mem_clock_mhz": float(mem_clock),
                    "util_gpu_pct": float(util.gpu),
                    "util_mem_pct": mem_util_pct,
                }
                with self._lock:
                    self.records.append(row)
            except pynvml.NVMLError:
                # 某些时刻读不到指标时，忽略单点，避免打断整次实验。
                pass

            time.sleep(self.interval_s)

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5)

    def to_dataframe(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(self.records)


def build_engine(
    model_path: str,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_model_len: int | None = None,
) -> LLMEngine:
    engine_kwargs = {
        "model": model_path,
        "enforce_eager": True,
        "enable_chunked_prefill": False,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
    }

    # vLLM 不同版本的 EngineArgs 参数集合不同，按实际签名兼容传参。
    arg_names = inspect.signature(EngineArgs).parameters
    if "disable_log_requests" in arg_names:
        engine_kwargs["disable_log_requests"] = True
    if "max_model_len" in arg_names:
        # 避免出现 max_num_batched_tokens < max_model_len 的调度配置报错。
        engine_kwargs["max_model_len"] = max_model_len if max_model_len is not None else max_num_batched_tokens

    engine_args = EngineArgs(**engine_kwargs)
    return LLMEngine.from_engine_args(engine_args)


def save_system_info(model_name: str, script_name: str = "e2e_profile_ttft") -> None:
    os.makedirs("./log", exist_ok=True)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")

    driver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes):
        driver = driver.decode("utf-8")

    info_lines = [
        "=== Environment Info ===",
        f"GPU: {gpu_name} | Driver: {driver}",
        f"Model: {model_name} | vLLM: {vllm.__version__}",
        "========================",
    ]
    print("\n" + "\n".join(info_lines) + "\n")

    out_path = f"./log/system_info_{script_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")
    print(f"Environment info saved to: {out_path}")


def run_one_request_with_ttft(
    engine: LLMEngine,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    """执行一次端到端推理，返回 TTFT、输出 token 数和请求结果。"""
    request_id = f"req-{uuid.uuid4()}"
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    engine.add_request(request_id=request_id, prompt=prompt, params=sampling_params)

    t0 = time.perf_counter()
    ttft_s = None
    final_output = None

    while True:
        step_outputs = engine.step()
        now = time.perf_counter()

        for out in step_outputs:
            if out.request_id != request_id:
                continue

            # 严格以“首次看到生成 token”定义 TTFT。
            generated_len = 0
            if out.outputs:
                generated_len = len(out.outputs[0].token_ids)
            if ttft_s is None and generated_len > 0:
                ttft_s = now - t0

            if out.finished:
                final_output = out
                break

        if final_output is not None:
            break

    total_duration_s = time.perf_counter() - t0

    if final_output is None or not final_output.outputs:
        raise RuntimeError("请求结束但未拿到输出结果。")

    output_token_count = len(final_output.outputs[0].token_ids)
    prompt_token_count = len(final_output.prompt_token_ids) if getattr(final_output, "prompt_token_ids", None) else None

    # 当 max_tokens=0 或异常返回时，兜底处理 TTFT，避免后续分析报错。
    if ttft_s is None:
        ttft_s = total_duration_s

    return {
        "request_id": request_id,
        "ttft_s": float(ttft_s),
        "total_duration_s": float(total_duration_s),
        "output_token_count": int(output_token_count),
        "prompt_token_count": int(prompt_token_count) if prompt_token_count is not None else -1,
        "text": final_output.outputs[0].text,
    }


def annotate_and_save(
    df: pd.DataFrame,
    out_csv: str,
    meta: dict,
    sample_interval: float,
) -> None:
    if df.empty:
        raise RuntimeError("GPU 采样为空，请检查 NVML 是否可用或采样间隔是否过大。")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_interval"] = df["timestamp"].diff().fillna(sample_interval)
    df["energy_j"] = df["power_w"] * df["time_interval"]

    ttft_s = meta["ttft_s"]
    df["phase"] = df["time_offset"].apply(lambda t: "prefill" if t < ttft_s else "decode")
    df["event"] = ""

    # 用最接近 TTFT 的采样点打标记，便于后续绘图精确画分界线。
    marker_idx = (df["time_offset"] - ttft_s).abs().idxmin()
    df.at[marker_idx, "event"] = "TTFT"

    total_energy = float(df["energy_j"].sum())
    output_tokens = max(meta["output_token_count"], 1)
    j_per_token = total_energy / output_tokens

    df["request_id"] = meta["request_id"]
    df["ttft_s"] = ttft_s
    df["total_duration_s"] = meta["total_duration_s"]
    df["prompt_tokens"] = meta["prompt_token_count"]
    df["output_tokens"] = meta["output_token_count"]
    df["total_energy_j"] = total_energy
    df["j_per_output_token"] = j_per_token

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    prefill_df = df[df["phase"] == "prefill"]
    decode_df = df[df["phase"] == "decode"]

    prefill_energy = float(prefill_df["energy_j"].sum()) if not prefill_df.empty else 0.0
    decode_energy = float(decode_df["energy_j"].sum()) if not decode_df.empty else 0.0

    print("\n========== E2E Profiling Summary ==========")
    print(f"Output CSV: {out_csv}")
    print(f"TTFT: {ttft_s:.4f} s")
    print(f"Total duration: {meta['total_duration_s']:.4f} s")
    print(f"Prompt tokens: {meta['prompt_token_count']}")
    print(f"Output tokens: {meta['output_token_count']}")
    print(f"Total energy: {total_energy:.4f} J")
    print(f"J/token: {j_per_token:.6f}")
    print(f"Prefill energy: {prefill_energy:.4f} J")
    print(f"Decode energy: {decode_energy:.4f} J")
    print("==========================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端 LLM 推理功耗采样（含 TTFT 打点）")
    parser.add_argument("--model", type=str, default="./mistral_7b_model/LLM-Research/Mistral-7B-v0.3")
    parser.add_argument("--prompt-file", type=str, default="./prompt/long_prompt.txt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sample-interval", type=float, default=0.05, help="采样周期（秒），建议 0.01~0.1")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-csv", type=str, default="./log/e2e_ttft_profile.csv")
    parser.add_argument("--warmup", action="store_true", help="先做一次短预热，减少首次加载抖动")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="可选：显式覆盖 vLLM max_model_len；默认与 max-num-batched-tokens 对齐以避免调度报错",
    )
    parser.add_argument("--load-levels", type=int, nargs="*", help="不同负载水平（输入token数），如果指定则执行负载扫描实验")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Arguments: {args}")
    save_system_info(args.model, script_name="e2e_profile_ttft")

    engine = build_engine(
        model_path=args.model,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
    )

    if args.warmup:
        warmup_params = SamplingParams(temperature=0.0, max_tokens=8)
        engine.add_request(request_id="warmup", prompt="hello", params=warmup_params)
        while True:
            out = engine.step()
            done = any(item.request_id == "warmup" and item.finished for item in out)
            if done:
                break

    # 存储所有负载水平的结果
    all_results = []

    if args.load_levels:
        print(f"\n=== Starting load level scan: {args.load_levels} tokens ===")
        for load_level in args.load_levels:
            print(f"\n--- Testing load level: {load_level} tokens ---")
            
            # 生成指定长度的提示词
            base_prompt = "The quick brown fox jumps over the lazy dog. "
            prompt = base_prompt * (load_level // len(base_prompt) + 1)
            prompt = prompt[:load_level]
            
            sampler = GPUSampler(gpu_index=args.gpu_index, interval_s=args.sample_interval)
            sampler.start()
            try:
                meta = run_one_request_with_ttft(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            finally:
                sampler.stop()

            df = sampler.to_dataframe()
            # 计算 energy_j，与 annotate_and_save 保持一致
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["time_interval"] = df["timestamp"].diff().fillna(args.sample_interval)
            df["energy_j"] = df["power_w"] * df["time_interval"]
            
            out_csv = f"./log/e2e_load_profile_{load_level}.csv"
            annotate_and_save(df=df, out_csv=out_csv, meta=meta, sample_interval=args.sample_interval)
            
            # 计算并存储关键指标
            peak_power = df["power_w"].max()
            avg_power = df["power_w"].mean()
            total_energy = df["energy_j"].sum()
            
            all_results.append({
                "load_level": load_level,
                "peak_power_w": peak_power,
                "avg_power_w": avg_power,
                "total_energy_j": total_energy,
                "ttft_s": meta["ttft_s"],
                "total_duration_s": meta["total_duration_s"],
                "output_tokens": meta["output_token_count"]
            })

        # 保存汇总结果
        summary_df = pd.DataFrame(all_results)
        summary_csv = "./log/e2e_load_profile_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSummary saved to: {summary_csv}")
        print("\n========== Load Profile Summary ==========")
        print(summary_df)
        print("=======================================")
    else:
        # 原始单请求模式
        prompt = load_long_prompt(args.prompt_file)
        sampler = GPUSampler(gpu_index=args.gpu_index, interval_s=args.sample_interval)
        sampler.start()
        try:
            meta = run_one_request_with_ttft(
                engine=engine,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        finally:
            sampler.stop()

        df = sampler.to_dataframe()
        annotate_and_save(df=df, out_csv=args.output_csv, meta=meta, sample_interval=args.sample_interval)


if __name__ == "__main__":
    main()
