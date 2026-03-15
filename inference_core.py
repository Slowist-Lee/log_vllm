import os
import time
import argparse
import threading
import pynvml
import pandas as pd
from vllm import LLM, SamplingParams


def ensure_metrics(metrics):
    if metrics is not None:
        return metrics
    return {
        "duration_s": 0.0,
        "ttft_s": 0.0,
        "tpot_s": 0.0,
        "avg_power_w": 0.0,
        "total_energy_j": 0.0,
        "throughput_tps": 0.0,
        "j_per_token": 0.0,
        "total_output_tokens": 0,
    }


def append_csv_row(file_path, row_values, header=None):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", encoding="utf-8") as f:
        if (not file_exists) and header is not None:
            f.write(",".join(header) + "\n")
        f.write(",".join(map(str, row_values)) + "\n")


def _metric_attr(metrics, key):
    if metrics is None:
        return None
    if hasattr(metrics, key):
        return getattr(metrics, key)
    if isinstance(metrics, dict):
        return metrics.get(key)
    return None


def extract_ttft_tpot(outputs):
    ttft_values = []
    tpot_values = []

    for out in outputs:
        metrics = getattr(out, "metrics", None)
        arrival_time = _metric_attr(metrics, "arrival_time")
        first_token_time = _metric_attr(metrics, "first_token_time")
        finished_time = _metric_attr(metrics, "finished_time")

        token_count = 0
        if getattr(out, "outputs", None):
            token_count = len(out.outputs[0].token_ids)

        if (arrival_time is not None) and (first_token_time is not None):
            ttft = first_token_time - arrival_time
            if ttft >= 0:
                ttft_values.append(ttft)

        if (first_token_time is not None) and (finished_time is not None) and token_count > 1:
            decode_span = finished_time - first_token_time
            if decode_span >= 0:
                tpot_values.append(decode_span / (token_count - 1))

    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0
    avg_tpot = sum(tpot_values) / len(tpot_values) if tpot_values else 0.0
    return round(avg_ttft, 4), round(avg_tpot, 4)

class GPUMonitor:
    def __init__(self, interval=0.075):
        self.interval = interval
        self.data = []
        self.running = False
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except pynvml.NVMLError as e:
            print(f"NVML Initialization Failed: {e}")
            self.available = False

    def __enter__(self):
        if not self.available:
            return self
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self

    def _monitor(self):
        while self.running:
            try:
                current_time = time.time()
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.data.append({
                    "timestamp": current_time,
                    "time_offset": current_time - self.start_time,
                    "power_w": power
                })
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_metrics(self, total_tokens, duration_override_s=None, ttft_s=0.0, tpot_s=0.0):
        if not self.data:
            duration = duration_override_s if duration_override_s is not None else 0.0
            throughput = total_tokens / duration if duration > 0 else 0.0
            return {
                "duration_s": round(duration, 4),
                "ttft_s": round(ttft_s, 4),
                "tpot_s": round(tpot_s, 4),
                "avg_power_w": 0.0,
                "total_energy_j": 0.0,
                "throughput_tps": round(throughput, 2),
                "j_per_token": 0.0,
                "total_output_tokens": int(total_tokens),
            }

        df = pd.DataFrame(self.data)
        df['time_interval'] = df['timestamp'].diff().fillna(self.interval)
        df['energy_j'] = df['power_w'] * df['time_interval']
        
        total_energy = df['energy_j'].sum()
        sampled_duration = df['time_offset'].iloc[-1] if not df.empty else 0.0
        duration = duration_override_s if duration_override_s is not None else sampled_duration
        avg_power = df['power_w'].mean()
        throughput = total_tokens / duration if duration > 0 else 0
        j_per_token = total_energy / total_tokens if total_tokens > 0 else 0
        
        return {
            "duration_s": round(duration, 4),
            "ttft_s": round(ttft_s, 4),
            "tpot_s": round(tpot_s, 4),
            "avg_power_w": round(avg_power, 2),
            "total_energy_j": round(total_energy, 4),
            "throughput_tps": round(throughput, 2),
            "j_per_token": round(j_per_token, 4),
            "total_output_tokens": int(total_tokens),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["4a", "4b"])
    parser.add_argument("--freq", type=int, default=0)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()

    # 1. 使用内置 Prompts（不再依赖 CSV）
    long_base = "The quick brown fox jumps over the lazy dog "
    multipliers = [1112, 1222, 1333, 1444, 1555]
    long_prompts_list = [
        (long_base * m).strip() for m in multipliers
    ]

    prefill_prompt = long_prompts_list[0]
    decode_prompt = "Hello."

    # 2. 初始化 vLLM
    model_path = "./mistral_7b_model/"
    llm = LLM(model=model_path, enforce_eager=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "log")
    
    # 预热 GPU
    llm.generate([decode_prompt], SamplingParams(max_tokens=10), use_tqdm=False)
    time.sleep(2)

    # 3. 执行对应的 Task
    os.makedirs(output_dir, exist_ok=True)

    if args.task == "4a":
        # === Prefill 阶段测试 ===
        prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
        start = time.perf_counter()
        with GPUMonitor() as monitor:
            out = llm.generate([prefill_prompt], prefill_params, use_tqdm=False)
        duration = time.perf_counter() - start

        out_tokens = len(out[0].outputs[0].token_ids)
        ttft_s, tpot_s = extract_ttft_tpot(out)
        m_pre = ensure_metrics(monitor.get_metrics(out_tokens, duration_override_s=duration, ttft_s=ttft_s, tpot_s=tpot_s))

        time.sleep(2)

        # === Decode 阶段测试 ===
        decode_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
        start = time.perf_counter()
        with GPUMonitor() as monitor:
            out = llm.generate([decode_prompt], decode_params, use_tqdm=False)
        duration = time.perf_counter() - start

        out_tokens = len(out[0].outputs[0].token_ids)
        ttft_s, tpot_s = extract_ttft_tpot(out)
        m_dec = ensure_metrics(monitor.get_metrics(out_tokens, duration_override_s=duration, ttft_s=ttft_s, tpot_s=tpot_s))

        # 保存结果到 CSV
        task4a_file = os.path.join(output_dir, "task4a_results.csv")
        header_4a = ["phase", "frequency_mhz", "duration_s", "ttft_s", "tpot_s", "avg_power_w", "total_energy_j", "throughput_tps", "j_per_token", "total_output_tokens"]
        append_csv_row(
            task4a_file,
            ["Prefill", args.freq, m_pre["duration_s"], m_pre["ttft_s"], m_pre["tpot_s"], m_pre["avg_power_w"], m_pre["total_energy_j"], m_pre["throughput_tps"], m_pre["j_per_token"], m_pre["total_output_tokens"]],
            header=header_4a,
        )
        append_csv_row(
            task4a_file,
            ["Decode", args.freq, m_dec["duration_s"], m_dec["ttft_s"], m_dec["tpot_s"], m_dec["avg_power_w"], m_dec["total_energy_j"], m_dec["throughput_tps"], m_dec["j_per_token"], m_dec["total_output_tokens"]],
        )
        print(f"Saved results: {task4a_file}")

    elif args.task == "4b":
        # === Batch Size 测试 ===
        batch_params = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        prompts = [prefill_prompt] * args.bs  # 复制长输入构建 Batch
        
        start = time.perf_counter()
        with GPUMonitor() as monitor:
            outs = llm.generate(prompts, batch_params, use_tqdm=False)
        duration = time.perf_counter() - start
        
        total_tokens = sum([len(o.outputs[0].token_ids) for o in outs])
        ttft_s, tpot_s = extract_ttft_tpot(outs)
        m_batch = ensure_metrics(monitor.get_metrics(total_tokens, duration_override_s=duration, ttft_s=ttft_s, tpot_s=tpot_s))

        # 保存结果到 CSV
        task4b_file = os.path.join(output_dir, "task4b_results.csv")
        header_4b = ["batch_size", "duration_s", "ttft_s", "tpot_s", "avg_power_w", "total_energy_j", "throughput_tps", "j_per_token", "total_output_tokens"]
        append_csv_row(
            task4b_file,
            [args.bs, m_batch["duration_s"], m_batch["ttft_s"], m_batch["tpot_s"], m_batch["avg_power_w"], m_batch["total_energy_j"], m_batch["throughput_tps"], m_batch["j_per_token"], m_batch["total_output_tokens"]],
            header=header_4b,
        )
        print(f"Saved results: {task4b_file}")

if __name__ == "__main__":
    main()