import os
import time
import argparse
import threading
import pynvml
import pandas as pd
import vllm
from vllm import LLM, SamplingParams


def load_long_prompt(prompt_path="./prompt/long_prompt.txt"):
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
        if prompt_text:
            return prompt_text
    # 回退逻辑：避免文件缺失时实验脚本直接失败。
    return ("The quick brown fox jumps over the lazy dog " * 1112).strip()


def extract_ttft_tpot(output_item, duration_s, output_tokens):
    """尽量从 vLLM 输出对象提取 TTFT/TPOT；提取失败时用可解释兜底值。"""
    ttft_s = None

    metrics = getattr(output_item, "metrics", None)
    if metrics is not None:
        # 不同版本 vLLM 的字段名可能不同，这里按常见命名做兼容。
        for attr in ["time_to_first_token", "ttft", "first_token_time", "first_token_latency"]:
            value = getattr(metrics, attr, None)
            if value is not None:
                try:
                    ttft_s = float(value)
                    break
                except (TypeError, ValueError):
                    pass

    if ttft_s is None:
        # 兜底：无法直接拿到 TTFT 时，用总时长近似（prefill max_tokens=1 时最接近真实 TTFT）。
        ttft_s = float(duration_s)

    if output_tokens > 0:
        tpot_s = max(float(duration_s) - float(ttft_s), 0.0) / float(output_tokens)
    else:
        tpot_s = 0.0

    return round(ttft_s, 4), round(tpot_s, 6)


def save_system_info(model_name, script_name="inference_core"):
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

class GPUMonitor:
    def __init__(self, interval=0.075):
        self.interval = interval
        self.data = []
        self.running = False
        self.handles = []
        try:
            pynvml.nvmlInit()
            deviceCount = pynvml.nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.handles.append(handle)
            # 默认监控第一张显卡，如果需要全卡监控，逻辑需改为循环所有 handles
            self.handle = self.handles[0] if self.handles else None
            print(f"Successfully initialized {deviceCount} GPU(s).")
            self.available = True
        except pynvml.NVMLError as e:
            print(f"NVML Initialization Failed: {e}")
            self.available = False

    def __enter__(self):
        if not self.available: return self
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

    def get_metrics(self, total_tokens):
        if not self.data:
            print("No GPU data collected.")
            return
        df = pd.DataFrame(self.data)
        df['time_interval'] = df['timestamp'].diff().fillna(self.interval)
        df['energy_j'] = df['power_w'] * df['time_interval']
        
        total_energy = df['energy_j'].sum()
        duration = df['time_offset'].iloc[-1]
        avg_power = df['power_w'].mean()
        throughput = total_tokens / duration if duration > 0 else 0
        j_per_token = total_energy / total_tokens if total_tokens > 0 else 0
        
        return {
            "duration_s": round(duration, 4),
            "avg_power_w": round(avg_power, 2),
            "total_energy_j": round(total_energy, 4),
            "throughput_tps": round(throughput, 2),
            "j_per_token": round(j_per_token, 4)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["4a", "4b"])
    parser.add_argument("--freq", type=int, default=0)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()

    # 1. 长 prompt 读取本地文件（Task 1）
    prefill_prompt = load_long_prompt()
    decode_prompt = "Hello."

    # 2. 初始化 vLLM
    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    save_system_info(model_path, script_name="inference_core")
    llm = LLM(model=model_path, enforce_eager=True)
    
    # 预热 GPU
    llm.generate([decode_prompt], SamplingParams(max_tokens=10), use_tqdm=False)
    time.sleep(2)

    # 3. 执行对应的 Task
    os.makedirs("./log", exist_ok=True)

    if args.task == "4a":
        # === Prefill 阶段测试 ===
        prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
        start_pre = time.perf_counter()
        with GPUMonitor() as monitor:
            out = llm.generate([prefill_prompt], prefill_params, use_tqdm=False)
        pre_duration = time.perf_counter() - start_pre
        in_tokens = len(out[0].prompt_token_ids)
        pre_out_tokens = len(out[0].outputs[0].token_ids)
        m_pre = monitor.get_metrics(in_tokens)
        pre_ttft_s, pre_tpot_s = extract_ttft_tpot(out[0], pre_duration, pre_out_tokens)

        time.sleep(2)

        # === Decode 阶段测试 ===
        decode_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
        start_dec = time.perf_counter()
        with GPUMonitor() as monitor:
            out = llm.generate([decode_prompt], decode_params, use_tqdm=False)
        dec_duration = time.perf_counter() - start_dec
        out_tokens = len(out[0].outputs[0].token_ids)
        m_dec = monitor.get_metrics(out_tokens)
        dec_ttft_s, dec_tpot_s = extract_ttft_tpot(out[0], dec_duration, out_tokens)

        # 保存结果到 CSV
        with open("./log/task4a_results.csv", "a") as f:
            f.write(
                f"Prefill,{args.freq},{m_pre['duration_s']},{pre_ttft_s},{pre_tpot_s},"
                f"{m_pre['avg_power_w']},{m_pre['total_energy_j']},{m_pre['throughput_tps']},"
                f"{m_pre['j_per_token']},{pre_out_tokens}\n"
            )
            f.write(
                f"Decode,{args.freq},{m_dec['duration_s']},{dec_ttft_s},{dec_tpot_s},"
                f"{m_dec['avg_power_w']},{m_dec['total_energy_j']},{m_dec['throughput_tps']},"
                f"{m_dec['j_per_token']},{out_tokens}\n"
            )

    elif args.task == "4b":
        # === Batch Size 测试 ===
        batch_params = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        prompts = [prefill_prompt] * args.bs  # 复制长输入构建 Batch
        
        start_batch = time.perf_counter()
        with GPUMonitor() as monitor:
            outs = llm.generate(prompts, batch_params, use_tqdm=False)
        batch_duration = time.perf_counter() - start_batch
        
        total_tokens = sum([len(o.outputs[0].token_ids) for o in outs])
        m_batch = monitor.get_metrics(total_tokens)

        # Batch 模式下做平均 TTFT/TPOT，便于横向比较。
        ttft_list = []
        for o in outs:
            o_tokens = len(o.outputs[0].token_ids)
            ttft_s, _ = extract_ttft_tpot(o, batch_duration, o_tokens)
            ttft_list.append(ttft_s)

        mean_ttft_s = round(sum(ttft_list) / len(ttft_list), 4) if ttft_list else round(batch_duration, 4)
        mean_tpot_s = round(max(batch_duration - mean_ttft_s, 0.0) / max(total_tokens, 1), 6)

        # 保存结果到 CSV
        with open("./log/task4b_results.csv", "a") as f:
            f.write(
                f"{args.bs},{m_batch['duration_s']},{mean_ttft_s},{mean_tpot_s},{m_batch['avg_power_w']},"
                f"{m_batch['total_energy_j']},{m_batch['throughput_tps']},{m_batch['j_per_token']},{total_tokens}\n"
            )

if __name__ == "__main__":
    main()
