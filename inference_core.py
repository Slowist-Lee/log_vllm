import os
import time
import argparse
import threading
import pynvml
import pandas as pd
from vllm import LLM, SamplingParams

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

    # 1. 使用内置 Prompts（不再依赖 CSV）
    long_base = "The quick brown fox jumps over the lazy dog "
    multipliers = [1112, 1222, 1333, 1444, 1555]
    long_prompts_list = [
        (long_base * m).strip() for m in multipliers
    ]

    prefill_prompt = long_prompts_list[0]
    decode_prompt = "Hello."

    # 2. 初始化 vLLM
    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    llm = LLM(model=model_path, enforce_eager=True)
    
    # 预热 GPU
    llm.generate([decode_prompt], SamplingParams(max_tokens=10), use_tqdm=False)
    time.sleep(2)

    # 3. 执行对应的 Task
    os.makedirs("./log", exist_ok=True)

    if args.task == "4a":
        # === Prefill 阶段测试 ===
        prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
        with GPUMonitor() as monitor:
            out = llm.generate([prefill_prompt], prefill_params, use_tqdm=False)
        in_tokens = len(out[0].prompt_token_ids)
        m_pre = monitor.get_metrics(in_tokens)

        time.sleep(2)

        # === Decode 阶段测试 ===
        decode_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
        with GPUMonitor() as monitor:
            out = llm.generate([decode_prompt], decode_params, use_tqdm=False)
        out_tokens = len(out[0].outputs[0].token_ids)
        m_dec = monitor.get_metrics(out_tokens)

        # 保存结果到 CSV
        with open("./log/task4a_results.csv", "a") as f:
            f.write(f"Prefill,{args.freq},{m_pre['duration_s']},{m_pre['avg_power_w']},{m_pre['total_energy_j']},{m_pre['throughput_tps']},{m_pre['j_per_token']}\n")
            f.write(f"Decode,{args.freq},{m_dec['duration_s']},{m_dec['avg_power_w']},{m_dec['total_energy_j']},{m_dec['throughput_tps']},{m_dec['j_per_token']}\n")

    elif args.task == "4b":
        # === Batch Size 测试 ===
        batch_params = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        prompts = [prefill_prompt] * args.bs  # 复制长输入构建 Batch
        
        with GPUMonitor() as monitor:
            outs = llm.generate(prompts, batch_params, use_tqdm=False)
        
        total_tokens = sum([len(o.outputs[0].token_ids) for o in outs])
        m_batch = monitor.get_metrics(total_tokens)

        # 保存结果到 CSV
        with open("./log/task4b_results.csv", "a") as f:
            f.write(f"{args.bs},{m_batch['duration_s']},{m_batch['avg_power_w']},{m_batch['total_energy_j']},{m_batch['throughput_tps']},{m_batch['j_per_token']}\n")

if __name__ == "__main__":
    main()
