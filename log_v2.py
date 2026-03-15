import csv
import os
import time
import threading
import pynvml
import pandas as pd
import shutil

import vllm
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
                # 获取功耗 (单位：毫瓦 -> 瓦)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                # 获取频率
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPH)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                # 获取利用率
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                
                self.data.append({
                    "timestamp": current_time,
                    "time_offset": current_time - self.start_time,
                    "power_w": power,
                    "gpu_clock_mhz": gpu_clock,
                    "mem_clock_mhz": mem_clock,
                    "util_pct": util
                })
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def save_and_calculate(self, filename, total_output_tokens):
        if not self.data:
            print("No GPU data collected.")
            return

        df = pd.DataFrame(self.data)
        # 计算实际时间间隔
        df['time_interval'] = df['timestamp'].diff().fillna(self.interval)
        # 能耗 (J) = 功率 (W) * 时间间隔 (s)
        df['energy_j'] = df['power_w'] * df['time_interval']
        total_energy = df['energy_j'].sum()
        
        # 确保保存路径存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
        duration = df['time_offset'].iloc[-1]
        avg_power = df['power_w'].mean()
        
        print(f"\n--- Energy Analysis ---")
        print(f"Total Time: {duration:.2f} s")
        print(f"Average Power: {avg_power:.2f} W")
        print(f"Total Energy: {total_energy:.2f} J")
        print(f"Total Output Tokens: {total_output_tokens}")
        if total_output_tokens > 0:
            print(f"Energy per Token: {total_energy/total_output_tokens:.4f} J/token")

def check_disk_space(min_gb=2):
    """防止磁盘再次写满导致系统崩溃"""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (2**30)
    if free_gb < min_gb:
        print(f"❌ Disk space too low ({free_gb:.2f} GB left). Aborting.")
        return False
    return True

def print_system_info(model_name):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode('utf-8')
    driver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes): driver = driver.decode('utf-8')
    
    print("\n=== Environment Info ===")
    print(f"GPU: {gpu_name} | Driver: {driver}")
    print(f"Model: {model_name} | vLLM: {vllm.__version__}")
    print("========================\n")

if __name__ == "__main__":
    if not check_disk_space():
        exit(1)

    # 路径配置
    input_csv = "./prompt/llama3_test_prompts.csv"
    output_csv = "./log/llama3_test_output.csv"
    power_log = "./log/inference_power_log.csv"
    model_path = "./mistral_7b_model/" 

    print_system_info(model_path)

    # 加载 Prompt
    prompts = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prompts = [row['prompt'] for row in reader if 'prompt' in row]

    # 推理
    print(f"Initializing vLLM with {model_path}...")
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    print("Running inference...")
    with GPUMonitor(interval=0.075) as monitor:
        outputs = llm.generate(prompts, sampling_params)

    # 结果处理
    results = []
    total_tokens = 0
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        total_tokens += tokens
        results.append({"prompt": prompt, "output": text, "output_tokens": tokens})

    # 保存
    pd.DataFrame(results).to_csv(output_csv, index=False)
    monitor.save_and_calculate(power_log, total_tokens)
    print(f"\nDone! Results saved to {output_csv}")