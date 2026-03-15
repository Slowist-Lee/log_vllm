import csv
import os
import time
import threading
import pynvml
import pandas as pd

import vllm
from vllm import LLM, SamplingParams

class GPUMonitor:
    def __init__(self, interval=0.075):
        self.interval = interval
        self.data = []
        self.running = False
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 默认监听第一张显卡

    def __enter__(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        return self

    def _monitor(self):
        while self.running:
            current_time = time.time()
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # 单位瓦特
            gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPH)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM) # 新增：显存频率
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            
            self.data.append({
                "timestamp": current_time,                 # 新增：绝对时间戳
                "time_offset": current_time - self.start_time,
                "power_w": power,
                "gpu_clock_mhz": gpu_clock,
                "mem_clock_mhz": mem_clock,                # 新增：显存频率
                "util_pct": util
            })
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

    def save_and_calculate(self, filename, total_output_tokens):
        df = pd.DataFrame(self.data)
        
        # 计算实际的采样时间间隔 (当前行时间戳 - 上一行时间戳)
        # 第一行的间隔默认取 self.interval
        df['time_interval'] = df['timestamp'].diff().fillna(self.interval)
        
        # 修改为按照作业要求：energy = sum of power * time interval across all samples
        df['energy_j'] = df['power_w'] * df['time_interval']
        total_energy = df['energy_j'].sum()
        
        df.to_csv(filename, index=False)
        
        duration = df['time_offset'].iloc[-1] if not df.empty else 0
        avg_power = df['power_w'].mean() if not df.empty else 0
        
        print(f"\n--- Energy Analysis ---")
        print(f"Total Time: {duration:.2f} s")
        print(f"Average Power: {avg_power:.2f} W")
        print(f"Total Energy: {total_energy:.2f} J")
        print(f"Total Output Tokens: {total_output_tokens}")
        if total_output_tokens > 0:
            print(f"Energy per Token: {total_energy/total_output_tokens:.4f} J/token")
        else:
            print("Energy per Token: N/A")

def print_system_info(model_name):
    """提取并打印 Task 1 要求的硬件与软件环境信息"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # 获取 GPU 名称并处理可能的 byte 字符串
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode('utf-8')
        
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver_version, bytes):
        driver_version = driver_version.decode('utf-8')
        
    cuda_version_int = pynvml.nvmlSystemGetCudaDriverVersion()
    cuda_version = f"{cuda_version_int // 1000}.{(cuda_version_int % 1000) // 10}"

    print("=== Task 1: Environment Info ===")
    print(f"GPU Model: {gpu_name}")
    print(f"Driver Version: {driver_version}")
    print(f"CUDA Version: {cuda_version}")
    print(f"Model Name: {model_name}")
    print(f"Inference Engine (vLLM) Version: {vllm.__version__}")
    print("================================\n")


if __name__ == "__main__":

    input_csv_path = "./prompt/llama3_test_prompts.csv"
    output_csv_path = "./log/llama3_test_output.csv"
    power_log_path = "./log/inference_power_log.csv"
    model_name = "./mistral_7b_model/" 

    # 1. 打印硬件与软件环境信息 (Task 1 核心要求)
    print_system_info(model_name)

    # 2. 从 CSV 读取 prompts
    prompts = []
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
        
    with open(input_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'prompt' in row:
                prompts.append(row['prompt'])
            else:
                raise ValueError("CSV file must contain a 'prompt' column.")

    print(f"Loaded {len(prompts)} prompts from {input_csv_path}")

    # 3. 初始化 vLLM
    print("Initializing vLLM engine...")
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # 4. 执行推理并同步监控功耗 (Task 1 & Task 2)
    print("Starting inference and power monitoring...")
    with GPUMonitor(interval=0.075) as monitor:
        # vLLM 批量推理
        outputs = llm.generate(prompts, sampling_params)

    # 5. 提取输出与 Token 统计
    results_to_save = []
    total_generated_tokens = 0

    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)
        total_generated_tokens += token_count
        
        results_to_save.append({
            "prompt": prompt,
            "output": generated_text,
            "output_tokens": token_count
        })

    # 6. 将输出写回新的 CSV 文件
    # 确保 log 文件夹存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "output", "output_tokens"])
        writer.writeheader()
        writer.writerows(results_to_save)
        
    print(f"\nInference complete! Outputs and token counts saved to {output_csv_path}")

    # 7. 保存功耗日志并计算能耗
    monitor.save_and_calculate(filename=power_log_path, 
                               total_output_tokens=total_generated_tokens)