# 测试PD分离

import os
import time
import threading
import pynvml
import pandas as pd
import shutil

import vllm
from vllm import LLM, SamplingParams


WARMUP_PROMPT = "Hello, this is a warm up prompt."
PAPER_STYLE_PROMPT_REPEAT = 3
PREFILL_PROMPT_MULTIPLIER = 128
DECODE_PROMPT_TEXT = "Write a concise explanation of why GPU power can drop during autoregressive decoding."

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
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, 0)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, 2)
                # 获取利用率
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                util_gpu = util_rates.gpu
                util_mem = util_rates.memory
                
                self.data.append({
                    "timestamp": current_time,
                    "time_offset": current_time - self.start_time,
                    "power_w": power,
                    "gpu_clock_mhz": gpu_clock,
                    "mem_clock_mhz": mem_clock,
                    "util_gpu_pct": util_gpu,
                    "util_mem_pct": util_mem,
                    "util_pct": util_gpu
                })
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def save_and_calculate(self, filename, total_output_tokens, input_text=None, output_text=None):
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
        
        # 保存输入输出对到单独的CSV文件
        if input_text is not None and output_text is not None:
            io_filename = filename.replace("_power_log.csv", "_io_log.csv")
            # 构造输入输出数据
            io_data = {
                "input_text": [input_text],
                "output_text": [output_text],
                "total_output_tokens": [total_output_tokens],
                "total_energy_j": [total_energy],
                "energy_per_token_j": [total_energy/total_output_tokens if total_output_tokens>0 else 0]
            }
            io_df = pd.DataFrame(io_data)
            # 如果文件已存在则追加，否则新建
            if os.path.exists(io_filename):
                io_df.to_csv(io_filename, mode='a', header=False, index=False)
            else:
                io_df.to_csv(io_filename, index=False)
        
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


def run_warmup(llm: LLM) -> None:
    print("Running Warm-up...")
    dummy_params = SamplingParams(temperature=0.0, max_tokens=16)
    llm.generate([WARMUP_PROMPT] * 2, dummy_params)
    print("Warm-up complete. Waiting for GPU to settle.\n")
    time.sleep(3)

if __name__ == "__main__":
    if not check_disk_space():
        exit(1)

    model_path = "./mistral_7b_model/" 
    print_system_info(model_path)

    print(f"Initializing vLLM with {model_path}...")
    # 关闭 chunked prefill，避免超长 prompt 被切碎后形成长时间锯齿功耗。
    llm = LLM(model=model_path, enforce_eager=True, enable_chunked_prefill=False) 
    print("Using built-in prompts...")
    long_base = "The quick brown fox jumps over the lazy dog "
    paper_style_long_prompt = (long_base * PREFILL_PROMPT_MULTIPLIER).strip()

    # 参考论文设定：同一类请求重复多次，而不是每次都换一条长度差异很大的 prompt。
    long_prompts_list = [paper_style_long_prompt] * PAPER_STYLE_PROMPT_REPEAT
    short_prompts_list = [DECODE_PROMPT_TEXT] * PAPER_STYLE_PROMPT_REPEAT
    
    print(f"Loaded {len(long_prompts_list)} long prompts and {len(short_prompts_list)} short prompts.")
    
    # ==========================================
    # 0. 极其关键的预热阶段 (Warm-up)
    # ==========================================
    run_warmup(llm)

    # ==========================================
    # 1. 测量 Prefill 阶段 (长输入, 极短输出)
    # ==========================================
    print("--- Starting Task 3: Prefill Isolation ---")
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1) 

    # 循环遍历所有 long_prompt
    for idx, long_prompt in enumerate(long_prompts_list):
        print(f"\n>>> Processing Long Prompt [{idx + 1}/{len(long_prompts_list)}]")
        
        with GPUMonitor(interval=0.075) as prefill_monitor:
            output_prefill = llm.generate([long_prompt], prefill_params)

        # 提取生成的 token 数量和输出文本
        prefill_tokens = len(output_prefill[0].outputs[0].token_ids)
        prefill_output_text = output_prefill[0].outputs[0].text
        
        # 传入输入输出文本保存 (注意：文件名加上了 idx，防止被下一次循环覆盖)
        filename = f"./log/prefill_{idx}_power_log.csv"
        prefill_monitor.save_and_calculate(filename, prefill_tokens, 
                                          input_text=long_prompt, output_text=prefill_output_text)
        
        time.sleep(3) # 每次跑完让 GPU 更充分回到稳定空闲态

    # ==========================================
    # 2. 测量 Decode 阶段 (短输入, 极长输出)
    # ==========================================
    print("\n--- Starting Task 3: Decode Isolation ---")
    decode_params = SamplingParams(temperature=0.0, max_tokens=512, ignore_eos=True) 

    # 循环遍历所有 short_prompt
    for idx, short_prompt in enumerate(short_prompts_list):
        print(f"\n>>> Processing Short Prompt [{idx + 1}/{len(short_prompts_list)}]")
        
        with GPUMonitor(interval=0.075) as decode_monitor:
            output_decode = llm.generate([short_prompt], decode_params)

        # 提取生成的 token 数量和输出文本
        decode_tokens = len(output_decode[0].outputs[0].token_ids)
        decode_output_text = output_decode[0].outputs[0].text
        
        # 传入输入输出文本保存 (注意：文件名加上了 idx)
        filename = f"./log/decode_{idx}_power_log.csv"
        decode_monitor.save_and_calculate(filename, decode_tokens,
                                         input_text=short_prompt, output_text=decode_output_text)
        
        time.sleep(3)

    print("\nAll Tasks completed successfully!")