import os
import time
import threading
import pynvml
import pandas as pd
import vllm


def load_long_prompt(prompt_path="./prompt/long_prompt.txt"):
    """加载长提示，如果文件不存在则使用默认文本"""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
        if prompt_text:
            return prompt_text
    # 回退逻辑：避免文件缺失时实验脚本直接失败。
    return ("The quick brown fox jumps over the lazy dog " * 1112).strip()


def save_system_info(model_name, script_name="gpu_utils"):
    """保存系统信息到文件"""
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
        "=======================",
    ]
    print("\n" + "\n".join(info_lines) + "\n")

    out_path = f"./log/system_info_{script_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines) + "\n")
    print(f"Environment info saved to: {out_path}")


def extract_ttft_tpot(output_item, duration_s, output_tokens):
    """提取TTFT和TPOT值"""
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
        # 兜底：无法直接拿到 TTFT 时，用总时长近似
        # 对于Prefill阶段（max_tokens=1），这最接近真实TTFT
        # 对于Decode阶段，这可能会高估TTFT，但TPOT计算仍准确
        ttft_s = float(duration_s)

    if output_tokens > 0:
        # TPOT = (总时长 - TTFT) / 输出token数
        # 对于Decode阶段，这能更准确反映持续生成的速度
        tpot_s = max(float(duration_s) - float(ttft_s), 0.0) / float(output_tokens)
    else:
        tpot_s = 0.0

    return round(ttft_s, 4), round(tpot_s, 6)


class GPUMonitor:
    """GPU监控类，支持功耗、频率等指标的监控"""
    def __init__(self, interval=0.075, monitor_clock=False):
        self.interval = interval
        self.monitor_clock = monitor_clock
        self.data = []
        self.running = False
        self.handles = []
        try:
            pynvml.nvmlInit()
            deviceCount = pynvml.nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.handles.append(handle)
            # 默认监控第一张显卡
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
                
                data_point = {
                    "timestamp": current_time,
                    "time_offset": current_time - self.start_time,
                    "power_w": power
                }
                
                # 如果需要监控时钟频率
                if self.monitor_clock:
                    # 获取频率
                    gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, 0)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, 2)
                    # 获取利用率
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    util_gpu = util_rates.gpu
                    util_mem = util_rates.memory
                    
                    data_point.update({
                        "gpu_clock_mhz": gpu_clock,
                        "mem_clock_mhz": mem_clock,
                        "util_gpu_pct": util_gpu,
                        "util_mem_pct": util_mem,
                        "util_pct": util_gpu
                    })
                
                self.data.append(data_point)
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def get_metrics(self, total_tokens):
        """获取监控指标"""
        if not self.data:
            print("No GPU data collected.")
            return
        df = pd.DataFrame(self.data)
        df['time_interval'] = df['timestamp'].diff().fillna(self.interval)
        df['energy_j'] = df['power_w'] * df['time_interval']
        
        total_energy = df['energy_j'].sum()
        duration = df['time_offset'].iloc[-1]
        avg_power = df['power_w'].mean()
        peak_power = df['power_w'].max()
        throughput = total_tokens / duration if duration > 0 else 0
        j_per_token = total_energy / total_tokens if total_tokens > 0 else 0
        
        return {
            "duration_s": round(duration, 4),
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(peak_power, 2),
            "total_energy_j": round(total_energy, 4),
            "throughput_tps": round(throughput, 2),
            "j_per_token": round(j_per_token, 4)
        }

    def save_and_calculate(self, filename, total_output_tokens, input_text=None, output_text=None):
        """保存监控数据并计算指标"""
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
