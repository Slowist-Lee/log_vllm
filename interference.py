# decode 和 prefill 干扰实验，没有修改测试过

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm import SamplingParams

def setup_engine(model_path):
    print(f"Initializing LLMEngine with model: {model_path}...")
    # 启用 eager mode 减少小 batch 下 CUDA Graph 的额外开销干扰
    # 关闭 chunked prefill 以确保完整的长序列 Prefill 会在单个 step 内计算
    engine_args = EngineArgs(
        model=model_path, 
        enforce_eager=True,
        enable_chunked_prefill=False,
        disable_log_requests=True, # 关闭请求日志防止刷屏
        max_num_seqs=512,          # 确保调度器容量足够
        max_num_batched_tokens=8192 # 确保能同时塞下长 Prefill 和几十个 Decode
    )
    return LLMEngine.from_engine_args(engine_args)

def run_interference_test(engine, batch_sizes, prefill_length=1024):
    """
    运行干扰测试，模拟纯 Decode 状态与插入长 Prefill 后的状态。
    """
    # 构造虚假的长短 Prompt
    # 假设平均 1 token ≈ 4-5 字符，这里用简单字符串近似，也可以用 tokenizer 传 token_ids
    short_prompt = "The quick brown fox " 
    long_prompt = "The quick brown fox " * (prefill_length // 4) 
    
    # 设置生成参数，ignore_eos=True 确保 Decode 任务不会提前结束
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100, ignore_eos=True)
    
    results = []

    for bs in batch_sizes:
        print(f"\n[Testing Batch Size: {bs}]")
        
        # 1. 清理引擎，确保干净的状态
        engine.abort_request("all")
        
        # 2. 注入 bs 个极短的请求
        for i in range(bs):
            engine.add_request(f"decode_req_{i}", short_prompt, sampling_params)
            
        # 3. 执行第一次 Step (消耗掉这 bs 个请求的 Prefill 阶段)
        # 此时这 bs 个请求正式进入 Decode 阶段
        engine.step()
        
        # 4. 测量 Baseline (纯 Decode 阶段的单步延迟)
        start_time = time.perf_counter()
        engine.step()
        decode_only_latency = (time.perf_counter() - start_time) * 1000 # 转 ms
        print(f"  -> Decoding-only Latency: {decode_only_latency:.2f} ms")
        
        # 5. 注入 1 个长文本请求 (制造干扰)
        engine.add_request("evil_prefill_req", long_prompt, sampling_params)
        
        # 6. 测量 Interference (Decode + 1个重度 Prefill 的单步延迟)
        start_time = time.perf_counter()
        engine.step()
        interference_latency = (time.perf_counter() - start_time) * 1000
        print(f"  -> Decoding-with-one-prefill Latency: {interference_latency:.2f} ms")
        
        # 7. 记录数据
        results.append({
            "batch_size": bs,
            "decoding_only_ms": decode_only_latency,
            "decoding_with_prefill_ms": interference_latency,
            "prefill_slowdown_ms": interference_latency - decode_only_latency
        })
        
        # 测试完清理该 batch 的所有请求
        engine.abort_request("all")

    return pd.DataFrame(results)

def plot_results(df, prefill_length):
    plt.figure(figsize=(8, 6))
    
    plt.plot(df['batch_size'], df['decoding_with_prefill_ms'], 
             marker='o', label='decoding-with-one-prefill', color='#1f77b4')
    plt.plot(df['batch_size'], df['decoding_only_ms'], 
             marker='o', label='decoding-only', color='#ff7f0e')
    
    # 添加水平参考线 (模拟只跑单个 Prefill 的时间，这里取 bs=1 时的差值作为基准参考)
    baseline_prefill_time = df.loc[df['batch_size'] == df['batch_size'].min(), 'prefill_slowdown_ms'].values[0]
    plt.axhline(y=baseline_prefill_time, color='#1f77b4', linestyle='--', alpha=0.6)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title(f'Batch Execution Time (Input Length $\\approx$ {prefill_length})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs("./log", exist_ok=True)
    save_path = f"./log/interference_len{prefill_length}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    # 使用你之前配置的模型路径
    MODEL_PATH = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    
    # 初始化引擎 (这个过程比较慢，只做一次)
    engine = setup_engine(MODEL_PATH)
    
    # 设定要测试的 Batch Size 梯度 (复现 X 轴)
    # 根据你的显存大小，如果 OOM 可以适当减小最大值
    test_batch_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    
    # --- 测试 1: 较短的 Prefill 干扰 (复现图 2a) ---
    print("\n" + "="*40)
    print("Starting Experiment 1: Short Prefill (128 tokens)")
    df_128 = run_interference_test(engine, test_batch_sizes, prefill_length=128)
    df_128.to_csv("./log/interference_data_len128.csv", index=False)
    plot_results(df_128, prefill_length=128)
    
    # --- 测试 2: 极长的 Prefill 干扰 (复现图 2b) ---
    print("\n" + "="*40)
    print("Starting Experiment 2: Long Prefill (1024 tokens)")
    df_1024 = run_interference_test(engine, test_batch_sizes, prefill_length=1024)
    df_1024.to_csv("./log/interference_data_len1024.csv", index=False)
    plot_results(df_1024, prefill_length=1024)
    
    print("\nAll experiments completed successfully!")