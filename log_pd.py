# 测试PD分离的实验

import os
import time
import shutil

import vllm
from vllm import LLM, SamplingParams
from gpu_utils import load_long_prompt, save_system_info, GPUMonitor


WARMUP_PROMPT = "Hello, this is a warm up prompt."
PAPER_STYLE_PROMPT_REPEAT = 3
PREFILL_PROMPT_MULTIPLIER = 512
DECODE_PROMPT_TEXT = "Write a concise explanation of why GPU power can drop during autoregressive decoding."


def check_disk_space(min_gb=2):
    """防止磁盘再次写满导致系统崩溃"""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (2**30)
    if free_gb < min_gb:
        print(f"❌ Disk space too low ({free_gb:.2f} GB left). Aborting.")
        return False
    return True


def run_warmup(llm: LLM) -> None:
    print("Running Warm-up...")
    dummy_params = SamplingParams(temperature=0.0, max_tokens=16)
    llm.generate([WARMUP_PROMPT] * 2, dummy_params)
    print("Warm-up complete. Waiting for GPU to settle.\n")
    time.sleep(3)

if __name__ == "__main__":
    if not check_disk_space():
        exit(1)

    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3" 
    save_system_info(model_path, script_name="log_pd")

    print(f"Initializing vLLM with {model_path}...")
    # 关闭 chunked prefill，避免超长 prompt 被切碎后形成长时间锯齿功耗。
    llm = LLM(model=model_path, enforce_eager=True, enable_chunked_prefill=False) 
    print("Loading long prompt from ./prompt/long_prompt.txt ...")
    paper_style_long_prompt = load_long_prompt()

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
        
        with GPUMonitor(interval=0.01, monitor_clock=True) as prefill_monitor:
            output_prefill = llm.generate([long_prompt], prefill_params)

        # 提取生成的 token 数量和输出文本
        prefill_tokens = len(output_prefill[0].outputs[0].token_ids)
        prefill_output_text = output_prefill[0].outputs[0].text
        
        # 传入输入输出文本保存 (注意：文件名加上了 idx，防止被下一次循环覆盖)
        filename = f"./log/prefill_{idx+3}_power_log.csv"
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
        
        with GPUMonitor(interval=0.01, monitor_clock=True) as decode_monitor:
            output_decode = llm.generate([short_prompt], decode_params)

        # 提取生成的 token 数量和输出文本
        decode_tokens = len(output_decode[0].outputs[0].token_ids)
        decode_output_text = output_decode[0].outputs[0].text
        
        # 传入输入输出文本保存 (注意：文件名加上了 idx)
        filename = f"./log/decode_{idx+3}_power_log.csv"
        decode_monitor.save_and_calculate(filename, decode_tokens,
                                         input_text=short_prompt, output_text=decode_output_text)
        
        time.sleep(3)

    print("\nAll Tasks completed successfully!")