#!/usr/bin/env python3
import csv
import os
import pandas as pd
from transformers import AutoTokenizer

def generate_exact_length_prompts(
    # 1. 修改为你 Mistral 模型的本地实际路径
    model_path="./mistral_7b_model/LLM-Research/Mistral-7B-v0.3", 
    parquet_file="./data/train-00000-of-00002.parquet",
    output_file="./prompt/mistral_test_prompts.csv",
    num_short=25, short_len=100,
    num_long=25, long_len=10000
):
    print(f"Loading tokenizer from: {model_path}")
    # Mistral 通常不需要 trust_remote_code，但加上也无妨
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 确保有起始符 <s> 的设置
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"

    print(f"Reading dataset: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    valid_texts = [t for t in df['text'] if isinstance(t, str) and len(t) > 10]
    
    prompts = []

    # --- 生成 Short Prompts ---
    print(f"Generating Short Prompts (Len: {short_len})...")
    short_collected = 0
    short_prefix_text = "Analyze the following text:\n"
    
    for text in valid_texts:
        if short_collected >= num_short:
            break
            
        # 编码前缀和文本
        # 我们手动处理 BOS，所以 add_special_tokens 设为 False
        prefix_tokens = tokenizer.encode(tokenizer.bos_token + short_prefix_text, add_special_tokens=False)
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        
        combined = prefix_tokens + text_tokens

        if len(combined) >= short_len:
            exact_tokens = combined[:short_len]
            exact_text = tokenizer.decode(exact_tokens, skip_special_tokens=False)
            
            prompts.append({
                "id": f"short_{short_collected:02d}",
                "category": "short",
                "exact_tokens": short_len,
                "prompt": exact_text
            })
            short_collected += 1
            if short_collected % 5 == 0:
                print(f"  [Short] {short_collected}/{num_short}")

    # --- 生成 Long Prompts ---
    print(f"\nGenerating Long Prompts (Len: {long_len})...")
    long_collected = 0
    long_prefix_text = "Summarize the following document in detail:\n\n"
    
    # 初始化缓冲区，放入 BOS 和前缀
    current_tokens = tokenizer.encode(tokenizer.bos_token + long_prefix_text, add_special_tokens=False)
    
    for text in valid_texts[::-1]: # 倒序增加多样性
        if long_collected >= num_long:
            break
        
        line_tokens = tokenizer.encode(text + "\n\n", add_special_tokens=False)
        current_tokens.extend(line_tokens)
        
        if len(current_tokens) >= long_len:
            exact_tokens = current_tokens[:long_len]
            exact_text = tokenizer.decode(exact_tokens, skip_special_tokens=False)
            
            prompts.append({
                "id": f"long_{long_collected:02d}",
                "category": "long",
                "exact_tokens": long_len,
                "prompt": exact_text
            })
            long_collected += 1
            print(f"  [Long] {long_collected}/{num_long}")

            # 重置缓冲区并添加前缀供下一条使用
            current_tokens = tokenizer.encode(tokenizer.bos_token + long_prefix_text, add_special_tokens=False)

    # --- 保存结果 ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "exact_tokens", "prompt"])
        writer.writeheader()
        writer.writerows(prompts)
            
    print(f"Success! Generated {len(prompts)} prompts.")

if __name__ == "__main__":
    # 在这里填入你本地 Mistral 文件夹的路径
    MY_MISTRAL_PATH = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3" 
    
    generate_exact_length_prompts(
        model_path=MY_MISTRAL_PATH,
        short_len=128,    # 可以根据需要调整
        long_len=8192     # Mistral v0.1 原生支持 8k，v0.3 支持更多，10k 也可以
    )