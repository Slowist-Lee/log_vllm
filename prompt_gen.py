#!/usr/bin/env python3
import csv
import os
import pandas as pd
from transformers import AutoTokenizer

def generate_exact_length_prompts(
    model_name="meta-llama/Meta-Llama-3-8B",
    # 你的 Tokenizer 本地路径
    tokenizer_path='./llama3_token/',
    parquet_file="./data/train-00000-of-00002.parquet",
    output_file="./prompt/llama3_test_prompts.csv",
    num_short=25, short_len=100,
    num_long=25, long_len=10000
):
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    print(f"Reading dataset: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    # 过滤掉非字符串和空行，确保数据干净
    valid_texts = [t for t in df['text'] if isinstance(t, str) and t.strip()]
    
    prompts = []
    
    print("Generating Short Prompts...")
    short_collected = 0
    short_prefix_text = "Analyze the following:\n\n"
    short_prefix_tokens = tokenizer.encode(short_prefix_text, add_special_tokens=False)
    for text in valid_texts:
        if short_collected >= num_short:
            break
            
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        
        tokens = short_prefix_tokens + text_tokens

        if len(tokens) >= short_len:
            exact_tokens = tokens[:short_len]
            exact_text = tokenizer.decode(exact_tokens)
            
            prompts.append({
                "id": f"short_{short_collected:02d}",
                "category": "short",
                "exact_tokens": short_len,
                "prompt": exact_text
            })
            short_collected += 1
            print(f"  [Short] Collected {short_collected}/{num_short}")

    print("\nGenerating Long Prompts (Summarize + Concatenation)...")
    long_collected = 0
    
    # 定义前缀
    prefix_text = "Summarize the following text:\n\n"
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    
    # 初始化当前缓冲区（放入前缀）
    current_tokens = list(prefix_tokens)
    
    for text in valid_texts[::-1]: # 倒序遍历，增加一点数据的多样性
        if long_collected >= num_long:
            break
        
        # 获取当前行的 token
        # 注意：在拼接时最好加一个换行符，防止单词粘连
        line_tokens = tokenizer.encode(text + "\n", add_special_tokens=False)
        
        # 拼接到当前缓冲区
        current_tokens.extend(line_tokens)
        
        # 检查长度是否达标
        if len(current_tokens) >= long_len:
            # 1. 截取到精确长度
            exact_tokens = current_tokens[:long_len]
            
            # 2. 解码回文本
            exact_text = tokenizer.decode(exact_tokens)
            
            # 3. 保存
            prompts.append({
                "id": f"long_{long_collected:02d}",
                "category": "long",
                "exact_tokens": long_len,
                "prompt": exact_text
            })
            long_collected += 1
            print(f"  [Long] Collected {long_collected}/{num_long}")

            current_tokens = list(prefix_tokens)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ["id", "category", "exact_tokens", "prompt"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for p in prompts:
            writer.writerow(p)
            
    print(f"Done! Saved {len(prompts)} prompts to {output_file}")

if __name__ == "__main__":
    generate_exact_length_prompts()