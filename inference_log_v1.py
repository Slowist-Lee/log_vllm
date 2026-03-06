import csv
import os

if __name__ == "__main__":

    input_csv_path = "./prompt/llama3_test_prompt.csv"
    output_csv_path = "./log/llama3_test_output.csv"
    power_log_path = "./log/inference_power_log.csv"

    # 1. 从 CSV 读取 prompts
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

    # 2. 初始化 vLLM (Task 1)
    # 替换为你实际的模型路径，比如 "meta-llama/Meta-Llama-3-8B"
    model_name = "meta-llama/Meta-Llama-3-8B" 
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # 3. 执行推理并同步监控功耗 (Task 1 & Task 2)
    print("Starting inference and power monitoring...")
    with GPUMonitor(interval=0.075) as monitor:
        # vLLM 批量推理
        outputs = llm.generate(prompts, sampling_params)

    # 4. 提取输出与 Token 统计
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

    # 5. 将输出写回新的 CSV 文件
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "output", "output_tokens"])
        writer.writeheader()
        writer.writerows(results_to_save)
        
    print(f"Inference complete! Outputs and token counts saved to {output_csv_path}")

    # 6. 保存功耗日志并计算能耗
    monitor.save_and_calculate(filename=power_log_path, 
                               total_output_tokens=total_generated_tokens)