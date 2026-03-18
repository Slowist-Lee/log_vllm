#!/usr/bin/env python3
"""
Sweet Spot 频率扫描脚本

目标：找到 Prefill 和 Decode 阶段能效最优（TPJ 最高）的 GPU 频率
TPJ (Tokens Per Joule) = 生成的 Token 总数 / 消耗的总能量 (Total Energy)

使用方法:
    python sweet_spot.py --start 600 --end 1500 --step 50 --repeat 3
    python sweet_spot.py --frequencies 600 700 800 900 1000 1100 1200 1300 1400 1500 --repeat 5
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd
from typing import List, Dict, Tuple
from inference_core import build_engine, run_e2e_request
from gpu_utils import load_long_prompt, save_system_info, extract_ttft_tpot, GPUMonitor
from vllm import LLM, SamplingParams


def set_gpu_frequency(freq_mhz: int) -> bool:
    """设置 GPU 频率，返回是否成功"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-lgc", f"{freq_mhz},{freq_mhz}"],
            capture_output=True,
            text=True,
            check=True
        )
        time.sleep(2)  # 给 GPU 时间稳定频率
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 设置频率 {freq_mhz} MHz 失败: {e}")
        print(f"[ERROR] stderr: {e.stderr}")
        return False


def reset_gpu_frequency() -> bool:
    """重置 GPU 频率到默认值"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-rgc"],
            capture_output=True,
            text=True,
            check=True
        )
        time.sleep(1)
        print("[*] GPU 频率已重置为默认值")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] 重置 GPU 频率失败: {e}")
        return False


def run_prefill_test(llm, prompt: str, freq_mhz: int) -> Dict:
    """运行 Prefill 阶段测试"""
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
    
    start_time = time.perf_counter()
    with GPUMonitor() as monitor:
        output = llm.generate([prompt], prefill_params, use_tqdm=False)
    duration = time.perf_counter() - start_time
    
    in_tokens = len(output[0].prompt_token_ids)
    out_tokens = len(output[0].outputs[0].token_ids)
    metrics = monitor.get_metrics(out_tokens)
    ttft_s, tpot_s = extract_ttft_tpot(output[0], duration, out_tokens)
    
    # 计算 TPJ (Tokens Per Joule)
    tpj = out_tokens / metrics['total_energy_j'] if metrics['total_energy_j'] > 0 else 0
    
    return {
        'phase': 'Prefill',
        'frequency_mhz': freq_mhz,
        'duration_s': metrics['duration_s'],
        'ttft_s': ttft_s,
        'tpot_s': tpot_s,
        'avg_power_w': metrics['avg_power_w'],
        'peak_power_w': metrics['peak_power_w'],
        'total_energy_j': metrics['total_energy_j'],
        'throughput_tps': metrics['throughput_tps'],
        'j_per_token': metrics['j_per_token'],
        'total_output_tokens': out_tokens,
        'tpj': round(tpj, 4),  # Tokens per Joule - 能效指标
        'input_tokens': in_tokens
    }


def run_decode_test(llm, prompt: str, freq_mhz: int, max_tokens: int = 256) -> Dict:
    """运行 Decode 阶段测试"""
    decode_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    
    start_time = time.perf_counter()
    with GPUMonitor() as monitor:
        output = llm.generate([prompt], decode_params, use_tqdm=False)
    duration = time.perf_counter() - start_time
    
    out_tokens = len(output[0].outputs[0].token_ids)
    metrics = monitor.get_metrics(out_tokens)
    ttft_s, tpot_s = extract_ttft_tpot(output[0], duration, out_tokens)
    
    # 计算 TPJ (Tokens Per Joule)
    tpj = out_tokens / metrics['total_energy_j'] if metrics['total_energy_j'] > 0 else 0
    
    return {
        'phase': 'Decode',
        'frequency_mhz': freq_mhz,
        'duration_s': metrics['duration_s'],
        'ttft_s': ttft_s,
        'tpot_s': tpot_s,
        'avg_power_w': metrics['avg_power_w'],
        'peak_power_w': metrics['peak_power_w'],
        'total_energy_j': metrics['total_energy_j'],
        'throughput_tps': metrics['throughput_tps'],
        'j_per_token': metrics['j_per_token'],
        'total_output_tokens': out_tokens,
        'tpj': round(tpj, 4),  # Tokens per Joule - 能效指标
        'input_tokens': len(output[0].prompt_token_ids)
    }


def run_e2e_test(engine, prompt: str, freq_mhz: int, max_tokens: int = 256) -> Dict:
    """运行端到端测试（使用 LLMEngine.step 精确测量 TTFT）"""
    with GPUMonitor() as monitor:
        e2e_result = run_e2e_request(engine, prompt, max_tokens=max_tokens)
    
    metrics = monitor.get_metrics(e2e_result['output_tokens'])
    
    # 计算 TPJ (Tokens Per Joule)
    tpj = e2e_result['output_tokens'] / metrics['total_energy_j'] if metrics['total_energy_j'] > 0 else 0
    
    return {
        'phase': 'E2E',
        'frequency_mhz': freq_mhz,
        'duration_s': metrics['duration_s'],
        'ttft_s': e2e_result['ttft_s'],
        'tpot_s': e2e_result['tpot_s'],
        'avg_power_w': metrics['avg_power_w'],
        'peak_power_w': metrics['peak_power_w'],
        'total_energy_j': metrics['total_energy_j'],
        'throughput_tps': metrics['throughput_tps'],
        'j_per_token': metrics['j_per_token'],
        'total_output_tokens': e2e_result['output_tokens'],
        'tpj': round(tpj, 4)  # Tokens per Joule - 能效指标
    }


def find_sweet_spot(results: List[Dict], phase: str) -> Tuple[Dict, List[Dict]]:
    """
    找到指定阶段的 Sweet Spot（TPJ 最高的频率）
    返回: (最佳结果, 所有结果按 TPJ 排序)
    """
    phase_results = [r for r in results if r['phase'] == phase]
    if not phase_results:
        return None, []
    
    # 按 TPJ 降序排序
    sorted_results = sorted(phase_results, key=lambda x: x['tpj'], reverse=True)
    return sorted_results[0], sorted_results


def print_sweet_spot_table(results: List[Dict], title: str = "Sweet Spot 扫描结果"):
    """打印 Sweet Spot 结果表格"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    
    for phase in ['Prefill', 'Decode', 'E2E']:
        best, sorted_results = find_sweet_spot(results, phase)
        if not sorted_results:
            continue
        
        print(f"\n【{phase} 阶段】")
        print(f"{'Phase':<10} {'Freq(MHz)':<12} {'J/Token':<12} {'TPJ':<12} {'Note'}")
        print(f"{'-'*70}")
        
        for i, r in enumerate(sorted_results):
            note = ""
            if i == 0:
                note = "💡 最高能效 (Sweet Spot)"
            elif i == 1:
                note = "能效下降"
            elif i == len(sorted_results) - 1:
                note = "能效最低"
            elif r['tpj'] < best['tpj'] * 0.9:
                note = "能效显著下降"
            elif r['tpj'] < best['tpj'] * 0.95:
                note = "能效开始下降"
            
            print(f"{r['phase']:<10} {r['frequency_mhz']:<12} {r['j_per_token']:<12.4f} "
                  f"{r['tpj']:<12.4f} {note}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweet Spot 频率扫描 - 找到能效最优的 GPU 频率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 扫描 600-1500 MHz，步长 50 MHz，每个频率重复 3 次
  python sweet_spot.py --start 600 --end 1500 --step 50 --repeat 3
  
  # 指定特定频率列表
  python sweet_spot.py --frequencies 600 800 1000 1100 1200 1400 1500 --repeat 5
  
  # 只扫描 Prefill 阶段
  python sweet_spot.py --start 600 --end 1500 --step 100 --phases prefill
  
  # 包含 E2E 测试
  python sweet_spot.py --start 600 --end 1500 --step 100 --phases prefill decode e2e
        """
    )
    
    # 单个频率参数
    parser.add_argument("--freq", type=int, required=True, help="测试频率 (MHz)")
    
    # 测试参数
    parser.add_argument("--repeat", type=int, default=3, help="每个频率重复测试次数，默认 3")
    parser.add_argument("--phases", type=str, nargs="+", default=["prefill", "decode"],
                        choices=["prefill", "decode", "e2e"],
                        help="测试阶段，默认: prefill decode")
    parser.add_argument("--decode-tokens", type=int, default=256, help="Decode 阶段生成的 token 数，默认 256")
    parser.add_argument("--warmup", action="store_true", help="是否在正式测试前进行预热")
    
    # 输出参数
    parser.add_argument("--output", type=str, default="./log/sweet_spot_results.csv",
                        help="结果输出文件路径")
    parser.add_argument("--summary", type=str, default="./log/sweet_spot_summary.txt",
                        help="摘要输出文件路径")
    
    args = parser.parse_args()
    
    # 使用单个频率
    freq = args.freq
    
    print(f"[*] Sweet Spot 频率测试")
    print(f"[*] 测试频率: {freq} MHz")
    print(f"[*] 重复测试: {args.repeat} 次")
    print(f"[*] 测试阶段: {args.phases}")
    print(f"[*] 总测试次数: {args.repeat * len(args.phases)}")
    
    # 加载 prompt
    prefill_prompt = load_long_prompt()
    decode_prompt = "Hello."
    
    # 模型路径
    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    save_system_info(model_path, script_name="sweet_spot")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    all_results = []
    
    try:
        # 初始化模型（只需初始化一次）
        print("\n[*] 初始化模型...")
        llm = None
        engine = None
        
        if "prefill" in args.phases or "decode" in args.phases:
            llm = LLM(model=model_path, enforce_eager=True)
            # 预热
            if args.warmup:
                print("[*] 预热中...")
                llm.generate([decode_prompt], SamplingParams(max_tokens=10), use_tqdm=False)
                time.sleep(2)
        
        if "e2e" in args.phases:
            engine = build_engine(model_path)
            if args.warmup:
                print("[*] E2E 预热中...")
                warmup_params = SamplingParams(temperature=0.0, max_tokens=8)
                engine.add_request(request_id="warmup", prompt="hello", params=warmup_params)
                while True:
                    outs = engine.step()
                    if any(o.request_id == "warmup" and o.finished for o in outs):
                        break
                time.sleep(2)
        
        # 开始频率测试
        print(f"\n{'='*80}")
        print(f" 开始测试频率: {freq} MHz")
        print(f"{'='*80}")
        
        for repeat_idx in range(args.repeat):
            print(f"\n[重复 {repeat_idx+1}/{args.repeat}]")
            
            # Prefill 测试
            if "prefill" in args.phases:
                result = run_prefill_test(llm, prefill_prompt, freq)
                all_results.append(result)
                print("  Prefill 测试完成")
            
            time.sleep(1)
            
            # Decode 测试
            if "decode" in args.phases:
                result = run_decode_test(llm, decode_prompt, freq, args.decode_tokens)
                all_results.append(result)
                print("  Decode 测试完成")
            
            time.sleep(1)
            
            # E2E 测试
            if "e2e" in args.phases:
                result = run_e2e_test(engine, prefill_prompt, freq, args.decode_tokens)
                all_results.append(result)
                print("  E2E 测试完成")
            
            time.sleep(1)
        
        # 保存原始结果到 CSV（追加模式）
        if all_results:
            df = pd.DataFrame(all_results)
            mode = 'a' if os.path.exists(args.output) else 'w'
            header = not os.path.exists(args.output)
            df.to_csv(args.output, index=False, mode=mode, header=header)
            print(f"\n[*] 原始结果已保存: {args.output}")
        
        # 计算每个频率的平均值（仅针对当前频率）
        if all_results:
            print(f"\n{'='*80}")
            print(f" 频率 {freq} MHz 测试结果（平均值）")
            print(f"{'='*80}")
            
            df = pd.DataFrame(all_results)
            
            # 按 phase 分组计算平均值
            summary_rows = []
            for phase in df['phase'].unique():
                phase_df = df[df['phase'] == phase]
                avg_row = {
                    'phase': phase,
                    'frequency_mhz': freq,
                    'duration_s': round(phase_df['duration_s'].mean(), 4),
                    'ttft_s': round(phase_df['ttft_s'].mean(), 4),
                    'tpot_s': round(phase_df['tpot_s'].mean(), 6),
                    'avg_power_w': round(phase_df['avg_power_w'].mean(), 2),
                    'peak_power_w': round(phase_df['peak_power_w'].mean(), 2),
                    'total_energy_j': round(phase_df['total_energy_j'].mean(), 4),
                    'throughput_tps': round(phase_df['throughput_tps'].mean(), 2),
                    'j_per_token': round(phase_df['j_per_token'].mean(), 4),
                    'total_output_tokens': int(phase_df['total_output_tokens'].mean()),
                    'tpj': round(phase_df['tpj'].mean(), 4),
                    'repeat_count': len(phase_df)
                }
                summary_rows.append(avg_row)
            
            # 保存汇总结果到 CSV（追加模式）
            summary_csv = args.output.replace('.csv', '_summary.csv')
            summary_df = pd.DataFrame(summary_rows)
            mode = 'a' if os.path.exists(summary_csv) else 'w'
            header = not os.path.exists(summary_csv)
            summary_df.to_csv(summary_csv, index=False, mode=mode, header=header)
            print(f"[*] 汇总结果已保存: {summary_csv}")
            
            # 打印当前频率的 Sweet Spot 表格
            print_sweet_spot_table(summary_rows, f"频率 {freq} MHz 测试结果")
        
        print(f"\n{'='*80}")
        print(" Sweet Spot 扫描完成!")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n\n[!] 用户中断测试")
        reset_gpu_frequency()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] 测试出错: {e}")
        reset_gpu_frequency()
        raise


if __name__ == "__main__":
    main()
