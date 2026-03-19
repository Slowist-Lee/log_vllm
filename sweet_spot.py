#!/usr/bin/env python3
"""
Sweet Spot 频率扫描脚本

目标：找到 Prefill 和 Decode 阶段能效最优（TPJ 最高）的 GPU 频率
TPJ (Tokens Per Joule) = 生成的 Token 总数 / 消耗的总能量 (Total Energy)

使用方法:
    python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 --phases prefill
    python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 --phases decode
    python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 --phases e2e
    python sweet_spot.py --frequencies 750 900 1050 1200 --repeat 1 --phases prefill
    python sweet_spot.py --freq 1000 --repeat 1 --phases e2e
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


def generate_prompts(base_prompt: str, target_tokens: int, num_prompts: int = 1) -> List[str]:
    """
    生成指定长度的 prompt 列表。
    如果 target_tokens > 实际 prompt 的 token 数，则重复或截断 prompt。
    """
    # 简单估计：1 token ≈ 4 个字符（英文）
    target_chars = target_tokens * 4
    actual_len = len(base_prompt)

    if actual_len == 0:
        return [base_prompt] * num_prompts

    # 通过重复或截断来匹配目标长度
    if actual_len < target_chars:
        multiplier = (target_chars // actual_len) + 1
        adjusted_prompt = (base_prompt + " ") * multiplier
        adjusted_prompt = adjusted_prompt[:target_chars].strip()
    else:
        adjusted_prompt = base_prompt[:target_chars].strip()

    return [adjusted_prompt] * num_prompts


def set_gpu_frequency(freq_mhz: int) -> bool:
    """设置 GPU 频率，返回是否成功"""
    try:
        subprocess.run(
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
        subprocess.run(
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


def run_prefill_test(llm, prompt: str, freq_mhz: int, requests_per_measure: int = 10) -> Dict:
    """运行 Prefill 阶段测试（单次测量窗口内累计多个请求，降低短任务采样噪声）"""
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)

    total_input_tokens = 0
    total_output_tokens = 0
    ttft_list = []
    tpot_list = []

    start_time = time.perf_counter()
    with GPUMonitor() as monitor:
        for _ in range(max(requests_per_measure, 1)):
            req_start = time.perf_counter()
            output = llm.generate([prompt], prefill_params, use_tqdm=False)
            req_duration = time.perf_counter() - req_start
            in_tokens = len(output[0].prompt_token_ids)
            out_tokens = len(output[0].outputs[0].token_ids)
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens

            # 每个请求仍单独提取 TTFT/TPOT，再对测量窗口取平均
            ttft_s, tpot_s = extract_ttft_tpot(output[0], req_duration, out_tokens)
            ttft_list.append(ttft_s)
            tpot_list.append(tpot_s)

    duration = time.perf_counter() - start_time
    metrics = monitor.get_metrics(total_input_tokens)

    avg_ttft_s = sum(ttft_list) / len(ttft_list) if ttft_list else duration
    avg_tpot_s = sum(tpot_list) / len(tpot_list) if tpot_list else 0.0

    # Prefill 的能效应按处理输入 token 来衡量，而非仅 1 个输出 token。
    tpj = total_input_tokens / metrics['total_energy_j'] if metrics['total_energy_j'] > 0 else 0

    return {
        'phase': 'Prefill',
        'frequency_mhz': freq_mhz,
        'duration_s': metrics['duration_s'],
        'ttft_s': round(avg_ttft_s, 4),
        'tpot_s': round(avg_tpot_s, 6),
        'avg_power_w': metrics['avg_power_w'],
        'peak_power_w': metrics['peak_power_w'],
        'total_energy_j': metrics['total_energy_j'],
        'throughput_tps': metrics['throughput_tps'],
        'j_per_token': metrics['j_per_token'],
        'total_output_tokens': total_output_tokens,
        'tpj': round(tpj, 4),
        'input_tokens': total_input_tokens
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
        'tpj': round(tpj, 4),
        'input_tokens': len(output[0].prompt_token_ids)
    }


def run_e2e_test(engine, prompts: list, freq_mhz: int, max_tokens: int = 256, batch_size: int = 1) -> Dict:
    """运行端到端批处理测试（使用 LLMEngine.step 精确测量 TTFT）"""
    from inference_core import run_batch_e2e_requests
    
    with GPUMonitor() as monitor:
        if batch_size > 1 and len(prompts) > 1:
            # 批处理模式
            e2e_result = run_batch_e2e_requests(engine, prompts[:batch_size], max_tokens=max_tokens)
            output_tokens = e2e_result['total_output_tokens']
            input_tokens = e2e_result.get('total_input_tokens', 0)
        else:
            # 单请求模式
            from inference_core import run_e2e_request
            e2e_result = run_e2e_request(engine, prompts[0], max_tokens=max_tokens)
            output_tokens = e2e_result['output_tokens']
            input_tokens = e2e_result.get('input_tokens', 0)

    total_processed_tokens = input_tokens + output_tokens
    metrics = monitor.get_metrics(total_processed_tokens)

    tpj = total_processed_tokens / metrics['total_energy_j'] if metrics['total_energy_j'] > 0 else 0

    return {
        'phase': 'E2E',
        'frequency_mhz': freq_mhz,
        'batch_size': batch_size,
        'duration_s': metrics['duration_s'],
        'ttft_s': e2e_result.get('ttft_s') or e2e_result.get('mean_ttft_s', 0),
        'tpot_s': e2e_result.get('tpot_s') or e2e_result.get('mean_tpot_s', 0),
        'avg_power_w': metrics['avg_power_w'],
        'peak_power_w': metrics['peak_power_w'],
        'total_energy_j': metrics['total_energy_j'],
        'throughput_tps': metrics['throughput_tps'],
        'j_per_token': metrics['j_per_token'],
        'total_output_tokens': output_tokens,
        'tpj': round(tpj, 4),
        'input_tokens': input_tokens
    }


def find_sweet_spot(results: List[Dict], phase: str) -> Tuple[Dict, List[Dict]]:
    """
    找到指定阶段的 Sweet Spot（TPJ 最高的频率）
    返回: (最佳结果, 所有结果按 TPJ 排序)
    """
    phase_results = [r for r in results if r['phase'] == phase]
    if not phase_results:
        return None, []

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
                note = ">>> Sweet Spot (最高能效)"
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
  # 扫描 750-1300 MHz，步长 50 MHz，重复 1 次，仅 Prefill 阶段
  python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 --phases prefill

  # E2E 测试：Batch Size=16, Input Length=1024 tokens, Max Tokens=256
  python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 --phases e2e \\
    --batch-size 16 --input-length 1024 --decode-tokens 256

  # 指定频率列表，E2E 批处理
  python sweet_spot.py --frequencies 750 900 1050 1200 --repeat 1 --phases e2e --batch-size 32

  # 单个频率测试，E2E 配置
  python sweet_spot.py --freq 1000 --repeat 1 --phases e2e --batch-size 16 --input-length 512
        """
    )

    # 频率参数（三选一）
    freq_group = parser.add_mutually_exclusive_group(required=True)
    freq_group.add_argument("--freq", type=int, help="单个测试频率 (MHz)")
    freq_group.add_argument("--start", type=int, help="起始频率 (MHz)，与 --end/--step 配合使用")
    freq_group.add_argument("--frequencies", type=int, nargs="+", help="指定频率列表 (MHz)")
    parser.add_argument("--end", type=int, default=1300, help="结束频率 (MHz)，默认 1300")
    parser.add_argument("--step", type=int, default=50, help="频率步长 (MHz)，默认 50")

    # 测试参数
    parser.add_argument("--repeat", type=int, default=1, help="每个频率重复测试次数，默认 1")
    parser.add_argument("--phases", type=str, nargs="+", default=["prefill", "decode"],
                        choices=["prefill", "decode", "e2e"],
                        help="测试阶段，默认: prefill decode")
    
    # E2E 专用参数
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="E2E 测试的批处理大小（默认 1），推荐 16-32")
    parser.add_argument("--input-length", type=int, default=512, 
                        help="输入 prompt 的 token 数（默认 512），推荐 512-1024")
    parser.add_argument("--decode-tokens", type=int, default=256, 
                        help="Decode/E2E 阶段生成的 token 数（默认 256），推荐 128-256")
    parser.add_argument("--warmup", action="store_true", help="是否在正式测试前进行预热")
    parser.add_argument("--prefill-requests", type=int, default=10,
                        help="Prefill 单次测量窗口内连续请求数（默认 10，用于降低短任务采样抖动）")

    # 输出参数
    parser.add_argument("--output", type=str, default="./log/sweet_spot_results.csv",
                        help="结果输出文件路径")

    args = parser.parse_args()

    # 构建频率列表
    if args.freq is not None:
        frequencies = [args.freq]
    elif args.frequencies is not None:
        frequencies = args.frequencies
    else:
        frequencies = list(range(args.start, args.end + 1, args.step))

    print(f"[*] Sweet Spot 频率扫描")
    print(f"[*] 测试频率列表: {frequencies}")
    print(f"[*] 重复测试: {args.repeat} 次")
    print(f"[*] 测试阶段: {args.phases}")
    if "e2e" in args.phases:
        print(f"[*] E2E 配置: Batch Size={args.batch_size}, Input Length={args.input_length} tokens, Max Tokens={args.decode_tokens}")
    print(f"[*] 总测试次数: {len(frequencies) * args.repeat * len(args.phases)}")

    # 加载 prompt
    base_prompt = load_long_prompt()
    prefill_prompts = generate_prompts(base_prompt, args.input_length, num_prompts=max(args.batch_size, 1))
    decode_prompt = "Hello."

    # 模型路径
    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    save_system_info(model_path, script_name="sweet_spot")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_summary_rows = []

    try:
        # 初始化模型（只需一次，复用于所有频率）
        print("\n[*] 初始化模型...")
        llm = None
        engine = None

        if "prefill" in args.phases or "decode" in args.phases:
            llm = LLM(model=model_path, enforce_eager=True, enable_prefix_caching=False)
            if args.warmup:
                print("[*] 预热中...")
                llm.generate([decode_prompt], SamplingParams(max_tokens=10), use_tqdm=False)
                time.sleep(2)

        if "e2e" in args.phases:
            engine = build_engine(model_path, max_num_seqs=args.batch_size)
            if args.warmup:
                print("[*] E2E 预热中...")
                warmup_params = SamplingParams(temperature=0.0, max_tokens=8)
                engine.add_request(request_id="warmup", prompt="hello", params=warmup_params)
                while True:
                    outs = engine.step()
                    if any(o.request_id == "warmup" and o.finished for o in outs):
                        break
                time.sleep(2)

        # 遍历所有频率
        for freq in frequencies:
            print(f"\n{'='*80}")
            print(f" 测试频率: {freq} MHz")
            print(f"{'='*80}")

            if not set_gpu_frequency(freq):
                print(f"[WARNING] 跳过频率 {freq} MHz（设置失败）")
                continue

            freq_results = []

            for repeat_idx in range(args.repeat):
                print(f"\n[重复 {repeat_idx+1}/{args.repeat}]")

                if "prefill" in args.phases:
                    result = run_prefill_test(
                        llm,
                        prefill_prompts[0],
                        freq,
                        requests_per_measure=args.prefill_requests,
                    )
                    freq_results.append(result)
                    print("  Prefill 测试完成")
                    time.sleep(1)

                if "decode" in args.phases:
                    result = run_decode_test(llm, decode_prompt, freq, args.decode_tokens)
                    freq_results.append(result)
                    print("  Decode 测试完成")
                    time.sleep(1)

                if "e2e" in args.phases:
                    result = run_e2e_test(engine, prefill_prompts, freq, args.decode_tokens, args.batch_size)
                    freq_results.append(result)
                    print("  E2E 测试完成")
                    time.sleep(1)

            # 追加原始结果到 CSV（每个频率测完立即保存）
            if freq_results:
                df = pd.DataFrame(freq_results)
                mode = 'a' if os.path.exists(args.output) else 'w'
                header = not os.path.exists(args.output)
                df.to_csv(args.output, index=False, mode=mode, header=header)
                print(f"[*] 频率 {freq} MHz 结果已追加: {args.output}")

            # 汇总当前频率的平均值
            if freq_results:
                df = pd.DataFrame(freq_results)
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
                        'input_tokens': int(phase_df['input_tokens'].mean()) if 'input_tokens' in phase_df else 0,
                        'repeat_count': len(phase_df)
                    }
                    all_summary_rows.append(avg_row)

        # 重置 GPU 频率
        print("\n[*] 重置 GPU 频率...")
        reset_gpu_frequency()

        # 保存汇总 CSV（覆盖，因为每次 phase 扫描都会单独跑）
        if all_summary_rows:
            summary_csv = args.output.replace('.csv', '_summary.csv')
            summary_df = pd.DataFrame(all_summary_rows)
            mode = 'a' if os.path.exists(summary_csv) else 'w'
            header = not os.path.exists(summary_csv)
            summary_df.to_csv(summary_csv, index=False, mode=mode, header=header)
            print(f"[*] 汇总结果已保存: {summary_csv}")

            # 打印 Sweet Spot 结果表
            print_sweet_spot_table(all_summary_rows, f"Sweet Spot 扫描结果 ({', '.join(args.phases)})")

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
