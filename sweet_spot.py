#!/usr/bin/env python3
"""
Sweet Spot 频率扫描脚本

目标：找到 Prefill 和 Decode 阶段能效最优（TPJ 最高）的 GPU 频率
TPJ (Tokens Per Joule) = 生成的 Token 总数 / 消耗的总能量 (Total Energy)

使用方法:
    python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 --phases prefill
    python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 --phases decode
    python sweet_spot.py --frequencies 750 900 1050 1200 --repeat 1 --phases prefill
    python sweet_spot.py --freq 1000 --repeat 1 --phases decode
"""

import os
import sys
import time
import uuid
import argparse
import pandas as pd
from typing import List, Dict, Tuple
from inference_core import build_engine
from gpu_utils import load_long_prompt, save_system_info, GPUMonitor
from vllm import SamplingParams


def _build_monitor_df(monitor: GPUMonitor) -> pd.DataFrame:
    """将 GPUMonitor 的原始数据转换为带能量列的 DataFrame。"""
    if not monitor.data:
        return pd.DataFrame(columns=["timestamp", "time_offset", "power_w", "time_interval", "energy_j"])

    df = pd.DataFrame(monitor.data).copy()
    df["time_interval"] = df["timestamp"].diff().fillna(monitor.interval)
    df["energy_j"] = df["power_w"] * df["time_interval"]
    return df


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


def run_backlog_requests_with_ttft(
    engine,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    backlog_multiplier: int = 4,
) -> Dict[str, float]:
    """
    运行 backlog 压测，同时记录每个请求的 TTFT 和完成时刻。
    返回汇总和每个请求的时序数据，供后续 prefill/decode 分段计量。
    """
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    total_requests = max(concurrency * backlog_multiplier, concurrency)

    request_ids = []
    for idx in range(total_requests):
        req_id = f"backlog-{uuid.uuid4()}-{idx}"
        engine.add_request(request_id=req_id, prompt=prompt, params=sampling_params)
        request_ids.append(req_id)

    request_id_set = set(request_ids)
    seen_ttft: Dict[str, float] = {}
    seen_finish: Dict[str, float] = {}
    request_tokens: Dict[str, Dict] = {rid: {"input": 0, "output": 0} for rid in request_ids}
    
    t0 = time.perf_counter()
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        now = time.perf_counter()
        for out in step_outputs:
            rid = out.request_id
            if rid not in request_id_set:
                continue

            generated_len = len(out.outputs[0].token_ids) if out.outputs else 0
            if rid not in seen_ttft and generated_len > 0:
                seen_ttft[rid] = now - t0
            
            if out.finished:
                seen_finish[rid] = now - t0
                request_tokens[rid]["input"] = len(out.prompt_token_ids) if out.prompt_token_ids else 0
                request_tokens[rid]["output"] = generated_len

    total_duration_s = time.perf_counter() - t0
    mean_ttft_s = sum(seen_ttft.values()) / len(seen_ttft) if seen_ttft else total_duration_s
    mean_output_tokens = sum(t["output"] for t in request_tokens.values()) / max(len(seen_finish), 1)
    mean_tpot_s = max(total_duration_s - mean_ttft_s, 0.0) / max(mean_output_tokens, 1.0)

    return {
        'total_duration_s': round(total_duration_s, 4),
        'mean_ttft_s': round(mean_ttft_s, 4),
        'mean_tpot_s': round(mean_tpot_s, 6),
        'total_input_tokens': sum(t["input"] for t in request_tokens.values()),
        'total_output_tokens': sum(t["output"] for t in request_tokens.values()),
        'finished_requests': len(seen_finish),
        'backlog_requests': total_requests,
        'request_ttft_dict': seen_ttft,
        'request_finish_dict': seen_finish,
        'request_tokens': request_tokens,
    }



def run_unified_test(
    engine,
    prompt: str,
    freq_mhz: int,
    max_tokens: int,
    concurrency: int,
    backlog_multiplier: int = 4,
) -> Dict:
    """
    运行统一的长 prompt + 长 decode 测试，通过 TTFT 分段计量 prefill/decode 能耗。
    
    原理：
    - 所有请求使用同一长 prompt（prefill）和足够长的 max_tokens（decode）
    - 根据 TTFT 分割时间轴：[0, mean_ttft) = prefill 阶段，[mean_ttft, total_duration) = decode 阶段
    - 根据各请求的 TTFT 和完成时刻，将整体能耗分配给两个阶段
    """
    with GPUMonitor() as monitor:
        batch = run_backlog_requests_with_ttft(
            engine,
            prompt=prompt,
            max_tokens=max_tokens,
            concurrency=concurrency,
            backlog_multiplier=backlog_multiplier,
        )

    total_duration_s = float(batch['total_duration_s'])
    mean_ttft_s = float(batch['mean_ttft_s'])
    total_input_tokens = int(batch['total_input_tokens'])
    total_output_tokens = int(batch['total_output_tokens'])

    # 用真实监控时间戳分段：time_offset < TTFT => Prefill；其余 => Decode
    monitor_df = _build_monitor_df(monitor)
    prefill_df = monitor_df[monitor_df["time_offset"] < mean_ttft_s]
    decode_df = monitor_df[monitor_df["time_offset"] >= mean_ttft_s]

    # 若边界导致某段为空，避免除零，回退到全量数据的一半分配
    total_energy_j = float(monitor_df["energy_j"].sum()) if not monitor_df.empty else 0.0
    if prefill_df.empty and decode_df.empty:
        prefill_energy_j = 0.0
        decode_energy_j = 0.0
        prefill_duration_s = mean_ttft_s
        decode_duration_s = max(total_duration_s - mean_ttft_s, 0.0)
        prefill_avg_power_w = 0.0
        decode_avg_power_w = 0.0
        prefill_peak_power_w = 0.0
        decode_peak_power_w = 0.0
    else:
        prefill_energy_j = float(prefill_df["energy_j"].sum())
        decode_energy_j = float(decode_df["energy_j"].sum())
        prefill_duration_s = float(prefill_df["time_interval"].sum()) if not prefill_df.empty else 0.0
        decode_duration_s = float(decode_df["time_interval"].sum()) if not decode_df.empty else 0.0
        prefill_avg_power_w = float(prefill_df["power_w"].mean()) if not prefill_df.empty else 0.0
        decode_avg_power_w = float(decode_df["power_w"].mean()) if not decode_df.empty else 0.0
        prefill_peak_power_w = float(prefill_df["power_w"].max()) if not prefill_df.empty else 0.0
        decode_peak_power_w = float(decode_df["power_w"].max()) if not decode_df.empty else 0.0

    # 防止因采样误差导致两段和不等于总能耗
    if total_energy_j > 0 and (prefill_energy_j + decode_energy_j) > 0:
        scale = total_energy_j / (prefill_energy_j + decode_energy_j)
        prefill_energy_j *= scale
        decode_energy_j *= scale

    prefill_tpj = total_input_tokens / prefill_energy_j if prefill_energy_j > 0 else 0.0
    decode_tpj = total_output_tokens / decode_energy_j if decode_energy_j > 0 else 0.0

    # 返回两行结果（prefill 和 decode），并记录分段边界
    return [
        {
            'phase': 'Prefill',
            'frequency_mhz': freq_mhz,
            'duration_s': round(prefill_duration_s, 4),
            'ttft_s': round(mean_ttft_s, 4),
            'tpot_s': round(0.0, 6),  # Prefill 没有 TPOT
            'avg_power_w': round(prefill_avg_power_w, 2),
            'peak_power_w': round(prefill_peak_power_w, 2),
            'total_energy_j': round(prefill_energy_j, 4),
            'throughput_tps': (float(total_input_tokens) / max(prefill_duration_s, 1e-9)),
            'j_per_token': (prefill_energy_j / total_input_tokens) if total_input_tokens > 0 else 0.0,
            'total_output_tokens': 0,  # Prefill 不生成 tokens
            'tpj': round(prefill_tpj, 4),
            'input_tokens': total_input_tokens,
            'concurrency': int(concurrency),
            'backlog_requests': int(batch['backlog_requests']),
            'finished_requests': int(batch['finished_requests']),
            'split_by': 'timestamp_ttft',
            'split_boundary_s': round(mean_ttft_s, 4),
            'samples_in_phase': int(len(prefill_df)),
        },
        {
            'phase': 'Decode',
            'frequency_mhz': freq_mhz,
            'duration_s': round(decode_duration_s, 4),
            'ttft_s': round(mean_ttft_s, 4),
            'tpot_s': round(float(batch['mean_tpot_s']), 6),
            'avg_power_w': round(decode_avg_power_w, 2),
            'peak_power_w': round(decode_peak_power_w, 2),
            'total_energy_j': round(decode_energy_j, 4),
            'throughput_tps': (float(total_output_tokens) / max(decode_duration_s, 1e-9)),
            'j_per_token': (decode_energy_j / total_output_tokens) if total_output_tokens > 0 else 0.0,
            'total_output_tokens': total_output_tokens,
            'tpj': round(decode_tpj, 4),
            'input_tokens': total_input_tokens,
            'concurrency': int(concurrency),
            'backlog_requests': int(batch['backlog_requests']),
            'finished_requests': int(batch['finished_requests']),
            'split_by': 'timestamp_ttft',
            'split_boundary_s': round(mean_ttft_s, 4),
            'samples_in_phase': int(len(decode_df)),
        }
    ]


def run_prefill_test(engine, prompt: str, freq_mhz: int, concurrency: int, backlog_multiplier: int = 4) -> Dict:
    """运行 Prefill 阶段测试（backlog 并发压测）"""
    with GPUMonitor() as monitor:
        batch = run_backlog_requests_with_ttft(
            engine,
            prompt=prompt,
            max_tokens=1,
            concurrency=concurrency,
            backlog_multiplier=backlog_multiplier,
        )

    total_input_tokens = int(batch['total_input_tokens'])
    total_output_tokens = int(batch['total_output_tokens'])
    metrics = monitor.get_metrics(total_input_tokens)
    energy = float(metrics['total_energy_j'])
    tpj = total_input_tokens / energy if energy > 0 else 0.0

    return {
        'phase': 'Prefill',
        'frequency_mhz': freq_mhz,
        'duration_s': float(batch['total_duration_s']),
        'ttft_s': float(batch['mean_ttft_s']),
        'tpot_s': float(batch['mean_tpot_s']),
        'avg_power_w': float(metrics['avg_power_w']),
        'peak_power_w': float(metrics['peak_power_w']),
        'total_energy_j': energy,
        'throughput_tps': (float(total_input_tokens) / max(float(batch['total_duration_s']), 1e-9)),
        'j_per_token': (energy / total_input_tokens) if total_input_tokens > 0 else 0.0,
        'total_output_tokens': total_output_tokens,
        'tpj': round(tpj, 4),
        'input_tokens': total_input_tokens,
        'concurrency': int(concurrency),
        'backlog_requests': int(batch['backlog_requests']),
        'finished_requests': int(batch['finished_requests']),
    }


def run_decode_test(
    engine,
    prompt: str,
    freq_mhz: int,
    max_tokens: int,
    concurrency: int,
    backlog_multiplier: int = 4,
) -> Dict:
    """运行 Decode 阶段测试（backlog 并发压测）"""
    with GPUMonitor() as monitor:
        batch = run_backlog_requests_with_ttft(
            engine,
            prompt=prompt,
            max_tokens=max_tokens,
            concurrency=concurrency,
            backlog_multiplier=backlog_multiplier,
        )

    total_output_tokens = int(batch['total_output_tokens'])
    total_input_tokens = int(batch['total_input_tokens'])
    metrics = monitor.get_metrics(total_output_tokens)
    energy = float(metrics['total_energy_j'])
    tpj = total_output_tokens / energy if energy > 0 else 0.0

    return {
        'phase': 'Decode',
        'frequency_mhz': freq_mhz,
        'duration_s': float(batch['total_duration_s']),
        'ttft_s': float(batch['mean_ttft_s']),
        'tpot_s': float(batch['mean_tpot_s']),
        'avg_power_w': float(metrics['avg_power_w']),
        'peak_power_w': float(metrics['peak_power_w']),
        'total_energy_j': energy,
        'throughput_tps': (float(total_output_tokens) / max(float(batch['total_duration_s']), 1e-9)),
        'j_per_token': (energy / total_output_tokens) if total_output_tokens > 0 else 0.0,
        'total_output_tokens': total_output_tokens,
        'tpj': round(tpj, 4),
        'input_tokens': total_input_tokens,
        'concurrency': int(concurrency),
        'backlog_requests': int(batch['backlog_requests']),
        'finished_requests': int(batch['finished_requests']),
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

    for phase in ['Prefill', 'Decode']:
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
        description="Sweet Spot 频率扫描 - 用 TTFT 分段方法找到能效最优的 GPU 频率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（新：统一模式，用 TTFT 分段）:
  # 长 prompt + 长 decode，通过 TTFT 分段 prefill/decode，测试所有频率
  python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 1 \
    --input-length 512 --decode-tokens 512

  # 单个频率，统一模式
  python sweet_spot.py --freq 1000 --repeat 1 \
    --input-length 512 --decode-tokens 512 --concurrency 32

示例（旧：分离模式，仍保留向后兼容）:
  # 仅 Prefill 测试（长prompt + max_tokens=1）
  python sweet_spot.py --freq 1000 --phases prefill --concurrency 32

  # 仅 Decode 测试（短prompt + 长 decode）
  python sweet_spot.py --freq 1000 --phases decode --concurrency 32
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
    parser.add_argument("--phases", type=str, nargs="+", default=None,
                        choices=["prefill", "decode"],
                        help="[旧模式] 指定测试阶段（prefill/decode），留空使用新«统一模式»")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="并发压测请求数（默认 32，显存不足可降到 16）")
    parser.add_argument("--input-length", type=int, default=512, 
                        help="输入 prompt 的 token 数（默认 512），推荐 512-1024")
    parser.add_argument("--decode-tokens", type=int, default=512, 
                        help="Decode 阶段生成的 token 数（默认 512），推荐 256-1024")
    parser.add_argument("--warmup", action="store_true", help="是否在正式测试前进行预热")

    # 输出参数
    parser.add_argument("--output", type=str, default="./log/sweet_spot_results.csv",
                        help="结果输出文件路径")
    parser.add_argument("--mode", type=str, default="unified", choices=["unified", "legacy"],
                        help="测试模式：unified = 统一模式（推荐），legacy = 分离模式（兼容旧脚本）")

    args = parser.parse_args()

    # 兼容性处理：如果指定了 --phases，则切换到 legacy 模式
    if args.phases is not None and args.phases != [] and args.phases != ["prefill", "decode"]:
        args.mode = "legacy"
        print(f"[*] 检测到 --phases 参数，切换到 legacy 模式: {args.phases}")
    elif args.phases is None:
        args.phases = []

    # 构建频率列表
    if args.freq is not None:
        frequencies = [args.freq]
    elif args.frequencies is not None:
        frequencies = args.frequencies
    else:
        frequencies = list(range(args.start, args.end + 1, args.step))

    print(f"[*] Sweet Spot 频率扫描 - 模式: {args.mode.upper()}")
    print(f"[*] 测试频率列表: {frequencies}")
    print(f"[*] 重复测试: {args.repeat} 次")
    if args.mode == "unified":
        print(f"[*] 输入长度: {args.input_length} tokens, Decode 长度: {args.decode_tokens} tokens")
        print(f"[*] [统一模式] 同一请求内，通过 TTFT 分段 prefill/decode")
    else:
        print(f"[*] 测试阶段: {args.phases}")
    print(f"[*] 并发配置: concurrency={args.concurrency}, backlog={args.concurrency * 4}")
    print(f"[*] 总测试次数: {len(frequencies) * args.repeat * (1 if args.mode == 'unified' else len(args.phases))}")

    # 加载 prompt
    base_prompt = load_long_prompt()
    if args.mode == "unified":
        prefill_prompt = generate_prompts(base_prompt, args.input_length, num_prompts=1)[0]
    else:
        prefill_prompts = generate_prompts(base_prompt, args.input_length, num_prompts=1)
    decode_prompt = "Hello."

    # 模型路径
    model_path = "./mistral_7b_model/LLM-Research/Mistral-7B-v0.3"
    save_system_info(model_path, script_name="sweet_spot")

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_summary_rows = []
    all_raw_rows = []

    try:
        # 初始化模型（只需一次，复用于所有频率）
        print("\n[*] 初始化模型...")
        max_input_tokens = int(args.input_length)
        required_batched_tokens = max_input_tokens * int(args.concurrency)
        engine = build_engine(
            model_path,
            max_num_seqs=max(256, int(args.concurrency) * 2),
            max_num_batched_tokens=max(32768, required_batched_tokens),
            gpu_memory_utilization=0.95,
        )
        if args.warmup:
            print("[*] 预热中...")
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

            freq_results = []

            for repeat_idx in range(args.repeat):
                print(f"\n[重复 {repeat_idx+1}/{args.repeat}]")

                if args.mode == "unified":
                    # 新模式：统一长 prompt + 长 decode，通过 TTFT 分段
                    results = run_unified_test(
                        engine,
                        prefill_prompt,
                        freq,
                        max_tokens=args.decode_tokens,
                        concurrency=args.concurrency,
                    )
                    freq_results.extend(results)  # results 是 [prefill_row, decode_row]
                    print("  统一测试完成（Prefill + Decode 已通过 TTFT 分段）")
                    time.sleep(1)
                else:
                    # 旧模式：分离 prefill/decode
                    if "prefill" in args.phases:
                        result = run_prefill_test(
                            engine,
                            prefill_prompts[0],
                            freq,
                            concurrency=args.concurrency,
                        )
                        freq_results.append(result)
                        print("  Prefill 测试完成")
                        time.sleep(1)

                    if "decode" in args.phases:
                        result = run_decode_test(
                            engine,
                            decode_prompt,
                            freq,
                            args.decode_tokens,
                            concurrency=args.concurrency,
                        )
                        freq_results.append(result)
                        print("  Decode 测试完成")
                        time.sleep(1)

            # 追加原始结果到 CSV（每个频率测完立即保存）
            if freq_results:
                df = pd.DataFrame(freq_results)
                mode = 'a' if os.path.exists(args.output) else 'w'
                header = not os.path.exists(args.output)
                df.to_csv(args.output, index=False, mode=mode, header=header)
                print(f"[*] 频率 {freq} MHz 结果已追加: {args.output}")
                all_raw_rows.extend(freq_results)

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
                        'concurrency': int(phase_df['concurrency'].mean()) if 'concurrency' in phase_df else int(args.concurrency),
                        'backlog_requests': int(phase_df['backlog_requests'].mean()) if 'backlog_requests' in phase_df else int(args.concurrency) * 4,
                        'finished_requests': int(phase_df['finished_requests'].mean()) if 'finished_requests' in phase_df else 0,
                        'repeat_count': len(phase_df)
                    }
                    all_summary_rows.append(avg_row)

        # 等全部频率记录结束后，再统一计算汇总和 sweet spot
        if all_raw_rows:
            raw_df = pd.DataFrame(all_raw_rows)
            grouped = raw_df.groupby(['phase', 'frequency_mhz'], as_index=False)
            summary_df = grouped.agg(
                duration_s=('duration_s', 'mean'),
                ttft_s=('ttft_s', 'mean'),
                tpot_s=('tpot_s', 'mean'),
                avg_power_w=('avg_power_w', 'mean'),
                peak_power_w=('peak_power_w', 'mean'),
                total_energy_j=('total_energy_j', 'mean'),
                throughput_tps=('throughput_tps', 'mean'),
                j_per_token=('j_per_token', 'mean'),
                total_output_tokens=('total_output_tokens', 'mean'),
                input_tokens=('input_tokens', 'mean'),
                concurrency=('concurrency', 'mean'),
                backlog_requests=('backlog_requests', 'mean'),
                finished_requests=('finished_requests', 'mean'),
                repeat_count=('phase', 'count'),
            )

            # 统一在所有记录结束后计算 TPJ
            summary_df['total_output_tokens'] = summary_df['total_output_tokens'].round().astype(int)
            summary_df['input_tokens'] = summary_df['input_tokens'].round().astype(int)
            summary_df['concurrency'] = summary_df['concurrency'].round().astype(int)
            summary_df['backlog_requests'] = summary_df['backlog_requests'].round().astype(int)
            summary_df['finished_requests'] = summary_df['finished_requests'].round().astype(int)

            summary_df['tpj'] = summary_df.apply(
                lambda row: (row['input_tokens'] / row['total_energy_j']) if row['phase'] == 'Prefill' and row['total_energy_j'] > 0
                else ((row['total_output_tokens'] / row['total_energy_j']) if row['total_energy_j'] > 0 else 0.0),
                axis=1,
            )

            for col in ['duration_s', 'ttft_s', 'tpot_s', 'avg_power_w', 'peak_power_w', 'total_energy_j', 'throughput_tps', 'j_per_token', 'tpj']:
                summary_df[col] = summary_df[col].astype(float)

            summary_df['duration_s'] = summary_df['duration_s'].round(4)
            summary_df['ttft_s'] = summary_df['ttft_s'].round(4)
            summary_df['tpot_s'] = summary_df['tpot_s'].round(6)
            summary_df['avg_power_w'] = summary_df['avg_power_w'].round(2)
            summary_df['peak_power_w'] = summary_df['peak_power_w'].round(2)
            summary_df['total_energy_j'] = summary_df['total_energy_j'].round(4)
            summary_df['throughput_tps'] = summary_df['throughput_tps'].round(2)
            summary_df['j_per_token'] = summary_df['j_per_token'].round(4)
            summary_df['tpj'] = summary_df['tpj'].round(4)

            summary_csv = args.output.replace('.csv', '_summary.csv')
            mode = 'a' if os.path.exists(summary_csv) else 'w'
            header = not os.path.exists(summary_csv)
            summary_df.to_csv(summary_csv, index=False, mode=mode, header=header)
            print(f"[*] 汇总结果已保存: {summary_csv}")

            all_summary_rows = summary_df.to_dict(orient='records')
            phases_tested = "All (TTFT Split)" if args.mode == "unified" else ", ".join(args.phases)
            print_sweet_spot_table(all_summary_rows, f"Sweet Spot 扫描结果 ({phases_tested})")

        print(f"\n{'='*80}")
        print(" Sweet Spot 扫描完成!")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print("\n\n[!] 用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] 测试出错: {e}")
        raise


if __name__ == "__main__":
    main()
