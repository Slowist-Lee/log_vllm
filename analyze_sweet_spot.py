#!/usr/bin/env python3
"""
Sweet Spot 分析脚本

目标：分析所有频率的测试结果，生成完整的 Sweet Spot 摘要

使用方法:
    python analyze_sweet_spot.py --input ./log/sweet_spot_results_summary.csv --output ./log/sweet_spot_final_summary.txt
"""

import argparse
import pandas as pd
from typing import List, Dict, Tuple


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
        description="Sweet Spot 分析 - 分析所有频率的测试结果",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", type=str, default="./log/sweet_spot_results_summary.csv",
                        help="输入汇总结果文件路径")
    parser.add_argument("--output", type=str, default="./log/sweet_spot_final_summary.txt",
                        help="输出摘要文件路径")
    
    args = parser.parse_args()
    
    # 读取汇总结果
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return
    
    # 转换为字典列表
    results = df.to_dict('records')
    
    if not results:
        print("[ERROR] 没有测试结果")
        return
    
    # 打印完整的 Sweet Spot 表格
    print_sweet_spot_table(results, "完整 Sweet Spot 扫描结果")
    
    # 保存最终摘要
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("Sweet Spot 频率扫描最终摘要\n")
        f.write("="*80 + "\n\n")
        
        # 获取所有测试频率
        frequencies = sorted(df['frequency_mhz'].unique())
        f.write(f"测试频率: {frequencies}\n")
        f.write(f"总测试频率数: {len(frequencies)}\n\n")
        
        for phase in ['Prefill', 'Decode', 'E2E']:
            best, sorted_results = find_sweet_spot(results, phase)
            if best:
                f.write(f"【{phase} 阶段 Sweet Spot】\n")
                f.write(f"  最优频率: {best['frequency_mhz']} MHz\n")
                f.write(f"  TPJ (Tokens/Joule): {best['tpj']:.4f}\n")
                f.write(f"  J/Token: {best['j_per_token']:.4f}\n")
                f.write(f"  平均功耗: {best['avg_power_w']:.2f} W\n")
                f.write(f"  TTFT: {best['ttft_s']:.4f} s\n")
                f.write(f"  TPOT: {best['tpot_s']:.6f} s\n")
                f.write(f"  吞吐: {best['throughput_tps']:.2f} tokens/s\n\n")
                
                # 写入该阶段所有频率的排序结果
                f.write(f"  该阶段频率排序（按 TPJ 降序）:\n")
                for i, r in enumerate(sorted_results):
                    f.write(f"    {i+1}. {r['frequency_mhz']} MHz - TPJ: {r['tpj']:.4f}, J/Token: {r['j_per_token']:.4f}\n")
                f.write("\n")
    
    print(f"\n[*] 最终摘要已保存: {args.output}")
    print(f"\n{'='*80}")
    print(" Sweet Spot 分析完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
