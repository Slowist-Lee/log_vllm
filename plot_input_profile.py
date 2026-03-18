import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_input_profile():
    summary_file = "./log/log_03_17/input_profile_summary.csv"
    
    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found")
        return
    
    df = pd.read_csv(summary_file)
    
    input_lengths = df['input_length'].values
    peak_power = df['peak_power_w'].values
    ttft = df['ttft_s'].values
    total_duration = df['total_duration_s'].values
    prefill_energy = df['prefill_energy_j'].values
    decode_energy = df['decode_energy_j'].values
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Input Size Impact on Performance', fontsize=16, fontweight='bold')
    
    x_pos = range(len(input_lengths))
    x_labels = [str(length) for length in input_lengths]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    axes[0, 0].bar(x_pos, peak_power, color=colors[0])
    axes[0, 0].set_xlabel('Input Length', fontweight='bold')
    axes[0, 0].set_ylabel('Peak Power (W)', fontweight='bold')
    axes[0, 0].set_title('Peak Power vs Input Size', fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(x_labels)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(peak_power):
        axes[0, 0].text(i, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[0, 1].bar(x_pos, ttft, color=colors[1])
    axes[0, 1].set_xlabel('Input Length', fontweight='bold')
    axes[0, 1].set_ylabel('TTFT (s)', fontweight='bold')
    axes[0, 1].set_title('Time to First Token vs Input Size', fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(x_labels)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(ttft):
        axes[0, 1].text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    axes[0, 2].bar(x_pos, total_duration, color=colors[2])
    axes[0, 2].set_xlabel('Input Length', fontweight='bold')
    axes[0, 2].set_ylabel('Total Duration (s)', fontweight='bold')
    axes[0, 2].set_title('Total Duration vs Input Size', fontweight='bold')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(x_labels)
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(total_duration):
        axes[0, 2].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    axes[1, 0].bar(x_pos, prefill_energy, color=colors[3])
    axes[1, 0].set_xlabel('Input Length', fontweight='bold')
    axes[1, 0].set_ylabel('Prefill Energy (J)', fontweight='bold')
    axes[1, 0].set_title('Prefill Energy vs Input Size', fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(x_labels)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(prefill_energy):
        axes[1, 0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    axes[1, 1].bar(x_pos, decode_energy, color=colors[4])
    axes[1, 1].set_xlabel('Input Length', fontweight='bold')
    axes[1, 1].set_ylabel('Decode Energy (J)', fontweight='bold')
    axes[1, 1].set_title('Decode Energy vs Input Size', fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x_labels)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(decode_energy):
        axes[1, 1].text(i, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    stacked_prefill = [prefill_energy[i] / (prefill_energy[i] + decode_energy[i]) * 100 for i in range(len(prefill_energy))]
    stacked_decode = [decode_energy[i] / (prefill_energy[i] + decode_energy[i]) * 100 for i in range(len(decode_energy))]
    
    axes[1, 2].bar(x_pos, stacked_prefill, label='Prefill %', color=colors[3], alpha=0.8)
    axes[1, 2].bar(x_pos, stacked_decode, bottom=stacked_prefill, label='Decode %', color=colors[4], alpha=0.8)
    axes[1, 2].set_xlabel('Input Length', fontweight='bold')
    axes[1, 2].set_ylabel('Energy Percentage (%)', fontweight='bold')
    axes[1, 2].set_title('Energy Distribution: Prefill vs Decode', fontweight='bold')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(x_labels)
    axes[1, 2].legend(loc='upper right')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    for i, (pref, dec) in enumerate(zip(stacked_prefill, stacked_decode)):
        axes[1, 2].text(i, pref/2, f'{pref:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        axes[1, 2].text(i, pref + dec/2, f'{dec:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = "./log/log_03_17/input_profile_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()
    
    print("\n========== Summary Statistics ==========")
    print(f"Input Lengths: {input_lengths}")
    print(f"Peak Power (W): {peak_power}")
    print(f"TTFT (s): {ttft}")
    print(f"Total Duration (s): {total_duration}")
    print(f"Prefill Energy (J): {prefill_energy}")
    print(f"Decode Energy (J): {decode_energy}")
    print(f"Prefill Energy Ratio: {stacked_prefill}")
    print(f"Decode Energy Ratio: {stacked_decode}")
    print("========================================")

if __name__ == "__main__":
    plot_input_profile()
