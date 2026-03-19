#!/bin/bash
# 测 210,420,...,1470 MHz 下 bs=16 的 avg power，并保存到 CSV
set -euo pipefail

mkdir -p ./log

# inference_core.py --task 4b 的原始输出文件（用于提取 avg_power_w）
RAW_TASK4B_CSV="./log/task4b_results.csv"
TASK4B_HEADER="batch_size,duration_s,ttft_s,tpot_s,avg_power_w,peak_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens"

# 最终结果文件：只保留频率、batch_size 和 avg_power_w
OUTPUT_CSV="./log/task4_bs16_avg_power_vs_freq.csv"
echo "frequency_mhz,batch_size,avg_power_w" > "$OUTPUT_CSV"

# 重新初始化原始结果，确保 tail -n 1 始终是本次实验数据
echo "$TASK4B_HEADER" > "$RAW_TASK4B_CSV"

# 频率序列：210, 420, ..., 1470
FREQUENCIES=($(seq 210 210 1470))
BATCH_SIZE=16

echo "=========================================================="
echo " Start: bs=${BATCH_SIZE}, freq=210..1470 step 210"
echo "=========================================================="

for FREQ in "${FREQUENCIES[@]}"; do
    echo "----------------------------------------------------------"
    echo "[*] Locking GPU frequency to ${FREQ} MHz..."
    sudo nvidia-smi -lgc ${FREQ},${FREQ}
    sleep 2

    echo "[*] Running Inference: task=4b, bs=${BATCH_SIZE}, freq=${FREQ}"
    python inference_core.py --task 4b --bs ${BATCH_SIZE}

    # 从 task4b 最新一行提取 avg_power_w（第 5 列）
    LAST_LINE=$(tail -n 1 "$RAW_TASK4B_CSV")
    AVG_POWER=$(echo "$LAST_LINE" | awk -F',' '{print $5}')

    if [[ -z "$AVG_POWER" ]]; then
        echo "[!] Failed to parse avg_power_w at freq=${FREQ}. Last line: $LAST_LINE"
        exit 1
    fi

    echo "${FREQ},${BATCH_SIZE},${AVG_POWER}" >> "$OUTPUT_CSV"
    echo "[+] Saved: freq=${FREQ}, bs=${BATCH_SIZE}, avg_power_w=${AVG_POWER}"
done

echo "[*] Resetting GPU frequency limits..."
sudo nvidia-smi -rgc
sleep 2

echo "=========================================================="
echo " Done. Result saved to ${OUTPUT_CSV}"
echo "=========================================================="
