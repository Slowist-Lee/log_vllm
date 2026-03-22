#!/bin/bash
# 扫描 SS/SL/LS/LL 四种 workload 的 Frequency vs TPJ（锁频操作在 Bash 中完成）
set -euo pipefail

mkdir -p ./log

RAW_CSV="./log/workload_tpj_freq_raw.csv"
SUMMARY_CSV="./log/workload_tpj_freq_summary.csv"
PLOT_DIR="./log/workload_tpj_plots"
PLOT_SUMMARY_CSV="./log/workload_tpj_sweet_spot_summary.csv"

# 频率序列：750, 800, ..., 1300
FREQUENCIES=($(seq 750 50 1300))
REPEAT=3
LATENCY_SLO_S=8.0
WORKLOADS="SS,SL,LS,LL"

# 先清理旧结果，避免混入历史数据
rm -f "$RAW_CSV" "$SUMMARY_CSV" "$PLOT_SUMMARY_CSV"

echo "=========================================================="
echo " Start workload TPJ scan"
echo " Workloads: ${WORKLOADS}"
echo " Frequencies: 750..1300 step 50"
echo " Repeat per frequency: ${REPEAT}"
echo "=========================================================="

for FREQ in "${FREQUENCIES[@]}"; do
    echo "----------------------------------------------------------"
    echo "[*] Locking GPU frequency to ${FREQ} MHz..."
    sudo nvidia-smi -lgc ${FREQ},${FREQ}
    sleep 2

    echo "[*] Running workloads at ${FREQ} MHz"
    python workload_tpj_freq_scan.py \
        --frequency-mhz ${FREQ} \
        --workloads "${WORKLOADS}" \
        --repeat ${REPEAT} \
        --latency-slo-s ${LATENCY_SLO_S} \
        --append \
        --raw-out "$RAW_CSV" \
        --summary-out "$SUMMARY_CSV"
done

echo "[*] Resetting GPU frequency limits..."
sudo nvidia-smi -rgc
sleep 2

echo "[*] Plotting per-workload figures..."
python plot/plot_workload_tpj_vs_freq.py \
    --csv "$SUMMARY_CSV" \
    --out-dir "$PLOT_DIR" \
    --summary-out "$PLOT_SUMMARY_CSV"

echo "=========================================================="
echo " Done"
echo " Raw CSV:        ${RAW_CSV}"
echo " Summary CSV:    ${SUMMARY_CSV}"
echo " Plot Dir:       ${PLOT_DIR}"
echo " Sweet Spot CSV: ${PLOT_SUMMARY_CSV}"
echo "=========================================================="
