#!/bin/bash
# 扫描 SS/SL/LS/LL 四种 workload 的 Frequency vs TPJ（锁频操作在 Bash 中完成）
set -euo pipefail

mkdir -p ./log

RAW_CSV="./log/log_freq/workload_tpj_freq_raw.csv"
SUMMARY_CSV="./log/log_freq/workload_tpj_freq_summary.csv"
PLOT_DIR="./plot/workload_tpj_plots"
PLOT_SUMMARY_CSV="./plot/workload_tpj_sweet_spot_summary.csv"

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

python plot/plot_workload_tpj_vs_freq.py     --csv "./log/log_freq_large/workload_tpj_freq_summary.csv"     --out-dir "./plot/workload_tpj_plots_large"     --summary-out "./plot/workload_tpj_sweet_spot_summary_large.csv"