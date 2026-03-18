#!/bin/bash
# Task 5: Batch Size × GPU Frequency 2D Sweep
# Generates data for a heatmap of (freq, bs) vs performance metrics.
set -e

LOG_DIR="./log/task5_heatmap"
SUMMARY_HEADER="freq_mhz,batch_size,duration_s,ttft_s,tpot_s,avg_power_w,peak_power_w,total_energy_j,throughput_tps,j_per_token,tpj,total_output_tokens"

# Create directories and write CSV header fresh
mkdir -p "${LOG_DIR}/power_logs"
echo "$SUMMARY_HEADER" > "${LOG_DIR}/summary.csv"

echo "=========================================================="
echo " Task 5: GPU Frequency × Batch Size Heatmap Experiment"
echo " Log dir: ${LOG_DIR}"
echo "=========================================================="

# Frequencies to sweep (MHz) — from low to high
FREQUENCIES=(210 420 630 840 1050 1260 1410)

# Batch sizes to sweep
BATCH_SIZES=(1 4 8 12 16 20 24 28 32)

for FREQ in "${FREQUENCIES[@]}"; do
    echo ""
    echo "=========================================================="
    echo " Locking GPU frequency to ${FREQ} MHz"
    echo "=========================================================="
    sudo nvidia-smi -lgc ${FREQ},${FREQ}
    sleep 2  # Allow GPU to stabilize at new frequency

    for BS in "${BATCH_SIZES[@]}"; do
        echo "----------------------------------------------------------"
        echo "[*] freq=${FREQ} MHz | batch_size=${BS}"
        python task5_heatmap_core.py \
            --freq  ${FREQ} \
            --bs    ${BS}   \
            --log-dir "${LOG_DIR}"
        # Brief pause between runs to let GPU cool / power settle
        sleep 1
    done
done

echo ""
echo "[*] Resetting GPU frequency limits to hardware defaults..."
sudo nvidia-smi -rgc
sleep 2

echo ""
echo "=========================================================="
echo " All experiments completed!"
echo " Summary CSV : ${LOG_DIR}/summary.csv"
echo " Power logs  : ${LOG_DIR}/power_logs/freq<F>_bs<B>_power_log.csv"
echo " Plot with   : python task5_heatmap_plot.py --log-dir ${LOG_DIR}"
echo "=========================================================="
