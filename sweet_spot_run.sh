#!/bin/bash
# 遇到错误终止执行
set -e

# 确保 log 文件夹存在
mkdir -p ./log

CONCURRENCY=32
START_FREQ=750
END_FREQ=1300
STEP_FREQ=50
REPEAT=3

INPUT_LENGTH=512
DECODE_TOKENS=512

# 统一模式：同一请求内，通过 TTFT 分段 prefill/decode
UNIFIED_OUTPUT=./log/sweet_spot_unified.csv

reset_gpu_freq() {
    echo ""
    echo "[*] Restoring GPU graphics clock to default..."
    if nvidia-smi -rgc; then
        echo "[*] GPU graphics clock restored."
    else
        echo "[WARN] Failed to restore GPU graphics clock."
    fi
}

trap reset_gpu_freq EXIT

rm -f "$UNIFIED_OUTPUT"
rm -f "${UNIFIED_OUTPUT%.csv}_summary.csv"

echo "=========================================================="
echo " Sweet Spot Frequency Scanning (TTFT Split Mode)"
echo " Frequencies: ${START_FREQ}-${END_FREQ} MHz, step ${STEP_FREQ} MHz, repeat ${REPEAT}"
echo " Concurrency: ${CONCURRENCY}"
echo " Input Length: ${INPUT_LENGTH} tokens, Decode Length: ${DECODE_TOKENS} tokens"
echo "=========================================================="

for ((freq=START_FREQ; freq<=END_FREQ; freq+=STEP_FREQ)); do
    echo ""
    echo "=========================================================="
    echo " Locking GPU graphics clock at ${freq} MHz"
    echo "=========================================================="
    nvidia-smi -lgc "${freq},${freq}"
    sleep 2

    echo ""
    echo "========== Scanning UNIFIED (TTFT Split) @ ${freq} MHz =========="
    python sweet_spot.py --freq "$freq" --repeat "$REPEAT" \
        --concurrency "$CONCURRENCY" \
        --input-length "$INPUT_LENGTH" \
        --decode-tokens "$DECODE_TOKENS" \
        --mode unified \
        --output "$UNIFIED_OUTPUT"
done

echo ""
echo "=========================================================="
echo " Sweet Spot Scanning completed!"
echo " Results:"
echo "   Unified (TTFT Split):   ./log/sweet_spot_unified_summary.csv"
echo "=========================================================="

