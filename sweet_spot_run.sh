#!/bin/bash
# 遇到错误终止执行
set -e

# 确保 log 文件夹存在
mkdir -p ./log

PREFILL_REQUESTS_PER_MEASURE=10

echo "=========================================================="
echo " Sweet Spot Frequency Scanning"
echo " Frequencies: 750-1300 MHz, step 50 MHz, repeat 1"
echo " Prefill requests/measure: ${PREFILL_REQUESTS_PER_MEASURE}"
echo "=========================================================="

echo ""
echo "========== [2/4] Scanning Decode Sweet Spot =========="
python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 3 --phases decode \
    --output ./log/sweet_spot_decode.csv

# -------------------------------------------------------
# Phase 1: Prefill
# 模型只加载一次，扫描所有频率
# -------------------------------------------------------
echo ""
echo "========== [1/4] Scanning Prefill Sweet Spot =========="
python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 3 --phases prefill \
    --prefill-requests "$PREFILL_REQUESTS_PER_MEASURE" \
    --output ./log/sweet_spot_prefill.csv


# -------------------------------------------------------
# Phase 4: E2E (Batch Processing)
# BS=16, Input Length=1024, Max Tokens=256
# -------------------------------------------------------
# echo ""
# echo "========== [4/4] Scanning E2E Sweet Spot (Batch Size=16) =========="
# python sweet_spot.py --start 750 --end 1300 --step 50 --repeat 3 --phases e2e \
#     --batch-size 16 --input-length 1024 --decode-tokens 256 \
#     --output ./log/sweet_spot_e2e_batch16.csv

echo ""
echo "=========================================================="
echo " Sweet Spot Scanning completed!"
echo " Results:"
echo "   Prefill:        ./log/sweet_spot_prefill_summary.csv"
echo "   Decode:         ./log/sweet_spot_decode_summary.csv"
echo "   E2E (Single):   ./log/sweet_spot_e2e_summary.csv"
echo "   E2E (Batch 16): ./log/sweet_spot_e2e_batch16_summary.csv"
echo "=========================================================="

