#!/bin/bash
# 遇到错误终止执行
set -e

# 确保 log 文件夹存在
mkdir -p ./log

# 初始化 CSV 文件并写入表头（与 inference_core.py 写入顺序严格一致）
TASK4A_HEADER="phase,frequency_mhz,duration_s,ttft_s,tpot_s,avg_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens"
TASK4B_HEADER="batch_size,duration_s,ttft_s,tpot_s,avg_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens"
echo "$TASK4A_HEADER" > ./log/task4a_results.csv
echo "$TASK4B_HEADER" > ./log/task4b_results.csv

echo "=========================================================="
echo " Starting Task 4a: GPU Frequency Scaling Experiments"
echo "=========================================================="
# 这里定义你要测试的四个频率 (针对云端服务器显卡如 A100，通常最高在 1410 左右)
# 如果是消费级显卡 (3090/4090)，可以设为 1200 1500 1800 2100
FREQUENCIES=(900 1100 1300 1500)

for FREQ in "${FREQUENCIES[@]}"; do
    echo "----------------------------------------------------------"
    echo "[*] Locking GPU frequency to ${FREQ} MHz..."
    sudo nvidia-smi -lgc ${FREQ},${FREQ}
    sleep 2 # 给 GPU 几秒钟稳定频率
    
    echo "[*] Running Inference for Frequency: ${FREQ} MHz"
    # 调用 Python 脚本，传入 task 和 freq 参数
    python inference_core.py --task 4a --freq ${FREQ}
done

# 测试完 4a 后，务必恢复 GPU 默认频率限制
echo "[*] Resetting GPU frequency limits..."
sudo nvidia-smi -rgc
sleep 2


echo "=========================================================="
echo " Starting Task 4b: Batch Size Scaling Experiments"
echo "=========================================================="
# 定义要测试的 Batch Size
BATCH_SIZES=(1 2 4 8 16)

for BS in "${BATCH_SIZES[@]}"; do
    echo "----------------------------------------------------------"
    echo "[*] Running Inference for Batch Size: ${BS}"
    # 调用 Python 脚本，传入 task 和 bs 参数
    python inference_core.py --task 4b --bs ${BS}
done

echo "=========================================================="
echo " All tasks completed successfully!"
echo " Results saved in:"
echo "   - ./log/task4a_results.csv"
echo "   - ./log/task4b_results.csv"
echo "=========================================================="