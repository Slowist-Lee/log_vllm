#!/bin/bash
# 遇到错误终止执行
set -e

# 确保 log 文件夹存在
mkdir -p ./log

# 初始化 CSV 文件并写入表头
HEADER="phase,frequency_mhz,duration_s,ttft_s,tpot_s,avg_power_w,peak_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens,tpj,input_tokens,repeat_count"
# 确保重新创建文件，避免表头不一致
echo "$HEADER" > ./log/sweet_spot_results.csv

echo "=========================================================="
echo " Starting Sweet Spot Frequency Scanning"
echo "=========================================================="

# 定义测试频率范围
FREQUENCIES=(700 900 1100 1300 1410 1500)

for FREQ in "${FREQUENCIES[@]}"; do
    echo "----------------------------------------------------------"
    echo "[*] Locking GPU frequency to ${FREQ} MHz..."
    sudo nvidia-smi -lgc ${FREQ},${FREQ}
    sleep 2 # 给 GPU 几秒钟稳定频率
    
    echo "[*] Running Sweet Spot Test for Frequency: ${FREQ} MHz"
    # 调用 Python 脚本，传入 freq 参数
    python sweet_spot.py --freq ${FREQ} --repeat 3 --phases prefill decode e2e
done

# 测试完后，务必恢复 GPU 默认频率限制
echo "[*] Resetting GPU frequency limits..."
sudo nvidia-smi -rgc
sleep 2

echo "=========================================================="
echo " Sweet Spot Scanning completed successfully!"
echo " Results saved in: ./log/sweet_spot_results.csv"
echo "=========================================================="
