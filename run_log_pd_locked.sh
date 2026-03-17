#!/bin/bash

# 锁频运行 log_pd 脚本的 bash 脚本
# 用法: ./run_log_pd_locked.sh <frequency_mhz>
# 示例: ./run_log_pd_locked.sh 1000

set -e

# 检查参数
if [ $# -ne 1 ]; then
    echo "错误: 请提供 GPU 频率 (MHz)"
    echo "用法: ./run_log_pd_locked.sh <frequency_mhz>"
    exit 1
fi

FREQ_MHZ=$1

# 检查频率是否为数字
if ! [[ "$FREQ_MHZ" =~ ^[0-9]+$ ]]; then
    echo "错误: 频率必须是数字"
    exit 1
fi

echo "================================================"
echo "开始锁频运行 log_pd 测试"
echo "目标频率: ${FREQ_MHZ} MHz"
echo "================================================"

# 函数: 重置 GPU 频率
reset_gpu_freq() {
    echo "\n[*] 重置 GPU 频率到默认值..."
    nvidia-smi -rgc > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "[✓] GPU 频率已重置为默认值"
    else
        echo "[⚠️] 重置 GPU 频率失败，请手动执行: nvidia-smi -rgc"
    fi
}

# 捕获 SIGINT 和 SIGTERM 信号，确保脚本被中断时也能重置频率
trap reset_gpu_freq SIGINT SIGTERM

# 设置 GPU 频率
echo "[*] 设置 GPU 频率为 ${FREQ_MHZ} MHz..."
nvidia-smi -lgc ${FREQ_MHZ},${FREQ_MHZ}
if [ $? -ne 0 ]; then
    echo "[✗] 设置频率失败，请检查权限和频率范围"
    exit 1
fi

# 等待频率稳定
echo "[*] 等待频率稳定..."
sleep 2

# 运行 log_pd 脚本
echo "\n[*] 运行 log_pd.py 脚本..."
python3 log_pd.py
# python3 e2e_profile_ttft.py

# 重置频率
reset_gpu_freq

echo "\n================================================"
echo "锁频测试完成!"
echo "================================================"
