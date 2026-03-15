#!/bin/bash
set -e # 遇到错误停止执行

# ==========================================
# 定义断点记录文件和辅助函数
# ==========================================
STATE_FILE="$HOME/.setup_state_vllm"

# 检查步骤是否完成的函数
is_completed() { grep -Fxq "$1" "$STATE_FILE" 2>/dev/null; }
# 标记步骤为已完成的函数
mark_completed() { echo "$1" >> "$STATE_FILE"; }

echo "=== 初始化 VLLM 环境配置脚本 ==="

# ------------------------------------------
# 1. 从创建 Conda vllm 环境开始
# ------------------------------------------
CONDA_BASE=$(conda info --base 2>/dev/null || true)

if [ -z "$CONDA_BASE" ] || [ ! -d "$CONDA_BASE" ]; then
    for path in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
        if [ -d "$path/bin" ] && [ -f "$path/bin/conda" ]; then
            CONDA_BASE="$path"
            break
        fi
    done
fi

if [ -z "$CONDA_BASE" ] || [ ! -d "$CONDA_BASE" ]; then
    echo "[错误] 未能找到 Conda 安装路径，请确认你是否已安装 miniconda 或 anaconda！"
    exit 1
fi

echo "-> 成功定位到 Conda 安装路径: $CONDA_BASE"

# **重要**：加载 conda 环境供当前 bash 脚本调用
source $CONDA_BASE/etc/profile.d/conda.sh

# ------------------------------------------
# 2. 创建 Conda 环境并安装 vLLM (匹配新版 CUDA)
# ------------------------------------------
if ! is_completed "conda_env_vllm"; then
    echo "[1/5] 正在创建 Conda vllm 环境并安装依赖..."
    conda create -n vllm python=3.10 -y
    conda activate vllm
    
    # 配置 pip 清华镜像源
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

    # 直接通过 pip 安装 vllm，自动适配系统较高版本的 CUDA (12.x)
    pip install vllm

    mark_completed "conda_env_vllm"
else
    echo "[跳过] Conda 环境 vllm 已创建并完成安装"
fi

# 确保激活环境供后续 Python 脚本使用
conda activate vllm

# ------------------------------------------
# 3. 克隆项目和环境记录
# ------------------------------------------
mkdir -p ~/project
cd ~/project

if ! is_completed "project_clone"; then
    echo "[2/5] 正在克隆项目并记录环境信息..."
    echo "=== System & Environment Info ===" > basic_log.txt
    echo "GPU Model & Driver Version:" >> basic_log.txt
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> basic_log.txt 2>/dev/null || echo "NVIDIA SMI failed" >> basic_log.txt
    echo "CUDA Version (System):" >> basic_log.txt
    nvcc --version | grep "release" | awk '{print $5,$6}' >> basic_log.txt 2>/dev/null || echo "System NVCC not found" >> basic_log.txt
    echo "Model Name: mistralai/Mistral-7B-Instruct-v0.2" >> basic_log.txt
    echo "Inference Engine: vLLM $(pip show vllm | grep Version | awk '{print $2}')" >> basic_log.txt

    if [ ! -d "log_vllm" ]; then
        git clone https://github.com/Slowist-Lee/log_vllm.git
    fi
    mark_completed "project_clone"
else
    echo "[跳过] 项目代码已克隆"
fi

cd log_vllm
mkdir -p mistral_7b_model
mkdir -p log

if ! is_completed "pip_requirements"; then
    echo "[3/5] 正在安装项目 requirements.txt ..."
    pip install -r requirements.txt
    mark_completed "pip_requirements"
else
    echo "[跳过] 项目 requirements 依赖已安装"
fi


# ------------------------------------------
# 5. 运行 Python 脚本
# ------------------------------------------
if ! is_completed "run_prompt_gen"; then
    echo "[5/5] 正在运行 prompt_gen.py..."
    python prompt_gen.py || echo "[警告] prompt_gen.py 运行遇到问题"
    mark_completed "run_prompt_gen"
fi

if ! is_completed "run_log_v4"; then
    echo "正在运行 log_v4.py..."
    python log_v4.py || echo "[警告] log_v4.py 运行遇到问题"
    mark_completed "run_log_v4"
fi

echo "=== 所有任务执行完毕！环境准备就绪！ ==="
chmod +x task4_run.sh && ./task4_run.sh