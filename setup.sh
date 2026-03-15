#!/bin/bash
set -e # 遇到错误停止执行

sudo sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
sudo sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
sudo apt update
# 安装必要的系统库
sudo apt install -y git zsh curl wget vim gcc g++ make


# 阻止 oh-my-zsh 安装后自动进入 zsh
export RUNZSH=no 
export CHSH=yes
# 使用国内 gitee 镜像安装 oh-my-zsh 提速
sh -c "$(curl -fsSL https://gitee.com/pocmac/ohmyzsh/raw/master/tools/install.sh)" || true

# 下载插件 (使用国内 Gitee 镜像或 GitHub 官方)
ZSH_CUSTOM=~/.oh-my-zsh/custom
git clone https://gitee.com/phper98/zsh-syntax-highlighting.git ${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting
git clone https://gitee.com/phper98/zsh-autosuggestions.git ${ZSH_CUSTOM}/plugins/zsh-autosuggestions

# 修改 ~/.zshrc 启用插件
sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# 寻找 Conda 路径并初始化 (截图显示你已经装了，一般在 ~/anaconda3 或 ~/miniconda3)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ ! -d "$CONDA_BASE" ]; then
    CONDA_BASE="$HOME/anaconda3"
fi
# 初始化 shell
$CONDA_BASE/bin/conda init bash
$CONDA_BASE/bin/conda init zsh

# 配置 Conda 清华镜像源
cat <<EOF > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF

# 激活 conda 供当前脚本使用
source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n vllm python=3.10 -y
conda activate vllm

# 配置 pip 清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 为了绕过系统 CUDA 11.4 的限制，我们在 conda 里装一套 cudatoolkit 11.8
conda install -y -c nvidia cuda-toolkit=11.8
# 安装 vllm
pip install vllm

cd ~/project
echo "=== System & Environment Info ===" > basic_log.txt
echo "GPU Model & Driver Version:" >> basic_log.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> basic_log.txt 2>/dev/null || echo "NVIDIA SMI failed" >> basic_log.txt
echo "CUDA Version (System):" >> basic_log.txt
nvcc --version | grep "release" | awk '{print $5,$6}' >> basic_log.txt 2>/dev/null || echo "System NVCC not found" >> basic_log.txt
echo "Model Name: mistralai/Mistral-7B-Instruct-v0.2" >> basic_log.txt
echo "Inference Engine: vLLM $(pip show vllm | grep Version | awk '{print $2}')" >> basic_log.txt

git clone https://github.com/Slowist-Lee/log_vllm.git
cd log_vllm
pip install -r requirements.txt

mkdir -p mistral_7b_model
mkdir -p log

# 使用 Huggingface 官方工具配合国内 HF 镜像源下载模型，防止断连
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
echo "开始下载 Mistral 模型 (约 14GB)"
# 这里假设下载 Instruct-v0.2 版本，因为它是目前主流测试的
huggingface-cli download --resume-download mistralai/Mistral-7B-Instruct-v0.2 --local-dir mistral_7b_model

# 运行脚本
python log_v4.py || echo "Python 脚本运行遇到问题，请检查代码或配置"
