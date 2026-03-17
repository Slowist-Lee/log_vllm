import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 加载数据
# ==========================================
# 请确保这里的文件路径与你实际保存的一致
try:
    df_prefill = pd.read_csv("./log/log_03_16_2/prefill_3_power_log.csv")
    df_decode = pd.read_csv("./log/log_03_16_2/decode_3_power_log.csv")
except FileNotFoundError:
    print("找不到 CSV 文件，请检查路径是否正确。")
    exit()


def validate_columns(df: pd.DataFrame, name: str, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"{name} 缺少必要列: {missing}")
        print(f"当前可用列: {list(df.columns)}")
        exit()

def normalize_util_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """兼容不同版本日志列名：util_gpu_pct / util_pct / util_mem_pct。"""
    if "util_gpu_pct" not in df.columns:
        if "util_pct" in df.columns:
            df["util_gpu_pct"] = df["util_pct"]
        else:
            print(f"{name} 缺少必要列: ['util_gpu_pct'] (且未找到可回退列 'util_pct')")
            print(f"当前可用列: {list(df.columns)}")
            exit()

    # 某些日志没有显存利用率，补齐为空列，避免后续处理报错
    if "util_mem_pct" not in df.columns:
        df["util_mem_pct"] = pd.NA

    return df


df_prefill = normalize_util_columns(df_prefill, "./log/log_03_16/prefill_3_power_log.csv")
df_decode = normalize_util_columns(df_decode, "./log/log_03_16/decode_3_power_log.csv")

required_columns = [
    "time_offset",
    "power_w",
    "gpu_clock_mhz",
    "util_gpu_pct",
]

validate_columns(df_prefill, "./log/log_03_16/prefill_3_power_log.csv", required_columns)
validate_columns(df_decode, "./log/log_03_16/decode_3_power_log.csv", required_columns)

# 仅保留绘图需要的数值列，避免字符串或异常值导致绘图失败
for col in required_columns:
    df_prefill[col] = pd.to_numeric(df_prefill[col], errors="coerce")
    df_decode[col] = pd.to_numeric(df_decode[col], errors="coerce")

df_prefill = df_prefill.dropna(subset=required_columns)
df_decode = df_decode.dropna(subset=required_columns)

# 统一全局图表风格 (模拟学术论文质感)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# 图表 A：功率对比图 (Task 3 核心要求)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

# 绘制两条线，使用时间偏移 (time_offset) 作为 X 轴
ax1.plot(df_prefill['time_offset'], df_prefill['power_w'], 
         label='Prefill Phase (Long Input, 1 Token)', 
         color='#D62728', linewidth=2.5, zorder=3) # 红色代表高计算量的 Prefill

ax1.plot(df_decode['time_offset'], df_decode['power_w'], 
         label='Decode Phase (Short Input, 512 Tokens)', 
         color='#1F77B4', linewidth=2, alpha=0.8, zorder=2) # 蓝色代表长缓的 Decode

# 美化图表 A
ax1.set_title("Task 3: Power Consumption Profile (Prefill vs. Decode)", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel("Time (seconds)", fontsize=12)
ax1.set_ylabel("GPU Power (Watts)", fontsize=12)
ax1.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
ax1.set_xlim(left=0) # X轴从0开始
ax1.set_ylim(bottom=0) # Y轴从0开始

fig1.tight_layout()
fig1.savefig("./task3_power_comparison.png")
print("成功生成功率对比图：./task3_power_comparison.png")

# ==========================================
# 图表 A-zoom：0-2 秒局部放大图
# ==========================================
df_prefill_zoom = df_prefill[df_prefill["time_offset"] <= 2].copy()
df_decode_zoom = df_decode[df_decode["time_offset"] <= 2].copy()

if not df_prefill_zoom.empty and not df_decode_zoom.empty:
    fig1_zoom, ax1_zoom = plt.subplots(figsize=(10, 5), dpi=150)

    ax1_zoom.plot(
        df_prefill_zoom["time_offset"],
        df_prefill_zoom["power_w"],
        label="Prefill Phase (Long Input, 1 Token)",
        color="#D62728",
        linewidth=2.5,
        zorder=3,
    )
    ax1_zoom.plot(
        df_decode_zoom["time_offset"],
        df_decode_zoom["power_w"],
        label="Decode Phase (Short Input, 512 Tokens)",
        color="#1F77B4",
        linewidth=2,
        alpha=0.8,
        zorder=2,
    )

    zoom_power = pd.concat([df_prefill_zoom["power_w"], df_decode_zoom["power_w"]])
    y_min = max(0, zoom_power.min() - 10)
    y_max = zoom_power.max() + 10

    ax1_zoom.set_title("Task 3: Power Consumption Profile (0-2s Zoom)", fontsize=14, fontweight="bold", pad=15)
    ax1_zoom.set_xlabel("Time (seconds)", fontsize=12)
    ax1_zoom.set_ylabel("GPU Power (Watts)", fontsize=12)
    ax1_zoom.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)
    ax1_zoom.set_xlim(0, 2)
    ax1_zoom.set_ylim(y_min, y_max)

    fig1_zoom.tight_layout()
    fig1_zoom.savefig("./task3_power_comparison_zoom_2s.png")
    print("成功生成 2 秒放大图：./task3_power_comparison_zoom_2s.png")
else:
    print("2 秒内数据不足，未生成放大图。")

# ==========================================
# 图表 B：硬件底层特征图 (展示利用率和频率差异)
# 用于 PPT 中深度解释 "Why they are different"
# ==========================================
fig2, (ax_util, ax_clock) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)

# --- 子图 1: GPU 利用率 (util_gpu_pct) ---
ax_util.plot(df_prefill['time_offset'], df_prefill['util_gpu_pct'], label='Prefill Util %', color='#D62728', linewidth=2)
ax_util.plot(df_decode['time_offset'], df_decode['util_gpu_pct'], label='Decode Util %', color='#1F77B4', linewidth=2, alpha=0.8)
ax_util.set_ylabel("GPU Utilization (%)", fontsize=11)
ax_util.set_title("Hardware Micro-behaviors: Utilization & Clock Speeds", fontsize=13, fontweight='bold')
ax_util.legend(loc='lower right')
ax_util.set_ylim(-5, 105)

# --- 子图 2: 核心频率 (gpu_clock_mhz) ---
ax_clock.plot(df_prefill['time_offset'], df_prefill['gpu_clock_mhz'], label='Prefill GPU Clock', color='#FF7F0E', linewidth=2)
ax_clock.plot(df_decode['time_offset'], df_decode['gpu_clock_mhz'], label='Decode GPU Clock', color='#2CA02C', linewidth=2, alpha=0.8)
ax_clock.set_xlabel("Time (seconds)", fontsize=12)
ax_clock.set_ylabel("Clock Speed (MHz)", fontsize=11)
ax_clock.legend(loc='lower right')

fig2.tight_layout()
fig2.savefig("./task3_hardware_metrics.png")
print("成功生成底层硬件特征图：./task3_hardware_metrics.png")

# plt.show() # 如果你在本地带界面的环境，取消注释可以直接预览