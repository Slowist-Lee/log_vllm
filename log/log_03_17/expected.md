# 结论-代码-结果映射（expected）

本文档用于回答：每条 Benchmark 结论由哪个脚本产出、对应哪些结果文件、当前可验证到什么程度。

## 一、先跑哪些脚本

1. Task 3（Prefill vs Decode 分离）

- 脚本: `python log_pd.py`
- 主要产物: `log/prefill_*_power_log.csv`, `log/decode_*_power_log.csv`, `log/prefill_*_io_log.csv`, `log/decode_*_io_log.csv`
- 环境信息: `log/system_info_log_pd.txt`

1. Task 4（频率与 Batch）

- 脚本: `bash task4_run.sh`
- 主要产物: `log/task4a_results.csv`, `log/task4b_results.csv`
- 环境信息: `log/system_info_inference_core.txt`

1. Task 3 绘图

- 脚本: `python task3.py`
- 主要产物: `task3_power_comparison.png`, `task3_hardware_metrics.png`

1. Task 4 绘图

- 脚本: `python task4_plot.py`
- 主要产物: `task4_bar_charts.png`

1. 端到端 TTFT 分界（新增）

- 脚本: `python e2e_profile_ttft.py --warmup`
- 主要产物: `log/e2e_ttft_profile.csv`
- 可视化: `python e2e_plot_ttft.py`
- 图像: `log/e2e_ttft_profile.png`
- 环境信息: `log/system_info_e2e_profile_ttft.txt`

1. 空闲功耗采样（新增）

- 脚本: `python idle_sample.py --duration 180 --interval 0.05`
- 主要产物: `log/idle_power_log.csv`

1. 干扰实验（可选）

- 脚本: `python interference.py`
- 主要产物（GPU 端）: `log/interference_data_len128.csv`, `log/interference_data_len1024.csv`
- 环境信息: `log/system_info_interference.txt`

1. 干扰实验绘图（本地）

- 脚本: `python interference_plot.py --log-dir ./log`
- 主要产物（本地）: `log/interference_len128.png`, `log/interference_len1024.png`

---

task4需要重跑，因为有peak power的数据 √

1. Sweet Spot 频率扫描（新增）

- 脚本: `python sweet_spot.py --start 600 --end 1500 --step 50 --repeat 3 --warmup`
- 主要产物: `log/sweet_spot_results.csv`, `log/sweet_spot_summary.csv`, `log/sweet_spot_summary.txt`
- 环境信息: `log/system_info_sweet_spot.txt`
- 说明: 通过 TPJ (Tokens per Joule) 指标找到 Prefill/Decode/E2E 阶段能效最优的频率甜点

测 e2e: # 包含 E2E 测试
python sweet_spot.py --start 600 --end 1500 --step 100 --phases prefill decode e2e --warmup

```bash
  # 扫描 600-1500 MHz，步长 50 MHz，每个频率重复 3 次
  python sweet_spot.py --start 600 --end 1500 --step 50 --repeat 1 --warmup
  
  # 指定特定频率列表
  python sweet_spot.py --frequencies 600 800 1000 1100 1200 1400 1500 --repeat 1
  
  # 只扫描 Prefill 阶段 
  python sweet_spot.py --start 800 --end 1200 --step 50 --repeat 1 --warmup --phases prefill
  
  # 包含 E2E 测试
  python sweet_spot.py --start 600 --end 1500 --step 100 --phases prefill decode e2e
```

1. 输入大小扫描实验（新增）√

- 脚本: `python input_profile.py --warmup --input-lengths 128 256 512 1024 2048`
- 主要产物: `log/input_profile_*.csv`, `log/input_profile_summary.csv`
- 环境信息: `log/system_info_input_profile.txt`


1. 端到端负载扫描实验（新增）

- 脚本: `python e2e_profile_ttft.py --warmup --load-levels 128 256 512 1024 2048`
- 主要产物: `log/e2e_load_profile_*.csv` (每个负载水平), `log/e2e_load_profile_summary.csv` (汇总结果)
- 环境信息: `log/system_info_e2e_profile_ttft.txt`

锁频跑推理，需要跑两个，还有一个e2e的

```bash 
# 示例：以 1000 MHz 频率运行
./run_log_pd_locked.sh 1000 # [done]

```

干扰实验需要重跑，来测试PD分离

## 二、Benchmark 结论映射

| # | 结论 | 当前状态 | 对应脚本 | 结果文件 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | 训练过程经常 reach/exceed TDP | 文献结论（本仓库未测） | 无 | 无 | 你当前代码是推理实验，不含训练 loop。 |
| 2 | 训练过程 Power Swings 大 | 文献结论（本仓库未测） | 无 | 无 | 同上。 |
| 3 | 推理存在 Spike，Prefill 峰值高于 Decode | 可验证（已测） | `log_pd.py`, `task3.py`, `e2e_profile_ttft.py`, `e2e_plot_ttft.py` | `log/prefill_*_power_log.csv`, `log/decode_*_power_log.csv`, `task3_power_comparison.png`, `log/e2e_ttft_profile.csv` | Prefill/Decode 曲线对比与 TTFT 分界都能支撑该结论。 |
| 4 | Input size 增大使峰值功率升高，延迟变化不大 | 部分可验证（需补充 input sweep） | `log_pd.py`（现有仅长 vs 短），`e2e_profile_ttft.py`（可扩展） | 现有 `log/prefill_*`, `log/decode_*` | 若要严格证明，需要固定其它变量，系统扫描多个输入长度。 |
| 5 | Batch size 越大，峰值功率略升、延迟略升 | 部分可验证 | `inference_core.py`, `task4_run.sh` | `log/task4b_results.csv` | 目前 CSV 有平均功率/时长/TTFT/TPOT，可做趋势；若要“峰值功率”需记录 batch 实验的逐采样功率流。 |
| 6 | 频率锁定可显著降功率，对性能影响有限 | 可验证（已测） | `inference_core.py`, `task4_run.sh`, `task4_plot.py` | `log/task4a_results.csv`, `task4_bar_charts.png` | 看不同频率下 avg_power 与 throughput 变化。 |
| 7 | 显存利用率低而 MEM 利用率高 | 可验证（Task 3日志） | `log_pd.py`, `task3.py` | `log/prefill_*_power_log.csv`, `log/decode_*_power_log.csv`, `task3_hardware_metrics.png` | 需要重点看 `util_mem_pct` 与 `util_gpu_pct` 的对比。 |
| 8 | 空闲功耗高，功耗与负载非线性 | 可验证（新增 idle 采样） | `idle_sample.py`, `e2e_profile_ttft.py` | `log/idle_power_log.csv`, `log/e2e_ttft_profile.csv` | 先测 idle 平均/分位功耗，再与推理阶段功耗对比，验证“功耗与负载非线性”。 |
| 9 | 最高频不一定最省 J/token，存在 sweet spot | 可验证（已测） | `sweet_spot.py`, `inference_core.py`, `task4_run.sh`, `task4_plot.py` | `log/sweet_spot_summary.csv`, `log/task4a_results.csv`, `task4_bar_charts.png` | 使用 `sweet_spot.py` 精细扫描找到 TPJ 最优频率；或直接比较不同频率下 `j_per_token`。 |
| 10a | 吞吐：Batch 越大吞吐越高 | 可验证（已测） | `inference_core.py`, `task4_run.sh`, `task4_plot.py` | `log/task4b_results.csv`, `task4_bar_charts.png` | 当前数据已呈现单调上升。 |
| 10b | 延迟：提频有助于延迟，盲目增大 batch 会恶化延迟 | 可验证（已补字段） | `inference_core.py`, `task4_run.sh` | `log/task4a_results.csv`, `log/task4b_results.csv` | 关注 `ttft_s`、`tpot_s`、`duration_s`。 |
| 10c | 功耗主要受频率影响，同频下不同 batch 功耗变化不大 | 部分可验证 | `inference_core.py`, `task4_run.sh` | `log/task4a_results.csv`, `log/task4b_results.csv` | 当前是 avg_power 粗粒度验证；要更强证据建议保存 batch 实验逐采样功率曲线。 |
| 10d | 能效：大 batch 提升能效，频率存在甜点区 | 可验证（已测） | `sweet_spot.py`, `inference_core.py`, `task4_run.sh`, `task4_plot.py` | `log/sweet_spot_summary.csv`, `log/task4a_results.csv`, `log/task4b_results.csv`, `task4_bar_charts.png` | 使用 `sweet_spot.py` 通过 TPJ 指标精确定位频率甜点；看 `j_per_token` 随 batch/频率趋势。 |
| A1 | Prefill 对频率更敏感：降频会显著拉长 TTFT | 可验证（已补 TTFT） | `inference_core.py`, `task4_run.sh` | `log/task4a_results.csv` | 比较 Prefill 行的 `ttft_s` 随频率变化。 |
| A2 | Decode 对频率相对不敏感：TPOT 增幅较小 | 可验证（已补 TPOT） | `inference_core.py`, `task4_run.sh` | `log/task4a_results.csv` | 比较 Decode 行 `tpot_s` 随频率变化。 |
| A3 | Input size 影响 Prefill 更大，Decode 影响相对小 | 部分可验证（需 input sweep） | `log_pd.py`, `e2e_profile_ttft.py` | 需新增 input sweep 结果 | 建议扫描 128/256/512/1024 prompt tokens。 |
| A4 | Batch 增大降低 J/token，直至临界点趋稳 | 可验证（已测） | `inference_core.py`, `task4_run.sh`, `task4_plot.py` | `log/task4b_results.csv`, `task4_bar_charts.png` | 目前到 BS=16；若显存允许可继续 32/64 观察平台期。 |

## 三、字段对齐说明（已修复）

`inference_core.py` 现在写入字段与 `task4_run.sh` 头部完全一致：

- Task4a: `phase,frequency_mhz,duration_s,ttft_s,tpot_s,avg_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens`
- Task4b: `batch_size,duration_s,ttft_s,tpot_s,avg_power_w,total_energy_j,throughput_tps,j_per_token,total_output_tokens`

其中：

- `ttft_s`：优先从 vLLM 输出指标提取，提取失败时使用可解释兜底值。
- `tpot_s`：按 `(duration_s - ttft_s) / total_output_tokens` 计算。

## 四、你现在怎么跑（推荐顺序）

1. 跑 Task 4 主实验：

```bash
bash task4_run.sh
```

1. 画 Task 4 图：

```bash
python task4_plot.py
```

1. 跑 Prefill/Decode 分离实验：

```bash
python log_pd.py
python task3.py
```

1. 跑端到端 TTFT 分界实验（建议）：

```bash
python e2e_profile_ttft.py --warmup
python e2e_plot_ttft.py
```

1. 跑空闲功耗基线（建议在无其它作业时执行）：

```bash
python idle_sample.py --duration 180 --interval 0.05
```

1. 干扰实验（可选，排队效应）：

```bash
python interference.py
```

1. 本地绘制干扰实验图：

```bash
python interference_plot.py --log-dir ./log
```

1. Sweet Spot 频率扫描实验（建议，精细扫描频率甜点）：

```bash
# 精细扫描 600-1500 MHz，步长 50 MHz，每个频率重复 3 次
python sweet_spot.py --start 600 --end 1500 --step 50 --repeat 3 --warmup

# 或指定特定频率列表进行高精度测试
python sweet_spot.py --frequencies 600 700 800 900 1000 1050 1100 1150 1200 1300 1400 1500 --repeat 5 --warmup

# 只扫描 Prefill 和 Decode 阶段
python sweet_spot.py --start 600 --end 1500 --step 50 --phases prefill decode --warmup

# 包含 E2E 测试
python sweet_spot.py --start 600 --end 1500 --step 100 --phases prefill decode e2e --warmup
```

1. 输入大小扫描实验（建议）：

```bash
python input_profile.py --warmup --input-lengths 128 256 512 1024 2048
```

1. 端到端负载扫描实验（建议）：

```bash
python e2e_profile_ttft.py --warmup --load-levels 128 256 512 1024 2048
```

## 五、一句话总结

你现在已有一条完整链路：

- 分离实验（Task 3）解释 Prefill/Decode 差异；
- 参数扫描（Task 4a/4b）解释频率与 batch 对功耗、吞吐、J/token 的影响；
- 端到端 TTFT 打点把两者串成一张可解释曲线。
