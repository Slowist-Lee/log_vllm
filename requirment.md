# LLM Serving Energy
## Paper List
- https://arxiv.org/pdf/2602.18755
- https://dl.acm.org/doi/10.1145/3620666.3651329
- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10946802
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10946751
- https://arxiv.org/pdf/2310.03003
- https://dl.acm.org/doi/pdf/10.1145/3757892.3757900
- https://www.sciencedirect.com/science/article/pii/S2666792425000265

## Programming Tasks
### Task 1: Set Up the Environment
Install vLLM (or another inference engine you are familiar with). Download a model with at least 7B parameters, such as Llama-3-8B or Mistral-7B. Prepare at least 50 test prompts, including some short ones (~100 tokens) and some longer ones (~1000 tokens). Run inference once and make sure everything works. Write down your GPU model, driver version, CUDA version, model name, and inference engine version.

### Task 2: Measure Power During Inference
Write a script that records GPU power while inference is running. Use pynvml or nvidia-smi to sample power every 50–100 milliseconds. For each sample, record the timestamp, power in watts, GPU clock speed, memory clock speed, and GPU utilization. Save everything to a CSV file. Compute the total energy in joules (energy = sum of power × time interval across all samples) and the energy per output token (joules/token).

### Task 3: Compare Prefill vs. Decode Power
Separately measure the power consumed during prefill and during decode. A simple way: run inference with a long input and very short output (max_tokens=1) to isolate prefill, then run with a short input and long output (max_tokens=512) to get a decode-dominated trace. You can also look at the power trace over time to identify the two phases. Report the average power (watts) and energy per token for each phase. Plot power over time with the two phases labeled. Then explain why they are different.

### Task 4: Change Settings and See What Happens
Run the same workload under different settings. For each change, measure power draw, throughput (tokens/sec), and energy per token (J/token). Keep everything else the same when you change one variable.

#### 4a. GPU Frequency
Lock the GPU clock using `nvidia-smi -lgc` and try at least 4 different frequencies spread across the GPU’s range. Reset afterwards with `nvidia-smi -rgc`. Plot power, throughput, and J/token vs. frequency. Is there a sweet spot? Does frequency scaling affect prefill and decode differently? If you lower the clock by 30%, does power drop by 30% too? Why or why not?

#### 4b. Batch Size
Run inference with batch sizes of 1, 2, 4, 8, 16 (and higher if your GPU has enough memory). As batch size grows, model weights are reused across requests, so the GPU does more useful work per memory read. Plot J/token vs. batch size. At what batch size does energy per token stop improving? How does power draw change as you increase batch size?

## Prepare A Presentation
Organize what you have learned from those papers, assume the audiences have very limited knowledge about this topic, make sure the content are self-explained.

Summarize what you accomplished through the experiments, for every experiment above, don’t just show the numbers - explain why you see those results. Connect your findings across all tasks. For example, How does it explain why frequency or batch size changes in Task 4 affect energy the way they do? The analysis is the most important part of this test.

----

以下是我在论文中扒出来的一些结论，我希望如果满足上面的要求的话，能够在我的实验中复现他们:

# Benchmark 总结

## 一、单卡测试

1. 【测】GPU的训练过程会经常reach/exceed TDP. [Cited POLCA]

![](Pasted%20image%2020260316100604.png)

2. 【测】训练过程中，由于LLM在computation- and communication-intensive phases之间切换，Power Swings（功率波动）很大

3. 推理过程存在Spike。 这样会有一个尴尬，电力供应必须按“最高峰”设计：为了不让系统在 Prefill 瞬间因为过载而崩溃，数据中心必须按照那个物理尖峰（Spike）来配置电力，但 **实际利用率低**：因为 Prefill 时间很短，大部分时间都在进行低功耗的 Decode，这就导致你申请了 1000W 的电，结果大部分时间只用了 400W，**电力容量被严重浪费了**。

![](Pasted%20image%2020260316102752.png)

4. input size increases, peak power drastically increases， 但input size增加对latency的影响不大
5. batch size越大，peak power越大，latency略微变大
6. 【测】Frequency lock可以大大降功率，对性能影响不大

![|251](Pasted%20image%2020260316104004.png)

7. 显存利用率很低，MEM利用率高 [Cited From Words to Watts]

![|475](Pasted%20image%2020260316111633.png)

8. GPU 的功耗和负载不是完美成正比的。即使在空闲时，8 张 H100 GPU 也要消耗 550W 的待机功耗。这意味着，让 GPU 闲着比让它低效运行更浪费电。将负载整合到更少的 GPU 上，并关闭空闲的 GPU，是节能的关键。[Cited DynamoLLM]
9. 把 GPU 频率拉到最高（也就是 Race-to-idle 策略的做法），其能效（J/token）反而不如稍微降低一点频率时的能效好。因此，**对于 LLM 这种持续不断的计算流，“全速狂飙”是最愚蠢的耗能方式** 所以要选择一个性能的sweet spot [Cited throttLL’eM]


10. [Cited throttLL’eM] 作者测试了从 1 到 32 不等的 Batch Size，以及不同的 GPU 频率 。结论非常精彩（这直接对应你的 **Task 4a 和 4b**）：

![](Pasted%20image%2020260310122050.png)


- 【已测】**吞吐量 (Throughput)**：毫无悬念，<u>Batch Size 越大、GPU 频率越高，系统的整体吞吐量（TPS，每秒生成的 Token 数）就越高 </u>。(为什么？batch size越大，parallelism越大，TPS就越大)

- **延迟 (Latency)**：这里出现了分歧。提高 GPU 频率能有效降低端到端延迟（E2E）和词间延迟（TBT） 。但是，**盲目增大 Batch Size 会导致E2E和TBT延迟恶化**，因为更多的并发请求会导致 GPU 内部的资源竞争加剧 。

- **功耗 (Power)**：这是一个极其关键的发现！**对于给定的 GPU 频率，无论 Batch Size 是 1 还是 32，功耗几乎保持不变** 。功耗主要受 GPU 工作频率的直接影响 ,<u>GPU 频率越大，功率越大</u>。

- **能效 (Energy Efficiency)**：作者用“每焦耳生成的 Token 数 (TPJ)”来衡量能效 。因为大 Batch Size 下功耗不变但产出变多，所以**处理更大的批次总能提高能效** 。更反直觉的是，**并非频率越低越省电** 。实验发现了一个“甜点区”（比如 1050 MHz）：使用最大 Batch Size 配合 1050 MHz 的较低频率，能在对性能影响极小（吞吐量仅下降 6.25%）的情况下，实现高达 **37.4%** 的能效提升 。


> **给你的实验避坑指南**：在做 Task 4a 时，如果你把频率降得太低（比如低于 840 MHz），你会发现虽然瞬时功率（瓦特）降了，但因为算得太慢，完成任务花费的总时间大幅拉长，最终计算出的每 Token 能耗（J/token）反而变差了


1. **Prefill vs. Decode 阶段特征对频率的敏感度差异（对应 Task 3 & 4a）：**  
    Prefill阶段属于Compute-bound（计算密集型），**当GPU频率（Frequency）降低时，计算吞吐量会显著下降，TTFT（首Token延迟）大幅延长**；Decode阶段属于Memory-bandwidth-bound（内存带宽密集型），**当GPU频率降低时，TPOT（单Token生成延迟）延长的幅度较小，受到的性能影响相对较弱**。
    
2. **Input size 对延迟和功耗的影响（对应 Task 3）：**  
    **Input size（输入长度）增加时，Prefill阶段的单次计算耗时和峰值功耗显著增加**；但由于现代GPU拥有极高的内存带宽，**Input size增加对Decode阶段的延迟（TPOT）和功耗表现影响不大**。
    
3. **频率（Frequency）对总功耗与能效的影响（对应 Task 4a）：**  
    **GPU锁定频率（Frequency lock）降低时，GPU的峰值功耗和总体平均功耗会大幅下降**；但是，**功耗下降的比例与频率下降的比例并不完全对等**（例如降频30%并不意味着功耗也精确下降30%），且在Decode阶段适度降频可以在不严重拖慢性能的情况下显著降低单Token能耗（J/token）。
    
4. **Batch Size 对总功耗、延迟与单Token能效（J/token）的影响（对应 Task 4b）：**  
    **Batch size增大时，GPU的总吞吐量增加，整体的峰值功耗和平均功耗随之变大，单次推理的延迟也会略微变大**；但是，由于模型权重被多个请求复用（减少了内存读取代价），**分摊到每个Token上的平均能耗（Energy per token，即J/token）会显著降低**，直到Batch size增加到某个临界点（sweet spot）后，单Token能耗才会停止改善并趋于平稳。
    