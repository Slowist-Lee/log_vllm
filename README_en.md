# Report

19/03/2026, Xin LI

> [!info] Reference
> - Paper Lists, Especially [throttLL’eM](https://arxiv.org/html/2408.05235)
> - [A Cuda Tutorial in Chinese](https://face2ai.com/CUDA-F-1-1-%E5%BC%82%E6%9E%84%E8%AE%A1%E7%AE%97-CUDA/)
> - [Transformers Videos From 3Blue1Brown](https://www.bilibili.com/video/BV13z421U7cs/?spm_id_from=0.0.homepage.video_card.click&vd_source=8b619f7067c22be821f5fd19b41b0eba)
> 
> My Test Scripts are in [Github](https://github.com/Slowist-Lee/log_vllm.git).
> 

> [!tip] Notice
> This report is focused on the completeness and coherence of the information  and only include some basic information I earned during my experiment. 
> 
> Some Paper Notes are not included in this Report. I'll add them later!  
> 

## 1. Transformer & KVCache

To understand LLM inference, we must first start with the Transformer model architecture. The core task of an LLM is to first split the input into tokens, embed each token into a vector, and then—using these vectors as inputs—pass them through various modules to output the most likely subsequent content. The Transformer architecture works as follows:

1. First, the original vector $\overrightarrow{E}_2$ is multiplied by $W_K$, $W_Q$, and $W_V$ to obtain $Q, K, V$. That is:

$$\begin{align}
K&=W_k \cdot X \\
Q&=W_Q \cdot X \\
V&=W_V \cdot X \\
\end{align}$$

2. Next, calculate the Attention score:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V$$
3. Finally, pass the result through an MLP layer.

Here, the most compute-intensive part is $QK^T$.

In mainstream large models, the GPT architecture generates tokens one by one (i.e., it is AutoRegressive). As shown in the figure below, before generating each word, the model needs to reprocess all existing tokens and perform the calculations mentioned above. In other words, generating 10 words requires 9 separate calculations. Thus, the number of computations is high, and the process takes a long time.

![|475](images/Pasted%20image%2020260310211545.png)

You might immediately realize that matrices can be partitioned, and the results of previous matrix multiplications can certainly be reused to some extent. Based on this intuition, the KV Cache was introduced.

We know that $K$ and $V$ are transformed from the initial vector $X$ via projection using weight matrices $W_K, W_V$. They represent the "information to be queried," meaning the content of $K_{t-1}$ and $V_{t-1}$ can be reused. However, the physical meaning of $Q_t$ is the "keyword" converted from the current query; it cannot be cached, so there is no possibility of reusing it. That is, $Q_t$ remains, while $K_t$ and $V_t$ will appear as an accumulated sequence, for example: $[K_1,\dots, K_{t-1},K_t]$.

Therefore, we can simplify the Attention formula as follows:

$$\begin{align}
\text{Attention}&(Q_t, [K_1,...,K_{t-1},K_t], [V_1,...,V_{t-1},V_t]) \\ &= \text{softmax}\left(\frac{Q_t \cdot [K_1,...,K_t]^T}{\sqrt{d_k}}\right) \cdot [V_1,...,V_t]\\ &= \text{softmax}\left(\frac{Q_t \cdot [K_{\text{cache}}, K_t]^T}{\sqrt{d_k}}\right) \cdot [V_{\text{cache}}, V_t]
\end{align}$$

This way, $[K_1,\dots, K_{t-1}]$ does not need to be recalculated. Note that the KV Cache only caches the results of the $W_k \cdot X$ step; it doesn't actually simplify the math when calculating the final Attention score.

At the algorithmic level, how does the introduction of KV Cache affect the characteristics of computation and inference? Let's review the KV Cache workflow. First, we feed the initial prompt to the LLM to obtain the initial KV Cache, calculate the complete Q/K/V matrices and Attention, and complete all calculations required before generating the first token. After that, we use $K_{Cache}$ and $V_{Cache}$ for autoregressive calculation to generate the output word by word. We call the former process **Prefill**, and the latter autoregressive generation **Decode**.

## 2. GPU

### 1. Heterogeneous Computing

In LLM inference, we typically use heterogeneous computing—namely, using the mysterious GPU. The architecture of heterogeneous computing can be illustrated by the diagram below:

![](images/Pasted%20image%2020260318203259.png)

- **Left Image:** A quad-core CPU generally has four ALUs. The ALU is the core for logical calculations (what we refer to when we say quad-core or octa-core). The Control unit and Cache are on-chip, while DRAM (memory) is usually off-chip. The CPU accesses memory via a bus.
- **Right Image:** A GPU. The small green squares are ALUs, and the red box represents an SM (Streaming Multiprocessor). A group of ALUs shares a Control unit, Register File, and Cache. This segment acts like a complete multi-core CPU, but with a difference: there are many more ALUs, the control unit is smaller, computational power is enhanced, and control logic capability is weakened. For programs with complex control logic, a single GPU SM cannot compare to a CPU. However, for tasks with simple logic and massive data, the GPU is highly efficient, and a single GPU contains many SMs.

The CPU and GPU are connected via the PCIe bus, which is used to transfer instructions and data—this also creates a performance bottleneck. A heterogeneous application always involves more than two architectures. Therefore, heterogeneous computing is divided into the following parts:
 $$\boxed{\text{CPU Control}} \leftrightarrow \boxed{\text{Bus Transfer}}\leftrightarrow \boxed{\text{GPU Computation}}$$

Each of these parts has its own bounds and constraints, which mutually restrict each other, leading to the performance characteristics we observe during LLM inference.

### 2. Memory/Compute Bound

![](images/Pasted%20image%2020260319165023.png)


We can use the previously mentioned Prefill and Decode stages to understand the GPU's working mechanism.

From the GPU's perspective, the tasks performed in these two stages are entirely different. During the initial Prefill, we need to calculate $K =W_k \cdot X, Q=W_Q \cdot X, V=W_V \cdot X$ and compute the Attention scores to output the first word. At this point, the GPU is executing massive matrix multiplications. 

In the LLM generation pipeline, Prefill and Decode are two core phases. Due to their distinct task focuses and computation patterns, they face completely different performance bottlenecks: **Compute Bound** and **Memory Bound**.

Let's look at the first execution phase, **Prefill**. Here, the GPU's core job is executing large-scale matrix operations. Since the input Prompt usually contains multiple tokens, the GPU executes massive matrix math, maximizing its computational capacity. If the number of tokens exceeds the GPU's compute limit, they simply have to queue up. This is a classic **Compute Bound** scenario: the GPU's calculation capability becomes the bottleneck. Improving GPU compute efficiency here will significantly shorten the Prefill time.

Once we enter the **Decode** phase, the GPU's working mode fundamentally changes. Each step in the Decode phase generates a single new token and relies heavily on the previous steps' $K_{Cache}$ and $V_{Cache}$. Now, the GPU's calculation task is very small—only small-scale matrix operations of $[1, d] \times [L, d]^T$. However, every single Decode step requires *reading* the entire historical K and V cache from VRAM. As the generated tokens increase and sequence length $L$ grows, the cache size grows linearly, taking longer and longer to read.

Furthermore, after each new token is generated, the new $K_{new}$ and $V_{new}$ must be written back to VRAM. Frequent read/write operations consume massive memory bandwidth. When the cache becomes too large, memory fragmentation can occur, further degrading access efficiency. Crucially, GPUs were designed to process massive parallel computations efficiently. The small-scale computation characteristic of the Decode phase leaves the GPU's compute cores idle most of the time, just waiting for memory read/writes to finish. This "cores waiting for data" scenario is a classic **Memory Bound**.

## 3. Benchmark

Based on the understanding above, we can preliminarily analyze the GPU's performance in LLM inference. Below is some test data and conclusions from my experiments on a V100 GPU, along with some analysis.

> [!info] Equipment Info
> GPU Model & Driver Version:
> Tesla V100-SXM2-32GB, 535.274.02
> CUDA Version (System): 12.8
> Model Name: mistralai/Mistral-7B-Instruct-v0.2
> Inference Engine: vLLM 0.17.1

During our measurements, we recorded Power, Throughput (token/s), and Energy per Token (TPJ).

By definition: $\text{TPJ} = \frac{\text{Power}}{\text{Throughput}}$.

System memory bandwidth info:

```bash
(base) ubuntu@VM-16-10-ubuntu:~$ /usr/local/cuda/extras/demo_suite/bandwidthTest
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla V100-SXM2-32GB
 Quick Mode

 Host to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     11141.4

 Device to Host Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     12858.4

 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     722322.1

Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
```

### 3.1 Prefill vs Decode

Let's first observe the LLM's Prefill and Decode phases. As discussed, Prefill is primarily Compute Bound (limited by SMs), while Decode is Memory Bound (limited by PCIe/Memory Bandwidth). What does this look like in practice?

We ran an inference session and recorded the GPU power. By splitting the graph at the time the first token was generated, we separate Prefill and Decode. The result is shown below:

![](images/Pasted%20image%2020260317200326.png)

As you can see, the power fluctuates wildly with multiple spikes during the Prefill phase, whereas power is much more stable during Decode. This is because the Power metric primarily measures the GPU's active power draw. When power hits a spike, it means the GPU is performing massive matrix calculations, and the compute units are fully saturated. The brief dips between spikes mean a calculation for a specific layer (or layers) has finished, and the system is preparing for the next layer.

Once Prefill ends, the calculation task becomes small, no longer saturating the GPU, so the power drops and becomes stable.

Now let's measure the actual energy consumed: **Energy per Token**. If Prefill draws higher power, does it mean it consumes more energy per token?

![](images/Pasted%20image%2020260318202123.png)

We can see that the Energy/Token for Decode is much higher than Prefill. This is because the high power during GPU computation comes from thousands of threads across different batches running in parallel—millions of transistors firing simultaneously consumes massive energy per second, but gets the job done incredibly fast. During memory transfer (Decode), even though the power isn't as high, memory access is inefficient and takes a long time. Generating a single token requires moving too much data, making the total Energy/Token in the Decode phase extremely large.

To observe this more intuitively, we ran the LLM exclusively in Prefill and Decode phases (Prefill: long prompt + 1 token output / Decode: short prompt + long output):

You can see that the Energy/Token for Prefill is very low, but it multiplies by nearly 50x during Decode. During Decode, GPU Utilization isn't maxed out at all, and Power is stable. However, during Prefill, GPU Utilization spikes straight to 100%, and (likely due to multiple model layers) Power fluctuations are much more obvious.

![](images/Pasted%20image%2020260317215447.png)

![](images/Pasted%20image%2020260317215518.png)

### 3.2 Batch Size & Frequency

For these tests, we only modified two variables: Batch Size and Frequency. Frequency was controlled using the following command (requires physical GPU privileges, so it's hard to do in Docker):

```bash
sudo nvidia-smi -lgc 1000,1000
```

Batching works by grouping multiple requests together, letting the model output results in parallel, and then separating them. This accelerates computation because during memory access, the most time-consuming part is transferring **model weights** and KV Cache to the SMs.

Since the total amount of KV Cache transferred per pass = `Batch Size × Current Sequence Length × Structural Constants (Layers × Attention Heads × Dimension × Data Precision)`, changing the Batch Size affects the time and energy spent moving the KV Cache, but **does not affect** the time and energy spent moving model weights.

Using throttLL'eM, I plotted a set of heatmaps. Due to limited time and budget, the dataset is relatively small. The results are as follows:

![|400](images/Pasted%20image%2020260319095452.png)

#### Conclusion 1: Higher Frequency -> Higher Power

This is easy to understand. Dynamic power consumption of a chip is $P \propto V^2 f$. To achieve higher frequencies, the GPU must increase voltage to maintain signal stability, naturally raising power consumption.

By plotting the $f - P$ relationship as a line graph, we find that GPU frequency and Power share a non-linear relationship. (It's less obvious here because the batch size is small):

![|575](images/Pasted%20image%2020260319100549.png)

Tested with BS=16:

![](2c9fefc47b2a01e584e2c305bd3251d3.png)

#### Conclusion 2: Batch Size Has Little Impact on Power (when Batch Size < 16)

The formula for average power is:
$$P_{avg} = \frac{E_{memory} + E_{compute}}{T_{memory} + T_{compute}}$$

This result doesn't mean batch size has *no* impact on power, but rather that the batches we are processing are currently too small. The system is still heavily Memory Bound. The time spent on memory access ($T_{memory}$) is vastly greater than compute time ($T_{compute}$), meaning $T_{memory} \gg T_{compute}$.

$E_{memory}$ can be further broken down into:
$$E_{memory} = E_{weight} \text{ (weights)} + E_{kv} \text{ (KV Cache)} + E_{act} \text{ (activations)}$$

When Batch Size is in a smaller range (e.g., < 16), increasing it linearly increases compute load (raising $E_{compute}$ and $T_{compute}$) and slightly increases dynamic memory access overhead ($E_{kv}$). However, because the accumulated **KV Cache volume is still far smaller than the model weights themselves** ($E_{weight} \gg E_{kv}$), the massive constant $E_{weight}$ totally dominates the numerator and denominator. Thus, $E_{memory}$ can be viewed as unchanged.

Therefore, the effect of changing batch_size is limited, and $P_{avg}$ barely changes.

![|450](images/Pasted%20image%2020260319095507.png)

#### Conclusion 3: The Larger the Batch Size, the Smaller the TPJ (Energy per Token) / Larger JPT

As Batch Size increases, the GPU generates more tokens in parallel for every single time it loads the model weights. According to our previous analysis, $E_{weight}$ dominates $E_{memory}$. If more tokens share this massive fixed energy cost, the energy allocated to each individual token becomes smaller. Therefore, the Energy per Token (TPJ) decreases.

#### Conclusion 4: GPU Noise/Static Power

This test checked the GPU's base static power to provide a foundation for further analysis. You can see there is quite a bit of baseline power draw, hovering around 23W.

![](images/Pasted%20image%2020260319113303.png)

#### Conclusion 5: Frequency has a "Sweet Spot"

Analyzing the heatmap vertically, you'll find that as Frequency increases, efficiency (TPJ) improves, peaks, and then drops. This is caused by the balance between Memory and Compute mentioned earlier. Increasing frequency only speeds up GPU computation. The system achieves maximum efficiency only when compute speed and memory access speed are perfectly synchronized.

When the frequency is too low, the GPU core runs too slowly. Compute speed is slower than memory speed, making it Compute Bound. This extends the overall runtime. Because the GPU has a static baseline power draw, taking longer wastes energy, resulting in poorer efficiency.

As frequency increases, compute speed catches up to memory speed, improving efficiency. There is a "Sweet Spot" where the GPU's compute speed perfectly matches the maximum bandwidth the VRAM can provide.

If frequency increases further, it hits the "Memory Wall." Beyond 900 MHz, no matter how much you raise the GPU Frequency, the cores are just waiting for VRAM to deliver data. Higher frequency yields no performance gains, but because $P \propto V^2 f$, it wastes more power, dropping efficiency.

We conducted more detailed tests below.

#### Conclusion 6: Increasing GPU Frequency and Batch Size Significantly Increases Throughput

During the compute phase, throughput can be increased by either raising the clock frequency or adding more compute units. During memory access, throughput can only be increased by raising memory bandwidth.

Here, because the Batch Size is relatively small, Throughput is primarily restricted by the compute cores. Therefore, raising frequency or adding compute units significantly boosts throughput.

![|400](images/Pasted%20image%2020260319143244.png)

> [!note] When will BS hit the Memory Wall?
> Hardware Machine Balance:
> $$\text{Machine Balance} = \frac{\text{Peak FLOPS}}{\text{Peak Memory Bandwidth}}$$
> 
> Since actual memory bandwidth is 722GB/s, calculations show the limit is reached when BS hits 173.

#### Conclusion 7: Increasing GPU Frequency Reduces Latency; Increasing Batch Size Sharply Increases End-to-End Latency

![](images/Pasted%20image%2020260319150026.png)

① Increasing GPU frequency speeds up calculation, naturally shortening the time to complete the whole task.
② As batch size grows, latency grows. This is mostly because the GPU has to process more tokens at once. Inside a single batch, it operates on a somewhat static mechanism (similar to static batching), where different requests must wait and be released together:

![](images/Pasted%20image%2020260319150453.png)

Furthermore, the GPU's VRAM determines the total amount of KV Cache vLLM can hold. Therefore, the larger the batch, the more blocks need to wait, leading to longer queue wait times.

#### Conclusion 8: Measuring the Sweet Spot

We measured the specific sweet spot of Frequency vs Efficiency for Prefill, Decode, and End-to-End (e2e) scenarios. Parameters:

- Frequency test range: 750 - 1300, intervals of 50
- Prefill: Input 512, Output 1
- Decode: Input 1, Output 256
- e2e (One full inference): Input 512, Output 256

The resulting line graph is shown below:

![](images/Pasted%20image%2020260319155523.png)

Just as analyzed earlier, at low frequencies, calculations are slow, and the GPU wastes a lot of **static power**. Compute speed lags behind memory speed, so raising frequency speeds up computation and improves efficiency. Even though Decode doesn't have a massive compute workload, if the GPU frequency is too low, it still suffers from being compute-bound.

Beyond 900 MHz, VRAM bottlenecking kicks in, so there are no more performance gains. At this point, Prefill efficiency drops mainly because the GPU voltage is pushed too high, causing dynamic power to surge. Decode efficiency drops mostly because of the memory wall—the GPU is forced to wait, hurting efficiency.

This Report is under Revision (I'm not very sure about some conclusions So I'm trying to solve them...)

%%
## 4. Batching

### 4.1 Orca: Continuous Batch

### 4.2 vllm: Continuous Batch & Memory Management

## 5. PD Disaggregation: Prefill & Decode Disaggregation

### 5.1 DistServe: Different GPUs/Parallel Methods for Prefill and Decode

### 5.2 SplitWise: Different Types of GPUs for Prefill and Decode

## 6. Scheduling

### 6.1 DynamoLLM

### 6.2 throttLL’eM

%%

%%
## 4. Parallelism

Next, we want to accelerate inference based on the characteristics above. Let's introduce a fundamental inference acceleration method: Parallelism. During the inference phase, we use parallelism to speed things up, focusing primarily on the following strategies:

- **Data Parallelism (DP):** Splitting data into multiple batches and processing them on different devices.
- **Pipeline Parallelism (PP):** Splitting the model into multiple stages and processing each stage on different devices.
- **Tensor Parallelism (TP):** Distributing the model parameters across different devices, with each device handling a portion of the parameters.

### 2.1 Data Parallelism

Data parallelism during inference is relatively simple. The key is using the *request* as the smallest unit for parallel splitting, distributing different requests across different devices for inference.

- **Memory Footprint:** Each device holds a complete copy of the model.
- **Communication:** Communication only happens at the start/end of the inference for each request, and only the final input/output needs to be transmitted. Communication volume is small.
- **Throughput:** When the number of simultaneous requests approaches infinity, increasing the degree of parallelism yields near-linear gains in throughput (due to low communication).
- **Latency:** Increasing parallelism has no positive effect on latency, because the compute resources allocated to each individual request remain the same.

### 2.2 Pipeline Parallelism

Pipeline Parallelism divides a single forward pass into multiple stages. It splits the workload by stage, distributing different stages to different devices for inference.

- **Memory Footprint:** Each device only holds the model parameters corresponding to the stage it is responsible for.
- **Communication:** Communication occurs after the device on a given machine finishes its task, passing the hidden states to the device handling the next stage.
- **Throughput:** In an ideal state, all devices in the pipeline are computing simultaneously, increasing total throughput (similar to a CPU pipeline).
- **Latency:** Latency will increase because, for each request, only one device's compute resources are participating in the forward pass at any given time, plus there is overhead from inter-device communication.

Compared to Data Parallelism, PP offers relatively lower improvements in throughput. However, its advantage lies in splitting model weights across different devices, allowing massive models to be deployed across multiple devices.

### 2.3 Tensor Parallelism

Like PP, Tensor Parallelism is a parallel method that splits model weights (Model Parallelism), but it operates at a much finer granularity. TP splits each individual model weight tensor.

![](images/Pasted%20image%2020260317152616.png)

Taking a simple Linear layer as an example, TP splits the weight along the output channel dimension and distributes it across multiple devices. After calculating the results, it gathers them back together.

- **Memory Footprint:** Capable of splitting model weights across multiple devices at a finer granularity than PP.
- **Communication:** TP involves massive amounts of collective communication, because outputs often need to be gathered along the output channel dimension.
- **Throughput:** TP can utilize more compute resources for a single request, thereby reducing the time per request and increasing throughput.
- **Latency:** As mentioned above, since a single request uses more compute resources, latency is drastically reduced.

Although TP offers excellent improvements in both throughput and latency, this assumes exceptionally good communication bandwidth and latency. Because TP incurs massive communication overhead, it usually requires high-speed interconnect networks (like InfiniBand, NVLink) to achieve significant scaling out benefits.
%%