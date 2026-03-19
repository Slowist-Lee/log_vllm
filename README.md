# LLM Serving

[English Version](./README_en.md)

本文参考了：
- https://blogs.erix025.me/EfficientAI/sct-llm-talk/sct-llm-talk/
- https://face2ai.com/CUDA-F-1-1-%E5%BC%82%E6%9E%84%E8%AE%A1%E7%AE%97-CUDA/
以及老师给的七篇文献。

我的测试脚本： https://github.com/Slowist-Lee/log_vllm.git
## 1. Transformer & KVCache

要理解LLM的推理首先要从Transformer这个模型结构说起，整个LLM所干的事情就是，先将输入切成token，再将每个token编码（embedding）成一组向量（vector），作为一组输入，使用相应的模块，给出最有可能的接下来的内容。Transformers 架构做的事情如下：

1. 首先先将原先的向量$\overrightarrow{E}_2$分别乘以$W_K$,$W_Q$,$W_V$，得到$Q,K,V$.  即：

$$\begin{align}
K&=W_k \cdot X \\
Q&=W_Q \cdot X \\
V&=W_V \cdot X \\
\end{align}$$

2. 再计算注意力分数：
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V$$
3. 在后面接一层MLP

其中这里最耗费算力的是$QK^T$。

在主流的大模型中，GPT 架构是逐个token生成的（即自回归-AutoRegressive 的），如下图，模型每生成一个单词之前，都需要重新处理目前已有的所有token内容，并进行上述计算。也就是说，如果要生成10个单词，就要进行9次计算，因此计算的次数多，时间很长。

![|475](images/Pasted%20image%2020260310211545.png)

于是，聪明的你会想到了矩阵是可以分块的，之前计算的矩阵积的结果在一定程度上一定可以复用。出于这样的直觉，程序员提出了KV Cache。

我们知道，$K$ 和 $V$ 都由初始的向量 $X$ 根据一定的权重矩阵 $W_K, W_V$ 通过 projection 转化而来，代表【被查询的信息】，是可以复用 $K_{t-1}$ 和 $V_{t-1}$ 的内容的。而 $Q_t$ 的物理意义是需求所转换的“关键词”，它没有办法缓存，因此没有复用的可能性。即 $Q_t$ 就是 $Q_t$, 而 $K_t$ 和 $V_t$ 会表现为例如：$[K_1,\dots, K_{t-1},K_t]$。

因此我们回顾注意力公式，可以进行如下化简：

$$\begin{align}
\text{Attention}&(Q_t, [K_1,...,K_{t-1},K_t], [V_1,...,V_{t-1},V_t]) \\ &= \text{softmax}\left(\frac{Q_t \cdot [K_1,...,K_t]^T}{\sqrt{d_k}}\right) \cdot [V_1,...,V_t]\\ &= \text{softmax}\left(\frac{Q_t \cdot [K_{\text{cache}}, K_t]^T}{\sqrt{d_k}}\right) \cdot [V_{\text{cache}}, V_t]
\end{align}$$

这样，$[K_1,\dots, K_{t-1}]$ 就不需要重复计算了。KVCache只是计算了之前的$W_k \cdot X$这一步
，在算Attention分数的时候实际上没有化简。

在算法层面上 KV Cache 的引入，会对计算和推理的特点有什么影响吗？我们回顾KV Cache的过程。首先我们会先将初始的prompt喂给LLM，得到初始的KV Cache，计算完整的 Q/K/V 矩阵和 Attention，得到第一个 token 前的所有计算。后面我们会利用 $K_{Cache}$ 和 $V_{Cache}$ 自回归计算，进行逐字输出。我们将前者的过程叫做Prefill，后者的自回归生成叫做Decode。

## 2.  GPU

### 1. 异构计算

在LLM推理中，我们往往使用异构计算，也就是神秘的GPU，异构计算的结构以下面的示意图为例：

![](images/Pasted%20image%2020260318203259.png)

- 左图：一个四核CPU一般有四个ALU，ALU是完成逻辑计算的核心，也是我们平时说四核八核的核，控制单元，缓存也在片上，DRAM是内存，一般不在片上，CPU通过总线访问内存。
- 右图：GPU，绿色小方块是ALU，红框代表SM（Streaming Multiprocessor）。这一组ALU公用一个Control单元、Register File和Cache，这个部分相当于一个完整的多核CPU，但是不同的是ALU多了，control部分变小，计算能力提升了，控制能力减弱了。对于控制（逻辑）复杂的程序，一个GPU的SM是没办法和CPU比较的，但是对了逻辑简单，数据量大的任务，GPU高效，并且一个GPU有好多个SM。

CPU和GPU之间通过PCIe总线连接，用于传递指令和数据，这也造成了一部分性能瓶颈。 然后一个异构应用一定包含两种以上架构。因此，异构计算分为以下部分：
 $$\boxed{\text{CPU控制部分}} \leftrightarrow \boxed{\text{总线传输}}\leftrightarrow \boxed{\text{GPU计算}}$$

这里的每个部分都有自己的bound，他们之间会相互制约，所以才存在了以下LLM推理时表现出来的特性。

### 2. Memory/Compute Bound

![](images/Pasted%20image%2020260319165023.png)

我们就拿之前说的Prefill和Decode来理解GPU的工作机制。

从GPU的角度来看，这两个阶段所做的事情是不一样的。第一次Prefill的时候，我们需要计算 $K =W_k \cdot X, Q=W_Q \cdot X, V=W_V \cdot X$ 并计算Attention分数，输出第一个词，这时候GPU做的是大规模的矩阵相乘。而Decode的时候，

在大语言模型（LLM）的生成流程中，Prefill（预填充）和Decode（解码）是两个核心阶段，从GPU的工作机制来看，这两个阶段的任务重心、计算模式有着天壤之别，也直接导致了它们分别面临着不同的性能瓶颈——Compute Bound（计算受限）与Memory Bound（内存受限）。

先看第一次执行的Prefill阶段，此时，GPU的核心工作是执行大规模的矩阵相乘运算：我们需要计算 $K =W_k \cdot X$, $Q=W_Q \cdot X$, $V=W_V \cdot X$ 并计算Attention分数，再经过一定计算输出第一个词。

由于输入Prompt通常包含多个token，GPU会执行大规模的矩阵运算，计算量拉满，即如果token超过GPU的【计算限制】只能排队，这就是典型的Compute Bound（计算受限）场景：GPU的计算能力成为了整个阶段的性能瓶颈，此时提升GPU的计算效率，能显著缩短Prefill阶段的耗时。

而进入Decode阶段后，GPU的工作模式会发生根本性转变，Decode阶段的每一步生成一个新的token，且必须依赖上一步的$K_{Cache}$和$V_{Cache}$。此时，GPU的计算任务很小，只有$[1, d]×[L, d]^T$的小规模矩阵运算，但每一步Decode都需要从显存中【读取】全部历史K、V缓存——随着生成的token数量增多，序列长度L不断增加，缓存的体积也线性增长，读取这些数据的时间会越来越长。

另一方面，每次生成新token后，都要将新的$K_{new}$、$V_{new}$写入显存，频繁的读写操作会占用大量的显存带宽，尤其是当缓存体积过大时，还可能出现显存碎片，进一步降低访问效率。更关键的是，GPU的设计初衷是为了高效处理大规模并行计算，而Decode阶段的小规模计算的特点，会导致GPU的计算核心大部分时间处于空闲状态，只能等待内存读写完成——这种“核心等数据”的场景，就是典型的Memory Bound（内存受限）。

## 3. Benchmark

根据上面的这些理解，我们就可以初步分析GPU在LLM推理上的一些表现。这里放一些我在V100的测试数据和结论，尝试进行一些分析。

> [!info] 设备信息
> GPU Model & Driver Version:
> Tesla V100-SXM2-32GB, 535.274.02
> CUDA Version (System): 12.8
> Model Name: mistralai/Mistral-7B-Instruct-v0.2
> Inference Engine: vLLM 0.17.1
> 

我们在测量过程中会记录功率（Power），吞吐量（Throughput, token/s）以及 每个Token的能耗（TPJ）。

根据定义，有 $\text{TPJ} = \frac{\text{Power}}{\text{Throughput}}$ 。

系统的内存相关信息：

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

首先先观察LLM的Prefill和Decode阶段，根据我们之前讲的，Prefill主要是Compute Bound（受限于SM），Decode是Memory bound（受限于PCle）。那具体的表现是怎么样的呢？

我们执行一次推理，记录GPU的功率。通过第一次生成token的时间来分割Prefill和Decode，结果图如下：

![](images/Pasted%20image%2020260317200326.png)

可以看到模型在Prefill阶段的功率波动很大，并存在多个尖峰（Spike），而Decode阶段的功率波动没那么明显。这是因为Power测的主要是GPU的功率，当功率到达尖峰的时候，说明GPU在进行大矩阵计算，计算单元是完全跑满的；尖峰之间的短暂下降，代表着某一层（或某几层）的计算已经完成，系统正在为下一层做准备。

当Prefill结束后，模型的计算任务变小，不再让GPU饱和运行，所以功率就变低变稳定了。

我们再具体测量GPU耗费的能量 Energy per Token。Prefill阶段的推理功率比较高，是否意味着它也会更耗电呢？

![](images/Pasted%20image%2020260318202123.png)

可以看到Decode的Energy/Token远高于Prefill/Token，这是因为GPU计算时的功率大，主要是来源于不同的batch，不同的线程全部在并行地计算，成千上万个晶体管一起工作导致的每秒消耗的能量巨大。搬运时（访存过程中）虽然功率没那么大，但是访存效率低，时间长，每生成一个token需要搬运的东西太多了，因此Decode阶段的Energy/Token很大。

为了更直观地观察，我们分别让LLM跑Prefill和Decode阶段：（Prefill：long prompt + 1 token output / Decode: short prompt + long output）

可以看到Prefill的Energy/Token很低，但是Decode阶段将近翻了50倍。在跑Decode阶段时GPU的Utilization根本没跑满，Power也比较平稳；而Prefill的GPU Utilization直接飙到100%，（可能因为计算层数比较多）Power的上下波动就更明显了。

![](images/Pasted%20image%2020260317215447.png)

![](images/Pasted%20image%2020260317215518.png)

### 3.2 Batch Size & Frequency

我们在这里根据要求只修改Batch Size和Frequency这两个变量。频率我们使用以下命令控制（需要对GPU有物理权限，所以不太能开docker）：

```bash
sudo nvidia-smi -lgc 1000,1000
```

Batch会将整个Prompt先分成若干个batch，让模型并行地输出结果，然后再将结果拼到一起。这是为了加速计算的过程，在访存过程中，主要是**模型权重**/KV Cache搬运到SM比较耗费时间。

由于单次搬运的 KV Cache 总量 = Batch Size × 当前上下文长度 (Sequence Length) × 结构常数【 (层数 × 注意力头数 × 维度 × 数据精度大小)】，所以Batch Size 会影响搬运KV Cache的时间和能量，但不会影响搬运权重的时间和能量。

按照 throttLL'eM 画了一组热力图，因为时间财力都有限，所以我测的数据也比较少，结果如下：

![|400](images/Pasted%20image%2020260319095452.png)

#### 结论1: 频率越高, 功率越高

频率越高功率越高这件事情还算好理解，芯片的动态功耗 $P \propto V^2 f$，而为了达到更高的频率GPU也需要增加电压来保持信号稳定，所以功率自然会升高。

将 $f - P$ 图象绘制成折线图，可以发现 GPU 频率和 Power 是非线性的。因为这里的batch size比较小体现的不明显：

![|575](images/Pasted%20image%2020260319100549.png)

BS=16测了一组：

![](2c9fefc47b2a01e584e2c305bd3251d3.png)
#### 结论2: Batch Size 对于功率影响很小（Batch Size < 16）

平均功率计算公式：
$$P_{avg} = \frac{E_{memory} + E_{compute}}{T_{memory} + T_{compute}}$$

这个结果并不是说batch size对power没有影响，只是目前需要处理的batch都还太小了，此时还是Memory Bound，Memory-访存的时间远远大于Compute-计算的时间，即上式 $T_{memory} >> T_{compute}$。

$E_{memory}$也可以进一步拆分：
$$E_{memory} = E_{weight} \text{ (搬运权重)} + E_{kv} \text{ (搬运KV Cache)} + E_{act} \text{ (搬运激活值)}$$

当 Batch Size 处于较小区间（如 < 16）时，虽然增大 Batch Size 会线性增加计算量（使 $E_{compute}$ 和 $T_{compute}$ 变大），并略微增加动态访存开销（$E_{kv}$）。但由于此时积累的 **KV Cache 体积远小于模型权重本身**（即 $E_{weight} \gg E_{kv}$），极其庞大的常数项 $E_{weight}$ 在公式的分子和分母中占据了绝对统治地位，可以视作$E_{memory}$ 没什么变化

因此 batch_size改变的效果实际上是有限的，$P_{avg}$的变化就不大了。

![|450](images/Pasted%20image%2020260319095507.png)

#### 结论3：Batch Size 越大，TPJ 越大（JPT 越小）

Batch Size变大，GPU每搬运一次权重，所并行能够完成任务、生成的token变多。根据之前的分析，我们知道目前占大头的主要还是$E_{memory}$中的$E_{weight}$。有更多的token分担一定的能量，那每个token分到的能量就更小，即TPJ越小，JPT越大。

#### 结论4：GPU的噪声功耗

这条主要测试了GPU的噪声功耗，为下列分析提供一下基础。可以看到噪声还是不少的，噪声的功率大概在23W左右。

![](images/Pasted%20image%2020260319113303.png)
#### 结论5：频率存在Sweet Spot

纵向分析上述热力图，会发现 Frequency vs TPJ 呈现先增大后减小。这也是由于我们之前所说的 Memory vs Compute 的制衡引起的。频率的变大只会影响GPU compute的速度，当compute的速度和访存速度配合的最好的时候才是TPJ最大的。

这是因为当频率比较小的时候，GPU 核心频率太低导致计算速度太慢，compute速度小于memory速度，此时为compute bound，拉长了整体运行时间，由于GPU 的静态底噪功耗的存在，导致能量流失，TPJ较小。

随着频率提升，计算速度赶上访存速度，TPJ变大。存在一个Sweet Spot，GPU 核心计算和消化数据的速度恰好等于显存带宽提供数据的极限速度。

当频率再增大时会撞到内存墙，超过 900 MHz 后，无论怎么提升 GPU Frequency，核心只能等着显存把数据搬过来，提升频率无法带来性能收益，但却因为 $P \propto V^2 f$ 会消耗更多能量，TPJ变小。

后续我们进行了更详细的测试。
#### 结论6：提高GPU频率、增加Batch Size会显著增加吞吐量

Throughput在计算过程中，只要提高计算频率，或增加计算单元，都可以增加计算过程中的Throughtput。在访存中，只能提升显存带宽，才可以提升Throughput。

在这里，由于Batch_Size较小，Throughput主要受制于计算核心。因此提高频率或增加计算单元数量时，吞吐量都会显著增长。

![|400](images/Pasted%20image%2020260319143244.png)

> [!note] 什么时候BS才会撞到内存墙？
> 硬件的算力带宽比 (Machine Balance)：
> $$\text{Machine Balance} = \frac{\text{Peak FLOPS}}{\text{Peak Memory Bandwidth}}$$
> 
> 由于实际显存带宽是722GB/s，计算出来BS到173之后会达到极限。

#### 结论7：提高GPU频率会降低延迟，增加Batch Size会急剧增加端到端延迟

![](images/Pasted%20image%2020260319150026.png)

① 提高GPU频率，计算速度快了，完成整个任务的时间自然就短了；
② batch变大，latency变大，主要是因为在batch变大时，GPU需要一次性处理的token变多了。而单个batch内还是static的机制，有点类似static batch，不同的请求之间还是需要等待然后一起释放：

![](images/Pasted%20image%2020260319150453.png)

而GPU的显存决定了vLLM能同时容纳的KV Cache总量，因此batch越大，需要等待的块也就越多，队列的等待时间还是会越长。

#### 结论8：Sweet Spot 的具体测量

测量Frequency vs TPJ  的具体sweet spot，分别对prefill, decode 以及 e2e 的场景进行测量。具体参数如下：

- 测试频率范围：750 - 1300，间隔为50
- Prefill: Input 512, Output 1;
- Decode: Input 1. Output: 256;
- e2e (一次正常推理): Input 512, Output 256;

生成的折线图如图所示：

![](images/Pasted%20image%2020260319164241.png)

和之前分析的一样，由于频率低，计算慢，GPU花费大量**静态功耗**，计算速度赶不上访存速度，因此提升频率能让计算速度增加，TPJ上升。即使Decode需要计算的量并不大，但当GPU频率太低时依然受限于compute bound. 

超过900之后显存等访存，所以不再存在性能收益。此时，Prefill的TPJ下降的主要原因是因为GPU增加的电压太大，导致动态功耗增长太多，TPJ下降；Decode更多是因为内存墙缘故，GPU不得不进行等待，导致TPJ下降。

