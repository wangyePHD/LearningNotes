# Sol-RL：FP4 Explore, BF16 Train —— 扩散 RL 的高效 Rollout Scaling (NVIDIA)

> **标签**：`Vision` `RL` `GRPO` `Diffusion RL` `FP4` `NVFP4` `Quantization` `Rollout Scaling` `Systems`
> **更新时间**：2026-10-06
> **原文**：`Papers/Sol-RL.pdf`（20 页，NVIDIA / HKU / MIT）
> **标题**：*FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling*
> **arXiv**：[2604.06916](https://arxiv.org/abs/2604.06916) v1（2026-04-08）
> **作者**：Yitong Li, Junsong Chen, Shuchen Xue, Pengcuo Zeren, Siyuan Fu, Dinghao Yang, Yangyang Tang, Junjie Bai, Ping Luo, Song Han, Enze Xie
> **实验基模**：SANA / FLUX.1 / SD3.5-Large，8× NVIDIA B200
> **精读进度**：§1–§5 全完结（Data 无关，纯 RL 方法论）
> **相关笔记**：[Qwen-Image-2.0 RLHF 统一对齐](./qwen-image-2-rl.md)（Hybrid CFG 同族思想）｜ [Swift-Image 并行专家 RL](./swift-image-rl.md)（CFG 非对称同构）｜ [算法 × 奖励 × 基模对比](./rl-comparison-2026.md) ｜ [训练 Playbook §4](../training-playbook.md)

---

## 1. 总体

Sol-RL 的出发点是一个已经成立的观察：在 group-based diffusion RL 里，**扩大同一个 prompt 的 rollout 数量 $N$ 就能提升性能，即使真正参与训练的样本数 $K$ 不变**。但大规模 rollout 极贵，而且论文把瓶颈转移这件事说得很准——由于只有一小撮高对比度样本真正进入梯度更新，其余样本在 rollout 之后直接被丢弃：

> <mark class="hl-trick">**Because only a small set of highly contrastive samples is ultimately used for optimization, scaling the candidate pool shifts the training bottleneck from policy optimization to candidate generation.**</mark>

<mark class="hl-trick">**换句话说：卷 massive rollout 会把瓶颈从 backward 搬到 forward，而这个 forward 的绝大部分算力是浪费的。**</mark>

它的解法是把 <mark class="hl-key">**Exploration 与 Policy Training 彻底拆开**</mark>：先用低精度大批量探索，再把昂贵的高精度计算集中到少量 informative samples 上。

$$
\boxed{
\text{FP4 Explore}
\rightarrow
\text{Reward Ranking}
\rightarrow
\text{Select Extreme Seeds}
\rightarrow
\text{BF16 Regeneration}
\rightarrow
\text{Policy Update}
}
$$

论文自己的框架全称是 <mark class="hl-trick">**Sol-RL (Speed-of-light RL), a novel FP4-empowered Two-stage Reinforcement Learning framework**</mark>，标准配置可直接记成：

$$
\boxed{
96\ \text{个 FP4 candidates}
\;\longrightarrow\;
\text{Top-12} + \text{Bottom-12}
\;\longrightarrow\;
24\ \text{个 BF16 rollout}
\;\longrightarrow\;
\text{RL Update}
}
$$

<mark class="hl-trick">**`top-12 and bottom-12` 是论文原话**</mark>，论文把这个配置称为 **`24-in-96` rollout setting**，对照组是 `24-in-24 (BF16)`（不扩展 exploration）与 `24-in-96 (FP4)`（直接拿 FP4 训）。

**关键在于 FP4 阶段不负责产出训练 trajectory，只负责「找 seed」**。因为低精度量化会影响最终图像细节，但作者发现它对候选样本的 <mark class="hl-key">**relative reward ranking**</mark> 仍然可靠，尤其能较稳定地识别 Top-K 与 Bottom-K。所以 Sol-RL <mark class="hl-key">**并不要求**</mark> $I_i^{\rm FP4}\approx I_i^{\rm BF16}$，<mark class="hl-key">**只要求**</mark>：

$$
\boxed{
\operatorname{Rank}\!\left(R(I_i^{\rm FP4})\right)
\approx
\operatorname{Rank}\!\left(R(I_i^{\rm BF16})\right)
}
$$

一句话就是 <mark class="hl-trick">**FP4 不需要生成得足够准，只需要「挑得准」**</mark>。Table 8 给了这条的量化证据（CLIPScore reward，NVFP4 proxy vs BF16 真值）：$\tau=0.752$、$\rho=0.900$，Top-4 命中 **95.7%** → Top-8 **93.9%** → Top-12 **92.2%**。<mark class="hl-key">**注意保真度随 $K$ 单调下降，所以「挑得准」不是免费的**</mark>——<mark class="hl-trick">**$K$ 是需要自己消融的量，不是照抄参数**</mark>。被选中的 seed 随后用 BF16 policy 重新 rollout，真正进入 policy gradient 的仍是高精度 trajectory，从而避免量化污染。

Figure 2 的耗时分解说明了加速来自哪里（同一 prompt）：

| 阶段 | 耗时 |
| :--- | --- |
| <mark class="hl-trick">BF16 全量 rollout（naïve scaling，96 张）</mark> | <mark class="hl-trick">**451s**</mark> |
| <mark class="hl-trick">FP4 直接量化 rollout（96 张）</mark> | 184s |
| <mark class="hl-trick">**FP4 Explore**（Sol-RL 探索阶段，6 denoise steps）</mark> | <mark class="hl-key">**125s**</mark> |
| <mark class="hl-key">**BF16 Re-gen**（只重跑 24 张，10 denoise steps）</mark> | <mark class="hl-key">**62s**</mark> |

<mark class="hl-key">**⚠️ 加速有三个不同口径，权威来源是 Table 5，引用时请直接用它**</mark>：

$$
\underbrace{1.41\text{–}2.41\times}_{\text{rollout 段}}
\qquad
\underbrace{1.25\text{–}1.62\times}_{\text{单次 iteration 端到端}}
\qquad
\underbrace{1.91\text{–}4.64\times}_{\text{收敛（time-to-target）}}
$$

<mark class="hl-key">**论文摘要与 Figure 1/2 给的 `2.4×` / `4.64×` 分属 rollout 与收敛两个口径；Figure 1 的三条 ImageReward 曲线分别是 SANA `2.42×`、FLUX.1 `3.00×`、SD3.5-Large `4.64×`**</mark>。<mark class="hl-key">**注意「扩 rollout」本身有代价**</mark>——<mark class="hl-trick">**`24-in-24 (BF16)` 单次 iteration 只要 353s，`24-in-96 (BF16)` 却要 691s，rollout 段本身就是 `×4`**</mark>；<mark class="hl-key">**Sol-RL 的价值在于「探索空间扩了 4 倍之后，总时长仍被压回接近原量级」**</mark>。

<mark class="hl-key">**⚠️ 另外 Figure 3(a) 的图例有一处会误导人的标注，建议以 Table 5 为准**</mark>：<mark class="hl-trick">**图中把 184s 那一根标为 `24-in-96 (FP4)`，但 184s 其实是 Naïve Quant（直接在 FP4 上训练）的耗时；Sol-RL 自己的 rollout 是 `125s + 62s = 187s`，对应 Table 5 的 2.41×（$451/187$），而不是图上标的 2.45×（$451/184$）**</mark>。<mark class="hl-key">**两者只差 3 秒，但结论完全不同——图上那根柱子的 `Train with FP4 Rollout` 曲线正是 Naïve Quant，而不是 Sol-RL**</mark>。

真正可迁移的不是 FP4 本身，而是这个抽象——<mark class="hl-key">**把大量、便宜、近似的计算用于「搜索哪里值得学」，把昂贵的高精度计算集中到少量 informative samples 上**</mark>：

$$
\boxed{
\text{Cheap Approximation for Exploration}
+
\text{Accurate Computation for Learning}
}
$$

<mark class="hl-trick">**这个抽象远超量化本身**</mark>——<mark class="hl-key">**任何「便宜代理足以完成排序 / 筛选，但不足以完成学习」的领域都适用**</mark>：候选检索、prompt 筛选、数据清洗、reward model 粗筛。相应地，<mark class="hl-key">**Sol-RL 的准确定位是 Diffusion RL 的 Exploration Efficiency 方法，而不是一个新的 GRPO objective**</mark>——<mark class="hl-trick">**它不动 reward、不动 advantage 公式、不动组内相对比较机制，改的是「花多少算力、在什么精度上、采多少候选」这三个系统量**</mark>。论文对这个协同关系的表述是 <mark class="hl-trick">**"integrates the algorithmic mechanisms of rollout scaling with the system-level throughput gains of NVFP4"**</mark>：<mark class="hl-key">**因为极端样本占比小，量化误差被限制在少量样本上，而这少量样本还会用 BF16 重跑一遍**</mark>。

::: warning 引用前必须记住的三个限制
<mark class="hl-trick">**一、这套方法高度依赖 Blackwell 世代硬件**</mark>——论文实测 8× NVIDIA B200，NVFP4 稠密算力约为 BF16 的 4× TFLOPs，NVFP4 采用 16 元素 / E4M3 scale 分组（OCP MXFP4 是 32 / E8M0）。<mark class="hl-key">**在 Ampere / Ada 上 NVFP4 路径未必有同等吞吐，论文完全没讨论跨硬件可移植性**</mark>。<mark class="hl-trick">**二、$K=12$ 不是普适最优**</mark>，见上文命中率衰减。<mark class="hl-trick">**三、加速有三种口径，且以 Table 5 为准**</mark>：rollout 段 `1.41–2.41×`、单次 iteration 端到端 `1.25–1.62×`、收敛 `1.91–4.64×`。<mark class="hl-key">**Figure 3(a) 把 Naïve Quant 的 184s 误标成了 Sol-RL，会让人把加速算高一点**</mark>。
:::

## 2. 为什么 Rollout Scaling 有效

Sol-RL 的出发点是：在 group-based diffusion RL 中，**扩大同一个 prompt 的 rollout 数量 $N$，即使真正参与训练的样本数 $K$ 不变，模型性能仍然会提升**。原因不是「训练数据更多了」，而是 <mark class="hl-key">**更大的 rollout pool 更容易找到 reward 分布两端的极端样本**</mark>。例如只采 24 张时，最高和最低 reward 可能差距不大；采到 96 张以后，更容易同时找到明显更好的正样本和明显更差的负样本，从而形成更强的 relative learning signal。

这也是为什么 selective training 往往优先保留 **Top-K 和 Bottom-K**。在 GRPO 这类 group-relative optimization 中，高 reward 样本对应较大的正 advantage，低 reward 样本对应较大的负 advantage，而<mark class="hl-key">**接近 group 平均 reward 的样本 advantage 接近 0，训练价值相对较低**</mark>——论文原话是其他样本 `provide limited gradient due to the near-zero advantages`。因此真正昂贵的 backward 没必要覆盖所有 rollout，只需集中在最 informative 的两端样本上。

<mark class="hl-trick">**Sol-RL 由此把 rollout 数量和训练数量彻底分开**</mark>：$N$ 控制 exploration space，$K$ 控制 optimization cost，目标是做到

$$
\boxed{
N_{\rm explore} \gg K_{\rm train}
}
$$

也就是「大量探索、少量训练」。所以 Rollout Scaling 的核心并不是把 batch 做大，而是：

$$
\boxed{
\text{Larger Rollout Pool}
\rightarrow
\text{More Extreme Positive/Negative Samples}
\rightarrow
\text{Stronger Relative Advantage Signal}
}
$$

<mark class="hl-trick">**这里有一个必须记清的点：扩大的不是训练数据量，而是 advantage 的对比度**</mark>。在 GRPO 里 advantage 是组内相对的，中间那些 reward 接近均值的样本贡献的梯度接近零——<mark class="hl-key">**扩池子扩的是「两端更极端」的样本，不是「样本总数」**</mark>。也正因如此，<mark class="hl-trick">**这个「只训极端样本」的 selective training 框架是引用 Xue et al. [7]（DanceGRPO）已有的，Sol-RL 的算法前提是别人的**</mark>，<mark class="hl-key">**它的原创贡献在 NVFP4 与硬件吞吐的协同设计**</mark>。

但这直接带来一个问题：<mark class="hl-trick">**如果 96 个 candidate 全都用 BF16 完整 rollout，而最后只训练其中 24 个，那么剩下的大量高精度 rollout 计算都浪费了**</mark>。Sol-RL 后面的核心方法正是解决这个问题——**能不能用更便宜的 FP4 先做大规模探索，只把最值得训练的 seed 再用 BF16 重生成**。

## 3. 为什么 FP4 不能直接训练，但可以用于筛选

Sol-RL 这里最关键的观察是：<mark class="hl-key">**FP4 rollout 有两种完全不同的使用要求——作为训练 target，要求非常高；作为候选排序 proxy，要求其实低很多。**</mark>把这两条要求拆开，就同时解释了「为什么量化 rollout 是危险的」和「为什么量化 rollout 又是有用的」。

**为什么直接拿来训练会崩。** Sol-RL 自己用的 objective 是 <mark class="hl-key">**DiffusionNFT**</mark>，论文明确写 `The policy is optimized using the DiffusionNFT [8] objective based on the 24 high-fidelity samples`。<mark class="hl-trick">**而这类 forward-process 算法（含 AWM、DiffusionNFT）是基于 denoising score matching loss 的**</mark>——<mark class="hl-key">**它把 rollout sample 直接当作回归目标**</mark>。于是低比特量化带来的数值误差会直接进入监督信号：<mark class="hl-trick">**`the numerical noise forces the high-precision policy to mimic distorted, low-fidelity semantics`**</mark>，即让 BF16 policy 去拟合一个被量化扰动过的生成分布。

论文点出两个使问题恶化的因素：<mark class="hl-key">**一是 off-policy gap**</mark>——量化 policy 采出的 trajectory 相对高精度 target policy 存在固有的分布偏移，会扰乱精细的 policy 更新；<mark class="hl-key">**二是扩散的状态空间是连续的**</mark>，这会进一步放大这种 degradation。结论很狠：<mark class="hl-trick">**这种朴素替换会「inherently cap the achievable alignment quality, finally neutralizing the benefits of rollout scaling」**</mark>——<mark class="hl-key">**也就是说它不只是变差，而是让 rollout scaling 的收益归零**</mark>。

![Qwen-Image-2.0 之外的 Sol-RL Fig.3：NVFP4 rollout 的陷阱与潜力，三 panel。(a) Iteration Time Breakdown：堆叠柱状图，横轴标签格式为 $K$-in-$N(P)$（从 $P$ 精度生成的 $N$ 个 rollout 中选 $K$ 个训练），绿色为 Rollout Time、灰色为 Train Time。`24-in-24 (BF16)` = 113s + 240s；`24-in-96 (BF16)` = 451s + 240s，旁注 `rollout ×4`；`24-in-96 (FP4)` = 184s + 240s，旁注 `speed ×2.45`。**三根柱子的灰色 Train Time 都是 240s，完全不变**。(b) Training Performance Comparison：HPSv2 随 Training Steps 曲线，绿线 `Train with BF16 Rollout` 从约 0.355 单调升到 200 步的约 0.3705；红线 `Train with FP4 Rollout` 起点约 0.337、前 30 步先跌到约 0.332（不稳定），随后爬升但在 180 步达峰约 0.3595 后**回落**到 200 步的约 0.357。(c) Proxy Ranking Reliability：条件概率密度热力图，横轴 BF16 Ranking Percentile、纵轴 NVFP4 Ranking Percentile，虚线为 `ideal ranking` 对角线，右侧色条为 Frequency Density 0.0–1.0；左下角绿框标 `top-k`、右上角绿框标 `bottom-k`，概率密度沿对角线集中。](/solrl-fig3-pitfalls-and-potential.png)

<mark class="hl-key">**Figure 3(b) 是这一节最直接的实证**</mark>：绿线（BF16 rollout 训练）全程单调上升；红线（FP4 rollout 训练）起点更低、<mark class="hl-trick">**早期先出现一次下跌（不稳定）**</mark>、后期爬升但<mark class="hl-key">**在 180 步左右达峰后反而回落**</mark>。<mark class="hl-trick">**「先崩一下、再涨一段、最后退化」这个形状，比单纯「效果差」更能说明量化样本污染了回归目标**</mark>——因为污染的监督会让策略在一个错误的方向上被短暂推着走。

**为什么排序就够用。** 论文的 key observation 建立在 <mark class="hl-key">**ODE-style diffusion sampling 的确定性**</mark> 上：给定相同 initial noise，大尺度的语义布局与结构结果**从根本上由这个 seed 决定**，而这又影响该样本的 reward 水平。NVFP4 会改变最终像素与局部纹理，但通常能保住这个 semantic structure。

![Sol-RL Fig.6：NVFP4（上排）与 BF16（下排）在相同 seed 下的 rollout 对比，三组 prompt。(1) 红色谷仓门口站着一头鹿的插画风画面；(2) `Wish you were Here` 手写体复古海岸路牌；(3) 祖母与孙女在厨房一起揉面做面包的写实照片。**三组的构图、主体、镜头视角、文字内容与整体语义完全一致**，差异集中在植被笔触、海浪泡沫的刻画、面粉与面包的纹理、以及色调与对比度等局部 photometric 细节上。](/solrl-fig6-nvfp4-vs-bf16-rollouts.png)

<mark class="hl-key">**Figure 6 把上面那句话变成了可以一眼看出的事实**</mark>：<mark class="hl-trick">**三组对比里构图、主体、文字、色调布局全都对得上，差别只在植被笔触、海浪泡沫、面粉纹理这类局部细节**</mark>。<mark class="hl-key">**这正是「语义结构由 seed 决定、量化只扰动局部」的直观证据**</mark>。

Table 4 给出了对应的定量版本（三个基模，BF16 vs NVFP4）：

| Base Model | IS BF16 | IS NVFP4 | CLIP BF16 | CLIP NVFP4 |
| :--- | ---: | ---: | ---: | ---: |
| FLUX.1 | 16.84 | **17.85** | 27.44 | 27.10 |
| SANA | 16.02 | 15.94 | **29.53** | **29.43** |
| SD3.5-Large | 16.42 | **17.60** | 28.37 | 28.34 |

<mark class="hl-trick">**CLIPScore 几乎完全一致（最大差 0.34），说明语义质量确实被保住了。但有一个反直觉的点必须记：IS 在 FLUX.1 与 SD3.5-L 上反而是 NVFP4 更高（+1.01 / +1.18）。**</mark><mark class="hl-key">**所以不要把「NVFP4 与 BF16 语义接近」理解成「NVFP4 在所有指标上都略低」**</mark>——<mark class="hl-trick">**IS 并未随量化下降，论文也没有解释这个反常现象**</mark>。

**因此 Sol-RL 不要求**

$$
I_i^{\rm FP4}\approx I_i^{\rm BF16},
$$

**只需要满足更弱的条件**

$$
\boxed{
\operatorname{Rank}\!\big(R(I_i^{\rm FP4})\big)
\approx
\operatorname{Rank}\!\big(R(I_i^{\rm BF16})\big)
}
$$

也就是说，同一个 seed 如果在 FP4 下属于「特别好的那批」，重新用 BF16 生成后大概率仍属于比较好的那批；特别差的 seed 同理。

Table 8 是这条的完整证据，四个 reward 分别统计：

| Reward | $\tau$ | $\rho$ | Top/Btm 4 | Top/Btm 8 | Top/Btm 12 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| CLIPScore | <mark class="hl-trick">0.752</mark> | <mark class="hl-trick">0.900</mark> | <mark class="hl-trick">95.7% / 4.5%</mark> | 93.9% / 6.2% | 92.2% / 8.2% |
| HPSv2 | 0.827 | 0.943 | 97.6% / 3.4% | 95.5% / 5.3% | 93.9% / 7.1% |
| ImageReward | 0.807 | 0.932 | 97.2% / 3.9% | 95.1% / 5.9% | 93.4% / 7.6% |
| PickScore | 0.806 | 0.934 | 97.1% / 3.8% | 95.4% / 5.6% | 93.6% / 7.2% |
| <mark class="hl-key">**Overall**</mark> | <mark class="hl-key">**0.798**</mark> | <mark class="hl-key">**0.927**</mark> | <mark class="hl-key">**96.9% / 3.9%**</mark> | 95.0% / 5.7% | 93.3% / 7.5% |

<mark class="hl-key">**⚠️ 这张表里有一个必须单独拎出来的实用结论：CLIPScore 是四个 reward 里排序一致性最差的一个**</mark>——<mark class="hl-trick">**$\tau$ 最低（0.752）、Top-4 命中最低（95.7%）、Bottom-4 误纳最高（4.5%），全项都是最差**</mark>。<mark class="hl-key">**也就是说：如果你用 CLIPScore 当 reward，FP4 筛选的可靠性是这四种方案里最低的；换成 HPSv2 会明显更好（$\tau$ 0.827、Top-4 97.6%）**</mark>。<mark class="hl-trick">**论文自己没有点出这个差异，只是在 Overall 行给了一个平均值**</mark>，<mark class="hl-key">**但选 reward 时这比平均值有用得多**</mark>。

<mark class="hl-key">**另一个趋势同样重要：Top-K 命中率随 $K$ 单调下降**</mark>（4→96.9%、8→95.0%、12→93.3%），Bottom 误纳率同步上升（3.9% → 5.7% → 7.5%）。<mark class="hl-trick">**所以「挑得准」不是免费的，$K$ 越大越难挑准**</mark>——<mark class="hl-key">**$K=12$ 是一个需要自己消融的取舍点，不是照抄参数**</mark>。

<mark class="hl-key">**Figure 3(c) 的读法**</mark>：它是<mark class="hl-trick">**一个条件概率密度图**</mark>——给定样本的真实 BF16 排名百分位 $x$，看它 NVFP4 proxy 排名的分布（即在 $x$ 处切一刀看纵向密度）。<mark class="hl-key">**密度沿对角线集中，说明排序被保住了；而论文特别强调质心偏向左下 `top-k` 与右上 `bottom-k` 两个绿框**</mark>——<mark class="hl-trick">**也就是最需要准确的那两个象限保真度最好，中间区域反而可以马虎**</mark>。<mark class="hl-key">**这正好解释了为什么 $K=4$ 时命中率（96.9%）远高于 $K=12$（93.3%）**</mark>。

**这套机制落到操作上就是：保存 seed，不保存图像。**

$$
z_1,\dots,z_{96}
\;\longrightarrow\;
I^{\rm FP4}_1,\dots,I^{\rm FP4}_{96}
\;\longrightarrow\;
R_1,\dots,R_{96}
$$

只保留 <mark class="hl-key">**Top-12 seeds + Bottom-12 seeds**</mark>，<mark class="hl-trick">**随后把 FP4 图像本身全部丢掉**</mark>，只保存这 24 个 **initial noises**，再从这些相同 seed 出发用 BF16 policy 完整重生成真正的训练样本。<mark class="hl-key">**这样既利用 FP4 找到了高价值区域，又完全避免让 policy 拟合 FP4 的量化误差**</mark>。

<mark class="hl-key">**所以这一块最值得记的一句话是**</mark>：

$$
\boxed{
\text{FP4 不够准确到可以作为 Training Target，}
\quad
\text{但足够准确到可以作为 Ranking Proxy。}
}
$$

<mark class="hl-trick">**Sol-RL 真正聪明的地方就是把这两个需求拆开了**</mark>：<mark class="hl-key">**探索只需要「排序正确」，训练才需要「样本准确」**</mark>。于是前者交给 FP4，后者继续保留 BF16。

::: warning 这一节的三个可操作结论
<mark class="hl-trick">**一、不要把 Overall 平均值当成你的 reward 的表现**</mark>——<mark class="hl-key">**CLIPScore 的排序保真度明显低于其他三个**</mark>，选 reward 时应查对应行而不是查 Overall。

<mark class="hl-trick">**二、$K$ 必须自己消融**</mark>。命中率随 $K$ 单调下降，$K=4$ 到 $K=12$ 差 3.6 个百分点；<mark class="hl-key">**如果你的 reward 噪声更大，这个差距会更糟**</mark>。

<mark class="hl-trick">**三、这套论证依赖「确定性 ODE 采样 + seed 决定粗结构」**</mark>。<mark class="hl-key">**如果换成 SDE 采样器、或 seed 之外还有明显随机性来源，那么「seed 决定语义结构」这个前提就不成立，排序保真度需要重新测**</mark>。<mark class="hl-trick">**论文完全没讨论这个边界**</mark>。
:::

## 4. Sol-RL 完整训练 Pipeline

Sol-RL 的一次 iteration 可以理解成 <mark class="hl-key">**两次 rollout，但两次 rollout 的目的完全不同**</mark>：第一阶段只负责**便宜地搜索哪些 seed 值得训练**，第二阶段才真正生成高质量 trajectory 做 policy update。

![Sol-RL Fig.2：两阶段解耦 pipeline。上半部三栏 —— **(a) FP4 Exploration**：四行候选，从噪声图生成马匹图像并标出 reward，红框 0.09（最差，✓ 选中）与绿框 0.98（最好，✓ 选中），灰框 0.35 / 0.56（✗ 淘汰）；**(b) BF16 Re-generation**：标注 `regeneration from selected noise seeds`，只有被选中的红/绿两行被重新生成出干净图像，淘汰的两行显示为带斜杠的空框；**(c) GRPO Training**：标注 `train on most contrastive samples (eg. 24 in 96 samples)`，右侧图例把样本分成 `Best-K Samples`（绿框）、`Worst-K Samples`（红框）、`Other Samples`（灰框），下方写出损失 $\mathcal{L}_{\rm GRPO}(\theta)=\frac{1}{N}\sum_{i\in\mathcal{S}_{\rm contrast}}\mathcal{J}^{(i)}_{\rm clip}(\theta)+\frac{1}{N}\sum_{j\in\mathcal{S}_{\rm norm}}\mathcal{J}^{(j)}_{\rm clip}(\theta)$。左侧竖排流程为 (a) FP4 Exploration → (b) BF16 Re-generation → (c) GRPO Training → Policy Update & Re-quant → 回流到 (a)。下半部三条横向耗时条：**Naïve Scaling** = 10 denoise steps / 96 samples / BF16 Precision Rollout **451s** / HPSv2 **0.370**；**Naïve Quant** = 10 steps / 96 samples / FP4 Naïve Quantized Rollout **184s** / HPSv2 **0.354**；**Sol-RL** = 6 steps / 96 samples / FP4 Explore **125s** + 10 steps / 24 samples / BF16 Re-gen **62s**，旁注 **2% overhead** 与 **2.4× speedup** / HPSv2 **0.369**。](/solrl-fig2-two-stage-pipeline.png)

**Stage 1（Accelerated Exploration at Scale via FP4）。** 论文的做法是采样 $N$ 个**互相独立**的 initial noises $\{z^{(i)}\}_{i=1}^{N}$（标准配置 $N=96$），用一个 <mark class="hl-key">**NVFP4 model ODE solver**</mark> 生成样本，并且<mark class="hl-trick">**刻意减少 inference steps**</mark>（标准是 6 步），快速算出对应的 <mark class="hl-key">**proxy rewards $\{\tilde R_i\}$**</mark>——注意论文对它用的是带波浪号的 $\tilde R$，本身就表明「这是估计值而非真值」。然后按 proxy reward 排序，隔离出 $K$ 个高对比度 seed。

**Stage 2（High-Fidelity Regeneration and Policy Update）。** 被选中的 $K$ 个 seed（标准 $K=24$）用**原始高精度 BF16 diffusion loop** 重新生成。这里有一个比「用 BF16」更强的表述，值得单独记：

> <mark class="hl-trick">**By completely shielding the underlying vector field $v_\theta$ from low-precision quantization, this phase allows the ODE solver to reliably reconstruct high-fidelity samples $x_0$**</mark>

<mark class="hl-key">**也就是说不只是采样器用 BF16，而是把向量场本身完全隔离在低精度量化之外**</mark>，然后 policy network **只在这 $K$ 个高保真样本上做标准梯度优化**。

**每轮都要重量化，这是最容易被误解的一点。** FP4 exploration policy 和 BF16 training policy <mark class="hl-key">**并不是两个长期独立训练的模型**</mark>。BF16 policy 更新之后，其权重会被<mark class="hl-trick">**重新量化回 NVFP4**</mark>，供下一轮 exploration 使用：

$$
\theta_t^{\rm BF16}
\;\longrightarrow\;
Q_{\rm FP4}(\theta_t)
\;\longrightarrow\;
\text{Cheap Explore}
\;\longrightarrow\;
\text{Select Seeds}
\;\longrightarrow\;
\theta_t^{\rm BF16}\text{ Regenerate}
\;\longrightarrow\;
\text{RL Update}
\;\longrightarrow\;
\theta_{t+1}^{\rm BF16}
$$

$$
\theta_{t+1}^{\rm BF16}\;\longrightarrow\;Q_{\rm FP4}(\theta_{t+1})\;\longrightarrow\;\cdots
$$

<mark class="hl-key">**所以 FP4 proxy 是持续跟着当前 policy 更新的，不是拿一个很旧的量化模型一直帮新 policy 挑样本**</mark>。附录给了工程细节：<mark class="hl-trick">**权重先经 NVIDIA Transformer Engine 量化成 NVFP4，部署到一个「预编译」推理引擎上；每轮梯度更新后的新权重 in-place 重新量化并拷回这个已编译引擎，`without recompilation`**</mark>——<mark class="hl-key">**这就是 Figure 2 里那个 `2% overhead` 的来源**</mark>。

**一个 iteration 的真实开销不是「96 FP4 + 96 BF16」。** 这是 $N$ 个便宜的加 $K$ 个贵的：

$$
\boxed{
96\times\text{Cheap FP4}\;+\;24\times\text{Expensive BF16}
}
$$

而 naive rollout scaling 相当于 $96\times\text{Expensive BF16}$。<mark class="hl-key">**所以 Sol-RL 节省计算的真正来源非常直观：72 个最终本来就不会参与 gradient update 的 candidate，不再浪费 BF16 完整 rollout。FP4 只负责快速淘汰它们。**</mark>

**最后要避免一个概念混淆：Sol-RL 本身不是新的 RL loss。** <mark class="hl-trick">**它可以建立在已有的 diffusion RL optimization 之上**</mark>，论文标准配置用的是 DiffusionNFT（见 §3）；<mark class="hl-key">**Sol-RL 改的是 rollout / sample-selection infrastructure**</mark>：

$$
\boxed{
\text{它解决「哪些样本值得花高精度训练成本」，而不是重新发明 policy objective。}
}
$$

<mark class="hl-key">**这也是它工业价值高的原因：理论上的 RL 方法可以继续换，但「cheap exploration + selective high-precision training」这个结构可以独立复用。**</mark>

::: warning 这一节有一个我没能从论文里确认的关键歧义
<mark class="hl-trick">**Figure 2(c) 画出的损失有两项**</mark>

$$
\frac{1}{N}\sum_{i\in\mathcal{S}_{\rm contrast}}\mathcal{J}^{(i)}_{\rm clip}(\theta)
\;+\;
\frac{1}{N}\sum_{j\in\mathcal{S}_{\rm norm}}\mathcal{J}^{(j)}_{\rm clip}(\theta)
$$

<mark class="hl-key">**第二项是对「非选中集合」$\mathcal{S}_{\rm norm}$ 的求和**</mark>，<mark class="hl-trick">**与「只用最对比样本训练」的叙述并不完全一致**</mark>——<mark class="hl-key">**如果被淘汰的 72 个样本仍然以 $1/N$ 权重进入损失，那么「其余样本被丢弃」这个说法就需要修正**</mark>。

<mark class="hl-trick">**论文正文没有解释 $\mathcal{S}_{\rm norm}$ 项的作用**</mark>（可能用于归一化统计、可能对应 DiffusionNFT 的某种加权），<mark class="hl-key">**而且 Figure 2(c) 标注的是 `GRPO Training` 而 §4.1 明确说实际用的是 DiffusionNFT objective，两者未对齐**</mark>。<mark class="hl-trick">**这一项需要查官方实现才能确认，笔记里先按「未确认」保留**</mark>。
:::

## 5. 实验、消融与最终 Recipe

这篇的实验真正要验证的只有三件事：更大的 rollout pool 有没有持续收益；FP4 探索要跑多少步才排得准；以及这套两阶段方案是真省钱，还是只是把计算从一个地方搬到另一个地方。

**① rollout scaling 确实有效，而且收益来自 exploration space 而不是训练 batch。** 作者固定真正参与训练的样本为 $K=24$，只改探索池 $N$：

| Exploration Pool $N$ | HPSv2 |
| ---: | ---: |
| 24 | 0.3569 |
| 48 | 0.3622 |
| 72 | 0.3663 |
| 96 | <mark class="hl-key">**0.3686**</mark> |

<mark class="hl-key">**真正 backward 的仍然只有 24 个样本，性能却单调上升**</mark>——<mark class="hl-trick">**这就是 §2 那件事的直接实验证据**</mark>。<mark class="hl-key">**但在作者测试的 $N\le96$ 范围内还没有看到饱和**</mark>，<mark class="hl-trick">**所以 $N=96$ 是「还没饱和的最后一个点」，不是「已找到的最优值」**</mark>。

**② FP4 探索不需要跑完整的步数。** Table 2 的 HPSv2：$2$ 步 0.3587、$4$ 步 0.3650、<mark class="hl-key">**$6$ 步 0.3686**</mark>、$8$ 步 0.3659——<mark class="hl-trick">**6 步之后饱和，8 步甚至略降**</mark>。所以标准 recipe 是：

$$
\boxed{
96\times6\text{-step FP4}
\;\longrightarrow\;
24\times10\text{-step BF16}
\;\longrightarrow\;
\text{DiffusionNFT update}
}
$$

<mark class="hl-key">**这个结果的意义不是「6 是神奇最优值」，而是证明了一件更普适的事**</mark>：

$$
\boxed{
\text{Exploration sample 不需要达到 final-generation quality}
}
$$

<mark class="hl-key">**它只需要达到 Enough Quality for Reliable Ranking**</mark>。<mark class="hl-trick">**步数太少时图像连基本 semantic layout 都还没形成，排序不可靠；到某个程度后主体与构图已足够用于 Top/Bottom 判定，再往上堆细节对筛 seed 帮助很小**</mark>。

**③ 三个 speedup 口径必须分开，这是最容易混淆的地方。** Table 5 是权威来源：

| Base Model | Rollout Naive | Rollout Ours | 加速 | E2E Naive | E2E Ours | 加速 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLUX.1 | 184s | 79s | <mark class="hl-key">**2.33×**</mark> | 274s | 169s | <mark class="hl-key">**1.62×**</mark> |
| SD3.5-Large | 451s | 187s | <mark class="hl-key">**2.41×**</mark> | 691s | 427s | <mark class="hl-key">**1.61×**</mark> |
| SANA | 65s | 46s | <mark class="hl-trick">1.41×</mark> | 95s | 76s | <mark class="hl-trick">1.25×</mark> |

$$
\boxed{2.4\times \approx \text{Rollout Forward 加速}}
\qquad
\boxed{1.6\times \approx \text{单次完整 RL Iteration 加速}}
\qquad
\boxed{4.64\times \approx \text{达到相同性能所需时间的加速}}
$$

<mark class="hl-trick">**端到端拿不到 2.4×，是因为一次完整 iteration 还包含 BF16 regeneration、backward、optimizer step，而这三段都没被加速**</mark>。<mark class="hl-key">**所以拿 2.4× 当「训练快多少」去汇报是会夸大的**</mark>。

![Sol-RL Fig.1（右栏）：ImageReward 训练曲线，横轴 GPU Hours、纵轴 ImageReward Score，绿线 Sol-RL、灰线 DiffusionNFT，红虚线标出 DiffusionNFT 的最终水平。三条曲线自上而下为 **SANA speedup 2.42×**、**FLUX.1 speedup 3.00×**、**SD3.5 Large speedup 4.64×**。三条曲线形状一致：Sol-RL 用明显更少的 GPU Hours 追上 DiffusionNFT 的终值，并最终停在更高的天花板。](/solrl-fig1-convergence-curves.png)

![Sol-RL Fig.4：跨基模的对齐性能，3×3 网格。行 = SD3.5 Large / SANA / FLUX.1，列 = CLIPScore / HPSv2 / PickScore，横轴 GPU Hours，绿线 Sol-RL、灰线 DiffusionNFT，每格左上角红字标注该格的加速倍数：**SD3.5 Large** CLIPScore 2.7× / HPSv2 3.0× / PickScore 2.0×；**SANA** CLIPScore 2.7× / HPSv2 3.6× / PickScore 2.0×；**FLUX.1** CLIPScore 1.9× / HPSv2 3.9× / PickScore 3.3×。](/solrl-fig4-alignment-perf.png)

<mark class="hl-key">**⚠️ 这里必须补一个论文没有明说的观察：headline 的 `4.64×` 只出现在 ImageReward 曲线上（即 SD3.5-Large 那一格）。**</mark>在 Figure 4 覆盖的三个指标上，加速区间是 <mark class="hl-key">**1.9× – 3.9×**</mark>，明显低于 4.64×。<mark class="hl-trick">**论文正文写的是「surpasses the baseline's equivalent performance by 1.91× to 4.64×」，这个区间横跨两个 figure**</mark>，<mark class="hl-key">**只报上限会明显夸大**</mark>。

**④ 质量不是「更快但略掉点」——这一点要靠 Table 6 澄清。** Sol-RL 对 BF16 naive scaling 的 HPSv2 差距是：

| Base Model | Naive | Sol-RL (Δ) |
| :--- | ---: | ---: |
| FLUX.1 | 0.3699 | 0.3688（<mark class="hl-trick">−0.29%</mark>） |
| SD3.5-Large | 0.3803 | 0.3762（<mark class="hl-trick">−1.08%</mark>） |
| SANA | 0.3682 | <mark class="hl-key">0.3686（+0.11%）</mark> |

<mark class="hl-key">**所以相对 naive scaling，Sol-RL 基本是「打平」，不是「更好」**</mark>，论文自己的措辞也是 `maintains on-par results with a marginal gap of at most 1%`。<mark class="hl-key">**它的收益是「同等质量、更便宜」，而不是「更便宜、还更好」**</mark>——<mark class="hl-trick">**这个区分在汇报时不能含糊**</mark>。

**⑤ 那 Table 1 里相对其他方法的优势是哪来的？** Table 1 在 FLUX.1 上、**相同 GPU-hour 预算**下比较：

| Method | ImageReward | CLIPScore | PickScore | HPSv2 |
| :--- | ---: | ---: | ---: | ---: |
| Base (w/o CFG) | 0.455 | 0.2630 | 0.8096 | 0.2566 |
| DanceGRPO | 1.4937 | 0.2898 | 0.8807 | 0.3552 |
| FlowGRPO | 1.5331 | 0.2884 | 0.8743 | 0.3501 |
| AWM | 1.6693 | <mark class="hl-trick">0.3039</mark> | 0.8842 | <mark class="hl-trick">0.3664</mark> |
| DiffusionNFT | 1.6707 | 0.2991 | <mark class="hl-trick">0.8852</mark> | 0.3613 |
| <mark class="hl-key">**Sol-RL (Ours)**</mark> | <mark class="hl-key">**1.7636**</mark> | <mark class="hl-key">**0.3089**</mark> | <mark class="hl-key">**0.8932**</mark> | <mark class="hl-key">**0.3688**</mark> |

<mark class="hl-key">**Sol-RL 四项全部最优，但要注意两点论文没强调的细节：**</mark>

- <mark class="hl-trick">**AWM 是比 DiffusionNFT 更强的 baseline**</mark>——<mark class="hl-key">**它在 CLIPScore（0.3039 vs 0.2991）与 HPSv2（0.3664 vs 0.3613）上都高于 DiffusionNFT**</mark>，而论文的超参是对齐 DiffusionNFT 的。<mark class="hl-trick">**在 HPSv2 上 Sol-RL 相对最强 baseline（AWM）的优势只有 +0.0024**</mark>，<mark class="hl-key">**远小于表观上的 +0.0075（对 DiffusionNFT）**</mark>。
- <mark class="hl-key">**真正起作用的是 rollout scaling + selective training，FP4 只是让这个更大的探索池付得起**</mark>——<mark class="hl-trick">**因为 Table 6 显示 Sol-RL 与 naive scaling 基本打平，两者的区别只有探索池大小**</mark>。

**⑥ 最后提炼进 Image RL Recipe 的两条。**

<mark class="hl-key">**第一条：Exploration Budget 与 Training Budget 应该解耦。**</mark> 以后不要默认「生成多少就训练多少」：

$$
\boxed{
\text{大量廉价探索}
\;\longrightarrow\;
\text{筛出高信息量样本}
\;\longrightarrow\;
\text{少量高质量训练}
}
$$

<mark class="hl-key">**第二条（比「FP4」这个具体技术普适得多）：探索阶段只需要满足决策所需的最低精度。**</mark>

$$
\boxed{
\text{探索的目的只是判断「哪个 candidate 值得训练」，它不需要达到训练样本所要求的 fidelity}
}
$$

<mark class="hl-trick">**可以用更低 precision、更少 denoising steps，甚至未来用其他 surrogate model 来做。**</mark>

**所以 Sol-RL 最终压成一句放进总 Recipe：**

$$
\boxed{
\textbf{Cheap Explore, Expensive Learn：把算力花在寻找高价值样本，而不是把所有 rollout 都高精度算到底。}
}$$

::: warning 引用这篇时必须一起带走的四个限定
<mark class="hl-trick">**一、这是工程 / 系统贡献，不是算法贡献**</mark>——algorithm 是 DiffusionNFT，selective training 来自 DanceGRPO。

<mark class="hl-trick">**二、加速上限依赖 Blackwell 硬件**</mark>，且 <mark class="hl-key">**SANA 只有 1.41× / 1.25×**</mark>——<mark class="hl-trick">**说明在算力较弱或模型较小的场景，收益会显著缩水**</mark>。

<mark class="hl-trick">**三、$N=96$ 只是「没饱和的最后一个测试点」**</mark>，论文没给 $N>96$ 的数据，<mark class="hl-key">**也没有回答 rollout scaling 最终会在哪里饱和**</mark>。

<mark class="hl-trick">**四、Figure 3(a) 把 Naïve Quant 的耗时标成了 Sol-RL**</mark>，<mark class="hl-key">**算加速请一律以 Table 5 为准**</mark>。
:::

---

## 附：已核实的关键数字

| 项目 | 数值 |
| :--- | :--- |
| Exploration pool $N$ | 96 / prompt |
| Selected samples | top-12 + bottom-12 = 24（`24-in-96`） |
| FP4 探索 / BF16 rollout denoise steps | 6 / 10 |
| FP4 吞吐 | up to **4×** TFLOPs of BF16 |
| 重量化开销 | **2%**（in-place 拷回已编译引擎，无需 recompile） |
| Rollout 加速（Table 5） | FLUX.1 **2.33×**、SD3.5-L **2.41×**、SANA **1.41×** |
| 端到端 iteration 加速（Table 5） | FLUX.1 **1.62×**、SD3.5-L **1.61×**、SANA **1.25×** |
| 收敛加速 time-to-target | **1.91–4.64×**（ImageReward 曲线：SANA 2.42 / FLUX.1 3.00 / SD3.5-L **4.64**） |
| Kendall $\tau$ / Spearman $\rho$（Overall） | 0.798 / 0.927（CLIPScore 单项最差：0.752 / 0.900） |
| Top-K match（Overall） | 4→96.9%、8→95.0%、12→93.3% |
| FP4 vs BF16 语义指标（Table 4） | IS 16.84→17.85 / 16.02→15.94 / 16.42→17.60；CLIP 27.44→27.10 / 29.53→29.43 / 28.37→28.34 |
| FP4 探索步数 $T$（Table 2） | 2→0.3587、4→0.3650、**6→0.3686**、8→0.3659（6 步后饱和甚至回落） |
| 探索池 $N$（Table 3） | 24→0.3569、48→0.3622、72→0.3663、**96→0.3686**（到 96 仍无饱和） |
| 硬件 / 基模 | 8× B200，NVFP4 backend = Transformer Engine；SANA / FLUX.1 / SD3.5-L |
| KL 系数 / old-model decay | $\beta_{\rm kl}=1.0$ / 0.9 |
| 对 naive scaling 的 HPSv2 差距（Table 6） | FLUX.1 −0.29%、SD3.5-L −1.08%、SANA +0.11% |
| 最强 baseline（Table 1, FLUX.1） | AWM 的 CLIPScore 0.3039 / HPSv2 0.3664 优于 DiffusionNFT；Sol-RL 在 HPSv2 上仅比 AWM 高 0.0024 |

<mark class="hl-trick">**尚未核实 / 未解决**</mark>：
- **Figure 2(c) 损失里 $\mathcal{S}_{norm}$ 项的作用**——被淘汰样本是否仍进梯度、系数为何（§4 末尾的 warning）
- **BF16 重跑后 reward 是否重算**，还是沿用 FP4 阶段的 $\tilde R$ 排序（§3 已标，论文始终用带波浪号的 proxy reward，未说明）
- **Figure 3(a) 把 Naïve Quant 的 184s 误标为 Sol-RL**（§1 / §5 已标，以 Table 5 为准）
- **Figure 2(c) 标 `GRPO Training` 而正文用 DiffusionNFT**，两者未对齐
- **$N>96$ 的 scaling 行为**未给；附录 B 的完整超参表（Table 7）未逐项抄录
- **Figure 5 / 7 / 8 / 9**（定性对比图）未查看；跨硬件可移植性论文未讨论