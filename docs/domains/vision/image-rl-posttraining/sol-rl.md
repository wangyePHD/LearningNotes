# Sol-RL：FP4 Explore, BF16 Train —— 扩散 RL 的高效 Rollout Scaling (NVIDIA)

> **标签**：`Vision` `RL` `GRPO` `Diffusion RL` `FP4` `NVFP4` `Quantization` `Rollout Scaling` `Systems`
> **更新时间**：2026-10-06
> **原文**：`Papers/Sol-RL.pdf`（20 页，NVIDIA / HKU / MIT）
> **标题**：*FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling*
> **arXiv**：[2604.06916](https://arxiv.org/abs/2604.06916) v1（2026-04-08）
> **作者**：Yitong Li, Junsong Chen, Shuchen Xue, Pengcuo Zeren, Siyuan Fu, Dinghao Yang, Yangyang Tang, Junjie Bai, Ping Luo, Song Han, Enze Xie
> **实验基模**：SANA / FLUX.1 / SD3.5-Large，8× NVIDIA B200
> **精读进度**：§1 ✅ ｜ §2 ✅ ｜ §3 ⬜ ｜ §4 ⬜ ｜ §5 ⬜
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

<mark class="hl-key">**所以 pipeline 级加速是 `2.4×`、额外开销仅 `2%`；而摘要里的 `4.64×` 指的是「达到同等 reward 水平的 wall-clock 收敛加速」**</mark>，Figure 1 给出的三模型分别是 `2.42×` / `4.64×` / `3.00×`。<mark class="hl-trick">**两个口径引用时必须带清楚**</mark>。

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
<mark class="hl-trick">**一、这套方法高度依赖 Blackwell 世代硬件**</mark>——论文实测 8× NVIDIA B200，NVFP4 稠密算力约为 BF16 的 4× TFLOPs，NVFP4 采用 16 元素 / E4M3 scale 分组（OCP MXFP4 是 32 / E8M0）。<mark class="hl-key">**在 Ampere / Ada 上 NVFP4 路径未必有同等吞吐，论文完全没讨论跨硬件可移植性**</mark>。<mark class="hl-trick">**二、$K=12$ 不是普适最优**</mark>，见上文命中率衰减。<mark class="hl-trick">**三、$2.4\times$ 与 $4.64\times$ 口径不同**</mark>，不可混用。
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

## 3. 为什么 FP4 不能直接训练，但可以用于筛选 ⬜

待填（论文 §3.2 + §3.3）。要回答的是：直接拿 quantized rollout 当训练目标为什么会 `training degradation`，机理是什么；量化误差对**图像质量 / trajectory / reward ranking** 三者的影响是否一致；以及 Figure 6 的 ranking percentile 散点为什么能证明「排序保真」而非「质量保真」。

## 4. Sol-RL 完整训练 Pipeline ⬜

待填（论文 §3.4 + 附录 B）。要回答的是：两阶段的算法描述（seed 如何保存复用、advantage 如何在 24 张子集上重算）与工程描述（NVFP4 引擎部署、权重量化/反量化、init mode）。

<mark class="hl-key">**⚠️ 已标记的待核实关键点**</mark>：论文保留的是 <mark class="hl-trick">**initial noise seed**</mark>，所以 BF16 重跑后 reward 必然会变——<mark class="hl-key">**advantage 是用 BF16 重算的 reward，还是沿用 FP4 阶段的排序？这是整套方法最容易实现错的地方**</mark>。

## 5. Experiments / Ablation / Recipe ⬜

待填（论文 §4）。Table 1 是 FLUX.1 上对 FlowGRPO / DanceGRPO / AWM / DiffusionNFT 的定量对比；Table 3 是 exploration pool $N\in\{24,48,72,96\}$ 的 scaling；Table 4 是 NVFP4 rollout 的定量评估；Table 5 是 $N=96$ 的耗时分解；Table 7 是训练超参（rollout 10 步、eval 40 步、timestep fraction 0.6、num train timesteps 6、$\beta_{\rm kl}=1.0$、old-model decay 0.9）。<mark class="hl-key">**最后只提炼真正值得加入 [Image RL Recipe](../training-playbook.md) 的部分**</mark>。

---

## 附：已核实的关键数字

| 项目 | 数值 |
| :--- | :--- |
| Exploration pool $N$ | 96 / prompt |
| Selected samples | top-12 + bottom-12 = 24（`24-in-96`） |
| FP4 探索 / BF16 rollout denoise steps | 6 / 10 |
| FP4 吞吐 | up to **4×** TFLOPs of BF16 |
| Pipeline 加速 / 额外开销 | **2.4×** / **2%** |
| 收敛加速 | up to **4.64×**（三模型 2.42 / 4.64 / 3.00×） |
| Kendall $\tau$ / Spearman $\rho$ | 0.752 / 0.900 |
| Top-K match | 4→95.7%、8→93.9%、12→92.2% |
| 硬件 / 基模 | 8× B200，NVFP4 backend = Transformer Engine；SANA / FLUX.1 / SD3.5-L |
| KL 系数 / old-model decay | $\beta_{\rm kl}=1.0$ / 0.9 |

<mark class="hl-trick">**尚未核实**</mark>：§3.2 的 degradation 机理、BF16 重跑后 reward 是否重算、Table 1/3/4/5 具体数值、Figure 1–7 全部未查看（本文暂无配图）、跨硬件可移植性论文未讨论。