# Sol-RL：FP4 Explore, BF16 Train —— 扩散 RL 的高效 Rollout Scaling (NVIDIA)

> **标签**：`Vision` `RL` `GRPO` `Diffusion RL` `FP4` `NVFP4` `Quantization` `Rollout Scaling` `Systems`
> **更新时间**：2026-10-06
> **原文**：`Papers/Sol-RL.pdf`（20 页，NVIDIA / HKU / MIT）
> **标题**：*FP4 Explore, BF16 Train: Diffusion Reinforcement Learning via Efficient Rollout Scaling*
> **arXiv**：[2604.06916](https://arxiv.org/abs/2604.06916) v1（2026-04-08）
> **作者**：Yitong Li, Junsong Chen, Shuchen Xue, Pengcuo Zeren, Siyuan Fu, Dinghao Yang, Yangyang Tang, Junjie Bai, Ping Luo, Song Han, Enze Xie
> **实验基模**：SANA / FLUX.1 / SD3.5-Large，8× NVIDIA B200
> **精读进度**：§1 总体 ✅ ｜ §2 ⬜ ｜ §3 ⬜ ｜ §4 ⬜ ｜ §5 ⬜
> **相关笔记**：[Qwen-Image-2.0 RLHF 统一对齐](./qwen-image-2-rl.md)（Hybrid CFG 同族思想）｜ [Swift-Image 并行专家 RL](./swift-image-rl.md)（CFG 非对称同构）｜ [算法 × 奖励 × 基模对比](./rl-comparison-2026.md) ｜ [训练 Playbook §4](../training-playbook.md)

---

## 1. 总体

### 1.1 核心问题：rollout scaling 有效，但太贵

<mark class="hl-key">**Sol-RL 要解决的问题非常明确：图像生成 RL 里，rollout 数量越大，越容易从同一个 prompt 下找到高 reward 和低 reward 的极端样本，从而得到更强的 relative learning signal；但大规模 rollout 本身又非常昂贵，逐渐成为训练瓶颈。**</mark>

<mark class="hl-trick">**论文把瓶颈转移这件事说得比一般论文更准**</mark>——因为在选择性训练范式下，<mark class="hl-key">**只有一小撮高对比度样本真正进入梯度更新，其余样本在 rollout 之后直接被丢弃**</mark>：

> <mark class="hl-trick">**Because only a small set of highly contrastive samples is ultimately used for optimization, scaling the candidate pool shifts the training bottleneck from policy optimization to candidate generation.**</mark>

<mark class="hl-trick">**换句话说：卷 massive rollout 会把瓶颈从 backward 搬到 forward，而这个 forward 的绝大部分算力是浪费的。**</mark>

### 1.2 核心思想：把 Exploration 和 Policy Training 彻底拆开

$$
\boxed{
\text{Exploration}
\quad\text{和}\quad
\text{Policy Training}
}
$$

<mark class="hl-key">**普通做法**</mark>如果想从一个 prompt 采 96 个 candidate，<mark class="hl-trick">**可能需要全部用 BF16 完整 rollout，成本很高**</mark>。<mark class="hl-key">**Sol-RL 改成**</mark>：

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

<mark class="hl-key">**论文自己给这个框架取的全称是**</mark>：

> <mark class="hl-trick">**Sol-RL (Speed-of-light RL), a novel FP4-empowered Two-stage Reinforcement Learning framework**</mark>

<mark class="hl-key">**标准设置可以直接记成**</mark>：

$$
\boxed{
96\ \text{个 FP4 candidates}
\;\rightarrow\;
\text{Top-12} + \text{Bottom-12}
\;\rightarrow\;
24\ \text{个 BF16 rollout}
\;\rightarrow\;
\text{RL Update}
}
$$

<mark class="hl-trick">**`top-12 and bottom-12` 是论文原话**，论文另外把这个配置称为 **`24-in-96` rollout setting**；配套的对照组还有 `24-in-24 (BF16)`（不扩展 exploration）与 `24-in-96 (FP4)`（直接用 FP4 训）。</mark>

<mark class="hl-key">**速度收益来自哪里 —— Figure 2 的实测分解**</mark>（同一 prompt 的耗时）：

| 阶段 | 耗时 |
| :--- | --- |
| <mark class="hl-trick">BF16 Precision Rollout（naïve scaling，96 张）</mark> | <mark class="hl-trick">**451s**</mark> |
| <mark class="hl-trick">FP4 Naïve Quantized Rollout（直接量化，96 张）</mark> | <mark class="hl-trick">**184s**</mark> |
| <mark class="hl-trick">FP4 Explore（Sol-RL 的探索阶段）</mark> | <mark class="hl-key">**125s**</mark> |
| <mark class="hl-trick">BF16 Re-gen（只重跑 24 张）</mark> | <mark class="hl-key">**62s**</mark> |

<mark class="hl-key">**所以 Sol-RL 的 pipeline 级加速是 `2.4×`，而额外开销只有 `2%`**</mark>（指把 policy 权重量化回 NVFP4 的开销）。摘要里说的 <mark class="hl-trick">**最高 `4.64×`**</mark> <mark class="hl-key">**指的是「达到同等 reward 水平的 wall-clock 收敛加速」**</mark>，<mark class="hl-trick">**这两个数字口径不同，不要混用**</mark>——Figure 1 给出的三个模型分别是 `2.42×` / `4.64×` / `3.00×`。

### 1.3 FP4 的作用不是生成，而是「挑得准」

<mark class="hl-key">**FP4 阶段的目的不是直接产出训练 trajectory，而只是低成本扩大 exploration space，帮模型找到「最值得训练」的 initial noise / seed。**</mark>

<mark class="hl-trick">**因为低精度量化虽然会影响最终图像细节，但作者发现它对候选样本的**</mark> <mark class="hl-key">**relative reward ranking**</mark> <mark class="hl-trick">**仍然比较可靠，尤其能够较稳定地识别 Top-K 和 Bottom-K。**</mark>

因此 Sol-RL <mark class="hl-key">**并不要求**</mark>

$$
I_i^{\rm FP4}\approx I_i^{\rm BF16}
$$

<mark class="hl-key">**它只要求**</mark>：

$$
\boxed{
\operatorname{Rank}
\left(R(I_i^{\rm FP4})\right)
\approx
\operatorname{Rank}
\left(R(I_i^{\rm BF16})\right)
}
$$

<mark class="hl-key">**一句话概括这个区别**</mark>：

> <mark class="hl-trick">**FP4 不需要生成得足够准，只需要「挑得准」。**</mark>

<mark class="hl-key">**被选中的 seed 随后用 BF16 policy 重新 rollout**</mark>，<mark class="hl-trick">**真正用于 policy gradient update 的仍然是高精度 trajectory，从而避免直接使用 FP4 样本训练带来的量化污染。**</mark>

<mark class="hl-key">**论文给的量化证据（Table 8，CLIPScore reward，NVFP4 proxy vs BF16 真值）**</mark>：

$$
\text{Kendall } \tau = 0.752,
\qquad
\text{Spearman } \rho = 0.900
$$

$$
\text{Top-4 match} = 95.7\% \;(<\text{4\% Bottom-4 error}),
\quad
\text{Top-8} = 93.9\%,
\quad
\text{Top-12} = 92.2\%
$$

<mark class="hl-key">**关键在于：保真度是随 K 增大而下降的**</mark>——<mark class="hl-trick">**Top-4 命中率 95.7%，到 Top-12 就降到 92.2%。这说明「挑得准」不是免费的，K 越大越难挑准，这正是 §5 需要消融的 tradeoff**</mark>。

### 1.4 真正可迁移的不是 FP4，而是这个抽象

<mark class="hl-key">**这篇最有价值的地方不是「把 BF16 换成 FP4」这么简单，而是提出了一种非常通用的思路**</mark>：

$$
\boxed{
\text{Cheap Approximation for Exploration}
+
\text{Accurate Computation for Learning}
}
$$

<mark class="hl-key">**也就是把大量、便宜、近似的计算用于「搜索哪里值得学」，然后把昂贵的高精度计算集中到少量 informative samples 上。**</mark>

<mark class="hl-trick">**这个抽象远超量化本身**</mark>——<mark class="hl-key">**任何「便宜代理足以完成排序 / 筛选，但不足以完成学习」的领域都适用**</mark>：候选检索、prompt 筛选、数据清洗、reward model 的粗筛。

<mark class="hl-key">**而 rollout scaling 的真正价值也不是「产生更多训练数据」，而是**</mark>：

$$
\boxed{
\text{Larger Rollout Pool}
\rightarrow
\text{More Extreme Positive / Negative Samples}
\rightarrow
\text{Stronger Relative Advantage Signal}
}
$$

<mark class="hl-trick">**注意这里的关键：扩大的不是训练数据量，而是 advantage 的对比度。**</mark>在 GRPO 里，<mark class="hl-key">**组内 advantage 是相对的**</mark>，所以<mark class="hl-trick">**中间那些 reward 接近均值的样本贡献的梯度接近零——扩池子扩的是「两端更极端」的样本，不是「样本总数」**</mark>。

### 1.5 定位：不是新 GRPO objective

<mark class="hl-key">**Sol-RL 更准确的定位是**</mark>：

$$
\boxed{
\textbf{Diffusion RL 的 Exploration Efficiency / Rollout Scaling 方法}
}
$$

<mark class="hl-trick">**而不是一个新的 GRPO objective。**</mark><mark class="hl-key">**它不动 reward、不动 advantage 公式、不动 GRPO 的组内相对比较机制**</mark>——<mark class="hl-trick">**它改的是「花多少算力、在什么精度上、采多少候选」这三个工程/系统量**</mark>。

<mark class="hl-key">**论文对这个协同关系的表述值得记**</mark>：

> <mark class="hl-trick">**Sol-RL integrates the algorithmic mechanisms of rollout scaling with the system-level throughput gains of NVFP4. This synergistic algorithm-hardware design...**</mark>

<mark class="hl-trick">**即：算法侧的「只训极端样本」正好与硬件侧的「NVFP4 吞吐」互补——因为极端样本占比小，量化误差带来的损失被限制在少量样本上，而这少量样本还会用 BF16 重跑一遍。**</mark>

### 1.6 为什么 FP4 能在 2026 年成立（硬件前提）

<mark class="hl-key">**这一节放进来是因为它决定了方法的适用边界**</mark>。论文用 NVIDIA Transformer Engine 作为 NVFP4 backend，<mark class="hl-trick">**全部实验在 8× NVIDIA B200 上完成**</mark>。

NVFP4 的格式细节（论文与 OCP MXFP4 的对比）：

$$
\text{NVFP4} : \text{16 elements / E4M3 scale}
\qquad\text{vs.}\qquad
\text{OCP MXFP4} : \text{32 elements / E8M0 scale}
$$

<mark class="hl-key">**硬件收益的来源论文写得很直接**</mark>：

> <mark class="hl-trick">**NVFP4 dense operations deliver up to 4× the TFLOPs of standard BF16 arithmetic.**</mark>

<mark class="hl-trick">**工程上还有两个细节值得注意**</mark>：policy 权重更新后<mark class="hl-key">**会被 in-place 重新量化进 NVFP4 推理引擎，且不需要重新编译**</mark>；探索阶段用 **6 个 denoise steps**，而 BF16 rollout 用 **10 步**。

::: warning §1 就必须说清的三个前提
<mark class="hl-trick">**第一，这套方法高度依赖 Blackwell 世代硬件**</mark>（论文实测 B200）。<mark class="hl-key">**在 Ampere / Ada 上 NVFP4 路径未必有同等吞吐，方法能否照搬取决于你的硬件是否有对应的低精度张量核**</mark>。<mark class="hl-trick">**这一点论文没有讨论跨硬件的可移植性**</mark>。

<mark class="hl-trick">**第二，「top-12 / bottom-12」是这套框架里的具体取值，不是普适最优**</mark>。<mark class="hl-key">**如前所述，Top-12 命中率已降到 92.2%，K 往上调会持续损失挑选精度**</mark>——<mark class="hl-trick">**所以 K 是一个需要自己消融的量，不是照抄参数**</mark>。

<mark class="hl-trick">**第三，Fig. 2 的 2.4× 是 pipeline 级加速，摘要的 4.64× 是收敛加速**</mark>，<mark class="hl-key">**两个数字口径不同，引用时必须带口径**</mark>。
:::

## 2. 为什么 Rollout Scaling 有效 ⬜

<mark class="hl-trick">**待填**</mark>：论文 §3.1 *Promise and Bottleneck of Rollout Scaling*。核心问题：

- GRPO 训练动力学下，为什么 top-K / worst-K 提供的学习信号「more reliable and informative」，而其余样本 `near-zero advantages`
- 引用了 Xue et al. [7]（DanceGRPO）的 selective training framework —— <mark class="hl-key">**Sol-RL 的算法前提是别人的，贡献在 FP4 与硬件协同**</mark>
- rollout scaling 的收益曲线是<mark class="hl-trick">**饱和的还是持续增长的**</mark>

## 3. 为什么 FP4 不能直接训练，但可以用于筛选 ⬜

<mark class="hl-trick">**待填**</mark>：论文 §3.2 *Training Degradation with Direct Quantized Rollouts* + §3.3 *Proxy Reward Ranking via FP4 Exploration*。核心问题：

- 直接把 quantized rollout 当训练目标，为什么会导致 `training degradation`，机理是什么
- 量化误差分别对<mark class="hl-key">**图像质量 / trajectory / reward ranking**</mark>三者的影响是否一致
- Figure 3 的三档对照：`24-in-24 (BF16)` vs `24-in-96 (BF16)` vs `24-in-96 (FP4)`
- Figure 6 的 ranking percentile 散点为什么能证明「排序保真」而非「质量保真」

## 4. Sol-RL 完整训练 Pipeline ⬜

<mark class="hl-trick">**待填**</mark>：论文 §3.4 *FP4-Empowered Two-Stage Framework* + 附录 B。核心问题：

- 两阶段的**算法**描述（seed 如何保存与复用、advantage 如何在 24 张子集上重算）
- 两阶段的**工程**描述（NVFP4 引擎部署、权重量化/反量化、init mode）
- ⚠️ 待核实的关键细节：<mark class="hl-key">**保留的是 initial noise seed，那么同一批 24 张在 BF16 重跑后 reward 会变**</mark>——<mark class="hl-trick">**重算后的 reward 是直接用于 advantage，还是沿用 FP4 阶段的排序？**</mark><mark class="hl-key">**这是这套方法最容易出 bug 的地方**</mark>

## 5. Experiments / Ablation / Recipe ⬜

<mark class="hl-trick">**待填**</mark>：论文 §4。核心问题：

- **Table 1**：FLUX.1 上对比 FlowGRPO / DanceGRPO / AWM / DiffusionNFT 的定量结果
- **Table 3**：exploration pool $N \in \{24, 48, 72, 96\}$ 的 scaling 行为
- **Table 4**：NVFP4 rollout 的定量评估
- **Table 5**：$N=96$ 下的耗时分解（rollout time vs 整体训练时间）
- **Table 7**：训练超参（rollout steps 10、eval steps 40、timestep fraction 0.6、num train timesteps 6、KL $\beta_{\rm kl}=1.0$、old-model decay 0.9）
- **Table 8**：ranking 保真度（Kendall $\tau$ / Spearman $\rho$ / Top-Bottom-K match）
- <mark class="hl-key">**最后只提炼真正值得加入 [Image RL Recipe](../training-playbook.md) 的部分**</mark>

---

## 附：已核实的关键数字（供后续 §2–§5 引用时对照）

| 项目 | 数值 | 出处 |
| :--- | :--- | :--- |
| Exploration pool $N$ | 96 / prompt | §3.4、Fig.2 |
| Selected samples | top-12 + bottom-12 = 24（`24-in-96`） | §3.4 原话 |
| FP4 探索 denoise steps | 6 | Fig.2、Table 7 |
| BF16 rollout steps | 10 | Table 7 |
| FP4 吞吐 | up to **4×** TFLOPs of BF16 | §3.4 |
| Pipeline 加速 | **2.4×**，额外开销 **2%** | Fig.2 |
| 收敛加速 | up to **4.64×**（三模型 2.42 / 4.64 / 3.00×） | Abstract、Fig.1 |
| Kendall $\tau$ / Spearman $\rho$ | 0.752 / 0.900 | Table 8 |
| Top-K match | 4→95.7%、8→93.9%、12→92.2% | Table 8 |
| 硬件 | 8× NVIDIA B200，NVFP4 backend = Transformer Engine | §4.1 |
| 基模 | SANA / FLUX.1 / SD3.5-L | Abstract |
| KL 系数 / old-model decay | $\beta_{\rm kl}=1.0$ / 0.9 | Table 7 |

## 附：本篇尚未核实的内容

- <mark class="hl-trick">**§3.2 的 training degradation 具体机理**</mark>——只读到标题与摘要级描述
- <mark class="hl-trick">**BF16 重跑后 reward 是否重算**</mark>（见 §4 待核实项）
- <mark class="hl-trick">**Table 1 / 3 / 4 / 5 的具体数值**</mark>尚未抽表
- <mark class="hl-trick">**Figure 1–7 全部未查看**</mark>——本文档暂无配图
- <mark class="hl-trick">**跨硬件可移植性**</mark>论文未讨论