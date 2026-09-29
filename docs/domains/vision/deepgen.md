# 轻量统一多模态模型 DeepGen 1.0 (SCB + 三阶段训练 + MR-GRPO)

> **标签**：`Vision` `Unified Model` `VLM-DiT` `Flow Matching` `RL` `GRPO` `Data-centric`
> **更新时间**：2026-09-29
> **参考来源**：[DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing (arXiv:2602.12205v2)](https://arxiv.org/abs/2602.12205) · [GitHub: DeepGenTeam/DeepGen](https://github.com/DeepGenTeam/DeepGen) · [HuggingFace: DeepGenT](https://huggingface.co/DeepGenT) · [Datasets](https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0)
> **原文**：本地 `Papers/DeepGen.pdf`（21 页，正文 18 页 + 附录 A/B）
> **精读重点**：§3 Training（data train）+ §3.3 RL + §4 Data
> **精读进度**：§4 Data ★ ✅ ｜ §3.1 Alignment Pre-Training ✅ ｜ §3.2 Joint SFT ✅ ｜ **§3.3 MR-GRPO ★ ✅**（Eq. 2–7 全部核对）｜ §5.3.2 RL 消融 ✅ ｜ §2 Architecture ｜ §5.1/§5.2 ｜ §6 Conclusion（笔记随学习逐节增补）

---

<!-- 精读导航：按论文章节序推进，重点节已标注 ★ -->

## 1. Introduction

## 2. Model Architecture

## 3. Training ★

### 3.1 Stage 1: Alignment Pre-Training

<mark class="hl-trick">第一阶段只训练 **SCB connector 和 128 个 learnable think tokens**，其余参数全部冻结，也就是 VLM 和 DiT 都不更新</mark>。可以把这一步理解成：<mark class="hl-trick">Qwen2.5-VL 和 SD3.5-Medium 本身都已经是 pretrained module，但它们的 representation space 并不是天然对齐的，所以先只训练中间桥梁</mark>，让 VLM 的语义、视觉、推理信息能够变成 DiT 能使用的 condition。

这一阶段只用两种基础任务：<mark class="hl-trick">**general text-to-image generation 和 general image editing**</mark>，也就是 §4 里的约 **35M generation image-text pairs + 6.6M editing triplets**。<mark class="hl-key">这里还没有加入 reasoning generation、reasoning editing、text rendering 这些专项任务，所以它本质上是在打"统一 generation/editing 的底座"。</mark>

| 项 | Stage 1 配置 |
| :--- | :--- |
| 训练步数 | <mark class="hl-trick">**200,000 iterations**</mark> |
| 分辨率 | <mark class="hl-trick">固定 **512×512**</mark>，<mark class="hl-trick">**不做 arbitrary resolution**</mark>（Table 9） |
| Learning rate | <mark class="hl-trick">**1×10⁻⁴**</mark> |
| Warm-up | <mark class="hl-trick">正文写 **20,000 steps**</mark> ⚠️ 见下方矛盾 |
| Optimizer / Scheduler | AdamW / cosine |
| Weight decay / Clip | 0.05 / 1.0 |
| Batch size / GPU | 512 / **64×H200** |
| <mark class="hl-trick">可训练参数</mark> | <mark class="hl-trick">**仅 SCB connector**</mark> |

### 3.2 Stage 2: Joint Supervised Fine-Tuning

Stage 2 才是真正的 <mark class="hl-trick">**Joint SFT**</mark>。这时候作者开始扩大可训练范围：<mark class="hl-trick">**DiT 解冻并参与训练，VLM 不直接 full fine-tune，而是通过 LoRA 做轻量更新，SCB connector 继续训练**</mark>。

<mark class="hl-key">这么做的目的很明确——既希望 VLM 能适应 generation/editing 任务，又不希望 joint optimization 把 VLM 原有的 multimodal understanding 和 world knowledge 破坏掉，所以作者选择 LoRA，而不是直接把整个 VLM 全参数打开。</mark>原文措辞是 *"To mitigate potential **degradation of the VLM's multimodal comprehension** during joint optimization, we apply LoRA for **efficient** fine-tuning of the VLM."*

这时数据也从"基础对齐"升级成真正的多任务混训，包括 <mark class="hl-trick">**general generation、general editing、reasoning-based generation、reasoning-based editing、text rendering**</mark>。<mark class="hl-key">也就是说 DeepGen 的 omni-capability 主要是在这个阶段形成的</mark>。

::: tip 与 Z-Image 的路线对照
<mark class="hl-key">DeepGen 并不是给每个能力单独开一个后训练 branch，而是在 Joint SFT 阶段把这些能力一起喂给统一模型。</mark>Z-Image 则是「SFT → 蒸馏 → RLHF」串行、编辑另开一支继续训练。两种统一模型的组织方式。
:::

| 项 | Stage 2 配置 |
| :--- | :--- |
| 训练步数 | <mark class="hl-trick">**400,000 iterations**</mark> |
| 分辨率 | <mark class="hl-trick">正文写固定 **512×512**，同时通过 **dynamic resizing 保持原始 aspect ratio**</mark> ⚠️ 见下方矛盾 |
| Arbitrary Resolution | <mark class="hl-trick">Table 9 标 **✓**</mark> ⚠️ |
| Learning rate | <mark class="hl-trick">**5×10⁻⁵**</mark> |
| Warm-up | <mark class="hl-trick">正文写 **20,000 steps**</mark> ⚠️ |
| Batch size / GPU | 768 / 64×H200 |
| <mark class="hl-trick">LoRA</mark> | <mark class="hl-trick">**rank 64 / α 128 / dropout 0.05**</mark> |
| <mark class="hl-trick">可训练参数</mark> | <mark class="hl-trick">**SCB connector + DiT + VLM 的 LoRA**</mark> |

两阶段可以压成：

$$
\boxed{
\text{Stage 1}:\ \text{Frozen VLM + Frozen DiT}\rightarrow\text{Train Connector + Think Tokens}
}
$$

$$
\boxed{
\text{Stage 2}:\ \text{Train DiT + Connector + VLM-LoRA}\rightarrow\text{Joint Gen/Edit/Reasoning/Text SFT}
}
$$

::: tip 真正该记住的
<mark class="hl-key">**DeepGen 先解决"VLM 和 DiT 能不能顺畅交流"，再解决"统一模型能不能学好多种能力"。Stage 1 是 representation alignment，Stage 2 才是 capability learning。**</mark>
:::

![DeepGen Fig.3：DeepGen 1.0 架构（VLM-DiT + SCB）。左半是 VLM：System Prompt 与编辑指令各经 Text tokenizer、参考图经 ViT Encoder，序列里 Visual Token（橙）/ Text Token（灰）/ Learnable Think Token（黄）三类拼接，token 序列从 **VLM Block 1、Block 2、…、Block N-R、Block N** 共 6 层均匀抽取后送入 Connector（SigLIP 视觉编码器 + 6 个 transformer 层）。右半是 DiT：DiT 输入由三路拼接——Connector 输出的 **Multimodal Condition**、参考图经 **VAE Encoder** 的 latent、以及 **Noisy Input** 经 Noisy Refiner 的噪声 token，统一做 self-attention，末端经 VAE Decoder 出图。每个 block 右侧的 🔥/❄ 图标按 caption 说明**依次表示该模块在 Pre-Training / SFT / RL 三阶段是否可训练**。](/deepgen-fig3-architecture.png)

::: info Fig. 3 揭示的完整可训练矩阵（正文没写这张表）
caption 明确：图标 *"indicate whether the corresponding module is frozen or trainable during the **Pre-Training, SFT, and RL stages, respectively**"*。据此可读出：

| 模块 | Pre-Training | SFT | RL |
| :--- | :--- | :--- | :--- |
| VLM Blocks（含 ViT Encoder） | ❄ 冻结 | 🔥 可训练（LoRA） | ❄ 冻结 |
| <mark class="hl-trick">Connector</mark> | <mark class="hl-trick">🔥 可训练</mark> | 🔥 可训练 | <mark class="hl-trick">❄ 冻结</mark> |
| DiT Blocks | ❄ 冻结 | 🔥 可训练 | 🔥 可训练 |

<mark class="hl-key">**由此得到一条正文没明说的结论：RL 阶段只更新 DiT，Connector 与 VLM 都是冻结的。**</mark>§3.3 正文只说 *"we apply reinforcement learning after supervised fine-tuning"*，没有交代可训练范围 —— 这是靠 Fig. 3 的图标读出来的。实践上这意味着 <mark class="hl-key">**RL 阶段不碰条件编码路径，奖励信号只经 DiT 影响输出**</mark>，这也解释了为什么 RL 阶段的改动比 SFT 阶段安全得多（§5.3.2 的 RL 消融全部只训 1,000 steps）。
:::

::: warning 论文内部的两处不一致（照记录，不自行校正）
**① Warm-up 数字对不上。** 正文两阶段都写 <mark class="hl-trick">**20,000 warm-up steps**</mark>，但 Appendix Table 9 写 <mark class="hl-trick">**warmup ratio = 0.01**</mark>。按 200K iteration 算 0.01 只有 2K，按 400K 算只有 4K —— <mark class="hl-key">**无论哪个阶段都对不上 20,000**</mark>。论文没有解释。

**② 分辨率表述不一致。** §3.2 正文写 *"fixed resolution of 512×512 **while preserving the original aspect ratio via dynamic resizing**"* —— "固定 512×512"与"保持原始宽高比"本身互相矛盾；而 Table 9 又把 Stage 2 的 **Arbitrary Resolution 标为 ✓**。<mark class="hl-key">论文没有进一步说明具体的 resize / bucket 机制。</mark>
:::

::: info 原文补充（笔记核对时添加，论文 §2 + §3.1/3.2 + Table 9 可查）
- **两阶段的可训练参数，Table 9 是权威口径**：Stage-I 写 *"Trainable Param: **SCB connector**"*；Stage-II 写 *"SCB connector, **DiT**, **LoRA in VLM**"*。§3.1/§3.2 正文与之一致。
- **底座模型（§2 给出）**：VLM = **Qwen-2.5-VL (3B)**，DiT = **SD3.5-Medium (2B)**（*"initialized from [11] with **joint generation–editing capability**"* —— 注意 DiT 底座本身就自带生编一体能力）。Connector = **SigLIP 视觉编码器 + 6 个 transformer 层**。合计约 **5B**。
- **LoRA 引用 [24]**，即 Hu et al. 的 LoRA 原文。
- **双分支视觉编码（Fig. 3 caption 强调）**：*"a **ViT encoder** captures high-level semantics for the VLM, while a **VAE encoder** extracts compressed latent features for the DiT"* —— 参考图被**两条路**编码：高层语义走 ViT→VLM，压缩 latent 走 VAE→DiT。
- **DiT 位置编码区分 reference 与 target**：caption 写 *"DiT positional encodings **explicitly distinguish reference tokens from target tokens**"* —— 与 Z-Image §4.1 用 3D RoPE 时间维偏移区分 reference/target 是同类设计。
- **RL 阶段的三个改动预告**（§3.3 开头，属下一节内容）：MR-GRPO（扩展自 Pref-GRPO [27]）、**novel auxiliary supervised diffusion loss** 补充 KL 正则以缓解长期 RL 的能力退化、以及 **noise-preserving stochastic sampling** [29]。
:::

### 3.3 Stage 3: Reinforcement Learning ★

RL 放在 Joint SFT 之后，提出 <mark class="hl-trick">**MR-GRPO**</mark>，官方定位是 *"the MR-GRPO framework, a variant of **Pref-GRPO** [27], which extends **GRPO** [28] to **flow matching** models"*。Abstract 里的措辞是 *"MR-GRPO, which leverages a **mixture of rewards** and supervision signals"*。<mark class="hl-key">因此 MR 应理解为 Multi-Reward / Mixture-of-Rewards，但论文并没有任何一句把它正式展开成某个固定缩写短语</mark>，引用时按"多奖励 + 多监督信号"理解即可，不要当成作者明确定义的缩写。

论文同时说明这两个改进是 *"two **concurrent** improvements into our pipeline"*（即**并行引入**、非原创）：<mark class="hl-trick">noise-preserving stochastic sampling [29]</mark> 与 <mark class="hl-trick">decoupled advantage normalization [14]</mark>。所以 MR-GRPO 相对 Pref-GRPO 的**真正增量**只有两项：<mark class="hl-key">**多奖励解耦归一化** + **auxiliary supervised diffusion loss**</mark>。

<mark class="hl-trick">MR-GRPO 要同时解决三件事</mark>：

1. <mark class="hl-trick">flow matching 原本是确定性 ODE 推理，没有足够 exploration，而 GRPO 需要同一 prompt 采样出 G 个不同候选</mark>；
2. <mark class="hl-trick">多个 reward 的数值尺度与方差差异巨大，直接相加会让高方差 reward 独占梯度</mark>；
3. <mark class="hl-trick">RL 训练拉长后会破坏 SFT 阶段已学到的复杂指令理解 / reasoning generation 能力</mark>。

#### 3.3.1 多奖励解耦归一化（MR 的核心）

给定文本条件 $h$，flow model 采样一组 $G$ 张图 $\{x_0^i\}_{i=1}^{G}$，<mark class="hl-trick">并保留每张图完整的 denoising trajectory $\{x_T^i, x_{T-1}^i, \dots, x_0^i\}$</mark>（DeepGen 取 $G=8$）。对 $K$ 个 reward $R_k$，<mark class="hl-key">**不是先把 raw reward 相加，而是每个 reward 先在当前 group 内独立标准化**</mark>（Eq. 2）：

$$
A_k^i=
\frac{
R_k(x_0^i,h)-\operatorname{mean}_{j}\big(R_k(x_0^j,h)\big)
}{
\operatorname{std}_{j}\big(R_k(x_0^j,h)\big)
},\qquad j=1,\dots,G
$$

然后加权聚合，再对聚合结果做一次 **batch-wise normalization**：

$$
\hat A^i=\mathrm{BatchNorm}\Big(\textstyle\sum_{k} w_k\,A_k^i\Big)
$$

<mark class="hl-key">**为什么必须这么做**：Preference（VLM 打分）、OCR、CLIP similarity 三者的数值范围和方差可以差好几个数量级。先直接求和的话，高方差 reward 会主导 advantage，低方差 reward 事实上失效。"每个 reward 先 normalize、再加权"才能保留各 reward 的相对信息 —— 论文原话是 *"better preserves **multi-reward signal granularity**"*。</mark>论文自己的解释也印证了这点：<mark class="hl-trick">*"high-variance rewards can **dominate the policy updates** and impede progress on specific objectives when normalization is not applied per reward"*</mark>。

| 消融（1,000 steps，Table 7） | GenEval | DPGBench | GEdit-EN | UniGenBench (Text) | UniGenBench (Overall) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| <mark class="hl-key">DeepGen 1.0 (RL) 完整版</mark> | 0.87 | 87.75 | 7.05 | <mark class="hl-key">**35.06**</mark> | <mark class="hl-key">**75.69**</mark> |
| w/o Reward-wise Norm | 0.86 (−0.01) | 87.73 (−0.02) | 7.02 (−0.03) | <mark class="hl-trick">**32.18 (−2.88)**</mark> | 75.27 (−0.42) |
| w/o Velocity KL | 0.87 | 87.32 (−0.43) | 7.02 (−0.03) | 32.47 (−2.59) | 75.07 (−0.62) |
| w/o Auxiliary SFT Loss | 0.87 | 87.40 (−0.35) | 6.99 (−0.06) | 33.33 (−1.73) | 74.33 (−1.36) |

<mark class="hl-trick">**掉点最狠的一列恰恰是 UniGenBench (Text)**：去掉 reward-wise norm 后 text score 掉 2.88 分，远超 GenEval / DPGBench 的 0.01–0.02。</mark><mark class="hl-key">说明多奖励场景下，text rendering 这类"由单一高方差 OCR reward 主导"的目标恰恰是最容易被其他 reward 挤掉的 —— 这正是解耦归一化要保护的对象。</mark>曲线（Fig. 6a）显示：换成跨 reward 的 joint normalization，<mark class="hl-trick">前 600 steps 与 baseline 几乎无差，约 600 steps 后差距开始明显拉大，1,000 steps 终值明显落后</mark>。这也意味着 <mark class="hl-key">**RL 训得越久，advantage 归一化方式的影响越大**</mark>。

#### 3.3.2 GRPO 目标与 velocity-space KL

更新沿整条 flow trajectory 进行，每个 denoising step 都算 importance ratio（Eq. 3）：

$$
r_t^i(\theta)=\frac{p_\theta\big(x_{t-\Delta t}^i\,\big|\,x_t^i,h\big)}{p_{\theta_{\rm old}}\big(x_{t-\Delta t}^i\,\big|\,x_t^i,h\big)}
$$

$$
\mathcal L_{\rm GRPO}(\theta)=\mathbb E_{h\sim\mathcal D}\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T}\sum_{t=0}^{T-1}
\Big(
\min\big(r_t^i(\theta)\hat A^i,\ \operatorname{clip}(r_t^i(\theta),1-\epsilon,1+\epsilon)\hat A^i\big)
-\beta D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})
\Big)
\right]
$$

<mark class="hl-trick">注意 KL 项写在 $\frac{1}{T}\sum_t$ 内部，即**每个 denoising step 都扣一次**，而不是在 trajectory 外扣一次</mark>（这是与 LLM GRPO 把 KL 放在整条 response 外层的一个实现差异）。

由于 flow matching 没有 token probability，KL 被写成 **velocity 空间**的平方距离（Eq. 4）：

$$
D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})=\big\|\hat v_\theta(x_t,t)-\hat v_{\rm ref}(x_t,t)\big\|_2^2
$$

<mark class="hl-key">作者对 KL 的定位非常明确：*"KL divergence acts as **process-level guidance**, constraining the denoising trajectory to stay close to the reference policy **at each step**"*。</mark>参考策略 $\pi_{\rm ref}$ 就是 SFT 之后的模型（与 Fig. 3 中 RL 阶段冻结 VLM/Connector、只训 DiT 一致）。

#### 3.3.3 Auxiliary SFT Diffusion Loss（KL 不够用）

<mark class="hl-trick">**论文观察到的现象**：只用 KL 约束，RL 训练超过约 $\sim 1000$ steps 后，*"the model exhibits a notable performance drop on tasks requiring **complex instruction comprehension**, such as **reasoning-based generation**"*。</mark>

作者的归因是两种正则**作用层级不同**：

| 约束 | 层级 | 作用 | 局限 |
| :--- | :--- | :--- | :--- |
| <mark class="hl-trick">Velocity KL</mark> | <mark class="hl-trick">**process-level**</mark> | 逐步把 denoising trajectory 拉回参考策略附近 | <mark class="hl-trick">只约束"路径"，不直接保证"终点落回高质量分布"</mark> |
| <mark class="hl-trick">$\mathcal L_{\rm SFT}$</mark> | <mark class="hl-trick">**outcome-level**</mark> | 把最终生成质量直接锚到 SFT 分布 | 需额外采样监督 batch |

<mark class="hl-key">原文措辞：*"**Process-level constraints alone, without outcome-level anchoring, leave the model susceptible to gradual drift during prolonged training.**"*</mark>

所以每个 RL step 都额外从一套**独立的高质量 SFT image-text corpus** 采样监督数据，算标准 flow-matching loss（Eq. 5）：

$$
\mathcal L_{\rm total}=(1-\lambda)\mathcal L_{\rm GRPO}+\lambda\mathcal L_{\rm SFT},\qquad \lambda=1\times 10^{-4}
$$

<mark class="hl-trick">Table 10 明确写 **SFT auxiliary frequency = Every step**，即 $\mathcal L_{\rm SFT}$ 每一步都算，不是周期性插入。</mark>消融（Table 7 / Fig. 6）：去掉后 <mark class="hl-key">**约 300 steps 起整体性能开始下降，到训练后期明显低于起始 checkpoint**；UniGenBench overall 从 75.69 掉到 **74.33**，text 从 35.06 掉到 **33.33**</mark>，DPGBench 87.75 → 87.40。Fig. 6(b) 显示 text rendering 的提升也更慢、更抖。

<mark class="hl-key">**KL 与 SFT loss 是互补而非二选一**：论文明确说 *"The combination of KL regularization and auxiliary SFT loss provides **complementary** constraints"*。</mark>去掉 KL 的组（overall 75.07、text 32.47）掉点反而比去掉 SFT loss 的组（74.33 / 33.33）在 overall 上更小，说明 <mark class="hl-trick">**SFT loss 是主力，KL 是补充**</mark>，二者都不可省。

#### 3.3.4 Noise-Preserving Stochastic Sampling

<mark class="hl-trick">**要解决的问题**：flow matching 的常规推理是确定性 ODE，没有 exploration；而直接换成标准 Flow-SDE 又可能在某些 timestep 注入**超过 scheduler 预期**的噪声，导致 rollout 图像质量下降、reward 信号随之不可靠。</mark>DeepGen 采用 noise-preserving stochastic sampling（Eq. 6），保证 <mark class="hl-key">**每个 timestep 的噪声水平与 flow matching scheduler 保持一致**</mark>：

$$
x_{t-\Delta t}=
\big(1-(t-\Delta t)\big)\hat x_0
+(t-\Delta t)\cos\!\left(\frac{\eta\pi}{2}\right)\hat x_1
+(t-\Delta t)\sin\!\left(\frac{\eta\pi}{2}\right)\epsilon
$$

$$
\hat x_0=x_t-t\,\hat v_\theta,\qquad \hat x_1=x_t+(1-t)\,\hat v_\theta,\qquad \epsilon\sim\mathcal N(0,I)
$$

$\eta\in[0,1]$ 控制随机性强度，DeepGen 取 <mark class="hl-trick">$\eta=1.0$</mark>。当 $\eta=0$ 时 $\cos(0)=1,\sin(0)=0$，公式退化为确定性 step；$\eta=1$ 时 $\cos(\pi/2)=0$，随机性最大。

为算 importance ratio，log-probability 被简化为 squared-distance 形式（Eq. 7）：

$$
\log p_\theta\big(x_{t-\Delta t}\,\big|\,x_t\big)=-\big\|x_{t-\Delta t}-\mu_\theta(x_t,t)\big\|_2^2
$$

$$
\mu_\theta(x_t,t)=\big(1-(t-\Delta t)\big)\hat x_0+(t-\Delta t)\cos\!\left(\frac{\eta\pi}{2}\right)\hat x_1
$$

<mark class="hl-key">$\mu_\theta$ 就是采样 step 的**确定性分量**，噪声项被单独提出。论文明确：这一形式 *"**removes the variance normalization term** present in the standard log-probability, avoiding **numerical instability at small noise levels**"*。</mark>这里 $\mu_\theta$ 的写法本身就是 <mark class="hl-trick">**ODE 与 SDE 的统一表示**</mark>：flow-GRPO 相关的后续工作大多沿用这一范式。

::: tip 这一节真正该记住的
<mark class="hl-key">**RL rollout 需要随机性，但新增的随机性又不能破坏原 flow scheduler 的 noise level。**</mark>噪声-保持采样 + squared-distance log-prob（去掉 variance normalization）就是 DeepGen 给出的解法。**这两条是 [Flow-GRPO 综述](./image-rl-posttraining/flow-grpo-survey-2026.md) 里 flow matching RL 通用底座的一部分，不是 DeepGen 的原创贡献**（论文自己标注为 [29] 的 concurrent improvement）。
:::

#### 3.3.5 三路 Reward 与按任务分组的权重配方

| # | Reward | 来源 | 监督对象 | 机制 |
| :-: | :--- | :--- | :--- | :--- |
| 1 | <mark class="hl-trick">$R_{\rm Pref}$ — VLM pairwise preference</mark> | **Unified-Reward-Think** [67]（作者自建） | <mark class="hl-trick">image-text alignment + visual quality</mark> | 在 group 内两两比较全部生成图，<mark class="hl-key">**以 per-sample win rate 作为 reward score**</mark> |
| 2 | <mark class="hl-trick">$R_{\rm OCR}$</mark> | [68] | text rendering accuracy | 检测生成图中的文字，与 prompt 指定的目标文本比对 |
| 3 | <mark class="hl-trick">$R_{\rm CLIP}$</mark> | [69] | semantic consistency | 图像与文本条件的整体 CLIP 相似度 |

<mark class="hl-trick">**关键设计：不同任务类别用不同 reward composition（Table 11）**</mark>

| Prompt Category | Preference | CLIP Sim | OCR |
| :--- | ---: | ---: | ---: |
| <mark class="hl-trick">Text rendering</mark> | 0.2 | 0.1 | <mark class="hl-key">**0.7**</mark> |
| <mark class="hl-trick">General T2I</mark> | <mark class="hl-key">**0.7**</mark> | 0.3 | <mark class="hl-key">**—（不用）**</mark> |

<mark class="hl-key">**这条比"多加几个 reward"更值得记：multi-reward 不只是堆数量，还可以按 task type 动态切换 reward 配方。**</mark>text rendering 把权重压到 OCR 上以"directly optimize text accuracy"；general T2I 则完全不挂 OCR，靠 preference reward 做整体质量判断 —— <mark class="hl-trick">因为通用生成 prompt 里通常没有指定目标文本，OCR reward 在那里是无意义的噪声源。</mark>这也正好解释了 §3.3.1 消融中 text score 掉点最狠：<mark class="hl-trick">OCR 在 text rendering 上权重高达 0.7，一旦归一化被破坏，它就第一个崩。</mark>

#### 3.3.6 RL Prompt 分布与辅助 SFT 语料（一个关键事实）

| Prompt 类别 | 采样权重 | 数据来源 |
| :--- | :-: | :--- |
| <mark class="hl-trick">Text-rendering</mark> | <mark class="hl-key">**3.0×**</mark> | UniGenBench text data、Qwen-Image text rendering captions、curated text rendering prompts |
| <mark class="hl-trick">General T2I</mark> | **1.0×** | UniGenBench general data、BLIP3-o captions、ShareGPT-4o image descriptions、CoREBench prompts |

辅助 SFT 语料是**独立**的高质量 image-text pair corpus，采样权重与 RL prompt 分布对齐：

| 辅助 SFT 语料类别 | 采样权重 | 数据来源 |
| :--- | :-: | :--- |
| General T2I pairs | 1.0× | BLIP3-o、ShareGPT-4o、Echo-4o、OpenGPT-4o、GenEval、Self-Banana-50K |
| <mark class="hl-trick">Text rendering pairs</mark> | <mark class="hl-key">**3.0×**</mark> | <mark class="hl-trick">*"to **match the emphasis on text rendering in the RL prompts**"*</mark> |

::: danger 关键事实：DeepGen 的 RL 阶段完全没有 Edit 数据
<mark class="hl-key">**RL prompt 只有 general T2I 和 text rendering 两类，auxiliary SFT corpus 里也**没有 editing triplet**（Appendix B 全文未出现任何 editing 数据源）。**</mark>

于是形成了一个非常明确的**能力收缩**：

$$
\boxed{
\text{Stage 1 / Stage 2} = \text{Generation} + \text{Editing} \ \text{联合训练}
}
$$

$$
\boxed{
\text{Stage 3 (RL)} = \underbrace{\text{General T2I}}_{\times 1} + \underbrace{\text{Text Rendering}}_{\times 3}\qquad \textbf{无 Editing}
}
$$

<mark class="hl-key">**对做图像 RL 的直接启示：如果某种能力完全不进入 RL 数据，就要警惕 RL 对该能力产生 trade-off。**</mark>这也是 §3.3.9 里 editing 指标下降的机制来源。
:::

#### 3.3.7 RL 工程配置（Table 10 全文）

| 项 | 值 |
| :--- | :--- |
| <mark class="hl-trick">Group size $G$</mark> | **8** |
| Image resolution | 512×512 |
| Denoising steps | 50 |
| <mark class="hl-trick">SDE stochasticity $\eta$</mark> | **1.0** |
| <mark class="hl-trick">Timestep fraction</mark> | **0.6** ⚠️ 见下方说明 |
| Learning rate | 2×10⁻⁶ |
| Total training steps | 1,500 |
| <mark class="hl-trick">KL coefficient $\beta$</mark> | **5×10⁻⁷** |
| <mark class="hl-trick">Clip range $\epsilon$</mark> | **1×10⁻⁴**（<mark class="hl-trick">极小，说明策略更新非常保守</mark>） |
| SFT auxiliary coefficient $\lambda$ | 1×10⁻⁴ |
| <mark class="hl-trick">SFT auxiliary frequency</mark> | <mark class="hl-key">**Every step**</mark> |
| Global batch size | 256 |
| DeepSpeed stage | ZeRO-2 |
| Precision | BF16 |

<mark class="hl-trick">RL 全程只有 **1,500 steps**、lr **2×10⁻⁶**、clip range **1×10⁻⁴** —— 三个数都指向同一件事：</mark><mark class="hl-key">**这是一次极其保守的轻量级 RL 微调，不是长周期 RL。**</mark>这也解释了为什么辅助 SFT loss（$\lambda$ 同为 1×10⁻⁴）能在这么短的训练里起决定性作用 —— <mark class="hl-trick">当策略更新幅度本身很小时，分布锚定的相对权重就被放大了。</mark>对照 Stage 1/2 的 200K / 400K iterations，RL 只占训练总量的 <mark class="hl-trick">0.25%</mark>。

::: warning Timestep fraction = 0.6 的含义论文未解释
Table 10 列出 **timestep fraction = 0.6**，<mark class="hl-key">**论文没有给出任何进一步定义或实现说明**</mark>。从 flow RL 的常见做法推测可能是"在时间轴上前 60% 的 timestep 上施加 GRPO 更新、后 40% 只做 SFT loss"，但这是**推断，论文未说明，不可作为事实引用**。
:::

#### 3.3.8 消融（§5.3.2，Table 7 + Fig. 6）

三个消融变体，<mark class="hl-trick">**全部只训 1,000 steps**</mark>，其余配置完全相同，评测集为 UniGenBench（+ GenEval / DPGBench / GEdit-EN）：

| 变体 | 论文结论 | 定量表现 |
| :--- | :--- | :--- |
| <mark class="hl-trick">w/o Auxiliary SFT Loss</mark> | <mark class="hl-key">**"critical for maintaining generation quality during extended RL training"**</mark>；*"performance degradation after approximately **300 steps**, eventually dropping **well below the initial checkpoint**"* | overall **75.69 → 74.33**，text **35.06 → 33.33** |
| <mark class="hl-trick">w/o Velocity KL</mark> | *"unconstrained policy updates can lead to **forgetting** of capabilities acquired during supervised fine-tuning"* | overall **75.69 → 75.07**，DPGBench **87.75 → 87.32**，text → 32.47 |
| <mark class="hl-trick">w/o Reward-wise Norm</mark> | *"replacing reward-wise normalization with **joint normalization** across all rewards yields comparable performance in the **early stages** but leads to a **growing gap after approximately 600 steps**"* | overall **75.69 → 75.27**，text **35.06 → 32.18** |

::: info 三条曲线的时间尺度各不相同，值得记
<mark class="hl-trick">**SFT loss 是 300 steps 就要命，reward-wise norm 是 600 steps 才显形，KL 则全程缓慢落后。**</mark>

这三条失效的时间尺度差异，正好对应三种不同的失效机制：
- <mark class="hl-key">**SFT loss 缺失 → 快速 drift**</mark>：没有 outcome anchor，曲线在数百步内就跌破起点；
- <mark class="hl-key">**reward-wise norm 缺失 → 缓慢的 reward 挤压**</mark>：前期各 reward 分布尚可区分，后期高方差 reward 逐渐压倒专项目标；
- <mark class="hl-key">**KL 缺失 → 缓慢的能力遗忘**</mark>：全程小幅落后，Fig. 6(a) 中 w/o KL 曲线始终在 baseline 下方。
:::

#### 3.3.9 RL 的收益与代价

**收益**（RL vs SFT，Fig. 5，1,500 steps 曲线）：

| 指标 | SFT | RL | 变化 |
| :--- | ---: | ---: | ---: |
| <mark class="hl-trick">UniGenBench Overall</mark> | $\sim 0.747$ | $\sim 0.756$ | <mark class="hl-key">**+0.9 pt**</mark> |
| <mark class="hl-trick">UniGenBench (Text Generation)</mark> | $\sim 0.25$ | $\sim 0.34$ | <mark class="hl-key">**+9 pt，涨了 36%**</mark> |
| <mark class="hl-trick">CVTG-2K Word Accuracy</mark> | 0.6605 | <mark class="hl-key">**0.7533**</mark> | <mark class="hl-key">**+0.093（+14%）**</mark> |
| <mark class="hl-trick">CVTG-2K CLIPScore</mark> | — | <mark class="hl-key">**0.8278（开源最高）**</mark> | 文本保真度提升未牺牲语义对齐 |
| WISE（reasoning gen）overall | 0.72 | 0.73 | +0.01 |
| T2I-CoREBench overall | 45.7 | 46.5 | +0.8 |

<mark class="hl-key">**CVTG-2K 上「Word Accuracy 大涨 + CLIPScore 仍是开源最高」这一组合，是 multi-reward 配方设计成功最直接的证据**</mark> —— 靠把 OCR 权重压到 0.7 精确地买到文字准确率提升，同时没有牺牲语义一致性。

**代价**（editing 侧指标全部下降）：

| 指标 | SFT | RL | 变化 |
| :--- | ---: | ---: | ---: |
| <mark class="hl-trick">RISE Overall</mark> | 13.3 | <mark class="hl-trick">**10.8**</mark> | <mark class="hl-trick">**−2.5（相对 −19%）**</mark> |
| <mark class="hl-trick">UniREditBench Overall</mark> | 77.5 | <mark class="hl-trick">**75.7**</mark> | −1.8 |
| GEdit-EN | 7.12 | 7.05 | −0.07 |
| RISE Temporal / Causal / Spatial / Logical | 15.3 / 18.9 / 14.0 / 4.7 | 12.9 / 14.4 / 13.0 / 2.4 | 四项全降 |

<mark class="hl-key">**这一组下降与 §3.3.6 的"RL 无 editing 数据"完全一致 —— 优化范围收缩到了 generation + text，模型在 RL 期间对 editing 的能力保持就自然漂移。</mark>

::: warning 因果关系要严格区分
<mark class="hl-trick">**"没有 Edit RL 数据导致 editing 下降"是合理推测，但论文并没有做专门的因果实验来证明这一点。**</mark>论文正文只笼统写 RL 后编辑能力 *"remains competitive"*、RISE *"remaining competitive under RL"*，<mark class="hl-key">**从未声称自己做过"加回 edit RL 数据"的对照实验**</mark>。替代解释（同样未被排除）包括：RL 与 SFT 的损失权重竞争、learning rate 差异、1,500 steps 本身不足以维持多任务能力。引用时必须标注这是推断。
:::

#### 3.3.10 MR-GRPO 全流程压缩

$$
\boxed{
\text{T2I / Text Prompt}
\xrightarrow{\ \text{8-sample noise-preserving stochastic rollout}\ }
\{x_0^i\}_{i=1}^{G}
\xrightarrow{\ \{R_{\rm Pref},R_{\rm CLIP},R_{\rm OCR}\}\ }
\underbrace{\text{per-reward group normalize}}_{\text{Eq. 2}}
\xrightarrow{\ w_k\ }
\text{weighted advantage}
\xrightarrow{\ \text{GRPO} + \|\hat v_\theta-\hat v_{\rm ref}\|^2\ }
\mathcal L_{\rm total}
}
$$

每一步并行注入的旁路：

$$
\boxed{
\text{High-quality T2I/Text SFT Corpus}\ (\times 1\ \text{general},\ \times 3\ \text{text})
\ \xrightarrow{\ \text{flow-matching loss},\ \lambda=10^{-4}\ }
\mathcal L_{\rm total}\quad(\textbf{every step})
}
$$

#### 3.3.11 DeepGen RL 值得带走的六条

::: tip 以后真正自己做图像 RL 时，DeepGen 最值得带走的不是"用了 GRPO"，而是这套完整 recipe
1. <mark class="hl-trick">**Rollout 要有随机性，但不能破坏 flow noise schedule**</mark> → noise-preserving stochastic sampling + squared-distance log-prob。
2. <mark class="hl-trick">**多 reward 必须先独立归一化再融合**</mark>，否则高方差 reward 独占梯度，专项目标最先崩。
3. <mark class="hl-trick">**不同任务可以用不同 reward 配方**</mark>，而不是全局一套权重；不相关的 reward（general T2I 上的 OCR）应当直接关掉。
4. <mark class="hl-trick">**KL 约束 denoising trajectory（process-level），SFT loss 锚定原有分布（outcome-level），二者互补**</mark>；只上 KL 在 $\sim 1000$ steps 后不够用。
5. <mark class="hl-trick">**RL prompt 分布与 auxiliary SFT 分布最好保持对应**</mark>（这里都是 text 3× / general 1×）。
6. <mark class="hl-trick">**某种能力完全不进 RL 数据，就必然要警惕 RL 对它产生 trade-off**</mark> —— DeepGen 的 editing 就是活生生的例子（RISE 13.3 → 10.8）。
:::

#### 3.3.12 公式速查卡（Eq. 2–7 一页回顾）

DeepGen §3.3 的全部数学内容就是下面 6 个式子。<mark class="hl-key">论文的 Eq. 1 属于 §2 的 SCB connector，RL 部分的编号是 Eq. 2 → Eq. 7 连续六式</mark>，复习时按这张表从左往右读一遍即可串起全流程。

| # | 记号 | 公式 | 在流程中的位置 | 去掉会怎样 |
| :-: | :--- | :--- | :--- | :--- |
| <mark class="hl-trick">Eq. 2</mark> | $A_k^i$ | $\dfrac{R_k(x_0^i,h)-\operatorname{mean}_j R_k(x_0^j,h)}{\operatorname{std}_j R_k(x_0^j,h)}$ | reward 打完分之后、聚合之前 | text score <mark class="hl-trick">35.06 → 32.18</mark>，约 600 steps 后差距显形 |
| <mark class="hl-trick">Eq. 2′</mark> | $\hat A^i$ | $\operatorname{BatchNorm}\big(\sum_k w_k A_k^i\big)$ | 聚合之后 | 论文未单列消融，随 Eq. 2 一起失效 |
| <mark class="hl-trick">Eq. 3</mark> | $\mathcal L_{\rm GRPO}$ | $\mathbb E_h\big[\tfrac1G\sum_i\tfrac1T\sum_t\big(\min(r_t^i\hat A^i,\ \mathrm{clip}(r_t^i,1{-}\epsilon,1{+}\epsilon)\hat A^i)-\beta D_{\rm KL}\big)\big]$ | 整条 trajectory 上逐 step 更新 | 主目标本身 |
| | $r_t^i(\theta)$ | $\dfrac{p_\theta(x_{t-\Delta t}^i\mid x_t^i,h)}{p_{\theta_{\rm old}}(x_{t-\Delta t}^i\mid x_t^i,h)}$ | Eq. 3 的 per-step importance ratio | 主目标本身 |
| <mark class="hl-trick">Eq. 4</mark> | $D_{\rm KL}$ | $\lVert\hat v_\theta(x_t,t)-\hat v_{\rm ref}(x_t,t)\rVert_2^2$ | Eq. 3 括号内，**每个 step 各扣一次** | overall 75.69 → 75.07，DPGBench 87.75 → 87.32 |
| <mark class="hl-trick">Eq. 5</mark> | $\mathcal L_{\rm total}$ | $(1-\lambda)\mathcal L_{\rm GRPO}+\lambda\mathcal L_{\rm SFT}$，$\lambda=10^{-4}$ | 每个 RL step 与 Eq. 3 同步计算 | overall 75.69 → **74.33**，约 300 steps 起就掉 |
| <mark class="hl-trick">Eq. 6</mark> | $x_{t-\Delta t}$ | $\big(1-(t{-}\Delta t)\big)\hat x_0+(t{-}\Delta t)\cos\frac{\eta\pi}{2}\,\hat x_1+(t{-}\Delta t)\sin\frac{\eta\pi}{2}\,\epsilon$ | rollout 采样，$\eta=1.0$ | 退化为确定性 ODE，无法 exploration |
| | $\hat x_0,\hat x_1$ | $\hat x_0=x_t-t\hat v_\theta$，$\hat x_1=x_t+(1-t)\hat v_\theta$ | Eq. 6 的预测 clean sample / noise | — |
| <mark class="hl-trick">Eq. 7</mark> | $\log p_\theta$ | $-\lVert x_{t-\Delta t}-\mu_\theta(x_t,t)\rVert_2^2$ | 算 $r_t^i$ 时的 log-prob 近似 | 含 variance normalization 时小噪声段数值不稳 |

<mark class="hl-key">**记忆锚点：Eq. 2 管"多 reward 怎么融合"，Eq. 3/4 管"策略怎么更新 + 别跑太远"，Eq. 5 管"别忘掉 SFT 学到的东西"，Eq. 6/7 管"怎么在 flow matching 上做有探索的采样"。**</mark>四个层次正好对应 §3.3.1 / §3.3.2-3 / §3.3.3 / §3.3.4 —— <mark class="hl-trick">Eq. 2 是 DeepGen 相对 Pref-GRPO 的真正增量，Eq. 5 是另一项增量，Eq. 4/6/7 都是并发引入的既有设计。</mark>

::: info §3.3 与本文档其他笔记的关系
- <mark class="hl-trick">**§3.3.4 的 noise-preserving SDE 与 squared-distance log-prob 是 flow-matching RL 的通用底座</mark>，详见 [Flow-GRPO 综述](./image-rl-posttraining/flow-grpo-survey-2026.md)；DeepGen 自己标注为 [29] 的并行引入。
- <mark class="hl-trick">**§3.3.1 的 reward-wise 归一化 ↔ ERNIE-Image DPO 的双 anchor、Qwen-Image-2.0-RL 的五奖励 GRPO</mark>，属于同一族"多信号对齐"设计，对照见 [算法 × 奖励 × 基模对比](./image-rl-posttraining/rl-comparison-2026.md)。
- <mark class="hl-trick">**§3.3.3 的 auxiliary SFT loss ↔ LLaDA-Image 用 TwinFlow 蒸馏替代对齐、Swift-Image 用多教师在线蒸馏</mark>，都是"RL 阶段防止能力漂移"的不同解法。
- <mark class="hl-trick">**§3.3.6 的无-edit-RL 权衡 ↔ SeFi-Image 的能力标签式 RL 采样</mark>，后者正是按能力标签控制 RL 覆盖面的做法。
:::

::: warning §3.3 论文未公开的细节
- <mark class="hl-trick">**Timestep fraction = 0.6</mark> 的具体实现（§3.3.7）。
- <mark class="hl-trick">**Unified-Reward-Think [67]</mark> 是作者自建模型，论文未给出其训练数据、标注方式、胜率如何映射为连续 reward。
- <mark class="hl-trick">**$\mu_\theta$ 的高斯方差假设</mark>：Eq. 7 直接把 log-prob 写成 $-\|\cdot\|^2$，隐含等方差的隐式高斯假设，论文未讨论。
- <mark class="hl-trick">**每个 timestep 的 KL 是否等权**</mark>：Eq. 3 中 $-\beta D_{\rm KL}$ 写在 $\frac{1}{T}\sum_t$ 内部且不带时变系数，论文未讨论是否应按时变加权。
- <mark class="hl-trick">**RL 阶段的训练算力**</mark>：与 Stage 1/2 一样未披露 GPU 数量 / 卡时。
- <mark class="hl-trick">**未披露 GRPO 的 $\mathcal D$ 中 general T2I 与 text rendering 的实际样本数</mark>，只有 1× / 3× 的相对采样权重。
:::

## 4. Data ★

DeepGen 的数据设计和 Z-Image 很不一样。它没有重点讲复杂的数据基础设施、去重或动态过滤，而是把重点放在 **"不同训练阶段分别需要什么类型的数据，以及 generation / editing / reasoning / text rendering 这些能力应该怎么组织"**。Figure 4 给出的整体思路就是把真实数据、合成数据和精心筛选的开源数据混在一起，覆盖 general generation、general editing、reasoning-based generation / editing、text rendering 和 application-oriented scenarios。

![DeepGen Fig.4：能力与评测的对应图（上半为 Multiple Scenarios，下半为 Evaluation Results 与 Tasks）。上半把能力分成生成侧（General Generation「基础语义与指令跟随」、Reasoning Generation「复杂推理与世界知识对齐」、Text Rendering「综合掌握文本结构要素」、Generative Applications「泛化到诗歌与海报」）与编辑侧（General Editing「图像一致性与指令跟随」、Reasoning Editing「复杂推理与世界知识对齐编辑」）。下半是五组能力各自对应的评测基准：General Generation → UniGenBench / GenEval / DPG Bench；Reasoning Generation → CoreBench Reason / WISE；Text Rendering → CVTG-2K；General Editing → ImgEdit / GEdit-EN；Reasoning Editing → UniREditBench / RISE。](/deepgen-fig4-data.png)

::: warning 一处图文不符：Fig. 4 并不画数据配比
§4 正文写 *"The overall composition of our training data is illustrated in **Fig. 4**"*，但 <mark class="hl-key">Fig. 4 实际是一张「能力分类 + 评测基准对应」图，完全没有画数据源或数量配比</mark>。**真实的数据明细与数量只在附录 Table 8 里。**引用"数据组成见图"会误导。
:::

### 4.1 General Generation

**Alignment Pre-training 阶段**主要用大规模公开 image-text pairs：<mark class="hl-trick">**text-to-image-2M、LAION-Aesthetic-6M、Megalith-10M、RedCaps-5M、CC-12M**</mark>。Appendix Table 8 统计为约 **35M general-generation samples**。

到了 **SFT，数据分布明显变了**，不再单纯依赖大规模 web pair，而是换成更高质量的 instruction-following generation data：<mark class="hl-trick">**BLIP-3o 60K、ShareGPT-4o-Image 45K、Echo-4o-Image 100K、OpenGPT4o-Image 40K**</mark>，再加入 <mark class="hl-trick">**10M in-house real samples**</mark>。内部数据同时包含 long-form 和 short-form prompts，<mark class="hl-trick">比例明确是 **3:1**</mark>。另外还用 <mark class="hl-trick">**Nano Banana 合成约 50K 高清 photorealistic images**</mark>，配套 fine-grained prompts，补充中英文细粒度写实生成能力。

<mark class="hl-key">正文列出了这些组成，而 Appendix Table 8 将 SFT general generation 总量报告为约 **11M**；论文没有进一步解释正文各子集之和与 11M 之间的差额，因此这里不要自行补。</mark>

### 4.2 General Editing

<mark class="hl-trick">**General Editing 的数据组织更值得记，因为它基本就是一张开源 editing 数据地图**</mark>。DeepGen 收集的是 <mark class="hl-trick">**image–instruction–image triplets**</mark>，**九个数据源**：

GPT-Image-Edit 1.5M ｜ X2I2 1.6M ｜ UniWorld-Edit set 1.2M ｜ NHR-Edit 720K ｜ Pico-Banana 250K ｜ Nano-banana-consist 150K ｜ ShareGPT-4o-Image-Edit set 50K ｜ OpenGPT4o-Image-Edit set 40K ｜ <mark class="hl-trick">in-house editing samples（中英文）1.1M</mark> → **合计 ≈ 6.6M**

逐条明细与引用编号见 [§4.6 附录 Table 8 完整数据明细](#sec-4-6-table8)。

<mark class="hl-key">最关键的一点：这 **6.6M editing 数据既用于 Alignment Pre-training，也继续用于 SFT**。</mark>也就是说 <mark class="hl-trick">DeepGen 不是先只做 generation、后面再加 editing，而是一开始对齐阶段就让模型见 generation + editing，两种任务在后续 SFT 里再继续联合训练</mark>。

### 4.3 Reasoning-based Generation and Editing

<mark class="hl-trick">这部分只在 SFT 阶段加入，不属于前面的基础 Alignment Pre-training</mark>。数据来自作者自己的 <mark class="hl-trick">**UniReason**</mark>：<mark class="hl-trick">reasoning generation **150K**，reasoning editing **100K**</mark>。覆盖五类知识领域：<mark class="hl-trick">**cultural commonsense、natural science、spatial、temporal、logical reasoning**</mark>。

<mark class="hl-key">这里的数据作用不是单纯提升"图画得好不好"，而是专门让模型学会利用 VLM 中已有的世界知识去完成需要推理的生成和编辑任务。</mark>Table 8 也明确把这 150K + 100K 放在 SFT，而没有放进 Pre-training。

### 4.4 Text Rendering + Application-oriented Data

<mark class="hl-trick">这一块非常适合以后自己造专项 SFT 数据</mark>。构造链路是：

1. <mark class="hl-trick">从 **document-centric 和 infographic-centric multimodal QA datasets** 中提取 captions</mark>；
2. <mark class="hl-trick">让 **Gemini 2.5 Pro** 随机组合各种 rendering attributes（**font styles、layouts、color schemes**）</mark>；
3. <mark class="hl-trick">再与面向 text rendering 的开源 prompt set 结合</mark>；
4. <mark class="hl-trick">**直接用 Qwen-Image 合成对应图像**</mark>，最终得到 <mark class="hl-trick">**500K text-rendering samples**</mark>；
5. 再扩展到应用型场景（<mark class="hl-trick">**Chinese poetry generation、poster design**</mark>），额外增加 <mark class="hl-trick">**60K samples**</mark>。

Table 8 最终把 text rendering / poster / Chinese poem 合并统计为 <mark class="hl-trick">**560K**</mark>。

### 4.5 数据 curriculum 全貌

$$
\boxed{
\text{Alignment Pre-training}
=
35M\ \text{General Generation}
+
6.6M\ \text{General Editing}
}
$$

$$
\boxed{
\text{Joint SFT}
=
11M\ \text{High-quality Generation}
+
6.6M\ \text{Editing}
+
150K\ \text{Reasoning Gen}
+
100K\ \text{Reasoning Edit}
+
560K\ \text{Text Rendering/Application}
}
$$

::: tip DeepGen 数据部分最值得记住的两点
**① Pre-training 解决"规模和基础对齐"，SFT 才明显转向高质量 instruction data** 并加入 reasoning、text rendering、application-specific 合成数据。**② editing 不是后加的能力，而是从 Alignment Pre-training 开始就和 generation 一起进入模型**（6.6M 两阶段复用）。
:::

::: info 对搭开源统一生成编辑模型的实用 recipe
**基础 generation 可以靠公开 image-text pairs；editing 可以直接拼现成 triplet 数据集；reasoning 用专项构造数据补；text rendering 则可以通过"LLM/VLM 生成结构化 prompt + 强图像模型合成 target"的方式造专项监督。**
:::

::: warning 本节未公开的细节
DeepGen **没有公开**更细的：数据过滤规则、质量打分、去重策略、caption 重写方法、**各数据源在 batch 内的具体采样比例**。除 long/short prompt 的 3:1 以及后续 RL 数据的权重外，Data Section 本身没有给出这些细节。**§4 的信息密度明显低于 Z-Image 的 §2。**
:::

::: info 原文补充（笔记核对时添加，论文 §4 + 附录 Table 8 可查）
- **三类数据来源的定位**：§4 开头明确说 Fig. 4 组合了 *"real-world, synthetic, and carefully curated **open-source** datasets"* —— 三类来源在论文里是并列的，没有说谁为主。
- **值得注意的教师模型选型**：造数据的三个强模型分别是 **Gemini 2.5 Pro**（编 caption/属性）、**Qwen-Image**（合成 text-rendering 图）、**Nano Banana**（合成 50K 写实图）。前两个是外部闭源，**与 §1 里"反对依赖专有模型"的路线相反** —— DeepGen 走的是"用强模型造专项数据"，而非"从零自建数据基建"。这与 Z-Image 的数据哲学是两种取向。
- **Table 8 的一个细节**：in-house 数据都标了 †，论文注明 *"† denotes covering both **Chinese and English** prompts"* —— 即内部数据是**双语**的。
- **规模对照（论文正文给出）**：DeepGen 全程 **~50M samples**，对比 **LongCat-Image 1.2B**、**HunyuanImage 3.0 5B**。这是它"小模型对抗大模型"的核心论据之一。
- **Fig. 4 顺带给出了完整基准清单**（后面 §5 会用到）：General Generation → UniGenBench / GenEval / DPG Bench；Reasoning Generation → CoreBench Reason / WISE；Text Rendering → CVTG-2K；General Editing → ImgEdit / GEdit-EN；Reasoning Editing → UniREditBench / RISE。
:::

### 4.6 附录 Table 8 完整数据明细（逐数据集） { #sec-4-6-table8 }

附录 Table 8 是**全部数据配比的唯一权威出处**（正文只给部分数字）。原表为 4 列（Stage / Task / Data source / Size），此处**按数据集逐条拆开**，并附求和校验。

原表结构：

| Stage | Task | Data source | Size |
| :--- | :--- | :--- | ---: |
| Pre-Training | General Generation | text-to-image-2M [30], LAION-Aesthetic-6M [31], Megalith-10M [32], RedCaps-5M [33], CC-12M [34] | 35M |
| Pre-Training | General Editing | NHR-Edit [38], GPT-Image-Edit [39], ShareGPT-4o-Image-Edit [35], OpenGPT4o-Image-Edit [37], Nano-banana-consist [40], Pico-Banana [41], X2I2 [12], UniWorld-Edit set [17], in-house editing data† | 6.6M |
| Supervised Fine-Tuning | General Generation | BLIP-3o [7], ShareGPT-4o-Image [35], Echo-4o-Image [36], OpenGPT4o-Image [37], Self-Banana-50K, in-house generation data† | 11M |
| Supervised Fine-Tuning | General Editing | （与 Pre-Training 同一份，见下） | 6.6M |
| Supervised Fine-Tuning | Reasoning Generation | UniReason-T2I set [42] | 150K |
| Supervised Fine-Tuning | Reasoning Editing | UniReason-Edit set [42] | 100K |
| Supervised Fine-Tuning | Text Rendering | General text rendering, poster design†, Chinese poem | 560K |

> 论文注明：*"† denotes covering both **Chinese and English** prompts"* —— 带 † 的内部数据与 poster design 均为**双语**。

#### ① Pre-Training / General Generation = 35M

| # | 数据集 | 引用 | 规模 |
| :-: | :--- | :-: | ---: |
| 1 | text-to-image-2M | [30] | 2M |
| 2 | LAION-Aesthetic-6M | [31] | 6M |
| 3 | Megalith-10M | [32] | 10M |
| 4 | RedCaps-5M | [33] | 5M |
| 5 | CC-12M | [34] | 12M |
| | **明细合计** | | <mark class="hl-key">**35M**</mark> |
| | **Table 8 报告值** | | **35M** ✅ 完全吻合 |

#### ② General Editing = 6.6M（**Pre-Training 与 SFT 两阶段复用同一份**）

| # | 数据集 | 引用 | 规模 |
| :-: | :--- | :-: | ---: |
| 1 | X2I2 | [12] | 1.6M |
| 2 | GPT-Image-Edit | [39] | 1.5M |
| 3 | UniWorld-Edit set | [17] | 1.2M |
| 4 | <mark class="hl-trick">in-house editing data†</mark> | — | 1.1M |
| 5 | NHR-Edit | [38] | 720K |
| 6 | Pico-Banana | [41] | 250K |
| 7 | Nano-banana-consist | [40] | 150K |
| 8 | ShareGPT-4o-Image-Edit set | [35] | 50K |
| 9 | OpenGPT4o-Image-Edit set | [37] | 40K |
| | **明细合计** | | 6.610M |
| | **Table 8 报告值** | | **6.6M** ✅ 吻合（四舍五入） |

<mark class="hl-key">**值得注意的是这份 editing 清单在 Pre-Training 与 SFT 两阶段逐字相同、规模也相同** —— 即 6.6M 编辑数据被完整复用两遍，不是"预训练用一部分、SFT 再补新的"。这与 §4.5 公式里 editing 项写两次 6.6M 是一致的。</mark>但也意味着 <mark class="hl-trick">DeepGen 在 SFT 阶段并没有为 editing 引入任何新数据源</mark>。

#### ③ SFT / General Generation = 11M

| # | 数据集 | 引用 | 规模 |
| :-: | :--- | :-: | ---: |
| 1 | <mark class="hl-trick">in-house generation data†</mark> | — | 10M |
| 2 | Echo-4o-Image | [36] | 100K |
| 3 | BLIP-3o | [7] | 60K |
| 4 | Self-Banana-50K | — | 50K |
| 5 | ShareGPT-4o-Image | [35] | 45K |
| 6 | OpenGPT4o-Image | [37] | 40K |
| | **明细合计** | | <mark class="hl-key">**10.295M**</mark> |
| | **Table 8 报告值** | | **11M** ⚠️ <mark class="hl-trick">**差 0.705M**</mark> |

<mark class="hl-key">**① 差额 0.705M**</mark>：公开子集只有 295K，剩下全靠 10M 内部数据。<mark class="hl-trick">Table 8 的 11M 比明细之和大出约 705K，论文未解释这一差额</mark>，不要自行补。

<mark class="hl-key">**② 命名不一致**</mark>：§4 正文写 *"we synthesize approximately 50k high-clarity photorealistic images ... using **Nano Banana**"*，而 Table 8 写的是 *"**Self-Banana-50K**"*。<mark class="hl-trick">两者规模都是 50K，应指同一批，但命名不一致，论文未说明</mark>。

#### ④ SFT / Reasoning + Text Rendering

| # | 任务 | 数据集 | 引用 | 规模 |
| :-: | :--- | :--- | :-: | ---: |
| 1 | Reasoning Generation | UniReason-T2I set | [42] | 150K |
| 2 | Reasoning Editing | UniReason-Edit set | [42] | 100K |
| 3 | Text Rendering | General text rendering | — | （正文 500K） |
| 4 | Text Rendering | poster design† | — | （正文 60K，含 application-oriented） |
| 5 | Text Rendering | Chinese poem | — | （同上 60K 内） |
| | **Text Rendering 合计** | | | **560K** ✅ 与正文 500K+60K 吻合 |

#### ⑤ 总量核对

| 口径 | 数值 |
| :--- | ---: |
| Pre-Training（35M + 6.6M） | 41.60M |
| SFT（11M + 6.6M + 150K + 100K + 560K） | 18.41M |
| <mark class="hl-key">**Table 8 两阶段直接相加**</mark> | <mark class="hl-key">**60.01M**</mark> |
| 若扣除两阶段复用的 6.6M editing | 53.41M |
| <mark class="hl-key">**论文 Intro 声称**</mark> | <mark class="hl-key">**~50M samples**</mark> |

::: danger Table 8 加总是 60M，但论文反复声称 ~50M
<mark class="hl-key">**这是本文档中最大的一处数字不自洽**</mark>：Table 8 逐条相加得 <mark class="hl-key">**60.01M**</mark>；即便扣除两阶段复用的 6.6M editing，仍有 <mark class="hl-key">**53.41M**</mark>，<mark class="hl-trick">无论怎么算都对不上 "~50M"</mark>。

这个数字是论文**最核心的对外论据**（"仅用 ~50M 样本即超越 80B / 1.2B / 5B 样本的模型"），所以差额值得警惕。可能的解释（<mark class="hl-trick">以下均为推断，论文未说明，不可作为事实引用</mark>）：

- "~50M" 可能只统计了 <mark class="hl-trick">unique images</mark> 而非样本对/triplet（editing triplet 与 generation pair 共享图片时会被重复计数）；
- 可能排除了某类数据（如 6.6M editing 或 10M in-house）；
- 也可能 "~50M" 是取整后的粗略说法。

<mark class="hl-key">**引用 "~50M samples" 这个卖点时请注明：与附录 Table 8 的明细求和存在约 7–10M 的差距，论文未给出解释。</mark>
:::

::: info Table 8 值得单独记住的三件事
1. <mark class="hl-trick">**editing 数据两阶段逐字复用**</mark>（同一批 6.6M），SFT 阶段没为 editing 引入任何新数据源 —— 这在多阶段训练里并不常见。
2. <mark class="hl-trick">**internal data 是绝对主力**</mark>：SFT generation 的 10M/11M 来自内部双语数据，公开子集只占 295K。所以「仅用 50M 样本」这个说法的可复现性主要取决于那 10M 内部数据，外部无法获得。
3. <mark class="hl-trick">**Pre-Training 的 35M 全部是公开 web-scale 图文对</mark>（text-to-image-2M / LAION / Megalith / RedCaps / CC-12M，合计精确等于 35M），这部分是完全可复现的。
:::

## 5. Experiments

### 5.1 Evaluation Setup

### 5.2 Model Performance

### 5.3 Ablation Study

#### 5.3.1 Architecture Design

#### 5.3.2 RL Settings

## 6. Conclusion

## 附录

### A. Pre-Training & SFT Details

### B. Reinforcement Learning Details

## 7. 讨论 / 开放问题
