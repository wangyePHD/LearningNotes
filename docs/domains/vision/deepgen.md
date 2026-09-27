# 轻量统一多模态模型 DeepGen 1.0 (SCB + 三阶段训练 + MR-GRPO)

> **标签**：`Vision` `Unified Model` `VLM-DiT` `Flow Matching` `RL` `GRPO` `Data-centric`
> **更新时间**：2026-09-27
> **参考来源**：[DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing (arXiv:2602.12205v2)](https://arxiv.org/abs/2602.12205) · [GitHub: DeepGenTeam/DeepGen](https://github.com/DeepGenTeam/DeepGen) · [HuggingFace: DeepGenT](https://huggingface.co/DeepGenT) · [Datasets](https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0)
> **原文**：本地 `Papers/DeepGen.pdf`（21 页，正文 18 页 + 附录 A/B）
> **精读重点**：§3 Training（data train）+ §3.3 RL + §4 Data
> **精读进度**：§4 Data ★ ✅ ｜ §3.1 Alignment Pre-Training ✅ ｜ §3.2 Joint SFT ✅ ｜ **§3.3 MR-GRPO ★** ｜ §2 Architecture ｜ §5 Experiments ｜ §6 Conclusion（笔记随学习逐节增补）

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
1. <mark class="hl-trick">**editing 数据两阶段逐字复用**（同一批 6.6M），SFT 阶段没为 editing 引入任何新数据源 —— 这在多阶段训练里并不常见。
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
