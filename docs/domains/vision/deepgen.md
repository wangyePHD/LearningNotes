# 轻量统一多模态模型 DeepGen 1.0 (SCB + 三阶段训练 + MR-GRPO)

> **标签**：`Vision` `Unified Model` `VLM-DiT` `Flow Matching` `RL` `GRPO` `Data-centric`
> **更新时间**：2026-09-27
> **参考来源**：[DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing (arXiv:2602.12205v2)](https://arxiv.org/abs/2602.12205) · [GitHub: DeepGenTeam/DeepGen](https://github.com/DeepGenTeam/DeepGen) · [HuggingFace: DeepGenT](https://huggingface.co/DeepGenT) · [Datasets](https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0)
> **原文**：本地 `Papers/DeepGen.pdf`（21 页，正文 18 页 + 附录 A/B）
> **精读重点**：§3 Training（data train）+ §3.3 RL + §4 Data
> **精读进度**：§4 Data ★ ✅ ｜ §3 Training（3.1–3.3，含 3.3 RL）★ ｜ §2 Architecture ｜ §5 Experiments ｜ §6 Conclusion（笔记随学习逐节增补）

---

<!-- 精读导航：按论文章节序推进，重点节已标注 ★ -->

## 1. Introduction

> 尚未记录。核心待答问题：5B 如何在 WISE / DPG-Bench / UniREditBench 上对抗 27B–80B？

## 2. Model Architecture

> 尚未记录。SCB（Stacked Channel Bridging）：3B VLM + 2B DiT = 5B，多层 hidden states + learnable think tokens。

## 3. Training ★

> 尚未记录。本节为重点。

### 3.1 Stage 1: Alignment Pre-Training

> 尚未记录。只训 SCB connector + think tokens，冻结 VLM 与 DiT。

### 3.2 Stage 2: Joint Supervised Fine-Tuning

> 尚未记录。解冻 DiT + VLM 上 LoRA（rank 64 / α 128）。

### 3.3 Stage 3: Reinforcement Learning ★

> 尚未记录。本节为重点。MR-GRPO：混合奖励函数 + 噪声保持随机采样。

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

<mark class="hl-trick">**General Editing 的数据组织更值得记，因为它基本就是一张开源 editing 数据地图**</mark>。DeepGen 收集的是 <mark class="hl-trick">**image–instruction–image triplets**</mark>：

| 数据源 | 规模 |
| :--- | ---: |
| GPT-Image-Edit | 1.5M |
| X2I2 | 1.6M |
| UniWorld-Edit set | 1.2M |
| NHR-Edit | 720K |
| Pico-Banana | 250K |
| Nano-banana-consist | 150K |
| ShareGPT-4o-Image-Edit set | 50K |
| OpenGPT4o-Image-Edit set | 40K |
| <mark class="hl-trick">in-house editing samples（中英文）</mark> | 1.1M |
| **合计（Table 8）** | **≈ 6.6M** |

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

## 5. Experiments

> 尚未记录。

### 5.1 Evaluation Setup

> 尚未记录。

### 5.2 Model Performance

> 尚未记录。

### 5.3 Ablation Study

> 尚未记录。

#### 5.3.1 Architecture Design

> 尚未记录。

#### 5.3.2 RL Settings

> 尚未记录。Table 7：各变体统一训练 1,000 steps 后评测。

## 6. Conclusion

> 尚未记录。

## 附录

### A. Pre-Training & SFT Details

> 尚未记录。Table 8 数据明细、Table 9 超参（lr / batch / LoRA / 分辨率）。

### B. Reinforcement Learning Details

> 尚未记录。噪声保持随机采样（式 6）、奖励函数组合、辅助 SFT 数据。

## 7. 讨论 / 开放问题

> 待学习过程中沉淀。
