# 微软原生分辨率基模 Mage-Flow (Mage-VAE + Native-Resolution MMDiT + Diffusion-NFT)

> **标签**：`Vision` `Diffusion` `MMDiT` `Flow Matching` `VAE` `RL` `DiffusionNFT` `Distillation` `Efficiency`
> **更新时间**：2026-10-03
> **参考来源**：[Mage-Flow: An Efficient Native-Resolution Foundation Model for Image Generation and Editing (arXiv:2607.19064v2)](https://arxiv.org/abs/2607.19064) · [Project Page](https://microsoft.github.io/Mage) · [GitHub](https://github.com/microsoft/mage) · [HuggingFace](https://huggingface.co/collections/microsoft/mage)
> **原文**：本地 `Papers/Mage-Flow.pdf`（59 页，微软 Mage Team，2026-07-22）
> **精读重点**：§4 Data → §5.1 Pre-train/SFT/Edit → §5.2 Diffusion-NFT → §5.3 Distillation → §3.2/§3.3 Native-Res + Infra → §3.1 Mage-VAE
> **精读进度**：目录已搭建，内容待逐节精读填充

---

<!--
精读顺序（按学习大纲，非论文原章节序）：
  1. §4   Data Collection & Curation   ★★★ 重点精读
  2. §5.1 Pre-training + SFT           ★★★ 重点精读
  3. §5.2 Diffusion-NFT                 ★★★ 最高优先级
  4. §5.3 Few-step Distillation         ★★☆ 重点学，与 Z-Image 对照
  5. §3.2/§3.3 Native-Res + Infra       ★★☆ 中等深度
  6. §3.1 Mage-VAE                      ★☆☆ 选择性学
主线：Data → Progressive Pretrain/SFT → Edit Training → Diffusion-NFT → 4-step Turbo
旁支：Native-Resolution Packing / Training Efficiency ／ Mage-VAE
-->

## 0. 精读导航

| # | 主题 | 论文位置 | 优先级 | 状态 |
| :-: | :--- | :--- | :-: | :--- |
| 1 | Data Collection & Curation | §4 (P16–19) | ★★★ | ⬜ |
| 2 | Pre-training + SFT Recipe | §5.1 (P20–21) | ★★★ | ⬜ |
| 3 | Diffusion-NFT Post-training | §5.2 (P21–24) | ★★★ | ⬜ |
| 4 | Few-step Distillation | §5.3 (P24–26) | ★★☆ | ⬜ |
| 5 | Native-Resolution + Infrastructure | §3.2 / §3.3 (P13–16) | ★★☆ | ⬜ |
| 6 | Mage-VAE | §3.1 (P10–13) | ★☆☆ | ⬜ |
| 7 | Ablation / Tricks 总结 | 全文 | ★★☆ | ⬜ |

## 1. Introduction

## 2. Data Collection and Curation ★

### 2.1 T2I 数据总览：10B raw → 1.3B curated

Mage-Flow 的 generation 语料来自约 <mark class="hl-trick">**10B raw image–text pairs**</mark>（大规模开源数据集聚合），经**四大阶段** curation 后保留约 <mark class="hl-key">**1.3B high-quality image–text pairs**</mark>，再从中采样各阶段的 pre-training 子集。

| 阶段 | 作用（论文原话） |
| :--- | :--- |
| <mark class="hl-trick">Sample-level filtering</mark> | 移除损坏、低质、不安全或视觉上不适合的图像 |
| <mark class="hl-trick">Cross-sample deduplication</mark> | 抑制近重复的视觉模式 |
| <mark class="hl-trick">Multi-granularity captioning</mark> | 标准化文本监督 |
| <mark class="hl-trick">Concept-aware synthesis</mark> | 补充长尾图像 |

$$
\boxed{
\text{10B raw pairs}
\xrightarrow{\ \text{filter} \to \text{dedup} \to \text{caption} \to \text{synthesis}\ }
\text{1.3B curated pairs}\ (\text{保留率}\approx 13\%)
}
$$

![Mage-Flow Fig.8(a)：Text-to-Image 数据处理流水线。从左到右：10B 开源图像数据集 → **Sample-level Filtering**（分两层虚线框：File Information Filter 含 Broken / File Size / Resolution / Aspect ratio / Rotation；Image Content Filter 含 Saturation / Brightness / Grayscale / Blurry / Texture / Watermark / NSFW / Aesthetic / OCR / Entropy）→ **Cross-sample Deduplication** → **Multi-Granularity Captioning** → **Concept-aware Synthesis** → 右侧数据柱「~1.3B High-Quality Image-Text Data」。注意四阶段的**串行顺序**：filter 在 dedup 之前，synthesis 在最后。](/mageflow-fig8a-pipeline.png)

::: warning Fig. 8(a) 里一个容易被略过的顺序细节
流程是**严格串行**的：**filter → dedup → captioning → synthesis**。这意味着 <mark class="hl-key">**去重是在 captioning 之前做的**</mark> —— 用的是原始 web caption，不是 VLM 重写后的 caption。这是有意义的工程选择：<mark class="hl-trick">先用便宜的 SSCD 描述符把 10B 砍掉一大半，再让昂贵的 Qwen3-VL-32B 跑剩下的 1.3B</mark>。如果顺序反过来（先 caption 再 dedup），VLM 的推理成本会按 10B 的量级计。
:::

<mark class="hl-trick">**这一节的量级对照值得单独记**：10B → 1.3B 是约 <mark class="hl-key">**7.7 倍的收缩**</mark>，而 DeepGen 全程声称 ~50M 样本、Z-Image 是 314K H800·h 的量级。三家的数据哲学完全不同——<mark class="hl-key">**Mage-Flow 是"海量粗筛 + 严格阈值"，DeepGen 是"少而精 + 内部数据"**</mark>，而 10B 这个起点决定了它必须依赖自动化阈值（Table 5），因为人工不可能审 10B。</mark>

### 2.2 Sample-level filtering 与四阶段阈值（256 / 512 / 1024 / SFT）

<mark class="hl-trick">**过滤分两层：File Information Filter 与 Image Content Filter。**</mark>

| 层 | 论文列举的过滤器 | 作用 |
| :--- | :--- | :--- |
| <mark class="hl-trick">File Information</mark> | Broken、File Size、Resolution、Aspect ratio、Rotation | 移除损坏/近空文件、分辨率或像素数不足、极端宽高比、<mark class="hl-trick">**方向元数据错误的样本**</mark> |
| <mark class="hl-trick">Image Content</mark> | Brightness、Saturation、Grayscale、Blurry、Entropy、Texture、Watermark、OCR、Aesthetic、NSFW | 对解码后的图像打分 |

<mark class="hl-key">**论文对每个 content filter 的动机都写了一句，值得抄下来**</mark>：

- brightness / saturation → 移除过曝、欠曝或** unnaturally saturated** 的图像
- grayscale / blurry → 移除近单色或低锐度样本
- entropy / texture → <mark class="hl-trick">**抑制近空图像和「纹理样」的非语义模式**</mark>
- watermark / aesthetic / OCR / NSFW → 移除带水印、低质、<mark class="hl-trick">**document-like**</mark>、不安全的图像

**Table 5 是本节最硬的证据 —— 阈值沿四个阶段逐步收紧**（论文 §4.1，P17）：

| 指标 | 256² | 512² | 1024² | SFT |
| :--- | :--- | :--- | :--- | :--- |
| Pixel count $h\times w$ | ≥256² | ≥512² | ≥1024² | ≥1024² |
| $\min(h,w)$ | ≥128 | ≥256 | ≥512 | ≥512 |
| Aspect ratio | [0.1, 10.0] | [0.1, 10.0] | [0.1, 10.0] | [0.1, 10.0] |
| File size | ≥1KB | ≥1KB | ≥1KB | ≥1KB |
| NSFW score | ≤0.1 | ≤0.1 | ≤0.1 | ≤0.1 |
| <mark class="hl-trick">Watermark score</mark> | <mark class="hl-key">**<0.5**</mark> | <mark class="hl-key">**<0.3**</mark> | <mark class="hl-key">**<0.1**</mark> | <mark class="hl-key">**<0.05**</mark> |
| <mark class="hl-trick">Aesthetic-V2.5 score</mark> | <mark class="hl-key">**≥4.5**</mark> | <mark class="hl-key">**≥5.5**</mark> | <mark class="hl-key">**≥6.0**</mark> | <mark class="hl-key">**≥6.5**</mark> |
| OCR text-area ratio | ≤0.3 | ≤0.3 | ≤0.3 | ≤0.3 |
| OCR num. regions | ≤5 | ≤5 | ≤5 | ≤5 |

<mark class="hl-key">**只有两个阈值在动，而且动得非常克制**</mark>：

$$\text{Watermark: } 0.5 \to 0.3 \to 0.1 \to 0.05 \qquad \text{Aesthetic: } 4.5 \to 5.5 \to 6.0 \to 6.5$$

<mark class="hl-trick">**其余七项四阶段完全不变**（resolution、$\min(h,w)$、aspect ratio、file size、NSFW、两项 OCR）。</mark>

::: tip 这里有个反直觉的设计，值得记
**OCR 的两个阈值从头到尾不变（text-area ratio ≤0.3、num. regions ≤5），而 Mage-Flow 恰恰把 text rendering 当作核心能力。**

<mark class="hl-key">看起来矛盾，其实不是 —— 因为「文字渲染能力」主要由 §4.1 的 **concept-aware synthesis** 里**合成的 long-text rendering 数据**承担，而不是靠 web 数据里恰好有文字的图。</mark>web 图里的文字要么是文档扫描（水印/文档样，OCR 区域多），要么是随机招牌（不可控）。<mark class="hl-trick">**OCR 过滤器的实际作用是「剔除文档图」，而不是「筛出文字图」**</mark> —— 文字能力靠合成端注入。

这也解释了为什么 OCR 阈值在 SFT 阶段不收紧：<mark class="hl-key">**真正带文字能力的样本根本不走 web 这条路。</mark>
:::

::: tip Aesthetic-V2.5 是唯一真正在动的旋钮
<mark class="hl-trick">**Aesthetic-V2.5 从 4.5 一路提到 6.5**是这个 pipeline 的核心旋钮</mark>：早期（256²）放宽到 4.5 换取视觉覆盖，后期（SFT）收紧到 6.5 只留精品。论文的措辞是 <mark class="hl-key">*"Early stages retain broad visual coverage, while later stages emphasize higher resolution, stronger aesthetics, lower watermark probability, and cleaner image content."*</mark>

<mark class="hl-key">**但要注意：论文明确说 "Many filters are threshold-based rather than binary"**</mark> —— <mark class="hl-trick">threshold-based 而非 binary 意味着超阈值不等于丢弃</mark>，具体是软权重还是采样概率调整，<mark class="hl-trick">**论文未说明，不要自行假设是硬过滤**</mark>。
:::

### 2.3 跨数据集去重：SSCD + FAISS

论文的动机写得很直接：<mark class="hl-trick">**web-scale 数据在单个来源内部和跨数据集之间都高度冗余**</mark>（*"highly redundant both within individual sources and across datasets"*）。

做法是<mark class="hl-key">**两级去重，用同一套机制，但索引的生命周期不同**</mark>：

| 级别 | 索引 | 阈值 | 动作 |
| :--- | :--- | :--- | :--- |
| <mark class="hl-trick">Within-dataset</mark> | 每个数据集内部建索引 | cosine sim **> 0.9** | 分组为重复，<mark class="hl-key">**只保留质量最高的代表**</mark> |
| <mark class="hl-trick">Across-dataset</mark> | <mark class="hl-key">**持久化已接受图像的描述符索引**</mark> | 同一 0.9 | 复现已有样本的**新样本直接拒绝** |

<mark class="hl-key">**SSCD 描述符的选择理由论文写明了**：*"robust to **re-encoding, cropping, resizing, and light edits**"*</mark> —— 这四种正是 web 图片的典型重复方式。

<mark class="hl-trick">**两个容易被漏掉的补充机制**</mark>：

1. <mark class="hl-key">**超大簇封顶（Very large clusters are capped）**</mark> —— 论文的动机写得很具体：*"to suppress repeated **web templates** such as **stock photos, banners, and product layouts**"*。<mark class="hl-trick">**注意这解决的不是"重复"问题，而是"模板"问题** —— 一张图有 500 个变体，SSCD 会全判为重复；但如果只是简单封顶，会误伤真正有细节差异的样本。论文没说封顶的具体策略（是随机采样还是按质量取 top-k）。</mark>
2. <mark class="hl-key">**对 held-out benchmark 建索引做匹配**</mark> —— *"We also match against a **held-out benchmark index** to reduce **evaluation contamination**"*。<mark class="hl-trick">**这是评测去污染，和训练去重是两个目的，但放在同一个索引机制里做**</mark> —— 成本几乎为零（只是多一批 query），收益是避免训练集泄漏到 benchmark。

::: info 这一节的可迁移结论
**「去重」在 web-scale 训练里不只是省算力，它同时是三种能力的保护措施**：

| 保护对象 | 机制 |
| :--- | :--- |
| <mark class="hl-trick">多样性</mark> | 簇内只留最优代表，簇上限封顶模板 |
| <mark class="hl-key">训练分布不被少数源主导</mark> | 跨数据集持久索引 |
| <mark class="hl-key">评测可信度</mark> | held-out benchmark 索引做去污染 |

<mark class="hl-key">**0.9 这个 cosine 阈值是可借的起点**，但依赖 SSCD 而非 CLIP —— <mark class="hl-trick">CLIP 描述符对"内容相同但风格不同"的图会误判为重复，而 SSCD 是 copy-detection 专用，对裁剪/缩放更鲁棒</mark>。</mark>
:::

### 2.4 Caption：Qwen3-VL 多粒度

<mark class="hl-trick">**captioner 是 Qwen3-VL-32B-Instruct**</mark>，为每张图生成**四个粒度**的 caption（论文说 "Following [22]"）：

| 粒度 | 内容 | 用途（论文原话） |
| :--- | :--- | :--- |
| <mark class="hl-key">**Phrase-level**</mark> | 概念短语列表 | <mark class="hl-trick">**用于 concept statistics**</mark> |
| <mark class="hl-trick">**Entity-level**</mark> | 主要物体与属性 | 训练 prompt |
| <mark class="hl-trick">**Composition-level**</mark> | 空间布局与关系 | 训练 prompt |
| <mark class="hl-trick">**Photographic**</mark> | 风格、光照、视角、氛围、精细视觉细节 | 训练 prompt |

<mark class="hl-key">**"During training, the model samples from these descriptive caption channels so that it learns to follow prompts with **different lengths and specificity**."**</mark>

![Mage-Flow Fig.8(b)：Multi-Granularity Caption Framework。左侧两个输入图（鸽子 / 画框）经同一个 System Prompt（*"You are a world-class multi-granularity image captioning expert... produce a structured, detailed, and objective description"*）送给 Qwen，右侧对每张图输出四列：**Phrase**（如 `four pigeons, urban scene, wet pavement, puddle reflection, green fence, overcast lighting, concrete steps, muted colors`）、**Entity**（`Four pigeons standing in a large puddle on a paved surface.`）、**Composition**（街级视角描述 + 水面倒影 + 背景木栅栏与混凝土台阶的相对位置）、**Photographic**（`This is a photorealistic, outdoor street photography shot taken at eye level...`）。画框那例的 Photographic 甚至捕捉到了 `'MARIANNA'` 印刷体与 `'Tommy Thompson'` 脚本体两种字体。](/mageflow-fig8b-caption-granularity.png)

<mark class="hl-key">**这个设计的真正价值是"一图四用"**：</mark>

- Phrase-level → 直接喂给 §2.6 的 concept-aware sampling 做分布统计（<mark class="hl-trick">**如果只有完整 caption，就没法低成本地给 1.3B 图做概念计数**</mark>）
- Entity / Composition / Photographic → 训练时按概率采样

<mark class="hl-trick">**"different lengths and specificity" 这个目标是可验证的**：</mark>Entity 级通常 1–2 句，Photographic 级 5–8 句。模型同时见过两端，就不会被绑死在一种 prompt 长度上 —— <mark class="hl-key">**这是 prompt-following 泛化的前置条件**，比后面任何 RL 都更基础。</mark>

::: warning 论文对 text-rich 图像只有一句处理
*"For text-rich images, the captioner is **explicitly prompted to recognize visible text and convert it into rendering instructions**."*

<mark class="hl-trick">**只有这一句**</mark>：没有给 system prompt 的具体措辞、没有说识别失败如何处理、没有给 text-rich 图的占比。<mark class="hl-key">从 Fig. 8(b) 画框那例能看出 Photographic caption 确实会记录可见文字（含字体描述），但这只是单个案例，不足以推断通用策略。</mark>
:::

### 2.5 Concept-aware synthesis

<mark class="hl-trick">**动机：web 数据在若干 capability-critical 领域是稀疏的**。论文列出的五类长尾（§4.1, P18）：</mark>

1. <mark class="hl-key">**long-text rendering**</mark>
2. <mark class="hl-trick">rare objects**</mark>
3. <mark class="hl-trick">uncommon attributes**</mark>
4. <mark class="hl-trick">structured layouts**</mark>
5. <mark class="hl-trick">under-represented styles**</mark>

<mark class="hl-key">**注意第 1 条和 §2.2 的 OCR 阈值不变是同一件事的两面**：web 端把文档图滤掉（OCR 阈值固定），文字能力改由这里合成的 long-text 数据承担。</mark>

构造的补充数据：

- <mark class="hl-trick">**synthetic text-rendering samples**，维度明确列出：<mark class="hl-key">**diverse fonts、layouts、languages、colors、backgrounds**</mark></mark>
- <mark class="hl-trick">**additional image–text pairs covering rare concepts and compositional cases**</mark>

<mark class="hl-key">**一个必须注意的工程约束：所有合成样本都要走一遍和 web 数据完全相同的过滤与质检流程**</mark>（*"All supplemental samples are passed through **the same filtering and quality-control pipeline** before being merged"*）。

<mark class="hl-trick">**这一点很容易被忽略但很关键** —— 它意味着 <mark class="hl-key">**Table 5 的 Aesthetic-V2.5 ≥6.5 和 watermark <0.05 同样在约束合成数据**</mark>。合成数据通常 aesthetic 分数虚高（生成图"太干净"），这条约束是反向的校准。DeepGen 的做法不同：它用 Gemini 2.5 Pro 编属性、Qwen-Image 合成图，<mark class="hl-trick">但没有说明合成数据是否过同一套过滤</mark>。</mark>

### 2.6 Balancing 与 concept reweighting

<mark class="hl-trick">**用 §2.4 的 phrase-level caption 估计合并后语料的概念分布**（Fig. 9a）</mark>。这正是 phrase-level caption 存在的理由。

![Mage-Flow Fig.9(a)：generation pre-training 数据的概念分布（太阳图，内环为大类，外环为细类）。**Object & Products 31.3%**（Apparel & Accessories 16.8% / Furniture 5.8% / Packaging 3.5% / Electronics 2.8% / Vehicles 2.4%）、**Scene & Place 26.1%**（Landscape 14.3% / Indoor 7.7% / Cityscape 4.0%）、**People 19.0%**（Person 7.2% / Appearance 6.2% / Portrait 5.7%）、**Living & Food 9.0%**（Food & Drink 4.1% / Plants 2.5% / Animals 2.4%）、**Design 8.8%**（Poster & UI 5.0% / Cartoon 2.1% / Art 1.8%）、**Synthetic 5.8%**（English Text 1.9% / Chinese Text 1.8% / Others 2.1%）。](/mageflow-fig9a-concept-dist.png)

<mark class="hl-key">**论文对这张图的判断很诚实**：*"The resulting distribution remains **long-tailed**"* —— 并没有说合成就均衡了。</mark>论文只点明两个最大的粗类：

- <mark class="hl-trick">**Object & Products（31.3%）和 Scene & Place（26.1%）是两个最大的粗域**</mark>
- <mark class="hl-key">**Design 和 Synthetic 提供 layout、product-style、poster-style 和 text-rendering 能力的关键覆盖**</mark>

<mark class="hl-key">**从外环能读出两个论文没点破的事实**</mark>：

| 观察 | 数字 | 含义 |
| :--- | :--- | :--- |
| <mark class="hl-trick">**Apparel & Accessories 单独占 16.8%**</mark> | 是第三大细类（≈8×） | 这是**纯长尾污染** —— 最大的细类和"通用视觉能力"关系不大，典型的 web 图文对偏置 |
| <mark class="hl-trick">**Synthetic 整类只 5.8%**</mark> | English 1.9% + Chinese 1.8% = **3.7%** | <mark class="hl-key">**文字渲染数据只占语料 3.7%**</mark>，却要撑起 CVTG-2K 这类榜单 |

<mark class="hl-key">**应对手段只有一招：concept-aware sampling**</mark>

> *"we apply **concept-aware sampling** to reduce the dominance of **frequent objects, natural scenes, and common photorealistic styles**."*

$$
\text{sampling weight} \propto \text{inverse function of concept frequency}
$$

<mark class="hl-trick">**论文只给了这句话的定性描述，没有给具体函数形式**</mark> —— 是 $\propto 1/f$、$\propto 1/\sqrt{f}$ 还是温度采样，<mark class="hl-key">**论文未说明，不要自行假设**</mark>。也没给 reweighting 后的实际分布（Fig. 9a 是 reweighting **之前**的原始分布）。

<mark class="hl-trick">**跨阶段还有一条统一的收紧线索**，论文在 §4.1 结尾总结：</mark>

> *"Across training stages, we progressively tighten **filtering thresholds** and strengthen **reweighting**, moving from broad visual-prior learning in early stages to cleaner and more capability-focused learning in later stages."*

<mark class="hl-key">**所以收紧的是两件事，不只是 Table 5 的阈值**</mark>：阈值（可查）+ reweighting 强度（<mark class="hl-trick">**不可查，论文没给任何数字**</mark>）。

::: warning 本节最关键的未公开细节
- <mark class="hl-trick">**concept-aware sampling 的具体函数形式与超参** —— 完全没给。这是本节唯一真正影响训练分布的旋钮，缺了它整个 balancing 就无法复现。</mark>
- <mark class="hl-trick">**"strengthen reweighting" 的量化** —— 早期到后期的 reweighting 强度比值。</mark>
- <mark class="hl-trick">**合成数据的占比** —— 论文只给了合并后的分布（Fig. 9a），**没有给 1.3B 里 web 与 synthetic 各占多少**。因此"Synthetic 5.8%"这个数字<mark class="hl-key">既包含合成也包含 web 里本来就有的合成风格图（AI 生成的艺术图等），无法反推合成注入量</mark>。</mark>
- <mark class="hl-trick">**Supplementary data 的来源与生成模型**</mark> —— <mark class="hl-trick">只说 "we construct targeted supplemental data"，<mark class="hl-key">**没有说是哪个 VLM 做的属性组合、哪个图像模型合成的图**</mark>。这和 DeepGen §4.4 明确写 Gemini 2.5 Pro + Qwen-Image 形成了鲜明对比 —— Mage-Flow 在这里更保守。</mark>
:::

::: info §2.1–2.6 对照 Z-Image / DeepGen 的数据观
| | **Mage-Flow** | **DeepGen** | **Z-Image** |
| :--- | :--- | :--- | :--- |
| 起点规模 | <mark class="hl-trick">**10B raw**</mark> | ~50M（声称） | 真实数据为主，无具体总量 |
| 收敛比 | <mark class="hl-trick">**10B → 1.3B（7.7×）**</mark> | 未给 | 未给 |
| 去重 | <mark class="hl-key">**SSCD + FAISS，cos>0.9，两级**</mark> | 未提 | Cross-modal Vector Engine（语义去重） |
| Caption | <mark class="hl-trick">**Qwen3-VL-32B 四粒度**</mark> | BLIP3-o / ShareGPT-4o 等现成 | 自建 Image Captioner（三级） |
| 阈值 | <mark class="hl-key">**Table 5 全表公开**</mark> | 无 | 过滤规则未公开 |
| 长尾补齐 | <mark class="hl-trick">**合成 + concept-aware sampling**</mark> | 无长尾机制（内部 10M 兜底） | Active Curation Engine 闭环 |
| 合成数据过同套过滤 | <mark class="hl-key">**是（明确写出）**</mark> | 未说明 | — |

<mark class="hl-key">**Mage-Flow 是三家里唯一把「过滤阈值」写成完整表格的**</mark> —— 这是它作为工业报告最诚实的地方，也是最直接可借鉴的产物。<mark class="hl-trick">而 concept-aware sampling 的函数形式缺失，是它相较 Z-Image Active Curation Engine 明显更弱的一环（Z-Image 那套是闭环 HITL，Mage-Flow 这套是开环加权）。</mark>
:::

### 2.7 Editing 数据：90M raw triples → 45M

### 2.8 Expert voting 过滤（三个 Qwen3.5-9B）

### 2.9 19 类 edit taxonomy 与 balancing

### 2.10 数据环节小结与未公开细节

## 3. Pre-training and Supervised Fine-tuning ★

### 3.1 T2I Progressive Curriculum 总览

### 3.2 逐阶段递进：resolution / quality / threshold / reweighting

### 3.3 Editing 两阶段混合训练

### 3.4 为什么 Edit 训练要混 Generation，比例怎么配

### 3.5 本节小结与未公开细节

## 4. Diffusion-NFT Post-training ★

### 4.1 Diffusion-NFT 机制：为什么不是 trajectory likelihood

### 4.2 T2I RL：20K RL prompts 与三类路由

### 4.3 Capability mixture 两阶段：1:1:1 → 2:4:1

### 4.4 Reward 配置

### 4.5 Editing RL：30K edit prompts + RationalRewards + 4:1

### 4.6 与 DeepGen「RL 无 Edit」的对照

### 4.7 与已有 NFT 笔记的对照（FireRed / Swift-Image）

### 4.8 本节小结与未公开细节

## 5. Few-step Distillation ★

### 5.1 Decoupled DMD（与 Z-Image 对照）

### 5.2 Adversarial Perceptual Guidance：DINOv2 / CLIP + discriminator

### 5.3 为什么 4-step 还需要 perceptual adversarial regularization

### 5.4 蒸馏数据：Generation 200K ／ Editing 250K + 3:1

### 5.5 Ablation：哪些提升、哪些没有

### 5.6 本节小结与未公开细节

## 6. Native-Resolution MMDiT + Training Infrastructure ★

### 6.1 为什么不用传统 resolution bucket

### 6.2 Native Packing：variable-length image + text token 混 pack

### 6.3 FlashAttention variable-length kernel + per-sample 2D RoPE

### 6.4 CFG cond/uncond 单次 packed forward

### 6.5 Fused CUDA Kernels（Mage-VAE / Qwen3-VL / MMDiT）

### 6.6 MFU 13.88% → 29.28% 与 2.48× step-time speedup

### 6.7 本节小结与未公开细节

## 7. Mage-VAE ★

### 7.1 为什么 VAE 是高分辨率训练 / 4-step 推理的瓶颈

### 7.2 One-step diffusion encoder / decoder

### 7.3 Anchor-latent KL regularization

### 7.4 三阶段 VAE training 与 latent space 对齐 FLUX.2-VAE

### 7.5 Encode / Decode MACs 与 1K / 2K / 4K 优势

### 7.6 本节小结与未公开细节

## 8. Ablation 与 Tricks 有效性总结

### 8.1 Generation 数据混进 Edit 是否有效

### 8.2 Adversarial guidance 是否有效

### 8.3 不同 RL capability mixture 是否合理

### 8.4 哪些 trick 真 work

## 9. Experiments

### 9.1 Experimental Setup

### 9.2 Quantitative Results

### 9.3 Qualitative Results

## 10. 讨论 / 开放问题

## 附录

### A. 论文章节 → 本笔记章节对照

### B. 三个变体（Base / RL-aligned / Turbo）
