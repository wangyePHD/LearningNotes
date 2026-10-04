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
| 1 | Data Collection & Curation | §4.1 (P17–18) + §4.2 (P18–19) | ★★★ | 🟡 §2.1–2.2 |
| 2 | Pre-training + SFT Recipe | §5.1 (P20–21) | ★★★ | ⬜ |
| 3 | Diffusion-NFT Post-training | §5.2 (P21–24) | ★★★ | ⬜ |
| 4 | Few-step Distillation | §5.3 (P24–26) | ★★☆ | ⬜ |
| 5 | Native-Resolution + Infrastructure | §3.2 / §3.3 (P13–16) | ★★☆ | ⬜ |
| 6 | Mage-VAE | §3.1 (P10–13) | ★☆☆ | ⬜ |
| 7 | Ablation / Tricks 总结 | 全文 | ★★☆ | ⬜ |

## 1. Introduction

## 2. Data Collection and Curation ★

::: info 笔记小节 ↔ 论文章节逐段对照（本节结构严格镜像论文 §4.1 / §4.2）
论文这两节的段落由**粗体导语**划分，本笔记的小节标题即采用同样的导语，因此可以逐段对照：

| 笔记小节 | 论文章节与段首粗体导语 | 论文页 | 状态 |
| :--- | :--- | :--- | :--- |
| §2.1 | §4.1 导语段（无粗体导语，以 *"The Mage-Flow generation corpus is built from..."* 起） | P17 | 🟡 部分 |
| §2.2 | §4.1 **`Sample-level filtering.`** | P17 | 🟡 部分 |
| §2.3 | §4.1 **`Cross-sample deduplication.`** | P17 | ⬜ |
| §2.4 | §4.1 **`Multi-granularity captioning.`** | P18 | ⬜ |
| §2.5 | §4.1 **`Concept-aware synthesis and balancing.`** | P18 | ⬜ |
| §2.6 | §4.2 导语段（以 *"The Mage-Flow-Edit corpus consists of..."* 起） | P18 | ⬜ |
| §2.7 | §4.2 **`Editing data synthesis.`** | P19 | ⬜ |
| §2.8 | §4.2 **`VLM-based dataset filtering.`** | P19 | ⬜ |
| §2.9 | §4.2 **`Edit-type tagging and balancing.`** | P19 | ⬜ |
| §2.10 | <mark class="hl-trick">**笔记自加，非论文章节**</mark> | — | ⬜ |
:::

::: warning 为什么 §2.1 与 §2.6 是「导语段」
<mark class="hl-trick">**论文这两节各有一段没有粗体导语的开场段**</mark>，它们的内容分别是总览数字与 Fig. 10 流程概述，<mark class="hl-key">**在笔记里被显式标为「导语段」以免和论文的粗体段落数对不上**</mark>。

所以：<mark class="hl-trick">**论文 §4.1 = 1 导语段 + 4 个粗体段，笔记 §2.1–§2.5；论文 §4.2 = 1 导语段 + 3 个粗体段，笔记 §2.6–§2.9**</mark>。总计论文 10 段（含 2 个导语段），笔记 §2.1–§2.9，另加 §2.10 为笔记自加。
:::

### 2.1 总览：10B raw image–text pairs → ~1.3B curated pairs

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

### 2.2 Sample-level filtering（`Sample-level filtering.`）

#### 阶段一内部还有两层：先文件层面，再图像内容

<mark class="hl-trick">**Fig. 8(a) 里 `Sample-level Filtering` 画成了一个带两层虚线框的大盒子 —— 这不是排版，是两级独立的过滤。**</mark>论文正文写得很明确：

| 层 | 过滤器（Fig. 8(a) 方框数） | 判定依据 | <mark class="hl-trick">**是否需要解码像素**</mark> |
| :--- | :--- | :--- | :--- |
| <mark class="hl-trick">**File Information Filter**</mark> | **5 个**：Broken、File Size、Resolution、Aspect ratio、Rotation | <mark class="hl-key">**文件元数据 / header**</mark> | <mark class="hl-key">**否**</mark> |
| <mark class="hl-trick">**Image Content Filter**</mark> | **10 个**：Saturation、Brightness、Grayscale、Blurry、Texture、Watermark、NSFW、Aesthetic、OCR、Entropy | <mark class="hl-key">**解码后的像素**</mark> | <mark class="hl-key">**是**</mark> |

<mark class="hl-key">**论文的关键措辞是这一句**</mark>（§4.1, P17）：

> *"File-information filters remove corrupted or near-empty files, images with insufficient resolution or pixel count, extreme aspect ratios, and samples with **incorrect orientation metadata**. **Image-content filters then score decoded images** for visual quality and safety."*

<mark class="hl-trick">**`then`（顺序）+ `decoded`（必须先解码）这两个词是全部证据**</mark>：<mark class="hl-key">**第一层的 5 个过滤器全部只需要读文件头，第二层的 10 个必须真正解码像素才能打分**</mark>。

- File size → 纯文件属性
- Resolution / Aspect ratio → 图像 header 里就写着
- Rotation → <mark class="hl-trick">**EXIF orientation 元数据**</mark>，论文说的 *"incorrect **orientation metadata**"* 就是它
- Broken → header 解析失败即可判定

$$
\underbrace{\text{Broken, File Size, Resolution, Aspect ratio, Rotation}}_{\text{header 级，5 个}}
\ \xrightarrow{\ \text{不解码}\ }\ 
\underbrace{\text{Brightness, ..., Entropy}}_{\text{像素级，10 个}}
$$

<mark class="hl-key">**所以这个先后顺序是一次成本排序，不是逻辑分类**</mark>：<mark class="hl-trick">先用几乎零成本的元数据检查把大批样本挡掉，只有存活下来的才付「解码 + 打分」的代价</mark>。

::: warning 「两层各自的淘汰量」论文完全没给
<mark class="hl-trick">这是一个很反常的缺口**</mark>：论文交代了总收缩（10B → 1.3B），也交代了四阶段的顺序，<mark class="hl-key">**但从未说明这两层各自筛掉了多少**</mark>。

$$
\underbrace{10B}_{\text{raw}}
\ \xrightarrow{\ \text{File Info 层}\ }\
\underbrace{?}_{\text{论文未给}}
\ \xrightarrow{\ \text{Image Content 层}\ }\
\underbrace{?}_{\text{论文未给}}
\ \xrightarrow{\ \text{dedup} \to \text{caption} \to \text{synthesis}\ }
\underbrace{1.3B}_{\text{curated}}
$$

<mark class="hl-trick">**而这个数字恰恰决定了「元数据优先」这套设计值不值**</mark>：如果 File Info 层已经砍掉 90%，那"先便宜后昂贵"是对的；如果它只砍掉 20%，<mark class="hl-key">**真正的算力大头在 Image Content 层，而那一层反而没有可优化的顺序可言**</mark>。论文不写，读者就无法判断这个设计的收益量级。
:::

<mark class="hl-trick">**另一个值得注意的点：两层都有「防病态」性质的检查，但分工不同**</mark>。File Info 层管的是<mark class="hl-key">**文件层面的异常**</mark>（坏文件、空文件、方向元数据错）；Image Content 层的 entropy / texture 管的是<mark class="hl-trick">**内容层面的空洞与无意义**</mark>（near-empty images、texture-like non-semantic patterns）。<mark class="hl-key">**前者防的是"文件坏了"，后者防的是"文件没坏但内容没信息"**</mark> —— 后者用纯元数据检查是做不到的，所以两层不能互相替代。


<mark class="hl-trick">**这一节的量级对照值得单独记**：10B → 1.3B 是约 <mark class="hl-key">**7.7 倍的收缩**</mark>，而 DeepGen 全程声称 ~50M 样本、Z-Image 是 314K H800·h 的量级。三家的数据哲学完全不同——<mark class="hl-key">**Mage-Flow 是"海量粗筛 + 严格阈值"，DeepGen 是"少而精 + 内部数据"**</mark>，而 10B 这个起点决定了它必须依赖自动化阈值（Table 5），因为人工不可能审 10B。</mark>

<mark class="hl-trick">**这一步的性质：逐条独立判断，不看其他样本。**</mark>论文原文 *"We first filter each **image–caption pair independently**"* —— 所以它和 §2.3 的 cross-sample dedup 是两种不同性质的过滤：<mark class="hl-key">**这里是「这条样本自身好不好」，那里是「这条样本是不是别条的重复」**</mark>。

过滤器分两层，共 <mark class="hl-trick">**15 个**</mark>（见 Fig. 8(a) 的方框数）：

| 层 | 过滤器 | 论文给出的动机 |
| :--- | :--- | :--- |
| <mark class="hl-trick">File Information</mark><br>（5 个） | Broken、File Size、Resolution、Aspect ratio、Rotation | 移除损坏/近空文件、分辨率或像素数不足、极端宽高比、<mark class="hl-trick">**方向元数据错误的样本**</mark> |
| <mark class="hl-trick">Image Content</mark><br>（10 个） | Brightness、Saturation、Grayscale、Blurry、Entropy、Texture、Watermark、OCR、Aesthetic、NSFW | 对**解码后**的图像打分 |

<mark class="hl-key">**论文给每个 content filter 都写了一句理由，值得抄**</mark>：

- brightness / saturation → 过曝、欠曝、<mark class="hl-trick">** unnaturally saturated**</mark>
- grayscale / blurry → 近单色、低锐度
- entropy / texture → <mark class="hl-trick">**抑制近空图像和「纹理样」的非语义模式**</mark>
- watermark / aesthetic / OCR / NSFW → 带水印、低质、<mark class="hl-trick">**document-like**</mark>、不安全

![Mage-Flow Table 5：四阶段 sample-level filtering 阈值完整表（论文 §4.1，P17）。九行四列：Pixel count、min(h,w)、Aspect ratio、File size、NSFW score 五行四阶段完全相同；**Watermark score** 0.5→0.3→0.1→0.05 与 **Aesthetic-V2.5 score** 4.5→5.5→6.0→6.5 是唯一逐阶段收紧的两行；OCR text-area ratio ≤0.3 与 OCR num. regions ≤5 也全程不变。](/mageflow-tab5-filter-thresholds.png)

| 指标 | 256² | 512² | 1024² | SFT | 是否变化 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pixel count $h\times w$ | ≥256² | ≥512² | ≥1024² | ≥1024² | <mark class="hl-trick">末段持平</mark> |
| $\min(h,w)$ | ≥128 | ≥256 | ≥512 | ≥512 | <mark class="hl-trick">末段持平</mark> |
| Aspect ratio | [0.1, 10.0] | [0.1, 10.0] | [0.1, 10.0] | [0.1, 10.0] | 冻结 |
| File size | ≥1KB | ≥1KB | ≥1KB | ≥1KB | 冻结 |
| NSFW score | ≤0.1 | ≤0.1 | ≤0.1 | ≤0.1 | 冻结 |
| <mark class="hl-key">Watermark score</mark> | <mark class="hl-key">**<0.5**</mark> | <mark class="hl-key">**<0.3**</mark> | <mark class="hl-key">**<0.1**</mark> | <mark class="hl-key">**<0.05**</mark> | <mark class="hl-trick">**全程收紧 10×**</mark> |
| <mark class="hl-key">Aesthetic-V2.5 score</mark> | <mark class="hl-key">**≥4.5**</mark> | <mark class="hl-key">**≥5.5**</mark> | <mark class="hl-key">**≥6.0**</mark> | <mark class="hl-key">**≥6.5**</mark> | <mark class="hl-trick">**全程 +2.0**</mark> |
| OCR text-area ratio | ≤0.3 | ≤0.3 | ≤0.3 | ≤0.3 | 冻结 |
| OCR num. regions | ≤5 | ≤5 | ≤5 | ≤5 | 冻结 |

#### 九行里只有两行在动

<mark class="hl-trick">**这是本节最重要的观察**：Table 5 看起来是一张「逐阶段收紧」的表，<mark class="hl-key">**但真正随阶段变化的只有 Watermark 和 Aesthetic-V2.5 两行**</mark>。</mark>其余七行四阶段逐字相同。

$$
\text{Watermark: } 0.5 \to 0.05 \ (\text{10× 收紧}) \qquad \text{Aesthetic: } 4.5 \to 6.5 \ (+2.0)
$$

<mark class="hl-key">**Table 5 的 caption 把这个意图写成了四个词**：*"higher resolution, **stronger aesthetics**, **lower watermark probability**, and cleaner image content"*</mark> —— <mark class="hl-trick">四个词里只有两个（aesthetics、watermark）有对应的可动阈值</mark>，"higher resolution" 靠前两行实现，而 "cleaner image content" <mark class="hl-key">**没有任何一行在动</mark>。

::: warning "cleaner image content" 这一项在表里是空的
<mark class="hl-trick">论文 caption 承诺了四个收紧方向，但 Table 5 只能兑现两个半。</mark>

| caption 承诺 | 对应行 | 状态 |
| :--- | :--- | :--- |
| higher resolution | Pixel count、$\min(h,w)$ | <mark class="hl-key">✅ 逐阶段收紧（但 SFT 段持平）</mark> |
| stronger aesthetics | Aesthetic-V2.5 | <mark class="hl-key">✅ 逐阶段收紧</mark> |
| lower watermark probability | Watermark score | <mark class="hl-key">✅ 逐阶段收紧</mark> |
| <mark class="hl-trick">cleaner image content</mark> | <mark class="hl-trick">**无对应行**</mark> | <mark class="hl-trick">**未兑现**</mark> |

<mark class="hl-key">**"cleaner image content" 本该由 brightness / saturation / blurry / grayscale / entropy / texture 这六个 filter 承担，但它们一个阈值都没公开**</mark>（见下）。所以这一项在表里是空的 —— <mark class="hl-trick">不是"不需要收紧"，是"没告诉你收紧到哪"</mark>。
:::

#### Table 5 只覆盖了 15 个过滤器中的 7 个

<mark class="hl-trick">**把 Fig. 8(a) 的方框和 Table 5 的行对一遍，会发现一个论文没提的缺口**</mark>：

| Fig. 8(a) 过滤器 | Table 5 是否有阈值 |
| :--- | :--- |
| Resolution | <mark class="hl-key">✅ Pixel count + $\min(h,w)$ 两行</mark> |
| Aspect ratio | ✅ |
| File Size | ✅ |
| NSFW | ✅ |
| Watermark | ✅ |
| Aesthetic | ✅ |
| OCR | <mark class="hl-key">✅ text-area ratio + num. regions 两行</mark> |
| <mark class="hl-trick">**Broken**</mark> | <mark class="hl-trick">**❌ 无**</mark> |
| <mark class="hl-trick">**Rotation**</mark> | <mark class="hl-trick">**❌ 无**</mark> |
| <mark class="hl-trick">**Saturation**</mark> | <mark class="hl-trick">**❌ 无**</mark> |
| <mark class="hl-trick">**Brightness**</mark> | <mark class="hl-trick">**❌ 无**</mark> |
| <mark class="hl-trick">**Grayscale**</mark> | <mark class="hl-trick">**❌ 无**</mark> |
| <mark class="hl-trick">**Blurry**</mark> | <mark class="hl-trick">**❌ 无**</mark> |
| <mark class="hl-trick">**Entropy**</mark> | <mark class="hl-trick">**❌ 无**</mark> |
| <mark class="hl-trick">**Texture**</mark> | <mark class="hl-trick">**❌ 无**</mark> |

$$
\underbrace{15}_{\text{Fig.8(a) 的过滤器}} = \underbrace{7}_{\text{Table 5 给了阈值}} + \underbrace{8}_{\text{完全没有阈值}}
$$

<mark class="hl-key">**8 个没有阈值公开的过滤器，恰好就是论文唯一给了文字动机的那批**</mark> —— brightness、saturation、grayscale、blurry、entropy、texture 六个，论文都写了"移除过曝/欠曝/近单色/低锐度/近空/纹理样"的说明，<mark class="hl-trick">**但一个数字都没给**</mark>。

<mark class="hl-key">**这意味着 Table 5 的可复现性是残缺的**：你能精确复现 watermark、aesthetic、resolution、NSFW、OCR 这五类过滤，<mark class="hl-trick">但「图像内容是否干净」这个维度整体缺失</mark>。</mark>而 §2.6 会看到 concept-aware synthesis 的合成图最容易被判 aesthetic 虚高 —— <mark class="hl-key">**恰好落在这 8 个没有阈值的过滤器里**</mark>。

#### $\min(h,w)$ 单独成行不是冗余

<mark class="hl-trick">**分辨率门槛被写成两行，看起来冗余，其实是一道防病态宽高比的独立闸门**</mark>：

$$
h\times w \ge 1024^2 \quad \text{但同时}\quad \min(h,w) \ge 512
$$

<mark class="hl-key">**光看总像素数会漏掉极端全景图**</mark>：一张 $512\times 2048$ 的图有 1048576 像素 $>1024^2$，但 <mark class="hl-trick">$\min(h,w)=512$</mark> <mark class="hl-trick">**恰好**</mark> 踩线（这里刚好通过）；而 $256\times 4096$ 像素数更多，$\min$ 边却只有 256，<mark class="hl-key">会被 $\min$ 那一行拦下</mark>。

<mark class="hl-trick">**这一行的存在正好和 Aspect ratio 冻结在 [0.1, 10.0] 形成互补**</mark>：宽高比允许 1:10 到 10:1，但总像素和短边同时设卡，<mark class="hl-key">**保证「允许极端比例」不等于「允许极端尺寸」</mark>。对照 Mage-Flow 主打的 native-resolution 能力（512–2048 任意宽高），<mark class="hl-trick">数据侧的短边下限 $\min(h,w)\ge 512$ 正好对应它推理侧要服务的最窄档位</mark>。

#### SFT 段的分辨率门槛不再提高

<mark class="hl-trick">**Pixel count 和 $\min(h,w)$ 都在 1024² 段停住，SFT 段与 1024² 完全相同</mark>（都是 ≥1024² / ≥512）。

$$
\text{分辨率门槛: } 256 \to 512 \to 1024 \to \textbf{1024 (停)}
\qquad
\text{质量门槛(aesthetic): } 4.5 \to 5.5 \to 6.0 \to \textbf{6.5 (继续涨)}
$$

<mark class="hl-key">**所以 SFT 阶段是一次纯质量筛选，不再是一次分辨率升级。**</mark>这个设计与 §3.1 的 progressive curriculum 里 SFT 仍在 1024² 训练是自洽的 —— <mark class="hl-trick">既然 SFT 不引入更高分辨率的数据，门槛自然不需要再提高</mark>。

::: tip 本节可以带走的四条
1. <mark class="hl-trick">**Table 5 是三家里唯一完整的过滤阈值表**</mark>（Z-Image、DeepGen 都没有），可直接抄的产物。
2. <mark class="hl-key">**真正在动的只有 watermark 和 aesthetic 两行</mark> —— 看这张表不要被"逐阶段收紧"的 caption 误导。
3. <mark class="hl-trick">**$\min(h,w)$ 独立成行是为了防病态宽高比</mark>，与冻结的 aspect ratio [0.1, 10.0] 互补。
4. <mark class="hl-key">**SFT 段分辨率门槛停住，只涨质量门槛</mark> —— SFT 是纯质量筛选。
:::

::: warning 本节未公开 / 未兑现的细节
- <mark class="hl-trick">**8 个过滤器没有任何阈值**</mark>：Broken、Rotation、Saturation、Brightness、Grayscale、Blurry、Entropy、Texture。其中六个论文写了文字动机但无数字。
- <mark class="hl-trick">**"threshold-based rather than binary" 的具体机制**</mark>：论文原句 *"Many filters are threshold-based rather than binary"*。<mark class="hl-key">**超阈值不等于丢弃**</mark> —— 是软权重、还是改成按概率采样、还是分桶，<mark class="hl-trick">论文完全没说。这是本节最大的实现层空白</mark>，且它直接决定 Table 5 能否复现。
- <mark class="hl-trick">**Aesthetic-V2.5 的取值范围未说明</mark>：阈值 4.5–6.5 暗示是 10 分制，但论文没写。<mark class="hl-key">同样没给这个打分器的具体版本/权重（是否公开权重）。</mark>
- <mark class="hl-trick">**Watermark 检测器是什么**</mark>：只给分数阈值，没说是分类器、检测器还是某种水印概率模型。
- <mark class="hl-trick">**OCR 两行为什么是两个指标**</mark>：text-area ratio ≤0.3 是文字面积占比，num. regions ≤5 是文字块数量上限。<mark class="hl-key">后者明显是 document-likeness 启发式</mark>，但论文没解释这个设计的意图，也没说两个阈值哪个起主导作用。
- <mark class="hl-trick">**各过滤器之间是 AND 还是可配置组合**</mark>：论文只说 "filters remove ... images"，默认理解为全部同时生效，未说明是否可配。
:::

::: info 留给 §2.5 的一个悬念
<mark class="hl-trick">**本节刻意不解释一件事**：OCR 的两项阈值（text-area ratio ≤0.3、num. regions ≤5）四阶段完全冻结，<mark class="hl-key">而 Mage-Flow 把 text rendering 当作核心能力</mark></mark>（CVTG-2K 这类榜单是它的卖点）。

<mark class="hl-trick">表面上这是矛盾的。但解法不在 §2.2，而在于 §2.5 —— concept-aware synthesis 里的 synthetic text-rendering 数据，以及 concept-aware sampling 对分布的修正。</mark><mark class="hl-key">**§2.2 先把现象立起来，§2.5 再解释原因，中间不要提前引用结论，否则会把两节的分析揉在一起。</mark>
:::

### 2.3 Cross-sample deduplication（`Cross-sample deduplication.`）

### 2.4 Multi-granularity captioning（`Multi-granularity captioning.`）

### 2.5 Concept-aware synthesis and balancing（`Concept-aware synthesis and balancing.`）

### 2.6 总览：~90M raw triples → ~45M retained（§4.2 导语段）

### 2.7 Editing data synthesis（`Editing data synthesis.`）

### 2.8 VLM-based dataset filtering（`VLM-based dataset filtering.`）

### 2.9 Edit-type tagging and balancing（`Edit-type tagging and balancing.`）

### 2.10 本节小结与未公开细节（笔记自加，非论文章节）

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
