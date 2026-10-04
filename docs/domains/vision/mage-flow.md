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
| 1 | Data Collection & Curation | §4.1 (P17–18) + §4.2 (P18–19) | ★★★ | 🟢 §2.1 + §2.2 完整 |
| 2 | Pre-training + SFT Recipe | §5.1 (P20–21) | ★★★ | ⬜ |
| 3 | Diffusion-NFT Post-training | §5.2 (P21–24) | ★★★ | ⬜ |
| 4 | Few-step Distillation | §5.3 (P24–26) | ★★☆ | ⬜ |
| 5 | Native-Resolution + Infrastructure | §3.2 / §3.3 (P13–16) | ★★☆ | ⬜ |
| 6 | Mage-VAE | §3.1 (P10–13) | ★☆☆ | ⬜ |
| 7 | Ablation / Tricks 总结 | 全文 | ★★☆ | ⬜ |

## 1. Introduction

## 2. Data Collection and Curation ★

::: info 笔记小节 ↔ 论文章节对照（本节结构镜像论文 §4）
笔记的 `###` 层级对应论文的**节**，笔记的 `####` 层级对应论文 §4.x 内部的**粗体段标题**，因此论文的每个段落都能在笔记里找到同名位置：

| 笔记层级 | 对应论文 | 论文段首粗体导语 | 页 |
| :--- | :--- | :--- | :--- |
| `### §2.1` | **§4.1** T2I 数据 | *（导语段，无粗体标题）* | P17 |
| └ `####` | §4.1 第 1 段 | **`Sample-level filtering.`** | P17 |
| └ `####` | §4.1 第 2 段 | **`Cross-sample deduplication.`** | P17 |
| └ `####` | §4.1 第 3 段 | **`Multi-granularity captioning.`** | P18 |
| └ `####` | §4.1 第 4 段 | **`Concept-aware synthesis and balancing.`** | P18 |
| `### §2.2` | **§4.2** Edit 数据 | *（导语段，无粗体标题）* | P18 |
| └ `####` | §4.2 第 1 段 | **`Editing data synthesis.`** | P19 |
| └ `####` | §4.2 第 2 段 | **`VLM-based dataset filtering.`** | P19 |
| └ `####` | §4.2 第 3 段 | **`Edit-type tagging and balancing.`** | P19 |
| `### §2.3` | *无对应* | <mark class="hl-trick">**笔记自加**：本节小结与未公开细节清单**</mark> | — |
:::

::: warning 两节各有一段「没有粗体标题」的开场段
<mark class="hl-trick">**论文 §4.1 与 §4.2 的第一段都没有粗体导语**</mark>：§4.1 以 *"The Mage-Flow generation corpus is built from roughly 10B raw image–text pairs..."* 起，§4.2 以 *"The Mage-Flow-Edit corpus consists of (source image, edit instruction, target image) triples..."* 起。

<mark class="hl-key">**它们是各节的总览段，不是独立小节**</mark>，所以笔记里直接放在对应 `###` 的开头，<mark class="hl-trick">**不另立 `####`**</mark> —— 这样论文的四个（§4.1）和三个（§4.2）粗体段标题在笔记里就是同名的 `####`，可以逐个对读。
:::

### 2.1 T2I 数据：10B raw pairs → ~1.3B curated pairs（论文 §4.1）

#### ① 导语段：四步固定顺序与收缩比

Mage-Flow 的 T2I 数据从大约 <mark class="hl-key">**10B raw image–text pairs**</mark> 开始，全部来自大规模开源数据集。<mark class="hl-trick">作者按照固定顺序进行四步处理</mark>：

$$
\boxed{
\text{Sample-level Filtering}
\rightarrow
\text{Cross-sample Deduplication}
\rightarrow
\text{Multi-granularity Captioning}
\rightarrow
\text{Concept-aware Synthesis \& Balancing}
}
$$

最终得到约 <mark class="hl-key">**1.3B high-quality image–text pairs**</mark>，后面的不同 pre-training stage 再从这个 curated corpus 中选择各自的数据子集。

| 步骤 | 作者的定位 |
| :--- | :--- |
| <mark class="hl-trick">Filtering</mark> | 去掉坏图、低质图、不安全图和视觉不合适的图片 |
| <mark class="hl-trick">Deduplication</mark> | 消除近重复视觉样本 |
| <mark class="hl-trick">Captioning</mark> | 重新标准化文本监督 |
| <mark class="hl-trick">Concept-aware synthesis</mark> | 补 raw web data 中天然缺少的长尾能力 |

![Mage-Flow Fig.8(a)：Text-to-Image 数据处理流水线。从左到右：10B 开源图像数据集 → **Sample-level Filtering**（分两层虚线框：File Information Filter 含 Broken / File Size / Resolution / Aspect ratio / Rotation；Image Content Filter 含 Saturation / Brightness / Grayscale / Blurry / Texture / Watermark / NSFW / Aesthetic / OCR / Entropy）→ **Cross-sample Deduplication** → **Multi-Granularity Captioning** → **Concept-aware Synthesis** → 右侧数据柱「~1.3B High-Quality Image-Text Data」。注意四阶段的**串行顺序**：filter 在 dedup 之前，synthesis 在最后。](/mageflow-fig8a-pipeline.png)

<mark class="hl-key">**这个串行顺序本身有成本含义**</mark>：<mark class="hl-trick">先用便宜的 SSCD 描述符把 10B 砍掉一大半，再让昂贵的 Qwen3-VL-32B 只 caption 剩下的 1.3B</mark>。若顺序反过来（先 caption 再 dedup），VLM 推理成本要按 10B 的量级计。

#### ② Sample-level filtering.

<mark class="hl-trick">**作者首先对每一个 image–caption pair 独立处理**</mark>，先做 <mark class="hl-key">**file-information filtering**</mark>，再做 <mark class="hl-key">**image-content filtering**</mark>。

| 层 | 过滤内容 |
| :--- | :--- |
| <mark class="hl-trick">**文件层面**</mark> | corrupted / near-empty files、分辨率或 pixel count 不够、极端 aspect ratio、<mark class="hl-trick">**orientation metadata 错误**</mark> |
| <mark class="hl-trick">**图像内容层**（decode 后）</mark> | brightness、saturation、grayscale、blur、entropy、texture、watermark、aesthetic、OCR、NSFW |

各 filter 的动机：

- <mark class="hl-trick">brightness / saturation</mark> → 去过曝、欠曝和异常高饱和
- <mark class="hl-trick">grayscale / blurry</mark> → 去近单色和低清晰度
- <mark class="hl-trick">entropy / texture</mark> → <mark class="hl-key">**压制接近空白、以及纯纹理缺少真实语义内容的图片**</mark>
- <mark class="hl-trick">watermark / aesthetic / OCR / NSFW</mark> → 分别去水印、低质量、<mark class="hl-key">**document-like**</mark>、unsafe samples

<mark class="hl-trick">**这里不是所有 filter 都是简单的 true/false**，作者明确说很多都是 <mark class="hl-key">**score + threshold**</mark></mark>。而且阈值随着训练不断变严格：

| Filter | 256² | 512² | 1024² | SFT |
| :--- | ---: | ---: | ---: | ---: |
| Pixel count $h\times w$ | ≥256² | ≥512² | ≥1024² | ≥1024² |
| $\min(h,w)$ | ≥128 | ≥256 | ≥512 | ≥512 |
| Aspect ratio | [0.1, 10] | [0.1, 10] | [0.1, 10] | [0.1, 10] |
| File size | ≥1 KB | ≥1 KB | ≥1 KB | ≥1 KB |
| NSFW | ≤0.1 | ≤0.1 | ≤0.1 | ≤0.1 |
| <mark class="hl-trick">Watermark</mark> | <mark class="hl-key">**&lt;0.5**</mark> | <mark class="hl-key">**&lt;0.3**</mark> | <mark class="hl-key">**&lt;0.1**</mark> | <mark class="hl-key">**&lt;0.05**</mark> |
| <mark class="hl-trick">Aesthetic-V2.5</mark> | <mark class="hl-key">**≥4.5**</mark> | <mark class="hl-key">**≥5.5**</mark> | <mark class="hl-key">**≥6.0**</mark> | <mark class="hl-key">**≥6.5**</mark> |
| OCR text-area ratio | ≤0.3 | ≤0.3 | ≤0.3 | ≤0.3 |
| OCR region number | ≤5 | ≤5 | ≤5 | ≤5 |

![Mage-Flow Table 5：四阶段 sample-level filtering 阈值完整表（论文 §4.1，P17）。](/mageflow-tab5-filter-thresholds.png)

因此作者真正采用的是一种 <mark class="hl-key">**progressive filtering**</mark>：<mark class="hl-trick">**256² 阶段不追求把数据洗到极致，而是保留 broad visual coverage**</mark>；随着 512² → 1024² → SFT，逐渐提高 resolution / aesthetic 要求、降低 watermark 容忍度，让数据越来越 clean。

<mark class="hl-trick">**注意九行里只有 Watermark 和 Aesthetic-V2.2 在动</mark>，其余七行四阶段不变；<mark class="hl-key">**Resolution 门槛在 1024² 段停住（SFT 不再提高），只有质量门槛继续收紧**</mark> —— 所以 SFT 是一次纯质量筛选，不是分辨率升级。

::: warning 本段未公开的细节
<mark class="hl-trick">**brightness、saturation、grayscale、blur、entropy、texture 虽然论文说用了，但具体模型、计算公式和 threshold 没公开**</mark>。<mark class="hl-key">这六个恰好是 Table 5 里没有对应行的部分</mark> —— 把 Fig. 8(a) 的 15 个过滤器与 Table 5 的行对一遍，会发现只有 7 个过滤器有公开阈值。

另外 <mark class="hl-trick">**"score + threshold" 超阈值之后的具体机制也没给**</mark>：是软权重、按概率采样还是分桶，论文未说明。
:::

#### ③ Cross-sample deduplication.

<mark class="hl-key">**Filtering 解决的是「单张图片好不好」，Deduplication 解决的则是「这张图片是不是已经在整个数据池里出现过」。**</mark>

Mage-Flow 首先把每张通过过滤的图片输入 <mark class="hl-key">**SSCD（Self-Supervised Copy Detection）**</mark>，得到一个 copy-detection descriptor。

::: info SSCD 原论文解决什么（arXiv:2202.10261）
<mark class="hl-trick">**SSCD 的目标不是判断「两张图语义上是不是都包含一只狗」，而是判断它们是否来自同一张 2D image source**</mark> —— 即使其中一张经历过 JPEG 压缩、裁剪、resize、加文字等变化，仍希望找到它。

SSCD 原论文也明确指出，<mark class="hl-key">**web-scale copy detection 最现实的方法就是把图片压成较短的 descriptor vector，再使用 approximate nearest-neighbor search**</mark>。这正是 Mage-Flow 选它而不选 CLIP 的理由：CLIP 对「内容相同但风格不同」的图会误判为重复，而 SSCD 是 copy-detection 专用，对裁剪/缩放更鲁棒。

$$
I_i \xrightarrow{\ \text{SSCD}\ } d_i
$$

其中 $d_i$ 是一个 dense vector。<mark class="hl-trick">SSCD 原论文中的代表性 ResNet-50 配置使用 **512-dimensional descriptor**</mark>，其方法包括 GeM pooling、linear projection 和 entropy regularization；论文实验中的 descriptor 还会做 <mark class="hl-trick">**whitening 和 L2 normalization**</mark>。
:::

::: warning 这里是 SSCD 原论文的实现细节，不代表 Mage-Flow 的配置
<mark class="hl-trick">**512-D / whitening / GeM 这些都是 SSCD 原论文的代表性配置**</mark>。<mark class="hl-key">**Mage-Flow 本身只说用了 SSCD copy-detection descriptor，没有进一步披露具体 variant**</mark> —— 不能假定它与原论文配置完全一致。
:::

然后才轮到 <mark class="hl-key">**FAISS**</mark>。

::: info FAISS 解决什么（facebookresearch/faiss）
<mark class="hl-trick">**FAISS 不是神经网络，它是一个专门做高维 dense-vector similarity search 和 clustering 的库**</mark>。官方定义就是 efficient similarity search of dense vectors。

它的假设是每个实例已经变成向量并有一个 integer ID，可以根据 L2 distance 或 dot product 搜索；<mark class="hl-key">**归一化向量的 dot product 又可以直接用于 cosine similarity**</mark> —— 这正是 §③ 里 $\cos(d_i,d_j)>0.9$ 判据能成立的前提。

FAISS 有不同 index，可以在搜索速度、搜索精度、显存/内存之间 trade-off；有些压缩索引能够扩展到 <mark class="hl-trick">**billions of vectors**</mark>，也支持 GPU。
:::

因此可以把 Mage-Flow 的 <mark class="hl-key">**descriptor index**</mark> 理解成：

```text
Image ID        SSCD descriptor
-----------------------------------------
image_00001  -> [d1, d2, ..., dD]
image_00002  -> [d1, d2, ..., dD]
image_00003  -> [d1, d2, ..., dD]
...
```

<mark class="hl-trick">**FAISS index 真正负责的是右边的向量检索，Image ID 再对应回原始图片和质量信息**</mark>。<mark class="hl-key">**Mage-Flow 没有告诉我们到底用了 Flat、IVF、HNSW、PQ 还是哪一种，也没有公开 index 参数**</mark>，所以这一点不能补。

有了这个 index 后，论文先做 <mark class="hl-trick">**within-dataset deduplication**</mark>。同一个 dataset 中，如果两张图的 SSCD descriptor cosine similarity：

$$
\cos(d_i,d_j)>0.9
$$

就把它们归入 duplicate group，<mark class="hl-key">**只留下 highest-quality representative**</mark>。特别大的 duplicate cluster 还会做 cap，以防 <mark class="hl-trick">**stock photos、banner、product layout 等 web template**</mark> 在训练数据里大量重复。

接着再做 <mark class="hl-trick">**cross-dataset deduplication**</mark>。作者维护一个 <mark class="hl-key">**persistent descriptor index**</mark>：前面已经接受进入训练池的图片，其 SSCD descriptor <mark class="hl-trick">**不会处理完一个 dataset 就扔掉，而是一直保存在这个全局 index 里**</mark>。之后一个新的 dataset 进来时：

$$
\text{new image}
\rightarrow
\text{SSCD descriptor}
\rightarrow
\text{query persistent FAISS index}
$$

<mark class="hl-trick">如果它和历史已经接受图片的 similarity > 0.9，就拒掉**</mark>。这样才能避免<mark class="hl-key">**「LAION 里一张图，另一个 web dataset 里又出现同一张图」的跨数据源重复**</mark>。

作者还另外维护一个 <mark class="hl-trick">**held-out benchmark index**</mark>。这个 index 里面<mark class="hl-key">**不是训练数据，而是最终 evaluation benchmark 中图像的 descriptors**</mark>。训练候选图片也会去和这个 index 做匹配，如果它和 evaluation image 太接近，就应该被排除，从而降低 <mark class="hl-key">**evaluation contamination / train-test leakage**</mark>。

因此实际上有两个概念完全不同的 index：

$$
\boxed{
\text{Persistent Training Index}\;\Longrightarrow\;\text{「这张图是不是训练库里已经有过？」}
}
$$

$$
\boxed{
\text{Held-out Benchmark Index}\;\Longrightarrow\;\text{「这张图是不是和未来测试集中的图重复或高度近似？」}
}
$$

::: warning 本段未公开的细节
- <mark class="hl-trick">**held-out benchmark index 到底包含哪些 benchmark**</mark> —— 未公开。
- <mark class="hl-trick">**benchmark matching 是否有独立的 similarity threshold**</mark> —— 论文只给了 0.9 一个数字，未说明是否对 benchmark 另设阈值。
- <mark class="hl-trick">**"highest-quality representative" 的质量排序公式**</mark> —— 未公开（用什么打分器、是否就是 Aesthetic-V2.5，都没说）。
- <mark class="hl-trick">**large duplicate cluster 的具体 cap 策略**</mark> —— 只说 capped，没说是随机采样还是按质量取 top-k。
- <mark class="hl-trick">**两层各淘汰多少、两层的先后淘汰量**</mark> —— 与 §② 同样地，论文只给了总收缩（10B → 1.3B）。
:::

#### ④ Multi-granularity captioning.

去完重复数据以后，Mage-Flow 再重新建立 text supervision。作者用 <mark class="hl-key">**Qwen3-VL-32B-Instruct**</mark> 给每张图片生成四种粒度的 caption，<mark class="hl-trick">**而且这四种不是单纯「长短不同」，而是负责不同信息**</mark>：

$$
\text{Phrase} \rightarrow \text{核心概念 / concept statistics}
$$

$$
\text{Entity} \rightarrow \text{主要 object + attribute}
$$

$$
\text{Composition} \rightarrow \text{spatial layout + object relation}
$$

$$
\text{Photographic} \rightarrow \text{style + lighting + viewpoint + atmosphere + fine details}
$$

![Mage-Flow Fig.8(b)：Multi-Granularity Caption Framework。左侧两个输入图（鸽子 / 画框）经同一个 System Prompt（*"You are a world-class multi-granularity image captioning expert... produce a structured, detailed, and objective description"*）送给 Qwen，右侧对每张图输出四列：**Phrase**（如 `four pigeons, urban scene, wet pavement, puddle reflection, green fence, overcast lighting, concrete steps, muted colors`）、**Entity**（`Four pigeons standing in a large puddle on a paved surface.`）、**Composition**（街级视角描述 + 水面倒影 + 背景木栅栏与混凝土台阶的相对位置）、**Photographic**（`This is a photorealistic, outdoor street photography shot taken at eye level...`）。画框那例的 Photographic 甚至捕捉到了 `'MARIANNA'` 印刷体与 `'Tommy Thompson'` 脚本体两种字体。](/mageflow-fig8b-caption-granularity.png)

<mark class="hl-trick">**Figure 8 给出的例子很典型**</mark>：同一张「水坑里的四只鸽子」图片，phrase caption 是一组高度浓缩的概念；entity caption 变成主体描述；composition caption 详细描述鸽子、水坑、围栏和楼梯之间的位置关系；photographic caption 又进一步描述 eye-level、overcast lighting、羽毛、反射和色调。<mark class="hl-key">**这就是多粒度 caption 的实际含义**</mark>。

<mark class="hl-trick">**训练时会从不同 caption channel 中 sampling**</mark>，让模型既能处理短 prompt，也能处理更长、更具体的 prompt。对于 text-rich image，Qwen3-VL 还被明确要求<mark class="hl-key">**识别 visible text，并把它转换为 rendering instruction**</mark>。

::: warning 本段未公开的细节
<mark class="hl-trick">**论文没有公开四种 caption 的 sampling ratio**</mark>。<mark class="hl-key">**因此不能假设四类等概率**</mark> —— 而这个比例直接影响「模型见长 prompt 还是短 prompt」的分布，是不能靠猜的。

同样未公开的还有完整 recaption system prompt 在正文中的所有实现细节。
:::

#### ⑤ Concept-aware synthesis and balancing.

最后一步是在解决：<mark class="hl-key">**10B web 数据已经非常多了，为什么还要自己造数据？**</mark>

原因是<mark class="hl-trick">**「大规模」不代表「能力覆盖均匀」**</mark>。作者明确发现 web corpus 中仍然缺少：

<mark class="hl-trick">**long-text rendering、rare objects、uncommon attributes、structured layouts、under-represented styles。**</mark>

因此他们主动构造 <mark class="hl-key">**targeted supplemental data**</mark>。Text rendering 会专门合成不同 <mark class="hl-trick">**fonts、layouts、languages、colors、backgrounds**</mark> 的图片；除此之外还额外构造 rare concepts 和 compositional cases。

<mark class="hl-key">**这里有一个非常值得保留的工程原则：synthetic data 并不是生成出来直接进入训练。**</mark>所有 supplemental samples 还要<mark class="hl-trick">**重新经过和原数据相同的 filtering + quality-control pipeline，通过之后才 merge 进最终 curated corpus**</mark>。<mark class="hl-key">**这一点 Mage-Flow 写得比 DeepGen 明确 —— 后者用了 Gemini 2.5 Pro + Qwen-Image 合成，但没说是否过同一套过滤。</mark>

接下来才是 <mark class="hl-key">**concept-aware balancing**</mark>。<mark class="hl-trick">**前面的 phrase-level captions 不只是训练 caption，它还有一个很重要的用途 —— 做 concept statistics**</mark>。作者根据这些 phrase-level captions 估计整个 merged corpus 的 concept distribution。

![Mage-Flow Fig.9(a)：generation pre-training 数据的概念分布（太阳图，内环为大类，外环为细类）。**Object & Products 31.3%**（Apparel & Accessories 16.8% / Furniture 5.8% / Packaging 3.5% / Electronics 2.8% / Vehicles 2.4%）、**Scene & Place 26.1%**（Landscape 14.3% / Indoor 7.7% / Cityscape 4.0%）、**People 19.0%**（Person 7.2% / Appearance 6.2% / Portrait 5.7%）、**Living & Food 9.0%**（Food & Drink 4.1% / Plants 2.5% / Animals 2.4%）、**Design 8.8%**（Poster & UI 5.0% / Cartoon 2.1% / Art 1.8%）、**Synthetic 5.8%**（English Text 1.9% / Chinese Text 1.8% / Others 2.1%）。](/mageflow-fig9a-concept-dist.png)

<mark class="hl-trick">**Figure 9(a) 展示的数据依旧是明显 long-tail**</mark>，其中 <mark class="hl-key">**Object & Products 约 31.3%、Scene & Place 约 26.1%、People 约 19.0%**</mark>，是最主要的几个大类；论文正文进一步强调 <mark class="hl-key">**Design 和 Synthetic 虽然占比不是最大的，但承担 layout、product-style、poster-style、text rendering 等重要专项能力**</mark>。

<mark class="hl-trick">如果直接按原始 web frequency 随机采样，那么 head concepts 会不断占据 batch</mark>，例如普通 object、natural scenes、常见 photorealistic images 会远多于长尾概念。因此作者进行 <mark class="hl-key">**concept-aware sampling**</mark>，明确降低：

- <mark class="hl-trick">**frequent objects**</mark>
- <mark class="hl-trick">**natural scenes**</mark>
- <mark class="hl-trick">**common photorealistic styles**</mark>

对训练的支配，让 <mark class="hl-key">**capability-critical / long-tail concepts 得到更大的训练曝光率**</mark>。

<mark class="hl-key">**这里需要非常明确地区分两个动作**</mark>：

$$
\boxed{
\text{Concept-aware Synthesis}=\text{「原来样本不够}\ \to\ \text{主动增加数据」}
}
$$

$$
\boxed{
\text{Concept-aware Sampling}=\text{「数据已经存在}\ \to\ \text{改变它被抽到的概率」}
}
$$

<mark class="hl-trick">**比如 rare-object 数据本来只有 10 万张，你可以先 synthetic 到 100 万张，这是增加数据量**</mark>；<mark class="hl-key">**但训练时又让这 100 万张比常见的千万级 street/photo 数据更频繁地进入 batch，这是 sampling reweighting**</mark>。

作者还明确说这个 balancing <mark class="hl-trick">**不是从头到尾固定**</mark>：随着训练从早期走向后期，一方面 filtering threshold 越来越严格，另一方面 <mark class="hl-key">**reweighting strength 也越来越强**</mark>。

$$
\boxed{
\text{Early training}\ \approx\ \text{broad visual prior}
\quad\xrightarrow{\ \text{逐阶段加强}\ }\quad
\text{Later training}\ \approx\ \text{clean + capability-focused distribution}
}
$$

这也正好和 <mark class="hl-trick">**256² → 512² → 1024² → SFT 的 filtering curriculum 对应起来**</mark>。

::: warning 本段最关键的未公开细节
<mark class="hl-trick">**Mage-Flow 没有公开 concept-aware sampling 的数学公式**</mark>，也没有公开各 concept 的具体 sampling weights、<mark class="hl-trick">**frequency → weight 的映射方式**</mark>以及每个 stage 的 reweighting strength。

<mark class="hl-key">**因此像 inverse-frequency sampling、temperature sampling、$1/f^{\alpha}$ 之类都只能作为一般方法理解，不能写成 Mage-Flow 的实际实现。**</mark><mark class="hl-trick">这是本段唯一真正影响训练分布的旋钮，缺了它整个 balancing 就无法复现。</mark>
:::

#### ⑥ §4.1 全流程压缩（笔记自加）

$$
\boxed{
10B\ \text{Raw Pairs}
\rightarrow
\text{单样本质量过滤}
\rightarrow
\text{SSCD Descriptor}
\rightarrow
\text{FAISS 全局近邻检索与去重}
\rightarrow
\text{Qwen3-VL 多粒度 Caption}
\rightarrow
\text{长尾专项数据合成}
\rightarrow
\text{再次 QC}
\rightarrow
\text{Concept-aware Sampling}
\rightarrow
1.3B\ \text{Curated Pairs}
}
$$

::: tip 一句话记住 SSCD + FAISS 的分工
> <mark class="hl-key">**SSCD 负责回答「图片应该怎样表示，才能识别经过裁剪、压缩、resize 等变化后的同源副本」；FAISS 负责回答「拿到几十亿个这种向量以后，怎么快速从里面找到最相似的几个」。**</mark>

这才是 Mage-Flow 这一段 cross-sample deduplication 真正完整的含义 —— <mark class="hl-trick">**没有 SSCD，web 副本识别不出来；没有 FAISS，10B × 512 维的向量检索成本不可接受。**</mark>
:::


### 2.2 Edit 数据：~90M raw triples → ~45M retained（论文 §4.2）

#### ① 导语段：训练单元与 90M 来源构成

Mage-Flow-Edit 的基本训练单元是一个三元组：

$$
(\text{source image},\ \text{edit instruction},\ \text{target image})
$$

也就是<mark class="hl-trick">**「原图 + 编辑指令 + 编辑后目标图」**</mark>。<mark class="hl-key">**这个三元组是 pair 级的 —— 质量判断的对象不是单张图，而是「这一组(source, instruction, target) 是否自洽」**</mark>，这是 editing 数据和 T2I 数据最本质的差别。

作者一开始就给出规模：

| 来源 | 规模 | 细分 |
| :--- | ---: | :--- |
| <mark class="hl-trick">开源 instruction-based editing datasets</mark> | <mark class="hl-key">**~50M**</mark> | — |
| <mark class="hl-trick">内部合成数据</mark> | <mark class="hl-key">**~40M**</mark> | <mark class="hl-trick">**~10M low-level image-processing**</mark> + <mark class="hl-trick">**~30M general semantic-editing**</mark> |
| <mark class="hl-key">**raw pool 合计**</mark> | <mark class="hl-key">**~90M triples**</mark> | |

整个流程是<mark class="hl-trick">**先收集/合成 → 再做 VLM voting filter → 之后 edit-type tagging → 最后 category balancing**</mark>。

#### ② Editing data synthesis.

论文先讲这 40M 内部 synthetic 数据是怎么构成的。重点是那 <mark class="hl-key">**30M semantic-editing data**</mark>，<mark class="hl-trick">**它不是泛泛地「用模型造编辑数据」，而是有意识地覆盖真实用户会用到的 editing skills**</mark>。

作者明确列出的能力包括：<mark class="hl-trick">**background replacement、color modification、material modification、tone transfer、style transfer、subject addition、subject removal、subject replacement、object-count change、motion change、viewpoint change、text editing、portrait retouching、old-photo restoration、global adjustment**</mark>。

<mark class="hl-key">**也就是说，他们先从能力维度反推数据缺口，再专门造对应的 source–target pairs，而不是只依赖现成开源数据。**</mark>

每一种 edit type 都有自己的 <mark class="hl-key">**type-specific synthesis pipeline**</mark>。论文明确说这些 pipeline 会组合几类现成技术：<mark class="hl-trick">**off-the-shelf generation、inpainting、segmentation、image processing，以及 template-based instruction generation**</mark>。

这样做的目的有两个：

1. <mark class="hl-key">**可以获得明确的 source–target 对应关系以及清晰的 editing instruction**</mark>
2. <mark class="hl-key">**专门补那些在开源数据里数量少或者质量不稳定的 editing category**</mark>

$$
\boxed{
\text{先定义需要的 Edit 能力}
\rightarrow
\text{针对每种能力设计合成 pipeline}
\rightarrow
(\text{source},\ \text{instruction},\ \text{target})
}
$$

<mark class="hl-trick">**这里和 T2I 的 concept-aware synthesis 其实有相似思想：缺什么能力，就主动造什么数据。**</mark>§2.1⑤ 的 long-tail 概念补齐和这里的 edit capability 补齐，是同一个方法论在两个模态上的应用。

::: warning 本段未公开的细节
<mark class="hl-trick">**作者没有进一步公开每一种 edit type 具体用了哪个 generation / inpainting / segmentation 模型**</mark>，也<mark class="hl-trick">**没有公开 30M semantic 数据内部各类的原始合成比例**</mark>。<mark class="hl-key">注意这和 §2.1⑤ 是同一类缺口 —— T2I 侧没给 synthetic 占比，editing 侧连各类内部比例也没给。</mark>
:::

#### ③ VLM-based dataset filtering.

有了 90M triples 后不能直接训练，因为 <mark class="hl-key">**editing pair 的「质量」比普通 T2I 更难判断**</mark>。作者明确指出 raw editing triples 常见的问题包括：

| 问题 | 说明 |
| :--- | :--- |
| <mark class="hl-trick">**instruction 未被执行**</mark> | target 根本没有正确执行 instruction |
| <mark class="hl-trick">**修改了不该变的区域**</mark> | 动了 unrelated regions |
| <mark class="hl-trick">**source identity / layout 被无必要改变**</mark> | — |
| <mark class="hl-trick">**引入明显 visual artifacts**</mark> | — |

所以 Mage-Flow 使用 <mark class="hl-key">**3 个独立的 Qwen3.5-9B experts**</mark> 来做过滤。

<mark class="hl-key">**这里有一个很关键的细节：三个 expert 不是完全相同的 judge。**</mark>每一个都有<mark class="hl-trick">**不同的 system prompt，而且评价 criteria 只部分重叠**</mark>，作者明确说这样设计是为了让三个判断 <mark class="hl-key">**"complementary rather than identical"**</mark>。

每个 expert 同时看到：

$$
(\text{source image},\ \text{target image},\ \text{edit instruction})
$$

然后重点判断三个核心问题：

1. <mark class="hl-trick">**requested edit 是否正确执行**</mark>
2. <mark class="hl-trick">**unrelated regions 是否保留**</mark>
3. <mark class="hl-trick">**最终 edited result 是否 visually plausible**</mark>

<mark class="hl-trick">**Expert 首先输出 reasoning，然后系统把 reasoning 解析成 criterion-level assessment，最后通过预先设定的 threshold 转成 pass / fail。**</mark>

![Mage-Flow Fig.10：Editing data filtering pipeline。左侧蓝色框 `~90M triples total` 分成 `~50M Open-source Editing Triples` 与 `~40M Synthesized In-house Triples`（后者再分 `~10M Low-level processing` / `~30M General semantic editing`）。中间橙色虚线大框 `VLM Voting`：三张叠放的 Qwen 卡片标 `Experts "You are an expert evaluator..."`，每张内含 `LLM Analyze → Result Parsing`，随后进入六边形 `Majority Vote (≥ 2/3 pass → admit)`。框内给了两个实例——**Pass Example**：`Replace the basket of bread with a vibrant bowl of fresh fruit`，三位 expert 打分 **Expert1: 10 / Expert2: 10 / Expert3: 6**（前两位判"perfectly done / matching pedestal, lighting, and shadows exactly"，第三位指出"bowl's shape and position differ, causing a mismatch in scale and placement"），结果 **✓ ADMIT**；**Fail Example**：`change the color of tree leaves to orange`，打分 **Expert1: 7 / Expert2: 4 / Expert3: 4**（意见包括"leaves on the left side have been changed while the rest remains green"、"partial coverage with visible artifacts"、"only partial recolor with visible hue mismatch and spill"），结果 **✗ DISCARD**。右侧纵向流程：`~45M Triples Survive (~20M open + ~25M synthetic)` → `Edit-type Tagging`（标注 *background, style, color, subject add/remove/replace...*）→ `Edit-category Balancing` → `Training Set`。](/mageflow-fig10-edit-filter-pipeline.png)

$$
\text{Triple}
\rightarrow
3\times\text{Qwen3.5-9B Expert}
\rightarrow
\text{Reasoning}
\rightarrow
\text{Result Parsing}
\rightarrow
\text{Pass / Fail}
$$

最后采用 <mark class="hl-key">**majority vote**</mark>：

$$
\boxed{
\ge 2/3\ \text{experts pass} \Rightarrow \text{保留}
}
$$

否则丢弃。<mark class="hl-key">**图里两个示例的对比很能说明问题**</mark>：Pass 那一例三位 expert <mark class="hl-trick">**对 scale / position 的判断并不完全一致**</mark>（第三位明确指出形状和位置有偏差），但总体多数通过；Fail 那一例因为<mark class="hl-trick">**存在绿色残留、partial recolor 和 visible artifacts**</mark>，多数 expert 判失败。

<mark class="hl-key">**所以这套 voting 不是只看「有没有发生变化」，而是在联合判断 instruction execution + preservation + visual quality。**</mark>

最终过滤非常狠：

$$
90M \rightarrow 45M
$$

$$
50M\ \text{open-source} \rightarrow 20M \qquad\qquad 40M\ \text{synthetic} \rightarrow 25M
$$

<mark class="hl-trick">**注意两个来源的通过率差异很大**</mark>：open-source 只有 $20/50 = 40\%$ 存活，而 synthetic 高达 $25/40 = 62.5\%$。<mark class="hl-key">这个差异合乎直觉 —— 自合成数据的 target 是由 pipeline 按指令生成的，天然更「指令一致」；而开源数据的 source–target 配对质量参差不齐。</mark>

::: warning 本段未公开的细节
<mark class="hl-trick">**论文没有公开三个 expert 的完整 system prompts，也没有公开每个 criterion 的具体 threshold**</mark>。<mark class="hl-key">因此我们知道它是「三 expert + reasoning + criterion-level parsing + predefined threshold + majority voting」，但**不能进一步写出具体评分公式**</mark>。

Fig. 10 里出现的是 10 / 6 / 7 / 4 这类整数分，但<mark class="hl-trick">**图上没有说明分数区间、量纲，也没有说明 threshold 落在哪里**</mark>，不能反推。
:::

#### ④ Edit-type tagging and balancing.

过滤完的 45M 数据仍然存在另一个问题：<mark class="hl-trick">**不同 edit operation 的数量非常不均匀**</mark>。所以作者又人工定义了一个统一的 <mark class="hl-key">**19-category edit taxonomy**</mark>，要求所有保留下来的 editing data 最终都映射到这个统一 taxonomy。

<mark class="hl-key">**但如果拿 VLM 给 45M triples 逐样本分类，成本太高，所以 Mage-Flow 做了一个很实用的工程简化：annotation unit 不是单条样本，而是根据原始 dataset 的组织方式决定。**</mark>

| 原始数据组织方式 | annotation unit |
| :--- | :--- |
| <mark class="hl-trick">整个 dataset 只包含一种稳定的 editing operation</mark> | <mark class="hl-key">**整个 dataset 当一个 unit**</mark> |
| <mark class="hl-trick">dataset 已划成语义一致的 sub-datasets</mark> | <mark class="hl-key">**每个 sub-dataset 当一个 unit**</mark> |
| <mark class="hl-trick">数据里本身有明确的 edit-type field</mark> | <mark class="hl-key">**具有同一个 field value 的样本归成一个 unit**</mark> |

最后再由<mark class="hl-trick">**人工把这些 units 映射到 19 个统一 category**</mark>。<mark class="hl-key">**这样就不用对 45M 张数据逐条跑分类器**</mark>。

完成统一 taxonomy 后，作者对<mark class="hl-trick">**每个 constituent dataset 和每个 edit category 的 sampling rate**</mark>进行调整，目的是让最后的 training mixture 在 taxonomy 上覆盖得更广、更合理。作者明确说这样可以<mark class="hl-key">**避免高频操作 dominating gradient，同时保证 rare but important edit types 被充分采到**</mark>。

<mark class="hl-key">**注意这里本质和前面 T2I concept-aware sampling 很像：不是简单删除头部数据，而是调训练时的 sampling rate。**</mark>§2.1⑤ 的 concept-aware sampling 和这里的 edit-category balancing 是同一个机制在两个模态上的落地。

![Mage-Flow Fig.9(b)：editing pre-training 数据的分布（太阳图，内环为 coarse domain，外环为细类）。**Scene & Spatial 20.5%**（Viewpoint Shift 15.0% / Background Change 5.5%）、**Attribute Editing 20.3%**（Attribute Tweak 8.6% / Portrait Retouch 5.4% / Recolor 2.6% / Action & Pose 2.2% / Material Change 2.2%）、**Text Editing 18.6%**（Poster 6.2% / Other Text 5.5% / Text Replacement 4.5% / Text Addition 1.61% / Text Removal 0.87%）、**Object Editing 15.7%**（Object Removal 5.7% / Object Replacement 4.5% / Object Addition 4.2% / Count Change 1.3%）、**Global Stylization 10.4%**（Style Transfer 8.4% / Tone & Lighting 2.0%）、**Low Level 9.3%**（Other Low Level 4.4% / Depth Map 2.1%）、**Reference & Composition 5.0%**（Multi Reference 2.1% / Subject Extraction 2.0% / Composition 0.91% / Colorization 0.66% / Normal Map 0.63% / Canny Edge 0.61% / Old Photo 0.90%）。](/mageflow-fig9b-edit-dist.png)

Figure 9(b) 展示 balance 后真正进入 pre-training 的编辑数据分布：

| coarse domain | 占比 |
| :--- | ---: |
| <mark class="hl-trick">Scene & Spatial</mark> | <mark class="hl-key">**20.5%**</mark> |
| <mark class="hl-trick">Attribute Editing</mark> | <mark class="hl-key">**20.3%**</mark> |
| <mark class="hl-trick">Text Editing</mark> | <mark class="hl-key">**18.6%**</mark> |
| <mark class="hl-trick">Object Editing</mark> | <mark class="hl-key">**15.7%**</mark> |
| <mark class="hl-trick">Global Stylization</mark> | 10.4% |
| <mark class="hl-trick">Low Level</mark> | 9.3% |
| <mark class="hl-trick">Reference & Composition</mark> | 5.0% |

更细的组成包括 <mark class="hl-trick">**Viewpoint Shift 15.0%、Background Change 5.5%、Attribute Tweak 8.6%、Portrait Retouch 5.4%、Style Transfer 8.4%、Poster 6.2%、Object Removal 5.7%**</mark> 等。

<mark class="hl-trick">**值得注意的是 Viewpoint Shift 单独占 15.0%**</mark>，<mark class="hl-key">**与 Scene & Spatial 合计 20.5% 里的四分之三 —— 这是全图最大的单一细类</mark>。而 §2.1⑤ 的 T2I 侧最大细类是 Apparel & Accessories 16.8%（纯长尾污染），<mark class="hl-trick">**两边的「最大细类」性质完全不同：editing 侧的 Viewpoint Shift 是一个真实且高价值的编辑能力，T2I 侧的 Apparel 则不是**</mark>。

::: warning 论文自身需要谨慎记录的地方：19 类与 Figure 9(b) 对不上
<mark class="hl-trick">**正文明确说人工 taxonomy 是 19 edit categories，但 Figure 9(b) 又画出了 7 个 coarse domains 以及更多细分 label**</mark>，<mark class="hl-key">**图中的可见细分类数量并不能直接和「19」一一对应**</mark>。

论文正文<mark class="hl-trick">**没有给出完整的「19 类名称列表及其和 Figure 9(b) 各层标签的映射关系」**</mark>，所以<mark class="hl-key">**这里不能擅自从饼图反推出那 19 类到底是哪 19 个**</mark>。我们只能准确记录：

- <mark class="hl-trick">**存在一个手工定义的 19-category unified taxonomy**</mark>（正文原文）
- <mark class="hl-trick">**Figure 9(b) 展示的是 balancing 后的数据组成**</mark>（图注原文）

<mark class="hl-key">**这两句话都成立，但它们之间的映射关系论文没有给出**</mark> —— 这是引用时必须标注的边界。
:::

#### ⑤ §4.2 全流程压缩（笔记自加）

$$
\boxed{
90M\ \text{Raw Editing Triples}
=
50M\ \text{Open}
+
40M\ \text{Synthetic}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Editing Data Synthesis}
\rightarrow
\text{3-Expert VLM Voting}
\rightarrow
45M\ \text{Retained}
\rightarrow
\text{19-category Tagging}
\rightarrow
\text{Sampling Balancing}
}
$$

::: tip 这一节真正值得记住的一条
<mark class="hl-key">**editing 数据不能像 T2I 一样只检查「图好不好」，还必须判断「指令有没有执行、该保留的区域有没有保留、source identity / layout 有没有被破坏、结果是否自然」。**</mark>

所以 Mage-Flow 用 multi-VLM expert voting 做 <mark class="hl-trick">**pair-level QC**</mark>；之后再统一 taxonomy 和调采样比例，解决不同 edit capability 分布严重不均的问题。

<mark class="hl-trick">**和 §2.1 的 T2I pipeline 对照着记，差异只在第二、三步**</mark>：

| 步骤 | T2I（§2.1） | Editing（§2.2） |
| :--- | :--- | :--- |
| 单样本质量 | 15 个 filter 的 score + threshold | <mark class="hl-trick">**pair 级：instruction 执行 + 区域保留 + visual plausibility**</mark> |
| 去重 | SSCD + FAISS，cos > 0.9 | <mark class="hl-key">**论文未提是否对 editing 也做去重**</mark> |
| 平衡 | concept-aware sampling | edit-category sampling rate 调整 |
| 共同点 | <mark class="hl-trick">**都不删头部数据，只调 sampling rate**</mark> | 同 |
:::


### 2.3 本节小结与未公开细节（笔记自加，非论文章节）


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
