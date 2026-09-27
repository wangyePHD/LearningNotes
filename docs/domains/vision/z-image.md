# 单流扩散基模 Z-Image (S3-DiT + 全链路后训练)

> **标签**：`Vision` `Diffusion` `DiT` `Flow Matching` `Distillation` `RLHF`
> **更新时间**：2026-09-26
> **参考来源**：[Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer (arXiv:2511.22699v5)](https://arxiv.org/abs/2511.22699) · [arXiv HTML 全文](https://arxiv.org/html/2511.22699v5) · [GitHub: Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) · [HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [ModelScope](https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
> **精读进度**：§1 Introduction ✅ ｜ §2 Data Infrastructure ✅（2.1–2.5）｜ §3 Image Captioner ✅（总览 + 3.1–3.3 全）｜ §4 Model Training 进行中（4.1–4.3 ✅，4.4–4.8 待展开）｜ §5 Evaluation

---

## 1. 问题定义与控制目标

> 本节对应论文 §1 Introduction。Z-Image 不是「又一个基模」，它的定位是**工业级全链路方法论样板**——真正的研究对象是「一个 6B 开源基模要花哪些钱、按什么顺序花」。

### 1.1 任务定位：先看清要打的三个靶子

论文开篇把 T2I 现状劈成 **两条 divergent trends**，然后明确表态「本工作两条都不走」：

| 现状路线 | 代表 | 作者的指控 |
| :--- | :--- | :--- |
| **闭源黑箱** | Nano Banana Pro、Seedream 4.0 | 性能高，但**不可复现、无透明度**，学术与工业都拿不到 |
| **开源靠堆参数**（scale-at-all-costs） | Qwen-Image 20B、FLUX.2 32B、Hunyuan-Image-3.0 80B | 训练与推理成本都 **prohibitive**；消费级硬件上无法推理、无法微调 |
| **第三条捷径：蒸合成数据** | 学术界为省算力的常规做法 | ⚠️ **闭环反馈 → 误差累积 + 数据同质化**，并**阻断教师模型之外的新能力涌现** |

::: danger 第三条指控是全文最有立场的部分
「蒸馏专有模型的合成数据」这条捷径在学术界很流行，但作者认为它有**三个结构性缺陷**：

1. **闭环反馈（closed feedback loop）**：学生学的是教师的输出，误差在反复蒸馏中累积；
2. **数据同质化（data homogenization）**：学生被锁死在教师已覆盖的分布内；
3. **能力天花板**：**学生永远学不到教师没有的东西**，这直接「阻断新视觉能力的涌现」。

Z-Image 的立场是 **purely real-world data**（含内部版权数据），**全程不蒸馏任何他人模型**。这是 §1.2 那 \$628K 成本能成立的方法论前提。
:::

### 1.2 核心矛盾与主张

$$
\underbrace{\text{参数量}}_{\text{Qwen-Image }20\text{B} \sim \text{Hunyuan-3.0 }80\text{B}}
\;\gg\;
\underbrace{\text{Z-Image }6\text{B}}_{\text{1/3} \sim 1/13}
\qquad\text{却要}\qquad
\underbrace{\text{Elo 人类偏好}\approx\text{商业 SOTA}}_{\text{Table 3/4}}
$$

作者主张 **principled design can effectively rival brute-force scaling**，并给出一个端到端解法的雏形：

> "the first comprehensive end-to-end solution that systematically optimizes every stage of the model lifecycle — from data curation and architecture design to training strategies and inference acceleration"

**成本锚点（Table 1）**：

| 阶段 | H800 GPU·h | 成本 (@\$2/h) | 占比 |
| :--- | ---: | ---: | ---: |
| 低分辨率预训练（$256^2$，仅 T2I） | 147.5K | \$295K | 47.0% |
| Omni 预训练（任意分辨率 + T2I/I2I 联合） | 142.5K | \$285K | 45.4% |
| 后训练（SFT + 蒸馏 + RLHF + Edit） | 24K | \$48K | 7.6% |
| **总计** | **314K** | **\$628K** | 100% |

::: info 第一个反直觉结论
**预训练占 92.4% 算力，后训练只占 7.6%。** 这与「效果主要来自后训练/RLHF」的社区直觉相反。Z-Image 的钱花在**数据与分布覆盖**上，后训练是廉价的「整形」环节。这一点对做资源规划很关键——如果你只有 10% 的算力预算，几乎不可能复刻它的路线。
:::

::: warning 最大的可复现性漏洞
论文 §2.1 明说数据来自 **"large-scale internal copyrighted collections"**。这些数据不公开，因此 **\$628K 里没有计入数据获取成本**（爬取/清洗/授权/存储）。对外部复现者而言，这个数字应理解为 **「训练算力」而非「总成本」**。另外 Table 1 的 \$2/GPU·h 是**租卡价格**，若自建集群则完全是另一笔账。
:::

### 1.3 四大支柱：全文的骨架

这是 Introduction 给出的**方法论总纲**，后面 §2–§4 全部是这四根支柱的展开：

| 支柱 | 核心动作 | 关键洞见 |
| :--- | :--- | :--- |
| **① Efficient Data Infrastructure** | 4 模块：Data Profiling Engine（多维特征抽取）+ Cross-modal Vector Engine（语义去重 / 定向检索）+ World Knowledge Topological Graph（概念组织）+ Active Curation Engine（闭环精炼） | 目标不是「筛干净」，而是 **让"对的数据"对上"对的训练阶段"**（"right data" aligned with "right stage"）。数据基建同时**决定能力上限**与**训练效率** |
| **② Efficient Architecture** | **S3-DiT**：单流 early-fusion，文本 / VAE token / 语义 token **统一处理** | 借鉴 LLM decoder-only 的 scaling 成功 → **每一层都是稠密跨模态交互**（而非双流各算各的），参数效率高；6B 打赢 20B/32B。PE 补世界知识弥补参数不足 |
| **③ Efficient Training Strategy** | 三段渐进：低分辨率预训练 ($256^2$) → Omni 预训练（任意分辨率 + T2I + I2I **摊薄预算**）→ **PE-aware SFT** | 关键：**PE-aware SFT 让 Z-Image 对齐 PE 的输出**，而**不是去训练 PE** → 因此 **零额外 LLM 训练成本** |
| **④ Efficient Inference** | **Z-Image-Turbo：8 NFE** | Decoupled DMD（解耦「质量增强」与「训练稳定」两个角色）+ DMDR（把分布匹配项当作 RL 的**内在正则**）。亚秒延迟、**<16GB 显存**可跑 |

::: info 支柱③ 的「摊薄」思想
Omni 预训练把**任意分辨率生成 + 文生图 + 图生图**合并成一个多任务阶段（amortizing the heavy pre-training budget across these diverse capabilities），从而**消除了独立、昂贵的分阶段训练**。这正是 Z-Image-Edit 得以低成本诞生的原因——它不是另起炉灶，而是**从 base model 继续训练**（§4.7）。
:::

### 1.4 两个衍生变体与核心能力主张

| 变体 | 来源 | 卖点 |
| :--- | :--- | :--- |
| **Z-Image-Turbo** | 少步蒸馏 + 奖励后训练 | **8 NFE**、亚秒延迟（企业级 GPU）、**<16GB VRAM** 消费级硬件可跑 |
| **Z-Image-Edit** | 复用 omni-pre-training 的多任务性 → 继续训练 | 精确指令跟随的编辑能力 |

核心能力主张（对应 Fig. 1–4 与 §5）：**写实生成** + **中英双语文字渲染** 达到甚至超越更大模型 / 顶级商业系统。脚注还点明：**FlashAttention-3 + torch.compile 是达成亚秒延迟的必要条件**。

### 1.5 本节留下的三个待追问点

1. **「6B + 纯真实数据打赢 20B/32B」可信吗？** 关键变量是**内部版权数据**（见上方 warning），这不可外部验证。
2. **PE-aware SFT 的隐忧**：把世界知识外挂给一个**冻结 VLM**，等于把「推理能力」的税交给上游模型。用户 prompt 的质量上限受制于 PE，而 PE 自身也可能幻觉（论文 §4.8 承认 VLM 全程冻结、不做对齐）。
3. **Introduction 说的「三段式」其实是简化版**：真实流水线是 预训练 → SFT → **少步蒸馏** → **RLHF** → Edit 继续训练（见 §4.3–§4.7 与 Fig. 11）。

---

## 2. Data Infrastructure（论文 §2）

> 论文 §2 的核心立场：资源受限时必须**从「数据数量」转向「数据效率」**——*"maximizes the information gain per computing unit"*。单纯扩大数据集往往收益递减；理想的数据系统要**概念广而不冗余**、**多语言图文对齐稳健**、且**为动态课程学习而结构化**（数据构成随训练阶段演化）。
>
> 四模块：**Data Profiling Engine**（多维特征抽取，本节）· **Cross-modal Vector Engine**（语义去重与定向检索，§2.2）· **World Knowledge Topological Graph**（概念组织，§2.3）· **Active Curation Engine**（闭环精炼，§2.4）。
>
> 这套基建同时**反哺训练了 captioner、奖励模型和 Z-Image-Edit**——数据侧与模型侧是同一个闭环。

### 2.1 Data Profiling Engine

**核心思想：**
Data Profiling Engine 不是简单做一次"留/删"过滤，而是先给每条 image-text pair 建立多维数据画像，后续再根据不同训练阶段做 hard filtering、sampling、balancing 和 curriculum。论文强调，不同数据源存在不同 bias，因此还支持 source-specific heuristics 和 sampling。

整体流程可以记成：

$$
\boxed{
\text{Raw Image-Text Pair}
\rightarrow
\text{Metadata}
\rightarrow
\text{Technical Quality}
\rightarrow
\text{Semantic/Aesthetic}
\rightarrow
\text{Cross-modal Consistency}
\rightarrow
\text{Multi-level Captioning}
}
$$

#### 1. Image Metadata

记录 resolution、width/height、file size，并计算 pHash。前者用于分辨率、宽高比等基础筛选，pHash 用于 identical / near-duplicate 的低层去重。

#### 2. Technical Quality Assessment

主要检测三类问题：

- **Compression artifacts**：通过理想未压缩大小与实际文件大小的比例，判断是否过度压缩；
- **Visual degradations**：内部质量模型检测 color cast、blur、watermark、excessive noise；
- **Information entropy**：用 border pixel variance 检测大面积纯色/边框，用 JPEG re-encoding 后的 BPP 作为 image complexity proxy，过滤信息密度过低的图。

#### 3. Semantic and Aesthetic Content

- 用专业标注数据训练的 aesthetic scoring model 判断视觉吸引力；
- 用 AIGC classifier 检测并过滤 AI-generated images；
- 用专门 VLM 做高层 semantic tagging，包括物体、人数、中国文化相关概念等；
- 同一 VLM 还输出 NSFW score。

#### 4. Cross-Modal Consistency

用 CN-CLIP 计算 image 与原始 alt caption 的相关性，低相关 pair 直接丢弃，避免错误图文对应污染训练。

#### 5. Multi-Level Captioning

对最终进入 pretraining 的图重新生成多粒度 caption，包括 tags、short phrases、long-form descriptions。VLM 还显式识别图中 visible text 和 watermark，并将其写入 caption，为后续 text rendering 提供监督。

::: info 最需要记住的结论

Z-Image 的数据思想不是"算一个 quality score 然后过滤"，而是：

$$
\boxed{
\text{对每条数据保存多个独立属性}
}
$$

比如：

$$
\text{technical quality}
+
\text{aesthetic}
+
\text{AIGC flag}
+
\text{semantic tags}
+
\text{safety}
+
\text{text-image alignment}
+
\text{caption}
$$

这些属性后面既可以用于 **hard filter**，也可以用于 **重采样、长尾平衡和 curriculum learning**。论文明确说，这些 profile 不只是为 basic filtering，而是为了量化 data complexity / quality，并支持动态训练阶段的数据构建。
:::

::: tip 一句话总结

> Z-Image 2.1 的核心不是"怎么筛掉坏图"，而是先建立一个多维、可复用的数据画像系统，为后续过滤、采样、平衡和课程学习提供统一的数据基础。
:::

::: warning 原文补充（笔记核对时添加，论文 §2.1 可查）
- **AIGC classifier 的依据**：论文明说 *"Following the findings of Imagen 3 [3]"*——即过滤 AI 生成内容有**已发表的先例依据**，不是为了去重而顺手加的。
- **OCR / 水印检测是 VLM 顺带做的**，这是一个**被明确强调的差异点**：*"diverging from prior works [21, 64, 76] that use separate modules for OCR and watermark detection, our approach leverages the powerful inherent capabilities of our VLM."* 即 Qwen-Image / Seedream 3.0 等用**独立模块**做 OCR 与水印，Z-Image 靠 VLM 的固有能力，省了模块与流水线。
- **AIGC 过滤的真实目的**：论文写明是 *"crucial for preventing degradation in the model's output quality **and physical realism**"*。这条对做真实感生成很重要——**用 AIGC 图训练会同时损伤物理真实性**。
- **跨模态一致性只查 alt caption**：CN-CLIP 算的是 image 与**原始 alt caption** 的相关性，在重生成 caption **之前**。所以这一关是过滤「图文配错」，不是过滤「描述不详细」。
:::

### 2.2 Cross-modal Vector Engine

这一节只需要抓住两个核心：**语义去重** 和 **定向检索**。2.1 的 Data Profiling Engine 是在回答"单条数据本身怎么样"，而 2.2 是在回答"这条数据和整个数据池里的其他数据是什么关系"。因此它不再只看单张图的质量，而是把海量样本放进一个统一的 multimodal embedding space 里，研究哪些样本彼此相似、哪些区域过密、哪些概念稀缺。

在语义去重上，Z-Image 沿用了 Stable Diffusion 3 的思路，但把原来的 `range_search` 换成了更适合超大规模数据的 **kNN search**。`range_search` 是"把某个相似度阈值内的所有邻居都找出来"，在十亿级数据上扩展性很差；kNN 则是"固定找每个样本最近的 $k$ 个邻居"。拿到这些近邻以后，Z-Image 根据 kNN 距离构建 **proximity graph**，再在图上做 **community detection**，把高度相似的数据组织成一个个语义社区。这样就不只是判断两个样本是不是重复，而是能识别一整片高度冗余的数据簇。论文还明确说，当 $k$ 足够大时，这种 kNN 图可以很好地近似原来的 range-search 结果，但工程效率高得多。

它的工程规模也比较值得记：论文给出的结果是，**1 billion items**，在 **8 张 H800** 上完成 index construction 和 **100-NN querying**，大约需要 **8 小时**，而且整个流程是 GPU 加速的。更重要的是，community detection 不只是为了删重复数据，它产生的 semantic structure 和 modularity levels 还可以用于 **fine-grained data balancing**。也就是说，embedding space 某个区域特别密，说明这一类数据过多；某个区域很稀，可能代表长尾概念。这样后续采样时就可以对过密区域降权、对稀缺区域补数据。

Cross-modal Vector Engine 的另一个核心作用是 **retrieval**。Z-Image 可以拿模型生成失败的图片或者 problematic prompt 去向量库里检索相关训练数据。如果发现某个概念的数据太少，就做 targeted augmentation；如果发现某些错误行为和某一类训练数据高度相关，就可以定位并 prune 相应的数据簇。论文明确把这个系统用于发现 **distributional voids**、补长尾数据，以及诊断并清理导致模型错误的数据。

所以这一节最终可以压成一句话：

$$
\boxed{
\text{Multimodal Embedding}
\rightarrow
\text{kNN}
\rightarrow
\text{Proximity Graph}
\rightarrow
\text{Community Detection}
\rightarrow
\text{Dedup + Balancing + Retrieval}
}
$$

::: tip 一句话总结

> Z-Image 的向量引擎不是单纯拿来去重，而是把整个训练数据池变成一个可搜索、可聚类、可诊断的语义空间，最终服务于数据去重、分布平衡、长尾补齐和模型失败修复。
:::

::: info 原文补充（笔记核对时添加，论文 §2.2 可查）
- **社区检测用的是外部算法**：论文标注了引用 [68]，即在 proximity graph 上套用现成的 community detection，而不是自研。
- **kNN 的双重身份**：kNN 距离**既是去重的依据**（构造 proximity graph），**又是检索的索引**（配 SOTA index 算法 [54]）。论文写明检索侧 *"leveraging multimodal features [86] combined with a state-of-the-art index algorithm [54]"*，其中 [86] 正是 §2.1 用于图文一致性打分的同一个 **CN-CLIP**。
- **retrieval 服务的两个对象**：论文明确列出 data curation（找 distributional voids → 定向采样补概念空洞）与 **active model remediation**（用 failure case 反查并 prune 责任数据簇）两条线，后者与 §2.4 的 Active Curation Engine 闭环。
:::

### 2.3 World Knowledge Topological Graph

2.3 **World Knowledge Topological Graph** 的核心其实比名字简单：它是在解决"**训练数据的概念分布怎么被系统地控制**"这个问题。前面的 2.1 告诉你每条数据有什么属性，2.2 告诉你数据之间谁和谁相似；到了 2.3，Z-Image 更进一步，想知道整个数据池里"哪些概念多、哪些概念少、这些概念之间是什么层级关系"，然后据此做更精细的数据采样。论文把这个知识图谱称为整个数据基础设施的"semantic backbone"。

它的构建过程大致分三步。第一步，从 **Wikipedia entities 和 hyperlink structure** 出发，先搭一个非常大但也很冗余的知识图谱；然后做两类 pruning：一类是根据 **PageRank centrality** 去掉特别边缘、几乎没人引用的概念，另一类是用 VLM 判断这个概念是否"可视觉化"，把太抽象、很难稳定生成图像的概念删掉。这里很重要，因为它不是要做一个百科全书式知识图谱，而是要做一个**服务图像生成的数据概念图**。

第二步，他们发现只靠 Wikipedia 还是不够覆盖真实视觉数据，所以又用内部大规模 captioned image data 来补图谱。具体做法是从 caption 里抽取 tags 和 text embeddings，然后做自动层次化组织，再让 VLM 给每个 parent node 总结/命名。这样就把大量离散 tag 整理成一个有父子关系的 taxonomy tree。也就是说，一个"拉面"样本不会只是孤零零一个 tag，它可能会挂在"日本料理 → 面食 → 拉面"这样的层级关系里。这个层级结构后面做 balance 时就比单纯统计 tag 频次更有用。

第三步是让这个图谱和真实产品需求对齐。他们会人工挑选并 **up-weight 高频用户 prompt 对应的概念**，同时主动加入训练池里原本没有、但最近新出现的 trending concepts。这个设计说明图谱不是一次构完就不动，而是会随着真实用户需求和数据分布动态更新。

真正落到训练采样时，Z-Image 会把每条训练 caption 里的 tags 映射到知识图谱节点，再综合 **BM25 score + 图谱里的层级关系**，为每条训练数据算一个 semantic-level sampling weight。这个 weight 再交给数据引擎，决定训练时哪些样本更应该被采到。这样就能避免数据量特别大的常见概念一直占满 batch，同时给长尾概念更高采样概率。论文明确说，这个机制用于 fine-grained control over training data distribution。

所以 2.3 你其实记住这条链就够了：

$$
\boxed{
\text{Wikipedia + Caption Tags}
\rightarrow
\text{Concept Graph / Taxonomy}
\rightarrow
\text{Prune + Expand + Reweight}
\rightarrow
\text{Semantic Sampling Weight}
\rightarrow
\text{Balanced Training Distribution}
}
$$

::: tip 真正需要掌握的不是 PageRank 或 BM25 的公式，而是这个思想

> Z-Image 不只是"数据多不多"地做平衡，而是把概念放进有层级结构的知识图谱里，再按概念稀缺度、层级关系和真实用户需求去控制采样。这就是 2.3 最值得你带走的东西。
:::

::: info 原文补充（笔记核对时添加，论文 §2.3 可查）
- **两类 pruning 的分工**：PageRank 去的是「**没人引用**」（统计孤岛），VLM 去的是「**引用多但画不出**」（概念不可视觉化）。后者是图像生成特有的过滤器，纯 NLP 知识图谱不会做这一刀。
- **层次化是 automatic hierarchical strategy**（论文标注借鉴 [71]），且**每个 parent node 由 VLM 对其子节点做总结命名**——所以 taxonomy 的层级标签本身也是生成的，不是人工给定的。
- **sampling 是 staged 的**：论文原文 *"perform principled, **staged** sampling from the data pool"*，即采样权重不仅决定「采不采」，还决定「在哪个训练阶段采」。
- **它同时是 SFT 的前置设施**：§4.4 的 Concept Balancing with Tagged Resampling 用的 rarity score 正是靠本节的图谱 + BM25 检索算出来的。两节是同一套设施在预训练与 SFT 两个阶段的复用。
:::

### 2.4 Active Curation Engine

2.4 **Active Curation Engine** 是第二节里最值得真正吃透的一块，因为它把前面的 2.1、2.2、2.3 全部串成了一个"数据—模型—再数据"的闭环。

它的核心思想是：**数据集不是一次性清洗完就固定不变，而是随着模型能力不断暴露问题，再反向补数据、修标注、更新数据分布。** 论文里给的例子很典型：模型对"松鼠鳜鱼"这个概念生成失败，说明它可能把"松鼠"和"鳜鱼"做了字面组合，而没有真正学会这道菜。于是系统会把这个 failure case 当成诊断信号，通过前面 2.2 的 cross-modal retrieval 去数据池里找相关样本，同时用规则过滤和去重筛掉低质量数据，再补充这一类长尾概念的数据。也就是说，模型的失败本身变成了下一轮数据采集的触发器。

![Z-Image Fig.5：Active Curation Engine 总览。Z-Image 诊断出长尾概念「松鼠鳜鱼」生成失败 → 文本 embedding 检索 → 去重与规则过滤 → 定向补数据 → Continual Pretraining 回灌模型，构成闭环。](/zimage-fig5-active-curation.png)

另一条闭环是 **captioner 和 reward model 的主动学习**。论文里 Figure 6 讲得比较清楚：系统先用 2.3 的 topology graph 和当前 reward model，从未标注的 media pool 里挑一批概念分布更合理、质量更合适的数据；然后当前的 captioner 和 reward model 给这些数据自动生成 pseudo-label，包括 caption 和 score。接下来不是直接拿去训练，而是经过 **Human verifier + AI verifier** 双重检查；通过的样本直接进入下一步，失败的样本会进入人工修正，专家重新改 caption 或 score。修正后的高质量标注数据再拿回来重新训练 captioner 和 reward model，于是下一轮自动标注会更准。

![Z-Image Fig.6：Human-in-the-Loop 主动学习循环。Media Pool 经 Concept/Quality Balance 挑样 → Reward+Captioner 打 pseudo-label → Human/AI Verifier 双重校验 → pass 直接用，fail 走 Human Correct → 回流重训 Reward/Captioner（虚线）。注意图中 score 被人工从 7/8 改成 9/4，caption 从「精致的」改成「平平无奇的」。](/zimage-fig6-hitl-active-learning.png)

所以整个 2.4 其实可以压成两个闭环。第一个是：

$$
\text{Model Failure}
\rightarrow
\text{Retrieval / Diagnosis}
\rightarrow
\text{Targeted Data Augmentation}
\rightarrow
\text{Retraining}
$$

第二个是：

$$
\text{Media Pool}
\rightarrow
\text{Pseudo Label}
\rightarrow
\text{Human + AI Verification}
\rightarrow
\text{Manual Correction}
\rightarrow
\text{Update Captioner / Reward Model}
$$

这两条链合起来，就是 Z-Image 所谓的 Active Curation。它不再是"数据工程服务模型训练"这种单向关系，而是：

$$
\boxed{
\text{模型暴露问题}
\rightarrow
\text{数据系统响应}
\rightarrow
\text{数据质量提升}
\rightarrow
\text{模型再提升}
}
$$

::: tip 真正需要掌握的分工

> 2.1 负责看单条数据质量，2.2 负责在语义空间里找相似和缺口，2.3 负责控制概念分布，2.4 则把这些能力变成一个持续迭代的 active data loop。这其实非常接近工业界真正的数据飞轮。
:::

这一节学到这里基本就够了，不需要再继续抠太多实现细节。最值得长期记住的一句话是：

::: danger Active Curation 的本质
**不是"主动采样"，而是让模型失败成为下一轮数据构建的监督信号。**
:::

::: info 原文补充（笔记核对时添加，论文 §2.4 可查）
- **论文把 Active Curation 拆成两个明确定义的职能**：一是 *"frontier exploration engine"*（用自动采样找出模型表现差或缺知识的 hard cases），二是 *"closed-loop data annotation pipeline"*（持续精炼数据质量）。前者找问题，后者修标注。
- **松鼠鳜鱼案例的诊断结论是原文的**：模型 *"lacks the specific concept for this dish and may rely on compositional reasoning (combining 松鼠 and 鳜鱼), leading to erroneous generations **absent of domain-specific training data**"*。注意归因是**缺少领域特异训练数据**，不是模型能力不足——所以修法是补数据而不是加参数。
- **Fig. 6 里有一处容易被忽略的细节**：人工修正不只是改 caption，**score 也被改**（图中 7→9、8→4），且改的方向是**下调**。这说明主动学习的目标不是让 reward model 打高分，而是**校准**它。
- **双重校验的分工**：AI Verifier 用的是 **Reward**（图中明确标出），即用奖励模型做自动校验；Human Verifier 处理机器判不准的部分。失败样本不是丢弃，而是走 **Human Correct** 修正后**回流**去重训 Reward/Captioner（图中虚线）。
- **人机分工的隐含成本**：这条闭环里 human verifier 和 manual correction 都是**不可并行扩展**的人力环节，与 §1.2 里「\$628K 未计入人工成本」的判断一致——**数据飞轮转得越快，人力投入越大**。
:::

### 2.5 Efficient Construction of Editing Pairs with Graphical Representation

2.5 这一节和前面不太一样，前面 2.1–2.4 讲的是通用数据基础设施，2.5 开始专门讲 **图像编辑数据怎么构造**。它要解决的核心问题是：高质量编辑 pair 很难大规模获得，因为不仅要有 source/target 两张图，还要有准确的 edit instruction，而且编辑后还得尽量保持没改的区域一致。Z-Image 的做法不是只依赖一种数据来源，而是组合了几条路线。

![Z-Image Fig.7：编辑数据的三种构造策略。(a) Graphical Representation——一张输入图经编辑 2/3 得到两个版本，两版本间还能互为 source/target（箭头 5/6），实线是编辑操作、虚线是反向 pair；(b) Paired Image from Videos——同一庭院的前后两帧，指令是「替换橙色圆形踏步石为 5 块长方形板岩、移除小树换成低矮松树等」的长指令；(c) Rendering for Text Editing——可控渲染生成的文字编辑 pair，指令直接由渲染操作产生。](/zimage-fig7-editing-pairs.png)

第一条是 **Mixed Editing with Expert Models**。他们先定义一套编辑任务 taxonomy，然后用不同的 task-specific expert model 去生成高质量编辑数据。关键点在于，他们不满足于"一对图只学一个编辑动作"，而是会把多个编辑操作合并进同一个 pair，形成 mixed-editing data。比如同一张图里既换背景、又改颜色、再增加物体，这样一条样本就能同时教模型多个操作，提高训练效率。论文明确说，这样可以让模型从一个 composite pair 里学习多个 editing task，而不是分别准备很多单任务 pair。

第二条也是这一节最有特点的，是 **Efficient Graphical Representation**。对于同一个 input image，他们先生成多个不同编辑版本。然后这些版本之间可以继续两两组合，构造新的 source-target pair。论文的意思是：原始图和 $N$ 个编辑版本之间，不只是有 $N$ 对关系，还可以通过不同 edited versions 之间的组合进一步扩增 pair 数量。这样一来，一组已经生成好的编辑结果可以被反复重组，不需要重新调用 expert model，就能把训练数据规模继续放大。与此同时，这种重组天然会产生 mixed-editing pair，也会产生 inverse pair。<mark class="hl-key">作者特别强调 inverse pair 的意义：可以让"真实、未失真的图"作为 target，从而提升数据质量。</mark>

第三条是 **Paired Images from Videos**。预定义编辑任务的缺点是分布太人工、编辑类型有限，所以 Z-Image 又从大规模视频里取自然相邻或相关 frame。因为同一段视频里的不同帧往往共享主体、场景或风格，它们天然就有一定的 editing relation。论文再用 CN-CLIP 计算 frame pair 的语义相似度，筛掉关系太弱的 pair。这样得到的数据有三个优势：任务类型更丰富、很多 pair 天然包含多个同时变化的因素，而且规模更容易扩展。

最后一条是 **Rendering for Text Editing**。文字编辑数据天然特别稀缺，因为真实图里带文字的样本本来就不均衡，而且 source-target pair 很难有精准编辑标注。所以他们专门做了一个可控文字渲染系统，可以控制 text content、font、color、size、position。这样就能自动生成大规模 text-editing pair，而且 instruction 是由渲染操作本身直接产生的，因此标注精度很高。

所以 2.5 你最终记成一条链就够了：

$$
\boxed{
\text{Expert Models}
+
\text{Pair Recombination}
+
\text{Video Frames}
+
\text{Text Rendering}
\rightarrow
\text{Large-scale Editing Pairs}
}
$$

::: tip 真正值得记住的是四来源互补

> Z-Image 的编辑数据不是单一合成路线，而是"专家模型保证质量，图结构重组提高数据利用率，视频帧补自然多样性，渲染系统解决文字编辑"这四种来源互补。
:::

如果你后面要做统一生成/编辑模型，这一节其实很值得迁移，因为它最直接回答了一个工程问题：**编辑数据贵的时候，怎么把有限的高质量 pair 扩成足够大的训练集。**

::: info 原文补充（笔记核对时添加，论文 §2.5 可查）
- **组合数有明确公式**：论文写明一张输入图 + $N$ 个编辑版本可构造 $\binom{N+1}{2}$ 个 pair（引用 [41]），且原文用词是 *"scale the training data at **zero cost**"* —— 零成本指的是**不再调用 expert model**，不是零算力。
- **Fig. 7(a) 的箭头含义**：实线（如 2、3）指 editing operation，虚线（如 1、4、5、6）指反向/inverse pair。图里 $5 \leftrightarrow 6$ 那对**两个 edited version 互为 source/target**，正是"重组产生新 pair"的可视化。
- **四条路线的分工是互补而非冗余**：taxonomy + expert model 负责**任务覆盖广度**；graphical representation 负责**数据利用率**；video frames 负责**多样性**（论文原话：预定义任务 *"suffers from limited diversity"*）；rendering 负责**文字编辑**（自然图 *"suffer from the scarcity and imbalance of textual content"*）。
- **video pair 的筛选阈值是"高语义相关性"而非"相邻帧"**：论文用 CN-CLIP 算 image embedding 的 **cosine similarity**，在每个 image group 内筛出高相关对。也就是说采的是"同组语义相关的帧"，不要求时间上紧邻。
- **video pair 的天然优势是"多编辑类型耦合"**：论文举例 simultaneous changes in human pose and background —— 这类**同时发生**的多因素变化，用预定义 taxonomy 反而很难造出来。
- **渲染系统借用了既有工作**（论文标注 [76]，即 Qwen-Image 一系），不是自研；其价值在于 **ground-truth instruction 由渲染操作本身给出**，天然免去人工标注。
:::

---

## 3. Image Captioner（论文 §3） { #sec-3-captioner }

第 3 节 **Image Captioner** 的总览其实很清楚：Z-Image 不是把 caption 当成"给图片写一句描述"，而是把它当成**训练监督信号的设计问题**。这一节的目标，是让同一张图同时拥有适合不同训练需求的文本表达，并且让这些文本尽可能覆盖图里的文字、世界知识、细节信息，以及编辑前后的差异。Figure 8 里给出的整体结构就是：单图经过 Z-Captioner，结合 **World Knowledge、OCR Augmentation、Tagging**，生成多层级的 T2I captions；对于 image pair，则生成专门的 image editing instruction。

![Z-Image Fig.8：Z-Captioner 双路流水线。上路 Single Image → 生成 Tagging/Short/Long Caption 三类 T2I caption；下路 Image Pair → Step1 Caption / Step2 Analysis / Step3 Instruction 三步生成编辑指令。World Knowledge 从上方注入，OCR Augmentation 从下方注入。](/zimage-fig8-captioner-pipeline.png)

对于单张图，Z-Captioner 最终不是只输出一种 caption，而是会生成不同粒度的描述。论文后面明确说一共设计了 **5 种 caption：long、medium、short、tags 和 simulated user prompts**。长 caption 尽量完整描述图像内容，适合精细图文对齐；短 caption 和 tags 更简洁；simulated user prompt 则专门模拟真实用户那种"短、不完整、只说自己关心部分"的输入。这样做的目的，是让模型既能学会精确对应复杂长描述，也能适应真实用户比较随意、信息不完整的 prompt。

这一节还有两个很关键的增强。第一是 **OCR-aware captioning**：Z-Image 特别强调，图里如果有文字，要先显式识别出来，而且保留原语言，再把 OCR 结果写进 caption，因为他们认为这和最终文字渲染能力直接相关。第二是 **world knowledge injection**：caption 不只是描述"看到了什么"，还会结合 meta information 去识别具体实体、地标、事件，尽量减少 named entity 的错误和幻觉。论文明确把这两点作为 Z-Captioner 的核心设计。

对于图像编辑，逻辑又不一样。Z-Captioner 不直接看 source/target 就一句话猜 edit instruction，而是走三步：先分别给 source 和 target 做详细 caption，然后做 difference analysis，最后再把这些差异压缩成 concise editing instruction。也就是：

$$
\text{Source/Target Caption}
\rightarrow
\text{Difference Analysis}
\rightarrow
\text{Editing Instruction}
$$

这样做的好处是把"看懂两张图"和"总结编辑动作"拆开，能更系统地覆盖视觉变化和文字变化。

![Z-Image Fig.9：caption 实例。左半为单图的三类 caption——Tagging Caption 是一长串逗号分隔标签（含 OCR 转写与地名 'West Lake, Hangzhou, China, Leifeng Pagoda'、中英文字 '2025 杭州美食节'、'中雨香'），Long Caption 里把图中文字逐条转写并保留原语言；右半为差分 caption 的三步：source 是白猫特写，target 是拟人猫穿西装站在海滩，Step2 分析出 Subject modification / Element addition / Scene change 三类差异，Step3 压缩成一句自然语言指令。](/zimage-fig9-caption-examples.png)

::: tip 第 3 节的总体印象

> Z-Captioner 的核心不是 caption 越长越好，而是针对不同训练任务设计不同形式的监督文本。T2I 需要多粒度 caption、OCR 和 world knowledge；Editing 则需要 source-target difference caption。后面 3.1、3.2、3.3 其实就是分别把这三块展开。
:::

::: info 原文补充（笔记核对时添加，论文 §3 可查）
- **Z-Captioner 是"all-in-one"**：论文明确说它 *"by incorporating **multiple types** of image caption"* 建成一个全功能 captioner，依据是 *"different captioning tasks can benefit each other as they share the same goal of understanding and depicting images"*（引用 [49]）。**多任务 captioner 反而互相增益**，不是简单堆功能。
- **OCR 的因果是论文用实验断言的**：原文 *"according to our experiments, including explicit OCR information in image captions is **inextricably bound** with accurate text rendering"* —— 不可分割。但论文**未给该实验的消融表**，属未验证细节。
- **OCR 强制不翻译**：*"we **force the OCR results to remain in their original languages without any translation**, avoiding them being falsely rendered in their translated languages"* —— 这条对中英混排海报很关键。
- **§3.2 的两个反直觉设计**（原文可查，是本节最容易被漏掉的）：① 长 caption *"deliberately adopt a **plain and objective** linguistic style ... strictly confining them to factual information"*，主动**抑制主观想象**以提升数据效率；② 模拟用户 prompt *"are **incomplete** prompts"*，与 short caption 有本质区别 —— short caption 描述全图，模拟 prompt *"focusing only on specific parts of interest to the user, while making no mention of the rest of the image"*。
- **差分 caption 的三步 CoT 借鉴了 [100]**，且 Step2 明确 *"leveraging both the raw images and their generated captions"* —— 即比较时**既看图也读 caption**，不是纯文本比对。
- **Fig. 9 左图暴露了一个 caption 设计风险**：Tagging Caption 里 OCR 转写、地名、风格标签、杂志刊名、页码文字**全部平铺进同一串逗号列表**，长尾且无语义结构。论文把它当 tag 用途是合理的，但这类监督若占比过高，可能诱导模型输出堆砌式 prompt。这是论文未讨论的潜在副作用。
:::

### 3.1 Detailed Caption with OCR Information

3.1 **Detailed Caption with OCR Information** 这一节其实很短，但它对 Z-Image 的文字生成能力很关键。论文最明确的结论是：**如果图像里本身有文字，那么 caption 里必须显式包含这些 OCR 信息，否则模型很难学好文字渲染。** 作者甚至直接说，他们实验中观察到，把 OCR 信息明确写进 image caption，和最终生成图像中的准确 text rendering 是"inextricably bound"的。

它的做法不是"先生成整段 caption，再顺手补 OCR"，而是采用一个类似 CoT 的两步流程：**先把图像里所有可见文字识别出来，再基于这些 OCR 结果去生成完整 caption**。这么做的原因是，如果直接让 captioner"一次性描述整张图"，当图中文字很多、很密时，模型很容易漏掉部分文字；先单独做 OCR，相当于先把最容易丢失的文本信息显式抽出来，再让后续 caption 必须基于这些结果生成。论文特别强调，这对长文本、密集文字场景尤其重要。

还有一个很容易忽略但很关键的细节：**OCR 结果保持原语言，不做翻译。** 比如图片里写的是中文，就保留中文；写的是英文，就保留英文。作者这么做是为了避免 caption 阶段把文字翻译掉，导致后面生成时模型学成"看到中文场景却输出英文翻译"这种错误映射。这个设计对 Z-Image 的中英文双语 text rendering 很重要。

所以 3.1 你可以直接记成这一条：

$$
\boxed{
\text{Image}
\rightarrow
\text{Explicit OCR First}
\rightarrow
\text{Caption conditioned on OCR}
}
$$

真正要掌握的点只有两个：**第一，图中文字要作为显式监督信号进入 caption；第二，OCR 要先做、且保留原语言。** 这一节不需要再往下抠，因为论文这里没有公开 OCR 模型结构、OCR loss、识别阈值或者具体训练细节。

::: warning 这一节的论证强度
*"inextricably bound"* 是很强的措辞，但**论文没有给出对应的消融表或数字**（CVTG-2K / LongText-Bench 只报了最终成绩，没报"去掉 OCR 增强会掉多少"）。所以这是一个**实验断言但未量化**的结论。可以放心当作设计原则接受，但引用时不宜说成"论文证明了"。
:::

实例见上文 [Fig. 9 左半](#sec-3-captioner) —— Long Caption 里把围裙上的「2025 杭州美食节」和挂签上的「中雨香」逐字转写并保留原语言，正是这条规则的产物。

### 3.2 Multi-Level Caption with World Knowledge

3.2 **Multi-Level Caption with World Knowledge** 这一节的核心，不是"把 caption 写得更长"，而是让同一张图同时拥有**不同粒度、不同用途**的文本监督。论文明确说，Z-Captioner 一共设计了 5 类 caption：**long、medium、short、tags、simulated user prompts**。其中 long/medium caption 尽可能覆盖图像里的主体、物体、背景、位置、OCR 等完整信息，用来建立精细的 text-image mapping；而 short、tags 和 simulated user prompts 更接近真实用户输入，尤其 simulated user prompt 故意是不完整的，只描述用户可能真正关心的局部内容，而不是把整张图都说一遍。

这里最值得理解的是为什么要多粒度。论文的逻辑是：如果训练时永远只给超详细长 caption，模型会很擅长"按说明书作图"，但真实用户往往只会输入很短、很模糊的 prompt。反过来，如果只用短 prompt，模型又学不到足够细的视觉-语言对应关系。所以 Z-Image 同时保留不同粒度，本质是在兼顾两种能力：**精细对齐** 和 **真实用户 prompt 适应性**。论文在 omni-pretraining 部分也明确说，多粒度 caption 和不同视角的描述能够提供更广的 mode coverage，有利于后续训练。

"with World Knowledge" 则是另一层增强。Z-Image 不希望 captioner 只描述"这里有一栋塔、一片湖"，而是希望它能在有足够证据时识别出"这是杭州西湖、雷峰塔"这种具体实体。因此他们在 caption 生成时引入 meta information，把 world knowledge 注入到所有 5 类 caption 中。论文明确说，这么做是为了减少对 public figures、famous landmarks、known events 这类 named entities 的 hallucination。也就是说，world knowledge 的作用不是让 caption 更文学，而是让它在具体实体命名上更准确。

同时，Z-Image 对长 caption 的风格还有一个很明确的限制：**plain and objective**。它要求描述尽量基于图中可观察事实，避免主观解释和想象性联想。作者认为这样可以减少无关信息，提高 image generation 训练的数据效率。

所以 3.2 最后你可以记成：

$$
\boxed{
\text{同一张图}
\rightarrow
\text{多粒度 Caption}
+
\text{World Knowledge}
}
$$

其中，多粒度解决"详细监督"和"真实用户 prompt"之间的差异；world knowledge 解决 named entity 和具体世界概念的准确识别；而整体 caption 风格保持客观、事实化，避免无关想象。

::: info 原文补充（笔记核对时添加，论文 §3.2 可查）
- **5 类 caption 的原文措辞**：*"We design five different types of image captions in total, including long, medium and short captions, as well as tags and simulated user prompts."* 注意是**五类**，"simulated user prompts" 单列一类。
- **长 caption 塞的东西有清单**：原文列了 *"full OCR results as mentioned above, along with subjects, objects, background, location information, et al."*
- **客观文风的目的是数据效率不是可读性**：原文 *"By **inhibiting subjective interpretations and imaginative associations**, our purpose is to **enhance data efficiency** ... by eliminating non-essential information."* —— 这是主动**牺牲 caption 的文学性**换训练信号纯度。
- **world knowledge 的注入条件是"条件于 meta information"**：§2.1 Data Profiling Engine 产出的元信息在这里被 captioner 消费。**§2.1 → §3.2 是一条直接的数据流**：没有 profiling engine 的元信息，captioner 无从做实体命名。
- **与 §4.3 的呼应**：你提到的 "mode coverage 有利于后续训练" 在论文 §4.3 有对应原句 —— *"The use of captions at different granularities and from diverse perspectives provides **broad mode coverage**, which is beneficial for subsequent stages of training."* 这条在 §4.3 讲 omni 预训练时会再次出现。
- **medium caption 的正例在附录**：§3 总览的 Fig. 9 只展示了 long / short / tags 三类的实例，**medium caption 与 simulated user prompt 没有给实例**。
:::

多粒度实例见上文 [Fig. 9 左半](#sec-3-captioner)。下一节 3.3 会转到 **source-target pair 怎么生成 editing instruction**。

### 3.3 Difference Caption for Image Editing

3.3 **Difference Caption for Image Editing** 这一节的核心，就是把一对 source image / target image，转换成一条尽可能准确、简洁的编辑指令。Z-Image 没有直接让 captioner 看两张图然后"一步到位"生成 instruction，而是采用一个三阶段过程：**先分别详细描述 source 和 target，再显式分析两者差异，最后把差异压缩成 editing instruction。** 论文把这个过程称为一种 step-by-step / CoT 风格的生成方式。

第一步是 **Detailed Captioning**。对 source 和 target 两张图分别生成完整 caption，而且这些 caption 会包含 OCR 信息。目的就是先把两张图各自的内容尽量"说清楚"，避免直接比较原图时漏掉细节。第二步是 **Difference Analysis**，模型同时参考原始图像和两边的详细 caption，从视觉和文字两个角度把所有变化找出来，比如主体变化、物体增删、背景变化、姿态变化、文字变化等。第三步才是 **Instruction Synthesis**，把前面的差异总结成一条简洁、可执行的编辑指令。论文给出的例子就是：source 里是一只普通猫，target 里这只猫被放到海滩、换成人形西装身体、手里多了酒杯；最终 instruction 会把这些变化统一组织成一条自然语言编辑命令。

这个设计最值得你学的是：**把"理解两张图"和"生成编辑指令"拆开。** 如果直接一步生成 instruction，很容易漏掉小变化，或者把 source/target 里共同存在的内容也误写成编辑操作。Z-Image 先让模型分别理解两边，再做 difference analysis，相当于先得到一个结构化的变化列表，再压缩成 instruction，所以监督会更干净。

你可以把 3.3 记成这条链：

$$
\boxed{
(I_s, I_t)
\rightarrow
(C_s, C_t)
\rightarrow
\Delta(I_s,I_t)
\rightarrow
\text{Edit Instruction}
}
$$

其中 $C_s, C_t$ 是 source/target 的详细 caption，$\Delta$ 是差异分析。论文没有公开这三步各自用什么 prompt 模板、具体模型结构或过滤阈值，所以学到这个程度就够了。

::: tip 最需要记住的一句话

**Difference Caption 的本质，是把 source-target pair 先转成"可解释的差异"，再生成最终编辑指令。**
:::

实例见上文 [Fig. 9 右半](#sec-3-captioner) —— Step2 把差异归成 Subject modification / Element addition / Scene change 三类，Step3 压缩成一句指令。

::: info 原文补充（笔记核对时添加，论文 §3.3 可查）
- **原文对三步的定位**：论文把 Step2 称为 *"Difference Analysis"*，并写明它 *"leveraging both the raw images and their generated captions, to tell all discrepancies from **visual and textual** perspectives"* —— 即比较时**图和 caption 同时看**，且显式区分**视觉差异**与**文字差异**两类。
- **三步 CoT 借鉴 [100]**：原文 *"we employ a three-step CoT process that systematically breaks down the comparative task [100]"*。
- **Step1 复用 §3.1 的 OCR-inclusive caption**：论文写 Step1 是 *"generate a comprehensive, **OCR-inclusive** caption for both the source and target images respectively"* —— 所以文字变化是被**显式**纳入差异分析的，这也是为什么文字编辑（§2.5 的 rendering 路线）能有精确监督。
- **与 §2.5 是一条闭环**：§2.5 用渲染系统造出 source/target 像素对，§3.3 用这三步为它们生成 instruction 文本。**前者保证像素级 ground-truth，后者保证语言级 ground-truth**，两条路线的产物在这里汇合。
- **一个论文未讨论的依赖**：Step2 的差异质量取决于 Step1 caption 的质量。若 captioner 漏掉了某个细微变化（§3.1 承认密集文字场景会漏字），**这个变化在后续两步里就彻底不可见了** —— 误差被前置步骤静默传递。这是 CoT 式 pipeline 的共性风险，论文没有分析。
:::

---

## 4. Model Training（论文 §4） { #sec-4-training }

> 本节是全文最重的一节，8 个小节，按流水线顺序：
>
> | 小节 | 内容 | 配图 |
> | :--- | :--- | :--- |
> | §4.1 | 架构 S3-DiT | Fig. 10 架构图 |
> | §4.2 | 训练效率优化（纯工程） | 无 |
> | §4.3 | 预训练（Flow Matching） | 无 |
> | §4.4 | SFT 三件套 | 无 |
> | §4.5 | 少步蒸馏 D-DMD / DMDR | Fig. 13 蒸馏对照 |
> | §4.6 | RLHF（DPO → GRPO） | Fig. 14 RLHF 对照 |
> | §4.7 | Z-Image-Edit 继续训练 | 无 |
> | §4.8 | Prompt Enhancer | Fig. 15 PE 可视化 |
>
> **真实流水线顺序（Fig. 11 坐标重建）**：低分辨率预训练 → Omni 预训练 → SFT → **少步蒸馏** → **RLHF**；编辑分支从 SFT 处向下分出（Continued PT → SFT for Editing）。

### 4.1 Architecture

Z-Image 的主干是一个 **6.15B 参数的 Scalable Single-Stream Diffusion Transformer（S3-DiT）**。文本侧使用 **Qwen3-4B** 作为 text encoder，图像侧使用 **Flux VAE** 把 RGB 图像编码成 latent；只有在 image editing 任务中，才额外加入 **SigLIP 2** 提取 reference image 的高层语义特征。整个模型采用 single-stream 设计：不同模态先分别经过很轻量的 modality-specific processor，再把 text token、image VAE token，以及 editing 场景中的 semantic token 拼接成同一个序列，送入统一 Transformer backbone。这样做的目的，是让不同模态在每一层里直接交互，同时提高参数利用率。

S3-DiT 的具体规模是 **30 层、hidden dimension 3840、32 个 attention heads、FFN intermediate dimension 10240，总参数 6.15B**。每种输入模态先经过由 2 个 Transformer block 构成的轻量 processor 做初步对齐，然后进入统一主干。为了保证训练稳定，模型使用 **QK-Norm、Sandwich-Norm 和 RMSNorm**；条件信息会被投影成 scale 和 gate 去调制 Attention / FFN。这个条件投影还采用 low-rank 形式，即共享一个 layer-agnostic down-projection，再接每层自己的 up-projection，以降低额外参数量。

在位置编码上，Z-Image 使用 **3D Unified RoPE**。图像 token 使用空间坐标，文本 token 沿 temporal 维度递增。对于 image editing，reference image 和 target image 的空间 RoPE 坐标是对齐的，也就是说 reference 左上位置和 target 左上位置保持空间对应；但两张图在 temporal 维度上额外加入一个 unit interval offset，从而让模型知道"这两个 token 虽然空间位置对应，但属于不同图像角色"。因此 RoPE 同时承担了两个作用：保持 source-target 的空间对应关系，同时区分 reference 和 target。

editing 场景里还有一个非常关键的设计，就是 **reference 和 target 使用不同的 diffusion time-conditioning**。Z-Image 使用 flow matching：

$$
x_t=t\,x_1+(1-t)\,x_0
$$

其中 $x_1$ 是 clean image，$x_0$ 是 Gaussian noise，因此在它的定义里：

$$
t=1 \Rightarrow \text{clean image}, \qquad t=0 \Rightarrow \text{pure noise}
$$

因此在 editing 训练时，reference image 始终作为干净条件输入：

$$
t_{\text{ref}}=1
$$

而 target image 正常参与 flow-matching 加噪：

$$
t_{\text{target}}\in[0,1]
$$

Figure 10 里明确画出了 reference 的 $t=1$，target 的 $t\in[0,1]$。模型实际看到的可以抽象成：

$$
\big[\ \text{text tokens},\ \text{clean reference tokens}(t=1),\ \text{noisy target tokens}(t)\ \big]
$$

然后根据 clean reference 和编辑指令，去预测 noisy target 的 velocity。

所以这里实际上有两套机制共同区分 reference / target：

$$
\boxed{\ \text{3D RoPE}\ }
$$

负责"**空间对齐 + 图像角色区分**"；

$$
\boxed{\ t_{\text{ref}}=1,\quad t_{\text{target}}\in[0,1]\ }
$$

负责"**clean condition + noisy generation target 的区分**"。

整个 4.1 最后可以压成：

$$
\boxed{
\text{Qwen3-4B}+\text{Flux VAE}+\text{SigLIP2(edit only)}
\rightarrow \text{Single-Stream S3-DiT}
}
$$

::: tip 最值得记住的三个核心设计
**单流统一处理多模态 token；3D RoPE 保持 reference-target 的空间对应并区分角色；editing 中 reference 固定 $t=1$，target 随机采样 $t\in[0,1]$，从而把"条件图"和"需要生成的图"明确分开。**
:::

![Z-Image Fig.10：S3-DiT 架构总览。左侧为模态处理器（Text Processor←Qwen3-4B、Image Processor←Noised VAE Embedding、Semantic Processor←SigLip-2 Embedding 仅编辑用），各自带 Timestep Condition 嵌入，经 Embed 后 ⊕ 拼接成统一序列进入主干。中部为 ×N 重复的 Single-Stream Attention Block 与 FFN Block。右侧放大两个 block 内部：Attention Block 为 RMS Norm → Scale → Q-Norm/K-Norm → U-RoPE（仅作用于 Q 与 K）→ Multi-head Self-Attention → Zero-init. Gate → 残差；FFN Block 为 RMS Norm → Scale → Feed Forward → RMS Norm → Zero-init. Gate → 残差。底部两行是输入示例：# Z-Image 行「文字 prompt + 单张目标图，t = [0,1]」；# Z-Image-Edit 行「两张 reference 图 t = 1 + 编辑指令 + 目标图 t = [0,1]」。](/zimage-fig10-architecture.png)

::: info 原文补充（笔记核对时添加，论文 §4.1 可查）
- **Table 2 完整配置**：除你列的 5 项外还有 RoPE 频率三元组 $(d_t,d_h,d_w)=(32,48,48)$，即 3D RoPE 在时间/高/宽三个轴上用不同频率。
- **引文出处**：single-stream MM-DiT 范式引 [18]（SD3 系），3D Unified RoPE 引 [58, 78]，Qwen3-4B 引 [85]，Flux VAE 引 [34]，SigLIP 2 引 [69]，低秩条件投影思路引 [1]。
- **Fig. 10 揭示了正文没写的 block 内部顺序**（已放大核对）：**Scale 调制的是归一化之后的输入**（adaLN 式），而 **Zero-init. Gate 位于 block 输出侧、残差相加之前**。即 $h' = h + g\odot F(\mathrm{Norm}(h)\odot(1+s))$，且 $g^{(0)}=0$ ⇒ 初始为恒等映射。这是小模型稳定性的关键一环。
- **U-RoPE 只作用于 Query 与 Key**，不作用于 Value —— 图中 U-RoPE 框只接在 Q、K 两条路上。
- **编辑态有两个独立的 Timestep Condition 嵌入**：Fig. 10 中 Timestep Condition 出现两次，各带一个 Embed —— 一个恒为 $t=1$ 服务 reference，一个随机 $t\in[0,1]$ 服务 target。这就是「两套机制」在实现上的落点。
- **editing 的 token 序列比 T2I 多一段**：编辑态序列为 $[\text{SigLip 语义}\oplus\text{VAE ref}(t{=}1)\oplus\text{文本}\ldots\oplus\text{含噪 VAE target}]$，即 reference 同时经过 SigLip-2（取抽象语义）和 VAE（取像素细节）两条路，而 target 只走 noised VAE。
- **teacher 的推理开销**：§4.5 明确 *"our standard SFT model requires approximately **100 NFEs**"*，Turbo 压到 8 NFE（约 1/12.5）。**注意论文全文没有出现「50 步」这种表述**，不要把 CFG 的两次前向和步数混算。
:::

### 4.2 Training Efficiency Optimization

这一节主要解决的不是模型能力问题，而是一个很典型的工业训练问题：**同样的模型和数据，怎么尽可能降低显存占用、减少无效计算，并把 GPU 吃满。** Z-Image 主要从两方面做优化：一方面是根据不同模块的训练状态选择不同的并行和显存策略；另一方面是针对图像模型多分辨率、序列长度变化大的特点，重新设计 batch 构造方式。

首先是 <mark class="hl-trick">**hybrid parallelization strategy**</mark>。Z-Image 并不是整个模型统一使用一种并行方式，而是分模块处理。<mark class="hl-trick">VAE 和 Text Encoder 在训练过程中保持 frozen，只负责 forward，因此它们没有梯度和 optimizer state 的大额显存开销，所以使用普通 **Data Parallelism（DP）** 即可</mark>：每张 GPU 都保留完整模块副本，不同 GPU 处理不同的数据。<mark class="hl-trick">真正需要更新的是大规模 DiT backbone，它的参数、梯度和 optimizer states 会占据大量显存，因此使用 **FSDP2**</mark>。FSDP2 的核心就是把这些训练状态 shard 到不同 GPU 上，而不是让每张 GPU 都保存完整副本，从而降低单卡显存压力。

除了模型状态，训练时另一个显存大户是 <mark class="hl-trick">**activation**</mark>。正常情况下，forward 经过每一层时产生的中间 activation 都要保存，backward 时再用这些中间结果计算梯度。<mark class="hl-trick">对于 DiT 来说，层数多、图像 token 序列又长，这部分显存会非常大</mark>。因此 Z-Image 对所有 DiT layers 都采用 <mark class="hl-trick">**Gradient Checkpointing**</mark>：只保存一部分关键中间节点，其余 activation 不保留，等 backward 真正需要时再重新执行对应的一段 forward。它本质上是在做：

$$
\boxed{\ \text{更多计算} \rightarrow \text{更少 activation 显存}\ }
$$

因此 checkpointing <mark class="hl-trick">不会让模型变小，也不会改变训练目标，只是通过"反向时重新算"换取更低显存，从而支持更大的 batch 或更长的序列</mark>。

Z-Image 还对 DiT blocks 使用了 <mark class="hl-trick">**`torch.compile`**</mark>。<mark class="hl-trick">这个东西解决的不是显存切分，而是运行效率</mark>。普通 PyTorch eager execution 会把很多操作逐个调度到 GPU，存在 Python 调度、kernel launch 等额外开销；`torch.compile` 会尝试把计算图编译优化，把可以融合的操作合并起来，减少这些调度开销。因此它的目标可以简单理解为：<mark class="hl-trick">**同样的 DiT forward/backward，尽可能让 GPU 更高效地执行，提高整体训练 throughput。**</mark>

第二部分是 Z-Image 针对 <mark class="hl-trick">**mixed-resolution training**</mark> 做的优化。图像分辨率和宽高比不同，经过 VAE 后得到的 image token 数量也不同，<mark class="hl-trick">也就是说每个样本的 sequence length 差别可能非常大</mark>。如果一个 batch 里同时放一个很短的序列和一个很长的序列，为了能够并行计算，短序列通常需要 padding 到最长序列长度，<mark class="hl-trick">这样大量计算实际上都浪费在 padding token 上</mark>。Z-Image 因此会<mark class="hl-trick">根据 metadata 里的 height 和 width，提前估计每个样本的 sequence length，然后把长度相近的数据分到同一个 batch，这就是 **sequence-length-aware batch construction**</mark>。

在此基础上，他们又做了 <mark class="hl-trick">**dynamic batch sizing**</mark>。<mark class="hl-trick">长序列 batch 使用较小的 batch size，避免 OOM；短序列就可以使用更大的 batch size，避免 GPU 显存和计算资源闲置</mark>。因此可以简单记成：

$$
\text{Long sequence} \Rightarrow \text{Small batch}
$$

$$
\text{Short sequence} \Rightarrow \text{Large batch}
$$

这样不同分辨率的数据虽然 token 数量不同，但<mark class="hl-trick">每个 batch 都尽可能接近硬件承载上限，提高整体 GPU utilization</mark>。

所以 4.2 的完整逻辑可以压缩成：

$$
\boxed{\ \text{VAE/Text Encoder frozen} \Rightarrow \text{DP}\ }
$$

$$
\boxed{\ \text{Trainable DiT} \Rightarrow \text{FSDP2} + \text{Gradient Checkpointing} + \text{torch.compile}\ }
$$

再加上：

$$
\boxed{\ \text{Multi-resolution Data} \Rightarrow \text{Length-aware Batching} + \text{Dynamic Batch Size}\ }
$$

::: tip 真正需要长期记住的
**Z-Image 会<mark class="hl-trick">先根据"模块是否需要训练"决定并行策略，再根据"序列到底有多长"决定 batch 怎么组</mark>。FSDP2 解决模型训练状态太占显存，Gradient Checkpointing 解决 activation 太占显存，`torch.compile` 提升计算吞吐，length-aware batching 和 dynamic batch sizing 则减少多分辨率训练里的 padding 和显存浪费。** 这就是 4.2 的全部核心内容。
:::

::: info 原文补充（笔记核对时添加，论文 §4.2 可查）
- **引用出处**：FSDP2 引 [96]，torch.compile 作为 JIT 编译器引 [1]。论文只给了这两处引用，**没有给任何加速比或吞吐数字** —— 全节是纯工程描述，**零量化指标**。
- **原文对 gradient checkpointing 的措辞值得记**：*"This technique trades an **acceptable increase in computational cost** for significant memory savings, **enabling larger batch sizes and improved overall throughput**."* 注意它把收益定义成"**能做更大的 batch**"，而不是"跑得更快"——这个收益最终又被后面的 dynamic batch sizing 放大了。
- **dynamic batch 的原始动机有两个，不止防 OOM**：论文写的是小 batch 防 *"Out-Of-Memory (OOM) errors"*，大 batch 防 *"resource vacancy"*（**资源闲置**）。后者常被忽略 —— 它的真正目标是**避免 GPU 空转**，而不只是避免显存溢出。
- **sequence length 是"估计"而非"精确"**：论文说 *"we **estimate** the sequence length of each sample **based on the resolution** recorded in the metadata"*，即只用 H×W 推算，**不读 VAE 实际输出**。这是纯 metadata 侧的近似，代价极小，但对自由宽高比的图只能估个均值。
- **本节与 §4.1 的联系**：§4.1 的 3D Unified RoPE 意味着图像 token 数随 (H, W) 变化，§4.3 的 arbitrary-resolution 训练又把分辨率彻底放开 —— **正是这两点让 §4.2 的 length-aware batching 成为必需项，而不是可选优化**。
- **论文未讨论的取舍**：gradient checkpointing（多算）与 dynamic batch（多填显存）**方向相反**，两者叠加后的最优组合依赖具体硬件，论文没有给出任何调参指引或消融。
:::

### 4.3 Pre-training

4.3 **Pre-training** 是 Z-Image 训练流程里非常关键的一节，因为它真正把前面的 Data Infrastructure、Captioner 和模型结构接起来了。整体上预训练分成两个阶段：**Low-resolution Pre-training → Omni-pre-training**。前者先在低分辨率上高效地把基础视觉知识和图文对齐学起来，后者再扩展到任意分辨率、生成+编辑联合训练以及多粒度 caption。

![Z-Image Fig.12：训练各阶段的生成结果演进（实际横跨 §4.1–§4.6 全流程）。列为 (a) Pre-train → (b) SFT → (c) PE → (d) FSD → (e) RLHF。可观察到 pre-train 阶段构图/文字尚不准确但语义已成立，SFT 后画面质量与美学跃升，PE 阶段推理链补齐了复杂构图，蒸馏阶段保住质量，RLHF 阶段写实感与光影进一步收敛。](/zimage-fig12-training-stages.png)

先看基础训练目标。Z-Image 使用 <mark class="hl-trick">**Flow Matching**</mark>。从高斯噪声 $x_0$ 和真实图像 latent $x_1$ 之间做线性插值：

$$
x_t=t\,x_1+(1-t)\,x_0
$$

然后让模型预测这条路径上的 velocity：

$$
v_t=x_1-x_0
$$

训练 loss 就是模型输出的 vector field 与真实 velocity 之间的 MSE：
$$
\mathcal{L}=\mathbb{E}_{t,x_0,x_1,y}\Big[\big\|u(x_t,y,t;\theta)-(x_1-x_0)\big\|_2^2\Big] \tag{1}
$$

这里 $y$ 是条件嵌入、$\theta$ 是可学习参数。还用了两个比较重要的训练技巧：一个是跟 SD3 一样使用 <mark class="hl-trick">**logit-normal timestep sampler**</mark>，<mark class="hl-trick">让训练更多集中在中间 timestep</mark>；另一个是跟 Flux 类似的 <mark class="hl-trick">**dynamic time shifting**</mark>，因为<mark class="hl-trick">不同图像分辨率的 SNR 分布不同，需要根据分辨率调整实际噪声时间，从而让多分辨率训练更稳定</mark>。

第一阶段是 <mark class="hl-trick">**Low-resolution Pre-training**</mark>。这个阶段非常纯粹：<mark class="hl-trick">只做 **$256\times256$ 的 text-to-image generation**</mark>。目标不是追求最终高清效果，而是用较低计算成本先把最基础的东西学出来，包括 cross-modal alignment、基本视觉知识、各种 concepts、styles、compositions。论文明确说，<mark class="hl-trick">这一阶段占了整个 pre-training compute 的一半以上</mark>，因为作者认为<mark class="hl-trick">模型的大部分 foundational visual knowledge，包括 **Chinese text rendering**，其实都可以在低分辨率阶段先学到</mark>。

然后进入 <mark class="hl-trick">**Omni-pre-training**</mark>。这里的"Omni"主要指三个维度。而且 Omni-pre-training 本身<mark class="hl-trick">**是多阶段（multiple stages）推进的**</mark>——论文原文 *"the omni-pre-training phase is conducted in multiple stages"*，并明确说是到<mark class="hl-trick">"最后一个阶段完成（upon completion of the final stage）"后</mark>模型才具备 1K–1.5K 的任意分辨率能力。<mark class="hl-trick">具体分几个阶段、每阶段各承担什么设置，论文没有披露</mark>。

第一个是 <mark class="hl-trick">**Arbitrary-Resolution Training**</mark>：<mark class="hl-trick">不再固定 $256$，而是把原始图像通过 resolution-mapping function 映射到预定义的 training resolution range</mark>，允许不同分辨率和宽高比一起训练。这样可以<mark class="hl-trick">减少强行 downsample 带来的信息损失</mark>，也为最后支持大约 1K–1.5K 分辨率做准备。

第二个是 <mark class="hl-trick">**Joint Text-to-Image and Image-to-Image Training**</mark>。这点很重要：<mark class="hl-trick">Z-Image 不是先把 T2I foundation model 完整训完，再单独从头搞 editing，而是在 omni-pretraining 阶段就已经把 image-to-image task 混进来了</mark>。这里使用前面 2.5 构建的大规模、自然的、弱对齐 image pairs，让模型在大规模 pretrain compute 下<mark class="hl-trick">提前学"两个图像之间的关系"</mark>。论文明确说，这给后续 editing 提供了很好的 initialization，而且他们观察到这种联合预训练**没有明显损害 T2I 性能**。

第三个是 <mark class="hl-trick">**Multi-level and Bilingual Caption Training**</mark>。前面第 3 节学到的 Z-Captioner 终于在这里真正用起来：训练时会混合 bilingual 的 long / medium / short captions、tags、simulated user prompts，同时<mark class="hl-trick">还会以较小概率使用原始 textual metadata，用来增强 world knowledge</mark>。作者强调，不同粒度和不同视角 caption 能提供更广的 mode coverage，为后续阶段打基础。对于 image-to-image 数据，他们还会随机选择两种文本条件：一种是 **target image caption**，对应 reference-guided generation；另一种是 **pairwise difference caption**，对应 image editing。至于这两个条件各自的采样概率，论文没有公开。

所以 4.3 最重要的训练主线可以直接记成：

$$
\boxed{\ 256^2\ \text{T2I Low-res Pretraining} \rightarrow \text{Omni-pretraining}\ }
$$

而 Omni-pretraining 再同时加入：

$$
\boxed{\ \text{Arbitrary Resolution}+\text{T2I/I2I Joint Training}+\text{Multi-level Bilingual Captions}\ }
$$

最终完成 omni-pre-training 后，模型已经能生成 **约 1K–1.5K arbitrary-resolution images**，并且同时接受 text 和 image condition，这时候才成为后续 **Z-Image generation SFT** 和 **Z-Image-Edit continued training** 的共同基础模型。

::: tip 这一节真正值得记住的工业思想
**不要一上来就用最高分辨率、最复杂任务烧算力。先在 $256^2$ 用便宜计算学绝大部分基础知识，再逐渐引入高分辨率、多宽高比、I2I 和复杂 caption，把昂贵计算留给真正需要高分辨率与多任务能力的阶段。**
:::

::: info 原文补充（笔记核对时添加，论文 §4.3 可查）
- **引用出处**：flow matching 引 [44, 48]；logit-normal 采样 *"Following SD3 [18]"*；dynamic time shifting *"as used in Flux [34]"*；caption 重要性引 [4]。**采样器超参（logit-normal 的均值/标准差、time shifting 的分辨率映射指数）论文一个都没给。**
- **"一半以上"精确是 50.9%**：Table 1 里低分辨率预训练 147.5K / 预训练总计 290K。注意论文写的是 *"over half of our total **pre-training** compute"*，**是预训练内部占比，不是全流程占比**（全流程口径是 47.0%）。引用时注意别混。
- **原文有一处笔误**：该段写 *"As shown in **Figure 1**, this phase accounts for over half..."*，但这个数字在 **Table 1** 里，Figure 1 是写实效果 showcase。
- **Omni-pre-training 是多阶段的（见正文）**：阶段数、每阶段的分辨率范围与任务配比、每阶段训练多久——**全部未披露**。这是 §4.3 复现缺口的一部分。
- **任意分辨率的动机不止省算力**：原文列了三条 —— 学 cross-scale visual information、*"mitigates information loss caused by downsampling to a fixed resolution"*、*"improves overall data efficiency"*。
- **I2I 混训是 Z-Image-Edit 低成本的根源**：§4.7 的 edit 继续训练是"从 base model 继续训练"，而 base model 的编辑先验**正是本节种下的**。这解释了 §1.3 支柱③说的"摊薄重预训练预算、不需要独立昂贵阶段"。
- **本节最大的复现性缺口**：预训练**数据量（图像数）、batch size、学习率、优化器、阶段划分、训练时长对应的迭代步数——全部未公开**。相比 §1.2 的 Table 1 只给了算力总量，这里的信息密度低得多。
:::
