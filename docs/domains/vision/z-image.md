# 单流扩散基模 Z-Image (S3-DiT + 全链路后训练)

> **标签**：`Vision` `Diffusion` `DiT` `Flow Matching` `Distillation` `RLHF`
> **更新时间**：2026-09-26
> **参考来源**：[Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer (arXiv:2511.22699v5)](https://arxiv.org/abs/2511.22699) · [arXiv HTML 全文](https://arxiv.org/html/2511.22699v5) · [GitHub: Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) · [HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [ModelScope](https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
> **精读进度**：§1 Introduction ✅ ｜ §2 Data Infrastructure ✅（2.1–2.5 全五小节）｜ §3 Image Captioner ｜ §4 Model Training ｜ §5 Evaluation（笔记随学习逐节增补）

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

第二条也是这一节最有特点的，是 **Efficient Graphical Representation**。对于同一个 input image，他们先生成多个不同编辑版本。然后这些版本之间可以继续两两组合，构造新的 source-target pair。论文的意思是：原始图和 $N$ 个编辑版本之间，不只是有 $N$ 对关系，还可以通过不同 edited versions 之间的组合进一步扩增 pair 数量。这样一来，一组已经生成好的编辑结果可以被反复重组，不需要重新调用 expert model，就能把训练数据规模继续放大。与此同时，这种重组天然会产生 mixed-editing pair，也会产生 inverse pair。作者特别强调 inverse pair 的意义：可以让"真实、未失真的图"作为 target，从而提升数据质量。

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
