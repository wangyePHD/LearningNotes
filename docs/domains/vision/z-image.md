# 单流扩散基模 Z-Image (S3-DiT + 全链路后训练)

> **标签**：`Vision` `Diffusion` `DiT` `Flow Matching` `Distillation` `RLHF`
> **更新时间**：2026-09-26
> **参考来源**：[Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer (arXiv:2511.22699v5)](https://arxiv.org/abs/2511.22699) · [arXiv HTML 全文](https://arxiv.org/html/2511.22699v5) · [GitHub: Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image) · [HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [ModelScope](https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)
> **精读进度**：§1 Introduction ✅ ｜ §2.1 Data Profiling Engine ✅ ｜ §2.2–§2.5 其余三模块 ｜ §3 Image Captioner ｜ §4 Model Training ｜ §5 Evaluation（笔记随学习逐节增补）

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
