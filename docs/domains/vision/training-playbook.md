# 图像基模训练 Playbook / Recipe v1.0

> **标签**：`Vision` `Diffusion` `Flow Matching` `RL` `GRPO` `Distillation` `Data-centric` `Playbook`
> **更新时间**：2026-10-05
> **性质**：<mark class="hl-trick">**方法论字典，不是论文笔记**</mark>。本文是读完 Z-Image / SeFi-Image / Mage-Flow / DeepGen / Qwen-Image-2.0 五篇技术报告后蒸馏出的操作手册，按「数据 → 预训练/SFT → 生成与编辑 → 后训练 → 评估 → 实验系统 → 工业流程」组织。
> **怎么用**：<mark class="hl-key">**当模型出现某种症状时，知道应该打开哪一个抽屉**</mark>。不是用来机械复刻某篇论文的。
> **来源笔记**：[Z-Image](./z-image.md) · [SeFi-Image](./image-rl-posttraining/sefi-image-rl.md) · [Mage-Flow](./mage-flow.md) · [DeepGen 1.0](./deepgen.md) · [Qwen-Image-2.0](./qwen-image-2.md) · [语义先行扩散范式 SFD](./sfd-semantic-first-diffusion.md) · [图像 RL 后训练专辑](./image-rl-posttraining/)

---

::: info 证据等级约定
本文的每条经验都标注了证据强度，**引用时请连同等级一起带走**：

| 等级 | 含义 |
| :--- | :--- |
| <mark class="hl-key">**A**</mark> | 论文有<mark class="hl-key">**比较直接的 ablation 或对照**</mark>支持 |
| <mark class="hl-trick">**B**</mark> | 多个工业报告<mark class="hl-trick">**反复采用**，逻辑和结果支持，但**没有严格证明是最优**</mark> |
| <mark class="hl-trick">**C**</mark> | 工程上值得尝试，但<mark class="hl-trick">**具体阈值和比例必须自己验证**</mark> |

:::

---

## 1. Data Engine：先把模型未来能学到什么设计好

工业图像基模的数据流程不应该理解成“拿十亿张图片过滤一下”。真正的数据引擎要同时解决四个问题：**数据质量、概念覆盖、监督信号质量和最终采样分布**。这四件事里，过滤只解决第一件事，真正影响模型能力结构的往往是后面三件。

一个比较稳的顺序是：

$$
\text{Raw Data}
\rightarrow
\text{Cheap Filter}
\rightarrow
\text{Model-based Filter}
\rightarrow
\text{Dedup}
\rightarrow
\text{Recaption}
\rightarrow
\text{Tag/Profile}
\rightarrow
\text{Reweight/Synthesis}
\rightarrow
\text{Training Mixture}
$$

第一层过滤尽量用便宜规则处理明显垃圾：文件损坏、极低分辨率、极端 aspect ratio、空白图、方向异常等。第二层再运行成本高一些的 aesthetic、watermark、NSFW、blur、OCR、image-text alignment 等模型。<mark class="hl-trick">**不要一上来给所有图片跑最贵的 VLM，这是数据流水线里非常实际的成本问题。**</mark>

### 从 Qwen-Image-2.0 补进 §1 的两条

#### 增量 A：Capability-driven Data Taxonomy（先定义能力，再反推数据域）

<mark class="hl-trick">**很多团队做数据时起点是“我有什么数据”，Qwen 的起点是“我希望模型会什么”。**</mark>两者的差别是<mark class="hl-key">**前者只能在既有数据里做取舍，后者会告诉你还缺哪一类数据、该去补什么**</mark>：

$$
\boxed{
\text{我希望模型会什么}
\;\rightarrow\;
\text{对应的数据域是什么}
}
$$

<mark class="hl-key">**这个层次高于 concept balancing。**</mark>concept balancing 解决的是<mark class="hl-trick">**“已经选定的域之间怎么配比”**</mark>；capability-driven taxonomy 解决的是<mark class="hl-key">**“域本身该怎么划、漏了哪些域”**</mark>——<mark class="hl-trick">**这是前一步，做错了后面配比再优化也补不回来**</mark>。

<mark class="hl-key">**可操作的做法**</mark>：把模型要具备的能力列成 taxonomy，然后<mark class="hl-trick">**对每个能力域标注三件事**</mark>——真实业务里出现频率、现有数据量、当前评测分数。<mark class="hl-key">**三者一比，缺口自然浮现**</mark>。→ [Qwen-Image-2.0 §2.1](./qwen-image-2.md#21-data-collection) <mark class="hl-key">〔B〕</mark>

#### 增量 B：Task-specific Supervision Representation（不要只有一种 caption 模板）

<mark class="hl-trick">**“Recaption” 这个词本身藏了一个假设：所有图都用同一个 VLM 走一遍，出一段自然语言。**</mark>Qwen 明确否掉了这个假设——<mark class="hl-key">**监督表示形式应该由数据承担的 capability 决定**</mark>：

$$
\boxed{
\text{Image Type / Capability}
\;\rightarrow\;
\text{Appropriate Supervision Representation}
}
$$

它把 caption 分成四类：普通视觉内容用自然语言描述；<mark class="hl-key">**文字密集图像要显式建模文字内容 + 版面结构 + 视觉符号 + 语义关系**</mark>；需要世界知识的图像补充背景与上下文；<mark class="hl-trick">**关系图 / 流程图这类图像用 entity–attribute–relation 的结构化表示，因为一长段自然语言会丢结构**</mark>。

<mark class="hl-key">**这条与本节已有的 Z-Image「先 OCR 再生成 caption」不冲突而是互补**</mark>——<mark class="hl-trick">**Z-Image 解决的是「通用 captioner 会漏掉密集文字」，Qwen 解决的是「这类图像根本需要另一套 pipeline」**</mark>。

<mark class="hl-trick">**落地成本要说清楚：为不同类型数据分设 caption pipeline 是有额外工程代价的**</mark>，<mark class="hl-key">**优先对最高 ROI 的两类做特化（通常是 text-rich 与长尾知识型），不要一上来做全套**</mark>。→ [Qwen-Image-2.0 §2.2](./qwen-image-2.md#22-data-annotation) <mark class="hl-key">〔B〕</mark>

#### 增量 C（构造方法）：从精细标注反向造“坏 prompt”

<mark class="hl-key">**这一条同时是数据构造和 prompt 工程两边的工具，值得单独记。**</mark>不要手工收集成千上万对「差 prompt / 好 prompt」，而是<mark class="hl-trick">**从已有的高质量精细标注出发，反向把它降级成真实用户会输入的短 prompt**</mark>：

$$
\boxed{
\text{Detailed Annotation}
\;\rightarrow\;
\text{Controlled Degradation}
\;\rightarrow\;
\text{Realistic User Query}
\;\rightarrow\;
\boxed{\text{Inverse Reasoning Supervision}}
}
$$

<mark class="hl-key">**最巧的地方在于：因为系统知道“降级时到底删掉了什么”，这些降级操作的逆过程天然就构成一条 reasoning 轨迹**</mark>，<mark class="hl-trick">**所以监督信号可以做成 (短 prompt, CoT, 精细标注) 三元组，而不只是「短 → 长」的映射**</mark>。→ [Qwen-Image-2.0 §3.3](./qwen-image-2.md#33-prompt-enhancer-) <mark class="hl-key">〔B〕</mark>

<mark class="hl-key">**并且随机采样降级策略与比例，才能覆盖难度、模糊度、信息密度各不相同的输入**</mark>——<mark class="hl-trick">**固定一种降级方式得到的是一个点，不是一条分布**</mark>。


过滤阈值不要问“aesthetic 应该是 5.5 还是 6.0”，真正应该做的是<mark class="hl-key">**阈值扫描**</mark>。例如分别取 4.5、5.0、5.5、6.0 后观察：

$$
\text{Retained Ratio},
\quad
\text{Concept Coverage},
\quad
\text{Aesthetic Distribution},
\quad
\text{Validation Generation Quality}
$$

怎么变化。<mark class="hl-trick">**一个阈值如果让数据审美均值提高，却把 illustration、document、diagram、low-light photography 等大量合理类别一起删掉，它就不是好阈值。过滤不是分数越严格越好。**</mark>

尤其 OCR 数据不能简单按照“文字面积大就删除”。如果你的模型未来需要 text rendering，那么正确处理方式往往是<mark class="hl-key">**分桶**</mark>：

$$
\text{ordinary image}
\qquad\text{和}\qquad
\text{text-rich image}
$$

普通生成数据可以限制 OCR ratio，但高质量 poster、sign、menu、infographic、logo、UI 类数据应该进入专门的 Text Rendering bucket，而不是被全删掉。<mark class="hl-key">**这个思想比某一个 OCR 阈值重要得多。**</mark> <mark class="hl-trick">〔C〕</mark>

Dedup 建议至少做 intra-dataset 和 cross-dataset 两层。<mark class="hl-trick">**真正保留哪个 duplicate，不要随机选**</mark>，而应该根据 resolution、watermark、aesthetic、compression artifacts、caption quality 等选择 cluster 里最好的代表。Benchmark contamination 也应该单独建索引检查。→ [Mage-Flow §2.1③](./mage-flow.md#_3-cross-sample-deduplication) 使用 SSCD descriptor + FAISS，并在 duplicate cluster 中保留高质量代表，就是一个比较标准的工业做法。 <mark class="hl-key">〔A〕</mark>

Recaption 的目标<mark class="hl-trick">**不是“caption 越长越好”，而是提供不同粒度的监督**</mark>。比较合理的是同时维护短描述、实体/属性、构图描述、详细 photographic description。<mark class="hl-key">**短 caption 对简单 semantic binding 很有价值，长 caption 更适合复杂 composition 和细节。**</mark>如果全部替换成长达数百 token 的 VLM caption，模型很容易学习到一种非常啰嗦而非自然的人类 prompt distribution。→ [Mage-Flow §2.1④](./mage-flow.md#_4-multi-granularity-captioning) 的 Entity / Phrase / Composition / Photographic 多粒度 caption 就体现了这个思想。 <mark class="hl-key">〔B〕</mark>

真正完成过滤和 caption 以后，还必须做 <mark class="hl-trick">**Data Profiling**</mark>。至少要知道你的训练集里面 object、human、scene、style、text、count、spatial relation、product、food、document、rare concept 等的频率。<mark class="hl-key">**不要只知道“我有 1B 数据”，却不知道这 1B 数据实际上是什么。**</mark>

最终训练的数据分布也不要完全服从 Web 自然分布，因为：

$$
p(\text{cat}) \ \gg\ p(\text{rare instrument})
$$

$$
p(\text{single object}) \ \gg\ p(\text{complex relation})
$$

$$
p(\text{little text}) \ \gg\ p(\text{dense readable text})
$$

模型会忠实学习这种不均衡。<mark class="hl-trick">**长尾能力要通过 oversampling、reweighting、targeted synthesis 去补，但不能无限 oversample。**</mark>正确判断方式不是看“rare data 占比”，而是看对应 capability validation 是否还在上涨，以及 common capability 是否开始退化。

<mark class="hl-trick">**注意 §1/§2/§3 各自表内的小排查，与 §5 的全局诊断字典有部分重叠**</mark>——<mark class="hl-key">**这是有意的**：各章表格是「该阶段内部的细排查」，§5 是「跨章症状的入口索引」。用 §5 定位到章，再回该章的表做细分。</mark>

### Data 阶段真正的诊断字典

| 现象 | 第一怀疑项 | 推荐先做的实验 |
| :--- | :--- | :--- |
| 图越来越漂亮但越来越同质 | Late-stage aesthetic filtering 太窄 | 对比不同 aesthetic bucket 的 style entropy / sample grid |
| 颜色过艳、过亮、HDR 感严重 | SFT / high-quality data 本身偏饱和 | 统计 saturation、brightness、contrast 分布并分桶生成 |
| OCR 很差 | text-rich 数据量或 OCR supervision 不足 | 分 word / sentence / multiline 分析 coverage |
| 数量关系差 | count / compositional 数据稀缺 | 单独统计 2/3/5+ objects prompt 与训练样本 |
| 冷门物体不会生成 | long-tail concept coverage 不足 | 按 concept frequency 画能力曲线 |
| 模型有固定摄影风格 | caption/style distribution 或 aesthetic 筛选过窄 | style-conditioned validation |
| 同一类问题反复出现，但每次修的层次都不一样 | 没有做根因归因，坏案例默认被塞回重训 | 建立 bad case 归因路由表，先分 Knowledge / Alignment / Prompt 三类 |
| 训练 loss 好但生成语义差 | caption 本身错误或过度 hallucination | 人工抽查 caption-image consistency |
| 模型分不清「该按版式渲染的文字」与「画面里的文字」 | 用了同一套 caption 模板，text-rich 数据没有被特殊处理 | 单独统计 text-rich 子集，并为其做 text + layout-aware caption |
| 冷门领域 / 长尾概念整体偏弱 | 数据域划分本身没覆盖到，而非配比问题 | 按 capability taxonomy 列出「业务频率 × 数据量 × 当前分数」三列表 |
| 用户 prompt 很短但生成结果缺版式 / 缺细节 | 中间缺一层 prompt→condition 的转换，而不是模型能力不足 | 检查是否有 Prompt Enhancer；用降级构造补 (短 prompt, 精细标注) 训练对 |

数据问题排查时我会坚持一个原则：

$$
\boxed{
\text{先看分布，再看算法。}
}
$$

这是目前这些工业报告最值得带走的经验之一。

---

## 2. Pretrain 与 SFT：一个学世界，一个塑造最终输出分布

Pretrain 的首要目标<mark class="hl-trick">**不是把图片训得特别漂亮，而是建立尽可能广的 visual-language capability coverage**</mark>。因此早期阶段可以接受相对更宽的数据质量区间，用更低的分辨率获得更高的 token / image throughput，学习 object、attribute、scene、relation、composition、style、language-image correspondence。

随着训练进行，再逐渐提高 resolution 和数据质量，类似：

$$
\text{Broad + Noisier + Cheap}
\rightarrow
\text{Cleaner + Higher Resolution}
\rightarrow
\text{High-quality}
\rightarrow
\text{SFT}
$$

<mark class="hl-key">**常见的 256→512→1024 只是这种思想的一种实现，不应该当成硬规定。**</mark>如果模型本身从头就是 native-resolution，也可以不采用固定 bucket，但“早期追 coverage、后期追 quality”的 curriculum 仍然成立。 <mark class="hl-trick">〔B〕</mark>

<mark class="hl-key">**Qwen-Image-2.0 把这条原则推到了更完整的形态：阶段变化时变的不只是分辨率，而是多个维度同时变。**</mark>它把数据流水线做成六阶段，每一阶段同时调整任务混合、数据类型、过滤强度与分布：

$$
\boxed{
\text{Stage Change}
\;\Rightarrow\;
\text{Resolution}
+
\text{Task Mixture}
+
\text{Data Type}
+
\text{Filter Threshold}
+
\text{Distribution}
\quad \text{一起调整}
}
$$

<mark class="hl-key">**对照本节上面的四段式（Broad → Cleaner → High-quality → SFT），Qwen 的增量在于“分辨率升上去时，同步收紧质量门槛并引入新的数据类型”。**</mark>两个可直接复用的具体做法：

- <mark class="hl-trick">**高分辨率不等于大像素**</mark>——尺寸达标但低清放大、JPEG 重压、细节差的图必须单独拦掉：

$$
\boxed{
\text{High Resolution Data} \;\neq\; \text{Large Pixel Count Data}
}
$$

<mark class="hl-key">**真正要的是 high-resolution + high-fidelity 的交集。**</mark>→ [Qwen-Image-2.0 §2.3](./qwen-image-2.md#23-multi-stage-training-data-strategy-) <mark class="hl-key">〔B〕</mark>

- <mark class="hl-trick">**升到最高分辨率时不要丢弃低分辨率，而是多分辨率共存**</mark>，<mark class="hl-key">**避免训练分布只剩最贵的那一档、同时保住多个 scale 的能力**</mark>。

<mark class="hl-key">**另一条很好用的规律，越靠近最终模型，数据越精、LR 越小、训练越短**</mark>（Qwen 是 700K → 250K → 10K steps，LR 1e-4 → 2e-5 → 1e-5）：

$$
\boxed{
\text{越靠近最终模型}
\;\Rightarrow\;
\text{Data 更精、LR 更小、训练更短}
}
$$

<mark class="hl-trick">**它还印证了本节的另一条：SFT 往往不需要“重新发明一套任务和数据类别”**</mark>，<mark class="hl-key">**同样的能力混合 + 更高质量的数据 + 更严格的分布 + 更小的 LR，就已经足够改变最终输出风格**</mark>。

<mark class="hl-trick">**⚠️ 别把它的具体比例当通用最优值**</mark>——<mark class="hl-key">**Qwen 的 9:1 → 7:3 没有任何 ablation 支撑，真正可迁移的是「task mixture 随阶段变化」这个结构性判断**</mark>。 <mark class="hl-key">〔B〕</mark>

什么时候进入下一阶段，也不应该只看 training loss。最好同时观察 capability validation：basic semantic、composition、text、human、aesthetic、高分辨率细节。<mark class="hl-trick">**如果基础语义仍在快速提升，那么过早切换到非常窄的高质量数据，可能浪费 coverage；如果基础能力已经趋于平台，而高分辨率细节和质感明显落后，就应该进入下一阶段。**</mark>

真正重要的是：<mark class="hl-key">**每一次阶段切换，都应该知道数据分布发生了什么变化。**</mark>不要只是目录从 `data_v1` 换成 `data_v2`。至少记录各 bucket 的 sampling probability：

$$
p_{\rm human},
p_{\rm text},
p_{\rm product},
p_{\rm scene},
p_{\rm rare},
\dots
$$

否则模型能力发生变化以后，你无法追溯原因。

SFT 要和 Pretrain 严格区分。<mark class="hl-trick">**SFT 不是“再训练一会”，而是**</mark>

$$
\boxed{
\text{Distribution Shaping}
}
$$

它是在告诉已经具备大量能力的 Base Model：

> inference 时你最应该倾向生成什么。

因此 SFT 数据应该更精、更接近产品目标，同时<mark class="hl-trick">**必须非常警惕 style collapse**</mark>。如果 SFT 数据大量来自相似风格的高审美图，模型可能迅速变得“第一眼很好看”，但所有 prompt 都出现类似 lighting、contrast、skin texture、构图。

所以 SFT 期间不能只看 aesthetic score，还应该同时监控：

$$
\text{Aesthetic}
+
\text{Semantic}
+
\text{Style Diversity}
+
\text{Prompt Diversity}
+
\text{Text}
+
\text{Human}
$$

的变化。

### SFT 常见问题的诊断方式

| 现象 | 更可能是什么问题 | 优先怎么调 |
| :--- | :--- | :--- |
| Base 很丰富，SFT 后所有图一个味 | SFT distribution 太窄 | 扩 style / lighting / composition diversity，<mark class="hl-trick">**而不是先调 LR**</mark> |
| 质感变好但 prompt following 掉 | 高审美数据 semantic supervision 偏弱 | 增加 high-quality + high-alignment subset |
| SFT 很快涨、后来质量反而下降 | 过拟合 / distribution collapse | 缩短 SFT，降低重复采样，提高 diversity |
| 人物越来越好，非人物越来越差 | human bucket 过重 | 检查 capability sampling ratio |
| 文本能力掉 | SFT text-rich data 太少 | 保留专门 text bucket |
| 色彩变得过饱和 | SFT aesthetic bias | 做 color-statistics audit，<mark class="hl-trick">**不要先归因模型架构**</mark> |

所以如果以后你发现模型“GPT 感不够”“质感不对”“饱和度不对”，我会首先让你做一个 <mark class="hl-key">**SFT Distribution Audit**</mark>，<mark class="hl-trick">**而不是上 RL**</mark>。

---

## 3. Generation 与 Editing：不要把它们当成两套完全独立的能力

如果模型已经有一个比较强的 Generation Base，再训练 Editing 时最值得注意的是 <mark class="hl-key">**catastrophic forgetting**</mark>。Editing 训练的输入分布和 T2I 差别很大：

$$
(x_{\rm src},\ instruction,\ x_{\rm tgt})
$$

会让模型逐渐习惯“有 source image”的条件形式。如果长时间只训练这种数据，原来的 open-ended generation prior 可能下降。

因此比较稳妥的 recipe 是<mark class="hl-key">**保留 Generation replay**</mark>。<mark class="hl-trick">**具体比例没有通用真理**</mark>，可以从较高 Generation 比例开始，再随着 Editing 稳定逐渐增加 Edit 权重。→ [Mage-Flow §3.2②](./mage-flow.md#_2-instruction-based-editing) 在不同阶段使用了不同 Gen/Edit 配比，而其 [Table 7 消融](./mage-flow.md#_8-1-generation-数据混进-edit-是否有效)显示，Generation replay 对 full-step 编辑器收益较小，但对 four-step editor 的 broad editing robustness 更明显，因此这个 trick 是 <mark class="hl-trick">**B/A 之间：值得默认保留，但具体比例必须自己调**</mark>。 <mark class="hl-key">〔A/B〕</mark>

Editing 数据必须先做 taxonomy，否则你根本不知道模型“会不会 edit”。至少应该区分 object add/remove/replace、attribute、background、style、pose/viewpoint、spatial、text editing、restoration、low-level adjustment、multi-image 等类别。然后根据真实能力曲线调整采样，<mark class="hl-trick">**而不是按数据集原始大小采样**</mark>。 <mark class="hl-key">〔B〕</mark>

Edit 数据质量最关键的不是 target image 看起来漂不漂亮，而是同时满足：

$$
\boxed{
\text{Instruction Correctness}
+
\text{Unedited Region Preservation}
+
\text{Visual Quality}
}
$$

这是 Editing 和纯 T2I 最大的不同。

例如指令只是“把红车改成蓝车”，输出如果同时换了天空、道路和视角，<mark class="hl-trick">**即使最后蓝车很漂亮，这仍然是坏样本**</mark>。数据 filtering 和 Editing reward 都应该显式考虑 preservation。

以后如果出现：

$$
\text{Edit adherence 很好，但改动太大}
$$

优先查 preservation supervision，<mark class="hl-trick">**而不是给 instruction reward 加更大权重**</mark>。

如果：

$$
\text{什么都不太改}
$$

则可能相反：preservation 约束太强，或真正带明显 target difference 的训练样本不足。

Generation/Edit 的核心不是找到一个神奇 ratio，而是根据两条曲线动态调：

$$
S_{\rm gen}(t),
\qquad
S_{\rm edit}(t).
$$

当 Editing 上升而 Generation 明显下跌时，增加 Generation replay；当 Generation 保持稳定而 Editing 学不动时，再增加 Editing 权重。

---

## 4. Post-training / RL：不要先问用 GRPO 还是 Diffusion-NFT，先问“我要修哪种能力”

工业图像模型的 Post-training 最重要的第一步，是建立 <mark class="hl-key">**Capability Taxonomy**</mark>。

比如：

$$
\mathcal C=
\{
\text{Aesthetic},
\text{Semantic},
\text{Text},
\text{Human},
\text{Composition},
\text{Editing}
\}
$$

这只是一级能力。<mark class="hl-trick">**Text 下面还应该拆 word、phrase、sentence、multiline、dense text；Semantic 可以拆 object、attribute、count、spatial relation、action、interaction。**</mark>

RL prompt pool 必须基于这个 taxonomy 构建，<mark class="hl-trick">**而不是随机拿一堆用户 prompt**</mark>。

每个 capability 要同时有：

$$
\boxed{
\text{Training Prompt Pool}
+
\text{Reward}
+
\text{Held-out Eval}
}
$$

<mark class="hl-key">**三者不能使用完全同一套样本，否则很容易把 reward benchmark 本身训穿。**</mark>

在 reward 设计上，目前至少存在两种都合理的路线。

一种是 DeepGen 风格的 multi-reward：

$$
R=
\sum_k w_k \tilde R_k
$$

但一定先对不同 reward 做 normalization，否则 OCR、CLIP、VLM preference 的 scale 和 variance 完全不同，其中一个 reward 很可能吞掉其他 reward。→ [DeepGen §3.3.1](./deepgen.md#_3-3-1-多奖励解耦归一化-mr-的核心)

另一种是 Mage-Flow 风格：

$$
\boxed{
\text{Prompt Capability}
\rightarrow
\text{Dedicated Evaluator}
}
$$

例如 Text prompt 只走 OCR，Semantic prompt 只走 Semantic VLM。它的优点是 reward 更容易解释，也避免多 reward 互相打架；缺点是一个 prompt 真实质量可能本来就是多维的。→ [Mage-Flow §4.2](./mage-flow.md#_4-2-text-to-image-generation)

<mark class="hl-key">**目前没有理由认为两者有一种“永远正确”。这是需要根据任务选择的分支。**</mark>

Rollout 之后真正应该持续观察的不是 average reward，而是：

$$
\text{Reward Mean},
\quad
\text{Reward Variance},
\quad
\text{Within-group Variance},
\quad
\text{Zero/One Saturation Rate}.
$$

例如 GRPO 一组 8 张图全部 OCR=0，那么：

$$
A_i\approx0
$$

基本没有有效学习信号。

如果一组所有图 reward 都接近 1，也同样没有学习价值。此时不应该先改 optimizer，而应该考虑：

$$
\boxed{
\text{Prompt Difficulty Distribution}
}
$$

是不是不对。

一个非常实用的 RL curriculum 是：

$$
\text{easy}
\rightarrow
\text{medium}
\rightarrow
\text{hard}
$$

但不是简单把 easy 全部删除，而是逐渐提高 hard sample 的占比。→ [Mage-Flow §4.2⑤](./mage-flow.md#_5-两阶段-curriculum-1-1-1-→-2-4-1) 的 Text RL 就体现了这种思路：前期简单 OCR，后期增加完整句子、多行文本和复杂 scene text，同时一直保留 aesthetic / semantic capability，防止模型完全向 OCR 专门化。 <mark class="hl-key">〔B〕</mark>

### 从 Qwen-Image-2.0 补进 §4 的两条

#### 增量 D：Hybrid CFG —— 把 rollout 质量与训练成本分开优化

<mark class="hl-key">**这是图像 RL 里最实用的一个工程 trick，而且极易被误解，所以必须说准。**</mark>扩散模型 rollout 时通常要开 CFG：

$$
\epsilon_{\rm CFG}=\epsilon_u+s\big(\epsilon_c-\epsilon_u\big)
$$

<mark class="hl-trick">**关键事实：这个 trick 不省 rollout 的钱。**</mark>rollout 阶段 conditional 与 unconditional 两个分支<mark class="hl-key">**仍然都要算**</mark>，否则采样质量下降、reward 变噪声；<mark class="hl-trick">**它省的是 policy update 阶段的 backward**</mark>——把 uncond 分支从 policy objective 里摘掉，不必为它保留 activation、构图、反传。

$$
\boxed{
\text{Sampling 用完整 CFG 保证样本质量，Learning 只优化 conditional policy 降低训练成本}
}
$$

<mark class="hl-key">**要注意两点**</mark>：

- <mark class="hl-trick">**这本质是一个近似，不是精确的策略梯度**</mark>——它把 $\epsilon_u$ 当作 fixed guidance / baseline-like 组件，rollout 时用、优化时不让它承担“根据 prompt 改进结果”的责任。<mark class="hl-key">**直觉上成立，因为 uncond 分支本身不含 prompt-specific 信息**</mark>。
- <mark class="hl-trick">**“不更新 uncond 分支”不等于“有一个独立的无条件模型被冻结”**</mark>——<mark class="hl-key">**两者共用同一套参数，只是 uncond 分支的 forward 不贡献梯度；参数更新后它的行为仍会随之改变**</mark>。

<mark class="hl-key">**收益主要在显存而不只是 FLOPs**</mark>：省掉的是一份 activation 加上整条反向图。<mark class="hl-trick">**在 “多 rollout sample × 多 denoising step” 的扩散 RL 里，这个开销本来就极重。**</mark>→ [Qwen-Image-2.0 §4.2](./qwen-image-2.md#42-reinforcement-learning-with-human-feedback-) <mark class="hl-key">〔B〕</mark>

#### 增量 E：Prompt 分布与 reward 权重应当随能力缺口动态变化

<mark class="hl-key">**三篇报告在同一个方向上给出了三个刻度，值得并排看：**</mark>

| 工作 | 做法 | 颗粒度 |
| :--- | :--- | :--- |
| Mage-Flow | 两阶段显式配比，如 $P_{\rm aes}:P_{\rm text}:P_{\rm sem}$ 由 $1{:}1{:}1 \to 2{:}4{:}1$ | <mark class="hl-trick">**手工设计的固定 schedule**</mark> |
| DeepGen | 按 task type 切换 reward 配方（general T2I 完全不挂 OCR） | <mark class="hl-trick">**按任务分派，非训练中动态**</mark> |
| Qwen-Image-2.0 | <mark class="hl-key">**训练过程中动态调整**</mark> $p(\text{prompt})$ 与 $w_{\rm reward}$ | <mark class="hl-trick">**最动态，但也最不透明——只给了一句陈述，没有 schedule、没有数据**</mark> |

$$
\boxed{
\text{Capability-driven RL Curriculum}
}
$$

<mark class="hl-key">**注意一条边界**：论文说“动态调整”，不等于论文证明了某种动态 schedule 最优。</mark><mark class="hl-trick">**Qwen 这篇没有给任何 controlled ablation**</mark>，<mark class="hl-key">**所以可迁移的是「reward 组合要跟着能力缺口走」这个方向，而不是它的实现**</mark>。 <mark class="hl-key">〔B〕</mark>

<mark class="hl-key">**另外把 §4 开头那条 Capability Taxonomy 落到实处**：reward 必须按能力拆，而不是做一个万能 reward。</mark>例如人类主体生成值得单列一个 portrait reward——<mark class="hl-trick">**通用的美学 reward 往往只能说“整体还不错”，无法针对脸部结构、皮肤质感这类具体失效给出信号**</mark>；<mark class="hl-key">**editing 则天然是一对相反的 reward（该改的必须改 / 不该改的不能乱改），也就是一个 trade-off 而不是单一方向**</mark>。 <mark class="hl-key">〔B〕</mark>

<mark class="hl-key">**⚠️ 一个反例值得记住**：Qwen 的三个 T2I reward（美学 / 图文对齐 / 人像）**都没有专门度量字形正确性**</mark>——<mark class="hl-trick">**它的文字能力主要来自预训练数据与 caption 设计，RL 阶段并没有再单独优化文字准确率**</mark>。<mark class="hl-key">**这与 Mage-Flow 把 OCR 权重压到 0.7 形成鲜明对比，说明“是否需要专门的 text reward”取决于该模型文字能力的来源阶段，不能默认照抄。**</mark> <mark class="hl-key">〔B〕</mark>

RL 最大的风险是：

$$
R\uparrow,
\qquad
Q_{\rm human}\downarrow.
$$

这就是 reward hacking / capability drift。

因此 DeepGen 采用 auxiliary SFT loss、KL 等约束的思路非常值得保留：

$$
L=
L_{\rm RL}
+
\lambda_{\rm SFT}L_{\rm SFT}
+
\lambda_{\rm KL}L_{\rm KL}.
$$

<mark class="hl-trick">**并不是每套算法都必须三个全加，而是要知道它们分别解决什么**</mark> → [DeepGen §3.3.2 / §3.3.3](./deepgen.md#_3-3-2-grpo-目标与-velocity-space-kl)

| 现象 | 首先考虑的手段 |
| :--- | :--- |
| RL capability 涨，但整体画质掉 | SFT replay / auxiliary SFT |
| policy 与 Base 快速偏离 | KL |
| 多 reward 一项独大 | reward-wise normalization |
| OCR 涨、aesthetic 掉 | 调 capability mixture，<mark class="hl-trick">**不一定调 RL 算法**</mark> |
| reward 很高但人工很差 | reward model / prompt distribution 出问题 |
| reward 长期几乎不动 | rollout diversity、reward variance、prompt difficulty |
| Edit RL 后 Generation 掉 | Generation replay |
| 某一个 capability 狂涨其他都掉 | curriculum 权重失衡 |

这里最重要的一句话是：

$$
\boxed{
\text{RL 失败时，第一检查项往往不是 RL 公式。}
}
$$

<mark class="hl-key">**检查顺序：Prompt Pool → Reward → Rollout Distribution → Capability Ratio → Regularization，最后才检查优化算法。**</mark>

---

## 5. Evaluation：没有自己的评估系统，就不可能真正做 Post-training

训练 Image Foundation Model 最危险的一件事，<mark class="hl-trick">**就是只看论文 benchmark 或单一 reward**</mark>。

真正工业训练应该维护一个固定的 <mark class="hl-key">**Capability Dashboard**</mark>：

$$
\mathbf S=
[
S_{\rm aesthetic},
S_{\rm semantic},
S_{\rm text},
S_{\rm human},
S_{\rm composition},
S_{\rm edit},
S_{\rm preservation}
].
$$

每次 checkpoint 都跑同一套固定 evaluation，<mark class="hl-trick">**同时保留随机 seed 和 sampler setting，否则不同 checkpoint 的图不能直接比较**</mark>。

此外要再维护一套 challenge set，专门包含历史 failure case。例如模型以前经常失败的：

$$
5\text{ fingers},
\quad
3\text{ red cups},
\quad
A\text{ left of }B,
\quad
\text{multiline poster},
\quad
\text{small text},
\quad
\text{rare material}
$$

一旦修过一个问题，就把它加入 regression set。

这样后面任何训练都能回答：

> 这轮训练到底修了什么，又破坏了什么？

这是工业 post-training 和单篇论文实验非常不一样的地方。

### 全局诊断字典

<mark class="hl-key">**用法：先在这里按症状定位到章，再回该章的细表做具体排查。**</mark>（§1 / §2 / §3 各自表内是更细的阶段内排查，与本表有意重叠）

| 模型症状 | 最可能原因 | 第一检查项 | 第一轮实验 |
| :--- | :--- | :--- | :--- |
| 过饱和、过亮 | SFT / aesthetic preference 偏 | SFT color distribution | 降低对应 bucket 权重做小规模 SFT |
| 图都很好看但一个味 | quality filter 过窄 | style / composition entropy | 混入 diverse high-quality data |
| Prompt following 差 | caption / semantic supervision 不够 | caption consistency | high alignment subset SFT |
| 数量错误 | count data 不足 | count prompt frequency | targeted count SFT/RL |
| 空间关系错误 | compositional data 不足 | relation distribution | relation-specific data |
| OCR 完全不会 | Pretrain/SFT text coverage 不够 | text-rich bucket | **先补数据，再考虑 RL** |
| OCR 基础会但复杂文本差 | RL curriculum 太简单 | reward by difficulty | 提高 sentence/multiline 比例 |
| OCR 涨但画面像海报模板 | Text RL 过度 | aesthetic/general eval | 减 Text ratio + general replay |
| 人脸好但手差 | supervision / evaluation 不平衡 | hand-specific subset | human hard-case SFT/reward |
| Edit 改太多 | preservation 不足 | unchanged-region score | 提高 preservation supervision |
| Edit 几乎不改 | preservation 太强 / edit signal 弱 | instruction adherence | 增强有效 edit pairs |
| Edit 后 T2I 下降 | catastrophic forgetting | Gen benchmark curve | 增 Generation replay |
| RL reward 涨但人工质量下降 | reward hacking | reward-human correlation | **重训/换 reward，不要继续跑** |
| Diversity 下降 | SFT/RL distribution collapse | inter-sample diversity | 加 broad replay / 降训练强度 |
| 1K 好、2K/4K 差 | high-res data / tokenizer / native training 不足 | resolution-wise eval | 单独分析 high-res data & VAE |
| 长 prompt 后半段被忽略 | caption/prompt distribution 太短 | prompt length distribution | 增长文本 compositional supervision |

这张表以后可以不断扩，但主逻辑不会变化：

$$
\boxed{
\text{症状}
\rightarrow
\text{定位能力}
\rightarrow
\text{定位数据/Reward}
\rightarrow
\text{最小修改实验}
}
$$

而不是：

$$
\text{症状}
\rightarrow
\text{重新设计模型}.
$$

---

## 6. 实验与训练系统：每次只回答一个问题

真正做大模型时，<mark class="hl-trick">**最容易浪费卡的不是模型不好，而是实验设计不好**</mark>。

每一轮实验开始之前必须能写出一句话：

> **这轮实验到底在验证什么？**

例如：

> “将 Text Rendering sampling 从 10% 提高到 20%，是否能提升 multiline OCR，同时保持 Aesthetic 不下降超过某个容忍范围？”

这就是一个有效实验。

而：

> “换数据、换 LR、换 reward、顺便多训练 3k steps 看看”

基本没有实验价值。

工业训练最好先在 <mark class="hl-key">**proxy setting**</mark> 上完成方向判断。例如少量数据、低分辨率、短 schedule、较小 model 或冻结部分参数。<mark class="hl-trick">**Proxy 不一定能准确预测最终 gain，但特别适合提前杀掉明显无效的方向。**</mark> <mark class="hl-key">〔B〕</mark>

真正大规模训练前，应该至少固定这些东西：

$$
\text{Dataset Version}
\qquad
\text{Sampling Config}
\qquad
\text{Model Checkpoint}
$$

$$
\text{Optimizer/LR}
\qquad
\text{Resolution Distribution}
\qquad
\text{Evaluation Version}
\qquad
\text{Random Seed}
$$

否则一个月以后，你会发现没人知道某个最好 checkpoint 为什么好。

系统效率也不能完全忽略。→ [Mage-Flow §6.6](./mage-flow.md#_6-6-mfu-13-88-→-29-28-与-2-48×-step-time-speedup) 的一个很好的提醒是：**模型训练速度不是单靠 DiT FLOPs 决定的**。Tokenizer、text encoder、attention packing、kernel launch、memory traffic 都可能是瓶颈。其系统消融里，完整轻量 tokenizer + kernel fusion 把 MFU 从 **13.88% 提到 29.28%**、step time 从 **1.9285s 降到 0.7775s**。 <mark class="hl-key">〔A〕</mark>

但这类 optimization <mark class="hl-trick">**应该有优先级**</mark>。如果 GPU profile 表明：

$$
70\%\text{时间在 DiT}
$$

就优化 DiT；如果 VAE preparation 占比很大，再处理 VAE。<mark class="hl-trick">**不要因为论文做了 fused kernel，就先花两周写 CUDA。**</mark>

---

## 7. 真正开始一个工业项目时，我会怎么跑

假设有一天你进入团队，老板给你一个 Base checkpoint、一套训练代码和几百/上千张卡，告诉你“把质量再做上去”，<mark class="hl-trick">**我不会第一天就开始训**</mark>。

首先应该<mark class="hl-key">**冻结一个 Baseline，跑完整 capability evaluation，并把生成结果人工看一遍**</mark>。然后把问题写成：

$$
\text{Strength}
/
\text{Weakness}
/
\text{Regression Risk}.
$$

例如：

> Semantic 已经不错；Aesthetic 中等；Text 明显弱；Editing adherence 好但 preservation 差。

然后去反查对应训练分布。

<mark class="hl-key">**如果 Text 占数据只有 1%，那现在不应该讨论 GRPO。**</mark>

<mark class="hl-key">**如果 Editing preservation 差，而 Edit triples 本身就大量改变背景，那也不要先换 reward model。**</mark>

把数据和 supervision 明显的问题先解决，再做 SFT。<mark class="hl-trick">**只有当模型已经具备基本能力，但“能力存在却很难进一步推高”，这时候 RL 的性价比才真正开始高。**</mark>

所以一个工业团队比较健康的迭代闭环应该是：

$$
\boxed{
\text{Evaluate}
\rightarrow
\text{Diagnose}
\rightarrow
\text{Audit Data}
\rightarrow
\text{Minimal Intervention}
\rightarrow
\text{Train}
\rightarrow
\text{Regression Test}
}
$$

而不是每次：

$$
\text{Train More}.
$$

### 从 Qwen-Image-2.0 补进 §7 的增量 F：Bad Case 先归因，再选最小干预

<mark class="hl-key">**上面这个闭环里的 “Diagnose → Minimal Intervention” 一步，Qwen-Image-2.0 给出了一个可以直接照抄的路由表。这是它对工业流程最有增量的贡献。**</mark>它的关键判断是：<mark class="hl-trick">**坏案例绝不能默认“塞回去重训”，更不能默认“都上 RL”——先做错误归因。**</mark>

$$
\boxed{
\text{Bad Case}
\rightarrow
\text{Root Cause}
\rightarrow
\boxed{\text{Minimal Intervention}}
}
$$

| 根因 | 优先动作 | 对应轨道 |
| :--- | :--- | :--- |
| <mark class="hl-trick">模型从没见过这类概念 / 知识</mark> | <mark class="hl-trick">补 Pretrain / Continual-pretrain 数据</mark> | Pre-training track |
| <mark class="hl-trick">模型会，但输出偏好或执行方式不对</mark> | <mark class="hl-key">RL / Reward 调整</mark> | RL track |
| <mark class="hl-trick">模型有能力，但用户 prompt 表达不足</mark> | <mark class="hl-key">**Prompt Enhancer（不必重训模型）**</mark> | PE track |

<mark class="hl-key">**其中最反直觉、也最省钱的是第三行**</mark>——<mark class="hl-trick">**如果失败只是 specification 不充分，那么模型本身完全不用重训，只要把输入经一层 Prompt Enhancer 转成更适合生成模型消费的形式即可**</mark>：

$$
\boxed{
\text{生成失败} \;\not\Rightarrow\; \text{一定要改 Generator}
}
$$

<mark class="hl-key">**最重要的一条规则，也是这条路由表背后的思想**</mark>：

$$
\boxed{
\text{不要拿一种训练方法解决所有问题。}
}
$$

<mark class="hl-trick">**很多团队最容易犯的错就是：看见某个能力不行就继续加 RL——但有些问题根本不是 RL 问题。**</mark><mark class="hl-key">**尤其是“知识缺失”这类问题，再复杂的 reward 也很难把根本没有见过的东西 RL 出来，应该先补数据。**</mark>

<mark class="hl-trick">**Qwen 的实现里还有两点工程设计值得借鉴**</mark>：

- <mark class="hl-key">**Pre-training track 用向量检索做两件事：先诊断“是不是某类数据太少”，再检索并泛化出更多 prompt 与 instruction-image pair（含 base image）**</mark>，<mark class="hl-trick">**最终形成 curated dataset 补缺口**</mark>。
- <mark class="hl-key">**它把人工干预压缩到几乎只有一个卡点：数据进模型前的人工 review & filtering**</mark>，<mark class="hl-trick">**其余（评测、归因、检索、prompt 改写、启动训练）全部自动**</mark>。→ [Qwen-Image-2.0 §2.4](./qwen-image-2.md#24-closed-loop-data-flywheel-system-) <mark class="hl-key">〔B〕</mark>

::: warning 这套路由表有两个论文自己没解决的缺口
<mark class="hl-trick">**第一，归因本身可能出错，而且出错代价最高**</mark>——<mark class="hl-key">**把“数据缺失”误判成“对齐问题”，就会在错误的轨道上烧掉整轮预算**</mark>。论文完全没讨论归因的可靠性或兜底策略。

<mark class="hl-trick">**第二，论文只定义了三条轨道，没有“归因不确定”这一种情况**</mark>，而工业上这恰恰是常态。<mark class="hl-key">**我的补充做法是：归因不确定时先做检索与相似 case 分析，用数据决定轨道，而不是先选一条路。**</mark>
:::

**这套方法本身可能比记住某一个 GRPO 公式更重要。**

---

## 8. 各技术报告真正独有的 Recipe

前面 1–7 章是**跨论文的共性 Recipe**，到这里结束。<mark class="hl-key">**下面只看每篇工作额外往工具箱里新增了什么工具。**</mark>

| 工作 | 最值得保留的独有 Recipe | 它主要解决什么 | 效果 / 证据怎么看 |
| :--- | :--- | :--- | :--- |
| **[Z-Image](./z-image.md)** | 完整 Data→Pretrain→SFT→Distillation→RLHF 工业链；Generation/Edit 联合能力建设；分阶段组织训练数据 | 如何把大量异质数据组织成连续训练 curriculum | <mark class="hl-trick">〔B〕</mark> 更适合作为“完整工业流水线模板”；<mark class="hl-trick">**具体某些 ratio 不应当作通用最优值**</mark> |
| **[SeFi-Image](./image-rl-posttraining/sefi-image-rl.md)** | Semantic-first、dual VAE、SFD，再接 SFT、DiffusionNFT、DMD2 | latent 表示、语义能力以及少步生成之间的协同 | <mark class="hl-trick">〔B〕</mark> 最大启发是：<mark class="hl-key">**Tokenizer / latent 设计本身会决定后面模型好不好学**</mark>，不能永远只从数据/RL 找问题 |
| **[DeepGen 1.0](./deepgen.md)** | Alignment Pretrain + Joint SFT；MR-GRPO；multi-reward；auxiliary SFT + KL；reward-wise normalization | 多能力联合 post-training，同时抑制 RL drift | <mark class="hl-key">〔A〕</mark> 对做后训练尤其重要：<mark class="hl-key">**RL 中保留 supervised signal、reward normalization、防漂机制确实值得优先考虑**</mark>；<mark class="hl-trick">且它给出了唯一可直接借的失效时钟表（300/600/1000 steps）</mark> |
| **[Mage-Flow](./mage-flow.md)** | Mage-VAE；Native Resolution Packing；Edit 中 Generation replay；capability-routed Diffusion-NFT；two-stage capability curriculum；系统级 kernel optimization | 高分辨率效率、Edit forgetting、能力定向 alignment | <mark class="hl-key">〔A〕</mark> Mage-VAE/系统优化有很强实证；Generation replay 有一定 ablation 支持；<mark class="hl-trick">**2:4:1 等具体 RL ratio 没有证明是最优，不应机械照搬**</mark> |
| **[Qwen-Image-2.0](./qwen-image-2.md)** | Capability-driven data taxonomy；四类 task-specific caption；六阶段数据 curriculum；<mark class="hl-trick">**错误归因驱动的 Data Flywheel（三轨路由）**</mark>；逆向退化构造 PE 数据 + PE 侧 GRPO；<mark class="hl-key">**Hybrid CFG（rollout 全开、优化只走 conditional）**</mark>；五维分任务 reward + scale calibration | 统一生编基模的数据系统设计；prompt 重写的可训练化；扩散 RL 的成本控制 | <mark class="hl-key">〔B〕</mark> 系统设计 unusually 完整，是四篇里<mark class="hl-key">**最接近“可直接照搬的流程”**</mark>的一篇；但<mark class="hl-trick">**全篇零 ablation，所有阈值、比例、reward 权重、数据量均未公开，不能当可复现 recipe**</mark>。<mark class="hl-trick">其中「多 reward 先统一 scale」在 DeepGen 侧是 〔A〕，在本篇只是 〔B〕</mark> |

---

## 9. 工具箱

因此，这几篇读完以后不会得到一个“最佳训练配方”，而是得到一个 <mark class="hl-key">**工具箱**</mark>：

$$
\boxed{
\begin{array}{ll}
\text{数据有问题} & \rightarrow \text{Data Engine / Curriculum}\\
\text{最终视觉分布不对} & \rightarrow \text{SFT}\\
\text{特定能力到瓶颈} & \rightarrow \text{Capability-specific RL}\\
\text{RL 开始漂} & \rightarrow \text{SFT/KL/Replay}\\
\text{Edit 忘记 Generation} & \rightarrow \text{Generation Replay}\\
\text{多 Reward 打架} & \rightarrow \text{Normalization / Routing}\\
\text{高分辨率太贵} & \rightarrow \text{Tokenizer / Packing / System}\\
\text{latent 本身不好学} & \rightarrow \text{重新审视 VAE / latent design}
\end{array}
}
$$

<mark class="hl-key">**目标不是机械复刻某篇论文，而是当模型出现某种症状时，知道应该打开哪一个抽屉。**</mark>

---

## 10. 怎么用这份 Playbook 继续扩展

后续再看新的技术报告（比如 LLaDA-Image 或更新的报告）时，<mark class="hl-trick">**不需要重新写一套笔记，只需要问两个问题**：</mark>

1. <mark class="hl-key">**它有没有修改这套共性 Recipe？**</mark>
2. <mark class="hl-key">**如果没有，它往这个工具箱里新增了什么工具？**</mark>

<mark class="hl-trick">**这样以后读十篇、二十篇论文，知识不会继续散掉。**</mark>

<mark class="hl-key">**同一套提问方式也可以用来审视自己的工作**</mark>：如果你的项目里出现了一个新问题，先问它落在 §9 工具箱的哪一格；如果哪一格都套不上，那可能就是一个值得写进 Playbook 的新工具。

::: warning 这份 Playbook 的已知局限
- <mark class="hl-trick">**所有比例、阈值都是别人的经验值，不是推导结果**</mark>。§8 里已逐条标注证据强度，<mark class="hl-key">**引用时请连同等级一起带走**</mark>。
- <mark class="hl-trick">**缺 Video / 3D / Action 域**</mark>。本文只覆盖 2D 图像 T2I + Editing，<mark class="hl-key">**视频生成的 curriculum、reward 和防漂机制都不同**</mark>（参见 `docs/domains/video/`）。
- <mark class="hl-trick">**缺预训练规模化细节**</mark>。并行策略、FSDP/ZeRO 配置、序列打包与 checkpoint 策略只在 §6 提了一句，<mark class="hl-key">**系统性内容应该看 `docs/infra/`**</mark>。
- <mark class="hl-trick">**没有讨论评测的可信度问题**</mark>。§5 假设 benchmark 本身可信，<mark class="hl-key">**但 Mage-Flow 专门做了 held-out benchmark index 做训练去污染**</mark>（[§2.1③](./mage-flow.md)），<mark class="hl-trick">**说明 benchmark 污染是真实存在的**</mark>。
:::

---

## 来源笔记

| 笔记 | 覆盖内容 |
| :--- | :--- |
| [Z-Image](./z-image.md) | §2 数据四模块 / §4.1 S3-DiT / §4.5 蒸馏 / §4.6 RLHF |
| [SeFi-Image](./image-rl-posttraining/sefi-image-rl.md) | SFD 语义先行、dual VAE、DiffusionNFT |
| [Mage-Flow](./mage-flow.md) | §2 Data / §3 训练 / §4 Diffusion-NFT / §6 系统效率 |
| [DeepGen 1.0](./deepgen.md) | §3.3 MR-GRPO / §4 Data / 三条失效时钟 |
| [语义先行扩散范式 SFD](./sfd-semantic-first-diffusion.md) | SeFi-Image 的前置篇 |
| [图像基模 RL 后训练专辑 (2026)](./image-rl-posttraining/) | 9 篇 RL/DPO/NFT 单篇 + [算法 × 奖励 × 基模对比](./image-rl-posttraining/rl-comparison-2026.md) |
| [AIGC 数据集专题](./datasets/) | 预训练/持续训练/SFT/评测集全生命周期 |