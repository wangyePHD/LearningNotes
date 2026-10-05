# 千问 Qwen-Image-2.0 技术报告精读（统一生成 + 指令编辑 + Flywheel）

> **标签**：`Vision` `Unified Model` `MMDiT` `Flow Matching` `RL` `GRPO` `Data-centric` `Data Flywheel`
> **更新时间**：2026-10-05
> **原文**：本地 `Papers/Qwen-image-2.0.pdf`（30 页，Qwen Team）
> **精读重点**：§2 Data ✅ → §3.3 Prompt Enhancer ✅ → §4.1 Training Recipe ✅ → §4.2 RLHF / GRPO ✅
> **精读进度**：§2 全五小节 ✅ ｜ §3.3 ✅ ｜ §4.1 ✅ ｜ §4.2 ✅ ｜ §3.1–3.2 ⬜ ｜ §4.3 ⬜ ｜ §5 ⬜
> **已有相关笔记**：[Qwen-Image-2.0 RLHF 专辑](./image-rl-posttraining/qwen-image-2-rl.md)（RL 专题，spec 格式）｜ [图像基模训练 Playbook](./training-playbook.md)（跨论文方法论字典）

---

<!--
本篇结构严格镜像论文章节序，不自行重排：
  §2.1 Data Collection              → §2.1
  §2.2 Data Annotation              → §2.2
  §2.3 Multi-Stage Training Data Strategy → §2.3
  §2.4 Closed-loop Data Flywheel System  → §2.4
  §3   Architecture (3.1/3.2/3.3)  → §3
  §4   Training (4.1/4.2/4.3)      → §4
  §5   Benchmark and Qualitative Evaluation → §5
-->

## 1. Introduction

## 2. Data ★

### 2.1 Data Collection

#### ① 三个数据建设原则

<mark class="hl-trick">**Qwen-Image-2.0 的数据设计从一开始就不是「单纯做 T2I，然后后面再外挂一个 Editing 模型」，而是为了**</mark>

$$
\boxed{
\text{Unified T2I Generation} + \text{Instruction-based Editing}
}
$$

<mark class="hl-trick">**来建设统一的数据基础设施。**</mark>论文自己总结数据建设有三个原则：

$$
\boxed{
\text{Broad Domain Coverage}
+
\text{Strong Instruction Quality}
+
\text{Reliable Source-Target Consistency}
}
$$

<mark class="hl-key">**覆盖要广、指令要好、编辑前后的 source-target 关系必须可靠。**</mark>这三个词其实分别对应三个很具体的问题：

| 原则 | 对应的问题 |
| :--- | :--- |
| <mark class="hl-trick">**Broad coverage**</mark> | 模型见没见过足够多的世界 |
| <mark class="hl-trick">**Instruction quality**</mark> | 文本监督是不是足够准确、足够细 |
| <mark class="hl-trick">**Source-target consistency**</mark> | Editing 里到底是不是「按要求改了该改的地方，同时没乱改」 |

<mark class="hl-key">**这三点已经和 Playbook §1 的工业 Recipe 高度一致**</mark> → [Playbook §1](./training-playbook.md)：<mark class="hl-trick">过滤解决「质量」，但真正决定能力结构的是「覆盖」和「监督信号质量」</mark>。

#### ② T2I 数据：不是只有「自然照片」

论文给出 T2I 的四大类：

$$
\boxed{
\text{Realistic Photography}
+
\text{Graphic Design}
+
\text{Artistic Content}
+
\text{Synthetic Imagery}
}
$$

Realistic 部分包括 portraits、landscapes、objects 等常见自然视觉场景，同时<mark class="hl-trick">**刻意保留 long-tail concepts 和 diverse scene compositions**</mark>。

<mark class="hl-key">**这里很重要的一点：它没有把「真实摄影图」当成唯一主干。**</mark>论文还专门加入：

$$
\text{slides}
\qquad
\text{posters}
\qquad
\text{rendered assets}
$$

<mark class="hl-trick">**这些是对 layout、typography、graphic design、composition 更敏感的数据。**</mark>论文明确说这些数据是为了提升：

$$
\text{aesthetic controllability}
+
\text{composition}
+
\text{visual intent}
$$

<mark class="hl-key">**这解释了 Qwen-Image-2.0 为什么特别强调 poster、slide、infographic 和复杂文字生成——能力不是后面 RL 硬推出来的，数据源阶段就已经把这类能力作为独立分布去建设。**</mark>

$$
\boxed{
\text{想让模型会某类复杂任务，首先要在 Pretrain 数据分布里让它「见过这种世界」。}
}
$$

<mark class="hl-trick">**而不是等 SFT / RL 再补。**</mark>→ 这条可以直接写进 [Playbook §1](./training-playbook.md)。

#### ③ Figure 5：正文没说全的真实分类体系

<mark class="hl-key">**Fig. 5 的向日葵图比正文 §2.1 的文字描述细得多**</mark>——这是论文里信息量最高、但完全没被文字提及的一张图。

![Qwen-Image-2.0 Fig.5：T2I / TI2I 数据分布向日葵图。中心二分 T2I 与 TI2I。**T2I 侧内环三类**：Realistic Photography、Design、Synthetic；外环细类 —— Realistic Photography → Landscape / Portrait / Object / Fauna / Flora / Others；Design → Art / Anime / Poster / Slide / UI；Synthetic → Others / Chart / Text / Color。**TI2I 侧内环两类**：Single Image Edit、Multi Image Edit；外环细类 —— Single Image Edit → Editing Context & Interpretation / Image Restoration & Enhancement / Spatiotemporal Reasoning / Style & Palette Transfer / Compositional Adjustment / Typography Editing & Symbolic Editing / Identity-Consistent Editing / Global Color Transformation / Object Manipulation；Multi Image Edit → Reasoning-based Editing / Reference-based Editing / Composition-based Editing。周围是各细类对应的示例图（藏猫、长椅特写、水花、动画场景、抽象油画、花环文字、静物色调、船、标签牌建筑、蒙娜丽莎、卡通吉祥物、绘本封面、酒店房间、婴儿合照、车、绿地招牌、中英双语贺卡）。](/qwen2-fig5-data-distribution.png)

**完整分类（按 Fig. 5 读出）：**

| 大类 | 子类（Fig. 5 外环原文标签） |
| :--- | :--- |
| <mark class="hl-trick">**Realistic Photography**</mark> | Landscape、Portrait、Object、Fauna、Flora、Others |
| <mark class="hl-trick">**Design**</mark> | Art、Anime、Poster、Slide、UI |
| <mark class="hl-trick">**Synthetic**</mark> | Others、Chart、Text、Color |
| <mark class="hl-key">**Single Image Edit**</mark> | Editing Context & Interpretation、Image Restoration & Enhancement、<mark class="hl-trick">**Spatiotemporal Reasoning**</mark>、Style & Palette Transfer、Compositional Adjustment、<mark class="hl-trick">**Typography Editing & Symbolic Editing**</mark>、<mark class="hl-trick">**Identity-Consistent Editing**</mark>、Global Color Transformation、Object Manipulation |
| <mark class="hl-key">**Multi Image Edit**</mark> | <mark class="hl-trick">**Reasoning-based Editing**</mark>、<mark class="hl-trick">**Reference-based Editing**</mark>、<mark class="hl-trick">**Composition-based Editing**</mark> |

::: warning Fig. 5 与正文 §2.1 的分类对不上——论文没有交代
<mark class="hl-trick">**正文与图存在三处不一致，论文全文没有解释：**</mark>

| # | 正文 §2.1 | Fig. 5 | 问题 |
| :-: | :--- | :--- | :--- |
| 1 | T2I 分 <mark class="hl-key">**4 类**</mark>（realistic photography / graphic design / <mark class="hl-trick">**artistic content**</mark> / synthetic imagery） | T2I 只有 <mark class="hl-trick">**3 类**</mark>内环（Realistic Photography / Design / <mark class="hl-trick">**Synthetic**</mark>） | <mark class="hl-key">**「artistic content」在图里没有对应节点**</mark>，最接近的是 Design 下的 Art / Anime |
| 2 | Single-image 列 <mark class="hl-key">**6 项**</mark>（attribute modification / background replacement / style transfer / text editing / restoration / structure-aware manipulation） | <mark class="hl-trick">**9 类**</mark> | <mark class="hl-key">**图比正文多出 Spatiotemporal Reasoning、Identity-Consistent Editing、Global Color Transformation 等**</mark> |
| 3 | <mark class="hl-key">**「background replacement」在 Fig. 5 里找不到对应细类**</mark> | — | 正文点名的能力，图里没有 |

<mark class="hl-key">**这意味着：正文 §2.1 的 6 项 single-image 清单不能当作完整 taxonomy 使用。**</mark><mark class="hl-trick">若要引用 Qwen-Image-2.0 的编辑能力分类，应该用 Fig. 5 的 9 类，而不是正文的 6 项。</mark>

<mark class="hl-trick">**图里三个值得单独注意的细类**</mark>：

- <mark class="hl-key">**Spatiotemporal Reasoning**</mark> —— <mark class="hl-trick">**一个纯图像模型的编辑 taxonomy 里出现「时空推理」类目**</mark>，说明编辑指令里包含运动/时序类描述（可与 [Playbook §3](./training-playbook.md) 的 multi-image / video-editing 边界对照）
- <mark class="hl-key">**Identity-Consistent Editing**</mark> —— <mark class="hl-trick">**与 §3.2 的 3D RoPE 身份区分设计直接呼应**</mark>，说明这个能力是数据侧显式建设的，不是模型侧偶然涌现的
- <mark class="hl-key">**Reasoning-based Editing**</mark> —— <mark class="hl-trick">**被归在 Multi Image Edit 下而非 Single Image Edit 下**</mark>，即「需要推理的编辑」在 Qwen 的划分里等价于「需要多图参考的编辑」

<mark class="hl-trick">**Fig. 5 没有给任何数字。**</mark>各扇区的**角度看起来编码了占比**，但<mark class="hl-key">**论文未标注任何百分比或绝对数量**</mark>。<mark class="hl-trick">因此「T2I 与 TI2I 谁占多数」「Realistic Photography 占多少」都无法从这张图定量读出</mark>——<mark class="hl-key">**唯一可查的比例在 §4.1 Table 2，只有训练阶段的 T2I:TI2I = 9:1 / 7:3，不是语料构成比例**</mark>。
:::

#### ④ TI2I Editing 数据：single-image + multi-image 两条线

<mark class="hl-trick">**Editing 数据不是只有最常见的**</mark>

$$
(x_{\rm source},\ instruction,\ x_{\rm target})
$$

<mark class="hl-key">**单图编辑。Qwen-Image-2.0 明确把 Editing 分成**</mark>：

$$
\boxed{
\text{Single-image Editing} + \text{Multi-image Editing}
}
$$

<mark class="hl-trick">**Single-image 包括**</mark>：Attribute Modification、Background Replacement、Style Transfer、Text Editing、Restoration、Structure-aware Manipulation。

<mark class="hl-trick">**但 Multi-image 更值得注意**</mark>，包括：Reference-based Generation / Editing、Subject Consistency、Cross-image Style Transfer、Compositional Merging。

<mark class="hl-key">**这已经不是「编辑一张图」这么简单，而是在训练：**</mark>

$$
\boxed{
\text{从多张 reference 中抽取不同条件，再组合成新的 output}
}
$$

例如可以想象：

$$
I_1=\text{人物 reference}
\qquad
I_2=\text{服装 reference}
\qquad
I_3=\text{style reference}
$$

然后 instruction 要求：

$$
(I_1,I_2,I_3,\text{text}) \rightarrow I_{\rm target}
$$

::: warning 这里论文给得很少
<mark class="hl-trick">**论文没有公开 multi-image 子集的数据合成 pipeline**</mark>，上面这个三图组合的例子只是帮助理解。<mark class="hl-key">论文明确说的只有 multi-image subset 训练的是 reference、subject consistency、style transfer、compositional merging 这四类能力</mark>，此外：

- <mark class="hl-trick">**multi-image 数据的规模、来源、过滤方式全部未给**</mark>
- <mark class="hl-trick">**「unified model naturally supports interleaved multi-image inputs」（§3.2）在 §2.1 没有对应的数据侧展开**</mark> —— 即模型侧的交错多图能力，<mark class="hl-key">**在数据章节里找不到对应的支撑细节**</mark>
- <mark class="hl-trick">**§4.1 Table 2 的 TI2I 比例（0.1 → 0.3）是否包含 multi-image 数据、multi-image 占比多少，均未说明**</mark>
:::

#### ⑤ 为什么这部分值得学：反向设计而非正向观察

<mark class="hl-trick">**Mage-Flow 的 Editing 数据是比较典型的 edit taxonomy + synthetic edit triples + VLM filtering；Qwen-Image-2.0 在更上游就把数据分布设计成**</mark>：

$$
\boxed{
\text{Natural Image}
+
\text{Design / Layout}
+
\text{Text-rich}
+
\text{Synthetic}
+
\text{Single-image Edit}
+
\text{Multi-image Edit}
}
$$

<mark class="hl-key">**它不是「先有数据，再看看模型学出什么」，而是先定义目标能力，然后反向设计数据域。**</mark>这是一个很强的工业思路：

$$
\boxed{
\text{Capability Taxonomy}
\rightarrow
\text{Data Taxonomy}
\rightarrow
\text{Training Mixture}
}
$$

而不是：

$$
\text{Raw Data} \rightarrow \text{Train} \rightarrow \text{看看会什么}
$$

<mark class="hl-key">**这条可以直接加进 [Playbook §1](./training-playbook.md)。**</mark>

#### ⑥ 提前记住的一个后续数字（此处不展开）

<mark class="hl-trick">**Qwen-Image-2.0 从 Data section 开始就已经把 T2I 和 TI2I 放在同一套数据体系里设计**</mark>，后面训练阶段也确实会把二者混起来，比例会从：

$$
0.9 : 0.1
$$

逐步变成：

$$
0.7 : 0.3
$$

<mark class="hl-trick">**但这个比例属于后面的 §4.1（Pre-training / Continual Pre-training / SFT），本节不提前展开。**</mark>已核实 Table 2 的完整口径 → [§4.1](#41-multistage-training)。

#### §2.1 一句话总结

$$
\boxed{
\text{Qwen-Image-2.0 的数据不是按「来源」组织，而是按「最终能力」组织。}
}
$$

<mark class="hl-key">**T2I 不是只有摄影图，Editing 也不是只有单图局部编辑；layout、text、long-tail、synthetic、reference-based、multi-image 从数据建设阶段就已经被当成独立能力方向。**</mark>

### 2.2 Data Annotation

#### ① 出发点：单一 caption 模板不够用

<mark class="hl-trick">**论文的判断很直接**</mark>：面对 diverse scenarios 下复杂度差异极大的图像，用<mark class="hl-key">**同一种自然语言 caption**</mark> 去描述是<mark class="hl-trick">**不够的**</mark>。普通照片、海报、流程图、知识型图像，需要模型学到的东西完全不同。

因此作者构建了一个 <mark class="hl-key">**fine-grained captioning framework**</mark>，其依据被论文明确写成两个维度：

$$
\boxed{
\text{tailored to different } \textbf{task types} \quad \text{and} \quad \textbf{image characteristics}
}
$$

<mark class="hl-key">**注意是「task type」+「image characteristic」两个维度共同决定**</mark>，不是单看图像类型。据此设计四套专用 captioning scheme：

$$
\boxed{
\text{General Caption}
+
\text{Text Caption}
+
\text{Knowledge Caption}
+
\text{Structured Caption}
}
$$

<mark class="hl-trick">**这一节最值得记的思想：caption 本身就是数据建模的一部分，而不是「数据清洗结束后统一调用一个 VLM 出一段文字」。**</mark>

$$
\boxed{
\text{Caption 设计} \in \text{Data Design}, \qquad \text{not} \in \text{Post-cleanup Annotation}
}
$$

#### ② General Caption：最基础的一类

<mark class="hl-key">**适用对象**</mark>：<mark class="hl-trick">**任意分辨率、任意复杂度的图像**</mark>。目的是给出 comprehensive and detailed 的自然语言视觉描述。

论文列出它覆盖的内容（这里论文用的措辞是 <mark class="hl-trick">**"not only ... but also"**</mark>，即前几项是主体、后几项是额外叠加）：

$$
\boxed{
\text{Main Objects}
+
\text{Scene Context}
+
\text{Spatial Relationships}
+
\text{Textual Content \& Its Semantics (whenever present)}
}
$$

<mark class="hl-key">**外加两个能力**</mark>：

$$
\text{Multilingual Generation}
\qquad
\text{Varying Caption Lengths}
$$

<mark class="hl-key">**这是最容易被忽略、但对训练最有价值的一条：论文明确写了 `varying caption lengths`。**</mark><mark class="hl-trick">**也就是说 Qwen 并没有假设「caption 越长越好」，而是允许不同长度的描述同时存在**</mark>——这与 [Playbook §1](./training-playbook.md) 里「长 prompt 后半段被忽略 → 需要增长文本 supervision」的诊断是同一族问题，只是 Qwen 的解法是<mark class="hl-key">**让长度本身成为一个可变量，而不是单调拉长**</mark>。

可以理解为：

$$
\boxed{
\text{Object}
+
\text{Attribute}
+
\text{Scene}
+
\text{Spatial Relation}
+
\text{Visible Text}
}
$$

<mark class="hl-key">**它承担的是最基础的 image-text alignment。**</mark>一张普通街景不应该只写 `a city street`，而要把人物、车辆、建筑、相对位置、环境说清楚。

::: warning 未公开
<mark class="hl-trick">**论文提到 `varying caption lengths`，但完全没有公开有哪些长度档位、训练时如何采样不同长度、以及不同长度与不同 resolution stage 如何对应。**</mark>这是复现时最想知道的参数之一。
:::

#### ③ Text Caption：本篇最有特色的一类

<mark class="hl-trick">**适用对象**</mark>：<mark class="hl-key">**包含 dense text 或 abstract symbols 的图像**</mark>——论文列举了 <mark class="hl-trick">**presentation slides、comics、posters、educational materials**</mark> 等 text-centric visual materials。

<mark class="hl-key">**实现方式是 `multiple prompting templates`**</mark>：为这类图像单独设计多套 prompt 模板，而不是复用 General 的模板。

<mark class="hl-trick">**与 General 相比，Text Caption 更强调四件事**</mark>（论文用 <mark class="hl-trick">**`greater emphasis on`**</mark> 明确对比）：

$$
\boxed{
\text{Dense Textual Content}
+
\text{Layout Structure}
+
\text{Visual Symbols}
+
\text{Semantic Relations}
}
$$

<mark class="hl-key">**由此它的适用范围被论文定义为**</mark>：<mark class="hl-trick">**text-rich、structurally complex、semantically organized**</mark> 的图像——注意这三个限定词是递进的：文字多、结构复杂、且有语义组织性。

##### `layout structure` 为什么关键

<mark class="hl-trick">**`layout structure` 是这一类里最容易被低估的字段。**</mark>以一张 PPT 为例，仅仅知道里面有 `Model Architecture` / `Data Pipeline` / `Experimental Results` 这些字符串是**远远不够**的，还需要理解：

- 哪个是**标题**；
- 哪些属于**左栏 / 右栏**；
- 哪个文本框**对应哪张图**；
- **箭头**连接了什么；
- 哪一块是 **caption**；
- 哪些文字属于**图表本身**。

<mark class="hl-trick">**否则模型即使学会了「写字」，也很难学会**</mark>

$$
\boxed{
\text{文字应该放在哪里、和视觉元素怎样组织在一起}
}
$$

<mark class="hl-key">**所以 Text Caption 真正解决的不是 OCR，而是三件事的组合**</mark>：

$$
\boxed{
\text{Text Rendering}
+
\text{Layout Understanding}
+
\text{Text-Visual Binding}
}
$$

<mark class="hl-key">**这直接对应论文摘要里就作为重点宣传的能力**</mark>——原文两处：

> Qwen-Image-2.0 enables <mark class="hl-key">**ultra-long text rendering with instructions of up to 1K tokens**</mark>, and can directly produce text-dense visual outputs such as slides, posters, ...

<mark class="hl-key">**`up to 1K tokens` 这个数字已核实，确实出自论文摘要，不是外部推测。</mark>它把「Text Caption 监督」与「1K-token 超长文字渲染」这两个目标直接连起来了：<mark class="hl-trick">**如果 captioner 只教模型「写 20 个字」，模型不可能在推理时稳定渲染 1K token 的文字内容**</mark>。

<mark class="hl-key">**对我们的 Recipe 最有启发的一条：想做好 OCR / infographic / poster，仅仅给普通 caption 追加 OCR transcription 是不够的，最好建立独立的 Text-rich Annotation Pipeline。**</mark>

<mark class="hl-trick">**这一点和仓库里已有的笔记是同一共识但不同做法**</mark>：Z-Image 是「先 OCR、再据 OCR 生成 caption」的两步 CoT 式流程 → [Z-Image 长文本处理](./z-image.md)；Qwen 则是<mark class="hl-key">**为 text-heavy 图像单独开一套专用 captioning scheme**</mark>。<mark class="hl-trick">**两者都在解决「通用 captioner 会漏掉密集文字」这个同一个问题。**</mark>

#### ④ Knowledge Caption：从「看得见」到「知道是什么」

<mark class="hl-trick">**这一类与前两类有本质区别。**</mark>General 和 Text 基本都在描述：

$$
\text{“图里能直接看到什么”}
$$

<mark class="hl-key">**Knowledge Caption 则进一步注入 image-related 的**</mark>：

$$
\boxed{
\text{Background Information}
+
\text{Contextual Cues}
+
\text{Auxiliary Conditions}
}
$$

<mark class="hl-trick">**论文对这三者的措辞是 `in the form of conditions`**</mark>——即以 condition 的形式注入，而非混在自然语言描述里。<mark class="hl-key">**目的是让模型不仅学视觉内容，还把视觉内容与 world knowledge 建立联系**</mark>：

$$
\boxed{
\text{capture image semantics} + \text{relevant world knowledge}
}
$$

<mark class="hl-key">**论文最关键的一句对比**</mark>是 <mark class="hl-trick">**`Unlike captions that focus only on explicitly visible content`**</mark>——即它与「只关注显式可见内容」的 caption 相对立，通过引入 supplementary information 建立 richer semantic connections。

<mark class="hl-trick">**下面这个例子纯粹帮助理解，不是论文原例：**</mark>若图片是埃菲尔铁塔，General Caption 可能只写 `A tall iron lattice tower stands in an urban landscape...`，而 Knowledge Caption 的监督里可能额外带入 `Eiffel Tower, Paris, France, landmark...`。<mark class="hl-key">**它不只说「看起来是什么」，还告诉模型「它在知识世界里是什么」。**</mark>

这类监督潜在解决的是一个很实际的问题：

$$
\boxed{
\text{Visual Appearance}
\;\longleftrightarrow\;
\text{Semantic / World Knowledge}
}
$$

<mark class="hl-key">**如果模型未来要理解 landmark、historical artifact、scientific diagram、文化符号、特定人物或物体，这类监督会比单纯 visual caption 丰富得多。**</mark>

::: warning 未公开（这里必须严格）
<mark class="hl-trick">**论文对 Knowledge Caption 的实现几乎没有交代**</mark>，因此<mark class="hl-key">**目前只能确认「它加入图像相关的背景知识与上下文」这一层，不能再往下假设任何具体 pipeline**</mark>。具体未公开的是：

- 由<mark class="hl-trick">**哪个模型**</mark>生成（论文全篇未指定 captioner）
- 外部知识从<mark class="hl-trick">**哪里检索**</mark>
- <mark class="hl-trick">**是否使用搜索 / RAG**</mark>
- `auxiliary condition` 的<mark class="hl-trick">**具体格式是什么**</mark>
:::

#### ⑤ Structured Caption：为复杂关系图换一种监督语言

<mark class="hl-trick">**适用对象**</mark>：<mark class="hl-key">**关系复杂、元素众多的图像**</mark>——论文列举 <mark class="hl-trick">**relation graphs、flowcharts、diagrams**</mark>。

<mark class="hl-key">**论文对这一类的动机说得最重**</mark>：<mark class="hl-trick">**natural language descriptions alone are often insufficient to fully and clearly represent the objects and their interactions**</mark>——自然语言本身存在表达瓶颈。

<mark class="hl-trick">**一张图可能有几十个 entity，它们之间存在复杂的**</mark> hierarchy / topology / dependency，<mark class="hl-trick">**强行写成一长段自然语言很容易丢结构。**</mark>

因此 Structured Caption 显式建模三元组：

$$
\boxed{
\text{Entity}
+
\text{Attribute}
+
\text{Relation}
}
$$

<mark class="hl-key">**它希望模型真正学到的是关系本身，而不是「图中有 A、B、C」**</mark>：

$$
A \rightarrow B \rightarrow C
\qquad\text{或}\qquad
\text{Entity}_1 \xrightarrow{\ \text{relation}\ } \text{Entity}_2
$$

<mark class="hl-key">**论文明确列出 Structured Caption 有助于学习的三件事**</mark>：

$$
\boxed{
\text{Hierarchical Relations}
+
\text{Topological Dependencies}
+
\text{Semantic Interactions Among Visual Elements}
}
$$

<mark class="hl-key">**所以对 flowchart、diagram、infographic 这类图像，这种 supervision 明显比普通 long caption 更自然。**</mark>

::: warning 未公开
<mark class="hl-trick">**Structured Caption 的具体 schema 论文完全没给**</mark>——是 JSON、triplet、还是某种 DSL 表达，<mark class="hl-key">**无从判断**</mark>。另外也没说 entity/relation 是人工标注、规则抽取还是 VLM 抽取。
:::

#### ⑥ 四类放在一起看：与 Mage-Flow 的「多粒度」是两个不同维度

<mark class="hl-key">**Qwen 的设计不是**</mark> $\text{Image} \rightarrow \text{One Caption}$ <mark class="hl-key">**而是**</mark>：

$$
\boxed{
\text{先判断这张图需要模型学习什么}
\;\rightarrow\;
\text{选择合适的 supervision representation}
}
$$

| 图像特征 / 训练目标 | 路由到 |
| :--- | :--- |
| 普通视觉内容 | <mark class="hl-trick">General Caption</mark> |
| 文字密集、版面复杂 | <mark class="hl-key">Text Caption</mark> |
| 需要世界知识联系 | <mark class="hl-key">Knowledge Caption</mark> |
| 复杂结构与关系 | <mark class="hl-key">Structured Caption</mark> |

<mark class="hl-key">**这和 Mage-Flow 的 multi-granularity caption 是两个正交的维度，不冲突。**</mark>Mage-Flow 的做法（→ [Mage-Flow §2.1④](./mage-flow.md)）是：

$$
\text{Phrase} \rightarrow \text{Entity} \rightarrow \text{Composition} \rightarrow \text{Photographic}
$$

<mark class="hl-trick">**即对同一张图产出四列不同粒度的描述**</mark>，因此可以理解为：

$$
\boxed{
\text{Mage-Flow} = \text{同一种视觉内容的} \textbf{描述粒度} \text{不同}
}
$$

$$
\boxed{
\text{Qwen-Image-2.0} = \text{不同类型视觉任务需要} \textbf{不同形式的监督语言}
}
$$

<mark class="hl-key">**两个思路可以同时存在**：前者管「描述多细」，后者管「用什么语言描述」。</mark>

#### ⑦ §2.2 最该进 Recipe 的一条

<mark class="hl-key">**§1.2 最值得放进 [Playbook](./training-playbook.md) 的结论不是「有四种 caption」这个事实，而是**</mark>：

$$
\boxed{
\text{Caption 不应该只有一个通用模板。}
}
$$

<mark class="hl-key">**应该根据数据本身承担的 capability，决定 supervision representation**</mark>：

$$
\boxed{
\begin{aligned}
\text{General visual understanding} &\rightarrow \text{natural-language caption}\\
\text{Text-rich generation} &\rightarrow \text{text + layout-aware caption}\\
\text{World knowledge} &\rightarrow \text{knowledge-enriched caption}\\
\text{Diagram / complex relation} &\rightarrow \text{structured entity-relation representation}
\end{aligned}
}
$$

#### ⑧ 未公开细节汇总

<mark class="hl-trick">**以下全部未公开，而这恰恰是真正复现时最想知道的部分：**</mark>

- <mark class="hl-trick">**四类 Caption 的数据量与采样比例**</mark>
- <mark class="hl-trick">**一张图是否可能同时拥有多种 Caption**</mark>（§4.1 只有一个 T2I/TI2I 粗粒度比例，没有 caption 类型维度）
- <mark class="hl-trick">**训练时是随机选 caption 类型，还是按 image category 固定 routing**</mark>
- <mark class="hl-trick">**用哪个 VLM 生成这些 caption**</mark>（论文全篇未指定 captioner）
- <mark class="hl-trick">**具体 system prompt 与 Text Caption 的 `multiple prompting templates` 内容**</mark>
- <mark class="hl-trick">**Knowledge Caption 的知识来源与检索方式**</mark>
- <mark class="hl-trick">**Structured Caption 的 schema 格式**</mark>
- <mark class="hl-trick">**Caption quality 如何过滤**</mark>（§2.3 的八个 filter 里<mark class="hl-key">**只有 CLIP 与 Token Length 作用于文本侧，且二者都不检查事实正确性**</mark>）
- <mark class="hl-trick">**不同 caption 类型在不同 Pretrain Stage 是否使用不同权重**</mark>

### 2.3 Multi-Stage Training Data Strategy ★

#### ① 核心：同时存在两条 curriculum

<mark class="hl-trick">**Qwen-Image-2.0 没有先把全量数据一次性清洗成一个固定训练集，然后从 256 一路训到 2K。**</mark>论文明确说的是：<mark class="hl-key">**过滤策略随训练过程逐阶段执行，数据分布也持续被重新精炼**</mark>（<mark class="hl-trick">**"applied progressively throughout the training process, with data distributions continuously refined over time"**</mark>）。

因此这一节实际上同时存在两条 curriculum：

$$
\boxed{
\text{Resolution Curriculum}
\;+\;
\text{Data Quality / Composition Curriculum}
}
$$

<mark class="hl-key">**即：分辨率越往后升，数据质量门槛也越来越高，数据类型也越来越丰富。**</mark>这个 design 基于 Qwen-Image（Wu et al., 2025），整条流水线见论文 Fig. 6。

#### ② 六个阶段总览

| Stage | Resolution | 数据组成 / 新变化 | 主要数据处理 |
| :-: | :--- | :--- | :--- |
| <mark class="hl-trick">**S1**</mark> | 256P | T2I | <mark class="hl-key">**8 个基础过滤器**</mark> |
| <mark class="hl-trick">**S2**</mark> | 256P | <mark class="hl-key">**T2I + Edit（首次引入 TI2I）**</mark> | 沿用 S1 过滤后的 T2I |
| <mark class="hl-trick">**S3**</mark> | 512P | <mark class="hl-key">**T2I + Edit + Synthetic**</mark> | 引入 Synthetic Data |
| <mark class="hl-trick">**S4**</mark> | 512P / 1024P | T2I + TI2I | <mark class="hl-key">**针对 1024P 增加 4 个高分辨率质量筛选**</mark> |
| <mark class="hl-trick">**S5**</mark> | 512P / 1024P / 2048P | T2I + TI2I | <mark class="hl-key">**为 2048P 增加专门的 Resolution Filter**</mark> |
| <mark class="hl-trick">**S6**</mark> | 目标高分辨率 | 更精的数据分布 | <mark class="hl-key">**复用前序算子 + 更严阈值 + Distribution Filter**</mark> |

<mark class="hl-key">**这六个阶段不是简单的**</mark> $256 \rightarrow 512 \rightarrow 1024 \rightarrow 2048$ <mark class="hl-key">**，而是**</mark>：

$$
\boxed{
\text{建立基础语义}
\rightarrow
\text{加入 Editing}
\rightarrow
\text{补 Synthetic 能力}
\rightarrow
\text{渐提高分辨率与质量}
\rightarrow
\text{最后用严格分布做 SFT}
}
$$

![Qwen-Image-2.0 Fig.6：六阶段数据流水线 Sankey 全景图（横轴＝训练阶段推进，纵轴＝数据流量）。**最左** 红色竖条 `T2I Data` 为唯一原始输入，向右分出 **8 条灰色支流** 分别流向 8 个过滤器节点：`S1 256p Broken Files Filter`、`Resolution Filter`、`Deduplication Filter`、`NSFW Filter`、`Rotation Filter`、`Entropy Filter`、`CLIP Filter`、`Token Length Filter`（自上而下，灰色＝被剔除的样本）。**主训练节点自左向右依次为** `S1 256p Training` → `S2 256p Training` → `S3 512p Training` → `S4 512p, 1024p Training` → `S5 512p, 1024p, 2048p Training` → 最右 `S6 2048p SFT`（紫色窄条）。**两股新数据在中途汇入**：黄色 `Edit Data` 在 S2 位置汇入（并单独标注在 S3 下方黄色块），绿色 `S3 512p Synthetic Data` 在 S3 位置从底部汇入。**过滤器按阶段分组**：S1 组 8 个、S4 组 4 个（`S4 1024p Resolution Filter` / `Image Quality Filter` / `Image Aesthetic Filter` / `Compression Quality Filter`）、S5 组 1 个（`S5 2048p Resolution Filter`）、S6 组 1 个（`S6 Distribution Filter`，其灰色支流在图中体积最大）。](/qwen2-fig6-data-curriculum.png)

<mark class="hl-key">**这张图比正文 §2.3 的文字描述精确得多，有三处正文没写的细节：**</mark>

| # | 图中额外信息 | 正文 |
| :-: | :--- | :--- |
| 1 | <mark class="hl-trick">**每个阶段的确切训练分辨率标签**</mark>：`[S1 256p]`、`[S2 256p]`、`[S3 512p]`、`[S4 512p, 1024p]`、`[S5 512p, 1024p, 2048p]`、<mark class="hl-key">**`[S6 2048p] SFT`**</mark> | 正文只说「256p→512p→2048p」，未给每阶段确切组合 |
| 2 | <mark class="hl-trick">**过滤器带阶段前缀**</mark>：Resolution Filter 在 S1 和 S4 和 S5 各有一个，<mark class="hl-key">**是三个不同实现，不是同一个**</mark> | 正文未区分 |
| 3 | <mark class="hl-trick">**`Edit Data` 与 `Synthetic Data` 作为独立数据流在图中显式标注**</mark> | 正文只在 S2/S3 顺带一提 |

<mark class="hl-key">**注意第 2 点：`[S6 2048p] SFT` 这个标签与 §4.1 Table 2 有轻微张力**</mark>——<mark class="hl-trick">**图把 S6 标为 2048p，而 Table 2 的 SFT resolution 写的是 512/1024/2048**</mark>。<mark class="hl-key">**推测图只标注了 SFT 的最高分辨率，但论文没有明确说明，因此不能断言 SFT 只在 2048p 训练。**</mark>

<mark class="hl-key">**关于「灰色支流 = 被剔除的样本」这个读法**</mark>：图中所有过滤器节点都只接收灰色带、没有彩色带流出，<mark class="hl-trick">**而各阶段训练节点的入流都是彩色带**</mark>。<mark class="hl-key">**因此灰色 = 被 filter 剔除、彩色 = 保留并进入训练**</mark>，这是唯一自洽的读法。

::: warning 带宽编码了相对量，但无法定量引用
<mark class="hl-trick">**Sankey 的带宽显然编码了相对数据量**</mark>——<mark class="hl-key">**S6 的 Distribution Filter 灰色支流在图中是体积最大的一股**</mark>，直观对应「SFT 阶段剔除量最大」。

<mark class="hl-trick">**但论文没有给图例、坐标轴刻度、任何数字或百分比**</mark>。<mark class="hl-key">**因此这只能作为定性的量级感知，不能反推出任何具体比例或样本量**</mark>——<mark class="hl-trick">**与 Fig. 5 的情况一样：图能告诉你「哪些量级大」，不能告诉你「是多少」**</mark>。
:::

::: warning Synthetic Data 是否延续到 S4 之后：图能确认一半
<mark class="hl-trick">**这个疑问现在可以部分收敛了。**</mark>放大 Fig. 6 底部可见：<mark class="hl-key">**绿色 `[S3 512p] Synthetic Data` 色带确实向右上方汇入 `[S4 512p, 1024p] Training` 的入流**</mark>，所以

<mark class="hl-key">**可以确认**</mark>：Synthetic Data <mark class="hl-trick">**至少延续进了 S4**</mark>（此前仅凭正文无法判断）
- <mark class="hl-trick">**仍然无法确认 S5 / S6**</mark>：因为 <mark class="hl-key">**从 S4 节点开始，Fig. 6 的色带就不再按数据来源拆分**</mark>——S4、S5、S6 各自是<mark class="hl-trick">**一整块单色**</mark>（青、蓝、紫），<mark class="hl-key">**无法再把 Synthetic 单独追踪出来**</mark>

<mark class="hl-trick">**另外六个阶段的正式标题（Stage 1–6 的命名）里 Synthetic Data 从未出现**</mark>，只有 S3 正文提到它——<mark class="hl-key">**标题的沉默与图的证据是矛盾的，论文没有解释这个不一致。**</mark>
:::

#### ③ S1：256P T2I，先把最基础的数据底座清干净

<mark class="hl-trick">**Stage 1 只做 256P T2I pre-training，而且是六个阶段里过滤流程写得最具体的一个**</mark>——论文明确说这一阶段的目标分辨率是 $256 \times 256$，因此先做物理层面的筛选。

原始 T2I 数据依次经过<mark class="hl-key">**八个 filter**</mark>：

$$
\boxed{
\text{Broken Files}
\rightarrow
\text{Resolution}
\rightarrow
\text{Deduplication}
\rightarrow
\text{NSFW}
\rightarrow
\text{Rotation}
\rightarrow
\text{Entropy}
\rightarrow
\text{CLIP}
\rightarrow
\text{Token Length}
}
$$

<mark class="hl-key">**这八个可以分成三类理解，其中顺序不是随意的——论文是按「先便宜后昂贵」排列的：**</mark>

| 类别 | Filter | 作用 |
| :--- | :--- | :--- |
| <mark class="hl-trick">**① 物理合法性**</mark>（最便宜） | Broken Files、Resolution、Rotation | 图片能读、尺寸够 256×256、方向正常（Rotation 是<mark class="hl-key">**纠正或丢弃**</mark>朝向不当的图） |
| <mark class="hl-trick">**② 内容质量与安全**</mark> | NSFW、Entropy | 移除不当内容；Entropy Filter 排除<mark class="hl-key">**信息量异常低或异常高**</mark>的样本 |
| <mark class="hl-key">**③ 监督质量**</mark>（最贵） | Deduplication、CLIP、Token Length | 去重；CLIP Filter 删掉<mark class="hl-trick">**image-text 相似度过低**</mark>的配对；Token Length Filter 限制<mark class="hl-trick">**文本描述长度超出可接受区间**</mark>的样本 |

<mark class="hl-key">**这个「先便宜后昂贵」的排序本身就是一条可直接复用的工业经验**</mark> → [Playbook §1 诊断字典](./training-playbook.md)：<mark class="hl-trick">**把需要跑 VLM/CLIP 的检查放到最后，前面先用几乎零成本的规则先砍掉大部分垃圾**</mark>。

<mark class="hl-trick">**关于 Entropy Filter 必须谨慎**</mark>：论文只说「异常低或异常高」，<mark class="hl-key">**没有给 entropy 的定义（是图像信息熵？还是 CLIP embedding 的熵？）和阈值**</mark>，不要自行补。

所以 S1 的本质非常清楚：

$$
\boxed{
\text{低分辨率} + \text{大规模干净 T2I}
\;\rightarrow\;
\text{建立基础 image-text semantic mapping}
}
$$

<mark class="hl-trick">**此时还没有急着上 Editing，也没有急着用高分辨率。**</mark>

#### ④ S2：仍是 256P，但开始把 Editing 加进来

<mark class="hl-key">**Stage 2 最值得记的一点：作者没有先把 T2I 在高分辨率上训完、再单独做 Edit。**</mark>而是在<mark class="hl-trick">**仍然是 256P**</mark> 时，就把 S1 过滤后的 T2I 与 TI2I 混起来：

$$
\boxed{
\text{Stage 2} = \text{filtered T2I (from S1)} + \text{TI2I}, \qquad \text{all at } 256\text{p}
}
$$

<mark class="hl-key">**论文的措辞是 `combined and used directly for Stage 2 training`，并明确说这一阶段让模型在「统一的低分辨率预训练设定下」同时学会 text-to-image generation 与 text-guided editing。**</mark>

这反映的思想是：

> <mark class="hl-key">**Generation 与 Editing 的统一，不是最后 SFT 时才做，而是在非常早期的低分辨率阶段就开始联合建立。**</mark>

<mark class="hl-trick">**为什么合理（这是帮助理解的解释，不是论文结论）**</mark>：256P 时算力便宜，模型主要在学 $\text{semantic understanding} + \text{basic correspondence} + \text{edit instruction semantics}$，此时让它同时接触 T2I 和 TI2I，可以更早建立统一条件形式。

<mark class="hl-trick">**必须严格区分**</mark>：论文明确公开的是「256P 时就开始 T2I + TI2I 联训」这个事实，<mark class="hl-key">**但论文没有做任何 ablation 证明「越早加入 Edit 一定更好」**</mark>。<mark class="hl-trick">**不要把它当成被验证过的结论。**</mark>

#### ⑤ S3：升到 512P，第一次加入 Synthetic Data

<mark class="hl-key">**Stage 3 有两个变化同时发生**</mark>：分辨率 $256\text{p} \rightarrow 512\text{p}$，数据组成从 $\text{T2I} + \text{Edit}$ 扩成：

$$
\boxed{
\text{T2I}
+
\text{Edit}
+
\text{Synthetic}
}
$$

<mark class="hl-key">**论文对 Synthetic 的目的写得很直接**</mark>：<mark class="hl-trick">**`enrich the training distribution and improve data diversity`**</mark>——即它<mark class="hl-key">**不只是为了扩充总量，而是为了补充原始数据分布中的不足**</mark>。

结合 §2.1 的 Fig. 5，Synthetic 至少包含：

$$
\text{Chart},\qquad \text{Text},\qquad \text{Color},\qquad \text{Others}
$$

<mark class="hl-trick">**所以这里很可能是在模型已有基础视觉能力后，再逐渐加强结构化、文字、颜色控制等 Web 数据天然不足的能力。**</mark>

::: warning 未公开
<mark class="hl-trick">**论文没有解释为什么偏偏在 512P 才加入 Synthetic**</mark>（是 512P 下 synthetic 数据质量才够？还是顺序实验的结果？），也<mark class="hl-key">**没有公开各 synthetic category 的量、采样比例、生成器和 QC 流程**</mark>。只能确认它在 Stage 3 成为正式训练分布的一部分。
:::

#### ⑥ S4：512P / 1024P，第一次把「高分辨率质量」当独立问题

<mark class="hl-key">**这一阶段特别值得学。**</mark>Stage 4 开始混 $512\text{p} + 1024\text{p}$，但作者<mark class="hl-trick">**并不是「只要原图尺寸够 1024 就拿来训」**</mark>。针对 1024P，额外增加四个 filter：

$$
\boxed{
\text{Resolution Filter}
+
\text{Image Quality Filter}
+
\text{Image Aesthetic Filter}
+
\text{Compression Quality Filter}
}
$$

论文对四个的分工是：Resolution Filter 保留<mark class="hl-trick">**空间分辨率足够**</mark>的图；Image Quality Filter 移除<mark class="hl-trick">**低保真**</mark>图；Image Aesthetic Filter 挑选<mark class="hl-key">**视觉上美观**</mark>的样本；Compression Quality Filter 丢弃<mark class="hl-trick">**压缩严重或带 artifact**</mark>的图。

<mark class="hl-key">**这个设计很工业，因为一张图满足 $\text{resolution} \ge 1024$ 并不意味着它适合高分辨率训练**</mark>——它可能是：

- 本来是<mark class="hl-trick">**低清图放大的**</mark>
- <mark class="hl-trick">**JPEG 压缩很重**</mark>的
- 尺寸大但<mark class="hl-trick">**细节很差**</mark>的
- <mark class="hl-trick">**aesthetic 很差或 artifact 很多**</mark>的

所以高分辨率阶段需要额外筛选：

$$
\boxed{
\text{尺寸真的够}
+
\text{视觉质量够}
+
\text{审美够}
+
\text{压缩损伤可接受}
}
$$

<mark class="hl-key">**由此可以提炼出一条非常直接的原则**</mark>：

$$
\boxed{
\text{High Resolution Data} \;\neq\; \text{Large Pixel Count Data}
}
$$

<mark class="hl-key">**真正应该追求的是 high-resolution + high-fidelity 的交集，而不是分辨率这一个维度。**</mark>→ 可补入 [Playbook §1](./training-playbook.md) 的诊断字典。

#### ⑦ S5：扩到 2048P，形成真正的 Multi-Resolution Pretraining

Stage 5 进入多分辨率设定：

$$
\boxed{
512\text{p} + 1024\text{p} + 2048\text{p}
}
$$

<mark class="hl-key">**对于新加入的 2048P 数据，作者又专门增加了一个 Resolution Filter**</mark>，以保证这些图满足<mark class="hl-trick">**更严格的 2048p 分辨率要求**</mark>。论文说这一阶段让模型跨多个 scale 学习，从而<mark class="hl-key">**进一步加强 high-resolution generation 与 high-resolution editing**</mark>。

<mark class="hl-key">**这里有一条非常值得记住的 recipe：他们没有升到 2K 后把 512 / 1K 全扔掉，而是让三者共存。**</mark>即 curriculum <mark class="hl-trick">**不是**</mark> $256 \rightarrow 512 \rightarrow 1024 \rightarrow 2048$ <mark class="hl-trick">**的后者替代前者，而是后期进入**</mark>：

$$
\boxed{
\text{Multi-resolution mixture}
}
$$

<mark class="hl-key">**这样避免训练分布只剩下昂贵的 2K，同时维持多个 resolution scale 的能力**</mark>——对推理时不同尺寸的请求都要有覆盖。

::: warning 未公开
<mark class="hl-trick">**$p_{512} : p_{1024} : p_{2048}$ 的真实采样比例论文没有给。**</mark>§4.1 Table 2 给了不同 resolution 对应的 batch size（如 continual pre-training 是 16/8/4），<mark class="hl-key">**但 batch size 不等于数据采样概率，不能反推比例**</mark>。
:::

#### ⑧ S6：SFT 不是「拿 Stage 5 数据跑 10K step」

<mark class="hl-key">**Stage 6 的定位与前面所有预训练阶段都不同，论文用 `Unlike the preceding pre-training stages` 明确转折**</mark>：前序阶段是<mark class="hl-trick">**逐步把分辨率范围从 256p 扩到 2048p**</mark>，而 Stage 6 <mark class="hl-trick">**不再重点扩展 resolution range，而是重点**</mark>：

$$
\boxed{
\text{Refine Data Distribution}
+
\text{Improve Sample Quality}
}
$$

<mark class="hl-key">**具体做法是「复用前序阶段的 filtering operators，但用更严的阈值」**</mark>：

$$
\boxed{
\text{Distribution Filter} = \text{reuse previous operators} + \text{stricter thresholds}
}
$$

它移除两类样本：

$$
\text{low-quality samples}
\qquad \text{和} \qquad
\boxed{\text{imbalanced samples}}
$$

<mark class="hl-key">**`imbalanced` 这个词尤其值得注意**</mark>——它说明 SFT 不只是「把 aesthetic threshold 从 5.5 调到 6.5」这种单点收紧，而<mark class="hl-trick">**还在主动调整训练数据的类别分布**</mark>。所以 SFT 真正做的是两件事：

$$
\boxed{
\text{Quality Selection}
+
\text{Distribution Shaping}
}
$$

<mark class="hl-key">**这与仓库里已有的结论几乎完全一致**</mark>——<mark class="hl-trick">**Pretrain 学 coverage，SFT 塑造最终输出 distribution**</mark> → [Playbook §2](./training-playbook.md)。<mark class="hl-key">**Qwen 这一篇为它提供了直接的工业证据。**</mark>

#### ⑨ 六阶段压缩成一条逻辑

$$
\boxed{
\begin{aligned}
S1 &: \text{先把 T2I 基础语义学起来} \\
S2 &: \text{尽早引入 Editing，建立统一能力} \\
S3 &: \text{升到 512，并用 Synthetic 补能力分布} \\
S4 &: \text{进入 1K，开始严格控制高分辨率质量} \\
S5 &: \text{扩到 2K，同时保持 multi-resolution 共存} \\
S6 &: \text{不再追规模，严格控制质量与分布做 SFT}
\end{aligned}
}
$$

<mark class="hl-key">**这比简单记 $256 \rightarrow 512 \rightarrow 1024 \rightarrow 2048$ 有价值得多**</mark>——<mark class="hl-trick">**分辨率只是表层，真正的 curriculum 是「能力引入时机」与「质量门槛」两条线在同时推进**</mark>。

#### ⑩ 极易混淆：六阶段 pipeline vs 三阶段 training

<mark class="hl-trick">**Data Section（§2.3）写的是 6-stage data pipeline；而 §4.1 Training 又把整个训练概括成三段**</mark>：

$$
\boxed{
\text{Pre-training}
\rightarrow
\text{Continual Pre-training}
\rightarrow
\text{SFT}
}
$$

<mark class="hl-key">**这两个描述不一定矛盾，但它们处于完全不同的颗粒度**</mark>：

| | §2.3 Data Pipeline | §4.1 Training |
| :--- | :--- | :--- |
| 描述对象 | <mark class="hl-trick">**数据被过滤/引入的六次演化**</mark> | <mark class="hl-trick">**模型参数更新的三个宏观 phase**</mark> |
| 阶段数 | 6 | 3 |
| 是否含分辨率 | 是（256→2048） | 是（256/512→2048） |

<mark class="hl-trick">**关键：论文没有给出任何一张把二者对应起来的表**</mark>，因此<mark class="hl-key">**不能自己强行断言 $S1,S2,S3 \equiv$ Pre-training 之类的精确映射**</mark>。

<mark class="hl-trick">**能观察到的只是二者大体同向**</mark>：§2.3 的 S1–S5 全部标注为 `pre-training`，S6 标注为 `Supervised fine-tuning`；而 §4.1 的三段是 Pre-training / Continual pre-training / SFT。<mark class="hl-key">**S1–S5 大致落在 §4.1 的前两段，S6 对应 SFT，但 S1–S5 在 §4.1 内部如何切分，论文未给。**</mark>

#### ⑪ §2.3 最该进 Recipe 的一条

<mark class="hl-key">**§1.3 最应该加进工业训练字典的不是任何具体 filter 阈值，而是这条方法论**</mark>：

$$
\boxed{
\text{不要先构造一个「最终数据集」然后一直训练。}
}
$$

而应该：

$$
\boxed{
\text{训练阶段变化}
\;\Rightarrow\;
\text{Resolution、Quality Threshold、Task Mixture、Capability Distribution 一起变化}
}
$$

<mark class="hl-key">**这其实比某个具体 filter threshold 更有复用价值**</mark> —— 阈值会随数据集变化，但「阶段与数据策略必须同步演进」这个结构性认识是通用的。

#### ⑫ 未公开细节汇总

<mark class="hl-trick">**这一节把 curriculum 结构公开得很清楚，但 production thresholds 与 mixture recipe 几乎全缺：**</mark>

- 每个 Stage 的<mark class="hl-trick">**实际数据规模**</mark>
- S1 八个 filter 的<mark class="hl-trick">**具体阈值**</mark>
- <mark class="hl-trick">**CLIP filter 用哪个 checkpoint、阈值多少**</mark>
- <mark class="hl-trick">**Entropy 的定义与阈值**</mark>
- <mark class="hl-trick">**Synthetic 数据怎么生成、由什么模型生成、如何 QC**</mark>
- <mark class="hl-key">**$512 / 1024 / 2048$ 的真实 sampling probability**</mark>
- S4 的 <mark class="hl-trick">**Image Quality / Aesthetic / Compression 模型具体是什么**</mark>
- S6 Distribution Filter <mark class="hl-trick">**如何判定 “imbalance”**</mark>
- 不同 category 在 SFT 中的<mark class="hl-trick">**最终配比**</mark>
- <mark class="hl-key">**Synthetic Data 在 S4–S6 是否保留**</mark>（见本节开头的歧义提示）

### 2.4 Closed-loop Data Flywheel System ★★

<mark class="hl-trick">**这是全篇对我们现有 Recipe 最有增量的一节**</mark>，也是四篇报告里<mark class="hl-key">**唯一把「坏案例归因」写成显式路由表**</mark>的一节。

<mark class="hl-trick">**它不是简单说「收集 bad case，再加回训练集」，而是先做一件更重要的事情：**</mark>

$$
\boxed{
\text{对 bad case 进行错误归因}
\;\rightarrow\;
\text{把不同问题路由到不同优化路径}
}
$$

#### ① 三阶段闭环总览

<mark class="hl-key">**论文把整个 Flywheel 明确分成三个阶段**</mark>：

$$
\boxed{
\text{Signal Collection}
\rightarrow
\text{Case Routing \& Targeted Optimization}
\rightarrow
\text{Model Update}
}
$$

论文 Fig. 7 画得很清楚：<mark class="hl-trick">**当前 checkpoint 先经过 Model Evaluation、User Feedback、Bad Case Mining 收集失败案例，然后不是统一处理，而是分别进入 RL Track、Pre-training Track、PE Track**</mark>，最后形成新模型或新 Prompt Enhancer，进入下一轮评测。

![Qwen-Image-2.0 Fig.7：错误归因驱动的闭环 Data Flywheel。**顶部紫色框 `1. Signal Collection` 内是三个带箭头串联的节点**：`Model Evaluation`（柱状图图标）→ `User Feedback`（对话气泡图标）→ `Bad Case Mining`（放大镜加号图标），**箭头单向串联**。**中部大框 `2. Case Routing & Targeted Optimization` 内并列三个轨道**：左 `RL Track`（紫色标签）下挂 `RL Cases` 框，斜体说明 `current case fail because reinforcement learning is insufficient`，向下箭头指向 `Reward Policy Adjustment`（盾牌加号图标）；中间 `Pre-training Track` 下挂 `Pre-training Cases` 框，斜体说明 `current case fail because the model never saw this kind of data during pre-training`，向下依次串四个窄框 `Vector Retrieval Engine`（数据库放大镜）、`Data Augmentation`（双叠矩形）、`Human Review & Filtering`（人形图标）、`Curated Dataset`（数据库）；右 `PE Track` 下挂 `PE Cases` 框，斜体说明 `current case fail because prompt engineering is insufficient`，向下箭头指向 `Optimize Prompt Enhancer`（文档放大镜）。**底部 `3. Model Update` 框内只有一个深蓝 `Model Training` 节点**（循环箭头图标）。**最左侧一条竖直回流箭头标注 `Next Checkpoint`**，从底部 Model Training 绕回顶部 Model Evaluation，闭合成环。](/qwen2-fig7-data-flywheel.png)

<mark class="hl-key">**图本身给出了三条正文没有强调的细节：**</mark>

- <mark class="hl-trick">**Stage 1 的三个节点之间有单向箭头，是串联而非并列**</mark>（见下文 ② 的修正）
- <mark class="hl-trick">**Stage 3 只有一个 `Model Training` 节点**</mark>——<mark class="hl-key">**三条轨道不各自训练，而是把产物聚合后统一走一次训练**</mark>
- <mark class="hl-trick">**三条轨道的步骤数差异极大**</mark>：Pre-training Track 有 <mark class="hl-key">**4 步**</mark>（Vector Retrieval → Data Augmentation → Human Review → Curated Dataset），而 <mark class="hl-trick">**RL Track 与 PE Track 各只有 1 步**</mark>。<mark class="hl-key">**图示上唯一的强制人工卡点 `Human Review & Filtering` 只存在于 Pre-training Track 这条线上**</mark>

<mark class="hl-trick">**三条轨道在图中的节点命名也值得记**</mark>：<mark class="hl-key">**`RL Cases` / `Pre-training Cases` / `PE Cases`**</mark>——即路由的<mark class="hl-trick">**不是「问题」，而是「case 集合」**</mark>，同一个 case 只会被送进其中一条轨道。

<mark class="hl-key">**闭环的本质是论文这句概括**</mark>：

$$
\boxed{
\text{failure discovery}
\;\rightarrow\;
\text{targeted remediation}
\;\rightarrow\;
\text{model update}
}
$$

论文称之为 <mark class="hl-trick">**`a self-reinforcing optimization loop`**</mark>——自我强化的优化循环。

#### ② Stage 1：Multi-source Signal Collection

<mark class="hl-key">**Qwen 不只看固定 benchmark，而是同时收集三类信号**</mark>：

$$
\boxed{
\text{Standardized Model Evaluation}
+
\text{Targeted Bad-case Mining}
+
\text{Real-world User Feedback}
}
$$

<mark class="hl-key">**论文明确提到还会收集模型内部 self-evaluated cases**</mark>——即<mark class="hl-trick">**一部分失败样本并不一定来自线上用户，而是可以通过自动化测试自己挖出来**</mark>。

<mark class="hl-key">**这一点很重要，因为工业模型不可能只靠 benchmark 驱动。**</mark>Benchmark 能告诉你平均能力，但<mark class="hl-trick">**真实线上 bad case 往往会暴露完全不同的问题**</mark>，例如某种特殊文字布局、长尾物体、人物身份保持、复杂多图组合等等。

::: warning Fig. 7 的图示与 caption 在信号采集顺序上不一致
<mark class="hl-key">**Fig. 7 的图示与 Fig. 7 的 caption 文字在顺序上不一致：**</mark>

| 来源 | 顺序 |
| :--- | :--- |
| <mark class="hl-trick">**图内箭头**</mark> | Model Evaluation → User Feedback → Bad Case Mining（<mark class="hl-key">**单向串联**</mark>） |
| <mark class="hl-trick">**caption 文字**</mark> | `standardized model evaluation, targeted bad-case mining, and user feedback`（<mark class="hl-trick">**bad-case mining 在 user feedback 之前**</mark>） |

<mark class="hl-trick">**两种读法含义不同：图示是「采集→补充→深挖」的递进流水线，caption 更像三类并列信号源。**</mark><mark class="hl-key">**论文没有解释，因此不确定是「先收反馈再定向挖掘」这一串行流程，还是三种并列来源的图示简写。</mark>

<mark class="hl-key">**实务上按并行理解更稳妥**</mark>——<mark class="hl-trick">**因为 caption 的措辞是 `from diverse sources`（来自多种来源），而「多种来源」通常是并行采集的意思**</mark>。
:::

#### ③ Stage 2：Case Routing —— 本节真正的核心

<mark class="hl-key">**论文的第一句就否定了「统一处理」**</mark>：<mark class="hl-trick">**`The collected failure cases are not processed in a uniform manner.`**</mark>而是<mark class="hl-key">**依据一套 error attribution mechanism 自动路由到三条不同的优化轨道**</mark>。

<mark class="hl-trick">**这是全节最重要的判断：Qwen 明确没有把 RL 当成默认答案。**</mark>

##### 三条轨道的判定条件与动作

| 轨道 | 论文的判定条件 | 采取的动作 |
| :--- | :--- | :--- |
| <mark class="hl-key">**RL track**</mark> | <mark class="hl-trick">**alignment / policy 问题，源于 `insufficient reinforcement learning`**</mark> | <mark class="hl-key">**automated reward policy adjustment**</mark> |
| <mark class="hl-trick">**Pre-training track**</mark> | <mark class="hl-key">**`missing knowledge`，即预训练时对相似数据暴露不足**</mark> | <mark class="hl-trick">**Vector Retrieval Engine + Data Augmentation → Human Review → Curated Dataset**</mark> |
| <mark class="hl-key">**PE track**</mark> | <mark class="hl-trick">**模型已具备所需能力，但 `inaccurate instruction understanding` 或 `suboptimal prompt formulation`**</mark> | <mark class="hl-key">**通过优化后的 prompt enhancer 自动改进输入**</mark> |

##### RL track：有能力，但偏好没对齐

<mark class="hl-key">**进入 RL track 的前提是「模型其实有这个能力」**</mark>，问题出在<mark class="hl-trick">**输出行为不符合偏好、奖励信号不够好**</mark>。论文的动作被明确限定为 <mark class="hl-key">**`automated reward policy adjustment`**</mark>——<mark class="hl-trick">**只调 reward policy，而不是重训模型或改数据**</mark>。

<mark class="hl-trick">**这个例子属于帮助理解（论文未给具体 case）**：</mark>模型本来会生成人脸，但皮肤质感不够自然；或者会执行 Editing，但总喜欢把没要求修改的区域一起改掉。<mark class="hl-key">**这类问题不是「模型没见过」，而是已有能力的输出偏好没对齐好**</mark>，所以适合 RL 而非补数据。

##### Pre-training track：根本没见过，走 Vector Retrieval Engine

<mark class="hl-trick">**这条路径是本节最有意思的设计。**</mark>论文说系统会调用 <mark class="hl-key">**Vector Retrieval Engine**</mark>，而它有<mark class="hl-trick">**两个明确的目的（论文用 `two objectives` 逐条列出）**</mark>：

$$
\boxed{
\begin{aligned}
\text{①} &\quad \text{诊断失败是否由特定数据类别的稀缺（scarcity）造成}\\
\text{②} &\quad \text{检索并泛化多样化的生成 prompt 与编辑指令-图像对}
\end{aligned}
}
$$

<mark class="hl-key">**② 里论文列举的检索/泛化对象是有分类的**</mark>：

$$
\boxed{
\text{diverse text prompts for image generation}
\quad+\quad
\text{comprehensive instruction-image pairs for image editing}
}
$$

<mark class="hl-trick">**其中 instruction-image pairs 论文明确说明包含 `editing prompts` 与 `their corresponding base images`**</mark>——即多图编辑所需的 base image 也要一并检索出来。

<mark class="hl-key">**因此这里实际形成的链条是**</mark>：

$$
\boxed{
\text{Bad Case}
\rightarrow
\text{Vector Retrieval 诊断数据稀缺}
\rightarrow
\text{检索 / 泛化更多 prompt 与样本}
\rightarrow
\boxed{\text{Data Augmentation}}
\rightarrow
\boxed{\text{Human Review \& Filtering}}
\rightarrow
\boxed{\text{Curated Dataset}}
\rightarrow
\text{补上 knowledge gap}
}
$$

<mark class="hl-key">**这条路径直接回答了我们之前反复讨论的一个问题：模型某个能力不行，到底该不该直接上 RL？**</mark>Qwen 的答案非常明确：<mark class="hl-key">**不一定**</mark>。如果根因是 <mark class="hl-trick">**Data Missing**</mark>，那 RL 往往治标不治本，应先补 Pretrain / continual-pretrain 数据。

<mark class="hl-trick">**比如模型不会生成一种极冷门的工业机械**</mark>——如果训练集里这种机械本来几乎没有，<mark class="hl-key">**那么再复杂的 reward 也很难把「根本没有的知识」凭空 RL 出来**</mark>。

由此得到一条可直接写进工业字典的规则：

$$
\boxed{
\text{Knowledge Missing} \;\Rightarrow\; \text{Data / Pretrain}
}
$$

$$
\text{Knowledge Missing} \;\not\Rightarrow\; \text{RL}
$$

::: warning 措辞需要精确：不是「从已有数据中检索」
<mark class="hl-trick">**论文原文是 `retrieve and generalize diverse text prompts ... as well as comprehensive instruction-image pairs`**</mark>，用的是 <mark class="hl-key">**`retrieve and generalize`**</mark>——<mark class="hl-trick">**而非单纯「从已有数据里检索」**</mark>。`generalize` 意味着<mark class="hl-key">**会泛化出新的 prompt 表述，不只是复用现成样本**</mark>。

<mark class="hl-trick">**这一点直接决定了 §2.3 那条「Synthetic Data 未公开生成器」的空白**</mark>：<mark class="hl-key">**这里很可能就是 Qwen 大规模合成数据的一个主要入口**</mark>，<mark class="hl-trick">**但论文没有把两者显式关联，也没有说明 generalize 用的是什么模型与 prompt 策略**</mark>。<mark class="hl-key">**两节之间存在一个未被论文打通的推断缺口，不要当成已证实的结论。**</mark>
:::

##### PE track：模型会，只是用户表达不够好

<mark class="hl-trick">**这是三条轨道里最反直觉、也最省钱的一条。**</mark>如果模型<mark class="hl-key">**其实已经有能力**</mark>，但失败是因为 <mark class="hl-trick">**instruction understanding 不够准确**</mark>，或者 <mark class="hl-trick">**用户 prompt 本身太短 / 太模糊 / specification 不充分**</mark>，那么<mark class="hl-key">**模型本身根本不用重新训练**</mark>，而是把 case 送到 Prompt Enhancer，通过<mark class="hl-trick">**优化输入 prompt**</mark> 去解决。

<mark class="hl-key">**论文的措辞是 `the system automatically refines the input through an optimized prompt enhancer`**</mark>——<mark class="hl-trick">**改的是输入，不是模型**</mark>。这条与 §3.3 里「PE 从 Qwen3.5-9B 初始化、且编辑场景刻意让 PE 输出简洁」的决策形成呼应：<mark class="hl-key">**Flywheel 的第三条轨道直接复用了 §3.3 那个 Prompt Enhancer**</mark>。

#### ④ 三条轨道汇总成一张诊断表

<mark class="hl-key">**Qwen 的 Flywheel 真正形成的是一个非常实用的诊断逻辑**</mark>：

$$
\boxed{
\text{这个 bad case 为什么失败？}
}
$$

<mark class="hl-key">**以及由此得到的完整路由表**</mark>：

| Bad case 根因 | 应优先做什么 | 对应 Qwen 轨道 |
| :--- | :--- | :--- |
| <mark class="hl-trick">模型从没见过类似概念 / 知识</mark> | <mark class="hl-trick">补 Pretrain / Continual-pretrain 数据</mark> | Pre-training track |
| <mark class="hl-trick">模型会，但输出偏好或执行方式不对</mark> | <mark class="hl-key">RL / Reward 调整</mark> | RL track |
| <mark class="hl-trick">模型有能力，但用户 prompt 表达不足</mark> | <mark class="hl-key">Prompt Enhancer</mark> | PE track |
| <mark class="hl-trick">不确定原因</mark> | <mark class="hl-trick">先做 retrieval / similar-case analysis 再决定</mark> | <mark class="hl-trick">论文未覆盖</mark> |

<mark class="hl-key">**最后一行值得注意：论文只写了「归因到三条轨道」，并没有讨论「归因不确定」的情况。**</mark>而工业上<mark class="hl-trick">**归因本身出错（把 data missing 误判成 alignment issue）恰恰是最贵的失败**</mark>——它会导致在错误的轨道上投入预算。<mark class="hl-key">**这个鲁棒性问题论文没有触及。**</mark>

<mark class="hl-key">**这张表背后的工程思想**</mark>：

$$
\boxed{
\text{不要拿一种训练方法解决所有问题。}
}
$$

<mark class="hl-trick">**这是学完前几篇报告后一直在隐约形成的判断，而 Qwen-Image-2.0 把它系统化了。**</mark>→ 可直接补入 [Playbook §1 诊断字典](./training-playbook.md) 与 [Playbook §7 工业闭环](./training-playbook.md)。

#### ⑤ Stage 3：Model Update & Closed Loop

<mark class="hl-key">**三条轨道各自的产出被聚合起来**</mark>——论文说是 <mark class="hl-trick">**`new datasets, and parameter updates`**</mark>——然后系统<mark class="hl-key">**自动启动下一轮训练**</mark>：

$$
\boxed{
\text{聚合三条轨道的 datasets + parameter updates}
\rightarrow
\text{next training round}
\rightarrow
\text{new checkpoint}
\rightarrow
\boxed{\text{feeded back to Stage 1}}
}
$$

<mark class="hl-trick">**注意闭环回的是 Stage 1 而非终点——评测与反馈会持续产生新的 bad case，所以这是一个永不停歇的循环，而非一次性修补。**</mark>

#### ⑥ 人工干预被刻意压缩

<mark class="hl-key">**论文强调这套系统 `highly automated`，且把人工干预限制在关键位置**</mark>：

$$
\boxed{
\text{limiting manual intervention to critical data filtering}
}
$$

<mark class="hl-trick">**在 Pre-training track 里，人工环节被论文明确称为 `the only manual intervention in the pipeline, namely necessary human review & filtering`**</mark>——即<mark class="hl-key">**数据进模型之前必须有人工过滤，这是全流程唯一的强制人工卡点**</mark>。其余（评测、归因、检索、prompt 改写、训练启动）论文都宣称是自动的。

<mark class="hl-key">**论文给出的理由有两条**</mark>：

1. <mark class="hl-trick">**`substantially reduces engineering overhead while preserving data reliability`**</mark> —— 降低工程开销的同时保住数据可靠性
2. <mark class="hl-trick">**error attribution 实现 `targeted and resource-efficient optimization`**</mark> —— 归因让优化更定向、更省资源
3. <mark class="hl-trick">**vector retrieval engine `continuously enriches the diversity of training data`**</mark> —— 检索引擎持续提升训练数据多样性

<mark class="hl-key">**第三条值得单独留意：论文把「提升数据多样性」这件事主要归功于 Vector Retrieval Engine，而不是归功于任何生成模型**</mark>。<mark class="hl-trick">**这与 §2.3 中 Synthetic Data 的生成器未知形成呼应，两处都指向「检索+泛化」而非「大规模生成」是 Qwen 的主要数据扩增手段。**</mark>

#### ⑦ §2.4 最该进 Recipe 的一条

<mark class="hl-key">**这一节最该进 [Playbook](./training-playbook.md) 的不是「Data Flywheel」这个名字，而是整套错误归因机制。**</mark>以后模型出现 bad case，<mark class="hl-trick">**第一问不应该是「我要不要再做一轮 SFT / RL？」**</mark>而应该先问：

$$
\boxed{
\text{这是 Knowledge Problem、Alignment Problem，还是 Prompt Problem？}
}
$$

<mark class="hl-key">**这一点对后训练尤其重要——很多团队最容易犯的错误就是：看见某个能力不行，就继续加 RL。实际上有些问题根本不是 RL 问题。**</mark>

#### ⑧ 未公开细节汇总

<mark class="hl-trick">**本节把归因框架公开得很清楚，但让它能真正跑起来的工程细节全部缺失：**</mark>

- <mark class="hl-trick">**error attribution classifier 是什么模型**</mark>（论文只说 `an error attribution mechanism`，未指定是 VLM、分类器还是规则）
- <mark class="hl-trick">**如何自动判断「RL insufficient」还是「pretrain data missing」**</mark>——这是整条路由链最关键的一步，却是黑盒
- <mark class="hl-trick">**Vector Retrieval 用什么 embedding 模型、索引什么语料**</mark>
- <mark class="hl-trick">**similarity threshold 是多少**</mark>
- <mark class="hl-trick">**`generalize` 具体如何做**</mark>（用什么模型、什么 prompt 策略、生成多少）
- <mark class="hl-trick">**Data Augmentation 的具体手法**</mark>（§2.3 的 Synthetic generator 同样未知，两者是否同一套未说明）
- <mark class="hl-trick">**线上 user feedback 如何去噪**</mark>（真实用户反馈天然含噪声与恶意输入，论文未提）
- <mark class="hl-trick">**每个 track 的触发比例与预算分配**</mark>
- <mark class="hl-trick">**「归因不确定」时的兜底策略**</mark>——论文只定义了三条轨道，没有第四种情况

### 2.5 Data 部分总结：真正给 Recipe 增加了什么 ★

#### ① 整体结构：三层逻辑

<mark class="hl-key">**把 §2.1–§2.4 放在一起看，Qwen-Image-2.0 的数据部分不是简单「数据更多、过滤更严格」，而是把工业图像基模的数据建设拆成了三层递进逻辑**</mark>：

$$
\boxed{
\begin{aligned}
\text{第一层} &: \text{先按能力定义数据} \\
\text{第二层} &: \text{按训练阶段改变数据分布} \\
\text{第三层} &: \text{用 bad case 反向驱动下一轮数据和后训练}
\end{aligned}
}
$$

<mark class="hl-trick">**注意「按任务设计监督」这一层在论文里是被写进 §2.2 的，但在方法论上它与「按能力定义数据」同源，因此下面并列为四层来表述。**</mark>

#### ② 第一层：Capability-driven Data Design

<mark class="hl-key">**Qwen 不是按「这个数据来自哪个网站」组织数据，而是按最终模型要获得什么能力来建 taxonomy。**</mark>T2I 里不仅有普通摄影，还显式覆盖 Design、Poster、Slide、UI、Synthetic、Chart、Text；TI2I 里又拆出 single-image、multi-image、identity consistency、typography、reasoning-based editing、composition-based editing（→ §2.1③ 的 Fig. 5）。

<mark class="hl-key">**这意味着数据工程的起点不应该是「我有什么数据」，而应该是**</mark>：

$$
\boxed{
\text{我希望模型会什么}
\;\rightarrow\;
\text{对应的数据域是什么}
}
$$

<mark class="hl-trick">**这个思想是现有 Recipe 里应该正式加入的一条，而不是只停留在「做 concept balancing」层面**</mark>——<mark class="hl-key">**concept balancing 解决的是「已选定的域之间配比」，而 capability-driven design 解决的是「域本身该怎么划」**</mark>，是前一步。

#### ③ 第二层：Task-specific Supervision Representation

<mark class="hl-key">**四类 Caption 说明 caption 不能只理解成「长描述」与「短描述」的区别**</mark>——不同任务本身需要不同的监督形式（→ §2.2）。

<mark class="hl-key">**因此 Recipe 可以从**</mark>

$$
\text{Image} \rightarrow \text{Recaption}
$$

<mark class="hl-key">**升级为**</mark>：

$$
\boxed{
\text{Image Type / Capability}
\;\rightarrow\;
\text{Appropriate Supervision Representation}
}
$$

<mark class="hl-key">**这是本篇真正的新东西**</mark>——<mark class="hl-trick">**「recaption」隐含了「所有图用同一种 VLM 走一遍」的假设，而 Qwen 明确否定了这个假设**</mark>。

#### ④ 第三层：Stage-aware Data Curriculum

<mark class="hl-key">**六阶段 pipeline 说明不能先做一个「最终训练集」然后从头用到尾**</mark>：随训练阶段同时改变 resolution、数据类型、过滤强度和 distribution（→ §2.3）。

<mark class="hl-key">**因此 Recipe 里「Pretrain 前期追 coverage、后期追 quality」这句可以写得更具体**</mark>：

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

<mark class="hl-trick">**而不是只改 resolution 这一个维度**</mark>——<mark class="hl-key">**这正是 §2.3 最容易被简化记忆的地方**</mark>。

#### ⑤ 第四层：Closed-loop Error Attribution

<mark class="hl-key">**这是本篇最值得加进工业 Playbook 的部分**</mark>——bad case 不能统一塞回训练，而要先判断属于什么问题（→ §2.4）：

$$
\boxed{
\text{Missing Knowledge} \rightarrow \text{Pre-training Track}
}
$$

$$
\boxed{
\text{Alignment / Policy Problem} \rightarrow \text{RL Track}
}
$$

$$
\boxed{
\text{Prompt Specification Problem} \rightarrow \text{Prompt Enhancer Track}
}
$$

<mark class="hl-key">**这把我们之前比较模糊的经验变成了一套可执行逻辑**</mark>：

> <mark class="hl-trick">**模型不会某个东西，不代表一定需要 RL；有时是没见过，有时是偏好没对齐，有时根本只是 prompt 写得不好。**</mark>

<mark class="hl-key">**因此以后真的做工业基模时，第一步诊断可以固定成**</mark>：

$$
\boxed{
\text{Bad Case}
\rightarrow
\text{Root Cause}
\rightarrow
\boxed{\text{Minimal Intervention}}
}
$$

$$
\text{Bad Case} \;\not\rightarrow\; \text{Train More}
$$

<mark class="hl-key">**`Minimal Intervention` 这个词是本节的收敛点：三条轨道的本质是按最小代价选择干预层级**</mark>——<mark class="hl-trick">**改 prompt 比改 reward 便宜，补数据比改 reward 更治本，而三者都不如先归因便宜**</mark>。

#### ⑥ 四条增量 vs 只是再次印证

| Qwen-Image-2.0 增量 | 对我们 Recipe 的意义 | 所在小节 |
| :--- | :--- | :-: |
| <mark class="hl-key">**Capability-driven data taxonomy**</mark> | <mark class="hl-trick">先定义能力，再反向定义数据域</mark> | §2.1③ |
| <mark class="hl-key">**Task-specific caption / supervision**</mark> | <mark class="hl-trick">不同视觉任务不应共用一种 caption representation</mark> | §2.2⑥ |
| <mark class="hl-key">**Stage-aware data curriculum**</mark> | <mark class="hl-trick">质量、类型、比例、分辨率随阶段一起变化</mark> | §2.3① |
| <mark class="hl-key">**Error-attribution Data Flywheel**</mark> | <mark class="hl-trick">bad case 先归因，再决定 Pretrain / RL / PE</mark> | §2.4④ |

<mark class="hl-trick">**以下几条不是全新的，只是再次印证前面已得到的共识**</mark>（Z-Image、Mage-Flow 等工作中已有相似思想，Qwen 的价值在于组织得更系统）：

- <mark class="hl-trick">**高分辨率数据不能只看像素大小，还要做质量与压缩筛选**</mark>（→ §2.3⑥，Z-Image 也有质量分级）
- <mark class="hl-trick">**SFT 应该使用更严格的高质量数据**</mark>（→ §2.3⑧）
- <mark class="hl-trick">**Editing 要尽早进入统一训练**</mark>（→ §2.3④）
- <mark class="hl-trick">**Synthetic Data 应针对数据分布缺口去补**</mark>（→ §2.3⑤）

<mark class="hl-key">**把「新增」与「印证」分开标注，本身是有价值的**</mark>——<mark class="hl-trick">**因为真正该写进 Playbook 的只有那四条；另外四条虽然对，但写进去只是增加篇幅而不增加信息**</mark>。

#### ⑦ 必须保留的判断：这是设计原则，不是可复现 recipe

<mark class="hl-trick">**技术报告没公开的内容要继续保留在附录里**</mark>（→ [附录 A](#a-本篇未公开的细节持续累积)）。Qwen 虽然把数据框架讲得很完整，但真正生产级最关键的参数仍然没有公开：

$$
\text{各类数据总量}\quad
\text{各类别 sampling ratio}\quad
\text{filter model 与阈值}\quad
\text{Synthetic pipeline}
$$

$$
\text{四类 Caption 的生成模型与 prompt}\quad
\text{Structured schema}\quad
\text{error attribution 模型}\quad
\text{retrieval embedding}\quad
\text{SFT 最终 distribution 组成}
$$

<mark class="hl-key">**因此这一篇给我们的主要是数据系统的设计原则和组织方式，不是一份可以逐参数复现的数据 recipe。**</mark>

$$
\boxed{
\text{可复现} \;\ne\; \text{可迁移}
}
$$

<mark class="hl-trick">**这句话需要说清楚**</mark>：§2.3 的阈值、§2.4 的归因模型都拿不到，所以照抄 Qwen 的 pipeline 是做不到的；<mark class="hl-key">**但「阶段与数据策略必须同步演进」「bad case 必须先归因」这两条判断，换任何数据集都成立**</mark>。

#### ⑧ Data 部分最终压缩成一句话

$$
\boxed{
\text{先按能力设计数据}
\rightarrow
\text{按任务设计监督}
\rightarrow
\text{按阶段改变分布}
\rightarrow
\text{按失败原因继续补数据或做后训练}
}
$$

<mark class="hl-key">**这四步值得正式写进 [Image Foundation Model Training Playbook](./training-playbook.md)。**</mark>

## 3. Architecture（§3.1–3.2 待填）

### 3.1 Variational AutoEncoder

### 3.2 Multi-modal Diffusion Transformer

<mark class="hl-trick">**已核实的关键结构（后续填写时展开）**</mark>：Qwen3-VL 同时编码视觉与文本输入得到 $h_x, h_y$；其中 $h_x$ <mark class="hl-key">**被 VAE latent $E_x$ 替换**</mark>，再拼接

$$
h=\mathrm{Concat}\big(E_x,\ h_y\big)
$$

送入 Qwen-Image-2.0 block。架构为 MMDiT（引 Esser et al. 2024），text 与 image token 在**共享 backbone** 内处理。

### 3.3 Prompt Enhancer ★★★★

<mark class="hl-trick">**这一节是全篇对工业落地最有增量的工具之一**</mark>。<mark class="hl-key">**它不是一个「把用户 prompt 改写得更长」的 LLM，而是一个专门训练、并且最终用图像生成质量做 RL 对齐的 prompt 重写模型**</mark>。

#### ① 动机：复杂任务的瓶颈在 specification，不在模型容量

<mark class="hl-trick">**论文的判断是：对于复杂图像生成任务——**</mark>

$$
\text{infographics}
\qquad
\text{posters}
\qquad
\text{typographic layouts}
\qquad
\text{multi-panel storyboards}
\qquad
\text{data visualizations}
$$

<mark class="hl-trick">**生成质量同时取决于两件事：**</mark>模型的视觉合成能力，<mark class="hl-key">**以及 prompt 对以下四类信息的表达能力**</mark>：

$$
\boxed{
\text{layout}
+
\text{object relations}
+
\text{visual hierarchy}
+
\text{compositional intent}
}
$$

<mark class="hl-key">**但真实用户 prompt 在「granularity 与 explicitness」上差异极大**</mark>——论文称之为 <mark class="hl-trick">**`a key bottleneck for high-complexity visual creation`**</mark>。因此 PE 的定位是一个独立的重写模块：

$$
\boxed{
\text{PE} : \text{user queries of varying specificity}
\;\rightarrow\;
\text{structured, detail-rich prompts}
}
$$

<mark class="hl-trick">**论文用的动词是 `converts`，目标是让下游生成器 `better capture the intended visual design across diverse tasks`**</mark>——<mark class="hl-key">**注意落脚点在「下游生成器能否消费」，不是「文本本身好不好」**</mark>，这个区别在 ⑥ 会变成 RL 的动机。

#### ② 数据构造：逆向退化流水线（本节最值得学的一招）

<mark class="hl-key">**作者不是去收集大量「短 prompt → 长 prompt」的人工改写对，而是从一条已经非常详细的 annotation 出发，反向制造真实用户可能输入的简短 prompt。**</mark>论文把这条流水线称为 <mark class="hl-trick">**`a reverse-engineering pipeline that atomically degrades fine-grained annotations`**</mark>。

完整链条：

$$
\boxed{
P_{\rm fine}
\;\xrightarrow{\ \text{LLM 分类}\ }\;
\text{category}
\;\xrightarrow{\ \text{采样策略}\ }\;
S'\subseteq S
\;\xrightarrow{\ \text{施加}\ }\;
P_{\rm short}
}
$$

##### 第一步：用 LLM 做 task-aware 四分类

给定一条详细标注 $P_{\rm fine}$，先用一个 LLM 把它分到四类生成任务之一：

| 类别 | 论文原文 | 这一类的退化重点 |
| :--- | :--- | :--- |
| <mark class="hl-trick">General</mark> | General | 场景、光照、材质、构图 |
| <mark class="hl-trick">Portrait</mark> | Portrait | 人像特征、肤质、光位 |
| <mark class="hl-key">Text</mark> | Text | 文字内容与版面 |
| <mark class="hl-key">Complex Text</mark> | Complex Text | 长文本、多分栏、复杂排版 |

<mark class="hl-key">**论文明确说明了这个分类的作用**</mark>：<mark class="hl-trick">**`This task-aware classification ensures that the subsequent degradation process is semantically grounded and adapted to the characteristics of each prompt type`**</mark>——<mark class="hl-key">**即先分型再退化，退化才不会退到无关维度上**</mark>。

::: warning 这四类与 §2.2 的四类 caption 不是同一套 taxonomy
<mark class="hl-trick">**这是本篇最容易混淆的一点，论文没有解释两套分类的关系：**</mark>

| | 分类 | 决定什么 |
| :--- | :--- | :--- |
| <mark class="hl-trick">**§2.2 Data Annotation**</mark> | General / **Text** / Knowledge / Structured | <mark class="hl-key">**监督表示形式**</mark>（写什么形状的 caption） |
| <mark class="hl-trick">**§3.3 Prompt Enhancer**</mark> | General / Portrait / **Text** / **Complex Text** | <mark class="hl-trick">**施加哪些退化操作**</mark> |

<mark class="hl-trick">**只有 `General` 与 `Text` 两项重合，`Knowledge` / `Structured` 在 PE 侧没有对应，`Portrait` / `Complex Text` 在 caption 侧没有对应。**</mark><mark class="hl-key">**引用时不要把这两套四分类混为一谈。**</mark>
:::

##### 第二步：从策略池采样并施加退化

<mark class="hl-trick">**根据类别，从预定义的 strategy pool 中采出一组适用的退化策略**</mark>，记为 $S$。论文明确列出的三类策略：

$$
\boxed{
\begin{aligned}
&\text{Stylistic Simplification} && \text{（文体简化，去掉专业术语）}\\
&\text{Colloquialization} && \text{（口语化）}\\
&\text{Removal / Underspecification of Visual Details} && \text{（删除或弱化 } \textbf{lighting, texture, layout, background}\text{）}
\end{aligned}
}
$$

##### 第三步：把 stochasticity 塞进去，让分布贴近长尾

<mark class="hl-key">**第三步是这一招真正的关键，论文措辞是 `To approximate the long-tail distribution of real-world user inputs, we introduce stochasticity into the degradation process`。**</mark>

$$
\boxed{
\text{从 } S \text{ 中按预定义概率分布采样一个子集，然后施加，得到 } P_{\rm short}
}
$$

<mark class="hl-key">**而且采样比例本身是可调的**：论文说通过调整 sampling proportions，这条流水线能产出</mark>

$$
\boxed{
\text{varying difficulty}
\qquad
\text{varying ambiguity}
\qquad
\text{varying information density}
}
$$

<mark class="hl-trick">**所以得到的不是一个「短 prompt 分布」，而是一个<mark class="hl-key">难度连续谱</mark>——从轻度口语化到几乎不含视觉信息。**</mark>

#### ③ Inverse Reasoning CoT：退化操作的逆过程天然就是推理链

<mark class="hl-trick">**这个设计最巧的地方在于：因为系统自己知道「从 $P_{\rm fine}$ 到 $P_{\rm short}$ 时到底删掉了什么」，所以这些退化操作反过来天然构成一条 inverse reasoning chain。**</mark>论文的论证是：

$$
\boxed{
\forall s \in S:\quad s \text{ 移除了信息} \;\Longrightarrow\; s^{-1} \text{ 定义一条 prompt 恢复轨迹}
}
$$

原文措辞：<mark class="hl-trick">**`its reverse defines a principled trajectory for prompt recovery and enrichment`**</mark>。论文称其为 <mark class="hl-key">**`a Chain-of-thought (CoT) for prompt enhancement`**</mark>。

最终训练样本是一个三元组：

$$
\boxed{
\left(P_{\rm short},\ \mathrm{CoT},\ P_{\rm fine}\right)
}
$$

<mark class="hl-key">**论文明确说这个三元组让模型同时学到两件事**</mark>：增强后的 prompt 本身，**以及底层的 intent-expansion 过程**——原文举例是 <mark class="hl-trick">**`inferring lighting, material, spatial, and stylistic cues from the remaining attributes`**</mark>。

<mark class="hl-trick">**所以 PE 不是学「把短 prompt 抄成长 prompt」，而是学一条从模糊需求逐步恢复视觉意图的 reasoning trajectory。**</mark>

#### ④ T2I 与 Editing 的构造方式不同，这是本节最重要的判断

<mark class="hl-key">**逆向退化流水线只用于 T2I。论文对 Editing 给的是完全不同的做法**</mark>：

$$
\boxed{
\begin{aligned}
\text{T2I} &: \text{stochastic degradation（从 } P_{\rm fine} \text{ 退化到 } P_{\rm short}\text{）}\\
\text{TI2I} &: \text{MLLM summarize（把 long-form annotation 总结为 concise editing prompt）}
\end{aligned}
}
$$

<mark class="hl-trick">**论文给出的理由是**</mark> <mark class="hl-key">**`where the input image already provides rich visual context`**</mark>，因此 Editing 侧 <mark class="hl-trick">**`avoiding unnecessary stochastic degradation`**</mark>。

<mark class="hl-key">**这背后是很本质的一个区别**</mark>：

$$
\boxed{
\begin{aligned}
\text{T2I 的 PE} &: \text{适度补充视觉细节} \\
\text{Editing 的 PE} &: \boxed{\text{Instruction Preservation}}
\end{aligned}
}
$$

<mark class="hl-trick">**因为用户说「把杯子变红」，你不能为了「增强 prompt」顺便把背景、光照、构图全部重新设计。**</mark><mark class="hl-key">**这是 Generation 与 Editing 在 Prompt Engineering 上最本质的一条区别。**</mark>

<mark class="hl-trick">**对照 [Playbook §3](./training-playbook.md) 的 catastrophic forgetting 讨论：Editing 侧的 PE 面临的是同一类风险——过强的条件改写会破坏原有条件。**</mark>

#### ⑤ PE Training：Qwen3.5-9B 初始化 + SFT → RL 两阶段

<mark class="hl-trick">**PE 模块初始化自 Qwen3.5-9B（引 Team, 2026），并作为 T2I 与 TI2I 统一的 prompt enhancement 模型训练**</mark>——论文用 `a unified prompt enhancement model for both image generation and image editing`。

$$
\boxed{
\text{SFT}
\;\rightarrow\;
\text{RL}
}
$$

##### Stage 1：SFT

<mark class="hl-trick">**SFT 用标准 next-token prediction objective，学的三项能力论文写得很明确**</mark>：

$$
\boxed{
\text{Intent Preservation}
+
\text{Scene Enrichment}
+
\text{Compositional Organization}
}
$$

<mark class="hl-key">**而 T2I 与 Editing 在 SFT 阶段的侧重不同**</mark>，论文用 `While...` 明确对比：

| | 侧重 |
| :--- | :--- |
| <mark class="hl-trick">Generation prompt</mark> | <mark class="hl-key">`require richer visual elaboration`</mark> |
| <mark class="hl-trick">Editing prompt</mark> | <mark class="hl-key">**`demand faithful instruction preservation and sensitivity to the existing visual context`**</mark> |

##### Stage 2：RL（这是 PE 真正的差异化之处）

<mark class="hl-key">**论文指出 SFT 的根本局限，措辞很直接**</mark>：

> <mark class="hl-trick">**`Since SFT relies on static textual references and cannot directly optimize downstream image quality`**</mark>

<mark class="hl-key">**也就是说：SFT 只能告诉 PE「你写得像不像 $P_{\rm fine}$」，却无法回答「这条增强后的 prompt 最后是不是真的让图生成得更好」。**</mark>因此第二阶段引入 <mark class="hl-key">**基于 GRPO 的 RL**</mark>。

<mark class="hl-trick">**注意这一处 GRPO 的引用是 `GRPO (Shao et al., 2024)`，即 DeepSeekMath 的原始 GRPO**</mark>——<mark class="hl-key">**与 §4.2 主模型 RLHF 引用的 Flow-GRPO / GRPO-Guard / DiffusionNFT 不是同一套，见 §4.2⑤**</mark>。

##### PE 的 RL 回路

$$
P_{\rm short}
\;\rightarrow\;
\text{Prompt Enhancer}
\;\rightarrow\;
P_{\rm enhanced}
\;\rightarrow\;
\boxed{\text{Frozen Image Generator}}
\;\rightarrow\;
I_{\rm generated}
\;\rightarrow\;
\text{Reward}
$$

<mark class="hl-key">**注意这里被冻结的是图像生成器，被更新的是 PE。**</mark>论文说 PE `generates candidate enhanced prompts, which are fed into a frozen image generator, and is optimized with rewards combining...`。

Reward 由三部分组成：

$$
\boxed{
R_{\rm PE} = R_{\rm visual\ consistency}^{\rm MLLM}
\;+\;
R_{\rm aesthetic}^{\rm MLLM}
\;+\;
R_{\rm textual\ constraint}^{\rm rule\text{-}based}
}
$$

<mark class="hl-trick">**注意论文的措辞是 `rewards combining`，它没有写出任何求和公式，也没有给三者权重**</mark>，<mark class="hl-key">**上面的加号是帮助理解的记法，不是论文公式**</mark>。<mark class="hl-trick">**完整 reward prompt 与权重均未公开。**</mark>

<mark class="hl-key">**这一节真正漂亮的地方在于：SFT 保证 PE「会改写、不会乱写」，RL 再让它学会「什么样的改写真的对下游生成有帮助」。**</mark>所以 PE 优化的是

$$
\boxed{
\text{Prompt Utility for Generator}
}
$$

<mark class="hl-trick">**而不是**</mark> $\text{Text Similarity}$ <mark class="hl-trick">**或** $\text{Text Richness}$。</mark><mark class="hl-key">**普通 LLM prompt rewriting 与它的差别正在这里——把「一个雨天的东京街道」扩写成一段华丽文字，对图像生成未必有任何帮助。**</mark>

#### ⑥ Figure 9：定性证据，以及它的三个局限

![Qwen-Image-2.0 Fig.9：Prompt Enhancer 定性对比（论文 caption 标注为 T2I）。两列布局，左列 `Original`、右列 `PE`，共 5 组案例，每组上方灰底横条是**原始 prompt 原文**。(1) `A massive waterfall formed by melting glaciers pours down from cliffs thousands of meters high, kicking up widespread mist and rainbows.` —— Original 是一张暖调、近距离的峡谷瀑布特写；PE 变成远景大全景，出现完整的雪山、彩虹横跨画幅、明显更冷更蓝的色调。(2) `A grand medieval castle stands atop a high mountain peak, surrounded by a rolling sea of clouds.` —— Original 是逆光下几乎剪影的远景小城堡；PE 变成近距离、明亮的城堡全景，可见石墙纹理、塔楼旗帜与云海层次。(3) `Paint the Mona Lisa as a Japanese ukiyo-e style geisha, keeping her original smile and pose unchanged.` —— **Original 列放的是真实的《蒙娜丽莎》**（因此这一行实际是编辑任务，尽管 caption 写 T2I）；PE 输出浮世绘风格艺伎，保留原画背景的山水与松树。(4) `A Chinese ink wash painting, with complete text of 《黄鹤楼》on the top left.` —— Original 侧栏文字是「黄鹤楼 / 黄鹤楼 / 云…」一类错字与重复字符；PE 输出竖排右起的《黄鹤楼》全诗（昔人已乘黄鹤去…烟波江上使人愁），字形与竖排版式均正确。(5) `A partially filled 4x4 sudoku grid with numbers 1 to 4 and three empty cells remaining.` —— Original 输出 `1 2 6 / 4 6 7 4 / 4 3 3 2 / 7 3 4`，数字重复且不构成合法数独、格子数也不对；PE 输出干净的黑框 4×4 网格，数字合法且恰好留下三个空格。](/qwen2-fig9-prompt-enhancer.png)

<mark class="hl-key">**五组案例里有两组（黄鹤楼、数独）直接印证了 §2.2 Text Caption 的价值判断**</mark>——<mark class="hl-trick">**PE 的收益主要发生在「需要精确遵循文字与结构约束」的任务上，而不是普遍的美观提升。**</mark>

##### 三个必须一起说清的局限

<mark class="hl-trick">**局限一：图里从不显示 PE 增强后的 prompt。**</mark>每组只有<mark class="hl-key">**原始 prompt 的原文**</mark>，右列只是「用它渲染出的结果」。<mark class="hl-key">**所以读者无法看出 PE 究竟补了哪些信息**</mark>——<mark class="hl-trick">**而这恰恰是逆向退化流水线最想展示的东西。**</mark>

<mark class="hl-trick">**局限二：caption 标注 `T2I results`，但第 (3) 行是编辑任务。**</mark>Original 列放的是真实《蒙娜丽莎》，<mark class="hl-key">**说明这一组是「给一张图按指令改」**</mark>。<mark class="hl-trick">**而 §3.3 的退化流水线本来只用于 T2I，所以这张图同时也没有为 Editing 侧的 MLLM summarize 设计提供任何视觉证据。**</mark>

<mark class="hl-trick">**局限三：五组案例全部是定性对比，论文未给任何定量数字。**</mark><mark class="hl-key">**没有 win rate、没有 CLIPScore、没有人工评分**</mark>——<mark class="hl-trick">**§3.3 声称 PE 改善了 `generation quality, prompt following, and reasoning performance` 三项，但三项都没有量化支撑。**</mark>

<mark class="hl-key">**一个容易被忽略但很有信息量的观察**</mark>：瀑布与城堡两组里，PE 改变的主要是<mark class="hl-trick">**构图与色调本身**</mark>（远景 vs 近景、冷调 vs 暖调），<mark class="hl-key">**不只是「多加了细节」**</mark>。<mark class="hl-trick">**这说明退化操作删掉的是 layout / background 这一整层信息，而 PE 的恢复是整体重写而非局部补充。**</mark>

#### ⑦ §3.3 最该进 Recipe 的一条

<mark class="hl-key">**§3.3 真正该加进 [Playbook](./training-playbook.md) 的不是「多加一个 LLM」，而是这条数据构造方法**</mark>：

$$
\boxed{
\text{High-quality detailed annotation}
\rightarrow
\text{controlled degradation}
\rightarrow
\text{realistic user query}
\rightarrow
\boxed{\text{inverse reasoning supervision}}
}
$$

<mark class="hl-trick">**以及与之配套的那个判断**</mark>：

$$
\boxed{
\text{生成失败} \;\not\Rightarrow\; \text{一定要改 Generator}
}
$$

<mark class="hl-key">**如果失败来自 specification 不充分，可以只优化中间这层接口**</mark>：

$$
\boxed{
\text{User Intent}
\;\rightarrow\;
\text{Generator-friendly Condition}
}
$$

<mark class="hl-trick">**这一条与 §2.4 的 PE Track 是同一件事的两个视角**</mark>：Flywheel 把它当成<mark class="hl-key">**一条不需要重训模型的修复路径**</mark>，§3.3 把它做成<mark class="hl-trick">**一个独立训练、独立 RL 的模块**</mark>。

#### ⑧ 未公开细节汇总

<mark class="hl-trick">**这一节把方法框架给得很完整，但 production recipe 全部缺失：**</mark>

- <mark class="hl-trick">**degradation strategy pool 的完整列表，以及每种策略的采样概率**</mark>
- <mark class="hl-trick">**$P_{\rm fine}$ 最初是怎么生成的**</mark>（用什么模型、什么 prompt）
- <mark class="hl-trick">**做四分类的那个 LLM 是什么**</mark>
- <mark class="hl-trick">**CoT 的具体格式**</mark>——是逐条列出被删掉的属性，还是自然语言推理段落
- <mark class="hl-trick">**SFT 数据规模、训练 step、LR、batch size**</mark>
- <mark class="hl-trick">**GRPO 的 rollout group size**</mark>
- <mark class="hl-trick">**三个 PE reward 的权重与完整 reward prompt**</mark>
- <mark class="hl-trick">**frozen image generator 用的是哪个 checkpoint**</mark>（Base 还是 RL 版？论文未说）
- <mark class="hl-trick">**Editing 侧 MLLM summarize 用的是哪个模型**</mark>
- <mark class="hl-trick">**PE 相对原始 prompt 的定量提升幅度**</mark>——论文只给定性图

## 4. Training ★

### 4.1 Multistage Training

<mark class="hl-key">**这一节描述的是模型训练的三个宏观 phase，与 §2.3 的六阶段数据流水线处于不同颗粒度。**</mark>论文没有给出二者的对应关系，<mark class="hl-trick">**不要自己强行配对**</mark>（→ §2.3⑩）。

$$
\boxed{
\text{Pre-training}
\rightarrow
\text{Continual Pre-training}
\rightarrow
\text{SFT}
}
$$

#### ① Table 2 完整配置

| Configuration | <mark class="hl-trick">Pre-training</mark> | <mark class="hl-trick">Continual Pre-training</mark> | <mark class="hl-trick">SFT</mark> |
| :--- | :--- | :--- | :--- |
| **Steps (K)** | <mark class="hl-key">**700**</mark> | <mark class="hl-key">**250**</mark> | <mark class="hl-key">**10**</mark> |
| <mark class="hl-trick">Resolution</mark> | 256 / 512 | 512 / 1024 / 2048 | 512 / 1024 / 2048 |
| <mark class="hl-trick">Batch Size (K)</mark> | 32 / 16 | 16 / 8 / 4 | 16 / 8 / 4 |
| **Data Distribution** | | | |
| Type | T2I / TI2I | T2I / TI2I | T2I / TI2I |
| <mark class="hl-key">Ratio</mark> | <mark class="hl-key">**0.9 / 0.1**</mark> | <mark class="hl-key">**0.7 / 0.3**</mark> | <mark class="hl-key">**0.7 / 0.3**</mark> |
| Optimizer | Adam | Adam | Adam |
| Weight Decay | 0.001 | 0.001 | 0.001 |
| Grad. Norm Clip | 1.0 | 1.0 | 1.0 |
| Uncond. Dropout | 0.1 | 0.1 | 0.1 |
| <mark class="hl-key">Learning Rate</mark> | <mark class="hl-key">**1×10⁻⁴**</mark> | <mark class="hl-key">**2×10⁻⁵**</mark> | <mark class="hl-key">**1×10⁻⁵**</mark> |

<mark class="hl-trick">**注意 Batch Size 一栏是按分辨率顺序给的**</mark>：pre-training 的 `32 / 16` 对应 256P / 512P，continual 与 SFT 的 `16 / 8 / 4` 对应 512P / 1024P / 2048P。<mark class="hl-key">**即分辨率越高、batch 越小，这是显存与算力的常规 trade-off**</mark>。

#### ② 这张表暴露的三个单调变化

<mark class="hl-key">**把三个阶段横着看，整套训练策略最核心的三个趋势一目了然**</mark>：

$$
\boxed{
\text{Resolution} \;\uparrow
\qquad
\text{Editing Ratio} \;\uparrow
\qquad
\text{Learning Rate} \;\downarrow
}
$$

<mark class="hl-trick">**也就是模型越往后训练，越从「大范围广覆盖学习」转向「高分辨率、更多编辑、更精细地调整分布」。**</mark>

#### ③ Pre-training：700K steps，先把通用生成能力立起来

<mark class="hl-key">**这是训练量最大的一段**</mark>，分辨率 $256\text{P} + 512\text{P}$，且

$$
\text{T2I} = 90\%,
\qquad
\text{TI2I} = 10\%
$$

<mark class="hl-key">**所以这一阶段主任务非常明确**</mark>：

$$
\boxed{
\text{先建立通用生成能力（general-purpose visual representation）}
}
$$

<mark class="hl-trick">**论文说 LR 设为 $1\times10^{-4}$ 是为了让模型从 large-scale image-text data 里学到 robust 的视觉表示。**</mark>

<mark class="hl-key">**这个比例很值得注意，因为它与 §2.3 的观察一致**</mark>——Qwen 很早就让 Generation 与 Editing 联合训练，<mark class="hl-key">**但「联合训练」并不意味着一开始就 1:1**</mark>。实际 recipe 更接近：

$$
\boxed{
\text{Generation 为主干}
+
\text{少量 Editing 提前建立统一条件接口}
}
$$

<mark class="hl-trick">**而不是先训一个纯 T2I Base，最后再突然往里塞 Editing。**</mark><mark class="hl-key">**对照 [Playbook §3](./training-playbook.md) 的 Generation replay 讨论：两篇都说明 Editing 不应完全独立于 Generation，但具体 mixture 并不相同。**</mark>

<mark class="hl-trick">**真正该学走的是**</mark>：**早期保留强 Generation 主分布，同时尽早给模型 Editing exposure**，<mark class="hl-key">**而不是死记 9:1**</mark>。

#### ④ Continual Pre-training：250K steps，把分辨率与 Editing 权重同时抬上去

<mark class="hl-key">**这一段有三个同时发生的变化**</mark>：

##### 变化一：最低分辨率从 256P 消失

$$
256/512 \;\longrightarrow\; 512/1024/2048
$$

<mark class="hl-key">**模型已经学完基础低分辨率语义之后，训练资源集中到真正的高分辨率生成与编辑。**</mark><mark class="hl-trick">**注意 512 仍然保留——与 §2.3⑦ 一样是多分辨率共存，不是逐级替代。**</mark>

##### 变化二：batch size 随分辨率下降

$$
512\text{P}:16\text{K}
\qquad
1024\text{P}:8\text{K}
\qquad
2048\text{P}:4\text{K}
$$

##### 变化三：Editing 权重从 10% 直接抬到 30%

$$
0.9 : 0.1 \;\longrightarrow\; \boxed{0.7 : 0.3}
$$

<mark class="hl-key">**而 LR 同时从 $10^{-4}$ 降到 $2\times10^{-5}$，论文给的理由是 `to ensure stable optimization during this stage`。**</mark>

<mark class="hl-key">**因此 Continual Pre-training 不是另起炉灶重新学，而是在已有 base 上扩大 resolution 与 capability boundary**</mark>，<mark class="hl-key">**所以学习率明显更保守**</mark>。

<mark class="hl-trick">**为什么抬 Editing 权重是合理的**</mark>：Editing 相比普通 T2I，需要模型同时完成

$$
\boxed{
\text{理解 source image}
+
\text{理解 instruction}
+
\text{保留无关区域}
+
\text{修改目标区域}
}
$$

<mark class="hl-key">**基础视觉生成能力成熟后再增加 TI2I exposure，符合 curriculum learning。**</mark>

#### ⑤ SFT：只有 10K steps

<mark class="hl-key">**最值得注意的对比在这里**</mark>：

$$
\boxed{
10\text{K} \;\ll\; 700\text{K} + 250\text{K} = 950\text{K}
}
$$

<mark class="hl-key">**SFT 的训练量比前面两段加起来小两个数量级。**</mark>所以它的作用显然不是「模型不会，靠 SFT 再重新学一遍」，而是：

$$
\boxed{
\text{用少量高质量数据重新塑造最终输出分布}
}
$$

<mark class="hl-trick">**论文对 SFT 的定位是 `improving the aesthetic quality of generated images`**</mark>；<mark class="hl-key">**降 LR 的理由是 `To enhance fine-grained visual details while preserving the model's world knowledge`**</mark>——即<mark class="hl-trick">**SFT 的风险是冲掉 world knowledge，所以 LR 必须小**</mark>。

<mark class="hl-trick">**SFT 数据方面，论文说从 diverse categories 采样，并施加 `strict filtering together with manual curation`。**</mark><mark class="hl-key">**这与 §2.3⑧ 的 S6 完全对应，数据侧与训练侧是同一件事的两面。**</mark>

#### ⑥ SFT 没有继续改 task mixture

<mark class="hl-key">**这一条很容易被忽略：SFT 并没有改变 T2I/TI2I 比例，而是保持 $0.7 : 0.3$。**</mark>也就是说从 Continual Pre-training 到 SFT，作者主要调整的<mark class="hl-trick">**不是 task mixture，而是**</mark>

$$
\boxed{
\text{Data Quality / Distribution}
\quad+\quad
\text{Learning Rate } 2\times10^{-5} \rightarrow 1\times10^{-5}
}
$$

<mark class="hl-key">**这意味着 SFT 的设计不一定非要「重新发明一套数据类别和训练任务」**</mark>，有时候只是

$$
\boxed{
\text{同样的能力 mixture}
+
\text{更高质量的数据}
+
\text{更严格的分布}
+
\text{更小的 LR}
}
$$

<mark class="hl-key">**就已经足够改变最终生成风格和质量。**</mark>→ 这条与 [Playbook §2](./training-playbook.md) 的「SFT 塑造最终输出分布」是同一判断，<mark class="hl-key">**而 Z-Image、Mage-Flow、Qwen 三篇都给出了相似方向的证据，现在可以相当确信地写进 Recipe**</mark>。

<mark class="hl-key">**这条也直接对应你之前关心的现象**</mark>——<mark class="hl-trick">**「为什么模型会偏亮、过饱和、GPT 独有质感学不到」这类最终 aesthetic distribution 问题，应当优先检查 SFT data distribution，而不是先改网络结构。**</mark>

#### ⑦ 三段能力变化

$$
\boxed{
\underbrace{
\text{700K Pretrain}
}_{\substack{
256/512\\
90\%\ \text{T2I}\\
10\%\ \text{Edit}\\
\text{LR}=10^{-4}
}}
}
\;\longrightarrow\;
\boxed{
\underbrace{
\text{250K Continual Pretrain}
}_{\substack{
512/1024/2048\\
70\%\ \text{T2I}\\
30\%\ \text{Edit}\\
\text{LR}=2\times10^{-5}
}}
}
\;\longrightarrow\;
\boxed{
\underbrace{
\text{10K SFT}
}_{\substack{
512/1024/2048\\
70\%\ \text{T2I}\\
30\%\ \text{Edit}\\
\text{High-quality curated}\\
\text{LR}=10^{-5}
}}
}
$$

<mark class="hl-key">**压缩成一句**</mark>：

$$
\boxed{
\text{先大规模学 Coverage}
\rightarrow
\text{再提高 Resolution + Editing}
\rightarrow
\boxed{\text{最后短程高质量 SFT 塑 Distribution}}
}
$$

<mark class="hl-trick">**而不是从头到尾拿同一批数据、同一个比例、同一个 learning rate 训练。**</mark>

#### ⑧ 一条很好用的规律

<mark class="hl-key">**这三条曲线的方向完全一致，合并成一条通用规律**</mark>：

$$
\boxed{
\text{越靠近最终模型}
\;\Rightarrow\;
\text{Data 更精、LR 更小、训练更短}
}
$$

$$
700\text{K} \rightarrow 250\text{K} \rightarrow 10\text{K}
\qquad
10^{-4} \rightarrow 2\times10^{-5} \rightarrow 10^{-5}
$$

<mark class="hl-key">**这是一个标准的 Broad Learning → Capability Refinement → Distribution Shaping 三段式。**</mark>→ 可补入 [Playbook §2](./training-playbook.md)。

::: warning 不要把 9:1 与 7:3 当成通用最优值
<mark class="hl-trick">**论文只是告诉我们它这么用了，并没有证明这些比例是最优的**</mark>。<mark class="hl-key">**全文没有任何针对 T2I/TI2I mixture 的 ablation**</mark>。

<mark class="hl-trick">**真正该进 Playbook 的是两条结构性判断**</mark>：

$$
\boxed{
\text{Task mixture 应该随训练阶段变化}
}
$$

$$
\boxed{
\text{SFT 的核心是高质量 distribution shaping，而不是继续堆训练量}
}
$$

<mark class="hl-trick">**而不是 9:1 和 7:3 这两个具体数字。**</mark>
:::

#### ⑨ 未公开细节汇总

- <mark class="hl-trick">**不同 resolution 在 continual / SFT 中实际的采样概率**</mark>——Table 2 给了 batch size，<mark class="hl-key">**但 batch size ≠ 数据采样概率，不能反推**</mark>
- <mark class="hl-trick">**每个 resolution 是否单独 queue / dataloader**</mark>
- <mark class="hl-trick">**T2I / TI2I 内部各 capability category 的比例**</mark>
- <mark class="hl-trick">**SFT 精确数据量与 manual curation 的标准**</mark>
- <mark class="hl-trick">**不同 caption 类型（§2.2 四类）如何采样**</mark>
- <mark class="hl-trick">**不同 resolution 是否有 loss reweight**</mark>
- <mark class="hl-trick">**$0.9:0.1$ 与 $0.7:0.3$ 是否经过任何 ablation**</mark>
- <mark class="hl-trick">**§2.3 Fig.6 标注的 `[S6 2048p] SFT` 与本节 Table 2 的 `512/1024/2048` 之间的口径差异**</mark>（→ §2.3②）

### 4.2 Reinforcement Learning with Human Feedback ★★

<mark class="hl-trick">**这一节是 Qwen-Image-2.0 真正的后训练核心。它的思路并不是发明一个新的 RL 算法，而是把 RL 做成一个多能力、多 reward、动态调度的系统。**</mark>

$$
\boxed{
\text{Prompt Pool}
\rightarrow
\text{Rollout}
\rightarrow
\text{Multi-dimensional Reward}
\rightarrow
\text{GRPO Update}
}
$$

#### ① T2I 与 TI2I 不共用同一套 Reward

<mark class="hl-key">**这是本节第一件最重要的事。**</mark>论文对 Generation 与 Editing 分别设计了不同的 task-specific composite reward models，<mark class="hl-trick">**每个 reward model 针对一个特定评估维度**</mark>。

##### T2I：三类 reward

$$
\boxed{
R_{\rm T2I} = R_{\rm aesthetic} + R_{\rm alignment} + R_{\rm portrait}
}
$$

| Reward | 论文原文的评价维度 |
| :--- | :--- |
| <mark class="hl-trick">Aesthetic</mark> | <mark class="hl-key">`compositional balance, realistic illumination, texture fidelity, and overall artistic coherence`</mark> |
| <mark class="hl-trick">Image-text alignment</mark> | <mark class="hl-key">**`explicitly penalizing outputs that omit, misinterpret, or contradict user-specified requirements`**</mark> |
| <mark class="hl-trick">Portrait</mark> | <mark class="hl-key">`anatomical plausibility, facial proportion accuracy, identity-preserving facial details, and fine-grained skin and hair texture realism`</mark> |

<mark class="hl-key">**所以 T2I 的优化目标不是单纯「越美越好」，而是**</mark>：

$$
\boxed{
\text{Quality}
+
\text{Semantic Compliance}
+
\text{Human-specific Capability}
}
$$

<mark class="hl-trick">**为什么单独拆出 Portrait Reward**</mark>：人物是图像生成里非常特殊的一类——脸部结构、皮肤质感、肢体、手、眼睛的失败非常敏感。<mark class="hl-key">**一个通用 aesthetic reward 很可能只能说「整体还不错」，但无法针对人物的具体失效模式给出信号**</mark>，所以需要独立维度。

::: warning 不能把 reward 脑补成某个具体模型
<mark class="hl-trick">**论文明确给出了这三类 reward 的评价维度，但没有公开完整 reward model、训练数据、具体打分 prompt 和权重**</mark>。<mark class="hl-key">**上面的加号是帮助理解的记法，论文并没有写出求和公式**</mark>。<mark class="hl-trick">**不能据此推断它用的是某个特定的 VLM 或 aesthetic scorer。**</mark>
:::

##### TI2I：两类 reward

$$
\boxed{
R_{\rm Edit} = R_{\rm instruction} + R_{\rm consistency}
}
$$

| Reward | 论文原文的评价维度 |
| :--- | :--- |
| <mark class="hl-trick">Instruction-following</mark> | <mark class="hl-key">`evaluates whether user-specified modifications are accurately executed, covering editing operations such as object replacement and style transfer`</mark> |
| <mark class="hl-trick">Visual consistency</mark> | <mark class="hl-key">**`preserves the identity and structural integrity of unmodified regions`，强制 source 与 edited 之间在 `geometric layout, spatial topology, and semantic features` 上严格一致**</mark> |

<mark class="hl-key">**这两个 reward 几乎就是 Editing 的两个核心矛盾**</mark>：

$$
\boxed{
\text{该改的必须改}
\qquad\text{同时}\qquad
\boxed{
\text{不该改的不能乱改}
}
$$

<mark class="hl-trick">**以「把衣服改成红色」为例**</mark>：Instruction reward 检查衣服到底红没红；Consistency reward 检查人物身份、背景、姿态、其他区域有没有被破坏。

<mark class="hl-key">**所以 Editing RL 的目标不是简单提高图像美观，而是处理一个更难的 trade-off**</mark>：

$$
\boxed{
\text{Edit Strength}
\;\longleftrightarrow\;
\text{Preservation}
}
$$

<mark class="hl-key">**这条应该进 Playbook**</mark>：以后碰到 Editing 出现「完全不改 / 改得不够 / 把整张图一起改了」这三类症状，<mark class="hl-trick">**第一反应不应该是统一地「加强 reward」，而应该先看 $R_{\rm instruction}$ 与 $R_{\rm consistency}$ 之间是不是失衡了**</mark>。

<mark class="hl-trick">**T2I 三类 + TI2I 两类 = 5 个任务专用 reward model。**</mark>

#### ② Scale Calibration 必须先于权重调节

<mark class="hl-key">**这是本节最容易被忽略、但工程上最关键的一条。**</mark>论文只有一句话交代：

> <mark class="hl-key">**`All reward models are calibrated to operate on comparable scales, and their weights are dynamically adjusted throughout training to avoid over-optimization toward any single dimension.`**</mark>

<mark class="hl-key">**「校准到可比尺度」与「动态调权重」是两个独立且有先后关系的动作**</mark>：

$$
\boxed{
\hat R_i = \operatorname{Calibrate}(R_i)
\qquad
R_{\rm final} = \sum_i w_i\,\hat R_i
}
$$

<mark class="hl-trick">**上面的公式是帮助理解的记法。论文只给了 `calibrated to operate on comparable scales` 这个陈述，没有给出 calibration 的数学形式**</mark>——<mark class="hl-key">**是 z-score、min-max 还是别的，论文未说明**</mark>。

##### 为什么这一步不能跳过

<mark class="hl-trick">**假设 $R_{\rm aesthetic}\in[0,1]$ 而 $R_{\rm alignment}\in[0,100]$，那么即使形式上写**</mark>

$$
R = R_{\rm aesthetic} + R_{\rm alignment}
$$

<mark class="hl-trick">**实际优化几乎完全由 alignment reward 控制**</mark>，因为它的尺度大两个数量级。

<mark class="hl-key">**所以在 multi-reward RL 里必须先统一 reward scale，否则「权重」这个概念本身没有真实含义**</mark>：

$$
\boxed{
\text{Multi-reward 的第一步不是调权重，而是先统一 scale}
}
$$

<mark class="hl-key">**这与 [DeepGen §3.3.1](./deepgen.md) 的结论完全一致**</mark>——<mark class="hl-trick">**那里给出了实证：Preference / OCR / CLIP similarity 三者数值范围与方差可以差好几个数量级，未做 per-reward 归一化时高方差 reward 会 dominate policy updates；去掉 reward-wise normalization 后 UniGenBench (Text) 掉 2.88 分，远超 GenEval / DPGBench 的 0.01–0.02**</mark>。

<mark class="hl-key">**证据等级要分开标注**</mark>：<mark class="hl-trick">**「必须做 per-reward 归一化」在 DeepGen 侧是 〔A〕（有消融），在 Qwen 这篇技术报告里只是 〔B〕（只有一句陈述，无任何数据）**</mark>。<mark class="hl-trick">**不要因为 Qwen 也这么说就把它当成 A 级证据。**</mark>

#### ③ Adapted GRPO，以及两处 GRPO 引用不同

<mark class="hl-trick">**算法层面 Qwen 并没有声称提出新的 RL paradigm。**</mark>论文写的是 `an adapted GRPO framework`，引三篇工作：

| 引用 | 工作 |
| :--- | :--- |
| Liu et al., 2026 | Flow-GRPO（flow matching 上的在线 RL） |
| Wang et al., 2025 | GRPO-Guard（regulated clipping 防过优化） |
| Zheng et al., 2025 | DiffusionNFT（forward-process 在线扩散 RL） |

<mark class="hl-trick">**而 §3.3 里 Prompt Enhancer 的 RL 引的是 `GRPO (Shao et al., 2024)`，即 DeepMath 的原始 GRPO。**</mark>

$$
\boxed{
\begin{aligned}
\text{PE 的 RL} &: \text{原始 GRPO（Shao et al., 2024）}\\
\text{生成器的 RL} &: \text{adapted GRPO，面向 flow matching}
\end{aligned}
}
$$

<mark class="hl-key">**这个区别值得记**</mark>：<mark class="hl-trick">**PE 优化的是文本输出，用原始 GRPO 就够；生成器优化的是 flow matching 模型的噪声预测，才需要上面那批适配工作**</mark>。

GRPO 的核心直觉仍然是：对同一 prompt 采样多个 candidate，按 reward 比较组内结果，用相对 advantage 更新 policy。

$$
\boxed{
\text{同一个 Prompt}
\rightarrow
\text{生成多张}
\rightarrow
\text{比较谁更好}
\rightarrow
\text{提高好样本概率}
}
$$

<mark class="hl-key">**Qwen 真正有意思的不是 GRPO 本身，而是它如何把 CFG 与 GRPO rollout 结合起来**</mark> → 下一节。

#### ④ Hybrid CFG：本节最值得记住的工程 trick

##### 先说结论

<mark class="hl-key">**Hybrid CFG 不减少 CFG rollout 的计算量。它仍然需要 conditional + unconditional 两次 forward；它真正降低的是 RL policy update 阶段的成本，因为 unconditional branch 不进入 policy objective，不需要为它构建梯度图和执行 backward。**</mark>

$$
\boxed{
\text{Sampling 用完整 CFG 保证样本质量，Learning 只优化 conditional policy 降低训练成本}
}
$$

##### 论文怎么说的

<mark class="hl-trick">**论文的原文是**</mark>：

> <mark class="hl-trick">**CFG is used during rollout sampling to generate high-quality candidates for reward evaluation, while the**</mark> <mark class="hl-key">**unconditional branch is excluded from the policy optimization objective.**</mark> <mark class="hl-trick">**This design preserves the visual fidelity and structural coherence of sampled images, thereby providing more reliable reward signals, while**</mark> <mark class="hl-key">**substantially reducing the computational overhead associated with optimizing the unconditional model.**</mark>

##### 把两个阶段彻底拆开

<mark class="hl-trick">**Rollout 阶段**</mark>：conditional 与 unconditional <mark class="hl-key">**两个分支都必须算**</mark>，然后组合

$$
\epsilon_{\rm CFG} = \epsilon_u + s\big(\epsilon_c - \epsilon_u\big)
$$

<mark class="hl-trick">**所以 rollout 的计算量仍然近似**</mark> $\boxed{2F}$ <mark class="hl-trick">**（$F$ 为一次模型 forward）**</mark>，<mark class="hl-key">**Hybrid 与否都跑不掉**</mark>。Qwen 保留这一步，是因为它希望 rollout 图片质量高、reward 信号可靠。

<mark class="hl-trick">**RL update 阶段才是省钱的地方。**</mark>如果把整个 CFG policy 都当成要优化的 policy，梯度原则上就是

$$
\nabla_\theta \epsilon_{\rm CFG} = s\nabla_\theta\epsilon_c + (1-s)\nabla_\theta\epsilon_u
$$

<mark class="hl-trick">**于是你不只要重算 conditional，还要重算 unconditional，并且两个分支都要保留 activation、构建计算图、做 backward。这才贵。**</mark>

Hybrid CFG 相当于：

$$
\boxed{
\epsilon_u \text{ 只参与 rollout，policy loss 中 stop-gradient}
}
$$

##### 一个粗略的 FLOPs 估算

<mark class="hl-key">**⚠️ 重要限定：下面的 $8F \to 5F$ 是帮助理解的 toy estimate，不是论文报告的真实加速数字——论文完全没有公开 Hybrid CFG 的 wall-clock 或 FLOPs 节省比例。**</mark>设一次 forward 为 $F$，一次 backward 约为 $B \approx 2F$：

设一次 forward 为 $F$，一次 backward 约为 $B \approx 2F$：

$$
\begin{array}{l|c|c|c}
& \text{Rollout} & \text{Policy Update} & \text{合计} \\
\hline
\text{全 CFG policy update} & 2F & 2F + 2B \approx 6F & \approx 8F \\
\text{Hybrid CFG} & 2F & F + B \approx 3F & \approx 5F
\end{array}
$$

<mark class="hl-key">**所以省掉的不是 rollout 那 $2F$，而是反向传播那一大坨计算与显存。**</mark>

##### 省的不只是 FLOPs，还有显存

<mark class="hl-trick">**如果 unconditional branch 也参与 policy loss，它还意味着**</mark>

$$
\text{保存 unconditional activations}
+
\text{autograd graph}
+
\text{gradient computation}
+
\text{更多 activation memory}
$$

<mark class="hl-key">**把 unconditional branch 从 policy objective 中拿掉，既省计算也省大量训练显存**</mark>——<mark class="hl-trick">**这在 diffusion RL 里尤其重要，因为成本本身就是「多个 rollout samples × 多个 denoising steps」**</mark>。

##### 一个非常容易误解的点

<mark class="hl-key">**「不更新 unconditional branch」并不意味着存在一个独立的无条件模型被冻结了。**</mark>conditional 与 unconditional 通常还是**同一套参数** $\theta$：

$$
f_\theta(x_t, c)
\qquad\text{vs.}\qquad
f_\theta(x_t, \varnothing)
$$

<mark class="hl-key">**更准确的说法是**</mark>：

$$
\boxed{
\text{unconditional forward 不贡献 policy-gradient}
}
$$

<mark class="hl-trick">**而不是「有一个独立的 unconditional network 不更新」。**</mark><mark class="hl-key">**因为参数是共享的，conditional branch 更新 $\theta$ 后，无条件输出本身以后也可能随之改变。**</mark>

::: warning 论文那句措辞其实不够准确
<mark class="hl-trick">**论文写的是「省掉 optimizing the unconditional model 的开销」，这个措辞暗示存在一个独立的 unconditional model——但从 diffusion 的实现看并不存在这样一个独立模块。**</mark>

<mark class="hl-key">**这里应该读作对论文措辞的一次修正**</mark>：<mark class="hl-trick">**它省的是「无分支参与反向传播」的开销，不是「不训练一个额外模型」的开销。**</mark>
:::

##### 这本质上是一个近似

<mark class="hl-trick">**既然 rollout 时的 action 由 $\epsilon_c$ 和 $\epsilon_u$ 共同产生，为什么 policy gradient 可以只算 conditional 分支？**</mark>严格来说，真正的 CFG policy 是

$$
a_t \sim \pi_{\rm CFG}\big(a_t \mid s\epsilon_c + (1-s)\epsilon_u\big)
$$

<mark class="hl-trick">**严格求梯度时两边都应该进入 $\nabla_\theta \log \pi_{\rm CFG}$。**</mark>Hybrid CFG 相当于把 $\epsilon_u$ 视作一个 <mark class="hl-key">**fixed guidance / baseline-like component**</mark>：rollout 时用它提高样本质量，但 policy optimization 时让 $\epsilon_c$ 承担「如何根据 prompt 改进生成结果」的责任。

<mark class="hl-key">**直觉上也说得通**</mark>，因为 RL 真正想学的是

$$
\boxed{
\text{Prompt Condition} \rightarrow \text{怎样产生更高 reward 的图}
}
$$

<mark class="hl-trick">**而 unconditional branch 本身不包含 prompt-specific 信息。**</mark>所以它是在

$$
\boxed{
\text{rollout fidelity}
\;\longleftrightarrow\;
\text{optimization cost}
}
$$

<mark class="hl-key">**之间做的折中，而不是一个精确的策略梯度。**</mark>

#### ⑤ Dynamic Prompt Distribution + Dynamic Reward Weight

<mark class="hl-trick">**论文明确说这两者都不是固定的**：</mark> <mark class="hl-key">**`dynamically adjusting the prompt distribution across tasks and calibrating the relative weights of individual reward models`**</mark>。

$$
\boxed{
\text{Dynamic Prompt Distribution}
+
\text{Dynamic Reward Weighting}
}
$$

<mark class="hl-trick">**也就是说 RL 过程中不是永远**</mark> $30\%_{\rm aesthetic} + 30\%_{\rm alignment} + 40\%_{\rm portrait}$ <mark class="hl-trick">**这种固定 recipe，而是模型在不同阶段遇到什么能力短板，就可以动态调整**</mark>

$$
p(\text{prompt type})
\qquad\text{与}\qquad
w_{\rm reward}
$$

<mark class="hl-key">**例如**</mark>：评测发现 <mark class="hl-trick">**portrait regression**</mark>，理论上就可以提高 $P(\text{portrait prompt})$ 或 $w_{\rm portrait}$；发现 <mark class="hl-trick">**instruction following 弱**</mark>，就提高对应 Editing prompt 与 reward 权重。

$$
\boxed{
\text{Capability-driven RL Curriculum}
}
$$

<mark class="hl-trick">**而不是**</mark> $\text{固定 Prompt Pool} + \text{固定 Reward} + \text{一路跑到底}$<mark class="hl-trick">**。**</mark>

<mark class="hl-key">**这与 §2.4 的 Data Flywheel 是连起来的**</mark>——Flywheel 在系统层面发现 bad case 并路由，RL 在训练层面动态调整 prompt 分布与 reward 权重，<mark class="hl-trick">**两者是同一套「按能力缺口配置资源」的思想在不同层的实现**</mark>。

<mark class="hl-key">**与已读过的两篇对照**</mark>：

| 工作 | 做法 | 证据 |
| :--- | :--- | :--- |
| <mark class="hl-trick">[Mage-Flow §4.2⑤](./mage-flow.md)</mark> | <mark class="hl-trick">**两阶段显式配比**</mark> $P_{\rm aes}:P_{\rm text}:P_{\rm sem}$ 从 $1:1:1 \to 2{:}4{:}1$，通过提高 OCR 权重强化文字能力 | <mark class="hl-trick">**手工设计的固定 schedule**</mark> |
| <mark class="hl-trick">[DeepGen](./deepgen.md)</mark> | <mark class="hl-key">**按 task type 动态切换 reward 配方**</mark>：text rendering 把权重压到 OCR 上，general T2I 完全不挂 OCR | <mark class="hl-trick">**按任务分派，非训练中动态**</mark> |
| <mark class="hl-key">**Qwen-Image-2.0**</mark> | <mark class="hl-key">**训练过程中动态调整**</mark> $p(\text{prompt})$ 与 $w_{\rm reward}$ | <mark class="hl-trick">**只有一句陈述，无 schedule、无数据**</mark> |

<mark class="hl-trick">**三者是同一趋势的三个刻度**</mark>：<mark class="hl-key">**都在往「reward 组合应该随能力缺口变化」这个方向走，Qwen 做得最动态但也最不透明。**</mark>

#### ⑥ 完整 pipeline

$$
\begin{aligned}
&\text{Capability-tagged / Dynamic Prompts} \\
&\quad\Downarrow \\
&\text{GRPO Rollout with Hybrid CFG} \\
&\quad\Downarrow \\
&\text{T2I: Aesthetic} + \text{Alignment} + \text{Portrait} \\
&\text{Editing: Instruction Following} + \text{Visual Consistency} \\
&\quad\Downarrow \\
&\text{Reward Scale Calibration} \\
&\quad\Downarrow \\
&\text{Dynamic Reward Weighting} \\
&\quad\Downarrow \\
&\text{Relative GRPO Advantage} \\
&\quad\Downarrow \\
&\text{Conditional-branch Policy Update} \\
&\quad\Downarrow \\
&\text{Evaluate Capability Gaps} \rightarrow \text{重新调整 Prompt / Reward Distribution}
\end{aligned}
$$

<mark class="hl-key">**注意这个环是闭的**</mark>——<mark class="hl-trick">**最后一步又回到最上面，形成 capability-driven 的自我强化循环，与 §2.4 的 Flywheel 同构。**</mark>

![Qwen-Image-2.0 Fig.10：RL 对齐前后定性对比。上半部为 T2I，四列布局 `Qwen-Image-2.0-Base | Qwen-Image-2.0-RL | Qwen-Image-2.0-Base | Qwen-Image-2.0-RL`，共 4 行 2 组：绿谷瀑布、街头戴贝雷帽男子（人群虚化）、中文食品包装「靠啥靠啥 升天降地」与城市烟花、敞篷跑车与花海、棕榈海滩与落地窗前的西装男子。**下半部为 Editing，四列布局 `Input Image | Input Text | Qwen-Image-2.0-Base | Qwen-Image-2.0-RL`**，共 3 行：(1) 输入为梵高《星夜》，指令是一段结构化的中文商业广告文案（概念标题／艺术溯源／概念故事／工艺与触感／产品清单／生活方式呈现／视觉设计元素／以简洁专业的艺术感布局呈现），Base 与 RL 均产出《星夜·茶韵》茶具套装系列海报，RL 版排版更规整；(2) 输入为黑底橙花，指令为 `Enhance the image clarity by applying super-resolution and deblurring techniques, preserving the original orange flower structure, green stem details, and black background while removing pixelation and noise.`；(3) 输入为老人肖像，指令为长段中文漫画创作要求（主体从手绘漫画书破页而出、保持外貌姿态、`2026` 透明烟花字样、暖光串灯虚化背景），Base 与 RL 均生成漫画风格图。](/qwen2-fig10-rl-comparison.png)

<mark class="hl-trick">**Fig. 10 有一个值得注意的结构信息：上半部 T2I 是 Base|RL 并排对比，下半部 Editing 是 Input Image|Input Text|Base|RL 四列**</mark>——<mark class="hl-key">**即编辑任务把输入图和指令文本也一起展示出来，因为编辑的保真度只有对着输入看才判断得了。**</mark>

<mark class="hl-trick">**论文对 Fig. 10 的结论只有定性描述**</mark>：T2I 侧 `notable improvements in texture fidelity and overall image realism`，编辑侧 `enhances texture quality and visual consistency`。<mark class="hl-key">**「texture fidelity」在三处 reward 的定义里都出现了**</mark>（aesthetic 的 texture fidelity、portrait 的 skin and hair texture realism、consistency 的 semantic features），<mark class="hl-trick">**但论文没有给任何数值 benchmark 证明这一点**</mark>——<mark class="hl-key">**§4.2 全节没有一个数字**</mark>。

#### ⑦ 后训练诊断思维

<mark class="hl-trick">**把 DeepGen、Mage-Flow、Qwen 放在一起，一个越来越明确的趋势是**</mark>：

$$
\boxed{
\text{图像 RL 的核心越来越像「训练系统设计」，而不只是 RL 算法设计}
}
$$

$$
\boxed{
\text{Prompt Data}
+
\text{Reward Design}
+
\text{Reward Calibration}
+
\text{Sampling Distribution}
+
\text{Capability Curriculum}
+
\text{Regression Control}
}
$$

<mark class="hl-trick">**很多时候这些比「GRPO vs 某个 GRPO variant」更决定最终效果。**</mark>→ 可补入 [Playbook §4](./training-playbook.md)。

<mark class="hl-key">**落到日常排障上，比如线上发现 OCR 弱，第一反应不应该是「换一个更先进的 RL algorithm」，而是依次检查**</mark>：

$$
\begin{aligned}
&\text{OCR Prompt Pool 是否够难？}\\
&\text{Reward 是否真的能判对？}\\
&\text{Reward scale 是否合理？}\quad \text{（见 ②）}\\
&\text{OCR Sampling Ratio 是否足够？}\\
&\text{增加 OCR 后 aesthetic 有没有 regression？}
\end{aligned}
$$

#### ⑧ §4.2 最该进 Post-training Recipe 的四条

$$
\boxed{
1.\ \text{Reward 必须按 Capability 拆，而不是一个万能 Reward}
}
$$

$$
\boxed{
2.\ \text{多 Reward 组合前先做 Scale Calibration}
}
$$

$$
\boxed{
3.\ \text{Prompt Distribution 和 Reward Weight 应随 Capability Gap 动态变化}
}
$$

$$
\boxed{
4.\ \text{Rollout Quality 与 RL 计算成本要分开优化，Hybrid CFG 是典型例子}
}
$$

::: warning 「动态」≠「已证明最优」
<mark class="hl-trick">**论文说「动态调整」，不等于论文证明了某种动态 schedule 最优。**</mark><mark class="hl-key">**它没有给出任何 controlled ablation，也没有公布 schedule。**</mark>

<mark class="hl-trick">**尤其要记住：这一节虽然系统设计很清楚，但训练细节比 DeepGen 更不完整。**</mark>以下全部未公开：

- RL prompt pool 总规模
- rollout group size $G$
- 每个 reward 的具体 evaluator
- reward calibration 的数学形式
- 各 reward 初始权重
- dynamic weighting 的 schedule
- prompt distribution 的动态调整规则
- GRPO 超参数、KL coefficient
- RL step 数
- <mark class="hl-trick">**是否混入 SFT loss**</mark>（<mark class="hl-key">**DeepGen 明确保留了 auxiliary SFT loss，Qwen 这篇完全没提**</mark>）
- 是否有 replay data
- Hybrid CFG 实际节省了多少 wall-clock / FLOPs
- 各 reward 的单独 ablation

<mark class="hl-key">**还有一处 §3.3 与 §4.2 的空缺值得单独记**</mark>：§2.2 提到的 Text-rich 能力、以及本节说的 OCR 弱，<mark class="hl-trick">**Qwen 这篇的 RL reward 里没有 OCR / text-accuracy 这一维**</mark>。<mark class="hl-key">**三个 T2I reward（aesthetic / alignment / portrait）都不专门度量字形正确性**</mark>——<mark class="hl-trick">**这与 Mage-Flow 把 OCR 权重压到 0.7 形成鲜明对比**</mark>。<mark class="hl-key">**Qwen 的文字能力主要来自 pretrain 数据与 Text Caption，RL 阶段没有再单独优化文字准确率，论文也未解释这个选择。**</mark>
:::

<mark class="hl-key">**RLHF 的方法细节另见已有的 RL 专题笔记**</mark> → [Qwen-Image-2.0 RLHF 统一对齐](./image-rl-posttraining/qwen-image-2-rl.md)（<mark class="hl-trick">**其中包含本篇技术报告未展开的组内标准化融合实现**</mark>）。

### 4.3 Few-step Distillation

## 5. Benchmark and Qualitative Evaluation ⬜

### 5.1 LMArena Benchmark Evaluation

### 5.2 Qualitative Results on Text-to-image Generation

### 5.3 Qualitative Results on Image Editing

## 6. Conclusion

## 附录 / 讨论

### A. 本篇未公开的细节（持续累积）

- <mark class="hl-trick">**数据规模完全未披露**</mark>：T2I 与 TI2I 各自的样本量、过滤各阶段的保留率，论文一个字都没给。<mark class="hl-key">**§2.3 的六阶段过滤流水线只有过滤器名称，没有任何一个阈值**</mark> —— 与 Mage-Flow 公开 Table 5 全表形成鲜明对比。
- <mark class="hl-trick">**Fig. 5 的占比未标注**</mark>：扇区角度看似编码了份额，但无数字，无法定量引用。
- <mark class="hl-trick">**多图（multi-image）数据侧细节几乎空白**</mark>：规模、来源、合成 pipeline、过滤方式均未公开。
- <mark class="hl-trick">**正文与 Fig. 5 的 taxonomy 不一致**</mark>：T2I 4 类 vs 3 类、single-image 6 项 vs 9 类、且正文的 background replacement 在图中无对应（详见 §2.1③）。
- <mark class="hl-trick">**四类 Caption 的全部实现细节**</mark>：数据量、采样比例、是否一图多 caption、routing 方式、captioner 身份、system prompt、Knowledge 的知识来源、Structured 的 schema、质量过滤（详见 §2.2⑧）。
- <mark class="hl-trick">**六阶段过滤流水线的所有阈值与 mixture**</mark>：每个 Stage 的数据规模、8 个 S1 filter 的阈值、512/1024/2048 采样概率、S6「imbalance」判定（详见 §2.3⑫）。
- <mark class="hl-trick">**Synthetic Data 在 S5 / S6 是否保留**</mark>：Fig. 6 只能确认它延续进了 S4，S4 之后色带不再按来源拆分，无法追踪（详见 §2.3②）。
- <mark class="hl-trick">**三张 Data 配图都无数字**</mark>：Fig. 5 无百分比标注，Fig. 6 的 Sankey 带宽虽编码相对量但无图例与刻度，Fig. 7 无任何量。<mark class="hl-key">**因此 Data 全章没有任何一个可定量引用的比例或样本量**</mark>。
- <mark class="hl-trick">**§3.3 Prompt Enhancer 的全部生产细节**</mark>：degradation 策略池与概率、$P_{\rm fine}$ 的生成方式、分类用哪个 LLM、CoT 格式、SFT 规模与超参、GRPO group size、三个 reward 的权重与 prompt、frozen generator 用哪个 checkpoint、Editing 侧 summarize 用哪个模型（详见 §3.3⑧）。
- <mark class="hl-trick">**PE 只有定性证据**</mark>：Fig. 9 不显示增强后的 prompt，caption 标 T2I 但含编辑案例，无任何 win rate 或评分（详见 §3.3⑥）。
- <mark class="hl-trick">**§4.1 的 resolution 采样概率与 mixture 细节**</mark>：batch size ≠ 采样概率；T2I/TI2I 内部能力配比；SFT 精确数据量与人工筛选标准；不同 caption 类型如何采样；是否有 loss reweight；9:1 与 7:3 无任何 ablation（详见 §4.1⑨）。
- <mark class="hl-trick">**§4.2 RLHF 是全篇最不完整的一节**</mark>：prompt pool 规模、rollout group size、每个 reward 的 evaluator、calibration 的数学形式、初始权重、dynamic weighting schedule、GRPO 超参与 KL 系数、RL step 数、是否混入 SFT loss、Hybrid CFG 实际节省多少算力、各 reward 的单独 ablation，全部未公开（详见 §4.2⑧）。
- <mark class="hl-trick">**两套 GRPO 引用的差异未展开**</mark>：§3.3 的 PE 用原始 GRPO（Shao et al. 2024），§4.2 的生成器用 flow-matching 适配版（Flow-GRPO / GRPO-Guard / DiffusionNFT），论文未解释这个分工（详见 §4.2③）。
- <mark class="hl-trick">**§4.2 全节没有一个数字**</mark>：包括 LMArena 之外的任何定量 benchmark 都没给，Fig. 10 纯定性。
- <mark class="hl-trick">**§3.1 VAE 与 §3.2 MMDiT 完全未读**</mark>：是否沿用 Qwen-Image 原 VAE、latent 通道数与下采样率均未核；MMDiT 的 3D RoPE 身份区分设计只在一句话里出现过（详见 §2.1③）。
