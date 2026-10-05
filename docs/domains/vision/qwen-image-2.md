# 千问 Qwen-Image-2.0 技术报告精读（统一生成 + 指令编辑 + Flywheel）

> **标签**：`Vision` `Unified Model` `MMDiT` `Flow Matching` `RL` `GRPO` `Data-centric` `Data Flywheel`
> **更新时间**：2026-10-05
> **原文**：本地 `Papers/Qwen-image-2.0.pdf`（30 页，Qwen Team）
> **精读重点**：§2 Data（四小节）→ §4 Training（三阶段 + RLHF + 蒸馏）
> **精读进度**：§2.1 ✅ ｜ §2.2 ✅ ｜ §2.3 ✅ ｜ §2.4 ⬜ ｜ §3 ⬜ ｜ §4 ⬜ ｜ §5 ⬜
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

::: warning Synthetic Data 从 S4 起是否继续存在，论文没说清
<mark class="hl-trick">**六阶段的标题（Stage 1–6 的正式命名）里，只有 Stage 3 之外全部写成 `T2I & TI2I`，Synthetic Data 从未出现在任何阶段标题中**</mark>——`256P T2I`、`256P T2I & TI2I`、`512P T2I & TI2I`、`512P/1024P T2I & TI2I`、`Multi-Resolution T2I & TI2I`。

<mark class="hl-key">**Synthetic Data 只在 Stage 3 的正文里被显式引入，此后再未提及。**</mark>因此：

- <mark class="hl-trick">**可以确认**</mark>：Synthetic 在 Stage 3 首次成为正式训练分布的一部分
- <mark class="hl-trick">**无法确认**</mark>：它在 S4 / S5 / S6 是否被保留、占比多少、是否被降权

<mark class="hl-trick">**不要默认「引入了就一直有」。**</mark>阶段标题的沉默是一个真实的歧义点。
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

### 2.4 Closed-loop Data Flywheel System

<mark class="hl-trick">**三阶段闭环：Signal Collection → Case Routing & Targeted Optimization → Model Update**</mark> ⬜ 待填

::: info 已核对的结构（供后续填写时对照，尚未展开）
**论文明确的路由逻辑**（这一段是 §2.4 的核心，Fig. 7 画得很清楚）：

| 失败原因 | 送去哪条轨道 |
| :--- | :--- |
| 强化学习不足 | <mark class="hl-key">**RL Track**</mark> → Reward / Policy Adjustment |
| prompt engineering 不足 | <mark class="hl-key">**PE Track**</mark> → Optimize Prompt Enhancer |
| 预训练时没见过这类数据 | <mark class="hl-trick">**Pre-training Track**</mark> → Vector Retrieval Engine + Data Augmentation |

<mark class="hl-key">**三条轨道都要过 Human Review & Filtering，然后 Model Training 到 Next Checkpoint，再回到 Model Evaluation。**</mark>

<mark class="hl-trick">**这一节的独特价值：它是四篇报告里唯一把「坏案例归因」写成显式路由表的**</mark> → 可直接对照 [Playbook §1 诊断字典](./training-playbook.md) 与 [Playbook §7 工业闭环](./training-playbook.md)。
:::

## 3. Architecture ⬜

### 3.1 Variational AutoEncoder

### 3.2 Multi-modal Diffusion Transformer

<mark class="hl-trick">**已核实的关键结构（后续填写时展开）**</mark>：Qwen3-VL 同时编码视觉与文本输入得到 $h_x, h_y$；其中 $h_x$ <mark class="hl-key">**被 VAE latent $E_x$ 替换**</mark>，再拼接

$$
h=\mathrm{Concat}\big(E_x,\ h_y\big)
$$

送入 Qwen-Image-2.0 block。架构为 MMDiT（引 Esser et al. 2024），text 与 image token 在**共享 backbone** 内处理。

### 3.3 Prompt Enhancer

<mark class="hl-trick">**PE 从 Qwen3.5-9B 初始化**</mark>，作为 T2I 与 TI2I **统一的** prompt enhancement 模型训练。

<mark class="hl-key">**论文提到一个反直觉的设计**</mark>：编辑场景下「输入图本身已提供丰富视觉上下文」，因此<mark class="hl-trick">**用一个 MLLM 把长形式标注summarize 成简洁 editing prompt，以「avoid unnecessary stochastic degradation」**</mark>——即<mark class="hl-key">**编辑场景刻意不让 PE 长篇大论**</mark>。这一条与 [Playbook §1](./training-playbook.md) 的「caption 不是越长越好」是同一类判断。

## 4. Training ⬜

### 4.1 Multistage Training

<mark class="hl-trick">**三阶段：Pre-training → Continual Pre-training → SFT**</mark> ⬜ 待填

::: info 已核对的 Table 2 完整配置（供后续填写时对照，尚未展开）
**Training Process**

| Configuration | Pre-training | Continual Pre-training | Supervised Fine-tuning |
| :--- | :--- | :--- | :--- |
| <mark class="hl-trick">Steps (K)</mark> | <mark class="hl-key">**700**</mark> | <mark class="hl-key">**250**</mark> | <mark class="hl-key">**10**</mark> |
| <mark class="hl-trick">Resolution</mark> | 256 / 512 | 512 / 1024 / 2048 | 512 / 1024 / 2048 |
| <mark class="hl-trick">Batch Size (K)</mark> | 32 / 16 | 16 / 8 / 4 | 16 / 8 / 4 |
| **Data Distribution** | | | |
| Type | T2I / TI2I | T2I / TI2I | T2I / TI2I |
| <mark class="hl-key">Ratio</mark> | <mark class="hl-key">**0.9 / 0.1**</mark> | <mark class="hl-key">**0.7 / 0.3**</mark> | <mark class="hl-key">**0.7 / 0.3**</mark> |

**Hyperparameters**

| Configuration | Pre-training | Continual Pre-training | Supervised Fine-tuning |
| :--- | :--- | :--- | :--- |
| Optimizer | Adam | Adam | Adam |
| Weight Decay | 0.001 | 0.001 | 0.001 |
| Grad. Norm Clip | 1.0 | 1.0 | 1.0 |
| Uncond. Dropout | 0.1 | 0.1 | 0.1 |
| <mark class="hl-key">Learning Rate</mark> | <mark class="hl-key">**1×10⁻⁴**</mark> | <mark class="hl-key">**2×10⁻⁵**</mark> | <mark class="hl-key">**1×10⁻⁵**</mark> |

<mark class="hl-key">**值得注意：T2I:TI2I 比例在 continual pre-training 之后不再继续下降，7:3 一直保持到 SFT。**</mark>并且 <mark class="hl-trick">**三阶段的 learning rate 是严格单调递减的 1e-4 → 2e-5 → 1e-5**</mark>，<mark class="hl-key">**与 [Playbook §2](./training-playbook.md)「SFT 是 distribution shaping 而非再训练一会」的定位一致**</mark>。
:::

### 4.2 Reinforcement Learning with Human Feedback

<mark class="hl-trick">**详见已有的 RL 专题笔记**</mark> → [Qwen-Image-2.0 RLHF 统一对齐](./image-rl-posttraining/qwen-image-2-rl.md)

<mark class="hl-key">**本篇不重复该笔记的 reward model 与五奖励 GRPO 细节**</mark>，待本节展开时只补充与 §2 数据体系的联动关系。

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
- <mark class="hl-trick">**六阶段过滤流水线的所有阈值与 mixture**</mark>：每个 Stage 的数据规模、8 个 S1 filter 的阈值、512/1024/2048 采样概率、S6「imbalance」判定、Synthetic 是否延续到 S4–S6（详见 §2.3⑫）。
- <mark class="hl-trick">**§3.1 VAE 完全未读**</mark>：是否沿用 Qwen-Image 原 VAE、latent 通道数与下采样率均未核。
