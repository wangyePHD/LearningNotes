# 千问 Qwen-Image-2.0 技术报告精读（统一生成 + 指令编辑 + Flywheel）

> **标签**：`Vision` `Unified Model` `MMDiT` `Flow Matching` `RL` `GRPO` `Data-centric` `Data Flywheel`
> **更新时间**：2026-10-05
> **原文**：本地 `Papers/Qwen-image-2.0.pdf`（30 页，Qwen Team）
> **精读重点**：§2 Data（四小节）→ §4 Training（三阶段 + RLHF + 蒸馏）
> **精读进度**：§2.1 ✅ ｜ §2.2 ⬜ ｜ §2.3 ⬜ ｜ §2.4 ⬜ ｜ §3 ⬜ ｜ §4 ⬜ ｜ §5 ⬜
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

<mark class="hl-trick">**四类 Caption**</mark>：General / Text / Knowledge / Structured ⬜ 待填

### 2.3 Multi-Stage Training Data Strategy

<mark class="hl-trick">**六阶段过滤流水线（Stage 1–6）**</mark> ⬜ 待填

::: info 已核对的结构（供后续填写时对照，尚未展开）
<mark class="hl-key">**注意区分：论文 §2.3 的「六阶段」是过滤流水线，§4.1 的训练只有三段。**</mark>这是两个不同的东西，极易混淆。

**§2.3 的六个过滤阶段：**

| Stage | 名称 |
| :-: | :--- |
| 1 | 256P T2I pre-training |
| 2 | 256P T2I & TI2I pre-training |
| 3 | 512P T2I & TI2I pre-training |
| 4 | 512P/1024P T2I & TI2I pre-training |
| 5 | Multi-Resolution T2I & TI2I pre-training |
| 6 | Supervised fine-tuning |

**Stage 1 内部的 8 个顺序过滤器**：Broken Files → Resolution → Deduplication → NSFW → Rotation → Entropy → CLIP → Token Length

<mark class="hl-trick">**全文出现过的过滤器名称**（跨阶段）：Broken Files、Resolution、Deduplication、NSFW、Rotation、Entropy、CLIP、Token Length、Image Quality、Compression Quality、Image Aesthetic、Distribution</mark>
:::

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
- <mark class="hl-trick">**「Knowledge captions」与「Structured captions」的完整定义未读完**</mark>（§2.2 待填）。
- <mark class="hl-trick">**§3.1 VAE 完全未读**</mark>：是否沿用 Qwen-Image 原 VAE、latent 通道数与下采样率均未核。
