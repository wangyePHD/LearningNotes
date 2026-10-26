---
type: paper
tags: ['RL', 'Rubric', 'Reward Model', 'Reward Hacking', 'DiffusionNFT', 'Data-centric']
status: done
summary: 'Arena。算法不变、只动奖励侧。核心命题：高质量 RL = Reward Coverage + Reward Composition + Reward Robustness。FLUX.2-dev +69 Elo、Ideogram-4 达 1223.5。**Part 1–10 完结**：10K Arena Prompt 经 Gemini-3.5-Flash 六条 VALID 规则过滤（未过滤 100K 低 0.0360）；BT Preference RM 5M 偏好对（Score 为无界 Logit，BT Loss 只依赖差值）；Faithfulness Checklist **182,230 问 / 18.2 每 Prompt**，Gemini-3-Pro 离线生成**依赖 DAG**，父失败子自动 No；Intent 三分类 strict/neutral/open，26,213 约束问题仅 3,140 启用（31.4%），人工审计 89.5%；Anti-Hack = 205 失败案例审计出的**两种固定 detector** + Applicability Gate + **One-sided Veto**（additive 化从 .635 崩到 .526）；Ideogram-4 Graded OCR = 0.5/0.3/0.2 三分量、无文字一律 Mask；公式 (3)–(9) 分轴 z-score → Gate → Veto → A_i=(R_i-μ_group)/σ_batch → r̃=½clip(A_i/5,-1,1)+½；Stage 1/2/3（Stage 2 单跑不超 Stage 1，收益来自 LoRA soup，r64×2→r128 精确平均）。'
---

# Arena-RL：Preference × Rubric 复合奖励后训练前沿 T2I 基模 (Arena)

> **arXiv**：[2610.02967](https://arxiv.org/html/2610.02967)
> **标题**：*Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards*
> **涉及基模**：FLUX.2-dev · Ideogram-4
> **精读进度**：**Part 1–10 完结**
> **Part 1**：Abstract / §1 Introduction / §2 Related Work → §1–§8
> **Part 2**：§4.1 Training and Evaluation Prompts / 附录 A.2.2 → §9–§16
> **Part 3**：§3.2.1 / 附录 A.1 / §4.3 RM Scaling → §17–§25
> **Part 4**：§3.2.2 / 附录 A.3.1–A.3.2 / Figure 4 → §26–§33
> **Part 5**：§3.2.3 / §3.3.2 / 附录 A.3.3、A.3.6 → §34–§43
> **Part 6**：§3.2.4 / §3.3.2–3.3.3 / 附录 A.3.5 → §44–§52
> **Part 7**：§4.4 / 附录 A.3.4 Graded OCR Reward → §53–§63
> **Part 8**：§3.3.1–§3.3.4，公式 (3)–(9) → §64–§75
> **Part 9**：§4.1 Implementation details / §3.3.5 / Table 2 → §76–§80
> **Part 10**：§3.3.5 公式 (10) / 附录 A.6 → §81–§85
> **专区**：[🎲 AIGC 强化学习后训练](./index.md)

---

## 1. 总体：作者究竟提出了什么

这篇论文<mark class="hl-key">**并不是研究怎样发明一个比 DiffusionNFT 更强的 RL 算法，而是在已有 RL 算法的基础上，研究如何构建一套真正有效的文生图 Reward System**</mark>。

它要解决的并不是简单的「Reward Model 不准确」，而是三个相互关联的问题：

1. **Coverage**：单一 Reward 无法覆盖所有生成质量维度；
2. **Composition**：多个 Reward 直接相加又可能相互干扰；
3. **Robustness**：即使 Reward 设计合理，模型在持续优化过程中也可能发现漏洞。

<mark class="hl-trick">**这三个问题就是全文的逻辑起点。**</mark>

### 1.1 Abstract 的逻辑还原

当前文生图模型在视觉质量方面已经取得很大进展，但通过后训练继续提高性能仍然困难。主要原因是，<mark class="hl-key">**没有一种单独的奖励信号能够完整表达用户对图像生成的所有偏好**</mark>。

因此作者提出一套面向开放域文生图的后训练方案，将两种互补的奖励体系组合起来：

| 奖励体系 | 训练方式 | 职责 |
| :--- | :--- | :--- |
| **Preference Reward** | 大规模人类偏好比较数据 + **Bradley–Terry 目标**训练 RM | 整体人类审美与感知偏好（视觉质量、美感、感知偏好） |
| **Rubric-based Reward** | 依据明确的评价准则检查图片是否符合用户指令 | 指令符合度，**并对 Reward Hacking 提供防护** |

但把不同奖励简单做加权平均并不能得到理想的优化效果，因此作者进一步设计了 **Reward 组合机制**，使得模型在优化总体偏好的同时更好地满足具体指令。

### 1.2 核心贡献的抽象

可以抽象成：

$$
\boxed{
\text{高质量 RL} \neq \text{只需要一个高质量 Reward Model}
}
$$

而是：

$$
\boxed{
\text{高质量 RL}
=
\text{Reward Coverage}
+
\text{Reward Composition}
+
\text{Reward Robustness}
}
$$

- **Coverage**：Reward 是否覆盖了用户真正关心的所有质量维度？
- **Composition**：多种 Reward 如何组合，才不会互相破坏？
- **Robustness**：模型通过 RL 不断提高得分时，会不会利用 Reward 的漏洞？

<mark class="hl-key">**后面十几个 Part，实际上就是围绕这三个问题逐步展开。**</mark>

---

## 2. Introduction：为什么 Preference Reward 不够？

这是整篇论文最值得理解的动机。

### 2.1 一个概念性例子

假设使用当前 Base Model，对同一个 Prompt 生成两张图：

> *A cartoon illustration of three red apples on a wooden table, with a plain white background.*

用户要求的是：**卡通风格、三个苹果、红色、木桌、纯白背景**。

| 示例 | 特征 |
| :--- | :--- |
| **示例 A（更忠实）** | 满足数量、颜色、媒介和背景要求 |
| **示例 B（更精致但可能偏题）** | 视觉质量可能很高，却不符合卡通及白背景要求 |

::: warning 这两张只是概念性示例
不是论文中的对照实验。
:::

假设 Preference RM 给 A 打 $0.72$、给 B 打 $0.91$，那么 NFT 会倾向于强化 B。

<mark class="hl-trick">**这并不是因为 DiffusionNFT 的公式有问题。真正的问题是：Reward Model 没有充分表达用户的具体要求。**</mark>

在视觉偏好数据里，用户可能更容易选择「更漂亮、更精致」的图片，但这并不意味着他愿意牺牲明确要求的数量、构图、风格和文字。

### 2.2 一个容易忽略的机制

Preference Reward 通常把很多质量维度<mark class="hl-key">**压缩成一个分数**</mark>：

$$
R_{\mathrm{pref}}(x, p)
\approx
f\bigl(\text{Aesthetic},\ \text{Realism},\ \text{Composition},\ \text{Faithfulness},\ldots\bigr)
$$

这里的 $f$ 是奖励模型内部学到的复杂评价关系，<mark class="hl-trick">**并不是作者明确设定的线性公式**</mark>。

假设模型通过增强光照、提高饱和度、改善构图让总奖励提高了 $0.2$；但同时漏掉一个苹果，或把卡通改成摄影，导致部分 Faithfulness 下降。只要奖励模型对前面的收益更敏感，最终总分仍可能提高。

于是 DiT 收到的训练信号就是：<mark class="hl-key">**「这样的改变值得强化。」**</mark>

长期训练后，模型可能<mark class="hl-trick">**越来越擅长提高 Reward Model 喜欢的属性，却越来越不擅长严格满足用户要求**</mark>。

::: tip 这就是作者认为需要 Rubric Reward 的根本原因
:::

### 2.3 论文的实验证据（Figure 5）

论文 Figure 5 给出了直接实验：分别使用 **Arena Reward Model** 和 **PickScore** 作为唯一奖励来训练 FLUX.2-dev，随着训练进行，模型在 **Faithfulness 维度上的表现持续下降**。

- Arena RM 的最佳总体配对胜率仅约 **$50.65\%$**；
- PickScore 的整体表现**没有稳定超过 Base Model**。

![Arena-RL Figure 5（附录 A.9）：单一偏好奖励训练相对 base 的胜率随 checkpoint（30/60/90/120/150）变化。三张子图分别为 overall / faithfulness / aesthetics，蓝线为 arena RM only，红虚线为 PickScore only，虚线为 base parity 0.50。overall 子图中两条曲线全程低于或贴近 0.50；faithfulness 子图从约 0.46 单调下滑到 0.40（arena）与 0.25（PickScore），是三张图中退化最严重的维度；aesthetics 子图则基本持平在 0.5 附近。](/arena-fig5-single-preference-hurts.png)

这张图想说明的<mark class="hl-key">**并不是「Preference Reward 完全没有用」，而是：单一 Preference Reward 的优化方向，并不必然等于用户真正希望提高的方向。**</mark>

---

## 3. 作者提出的 Reward System 拓扑

作者虽然在摘要中只强调 Preference 和 Rubric 两大类，但实际方法包含**四种不同职责**。

$$
\text{用户 Prompt}
\;\rightarrow\;
\text{当前模型生成图像}
\;\rightarrow\;
\begin{cases}
\text{Preference} & \text{整体质量、审美、人类偏好} \\
\text{Faithfulness} & \text{要求的内容是否正确出现} \\
\text{Constraint} & \text{严格约束与多余内容} \\
\text{Anti-Hack} & \text{检测利用奖励漏洞的行为}
\end{cases}
\;\rightarrow\;
\text{Reward Composition}
\;\rightarrow\;
\text{DiffusionNFT 更新 DiT}
$$

| 奖励 | 提问方式 |
| :--- | :--- |
| Preference | 图片整体好不好？ |
| Faithfulness | 该有的东西有没有？ |
| Constraint | 不该有的东西有没有？ |
| Anti-Hack | 是不是在刷分？ |

### 3.1 Faithfulness vs Constraint：最容易混淆的一对

这两类 Reward 看起来类似，但<mark class="hl-key">**实际并不等价**</mark>。

| 类型 | 方向 | 检查什么 |
| :--- | :--- | :--- |
| **Faithfulness** | **Presence-Oriented** | 用户要求的东西是否出现：有没有三个苹果、苹果是不是红色、是不是在桌子上、是不是卡通风格 |
| **Constraint** | **Absence-Oriented** | 是否引入了与用户约束冲突的内容：用户只想要一个黑色圆圈，模型却加了三个装饰性图案 |

假设用户只说：

> *An Impressionist painting of a country road.*

那么模型在道路旁添加鲜花或人物，<mark class="hl-trick">**可能属于合理的艺术创作**</mark>。如果把所有「用户没有明确提到的物体」都视为错误，模型就会逐渐失去创作自由。

因此作者使用 <mark class="hl-key">**Intent-Gated Constraint Reward，只在需要严格遵守内容限制的 Prompt 上启用它**</mark>。

### 3.2 Anti-Hack 不是加项

<mark class="hl-key">**Anti-Hack 并不是和前三个 Reward 一起简单加分。**</mark> 它主要识别已经观察到的作弊行为，在触发时<mark class="hl-trick">**抑制有问题的正向 Preference 信号**</mark>，这种方法叫 **One-sided Veto**。

---

## 4. 为什么不直接把四个 Reward 加起来？

这是论文的第二个核心问题。假设我们获得：

| Reward 类型 | 原始分数 | 特点 |
| :--- | :--- | :--- |
| Preference | 7.8 | 可能是未限定范围的模型 Logit |
| Faithfulness | 0.65 | 满足的检查项占比 |
| Constraint | 1.00 | 约束满足比例 |
| Hack Detector | 0 或 1 | 是否发现特定问题 |

这些数值的尺度、分布、含义都不一样。如果直接相加：

$$
R = R_{\mathrm{pref}} + R_{\mathrm{faith}} + R_{\mathrm{constraint}}
$$

<mark class="hl-trick">**Preference 的数值可能主导优化，另外两个信号几乎不起作用。**</mark>

即使全部归一化到可比尺度，仍然有其他问题：

- **Constraint** 对开放式创作 Prompt 不一定适用；
- **Anti-Hack** 如果被当成正向加分项，也可能形成新的作弊目标。

因此，作者<mark class="hl-key">**不是简单调一个 $w_1, w_2, w_3$，而是设计了四个机制**</mark>：

| 机制 | 解决的问题 | 主要目的 |
| :--- | :--- | :--- |
| Reward-decoupled Normalization | 奖励尺度不同 | 避免单一数值尺度支配训练 |
| Prompt-conditioned Gating | 并非每个奖励适用于每个 Prompt | 防止错误的优化压力 |
| One-sided Veto | 生成器利用 Reward 漏洞 | 阻止不良结果获得正向激励 |
| Weight Ensembling | 不同 Reward 方案互有优势 | 合并不同 RL 策略的收益 |

::: warning 一个值得留意的论文表述细节
引言提到将 Reward 归一化到共同的 $[0,1]$ 尺度，但 **§3.3.1 的实际公式是逐维 z-score 标准化，结果并不限定在 $[0,1]$**；最终转换为 NFT 权重时才映射到 $[0,1]$。

<mark class="hl-key">**所以后面精读 Part 8 时应该以公式 (3)–(9) 为准，而不是直接按引言字面理解。**</mark>
:::

---

## 5. Introduction 里的实验与贡献声明

作者使用两个不同的基础模型验证方案：

| 基模 | Arena Elo（Base） | Arena Elo（Ours） | 增益 |
| :--- | :---: | :---: | :---: |
| **FLUX.2-dev** | 1132.8 | 1202 | 约 **+69 Elo** |
| **Ideogram-4** | 1204 | 1224 | 约 **+20 Elo** |

Table 1，**2026 年 9 月 4 日榜单快照**。论文另报告 Ideogram-4 的更精确 Elo 值为 **1223.5**。

<mark class="hl-key">**这里作者强调该方法不仅对开源 FLUX 有效，对另一个强大的文生图模型 Ideogram-4 也有效。**</mark>

另外，作者计划公开 **Arena-T2I-Training 1K 数据子集**，使研究者能够以更小的数据规模测试这套方法。

### 5.1 需要保留的两点判断

::: danger 研究者视角的保留意见
1. 这些 Elo 提升是<mark class="hl-key">**作者报告的结果**</mark>；具体收益仍然依赖评测平台、Prompt 分布、模型版本和部署设置。

2. 我们之前已经讨论过，<mark class="hl-key">**Arena 历史用户 Prompt 和投票数据被用于训练，最终又在 Arena 平台进行在线评测**</mark>。因此<mark class="hl-trick">**同平台偏好适配与真正的跨平台泛化需要区分**</mark>。

<mark class="hl-key">**这不直接证明测试数据泄露，但意味着后面的评测和数据隔离细节必须仔细审查。**</mark>
:::

---

## 6. Related Work：作者如何定位自己的贡献

这一节很重要，因为它实际上说明了作者如何定位自己的贡献。

### 6.1 第一类：Diffusion / Flow RL 算法

| 工作 | 核心方向 |
| :--- | :--- |
| **Diffusion-DPO** | 将直接偏好优化应用于扩散模型，不必显式训练 Reward Model |
| **Flow-GRPO** | ODE-to-SDE，使 Flow Matching 可以使用 GRPO 类策略优化 |
| **DiffusionNFT** | 在前向加噪过程上做策略改进，避免沿反向采样器优化 |
| **MixGRPO** | 混合 ODE / SDE Rollout 调度 |
| **TempFlow-GRPO** | 考虑噪声时间步的差异，采用噪声感知权重 |

<mark class="hl-key">**作者明确选择 DiffusionNFT 作为后训练算法。**</mark> 也就是说，这篇论文希望保持 Policy Optimization 算法相对稳定，<mark class="hl-trick">**把主要研究变量放在 Reward 一侧**</mark>。

结合你现在已经熟悉 NFT、也在实际训练，这一点尤其重要：<mark class="hl-key">**你完全可以把本文看作一篇 RL Reward 与数据系统论文，而不是算法论文。**</mark>

### 6.2 第二类：Preference Reward 和 Quality Reward

作者又把图像奖励模型分为几种：

| 类别 | 代表工作 | 特点 |
| :--- | :--- | :--- |
| 总体偏好模型 | ImageReward、PickScore、HPSv2/v3 | 利用人类图像偏好数据学习一个总体分数；评价范围广、使用方便，但**无法保证每个细粒度要求都被充分关注** |
| 统一 VLM Judge | UnifiedReward | 同时处理图像理解和生成质量评价 |
| 感知/可验证评分器 | LAION Aesthetic Predictor、GenEval 检测器、OCR 文字奖励、基于依赖检查的 Faithfulness Reward | 往往只覆盖特定能力，但**更容易得到有明确含义的反馈** |

作者在实验中具体比较两种 Preference Reward：公开的 **PickScore**，以及<mark class="hl-key">**自己利用约 500 万对 Arena 人类偏好投票训练的 Reward Model**</mark>。

这里其实隐含了一个很有价值的判断：

::: tip 总体偏好模型与可验证能力奖励不是互相替代的关系
前者负责覆盖很宽的用户审美，后者负责修补总体分数中被忽略的细节。
:::

### 6.3 第三类：Rubric-based Reward

<mark class="hl-key">**这是我们后续最值得深入的一类工作。**</mark>

Rubric 的意思不是简单给图像打一个质量分，而是<mark class="hl-trick">**把原本模糊的评价目标拆成若干明确标准**</mark>。例如：

> *A blue car parked behind a red bus on a rainy city street.*

可以拆成：

- 有没有蓝色汽车？
- 有没有红色公交车？
- 汽车是否位于公交车后方？
- 场景是否是城市街道？
- 是否表现出下雨的状态？

然后逐项评价。作者引用了几项与 Rubric 自动生产相关的工作：

| 工作 | 研究思路 |
| :--- | :--- |
| **OpenRubrics** | 对比偏好与拒绝样本，推导评价规则 |
| **AutoRule** | 从偏好样本和推理过程提取候选规则 |
| **Autorubric-T2I** | 利用图像偏好数据学习 Rubric 权重，选择有区分度的评价项 |
| **Chasing the Tail** | 迭代细化 Rubric，提高对优秀候选之间细微差异的区分能力 |
| **Auto-Rubric** | 通过信息论式压缩构建简洁、低冗余的通用评价标准 |

::: warning 这篇论文的贡献定位
Arena 这篇<mark class="hl-key">**不是最早提出 Rubric Reward，也不是最早提出基于依赖关系的图像 Faithfulness Checklist**</mark>。它真正做的是<mark class="hl-trick">**把这些已有思路组合起来，加入大规模真实人类偏好 Reward、适用条件判断和 Reward Hacking 防护，然后验证能否形成一套在较强文生图基模上有效的 RL Recipe**</mark>。

换句话说，<mark class="hl-key">**它的贡献更偏系统集成与工程验证。**</mark>
:::

---

## 7. 从我们的角度：Part 1 最应该学到什么

读完动机和相关工作后，我认为有三个重要认识。

::: tip 第一，文生图 RL 的性能上限，不只是由 NFT 或 GRPO 决定，也受到 Reward 覆盖范围和可靠性的限制
这与你之前怀疑「图像 RL 增益有限」的观点并不冲突。<mark class="hl-trick">**算法可以持续优化一个目标，但如果这个目标没有充分表达用户真正关心的质量，增加训练算力也未必有帮助。**</mark>
:::

::: tip 第二，RL Reward 与 RL Prompt 数据是紧密联系的
这篇论文后续使用真实用户 Prompt，每个 Prompt 再生成对应的 <mark class="hl-key">**Faithfulness Checklist、Constraint Checklist 和适用条件标签**</mark>。也就是说，<mark class="hl-trick">**训练数据不是简单的文本列表，而是一套带评价规则的数据系统。**</mark>
:::

::: tip 第三，Reward 应该关注两个不同问题：是否满足要求，以及是否引入不必要的内容
这个区分非常适合<mark class="hl-key">**信息图、场景文字、产品设计与高可控生成任务</mark>，但<mark class="hl-trick">**不能无条件用于开放式艺术创作**</mark>。
:::

::: danger 一个容易被忽略的研究机会
这篇论文试图通过组合 Reward 提升生成质量，但其<mark class="hl-key">**真正可迁移的核心，可能是如何自动生产、审核并维护这些 Reward 对应的数据标注**</mark>。

这正好与你现在想做的 RL 数据生产高度一致。
:::

---

## 8. Part 1 小结

**用一句话总结：**

这篇论文认为，强文生图模型的 RL 后训练，<mark class="hl-key">**关键瓶颈可能已经不在 Policy Optimization 算法本身，而在于如何构建覆盖用户偏好、可验证指令要求、能够避免模型钻空子的 Reward System**</mark>。

目前我们已经覆盖了 **Abstract、Introduction 的问题与贡献、Table 1 的主要结果，以及 Related Work 的三个方向**。

**下一部分是 Part 2：RL Prompt 数据收集与筛选。** 我们会进入你最关心的工程细节：Arena 真实 Prompt 如何获得、Validator 的每一条筛选标准是什么、如何使用 LLM 判断、怎样产生训练数据，以及论文到底通过什么实验说明过滤后的 10K Prompt 比未过滤的 100K 更有价值。

---

# Part 2：RL Prompt 数据收集与筛选

> **对应原文**：§4.1 Training and Evaluation Prompts、附录 A.2.2 Training-prompt Filtering

这一部分<mark class="hl-key">**只研究 RL 训练 Prompt 从哪里来、怎样过滤、过滤是否有效**</mark>。Faithfulness Checklist、Constraint、Reward Model 的具体生产方法放到后续 Part，不混在一起。

## 9. 最关键的结果

| 训练 Prompt | 验证 Prompt | 过滤模型 |
| :---: | :---: | :---: |
| **# 10K** | **# 1K** | **Gemini-3.5-Flash** |

作者将 **100K 未过滤 Prompt** 与 **10K 过滤后 Prompt** 进行 RL 对照，报告未过滤组在相同 Gemini 评价协议下，<mark class="hl-key">**比过滤组低 $0.0360$ 的配对评价胜率**</mark>。

---

## 10. 数据来源：直接使用 Arena 的真实用户 Prompt

论文 §4.1 写得很明确：作者从 **Arena Leaderboard 的真实用户 Prompt 分布**中采样训练和验证数据，<mark class="hl-key">**而不是从 LAION、COYO 之类的图文数据集中提取 Caption，也没有说主要使用 LLM 合成 Prompt**</mark>。

::: tip 这个区别很重要
SFT 中的 Caption 往往是在<mark class="hl-trick">**描述一张已经存在的图片**</mark>；而 Arena 用户 Prompt 是<mark class="hl-trick">**真实用户希望模型生成什么图片的直接表达**</mark>。作者选择后者，目的是让 RL 优化目标贴近真实用户需求。
:::

不过，原始用户输入不一定是合格的文生图训练指令。它可能是一条<mark class="hl-key">**图像编辑请求**</mark>，可能<mark class="hl-key">**依赖之前的聊天**</mark>，也可能只写了几个单词、包含<mark class="hl-key">**小众专有名词**</mark>，或者是<mark class="hl-key">**英文以外的语言**</mark>。

因此，作者没有把原始 Prompt 全部送进 RL，而是引入一个严格的 **Validator**。

$$
\text{Arena 真实用户 Prompt}
\;\rightarrow\;
\begin{cases}
\text{Gemini-3.5-Flash Validator} \text{（固定判定规则，输出 yes/no）} \\
\quad \text{yes} \Rightarrow \text{保留为合格候选} \\
\quad \text{no} \Rightarrow \text{丢弃}
\end{cases}
\;\rightarrow\;
\text{10K Training} + \text{1K Validation}
$$

::: danger 这里有一个必须留白的数字
<mark class="hl-key">**论文没有披露原始候选池一共有多少条、过滤掉多少条、每条规则分别删除多少条，以及是否还存在其他未说明的数据清洗步骤。因此我们不能计算真实保留率。**</mark>
:::

---

## 11. 完整的六条 Validator 规则

附录 A.2.2 把 Validator 的 System Prompt 公开了。其最核心的约束是：<mark class="hl-key">**以下六项必须全部满足，才允许回答 yes**</mark>。

<mark class="hl-trick">**我按原文顺序逐条解释，不额外引入其他过滤条件。**</mark>

### 11.1 Single-image Describability：必须能描述一张独立图片

Prompt 应直接描述某个主体、场景、人像、物体或抽象艺术作品，而且最终能够被绘制成单张画面。

重点是 <mark class="hl-key">**single frame**</mark>：作者希望输入是一条能够直接执行的单图生成指令，<mark class="hl-trick">**而不是同时要求生成多张图片或完成一组额外交付任务**</mark>。

### 11.2 Visual-first Content：内容必须以视觉描述为主

大部分文本应该描述实际可以看到的东西。例如主体、场景、外观、构图、颜色、光照。风格、媒介和相机信息可以包含在内，但不是必须。

这条规则还有一个容易忽略的地方：作者<mark class="hl-key">**不是简单要求「只要与图片有关就行」，而是要求主要内容本身就是可视化描述**</mark>。

在 System Prompt 最后的自查问题中，作者进一步要求，输入应当<mark class="hl-trick">**主要是视觉场景描述，而不是幻灯片、营销内容、Logo 或文字叠加任务**</mark>。

::: warning 这条过滤规则实际上对设计类和文字类请求比较严格
<mark class="hl-key">**并不意味着所有海报或文字图片一定被删除，但它们并不是这套 Validator 优先保留的任务类型。**</mark>
:::

### 11.3 No Dependency on Hidden Context：不能依赖上下文

如果 Prompt 需要查看一张参考图片、之前生成的图片，或者历史聊天内容才能理解，就不合格。

原文明确举了多种常见指代，例如：

```
in the picture
in this photo
provided image
as shown above/below
```

因此，作者主要保留的是<mark class="hl-key">**独立文生图请求，而不是将图像编辑 Prompt 转换成文生图 Prompt**</mark>。

<mark class="hl-trick">**这里没有公开任何自动重写、补全参考信息的步骤。**</mark>

### 11.4 No Strong External Knowledge Dependency：不能过度依赖冷门实体知识

::: warning 这条非常值得注意，因为它**不是「不能出现任何专有名词」**
:::

作者实际写了两个判断条件：

| 判定 | 条件 |
| :--- | :--- |
| **无效** | Prompt 主要依赖某个小众或冷门的专有实体才能确定画面 |
| **有效** | 专有名词广为人知，<mark class="hl-key">**或者即使不知道这个实体是什么，Prompt 本身依然提供了足够的普通视觉描述**</mark> |

也就是说，<mark class="hl-trick">**Validator 有例外，不是看到人物名、地名或品牌名就一律删除。**</mark>

::: warning 判定边界完全交给 LLM
这条规则<mark class="hl-key">**没有提供封闭的实体词表，也没有规定什么程度的知名度算作 widely recognizable**</mark>。实际判断由 Gemini-3.5-Flash 完成。
:::

### 11.5 Natural, Fluent English：必须是自然、流畅的英文

这一条<mark class="hl-key">**没有模糊空间**</mark>：作者明确要求 Prompt 使用自然、流畅的英语句子或短语。

因此，<mark class="hl-trick">**非英文 Prompt 在这一阶段被过滤，不属于本实验的 RL 数据分布。**</mark>

### 11.6 Sufficient Detail：必须具有足够视觉细节

这是第六条，也是<mark class="hl-key">**比较主观的一条**</mark>。

作者要求 Prompt 能支持一幅内容具体、清晰的图像；<mark class="hl-trick">**极短、通用且没有附加描述的 Prompt 会被拒绝**</mark>。

::: danger 不能把它当成硬阈值规则
作者<mark class="hl-key">**没有规定最少多少个 Token、多少个单词，或者最少包含几个物体和属性**</mark>。因此<mark class="hl-trick">**不能把它理解成诸如「少于 20 个 Token 就删除」这样的硬规则**</mark>。

它实际上是让 Validator <mark class="hl-key">**判断文本的视觉信息是否充分**</mark>。
:::

---

## 12. Validator 的实际输入输出格式

除了六条标准，附录还明确规定了模型应该怎样工作。

它使用**固定的 System Prompt**，将候选 RL Prompt 交给 `gemini-3.5-flash` 判断。模型只允许返回两个结果：

```
yes
no
```

<mark class="hl-key">**不允许解释、不允许标点、不允许其他文本，也不允许改变格式。**</mark>

这意味着该模块的输出是<mark class="hl-trick">**二值过滤结果，不是质量分数，也不是多标签分类**</mark>。

论文<mark class="hl-key">**没有说 Validator 要生成修改后的 Prompt**</mark>。对于不满足条件的数据，它的处理方式是<mark class="hl-trick">**拒绝，而不是自动补充、改写后再尝试通过**</mark>。

### 12.1 六条规则之外的两类特别拒绝条件

原文另设了一个 `INVALID ("no") IF ANY ARE TRUE` 部分，特别强调两类情况：

| 编号 | 条件 | 说明 |
| :--- | :--- | :--- |
| **A** | **图像编辑或依赖外部图片** | 比如要求修改已有图片中的物体、引用上面的图片，或者根据提供的图片进行生成。<mark class="hl-key">**这相当于再次强调第三条规则**</mark> |
| **B** | **Prompt 太短，无法支持具体的图像生成** | <mark class="hl-key">**这相当于再次强调第六条规则**</mark> |

最后还有一个整体自查：<mark class="hl-trick">**文本本身是否足以让绘图者画出一张具体图片？其主要内容是不是视觉场景，而非幻灯片、营销、Logo 或文字叠加等请求？**</mark>

### 12.2 附录结构的准确概括

$$
\begin{aligned}
&\textbf{固定 Validator System Prompt} \\
&\quad \bullet\ \text{目标：判断是否是一条独立、可生成的单图 Prompt} \\
&\quad \bullet\ \textbf{全部满足六条正向条件} \\
&\quad \bullet\ \textbf{不得触发两类特别强调的拒绝条件 (A / B)} \\
&\quad \bullet\ \textbf{通过整体可绘制性与视觉优先的最终自查} \\
&\quad \bullet\ \textbf{只输出 } \texttt{yes} \textbf{ 或 } \texttt{no}
\end{aligned}
$$

::: warning 这里并没有出现我们之前研究 RL 时常见的筛选手段
<mark class="hl-key">**Reward Range、Reward Variance、图像 IQA、CLIP Score、Base Model Probe —— 这些都不是本文 A.2.2 所披露的 Prompt 过滤方法。**</mark>
:::

---

## 13. 训练数据与验证数据如何使用？

过滤后，论文得到：

| 数据集 | 数量 | 用途 |
| :--- | :---: | :--- |
| **RL Training Prompts** | 10,000 | 在线 RL 时采样 Prompt |
| **Arena Validation Prompts** | 1,000 | 评估训练中的 Checkpoint |
| **100K Unfiltered Pool** | 100,000 | 专门用于过滤有效性对照实验 |

<mark class="hl-key">**作者没有公布训练与验证的具体抽样种子、时间切分方式及近重复去重方法。**</mark>

在 RL 训练中，对于 FLUX.2-dev，每次使用 <mark class="hl-key">**48 个 Prompt Group**</mark>，每个 Prompt 生成 <mark class="hl-key">**24 张图片**</mark>，因此每次更新对应：

$$
48 \times 24 = 1{,}152 \ \text{张生成图像}
$$

作者对同一运行中的第 **30、60、90、120、150** 个 Checkpoint 进行验证，最后报告验证表现最佳的 Checkpoint。验证时使用 **Gemini-3.5-Flash** 进行相对于冻结 Base Model 的成对比较，并使用 **GPT-5.4** 做另一套 Judge 检验。

::: danger 1K Arena Prompt 不是完全独立的最终测试集
<mark class="hl-key">**1K Arena Prompt 是开发验证集，参与了 Checkpoint 选择**</mark>，不是完全独立的最终测试集。

<mark class="hl-trick">**我们此前讨论的数据泄露风险，在这里应该继续保留关注。**</mark>
:::

::: tip 特别要区分两套数据
论文另外提到的<mark class="hl-key">**约 500 万对 Arena 用户投票，是训练 Preference Reward Model 的数据，不是这 10K 条 RL Prompt。两套数据不能混为一谈。**</mark>
:::

---

## 14. 过滤是否真的有效？100K 与 10K 的对照实验

附录 A.2.2 最后有一段专门讨论过滤效果。

作者做了两次 Stage 1 RL 训练，采用 Arena Reward Model，分别使用：

| 对照 | Prompt 数据 | Reward |
| :--- | :--- | :--- |
| **实验 A** | 100K 未过滤 Arena Prompt | Arena RM |
| **实验 B** | 10K 过滤后 Arena Prompt | Arena RM |

两次实验使用相同的 Gemini Judge 评价协议。论文报告：

$$
\boxed{
\text{实验 A 相对实验 B 的胜率差} = -0.0360
}
$$

即<mark class="hl-key">**未过滤数据训练的模型在配对评价中，比过滤数据训练的模型低 3.60 个百分点**</mark>。

作者据此认为，<mark class="hl-trick">**低质量 Prompt 会在相同的优化压力下损害 RL 训练，因此后续实验全部采用过滤后的 Prompt 集合**</mark>。

### 14.1 这个实验到底证明了什么？

它确实提供了证据：<mark class="hl-key">**在论文的训练与评测设置下，使用过滤后的 10K Prompt 比使用未过滤的 100K Prompt 得到更好的结果。**</mark>

但是有两个重要细节不能忽略。

::: warning 第一，实验同时改变了数据数量和数据内容
100K 与 10K<mark class="hl-key">**不仅质量不同，规模也差了十倍**</mark>。

因此<mark class="hl-trick">**它不能单独证明「更严格的质量过滤一定比更多数据好」，也不能精确分离数据规模、Prompt 分布与质量筛选的作用**</mark>。
:::

::: danger 第二，论文没有在这一段给出完整的过滤实验配置
例如<mark class="hl-key">**没有单独列出这两次运行的逐步训练曲线、独立的多随机种子统计、每个类别的保留率，或者更细致的过滤阈值消融**</mark>。

正文 §4.1 说未过滤数据导致训练表现更快退化，但附录提供的定量信息主要就是上述 $0.0360$ 的差距。<mark class="hl-trick">**我们不能据此编造具体第几步开始退化、峰值出现在哪里。**</mark>
:::

---

## 15. 哪些工程细节论文没有公开？

<mark class="hl-key">**不能遗漏，也不能乱加，所以把公开内容的边界单独列出来。**</mark>

| 具体问题 | 原文是否披露 |
| :--- | :--- |
| RL Prompt 来自 Arena 真实用户分布 | ✅ 明确 |
| Validator 模型 | Gemini-3.5-Flash |
| Validator 完整 System Prompt | ✅ 明确公开 |
| 判定形式 | `yes` / `no` |
| 六条主要筛选规则 | ✅ 明确公开 |
| 训练和验证 Prompt 数量 | 10K / 1K |
| 原始 Arena Prompt 池总量 | ❌ 未公开 |
| 原始 Prompt 的采集时间范围 | ❌ 未针对该 Prompt 池明确列出 |
| 各种过滤原因分别删除多少条 | ❌ 未公开 |
| Validator 推理 Temperature / Sampling 参数 | ❌ 未公开 |
| 每条 Prompt 是否重复判断、多数投票 | ❌ 未公开 |
| Validator 的人工标注准确率、误杀率 | ❌ 未公开 |
| Prompt 是否做语义去重 | ❌ 未明确说明 |
| 训练与验证的具体切分算法 | ❌ 未公开 |
| 是否根据 Base Model 的 Rollout Reward 进一步预筛 | ❌ 本节未披露这种做法 |
| 100K 未过滤与 10K 过滤后的训练比较 | ✅ 有，报告 0.0360 差距 |

<mark class="hl-key">**这意味着我们可以按照公开文本复现主要规则，但还不能声称已经拿到了 Arena 的完整 RL Prompt 数据生产流水线。**</mark>

---

## 16. Part 2 的结论

::: tip 作者的 RL Prompt 生产并不复杂
从真实用户请求采样，使用 <mark class="hl-key">**Gemini-3.5-Flash 做一次非常严格的文本级可用性过滤**</mark>，得到 10K 训练和 1K 验证 Prompt。

它<mark class="hl-trick">**没有在这一阶段设计复杂的能力 Tag、没有根据当前 Base Model 计算 Reward 方差，也没有披露动态 Prompt 课程学习**</mark>。
:::

<mark class="hl-key">**这篇论文真正复杂的地方，是接下来如何为这 10K 条 Prompt 离线生成细粒度的评价规则。**</mark>

::: danger 一个值得保留的问题
它采用较严格的<mark class="hl-key">**视觉优先、英文和充分细节**</mark>要求，实际上<mark class="hl-trick">**缩窄了训练 Prompt 的覆盖范围**</mark>。

因此论文中「10K 过滤后更好」的结论，<mark class="hl-key">**应限定在其任务与评测条件下理解，不能直接推广为通用图像基模应该删除所有短 Prompt、非英文 Prompt 和设计类请求**</mark>。
:::

**Part 2 到这里结束。** 对应的 §4.1 和 A.2.2 已经覆盖，包括完整筛选逻辑、输出格式、两类特别拒绝条件、数据规模、用途、对照实验与未披露细节。

**下一部分 Part 3：Human Preference Reward Model 的数据生产与训练。** 那里我们会具体研究 Arena 的约 500 万对人类投票怎样组织成 Reward Model 训练数据，以及 Qwen3.6-27B 如何通过 Bradley–Terry Loss 学习人类偏好。

---

# Part 3｜Human Preference Reward Model 学习笔记

> **对应原文**：§3.2.1、附录 A.1、§4.3（RM Scaling）

## 17. 核心问题

作者通过 Arena 上<mark class="hl-key">**同一个 Prompt 对两张图的用户投票**</mark>，训练一个输入 $(\text{prompt}, \text{image})$、输出实数 $R(p, x)$ 的 Reward Model。

<mark class="hl-trick">**500 万左右的偏好对用于训练 RM，不是直接训练 DiT。**</mark>

::: tip 关键区分
<mark class="hl-key">**Pairwise 是监督形式；单图打分是模型的输出结构。**</mark>

这两件事经常被混为一谈：训练时的监督信号天然是成对的（winner / loser），但**推理时模型只需要输入一张图就能输出一个实数**。§19 会说明为什么这在结构上是成立的。
:::

---

## 18. 数据

| 项 | 设置 |
| :--- | :--- |
| 来源 | Arena 人类偏好投票 |
| 时间跨度 | **2025-04-16 → 2026-06-07** |
| 覆盖模型 | **100+ 个 T2I 模型** |
| 正文规模 | 约 **500 万** 对 |
| 数据规模消融中的完整数据 | 约 **560 万（5.6M）** 对 |

记录可表示为：

$$
(p,\ \text{winner\_image},\ \text{loser\_image})
$$

<mark class="hl-key">**人类只告诉模型前者优于后者，不给绝对 0–10 分。**</mark>

::: warning 清洗规则的披露程度
论文<mark class="hl-key">**只提及剔除不认真投票者和低质量 Prompt 对应的记录，未公布可复现的详细清洗规则**</mark>。
:::

---

## 19. 模型结构：为什么能给一张图打分？

$$
\text{Prompt} + \text{单张 Image}
\;\rightarrow\;
\text{Qwen3.6-27B}
\;\rightarrow\;
\text{末尾 } \texttt{\textless{}|cls|\textgreater{}} \text{ Hidden State}
\;\rightarrow\;
\text{Linear Head}
\;\rightarrow\;
\text{Score}
$$

数学形式：

$$
h = \mathrm{Encoder}(p, x)_{\mathrm{cls}},
\qquad
R(p, x) = w^{\top} h + b \in \mathbb{R}
$$

::: info 为什么单张图也能拿到一个标量
<mark class="hl-key">**即使未经训练，线性层也能给单张图输出一个数字**，只不过当时没有可靠的偏好意义。</mark>

<mark class="hl-trick">**训练使这一评分函数学会如何正确排序。**</mark>
:::

---

## 20. Pairwise 训练过程

$$
\begin{aligned}
r_w &= R(p, x_w) && \text{（Winner 单独前向）} \\
r_l &= R(p, x_l) && \text{（Loser 单独前向，两次共享同一模型参数）}
\end{aligned}
$$

**Bradley–Terry 概率模型：**

$$
P(x_w \succ x_l \mid p) = \mathrm{sigmoid}\bigl(r_w - r_l\bigr)
$$

**Bradley–Terry Loss：**

$$
\mathcal{L} = -\log \mathrm{sigmoid}\bigl(r_w - r_l\bigr) = \mathrm{softplus}\bigl(r_l - r_w\bigr)
$$

**对评分的梯度：**

$$
\frac{\partial \mathcal{L}}{\partial r_w} = \mathrm{sigmoid}(r_w - r_l) - 1 < 0,
\qquad
\frac{\partial \mathcal{L}}{\partial r_l} = 1 - \mathrm{sigmoid}(r_w - r_l) > 0
$$

::: info 这两个梯度的符号本身就是在陈述监督信号
<mark class="hl-key">**梯度下降使赢家相对更高、输家相对更低**</mark>——$r_w$ 的梯度恒为负（上升），$r_l$ 的梯度恒为正（下降）。
:::

**例子**：人类认为 $A > B$，但模型给 $A = 3$、$B = 5$，Loss 较大；反向传播就会调整评分函数，<mark class="hl-trick">**让 $A$ 的相对得分提高**</mark>。

::: tip 推理不需要两张图
$$
\mathrm{score} = R(\text{prompt},\ \text{new\_image})
$$

<mark class="hl-key">**只需一张图即可**</mark>，这也是它能直接充当 RL Reward 的前提。
:::

---

## 21. Score 范围（本轮问答新增）

| 类型 | 范围 | 解释 |
| :--- | :--- | :--- |
| **原始单图 Score** $R(p,x)$ | $(-\infty, +\infty)$ | Linear Head 的<mark class="hl-key">**实数 Logit，不是 0–10 分**</mark> |
| **偏好概率** $\mathrm{sigmoid}(r_A - r_B)$ | $(0, 1)$ | A 相对 B 获胜的概率 |
| **最终 NFT 权重** $\tilde{r}_i$ | $[0, 1]$ | 多 Reward 组合、组相对归一化后映射而来 |

**示例**：若 $R(A) = 3$、$R(B) = 1$，则

$$
P(A \succ B) = \mathrm{sigmoid}(3 - 1) = \mathrm{sigmoid}(2) \approx 0.881
$$

<mark class="hl-trick">**`Score = 3` 本身并不意味着 $3/10$，也不是 30%。**</mark>

::: danger 原始分数不是唯一的绝对质量尺度
BT Loss 只依赖两个 Score 的<mark class="hl-key">**差**</mark>：

$$
\mathcal{L} = \mathrm{softplus}\bigl(r_l - r_w\bigr)
\;\Longrightarrow\;
\mathcal{L}\bigl(r_w + c,\ r_l + c\bigr) = \mathcal{L}\bigl(r_w,\ r_l\bigr)
$$

<mark class="hl-key">**同时给两个 Score 加上 $100$，概率和 Loss 都不变。**</mark>

所以<mark class="hl-trick">**原始分数并非唯一的绝对质量尺度，不同 Prompt 下的分数不宜直接比较。**</mark>
:::

---

## 22. RM 训练参数

| 项 | 设置 |
| :--- | :--- |
| Backbone | **Qwen3.6-27B**，末尾 `<\|cls\|>` 接 Linear Head |
| Loss | **Bradley–Terry** |
| Epoch | **1 Epoch** |
| **Global Batch** | 附录 A.1 为 **1,032 pairs**；正文 §3.2.1 为 **1,024**，<mark class="hl-key">**原文不一致**</mark> |
| LR | $2\times10^{-6}$，**Constant**，**No Warmup** |
| Max Length | **8,192 Tokens** |

**图像增强**（宽高比修正后应用于每张训练图像）：

| 变换 | 幅度 |
| :--- | :--- |
| Brightness / Contrast / Saturation | 各 $\pm 0.05$ |
| Hue | $\pm 0.025$ |
| Gaussian Noise | $\sigma = 0.005$（像素归一到 $[0,1]$） |

**Collator 格式**：

$$
\underbrace{\text{固定 System Instruction}}_{\text{reward-model system}}
\;+\;
\underbrace{\text{User Prompt}}_{p}
\;+\;
\underbrace{\text{Winner 或 Loser Image}}_{x}
\;+\;
\texttt{\textless{}|cls|\textgreater{}}
$$

---

## 23. RM 数据规模与下游 RL 表现

| RM 人类偏好数据 | MMRBv2 Stage-1 RL 胜率 |
| :---: | :---: |
| $\sim 10$K pairs | **53.7%** |
| $\sim 100$K pairs | **60.7%** |
| $\sim 500$K pairs | **61.1%** |
| $\sim 5.6$M pairs | **64.2%** |

![Arena-RL Figure 3：下游 RL 表现随奖励模型训练数据量缩放。横轴为训练 RM 所用人类偏好对数（对数刻度 10⁴–10⁷），纵轴为 Stage-1 RL 模型相对冻结 base 的 MMRBv2 Gemini 胜率，灰虚线为 base parity 0.50。四个点依次为 10K→0.537、100K→0.607、500K→0.611、5.6M→0.642。曲线在 10⁴→10⁵ 陡升后进入长平台（10⁵→5×10⁵ 仅 +0.004），直到接近 5.6M 才出现第二次跃升。](/arena-fig3-rm-data-scaling.png)

::: warning 这里 Scale 的是什么
<mark class="hl-key">**Scale 的是训练 RM 的偏好数据**</mark>，<mark class="hl-trick">**不是 RL Prompt，也不是 Rollout 数量**</mark>。

它与 Part 2 的 10K RL Prompt 是<mark class="hl-key">**完全不同的两套数据**</mark>：前者是「人类选哪张图」，后者是「用户想让模型生成什么」。
:::

---

## 24. 适用边界和披露不足

::: warning 适用边界
偏好 RM 能提供<mark class="hl-key">**整体美学 / 构图 / 视觉质量信号，但不能保证计数、关系、文字等逐条遵循**</mark>，因此论文进一步使用 Rubric Faithfulness / Constraint Reward。
:::

::: danger 披露不足清单
- 投票清洗规则
- 完整 System Prompt
- 优化器与 GPU
- <mark class="hl-key">**RM 是否全参训练**</mark>
- 完整校准指标
- 数据近重复审计
:::

---

## 25. 最简伪代码

```python
r_w = reward_model(prompt, winner_image)
r_l = reward_model(prompt, loser_image)
loss = F.softplus(r_l - r_w)
loss.backward()
optimizer.step()
optimizer.zero_grad()

with torch.no_grad():
    score = reward_model(prompt, generated_image)
```

> **一句话记忆**：<mark class="hl-key">**模型结构决定单图输出实数，Pairwise BT 训练决定这个实数怎样反映相对偏好。**</mark>

**原文**：https://arxiv.org/html/2610.02967

---

# Part 4：Faithfulness Checklist

> **对应原文**：§3.2.2、附录 A.3.1–A.3.2，以及 Figure 4 中的真实案例

这一部分<mark class="hl-key">**是整篇论文中与你现在研究 RL 训练数据生产最相关的内容之一**</mark>。

在 Part 3 中，作者利用人类的 Pairwise Preference 数据训练一个 Reward Model，输入 Prompt 和图片，输出一个分数。但这个 RM 有一个局限：<mark class="hl-trick">**它可能认为一张图片很好看，却没有发现图片漏掉了用户要求的物体、数量或空间关系**</mark>。

因此 Part 4 要解决的是：

<mark class="hl-key">**不依赖人类逐张标注，如何自动生成一套针对每条 Prompt 的检查标准，并将生成图片的指令遵循程度变成一个数值 Reward？**</mark>

$$
\boxed{
\text{Prompt} \xrightarrow{\text{Gemini}} \text{Faithfulness Checklist}
}
\qquad
\boxed{
(\text{Prompt},\ \text{Generated Image},\ \text{Checklist}) \xrightarrow{\text{Qwen VLM}} R_{\mathrm{faith}} \in [0, 1]
}
$$

::: warning 注意，这是两个不同的阶段
<mark class="hl-key">**Gemini 在训练之前生成检查问题，Qwen 在 RL 过程中根据生成图片回答这些问题。**</mark>
:::

---

## 26. 完整流程：离线生产问题，在线执行检查

**离线阶段**：每条训练 Prompt 执行一次。

$$
\begin{aligned}
&\textbf{10K 条 RL Training Prompts} \\
&\quad \downarrow\ \text{Gemini-3-Pro-Preview} \\
&\quad \bullet\ \textbf{将每条 Prompt 拆成原子 Yes/No 问题} \\
&\quad \bullet\ \textbf{生成问题之间的依赖关系 DAG} \\
&\quad \downarrow \\
&\quad \textbf{将 Checklist 和 Prompt 一起保存}
\end{aligned}
$$

**在线阶段**：每次 RL Rollout 后执行。

$$
\begin{aligned}
&\textbf{当前 DiT 生成图片} \\
&\quad \downarrow\ \text{Qwen3.6-27B Judge · vLLM} \\
&\quad \bullet\ \textbf{按 DAG 的依赖层级批量回答问题} \\
&\quad \bullet\ \textbf{父问题失败时，子问题直接记为 No} \\
&\quad \downarrow \\
&R_{\mathrm{faith}} = \frac{N_{\mathrm{yes}}}{N_{\mathrm{questions}}}
\end{aligned}
$$

原文明确说明：<mark class="hl-key">**所有 Checklist 和 Intent Label 都是离线生成、随 Prompt 保存的；在 RL 训练循环里，只调用通过 vLLM 部署的 Checklist Judge**</mark>。

::: warning 这里说的是 Rubric 生成流程
<mark class="hl-trick">**不代表 Preference RM 也由该 Judge 取代**</mark>，两者是并行的不同模块（见 Part 3）。
:::

---

## 27. Checklist 到底如何生成？

作者对 10K 条训练 Prompt，每条调用一次 `gemini-3-pro-preview`，生成包含明确依赖关系的原子问题。原文规定了<mark class="hl-key">**七个关键要求**</mark>：

1. <mark class="hl-key">**每个问题只检查一个视觉事实。**</mark>
2. <mark class="hl-key">**检查对象包括物体存在、颜色、数量、空间关系、动作、风格、文字内容等。**</mark>
3. <mark class="hl-key">**覆盖原始 Prompt 中的每一个细节。**</mark>
4. <mark class="hl-key">**从核心主体逐渐覆盖到次要风格细节。**</mark>
5. <mark class="hl-key">**不能添加原始 Prompt 没有提到、也没有合理隐含的检查要求。**</mark>
6. <mark class="hl-key">**物体是否存在的检查放在 DAG 根节点；属性检查依赖对应物体的存在性。**</mark>
7. <mark class="hl-key">**两个物体之间的关系检查，要依赖两个物体都存在。**</mark>

::: warning 这七项来自附录 A.3.2 的描述
论文<mark class="hl-key">**没有在这里逐字刊出 Gemini 用于生成 Faithfulness Checklist 的完整 System Prompt**，而是概述其中强制执行的规则。</mark>

<mark class="hl-trick">**我们能复现设计原则，但不能声称已经拿到作者的完整生成模板。**</mark>
:::

### 实际生成了多少问题？

| 指标 | 数值 |
| :--- | ---: |
| **Faithfulness 问题总数** | **182,230** |
| **平均问题数 / Prompt** | **18.2** |
| **问题数中位数** | **17** |
| **单条 Prompt 最大问题数** | **70** |

<mark class="hl-key">**这个粒度比传统的 `counting`、`spatial` 等能力标签细得多**</mark>。一条 Prompt 不是只打一个能力 Tag，而是被转化成十几个可判断的具体条件。

::: danger 但 18.2 不是作者规定
<mark class="hl-key">**18.2 是实际统计均值，不是作者规定每条 Prompt 必须生成约 18 个问题**</mark>。短 Prompt 和复杂 Prompt 的实际问题数不同。
:::

---

## 28. 为什么要使用 DAG，而不是普通的问题列表？

<mark class="hl-key">**这里有一个很重要的问题：不同检查项之间并不是独立的。**</mark>

例如，一张图片要求有三个红苹果，只有先确认苹果存在，再判断数量和颜色才有意义。

假设没有苹果，但 VLM 被迫单独回答「苹果是不是红色」，<mark class="hl-trick">**就可能产生前后矛盾的结果**</mark>。

所以作者采用 **DAG**（Directed Acyclic Graph，有向无环图）表示问题之间的依赖关系。

### 真实案例：Mandala 涂色页

用户要求一张竖版涂色页，包含对称的曼陀罗图案、重复纹样，以及月亮、小骷髅、磁带和钥匙等元素，并要求边缘留白。

论文给出了这条 Prompt 对应的完整 Faithfulness Checklist，共有 <mark class="hl-key">**16 个问题**</mark>。

![Arena-RL Figure 4 Example B：8.5×11 英寸竖版涂色页（300 DPI、0.5 in margin）Prompt 的完整奖励包。左侧为 FLUX.2 base 与 Stage 1 的 as-served 出图。右侧上半为 presence-oriented Faithfulness DAG（F0–F15，括号内为 depends-on 索引，可见 F4 [0,3]、F7 [0,5]、F9 [0,8]、F11 [0,10] 均为多父节点），下半为 absence-oriented Constraint checklist（C0–C4）。元数据显示 intent: strict、style_lock: coloring book page、background_spec: plain white、detail_budget: minimal。](/arena-fig4b-reward-battery-mandala.png)

以下是论文真实检查项的中文概括，F 编号及依赖关系保持与原文一致。

| ID | 依赖 | 检查什么 |
| :--- | :--- | :--- |
| F0 | 无 | 是否有曼陀罗图案 |
| F1 | F0 | 图案是否对称 |
| F2 | F0 | 是否有重复纹样 |
| F3 | 无 | 是否出现月亮 |
| F4 | **F0, F3** | 月亮是否属于曼陀罗的一部分 |
| F5 | 无 | 是否出现骷髅 |
| F6 | F5 | 骷髅是否很小 |
| F7 | **F0, F5** | 骷髅是否属于曼陀罗的一部分 |
| F8 | 无 | 是否出现磁带 |
| F9 | **F0, F8** | 磁带是否属于曼陀罗的一部分 |
| F10 | 无 | 是否出现钥匙 |
| F11 | **F0, F10** | 钥匙是否属于曼陀罗的一部分 |
| F12 | F0 | 曼陀罗元素是否像摇篮悬挂装饰 |
| F13 | 无 | 是否为涂色书页面风格 |
| F14 | 无 | 是否为竖版 |
| F15 | 无 | 作品边缘是否留白 |

你看其中的 F4：

$$
\boxed{ F4 \leftarrow \{F0, F3\} }
$$

意思是：<mark class="hl-key">**只有图中存在曼陀罗（F0），并且有月亮（F3），才有必要判断「月亮是不是曼陀罗的一部分」（F4）**</mark>。

同理：

$$
F7 \leftarrow \{F0, F5\}
$$

也就是「骷髅是否属于曼陀罗」同时依赖曼陀罗和骷髅存在。

::: tip 这说明作者使用的并不只是简单的单父节点树结构
从数据关系上看，它是<mark class="hl-key">**允许一个问题有多个父节点的 DAG**</mark>。

这也解释了为什么<mark class="hl-trick">**正文称为 Tree-structured Checklist，但附录用的是更准确的 Dependency DAG**</mark>。
:::

### 如果父节点失败，会发生什么？

以 F4 为例，假设 Qwen 对 F0 的回答是 `yes`，对 F3 的回答是 `no`，那么：

$$
F4 = \mathrm{no}
$$

系统**不再请求 Qwen 判断 F4**，而是直接把它记为 No。

原因是：<mark class="hl-trick">**图片里既然没有月亮，自然不可能正确表现「月亮属于曼陀罗」的关系**</mark>。

这既<mark class="hl-key">**减少了无意义的 Judge 调用**</mark>，也<mark class="hl-trick">**避免模型出现「没有月亮，但月亮的位置正确」这样的逻辑矛盾**</mark>。

::: danger 论文明确规定的是父问题失败时自动记 No
但<mark class="hl-key">**并没有细化每一种 `irrelevant` 父节点应该如何传播**</mark>。因此<mark class="hl-trick">**实现时不能擅自把所有异常情况都视为作者已公布的规则**</mark>。
:::

---

## 29. 在线训练时，Qwen 怎样执行这张 DAG？

附录 A.3.1 实际公开了两个 Judge Prompt 的内容：

| 模式 | 输入 |
| :--- | :--- |
| **Single-question Judge** | 一次输入一个问题 |
| **Batched Judge** | 一次输入同一依赖层级的多个问题 |

两种模式都会给 Qwen 三个输入：

$$
\boxed{
(\text{Original Prompt},\ \text{Question(s)},\ \text{Generated Image})
}
$$

而且作者明确规定只能输出：

```text
yes
no
irrelevant
```

<mark class="hl-trick">**这和我们前面使用 Gemini 的二值 Prompt Validator（Part 2）不同**</mark>。这里 Judge 允许第三种结果，代表问题不适用或者无法从图片判断。

### 具体怎样 Batch？

继续使用上面的真实 DAG。

最开始，可以先评估所有不依赖其他问题的<mark class="hl-key">**根节点**</mark>：

| Level 0（根节点） | F0 | F3 | F5 | F8 | F10 | F13 | F14 | F15 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

把同一层中需要评估的问题批量提交给 Qwen，根据回答决定哪些子节点需要继续评估。

<mark class="hl-key">**后续层级**：F1、F2、F4、F6、F7、F9、F11、F12</mark> —— <mark class="hl-trick">**满足父节点条件的问题才真正调用 Judge；其他问题自动记为 No**</mark>。

::: warning 分层方式说明
这里的分层方式是根据论文 DAG 机制对 Example B 的结构展开，<mark class="hl-key">**论文没有逐一打印每次 vLLM 请求的运行日志**</mark>。
:::

作者的 Batched Judge 输出格式是一个 JSON 数组：

```json
[
  {"id": 0, "answer": "yes"},
  {"id": 3, "answer": "no"},
  {"id": 5, "answer": "irrelevant"}
]
```

<mark class="hl-trick">**这三个结果只是原文规定的 JSON 格式示意，不是对上述 Mandala 图片的实际评分。**</mark>

如果某个依赖层级只有一个待评价问题，作者使用 Single-question 版本作为回退，<mark class="hl-key">**此时直接返回单个单词，而不是 JSON 数组**</mark>。

::: tip 这里非常值得注意
<mark class="hl-key">**Qwen 的任务不是给出一个 0–10 的 Faithfulness 分数，而是对每个事实做判断。最终分数由程序统计计算。**</mark>
:::

---

## 30. 最终 Faithfulness Reward 怎么计算？

公式实际上非常简单：

$$
\boxed{
R_{\mathrm{faith}}(p, x) = \frac{1}{N_p} \sum_{j=1}^{N_p} \mathbf{1}\bigl[a_j = \mathrm{yes}\bigr]
}
$$

其中：

| 符号 | 含义 |
| :--- | :--- |
| $p$ | 原始用户 Prompt |
| $x$ | 当前 DiT 生成的一张图片 |
| $N_p$ | 该 Prompt 的 Checklist 问题总数 |
| $a_j$ | 第 $j$ 个问题的最终判定，<mark class="hl-key">**包括自动传播的 No**</mark> |
| $\mathbf{1}[\cdot]$ | 满足条件记 1，否则记 0 |

因此：

$$
\boxed{ R_{\mathrm{faith}} \in [0, 1] }
$$

### 举个完整计算例子

假设我们使用上述 16 个检查问题，评估某张新生成的 Mandala 图片。以下回答是计算演示，<mark class="hl-trick">**不是作者对 Figure 4 图片的实际评分**</mark>。

| 问题组 | 结果 |
| :--- | :--- |
| F0、F1、F2：曼陀罗存在、对称、重复图案 | 3 个 Yes |
| F3、F4：月亮及其所属关系 | 2 个 Yes |
| F5、F6、F7：骷髅及其所属关系 | 3 个 Yes |
| F8、F9：磁带及其所属关系 | 2 个 No |
| F10、F11：钥匙及其所属关系 | 2 个 Yes |
| F12、F13：悬挂装饰特征、涂色书风格 | 2 个 Yes |
| F14、F15：竖版、留白 | 2 个 Yes |

这样总共 16 个问题，14 个 Yes：

$$
R_{\mathrm{faith}} = \frac{14}{16} = 0.875
$$

::: danger 一个必须注意的计数细节
假设 F8 判断没有磁带，那么 F9 就无需再次调用 Qwen，<mark class="hl-key">**自动记 No**</mark>。

但<mark class="hl-trick">**F9 仍然属于原 Checklist 的问题，不能因为被跳过就从总问题数中删除**</mark>。

这也意味着<mark class="hl-key">**一条 Prompt 中某个关键主体的缺失，可能一次造成多个相关检查项失败**</mark>。
:::

### `irrelevant` 怎么算？

论文的 Judge 模板允许这个回答，但<mark class="hl-key">**没有单独规定复杂的缺失值归一化策略**</mark>。

按照作者公布的「全部问题中 Yes 所占比例」，<mark class="hl-trick">**它不属于 Yes；不能自行假定 `irrelevant` 会被排除出分母**</mark>。

### 有没有给重要问题更大的权重？

<mark class="hl-key">**没有披露这种设计。**</mark>

这篇论文采用的是问题 Yes 比例，而不是：

$$
R_{\mathrm{faith}} = \sum_j w_j\, \mathbf{1}\bigl[a_j = \mathrm{yes}\bigr]
$$

也就是说，<mark class="hl-trick">**作者没有说明对「主体存在」「文字准确性」「风格」或「空间关系」分别赋予不同权重**</mark>。

::: tip 从我们研究 RL 数据的角度
这很重要，因为<mark class="hl-key">**如果 Gemini 把某个简单属性拆成五个问题，而把某个复杂关系只拆成一个问题，那么最终比例会受到 Checklist 拆分粒度影响**</mark>。

但<mark class="hl-trick">**这属于对评分机制的分析，不是作者已经验证过的失效案例**</mark>。
:::

---

## 31. 论文还给了另一个真实案例：Logo

Figure 4 的 Example A 是用户要求制作一个带有 `MaurzTV` 字样、圆形、超现代、具有太空和科学元素、无背景的 Logo。

![Arena-RL Figure 4 Example A：MaurzTV 圆形 Logo Prompt 的完整奖励包。左侧两栏为 base 与 Stage 1 的 as-served 出图。右侧为 presence-oriented Faithfulness F0–F7（注意 F1「是否有准确的 MaurzTV 文字」是根节点，F2–F6 全部依赖 F0）与 absence-oriented Constraint C0–C4。C3 显式检查「是否保持 Logo 风格而没有偏向摄影写实或电影化渲染」，C4 是兜底的「是否没有出现 Prompt 未要求的显著物体、场景或装饰细节」。](/arena-fig4a-reward-battery-logo.png)

作者生成了 8 个 Faithfulness 问题：

| ID | 检查内容 | 依赖 |
| :--- | :--- | :--- |
| F0 | 是否为 Logo | 无 |
| F1 | 是否有准确的 `MaurzTV` 文字 | 无 |
| F2 | Logo 是否为圆形 | F0 |
| F3 | 是否有太空相关元素 | F0 |
| F4 | 是否有科学相关元素 | F0 |
| F5 | 是否呈现超现代风格 | F0 |
| F6 | 是否具有精致的设计感 | F0 |
| F7 | 是否没有背景，或背景完全简单 | 无 |

::: tip 这个真实案例可以看出作者允许的问题粒度
不仅有非常明确的文字、形状、主体，<mark class="hl-key">**也包括 `ultra-modern`、`sophisticated` 这样的视觉风格判断**</mark>。它们最终都由同一个 Qwen3.6-27B Judge 判断。
:::

::: warning 这里还要区分一件事
Faithfulness 可以检查场景文字是否符合要求，但<mark class="hl-key">**不能因此认为作者只使用 Faithfulness 就完成了所有 OCR 优化**</mark>。论文在 Ideogram-4 实验中还单独加入了 **Graded OCR Reward**，这属于 Part 7。
:::

---

## 32. 这一部分有哪些实现细节没有公开？

| 技术细节 | 披露程度 |
| :--- | :--- |
| Checklist 生成模型 | `gemini-3-pro-preview` |
| 生成时机 | 每条训练 Prompt 离线调用一次 |
| 问题设计规则 | ✅ 明确说明 |
| 依赖结构 | ✅ 明确使用 DAG |
| 问题总数、平均、中位数、最大值 | ✅ 全部给出 |
| 生成 Checklist 的完整 System Prompt | ❌ 未逐字公开 |
| 完整 DAG JSON Schema | ❌ 未公开 |
| Judge 模型 | `Qwen3.6-27B` |
| Judge 部署方式 | vLLM |
| Judge 的单问题与批量 Prompt | ✅ 完整公开 |
| Judge 输出 | `yes`、`no`、`irrelevant` |
| DAG 失败传播 | 父节点失败，子节点自动 No |
| Faithfulness Reward | Yes 问题占全部问题的比例 |
| Judge 的 Temperature、图像输入分辨率 | ❌ 未在对应章节交代 |
| JSON 解析失败时的重试机制 | ❌ 未公开 |
| Faithfulness Checklist 的人工准确率审计 | ❌ 未公开 |
| 每项检查是否单独加权 | ❌ 未披露，公开公式采用等权比例 |

::: warning 这最后一点尤其需要记住
论文后面展示了 Intent Classification 的人工审核准确率，但那属于 <mark class="hl-key">**Constraint 的 Gating Label**</mark>，<mark class="hl-trick">**并不是对 182,230 个 Faithfulness 问题准确性的系统人工审计。不能混淆。**</mark>
:::

---

## 33. 从文生图 RL 数据生产角度，Part 4 最重要的结论

我们已经看到作者实际上建立了两层 RL 数据。

**第一层是普通 Prompt：**

```text
Prompt = "A circular space-themed logo with the text MaurzTV..."
```

**第二层则是围绕这条 Prompt 自动生成的、可以执行的评价监督：**

```json
{
  "prompt": "A circular space-themed logo with the text MaurzTV...",
  "faithfulness": [
    {"id": 0, "question": "Is this a logo?", "depends_on": []},
    {"id": 1, "question": "Is the text MaurzTV visible?", "depends_on": []},
    {"id": 2, "question": "Is the logo circular?", "depends_on": [0]}
  ]
}
```

<mark class="hl-key">**这是依据论文 DAG 规则整理的简化存储示意，并非作者公开的原始 JSON Schema。**</mark>

有了第二层数据，在线 RL 才能针对每张生成图片计算明确的 Faithfulness Reward。整个过程可以概括为：

$$
\boxed{
\underbrace{p \rightarrow Q(p)}_{\text{离线数据生产}}
\qquad
\underbrace{(p, x, Q(p)) \rightarrow R_{\mathrm{faith}}}_{\text{在线奖励计算}}
}
$$

::: tip Part 4 的关键
<mark class="hl-key">**关键不是「用 Qwen 给图片打分」，而是把每条自然语言 Prompt 转化为一套具有依赖关系、可以重复执行的评价程序。**</mark>

但它也带来一个新的问题：<mark class="hl-trick">**Checklist 自身是否正确、是否覆盖全部要求、是否存在冗余，以及 Judge 是否稳定。这些都是训练数据质量的一部分，不能因为使用了强 VLM 就默认可靠。**</mark>
:::

另外，<mark class="hl-key">**Faithfulness 主要检查用户要求出现的东西有没有正确出现，并不系统惩罚用户没有要求却被额外加入的东西**</mark>。

**下一部分 Part 5：Constraint Reward 与用户意图分类。** 就要研究作者如何识别哪些 Prompt 必须严格遵守内容边界，以及为什么只对 `strict` Prompt 启用额外的 Constraint Reward。

---

# Part 5：Constraint Reward 与用户意图分类（Intent Gating）

> **对应原文**：§3.2.3、§3.3.2、附录 A.3.3、A.3.6；结合 Figure 4 和 §4.3 的消融

这一部分重点回答一个问题：既然 Part 4 已经通过 Faithfulness Checklist 检查生成图像是否满足用户要求，<mark class="hl-key">**为什么还需要 Constraint Reward？**</mark>

答案是：<mark class="hl-trick">**满足用户要求的内容，不代表没有生成用户不希望出现的内容**</mark>。

而且作者发现，<mark class="hl-key">**Constraint Reward 如果使用不当，还可能压制模型合理的创作能力**</mark>。所以这一部分真正的关键并不仅仅是生成 Constraint Checklist，而是<mark class="hl-trick">**判断哪些 Prompt 应该使用它**</mark>。

---

## 34. Faithfulness 和 Constraint 到底有什么区别？

假设用户要求：

> *Generate a minimalist logo featuring a single black circle on a plain white background.*

即生成一个极简 Logo，只有一个黑色圆形，背景为纯白色。

模型生成的结果包含黑色圆形、白色背景，但<mark class="hl-key">**另外增加了一圈金色装饰、一些文字和其他图案**</mark>。

此时 Faithfulness 可能给出较高的分数，因为它检查的是「要求出现的东西有没有出现」。但问题是，<mark class="hl-trick">**这些额外装饰违反了极简设计的要求**</mark>。

| | **Faithfulness Reward** | **Constraint Reward** |
| :--- | :--- | :--- |
| 核心方向 | **Presence-oriented** | **Absence-oriented** |
| 主要关注 | 应该出现的东西是否出现 | 不应该出现的东西是否被加入 |
| 典型检查 | 圆形有没有？颜色对不对？ | 有没有多余文字、背景、装饰？ |
| 是否所有 Prompt 都启用 | 是 | <mark class="hl-key">**不是，仅 Strict**</mark> |

::: warning 这并不意味着 Constraint 只能检查「没有什么」
论文实际生成的 Constraint 问题也可能重复检查文字、形状或背景是否正确。它的总体设计意图是<mark class="hl-trick">**控制与用户预期不一致的额外内容和风格漂移**</mark>。
:::

---

## 35. 核心设计：为什么需要 Strict / Neutral / Open 三分类？

论文 §3.3.2 给了一个很直接的例子：

> *Create an image of a country road in an Impressionist style.*

假如模型额外生成一名路人，或者道路旁有几朵鲜花，这算不算错误？

作者的判断是：<mark class="hl-key">**不一定**</mark>。用户并没有要求只允许出现道路，印象派绘画本来就留有合理的创作空间。

所以，<mark class="hl-trick">**不能把「Prompt 没有明确提到的任何物体」都视为错误。否则模型可能越来越倾向于空白背景、稀少物体和极简构图**</mark>。

为此，作者定义三个 Intent 类别：

| Intent | 原文定义 | Constraint Checklist 生产方式 |
| :--- | :--- | :--- |
| `strict` | 用户已经把视觉规格限定得比较明确 | 生成完整约束检查项 |
| `neutral` | 有一些具体要求，但仍有一定创作空间 | 仅生成明确写出的约束 |
| `open` | 简短或富有启发性的描述，允许补充细节 | 不生成 Constraint 问题 |

::: warning 这里有一个特别重要的区别
<mark class="hl-key">**生成 Checklist 和真正启用 Reward 是两回事。**</mark>

虽然作者为 Neutral Prompt 生成了显式约束问题，<mark class="hl-trick">**但最终训练时仍然只对 `strict` Prompt 启用 Constraint Reward。Neutral 和 Open 都不启用**</mark>。
:::

### 作者公布的三个真实示例

| Intent | 原始 Prompt | 说明 |
| :--- | :--- | :--- |
| **Strict** | *top down pixel trader space ship with no background* | 用户明确要求俯视、像素风格、无背景，<mark class="hl-key">**模型不应该随意改变这些规定**</mark> |
| **Neutral** | *please generate a scene of Seattle in the style of Richard Scarry. Include the space needle, a ferry and pike place market* | 用户规定了主体和艺术风格，<mark class="hl-key">**但并没有完整限制整个场景中的其他内容**</mark> |
| **Open** | *cat in red moccasins* | 只要求穿红色软皮鞋的猫，<mark class="hl-trick">**其他背景和构图留给模型发挥**</mark> |

<mark class="hl-key">**这三条均来自附录 A.3.3，而不是编造的分类样本。**</mark>

::: danger Intent 的判断并不等于 Prompt 长度分类
作者<mark class="hl-key">**没有规定多少 Token 以下就是 Open，也没有公布基于关键词的硬规则**</mark>，而是让 LLM 理解整体用户意图。
:::

---

## 36. Intent Tag 具体怎样生产？一次 Gemini 调用做两件事

附录 A.3.3 写得很明确。对每一条已经通过 Part 2 过滤的 RL Prompt，作者离线调用一次：

```text
gemini-3.1-pro-preview
```

**一次调用同时完成两项任务。**

**第一项是分类与元数据提取：**

| 字段 | 含义 |
| :--- | :--- |
| `intent` | `strict`、`neutral` 或 `open` |
| `style_lock` | 需要保留的明确风格 |
| `background_spec` | 背景要求 |
| `detail_budget` | 粗粒度的画面细节预算 |

其中 `detail_budget` 可以理解为<mark class="hl-key">**用户允许画面包含多少额外复杂度**</mark>。不过原文没有提供它的完整枚举词表或量化定义，<mark class="hl-trick">**因此不能擅自认为它使用某个固定的 0–5 级评分系统**</mark>。

**第二项是按照这些信息生成该 Prompt 的 Constraint Checklist。**

$$
\begin{aligned}
&\textbf{一条 RL Prompt} \\
&\quad \downarrow\ \text{Gemini-3.1-Pro-Preview（每 Prompt 离线调用一次）} \\
&\quad \bullet\ \textbf{Intent Metadata} \\
&\qquad \texttt{intent} \mid \texttt{style\_lock} \mid \texttt{background\_spec} \mid \texttt{detail\_budget} \\
&\quad \bullet\ \textbf{Constraint Checklist} \\
&\qquad \text{明确禁止项} \mid \text{背景约束} \mid \text{风格约束} \mid \text{额外内容检查} \\
&\quad \downarrow \\
&\textbf{与 Prompt 一起存储，供后续 RL 使用}
\end{aligned}
$$

::: tip 这里和 Part 4 有两个实现区别
1. Faithfulness 使用的是 <mark class="hl-key">`gemini-3-pro-preview`</mark>；这一部分使用的是 <mark class="hl-key">`gemini-3.1-pro-preview`</mark>。论文确实给出了不同的模型名称。

2. <mark class="hl-trick">**Constraint 部分没有明确披露类似 Faithfulness 的依赖 DAG，因此不能直接将 Part 4 的 DAG 执行策略完整套到这里**</mark>。它使用相同的 VLM Yes/No Judge，但问题的生成方式不同。
:::

---

## 37. Strict、Neutral、Open 分别生成什么问题？

这是数据生产中最具体的一部分。

**Strict Prompt**：作者要求生成<mark class="hl-key">**四种约束检查**</mark>，包括每条明确的否定要求、背景、风格保留，以及一个始终存在的兜底检查，用来判断是否出现用户未要求的显著物体、场景或装饰。

<mark class="hl-trick">**这意味着一个 Strict Prompt 即使没有明确写 `do not add extra objects`，也可能有一项关于多余内容的检查。**</mark>

**Neutral Prompt**：<mark class="hl-key">**只检查 Prompt 中明确陈述的约束，不添加通用的「禁止额外物体」检查**</mark>。

**Open Prompt**：<mark class="hl-key">**不生成 Constraint 问题**</mark>。

论文给出的统计如下：

| 指标 | 数值 |
| :--- | ---: |
| **Constraint 问题总数** | **26,213** |
| **实际启用 Constraint 的 Prompt** | **3,140** |
| **占比** | **31.4%** |

::: danger 这里尤其不能误解 26,213 这个数字
<mark class="hl-key">**它是全部训练 Prompt 生成的 Constraint 问题总数，不代表每个训练 Prompt 都会启用 Constraint Reward。**</mark>

作者也<mark class="hl-trick">**没有提供 Strict 类样本的完整问题数分布，因此不能简单拿 26,213 除以 3,140，声称每个 Strict Prompt 平均有 8.35 个问题**</mark>。
:::

---

## 38. 看论文 Figure 4：真实 Constraint Checklist 到底长什么样？

这里不编造例子，直接使用作者在 Figure 4 中展示的两条真实 Prompt。

### 例子 A：`MaurzTV` 圆形 Logo

原始用户要求制作一个用于 YouTube 频道的 Logo，包含 `MaurzTV` 文字，采用超现代、精致、太空和科学主题的圆形设计，并且没有背景。

![Arena-RL Figure 4 Example A：MaurzTV Logo 的完整奖励包。Constraint 部分（图中下半）为五项 C0–C4：C0 精确文字、C1 圆形主体设计、C2 背景完全简单/白色/透明、C3 保持 Logo 风格而不偏向摄影写实或电影化渲染、C4 是否无 Prompt 未要求的显著物体/场景/装饰细节。](/arena-fig4a-reward-battery-logo.png)

Gemini 提取的元数据是：

```text
intent: strict
style_lock: logo
background_spec: no background
detail_budget: minimal
```

然后生成了 5 个 Constraint 问题：

| 编号 | 论文检查项的中文翻译 |
| :--- | :--- |
| C0 | 图片是否清晰包含准确的 `MaurzTV` 文字？ |
| C1 | Logo 的主体设计是否为圆形？ |
| C2 | 背景是否完全简单、白色或透明，没有添加其他内容？ |
| C3 | 图片是否保持 Logo 风格，而没有偏向摄影写实或电影化渲染？ |
| C4 | 图片是否没有出现 Prompt 未要求的显著物体、场景或装饰细节？ |

::: warning 你可能会立即发现一个问题
<mark class="hl-key">**C0、C1 和 Part 4 的 Faithfulness 问题有重复。**</mark>

这不是理解错了。<mark class="hl-trick">**论文 Figure 4 的真实 Checklist 确实这样写。**</mark>

所以 Constraint 与 Faithfulness 在理论上分别偏向 Absence 和 Presence，但<mark class="hl-key">**实际自动生成的问题可能重叠**</mark>。在最终 Reward 等权组合下，一项要求如果同时出现在两种 Checklist 中，就可能获得更高的实际监督权重。

<mark class="hl-trick">**作者并没有单独分析这种问题重叠的影响。**</mark>
:::

### 例子 B：曼陀罗涂色页

另一条真实 Prompt 要求生成一张 8.5 × 11 英寸的竖版涂色页，300 DPI，边距 0.5 英寸。画面主体是由月亮、小骷髅、磁带和钥匙组成的对称曼陀罗图案。

![Arena-RL Figure 4 Example B：曼陀罗涂色页的完整奖励包。Constraint 部分 C0–C4：C0 竖版/对称/重复纹样的曼陀罗含四种挂饰、C1 可见空白边距、C2 背景纯白无额外内容、C3 保持涂色书风格不漂向摄影写实、C4 无未要求的显著物体/背景场景/装饰。](/arena-fig4b-reward-battery-mandala.png)

作者提取出：

```text
intent: strict
style_lock: coloring book page
background_spec: plain white
detail_budget: minimal
```

随后生成了 5 个 Constraint 问题：

| 编号 | 论文检查项的中文翻译 |
| :--- | :--- |
| C0 | 是否为竖版、对称且具有重复纹样的曼陀罗，包含月亮、小骷髅、磁带和钥匙？ |
| C1 | 画面边缘是否留有明显的空白边距？ |
| C2 | 背景是否为纯白色，并且没有额外内容？ |
| C3 | 是否保持涂色书页面风格，没有转向摄影写实或电影化渲染？ |
| C4 | 是否没有出现 Prompt 未要求的显著物体、背景场景或装饰细节？ |

::: tip 这里又可以看出两个细节
**第一**，作者确实会从用户要求中提取 `style_lock`，然后检查生成结果有没有明显偏离原始媒介。<mark class="hl-key">**例如涂色书不应该变成真实摄影。**</mark>

**第二**，即使 Prompt 明确写了 `300 DPI` 和 `0.5 in margin`，<mark class="hl-trick">**Checklist 也只是检查是否存在可见的空白边距，没有逐项检查图像文件的实际 DPI 元数据和精确物理尺寸**</mark>。
:::

::: warning 这是 Figure 4 真实示例所揭示的可评价性边界
<mark class="hl-key">**Rubric Judge 检查的是图像视觉表现，而不是所有非视觉技术规格。**</mark>
:::

---

## 39. 在线 RL 时如何计算 Constraint Reward？

与 Faithfulness 一样，训练时使用 <mark class="hl-key">Qwen3.6-27B 作为 VLM Judge，通过 vLLM 提供服务</mark>。每个检查问题都设计成：

| Judge 回答 | 含义 |
| :--- | :--- |
| `yes` | 符合约束 |
| `no` | 不符合约束 |
| `irrelevant` | 问题不适用或无法判断 |

Constraint 原始奖励为：

$$
\boxed{
r_i^{(\mathrm{constr})} = \frac{N_{\mathrm{yes}}}{N_{\mathrm{questions}}}
}
$$

例如，对于前面的 Logo Prompt，一共有 5 个检查问题。假设某次生成满足 C0、C1、C3，但背景不符合 C2，且添加了额外的装饰，导致 C4 失败。那么：

$$
r_i^{(\mathrm{constr})} = \frac{3}{5} = 0.6
$$

<mark class="hl-key">**这是用于说明计算方式的假设评分，并不是论文 Figure 4 的实际 Judge 结果。**</mark>

### 真正重要的是 Gating

作者在公式 (4) 中定义：

$$
\boxed{
g_i^{(\mathrm{constr})} = \mathbf{1}\bigl[y_i^{\mathrm{intent}} = \mathrm{strict}\bigr]
}
$$

也就是说：

$$
g_i^{(\mathrm{constr})} =
\begin{cases}
1, & \text{Strict} \\
0, & \text{Neutral 或 Open}
\end{cases}
$$

在总 Reward 中对应：

$$
\boxed{
R_i = w_{\mathrm{pref}}\, \bar z_i^{(\mathrm{pref})}
+ w_{\mathrm{faith}}\, z_i^{(\mathrm{faith})}
+ w_{\mathrm{constr}}\, g_i^{(\mathrm{constr})} z_i^{(\mathrm{constr})}
}
$$

这是论文公式 (8)。<mark class="hl-key">**作者所有实验中三个权重均设为 1**</mark>。这里的 $z$ 是按各 Reward 维度分别进行 Batch 标准化的结果；详细标准化和 NFT 权重转换留到 Part 8。

因此，一条 Prompt 的训练信号是：

| Prompt 类型 | Preference | Faithfulness | Constraint |
| :--- | :---: | :---: | :---: |
| **Strict Prompt** | ✅ | ✅ | ✅ |
| **Neutral / Open** | ✅ | ✅ | ❌ |

::: warning 这里只展示 Part 5 讨论的三个主要 Reward 轴；Anti-Hack Veto 属于 Part 6
<mark class="hl-key">**Gating 不是把整个 Prompt 从 RL 训练中删除，而是决定对应 Reward 轴是否参与该样本的总 Reward。**</mark>
:::

::: danger 还有一个未公开的实现细节
原文并没有明确说明，对没有启用 Constraint 的样本，是<mark class="hl-key">**完全跳过在线 Judge 调用，还是为了统计仍然计算后再屏蔽**</mark>。

因此<mark class="hl-trick">**不能将节约的 Judge 算力作为已经得到证实的实现收益**</mark>。
:::

---

## 40. Intent 分类准确吗？作者进行了人工审计

这部分正好回应「技术报告中自动生成的 Tag 究竟可不可信」这个问题。

作者并没有完全相信 Gemini 的分类结果。他们从评测 Prompt 中抽取 <mark class="hl-key">**400 条**</mark>，由<mark class="hl-key">**两名作者阅读原始用户文本进行人工标注**</mark>，再与自动分类结果对比。

这是附录 A.3.6 给出的完整混淆矩阵：

| 人工 \ Gemini | Strict | Neutral | Open | 合计 |
| :--- | ---: | ---: | ---: | ---: |
| **Strict** | 100 | 21 | 1 | 122 |
| **Neutral** | 12 | 128 | 36 | 176 |
| **Open** | 8 | 11 | 83 | 102 |
| **合计** | 120 | 160 | 120 | **400** |

| 指标 | 数值 |
| :--- | ---: |
| 三分类一致率 | **77.8%** |
| Strict / 非 Strict | **89.5%** |
| Cohen's $\kappa$ | **0.66** |

### 为什么二分类可以达到 89.5%？

因为<mark class="hl-key">**真正进入公式 (4) 的判断只有一个：是否为 Strict**</mark>。

比如人工判断为 Neutral、模型判断为 Open，<mark class="hl-trick">**虽然在三分类中算错误，但两者都不启用 Constraint，因此不影响该 Gate**</mark>。

按照表格：

$$
\text{Strict/Non-strict Accuracy}
=
\frac{100 + 128 + 36 + 11 + 83}{400}
=
89.5\%
$$

### 但还有两种真正会影响 Gate 的错误

| 错误类型 | 数量 | 后果 |
| :--- | ---: | :--- |
| **假阳性** | 20 条 | 人工认为不是 Strict，模型却判断为 Strict，<mark class="hl-trick">**于是错误启用了 Constraint**</mark> |
| **假阴性** | 22 条 | 人工认为是 Strict，模型却判断为 Neutral 或 Open，<mark class="hl-trick">**于是漏掉了 Constraint**</mark> |

这 42 条占 400 条的 10.5%。

<mark class="hl-key">**所以作者报告的 89.5% 是有实际意义的，但也说明 Gating 并非完全可靠。**</mark>

::: warning 三种准确率不能混为一谈
这项审计验证的是 <mark class="hl-key">**Intent 分类标签**</mark>，<mark class="hl-trick">**不是 26,213 个 Constraint 问题的逐项正确率，也不是 Qwen Judge 的 Yes/No 正确率**</mark>。
:::

---

## 41. Intent Gating 真的提升了训练效果吗？

作者在 §4.3 专门做了消融，比较两个方案：

- **Ungated**：对全部 Prompt 使用 Constraint Reward；
- **Intent-gated**：只对 Strict Prompt 使用 Constraint Reward。

其他条件按论文的相应对照设置。

| Preference RM | Judge | 不加 Gate | 加 Intent Gate | 提升 |
| :--- | :--- | ---: | ---: | ---: |
| Arena RM | Gemini | 60.46% | 64.16% | **+3.70 pp** |
| PickScore | Gemini | 52.00% | 58.10% | **+6.10 pp** |
| Arena RM | GPT-5.4 | 55.55% | 56.15% | **+0.60 pp** |
| PickScore | GPT-5.4 | 49.85% | 61.05% | **+11.20 pp** |

::: danger 提升幅度并不一致
在作者测试的两种 Preference Reward 和两种 Judge 下，Intent Gating 都优于不做 Gating。

不过，<mark class="hl-key">**例如 Arena RM 在 GPT-5.4 Judge 下只提升了 0.60 个百分点。因此不应该把所有评测器下的改善描述成同等幅度。**</mark>
:::

![Arena-RL Figure 1：三种 Reward 配置与本文方法的定性对照，3 行 Prompt × 6 列模型（Base / Preference only / Faithfulness only / Arena-T2I-Hard / Ungated constraint / Stage 1 ours）。第一行可见 Ungated constraint 在移除不良装饰的同时把背景压成一片平坦白场，Stage 1 则保留了完整街景。第二行可见 Arena-T2I-Hard 会额外实体化出木质玩具马（prompt 只要求蒸汽成形的幽灵形状）。第三行为 "SPIRIT OF DEEN" 文字 Logo。](/arena-fig1-qualitative-comparison.jpg)

作者还在 Figure 1 观察到一种现象：<mark class="hl-key">**不使用 Gate 的 Constraint Reward 有时会过度消除背景细节，把合理的背景压缩成一片平坦的白色区域**</mark>。

<mark class="hl-trick">**这正是作者想解决的问题：惩罚多余内容不能以损失合理创作空间为代价。**</mark>

---

## 42. 哪些实现细节仍然没有披露？

| 问题 | 原文情况 |
| :--- | :--- |
| Intent / Constraint 生成模型 | ✅ 明确：Gemini-3.1-Pro-Preview |
| 每条 Prompt 调用次数 | ✅ 明确：一次，同时生成标签和 Checklist |
| 三种 Intent 的含义 | ✅ 明确 |
| `style_lock`、`background_spec`、`detail_budget` | ✅ 明确存在 |
| `detail_budget` 完整取值及定义 | ❌ 未公开 |
| Strict / Neutral / Open 的问题生成原则 | ✅ 明确 |
| 完整的 Constraint 生成 System Prompt | ❌ 未逐字公开 |
| Constraint Checklist 的官方存储 JSON Schema | ❌ 未公开 |
| Constraint 的问题总数 | 26,213 |
| Constraint 的人工问题质量审计 | ❌ 未公开 |
| Intent 分类人工审核 | ✅ 400 条，完整混淆矩阵 |
| Constraint Judge | Qwen3.6-27B |
| Constraint Reward 公式 | Yes 占全部问题比例 |
| 是否包含 Faithfulness 的依赖 DAG | ❌ 未明确说明 |
| 非 Strict 时是否跳过 Judge 的实际调用 | ❌ 未明确说明 |
| Gating 效果的实验消融 | ✅ 有，Table 3 |
| 分类失败的 12 条 Prompt 如何参与训练 | ❌ 未说明具体处理 |

::: warning 还有一个比较容易被忽略的细节
作者虽然提取了 `style_lock`、`background_spec` 和 `detail_budget`，但<mark class="hl-key">**最终决定是否启用 Constraint 的公式只使用 `intent == strict`**</mark>。

<mark class="hl-trick">**这些其他字段的作用主要体现在 Checklist 的生成内容，而不是作为独立、连续的 RL Reward 权重。**</mark>
:::

---

## 43. Part 5 应该记住的核心内容

这部分其实没有复杂的新 RL 算法，真正的贡献是<mark class="hl-key">**一套 Prompt-conditioned Reward Routing**</mark>。

$$
\boxed{
\text{Prompt} \xrightarrow{\text{Gemini 离线分类}} \text{Strict / Neutral / Open}
}
$$

随后，对 Strict Prompt 离线生成更完整的约束检查，并在在线训练时启用对应 Reward。

::: tip 与 Part 4 连起来看
<mark class="hl-key">**Faithfulness 负责让模型学会正确实现要求；Constraint 负责在用户需要精确控制时，抑制不符合用户意图的额外生成。**</mark>

这篇论文没有盲目追求更多 Reward，而是<mark class="hl-trick">**先判断 Reward 在什么场景下适用**</mark>。消融结果也支持这一策略在作者实验设置中的有效性。
:::

::: danger 两个可以进一步研究、但作者没有充分验证的问题
1. 自动 Checklist 中 <mark class="hl-key">**Faithfulness 和 Constraint 重复计分**</mark>会不会产生偏差？
2. <mark class="hl-key">**89.5% 准确率的 Intent Gate 错误**</mark>，会对训练造成什么影响？

<mark class="hl-trick">**这些是从论文结果出发可以提出的研究问题，并非已经得到证明的结论。**</mark>
:::

**下一部分 Part 6：Anti-Reward-Hacking。** 我们会进入另一个重要机制：为什么模型会通过添加乱码文字、过度摄影写实等方式骗取 Preference Reward，作者如何分别设计 Pointwise 和 Reference-based Detector，以及为什么最后选择 One-sided Veto，而不是再增加一个正向 Reward。

---

# Part 6：Anti-Reward-Hacking

> **对应原文**：§3.2.4、§3.3.2–3.3.3、§4.2–4.3、附录 A.3.5，Figure 2 / Figure 4，Table 2 / Table 3

先给一个结论：<mark class="hl-key">**作者并没有训练一个全新的 Anti-Hack Reward Model，也没有给每条 Prompt 生成一套新的反作弊问题**</mark>。作者是<mark class="hl-trick">**先观察 RL 生成结果中的实际失败模式，再为这些失败模式设计固定的检测问题，通过 VLM 判断是否发生作弊，最后用 One-sided Veto 修改 Preference Reward**</mark>。

$$
\boxed{
\text{发现异常} \rightarrow \text{设计 Detector} \rightarrow \text{判断是否适用} \rightarrow \text{检查 Rollout} \rightarrow \text{Veto 正向奖励}
}
$$

---

## 44. 为什么作者突然要加入 Anti-Reward-Hacking？

回忆前面的训练过程。Stage 1 已经使用三种奖励（这里暂时省略标准化和 Gating）：

$$
R = R_{\mathrm{pref}} + R_{\mathrm{faith}} + R_{\mathrm{constr}}
$$

作者原本希望模型既能生成质量较高的图片，又能满足用户描述、遵守严格约束。

但在 Stage 1 训练结束后，他们<mark class="hl-key">**人工检查了最佳 Checkpoint 的 205 个失败案例**</mark>，发现两种反复出现的问题：

| 模式 | 描述 |
| :--- | :--- |
| **Garbled Text** | <mark class="hl-key">**模型生成用户没有要求的乱码或无意义文字**</mark> |
| **Photo-style Drift** | <mark class="hl-key">**用户明确要求插画、卡通等非摄影风格，但模型逐渐变成电影感强烈的写实摄影**</mark> |

作者将这两种现象作为<mark class="hl-trick">**已观察到的 Reward Hacking 模式**</mark>，设计专门的检测器。

![Arena-RL Figure 2：两种真实失败模式。左：Prompt 为「一只长着蓬松橘色长毛的姜黄色猫的超写实特写照片」，RL 后画面在餐巾纸上出现手写乱码串——用户从未要求任何文字。右：Prompt 为「一位穿蓝裙的公主，Elsa Beskow 风格，全身，超写实，白底」，左侧 Base model 保留了童话插画特征，右侧 After RL 的皮肤与光照明显偏向摄影写实。](/arena-fig2-failure-modes.jpg)

| 模式 | 图中现象 |
| :--- | :--- |
| **Garbled Text** | <mark class="hl-trick">**用户没有要求文字，RL 后的图片却在餐巾纸上出现乱码字符串**</mark> |
| **Photo-style Drift** | 用户要求 Elsa Beskow 的故事书插画风格，<mark class="hl-trick">**Base 保留了插画特征，RL 后却偏向摄影写实**</mark> |

::: warning 范围声明
这里描述的是 Figure 2 实际展示的两类问题，<mark class="hl-key">**不代表作者系统发现了其他类型的 Reward Hack**</mark>。
:::

::: tip 你可以发现，这两个问题本质上不同
<mark class="hl-trick">**乱码是单张图片就能判断的缺陷；风格漂移却需要判断模型相对 Base 是否变得更写实。**</mark>

所以作者没有用一种检测方式解决全部问题，而是分成 Pointwise 和 Reference-based 两类。
:::

---

## 45. 第一种：Pointwise Anti-Reward-Hack

这一类最容易理解。

假设用户输入：

> *A white ceramic cup on a wooden table.*

模型生成一张图片，但桌面、杯子或者背景中出现了 `ASDJKL`、`XYZXYZ` 之类无意义的字符。

<mark class="hl-trick">**这种问题不需要再生成一张 Base Model 图片做比较。**</mark>只需要把当前 Rollout Image 交给 VLM，询问：图片是否没有乱码、拼写错误、无意义字符、键盘式随机字母序列，以及同一文字的非预期大量重复？

这对应 Figure 4 中作者公开的固定 **Text Gibberish Detector**。

![Arena-RL Figure 4 底部：两个固定的 hack-prevention detector 问题，标注为 fixed per failure mode, not generated per prompt，且 verdicts enter the objective only as one-sided vetoes, never as reward。上为 Text gibberish（pointwise; armed on every sample），下为 Photo-style drift（reference-based; armed on both prompts above; judged in both presentation orders）。](/arena-fig4c-hack-detector-questions.png)

### 输入输出具体是什么？

$$
\begin{aligned}
&\textbf{当前 DiT 的一张 Rollout 图片} \\
&\quad \downarrow\ \text{Qwen3.6-27B Judge，使用固定的 Text Gibberish 检查问题} \\
&\quad \bullet\ \texttt{Yes} \Rightarrow \textbf{未发现乱码问题} \\
&\quad \bullet\ \texttt{No} \Rightarrow \textbf{发现乱码问题}
\end{aligned}
$$

::: warning 有一个非常容易搞错的地方
<mark class="hl-key">**检测问题本身采用正向措辞——「图片是否没有乱码」；但后续公式中的 $r_i^{(h)} = 1$ 表示检测到违规。**</mark>

所以从 Judge 回答到违规标志，需要进行逻辑转换：

| Judge 回答 | 是否存在违规 | 进入公式的违规标志 |
| :--- | :---: | :---: |
| Yes，图片没有乱码 | 否 | $0$ |
| No，图片存在乱码 | 是 | $1$ |

<mark class="hl-key">**这里是对论文检测问题与公式的逻辑对应解释，作者没有公布完整的结果解析程序。**</mark>
:::

最重要的是，原文 Figure 4 明确指出 <mark class="hl-key">**Text Gibberish Detector 对每个样本都启用，而不是只对 `strict` Prompt 启用**</mark>。

<mark class="hl-trick">**它的作用不是促进模型生成更多正确文字，而是防止模型生成无意义的错误文字。**</mark>这与 Part 7 的 OCR Reward 是不同任务。

---

## 46. 第二种：Reference-based Anti-Reward-Hack

这才是作者比较有特点的设计。

假设 Prompt 是：

> *An illustration in a classic children's storybook style.*

Stage 1 之后生成的图片虽然更精致、光影更丰富，但<mark class="hl-key">**逐渐变成了真实摄影**</mark>。

如果只看这一张图片，Qwen 很可能觉得它「质量很好」。但问题并不是图片是否漂亮，而是：<mark class="hl-trick">**RL 是否让模型相对原来的 Base Model，朝着违背用户指定风格的方向发生了变化？**</mark>

因此作者引入一个<mark class="hl-key">**冻结的 Base Model Reference**</mark>。

$$
\begin{aligned}
&\textbf{同一条用户 Prompt} \\
&\quad \downarrow \\
&\quad \bullet\ \textbf{Frozen Base Model} \Rightarrow \text{Reference Image} \\
&\quad \bullet\ \textbf{Current RL Policy} \Rightarrow \text{Rollout Image} \\
&\quad \downarrow \\
&\quad \textbf{Qwen3.6-27B 对比两张图：哪张图片更偏摄影写实？} \\
&\quad \downarrow \\
&\textbf{判断当前 Policy 是否更偏离目标风格}
\end{aligned}
$$

作者 Figure 4 公开的检测问题，要求 Judge 比较哪张图片表现出<mark class="hl-key">**更强的摄影写实特征**，具体看<mark class="hl-key">**真实纹理、摄影式光照和相机般的细节**</mark></mark>。输出限制为：

```text
A
B
same
```

为了减少图像排列顺序造成的判断偏差，<mark class="hl-key">**作者对同一图像对调换 A/B 顺序，进行两次评价**</mark>。

::: danger 论文没有进一步公开两件事
1. <mark class="hl-key">**当正反两次回答不一致时如何最终合并成二值违规标志**</mark>；
2. <mark class="hl-key">**Frozen Base Reference 是预先缓存还是每轮重新生成**</mark>。

因此<mark class="hl-trick">**不能擅自补充具体的投票或缓存策略**</mark>。我们只能确定：对比同一 Prompt 下的 Base 和 RL 图像、双顺序 Judge，以及检测是否朝过度摄影写实方向漂移。
:::

---

## 47. 两个 Detector 是每条 Prompt 自动生成的吗？

<mark class="hl-key">**不是。这是非常重要的实现细节。**</mark>与 Part 4、Part 5 不同：

| 模块 | 问题从哪里来 | 是否每条 Prompt 不同 |
| :--- | :--- | :---: |
| **Faithfulness** | Gemini 根据 Prompt 生成 | 是 |
| **Constraint** | Gemini 根据 Prompt 和 Intent 生成 | 是 |
| **Text Gibberish Detector** | <mark class="hl-trick">**作者根据观察到的错误设计固定问题**</mark> | 否 |
| **Photo-style Drift Detector** | <mark class="hl-trick">**作者根据观察到的错误设计固定问题**</mark> | 否 |

也就是说，本文的 Anti-Hack 使用的是<mark class="hl-key">**固定的失败模式检测器**</mark>。

但是，<mark class="hl-trick">**即使检测问题固定，也不意味着每条 Prompt 都应该启用每一个检测器**</mark>。这就引出了下一步：**Applicability Gating**。

---

## 48. Applicability Gating：什么时候应该检测？

作者在 §3.3.2 中指出：<mark class="hl-key">**同一种视觉现象，在不同 Prompt 下可能具有完全不同的意义**</mark>。

例如：

| Prompt | 越来越接近真实摄影是否算问题？ |
| :--- | :--- |
| *A photorealistic portrait of a woman.* | <mark class="hl-key">**如果生成结果越来越接近真实摄影，那可能正是用户所希望的**</mark> |
| *A flat vector illustration of a woman.* | <mark class="hl-key">**但如果用户要求扁平矢量插画，RL 却让结果变成摄影作品，这才属于目标风格偏移**</mark> |

因此，在执行 Detector 前，作者先<mark class="hl-key">**离线判断这类错误是否适用于当前 Prompt**</mark>：

$$
\begin{aligned}
&\textbf{每条 Prompt 调用一次 Gemini-3.1-Pro-Preview} \\
&\quad \downarrow \\
&\textbf{覆盖所有已定义的失败模式}
\end{aligned}
$$

论文公式 (5)：

$$
\boxed{
g_i^{(h)} = \mathbf{1}\bigl[y_{i,h}^{\mathrm{app}} = \mathrm{applicable}\bigr]
}
$$

这里：

| 符号 | 含义 |
| :--- | :--- |
| $i$ | 当前 Prompt 或对应 Rollout |
| $h$ | 某一种 Hack Detector |
| $y_{i,h}^{\mathrm{app}}$ | <mark class="hl-key">**离线 LLM 对该错误模式是否适用的判断**</mark> |
| $g_i^{(h)} = 1$ | 该检测器在当前 Prompt 上启用 |

### 两种 Detector 的实际适用条件

| Detector | 启用条件 |
| :--- | :--- |
| **Text Gibberish** | <mark class="hl-key">**论文 Figure 4 明确注明对每个样本启用**</mark> |
| **Photo-style Drift** | <mark class="hl-key">**Prompt 锁定非摄影风格，并且存在 Frozen Base Reference Image**</mark> |

例如，前面 Part 5 的 `MaurzTV` Logo 和 Mandala 涂色页，Figure 4 都<mark class="hl-trick">**同时启用了两种 Detector**，因为它们都要求非摄影视觉形式，并具有 Base Reference</mark>。

而对于明确要求摄影写实的 Prompt，<mark class="hl-key">**Photo-style Drift Detector 不应该将更写实本身视为违规**</mark>。

::: warning Hack Applicability Gate 与 Intent Gate 并不相同
<mark class="hl-key">**前者是针对每一种已知失败模式分别判断是否适用；后者只决定是否启用 Constraint Reward。**</mark>
:::

---

## 49. 最关键：为什么采用 One-sided Veto，而不是额外加一个 Anti-Hack Reward？

这里是 Part 6 最重要的算法设计。

假设我们已经有三个正常 Reward：

$$
R_{\mathrm{pref}}, \quad R_{\mathrm{faith}}, \quad R_{\mathrm{constr}}
$$

一种非常自然的想法是，再加入 $R_{\mathrm{anti-hack}}$：如果图像没有乱码，就奖励 1 分；如果有乱码，就奖励 0 分。

::: warning 但作者认为这种方法存在问题
<mark class="hl-key">**模型可能开始专门优化「怎样满足 Anti-Hack Detector」，而不是优化真正的生成质量**</mark>。

例如某个 Detector 喜欢没有文字的画面，模型可能为了取得较高评分，减少本来应该存在的文字。<mark class="hl-trick">**这就相当于为模型创造了新的可利用奖励目标。**</mark>
:::

因此作者没有把 Anti-Hack 设计为额外的正向奖励，而是<mark class="hl-key">**让它只负责阻止已知违规图片获得正向 Preference 奖励**</mark>。

### 第一步：判断是否发生了适用的违规

论文公式 (6)：

$$
\boxed{
v_i = \max_{h \in \mathcal{H}} g_i^{(h)} r_i^{(h)}
}
$$

其中：

| 符号 | 含义 |
| :--- | :--- |
| $\mathcal{H}$ | 所有 Anti-Hack Detector 的集合 |
| $g_i^{(h)} \in \{0,1\}$ | 这个 Detector 是否适用 |
| $r_i^{(h)} \in \{0,1\}$ | 是否检测到这种违规 |
| $v_i \in \{0,1\}$ | <mark class="hl-key">**当前图片最终是否触发 Veto**</mark> |

因为用了 `max`，所以<mark class="hl-key">**只要任意一个适用的 Detector 发现违规，$v_i = 1$**</mark>。例如：

| 图片 | 乱码检测 | 风格漂移检测 | 最终 Veto |
| :--- | :---: | :---: | :---: |
| A | 0 | 0 | 0 |
| B | 1 | 0 | 1 |
| C | 0 | 1 | 1 |
| D | 1 | 1 | 1 |

<mark class="hl-key">**这张表假设两个 Detector 都适用，是公式的直接演示。**</mark>

### 第二步：只取消正向 Preference 信号

论文公式 (7)：

$$
\boxed{
\bar z_i^{(\mathrm{pref})} = z_i^{(\mathrm{pref})} - v_i \bigl[z_i^{(\mathrm{pref})}\bigr]_+
}
$$

其中 $[x]_+ = \max(x, 0)$。

这里的 $z_i^{(\mathrm{pref}})$ 是 <mark class="hl-key">**Batch 标准化之后的 Preference Reward，不是 Part 3 中 RM 直接输出的原始 Logit**</mark>。

为了看清楚，我们直接代入数字。

| 标准化 Preference | Veto | 修改后 Preference |
| :---: | :---: | :---: |
| $+1.20$ | 0 | $+1.20$ |
| $+1.20$ | **1** | **$0.00$** |
| $-0.80$ | 0 | $-0.80$ |
| $-0.80$ | 1 | $-0.80$ |

$$
\bar z = z - v\max(z, 0)
$$

<mark class="hl-key">**已触发 Veto，正向 Preference 信号被清零。**</mark>所以触发 Veto 时只有两种情况：

$$
\bar z_i^{(\mathrm{pref})} =
\begin{cases}
0, & v_i = 1,\ z_i^{(\mathrm{pref})} > 0 \\
z_i^{(\mathrm{pref})}, & \text{其他情况}
\end{cases}
$$

<mark class="hl-trick">**这与「发现违规就给图片 -1 分」完全不同。**</mark>如果图片本身已经得到负向 Preference 信号，Veto 并不会继续惩罚它。

更重要的是：<mark class="hl-key">**Faithfulness 和 Constraint Reward 并不会被同时清零**</mark>。完整的总奖励是论文公式 (8)：

$$
\boxed{
R_i = w_{\mathrm{pref}} \bar z_i^{(\mathrm{pref})}
+ w_{\mathrm{faith}} z_i^{(\mathrm{faith})}
+ w_{\mathrm{constr}} g_i^{(\mathrm{constr})} z_i^{(\mathrm{constr})}
}
$$

<mark class="hl-key">**三个权重在作者的实验中都设置为 1**</mark>。因此，<mark class="hl-trick">**一张触发 Hack 的图片仍然可能从 Faithfulness 或 Constraint 获得信号；它不是被彻底删除，也不是 NFT Loss 被直接置零**</mark>。

最终总奖励还会通过公式 (9) 转换为 NFT 的相对训练权重。我们在 Part 8 完整推导。

---

## 50. 实验：Anti-Hack 真的有效吗？

这里的结果比单纯「加了一个 Detector 就明显变好」复杂一些。

### 1. Stage 1 与 Stage 2 的直接比较

| FLUX.2-dev | Arena RM · Gemini | PickScore · Gemini |
| :--- | ---: | ---: |
| **Stage 1，无 Anti-Hack** | **64.2%** | **58.1%** |
| **Stage 2，有 Anti-Hack** | **63.5%** | **57.1%** |
| **Stage 3，两个模型权重融合** | **66.0%** | **62.6%** |

::: danger 这是一个非常重要的细节
<mark class="hl-key">**单独加入 Anti-Hack 后，Stage 2 并没有在 Gemini 整体胜率上超过 Stage 1。**</mark>

作者的解释是两个 Checkpoint 的优势互补。Stage 2 在部分指令遵循、干净文字渲染方面有优势，但可能损失其他方面。

所以最终作者使用 Stage 1 和 Stage 2 的 LoRA Weight Soup（见 Part 9 / Part 10）。<mark class="hl-trick">**这也是为什么不能简单宣称 Anti-Hack 必然提高所有评测指标。**</mark>
:::

### 2. Applicability Gating 是否必要？

| Reward / Judge | 不做 Applicability Gate | 使用 Gate |
| :--- | ---: | ---: |
| Arena RM · Gemini | 55.91% | **63.46%** |
| Arena RM · GPT-5.4 | 50.00% | **56.70%** |
| PickScore · Gemini | 56.95% | **57.10%** |
| PickScore · GPT-5.4 | 55.15% | **58.45%** |

::: tip 准确结论
Gate 对 Arena RM 的改善非常明显，但<mark class="hl-key">**对 PickScore 在 Gemini 评价下的改善只有 0.15 个百分点**</mark>。

因此：<mark class="hl-trick">**在作者测试的四种组合中，Applicability Gating 都优于无 Gate，但收益大小依赖 Reward Model 和 Judge。**</mark>
:::

### 3. One-sided Veto 为什么优于 Additive Reward？

作者比较：

- **Additive**：把 Anti-Hack Compliance 作为一个额外 Reward 轴，<mark class="hl-key">**权重为 1**</mark>；
- **One-sided Veto**：<mark class="hl-key">**不奖励无违规行为，只阻止违规结果获得正向 Preference 信号**</mark>。

| Reward / Judge | Additive | One-sided Veto |
| :--- | ---: | ---: |
| Arena RM · Gemini | 52.60% | **63.46%** |
| PickScore · Gemini | 41.94% | **57.10%** |
| Arena RM · GPT-5.4 | 47.30% | **56.70%** |
| PickScore · GPT-5.4 | 41.05% | **58.45%** |

::: tip 这是一组很强的实验结果
在作者的设置里，<mark class="hl-key">**Anti-Hack Detector 作为约束机制，比把 Detector Compliance 当成新的奖励目标更有效**</mark>。

但<mark class="hl-trick">**它只比较了论文中指定的两种形式及参数，不能据此证明所有加性惩罚方法都不如 Veto**</mark>。
:::

---

## 51. 从数据生产角度，作者实际做了什么？

把 Part 4、5、6 放在一起，会发现三种数据结构很不一样：

| 监督内容 | 生成方式 | 在线使用 |
| :--- | :--- | :--- |
| **Faithfulness** | 每条 Prompt 独立生成问题 DAG | Qwen 逐项检查 |
| **Constraint** | 每条 Prompt 生成 Intent 与问题 | 仅 Strict 启用奖励 |
| **Anti-Hack** | <mark class="hl-key">**从 205 个失败案例发现两种模式，固定 Detector**</mark> | 适用时检查并 Veto |

<mark class="hl-key">**Anti-Hack 这一环真正新增的数据，不是数十万条新的 RL Prompt，而是：固定的失败模式定义、固定检测问题，以及与每条 Prompt 关联的 Applicability Label。**</mark>

所以它是一个很清晰的工业闭环：

$$
\boxed{
\text{RL 训练} \rightarrow \text{失败案例审计} \rightarrow \text{固定检测器} \rightarrow \text{离线适用性标签} \rightarrow \text{带防护的下一轮 RL}
}
$$

::: danger 特别值得留意
论文<mark class="hl-key">**没有声称这套系统能够自动发现所有未来的 Reward Hack**</mark>。它只处理了人工观察到的两类模式，<mark class="hl-trick">**也没有提供这些 Detector 的完整人工准确率、误报率和漏报率**</mark>。

另外，论文<mark class="hl-key">**没有明确公开正反顺序判断冲突时的聚合规则、Reference 的缓存与匹配策略，以及每类 Detector 的触发频次**</mark>。因此<mark class="hl-trick">**不能把上述流程图当作已经完整开源的实现**</mark>。
:::

---

## 52. Part 6 的最终结论

这一部分应该记住三个关键点。

::: tip 首先，Reward Hacking 不是 NFT 公式出了问题
<mark class="hl-key">**而是模型可能找到让 Reward Model 给高分、却违背用户真实要求的生成方式。**</mark>作者在 205 个失败案例中发现乱码和摄影化漂移，并针对这两种实际问题建立检测器。
:::

::: tip 其次，不同 Hack 要采用不同检查机制
<mark class="hl-trick">**乱码可从单张图判断，而风格漂移更适合与 Frozen Base 的结果比较**</mark>；对于风格问题还要判断当前 Prompt 是否适用，<mark class="hl-trick">**避免惩罚用户真正希望获得的摄影效果**</mark>。
:::

::: tip 最后，Anti-Hack 的目的不是创造新的正向奖励
<mark class="hl-key">**而是阻止错误图像利用现有 Preference Reward 得益。**</mark>这就是 <mark class="hl-trick">**One-sided Veto 的本质**</mark>。

而从实验结果看，Anti-Hack 对部分能力有帮助，但<mark class="hl-key">**不会保证整体 Win Rate 必然超过未加入 Anti-Hack 的模型**</mark>。作者最终通过 Weight Soup 整合了两种策略的互补优势。
:::

**下一部分 Part 7：Graded OCR Reward。** 这部分比较短，但实现细节很实用：作者如何从 Prompt 提取准确文字、怎样不依赖原始 Prompt 对生成图片转录、如何计算 Exact / Partial / Recall，以及为什么不包含文字的 Prompt 必须从 OCR Reward 轴中 Mask 掉。

---

# Part 7：Graded OCR Reward

> **对应原文**：§4.4 A Second Base Model: Ideogram-4、附录 A.3.4 Graded OCR Reward for Ideogram-4

这一部分篇幅不长，但非常值得研究。尤其是作者在训练一个文字生成能力已经很强的基础模型时，发现了一个问题：<mark class="hl-key">**RL 可以提高模型的总体视觉偏好，却可能损害它原本很强的文字渲染能力**</mark>。

::: warning 适用范围提醒
这个 OCR Reward <mark class="hl-key">**不是 FLUX.2-dev 主实验里的常规 Reward 轴，而是作者在 Ideogram-4 的后训练实验中额外引入的**</mark>。论文明确说明，Table 4 中的 Ideogram-4 实验均使用了 OCR Reward。
:::

---

## 53. 为什么 Faithfulness Checklist 还不够？

我们在 Part 4 已经看到，Faithfulness 可以包含文字检查，例如：

> *图片中是否包含准确的 `MaurzTV`？*

但这种 Yes/No 判断比较粗糙。假设用户要求在海报上写：

```text
WELCOME HOME
```

模型生成的文字分别是：

| 图片 | 实际文字 | Faithfulness 二值判断 |
| :--- | :--- | :---: |
| A | `WELCOME HOME` | Yes |
| B | `WELC0ME HOME` | No |
| C | `WELCOME` | No |
| D | `WELCME HME` | No |

<mark class="hl-key">**虽然 B、C、D 都不完全正确，但显然它们的错误程度不同。**</mark>如果都给 0 分，Reward 就无法表达这些差异。

所以作者引入 <mark class="hl-key">**Graded OCR Reward，让部分正确的文字也能够获得部分奖励，同时继续鼓励完全正确的文字**</mark>。

---

## 54. 完整流程：离线提取期望文字，在线独立转录

$$
\begin{aligned}
&\textbf{离线：制作 RL Prompt 元数据} \\
&\quad \bullet\ \text{原始用户 Prompt} \\
&\quad \bullet\ \textbf{提取用户明确要求渲染的文字片段（例如引号内文字、指定名称）} \\
&\quad \bullet\ \textbf{将 Expected Text Spans 保存为 Prompt Metadata} \\[8pt]
&\textbf{在线：当前模型 Rollout 后} \\
&\quad \bullet\ \text{DiT 生成图片} \\
&\quad \bullet\ \textbf{VLM 转录图片中所有可见文字} \\
&\qquad \downarrow\ \textcolor{red}{\textbf{关键：转录时不提供原始 Prompt}} \\
&\quad \bullet\ \textbf{Expected Text 与 Transcribed Text 比较} \\
&\quad \bullet\ \text{Exact + Partial + Recall} \\
&\quad \downarrow \\
&\qquad R_{\mathrm{OCR}} \in [0, 1]
\end{aligned}
$$

::: danger 这里有一个特别关键的细节
<mark class="hl-key">**OCR 转录模型不能看到用户 Prompt。**</mark>

因为如果用户要求写 `WELCOME HOME`，然后你同时把这个字符串和图片都给 VLM，让它读图中文字，那么<mark class="hl-trick">**即使图片实际写成 `WELC0ME HOME`，VLM 也可能受到用户 Prompt 的影响，倾向于输出期望的正确文字**</mark>。

作者明确规定：<mark class="hl-key">**转录时只能依据图片本身，不能依赖 Prompt。**</mark>
:::

---

## 55. 离线数据究竟要准备什么？

论文说明，对每一条训练 Prompt，<mark class="hl-key">**提取用户明确要求绘制的文字片段，例如引号中的字符串或被指定为图中文字的名称，且只提取一次、存储为元数据**</mark>。

比如用户输入：

> *Create a retro-style poster with the title "SUMMER FEST" and a small subtitle "JULY 20".*

那么应该得到两个 Expected Text Spans：

```json
{
  "prompt": "Create a retro-style poster with the title \"SUMMER FEST\" and a small subtitle \"JULY 20\".",
  "expected_text_spans": [
    "SUMMER FEST",
    "JULY 20"
  ]
}
```

<mark class="hl-key">**这是根据论文描述构造的示例格式，不是作者公布的官方 JSON Schema。**</mark>

::: warning 需要特别强调
作者要求提取的是<mark class="hl-key">**明确需要出现在图像里的文字，不是把 Prompt 中每个单词都拿去进行 OCR 检查**</mark>。

比如 *A beautiful summer festival poster.* —— 这里的 `summer festival` 是<mark class="hl-trick">**对画面内容的描述，并不一定意味着用户要求把这两个单词印在海报上**</mark>。
:::

::: danger 提取环节的披露程度
论文<mark class="hl-key">**没有提供提取 Expected Text Span 所使用的具体模型、完整 System Prompt、文本规范化规则或人工准确率审计**</mark>。

因此我们能够确认它的提取原则与存储方式，<mark class="hl-trick">**但无法复原完整的文本抽取实现**</mark>。
:::

---

## 56. Graded OCR Reward 的完整计算公式

这是这一部分最重要的内容。对于一个期望文字片段 $s$，论文定义三个得分：

$$
\boxed{
r_{\mathrm{OCR}}(s, \hat T) = 0.5\, r_{\mathrm{exact}} + 0.3\, r_{\mathrm{partial}} + 0.2\, r_{\mathrm{recall}}
}
$$

其中 $\hat T$ 是 VLM 从生成图片中转录出的全部可见文字。

| 分量 | 权重 | 论文中的具体含义 |
| :--- | ---: | :--- |
| **Exact** | **0.5** | <mark class="hl-key">**期望字符串是否作为连续子串，完整出现在转录文本中**</mark> |
| **Partial** | **0.3** | 期望字符串和转录文本之间的模糊字符串相似度 |
| **Recall** | **0.2** | 期望文字片段中的 Token，有多大比例出现在转录文本的任意位置 |

### 1. Exact：完全匹配

这是权重最大的部分，占 50%。

假设期望文字是 `HELLO WORLD`，转录结果是 `WELCOME! HELLO WORLD`。由于 `HELLO WORLD` 完整出现：

$$
r_{\mathrm{exact}} = 1
$$

但如果图片实际转录为 `HELLO W0RLD`，由于 $O$ 变成了数字 $0$，完整子串不存在：

$$
r_{\mathrm{exact}} = 0
$$

::: warning 这里原文强调的是 verbatim substring（原样子串）
因此，<mark class="hl-key">**它不要求整张图片只能包含指定文字；只要期望字符串完整出现在转录内容中，就满足这个 Exact 条件**</mark>。

这也意味着，<mark class="hl-trick">**单靠这个 OCR Reward 本身，并不能保证图片没有多余文字**</mark>。
:::

### 2. Partial：模糊匹配

这是作者为了缓解 Exact Match 过于严格的问题引入的。

例如期望 `WELCOME HOME`，实际转录 `WELC0ME HOME`。虽然 Exact $= 0$，但两个字符串高度相似，因此 Partial 可以给出非零分数，<mark class="hl-trick">**避免把只错一个字符的结果和完全无关的乱码都视为同样差**</mark>。

::: danger 论文没有说明具体相似度实现
论文只说采用 <mark class="hl-key">**Fuzzy String Similarity**</mark>，并没有说明具体是 <mark class="hl-trick">**Levenshtein Ratio、RapidFuzz、归一化编辑距离还是其他相似度实现**</mark>。

所以<mark class="hl-key">**不能把某一个编辑距离公式当成作者的原始公式**</mark>。
:::

### 3. Recall：期望的词有没有出现

$$
r_{\mathrm{recall}} = \frac{\text{在转录文本中找回的期望 Token 数}}{\text{期望 Token 总数}}
$$

例如，假设按照空格划分词：

| | 文本 |
| :--- | :--- |
| **期望文字** | `HELLO BEAUTIFUL WORLD` |
| **实际转录** | `HELLO WORLD` |

那么找到两个期望词，遗漏一个：

$$
r_{\mathrm{recall}} = \frac{2}{3}
$$

::: warning 这里还要注意原文的措辞
<mark class="hl-key">**Token 可以出现在转录文本中的任意位置**</mark>。因此 Recall 主要关注文字是否被找回来，<mark class="hl-trick">**不强制要求全部 Token 按原始顺序连续排列**</mark>。

论文<mark class="hl-key">**没有进一步披露 Tokenizer、大小写归一化、标点处理、重复词计数及多语言分词规则**</mark>，所以<mark class="hl-trick">**不能断言作者就是按空格划词**</mark>。上述例子只是用最容易理解的分词方式解释 Recall。
:::

---

## 57. 完整算一个 Reward，看看它和二值 OCR 的区别

假设一条 Prompt 要求：

> *Create a poster displaying the words "HELLO WORLD".*

模型生成图片后，VLM 转录得到 `HELLO`。那么：

$$
r_{\mathrm{exact}} = 0
$$

因为 `HELLO WORLD` 没有完整出现。为了演示计算，假设采用的模糊字符串相似度得到：

$$
r_{\mathrm{partial}} = 0.5
$$

再假设按空格划词，两个词只找回 `HELLO`：

$$
r_{\mathrm{recall}} = \frac{1}{2}
$$

最后：

$$
\begin{aligned}
r_{\mathrm{OCR}} &= 0.5 \times 0 + 0.3 \times 0.5 + 0.2 \times 0.5 \\
&= \boxed{0.25}
\end{aligned}
$$

<mark class="hl-key">**这个 0.25 是根据假设的 Partial 值和分词方式计算出来的演示值，不是论文给出的具体评分案例**</mark>。

| 分量 | 数值 | 加权贡献 |
| :--- | ---: | ---: |
| Exact Match | $0$ | $0.5 \times 0 = 0$ |
| Partial 相似度 | $0.50$ | $0.3 \times 0.5 = 0.15$ |
| Token Recall | $0.50$ | $0.2 \times 0.5 = 0.10$ |
| **最终单片段 OCR Reward** | | **$0.250$** |

<mark class="hl-trick">**可以看出，作者让准确的完整字符串获得最高奖励，同时为部分正确的结果留下梯度区分信号。**</mark>

::: warning 但这并不是字符级可微分 OCR Loss
<mark class="hl-key">**而是先计算一个离散的 Reward，随后通过 NFT 更新图像生成模型**</mark>。
:::

---

## 58. 如果 Prompt 要求多个文字片段怎么办？

原文还明确规定：<mark class="hl-key">**先分别计算每个 Expected Text Span 的得分，然后对所有 Span 取算术平均**</mark>。

设一条 Prompt 有 $M$ 个期望文字片段：

$$
\boxed{
R_{\mathrm{OCR}}(p, x) = \frac{1}{M} \sum_{j=1}^{M} r_{\mathrm{OCR}}(s_j, \hat T)
}
$$

例如用户要求海报出现 `SUMMER FEST` 与 `JULY 20`，假设第一条文字的 OCR 分数为 $1.0$，第二条为 $0.4$，那么：

$$
R_{\mathrm{OCR}} = \frac{1.0 + 0.4}{2} = 0.7
$$

<mark class="hl-key">**这两个分数是为了展示平均方式而假设的，不是论文中的实测值**</mark>。

::: tip 这里的关键细节
作者采用 <mark class="hl-key">**Span 级等权平均，而不是按字符串长度加权，也没有公开为海报标题和副标题设置不同重要性权重**</mark>。

因此，<mark class="hl-trick">**较短的一个文字片段和较长的一个文字片段，在最终平均时具有相同的权重**</mark>。
:::

---

## 59. 非常重要：哪些样本不参与 OCR Reward？

论文明确规定了两种情况：

**情况一：Prompt 根本没有要求渲染文字。**例如：

> *A cat sleeping on a sofa in warm sunlight.*

<mark class="hl-key">**这条 Prompt 不应该因为图片里没有文字而得到 OCR = 0。用户本来就没有要求文字。**</mark>

**情况二：图片转录调用失败。**例如 VLM 服务异常，没有成功返回转录结果。<mark class="hl-trick">**这种情况属于评分失败，不能直接解释为生成图片没有正确文字**</mark>。

因此，作者规定：<mark class="hl-key">**无文字要求的 Prompt，以及 OCR 转录失败的样本，都从 OCR Reward 轴中排除，而不是给予负面奖励**</mark>。

| 情况 | OCR 如何处理 |
| :--- | :--- |
| Prompt 要求文字，图片文字正确 | 正常计算，得分高 |
| Prompt 要求文字，图片文字错误 | 正常计算，得分较低 |
| Prompt 要求文字，图片没有成功渲染文字 | <mark class="hl-key">**应依据成功转录的结果计算，可能很低**</mark> |
| **Prompt 没有要求文字** | **不参与 OCR Reward** |
| **OCR 转录调用失败** | **不参与 OCR Reward** |

::: danger 最后两项不是 Score = 0，而是 Mask 掉整个 Reward 轴
<mark class="hl-key">**这意味着作者并没有把 OCR 用作所有图片的全局质量奖励，而是采用条件适用的能力奖励**</mark>。

但论文<mark class="hl-trick">**没有给出 Mask 样本在 Batch 统计中如何精确实现有效样本均值与方差，也没有说明整个 Batch 都没有有效 OCR 样本时的回退方式**</mark>。
:::

---

## 60. OCR Reward 怎样与其他 Reward 合并？

论文明确说，Ideogram-4 的 OCR Reward 采用与其他奖励轴相同的 Per-batch z-normalization，权重设置为：

$$
\boxed{ w_{\mathrm{OCR}} = 1.0 }
$$

也就是说，对有效 OCR 样本先得到原始 OCR 分数，然后按 Batch 做该维度的标准化：

$$
z_i^{(\mathrm{OCR})} = \frac{r_i^{(\mathrm{OCR})} - \mu_{\mathrm{OCR}}}{\sigma_{\mathrm{OCR}} + \epsilon}
$$

<mark class="hl-key">**这里是根据论文标准化规则对 OCR 轴进行的公式展开。**</mark>结合 §3.3 的整体设计，Ideogram-4 的奖励可以概念性写成：

$$
\begin{aligned}
R_i = {} & \bar z_i^{(\mathrm{pref})} + z_i^{(\mathrm{faith})} \\
& + g_i^{(\mathrm{constr})} z_i^{(\mathrm{constr})} \\
& + m_i^{(\mathrm{OCR})} z_i^{(\mathrm{OCR})}
\end{aligned}
$$

其中 $m_i^{(\mathrm{OCR}})$ 表示该样本的 OCR 轴是否有效。

::: warning 注意这是扩展性写法
<mark class="hl-key">**这是将附录 A.3.4 的 OCR 规则扩展到正文公式 (8) 的解释性写法，不是论文另行编号给出的公式**</mark>。

论文仅明确说 OCR 以权重 $1.0$ 参与，并采用相同的 Batch 标准化；<mark class="hl-trick">**没有逐行公开 Mask 与标准化之间的实际代码**</mark>。
:::

之后仍然需要按论文公式 (9) 将综合 Reward 转为 NFT 的组相对权重，<mark class="hl-trick">**这不是对 OCR 单独进行一次 NFT 更新**</mark>。

---

## 61. 实际实验结果如何？有没有专门的 OCR 消融？

这部分必须客观。

作者报告，Ideogram-4 在 RL 后训练过程中可能出现文字渲染能力下降，因此<mark class="hl-key">**给所有 Ideogram-4 的 Table 4 实验增加了 OCR Reward**</mark>。

| 方法 | Arena RM + Gemini | PickScore + Gemini |
| :--- | ---: | ---: |
| Preference-only | 26.6% | 33.4% |
| Arena-T2I-Hard | 40.1% | 52.4% |
| **本文 Stage 1** | **55.3%** | **53.7%** |

<mark class="hl-key">**指标为对应实验相对冻结 Ideogram-4 Base 的配对胜率。全部这些实验均包含额外 OCR Reward。**</mark>

::: danger 不能把 Stage 1 的提升归因于 OCR Reward
因为所有对比方案均已经添加了 OCR，作者<mark class="hl-key">**没有报告「Ideogram RL without OCR」vs「Ideogram RL with OCR」的直接对照**</mark>，也<mark class="hl-trick">**没有在本篇论文中提供文字专用 Benchmark 上的完整 OCR 准确率曲线**</mark>。

因此，<mark class="hl-key">**作者明确给出了加入 OCR 的动机和算法，但没有单独量化 OCR Reward 的实际贡献**</mark>。
:::

另外，<mark class="hl-key">**作者没有在 Ideogram-4 上继续执行 FLUX.2-dev 的 Stage 2 Anti-Hack 和 Stage 3 Weight Soup，因为他们在 Stage 1 Checkpoint 中没有观察到明确的 Reward Hacking 模式**</mark>。

这也再次说明：<mark class="hl-trick">**本文并没有要求不同基模必须使用完全相同的 Reward 体系**</mark>。

---

## 62. 哪些关键细节没有公开？

| 问题 | 原文披露情况 |
| :--- | :--- |
| OCR 使用在哪个模型 | Ideogram-4 |
| 加入 OCR 的原因 | RL 可能损害文字渲染能力 |
| Expected Text 来源 | 用户明确要求绘制的文字 |
| Expected Text 提取次数 | 每 Prompt 一次，离线保存 |
| 在线 OCR 输入 | 生成图像，<mark class="hl-key">**转录时不看 Prompt**</mark> |
| 转录模型 | 说明使用 VLM，<mark class="hl-key">**未在 A.3.4 明确指定具体型号**</mark> |
| Exact 的判断方式 | 原样子串匹配 |
| Partial 的具体算法 | ❌ 未公开 |
| Recall 的分词与规范化方式 | ❌ 未公开 |
| 三个权重 | 0.5 / 0.3 / 0.2 |
| 多文字片段聚合 | 算术平均 |
| 无文字 Prompt | Mask |
| 转录失败 | Mask |
| OCR Reward 权重 | 1.0 |
| Reward 标准化 | Per-batch z-normalization |
| OCR 单独消融 | ❌ 未提供 |
| OCR 训练专用 Prompt 数量 | ❌ 未单独给出 |
| OCR 单独提升多少 | ❌ 无法从现有实验分离 |

::: danger 一个必须注意的界限
虽然正文中的其他 Rubric Judge 明确使用 Qwen3.6-27B，但附录 A.3.4 对 OCR 转录仅写 VLM。因此<mark class="hl-key">**不能直接断言 OCR 一定由 Qwen3.6-27B 完成，更不能据此认为使用了 PaddleOCR 或其他 OCR 专用模型**</mark>。
:::

---

## 63. Part 7 总结

这一部分最值得记住的，其实只有三件事。

::: tip 第一，文字生成 Reward 不一定采用严格的 Exact Match
作者同时使用 Exact、Partial、Token Recall，<mark class="hl-key">**使部分正确的文字仍然能被区分**</mark>。
:::

::: tip 第二，OCR 必须避免 Prompt 信息泄漏
用户指定的文字在离线阶段提取，而<mark class="hl-key">**在线转录模型只看生成图像，再由程序比较结果**</mark>。这样比直接把正确答案提示给 Judge <mark class="hl-trick">**更能避免受条件文本影响的错误转录**</mark>。
:::

::: tip 第三，OCR 是一种有适用范围的能力保护 Reward
<mark class="hl-key">**没有文字要求的 Prompt 不应该因为不存在文字而受到惩罚；OCR 服务失败也不应该直接被解释为图片生成失败**</mark>。
:::

::: warning 对图像基模 RL 的可迁移价值
当基础模型已有某项优势能力时，<mark class="hl-key">**可以设计针对性的 Reward 来防止后训练期间能力退化**</mark>。

但是，<mark class="hl-key">**这篇论文还没有证明 Exact + Partial + Recall 的权重组合是最优的，也没有提供 OCR 奖励本身的独立消融**</mark>。这部分应当作为<mark class="hl-trick">**作者公开的工程配方**</mark>来理解，而不是已被充分验证的通用最优方案。
:::

**下一部分是 Part 8：Multi-Reward Composition。** 这是整篇论文最核心的公式部分。我们会把 Preference、Faithfulness、Constraint、Anti-Hack 和 OCR 放到同一个真实训练 Batch 中，从原始分数开始，逐步计算 Per-axis Normalization、Intent Gate、Veto、Group Advantage 和最终 NFT Weight，完整解释公式 (3)–(9)，不再停留在「多奖励加起来」这个层面。

---

# Part 8：Multi-Reward Composition

> **对应原文**：§3.3.1–§3.3.4，公式 (3)–(9)；补充 §3.1、附录 A.3.4、§4.3 的相关实验

前面 Part 3–7 已经分别研究了 Preference、Faithfulness、Constraint、Anti-Hack 和 OCR，但它们最后<mark class="hl-key">**必须变成每张 Rollout 图片对应的一个训练权重，才能接入 DiffusionNFT**</mark>。

## 整体流程

$$
\boxed{
\begin{gathered}
\textbf{同一批 Prompt} \rightarrow \textbf{多张 Rollout 图片} \\
\begin{array}{c|c|c}
\text{Preference} & \text{Faithfulness} & \text{Constraint} \\
\text{无界 Logit} & \text{Yes 比例} & \text{合规比例}
\end{array}
\\[4pt]
\textbf{①} \text{ 每个 Reward 轴分别进行 Batch Z-score 标准化} \\
\textbf{②} \text{ Prompt-conditioned Gating（仅 Strict 启用 Constraint）} \\
\textbf{③} \text{ Anti-Hack One-sided Veto（仅清除违规图片的正向 Preference 信号）} \\
\textbf{④} \text{ 加权求和：总 Reward } R_i \\
\textbf{⑤} \text{ 同 Prompt 去均值 + Batch 标准差归一化} \\
\textbf{⑥} \text{ 映射到 } [0,1]\text{：}\tilde r_i \rightarrow \text{NFT Loss}
\end{gathered}
}
$$

::: warning 这不是「将三个原始分数直接加起来」
<mark class="hl-key">**作者使用了两个不同层级的归一化、两类 Gate 和一个 Veto。**</mark>下面严格按公式顺序拆解。
:::

---

## 64. 公式 (3)：为什么每一种 Reward 都要单独标准化？

先看三种原始评分。假设同一张生成图像，经过三个评价模块后得到：

| Reward | 原始数值 | 数值含义 |
| :--- | ---: | :--- |
| **Preference** | 4.8 | Part 3 的 BT Reward Model 输出 Logit |
| **Faithfulness** | 0.75 | 例如 12/16 个问题满足 |
| **Constraint** | 0.80 | 例如 4/5 个问题满足 |

如果简单求和：

$$
R = 4.8 + 0.75 + 0.80 = 6.35
$$

<mark class="hl-key">**Preference 的数值会占很大比例。但原始分数尺度大，并不说明它在最终训练中应该更重要。**</mark>

所以作者参考 GDPO 的思想，对每个 Reward 轴分别标准化：

$$
\boxed{
z_i^{(a)} = \frac{r_i^{(a)} - \mu^{(a)}}{\sigma^{(a)} + \epsilon}
}
\tag{3}
$$

其中：

| 符号 | 含义 |
| :--- | :--- |
| $a \in \{\mathrm{pref}, \mathrm{faith}, \mathrm{constr}\}$ | 不同 Reward 轴 |
| $r_i^{(a)}$ | 第 $i$ 张图在该 Reward 轴上的原始分数 |
| $\mu^{(a)}$ | <mark class="hl-key">**当前 Batch 中这个 Reward 轴的平均值**</mark> |
| $\sigma^{(a)}$ | <mark class="hl-key">**当前 Batch 中这个 Reward 轴的标准差**</mark> |
| $\epsilon$ | 防止分母为零的稳定项 |

::: danger 注意是每个轴分别求均值、标准差
<mark class="hl-key">**而不是先将三个 Reward 相加，再对总分标准化。**</mark>
:::

### 这里有一个容易理解错的地方

论文将 §3.3.1 称为 **Group Reward-Decoupled Normalization**，但公式 (3) 的文字明确定义均值和标准差来自 <mark class="hl-key">**current batch**</mark>。因此应当区分：

| 公式 | 分子统计量 | 分母统计量 |
| :--- | :--- | :--- |
| **公式 (3)** | <mark class="hl-key">**每个 Reward 轴在当前 Batch 上的均值**</mark> | **该轴在当前 Batch 上的标准差** |
| **公式 (9)** | <mark class="hl-key">**每个 Prompt 的组内总 Reward 均值**</mark> | **当前 Batch 总 Reward 标准差** |

<mark class="hl-trick">**这两个归一化不能混为一谈。**</mark>

还有一处原文表述不完全一致：<mark class="hl-key">**Introduction 曾将各 Reward 的预处理描述为映射到共同的 $[0,1]$ 尺度，但实际公式 (3) 是 Z-score，结果可以为负，也可以大于 1。直到公式 (9) 才将最终 NFT 权重限制到 $[0,1]$**</mark>（参见 Part 1 §4）。因此实现时应以公式为准。

---

## 65. 公式 (4)、(5)：Gating 不是选 Prompt，而是选择 Reward

我们在 Part 5 已经研究过 Intent 分类，在 Part 6 研究过 Anti-Hack Applicability。现在看它们如何真正进入总奖励。

首先，普通 Reward 轴的 Gate：

$$
\boxed{
g_i^{(a)} =
\begin{cases}
1, & a \in \{\mathrm{pref}, \mathrm{faith}\} \\[4pt]
\mathbf{1}\bigl[y_i^{\mathrm{intent}} = \mathrm{strict}\bigr], & a = \mathrm{constr}
\end{cases}
}
\tag{4}
$$

| Prompt 类型 | Preference | Faithfulness | Constraint |
| :--- | :---: | :---: | :---: |
| **Strict** | 启用 | 启用 | 启用 |
| **Neutral** | 启用 | 启用 | 关闭 |
| **Open** | 启用 | 启用 | 关闭 |

::: warning 需要特别强调
<mark class="hl-key">**Neutral/Open 不会被丢出 RL Batch，只是不贡献 Constraint Reward。**</mark>
:::

其次是 Anti-Hack 的适用性 Gate：

$$
\boxed{
g_i^{(h)} = \mathbf{1}\bigl[y_{i,h}^{\mathrm{app}} = \mathrm{applicable}\bigr]
}
\tag{5}
$$

这里的 $h$ 是某个固定检测器，例如乱码或摄影风格漂移。<mark class="hl-key">**只有这个 Detector 被判定适用于当前 Prompt，才允许触发后面的 Veto。**</mark>

::: tip 这两种 Gate 的功能不同
- **Constraint Gate** 决定一个评分维度是否生效；
- **Anti-Hack Gate** 决定某个异常检测器是否具有否决资格。
:::

---

## 66. 公式 (6)、(7)：One-sided Veto 怎么影响总 Reward？

先计算当前图片是否触发任何适用的 Hack：

$$
\boxed{
v_i = \max_{h \in \mathcal{H}} g_i^{(h)} r_i^{(h)}
}
\tag{6}
$$

<mark class="hl-key">**只要一个适用的 Detector 发现违规，$v_i = 1$。**</mark>接下来：

$$
\boxed{
\bar z_i^{(\mathrm{pref})} = z_i^{(\mathrm{pref})} - v_i \bigl[z_i^{(\mathrm{pref})}\bigr]_+
}
\tag{7}
$$

这句话可以直接翻译成：<mark class="hl-trick">**检测到作弊时，如果 Preference 标准化得分是正数，就将其清零；如果是负数，则保持原样。**</mark>例如：

| 原始标准化 Preference | Veto | 修改后 Preference |
| ---: | :---: | ---: |
| $+1.20$ | 0 | $+1.20$ |
| $+1.20$ | **1** | **$0$** |
| $-0.80$ | 0 | $-0.80$ |
| $-0.80$ | 1 | $-0.80$ |

::: tip 为什么作者要这样做
因为它<mark class="hl-key">**不想把「没有作弊」额外设计成一种正向奖励，只想阻止图片通过已知漏洞获得正向 Preference 激励**</mark>。

同时，这里<mark class="hl-trick">**不会直接修改 Faithfulness 或 Constraint 的分数，也不会直接将整张图片从训练中删除**</mark>。
:::

---

## 67. 公式 (8)：怎样合成最终总 Reward？

经过前面三个步骤，作者将有效奖励合并为：

$$
\boxed{
R_i = w_{\mathrm{pref}} \bar z_i^{(\mathrm{pref})}
+ w_{\mathrm{faith}} z_i^{(\mathrm{faith})}
+ w_{\mathrm{constr}} g_i^{(\mathrm{constr})} z_i^{(\mathrm{constr})}
}
\tag{8}
$$

论文明确说：

$$
\boxed{ w_{\mathrm{pref}} = w_{\mathrm{faith}} = w_{\mathrm{constr}} = 1 }
$$

所以在 FLUX.2-dev 主实验中，就是：

$$
R_i = \bar z_i^{(\mathrm{pref})} + z_i^{(\mathrm{faith})} + g_i^{(\mathrm{constr})} z_i^{(\mathrm{constr})}
$$

::: warning 这不意味着三个 Reward 在实际训练中的影响严格相同
它们的<mark class="hl-key">**原始分数分布、Gate 启用比例、检测误差、与其他评分的相关性都不同**</mark>。

作者的意思仅仅是：<mark class="hl-key">**经过分轴标准化后，三个 Reward 不再额外进行人工权重搜索，统一使用权重 1**</mark>。
:::

这里的 $R_i$ 仍然是一个<mark class="hl-trick">**可以为正、为负、甚至绝对值大于 1 的总分。它还不是 NFT 所要求的 $[0,1]$ 权重**</mark>。

---

## 68. 公式 (9)：为什么加完 Reward，还要再做一次组相对归一化？

<mark class="hl-key">**这一点是整篇论文最需要弄清楚的地方。**</mark>

前面的公式 (3) 是为了让不同 Reward 维度的数值尺度具有可比性。但公式 (9) 要解决另一个问题：<mark class="hl-trick">**对于同一个 Prompt 的 $K$ 张生成图片，哪几张相对更好，哪几张相对更差？**</mark>

$$
\boxed{
A_i = \frac{R_i - \mu_{\mathrm{group}}}{\sigma_{\mathrm{batch}}}
}
$$

接着：

$$
\boxed{
\tilde r_i = \frac{1}{2}\,\mathrm{clip}\left(\frac{A_i}{A_{\max}},\ -1,\ 1\right) + \frac{1}{2}
}
\tag{9}
$$

其中：

$$
\boxed{ A_{\max} = 5 }
$$

### 两个不同范围的统计量

| 统计量 | 定义 |
| :--- | :--- |
| $\mu_{\mathrm{group}}$ | <mark class="hl-key">**同一条 Prompt 的 $K$ 张图的总 Reward 均值**</mark> |
| $\sigma_{\mathrm{batch}}$ | <mark class="hl-key">**当前 Batch 所有样本的总 Reward 标准差**</mark> |

假设一个 Prompt Rollout 24 张图片：

$$
\mu_{\mathrm{group}}(p) = \frac{1}{24}\sum_{j=1}^{24} R_{p,j}
$$

::: danger 一个未公开的实现细节
论文 FLUX.2-dev 的公开配置是每次 48 个 Prompt Group，每组 24 张图片，即一次 Rollout 得到 1,152 张图。<mark class="hl-key">**公式使用当前 Batch 统计，但没有进一步明确 Batch 标准差在分布式实现中覆盖完整 1,152 张图，还是优化时划分的某个 Batch；因此不要擅自将这一实现细节固定下来。**</mark>
:::

### 为什么分子用 Group Mean，分母用 Batch Std？

**分子** $R_i - \mu_{\mathrm{group}}$ 衡量的是<mark class="hl-key">**当前图片相对同一个 Prompt 的其他图片好不好**</mark>。

**分母** $\sigma_{\mathrm{batch}}$ 衡量的是<mark class="hl-key">**这类 Reward 差异相对于整个 Batch 的整体波动有多大**</mark>。

假设：

| Prompt | 4 张图的总 Reward |
| :--- | :--- |
| **Prompt A（简单）** | $[2.8,\ 3.0,\ 3.2,\ 3.4]$ |
| **Prompt B（困难）** | $[-1.0,\ -0.8,\ -0.6,\ -0.4]$ |

如果直接按照绝对总分奖励，<mark class="hl-trick">**Prompt A 的图片会全部压过 Prompt B**</mark>。但在组内去均值后：

$$
A_{\mathrm{centered}} = [-0.3,\ -0.1,\ 0.1,\ 0.3]
$$

$$
B_{\mathrm{centered}} = [-0.3,\ -0.1,\ 0.1,\ 0.3]
$$

<mark class="hl-key">**两个 Prompt 的组内相对关系完全一样**</mark>。这意味着：<mark class="hl-trick">**即使一个困难 Prompt 的绝对 Reward 普遍较低，其中相对更好的 Rollout 仍然可以获得正向训练信号**</mark>。

这和 LLM GRPO 的组相对思想很像，<mark class="hl-key">**但是这里还没有结束：作者使用的是 Batch 标准差而不是每组各自的标准差，最后也不是直接把 $A_i$ 当成 Policy Gradient Advantage**</mark>。

### 为什么最后还要除以 5，映射成 $[0,1]$？

因为 DiffusionNFT 使用的是：

$$
\boxed{
\mathcal{L}_{\mathrm{NFT}} = \tilde r_i \lVert v^+ - v_{\mathrm{target}}\rVert_2^2 + (1 - \tilde r_i)\lVert v^- - v_{\mathrm{target}}\rVert_2^2
}
\tag{2}
$$

所以需要一个介于 0 与 1 之间的权重。

| Group Advantage $A_i$ | NFT 权重 $\tilde r_i$ |
| ---: | ---: |
| $-5$ | 0 |
| $-2$ | 0.30 |
| $-1$ | 0.40 |
| $0$ | 0.50 |
| $+1$ | 0.60 |
| $+2$ | 0.70 |
| $+5$ | 1 |

当 $A_i > 5$ 权重被 Clip 到 1；当 $A_i < -5$ 权重被 Clip 到 0。

::: warning $A_{\max} = 5$ 到底是什么
<mark class="hl-key">**它不是 Reward Range 过滤阈值，也不是 GRPO 的 PPO Clip 参数，而是控制 Advantage 转换为 NFT 权重的缩放和截断常数**</mark>。

同样的 $A_i = 1$，如果不除以 5，经过裁剪后 NFT 权重可能直接达到 1；现在只得到 0.6。

所以作者实际上<mark class="hl-trick">**让组相对优势比较温和地作用于 NFT 正负隐式策略的权重**</mark>。
:::

---

## 69. 为什么分子减 Group Mean，分母却不用 Group Std？

这一节对应公式 (9) 的分母选择。

作者明确使用 <mark class="hl-key">**Group Mean + Batch Std，而不是 Group Mean + Group Std**</mark>。核心原因是：<mark class="hl-trick">**两者虽然都能得到相对 Advantage，但 Group Std 会把不同 Prompt 组的奖励差异强行调整到近似相同的尺度，而 Batch Std 能保留不同组的奖励离散程度差异**</mark>。

### 情况 A：使用 Group Std（标准 GRPO 中常见的形式）

$$
A_i = \frac{R_i - \mu_{\mathrm{group}}}{\sigma_{\mathrm{group}}}
$$

假设一个 Batch 只有两个 Prompt，每条 Prompt Rollout 两张图片：

| Prompt | 图片 1 Reward | 图片 2 Reward | Group Mean | Group Std |
| :--- | ---: | ---: | ---: | ---: |
| **P1** | 0.49 | 0.51 | 0.50 | 0.01 |
| **P2** | 0.10 | 0.90 | 0.50 | 0.40 |

**P1**：

$$
A = [-1,\ +1]
$$

**P2**：

$$
A = [-1,\ +1]
$$

::: danger 问题出现了
<mark class="hl-key">**P1 只有 0.02 的 Reward 差异，P2 却有 0.80 的差异，但两组最终的 Advantage 完全一样。**</mark>

换句话说，<mark class="hl-trick">**Group Std 抹掉了不同 Prompt 组的相对分数差距大小**</mark>。
:::

### 情况 B：使用 Batch Std（Arena 论文）

四张图片的 Batch Std 约为：

$$
\sigma_{\mathrm{batch}} = 0.283
$$

**P1**：

$$
A = \left[-\frac{0.01}{0.283},\ +\frac{0.01}{0.283}\right] = [-0.035,\ +0.035]
$$

**P2**：

$$
A = \left[-\frac{0.40}{0.283},\ +\frac{0.40}{0.283}\right] = [-1.414,\ +1.414]
$$

| 归一化方式 | P1（低差异） | P2（高差异） |
| :--- | :--- | :--- |
| **Group Std** | $[-1,\ +1]$ | $[-1,\ +1]$ |
| **Batch Std** | $[-0.035,\ +0.035]$ | $[-1.414,\ +1.414]$ |

这就产生了一个重要区别：

- <mark class="hl-key">**Group Std：每个 Prompt 组内的相对好坏受到相似强度的优化。**</mark>
- <mark class="hl-key">**Batch Std：Reward 差异较大的组会产生更大的 Advantage；Reward 差异很小的组则产生更弱的信号。**</mark>

### 换成 DiffusionNFT，效果更加直观

代入 $\tilde r_i = \frac12\mathrm{clip}(A_i/5, -1, 1) + \frac12$：

| Prompt | Group Std 得到的 NFT 权重 | Batch Std 得到的 NFT 权重 |
| :--- | :--- | :--- |
| **P1** | $[0.40,\ 0.60]$ | $[0.4965,\ 0.5035]$ |
| **P2** | $[0.40,\ 0.60]$ | $[0.3586,\ 0.6414]$ |

也就是说，使用 Batch Std 时，<mark class="hl-key">**P1 两张几乎同分的图片会得到接近 0.5 的 NFT 权重，而 P2 两张差异明显的图片会得到区分更大的权重**</mark>。

这对于图像 RL 有实际意义：<mark class="hl-trick">**如果 Reward Model 对某个 Prompt 的几张图片只给出非常接近的分数，你未必希望模型像对待一个高差异组那样，强烈区分它们**</mark>。尤其当组内 Reward 方差很小时，<mark class="hl-key">**Group Std 可能放大评分噪声；Batch Std 能缓解这种情况**</mark>。

::: danger 但这是机制性解释，不是论文结论
<mark class="hl-key">**论文并没有单独通过实验验证「Batch Std 一定优于 Group Std」，也没有对公式 (9) 的分母选择提供专门消融。**</mark>
:::

### 那为什么分子还要减 Group Mean？

因为两个操作解决的是不同问题：

$$
\boxed{
A_i = \frac{\overbrace{R_i - \mu_{\mathrm{group}}}^{\text{消除同 Prompt 的奖励基线}}}{\underbrace{\sigma_{\mathrm{batch}}}_{\text{统一缩放 Advantage}}}
}
$$

- <mark class="hl-key">**减 Group Mean**：让每张图片和同 Prompt 的其他生成结果比较，而不是与其他 Prompt 的图片比较，避免简单 Prompt 因为整体得分高而天然占优势。</mark>
- <mark class="hl-key">**除 Batch Std**：让不同 Prompt 组使用同一个缩放尺度，同时保留各组内部奖励差异的相对幅度。</mark>

这其实和 LLM RL 中关于 Group Normalization 是否抹掉 Prompt 难度差异的讨论非常接近。

::: warning Batch Std 有它自己的局限
如果不同 Prompt 的平均 Reward 差异很大，<mark class="hl-key">**那么整个 Batch 的标准差可能主要来自 Prompt 之间的差异，而不是同 Prompt 的生成质量差异**</mark>。

因此，<mark class="hl-trick">**Batch Std 也不是理论上永远最优的选择**</mark>。

另外，原文<mark class="hl-key">**没有进一步明确分布式训练中 `batch` 具体覆盖哪些 GPU、是否进行全局 All-Gather、采用何种方差计算约定**</mark>。公式能够确定统计量的类型，但还不能完全复原底层实现。
:::

::: tip 最终结论
Arena 使用 Group Mean + Batch Std 是确定的；<mark class="hl-key">**这种设计能够避免 Group Std 对低差异组进行过度归一化，但论文没有提供它相对 Group Std 的单独性能消融**</mark>。

这其实给现有的 DiffusionNFT 训练系统提供了一个<mark class="hl-trick">**很有价值、且改动很小的实验**：保持 Prompt、Reward 和训练预算一致，直接比较 Group Mean + Group Std、Group Mean + Batch Std，以及低差异 Group Filtering，看看哪个真正能够改善训练稳定性和最终图像质量。</mark>
:::

---

## 70. 现在把整个 Batch 真正算一遍

<mark class="hl-key">**为了避免只看公式不清楚，我们做一个完整数值例子。**</mark>假设一次训练只有 2 条 Prompt，每条 Rollout 3 张图，总计 6 张图片。

::: warning 这是教学演示
<mark class="hl-key">**不是作者实际使用的 $48 \times 24$ 配置**</mark>。
:::

| Prompt | 原文 |
| :--- | :--- |
| **P1** | *A minimal circular logo with the exact text "ABC", on a plain white background.* — Intent = `strict`，因此三种 Reward 全部启用 |
| **P2** | *An Impressionist painting of a country road.* — Intent = `open`，因此只使用 Preference 和 Faithfulness |

### Step 1：六张图片获得原始 Reward

| 图片 | Prompt | Preference | Faithfulness | Constraint | 触发 Hack |
| :--- | :--- | ---: | ---: | ---: | :---: |
| A | P1 Strict | 3.0 | 0.50 | 0.25 | 否 |
| B | P1 Strict | 5.0 | 0.75 | 1.00 | **是** |
| C | P1 Strict | 4.0 | 1.00 | 0.75 | 否 |
| D | P2 Open | 2.0 | 0.75 | 不适用 | 否 |
| E | P2 Open | 3.5 | 0.50 | 不适用 | 否 |
| F | P2 Open | 5.5 | 1.00 | 不适用 | 否 |

<mark class="hl-key">**所有数值均为模拟数据。**</mark>

::: danger 还有一个边界要特别说明
原文<mark class="hl-key">**没有公开对非 Strict 样本的 Constraint 轴，究竟如何构造有效统计集合、如何处理缺失值**</mark>。

因此下面的示例<mark class="hl-trick">**假设 Constraint 均值和标准差只在启用该轴的样本上计算**</mark>，以说明数学机制。<mark class="hl-key">**这个 Mask 统计选择不是论文已经确认的代码实现**</mark>。
:::

### Step 2：对每个 Reward 轴分别 Z-score

**Preference 轴**，六张图片共同计算：

$$
\mu_{\mathrm{pref}} = 3.833, \qquad \sigma_{\mathrm{pref}} \approx 1.179
$$

**Faithfulness 轴**：

$$
\mu_{\mathrm{faith}} = 0.750, \qquad \sigma_{\mathrm{faith}} \approx 0.204
$$

**Constraint 轴**，在上述演示假设下使用 A、B、C：

$$
\mu_{\mathrm{constr}} = 0.667, \qquad \sigma_{\mathrm{constr}} \approx 0.312
$$

这里为演示采用总体标准差，忽略很小的 $\epsilon$。

| 图片 | $z_{\mathrm{pref}}$ | $z_{\mathrm{faith}}$ | $z_{\mathrm{constr}}$ |
| :--- | ---: | ---: | ---: |
| A | $-0.707$ | $-1.225$ | $-1.336$ |
| B | $+0.990$ | $0.000$ | $+1.069$ |
| C | $+0.141$ | $+1.225$ | $+0.267$ |
| D | $-1.556$ | $0.000$ | — |
| E | $-0.283$ | $-1.225$ | — |
| F | $+1.414$ | $+1.225$ | — |

::: tip 到这里，所有 Reward 都还没有被加在一起
:::

### Step 3：Gating 和 Anti-Hack Veto

P1 是 Strict，启用 Constraint；P2 是 Open，关闭 Constraint。

B 触发了 Hack 检测，所以其 Preference $z_{B,\mathrm{pref}} = 0.990$，经过 Veto：

$$
\bar z_{B,\mathrm{pref}} = 0.990 - 1 \times 0.990 = 0
$$

其他五张图片没有触发 Hack，Preference 保持原样。然后按公式 (8) 得到：

| 图片 | Veto 后 Preference | Faithfulness | 有效 Constraint | 总 Reward $R_i$ |
| :--- | ---: | ---: | ---: | ---: |
| A | $-0.707$ | $-1.225$ | $-1.336$ | **$-3.268$** |
| B | $0.000$ | $0.000$ | $+1.069$ | **$+1.069$** |
| C | $+0.141$ | $+1.225$ | $+0.267$ | **$+1.633$** |
| D | $-1.556$ | $0.000$ | $0$ | **$-1.556$** |
| E | $-0.283$ | $-1.225$ | $0$ | **$-1.508$** |
| F | $+1.414$ | $+1.225$ | $0$ | **$+2.639$** |

::: danger 你会注意到一个非常重要的细节
<mark class="hl-key">**B 虽然触发了 Anti-Hack Veto，但总 Reward 依然是正数。**</mark>

原因是 Veto 只取消了它的正向 Preference，而 Faithfulness 和 Constraint 仍然贡献奖励。

所以 <mark class="hl-trick">**One-sided Veto 并不等于禁止模型学习这张图片，更不等于将整张图片的 NFT Weight 设为 0**</mark>。
:::

### Step 4：Group Mean 和 Batch Std

**P1** 对应 A、B、C：

$$
\mu_{\mathrm{P1}} = \frac{-3.268 + 1.069 + 1.633}{3} \approx -0.189
$$

**P2** 对应 D、E、F：

$$
\mu_{\mathrm{P2}} = \frac{-1.556 - 1.508 + 2.639}{3} \approx -0.141
$$

**全部六张图片的总 Reward 标准差**：

$$
\sigma_{\mathrm{batch}} \approx 2.081
$$

于是按公式 (9)：

$$
A_i = \frac{R_i - \mu_{\mathrm{corresponding\ group}}}{2.081}
$$

再转换成：

$$
\tilde r_i = 0.5 + 0.5\,\mathrm{clip}(A_i/5,\ -1,\ 1)
$$

| 图片 | 总 Reward | Group Advantage | NFT 权重 $\tilde r_i$ |
| :--- | ---: | ---: | ---: |
| A | $-3.268$ | $-1.480$ | **0.3520** |
| B | $+1.069$ | $+0.604$ | **0.5604** |
| C | $+1.633$ | $+0.876$ | **0.5876** |
| D | $-1.556$ | $-0.680$ | **0.4320** |
| E | $-1.508$ | $-0.656$ | **0.4344** |
| F | $+2.639$ | $+1.336$ | **0.6336** |

<mark class="hl-key">**这是模拟 Batch，经过完整 Reward Composition 后的结果；不是论文真实训练日志。**</mark>

::: tip 两个可以自己验证的不变式
1. 每张图从 Preference、Faithfulness、Constraint、Hack Detector 的原始输出，到最后 NFT Weight，<mark class="hl-trick">**经历了两次不同目的的归一化**</mark>；
2. 按照这个公式，<mark class="hl-key">**每个 Prompt 的组内 $A_i$ 均值为 0**</mark>；若所有样本都没有触发 Clip，<mark class="hl-trick">**那么每组最终 $\tilde r_i$ 的平均值恰好是 0.5**</mark>。本例正属于这种情况。
:::

---

## 71. Ideogram-4 的 OCR 怎么加入这个流程？

前面我们只使用三个主 Reward，这是论文公式 (3)–(9) 针对通用主实验写出的形式。但在 Ideogram-4 实验中，作者增加了 OCR 轴。

附录 A.3.4 明确说明，OCR 先计算 $r_i^{(\mathrm{OCR})} \in [0, 1]$，然后使用与其他轴相同的 Batch Z-score 标准化，并设置权重 $w_{\mathrm{OCR}} = 1$。

因此，可以把 Ideogram 版本的总奖励理解成：

$$
\boxed{
R_i^{\mathrm{Ideogram}} = R_i^{\mathrm{main}} + m_i^{\mathrm{OCR}} z_i^{(\mathrm{OCR})}
}
$$

其中 $m_i^{\mathrm{OCR}}$ 表示当前 OCR 评分是否有效。<mark class="hl-key">**没有文字要求的 Prompt，以及转录调用失败的样本，不参加该评分轴**</mark>。

::: warning 这条公式是扩展写法
<mark class="hl-key">**它是根据论文的 OCR 描述对公式 (8) 进行的扩展，不是作者额外给出的编号公式**</mark>。

这里还有一个尚未公开的工程细节：<mark class="hl-trick">**计算 OCR 轴 Batch Mean/Std 时，如何处理被 Mask 的图片，论文没有明确给出完整实现**</mark>。Constraint 轴也存在类似问题。

因此在你自己的 NFT 训练系统里，可以设计有效样本 Mask 和无有效样本时的回退机制，但要区分：<mark class="hl-key">**那属于你的工程实现，而不是作者公开的精确配方**</mark>。
:::

---

## 72. 实际接入 NFT，代码应该长什么样？

下面用简化伪代码，把公式 (3)–(9) 串起来。

::: warning 这段代码适合帮助理解，但不是作者的官方实现
:::

```python
# 所有 Reward 均已完成离线/在线评分
# shape: [B]，B 是当前评分 Batch 的图片数量
pref   = preference_reward
faith  = faithfulness_reward
constr = constraint_reward

# 每张图片对应的 Prompt 元数据
strict_mask = (intent == "strict")

# 每个适用检测器的违规标志
veto = (applicable_hack_flags & detected_hack_flags).any(dim=-1)


def zscore(x, eps=1e-6):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


# 公式 (3)：分别标准化
z_pref  = zscore(pref)
z_faith = zscore(faith)

# 论文未明确 Constraint Mask 的统计细节；
# 这里采用一种示意性的 valid-only 计算。
z_constr = torch.zeros_like(pref)
if strict_mask.sum() > 1:
    z_constr[strict_mask] = zscore(constr[strict_mask])

# 公式 (7)：只清除作弊样本的正向 Preference
z_pref_safe = z_pref - veto.float() * z_pref.clamp(min=0)

# 公式 (8)：组合三个 Reward
R = (
    z_pref_safe
    + z_faith
    + strict_mask.float() * z_constr
)

# 公式 (9)：同 Prompt 组内去均值
# group_ids 标识每张图片属于哪个 Prompt。
group_mean = compute_group_mean(R, group_ids)

# 当前 Batch 的总 Reward 标准差
batch_std = R.std(unbiased=False).clamp_min(1e-6)

A = (R - group_mean) / batch_std

# 映射到 DiffusionNFT 所需的 [0, 1] 区间
r_nft = 0.5 * torch.clamp(A / 5.0, -1.0, 1.0) + 0.5
```

随后：

$$
\mathcal{L}_{\mathrm{NFT}} = \tilde r_i \lVert v^+ - v^\* \rVert^2 + (1 - \tilde r_i)\lVert v^- - v^\* \rVert^2
$$

<mark class="hl-key">**这里 $v^+$、$v^-$ 的构造仍然使用标准 DiffusionNFT，论文没有对该部分提出新 Loss。**</mark>

::: danger 代码里需要额外注意三件事
1. 如果 <mark class="hl-key">**Batch 总 Reward 方差为零**</mark>，需要数值稳定处理；
2. 如果<mark class="hl-key">**某个轴没有有效样本**</mark>，需要明确回退行为；
3. 如果<mark class="hl-key">**多 GPU 分布式训练**</mark>，必须确定均值、方差究竟在哪个统计集合上聚合。

<mark class="hl-trick">**论文没有逐项公开这些实现，不能从公式自动推断它的分布式通信策略。**</mark>
:::

---

## 73. 作者做了哪些实验来证明 Reward Composition 有效？

论文 §4.3 对三个核心设计分别进行了消融。

| 机制 | 比较设置 | Arena RM + Gemini 胜率 |
| :--- | :--- | ---: |
| **Constraint Gate** | 所有 Prompt 启用 Constraint | 60.46% |
| **Constraint Gate** | 仅 Strict 启用 | **64.16%** |
| **Anti-Hack Applicability** | 不判断适用性 | 55.91% |
| **Anti-Hack Applicability** | 使用 Prompt Gate | **63.46%** |
| **Anti-Hack 组合方式** | 将合规评分作为加性 Reward | 52.60% |
| **Anti-Hack 组合方式** | 使用 One-sided Veto | **63.46%** |

::: danger 需要严谨地理解这张表
<mark class="hl-key">**这是三组不同的消融，不代表六个数来自同一条连续叠加的训练曲线。**</mark>

它们支持三个结论：<mark class="hl-key">**Constraint 不能不分任务地启用；Hack 检测需要适用性判断；在作者比较的设置下，Veto 比直接增加合规奖励更有效**</mark>。

但<mark class="hl-trick">**论文没有单独提供一个完整的「原始 Reward 直接相加 vs 分轴 Z-score 标准化」的数值消融，因此不能把上述提升全部归因于公式 (3) 的标准化策略**</mark>。
:::

---

## 74. 必须注意的三个技术细节

::: tip 第一个：这里实际上包含两层标准化，但它们解决的是不同问题
$$

$$
\underbrace{ r_i^{(a)} \rightarrow z_i^{(a)} }_{\text{不同 Reward 维度可比较}}
\qquad
\underbrace{ R_i \rightarrow A_i \rightarrow \tilde r_i }_{\text{同 Prompt Rollout 的相对优劣}}
$$

<mark class="hl-key">**前者解决 Reward Scale，后者解决 Rollout Preference。不能省略其中一层，也不能随便交换它们的顺序，因为 Gating 和 Veto 都发生在中间。**</mark>

::: tip 第二个：Veto 并不是样本级硬过滤
§70 中 B 图片触发 Hack，但其最终 NFT 权重仍然是 **0.5604**。这说明<mark class="hl-key">**作者是在限制作弊行为能够获得的正向 Preference 信号，不是直接禁止该样本获得一切正向训练信号**</mark>。

如果你想实现「触发违规则整体不能成为正样本」，<mark class="hl-trick">**需要额外修改 Reward Composition，那已经不是本文的 One-sided Veto 公式了**</mark>。
:::

::: tip 第三个：这种 Reward 组合方法并没有自动解决所有数据问题
例如 <mark class="hl-key">**Faithfulness Checklist 漏了一个重要条件、Constraint Intent 误判、VLM Judge 对相同图片不稳定**</mark>，这些错误仍会通过公式 (8) 进入训练。

而公式 (9) 的组相对标准化，<mark class="hl-trick">**只能衡量同一组图片的相对 Reward，不能判断 Reward 自身是否正确**</mark>。

因此，<mark class="hl-key">**更复杂的 Reward 组合，并不能替代对 Checklist、Intent 标签和 Judge 的人工审计**</mark>。
:::

---

## 75. Part 8 最终总结

这篇论文的 Multi-Reward 方案可以压缩为五条真正需要记住的公式：

$$
\begin{aligned}
&\textbf{分轴标准化：} && z_i^{(a)} = \frac{r_i^{(a)} - \mu^{(a)}}{\sigma^{(a)} + \epsilon} \\[6pt]
&\textbf{作弊否决：} && \bar z_i^{\mathrm{pref}} = z_i^{\mathrm{pref}} - v_i \bigl[z_i^{\mathrm{pref}}\bigr]_+ \\[6pt]
&\textbf{总奖励：} && R_i = \bar z_i^{\mathrm{pref}} + z_i^{\mathrm{faith}} + g_i^{\mathrm{constr}} z_i^{\mathrm{constr}} \\[6pt]
&\textbf{组相对优势：} && A_i = \frac{R_i - \mu_{\mathrm{group}}}{\sigma_{\mathrm{batch}}} \\[6pt]
&\textbf{NFT 权重：} && \tilde r_i = \frac{1}{2}\operatorname{clip}\left(\frac{A_i}{5}, -1, 1\right) + \frac{1}{2}
\end{aligned}
$$

<mark class="hl-key">**Part 8 的关键贡献不是提出新的 NFT 优化器，而是将多种尺度不同、适用条件不同、可靠性不同的 Reward，组织成可以实际用于同一套图像 RL 训练的信号**</mark>。

至于这个方案是否是最优的 Reward Composition，论文还没有充分回答。例如<mark class="hl-trick">**Gate 的误判影响、Reward 之间的相关性、标准化统计集合、$A_{\max} = 5$ 的敏感性，都缺乏系统消融**</mark>。这些问题很值得后续在现有 NFT 训练中验证。

**下一部分 Part 9：Stage 1 / Stage 2 / Stage 3。** 我们会把这五个 Reward 轴（Part 3–7 研究的）放进实际训练流程，看作者如何把它们组织成三阶段训练，以及最后如何用一次权重融合得到部署模型。

---

# Part 9：Stage 1 / Stage 2 / Stage 3

> **对应原文**：§4.1 Implementation details、§3.3.5、Table 2

这篇论文的 Stage 1、Stage 2、Stage 3，<mark class="hl-key">**本质上是两次不同 Reward 配方的 RL 训练，加一次模型权重融合**</mark>。

$$
\begin{aligned}
&\textbf{FLUX.2-dev Base Model} \\
&\quad \downarrow \\
&\quad \textbf{Stage 1：常规 RL} \\
&\qquad \text{Preference + Faithfulness + Constraint Gate} \Rightarrow \textbf{LoRA A} \\
&\quad \downarrow \\
&\quad \textbf{Stage 2：Anti-Hack RL} \\
&\qquad \text{Stage 1 配方 + Anti-Hack Veto} \Rightarrow \textbf{LoRA B} \\
&\quad \downarrow \\
&\quad \textbf{Stage 3：Weight Soup} \\
&\qquad \Delta W = \tfrac{1}{2}\Delta W_A + \tfrac{1}{2}\Delta W_B
\end{aligned}
$$

<mark class="hl-key">**直接融合 LoRA，不再进行 RL 训练。**</mark>

---

## 76. Stage 1：正常的 Multi-Reward RL

使用三个 Reward（此处省略标准化细节）：

$$
R_1 = R_{\mathrm{pref}} + R_{\mathrm{faith}} + g_{\mathrm{strict}} R_{\mathrm{constr}}
$$

其中<mark class="hl-key">**每个 Reward 都先经过标准化，Constraint 只对 Strict Prompt 启用**</mark>。

<mark class="hl-key">**目标：同时提升美学质量、指令遵循能力和严格约束满足程度。**</mark>训练 150 次更新，从中选择最佳 Checkpoint。

---

## 77. Stage 2：引入 Anti-Reward-Hacking

作者检查 Stage 1 最佳 Checkpoint 的 205 个失败案例，发现乱码文字和过度摄影写实两种典型问题。

因此，使用原来的三个 Reward，<mark class="hl-key">**同时加入 Anti-Hack Detector 和 One-sided Veto**</mark>：

$$
R_2 = \underbrace{\bar R_{\mathrm{pref}}}_{\text{发生违规时清除正向信号}} + R_{\mathrm{faith}} + g_{\mathrm{strict}} R_{\mathrm{constr}}
$$

<mark class="hl-key">**目标：防止模型为了获得较高 Preference Score 而生成乱码、偏离用户指定风格。**</mark>

作者为这一套 Reward 配方训练第二个模型。

::: danger 这里有一个必须保留的提醒
<mark class="hl-key">**原文称其为第二个训练 Run，没有明确披露它是不是从 Stage 1 的权重继续训练**</mark>，因此<mark class="hl-trick">**不能直接理解为 Stage 1 之后接着训 150 步**</mark>。
:::

---

## 78. Stage 3：融合两个 LoRA

Stage 1 和 Stage 2 各自存在优势：

| 父模型 | 优势 |
| :--- | :--- |
| **Stage 1** | <mark class="hl-key">**整体构图和风格表现较好**</mark> |
| **Stage 2** | <mark class="hl-key">**指令遵循和干净的文字渲染更有优势**</mark> |

因此，作者把两个最佳 Checkpoint 的 LoRA 更新量按 $0.5 : 0.5$ 融合：

$$
\boxed{
\Delta W_{\mathrm{final}} = 0.5\, \Delta W_{\mathrm{Stage1}} + 0.5\, \Delta W_{\mathrm{Stage2}}
}
$$

<mark class="hl-key">**Stage 3 没有额外的 Rollout、Reward 打分或 `optimizer.step()`。它仅进行权重空间融合。**</mark>

<mark class="hl-trick">**Arena RM 版本通过 LoRA Rank Concatenation 精确实现更新量平均**</mark>（机制详见 Part 10）。

---

## 79. 效果如何？

FLUX.2-dev，使用 Arena RM 训练、Gemini 评估，<mark class="hl-key">**相对冻结 FLUX.2-dev 的配对胜率；数值来自论文 Table 2**</mark>：

| 模型 | Gemini 评估胜率 |
| :--- | ---: |
| **Stage 1** | 64.2% |
| **Stage 2** | 63.5% |
| **Stage 3 融合** | **66.0%** |

<mark class="hl-key">**可以看到，Stage 2 单独训练并没有超过 Stage 1，但融合后的 Stage 3 更好。**</mark>

::: warning 不要过度解读
这与 Part 6 §50 的结论一致：<mark class="hl-trick">**收益来自融合，而不是 Anti-Hack 单项**</mark>。
:::

---

## 80. 一句话总结

$$
\underbrace{\text{Stage 1 学整体生成质量}}_{\text{Pref + Faith + Constraint}}
\;\longrightarrow\;
\underbrace{\text{Stage 2 用 Anti-Hack Reward 修正已发现的问题}}_{\text{+ Veto}}
\;\longrightarrow\;
\underbrace{\text{Stage 3 将两套互补的 LoRA 融合成最终模型}}_{\text{权重平均}}
$$

另外，<mark class="hl-key">**这三个阶段是 FLUX.2-dev 实验的设计。Ideogram-4 只做了 Stage 1，因为作者没有观察到需要单独处理的明显 Reward-Hacking 模式**</mark>。

**下一部分 Part 10：LoRA 权重融合的原理。** Part 10 本身没那么复杂——作者就是训练了两个 LoRA，然后把两个 LoRA 对基础模型产生的修改取平均，得到一个新的 LoRA。

---

# Part 10：LoRA 权重融合与 Rank Concatenation

> **对应原文**：§3.3.5 公式 (10)、附录 A.6 Weight-soup merge recipes

<mark class="hl-key">**作者就是训练了两个 LoRA，然后把两个 LoRA 对基础模型产生的修改取平均，得到一个新的 LoRA。**</mark>我们先不管那些矩阵公式，直接看它怎么操作。

---

## 81. 整个过程只有三步

$$
\begin{aligned}
&\textbf{原始 FLUX.2 模型} \\
&\quad \downarrow \\
&\quad \textbf{Stage 1} \Rightarrow \text{正常 Multi-Reward 训练} \Rightarrow \textbf{LoRA A} \\
&\quad \downarrow \\
&\quad \textbf{Stage 2} \Rightarrow \text{Multi-Reward + Anti-Hack 训练} \Rightarrow \textbf{LoRA B} \\
&\quad \downarrow \\
&\quad \textbf{Stage 3：融合两个 LoRA} \\
&\qquad \textbf{50\% LoRA A + 50\% LoRA B} \Rightarrow \textbf{LoRA C}
\end{aligned}
$$

<mark class="hl-key">**Stage 3 不训练、不生成 Rollout、不计算 Reward。只是合并参数。**</mark>

---

## 82. 什么叫把两个 LoRA 融合？

假设 FLUX 某一层原始权重是：

$$
W_0 = 100
$$

经过 Stage 1 训练，LoRA A 对这个权重的修改是：

$$
\Delta W_A = +10
$$

经过 Stage 2 训练，LoRA B 对这个权重的修改是：

$$
\Delta W_B = +6
$$

那么 Stage 3 就是求平均：

$$
\Delta W_C = \frac{10 + 6}{2} = 8
$$

最终模型对应的权重就是：

$$
W_{\mathrm{final}} = 100 + 8 = 108
$$

::: tip 真实模型当然不是只有一个数字
而是<mark class="hl-key">**每层都有一个很大的权重矩阵。作者做的就是对这些矩阵修改量逐元素取平均**</mark>。
:::

---

## 83. 那论文里为什么还提 Rank Concatenation？

因为 LoRA 并不是直接保存完整的 $\Delta W$，而是用两个小矩阵来表示：

$$
\Delta W = BA
$$

<mark class="hl-key">**先忽略 LoRA 的缩放系数。**</mark>假设两个训练好的 LoRA 分别是：

$$
\Delta W_1 = B_1 A_1
$$

$$
\Delta W_2 = B_2 A_2
$$

作者想得到：

$$
\boxed{
\Delta W_C = \frac{1}{2}B_1 A_1 + \frac{1}{2}B_2 A_2
}
$$

最容易理解的做法：<mark class="hl-trick">**先把每个 LoRA 的 `B @ A` 算出来，再相加、除以 2**</mark>。

但是，<mark class="hl-key">**这样可能需要构造一个很大的完整权重矩阵**</mark>。

因此作者使用一个巧妙的办法：<mark class="hl-trick">**把两个 LoRA 的小矩阵拼接起来，让它们在相乘时自动完成这个平均**</mark>。这就叫 **Rank Concatenation**。

::: tip 你不需要记住拼接的具体维度
只需要知道：

- 原来有两个 <mark class="hl-key">Rank 64</mark> 的 LoRA；
- 拼接后得到一个 <mark class="hl-key">Rank 128</mark> 的 LoRA；
- 这个新 LoRA 对模型产生的权重修改，<mark class="hl-key">**恰好等于原来两个 LoRA 修改量的平均值**</mark>。

所以 <mark class="hl-trick">**Rank Concatenation 不是新的训练算法，只是一种不用展开完整大矩阵，就能精确合并 LoRA 的实现方法**</mark>。
:::

---

## 84. 为什么融合后效果会更好？

因为两个 LoRA 学到的东西有所不同：Stage 1 侧重整体生成质量；Stage 2 加入了防止乱码和风格漂移的机制。

作者发现<mark class="hl-key">**两者的失败案例并不完全重合**</mark>，所以尝试把两个模型的参数更新取平均。最后的实验结果见 Part 9 §79：

| 模型 | Gemini 评估胜率 |
| :--- | ---: |
| **Stage 1** | 64.2% |
| **Stage 2** | 63.5% |
| **Stage 3 融合** | **66.0%** |

融合后确实有所改善，<mark class="hl-trick">**不过这只是作者在该实验设置中的结果，不代表任何两个 LoRA 平均后都会变好**</mark>。

---

## 85. 核心总结

现在你只需要记住这一句话：

$$
\boxed{
\text{最终模型}
=
\text{原始 FLUX}
+ 0.5 \times \text{LoRA A 的修改量}
+ 0.5 \times \text{LoRA B 的修改量}
}
$$

其中 <mark class="hl-key">**Rank Concatenation 只是实现这个公式的一种技巧**</mark>。
---

## 86. 本笔记与 Reward Hacking 子专题的关系

本篇 **Part 1、Part 6、Part 8** 的内容（Goodhart 实证、Anti-Hack detector、One-sided Veto、$A_{\max}=5$ 的截断信任域）已在新建立的 [🚨 RL Reward Hacking 与分布偏移](./reward-hacking/) 里建立索引。

| 本笔记位置 | 归入该专题的哪个问题 |
| :--- | :--- |
| §2.3 / §39 | Goodhart：单一偏好 RM 训练让 faithfulness 胜率跌破 50% 并持续下滑 |
| §44–§52 | 攻击面：205 个失败案例审计出的两种固定 detector |
| §49 | 防御面：为什么用 One-sided Veto 而不是加一项 anti-hack reward |
| §64–§75 | 分布侧：分轴归一化 + 组内 $A_i$ + $\tilde r$ 的截断信任域 |
