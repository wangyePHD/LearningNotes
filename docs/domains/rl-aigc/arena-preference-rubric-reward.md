# Arena-RL：Preference × Rubric 复合奖励后训练前沿 T2I 基模 (Arena)

> **标签**：`RL` `Reward System` `Rubric` `Bradley–Terry` `Reward Hacking` `DiffusionNFT` `Vision`
> **更新时间**：2026-10-12
> **arXiv**：[2610.02967](https://arxiv.org/html/2610.02967)
> **标题**：*Post-Training Frontier Text-to-Image Models by Composing Preference and Rubric Rewards*
> **涉及基模**：FLUX.2-dev · Ideogram-4
> **精读进度**：**Part 1–2 完结**
> **Part 1**：Abstract / §1 Introduction / §2 Related Work → 见 §1–§8
> **Part 2**：§4.1 Training and Evaluation Prompts / 附录 A.2.2 Training-prompt Filtering → 见 §9–§15
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