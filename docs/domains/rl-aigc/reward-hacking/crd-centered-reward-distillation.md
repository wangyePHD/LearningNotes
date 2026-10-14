# CRD：用中心化奖励蒸馏做 Diffusion RL（Diffusion Reinforcement Learning via Centered Reward Distillation)

> **标签**：`Reward Hacking` `Distribution Drift` `Forward-Process RL` `Centered Reward Distillation` `KL` `Flow-GRPO` `DiffusionNFT` `SD3.5`
> **更新时间**：2026-10-13
> **arXiv**：[2603.14128](https://arxiv.org/abs/2603.14128)
> **涉及基模**：SD3.5-Medium
> **精读进度**：**Part 1 完结**（Abstract / §1 Introduction / §2 Related Work）
> **专题**：🚨 [RL Reward Hacking 与分布偏移](./index.md) · 🎲 [AIGC 强化学习后训练](../index.md)

::: tip 归属说明
本篇的核心问题**就是**「Forward-Process RL 会不会漂移、会不会被 reward 骗」，因此整篇收在本专题，而非 rl-aigc 本层（那里关心多奖励怎么组合）。
:::

---

## 1. 研究背景：为什么需要 Diffusion RL？

预训练主要让模型学习**数据分布**，而不是直接针对具体能力做优化。因此 Diffusion / Flow Matching 模型在文字渲染、复杂物体组合、细粒度 Prompt 遵循这几项上仍然不足。

于是引入 RL 做后训练：生成图片后用外部 Reward 打分（OCR 判文字、GenEval 判物体数量/颜色/空间关系），再用奖励信号更新模型。

<mark class="hl-key">**但随之出现的问题是：RL 能提高特定任务的 Reward，却可能破坏模型原本的图像生成质量。**</mark>

---

## 2. 现有 Diffusion RL 的两条路线

### 2.1 反向去噪过程 RL —— 代表 Flow-GRPO

把整个图像生成看作多步决策过程：从噪声出发逐步去噪，<mark class="hl-key">**每个去噪步骤是一次决策**</mark>，用最终图像的奖励来优化整条去噪过程。

**优势**：可直接复用传统 RL 的策略梯度、优势函数、重要性采样。

| 问题 | 说明 |
| :--- | :--- |
| **开销大** | 需要处理完整采样轨迹，计算与显存开销都大 |
| **梯度方差高** | 影响训练稳定性与收敛速度 |

### 2.2 Forward Process RL —— 代表 GPO / AWM / DiffusionNFT / CRD

<mark class="hl-key">**思路反过来：先让模型生成完整图像并算 Reward，再对生成结果重新加噪，用这些样本构造训练目标**</mark>——不必沿随机去噪轨迹算策略梯度。

| 对比项 | Flow-GRPO | Forward-Process RL |
| :--- | :--- | :--- |
| 梯度来源 | 整条去噪轨迹 | 完整图 + 重加噪 |
| 训练效率 | 低 | <mark class="hl-key">**高、收敛快**</mark> |
| 代价 | 方差高 | <mark class="hl-trick">**Distribution Drift + Reward Hacking**</mark> |

---

## 3. Distribution Drift 与 Reward Hacking 是两个不同的东西

| 概念 | 定义 | 危害对象 |
| :--- | :--- | :--- |
| **Distribution Drift（分布漂移）** | 多轮 RL 更新后，生成结果**逐渐偏离原始预训练模型的分布** | 模型整体 |
| **Reward Hacking（奖励投机）** | 模型找到**拿高 Reward 的捷径**，但这条捷径不一定是真正想完成的任务 | 优化目标 |

### 3.1 论文给的概念性例子

要求生成一间教室，墙上时钟写着 `Time for Lunch`。

- **正常**：桌椅 + 时钟 + 相关文字构成的完整教室场景。
- ** hacking 后**：直接在简单背景上写出 `Time for Lunch`。<mark class="hl-trick">**文字正确了，场景却消失了。**</mark>

### 3.2 论文的实证（Figure 5）

Figure 5 展示了更极端的退化：<mark class="hl-key">**模型满足了文字识别奖励，却丧失了场景一致性**</mark>。

结论：<mark class="hl-key">**Reward 高 ≠ 图像生成质量高。**</mark>

### 3.3 作者对漂移来源的判断

<mark class="hl-key">**现有 Forward-Process RL 中「参考模型不断更新」的机制，是长期分布漂移的重要原因之一。**</mark>

所以本文要解决的问题被表述为：**如何在提高 Reward 的同时控制生成分布的变化。**

---

## 4. CRD 的解决思路

**Centered Reward Distillation**，目标是让 Diffusion RL 既高效提升目标能力，又尽量避免训练中的分布漂移与图像质量退化。

作者从两个方面改进：

| 方向 | 做法 |
| :--- | :--- |
| **① 改进「模型怎么学 Reward」** | 不只看单张图的 Reward 绝对值，还利用**同一 Prompt 下多张生成图之间的相对奖励关系**构建更合理的训练目标 |
| **② 改进「怎么约束分布」** | 让模型持续学更高 Reward 的生成方式，同时不丢掉原始模型已具备的高质量生成能力 |

::: warning 我们自己的抽象（**非论文公式**）
「绝对值 → 组内相对关系」这一改动，可以理解为把训练目标从**逐样本的 reward 回归**换成**组内排序/对比式的目标**；
而分布约束则对应在目标里显式加入一个**把生成分布拉回参考分布的惩罚项**。
:::
具体数学原理与算法实现留给后面的 Parts。

---

## 5. 实验：SD3.5-Medium

两类实验：**GenEval**（复杂组合关系生成）与 **OCR**（文字渲染）。

Table 1（OCR）：

| 方法 | OCR ↑ | Aesthetic ↑ | ImageReward ↑ |
| :--- | :--- | :--- | :--- |
| 原始 SD3.5-M | 0.59 | 5.39 | 0.87 |
| Flow-GRPO（有 KL） | 0.92 | 5.32 | 0.95 |
| DiffusionNFT | **0.97** | 4.89 | **-0.81** |
| **CRD** | 0.92 | **5.33** | 0.87 |

> OCR 与图像质量指标采用**不同的评测 Prompt 集合**。

<mark class="hl-key">**DiffusionNFT 拿到最高 OCR，但其他图像质量指标明显下降；CRD 的 OCR 略低，却保住了基模的图像质量。**</mark>

CRD 保留了 Forward-Process RL 的效率优势（收敛快）。<mark class="hl-trick">**但 CRD 并非在所有指标上都超过 Flow-GRPO——论文强调的是奖励优化、训练效率与生成质量之间的综合权衡。**</mark>

---

## 6. 与已有方法的关系

| 方法 | 核心特点 | 主要关注的问题 |
| :--- | :--- | :--- |
| **Flow-GRPO** | 基于去噪轨迹的策略优化 | RL 策略梯度优化 |
| **GPO** | 基于 Forward Process 的概率估计 | 高效 RL 训练 |
| **AWM** | Advantage-Weighted Matching | 降低梯度方差 |
| **DiffusionNFT** | 基于最优速度方向进行更新 | 高效 Reward 优化 |
| **CRD** | Centered Reward Distillation | **稳定的 Reward 学习与分布漂移控制** |

<mark class="hl-key">**GPO、AWM、DiffusionNFT、CRD 同属 Forward-Process RL 路线，区别只在优化目标的构造方式。**</mark>

---

## 7. 从我们的角度：Part 1 最应该学到什么

::: tip Table 1 的 ImageReward = -0.81 是本篇最刺眼的一个数字
DiffusionNFT 的 **OCR 最高（0.97）但 ImageReward 掉到 -0.81**，而 Aesthetic 只从 5.39 降到 4.89。
<mark class="hl-key">**两个指标的反向幅度不对等，说明「奖励轴之间发生了此消彼长，而不只是整体变差」**</mark>——这正是我们此前在 Arena-RL 记录的「逐轴独立 z-score 后等权相加」可能掩盖的失效模式。
:::

::: danger Distribution Drift 与 Reward Hacking 不要混为一谈
漂移是**分布层面**的漂移（模型整体走样），hacking 是**目标层面**的投机（模型钻空子）。
<mark class="hl-key">**两者的干预点不同**：漂移要对齐参考分布，hacking 要重新审视奖励本身的可投机性。**</mark>
论文把「参考模型不断更新」判为漂移主因，意味着<mark class="hl-trick">**Freeze reference 可能是一个被低估的简单基线**</mark>。
:::

::: tip 这个子专题为什么值得单独存在
Arena-RL 的 anti-hack 是**事后检测 + 单边否决**（One-sided Veto），RationalRewards 那边是**阈值过滤**，两者都是在奖励链路上打补丁。
<mark class="hl-key">**CRD 走的是另一条路：在目标函数层面就避免模型有机会走偏**</mark>——「事前防」与「事后堵」是两条技术路线，值得并列比较。
:::

::: danger Part 1 未回答的关键问题
- 「组内相对奖励关系」具体怎么构成损失？与组内归一化后的 $A_i$ 是什么关系？
- 分布约束是 KL 惩罚还是其他形式？「参考模型不断更新」被换成了什么？
- CRD 与 DiffusionNFT **是否可以叠加**（都在 Forward-Process 线上）？
- GenEval 结果如何？OCR 好但组合关系退化是普遍现象吗？
:::

---

## 8. Part 1 小结与下一 Part 预告

**核心主张一句话**：Forward-Process RL 用效率换来了分布漂移与奖励投机，CRD 靠「组内相对奖励学习 + 分布约束」把这两点一起按住。

**已覆盖范围**：Abstract / §1 Introduction / §2 Related Work。

---

**下一部分 Part 2：KL-Regularized RL 的理论基础。**

我们再正式进入数学推导，理解为什么 Reward Maximization 可以转化为概率分布优化，以及它如何成为 CRD 的理论起点。