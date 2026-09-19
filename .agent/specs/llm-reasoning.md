# 💬 语言模型与推理思考 (LLM & Reasoning) 细粒度编写规范

当用户让 Agent 记录大语言模型架构、后训练（Post-Training）、强化学习（RLHF / DPO / GRPO）、长思维链（CoT）推理涌现等相关笔记时，必须遵循本规范。

---

## 1. 结构大纲标准 (必须包含以下 6 个小节)

```markdown
# [中文标题] ([英文机制/算法])

> **标签**：`LLM` `Reasoning` `RL` `[其他标签]`  
> **更新时间**：YYYY-MM-DD  
> **参考来源**：[论文标题/链接] · [技术报告]

---

## 1. 核心理论背景与思维链范式
- 阐述该研究突破了什么瓶颈（如从纯模仿学习 SFT 到强化学习自主探索、长思维链顿悟、搜索空间剪枝）。

## 2. 优化目标函数数学表达
- 完整列出策略更新的数学目标函数（如 PPO / GRPO / DPO 目标式）：
  $$ \mathcal{J}(\theta) = \mathbb{E}\left[ \dots \right] $$
- 明确标注重要性采样比率 $\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}$、裁剪因子 $\epsilon$ 以及 KL 散度惩罚项 $\beta D_{\text{KL}}$；
- 给出优势函数（Advantage $\hat{A}$）的组内归一化或价值预估公式。

## 3. 奖励机制设计与抗欺骗 (Reward Engineering)
- 详述奖励信号来源（规则奖励如编译通过/答案正确、模型奖励 RM、过程监督 PRM）；
- 分析如何防范“奖励作弊（Reward Hacking）”与无意义的长篇思维链膨胀（Length Bias）。

## 4. 显存开销与训练系统对比
- 深度对比 Actor、Critic、Reference、Reward 模型的显存占用公式；
- 用表格量化无 Critic 架构（如 GRPO）在单卡/分布式节点上的显存节约比例与吞吐提升。

## 5. 核心损失函数 PyTorch 实现片段
- 提供标准的损失计算函数（包含组内标准化、clip 截断与 KL 正则化计算）。

## 6. 自省反思与评测基准
- 包含典型 Benchmark（如 MATH-500、AIME 2024、GPQA Diamond、LiveCodeBench）；
- 用 `::: tip 关键发现` 记录模型自发涌现自我修正（Self-Correction）或顿悟（Aha Moment）的临界条件。
```
