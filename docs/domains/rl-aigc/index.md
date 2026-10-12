# 🎲 AIGC 强化学习后训练 (RL for AIGC)

> **标签**：`RL` `Reward System` `Rubric` `DiffusionNFT` `Reward Hacking` `Data Pipeline`
> **更新时间**：2026-10-12

本专区聚焦 **扩散 / 流匹配生成模型的后训练阶段**：优化器（NFT / GRPO / DPO）× 奖励体系（Preference RM / Verifiable / Rubric）× Prompt 与标注数据管线 × 评测协议。

与 `图像与视觉` 域的分工：本域不讨论基模架构、VAE、蒸馏结构，只讨论**「用什么信号把已经很强的基模继续推上去」**。基模侧论文里的 RL 章节若以奖励体系为主要研究变量，也收录于本域并在原笔记中链回。

---

## 收录标准

| 判据 | 说明 |
| :--- | :--- |
| ✅ 收录 | 奖励体系（多奖励组合 / Rubric / VLM Judge）为主要研究变量 |
| ✅ 收录 | RL Prompt / Checklist / 偏好对的生产与过滤管线 |
| ✅ 收录 | Reward Hacking 防御、One-sided Veto、权重空间集成等鲁棒性机制 |
| ❌ 不收 | 仅提出新 Policy Optimization 算法且奖励为现成单信号的（归 `vision/image-rl-posttraining/` 算法篇） |
| ❌ 不收 | RL 只作为基模训练三阶段中的一步、且无独立 RL 分析的（留在 `vision/` 原笔记） |

---

## 论文篇

| 篇目 | 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **Arena-RL：Preference × Rubric 复合奖励后训练前沿 T2I 基模** | **算法不变、只动奖励侧**。核心命题：高质量 RL = Reward Coverage + Reward Composition + Reward Robustness。四类奖励（Preference RM / presence-oriented Faithfulness / absence-oriented Intent-gated Constraint / Anti-Hack）+ 四个组合机制（Reward-decoupled Normalization / Prompt-conditioned Gating / One-sided Veto / Weight Ensembling）。FLUX.2-dev +69 Elo、Ideogram-4 达 1223.5。**Part 2**：Arena 真实用户 Prompt → Gemini-3.5-Flash 六条 VALID 规则二值过滤 → 10K 训练 / 1K 验证；100K 未过滤比 10K 过滤低 **0.0360** 胜率 | 🔄 Part 1–2 完结 | [阅读笔记 →](./arena-preference-rubric-reward.md) |

---

## 本域待补主线（阅读路线图）

1. **Part 3：Human Preference Reward Model 的数据生产与训练** —— Arena 约 500 万对人类投票怎样组织成 RM 训练数据，Qwen3.6-27B 如何通过 Bradley–Terry Loss 学习偏好
2. **Part 8：Reward Composition 公式精读** —— 以公式 (3)–(9) 为准（引言的 $[0,1]$ 表述与 §3.3.1 的逐维 z-score 不一致）
3. **Rubric 奖励体系** —— Faithfulness / Constraint checklist 与 Anti-Hack detector 的逐条定义

---

## 关联笔记

- [图像基模 RL 后训练专题（2026）](../vision/image-rl-posttraining/) — 算法篇 / 基模 RL 篇总索引
- [训练 Playbook v1.0](../vision/training-playbook.md) — 「多 reward 打架 → Normalization/Routing」诊断表
- [Sol-RL：FP4 Explore, BF16 Train](../vision/image-rl-posttraining/sol-rl.md) — 扩散 RL 的 rollout 预算解耦
- [GDRO 组级直接奖励优化](../vision/image-rl-posttraining/gdro-group-reward-2026.md) — 本文的 GDPO 式解耦归一化出处
- [Flow-GRPO 综述](../vision/image-rl-posttraining/flow-grpo-survey-2026.md) — OCR / GenEval 等可验证奖励的算法侧接入