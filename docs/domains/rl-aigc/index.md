# 🎲 AIGC 强化学习后训练 (RL for AIGC)

> **标签**：`RL` `Reward System` `Rubric` `DiffusionNFT` `Reward Hacking` `Data Pipeline`
> **更新时间**：2026-10-12

本专区聚焦 **扩散 / 流匹配生成模型的后训练阶段**：优化器（NFT / GRPO / DPO）× 奖励体系（Preference RM / Verifiable / Rubric）× Prompt 与标注数据管线 × 评测协议。

与 `图像与视觉` 域的分工：本域不讨论基模架构、VAE、蒸馏结构，只讨论**「用什么信号把已经很强的基模继续推上去」**。基模侧论文里的 RL 章节若以奖励体系为主要研究变量，也收录于本域并在原笔记中链回。

本专区下设两个**子专题**：

- 🏅 [偏好模型 (Preference Model)](./preference-model/)——研究 **Reward Model 本身怎么造**（结构、数据、蒸馏、接口）
- 🚨 [RL Reward Hacking 与分布偏移](./reward-hacking/)——研究 **优化过程里奖励与分布怎么被钻空子**（Goodhart 饱和、mode collapse、hack 检测与防御、分布塌缩）

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

## 子专题

| 子专题 | 关注问题 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **🏅 偏好模型 (Preference Model)** | **打分器本身怎么造**：结构（scalar vs reasoning-based）、pairwise→pointwise 投影、偏好标签转 Reasoning 监督、评测口径 | 🔄 建设中 | [进入子专题 →](./preference-model/) |
| **🚨 RL Reward Hacking 与分布偏移** | **优化过程里奖励与分布怎么被钻空子**：Goodhart 饱和、mode collapse、具体 hack 形态与检测、veto/信任域/rollout 过滤等防御、分布塌缩与 off-policy 失配 | 🔄 建设中 | [进入子专题 →](./reward-hacking/) |

---

## 论文篇

| 篇目 | 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **Arena-RL：Preference × Rubric 复合奖励后训练前沿 T2I 基模** | **算法不变、只动奖励侧**。核心命题：高质量 RL = Reward Coverage + Reward Composition + Reward Robustness。四类奖励（Preference RM / presence-oriented Faithfulness / absence-oriented Intent-gated Constraint / Anti-Hack）+ 四个组合机制（Reward-decoupled Normalization / Prompt-conditioned Gating / One-sided Veto / Weight Ensembling）。FLUX.2-dev +69 Elo、Ideogram-4 达 1223.5。**已完结 Part 1–10**：① 三问题动机与 Related Work 三分类定位；② Arena 真实用户 Prompt → Gemini-3.5-Flash 六条 VALID 规则二值过滤 → 10K/1K，100K 未过滤低 **0.0360**；③ BT Preference RM（Qwen3.6-27B + `<\|cls\|>` Linear Head，5M 偏好对），Score 为无界 Logit 且 BT Loss 只依赖差值；④ Faithfulness Checklist = **182,230 问 / 18.2 每 Prompt**，Gemini-3-Pro 离线生成**依赖 DAG**（F4←{F0,F3} 多父节点），父失败子自动 No，$R_{\mathrm{faith}}$ = Yes 比例；⑤ Intent 三分类 strict/neutral/open，**26,213 约束问题 / 3,140 启用（31.4%）**，人工审计三分类 77.8%、strict 二分类 89.5%、κ=0.66；⑥ Anti-Hack = **205 个失败案例审计出的两种固定 detector**（Text Gibberish pointwise / Photo-style Drift reference-based 双序判）+ Applicability Gate + **One-sided Veto**；⑦ Ideogram-4 的 Graded OCR $=0.5r_e+0.3r_p+0.2r_r$、无文字/转录失败一律 Mask；⑧ 公式 (3)–(9) 全量演算：分轴 z-score → Gate → Veto → $R_i$ → $A_i=(R_i-\mu_{group})/\sigma_{batch}$ → $\tilde r_i=\frac12\mathrm{clip}(A_i/5,-1,1)+\frac12$，含 6 图完整数值演算与 Group Std vs Batch Std 深挖；⑨ Stage 1/2/3（**Stage 2 单跑不超 Stage 1**，收益来自融合）；⑩ LoRA 权重融合与 Rank Concatenation（r64×2→r128 精确平均） | 🟢 Part 1–10 完结 | [阅读笔记 →](./arena-preference-rubric-reward.md) |

---

## 本域待补主线（阅读路线图）

1. **跨奖励轴相关性分析** —— 论文缺 Faithfulness / Constraint 重复计分（Figure 4 中 C0、C1 与 F1、F2 重叠）与 $w_a=1$ 敏感性的消融
2. **$A_{\max}=5$ 的敏感性** —— 与 Sol-RL 的 `24-in-96` 探索策略直接冲突，值得在现有 NFT 训练中对照
3. **Batch Std 的分布式口径** —— 论文未公开 `batch` 覆盖哪些 GPU、是否全局 All-Gather
4. **Rubric 自动生产谱系** —— 与 `Arena-T2I-Hard`、`Auto-Rubric`、`Autorubric-T2I` 的关系尚未成篇

---

## 关联笔记

- [图像基模 RL 后训练专题（2026）](../vision/image-rl-posttraining/) — 算法篇 / 基模 RL 篇总索引
- [训练 Playbook v1.0](../vision/training-playbook.md) — 「多 reward 打架 → Normalization/Routing」诊断表
- [Sol-RL：FP4 Explore, BF16 Train](../vision/image-rl-posttraining/sol-rl.md) — 扩散 RL 的 rollout 预算解耦
- [GDRO 组级直接奖励优化](../vision/image-rl-posttraining/gdro-group-reward-2026.md) — 本文的 GDPO 式解耦归一化出处
- [Flow-GRPO 综述](../vision/image-rl-posttraining/flow-grpo-survey-2026.md) — OCR / GenEval 等可验证奖励的算法侧接入