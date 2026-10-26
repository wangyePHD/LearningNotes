# 🚨 RL Reward Hacking 与分布偏移

> **标签**：`Reward Hacking` `Distribution Shift` `Mode Collapse` `One-sided Veto` `Rollout Filtering` `Trust Region`

RL-AIGC 专区的第三个子专题。本专题研究一个具体问题：**模型在长期优化中如何「骗」奖励，以及训练分布如何在压力下漂移。**

---

## 定位：与另两个子专题的分工

```
/domains/rl-aigc/                    ← 奖励体系怎么组合（多奖励 · gating · 归一化）
/domains/rl-aigc/preference-model/    ← 打分器本身怎么造（结构 · 蒸馏 · 评测）
/domains/rl-aigc/reward-hacking/      ← 🔴 本专题：优化过程中奖励与分布如何被钻空子
```

::: warning 判别边界
| 问题 | 归属 |
| :--- | :--- |
| 「**奖励怎么定义、怎么组合**」 | `rl-aigc/` |
| 「**打分器本身怎么训**」 | `preference-model/` |
| 「**模型怎么利用奖励的漏洞、训练分布怎么漂**」 | **本专题** |
:::

本专题关心的四个典型症状：

| 症状 | 表现 |
| :--- | :--- |
| **Goodhart / 奖励饱和** | reward 持续上涨，真实质量先饱和后崩塌 |
| **Mode Collapse** | 为了高奖励退化成单一模式（图像越来越像、动作策略越来越单一） |
| **Reward Hacking 具体形态** | 加乱码文字 · 风格漂移 · 过度写实 · 刷 OCR 分数 · 只改未编辑区 |
| **分布偏移** | rollout 分布相对 reward 分布塌缩 · online/off-policy 失配 · EMA 累积漂移 |

---

## 收录标准

| 判据 | 说明 |
| :--- | :--- |
| ✅ 收录 | **攻击面**：论文里给出了具体的 reward hacking 形态、检测器或度量 |
| ✅ 收录 | **防御面**：惩罚项 / veto / 正则 / 信任域 / rollout 过滤等抑制手段及其消融 |
| ✅ 收录 | **分布偏移**：rollout 分布塌缩、模式崩塌、off-policy 偏移、训练-评测不一致 |
| ✅ 收录 | **审计方法**：如何证明「reward 涨了但质量没涨」 |
| ❌ 不收 | 只是「奖励定义得不好」但没讨论被利用的风险（归 `rl-aigc/`） |
| ❌ 不收 | 只讨论打分器训练技巧、与 hacking 无关（归 `preference-model/`） |
| ❌ 不收 | 纯安全性对抗攻击（越狱 / 有害内容），除非明确针对 **reward** 机制 |

---

## 论文篇

<NoteList folder="domains/rl-aigc/reward-hacking" />

---

## 已有的分散内容（尚未独立成篇）

本专题的核心证据目前散落在已完结笔记里，先在这里建索引，后续再决定是否抽出成篇：

| 内容 | 位置 | 关键结论 |
| :--- | :--- | :--- |
| **Anti-Hack 闭环** | [Arena-RL Part 6 §44–§52](../arena-preference-rubric-reward.md) | 205 个失败案例审计出两种固定 detector；**加成型惩罚从 .635 崩到 .526**，单边否决才是对的；「防 hack 的 reward 会变成新的 hack 面」 |
| **组合公式与两级归一化** | [Arena-RL Part 8 §64–§75](../arena-preference-rubric-reward.md) | 分轴 z-score → Gate → Veto → 组内 $A_i$ → $\tilde r$；$A_{\max}=5$ 的截断信任域 |
| **Goodhart 实证** | [Arena-RL Part 1 §2.3](../arena-preference-rubric-reward.md) · [Part 8 §39](../arena-preference-rubric-reward.md) | 单一偏好 RM 训练让 faithfulness 胜率跌破 50% 并持续下滑（Fig. 5） |
| **Reward 饱和 + 分数上限** | [RationalRewards Part 5 §28、§30](../preference-model/rational-rewards.md) | 生成式评分非确定性；解析失败给 0.0 有被误当低质量图的风险 |
| **训练曲线里的退化** | [RationalRewards Part 6 §36](../preference-model/rational-rewards.md) · `rational-fig11-training-curves.png` | 最上行 eval reward 在 step≈200 明显下跌后回升，对应训练中期退化 |

---

## 待补主线 / 候选论文池

来自 2026-10-13 的四维扫描，均直接命中本专题：

| 论文 | 命中的症状 | 一句话 |
| :--- | :--- | :--- |
| **VideoAWR** `2610.13338` | <mark class="hl-key">**EMA 累积漂移**</mark> | 把采样用的引导场当优化锚点，EMA 更新会让引导强度几何级放大、300 GPU·h 内崩掉 |
| **Uni-TMPO** `2609.34688` | **Mode Collapse** | 「越训越像同一张图」是追最高分本身造成的；改成学一个被分数重新加权的分布 |
| **CAST** `2609.39441` | **Reward 饱和** | 现有基于人工标注的打分模型在最新 SOTA 上分数极高且几乎无法区分，优势估计失效 |
| **MATE** `2609.36224` | **自博弈偏移** | 让理解与生成分支互相挑战，挑战样本随模型演化 —— 用模型自身输出当对手 |
| **NowcastDiT** `2609.37038` | **分布/任务偏移** | 标准 DiT 也能做降水临近预报，端到端 RL + 感知质量与气象技巧双目标 |

其他方向（本专题尚未覆盖）：

- Reward Hacking 的**量化度量**（如何给「hack 程度」定标，而不只有定性案例）
- Rollout 过滤的**副作用**（我们已记下：低方差过滤可能把困难任务永久排除）
- **Off-policy / 分布外 rollout** 在扩散 RL 中的处理（VideoAWR 附录 C 提到 anchor 无关性允许任意行为分布，作者把 off-policy 数据整合列为 future work）

---

## 关联

- 🎲 [AIGC 强化学习后训练](../index.md) — 本层：奖励体系怎么组合
- 🏅 [偏好模型 (Preference Model)](../preference-model/) — 打分器本身怎么造
- 📥 [每日 arXiv 速览](/daily-arxiv/) — 本专题的候选池主要来源