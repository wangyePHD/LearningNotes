# 🔁 OPD 子专题：跨范式 On-Policy 蒸馏

> **关注问题**：当 teacher 与 student 的**生成范式、预测空间、条件上下文**三者都不一致时（AR-VLM → 离散扩散、连续流 → 离散扩散），on-policy 蒸馏里的“对齐”到底该对齐什么？哪些量可以直接拷，哪些必须换一种形式重造？

---

## 本子专题的核心矛盾

On-Policy Distillation 的初衷是消除 **训练分布与推理分布的偏移**：不再用 teacher 独立生成的样本当训练集，而是在 student 自己跑出来的轨迹上取 teacher 的监督。

但跨范式蒸馏会引入第二个偏移：

| 偏移 | Off-Policy 蒸馏 | On-Policy 蒸馏（同范式） | 跨范式 OPD |
| :--- | :--- | :--- | :--- |
| 训练轨迹来自谁 | Teacher | Student | Student |
| Teacher 的预测上下文与 student 是否一致 | 一致（都看 teacher 的轨迹） | 一致（都看 student 的轨迹） | **不一致**（AR 前缀 vs 双向 Mask 上下文；连续 latent vs 离散 code） |
| 分布级对齐是否可能 | 是 | 是 | **不一定**（空间都不同，KL 无从定义） |
| 应对方式 | — | 直接用 KL | ① 先跨范式搭桥（transitive）；② 或在 teacher 空间里造 refined 目标 |

所以本子专题里的论文，实质都在回答：**跨范式时，用什么中间量把“teacher 想要什么”翻译成“student 能吃的监督”。**

---

## 收录标准

- ✅ teacher 与 student 的范式 / 表征空间不同，且论文给出了跨范式的转换机制
- ✅ 必须包含 on-policy 采样环节（student roll out），并说明 rollout 的采样设置（温度、block size、NFE）
- ✅ 必须包含与 naive 直接拷贝 teacher 分数的对照实验
- ❌ 同范式、同空间的常规蒸馏（归入各领域笔记）

---

## 论文篇

论文笔记已统一归入后训练下的 [🔁 OPD 子专题](/domains/rl-aigc/opd/)，与 DPO 子专题同级。

---

## 相关子专题

- [🪞 OPSD：On-Policy 自蒸馏](../opsd/) —— 跨范式 OPD 常用“student 自身另一 view 当 teacher”作桥，这一做法本身就是 OPSD
