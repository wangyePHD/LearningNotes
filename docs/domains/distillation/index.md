# 🔁 蒸馏专题：OPD / OPSD

> **定位**：本专题只研究一类训练信号——**在 student 自己 roll out 出来的轨迹上，用 teacher 的分数做监督**。按 teacher 的来源分成两条主线：OPD（On-Policy Distillation，teacher 是独立的外部强模型）与 OPSD（On-Policy Self-Distillation，teacher 与 student 同源，靠不同上下文 / 不同视角自我监督）。

---

## 1. 两条主线

| | **OPD（On-Policy Distillation）** | **OPSD（On-Policy Self-Distillation）** |
| :--- | :--- | :--- |
| Teacher 来源 | 独立的外部强模型（可以是异构范式） | 同一个模型 / 同一份权重 |
| 训练轨迹 | Student 自己生成（on-policy） | Student 自己生成（on-policy） |
| Teacher 提供什么 | 逐 token 分布（连续空间下是 refined 样本） | 同一轨迹在另一上下文 / 另一视角下的分数 |
| 主要难点 | 训练-推理分布一致了，但 teacher 与 student 的**上下文、预测空间、范式**仍不一致 | 没有外部信号，只靠上下文差异制造“teacher 视角”，容易自我强化偏差 |
| 典型形态 | AR teacher → DLLM student；连续流 teacher → 离散扩散 student；多 teacher 能力合并 | AR view ↔ Diffusion view 互教；带提示 / 无提示双上下文 |

两条主线共同的机制是：**先把 off-policy 的分布偏移问题换成 on-policy 的上下文错配问题**，再想办法处理上下文错配。

---

## 2. 收录标准

| 判据 | 说明 |
| :--- | :--- |
| ✅ 收录 | 明确提出 on-policy 采样 + teacher 监督的蒸馏目标（含其 loss 推导与消融） |
| ✅ 收录 | 跨范式蒸馏：teacher 与 student 的生成范式 / 表征空间不同（AR ↔ 扩散 ↔ 连续流） |
| ✅ 收录 | 自蒸馏：同一模型内部两个 view / 两个上下文互为 teacher，含 stop-grad、双向 KL 等细节 |
| ✅ 收录 | 把“on-policy 轨迹 + teacher 修正目标”退化成 SFT 的做法（teacher-refined target），但要说明它为何仍算 on-policy |
| ❌ 不收 | 纯 off-policy 蒸馏（固定 teacher 输出当训练集）且不含 on-policy 部分 |
| ❌ 不收 | 只做量化 / 剪枝 / 结构蒸馏，不涉及分布级监督与轨迹采样 |

---

## 3. 子专题

| 子专题 | 关注问题 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **🔁 OPD：跨范式 On-Policy 蒸馏** | 外部强 teacher 与 student 范式不同（AR-VLM / 连续流 → 离散扩散）时，怎么把分布对齐做对：上下文错配、teacher 分数能不能直接拷、refined target 怎么造 | 🔄 建设中 | [进入子专题 →](./opd/) |
| **🪞 OPSD：On-Policy 自蒸馏** | 没有更强 teacher 时，用同一模型的不同 view / 不同上下文互教：stop-grad 目标、双向 KL、上下文差异如何制造有效监督 | 📋 目录已建 | [进入子专题 →](./opsd/) |

---

## 4. 论文篇

<NoteList folder="domains/distillation" />

---

## 5. 待补主线（阅读路线图）

以下篇目来自本专题第一篇笔记（NLD-Uni）的相关工作章节，作为后续补齐的锚点：

1. **OPD 原始工作** —— on-policy 蒸馏在语言模型上的提出与收敛性质（NLD-Uni §2.3 引 [29, 30]）
2. **M-OPD** —— 多 teacher 设定下合并领域专家能力（[31]）
3. **Diffusion-OPD** —— 把 OPD 搬到连续流模型的视觉生成任务（[16]）
4. **OPD-LM** —— AR → DLLM 的同源转换式蒸馏，与本文的最大对照组（[34]，差异见 [NLD-Uni 笔记 §七](./opd/nld-uni.md)）
5. **多模态理解侧的 OPD** —— VLM 上用 on-policy 蒸馏提分的两组工作（[32, 33]）

---

## 关联笔记

- [图像基模 RL 后训练专题](../vision/image-rl-posttraining/) —— 同属“后训练信号”一侧，对照 RL 与蒸馏两种信号来源
- [NLD-Uni：统一理解与生成的离散扩散模型](./opd/nld-uni.md) —— 本专题第一篇，跨范式 OPD + 渐进自细化扩散
