# 🪞 OPSD 子专题：On-Policy 自蒸馏

> **状态**：📋 目录已建，待补充篇目。
>
> **关注问题**：在没有“更强 teacher”可用时，能否让**同一个模型**在另一种上下文 / 另一种视角下当 teacher，仍然保住 on-policy 蒸馏的收益？此时监督信号来自自身，如何避免自我强化偏差（stop-grad、温度、只监督 masked 位置、上下文差异的设计）。

---

## 本子专题要回答的四个问题

1. **teacher 视角从哪来**：同一权重的不同模态分支（如 AR view vs Diffusion view）、不同上下文（有提示 / 无提示）、不同时间步的自身快照？
2. **目标要不要 stop-grad**：自蒸馏时梯度若双向流动，等价于在同一个函数上做一致性约束，可能坍缩到平凡解；哪些工作是单向、哪些是双向？
3. **监督位置**：只监督 masked 位置 / 只监督 student 采样到的 token / 全序列？
4. **与外部 teacher 的关系**：自蒸馏是替代外部 teacher，还是像跨范式 OPD 那样作为“外部 teacher → 自身 view → 目标 view”的中转桥？

---

## 已收录线索

| 篇目 | 自蒸馏形态 | 归属 |
| :--- | :--- | :--- |
| [NLD-Uni](../opd/nld-uni.md) | **AR view（stop-grad）→ Diffusion view** 的 consistency loss，作为 AR teacher → student 的中转桥 | 主体归 OPD 子专题，其 §4.2.2 是本子专题的第一个实例 |

---

## 论文篇

<NoteList folder="domains/distillation/opsd" />
