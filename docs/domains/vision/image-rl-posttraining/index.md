# 🖼️ 图像基模 RL 后训练 (2026)

> **标签**：`Vision` `RL` `GRPO` `DPO` `DiffusionNFT`
> **更新时间**：2026-09-23

2026 年开源文生图 / 图像编辑模型中与 RL / 偏好对齐强相关的精读笔记。范围：2026 年（含 25.11 跨年）报告。

## 算法篇

| 篇目 | 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **Flow-GRPO 综述** | 稀疏奖励稠密化 / 树状信用分配 / 离线化三主线 | 🟢 已完结 | [阅读笔记 →](./flow-grpo-survey-2026.md) |
| **GDRO 组级直接奖励优化** | 全离线组级优化 + corrected score 抗 hacking | 🟢 已完结 | [阅读笔记 →](./gdro-group-reward-2026.md) |

## 模型 RL 篇

| 篇目 | 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **FireRed-Image-Edit RL** | 非对称 DPO → DiffusionNFT + Layout-OCR 奖励 | 🟢 已完结 | [阅读笔记 →](./firered-image-edit-rl.md) |
| **Qwen-Image-2.0 RL** | 五奖励 GRPO + CFG 杂交 + PE 的 GRPO | 🟢 已完结 | [阅读笔记 →](./qwen-image-2-rl.md) |
| **LLaDA-Image（无 RLHF 参照）** | TwinFlow 蒸馏替代对齐 | 🟢 已完结 | [阅读笔记 →](./llada-image-recipe.md) |
| **Swift-Image RL** | 并行专家 NFT → 多教师在线蒸馏 | 🟢 已完结 | [阅读笔记 →](./swift-image-rl.md) |
| **ERNIE-Image DPO** | Flow Matching DPO + 双 anchor + MT-DMD | 🟢 已完结 | [阅读笔记 →](./ernie-image-dpo.md) |
| **SeFi-Image RL** | 在线 NFT + 双潜适配 + 能力标签；§4–7 已补全损失/Δt 调度/双流代码/评测 | 🟢 已完结 | [阅读笔记 →](./sefi-image-rl.md) |
| **i1（无 RL 参照）** | 建模 + 数据 recipe 对照组 | 🟢 已完结 | 见对比总结 |
| **算法 × 奖励 × 基模对比与演进** | 六条主线 + 工程 checklist | 🟢 已完结 | [阅读笔记 →](./rl-comparison-2026.md) |
