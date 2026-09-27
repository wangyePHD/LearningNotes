# 轻量统一多模态模型 DeepGen 1.0 (SCB + 三阶段训练 + MR-GRPO)

> **标签**：`Vision` `Unified Model` `VLM-DiT` `Flow Matching` `RL` `GRPO` `Data-centric`
> **更新时间**：2026-09-27
> **参考来源**：[DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing (arXiv:2602.12205v2)](https://arxiv.org/abs/2602.12205) · [GitHub: DeepGenTeam/DeepGen](https://github.com/DeepGenTeam/DeepGen) · [HuggingFace: DeepGenT](https://huggingface.co/DeepGenT) · [Datasets](https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0)
> **原文**：本地 `Papers/DeepGen.pdf`（21 页，正文 18 页 + 附录 A/B）
> **精读重点**：§3 Training（data train）+ §3.3 RL + §4 Data
> **精读进度**：待开始（笔记随学习逐节增补）

---

<!-- 精读导航：按论文章节序推进，重点节已标注 ★ -->

## 1. Introduction

> 尚未记录。核心待答问题：5B 如何在 WISE / DPG-Bench / UniREditBench 上对抗 27B–80B？

## 2. Model Architecture

> 尚未记录。SCB（Stacked Channel Bridging）：3B VLM + 2B DiT = 5B，多层 hidden states + learnable think tokens。

## 3. Training ★

> 尚未记录。本节为重点。

### 3.1 Stage 1: Alignment Pre-Training

> 尚未记录。只训 SCB connector + think tokens，冻结 VLM 与 DiT。

### 3.2 Stage 2: Joint Supervised Fine-Tuning

> 尚未记录。解冻 DiT + VLM 上 LoRA（rank 64 / α 128）。

### 3.3 Stage 3: Reinforcement Learning ★

> 尚未记录。本节为重点。MR-GRPO：混合奖励函数 + 噪声保持随机采样。

## 4. Data ★

> 尚未记录。本节为重点。预训练 35M 生成 + 6.6M 编辑；SFT 11M / 6.6M / 150K / 100K / 560K。

## 5. Experiments

> 尚未记录。

### 5.1 Evaluation Setup

> 尚未记录。

### 5.2 Model Performance

> 尚未记录。

### 5.3 Ablation Study

> 尚未记录。

#### 5.3.1 Architecture Design

> 尚未记录。

#### 5.3.2 RL Settings

> 尚未记录。Table 7：各变体统一训练 1,000 steps 后评测。

## 6. Conclusion

> 尚未记录。

## 附录

### A. Pre-Training & SFT Details

> 尚未记录。Table 8 数据明细、Table 9 超参（lr / batch / LoRA / 分辨率）。

### B. Reinforcement Learning Details

> 尚未记录。噪声保持随机采样（式 6）、奖励函数组合、辅助 SFT 数据。

## 7. 讨论 / 开放问题

> 待学习过程中沉淀。
