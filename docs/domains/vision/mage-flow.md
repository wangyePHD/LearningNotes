# 微软原生分辨率基模 Mage-Flow (Mage-VAE + Native-Resolution MMDiT + Diffusion-NFT)

> **标签**：`Vision` `Diffusion` `MMDiT` `Flow Matching` `VAE` `RL` `DiffusionNFT` `Distillation` `Efficiency`
> **更新时间**：2026-10-03
> **参考来源**：[Mage-Flow: An Efficient Native-Resolution Foundation Model for Image Generation and Editing (arXiv:2607.19064v2)](https://arxiv.org/abs/2607.19064) · [Project Page](https://microsoft.github.io/Mage) · [GitHub](https://github.com/microsoft/mage) · [HuggingFace](https://huggingface.co/collections/microsoft/mage)
> **原文**：本地 `Papers/Mage-Flow.pdf`（59 页，微软 Mage Team，2026-07-22）
> **精读重点**：§4 Data → §5.1 Pre-train/SFT/Edit → §5.2 Diffusion-NFT → §5.3 Distillation → §3.2/§3.3 Native-Res + Infra → §3.1 Mage-VAE
> **精读进度**：目录已搭建，内容待逐节精读填充

---

<!--
精读顺序（按学习大纲，非论文原章节序）：
  1. §4   Data Collection & Curation   ★★★ 重点精读
  2. §5.1 Pre-training + SFT           ★★★ 重点精读
  3. §5.2 Diffusion-NFT                 ★★★ 最高优先级
  4. §5.3 Few-step Distillation         ★★☆ 重点学，与 Z-Image 对照
  5. §3.2/§3.3 Native-Res + Infra       ★★☆ 中等深度
  6. §3.1 Mage-VAE                      ★☆☆ 选择性学
主线：Data → Progressive Pretrain/SFT → Edit Training → Diffusion-NFT → 4-step Turbo
旁支：Native-Resolution Packing / Training Efficiency ／ Mage-VAE
-->

## 0. 精读导航

| # | 主题 | 论文位置 | 优先级 | 状态 |
| :-: | :--- | :--- | :-: | :--- |
| 1 | Data Collection & Curation | §4 (P16–19) | ★★★ | ⬜ |
| 2 | Pre-training + SFT Recipe | §5.1 (P20–21) | ★★★ | ⬜ |
| 3 | Diffusion-NFT Post-training | §5.2 (P21–24) | ★★★ | ⬜ |
| 4 | Few-step Distillation | §5.3 (P24–26) | ★★☆ | ⬜ |
| 5 | Native-Resolution + Infrastructure | §3.2 / §3.3 (P13–16) | ★★☆ | ⬜ |
| 6 | Mage-VAE | §3.1 (P10–13) | ★☆☆ | ⬜ |
| 7 | Ablation / Tricks 总结 | 全文 | ★★☆ | ⬜ |

## 1. Introduction

## 2. Data Collection and Curation ★

### 2.1 T2I 数据总览：10B raw → 1.3B curated

### 2.2 Sample-level filtering 与四阶段阈值（256 / 512 / 1024 / SFT）

### 2.3 跨数据集去重：SSCD + FAISS

### 2.4 Caption：Qwen3-VL 多粒度

### 2.5 Concept-aware synthesis

### 2.6 Balancing 与 concept reweighting

### 2.7 Editing 数据：90M raw triples → 45M

### 2.8 Expert voting 过滤（三个 Qwen3.5-9B）

### 2.9 19 类 edit taxonomy 与 balancing

### 2.10 数据环节小结与未公开细节

## 3. Pre-training and Supervised Fine-tuning ★

### 3.1 T2I Progressive Curriculum 总览

### 3.2 逐阶段递进：resolution / quality / threshold / reweighting

### 3.3 Editing 两阶段混合训练

### 3.4 为什么 Edit 训练要混 Generation，比例怎么配

### 3.5 本节小结与未公开细节

## 4. Diffusion-NFT Post-training ★

### 4.1 Diffusion-NFT 机制：为什么不是 trajectory likelihood

### 4.2 T2I RL：20K RL prompts 与三类路由

### 4.3 Capability mixture 两阶段：1:1:1 → 2:4:1

### 4.4 Reward 配置

### 4.5 Editing RL：30K edit prompts + RationalRewards + 4:1

### 4.6 与 DeepGen「RL 无 Edit」的对照

### 4.7 与已有 NFT 笔记的对照（FireRed / Swift-Image）

### 4.8 本节小结与未公开细节

## 5. Few-step Distillation ★

### 5.1 Decoupled DMD（与 Z-Image 对照）

### 5.2 Adversarial Perceptual Guidance：DINOv2 / CLIP + discriminator

### 5.3 为什么 4-step 还需要 perceptual adversarial regularization

### 5.4 蒸馏数据：Generation 200K ／ Editing 250K + 3:1

### 5.5 Ablation：哪些提升、哪些没有

### 5.6 本节小结与未公开细节

## 6. Native-Resolution MMDiT + Training Infrastructure ★

### 6.1 为什么不用传统 resolution bucket

### 6.2 Native Packing：variable-length image + text token 混 pack

### 6.3 FlashAttention variable-length kernel + per-sample 2D RoPE

### 6.4 CFG cond/uncond 单次 packed forward

### 6.5 Fused CUDA Kernels（Mage-VAE / Qwen3-VL / MMDiT）

### 6.6 MFU 13.88% → 29.28% 与 2.48× step-time speedup

### 6.7 本节小结与未公开细节

## 7. Mage-VAE ★

### 7.1 为什么 VAE 是高分辨率训练 / 4-step 推理的瓶颈

### 7.2 One-step diffusion encoder / decoder

### 7.3 Anchor-latent KL regularization

### 7.4 三阶段 VAE training 与 latent space 对齐 FLUX.2-VAE

### 7.5 Encode / Decode MACs 与 1K / 2K / 4K 优势

### 7.6 本节小结与未公开细节

## 8. Ablation 与 Tricks 有效性总结

### 8.1 Generation 数据混进 Edit 是否有效

### 8.2 Adversarial guidance 是否有效

### 8.3 不同 RL capability mixture 是否合理

### 8.4 哪些 trick 真 work

## 9. Experiments

### 9.1 Experimental Setup

### 9.2 Quantitative Results

### 9.3 Qualitative Results

## 10. 讨论 / 开放问题

## 附录

### A. 论文章节 → 本笔记章节对照

### B. 三个变体（Base / RL-aligned / Turbo）
