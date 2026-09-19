# LLM 7B 模型 LoRA 微调与吞吐评测

> **标签**：`LLM` `LoRA` `Fine-tuning` `Benchmark`  
> **更新时间**：2026-09-19

---

## 1. 实验背景与目标

针对 7B 规模基座大模型进行特定垂域下游任务微调，对比 Full Fine-Tuning（全量参数微调）与 LoRA（低秩适配）在显存消耗、吞吐量（Tokens/sec）以及损失收敛方面的差异。

## 2. 硬件与超参配置

- **GPU**: 单卡 NVIDIA A100-SXM4-80GB
- **Sequence Length**: 2048
- **Batch Size per GPU**: 4 (Gradient Accumulation Steps = 4, Effective Batch Size = 16)
- **Optimizer**: AdamW (`lr = 2e-4`, `weight_decay = 0.01`)

## 3. 评测结果对照

| 微调策略 | 可训练参数比例 | 峰值显存 (VRAM) | 吞吐量 (Tokens/s) | 评测评估分 (Accuracy) |
| :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | 100% | 68.4 GB | 1,420 | 82.5% |
| **LoRA (r=16, alpha=32)** | 0.06% | **24.8 GB** (-63.7%) | **2,560** (+80.2%) | 81.9% (-0.6%) |

::: tip 实验结论
在保证最终任务评测分数仅微损 0.6% 的前提下，LoRA 将显存占用降低至 24.8 GB（单张 RTX 3090/4090 即可承载），且吞吐效率大幅提升。
:::
