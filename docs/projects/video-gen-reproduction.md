# 开源视频生成模型本地复现与评测报告

> **标签**：`Video Gen` `Wan2.1` `Benchmark` `FSDP`  
> **更新时间**：2026-09-19

---

## 1. 复现目标与模型规模

本次实验针对 14B 参数级别开源视频生成大模型进行分布式多卡推理与微调复现，评估在不同时序切分策略下的显存占用与显卡通讯消耗。

## 2. 评测硬件环境

- **GPU 集群**：8x NVIDIA H800 (80GB SXM5), NVLink 4.0 400GB/s
- **CPU / 内存**：Intel Xeon Platinum 8468, 1TB DDR5
- **框架版本**：PyTorch 2.4.0, CUDA 12.4, FlashAttention-3

## 3. 推理速度与显存对比

| 优化方案 | 分辨率 & 帧数 | 单视频端到端生成耗时 | 峰值显存 (Per GPU) | 视频流畅度评测 (VBench) |
| :--- | :--- | :--- | :--- | :--- |
| **原生 Baseline (8 卡 TP)** | 720P, 81 帧 (5s) | 68.2 秒 | 74.2 GB | 81.2 |
| **序列并行 (SP) + 算子重编译** | 720P, 81 帧 (5s) | **44.1 秒** (-35.3%) | **48.6 GB** (-34.5%) | 81.3 |

::: tip 关键发现
在视频模型中，由于序列长度长（$T \times H \times W$），采用序列并行（Sequence Parallelism, 如 DeepSpeed Ulysses / Ring Attention）相较于传统的 Tensor Parallelism（TP），能显著减少跨卡 All-Reduce 传输量，使得多卡扩展效率达到 92% 以上。
:::
