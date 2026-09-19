# CUDA 显存层级与 Roofline 性能模型

> **标签**：`CUDA` `Infra` `Roofline` `Memory Bound`  
> **更新时间**：2026-09-19

---

## 1. GPU 显存金字塔带宽与延迟

在现代 GPU（如 NVIDIA Hopper / Blackwell 架构）中，显存层级差异高达数十倍：

| 存储层级 | 典型容量 (H100) | 理论访问带宽 | 访问延迟 (Cycles) |
| :--- | :--- | :--- | :--- |
| **Registers (寄存器)** | ~256 KB / SM | ~30 TB/s | ~1 cycle |
| **Shared Memory / L1 Cache** | 228 KB / SM | ~15 TB/s | ~20-30 cycles |
| **L2 Cache (片上缓存)** | 50 MB (全局共享) | ~12 TB/s | ~200 cycles |
| **HBM3 (显存/全局内存)** | 80 GB | ~3.35 TB/s | ~400-800 cycles |

---

## 2. Roofline 模型判定公式

$$
P = \min \left( P_{\text{peak}}, \, I \times B W \right)
$$

其中：
- $P$: 实际计算性能 ($\text{TFLOPs}$)
- $P_{\text{peak}}$: 硬件峰值浮点算力
- $I$: 算术强度 ($\text{FLOPs/Byte}$)
- $BW$: 显存访问带宽 ($\text{TB/s}$)

::: tip 优化指导原则
- **如果 $I < \frac{P_{\text{peak}}}{BW}$ (Memory Bound)**：算子受限于访存带宽（如 LayerNorm, Softmax）。优化手段：Triton 算子融合（Fusion），将中间结果保存在 SRAM 中，避免来回读写 HBM。
- **如果 $I \ge \frac{P_{\text{peak}}}{BW}$ (Compute Bound)**：算子受限于计算核心（如大矩阵乘 GEMM）。优化手段：提升 Tensor Core 占空比，利用异步复制（Async Copy）隐藏流水线延迟。
:::
