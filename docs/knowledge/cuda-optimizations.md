# CUDA 算子优化与内存瓶颈分析

> **标签**：`CUDA` `GPU Architecture` `Optimization`  
> **更新时间**：2026-09-19

---

## 1. Roofline 模型与瓶颈判定

算子优化的核心第一步是判断算子是处于 **Memory-Bound（访存受限）** 还是 **Compute-Bound（计算受限）**：

$$
\text{Arithmetic Intensity (算术强度)} = \frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

- 若 $\text{Intensity} < I_{\text{knee}}$：属于访存密集型（如 Elementwise Add、Softmax、LayerNorm），优化重点在于合并访存、利用共享内存（Shared Memory）减少全局内存（HBM）访问。
- 若 $\text{Intensity} \ge I_{\text{knee}}$：属于计算密集型（如 GEMM 矩阵乘法），优化重点在于 Tensor Core 利用率与指令流水编排。

---

## 2. 经典优化手段清单

1. **Memory Coalescing（合并访存）**：确保 Warp 内 32 个线程访问连续对齐的显存地址。
2. **Shared Memory Tiling（共享内存分块）**：避免重复加载数据，降低 HBM 读写带宽压力。
3. **Kernel Fusion（算子融合）**：将多个连续小算子（例如 BiasAdd + ReLU）融合成单个 Kernel，减少显存换入换出。
