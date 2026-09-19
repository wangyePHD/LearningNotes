# ⚡ 算力基建与系统工程 (Infra & Systems) 细粒度编写规范

当用户让 Agent 记录 CUDA 算子优化、显存层级分析、Roofline 模型、分布式训练（Megatron/FSDP/Deepspeed/SP）、推理吞吐（vLLM/TensorRT-LLM）等底层工程知识时，必须遵循本规范。

---

## 1. 结构大纲标准 (必须包含以下 6 个小节)

```markdown
# [工程主题中文名] ([英文技术名称])

> **标签**：`Infra` `CUDA` / `Roofline` / `vLLM` `[其他标签]`  
> **更新时间**：YYYY-MM-DD  
> **目标硬件**：[如：NVIDIA A100 / H800 / B200 / RTX 4090]

---

## 1. 业务场景与系统性能瓶颈
- 明确指出该优化所处的实际场景（如超长视频 128 帧推理、大模型高并发 Serving、千卡分布式集群）；
- 阐明核心瓶颈到底是：访存带宽（Memory-Bound）、计算核心（Compute-Bound）还是跨节点网络通讯（Comm-Bound）。

## 2. 硬件层级指标与 Roofline 算术强度定量计算
- 明确目标硬件的关键物理上限（FP16/BF16 TFLOPs、HBM 带宽、SRAM 容量）；
- 严格推导并计算算子的**算术强度（Arithmetic Intensity）**：
  $$ I = \frac{\text{FLOPs}}{\text{Bytes Transferred}} $$
- 判定当前算子是否越过硬件拐点（Knee Point），给出理论性能上限。

## 3. 显存架构与访存编排解析
- 剖析 Registers、Shared Memory (SRAM)、L2 Cache 到 HBM 的数据搬运轨迹；
- 分析合并访存（Memory Coalescing）、避免 Bank Conflict、或解决碎片化（如 Paged 分页）的底层机理。

## 4. 优化技术方案与并行策略决策
- 阐述具体的优化路线（如 Kernel Fusion、Triton 算子重写、流水线隐藏异步拷贝 Async Copy、序列并行 SP vs 张量并行 TP）；
- 给出并行选型的决策树依据。

## 5. 核心高性能实现代码
- 提供 20~40 行 Triton JIT 算子核心或高性能 PyTorch/CUDA 关键逻辑，包含显存加载与 Tile 分块注释。

## 6. 实测吞吐与显存评测对比
- 必须包含实测对照表格：
  | 方案 | 显存峰值 (VRAM) | 吞吐量 (Tokens/s 或 FPS) | 加速比 (Speedup) |
- 用 `::: tip 调优总结` 给出参数调优经验（如 Block/Thread 配置、动态批处理策略）。
```
