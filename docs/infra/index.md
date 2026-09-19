# ⚡ 算力与系统工程 (Infra & Systems)

算法工程师从模型到生产的核心瓶颈往往在算力与显存。本模块记录 CUDA 算子极致优化、显存层级、分布式通信（NCCL/Megatron）与高性能推理引擎（vLLM / TensorRT-LLM）。

## 核心篇目

| 篇目 | 主题 / 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **CUDA 显存层级与 Roofline 模型** | HBM, SRAM, Warp 合并访存与瓶颈判定 | 🟢 已完结 | [阅读笔记 →](./cuda-memory-hierarchy.md) |
| **vLLM PagedAttention** | 类似虚拟内存分页管理的 KV Cache 极致复用 | 🟢 已完结 | [阅读笔记 →](./vllm-paged-attention.md) |
