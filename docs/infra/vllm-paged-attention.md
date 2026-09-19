# vLLM PagedAttention 与高并发吞吐优化

> **标签**：`LLM Infra` `vLLM` `PagedAttention` `KV Cache`  
> **更新时间**：2026-09-19

---

## 1. 传统 KV Cache 显存碎片痛点

在传统的自回归大模型推理服务中，必须为每个请求提前静态分配对应 `max_seq_len` 的连续显存空间。  
这导致了三大显存浪费：
1. **内部碎片 (Internal Fragmentation)**：用户请求可能只生成了 100 个 token，但按 4096 预分配。
2. **预留显存 (Reservation)**：为未来可能生成的 token 预留。
3. **共享瓶颈 (Sharing)**：在并行采样（Parallel Sampling）或束搜索时，相同的 Prompt 无法共享显存。

碎片率往往高达 **60% ~ 80%**，严重制约了并发并发度（Throughput）。

---

## 2. PagedAttention 核心架构

vLLM 借鉴了操作系统 **虚拟内存分页 (Virtual Memory Paging)** 的思想：

```text
逻辑块 (Logical Blocks)       块表映射 (Block Table)      物理块 (Physical Blocks in HBM)
[ Token 0 ~ 15 ]    ───────>  Logical #0 -> Physical #7    Physical #7: [非连续显存块 A]
[ Token 16 ~ 31 ]   ───────>  Logical #1 -> Physical #3    Physical #3: [非连续显存块 B]
[ Token 32 ~ 47 ]   ───────>  Logical #2 -> Physical #11   Physical #11: [非连续显存块 C]
```

- 将 KV Cache 拆解为固定大小的块（例如每个 Block 存放 16 个 Token 的 Key 和 Value）。
- 物理内存不要求连续，通过查表将连续的注意力计算分散到非连续的显存块上。
- **效果**：将显存碎片率降至 **4% 以下**，在相同显存下并发量提升 **2~4 倍**。
