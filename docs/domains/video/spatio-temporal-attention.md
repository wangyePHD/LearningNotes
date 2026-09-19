# 时空注意力机制 (Spatio-Temporal Attention) 数学推导

> **标签**：`Video Model` `Attention` `Spatio-Temporal` `Complexity`  
> **更新时间**：2026-09-19

---

## 1. 为什么全时空联合注意力 (Joint Attention) 会爆炸？

给定视频输入张量 $X \in \mathbb{R}^{T \times H \times W \times C}$，其中：
- $T$: 视频帧数 (Frames)
- $H \times W$: 每帧 Patch 空间大小
- $C$: 特征通道数 (Embedding Dimension)

若直接将所有时空 Token 展开为序列长度 $N = T \times H \times W$：
在自注意力机制中，计算复杂度为 $\mathcal{O}(N^2)$：

$$
\text{Complexity}_{\text{Joint}} = \mathcal{O}\left((T \cdot H \cdot W)^2 \cdot C\right)
$$

::: danger 显存与计算灾难
假设视频只有 16 帧，每帧 $32 \times 32$ 个 Patch：
$$
N = 16 \times 32 \times 32 = 16,384
$$
$$
N^2 \approx 2.68 \times 10^8 \quad (\text{单个注意力矩阵即需近 3 亿次浮点运算})
$$
对于长视频（如 128 帧），显存直接 OOM。
:::

---

## 2. 空间-时间因子分解注意力 (Factorized Attention)

为了降低复杂度，现代视频模型采用**分步解耦机制**（如 TimeSformer / Video DiT）：

```text
输入张量: [T, S, C] (S = H * W)
       │
       ▼
 ┌─────────────┐
 │ 空间注意力   │  在每帧内计算 S 个 Token 的注意力，保持 T 独立
 │ (Spatial)   │  复杂度: O(T * S^2)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │ 时间注意力   │  在同一空间位置沿 T 轴计算注意力，保持 S 独立
 │ (Temporal)  │  复杂度: O(S * T^2)
 └─────────────┘
```

### 总复杂度对比

$$
\text{Complexity}_{\text{Factorized}} = \mathcal{O}\left( T \cdot S^2 + S \cdot T^2 \right) = \mathcal{O}(T \cdot S (S + T))
$$

相较于全时空联合注意力的 $\mathcal{O}(T^2 \cdot S^2)$，复杂度直接从**二次方乘积降至线性乘积**，使得大分辨率长视频的训练成为可能。
