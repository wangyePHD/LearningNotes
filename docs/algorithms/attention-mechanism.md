# Transformer 缩放点积注意力数学推导

> **标签**：`Deep Learning` `Attention` `Transformer` `Math Derivation`  
> **更新时间**：2026-09-19

---

## 1. 核心公式与定义

在标准 Transformer 架构中，Scaled Dot-Product Attention（缩放点积注意力）的定义如下：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中各矩阵维度定义为：
- 查询矩阵：$Q \in \mathbb{R}^{n \times d_k}$
- 键矩阵：$K \in \mathbb{R}^{m \times d_k}$
- 值矩阵：$V \in \mathbb{R}^{m \times d_v}$
- 输出：$\text{Attention}(Q, K, V) \in \mathbb{R}^{n \times d_v}$

---

## 2. 为什么需要除以 $\sqrt{d_k}$？（方差推导）

很多初学者容易忽略缩放因子 $\frac{1}{\sqrt{d_k}}$ 的数学原因。我们从随机变量统计特性的角度进行证明：

::: tip 假设条件
假设 $q$ 和 $k$ 是独立的 $d_k$ 维向量，且向量各元素服从均值为 0、方差为 1 的独立同分布：
$$
\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \text{Var}(q_i) = \text{Var}(k_i) = 1 \quad (\forall i = 1, \dots, d_k)
$$
:::

### 点积的期望与方差计算

计算点积 $z = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$：

1. **期望计算**：
   $$
   \mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0
   $$

2. **方差计算**：
   因为各项独立：
   $$
   \text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)
   $$
   根据两独立随机变量乘积的方差公式：
   $$
   \text{Var}(XY) = \mathbb{E}[X^2]\mathbb{E}[Y^2] - (\mathbb{E}[X]\mathbb{E}[Y])^2
   $$
   代入得到：
   $$
   \text{Var}(q_i k_i) = (1) \times (1) - 0 = 1
   $$
   因此整个点积的方差为：
   $$
   \text{Var}(z) = \sum_{i=1}^{d_k} 1 = d_k
   $$

::: danger 梯度消失风险
当维度 $d_k$ 较大（例如 $d_k = 64$ 或 $128$）时，点积 $z$ 的方差达到 $d_k$。此时点积数值的绝对值会变得很大，导致通过 Softmax 函数时进入梯度饱和区（极度平坦）：
$$
\lim_{|z| \to \infty} \frac{\partial \text{Softmax}(z)_i}{\partial z_j} \approx 0
$$
:::

因此，将点积除以 $\sqrt{d_k}$，使得缩放后的方差被拉回到 $1$：
$$
\text{Var}\left(\frac{z}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(z) = \frac{d_k}{d_k} = 1
$$
从而确保了反向传播时梯度的稳定流动。

---

## 3. PyTorch 极简实现

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # q: [batch, heads, seq_len_q, d_k]
        # k: [batch, heads, seq_len_k, d_k]
        # v: [batch, heads, seq_len_k, d_v]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
```
