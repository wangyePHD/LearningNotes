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

::: tip 统计假设
假设 $q$ 和 $k$ 是独立的 $d_k$ 维向量，且向量各元素服从均值为 0、方差为 1 的独立同分布：
$$
\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \text{Var}(q_i) = \text{Var}(k_i) = 1 \quad (\forall i = 1, \dots, d_k)
$$
:::

### 点积方差计算

计算点积 $z = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$：

1. **期望计算**：
   $$
   \mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0
   $$

2. **方差计算**：
   $$
   \text{Var}(z) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left( \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - (\mathbb{E}[q_i]\mathbb{E}[k_i])^2 \right) = d_k
   $$

::: danger 梯度饱和风险
当 $d_k$ 很大时，点积结果的方差达到 $d_k$。Softmax 函数在自变量绝对值很大时进入极度平坦的饱和区，反向传播梯度几乎为 0。
:::

因此，除以 $\sqrt{d_k}$ 使缩放后的方差被拉回到 $1$：
$$
\text{Var}\left(\frac{z}{\sqrt{d_k}}\right) = \frac{1}{d_k} \text{Var}(z) = \frac{d_k}{d_k} = 1
$$
确保训练梯度的长期数值稳定性。
