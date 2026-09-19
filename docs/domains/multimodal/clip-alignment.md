# CLIP 跨模态对比表征学习数学解析

> **标签**：`Multimodal` `CLIP` `Contrastive Learning` `InfoNCE`  
> **更新时间**：2026-09-19

---

## 1. 对称对比学习损失函数 (Symmetric Cross Entropy)

给定一个 Batch 的 $N$ 个图文对 $(I_i, T_i)$：
- 视觉编码器提取特征并归一化：$v_i = \frac{f_v(I_i)}{\|f_v(I_i)\|_2}$
- 文本编码器提取特征并归一化：$t_i = \frac{f_t(T_i)}{\|f_t(T_i)\|_2}$

构造 $N \times N$ 相似度矩阵，点积项即为余弦相似度：
$$
S_{i, j} = v_i^T t_j \cdot \exp(\tau)
$$
其中 $\tau$ 为可学习的温度缩放因子（Temperature Parameter）。

---

## 2. 损失函数推导

总损失为图像到文本损失 $\mathcal{L}_{I \to T}$ 与文本到图像损失 $\mathcal{L}_{T \to I}$ 的平均：

$$
\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{i, i})}{\sum_{j=1}^N \exp(S_{i, j})}
$$

$$
\mathcal{L}_{T \to I} = -\frac{1}{N} \sum_{j=1}^N \log \frac{\exp(S_{j, j})}{\sum_{i=1}^N \exp(S_{i, j})}
$$

$$
\mathcal{L}_{\text{total}} = \frac{1}{2} (\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I})
$$

::: info 几何意义
拉近匹配对 $(I_i, T_i)$ 在多模态超球面上的距离，推远非匹配对 $(I_i, T_j)$，实现文本与视觉语义特征的零样本（Zero-Shot）迁移。
:::
