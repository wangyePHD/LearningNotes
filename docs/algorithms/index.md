# 📐 理论与公式推导 (Theory & Algorithms)

本模块专为算法研究与数学推导设计。主要收录核心算法的数学建模、损失函数推导、证明过程及复杂度分析。

## 最新笔记索引

| 算法/模型 | 核心数学思想 | 关键结论/公式 | 状态 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **Transformer Attention** | 缩放点积、方差保持 | $\text{Softmax}(QK^T/\sqrt{d_k})V$ | 🟢 已完结 | [阅读笔记 →](./attention-mechanism.md) |
| **DDPM 扩散模型** | 马尔可夫链、变分推断、重参数化 | $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ | 🟢 已完结 | [阅读笔记 →](./diffusion-derivation.md) |

::: tip 格式规范
推导过程推荐使用标准 LaTeX 块：
- 独立公式：`$$ ... $$`
- 行内公式：`$ ... $`
- 关键推论使用 `::: tip 关键定理` 或 `::: info 推导细节` 容器包裹。
:::
