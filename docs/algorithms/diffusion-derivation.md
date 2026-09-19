# DDPM 扩散模型变分推断与前向重参数化推导

> **标签**：`Generative Models` `Diffusion` `Math Derivation`  
> **更新时间**：2026-09-19

---

## 1. 前向扩散过程 (Forward Process)

前向过程为一个固定的马尔可夫链，逐步向真实数据 $x_0 \sim q(x_0)$ 中添加高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}\right)
$$

其中 $\beta_1, \beta_2, \dots, \beta_T$ 为预先设定的方差调度计划（Variance Schedule）。

---

## 2. 任意步 $x_t$ 的重参数化闭式解

由于每一步均为高斯分布，我们无需逐步迭代计算，可以直接从 $x_0$ 采样出任意时刻 $x_t$。

令 $\alpha_t = 1 - \beta_t$，且累乘项定义为：
$$
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

利用重参数化技巧（Reparameterization Trick）：
$$
x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_0, \quad \epsilon_0 \sim \mathcal{N}(0, \mathbf{I})
$$
$$
x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2} \epsilon_1 = \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_0) + \sqrt{1 - \alpha_2} \epsilon_1
$$

合并同分布的高斯变量 $\mathcal{N}(0, \sigma_1^2 \mathbf{I}) + \mathcal{N}(0, \sigma_2^2 \mathbf{I}) \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$：

$$
\sqrt{\alpha_2(1 - \alpha_1)}^2 + \sqrt{1 - \alpha_2}^2 = \alpha_2 - \alpha_1 \alpha_2 + 1 - \alpha_2 = 1 - \alpha_1 \alpha_2 = 1 - \bar{\alpha}_2
$$

::: info 核心闭式定理
数学归纳法易证，任意时刻 $t$ 的状态分布为：
$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)
$$
采样公式写为：
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$
:::
