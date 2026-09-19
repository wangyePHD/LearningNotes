# Flow Matching 与连续流生成动力学推导

> **标签**：`Generative Models` `Flow Matching` `ODE` `Optimal Transport`  
> **更新时间**：2026-09-19

---

## 1. 从扩散模型到流匹配 (Flow Matching)

传统 DDPM/SDE 扩散模型通过曲线路径（Curved Trajectories）向数据添加噪声，导致逆向采样需要较多步数（如 20~50 步）。  
**Flow Matching (FM)** 直接在噪声分布 $p_0 = \mathcal{N}(0, \mathbf{I})$ 与真实数据分布 $p_1 = q(x_1)$ 之间定义直线最优传输（Optimal Transport Displacement Interpolant）：

$$
x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]
$$

其导数（真实速度场 / Vector Field）极其简洁且为常数：
$$
u_t(x_t \mid x_0, x_1) = \frac{d x_t}{d t} = x_1 - x_0
$$

---

## 2. 条件流匹配损失 (Conditional Flow Matching)

由于直接从全局不可知真实边际分布回归速度场在计算上不可行，CFM 定理证明：学习**条件速度场**等价于学习边际速度场！

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], \, x_0 \sim p_0, \, x_1 \sim p_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

::: tip 现代生成模型的绝对主导
相比传统扩散模型，基于 Optimal Transport 的 Flow Matching 轨迹近乎直线，结合欧拉（Euler）解算器仅需 **4~8 步** 即可高质量生成视频与图像（如 SD3、Flux、Wan2.1）。
:::
