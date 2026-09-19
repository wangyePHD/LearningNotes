# ControlNet 空间结构可控生成原理解析

> **标签**：`Vision` `Diffusion` `ControlNet` `Zero-Conv`  
> **更新时间**：2026-09-19

---

## 1. 核心思想：锁定主干与零卷积 (Zero Convolution)

ControlNet 解决的核心问题是：**如何在引入强空间先验（如 Canny 边缘、OpenPose 姿态、Depth 图）的同时，不破坏预训练文生图大模型的生成能力？**

它的核心结构设计：
1. **Locked Copy**：锁定原始预训练模型的权重（如 SD 1.5 / SDXL / SD3），保持参数不更新。
2. **Trainable Copy**：复制一份 Encoder 结构，用于接收额外的条件向量 $c_f$。
3. **Zero Convolution**：利用权重和偏置全部初始化为 0 的 $1 \times 1$ 卷积层连接两个分支。

---

## 2. 零卷积的数学稳定性证明

定义零卷积操作为 $\mathcal{Z}(x; \mathcal{W}, \mathbf{b})$，在训练初始时刻：
$$
\mathcal{W} = 0, \quad \mathbf{b} = 0
$$

输入特征向量 $x$，初始前向输出恒为 0：
$$
\mathcal{Z}(x; \mathbf{0}, \mathbf{0}) = 0 \cdot x + 0 = 0
$$

因此，在第一步迭代时，注入主干网络的额外扰动量为 0，模型行为与原始预训练模型完全一致：
$$
y = \mathcal{F}(x; \Theta) + \mathcal{Z}(\mathcal{F}(x + c; \Theta_{\text{copy}}); \mathcal{W}, \mathbf{b}) \Big|_{\mathcal{W}=0} = \mathcal{F}(x; \Theta)
$$

### 反向传播梯度有效性

很多人误以为全 0 权重会导致梯度无法流动。其实不然：
$$
\frac{\partial \mathcal{Z}}{\partial \mathcal{W}} = x
$$
当反向传播时，如果外界损失传递回来的梯度为 $\frac{\partial \mathcal{L}}{\partial y}$：
$$
\frac{\partial \mathcal{L}}{\partial \mathcal{W}} = \frac{\partial \mathcal{L}}{\partial y} \cdot x \neq 0
$$
因此在第一轮梯度更新后，$\mathcal{W}$ 立即离开 0 点开始学到条件特征，整个训练平滑无突变。
