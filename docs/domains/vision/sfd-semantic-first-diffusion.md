# 语义先行扩散范式 (Semantic-First Diffusion, SFD)

> **标签**：`Vision` `Diffusion` `SFD` `Flow Matching` `Latent Space` `REPA` `DINOv2`
> **更新时间**：2026-09-26
> **参考来源**：[Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion (CVPR 2026)](https://arxiv.org/abs/2512.04926) · [GitHub](https://github.com/YuemingPan/SFD) · [项目页](https://yuemingpan.github.io/SFD.github.io/)

---

::: tip 这篇是 SeFi-Image 的前置知识
SeFi-Image 的 §3.1「Semantic-First Diffusion Modeling」几乎逐式复用了本文的公式 (1)–(20)，只是把类别条件 $y$ 换成 Qwen3-VL 文本条件、把 LightningDiT 换成双流 DiT。**没有本文的基础，读 SeFi 的方法章节会寸步难行。** 衔接细节见 [§7 与 SeFi-Image 的关系](#7-与-sefi-image-的关系前置--应用)。
:::

## 1. 问题定义与核心矛盾

### 1.1 根因：VAE 隐空间只擅长纹理，不擅长语义

LDM 的 VAE 为像素级重建而优化，其隐表示天然**偏向低层纹理**。于是扩散模型被迫在一个自相矛盾的目标上优化：

$$
\mathcal{L}_{\text{vel}}(\theta)=\int_0^1 \mathbb{E}_{x_1,x_0}\Big[\big\|v_\theta(x_t,t)-(x_1-x_0)\big\|_2^2\Big]\,dt
$$

它必须**同时**吃下高层语义理解与低层纹理细节 $\Rightarrow$ 收敛慢、生成质量次优。

### 1.2 前人路线的共同盲区：全都在「同步去噪」

| 路线 | 代表 | 做法 | 语义从哪来 |
| :--- | :--- | :--- | :--- |
| 特征级正则 | REPA / REPA-E | 对齐 DiT 中间特征与 VFM 特征 | 蒸馏监督，**不进 latent 空间** |
| 隐空间重设计 | VA-VAE | 让 VAE latent 本身带语义 | 语义与纹理**纠缠**在同一 latent |
| 表征直接替换 | RAE / SVG | 用 VFM 特征替掉 VAE latent | 纹理被牺牲 |
| 联合建模 | ReDi / REG | 拼接 [语义, 纹理] 一起扩散 | 有语义 latent，但**同步去噪** |

::: danger SFD 的核心指控
上述所有方法共享同一个范式：**语义与纹理在整条去噪轨迹上处于同一噪声水平（同一 timestep）被去噪**。这违背了扩散模型自身的 coarse-to-fine 本性——低频结构先于高频纹理生成。语义本该「先形成、再当锚点」，却被强行与纹理同步演化。
:::

### 1.3 为什么不能直接硬串行：Exposure Bias

最朴素的想法：先完全生成语义，再以此为条件生成纹理。**这是错的。** 硬串行等价于 teacher forcing：

- 训练时语义条件是 ground-truth $s_1$；
- 推理时语义条件是模型自己 imperfect 的预测 $\hat{s}_1$；
- 误差在纹理分支被放大 $\Rightarrow$ 训练/推理失配、性能退化（论文原文类比 exposure bias）。

论文用 $\Delta t$ 消融表（Fig. 5）实测证实了这一点：$\Delta t = 1.0$（即完全串行）时 FID 显著劣于 $\Delta t = 0.3$。

**SFD 的解法是「异步」而非「串行」**：两者同时去噪，但处在**不同噪声水平**，语义恒定领先纹理一个固定时间偏移 $\Delta t$。这样纹理分支从训练到推理都只见过「略领先、但 imperfect」的语义，曝光偏差被天然抹平。

## 2. 架构拓扑与特征注入机理

### 2.1 Semantic VAE (SemVAE)：把 768 维 VFM 特征压成 16 通道

VFM 特征太肥（$256\times768$），直接扩散代价高。SemVAE 专做压缩：

$$
f_s=\Phi(x_1)\in\mathbb{R}^{L\times C_{in}}=\mathbb{R}^{256\times 768},\quad
h_s=E_s(f_s)\in\mathbb{R}^{L\times 2C_s}=\mathbb{R}^{256\times 32}
$$

$E_s$ = Linear Proj → 4 × Transformer blocks → LayerNorm → Linear Proj。通道拆分为高斯后验参数：

$$
\mu,\sigma^2=h_s[:, :C_s],\ h_s[:, C_s:]\in\mathbb{R}^{L\times C_s}=\mathbb{R}^{256\times16}
$$

$$
s_1=\mu+\sigma\odot\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

**关键设计：SemVAE 保持 VFM 的 patch 空间排布不变，只压通道。** 这正是语义 latent 能与纹理 latent 沿通道直接拼接的前提。SemVAE 训完即冻结（58M 参数，encoder/decoder 各 29M）。

### 2.2 复合隐空间：通道维拼接

$$
c=[s_1,\ z_1]\in\mathbb{R}^{L\times(C_s+C_z)}=\mathbb{R}^{256\times 48},\qquad z_1=E_z(x_1)\in\mathbb{R}^{256\times 32}
$$

SD-VAE $f16\text{-}d32$ 把纹理编码为 32 通道、16× 下采样（$\mathbb{R}^{32\times16\times16}$，展平为 256 token）。$16+32=48$ 通道，$L=256$ token，对应 $256\times256$ 图像。

::: info 为什么必须保留 VFM 的 token 网格
如果 SemVAE 做了空间下采样，$s_1$ 的 token 数就与 $z_1$ 不匹配，拼接失效。SFD 明确选择**只压通道不压空间**，用语义容量换空间对齐。
:::

### 2.3 双时间步 DiT

$$
[\hat{v}_s,\ \hat{v}_z]=v_\theta\big([s_{t_s},z_{t_z}],\ [t_s,t_z],\ y\big)
$$

两个 **独立** 的 timestep embedder $\tau_s,\tau_z$，各自 hidden dim 降为 $H/2$ 后沿通道拼接注入：

$$
e=[\tau_s(t_s),\ \tau_z(t_z)]
$$

::: tip 双 embedder 反而更省参数
MLP 参数随 hidden dim 平方增长，两个 $H/2$ 宽 MLP 的参数量 $=2\times(H/2)^2=0.5H^2$，只有单个 $H$ 宽 MLP 的**一半**。实测 LightningDiT-XL 参数量从 683.39M **降到** 682.77M，FLOPs 从 116.479G → 116.487G（+0.007%），FID 却从 9.29 → 3.53。**近乎零成本的巨大收益。**
:::

### 2.4 三阶段异步去噪调度（推理）

$$
[M_s, M_z]=\begin{cases}
[\mathbf{1}, \mathbf{0}], & t_s\in[0,\Delta t),\ t_z=0 & \text{Stage I：语义初始化}\\
[\mathbf{1}, \mathbf{1}], & t_s\in[\Delta t,1],\ t_z\in[0,1-\Delta t) & \text{Stage II：异步生成}\\
[\mathbf{0}, \mathbf{1}], & t_s=1,\ t_z\in[1-\Delta t,1] & \text{Stage III：纹理收尾}
\end{cases}
$$

$$
\hat{v}=[M_s\odot\hat{v}_s,\ M_z\odot\hat{v}_z],\qquad M_s\in\{0,1\}^{B\times C_s\times H\times W},\ M_z\in\{0,1\}^{B\times C_z\times H\times W}
$$

::: warning 「不增加推理步数」的关键技巧
SFD 把去噪时间范围**从 $[0,1]$ 扩展到 $[0,1+\Delta t]$**（Stage III 需要 $t_z$ 从 $1-\Delta t$ 走到 1），但**同比拉大步长间隔**，总步数保持不变。所以三阶段调度**零额外推理成本**。采样完成后**只解码纹理隐变量 $z_1$**，$s_1$ 直接丢弃。
:::

## 3. 损失函数与数学稳定性推导

### 3.1 SemVAE 训练目标

$$
\mathcal{L}_{\text{MSE}}=\|\hat{f}_s-f_s\|_2^2,\qquad
\mathcal{L}_{\cos}=1-\frac{\hat{f}_s\cdot f_s}{\|\hat{f}_s\|\|f_s\|}
$$

$$
\mathcal{L}_{\text{KL}}=D_{\text{KL}}\big(q(s_1|f_s)\,\|\,\mathcal{N}(0,I)\big)=\frac{1}{2}\sum_i\big(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1\big)
$$

$$
\mathcal{L}_{\text{SemVAE}}=\mathcal{L}_{\text{MSE}}+\mathcal{L}_{\cos}+\lambda_{\text{kl}}\mathcal{L}_{\text{KL}},\qquad \lambda_{\text{kl}}=10^{-7}
$$

::: info 为什么 MSE 与余弦必须同时用
消融（Supp. Tab. 12）：仅 MSE → FID 10.79；仅余弦 → 10.71；**MSE + 余弦 → 10.14**。MSE 保特征幅值保真度，余弦保高维空间的方向对齐，两者互补，缺一不可。
:::

### 3.2 扩散主干：双流速度回归 + 语义权重 $\beta$

$$
\mathcal{L}_{\text{pred}}=\mathbb{E}_{s_0,s_1,z_0,z_1,t_s,t_z}\Big[\big\|\hat{v}_z-(z_1-z_0)\big\|_2^2+\beta\big\|\hat{v}_s-(s_1-s_0)\big\|_2^2\Big]
$$

$s_0,z_0\sim\mathcal{N}(0,I)$。$\beta$ 是语义损失权重，$\beta=2.0$ 最优（见 §4.2）。

### 3.3 REPA 损失的重解码解读（SFD 的关键理论增量）

$$
\mathcal{L}_{\text{REPA}}(\psi,\phi):=-\mathbb{E}_{s_{t_s},z_{t_z},t_s,t_z}\Big[\mathcal{L}_{\text{sim}}\big(y^*,\,h_\phi(h_t)\big)\Big]
$$

其中 $y^*=f(x_1)$ 为 VFM 输出，$h_t=f_\psi([s_{t_s},z_{t_z}],[t_s,t_z])$ 为 DiT 编码器输出。

::: tip 与原始 REPA 的本质区别
原版 REPA 把对齐当作**蒸馏**——逼扩散模型从零「分析理解」输入 latent，因此需要很深的对齐层（depth 8）。SFD 因为语义 latent 已经是**压缩过的显式语义**，对齐目标变成「**从噪声语义 latent 解码回干净表征**」——一个天然更容易的重建任务。所以最优对齐深度降到 **layer 2**，且对齐函数用 cosine + MSE 组合（与 SemVAE 训练度量一致）时最优。
:::

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{pred}}+\lambda\mathcal{L}_{\text{REPA}}
$$

（原文 Eq.18 写作 $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{vel}}+\lambda\mathcal{L}_{\text{REPA}}$，其中 $\mathcal{L}_{\text{vel}}$ 即上式的 $\mathcal{L}_{\text{pred}}$。）

### 3.4 双时间步采样与偏移的严格性质

$$
t_s\sim\mathcal{U}(0,\,1+\Delta t),\qquad t_z=\max(0,\,t_s-\Delta t),\qquad t_s\leftarrow\min(t_s,\,1)
$$

**采样区间为何要扩展到 $1+\Delta t$？** 因为 Stage III 需要 $t_s$ 停在 1（语义已干净）而 $t_z$ 继续从 $1-\Delta t$ 走到 1；若 $t_s$ 被硬截断在 1，$t_z$ 永远到不了 1，纹理无法完成。

::: info 证明：实际偏移恒为 $\Delta t$ 的下界满足 $0\le t_s-t_z\le\Delta t$
记原始采样 $u\sim\mathcal{U}(0,1+\Delta t)$，则 $t_s=\min(u,1)$，$t_z=\max(0,u-\Delta t)$。

- **情形 A**：$u\le 1$。此时 $t_s=u$，故
  $$t_s-t_z=u-\max(0,u-\Delta t)=\min(u,\ \Delta t)\in[0,\Delta t]$$
- **情形 B**：$u>1$。此时 $t_s=1$，$t_z=u-\Delta t\in(0,1]$，故
  $$t_s-t_z=1+\Delta t-u\in(0,\ \Delta t)$$

两情形合并得 $t_s,t_z\in[0,1]$ 且 $0\le t_s-t_z\le\Delta t$。**偏移恰好等于 $\Delta t$ 的区间是 $u\in[\Delta t,1]$**（情形 A 的后段），也正是 Stage II 的主体。$\blacksquare$
:::

由此，训练时模型见到的是「偏移在 $[0,\Delta t]$ 内连续分布」的样本对，而非单一固定偏移——这正是异步调度能学到**协同**而非死板串行的原因。

## 4. 保真度与生成权衡 (Trade-off Analysis)

所有消融在 LightningDiT-XL / 400K iters / lr $2\times10^{-4}$ / $\beta_2=0.95$ 下进行，指标为 FID-50K。

### 4.1 核心收益：近乎免费的加速

| Model | #Params | Iter. | FID↓ |
| :--- | :--- | :--- | :--- |
| DiT-XL/2 | 675M | 7M | 9.62 |
| LightningDiT-XL/1 | 675M | 400K | 9.29 |
| LightningDiT-XL/1 + REPA | 675M | 400K | 6.94 |
| LightningDiT-XL/1 + REPA | 675M | 4M | 5.84 |
| **+ SFD (Ours)** | 675M | **70K** | 8.79 |
| **+ SFD (Ours)** | 675M | **120K** | 6.22 |
| **+ SFD (Ours)** | 675M | **400K** | **3.53** |
| **+ SFD (Ours)** | 675M | 4M | **2.54** |

- 达到 DiT-XL 训练 7M 步的 FID 只需 **70K 步**（约 $100\times$ 加速）；
- 达到 LightningDiT-XL 训练 4M 步的 FID 只需 **120K 步**（约 $33.3\times$ 加速）；
- 400K 步时 3.53 已优于 LightningDiT+REPA 的 4M 步（5.84）与 DiT 的 7M 步（9.62），**训练成本仅 10% / 5.7%**。

### 4.2 $\Delta t$ 与 $\beta$ 双超参的博弈

| $\Delta t$ | 0（=ReDi/REG 同步） | **0.3（最优）** | 1.0（硬串行） |
| :--- | :--- | :--- | :--- |
| FID↓ | 退化为同步联合去噪 | **3.03** | 训练/推理失配，性能次优 |

| $\beta$ | 0.25 | 0.5 | 1.0 | **2.0** | 4.0 | 8.0 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FID↓ | 3.46 | 3.26 | 3.08 | **3.03** | 3.28 | 3.96 |

::: warning 两个超参共享同一个失效模式
$\Delta t$ 过大 $\to$ 语义跑太远，纹理失去协同，退化为 teacher forcing；$\beta$ 过大 $\to$ 语义监督压制纹理学习，细节丢失。**语义先行不等于语义至上**，两者都存在甜点区，$\Delta t=0.3$ / $\beta=2.0$。
:::

### 4.3 组件消融：谁在贡献性能

| REPA | SemVAE | Semantic-First | FID↓ |
| :--- | :--- | :--- | :--- |
| ✗ | ✗ | ✗ | 8.17 |
| ✓ | ✗ | ✗ | 7.08 |
| ✓ | ✓ | ✗ | 5.24 |
| ✓ | ✓ | ✓ | **3.03** |

**语义优先机制单独贡献 5.24 → 3.03，是最大的单项增益**，超过 REPA（8.17→7.08）与 SemVAE（7.08→5.24）之和。

### 4.4 语义压缩方式与容量

| 压缩方式 | FID↓ | | 语义通道数 $C_s$ | 2 | 4 | 8 | **16** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PCA（ReDi 式） | 4.06 | | FID↓ | 3.90 | 3.67 | 3.16 | **3.03** |
| **SemVAE** | **3.03** | | | | | | |

通道数从 2 增到 16 单调改善，**没有出现饱和拐点**——语义容量对生成质量是直接的正向作用。

### 4.5 VFM 选择与尺度：一个反直觉发现

| 目标表征 | DINOv2-B | MAE-B | CLIP-B | SigLip-B |
| :--- | :--- | :--- | :--- | :--- |
| FID↓ | **3.03** | 6.29 | 4.89 | 4.15 |

| DINOv2 尺度 | S | **B（默认）** | L |
| :--- | :--- | :--- | :--- |
| FID↓ | 4.14 | 3.03 | **2.97** |

::: tip 与 REG / RAE 的正面分歧
REG 与 RAE 都发现 **DINOv2-B 最优**，放大 VFM 反而因维度爆炸导致性能下降。SFD 的结论相反：**放大 VFM 持续受益**（L 优于 B）。原因正是「显式语义压缩」——SemVAE 吸收了高维特征的维度灾难。这条差异是 SFD 方法有效性的强证据。
:::

### 4.6 REPA 配置：浅层对齐即可

| 对齐深度 | 无 REPA | **2** | 4 | 6 | 8 | 10 | 12 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FID↓ | 4.15 | **3.03** | 3.07 | 3.24 | 3.16 | 3.19 | 3.28 |

| 权重 $\lambda$ | 0.25 | **0.5** | 1.0 | 2.0 | 4.0 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FID↓ | 3.30 | **3.03** | 3.18 | 3.25 | 3.20 |

对齐函数：cosine 3.16 / MSE 3.13 / **cosine+MSE 3.03**。

### 4.7 重建保真度：SFD 不用牺牲重建

| 隐空间方案 | rFID↓ | PSNR↑ | LPIPS↓ | SSIM↑ |
| :--- | :--- | :--- | :--- | :--- |
| VA-VAE（语义增强但纠缠） | 0.28 | 27.96 | 0.096 | 0.79 |
| RAE（纯 VFM 表征，纹理匮乏） | 0.57 | 18.86 | 0.256 | 0.42 |
| **SD-VAE（SFD 纹理分支）** | **0.26** | **28.59** | **0.089** | **0.80** |

SFD 靠**另开一条语义旁路**来拿语义引导，纹理 VAE 完全不用动——这是它区别于 VA-VAE / RAE 的结构性优势。论文明确指出这使 SFD **天然更适合 T2I 与一致性要求高的图像编辑**。

### 4.8 泛化性：作为插件嫁接他人

| 宿主方法 | w/o SFD | w/ SFD |
| :--- | :--- | :--- |
| ReDi（PCA 语义 + SD-VAE 纹理） | 5.33 | **4.41** |
| VA-VAE（语义纹理纠缠在同一 latent） | 4.52 | **4.14** |

::: warning 泛化增益的上限由「语义纹理是否解耦」决定
ReDi 拿到 0.92 增益，VA-VAE 只有 0.38。论文解释：VA-VAE 的 latent 本身纠缠语义与纹理，异步去噪**没有可操作的自由度**。**SFD 的收益本质上来自解耦，解耦程度决定收益上限。**
:::

### 4.9 无条件生成：异步调度的收益更大

| Method | Epochs | Params | FID↓ | IS↑ |
| :--- | :--- | :--- | :--- | :--- |
| ReDi | 80 | 675M | 25.10 | – |
| RAE (w/ AG) | 200 | 839M | 4.96 | 123.1 |
| RCG-G (MAGE-L) | 800 | 502M | 2.15 | 253.4 |
| **SFD (w/o AG)** | **80** | 675M | **10.24** | 78.5 |
| **SFD (w/ AG)** | **80** | 675M | **3.77** | 127.9 |
| **SFD (w/ AG)** | 200 | 675M | **2.90** | 148.5 |

无类别标签时语义表示更平滑、更易建模，异步机制的优势被放大。

## 5. 核心控制层代码实现

三阶段掩码调度的最小实现（`t` 在扩展区间 $[0,1+\Delta t]$ 上扫，掩码逻辑可化简为两个比较）：

```python
import torch

def sample_train_timesteps(bs: int, dt: float, device="cuda"):
    """训练侧双时间步采样 (Eq.12-14)。偏移在 [0, dt] 内连续分布。"""
    u = torch.rand(bs, device=device) * (1.0 + dt)   # 扩展区间
    t_z = torch.clamp(u - dt, min=0.0)               # 纹理滞后
    t_s = torch.clamp(u, max=1.0)                    # 语义截断
    return t_s, t_z


def sfd_masks(t: float, dt: float):
    """三阶段二元掩码 (Eq.19)：t 为标量时刻，返回 (M_s, M_z)。"""
    #  Stage I  (0 <= t < dt)      : M_s=1, M_z=0  语义初始化
    #  Stage II (dt <= t < 1)      : M_s=1, M_z=1  异步生成
    #  Stage III(1 <= t <= 1+dt)   : M_s=0, M_z=1  纹理收尾
    return (t < 1.0), (t >= dt)


@torch.no_grad()
def sfd_sample(v_theta, s_shape, z_shape, dt=0.3, n_steps=50, device="cuda", **cond):
    """SFD 异步去噪采样。步数不变，仅把时间范围拉长到 1+dt。"""
    s, z = torch.randn(s_shape, device=device), torch.randn(z_shape, device=device)
    grid = torch.linspace(0.0, 1.0 + dt, n_steps + 1, device=device)

    for i in range(n_steps):
        t, t_next = grid[i].item(), grid[i + 1].item()
        t_s, t_z = min(t, 1.0), max(0.0, t - dt)
        M_s, M_z = sfd_masks(t, dt)
        step = t_next - t

        v_s, v_z = v_theta(torch.cat([s, z], dim=-1), [t_s, t_z], **cond)
        s = s + step * M_s * v_s      # Stage III 后语义冻结
        z = z + step * M_z * v_z      # Stage I 期间纹理不动

    return z                           # 只解码纹理隐变量，s 丢弃
```

::: info 掩码化简的依据
由 §3.4 的证明，$t_s=\min(t,1)$ 在 $t\ge1$ 时恒为 1（语义已干净），$t_z=\max(0,t-\Delta t)$ 在 $t<\Delta t$ 时恒为 0（纹理尚未启动）。因此 Stage I/II/III 的划分**完全由 $t$ 与两个阈值 $\{1,\Delta t\}$ 的比较决定**，无需显式三段分支。
:::

## 6. 避坑指南与评测基准

### 6.1 六个必须避开的坑

1. **不要把 $\Delta t$ 开到 1.0**。硬串行触发 exposure bias，FID 明显退化。甜点区是 0.3。
2. **不要认为 $\Delta t$ 越大「语义先行」越彻底越好**。它与 $\beta$ 共享失效模式：语义跑太远 = 纹理失去协同。
3. **不要省掉 SemVAE 的余弦损失**。只上 MSE 会掉 ~0.65 FID（10.79 vs 10.14）。
4. **不要照搬 REPA 的 depth 8**。SFD 框架下最优是 depth 2，照搬会掉 0.13 FID。
5. **不要在纹理高维 latent（$f16\text{-}d32$）上期待无 guidance 的表现**。论文坦承这是短板：纹理 latent 维度是 $f8\text{-}d4$ 的两倍，无 guidance 时纹理收敛困难（800ep 无 guidance FID 2.54，明显弱于 ReDi 同期口径），必须依赖 AutoGuidance 补上。
6. **不要在语义与纹理纠缠的 latent 上指望异步机制生效**。VA-VAE 上只有 0.38 增益（见 §4.8）。

### 6.2 评测协议

- **指标**：FID（视觉真实性）、sFID（空间结构一致性）、IS（类别多样性）、Precision / Recall（分布覆盖）。50K 样本，遵循 ADM 标准评测流水线。
- **采样器**：dopri5 自适应 ODE，atol $10^{-6}$ / rtol $10^{-3}$。
- **引导**：**AutoGuidance**（非 CFG），用退化的 LightningDiT-B 作引导网络，scale 1.5–1.6。
- **timestep 采样**：logit-normal（沿用 LightningDiT）。
- **对比公平性**：采用 RAE 提出的 **class-balanced sampling**（每类 50 张，共 50K），并同时报告 class-random 口径。

### 6.3 主结果（ImageNet 256×256，with guidance）

| Method | Epochs | #Params | FID↓ | sFID↓ | IS↑ | Prec.↑ | Rec.↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DiT-XL | 1400 | 675M | 2.27 | 4.60 | 278.2 | 0.83 | 0.57 |
| VA-VAE | 800 | 675M | 1.35 | 4.15 | 295.3 | 0.79 | 0.65 |
| REPA | 800 | 675M | 1.42 | 4.70 | 305.7 | 0.80 | 0.65 |
| REPA-E | 800 | 675M | 1.12 | 4.09 | 302.9 | 0.79 | 0.66 |
| ReDi | 800 | 675M | 1.61 | 4.66 | 295.1 | 0.78 | 0.64 |
| REG | 800 | 677M | 1.36 | 4.25 | 299.4 | 0.77 | 0.66 |
| RAE (DiTDH-XL) | 800 | 839M | 1.13 | – | 262.6 | 0.78 | 0.67 |
| **SFD (XL)** | **80** | 675M | 1.30 | 3.87 | 233.4 | 0.78 | 0.64 |
| **SFD (XL)** | 800 | 675M | **1.06** | 3.89 | 267.0 | 0.78 | **0.67** |
| **SFD (XXL)** | 800 | 1.0B | **1.04** | **3.75** | 264.2 | 0.78 | 0.66 |

::: tip sFID 才是 SFD 的主场
sFID 衡量空间结构一致性。SFD-XXL 的 **3.75 显著优于所有对手**（次优 REPA-E 4.09），这直接验证了「显式压缩语义表征并保留空间布局 → 先建立鲁棒全局结构，再精修纹理」的设计。**评估 SFD 类方法时不要只看 FID，sFID 是更灵敏的探针。**
:::

### 6.4 少步采样友好

FID-10K，400K iters，Euler，无 guidance：

| 推理步数 | 250 | 200 | 150 | 100 | 80 | 60 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightningDiT | 12.50 | 12.58 | 12.67 | 12.91 | 13.03 | 13.40 |
| LightningDiT+REPA | 10.00 | 10.10 | 10.23 | 10.50 | 10.67 | 10.94 |
| LightningDiT+ReDi | 8.58 | 8.63 | 8.72 | 8.86 | 9.02 | 9.32 |
| **LightningDiT+SFD** | 6.32 | 6.26 | 6.41 | **6.35** | 6.81 | 6.77 |

with guidance（4M iters，SFD-XL）：dopri5 1.064 / 100 步 Euler **1.045** / 25 步 1.865（优于 SVG 的 1.920）。语义早期稳定 ⇒ 所需精修步数更少。

## 7. 与 SeFi-Image 的关系（前置 → 应用）

SFD 原文只在 ImageNet 256×256 类别条件下验证，并把「扩展到 T2I / T2V」列为 future work。**SeFi-Image 是 SFD 第一次大规模 T2I 化**：

| 维度 | SFD (CVPR 2026) | SeFi-Image |
| :--- | :--- | :--- |
| 条件信号 | 类别标签 $y$（1000 类） | Qwen3-VL LLM hidden states（多隐层拼接，context 512→1024） |
| 骨干 | LightningDiT 单流 | 双流 MMDiT → 单流（1B/2B/5B） |
| VFM | DINOv2-B-reg（消融显示 L 更好，2.97） | **DINOv2-Large**（与消融结论一致） |
| 纹理 VAE | SD-VAE $f16\text{-}d32$，32 ch，不动 | FLUX.2 VAE 微调，32 ch，Kodak PSNR 33.18→**36.40** |
| 语义通道 $C_s$ | 16（消融最优） | 未披露 |
| $\Delta t$ | 0.3 | 0.2（256/512px）→ **0.1**（768/1024px） |
| $\beta$ | 2.0 | 2（预训练）→ 1（CT/SFT） |
| 损失 | $\mathcal{L}_{\text{pred}}+\lambda\mathcal{L}_{\text{REPA}}$ | 同（Eq.6–8） |
| 三阶段掩码 | Eq.19 | Eq.9（同一套） |

::: warning 两个值得注意的偏移
1. **$\Delta t$ 随分辨率递减**（0.2 → 0.1）。合理解释：分辨率越高单步承载的信息越少，异步窗口应收窄；但论文未给出消融，属观察而非结论。
2. **$\beta$ 从 2 降到 1**。预训练重语义（先立结构），CT/SFT 阶段结构已稳，转为纹理精刻画。SeFi 笔记 §2.2 已记录此机制。
:::

SeFi 相对 SFD 的真正增量不在算法，而在**证明了 SFD 能突破「类别条件 + ImageNet + ≤1B」的规模边界**，且能在高保真 VAE（ aggressively fine-tune 到 PSNR 36.40）下仍快速收敛——这正是 SFD 原文 §4.5 主张的「重建-生成权衡的桥接作用」。

::: tip 面向视频的迁移视角
SFD 原文与 SeFi 均把 T2V 列为下一步。对视频 DiT 的天然适配点：**语义分支天然对应「低频时序结构 / 场景布局 / 角色身份」，纹理分支对应「高频细节 / 逐帧外观」**。$\Delta t$ 机制可直接映射为「语义帧领先纹理帧 $\Delta t$ 步」，且视频 VAE 的高压缩比使重建-生成矛盾比图像更尖锐，SFD 的解耦收益理论上更大。SeFi 原文 limitation 章节亦持此判断。
:::

## 附：关键超参速查（Supp. Table 1–3）

| 组件 | 配置 |
| :--- | :--- |
| VFM | DINOv2-B with registers，patch 对应 256×256 → $L=256$ |
| SemVAE | 4 enc + 4 dec Transformer blocks，hidden 768，6 heads，bottleneck 16 ch，$\lambda_{\text{kl}}=10^{-7}$，58M（29M+29M） |
| SemVAE 训练 | ImageNet-1K，1M iters，bs 64，AdamW lr $5\times10^{-5}$，warmup 500 + constant 800K + cosine 退火至 $5\times10^{-6}$ |
| 复合 latent | $16\times16\times48$（16 sem + 32 tex），256 token |
| DiT 尺度 | B 130M/12L/768dim/12head；L 458M/24L/1024/16；XL 675M/28L/1152/16；XXL 1.0B/32L/1280/16 |
| 主训练 | bs 256，AdamW lr $1\times10^{-4}$，$\beta=(0.9,0.999)$，800 epochs，logit-normal timestep |
| 采样 | dopri5，atol $10^{-6}$，rtol $10^{-3}$；AutoGuidance 用退化的 LightningDiT-B，scale 1.5–1.6 |
