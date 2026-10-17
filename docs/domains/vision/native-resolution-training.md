# 多分辨率图像高效训练：Native Resolution + Packing + VarLen Attention (Mage-Flow / MOSS-Image)

> **标签**：`Vision` `Diffusion` `DiT` `Native Resolution` `Packing` `VarLen Attention` `Progressive Resolution` `Efficiency`
> **更新时间**：2026-10-17
> **参考来源**：[Mage-Flow (arXiv:2607.19064)](https://arxiv.org/abs/2607.19064) · [Mage-Flow 精读笔记](./mage-flow.md) · [MOSS-Image Technical Report](https://github.com/OpenMOSS/MOSS-Image/blob/main/paper/MOSS_Image_Technical_Report.pdf) · [MOSS-Image GitHub](https://github.com/OpenMOSS/MOSS-Image) · [MOSS-Image Project Page](https://openmoss.ai/MOSS-Image/)
> **定位**：跨模型技术专题，不是单篇论文精读。把「不同尺寸图像怎么训练」拆成四个可以独立复用的机制，并标注哪些细节来自论文原文、哪些只是原理性说明。

---

::: tip 30 秒版本
$$
\boxed{
\begin{aligned}
&\text{Native Resolution} &&\text{解决「尺寸不一样怎么进模型」}\\
&\text{Variable-Length Tokens} &&\text{解决「每张图 token 数不一样」}\\
&\text{Packing + VarLen Attention} &&\text{解决「长度不一样怎么高效组成 batch」}\\
&\text{Progressive Resolution} &&\text{解决「高分辨率太贵怎么分阶段训」}
\end{aligned}
}
$$
:::

## 1. 问题定义：固定分辨率训练到底浪费在哪里

### 1.1 强制 resize / crop 的几何失真

把任意图像统一拉成 $S\times S$，等价于对高度和宽度分别使用不同的缩放因子：

$$
s_h=\frac{S}{H_0},\qquad s_w=\frac{S}{W_0}
$$

当 $s_h\neq s_w$ 时，图像内容被**各向异性缩放**：人脸被压扁、建筑被拉长、圆形变椭圆。若改用 center crop，则长图两端的内容被直接丢弃。两种做法都会改变训练分布与推理时的构图先验。

### 1.2 Resolution bucket 的量化误差

传统做法预先定义有限个分辨率 / 长宽比 bucket，再把每张图分配到最近的 bucket。设 bucket 中心为 $(H_b,W_b)$，一种便于理解的量化误差写法是（**不是论文原式**）：

$$
\Delta_{\rm bucket}
=
\left|\log\frac{H_0}{W_0}-\log\frac{H_b}{W_b}\right|
+
\lambda\left|\log\frac{H_0W_0}{H_bW_b}\right|
$$

第一项度量长宽比偏差，第二项度量面积偏差，$\lambda$ 控制两者的相对权重。

这个误差有三个直接后果：

| 问题 | 机理 |
| :--- | :--- |
| **Bucket-quantization mismatch** | 真实图像的尺寸 / 长宽比是连续分布，硬塞进有限 bucket 会产生系统性偏差 |
| **单步内 aspect-ratio diversity 受限** | 一个 optimization step 通常只从一个 bucket 取数据，模型每次 update 见到的长宽比种类很少 |
| **极端尺寸无法自然覆盖** | 想支持 $512\times2048$ 这类极端比例，必须提前专门增加 bucket，否则训练时根本见不到 |

### 1.3 Padding 的算力浪费

设一个 batch 有 $B$ 个样本，第 $i$ 个样本的真实长度为 $L_i$，batch 内最大长度为 $L_{\max}$。Padding 到统一长度后：

$$
\text{GPU 实际处理的 slot 数}=B L_{\max},
\qquad
\text{真正有效的 token 数}=\sum_{i=1}^{B}L_i
$$

padding 浪费率为：

$$
\eta_{\rm pad}
=
1-\frac{\sum_{i=1}^{B}L_i}{B L_{\max}}
$$

如果长度分布很宽（短 prompt + 长 prompt、小图 + 大图混在一起），$\eta_{\rm pad}$ 会非常高。更严重的是，标准 dense attention 的计算量按 $L_{\max}^2$ 计，padding token 也会被送进 attention：

$$
C_{\rm padded}=O\left(B L_{\max}^2 d_h\right)
$$

而真正需要的计算量只有：

$$
C_{\rm ideal}=O\left(\sum_{i=1}^{B}L_i^2 d_h\right)
$$

这就是 Packing + VarLen Attention 要解决的核心矛盾。

---

## 2. Native Resolution：保留长宽比，只约束像素预算

### 2.1 面积约束与对齐

Native Resolution 的关键不是「保持原始像素尺寸」，而是三件事同时满足：

$$
\boxed{
\text{保持长宽比}
\;+\;
\text{限制像素面积}
\;+\;
\text{对齐到模型要求的倍数}
}
$$

设原图为 $(H_0,W_0)$，像素面积上限为 $A_{\max}$，对齐倍数为 $a$（通常是 VAE 下采样倍数与 patch size 的乘积）。先计算缩放因子：

$$
s=\min\left(1,\sqrt{\frac{A_{\max}}{H_0W_0}}\right)
$$

再对齐到 $a$ 的整数倍：

$$
H'=a\left\lfloor\frac{sH_0}{a}\right\rfloor,
\qquad
W'=a\left\lfloor\frac{sW_0}{a}\right\rfloor
$$

由于向下取整，$H'W'\le A_{\max}$ 自动成立（前提是 $A_{\max}$ 本身是 $a^2$ 的倍数）。长宽比的相对误差上界约为：

$$
\left|\frac{H'/W'}{H_0/W_0}-1\right|
\lesssim
\frac{a}{\min(H',W')}
$$

也就是说，**对齐倍数越小、目标尺寸越大，长宽比保持得越精确**。

::: info 一个具体的缩放例子
原图 $1920\times1080$：

- 若像素预算约为 $0.92$M（720p 级别），缩放后得到 $1280\times720$，而不是强行拉成 $1024\times1024$；
- 若像素预算为 $1024^2\approx1.05$M，按上式会得到约 $1360\times768$（两者都是 16 的倍数），面积 $1{,}044{,}480\le1{,}048{,}576$。

两种结果都保留了 $16:9$ 的构图，只是像素预算不同。
:::

### 2.2 Token 数公式

设图像 tokenizer 的**有效空间步长**为 $p$，则图像 token 数为：

$$
N_{\rm img}
=
\left(\frac{H'}{p}\right)\left(\frac{W'}{p}\right)
=
\frac{H'W'}{p^2}
$$

对 $p=16$ 的模型：

| 分辨率 | Token 数 | 说明 |
| :--- | ---: | :--- |
| $256\times256$ | $16\times16=256$ | 低分辨率预训练阶段 |
| $512\times512$ | $32\times32=1024$ | 中分辨率阶段 |
| $1024\times1024$ | $64\times64=4096$ | 最终阶段常见预算 |
| $1280\times720$ | $80\times45=3600$ | 16:9，面积小于 $1024^2$ |
| $2048\times512$ | $128\times32=4096$ | 极端长宽比，token 数仍受控 |

::: warning Token 数由「面积」决定，不是由「最长边」决定
$2048\times512$ 的最长边是 2048，但 token 数只有 4096，和 $1024\times1024$ 一样。**面积约束天然适合处理极端长宽比**：宽图 / 长图可以保留构图，同时把 attention 成本控制在预算内。
:::

### 2.3 张量形状：从像素到 packed token

以 Mage-Flow 的 Mage-VAE（16× 空间下采样）为例：

$$
\underbrace{\mathbb{R}^{B\times3\times H\times W}}_{\text{image}}
\xrightarrow{\ \text{VAE encode}\ }
\underbrace{\mathbb{R}^{B\times C\times (H/16)\times (W/16)}}_{\text{latent}}
\xrightarrow{\ \text{patchify + project}\ }
\underbrace{\mathbb{R}^{B\times N_{\rm img}\times D}}_{\text{visual tokens}}
$$

MOSS-Image 的路径略有不同：FLUX.2 VAE 先做 8× 下采样，再把 $2\times2$ 个 latent cell 合成一个 DiT token，因此**有效步长仍然是 16 像素**：

$$
\text{effective stride}
=
8\times2
=
16
$$

两种实现最后都落在同一个公式 $N_{\rm img}=(H/16)(W/16)$ 上。

### 2.4 为什么 Native Resolution 不只是「支持更多尺寸」

Mage-Flow 的论点值得单独记住：

$$
\boxed{
\text{resolution diversity 本身可以成为 training signal}
}
$$

传统 bucket 训练把「尺寸」当成需要被消除的噪声；Native Resolution 把它当成一种需要被建模的条件。好处是：

- 单一 checkpoint 自然泛化到 flexible output sizes；
- 训练时见到的长宽比分布更接近真实图像分布；
- 不需要为极端比例手工设计 bucket。

::: warning 不要把「Native Resolution」理解成「原图直接喂进去」
它仍然会做缩放，只是缩放策略从「拉到固定方形」变成「保持长宽比 + 面积约束 + 对齐」。Mage-Flow 和 MOSS-Image 都会把图像限制在一个像素预算内，而不是无条件保留原始分辨率。
:::

---

## 3. Variable-Length Tokens：长度成为一条新的 batch 维度

### 3.1 单样本长度

对一个图文样本 $i$，其序列长度为：

$$
L_i
=
L_{\text{text},i}
+
N_{\text{img},i}
\quad
\left(
+
N_{\text{source},i}
\ \text{如果是 editing}
\right)
$$

其中：

- $L_{\text{text},i}$ 由 prompt / caption / instruction 决定；
- $N_{\text{img},i}$ 由图像面积与 tokenizer 步长决定；
- editing 任务还要加上 source image 的 token。

由于文本长度和图像面积都不固定，$L_i$ 天然是变长的：

$$
L_1\neq L_2\neq L_3\neq\cdots
$$

### 3.2 MOSS-Image 的具体数字

MOSS-Image 技术报告给出了一个非常有用的完整例子：

| 项目 | 数值 |
| :--- | ---: |
| 最终阶段文本上限 | 4096 tokens |
| $1024^2$ 图像 token 数 | 4096 |
| 满长文本 + $1024^2$ 图像 | $4096+4096=8192$ tokens |
| Multi-sample packing budget | 8576 tokens / pack |
| Static execution length | 8588 tokens |

这里有两个容易混淆的概念：

$$
\boxed{
\text{per-sample text cap}=4096
}
\qquad
\boxed{
\text{per-pack token budget}=8576
}
$$

前者限制**单条样本的文本长度**，后者限制**一个 pack 内所有样本的真实 token 总量**。两者不是同一个旋钮。

::: info 为什么 static execution length 比 budget 大？
8576 是真实 token 预算，8588 是实际执行的静态长度。多出来的部分通常用于对齐、padding 或 kernel / CUDA graph 的固定形状要求。**这也说明「token budget」和「实际执行长度」在工程上可以解耦**。
:::

### 3.3 Mage-Flow 的配置

Mage-Flow 的 Native Packing 同样使用 fixed token budget，把不同长度的 text / image 序列连续 pack 进同一个 batch。其精读笔记中记录了一个具体配置：

- 每个 GPU 一个 packed sample；
- 每个 packed sample 固定 50,000 tokens。

::: warning 这个 50K 是效率 benchmark 配置，不是所有训练阶段的固定值
Mage-Flow 论文 Table 4 的 training-efficiency benchmark 使用 50K tokens / packed sample。**不能据此推断正式训练的所有阶段都固定 50K**。引用时应保留这个限定。
:::

---

## 4. Packing：把 padding 换成真实 token

### 4.1 形式化定义

给定 $n$ 个样本，长度分别为 $L_1,\ldots,L_n$，pack 的 token 预算为 $T$。Packing 要把这些样本划分成若干个 pack $\mathcal{P}_1,\ldots,\mathcal{P}_K$，使得：

$$
\sum_{i\in\mathcal{P}_k}L_i\le T,
\qquad
k=1,\ldots,K
$$

并尽量减少 pack 的数量 $K$ 或总浪费。这是一个 **bin packing** 问题；一维 bin packing 是 NP-hard，实际训练中通常用启发式：

| 策略 | 做法 | 特点 |
| :--- | :--- | :--- |
| **First-Fit Decreasing (FFD)** | 按长度降序排序，依次放入第一个装得下的 pack | 简单、效果稳定 |
| **Best-Fit Decreasing (BFD)** | 按长度降序排序，放入剩余空间最小的 pack | 通常比 FFD 更紧凑 |
| **Random / Shuffle** | 先随机打乱再顺序装 | 实现最简单，但浪费可能更大 |

Packing 效率定义为：

$$
\eta_{\rm pack}
=
\frac{\sum_{i=1}^{n}L_i}{K T}
$$

$\eta_{\rm pack}=1$ 表示所有 pack 都被完全填满；最后一个 pack 通常填不满，所以实际效率略低于 1。

### 4.2 动态 batch size

Packing 带来一个非常重要的工程后果：**batch 中的样本数不再是固定的**。

$$
\sum_{i\in\mathcal{P}_k}\left(L_{\text{text},i}+N_{\text{img},i}\right)
\lesssim
T
$$

大图 token 多，一个 pack 装的样本就少；小图 token 少，一个 pack 装的样本就多。这样每次 forward / backward 的**真实 token 工作量**更接近，显存负载也更稳定。

对比一下：

$$
\text{固定 batch size：8 张图}
\quad\Longrightarrow\quad
\begin{cases}
8\ \text{张大图} & \Rightarrow\ \text{可能 OOM}\\
8\ \text{张小图} & \Rightarrow\ \text{GPU 吃不满}
\end{cases}
$$

$$
\text{固定 token budget：每个 pack 约 }T\text{ tokens}
\quad\Longrightarrow\quad
\text{每次 update 的工作量近似恒定}
$$

::: tip Packing 到底省了什么
$$
\text{Padding：}\ [100,300,600]\rightarrow[600,600,600]
$$
$$
\text{Packing：}\ [100,300,600]\rightarrow[100\,|\,300\,|\,600]
$$

前者为了 batching 把每个样本拉成一样长；后者只控制一个 pack 内真实 token 的总量。**Packing 优化的是内存布局与 batch 组成，不是单样本的 attention 复杂度。**
:::

### 4.3 Packing 不改变什么

这一点在面试里经常被追问：

$$
\boxed{
\text{Packing 消除的是 padding，不是 attention 的 }O(L^2)\text{ 成本}
}
$$

一个 $L_i$ token 的样本，即使被 pack 进一条长序列，它在自己的样本内部仍然要做 $L_i\times L_i$ 的 attention。Packing 不会把 $O(L_i^2)$ 变成 $O(L_i)$。真正降低高分辨率成本的是：

1. **Progressive Resolution**：先训低分辨率，减少高分辨率阶段的步数；
2. **更高效的 tokenizer**：降低 $N_{\rm img}$（例如 Mage-VAE）；
3. **更高效的 attention kernel**：降低常数因子与显存占用；
4. **更少的 padding**：Packing + VarLen Attention。

---

## 5. VarLen Attention：用 `cu_seqlens` 隔离样本

### 5.1 问题：packed sequence 会跨样本 attention

把多个样本拼成一条序列后，如果直接做标准 full attention：

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

那么样本 A 的 token 会 attend 到样本 B 的 token，造成**信息泄漏与训练信号污染**。这不是「效率问题」，而是「正确性问题」。

### 5.2 Block-diagonal mask 的等价写法

正确的 attention 应该只允许同一样本内部的 token 互相可见。定义 mask：

$$
M_{uv}
=
\begin{cases}
0, & u,v\ \text{属于同一个样本}\\[4pt]
-\infty, & \text{otherwise}
\end{cases}
$$

则：

$$
\tilde{A}
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)
$$

由于每一行至少包含自己所在样本的 token，softmax 不会出现整行 $-\infty$ 的数值问题。这个 mask 的形状是 $L\times L$，其中 $L=\sum_i L_i$；它的**语义**就是 block-diagonal。

### 5.3 `cu_seqlens`：用偏移量代替 mask

`cu_seqlens`（cumulative sequence lengths）是一个长度为 $B+1$ 的一维整数张量：

$$
\mathrm{cu\_seqlens}
=
\left[
0,\ L_1,\ L_1+L_2,\ \ldots,\ \sum_{i=1}^{B}L_i
\right]
\in\mathbb{Z}^{B+1}
$$

例如两个样本长度分别为 1000 和 2000：

```text
cu_seqlens = [0, 1000, 3000]
```

含义：

| 区间 | 样本 |
| :--- | :--- |
| `[0, 1000)` | Sample 1 |
| `[1000, 3000)` | Sample 2 |

注意 `cu_seqlens` 的长度是 **样本数 + 1**，最后一项等于 packed 序列的总长度。

### 5.4 FlashAttention variable-length kernel

FlashAttention 的 varlen 接口直接接收 `cu_seqlens`，在 kernel 内部按样本边界切分 attention，**不需要显式构造 $L\times L$ 的 block-diagonal mask**。

一个典型的调用形式（以 `flash-attn` 为例）：

```python
from flash_attn import flash_attn_varlen_func

# q, k, v: (total_tokens, num_heads, head_dim)
# cu_seqlens_q/k: (batch + 1,), dtype=torch.int32, device=cuda
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_len,
    max_seqlen_k=max_len,
    causal=False,          # 扩散模型通常是双向 attention
)
```

几个关键约定：

- `q`、`k`、`v` 的 shape 是 `(total_tokens, num_heads, head_dim)`，不是 `(batch, seq, heads, dim)`；
- `cu_seqlens` 必须是 `int32`，且在 GPU 上；
- `max_seqlen` 是该 batch 内最长的样本长度，用于 kernel 的 shared memory / tile 配置；
- `causal=False` 对应双向 attention；如果是自回归模型，需要按 `cu_seqlens` 正确设置 causal 语义；
- 所有样本必须按照 `cu_seqlens` 的顺序连续存放。

### 5.5 复杂度对比

| 方案 | Attention 计算量 | Mask / 索引内存 | 是否跨样本 |
| :--- | :--- | :--- | :--- |
| Padding + dense attention | $O\left(B L_{\max}^2 d_h\right)$ | $O\left(B L_{\max}^2\right)$ | 否 |
| Packed + full attention | $O\left(\left(\sum_i L_i\right)^2 d_h\right)$ | $O\left(\left(\sum_i L_i\right)^2\right)$ | **是（错误）** |
| Packed + 显式 block-diagonal mask | $O\left(\left(\sum_i L_i\right)^2 d_h\right)$ | $O\left(\left(\sum_i L_i\right)^2\right)$ | 否 |
| Packed + FlashAttention varlen | $O\left(\sum_i L_i^2 d_h\right)$ | $O\left(\sum_i L_i\right)$ | 否 |

最后一行是 Packing 真正发挥威力的地方：**它把跨样本的 $L_iL_j$ 计算块直接跳过，而不是算完再 mask 掉**。

### 5.6 位置编码必须按样本重置

Packing 之后，token 在物理上是一条连续序列。如果位置编码直接使用「packed 序列中的全局 index」，那么第二个样本的第一个 token 会继承第一个样本末尾的位置，导致：

- 图像的空间坐标错乱；
- 文本的位置偏移；
- 模型无法区分「新样本开始」。

正确的做法是**每个样本内部重新计算位置编码**：

| 模型 | 位置编码 |
| :--- | :--- |
| **Mage-Flow** | per-sample 2D RoPE $(h,w)$；editing 扩展为 3D RoPE $(h,w,f)$，$f$ 区分 source / target |
| **MOSS-Image** | 4D 坐标 $(r,t,w,h)$：reference index、序列位置、宽、高；部分维度做 RoPE |

::: danger 最常见的三个位置编码 bug
1. **全局 index 直接当 position id**：第二个样本的位置从第一个样本末尾继续；
2. **文本和图像共享同一套 1D 位置**：图像丢失 2D 空间结构；
3. **editing 的 source / target 没有用 $f$ 轴区分**：模型不知道哪个是输入、哪个是输出。
:::

### 5.7 一个纯 PyTorch 的参考实现

如果暂时不想引入 FlashAttention，可以用 block-diagonal mask 写一个**语义正确但效率较低**的参考实现，用来对齐数值：

```python
import torch
import torch.nn.functional as F

def varlen_attention_reference(q, k, v, cu_seqlens):
    """
    q, k, v: (total_tokens, num_heads, head_dim)
    cu_seqlens: (batch + 1,)
    """
    total_tokens = q.shape[0]
    # 构造 block-diagonal mask
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        device=q.device,
        dtype=q.dtype,
    )
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        mask[start:end, start:end] = 0.0

    # 转成 (num_heads, total_tokens, head_dim)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn = torch.softmax(q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5) + mask, dim=-1)
    out = attn @ v
    return out.transpose(0, 1)
```

这个实现的时间 / 显存都是 $O(L^2)$，只适合小规模数值验证；正式训练应使用 FlashAttention varlen 或其他支持变长序列的 kernel。

---

## 6. Progressive Resolution：先学全局，再学细节

### 6.1 为什么渐进分辨率有效

分辨率对成本的影响是**平方级**的：

$$
N_{\rm img}\propto H W,
\qquad
C_{\rm attn}\propto N_{\rm img}^2\propto (HW)^2
$$

也就是说，分辨率翻倍：

$$
H,W\rightarrow 2H,2W
\quad\Longrightarrow\quad
N_{\rm img}\rightarrow4N_{\rm img}
\quad\Longrightarrow\quad
C_{\rm attn}\rightarrow16C_{\rm attn}
$$

如果一开始就在最高分辨率上训练，绝大部分算力会花在「细节」上，而模型还没有学好全局语义、构图和文本对齐。渐进分辨率的思路是：

$$
\boxed{
\text{低分辨率阶段：用便宜的计算学全局语义与构图}
}
$$

$$
\boxed{
\text{高分辨率阶段：用较少的步数学细节、文字与质感}
}
$$

### 6.2 Mage-Flow 的 recipe

Mage-Flow 的 T2I 训练是典型的三段预训练 + 一段 SFT：

| 阶段 | 数据量 | 分辨率 | 目标 |
| :--- | ---: | :--- | :--- |
| Pre-train 1 | 1.2B | $256\times256$ | 低成本建立 broad visual–language alignment |
| Pre-train 2 | 600M | 512-pixel native aspect ratio | 保留长宽比，提升细节 |
| Pre-train 3 | 300M | 1024-pixel native aspect ratio | 细节、layout fidelity、文字渲染、美学 |
| SFT | 150M | 1024-pixel native aspect ratio | 产出 Mage-Flow-Base |

更重要的是，**数据过滤阈值与分辨率同步收紧**：

| 阶段 | Aesthetic-V2.5 | Watermark |
| :--- | ---: | ---: |
| Pre-train 1 ($256^2$) | $\ge4.5$ | $<0.5$ |
| Pre-train 2 ($512$ native) | $\ge5.5$ | $<0.3$ |
| Pre-train 3 ($1024$ native) | $\ge6.0$ | $<0.1$ |
| SFT ($1024$ native) | $\ge6.5$ | $<0.05$ |

$$
\boxed{
\text{分辨率}\uparrow
\quad+\quad
\text{数据质量}\uparrow
\quad+\quad
\text{concept-aware reweighting}\uparrow
}
$$

这三件事是**一起**发生的，不是只把分辨率调高。

### 6.3 MOSS-Image 的 recipe

MOSS-Image 技术报告 Table 1 的训练阶段如下：

| Stage | Resolution | Text-token cap | Training steps |
| :---: | :--- | ---: | ---: |
| 1 | 256-pixel level | 320 | 2,100,000 |
| 2 | 512, then 256/512 | 512 → 1024 → 2048 | 2,100,000 |
| 3 | Mixed, area $\le1024^2$ | 4096 | 370,000 |
| 4 | Mixed, up to $\sim2\times1024^2$ | 4096 | 520,000 |

报告正文对最终阶段的描述是：

> the final stage preserves aspect ratio, aligns image dimensions to multiples of 16, and caps target image area at $1024^2$ pixels.

按正文「三阶段」的口径，**Stage 3 就是这里的 final stage**：保留长宽比、面积上限 $1024^2$、16 对齐，并使用 multi-sample packing；Table 1 的 Stage 4 则是进一步放宽到约 $2\times1024^2$ 的扩展阶段。

::: warning 论文口径需要注意：正文说「three stages」，Table 1 却列了四行
MOSS-Image 正文写的是 *"Training increases image resolution and text-context capacity across three stages (Table 1)"*，但 Table 1 实际列出 Stage 1–4。**本文按「三阶段 = Stage 1–3，Stage 4 为扩展」来解读**；引用时不要把「三阶段」和「四行」混为一谈，也不要把 Stage 3 与 Stage 4 的配置互相替代。
:::

### 6.4 渐进分辨率的真正含义

Progressive Resolution 不是「把分辨率从 256 调到 1024」这么简单。它通常同时调整：

$$
\boxed{
\begin{aligned}
&\text{image resolution}\uparrow\\
&\text{text length / context cap}\uparrow\\
&\text{data quality threshold}\uparrow\\
&\text{aspect-ratio diversity}\uparrow\\
&\text{capability-specific sampling}\uparrow
\end{aligned}
}
$$

MOSS-Image 的 text cap 从 320 → 2048 → 4096，就是「文本上下文容量」也在渐进增长；Mage-Flow 的 aesthetic / watermark 阈值逐阶段收紧，就是「数据质量」在渐进增长。

::: tip 一个可以直接复用的判断
如果低分辨率阶段的基础语义仍在快速提升，不要过早切到高分辨率；如果基础语义已经趋于平台，而高分辨率细节、文字、质感明显落后，就应该进入下一阶段。**切换信号应该看 capability validation，而不是只看 training loss。**
:::

---

## 7. 核心伪代码

下面的代码是**原理性伪代码**，不是 Mage-Flow 或 MOSS-Image 的原始实现。它把前面四个机制串成一个完整的训练循环。

```python
# 原理性伪代码：省略分布式、混合精度、checkpoint 等工程细节
for resolution in [256, 512, 1024]:
    max_area = resolution * resolution

    for micro_batch in data_loader:
        samples = []

        for image, caption in micro_batch:
            # 1. Native Resolution：保持长宽比，限制面积，对齐到 16
            image = resize_keep_ratio(
                image,
                max_area=max_area,
                align=16,
            )

            # 2. 编码成变长 token 序列
            latent = vae.encode(image)              # (C, H/16, W/16)
            img_tokens = patchify_and_project(latent)   # (N_img, D)
            txt_tokens = text_encoder(caption)          # (N_txt, D)

            # 文本在前、图像在后；editing 还要拼 source tokens
            tokens = torch.cat([txt_tokens, img_tokens], dim=0)  # (L_i, D)
            samples.append(tokens)

        # 3. Packing：按 token budget 装包
        packs = pack_by_token_budget(
            samples,
            max_tokens=8576,
            strategy="best_fit_decreasing",
        )

        for pack in packs:
            lengths = [x.shape[0] for x in pack]

            # cu_seqlens 长度 = 样本数 + 1
            cu_seqlens = [0] + torch.cumsum(
                torch.tensor(lengths), dim=0
            ).tolist()
            max_seqlen = max(lengths)

            # 4. 拼接成一条 packed 序列
            tokens = torch.cat(pack, dim=0)          # (sum L_i, D)

            # 5. 每条样本独立采样 noise 与 timestep
            t = sample_timesteps(len(pack))          # (B_pack,)
            eps = torch.randn_like(tokens)           # (sum L_i, D)
            # 把 per-sample 的 t 展开成 per-token，长度与 packed 序列对齐
            t_per_token = repeat_by_lengths(t, lengths)   # (sum L_i, 1)

            # 只有 target image token 加噪；text 与 editing source 保持 clean
            img_mask = token_is_target_image(pack)   # (sum L_i, 1), bool
            x_t = torch.where(
                img_mask,
                (1 - t_per_token) * tokens + t_per_token * eps,
                tokens,
            )
            v_target = torch.where(
                img_mask,
                eps - tokens,
                torch.zeros_like(tokens),
            )

            # 6. 位置编码按样本重置：text 1D / image 2D (h, w)
            pos = build_per_sample_positions(pack)

            # 7. VarLen Attention：样本之间互不干扰
            v_pred = dit(
                x_t,
                t,
                pos,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attention="flash_attn_varlen",
            )

            # 8. 只对目标图像 token 计算 loss
            #    text token、padding、editing 的 source token 全部 mask 掉
            loss = flow_matching_loss(
                v_pred,
                v_target,
                mask=target_image_token_mask(pack),
                reduction="per_token",   # 或 "per_sample"，见 §8
            )

            (loss / gradient_accumulation_steps).backward()

        optimizer.step()
        optimizer.zero_grad()
```

::: info 伪代码里最容易漏掉的五件事
1. **`t` 和 `eps` 必须按样本独立采样**，不能整个 pack 共享一个 timestep；
2. **只有 target image token 加噪**，text token 和 editing 的 source token 是 clean condition；
3. **loss mask 必须排除 text token、padding token，以及 editing 的 source token**；
4. **位置编码必须按样本重置**，不能用 packed 序列的全局 index；
5. **loss 归一化方式要在训练前确定**：per-sample 还是 per-token，会改变大图 / 小图的相对权重。
:::

---

## 8. 工程细节与避坑清单

| # | 细节 | 正确做法 | 错误后果 |
| :-: | :--- | :--- | :--- |
| 1 | **位置编码** | 每个样本内部重置；图像用 2D / 3D RoPE | 第二个样本继承前一个样本的位置，空间结构错乱 |
| 2 | **独立噪声与 timestep** | 每个样本独立采样 $t_i,\epsilon_i$ | 整个 pack 共享一个 $t$，不同样本的监督信号被绑定 |
| 3 | **Loss mask** | 只对目标图像 token 计算；text / padding / source 全部 mask | 文本 token 参与生成损失，editing 的 source 被错误监督 |
| 4 | **Loss 归一化** | 明确 per-sample 或 per-token；与 batch 组成保持一致 | 大图支配梯度，或小图被过度加权 |
| 5 | **`cu_seqlens` 类型** | `int32`，GPU，长度 $B+1$ | kernel 报错或静默截断 |
| 6 | **`max_seqlen`** | 取当前 pack 的真实最大长度 | shared memory 配置错误，性能下降或越界 |
| 7 | **文本 padding / truncation** | 右 padding + attention mask；截断到阶段上限 | padding token 被当成有效文本 |
| 8 | **Packing 顺序** | 先 shuffle 再 pack；固定随机种子 | 同一数据源聚集，batch 分布有偏 |
| 9 | **Static execution length** | 与 token budget 解耦；为 CUDA graph 留对齐空间 | 动态 shape 导致频繁重编译 / 性能抖动 |
| 10 | **梯度累积** | loss 按 token 数或样本数正确加权 | 不同 accumulation step 的有效权重不一致 |
| 11 | **分布式训练** | 各 rank 的 local pack 大小可以不同，但 loss 归一化要全局一致 | 各 rank 梯度尺度不一致 |
| 12 | **Attention backend** | 扩散模型用 `causal=False`；确认 varlen kernel 支持当前 head dim | 错误的 causal mask 破坏双向建模 |
| 13 | **数值精度** | attention softmax 用 fp32 / 稳定实现；RoPE 注意 dtype | bf16 下出现 NaN 或长序列不稳定 |
| 14 | **评测** | 按长宽比分桶评测；检查极端比例与高分辨率文字 | 平均指标掩盖某些比例的系统性退化 |

::: warning 关于 loss 归一化：per-sample 还是 per-token，是一个真正的设计选择
- **Per-sample**：每个样本对 loss 的贡献相同，小图不会被大图淹没；
- **Per-token**：每个有效图像 token 的贡献相同，大图按 token 数比例贡献更多梯度。

MOSS-Image 技术报告明确采用 **per-token normalization**：不同分辨率的图像按有效图像 token 数成比例贡献；文本 token 不参与 loss；没有额外的 $\sigma$/SNR 加权。报告还提到 Stage 2 从 per-sample 过渡到 per-token，说明这个选择会随 packed training 的引入而变化。

**训练前必须明确选哪一种，并让数据分布、batch 组成和评估口径与之一致。**
:::

---

## 9. Mage-Flow vs MOSS-Image：各自披露了什么

| 维度 | Mage-Flow | MOSS-Image |
| :--- | :--- | :--- |
| Backbone | 4B Native-Resolution MMDiT | 5B single-stream DiT，42 blocks |
| VAE | Mage-VAE，16× 空间下采样，128 channels | FLUX.2 VAE，8× 下采样 + $2\times2$ patch，有效步长 16 |
| Native Resolution | 保留长宽比，支持 $512\sim2048$ 边长，包括 $512\times2048$ | 最终阶段保留长宽比，16 对齐，面积上限 $1024^2$ |
| Token 公式 | $N_{\rm img}=(H/16)(W/16)$ | $N_{\rm img}=(H/16)(W/16)$ |
| Packing | Native Packing，fixed token budget；效率 benchmark 中 50K tokens / packed sample | Multi-sample packing，8576 tokens / pack，static execution length 8588 |
| VarLen Attention | 明确使用 FlashAttention variable-length kernel + per-sample cumulative offsets | 使用 attention mask 处理 variable-length packing；**未明确披露底层 varlen kernel** |
| Position Encoding | per-sample 2D RoPE $(h,w)$；editing 3D RoPE $(h,w,f)$ | 4D 坐标 $(r,t,w,h)$，部分维度 RoPE |
| Progressive Resolution | $256^2\rightarrow512$ native $\rightarrow1024$ native $\rightarrow$ SFT | Stage 1 $256$ → Stage 2 $512$ / mixed → Stage 3 area $\le1024^2$ → Stage 4 $\sim2\times1024^2$ |
| Loss Normalization | 本次核对范围内未展开逐 token 归一化公式 | 明确 per-token normalization；text token 不参与；无 SNR 加权 |
| CFG Packing | cond / uncond 单次 packed forward | 未在本次核对中确认 |

::: warning 引用边界：不要把两篇论文的细节互相串用
1. **MOSS-Image 的 varlen kernel 未披露**：报告只提到 attention mask 用于 variable-length packing，没有说使用 FlashAttention varlen、`cu_seqlens` 或 block-diagonal kernel。**不能把 Mage-Flow 的 FlashAttention varlen 实现直接写成 MOSS-Image 的实现。**
2. **MOSS-Image 的 Stage 3 / Stage 4 口径**：正文说 three stages，Table 1 列四行；引用时保留原文矛盾，不要自行合并。
3. **Mage-Flow 的 50K token 配置**：来自 Table 4 的 efficiency benchmark，不能当作所有训练阶段的固定配置。
4. **Token 数例子**：$1024^2\rightarrow4096$ 是 MOSS-Image 报告明确给出的；$1280\times720\rightarrow3600$ 是按 $N=(H/16)(W/16)$ 推导出来的，属于公式的直接应用，不是报告原文逐字列出。
5. **Native Resolution 的思想来源**：Mage-Flow 明确引用 NiT [10]，不是它首次提出。
:::

---

## 10. 面试速答：五个问题

::: tip Q1：如何高效训练不同分辨率的图像？
**四个机制分层回答：**

1. **Native Resolution**：保持长宽比，限制像素面积，对齐到 16；
2. **Variable-Length Tokens**：$N_{\rm img}=(H/16)(W/16)$，文本长度也可变；
3. **Packing + VarLen Attention**：在固定 token budget 下把不同长度样本拼成一条序列，用 `cu_seqlens` 保证样本间不互相 attention；
4. **Progressive Resolution**：先 256 / 512 学全局，再 1024 学细节，同时收紧数据质量与文本长度。
:::

::: tip Q2：Packing 和 VarLen Attention 分别解决什么？
- **Packing 是内存布局**：把 padding 换成真实 token，提高 token 利用率，让动态 batch 的显存更稳定；
- **VarLen Attention 是计算语义**：告诉 attention「哪些 token 属于同一样本」，避免 packed 序列跨样本泄漏。

**没有 VarLen Attention，Packing 是不安全的；没有 Packing，VarLen Attention 的收益也有限。**
:::

::: tip Q3：Packing 能提速，但它不改变什么？
它不改变单样本 attention 的 $O(L_i^2)$ 复杂度。Packing 消除的是 padding 带来的额外 slot 和跨样本计算；真正降低高分辨率成本还要靠 tokenizer、progressive resolution 和更高效的 attention kernel。
:::

::: tip Q4：为什么渐进分辨率有效？
因为 $N_{\rm img}\propto HW$，而 attention 成本 $\propto N_{\rm img}^2$。分辨率翻倍会让单图 attention 成本变成约 16 倍。低分辨率阶段用便宜的计算学全局语义与构图，高分辨率阶段只花较少步数学细节和文字。
:::

::: tip Q5：最容易踩的坑是什么？
四个高频 bug：

1. 位置编码没有按样本重置；
2. 整个 pack 共享一个 timestep / noise；
3. text token 或 padding token 没有 mask；
4. loss 归一化在 per-sample 与 per-token 之间摇摆，导致大图 / 小图权重不一致。
:::

---

## 11. 参考与延伸

- **Mage-Flow**：arXiv:2607.19064 · [论文](https://arxiv.org/abs/2607.19064) · [本库精读笔记](./mage-flow.md)
- **MOSS-Image**：[Technical Report](https://github.com/OpenMOSS/MOSS-Image/blob/main/paper/MOSS_Image_Technical_Report.pdf) · [GitHub](https://github.com/OpenMOSS/MOSS-Image) · [Project Page](https://openmoss.ai/MOSS-Image/)
- **NiT (Native-Resolution Image Synthesis)**：arXiv:2506.03131 · [ar5iv](https://ar5iv.labs.arxiv.org/html/2506.03131)
- **FlashAttention**：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **本库相关笔记**：[图像基模训练 Playbook](./training-playbook.md) · [Mage-Flow 精读](./mage-flow.md)

---

::: info 维护备注
这篇笔记是「跨模型技术专题」，与 [Mage-Flow 精读](./mage-flow.md) §6 的分工是：

- **Mage-Flow §6**：以 Mage-Flow 论文为主线，逐段精读其 Native Packing、FlashAttention varlen、per-sample 2D RoPE、CFG packing 与系统优化；
- **本篇**：把 Mage-Flow 与 MOSS-Image 放在一起，抽取可复用的四个机制、公式、伪代码与工程避坑清单，便于面试速答与跨项目迁移。

两者互为索引，不重复替代。
:::
