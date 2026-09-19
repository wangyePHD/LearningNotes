# Softmax 注意力 vs 线性注意力：联想记忆容量极限与结合律数学本质对比

> **标签**：`Attention` `Linear Attention` `Math Derivation` `Hopfield Network` `Complexity`  
> **更新时间**：2026-09-19  
> **数学工具**：线性代数（结合律与矩阵秩）、核方法（Kernel Trick）、连续联想记忆理论（Modern Hopfield Networks）

---

## 0. 先补基石：究竟什么是 Softmax？（标量与注意力展开）

在理解结合律为什么失效之前，我们必须先彻底看清 $\text{Softmax}$ 的每一行标量公式。

### ① 基础定义：从任意实数到概率分布

给定一个包含 $K$ 个任意实数的输入向量 $z = [z_1, z_2, \dots, z_K]^T \in \mathbb{R}^K$（例如模型打出的相关度原始分数）：
$\text{Softmax}$ 函数对第 $i$ 个元素的定义为：

$$
\text{Softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}
$$

这个公式只做了两件极其纯粹的事：
1. **分子取指数 $\exp(z_i) = e^{z_i}$**：
   - 保证所有输出恒为正数（$>0$）；
   - **极度拉大差距（强者愈强）**：比如 $z_1 = 3, z_2 = 1$。取指数后 $e^3 \approx 20.08$，$e^1 \approx 2.72$，差距瞬间从 3 倍放大到了 7.4 倍！
2. **分母除以总和 $\sum_{j=1}^K \exp(z_j)$**：
   - 归一化，使得所有元素的和**恒等于 1**（$\sum_i \text{Softmax}(z)_i = 1$），成为一个合法的概率/权重分布。

---

### ② Softmax 在 Transformer 注意力中的“逐个 Token”展开式

当我们写矩阵形式 $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$ 时，它内部真实的标量运算如下：

假设我们要计算**第 $i$ 个 Token（Query 向量 $q_i$）** 对全视频所有 Token 的注意力：

1. **计算打分（点积相似度）**：
   第 $i$ 个 Token 和第 $j$ 个 Token 的相似度分数为：
   $$
   a_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} = \frac{1}{\sqrt{d}} \sum_{c=1}^d q_{i, c} k_{j, c}
   $$

2. **经过 Softmax 计算出归一化权重 $\alpha_{ij}$**：
   $$
   \alpha_{ij} = \frac{\exp(a_{ij})}{\sum_{m=1}^N \exp(a_{im})} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)}{\mathbf{\sum_{m=1}^N \exp\left(\frac{q_i \cdot k_m}{\sqrt{d}}\right)}}
   $$

3. **对 Value 向量 $v_j$ 加权求和，得到最终输出 $\text{Output}_i$**：
   $$
   \text{Output}_i = \sum_{j=1}^N \alpha_{ij} v_j = \sum_{j=1}^N \left( \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)}{\mathbf{\sum_{m=1}^N \exp\left(\frac{q_i \cdot k_m}{\sqrt{d}}\right)}} \right) v_j
   $$

---

### ③ 核心死结：为什么 Softmax 会彻底粉碎“结合律”？

这是很多算法工程师最容易产生盲区的地方。我们从**初等代数直觉**和**求和因式分解**两个层面彻底讲透。

#### 层级 1：代数直觉 —— 中间只要夹了非线性函数，结合律必然崩塌

在初中数学中，纯粹的乘法是满足结合律的：
$$
(a \times b) \times c = a \times (b \times c)
$$
但是，**如果中间套了一个非线性函数 $f(\cdot)$，结合律就会瞬间被粉碎**！
例如，令 $f(x) = x^2$（平方运算），我们看看括号还能不能随便换位置：
- 左边先算：$f(a \times b) \times c = (a \times b)^2 \times c$
- 右边先算：$a \times f(b \times c) = a \times (b \times c)^2$

代入具体数字测试（设 $a=2, b=3, c=4$）：
- 左边：$(2 \times 3)^2 \times 4 = 6^2 \times 4 = 36 \times 4 = \mathbf{144}$
- 右边：$2 \times (3 \times 4)^2 = 2 \times 12^2 = 2 \times 144 = \mathbf{288}$
- 结果：$\mathbf{144 \neq 288}$！两者根本不相等！

在标准注意力中，$\text{Softmax}$ 本质上就是那个非线性函数 $f(\cdot)$：
$$
\text{Attention} = \mathbf{f(Q \times K^T)} \times V \quad \mathbf{\neq} \quad Q \times \mathbf{f(K^T \times V)}
$$
**只要 $Q$ 和 $K^T$ 乘完之后被包裹在非线性函数 $f$ 里面，你就绝对不能伸手穿过这个函数，去擅自先乘后面的 $V$！**

---

#### 层级 2：因式分解视角 —— 能否把 $q_i$ 提公因子提出来？

我们把每一行 Token 的计算展开对比，看看为什么“没有 Softmax 可以提公因子”，而“有了 Softmax 绝对提不出来”：

##### 情况 A：如果没有 Softmax（纯线性乘法加法）
第 $i$ 个 Token 的输出是：
$$
\text{Output}_i = \sum_{j=1}^N (q_i \cdot k_j) v_j
$$
因为这里只有普通的加法和乘法，完全满足初中代数的**乘法分配律**。注意：求和符号 $\sum$ 是对指标 $j$ 求和，与 $q_i$ 无关！  
我们可以直接把公因子 $q_i$ **原封不动地提到求和符号外面**：
$$
\text{Output}_i = \mathbf{q_i} \times \left( \sum_{j=1}^N k_j v_j \right)
$$
::: tip 惊人的发现
看右边的括号：$\left( \sum_{j=1}^N k_j v_j \right)$！  
**这个括号里面，已经完全没有 $q_i$ 了！**  
这意味着：显卡根本不需要等待具体的 Query 是什么，它可以完全脱离 $q_i$，提前在后台把所有的 $k_j$ 和 $v_j$ 乘起来累加存好（这就是那块 $d \times d$ 的小海绵）！
:::

##### 情况 B：加上了 Softmax（真实 Transformer）
第 $i$ 个 Token 的输出被换成了：
$$
\text{Output}_i = \sum_{j=1}^N \frac{\exp(q_i \cdot k_j)}{\mathbf{\sum_{m=1}^N \exp(q_i \cdot k_m)}} v_j
$$
现在请你尝试把 $q_i$ 提到求和符号外面：
1. **分子的死锁**：$q_i$ 被扣在自然指数的幂上 $\exp(q_i \cdot k_j) = e^{q_i \cdot k_j}$，根据初等数学公式 $e^{ab} \ne a \cdot e^b$，**$q_i$ 无法从指数里拆出来**；
2. **分母的死锁**：分母本身就是一个关于 $m$ 的庞大累加和 $\sum_{m=1}^N \exp(q_i \cdot k_m)$，**每一个不同的 $q_i$，分母算出来的数值完全不同**！
3. 分子和分母同时被 $q_i$ 死死锁定，**在数学上不存在任何代数变形，能够把 $q_i$ 独立剥离出来**！

---

#### 总结结论：数学因果死结

$$
\text{必须计算 Softmax 分母} \implies \text{必须提前知道当前 } q_i \text{ 与所有 } k_m \text{ 的点积} \implies \mathbf{被迫算出整个\ N \times N\ 矩阵}
$$

这就是为什么我们说：**是 Softmax 这个函数本身的非线性指数和归一化分母，强行切断了结合律的数学通路。**

---

## 1. 核心矛盾：结合律的“奇迹”与“诅咒”

在标准 Transformer 的自注意力机制中，计算瓶颈源于一个看似简单的非线性函数 $\text{Softmax}(\cdot)$：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

- $Q, K, V \in \mathbb{R}^{N \times d}$，其中 $N$ 为序列长度（如视频中的 35 万个 Token），$d$ 为隐层通道维度（如 64 或 128）。
- 由于 $\text{Softmax}$ 的存在，矩阵乘法**无法使用结合律**：我们必须先算出 $N \times N$ 的点积矩阵，再与 $V$ 相乘，导致了死穴般的 $\mathcal{O}(N^2)$ 计算与显存开销。

### 线性注意力的“结合律奇迹”
为了消灭 $N^2$，线性注意力（如 Linear Transformer、Mamba、GDN、FLA）引入核映射 $\phi(\cdot)$ 去掉 Softmax：

$$
\text{Attention}_{\text{Linear}}(Q, K, V) = \phi(Q) \big(\phi(K)^T V\big)
$$

由于去掉了包裹在中间的非线性 Softmax，矩阵乘法可以使用**结合律**从左向右或从右向左移动：

$$
\underbrace{(\phi(Q) \phi(K)^T)}_{\text{传统先算前两项：} N \times N} V \quad \xrightarrow{\mathbf{结合律移位}} \quad \phi(Q) \underbrace{\big(\phi(K)^T V\big)}_{\mathbf{现在先算后两项：} d \times d}
$$

---

### 1.1 维度与计算量逐层拆解对比（为什么换个顺序就能省千倍算力？）

很多同学第一次看结合律推导都会困惑：“明明还是这三个矩阵，为什么仅仅换个相乘顺序，计算量就会天差地别？”  
我们用最严谨的高等代数矩阵乘法规则，一步一步跟踪**每一个矩阵在每一步前后的维度**。

::: tip 线性代数基础复习：矩阵乘法的计算量公式
若矩阵 $A \in \mathbb{R}^{m \times k}$，矩阵 $B \in \mathbb{R}^{k \times n}$，则乘积矩阵 $C = AB \in \mathbb{R}^{m \times n}$。  
其所需的标量浮点乘法计算次数精确为：
$$
\text{FLOPs} = m \times k \times n
$$
:::

#### 路线 A：传统先算前两项 $(Q K^T) V$

设序列长度 $N = 100,000$（十万个 Token），特征通道 $d = 64$。
输入维度：
- $Q \in \mathbb{R}^{N \times d} \quad (100,000 \times 64)$
- $K \in \mathbb{R}^{N \times d} \implies K^T \in \mathbb{R}^{d \times N} \quad (64 \times 100,000)$
- $V \in \mathbb{R}^{N \times d} \quad (100,000 \times 64)$

```text
【第 1 步】：算 A = Q × K^T
  维度变换：(N × d) × (d × N)  ──>  生成注意力打分大矩阵 A ∈ R^{N × N}
  真实数值：(100,000 × 64) × (64 × 100,000)  ──>  (100,000 × 100,000)
  计算次数：N × d × N = N^2 · d = 100,000 × 64 × 100,000 = 6,400 亿次！

【第 2 步】：算 Output = A × V
  维度变换：(N × N) × (N × d)  ──>  最终输出矩阵 Output ∈ R^{N × d}
  真实数值：(100,000 × 100,000) × (100,000 × 64)  ──>  (100,000 × 64)
  计算次数：N × N × d = N^2 · d = 6,400 亿次！

【路线 A 累计计算量】：
  总计算量 = N^2 · d + N^2 · d = 2 · N^2 · d = 12,800 亿次浮点运算！
```

#### 路线 B：线性注意力先算后两项 $Q (K^T V)$

同样的三个矩阵，同样的输入维度！我们利用结合律，**优先让 $K^T$ 和 $V$ 结合**：

```text
【第 1 步】：先算记忆状态 S = K^T × V
  维度变换：(d × N) × (N × d)  ──>  生成记忆矩阵 S ∈ R^{d × d}  (注意：N 在内侧被压缩抵消了！)
  真实数值：(64 × 100,000) × (100,000 × 64)  ──>  (64 × 64)
  计算次数：d × N × d = N · d^2 = 100,000 × 64 × 64 = 4.1 亿次！

【第 2 步】：再算 Output = Q × S
  维度变换：(N × d) × (d × d)  ──>  最终输出矩阵 Output ∈ R^{N × d}
  真实数值：(100,000 × 64) × (64 × 64)  ──>  (100,000 × 64)
  计算次数：N × d × d = N · d^2 = 100,000 × 64 × 64 = 4.1 亿次！

【路线 B 累计计算量】：
  总计算量 = N · d^2 + N · d^2 = 2 · N · d^2 = 8.2 亿次浮点运算！
```

---

### 1.2 两条路线真实数值暴击对比

最终得到的输出结果维度**完全一模一样**（都是 $N \times d = 100,000 \times 64$），但计算量对比：

| 运算顺序 | 中间生成的过渡矩阵尺寸 | 计算复杂度公式 | $N=100,000, d=64$ 时运算次数 | 速度倍率 |
| :--- | :--- | :--- | :--- | :--- |
| **路线 A：$(Q K^T) V$** | **$100,000 \times 100,000$ (百亿元素巨阵)** | $\mathcal{O}(N^2 \cdot d)$ | **12,800 亿次** | $1 \times$ (基准) |
| **路线 B：$Q (K^T V)$** | **$64 \times 64$ (仅 4096 个元素的小矩阵)** | $\mathcal{O}(N \cdot d^2)$ | **8.2 亿次** | **快了 1,562 倍！** ⚡ |

::: tip 为什么能快 1500 多倍的数学直觉
对比两者的比值：
$$
\frac{\text{路线 A}}{\text{路线 B}} = \frac{2 N^2 d}{2 N d^2} = \mathbf{\frac{N}{d}}
$$
- 只要序列长度 $N$ 远大于通道数 $d$（在视频中 $N \approx 100,000$ 甚至 35 万，而 $d$ 只有 64），加速比就是 $\frac{N}{d}$！
- **几何本质**：
  - 路线 A 产生了一个 $N \times N$ 的巨大中间平面，把显卡拖死在巨大平面的搬运上；
  - 路线 B 在内侧直接让 $N$ 相互求和抵消，中间只留下一个极其小巧的 $d \times d$ 小矩阵（小海绵），因此计算量被瞬间压缩了三个数量级！
:::

---

## 2. 为什么说线性注意力的容量是“有限”的？（几何与秩的瓶颈）

天下没有免费的午餐。带来 $\mathcal{O}(N)$ 线性速度的代价，就是**容量的毁灭性降维压缩**。

### ① 矩阵维度的物理现实
我们对比两种注意力最终在内部维护的“记忆载体”：

```text
【Softmax 注意力】
 维护未压缩的完整样本库：K ∈ R^{N × d}, V ∈ R^{N × d}
 实际记忆容量随序列长度 N 无限动态增长！
 N = 1,000      ──> 记忆规模包含 1,000 个独立特征
 N = 350,000    ──> 记忆规模包含 350,000 个独立特征 (无损存储)

【线性注意力】
 无论序列长度 N 有多长（哪怕 N 达到 100 万），
 所有历史 Token 全被累加压缩进单个状态矩阵：
 S = \sum_{j=1}^N \phi(k_j)^T v_j  ∈  R^{d × d}
 记忆矩阵尺寸恒定为 d × d，与 N 完全无关！
```

### ② 矩阵秩的硬上限 (Rank Bottleneck)
矩阵 $S = \phi(K)^T V \in \mathbb{R}^{d \times d}$ 的代数秩满足：
$$
\text{rank}(S) \le \min\big(\text{rank}(\phi(K)), \, \text{rank}(V), \, d\big) \le d
$$

- 无论序列中有几万还是几十万个 Token，矩阵 $S$ 中最多只能保留 **$d$ 个线性无关的特征方向**！
- 如果 $d = 64$ 或 $128$，当序列增长到 30 万个 Token 时（例如一段高清长视频），这 30 万个不同局部画面的向量被强行投影并累加在这仅仅 64 个基底方向上。
- **必然结果**：后来的向量不断冲刷、覆盖、污染旧的特征，向量之间不可避免地发生正交破坏与特征湮灭，导致**灾难性遗忘（Catastrophic Forgetting）**。

---

## 3. 联想记忆理论证明：经典 Hopfield vs 现代连续 Hopfield

从计算神经科学与联想记忆网络（Associative Memory）的角度，可以更深刻地证明二者在记忆容量上的**指数级代数鸿沟**。

### ① 线性注意力 = 经典离散 Hopfield 网络（容量线性）
在线性注意力中，记忆更新规则为赫布理论（Hebb's Rule）的外积累加：
$$
S = \sum_{i=1}^N v_i k_i^T
$$
根据 Hopfield（1982）与 Amit 等人的经典存储容量定理：
$$
C_{\text{Linear}} \approx \alpha_c \cdot d \approx \mathbf{0.14 \cdot d}
$$
- 若隐层通道 $d = 128$，一个线性记忆单元**最多只能精准检索和重构约 18 个不同的模式**！
- 超过这个数量后，伪吸引子（Spurious States）大量出现，读出的 Value 会变成全部历史的“混乱混合物（Blurry Soup）”。

### ② Softmax 注意力 = 现代连续 Hopfield 网络（容量指数级）
Krotov & Hopfield (2016) 以及 Ramsauer 等人 (2020) 在著名论文 *《Hopfield Networks is All You Need》* 中证明：
标准 Transformer 的 Softmax 注意力，本质上是一个**现代连续状态 Hopfield 能量模型**：

$$
E(z) = -\text{lse}\left(\beta, \, K z\right) + \frac{1}{2} \|z\|^2
$$
其中 $\text{lse}(\beta, x) = \frac{1}{\beta} \log \sum_j \exp(\beta x_j)$。  
通过引入指数非线性（Softmax），其极限联想记忆容量为：
$$
C_{\text{Softmax}} \propto \mathbf{\exp(c \cdot d)} \quad (\text{其中 } c > 0)
$$
- 记忆容量关于维度 $d$ 是**指数级爆炸增长**的！
- 即使 $d$ 很小，Softmax 也能在超球面上通过尖锐的非线性峰值，在上百万个模式中毫厘不差地精确定位目标 Token，互不干扰。

---

## 4. 全方位对比速查矩阵

| 对比维度 | 传统 Softmax 注意力 | 纯线性注意力 (Linear Attention) |
| :--- | :--- | :--- |
| **计算复杂度** | $\mathcal{O}(N^2)$（二次方爆炸） | $\mathcal{O}(N)$（严格线性，极速） |
| **显存占用 (KV)** | 随序列长度 $N$ 线性膨胀 $\mathcal{O}(N \cdot d)$ | 恒定尺寸 $\mathcal{O}(d^2)$，与序列长度无关 |
| **结合律适用性** | ❌ 无法使用（被非线性 Softmax 阻隔） | ✅ 完美适用：$(QK^T)V = Q(K^T V)$ |
| **记忆存储载体** | 未压缩的原生 Token 库 ($N \times d$) | 固定尺寸的状态转移矩阵 ($S \in \mathbb{R}^{d \times d}$) |
| **最大记忆容量** | 指数级：$C \propto \exp(d)$ | 严格受限线性：$C \le d$ |
| **检索行为表现** | **尖锐非线性检索**（像哈希查找，极精准） | **加权平均低通滤波**（像海绵吸水，易模糊） |
| **视频生成的长处** | 相邻帧高频纹理、人脸五官、物理一致性极佳 | 远距离背景漫游、超长序列推理速度极快 |
| **视频生成的死穴** | 序列长时显卡直接 OOM，推理慢如蜗牛 | 细节退化、人脸五官畸变、长时主体身份漂移 |

---

## 5. 结论：为什么“混合注意力 (Hybrid)”是视频生成的唯一正道？

通过上述数学与容量证明，我们可以得出两个不可动摇的底层结论：

1. **不能单用 Softmax**：因为视频时空 Token 动辄几十万，$N^2$ 的算力开销直接判了实时交互死刑；
2. **不能单用线性注意力**：因为固定 $d \times d$ 的有限容量在处理几十万帧画面时，无法支撑高频细节与主体保真度；
3. **唯一的物理与工程解 —— 混合注意力 (Hybrid Attention)**：
   - 用 **局部 Softmax** 在小窗口内发挥其指数级精准联想能力，锁死毛发、人脸与帧间物理连续性；
   - 用 **线性注意力** 在滑窗之外发挥其 $\mathcal{O}(N)$ 极速吞吐能力，以低频备忘录的形式传递全局背景与文字先验。
