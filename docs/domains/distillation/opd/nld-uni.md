---
type: paper
tags: ['Distillation', 'Diffusion', 'Unified Model', 'Multimodal', 'Image Editing', 'Reasoning']
status: reading
summary: '用 cross-paradigm OPD 把 AR-VLM 的理解能力和连续流模型的编辑能力蒸馏进统一离散扩散模型：AR Teacher → Student AR View（block size=1）→ Student Diffusion View 两段式 transitive distillation，避开把 AR 分数直接拷进双向 Mask 上下文；生成侧用 token age 中间态做 progressive self-refining diffusion；Elastic-MoT 按任务激活 8B / 12B / 16B。'
models: ['NLD-Uni', 'NLD', 'Nemotron-3 Nano Omni', 'Qwen-Image', 'LaViDa-O']
---

# Nemotron-Labs-Diffusion-Uni：统一理解与生成的离散扩散模型

> **论文**：[Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation (arXiv:2610.36466)](https://arxiv.org/abs/2610.36466)
> **作者**：Shufan Li, Hanrong Ye, Greg Heinrich, Yonggan Fu, Haotian Zhang, Sifei Liu, Aditya Grover, Pavlo Molchanov
> **机构**：NVIDIA ｜ **发布时间**：2026 年 10 月 27 日
> **研究方向**：Unified Multimodal Models、Discrete Diffusion Language Models、On-Policy Distillation、Multimodal Understanding and Generation
> **专题归属**：[🔁 OPD 子专题：跨范式 On-Policy 蒸馏](/domains/distillation/opd/)（本专题第一篇）

---

## 一、研究背景与问题定义

### 1.1 研究背景

现有多模态模型主要有两类能力：多模态理解和多模态生成。理解模型接收图像与文本，输出文本形式的回答；生成模型则根据文本或图像条件生成新的图像。

传统方案通常采用不同模型或不同建模范式分别实现这两种能力。随着统一多模态模型的发展，研究者开始探索能否在同一个模型体系中同时实现视觉理解、文本推理、图像生成与图像编辑。

目前主要存在三种建模范式：

1. **Unified Autoregressive Modeling：** 将文本和图像转换为离散 Token，统一使用自回归方式建模。代表工作包括 Janus-Pro、Emu3 等。
2. **AR + Continuous Diffusion：** 文本使用自回归建模，图像使用连续扩散或 Flow Matching 建模。代表工作包括 BAGEL、Transfusion、MetaQueries 等。
3. **Unified Discrete Diffusion：** 将文本和图像统一纳入离散扩散框架，通过 Masked Diffusion 实现多模态理解与生成。代表工作包括 LaViDa-O、MMaDA 等。

其中，统一离散扩散模型具有统一训练目标、多个 Token 并行预测等优势，但与专用 AR-VLM 和连续图像生成模型相比，仍然存在能力差距。

### 1.2 核心研究问题

NLD-Uni 主要解决两个问题：

**问题一：如何将不同模型范式的能力迁移到统一离散扩散模型？**

现有统一 DLLM 缺乏足够强的同范式 Teacher。直接使用 AR-VLM 或 Continuous Flow Model 蒸馏，又会面临模型生成范式、预测空间和条件上下文不一致的问题。

**问题二：如何提升离散扩散模型的图像生成质量？**

传统 Masked Diffusion 中，Token 直接从 Clean 状态跳转到 Mask 状态，推理时再从 Mask 恢复到 Clean。一旦 Token 被确定，通常就保持不变，缺少连续扩散模型那样逐步细化和修正生成结果的能力。

### 1.3 论文主要贡献

1. 提出 Cross-Paradigm On-Policy Distillation，将 AR-VLM 的视觉理解能力和 Continuous Flow Model 的视觉生成能力迁移到统一 DLLM。
2. 提出 Progressive Self-Refining Diffusion，通过多级 Token 状态实现渐进式加噪与恢复，提升图像生成质量。
3. 构建 NLD-Uni，通过 Elastic-MoT 实现视觉理解、推理、文本生成图像和图像编辑等任务的统一建模。

![](/nld-uni-fig01-teaser.png)

*图 1（论文 Fig. 1）：左——通过 cross-paradigm on-policy distillation 从异构 teacher（VLM 负责理解、连续扩散模型负责生成）构建统一 Diffusion LLM 的新范式；右——视觉理解与视觉生成的横向对比，NLD-Uni 在统一 DLLM 中取得与专用模型可比的结果。*

---

## 二、统一理解与生成模型架构

### 2.1 Elastic Mixture-of-Transformers

NLD-Uni 采用双塔 Elastic Mixture-of-Transformers（Elastic-MoT）架构，包括：

- Understanding Tower：负责视觉理解、文本推理等任务。
- Generation Tower：负责图像生成与编辑任务。
- Joint Attention：实现两个分支之间的信息交互。

两个分支共享统一的多模态建模体系，但保留不同的专家参数，以适应理解和生成任务的差异。

模型采用三种弹性工作模式：

| 工作模式 | 激活参数 | 主要任务 |
|---|---:|---|
| VLM View | 8B | 视觉理解与推理 |
| T2I View | 12B | 文本生成图像 |
| Interleaved View | 16B | 理解与生成交替进行 |

其中 T2I View 使用 4B Understanding 参数与 8B Generation 参数；Interleaved View 使用完整的两个分支。

这种设计能够根据任务动态加载所需参数，而不是每次都使用完整模型。

![](/nld-uni-fig02-elastic-mot.png)

*图 2（论文 Fig. 2）：Elastic-MoT 的三种弹性视图。VLM view 只用 8B understanding 分支；T2I view 由 4B understanding 参数 + 8B generation 分支组成；interleaved view 用满两塔共 16B。黄色 / 绿色块分别表示 understanding 与 generation 层，灰块为未加载层，蓝色箭头为 joint attention。右下展示 Reason → Locate → Edit 的交替推理链。*

> 初始化方式（§3.1）：understanding tower 由扩散语言模型 **NLD** 初始化（采用 block-causal SBD 形式），generation tower 由它的**同一份权重的拷贝**初始化。

### 2.2 图像表示与 Encoder

NLD-Uni 对输入图像采用双编码设计：

**Semantic Encoder：RADIO-H**

提取连续视觉语义特征，为模型提供图像理解所需的语义信息。

**VQ Encoder：预训练 IBQ**

将图像编码为离散视觉 Token，用于图像生成与离散扩散建模。

输入图像同时使用两种表示：

- 连续 Embedding：提供视觉语义信息。
- 离散 VQ Codes：提供离散视觉 Token 序列。

输出图像则仅使用 VQ Codes 表示。

VQ 编码的基本过程为：

$$
Z_e=E_{\mathrm{VQ}}(X)
$$

对每个连续特征向量进行 Codebook 最近邻量化：

$$
k_i=\arg\min_j\|z_{e,i}-c_j\|_2^2
$$

得到离散 Token：

$$
Z_q=[k_1,k_2,\ldots,k_N]
$$

NLD-Uni 的 VQ 词表包含约 131K 个离散 Code。

需要注意：论文采用的是预训练 IBQ，并不是在 NLD-Uni 中从零训练一个普通 VQ-VAE。

![](/nld-uni-fig03-framework.png)

*图 3（论文 Fig. 3）：整体输入输出框架。输入流里的图像同时给出 semantic encoder 的连续 embedding（青色）与 VQ encoder 的离散 token（黄色）；输出流里图像只用 VQ codes 表示，再由 VQ Decoder 还原。中间是 Elastic Unified Diffusion Model 在部分 mask 的输入上预测 clean token 分布；图像生成一侧额外引入 intermediate masking state（半实心块）。*

---

## 三、离散扩散模型基础

### 3.1 Autoregressive Modeling

自回归模型按照从左到右的顺序逐 Token 生成：

$$
p(x)=\prod_{i=1}^{L}p(x_i|x_{<i})
$$

每次根据已经生成的 Token 预测下一个 Token。

优点是训练与推理机制成熟，缺点是生成过程具有串行依赖。

### 3.2 Masked Discrete Diffusion

离散扩散模型通过随机 Mask 构造训练样本，学习恢复被 Mask 的 Token。

正向过程：

$$
x_0\rightarrow x_t
$$

其中 $x_0$ 为原始序列，$x_t$ 为部分 Mask 的序列。

最简单的 Masking 机制：

$$
q(x_{t,i}|x_{0,i})=
\begin{cases}
[MASK], & \text{以概率 }t\\
x_{0,i}, & \text{以概率 }1-t
\end{cases}
$$

模型学习预测 Clean Token：

$$
p_\theta(x_{0,i}|x_t,t)
$$

训练损失通常采用 Masked Cross-Entropy：

$$
\mathcal L_{\mathrm{MDM}}
=
-\mathbb E
\left[
\sum_{i\in M}
\log p_\theta(x_{0,i}|x_t,t)
\right]
$$

其中 $M$ 表示被 Mask 的位置集合。

推理时，模型从全 Mask 或部分 Mask 状态出发，通过多轮预测逐步恢复 Token。

与 AR 不同，DLLM 可以在一次前向计算中预测多个 Mask 位置。

### 3.3 Block Diffusion 与 SBD

普通全序列 Diffusion 需要对整个序列进行双向注意力计算，长序列下计算开销较大。

Block Diffusion 将序列划分为多个连续 Block：

$$
X=[B_1,B_2,\ldots,B_N]
$$

Block 之间按照从左到右的顺序生成；每个 Block 内部采用 Diffusion 进行并行预测。

因此：

$$
\text{Block Diffusion}
=
\text{Autoregressive Blocks}
+
\text{Parallel Diffusion within Blocks}
$$

NLD-Uni 采用 SBD（Semi-autoregressive Block Diffusion）框架，支持可变 Block Size。

- Block Size = 1：退化为标准 AR 模型。
- Block Size > 1：采用块内并行 Diffusion。
- 图像：将一张图像作为一个整体 Block，Block Size 随图像分辨率变化。

SBD 的训练目标可以表示为：

$$
\mathcal L_{\mathrm{SBD}}
=
\mathbb E
\left[
-\frac{1}{t}
\log p(B_i^0|B_{j<i}^0,B_i^t)
\right]
$$

其中 $B_i^0$ 是 Clean Block，$B_i^t$ 是部分 Mask 的 Block，前面的 Block 保持 Clean 状态。

![](/nld-uni-fig10-sbd.png)

*图 10（论文 Fig. 10）：SBD 的训练结构。(a) 双流训练——把 clean blocks 与 partially masked blocks 在序列维拼接，clean 部分算 AR CE loss，noisy 部分算 diffusion CE loss；(b) 专用注意力 mask——clean block 之间 block-causal（只看到自己与之前的 clean block），noisy block 之间互相不可见，只能看到全部 clean block 与自身。*

> 这个“Block Size 可调”的特性后面会被直接拿来当跨范式蒸馏的接口：把 Block Size 设为 1，同一个模型就退化出 AR view（§4.2）。

---

## 四、核心创新一：Cross-Paradigm On-Policy Distillation

### 4.1 Knowledge Distillation

知识蒸馏通过 Teacher 模型指导 Student 模型，使 Student 学习 Teacher 的知识与能力。

传统蒸馏通常使用 Teacher 独立生成的样本作为训练目标。

On-Policy Distillation（OPD）则使用 Student 自己生成的轨迹，让 Teacher 在这些 Student-generated Samples 上提供监督。

两者的区别：

| | Off-Policy Distillation | On-Policy Distillation |
|---|---|---|
| 训练轨迹来源 | Teacher | Student |
| Teacher 的作用 | 提供训练目标 | 指导 Student 自己的生成结果 |
| 训练分布 | 更接近 Teacher | 更接近 Student |
| 核心问题 | 训练与推理可能存在分布偏移 | 缓解 Student 自身生成分布带来的偏移 |

OPD 的核心思想是让 Teacher 直接指导 Student 在实际生成过程中遇到的状态。

### 4.2 AR-VLM → Discrete Diffusion：视觉理解能力迁移

#### 4.2.1 核心问题

AR Teacher 和 Diffusion Student 的预测上下文不同。

AR 模型根据左侧前缀逐 Token 预测：

$$
p(y_i|y_{<i})
$$

Diffusion 模型则根据部分 Mask 的双向上下文预测：

$$
p(y_i|y_{\mathrm{masked}},t)
$$

如果直接将 AR Teacher 的预测分布复制给 Diffusion Student，可能造成监督不匹配。

例如，Student 的回答为：

Jason visited San Francisco.

AR Teacher 根据前两个词，希望将 San 改成 New。

但如果 Diffusion Student 的上下文是：

Jason visited [MASK] Francisco

直接复制 Teacher 对 New 的高概率预测，就可能引导模型生成：

Jason visited New Francisco.

这是不合理的监督。

#### 4.2.2 Transitive Distillation

作者提出两阶段知识迁移方案。

**第一阶段：AR Teacher → Student AR View**

利用 SBD 的可变 Block Size 特性，当 Block Size = 1 时，Student 可以退化为 AR 模型。

因此，先让 Student 的 AR View 在自身生成轨迹上接受 AR Teacher 的指导。

$$
\mathcal L_{\mathrm{AR-OPD}}
=
D_{\mathrm{KL}}
\left(
p_s^{AR}\parallel p_t^{AR}
\right)
$$

**第二阶段：Student AR View → Student Diffusion View**

Student 的 AR View 已经学习到 Teacher 的理解与推理能力。

接下来让 AR View 充当内部 Teacher，指导 Diffusion View。

具体过程：

1. Student 的 AR View 自己生成一个完整 Block。
2. 对生成的 Block 进行随机 Mask。
3. 使用 Diffusion View 对 Mask 位置进行预测。
4. 通过 KL Divergence 约束 AR View 与 Diffusion View 的预测分布。

$$
\mathcal L_{\mathrm{Cons}}
=
D_{\mathrm{KL}}
\left(
p_s^{AR}\parallel p_s^{Diff}
\right)
$$

其中 AR View 的预测作为固定监督目标，采用 Stop-Gradient。

最终损失：

$$
\boxed{
\mathcal L_{\mathrm{OPD-und}}
=
\mathcal L_{\mathrm{AR-OPD}}
+
\mathcal L_{\mathrm{Cons}}
}
$$

这就是 Transitive Distillation：

$$
AR\ Teacher
\rightarrow
Student\ AR\ View
\rightarrow
Student\ Diffusion\ View
$$

其核心价值是通过 Student 自身的 AR View 建立跨范式知识迁移桥梁，避免直接使用 AR Teacher 分布指导不同上下文中的 Diffusion View。

![](/nld-uni-fig05-ar-opd.png)

*图 5（论文 Fig. 5）：从 AR-VLM 做 on-policy 蒸馏。上方：给定 prompt 与 clean prefix，用 block size=1 让 student 的 AR view 自己 roll out 当前 block。左侧：AR OPD loss 把 student AR 分布对齐冻结的 AR teacher；右侧：consistency loss 把 student 的 diffusion view 对齐它自己的 AR view——对 AR rollout 做 forward corruption 得到 $B_i^t$，只在意被 mask 的位置上算 KL，AR 一侧 stop-grad。*

> **为什么不能直接拷 teacher 分数（论文 §3.3 的补充说明）**：OPD-LM 的场景里 student 由 teacher 权重初始化（AR→Diff 转换），teacher 与 student AR view 的初始差距为 0；而这里的 teacher 是**更大更强的异构模型**，student 已经是训好的扩散模型，初始分布差距很大，naive 直接拷 teacher 分数会产生大量 ill-formed 监督。另外作者发现 **AR rollout 用小的非零温度**效果更好（Table 8：温度 0.4 时 diffusion 74.8、AR 76.0；温度 0.0 时 diffusion 73.9、AR 76.9）——轨迹多样性带来的收益超过 argmax 性质被破坏的代价。

### 4.3 Continuous Flow Teacher → Discrete Diffusion：图像编辑能力迁移

#### 4.3.1 核心问题

Continuous Flow Model 与 Discrete Diffusion Model 的图像表示空间不同：

- Teacher：连续 VAE Latent。
- Student：离散 VQ Token。

因此，无法简单使用 Token 级 KL Divergence 进行分布对齐。

#### 4.3.2 Teacher Refinement

作者使用 Qwen-Image 作为 Continuous Flow Teacher。

具体流程：

1. Student 根据输入图像和编辑指令生成离散 VQ Codes：

$$
X_s^0\sim p_\theta(X|V_{\mathrm{in}},p)
$$

2. 通过 Student 自己的 VQ Decoder 还原为图像：

$$
V_s=\mathrm{Decode}_{VQ}(X_s^0)
$$

3. 使用 Teacher 的连续 VAE Encoder，将 Student 图像转换为连续 Latent：

$$
z_s=\mathrm{Encode}_{VAE}(V_s)
$$

4. 对 Latent 加入高斯噪声：

$$
z_r=(1-r)z_s+r\delta,\qquad \delta\sim\mathcal N(0,I)
$$

其中 $r\in(0,1)$ 控制 refinement 强度。

5. 用 Teacher 在**多步 ODE** 上从 $z_r$ 去噪回 clean latent（关键：不是单步 teacher forward）：

$$
\hat z^0=\mathrm{ODESolve}_T\left(z_r;\,r\rightarrow 0,\,M,\,V_{\mathrm{in}},p\right)
$$

6. 解码回像素空间再用 Student 的 VQ Encoder 编码成离散目标，并停止梯度：

$$
\hat X^0=\mathrm{stopgrad}\left(\mathrm{Encode}_{VQ}\left(\mathrm{Decode}_{VAE}(\hat z^0)\right)\right)
$$

7. 对 $\hat X^0$ 采样部分腐蚀版本 $\hat X^t$，在 refined target 上做 SFT 式损失：

$$
\mathcal L_{\mathrm{OPD-Gen}}
\left(\hat X^0,\hat X^t,V_{\mathrm{in}},p,t\right)
=
-\frac{1}{t}\log p_\theta\left(\hat X^0\mid \hat X^t,V_{\mathrm{in}},p\right)
$$

整个“refine 目标”的构造来自 student 当前的 rollout，因此监督会随 student 的生成分布一起演化；论文明确说这**不是传统意义上的 on-policy**，但与本领域若干“用 teacher 修正 student rollout”的并发工作是同一思路。

![](/nld-uni-fig06-flow-refine.png)

*图 6（论文 Fig. 6）：从连续流模型做 on-policy 蒸馏。给定 student 输出，先在 teacher 的 latent 空间加噪、再用 teacher 重新去噪，得到 teacher-refined output；作为对照也给出 teacher 独立生成。可以观察到 teacher 的 refinement 与 student 输出高度对齐（语义与空间结构不变），但伪影更少、保真度与源图保持更好；而独立 teacher 生成与 student 输出的差距明显更大——这正是 refined 样本适合当训练目标的原因。*

> **两个实现细节**：噪声比例 $r$ 的消融（Table 9，ImgEdit 总分）为 $r=0.5\to 4.06$、$r=0.8\to 4.09$、$r=0.9\to \mathbf{4.17}$、$r=1.0\to 3.90$；$r=1.0$ 相当于完全换成 teacher 独立生成，掉分最多。另外 teacher 偶尔会 over-edit，此时 refined 结果反而不如纯 teacher 生成（附录 Figure 14）。

---

## 五、核心创新二：Progressive Self-Refining Diffusion

### 5.1 动机：二值 mask 状态的两个硬约束

Canonical mask discrete diffusion 的状态空间是 $S=\mathcal V\cup\{[m]\}$，token 只能在 clean 与 mask 之间**一步跳变**。由此带来两个问题：

1. **没有中间态就没有修正机会**：推理时 token 一旦被 unmask 就固定，后续步骤无法再改（token editing 类方法要靠置信度启发式去“回改”，对阈值敏感，且与训练设定不一致）。
2. **同一步采样的 token 被假设为条件独立**：MDM 从逐 token logit 独立采样，同一步内的 token 之间没有交互；而同一张图里的 token 高度相关，这个假设在实践中几乎不成立。

作者的解法：**在 clean 与 mask 之间插入中间状态**，把“一步跳变”拆成“多步转移”，从而让 token 之间产生更细粒度的交互与渐进式自我纠正。

### 5.2 状态空间：Token Age

引入 token age $a\in\mathcal A=\{0,1,2,\ldots,K\}$：

- $a=0$：clean 状态。
- $a=K$：完全 mask 状态。

关键点：$a$ 是**逐 token 的离散索引**，与全局连续时间步 $t\in[0,1]$ 是两个不同的量。状态空间从 $S=\mathcal V\cup\{[m]\}$ 扩展为：

$$
S=\mathcal V\times\mathcal A
$$

正向过程中，每个 token $(c,a)\in S$ 只能转移到 $(c',a+1)$，$(*,0)$ 为初始态、$(*,K)$ 为终止态。论文把这个过程拆成两部分的交互：

- **Per-token corruption process（离散）**：转移概率 $U_{k|k-1}(u|c)$，每个 token 从 clean 出发经过 $K$ 次跳变到 mask。
- **Per-token jump schedule（连续）**：$b_i^t(a)=\mathbb P(A_i^t=a)$ 决定这些跳变发生在 $[0,1]$ 区间内的什么时刻，需满足 $b_i^0(0)=1$、$b_i^1(K)=1$。

![](/nld-uni-fig04-self-refine.png)

*图 4（论文 Fig. 4）：自细化过程的设计。左：状态空间——Mask Diffusion 只有 clean / mask 两态，Self-Refining 模型多出 $a=1,\ldots,K-1$ 的中间态。中：扩散过程——每个 token 不是一步跳变，而是在区间 $(\tau,\tau')$ 内、长度 $\epsilon$ 的时间里逐级加噪；$t=1\to 0$ 时再逐级细化回来。右：用 VQ embedding 空间的 ball tree 实现 $a=0\to K$ 的随机跳变，token 分布由 one-hot 扩散为半径逐渐增大的球（示例树深 $D=4$）。*

### 5.3 粗到细的 Corruption：用 Codebook 几何代替学出来的子 token

中间态需要定义“一个 token 变成什么”。作者的答案是**利用 VQ codebook 的几何结构**：在 embedding 空间上建 **ball tree**（平衡二叉树，父节点球半径大于子节点，叶节点对应词表中每个唯一 token）。

- $N_d(c)$：与 $c$ 在深度 $d$ 共享同一祖先的所有 token 集合，$N_1(c)=\mathcal V$（根），$N_D(c)=\{c\}$（叶）。
- 单步转移定义为在 $N_{D-k}(c)$ 上的均匀分布，于是前向 / 反向边缘为：

$$
U_{j|k}(u|c)=\mathbf{1}\{u\in N_{D-j}(c)\}\frac{1}{|N_{D-j}(c)|};
\quad
U_{k|j}(c|u,\hat c)=U_{k|0}(c|\hat c),\ \forall u\in N_{D-j}(\hat c)
$$

其中 $\hat c$ 是 $a=0$ 时的初始 token。也就是：age 越大 → 所在节点越浅 → 球越大 → 采样到的 token 越“离谱”。这是一个 **coarse-to-fine 的语义腐蚀**，不需要学习显式的 sub-token 表示。

**为什么必须用树**：词表有 131K 个 VQ code，完整的两两距离矩阵需要约 **32GB VRAM**；对应的 18 层二叉树用一维数组表示只要 **524KB**。

### 5.4 Per-Token Jump Schedule：把离散跳变嵌进连续时间

沿用 DNDM 的思路：为每个 token 采一个跳变时刻 $\tau_i\sim U([0,1])$，则前向过程是确定性的（token 在 $t=\tau_i$ 从 clean 跳到 mask），且与 vanilla MDM 有相同的前向边缘。

扩展到多级中间态的方式是把这个时刻扩成一个**区间**：

$$
\tau_i\sim U([0,1-\epsilon]),\qquad \tau_i'=\tau_i+\epsilon
$$

- $\epsilon\to 0$：退化为 DNDM（所有跳变挤在无穷小时间窗内）。
- $\epsilon=1$：所有 token 以相同速率被渐进腐蚀。

为降低复杂度，所有中间 age $1,\ldots,K-1$ 在 $(\tau,\tau')$ 内均匀推进，每级占 $\epsilon/K$ 时间；给定 $\epsilon$，每个 per-token jump schedule 都与 DNDM 的单步 schedule 构成双射（做变量替换即可复用前向 / 反向边缘）。这个流程是**非马尔可夫**的（无法对 $\tau_i$ 边缘化），因此训练与推理都预采样 $(\tau_i,\tau_i')$，此后 age 的变化是确定性的：

$$
a_i(t,\tau_i,\epsilon)=
\begin{cases}
0 & 0\le t\le \tau_i\\
\left\lfloor K\frac{t-\tau_i}{\epsilon}\right\rfloor & \tau_i<t<\tau_i'\\
K & t\ge \tau_i'
\end{cases}
$$

推理时每个 token 的节奏是：$[\tau_i',1]$ 保持 masked → $[\tau_i,\tau_i']$ 内渐进细化 → $[0,\tau_i]$ 保持固定。

### 5.5 与 MDM-Prime 的区别

MDM-Prime 把 token 拆成 sub-token 来实现部分 mask/unmask，但其图像实验只做到 32×32 低分辨率，作者实测**无法 scale 到本文设定**。NLD-Uni 则直接用 codebook 几何（ball tree）在粗到细的粒度上做部分 mask/unmask，不需要学习显式的 sub-token 表示。

### 5.6 消融：自细化到底赢在哪

与 uniform diffusion、masked diffusion、masked diffusion + token editing 在相同预算下对比（64 diffusion steps，20k 步训练全程）：

![](/nld-uni-fig07-self-refine-ablation.png)

*图 7（论文 Fig. 7）：自细化扩散过程的消融。(a) GenEval（检测器口径，考高层语义）——NLD-Uni 与 uniform diffusion 相当；(b) DPG（细粒度对齐）——NLD-Uni 稳定优于 uniform；(c) HPSv3（人类偏好）——NLD-Uni 提升最大，说明自细化最直接的收益是视觉保真度。masked diffusion + token editing 全程垫底。*

**NFE 与 refinement 步数 $N$ 的交互**（Table 6，HPSv3）：$N$ 是每个被采样 token 经历的 refinement 次数（$N=1$ 即 vanilla mask diffusion，$N=\mathrm{NFE}$ 即 $\epsilon=1$）。任何 NFE 下 $\epsilon=1$ 都很差；小 NFE 时 $N=1$ 最好；大 NFE 时更大 $N$ 才开始有收益，默认 64 步推理下 **$N=8$ 最佳**，$N=16$ 无进一步提升且低 NFE 时会明显退化。作者的结论是：应当在不同 token 的 unmask/refine 之间**强制某种顺序**，而不是让大量 token 同时被处理。

---

## 六、实验结果

### 6.1 训练设置（§4.1 + Table 5）

- **三阶段**：Stage 1 以图文对为主（基础视觉对齐 + 图像生成）；Stage 2 加入指令跟随、图像编辑、object grounding、interleaved 生成；Stage 3 是统一的 OPD 阶段——理解侧 teacher 为 **Nemotron-3 Nano Omni**，编辑侧 teacher 为 **Qwen-Image**，同时混入 T2I SFT 批次保住文生图能力。
- **T2I 不设 OPD teacher**：因为基模在 GenEval / HPSv3 上已经强于 Qwen-Image、Flux 这些候选 teacher。
- **算力**：64×H100、26 天，其中最后的 OPD 阶段占 5 天。
- **超参**：Stage 3 学习率 $2\times10^{-6}$、10k steps、AdamW（$\beta_1=0.99,\beta_2=0.999$），objective 为 SFT+OPD。

### 6.2 四张主表

| 任务 | 指标 | NLD-Uni | 对照 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 图像理解 | 11 benchmark 平均 | **78.3** | 离散扩散 SOTA LLaDA2-Uni 61.2 / LaViDa-O — / NLD-VLM 58.3 | Δ vs. Diff. SOTA **+17.1**；Qwen3-VL-8B AR 参照为 78.1 |
| 多模态推理 | MathVista / DynaMath / MathVision / MathVerse / Geometry3k / LogicVista / We-Math | 77.0 / 74.7 / 45.4 / 45.3 / 67.9 / 55.9 / 74.9 | NLD-Uni-Base 71.5 / 68.3 / 35.8 / 35.5 / 54.4 / 52.4 / 59.8 | **naive OPD（直接拷 teacher 分数）全线崩到 47.7 / 46.7 / 17.9 / 33.4 / 23.0 / 38.7 / 42.0**，比 base 还低 |
| 文生图 | GenEval Overall / DPG / HPSv3 | 0.89 / 86.6 / **11.12** | NLD-Image 0.90 / 85.2 / 9.61；Qwen-Image 0.87 / 88.3 / 10.91 | HPSv3 为全表最高；12B（T2I view 只加载一半 understanding 层） |
| 图像编辑（ImgEdit） | Overall | **4.17** | NLD-Uni-Base 3.81 / +Off-Policy KD 3.90 / LaViDa-R1 3.90 / Qwen-Image 4.27 / GPT-4o 4.20 | teacher refinement 比 off-policy 直接用独立 teacher 输出高 0.27 |

![](/nld-uni-fig16-opd-ablation.png)

*图 16（论文 Fig. 16）：从 AR-VLM 做蒸馏的消融（MathVista 随训练步数）。左：diffusion 性能——AR-OPD + consistency（蓝）一路涨到 ~75.1；AR-OPD + SFT、AR-OPD-Only 都停在 72–73；**AR-OPD + Teacher-Diff-OPD（红）和纯 KD（紫）直接崩到 50 以下**，即把 teacher 分数直接搬到扩散 view 会让训练发散。右：AR 性能——所有 AR-OPD 变体都随步数上升，说明 AR view 本身学得没问题，崩的是扩散 view 那一侧的监督方式。*

### 6.3 推理效率（Table 10，H100）

| 模式 / 置信度阈值 | Tokens/Forward | Tokens/s/GPU | MathVista |
| :--- | ---: | ---: | ---: |
| AR | 1.00 | 202.74 | 76.1 |
| Diffusion 0.95 | 3.94 | 463.84 | 75.2 |
| Diffusion 0.90 | 4.64 | **552.58** | 75.0 |
| Diffusion 0.60 | 8.31 | 997.95 | 73.4 |
| Diffusion 0.50 | 7.23 | 734.03 | 70.8 |

也就是说：**阈值 0.90 时约 2.7× 于 AR 的吞吐，只掉 1.1 分**；阈值放到 0.60 能到 ~5× 吞吐，但掉到 73.4。

---

## 七、与 OPD 谱系的位置

### 7.1 和 OPD-LM 的逐条差异（附录 A.3.1）

| | OPD-LM | NLD-Uni |
| :--- | :--- | :--- |
| Rollout 来源 | **diffusion view** | **AR view**（block size=1） |
| 谁来监督 diffusion view | 直接用 **teacher** 分数 | 用**自己的 AR 分数**（stop-grad） |
| Teacher / student 关系 | student 由 teacher 权重初始化（未训好的扩散模型） | 两个不同大小、不同架构的模型；student 蒸馏前已有强扩散生成能力 |
| 目标 | 单一目标的 AR→Diff 后训练（语言域） | 多目标统一训练里的一个组件（理解 + T2I + 编辑同时进行） |

作者对 OPD-LM 的批评很具体：直接拷 teacher 分数会造成 **ill-formed 监督**。附录给的语言侧例子是——student 生成 “The number $i$ is called the multiplicative identity.”，teacher 在前缀 “...is called the” 之后会给 “imaginary” 高概率，但 masked 上下文 “...called the [m] identity” 下 “imaginary identity” 根本不成词。NLD-Uni 的 consistency loss 用 student 自己的 AR 预测当目标，绕开了这层 teacher–student 分布差，只保留 AR / diffusion 上下文差这一个变量。

### 7.2 这篇笔记里可迁移到 OPD / OPSD 专题的四点

1. **“范式接口”比“损失函数”更关键**：SBD 的 block size 可调，使得同一个模型能同时扮演 AR student、AR teacher、diffusion student 三个角色；跨范式蒸馏的第一步往往是先在 student 内部**造出一个可对齐的中间 view**。
2. **上下文错配不能靠“少监督一点”解决**：naive OPD 在 Table 2 / Figure 16 上是**低于 base** 的，不是小幅下降，而是训练崩坏；说明错配的监督信号会主动破坏模型。
3. **stop-grad 的自蒸馏是合法桥梁**：AR view 提供的是“同一条轨迹、同一个模型”的分数，分布差被压到只剩上下文差；这正好是 OPSD 子专题的核心机制，只是这里它被当作 OPD 的中转站。
4. **teacher 不必“生成得更好”**：continuous flow teacher 在编辑任务上偶尔会 over-edit，但**在 student 自己的 latent 上加噪再回噪**得到的 refined target 依然有用——把 teacher 当成“同分布上的去噪算子”，比当成“更好的生成器”更稳。

---

## 八、关键超参与配置速查

| 项 | 取值 | 出处 |
| :--- | :--- | :--- |
| VQ 词表 | ~131K codes（预训练 IBQ） | §2.2 |
| Ball tree 深度 / 默认 $K$ | $D=18$，默认 $K=17$；训练时在 $\{1,\dots,17\}$ 里随机取 $K$ | 附录 A.2.1 |
| 树存储开销 | 一维数组 524KB（完整距离矩阵需 ~32GB） | §3.2 |
| 中间态时长 $\epsilon$ | 控制 $(0,1]$；$\epsilon\to0$ 退化为 DNDM，$\epsilon=1$ 为全局同步腐蚀 | §3.2 |
| 推理默认设定 | NFE 64，refinement $N=8$ 最佳；$N=1$ 在低 NFE 下最好 | Table 6 |
| AR rollout 温度 | 0.4 最优（diffusion 74.8 / AR 76.0） | Table 8 |
| Teacher refinement 噪声 $r$ | 0.9 最优（ImgEdit 4.17） | Table 9 |
| Teacher 选择 | Nemotron（理解）、Kimi（理解对照）；Qwen-Image（编辑） | Table 7 |
| 三阶段 LR / steps | $1\times10^{-5}$ / 400k → $1\times10^{-5}$ / 100k×2 → $2\times10^{-6}$ / 10k | Table 5 |
| 训练成本 | 64×H100 × 26 天（OPD 阶段 5 天） | §4.1 |

---

## 关联笔记

- [🔁 蒸馏专题总览：OPD / OPSD](/domains/distillation/)
- [🪞 OPSD 子专题：On-Policy 自蒸馏](/domains/distillation/opsd/) —— §4.2.2 的 stop-grad 自蒸馏是它的第一个实例
- [🎯 DPO 子专题](/domains/rl-aigc/dpo/) —— 同样属于「后训练信号设计」的另一条路线：DPO 用偏好对，OPD 用 teacher 分数
- [📐 通用理论基石](/foundations/) —— Flow Matching / Diffusion 的 ODE 视角是 §4.3 teacher refinement 的前提
