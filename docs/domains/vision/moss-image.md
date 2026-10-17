# Part 1：MOSS-Image 模型整体架构

**本部分对应论文第 3–4 页的 Figure 2、公式（1）（2），以及附录 A.1（第 14 页）。**

这一部分我们要理解三个问题：MOSS-Image 的模型结构是什么？文本和图像信息如何在 DiT 中交互？模型如何通过扩散过程生成图像？

## 1. 整体架构：5B Single-Stream DiT

先看论文原始架构图。

<!-- 图片待补：MOSS-Image Figure 2 模型架构（原路径：MOSS_Part1_Figure2.png） -->

整个系统实际上由两个相对独立的部分组成。

**第一部分：Agentic Prompt Enhancer。** 负责理解用户意图、检索相关信息，产生详细的文字描述和 Layout JSON。这是一个推理阶段的模块，我们留到 Part 6 专门学习。

**第二部分：Layout-Controllable Diffusion Transformer。** 这才是真正的图像生成基座。它可以接受普通自然语言，也可以接受结构化布局描述，然后通过扩散生成图像。

这里有一个非常重要的概念：

**论文所说的 5B，是 DiT 生成器的参数量，不包括冻结的 Qwen3.5-9B Text Encoder 和 FLUX.2 VAE。** 作者从零预训练的是生成器，并不是重新训练整个文本编码器和 VAE。

### 核心模型配置

| 组件 | 配置 |
|---|---|
| DiT 类型 | Single-Stream Transformer |
| DiT 参数量 | 5B |
| DiT Blocks | 42 层 |
| Hidden Size | 3072 |
| Attention Heads | 24 |
| Head Dimension | 128 |
| FFN Hidden Size | 8192 |
| Text Encoder | Qwen3.5-9B，冻结 |
| Text Adapter | Linear + 2 层 Transformer |
| Image VAE | FLUX.2 VAE，冻结 |
| VAE Latent Channels | 32 |
| VAE 下采样倍率 | 8 |
| Latent Patch Size | 2×2 |
| 位置编码 | 4D RoPE |
| FFN | SiTU-GLU |

这些参数来自附录 Table 5。

下面沿着实际的数据流理解模型。

## 2. 输入是怎么处理的？

MOSS-Image 有两条输入处理路径：文本输入和图像 Latent 输入，最后汇合到同一个 Transformer。

### 2.1 文本路径

假设用户输入：

`A red circle below the word SALE.`

其处理过程为：

**Prompt / Layout JSON**

↓

**Qwen3.5-9B Text Encoder（冻结）**

输出最后一层的 Token Hidden States，4096 维

↓

**Linear Projection**

4096 → 3072

↓

**2-Layer Text-Only Transformer**

↓

**Text Tokens，维度 3072**

为什么在 Qwen3.5-9B 后面还要加一个 Linear 和两层 Transformer？

因为 Qwen 输出的文本特征为 4096 维，而 DiT 的隐藏维度只有 3072，首先需要完成维度适配。后面的两层 Text-Only Transformer 进一步处理文本特征，再把它们送进图文联合 Transformer。

需要注意的是，这里使用的是 Qwen 的 **Token-level Hidden States**，不是只用一个句子向量。因此不同文字、对象和空间关系的信息可以保留在不同的 Token 表示中。

论文还明确说明，普通自然语言和 Layout JSON 都会经过这条文本条件路径，不存在一个完全独立的 JSON 控制网络。

### 2.2 图像路径

图像生成模型不会直接在原始 RGB 像素上运行 42 层 DiT，而是先在 VAE Latent 空间中进行生成。

我们用一张 1024×1024 的图像举例。

**第一步：通过 FLUX.2 VAE 得到 Latent。**

VAE 的下采样倍率为 8，Latent Channel 为 32，因此：

$$
1024\times1024\times3
\longrightarrow
128\times128\times32
$$

**第二步：进行 Patchify。**

作者并不是将每个 Latent 位置直接作为 Token，而是把相邻的 2×2 个位置合并。

每个 Image Token 包含：

$$
2\times2\times32=128
$$

个 Latent 数值。

最终 Image Token 数量为：

$$
N_{\mathrm{img}}=\frac{H}{16}\times\frac{W}{16}
$$

| 图像分辨率 | Image Token 数量 |
|---|---:|
| 256×256 | 256 |
| 512×512 | 1,024 |
| 1024×1024 | 4,096 |

**第三步：通过 Linear 将 Image Token 投影到 3072 维。**

这样文本与图像 Token 就有了相同的 Hidden Size，可以进入同一个 DiT。

还有一个容易混淆的地方：训练阶段，图像先通过冻结 VAE 编码为干净 Latent，再加噪输入 DiT；而推理阶段，DiT 是从噪声 Latent 开始生成，并不需要先提供一张真实图像。

## 3. 核心：Single-Stream Transformer 如何工作？

这是架构中最重要的部分。

传统的部分多模态 DiT 会分别维护文本流与图像流，再进行跨模态交互。

但 MOSS-Image 采用 Single-Stream 设计：

$$
X=[X_{\text{text}};X_{\text{image}}]
$$

也就是将文本 Token 和图像 Token 沿序列维度拼接起来，然后一起经过 42 层 Transformer。

每一层主要包含三个部分：

- Joint Self-Attention
- Shared Modulation（Scale / Shift）
- SiTU-GLU FFN

### 3.1 Joint Self-Attention

对于拼接后的序列 $X$，正常计算：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

随后进行 Attention：

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{128}}\right)V
$$

上面是 Self-Attention 的基本数学形式。论文架构图还显示了 Q-Norm、K-Norm、4D RoPE、时间调制及残差 Gate，它们共同构成实际的 Attention Block。

这种联合注意力有什么意义？

假如 Prompt 是：

`A red circle below the word SALE.`

文本 Token 中包含 `SALE`、`red circle` 和 `below`，图像 Token 则对应不同的空间位置。

经过 Joint Attention，图像 Token 可以直接与描述颜色、文字和位置关系的文本 Token 进行交互，进而学习把这些语义信息转换成空间结构和可见文字。

**重点：这里的文本与图像是在同一条 Transformer 序列内交互，而不是通过额外的 Cross-Attention 分支连接。**

### 3.2 时间调制：Scale / Shift / Gate

图像扩散生成是一个逐步去噪过程。在不同噪声水平，网络需要执行不同程度的图像恢复。

因此，DiT 会接收当前扩散时间信息，通过 Time Embedding 影响 Transformer Block。

论文 Figure 2 显示两种主要操作：

- **Scale / Shift：** 对归一化后的隐藏状态进行时间条件调制。
- **Gate：** 调节 Attention 和 FFN 分支输出进入残差路径的强度。

可以把它理解成：同一个网络在不同去噪时刻，通过时间条件调整内部特征处理方式。

这里暂时不展开具体调制参数如何生成，因为报告没有给出完整的时间调制公式及其初始化细节。

## 4. 两个值得专门理解的架构细节

### 4.1 SiTU-GLU：为什么不用普通 FFN？

传统 Transformer FFN 可以简化为：

$$
\mathrm{FFN}(x)=W_2\phi(W_1x)
$$

MOSS-Image 使用的是 SiTU-GLU，其公式（1）为：

$$
\boxed{
\mathrm{FFN}(x)=
W_o\left[
c_{\beta_g}(g)
\odot\mathrm{sigmoid}(g)
\odot c_{\beta_u}(u)
\right]
}
$$

其中：

$$
g=W_gx,\qquad u=W_ux
$$

$$
c_\beta(v)=\beta\tanh(v/\beta)
$$

乍看比较复杂，其实只需理解三个步骤。

首先，将输入 $x$ 投影成两组向量 $g$ 和 $u$。

其次，对两组向量进行不同的非线性处理。其中 $c_\beta$ 使用 tanh 对数值作平滑限制，$\mathrm{sigmoid}(g)$ 提供门控作用。

最后，将三项逐元素相乘，再通过 $W_o$ 投影回原始 Hidden Size。

与普通 FFN 相比，它引入了更加明确的门控机制和受控的激活幅度。

**需要区分事实与推测：** 论文使用 SiTU-GLU，但没有提供针对 SiTU-GLU 与 SwiGLU、GELU FFN 的直接消融，因此目前不能从这篇论文得出它单独提升了多少图像质量。

### 4.2 4D RoPE：四个维度分别是什么？

作者定义了四个位置轴：

$$
p=(r,t,w,h)
$$

| 位置轴 | 含义 |
|---|---|
| $r$ | Reference Index，参考图像编号 |
| $t$ | Token Sequence Position，序列位置 |
| $w$ | Width Position，图像横向位置 |
| $h$ | Height Position，图像纵向位置 |

具体计算是把每个 Attention Head 的特征划分为四部分，然后分别进行位置旋转：

$$
\mathrm{RoPE}_{4D}(z,p)=
[R_r(r)z_r;\ R_t(t)z_t;\
R_w(w)z_w;\ R_h(h)z_h]
$$

这里最特别的是 $r$。

MOSS-Image 当前只支持文生图，不支持直接输入参考图。因此，**所有 Token 的 $r=0$**。

于是：

$$
R_r(0)=I
$$

对应的 32 维特征不发生位置旋转，这就是作者说的 Partial RoPE。

剩余 96 维用于序列和图像空间位置编码。

作者保留 $r$ 轴，主要是为了未来能够扩展多参考图像输入，而不必重新设计整个位置编码结构。

另外，**RoPE 中的 $t$ 是 Token 序列位置，不是扩散时间步**。这两个概念必须分开。

附录 D 还提供了 RoPE 的消融实验。总体而言，不同坐标分配方案在多个生成指标上有一定取舍，没有一个方案在所有指标上绝对领先。

## 5. DiT 如何通过 Flow Matching 生成图像？

这里先讲整体机制，具体噪声分布和训练参数留到 Part 4。

训练阶段，设干净图像 Latent 为 $z_0$，随机噪声为 $\epsilon$。

按照与论文速度预测目标一致的标准直线路径，可以写成：

$$
z_\sigma=(1-\sigma)z_0+\sigma\epsilon
$$

其中 $\sigma$ 表示噪声水平。

模型接收三个输入：

$$
v_\theta(z_\sigma,1000\sigma,c)
$$

其中：

- $z_\sigma$：加入噪声的图像 Latent
- $1000\sigma$：时间条件
- $c$：文本或布局条件

DiT 需要预测从干净图像指向噪声的速度：

$$
v_{\mathrm{target}}=\epsilon-z_0
$$

论文训练损失的核心就是：

$$
\boxed{
\mathcal L_{\mathrm{FM}}
\propto
\left\|v_\theta(z_\sigma,1000\sigma,c)
-(\epsilon-z_0)\right\|_2^2
}
$$

实际实现还包含有效 Image Token 的求和与归一化。值得注意的是，**Text Tokens 不参与预测损失计算，损失只作用在图像输出部分**。

推理时，过程反过来：

**采样随机噪声 Latent**

↓

**DiT 预测 Velocity**

结合文本条件，从高噪声水平向低噪声水平逐步更新

↓（重复多个 Sampling Steps）

**得到干净图像 Latent**

↓

**FLUX.2 VAE Decoder → RGB 图像**

论文附录 J 的案例使用了 30 步去噪与 Shift=3 的采样调度，但这是该案例的设置，不能据此认定所有评测都使用相同的采样配置。

## 6. 一个完整的生成例子

假设用户要生成：

`Design a poster with a red circle below the word SALE.`

首先，Prompt 经过冻结的 Qwen3.5-9B，得到 4096 维 Token 特征，再适配成 3072 维。

随后，假设目标分辨率为 1024×1024，则随机噪声 Latent 会被整理成 4096 个 Image Tokens，每个 Token 同样投影到 3072 维。

接下来，文本 Token 与图像 Token 拼接，进入 42 层 Single-Stream DiT。每一层通过 Joint Self-Attention 交互文本语义与图像空间信息，通过 4D RoPE 提供位置信息，通过时间调制适应当前噪声水平，再通过 SiTU-GLU 处理特征。

经过多次去噪，得到最终图像 Latent，然后使用 FLUX.2 VAE 解码为 RGB 图像。

假如把输入改为包含 `SALE` 的精确位置、红色圆形的 Bounding Box、字号和排版属性的 Layout JSON，**后面的生成器结构完全不需要变化**，因为这些结构化内容仍然会经过原有的文本条件通道。

这正是 MOSS-Image 的一个核心设计。

---

## Part 1 总结

学完这一部分，我认为应该牢牢记住四件事：

1. **模型主体是 5B Single-Stream DiT。** Qwen3.5-9B 和 FLUX.2 VAE 都是冻结组件，不计入 5B。
2. **图文 Token 共同进入 42 层 Transformer。** 文本和图像通过 Joint Self-Attention 交互。
3. **模型用了 SiTU-GLU、时间调制、4D RoPE。** 其中 Reference Index 轴为未来参考图像生成预留扩展空间。
4. **Layout Control 走普通文本条件通道。** 不需要另外增加 ControlNet 或专门的 Bounding Box Encoder；这种能力主要依赖后续要学习的结构化数据监督。

这里还有一个非常关键的认识：**仅凭这一架构，还不能解释 MOSS-Image 为什么有如此强的长文本生成和精确布局能力。** 这些能力究竟如何获得，需要进一步看它使用了什么训练数据、如何构建文字与空间标注，以及如何组织预训练。

所以下一个部分，我建议进入 **Part 2：预训练数据构建**，重点理解 16 个数据集怎么组合、32 亿数据如何保留，以及论文中那个很反直觉的结论——为什么额外的数据质量过滤反而让模型变差。

---

# Part 2：MOSS-Image 预训练数据构建

**对应论文：Section 3.1（第 4–5 页）、Appendix B（第 16–18 页），重点是 Figure 3、Table 8、Table 9、Table 10 和 Figure 5。**

这一部分最值得研究的是：**作者为什么在构建 32 亿规模的预训练数据时，选择尽量保留图像，而不是继续用质量分数大量过滤数据？**

我们按照真实的数据处理顺序来学习。

## 一、预训练数据整体流程

先看论文 Figure 3，左侧是数据构建，中间是 Caption，右侧是渐进训练。

<!-- 图片待补：MOSS-Image Figure 3 预训练方案（原路径：MOSS_Part2_Figure_3.png） -->

作者的整体流程可以概括成：

**16 个公开数据源**

约 55 亿条图文记录

↓

**各数据源自身的预处理**

部分数据源已经做过 NSFW、水印等筛选

↓

**SSCD 图像去重 + AI 生成图像检测**

↓

**保留约 32 亿条数据**

不再执行额外的启发式质量过滤

↓

**Caption 重标注与结构化布局标注**

给模糊、水印等属性添加文本描述

↓

**多阶段 DiT 预训练**

注意：上面的流程区分了数据清理与文本标注两个环节。我们这次主要讨论前四步；Caption 如何生成、如何采样，会在 Part 3 详细学习。

## 二、16 个预训练数据源分别是什么？

作者主要采用三个超大规模数据源：**OmniCorpus、ReLAION 和 DataComp**，另外补充 13 个规模较小的数据源。

附录 Table 8 把每个来源的输入数量和清洗后数量列出来了。

| 数据源 | 输入记录 | 保留记录 |
|---|---:|---:|
| OmniCorpus | 32.1 亿 | 18.2 亿 |
| ReLAION | 19.5 亿 | 11.9 亿 |
| DataComp | 2.91 亿 | 1.67 亿 |
| MedTrinity-25M | 1850 万 | 1360 万 |
| optical-rs | 1320 万 | 735 万 |
| marqo | 981 万 | 421 万 |
| git_10m | 981 万 | 972 万 |
| megalith | 856 万 | 815 万 |
| commoncatalog | 789 万 | 778 万 |
| text_atlas | 540 万 | 287 万 |
| humancaption | 490 万 | 114 万 |
| textrecap | 242 万 | 138 万 |
| google_landmarks | 126 万 | 121 万 |
| wikiart | 8.14 万 | 7.67 万 |
| place365 | 3.65 万 | 3.46 万 |
| poster100k | 7090 | 6370 |

这张表有两个重要信息。

**第一，主要数据来自三个大数据源。**

清洗后数据占比（按 Table 8 近似统计）：

| 数据源 | 占比 | 数量 |
|---|---:|---:|
| OmniCorpus | 56.3% | 18.2 亿 |
| ReLAION | 36.8% | 11.9 亿 |
| DataComp | 5.2% | 1.67 亿 |
| 其余 13 个数据源 | 1.7% | 约 0.57 亿 |

前三个数据源合计占最终记录数约 **98.2%**。所以 MOSS-Image 的预训练主体仍然依赖大规模通用图文数据。

**第二，作者没有因为某个数据集很小就不使用它。**

例如 TextAtlas、Poster100K、WikiArt 等专门来源仍然被保留下来。从数据名称和原始来源看，它们覆盖文字图像、海报、艺术作品等不同内容。不过论文没有单独给出这些小数据源的增益消融，因此不能断定具体哪一个带来了多少文字或设计能力。

还有一个关键细节：**Table 8 统计的是进入作者数据流水线的记录数量，不是这些公开数据集在互联网上最初发布的全部数据量。** 作者采用各数据源原有的混合比例，没有报告额外的类别重采样权重。

## 三、数据是如何清洗的？

作者真正执行的主要清理操作只有两类：**SSCD 去重和 AI 生成图像检测**。但这不代表原始数据完全没有过滤。

### 3.1 SSCD：图像去重

SSCD（Self-Supervised Copy Detection）是一种用于检测图像副本或近重复图像的特征方法。

简单理解，假设数据集中有三张图片：

- A：一张猫的照片。
- B：A 的裁剪、缩放或重新压缩版本。
- C：另一张不同的猫照片。

理想情况下，SSCD 应当判断 A、B 高度重复，而不应该仅仅因为 C 也是猫就把它删除。

基本流程是：

$$
\text{Image}
\xrightarrow{\text{SSCD Encoder}}
\text{Feature}
\xrightarrow{\text{Similarity}}
\text{Deduplication}
$$

其核心目的，是减少同一张图像及其变体在训练数据中的重复出现，而不是对所有语义相近的图像进行合并。

**但论文并没有公开 SSCD 去重的具体相似度阈值、索引和检索实现、重复簇保留规则。** 因此我们可以理解它的技术原理，却不能仅凭这篇报告完整复现作者的去重流水线。

### 3.2 AI 生成图像过滤

作者还使用了一个内部训练的 AI-image detector，移除检测到的 AI 生成图片。

这里需要区分两种问题：

- 一张图像是不是 AI 生成的？
- 一张图像的质量好不好？

这是两件不同的事情。MOSS-Image 主要针对前者进行过滤，并没有仅因为一张图像审美分数低，就在后续数据准备阶段把它删除。

不过，论文没有披露内部检测器的架构、训练集、阈值或误检率，所以这一部分的工程细节并不充分。

### 3.3 为什么说它没有进一步做质量过滤？

这是 Part 2 最重要的一点。

通常，图像预训练可能采用各种质量指标，比如：

- 分辨率是否太低；
- 是否模糊、有压缩瑕疵；
- 亮度、饱和度是否异常；
- 水印占比是否过大；
- 图像是否足够自然。

MOSS-Image 在去重和 AI 图像过滤之后，**没有再使用这些启发式质量条件筛掉更多图片**。

但作者明确强调，这里的 *No additional data filtering* 有前提：部分数据源在进入统一流水线之前，已经做过自己的过滤，例如 ReLAION 的 NSFW 和水印检查。

因此，“不过滤”指的是**不增加下一轮质量过滤**，而不是把原始互联网图片不经任何处理全部保留下来。

## 四、作者究竟测试了什么质量过滤规则？

这部分非常有价值，因为作者不是简单地提出观点，而是构建了一组对照实验。

他们使用一个额外的 `pretrain_v1` 数据过滤流水线。附录 Table 9 给出了全部七个筛选条件。

| 指标 | 保留条件 | 通俗解释 |
|---|---|---|
| Width / Height | 均 ≥192 px | 去除尺寸过小的图像 |
| Aspect Ratio | ≥0.33 且 <3 | 去除极端长宽比 |
| Entropy | ≥4 | 过滤信息变化过少的图像 |
| Brightness | ≥0.1 且 <0.9 | 过滤过暗和过亮的图像 |
| Saturation | <0.8 | 过滤过度饱和的图像 |
| Watermark Area Ratio | <0.1 | 水印区域总占比小于 10% |
| Natural Score | ≥0.2 | 根据自然度评分进行筛选 |

这些条件之间采用 **AND 逻辑**，即适用的条件必须全部满足，一张图像才能留下。

这里还有一个实现细节：如果某个指标值为 NULL，则该字段检查允许通过，但其他条件仍然必须通过。因此它不是“缺少任何指标就丢掉整张图像”。

需要注意，论文只给出了上述阈值，没有详细交代 Entropy、Brightness、Saturation 的完整计算方式，也没有说明 Natural Score 的评分模型和具体定义。不要把我们熟悉的某种自然度评价方法直接当成作者的实现。

**ReLAION 额外质量过滤实验**

| | 数量 |
|---|---:|
| 过滤前 | 11.84 亿 |
| 过滤后 | 5.13 亿 |
| 保留比例 | 43.35% |
| 删除比例 | 56.65% |

也就是说，在已经去重、重标注的 ReLAION 数据上继续进行严格质量过滤，会让超过一半的数据消失。

## 五、过滤与不过滤，最终实验结果如何？

### 5.1 对照实验的具体配置

这里必须先强调：**作者并没有直接拿最终 5B 模型做这组消融，而是使用一个早期的小模型进行验证。**

| 参数 | 实验配置 |
|---|---|
| 模型架构 | 24 层 Multi-Stream DiT |
| Hidden Size | 1280 |
| FFN Hidden Size | 3072 |
| Attention Heads | 20 |
| Text Encoder | Qwen3.5-0.8B |
| Autoencoder | ViViT2-REPAE |
| 图像分辨率 | 256×256 |
| GPU 数量 | 32 |
| Global Batch Size | 1024 |
| Learning Rate | 1e-4 |
| Recaption 概率 | 0.8 |
| 训练步数 | 1.6M |
| Checkpoint 间隔 | 50K |

两组模型采用相同的主要训练配置。区别在于，一组使用已经去重和重标注的 ReLAION 数据，另一组在这个基础上继续使用七个条件过滤数据。

### 5.2 最终评估结果

| 指标 | 不额外过滤 | 额外过滤 | 结果 |
|---|---:|---:|---|
| DPG-Bench ↑ | 82.420 | 82.411 | 基本持平 |
| GenEval ↑ | **0.6484** | 0.6244 | 不过滤更好 |
| OneIG ↑ | **0.4363** | 0.3482 | 不过滤更好 |
| FDR6-LAION ↓ | **2.4126** | 3.2184 | 不过滤更好 |

其中，OneIG 是五个维度的平均分，包含 Alignment、Text、Reasoning、Style、Diversity。

而文字能力的退化尤其明显：

**OneIG Text Score**

| 不额外过滤 | 额外过滤 |
|---:|---:|
| **0.5243** | **0.1180** |

过滤后文字维度评分降低约 77.5%。注意这不是文字准确率，而是 OneIG 的文字子指标。

这表明，在这组实验中，额外质量过滤对文字生成能力造成了非常明显的负面影响。

### 5.3 不只是最终指标，训练曲线也有差异

这是附录 Figure 5 的完整曲线。

<!-- 图片待补：MOSS-Image Figure 5 数据过滤消融实验（原路径：MOSS_Part2_Figure_5.png） -->

蓝线是不过滤，橙线是额外过滤。

左上角 DPG-Bench：两组曲线非常接近，说明这个指标没有明显区分两种数据策略。

右上角 GenEval：蓝线在后期总体更高。

左下角 OneIG：差异最为明显，随着训练继续进行，不过滤模型逐渐拉开差距。

右下角 FDR6-LAION：越低越好，不过滤模型在记录的所有 32 个 Checkpoint 上均取得更低的分数。

FDR6-LAION 不是简单的美学分数。它使用 Inception-v3、CLIP、ConvNeXt-v2、DINOv2、MAE 和 SigLIP2 六组视觉表征衡量生成图像与真实图像参考分布的差异。

为了评估该指标，作者在每个 Checkpoint 使用 50,000 张生成图像，并对 CFG 取值 `{1, 1.25, 1.5, 1.75, 2, 3, 4, 5}` 进行扫描，报告最佳分数。两组最终 Checkpoint 的最佳 CFG 都为 3。

论文也提醒，由于两组模型的训练数据分布不同，FDR6 的比较不一定完全公平，因而更适合作为趋势参考，而不是唯一结论。

## 六、为什么过滤高质量数据，反而让模型变差？

作者的直接实验结论是：

**大规模预训练中，保持广泛的图像分布覆盖，可能比继续进行严格的启发式质量筛选更重要。**

我认为可以从三个角度理解，但需要说明，这些属于对实验现象的机制分析，并不是论文逐项验证过的因果结论。

首先，很多复杂图像本身就不一定满足简单的自然图像质量标准。例如，一张信息图可能包含大量小字、纯色背景、不常见的长宽比以及复杂的颜色区域。这样的图像对文字和排版生成有价值，却可能因为部分统计特征而被错误过滤。作者观察到文字指标退化严重，与这种解释相容，但没有公布被删除图像的详细类别分布。

其次，过滤损失的不只是低质量图像，还有数据多样性。在相同训练步数和 Batch Size 下，5.13 亿数据集也更容易重复采样已有内容。因而这项实验实际上同时改变了数据质量分布和数据覆盖规模，无法单独区分二者的作用。

最后，作者提出了一种不同的监督思路：

### Annotate Rather Than Filter

例如，训练集中存在两张图像。

第一张图像清晰，色彩自然。第二张图像有水印，而且画面模糊。

传统严格质量过滤可能直接删除第二张，而 MOSS-Image 希望通过 Caption 显式描述它的属性，例如：

`The image quality is blurry and pixelated.`

或者：

`There is a watermark at the bottom left.`

这样，模型不仅看到图像，还能获得描述图像缺陷的条件信息。

其基本思路是：**不要完全依靠删除样本控制生成分布，而是通过文本监督让模型学会区分不同视觉属性。**

但这同样有一个边界：论文提供的是过滤策略的整体对照，没有单独进行“保留缺陷图像但不标注”与“保留缺陷图像且标注”之间的因果消融，因此尚不能量化缺陷标注本身的独立增益。

## 七、Part 2 最值得记住的结论

我认为这部分有三项核心收获。

**第一，MOSS-Image 的预训练数据规模非常大，但主要由三个通用数据源支撑。** 作者没有披露针对海报、文字图像等领域的大幅度重采样策略，所以不能简单认为它的文字能力来自某个小规模专门数据集。

**第二，它区分了清理无效数据与筛选图像质量。** 去重和 AI 图像过滤依然存在，主要争议在于是否应继续用亮度、饱和度、水印、自然度等指标严格筛选。

**第三，过滤策略必须经过实际训练验证。** MOSS-Image 的消融证明，这套额外过滤方案在早期模型上效果更差，尤其损害文字生成能力。但不能把这一结论无限扩展为“任何模型、任何阶段都不需要质量过滤”。

另外有个容易忽略的信息：作者在附录 B 中明确说明，**预训练 Caption 主要是英文，没有专门进行中文 Caption 标注**。这一点与模型最终展示的较强中文长文本渲染能力放在一起，非常值得在下一部分研究。

---

## 下一步：Part 3 — Caption 与结构化标注

我认为 Part 3 比 Part 2 更有技术含量，因为它直接关系到图像模型如何学会文字和空间控制。

下一部分我们会重点搞清楚：Gemma4 如何生成 Short、Medium、Long Caption，原始 Caption 与 Recaption 如何混合采样，Layout JSON 如何标注文字、Bounding Box 和字体属性，标注错误如何检查，以及为什么需要字段 Dropout、坐标变换和保留换行符。

**Part 2 的核心就是：先保留足够丰富的图像分布，再通过更精细的标注来提高监督信息质量。Part 3 就是研究后半句话究竟如何实现。**

---

# Part 3：Caption 生成与结构化标注

**对应论文：Section 3.2（第 5–6 页）、Appendix C（第 18–20 页）。**

这部分比模型架构更值得仔细学习。MOSS-Image 的核心设计是：**同一张图像，不仅使用不同详细程度的 Caption 训练，还使用包含文字、位置和排版信息的结构化 JSON 训练。**

我们要弄清楚四件事：Caption 怎么制作、训练时怎么采样、Layout JSON 怎么构建，以及为什么还要做坐标变换和字段 Dropout。

## 一、整体数据标注流程

先看 Figure 3 中的数据标注部分。

<!-- 图片待补：MOSS-Image Figure 3 数据标注流程（原路径：moss_part3_figure_3.png） -->

对一张训练图像，作者构建两类文本监督：

**原始图像 + 原始 Caption**

↓

**普通 Caption**：Short（一句话概括）、Medium（中等详细描述）、Long（完整视觉描述）

**结构化 Caption**：整体场景描述、对象 / 文字内容、BBox / 排版 / 颜色

↓

**训练时按相应策略选择文本条件**

统一通过 Qwen 文本编码器输入 DiT

需要注意，并不是所有图像都拥有全部标注。三种 Recaption 根据生成结果的可用情况采样，而结构化 JSON 只覆盖选中的部分数据。

## 二、普通 Caption：Short、Medium、Long 怎么生成？

### 2.1 用什么模型生成？

论文明确使用 **Gemma 4** 生成不同语义粒度的 Caption。

输入不仅是一张图像，还包含原始 Caption 和描述视角（descriptive perspective）。

生成流程是：先得到详细的 Long Caption，再从中浓缩出 Medium Caption，同时提供 Short Caption 作为简洁概括。论文没有公开完整的 Gemma 4 提示词模板。

三种 Caption 之间主要区别是描述的信息量，而不是严格限制 Token 数量。

举个便于理解的例子，假设图像是一张写着 `SALE` 的促销海报。

**Short Caption：**

> A promotional poster featuring a large SALE headline.

只描述最核心内容。

**Medium Caption：**

> A promotional poster with a bold SALE headline at the top, a red circle below it, and a clean white background.

增加颜色、对象、位置关系。

**Long Caption：**

> A minimalist promotional poster with a white background. The word SALE appears in large, bold black letters near the upper center. A solid red circle is positioned below the headline...

进一步描述构图、主体外观、文字内容、颜色及空间关系。

这三个是我为解释机制构造的例子，不是论文原始的 Gemma 4 输出。

### 2.2 三种 Caption 实际有多长？

作者从 94 个数据分片抽样了 5,792 张图像，其中 4,335 张具有完整的 Short、Medium、Long 三元组。

使用 Qwen3.5-9B Tokenizer 统计，得到：

| Caption | 平均 Token | 中位数 | P90 |
|---|---:|---:|---:|
| Short | 20.86 | 20 | 26 |
| Medium | 76.61 | 75 | 97 |
| Long | 235.12 | 193 | 293 |

这里有两个值得注意的地方。

首先，Long Caption 并没有刻意做成上千 Token，平均只有约 235 Token，但抽样中最长的 Long Caption 达到了 3,126 Token。

其次，这些数据是样本统计，不代表全部 32 亿图像都完成了三个粒度的重标注。

### 2.3 最关键的细节：可见文字要原样抄写

作者对图片中的文字提出了明确要求：

**保留原始文字，而不是翻译或重新组织。**

例如图像中写着：

```text
ALL YOU
NEED IS
LOVE
```

Caption 不能只写：

`The image contains the phrase "All you need is love".`

更精确的标注应该保留：

```text
ALL YOU
NEED IS
LOVE
```

此外，还要描述文字的阅读顺序、位置、对齐方式、字体、颜色和间距。

这很重要，因为文字渲染不仅要求模型知道有哪些字符，还要求模型知道这些字符应该怎么排列。

论文在 Section 3.2 特别强调：

**Text line breaks are also a form of layout.**

也就是：换行本身也是布局信息。

## 三、这些 Caption 在训练时怎么采样？

这是一个非常实用的训练细节。

在标准 Caption 分支，如果一张图同时有原始 Caption 和 Recaption，作者采用：

- 20% 概率使用原始 Caption；
- 80% 概率使用生成的 Recaption。

如果三个粒度的 Recaption 都存在，则在 80% 的分支内均匀采样 Short、Medium、Long。

因此，完整情况下的概率为：

| 文本条件 | 训练采样概率 |
|---|---:|
| Original Caption | 20% |
| Short Caption | 26.67% |
| Medium Caption | 26.67% |
| Long Caption | 26.67% |

如果某个 Caption 不存在，训练时会从当前可用的描述中选择，而不是强行构造缺失版本。

为什么要这样混合？可以从监督的角度理解：

Short Caption 让模型适应简洁指令；Medium Caption 提供更多视觉描述；Long Caption 提供更详细的视觉信息；Original Caption 则保留原始数据中的描述风格。

这里必须强调一点：**80:20 是普通 Caption 分支内部的比例，并不表示整个预训练有 80% 普通 Caption、20% Layout JSON。** 结构化标注另有按领域和标注可用性确定的采样策略，论文没有给出一个统一的全局混合比例。

## 四、结构化 Caption：Layout JSON 怎么制作？

这可能是 MOSS-Image 的关键技术之一。

普通 Caption 即使很长，也不一定能准确表达每个对象的位置。比如：

`A red circle below SALE.`

这句话虽然告诉模型圆形在文字下面，但没有给出文字区域、圆形大小和具体位置。

因此，作者对部分图像额外构建 **包含 Bounding Box 的结构化标注**。

### 4.1 两种主要 JSON 格式

论文 Appendix C.3 给出了两类主要的结构化格式。

<!-- 图片待补：MOSS-Image Figure 6 两种 Layout JSON 格式（原路径：moss_part3_figure_6.png） -->

左边是 **Scene/element**，右边是 **Summary/items**。

两种格式都描述同一张图像，只是组织字段的方式不同。

Scene/element 更强调全局场景、视觉元素、空间关系和配色。Summary/items 则把图像整理成总体描述和若干局部元素，包含每个 Item 的类别、描述、坐标等。

以左边的 Scene/element 为例：

| 字段 | 作用 |
|---|---|
| `scene_type` | 场景类型，例如 poster |
| `gen_brief` | 对整个画面的概括 |
| `layout` | 构图、视角等信息 |
| `elements` | 各个对象和文字元素 |
| `bbox` | 元素的具体位置 |
| `anchors` | 元素之间的空间关系 |
| `palette` | 画面的主要颜色 |

在 `elements` 内，`type=obj` 描述图像对象，包括身份、形状、颜色、显著性；`type=text` 描述文字内容、位置、字形和是否属于水印。

比如 Figure 6 中的 `SALE`，模型可以同时获得准确文字、顶部位置和粗体黑色字形这些信息。

### 4.2 Bounding Box 如何表示？

作者使用的坐标顺序是：

$$
[y_{\min},x_{\min},y_{\max},x_{\max}]
$$

注意，**这里是 YXYX，不是很多目标检测框架常用的 XYXY。**

原始存储的坐标被归一化到 `[0,1000]`。

例如，Figure 6 中的文字：

```json
{
  "type": "text",
  "text": "SALE",
  "bbox": [80, 250, 200, 750]
}
```

在 1024×1024 画布上，对应：

$$
x\in[256,768],\qquad y\in[82,205]
$$

也就是说，`SALE` 应该位于画面靠上的中间区域。

这种监督让模型同时学习**文字是什么，以及文字应该出现在哪里**。

## 五、训练前为什么还要变换坐标和丢弃字段？

这一部分是论文比较有价值的工程细节。

### 5.1 三种 Bounding Box 坐标表示

作者没有只用一种坐标格式，而是混合三种：

| 坐标形式 | 表达方式 | Scene/element 采样概率 |
|---|---|---:|
| 归一化坐标 | `[0,1000]` | 40% |
| 绝对像素坐标 | 实际像素位置 | 30% |
| 中心相对像素坐标 | 以图像中心为原点 | 30% |

中心相对坐标的计算是：

$$
[y_1-H/2,\ x_1-W/2,\ y_2-H/2,\ x_2-W/2]
$$

比如一张 1024×1024 图像，中心位置的坐标为 (512,512)。一个元素如果位于画面中心附近，它的中心相对坐标就接近零。

对于归一化坐标，论文还规定一半带明确的坐标格式标记，一半不带标记。

此外，作者还提供第三种补充标注方式：**在普通自然语言 Caption 后附加 Item-level 标注**，里面包含 BBox、Label、Tags 和 Caption。这一分支的坐标采样比例与前面不同：

$$
P(\mathrm{normalized},\mathrm{absolute},\mathrm{center})
=(0.2,0.4,0.4)
$$

作者在附录 K 的案例中观察到，在训练未见过的更高分辨率上，中心相对像素坐标比 `[0,1000]` 归一化坐标更好地保持了布局的空间尺度。不过这是定性案例，并非充分的大规模泛化消融。

### 5.2 字段 Dropout：不是每次都使用完整 JSON

如果训练时始终提供完整的布局、颜色、字体、坐标信息，模型可能过度依赖这些详细条件。

作者因此使用 Stochastic Field Dropout，让同一类结构化监督在训练中呈现不同的信息完整程度。

附录 Table 12 给出了具体规则。

| 标注格式 | 随机处理规则 |
|---|---|
| Scene/element | 以 50% 概率开启字段 Dropout；开启后，各个指定字段组以 50% 概率丢弃 |
| Scene/element | `gen_brief` 和 `elements` 始终保留 |
| Scene/element | 随机打乱元素顺序；以 50% 概率替换外观描述字段名 |
| Summary/items | 100% 删除 ID 与 Parent；10% 概率删除整个 Summary |
| Summary/items | 保留 Summary 时，`meta` 以 10% 概率丢弃，`canvas`、`caption` 各以 50% 概率丢弃 |
| Summary/items | 保留下来的 Summary Caption，有 50% 概率替换成基础 Prompt |
| Summary/items | Palette 等可选字段以 50% 概率丢弃；Tags/Caption 在“两者都有、只有 Tags、只有 Caption”之间等概率选择 |
| Summary/items | 50% 概率打乱 Items 的顺序 |

这里需要理解一个简单例子。

在 Scene/element 分支，字段 Dropout 首先以 50% 概率被激活，激活后指定字段组再以 50% 概率被删除。

所以，对于参与这两级随机操作的某个字段组，其总体删除概率是：

$$
0.5\times0.5=0.25
$$

也就是 25%。

**这种训练方式使模型既能处理非常详细的结构化 Prompt，也能处理缺少部分属性的 Prompt。**

另外，作者还会在图像执行 Resize、Crop、Flip 时同步修改 BBox 坐标，裁剪后完全消失的元素会被删除，避免图像内容与坐标监督不一致。

## 六、标注质量如何保证？

作者没有直接把所有大模型生成的 Caption 和 JSON 无条件送入训练。

附录 C.2 给出了明确的结构校验机制。

对于普通 Caption，主要检查 JSON 语法和三个非空描述是否存在。

对于 Layout JSON，检查内容更多，包括必需字段、BBox 是否由四个整数构成、坐标是否在有效范围内、BBox 面积是否大于零、元素 ID 是否唯一、父元素引用是否有效，以及颜色格式是否正确。

由 Gemma 和 Qwen 系模型生成的标注，如果不满足结构约束，最多允许 **3 次重试**。

需要说明的是，这些校验主要保证的是**格式合法性与几何合理性**。论文并没有证明它能完全消除图像文字识别错误、对象遗漏或语义幻觉。

## 七、最终这些标注怎么送入 DiT？

这是理解前面所有设计的最后一步。

**MOSS-Image 没有另设一个专门接收 BBox 的控制编码器。**

作者在训练时先选择相应的文本条件，将 Caption、结构化标注和相关元数据组合，然后应用 Qwen Chat Template，最终交给冻结的 Qwen3.5-9B Text Encoder 编码。

流程是：

$$
\text{Caption / JSON / Metadata}
$$

$$
\downarrow\quad\text{Serialization + Chat Template}
$$

$$
\text{Qwen3.5-9B Text Encoder}
$$

$$
\downarrow
$$

$$
\text{Text Adapter}\rightarrow\text{DiT}
$$

最终训练阶段的完整文本条件上限为 **4096 Tokens**，包含格式化后所有条件字段，而不仅仅是 Caption 本身。超过上限会截断。

这意味着模型通过同一条文本条件通道学习两种能力：一方面理解普通自然语言描述，另一方面理解包含精确位置、文字与元素信息的 JSON。

推理阶段的 Agent 只需要生成符合接口约定的布局描述，就可以调用已经学会执行这些条件的 DiT。

---

## Part 3 总结

我认为这一部分真正值得吸收的，不是简单地“使用了 Long Caption”，而是下面三个训练思路。

**第一，多粒度 Caption 不是越长越好。** MOSS-Image 将短、中、长描述混合训练，并保留 20% 原始 Caption，使模型能够适应不同信息密度的条件输入。

**第二，精确文字和布局要通过结构化监督学习。** 尤其是原样转录文字、保留换行、标注 Bounding Box，以及明确元素之间的空间关系。

**第三，标注本身也需要数据增强。** 作者不仅进行图像几何增强，还对坐标形式、字段、顺序和描述方式进行随机化，使模型不会只适应一种固定的 JSON 写法。

不过，这篇论文没有提供多粒度 Caption、结构化 JSON、字段 Dropout 各自独立贡献的完整消融。因此，我们能明确学习到它的训练方案，但还不能给每项技术精确分配性能增益。

**Part 3 到这里就完成了。**

接下来的 **Part 4：Pretraining Schedule and Optimization** 会深入具体训练过程，尤其是 Flow Matching 的损失归一化方式、噪声采样分布为什么不断变化、Stage 1/2/3 如何逐步增加分辨率与文本长度，以及 Packing 和有效 Batch 到底怎么计算。这部分会更偏训练代码和实际参数配置。

---

# Part 4：MOSS-Image 预训练流程与训练策略

**对应论文：Section 3.3（第 6 页）、Appendix A.1–A.2（第 14–16 页），重点是 Table 1、Table 6、Table 7，以及公式（3）（4）。**

这一部分的核心不是模型架构，而是**一个 5B DiT 究竟怎样从随机初始化开始训练，直到支持 1024 分辨率、4096 Text Tokens 和复杂图像生成。**

我们沿着一次真实的训练流程讲，重点解决五个问题：训练阶段怎么划分、Loss 怎么计算、噪声如何采样、Batch 如何组织，以及具体用了什么训练参数。

## 一、多阶段渐进式预训练

MOSS-Image 并不是一开始就采用高分辨率和长文本训练，而是逐渐增加图像分辨率与文本长度。

论文 Table 1 给出以下训练阶段。

| 阶段 | 图像分辨率 | Text Token 上限 | 训练步数 |
|---|---|---:|---:|
| Stage 1 | 256-pixel level | 320 | 2.1M |
| Stage 2 | 先 512，后混合 256/512 | 512 → 1024 → 2048 | 2.1M |
| Stage 3 | 混合分辨率，面积 ≤1024² | 4096 | 370K |
| Stage 4 | 混合分辨率，面积约 ≤2×1024² | 4096 | 520K |

来自论文 Table 1。这里有报告口径不一致的问题，下面专门解释。

### 1.1 Stage 1：256 分辨率预训练

第一个阶段使用 256 像素级别的图像，Text Token 上限为 320。

结合 Part 1 学过的 VAE 和 Patchify：

$$
N_{\text{image}}=\frac{256}{16}\times\frac{256}{16}=256
$$

因此，一条样本最多包含：

$$
256+320=\boxed{576\text{ Tokens}}
$$

这里训练序列比较短，Attention 的计算成本相对较低。

可以将这一阶段理解为基础视觉生成能力的学习阶段。不过论文并没有单独证明 Stage 1 各类能力的学习顺序，我们不能断言它在此阶段就完成了哪些具体能力。

### 1.2 Stage 2：提高分辨率和文本长度

第二阶段首先提高到 512 分辨率，然后混合 256、512 两种分辨率。

文本长度也逐渐增加：

$$
512\rightarrow1024\rightarrow2048
$$

一张 512×512 图像对应：

$$
N_{\text{image}}=1024
$$

所以最大训练序列达到：

$$
1024+2048=\boxed{3072\text{ Tokens}}
$$

这一阶段的主要变化有两个：图像可以包含更多细节，文本条件也可以包含更丰富的内容。

论文没有公开 Stage 2 中三档文本长度具体在第多少步切换，也没有披露 256/512 混合采样的完整比例。

### 1.3 Stage 3：高分辨率与长文本预训练

这一阶段的目标图像面积上限为：

$$
H\times W\leq1024^2
$$

注意，**并不是所有图像都被强制缩放到 1024×1024**。

作者会保留长宽比，并把图像尺寸对齐到 16 的整数倍。因此，横图、竖图和正方形图像都可以参与训练。

文本长度上限增加到 4096 Tokens。

对一张 1024×1024 图像：

$$
N_{\text{image}}=4096
$$

于是：

$$
N_{\text{total}}=4096+4096
=\boxed{8192\text{ Tokens}}
$$

这也解释了 Figure 3 中三个阶段的最大 Token 数量为什么分别是 576、3072、8192。

### 1.4 这里有一个值得注意的论文记录问题

正文和附录 A.2 都将训练描述为三个阶段，附录明确说明：

**最终 Stage 3 的 EMA Checkpoint 取自第 310,000 步。**

Figure 3 也写的是 Stage 3 训练 0.31M Steps。

但是 Table 1 却列出了 Stage 3 的 370K Steps，以及额外的 Stage 4（520K Steps）。

这几个记录并不完全一致。

因此，后续分析最终模型的训练设置时，我们应优先依据附录明确对应的 **Stage 3、310K Checkpoint**。Table 1 中的 Stage 4 可以作为报告列出的额外训练阶段，但不能未经证实就认为它一定对应论文所有最终评测使用的模型。

## 二、Flow Matching：每一步训练到底在做什么？

这一部分理解清楚，基本就知道了 MOSS-Image 的基础训练过程。

### 2.1 构造带噪 Latent

假设一张训练图片经过冻结的 FLUX.2 VAE，得到干净的 Latent：

$$
z_0=\mathrm{VAE.encode}(I)
$$

再采样标准高斯噪声：

$$
\epsilon\sim\mathcal N(0,I)
$$

然后采样一个噪声水平 $\sigma$，构造：

$$
\boxed{
z_\sigma=(1-\sigma)z_0+\sigma\epsilon
}
$$

这个表达式与论文的 Velocity Target 一致。

- $\sigma=0$：完全干净的图像 Latent。
- $\sigma=1$：完全随机的噪声。
- $\sigma=0.5$：图像与噪声的线性混合。

### 2.2 模型需要预测什么？

模型的输入是：

$$
v_\theta(z_\sigma,1000\sigma,c)
$$

其中 $c$ 是选中的 Caption 或结构化文本条件。

训练的目标速度为：

$$
\boxed{v_{\text{target}}=\epsilon-z_0}
$$

也就是说，模型不是直接预测 RGB 图像，也不是直接预测干净 Latent，而是预测 Flow Matching 路径上的速度。

这里的 `1000σ` 是输入模型的时间条件，而不是表示每张训练图片都要执行 1000 次去噪。

### 2.3 论文实际使用的 Loss

附录公式（3）是：

$$
\boxed{
\mathcal L_{\mathrm{FM}}
=
\mathbb E\left[
\frac{1}{\sum_i N_i}
\sum_i\sum_{p=1}^{N_i}
\frac{1}{C}
\left\|
v_{\theta,i,p}-
(\epsilon_i-z_{0,i})_p
\right\|_2^2
\right]
}
$$

其中：

- $i$：一个 Packed Batch 中的第 $i$ 张图像。
- $N_i$：该图像有效的 Image Token 数量。
- $p$：Image Token 索引。
- $C$：每个输出 Token 的通道维度。
- $v_{\theta,i,p}$：预测速度。

这里真正值得注意的，不是 MSE 本身，而是 **Loss 的归一化方式**。

作者在最终阶段采用 Per-token Normalization，而不是每张图片先分别求平均再平等加权。

假设一个 Batch 里只有两张图：

| 图像 | 分辨率 | Image Tokens | Loss 权重 |
|---|---|---:|---:|
| A | 512×512 | 1024 | 20% |
| B | 1024×1024 | 4096 | 80% |

如果采用 Per-sample Mean，两张图各占 50%。

而按照 MOSS-Image 最终阶段的 Per-token Normalization，1024 分辨率图像拥有更多有效 Image Tokens，因此对总体 Loss 的贡献更大。

**这意味着：混合分辨率训练不仅改变了计算量，也改变了不同分辨率样本在梯度中的实际权重。**

论文还说明，Stage 2 随着 Packing 配置扩展，从 Per-sample Normalization 逐步切换到 Per-token Normalization。

此外，Loss 有两个明确的实现细节：

1. 只在 Image Tokens 上计算预测误差，Text Tokens 不参与 Loss。
2. 不再额外乘以 $\sigma$ 或 SNR 相关的 Loss Weight。

第二点不代表不同噪声水平同样频繁出现，因为作者通过下一节的**噪声采样分布**控制不同噪声水平的训练频率。

## 三、最重要的训练 Trick：噪声采样策略

这一部分很值得仔细看。

在 Flow Matching 中，虽然可以从 0 到 1 均匀采样噪声水平，但不同噪声水平对应的学习任务不同。

$\sigma$ 接近 1 时，输入几乎是纯噪声；$\sigma$ 接近 0 时，输入已接近干净图像。

MOSS-Image 没有始终使用同一种采样分布，而是在 **Stage 3 的 310K 训练步中，分四个时间区间调整噪声分布**。

### 3.1 先看全部参数

| Stage 3 步数 | Shift | Logit-Normal $(\mu,s)$ | Cutoff | 全噪声额外采样 |
|---|---:|---|---:|---:|
| 1–140K | 3 | (0, 1) | 无 | 1% |
| 140K–207K | 1 | (-1.45, 1.05) | 无 | 1% |
| 207K–274K | 1 | (-1.45, 1.05) | 0.95 | 0 |
| 274K–310K | 1 | (-1.09, 0.95) | 0.95 | 0 |

论文 Appendix A.2，Table 6。最后一个区间另外混合 1% 离散均匀采样。

这里有三个技术概念需要理解：Logit-Normal、Shift、Cutoff。

### 3.2 Logit-Normal：控制不同噪声水平的出现概率

作者首先采样：

$$
Z\sim\mathcal N(\mu,s^2)
$$

然后构造：

$$
\boxed{\sigma=1-\operatorname{sigmoid}(Z)}
$$

这里为了便于理解，先不考虑后续的离散化和截断处理。

当 $\mu=0,s=1$ 时，噪声分布相对居中。

而当 $\mu=-1.45$ 时，$Z$ 整体向负数移动。

由于：

$$
\sigma=1-\operatorname{sigmoid}(Z)
$$

所以负的 $\mu$ 会使噪声水平整体偏高。

例如将 $Z$ 取为均值，可以得到：

$$
\begin{aligned}
\mu=0 &\Rightarrow \sigma=0.50\\
\mu=-1.45 &\Rightarrow \sigma\approx0.81\\
\mu=-1.09 &\Rightarrow \sigma\approx0.75
\end{aligned}
$$

注意，这里计算的是将 $Z$ 设为均值时对应的噪声水平，不是整个分布的平均噪声。

因此，我们可以理解作者为什么要改变 $\mu$：它能够调整模型看到高噪声和低噪声样本的频率。

### 3.3 Shift：进一步改变噪声时间分布

第一段训练使用 Shift=3，之后使用 Shift=1。

论文附录 J 给出过 Shift 的具体形式：

$$
\boxed{
\sigma'=\frac{s\sigma}{1+(s-1)\sigma}
}
$$

这里的 $s$ 是 Shift 系数，不是 Logit-Normal 的标准差。

例如，当基础噪声水平为 0.5：

$$
s=3\Rightarrow\sigma'=0.75
$$

$$
s=1\Rightarrow\sigma'=0.5
$$

因此，Shift=3 会把中间位置的噪声水平向高噪声侧移动。

对于 Stage 3 的第一段训练，作者同时使用 $(\mu,s)=(0,1)$ 的 Logit-Normal 和 Shift=3 的噪声网格。

这里需要区分：附录 J 直接给出的是采样过程中的 Shift 公式，而 Table 6 给出的是训练噪声调度的 Shift 配置。二者采用相同的 Shift 概念，但论文没有完整展示训练调度器的所有实现代码。

### 3.4 Cutoff=0.95 究竟是什么？

这是一个容易误解的地方。

Cutoff=0.95 **不代表模型从此不再训练大于 0.95 的噪声水平**。

论文的实际做法是先对 Logit-Normal 分布截断，再重新映射到完整的离散噪声网格。

具体来说，最终区间使用：

$$
S=1-\operatorname{sigmoid}(Z)
$$

截断至：

$$
S\leq a,\qquad a=0.95
$$

然后重新映射到包含 1000 个离散时间点的完整网格：

$$
\left\{0.001,0.002,\ldots,1.000\right\}
$$

最终的概率分布为：

$$
\boxed{
p_j=(1-\rho)
\frac{F(aj/T)-F(a(j-1)/T)}{F(a)}
+\frac{\rho}{T}
}
$$

其中 $T=1000$、$a=0.95$、$\rho=0.01$。

这个公式可以简单理解为：**99% 的概率来自经过截断和重新映射的 Logit-Normal 分布，1% 的概率来自均匀分布。**

因此，即使某些噪声水平在主分布中出现得比较少，均匀分布也能给它们提供一定的采样机会。

### 3.5 如何理解这四次调整？

从训练策略的角度，我会这样理解：

第一段使用 Shift=3，同时额外采样 1% 的完全噪声输入，让训练较多接触高噪声状态。

第二段把 Shift 降为 1，同时将 Logit-Normal 的均值改为 -1.45，重新调整噪声分布。

第三段保持均值和标准差不变，但加入 0.95 的 Cutoff，改变采样分布的尾部形态。

第四段再次调整 Logit-Normal 参数，并混入 1% 均匀采样。

这是对调度设计的解释，**论文并没有对四次改变分别进行独立消融**，所以不能直接说哪一段负责提升构图、文字或细节质量。

这一部分的重要收获是：MOSS-Image 不只采用 Flow Matching，还在训练后期持续调整时间步采样分布。

---

## 四、Packing、Batch 和梯度累积

这是另一个非常实用的工程细节。

为什么图像训练需要 Packing？

因为不同分辨率的图片、不同长度的 Caption，会产生长度差异很大的训练序列。

例如：

| 样本 | Image Tokens | Text Tokens | 总计 |
|---|---:|---:|---:|
| A：256² 图像 | 256 | 200 | 456 |
| B：512² 图像 | 1024 | 500 | 1524 |
| C：1024² 图像 | 4096 | 3000 | 7096 |

如果所有样本都填充到最大长度，会造成大量无效计算。

作者因此采用 Multi-sample Packing，把多个样本组织到一个固定 Token Budget 内。上述数值只是示意，并不是论文披露的实际 Batch 样例。

### 4.1 实际配置是多少？

附录明确给出了以下参数：

| 参数 | 配置 |
|---|---:|
| 单个样本最大 Text Tokens | 4096 |
| 单个 Pack 的 Token Budget | 8576 |
| Static Execution Length | 8588 |
| 设备数量 | 512 |
| Gradient Accumulation | 2 |

因此，每次参数更新的理论容量为：

$$
8576\times512\times2
$$

$$
\boxed{8,781,824\text{ Token Slots/Update}}
$$

注意，这不是说每次训练一定有 878 万个有效 Token，更不是 878 万张图像。

它表示理论上的 Token 容量。真实的有效 Token 数量，以及每次更新包含多少张图像，都取决于 Packing 的具体情况。

### 4.2 Packing 中的几个实现细节

论文说明，文本条件会先经过 Qwen Chat Template，再进行 Tokenization。

超过 4096 Tokens 的条件被截断。短条件使用 Right Padding，并使用 Attention Mask 标记有效文本 Token。

对于多个样本如何拼接、跨样本 Attention 如何隔离，以及具体采用什么 Pack 分配算法，报告没有披露完整实现。

因此，**可以确定作者使用了 Packing，但还不能从报告直接复原完整的 Packing DataLoader 和 Attention Mask 代码。**

另外，8588 是静态执行长度，而 8576 是 Packing Budget。论文没有明确解释两者相差的 12 个位置具体用于什么，不能直接认定它们全部是特殊 Token。

## 五、优化器和其他训练参数

这一部分先把配置记录下来。Muon 的详细原理和消融我们放到 Part 5 学习。

| 超参数 | MOSS-Image 配置 |
|---|---|
| Optimizer | Hybrid Muon–AdamW |
| AdamW Learning Rate | 5e-5 |
| Muon Learning Rate | 5e-5，RMS-matched |
| AdamW Betas | (0.9, 0.95) |
| AdamW Epsilon | 1e-8 |
| Muon Momentum | 0.95 |
| Muon Newton–Schulz | 5 Steps |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |
| EMA | 0.9999 |
| Warmup | 2000 Steps |

这里还有两个细节。

首先，Muon 并不是作用于全部参数，而是主要用于符合条件的二维权重矩阵；其他参数依然交给 AdamW。Fused QKV、Gated FFN 矩阵在进行 Muon 更新前会拆分。

其次，Bias、Normalization、Embedding 和一维参数等不使用普通 Weight Decay。最后一层 Transformer 的 Q/K Normalization 权重还有单独的 Decay-to-one 规则。

这两个细节在复现训练优化器时都不能忽略。

---

## 六、把一次训练迭代完整串起来

到这里，我们可以把 MOSS-Image 的训练过程还原成下面的逻辑。

**1. DataLoader**

读取图像，并按策略采样 Caption 或 Layout JSON

↓

**2. VAE + Text Encoder**

图像得到干净 Latent，文本得到 Token Embeddings

↓

**3. Noise Sampling**

根据当前训练阶段的分布采样 σ，并生成噪声 Latent

↓

**4. Packing + DiT Forward**

图文 Tokens 经过 42 层 Single-Stream Transformer

↓

**5. Flow Matching Loss**

只计算 Image Tokens 的 Velocity MSE，并进行有效 Token 归一化

↓

**6. Backward + Optimizer Step + EMA**

梯度累积、Muon–AdamW 更新与 EMA 权重维护

这就是当前论文所描述的核心预训练流程。

整个过程没有报告额外的 Reward Model、GRPO 或独立 RL 阶段。

## 七、Part 4 学习总结

我认为这一部分最值得掌握的是三个训练设计。

**第一个是 Progressive Training。** 作者从较小的图像与较短的文本条件开始，逐渐扩展到 1024 像素级别、4096 Text Tokens。它同时降低了前期计算负担，并逐步提高模型需要处理的输入复杂度。

**第二个是噪声时间分布的分阶段调整。** 尤其是 Stage 3 中从 Shift=3 到 Shift=1，以及 Logit-Normal 参数和 Cutoff 的变化。这是一套有具体参数、可以进一步研究的时间步采样策略。

**第三个是 Token-level Training。** 不同分辨率图像具有不同数量的 Image Tokens，最终阶段按照有效 Token 总量归一化 Loss，同时采用 Packing 来组织变长图文序列。这些设置直接影响梯度权重、计算效率和实际训练 Batch。

我对 Part 4 的判断是：**它披露了不少值得学习的工业级训练配置，但对于分阶段噪声采样为什么有效，还缺少充分的独立消融。** 训练配方是明确的，单项策略的因果收益则不能直接从最终模型性能中推断出来。

下一部分进入 **Part 5：Muon–AdamW 优化器与收敛效率**。重点是它为什么比 AdamW 收敛更快、矩阵参数如何更新，以及作者报告的减少约 29%–33% 训练步数究竟是在什么实验条件下获得的。
