# 理解优先的 Agentic 图像基模 Boogu-Image-0.1

> **标签**：`Vision` `Diffusion` `DiT` `Flow Matching` `Agentic` `Prompt Rewriter` `Data-centric` `Timestep Sampling` `RL` `Distillation`
> **更新时间**：2026-10-09
> **参考来源**：[Boogu-Image-0.1: Boosting Open Agentic Multimodal Generation via Understanding under a Minimal Budget (arXiv:2607.13125v2)](https://arxiv.org/abs/2607.13125) · [GitHub: Boogu-Project/Boogu-Image](https://github.com/Boogu-Project/Boogu-Image) · [HF: Boogu/Boogu-Image-0.1-Base](https://huggingface.co/Boogu/Boogu-Image-0.1-Base)
> **精读进度**：§1 Introduction ✅ ｜ §2.1 评测反思 ✅ ｜ §3.1.1–§3.1.4 ✅ ｜ §3.2.1–§3.2.3 ✅ ｜ 附录 B.2 宏观架构 / B.2.1 Reasoner / B.2.2 Encoder+DiT / B.3 微架构 / B.5 训练配置 ✅ ｜ 附录 C Boosted Orthogonal Guidance ✅
> **未覆盖**：§2.2.2–§2.2.5 全部评测数值表（Table 1–6 的完整对比列）、§2.3 图像编辑（ImgEdit-Bench）方法细节、附录 B.4 未在正文出现的部分微架构超参

---

## 0. TL;DR：六个阶段速查

| 阶段 | 论文位置 | 一句话主张 | 关键数字 / 图 |
| :--- | :--- | :--- | :--- |
| **① 动机** | §1 | 生成瓶颈不只在 DiT，也在「没听懂用户要什么」：Text-to-Image → **Requirement-to-Image** | Fig.1 |
| **② 架构** | 附录 B.2/B.2.1/B.2.2 | **Reasoner(≈32B) + Encoder(Qwen3-VL-8B, 冻结) + DiT(10B)** 解耦；原试过 72B VLM + 10B DiT，成本不可接受 | Fig.34 / 35 / 36 / 37，Table 7 |
| **③ 数据** | §3.1.3 / §3.2.2 | **208.62M 独立图**（187M 开源 + Syllabus 21.62M→重采样 47.19M）；缺陷图**标注而非删除**；分维度多 VLM Caption；Syllabus 三原则 | Fig.21–28，Table 8/9 |
| **④ 训练** | §3.2.3 / 附录 B.5 | 三阶段 512²→1024²→2048² 渐进；**Rectified Dynamic Time Shifting**（2K 不再继续加大偏移） | Fig.31/32，Table 11 |
| **⑤ Agentic 推理** | §3.1.2 / §3.1.4 / §3.2.1 | Rewriter 是 **Translator, Not an Enhancer**；五类能力定向 PE Skill；Router 按难度选 Base/Turbo | Fig.17/18/19 |
| **⑥ RL + BOG** | §3.2.3 / 附录 C | **刻意不做大规模美学 RL**（保多样性），只修 anatomy / 文字；BOG = 矩阵正交化 CFG，解过饱和 | Fig.30 / 33 |

::: info 全篇一句话
Boogu-Image 真正的贡献不是新网络、不是新 Loss，而是**把「理解」从 DiT 里拆出来做成一条独立的、可堆算力的系统流水线**，同时把几个被忽视的训练细节（timestep 采样、汉字曝光量、缺陷标注）修正到位——总成本仅 **≈ \$400K / 208.62M 图**。
:::

---

## 1. 问题定义与核心动机（论文 §1 Introduction）

### 1.1 主张：Text-to-Image → Requirement-to-Image

作者开篇不谈画质，先谈**任务定义**的迁移：

$$
\boxed{\ \text{Text-to-Image}\ \longrightarrow\ \text{Requirement-to-Image}\ }
$$

原文逻辑：用户已经不满意「一句描述性 prompt」，他们期待模型去解释复杂意图、隐含约束、多层指令和跨模态上下文线索；而真实需求还横跨「从简单到 intricate」、成本与延迟预期差异巨大，**单一模型配置几乎无法同时最优服务所有请求**。

![Boogu Fig.1：左为 Boogu Arena Elo 对比（Boogu-Image-0.1-Turbo-Thinking 1196 分，开源第一；GPT-Image-2 1087、Nano-Banana-Pro 1048、Seedream-5.0-Lite 1032、Qwen-Image-Max 1021、Boogu-Turbo 988、Z-Image-Turbo 960、Qwen-Image-2.0 946、HiDream-O1-Image 868）；右为 Turbo → Turbo-PE → Turbo-Thinking → Pro 的推理时间-质量权衡（论文标注 illustrative only）。](/boogu-fig01-arena-elo.png)

::: warning Fig.1 右图不是实测数据
论文原文明写 “illustrative only”。右图表达的是**趋势命题**（推理算力越高，T2I 质量越好），不能当作四个变体的实测对照表引用。
:::

### 1.2 反直觉的动机例子：海报里的牛顿三定律

用户输入「画一张海报，介绍牛顿三大运动定律，要有对应插图，文字准确，排版清晰」。即使画面光照/配色/质感都很好，仍可能出现：

- 第二定律公式写错；
- 三条定律的说明与插图对不上；
- 第三定律作用力/反作用力箭头方向画反。

::: tip 关键区分：这不是 DiT 的错
作者认为存在一个**更上游的失败**：用户只给了目标，没写海报该有哪些文字、每块放什么、公式是什么、插图怎么排。如果在生成前把内容确定下来，DiT 的任务会简单很多。这正是 **Fig.17** 里 Rewriter 要做的事。
:::

### 1.3 理解能力被拆成三个层次（§3.1 开头）

论文把「理解」显式拆成三条互补的轴，每条对应一类错误来源：

| 层次 | 问题 | 对应模块 |
| :--- | :--- | :--- |
| **① 理解用户意图** | prompt 又短又准时必须**忠实保留**；又长又糊时需要推理后**翻译成可生成的指令**。因为编码器与改写器都在生成器**上游**，改进会向下游全量传导 | Instruction Encoder + Prompt Rewriter |
| **② 理解训练图片** | caption 策略决定模型能学到什么：**caption 里没描述过的能力，模型几乎不可能习得**，推理期再多的改写/采样也补不回来 | 分维度 Caption + Syllabus |
| **③ 理解任务复杂度** | 不同任务所需生成能力差异巨大，一律用最强模型对简单请求是算力浪费 | Model Router |

论文 §3.1 还给出一条容易被忽略的观察：**即使最强的 VLM 也有固有短板**（计数、精确空间推理、属性绑定），且 system prompt 过长会让模型注意力涣散——这是后面「分能力 Skill」与「不用单一大模型」的动机来源。

### 1.4 附带的一把刀：公开 benchmark 已经不可信（§2.1）

作者在评测章开头先拆自己的台：把六个近期 T2I 模型的 **LMArena Elo** 与 **GenEval / DPG-Bench** 分数对 scatter，出现明显**秩反转**——人类偏好最强的 GPT-Image-2 在两个 benchmark 上都只排中游。

![Boogu Fig.5：LMArena 人类偏好 Elo vs（a）GenEval、（b）DPG-Bench 的秩反转散点；灰色虚线为理想相关。](/boogu-fig05-benchmark-inversion.png)

作者归因三点：① benchmark 不反映真实应用分布；② 普遍饱和，动态范围被压缩，差异来自方差/格式/过拟合；③ **数据污染与测试集泄漏**普遍存在（甚至有模型在含泄漏数据的训练条件下报分）。他们自己的模型也中招：一个 SANA-VAE 配置 GenEval 0.92，而实际部署的更强模型只有 0.85。

> **对做基模的人**：论文因此自建 **Boogu Arena**（§2.2.1）——三类场景（写实电影感 / 文字渲染 / 风格化艺术）× 每类约 400 个细粒度子场景关键词，用强 VLM 沿「prompt 长度 short:medium:long = 3:4:3」「用户 persona 27 档 novice:intermediate:professional = 5:3:2」两个轴**耦合**展开（长段式 prompt 不与 novice persona 配对），得到 **1,200 条中英双语 prompt**；严格盲测两两对战，投票后才揭示模型身份；按加权逆频率采样，用 **Bradley–Terry 模型**聚合成 Elo。共收集 **4,000+ 票**。与 LMArena 的一致性：**Pearson r = 0.986，Spearman ρ = 1.000**。

---

## 2. 架构拓扑：Reasoner + Encoder + DiT 的三段解耦（附录 B.2）

### 2.1 数据流总览

```text
用户原始 Prompt ──► ① Instruction Reasoner（强 VLM，推理期工作，不进 DiT 梯度路径）
                     │  分析意图 / 消歧 / 结构化 → 详细、明确的文本 Prompt
                     ▼
                  ② Instruction Encoder：Qwen3-VL-8B（冻结）→ 取最后层 hidden states
                     │  H_text ∈ R^{N×d}
                     ▼
                  ③ DiT 10B：Dual-Stream(N_d) → Single-Stream(N_s)，Flow Matching 多步去噪
                     ▼
                  ④ VAE Decoder（FLUX.1 VAE）：latent → RGB
```

### 2.2 两个最易混淆的模块

| 维度 | ① Instruction Reasoner | ② Instruction Encoder |
| :--- | :--- | :--- |
| 主要职责 | 理解并**改写**用户要求 | 把 Prompt **编码成条件特征** |
| 输入 | 原始用户 Prompt（+ 编辑时的参考图缩略图） | 改写后的 Prompt |
| 输出 | 可阅读的**文本 Prompt** | **Hidden States 张量** |
| 是否直接产生图像条件特征 | 否 | **是** |
| 是否需要复杂推理 | 需要 | 主要负责表示提取 |
| 是否属于 DiT 本体 | 否 | 否，是上游条件编码器 |
| 规模/状态 | 中等偏大 VLM（如 32B；结论章提到线上 agent 由 **DeepSeek-V4-Flash** 驱动） | Qwen3-VL-8B，**训练全程冻结** |

### 2.3 为什么从「72B VLM + 10B DiT」退回来

论文附录 B.2 的动机链很完整：对 GPT-Image 系列与 Nano Banana 系列观察到**「生成前存在稳定的延迟先验」**，推测它们都有一个强理解模块在拆解指令（这天然带来时间开销）。于是最初设计为 **72B VLM（冻结）驱动 10B DiT**，只更新 DiT——但训练与推理开销都**高到无法接受**。

最终方案是**解耦**：把「思考」与「编码」拆给两个不同规模的 VLM。

::: info 架构判断的正确读法
论文明确说这种**能力来自 encoder 的理解与编码能力，而非参数量本身**；且因资源限制**没有测更大的 encoder**，在 1.7B–14B 区间**未观察到饱和**。选 8B 是「质量 vs 开源可部署成本」的折中，不是「8B 最优」的实验结论。
:::

### 2.4 Instruction Encoder 输出什么

它不生成文字，而是取 Transformer **最后一层的 hidden states**。改写后 prompt 被 tokenizer 切成 $N$ 个 token：

$$
c=[w_1,w_2,\dots,w_N]
\qquad
\boxed{\ H_{\text{text}}=\operatorname{Qwen3VL}(c)\in\mathbb{R}^{N\times d}\ }
$$

$H_{\text{text}}$ 经独立 Embedder + 轻量 Refiner 映射到 DiT 所需维度，进入 DiT 的文本分支。Refiner 默认只有 **2 层**；可选的 **Prompt Tuning Transformer 固定 3 层、仅 32 个可训练 prompt token embedding、因果 mask**，其输出被 prepended 到 encoder 输入序列。

::: warning 「冻结」不等于「无影响」
因为 Instruction Encoder 全程冻结，Prompt Tuning Transformer 在数学上等价于**全局调制 encoder 输出的 hidden state 分布**，使其对齐 DiT 需要的语义空间——这是全篇唯一「加在冻结 VLM 上」的极轻量可训练模块（论文称其为 optional，未在正文给出它的独立消融）。
:::

### 2.5 DiT 微架构

![Boogu Fig.35：Instruction Encoder（左 a）与 DiT（右 b）的完整 pipeline。左半：Trainable Prompt Token Embeddings → Prompt Tuning Transformer，与 Text Tokenizer + Embedding Table、ViT Encoder（384×384 参考图缩略图）并行，文本/视觉/系统 prompt/用户指令拼成统一序列进入 VLM，输出「Instruction Hidden States」并 Look Up 到最后层。右半：三路独立 Embedder Projection + Refiner（Instruct. / Ref. / Noise.）→ N_d 层 Double-Stream → Merge → N_s 层 Single-Stream → Unpatchify → VAE Decoder；右侧 Time Embedding 经 scale 调制。](/boogu-fig35-pipeline.png)

- **双流 → 单流**：$N_d$ 层 Dual-Stream 里，instruction stream 与 image stream（噪声 latent + 可选参考图）分开处理；最后一层双流输出拼接后送入单流层，最终 unpatchify + VAE decode。
- **Dual-Stream 的关键改造**：不像常规双流只给两路 QKV 独立投影或只解耦少量模块，Boogu 采用**更完整的独立模块集**——三组独立 QKV 分别属于「文本模态」「图像模态（整体增强用）」「图像模态内部 self-attention」。右流负责加深图像 token 交互；左流做 instruction+image 的全模态注意力并带 instruction 残差，再把左流的图像信息与右流的残差/自注意力输出融合。
- **RoPE**：两路注意力各自使用 **3D RoPE**。
- **VAE**：使用开源 **FLUX.1 VAE**；无参考图时跳过 VAE encoder，但 decoder 始终需要。

![Boogu Fig.36：Dual-Stream Layer 详细结构。左路（instruction）：三组 QKV → RMS → 3D RoPE → 全模态 Softmax → out-proj → 带 gate/tanh 的门控残差 → FFN；右路（image）：内部自注意力分支与整体增强分支经 Merge 融合，含两组带 gate 的残差（Image Residue 1 / 2）。底部标注 instruction hidden states 的组成为 {Prompt Emb., System Prompt, Thumbnail, User Instruction}。](/boogu-fig36-dual-stream.png)

![Boogu Fig.37：轻量单流 Transformer 层（Refiner 与 Prompt Tuning Transformer 通用），结构为 RMS → QKV → RoPE → RMS → Softmax(Q·Kᵀ/√d)V → General Out-Projections → RMS → Feed Forward → RMS → Residue。](/boogu-fig37-lightweight-layer.png)

### 2.6 Instruction Reasoner 工作流

![Boogu Fig.34：Instruction Reasoner 工作流。输入侧由 Ref. Image Thumbnails + `[User] Make the Doraemon blue and remove his ears.` + `[System] You are a professional instruction reasoner...` 经 chat template 包装；输出侧是完整解析、分解、改写后的长指令（明确「Remove his ears completely, leaving the top of his head smooth and rounded, while preserving his overall shape, expression, and pose」）。](/boogu-fig34-reasoner-flow.png)

**任务模态由是否有视觉输入决定**：没有参考图 → 纯 T2I，套用 T2I system prompt；有参考图 → 编辑任务，启用编辑专用 system prompt。

### 2.7 「传感器」消融：Table 7

保持 DiT **固定 1B**、训练数据与超参不变，**只替换冻结的 instruction encoder**：

| Instruction Encoder | GenEval ↑ |
| :--- | ---: |
| Qwen3-1.7B | 0.6034 |
| Qwen3-4B | 0.6251 |
| Qwen3-14B | 0.6477 |

::: danger 这张表最容易被误读
1. 消融用的是 **Qwen3 纯文本模型**的三个规模，**不是 Qwen3-VL-8B 与其他 VLM 的直接对照**；不能推出「8B VLM 是最优配置」。
2. 论文原文明说这是**尚未充分训练的中间结果**，只用来看跨规模的**相对趋势**。
3. 「更强 encoder → 更好生成」这一因果链，在本文中**只有单变量对照，没有对 DiT 容量/训练步数的二次扫描**。
:::

论文还给了个很直观的失败案例（Fig.16）：`龙凤呈祥 剪纸风格` 与 `龙和凤呈祥 剪纸风格` 只差一个「和」字，但后者打破了模型学到的四字成语整体模式，生成结果显著不同——说明**文本编码器对「字符级」语义结构的敏感度本身就是瓶颈**。

![Boogu Fig.16：Prompt 1「龙凤呈祥 / 剪纸风格」与 Prompt 2「龙和凤呈祥 / 剪纸风格」仅差一个「和」字，生成结果明显不同。](/boogu-fig16-one-character.png)

---

## 3. 训练数据与 Caption（§3.1.3 + §3.2.2）

> 这是全篇工程价值最高的部分。核心问题：**在有限算力下，怎么用 2 亿张图训出强基模？** 作者的回答不是提高美学分并删掉低质图，而是围绕「覆盖什么能力 / caption 描述什么 / 每类训几次」重新设计。

### 3.1 数据规模：三个数字别搞混

| 名称 | 数量 | 含义 |
| :--- | ---: | :--- |
| 全流程训练用图 | **208.62M** 独立图片 | 论文口径的「unique images」总量 |
| 开源基础数据 | **187M** | COYO / DataComp / PixelProse / BLIP3-o / OmniCorpus |
| Boogu Syllabus 独立样本 | **21.62M** | 定向补齐重要能力的**独立图片** |
| Syllabus 训练样本 | **47.19M** | 按预设权重对 21.62M **重采样**得到，不是新增图片 |

::: tip Unique Images ≠ Training Samples
同一张图可在不同轮次重复出现、也可带不同 caption，但**不能因此算成多张独立图片**。47.19M 是采样权重作用后的训练规模。
:::

### 3.2 第一步：图片从哪来

| 数据源 | 作用 |
| :--- | :--- |
| COYO | 大规模网页图文对 |
| DataComp | 大规模图文配对 |
| PixelProse | 更丰富的图像描述 |
| BLIP3-o | 图像与文本监督 |
| OmniCorpus | **交错图文文档**（interleaved image-text documents），扩知识与场景覆盖 |

::: warning 可复现性边界
论文**没有**披露每个数据源的最终保留量、去重阈值、过滤器设置与逐步损耗，因此**无法据此还原一份完全相同的 187M 清单**。学习时要区分「作者公开了可直接复现的策略」与「仍是黑盒的工程细节」。
:::

### 3.3 第二步：不删缺陷图，而是**把缺陷写进 caption**

这是全篇最反常识、也最可直接复用的一条。

![Boogu Fig.22：四类常见缺陷图 + 显式描述缺陷的 caption（左上 watermark / 右上低照度彩色噪点 / 左下慢门运动模糊 / 右下严重过曝），caption 中**用红色标出**描述缺陷的句子。](/boogu-fig22-artifacts-caption.jpg)

论文立场：**几乎任何 2D 图都能用**，真正的问题不是「要不要留」而是「**怎么配比**」，而配比需要细粒度 tagging 或人工标注才能精确控制。

对照示例（原文 caption 的中文大意）：

- 夜景噪点图 → caption 明确写出「整张图片由于高 ISO 或长曝光拍摄，呈现出明显的数字噪点和颗粒感，尤其在天空和暗部区域」；
- 过曝静物 → 明确写出「透过铁丝网看到的景象几乎完全发白且过曝」。

::: danger 两条必须一起讲的限制
1. **准确标注 ≠ 普通 prompt 下就不产生缺陷**。若这类样本占比过高，仍会拉低默认生成质量。
2. 论文**未公布**各类缺陷样本的最终训练占比。因此「该删还是该留」是原则，**比例是未开源的工程细节**。
:::

与本库已读的两篇对照：[Mage-Flow §4.1 是「先便宜后昂贵」的十级硬过滤](/mageflow-fig8a-pipeline.png)（含 Blurry / Watermark / Saturation 等）；**Boogu 是「保留 + 显式标注」**。两种路线目标不同：Mage-Flow 在为通用基模清理分布，Boogu 在把缺陷**变成可控生成能力**。

### 3.4 第三步：分维度 Captioner，而不是一个 VLM 写到底

§3.1.3 的标题就是立场：**Good Supervision Requires Understanding User Demand**。

论文点名现有做法两个通病：

- **中等规模 VLM 能力有限**：数错物体、幻觉属性、误判空间关系，而**噪声监督会被生成模型忠实继承**；
- **强 VLM 对 system prompt 高度敏感**：同一张图换 captioning prompt 结果差异明显，不校准就会在训练集里制造**互相冲突的监督信号**。

做法：对每个维度**独立选出最优的 VLM + System Prompt 组合**，最后再汇总成一条统一 caption；多数 VLM 都做不好的维度，直接上专家模型或人工标注。

候选池与维度：

| 候选 Captioner | 维度 |
| :--- | :--- |
| Qwen2.5-VL-7B / Qwen2.5-VL-32B / Qwen3-VL-8B / InternVL3-8B / Gemini-2.5 / Gemini-3 | 物体计数、空间关系、属性绑定、风格、文字渲染 |

结论：**per-aspect 设计的 caption 一致性优于任何单一 VLM 基线**，且 image–caption 对齐的提升会**转化为对应生成能力提升**。

::: warning 三个工程问题（论文未给完整答案）
1. 各 VLM 的输入分辨率、提示词、视觉预处理都不同，**要逐维度而不是逐模型选型**；
2. **分维度准 ≠ 融合后准**：Counting 说 3 个人、Spatial 只描述 2 个人，融合阶段必须有冲突消解规则；
3. caption **变长不等于变好**：论文未公布每类 caption 的完整模板与 token 分布。
:::

### 3.5 第四步：Boogu Syllabus（教学大纲式数据组织）

**为什么 187M 不够**——论文给了三段递进论证：

1. 互联网数据严重长尾。以地标为例，Eiffel Tower 的搜索热度远高于长尾地标，**头部概念的额外样本收益递减**；
2. 开源数据有**严格上限**：语义噪声、事实错误、严重重复，且 caption 又短又浅，缺细粒度属性与空间关系；
3. 更狠的是一条**文化偏置**观察：即使用中文 prompt，模型仍高概率生成西方人脸与西方街景，**SFT 阶段加大规模中文数据也纠正不过来**——因为世界观在预训练阶段就已建立。

![Boogu Fig.21：100 个世界地标的 Google Trends 相对热度（锚定到第一名）。Top10 为 100 / 66 / 51.7 / 43.5 / 38.1 / 34.1 / 31.1 / 28.7 / 26.8 / 25.1，rank 91–100 全部挤在 6.3–6.7 区间。](/boogu-fig21-landmark-longtail.png)

![Boogu Fig.24：语义等价的中英文 prompt 生成结果都以西方文化元素为主（左二英文 prompt，右二中文 prompt：「画面中心有一个拾荒的老人，满脸沧桑」）。](/boogu-fig24-cultural-bias.jpg)

**三条数据组织原则**：

| 原则 | 含义 | 类比 |
| :--- | :--- | :--- |
| **Systematic & Fine-Grained Deconstruction** | 按人类先验把数据细拆成逻辑互联的组件，形成 syllabus；如 Graphic Design 继续拆成 Typography / Logo / Poster / Composition | 物理教材分册 |
| **Comprehensive Coverage** | 任何目标能力都不能落在计划之外，必须**主动采集**，不能指望模型靠规模「顿悟」 | 「没见过猫，就假设它画不出猫」 |
| **Ample Data per Category** | 每个细粒度类别的数据量要足以让模型**记住**这个概念（阈值经验证，见 §3.7） | 每章都要有习题量 |

![Boogu Fig.25：Syllabus 宏观分布——general 34.9%、design 34.7%、people 17.0%、scene 3.7%、animal 3.7%、style 3.4%、object 1.9%、unknown 0.7%。注意 design 与 general 几乎等量。](/boogu-fig25-syllabus-macro.png)

![Boogu Fig.26：Syllabus 微观（细粒度）分布。design 下 poster 11.3%、text rare 5.6%、text general 4.7%、product poster design 2.3%、diagram slides 1.8%、logo product 1.4%、web design 1.4%、logo object 1.3%、paper diagram 1.2%、logo design 1.0%、other 2.9%；people 6.8%（celebrity 3.6% / people-object interaction 1.6% / activity 1.4%）；animal 3.7%（birds 0.6% / fish 0.6% / other 2.6%）；scene 3.7%；style 3.3%（general style 2.0% / comic character 1.3%）；object 2.0%（object 1.3% / product 0.7%）；unknown 0.7%。横轴被刻意断成 0–12 与 32.5–35.0 两段，因为 general 类单条占 34.7% 会压扁其他类。](/boogu-fig26-syllabus-micro.png)

::: info 这里最值得抄的一条设计意图
**「graphic design 占 34.7% 且被切得极碎」是全篇数据配比里最有信号的一处**：海报、Logo、幻灯片、网页设计、图表排版被拆成 10 个子类，说明作者把「设计类生成」判断为**复合能力**（准确文字 + 排版层次 + 构图），而不是单一技能。
:::

**重采样公式**（作者视角）：

$$
\mathcal D_{\text{syllabus}}=\bigcup_{k=1}^{K}\mathcal D_k,\qquad
\text{sample}(k)\propto w_k
$$

`21.62M unique → 47.19M training samples` 就是权重作用的结果。论文明确表示**给复杂任务更高采样权重**，但**未公开完整权重表、类别查询规则与采集脚本**。

**Table 8 消融**（同模型、同训练设置）：

| 训练数据 | Qwen-Image-Bench (CN+EN) ↑ | LongText EN ↑ | LongText ZH ↑ |
| :--- | ---: | ---: | ---: |
| 仅开源 187M | 48.45 | 0.828 | 0.869 |
| + Boogu Syllabus | **53.65** | **0.952** | **0.969** |

![Boogu Fig.23：同设置下，仅用开源数据的模型（左二）与用 Boogu Syllabus 训练的模型（右二）的定性对比——后者文字渲染更准、构图更连贯、指令遵循更好。](/boogu-fig23-syllabus-quality.jpg)

::: danger 不能把 Table 8 的收益全归给「重采样权重」
这个消融同时改变了**样本质量、能力覆盖、数据来源**三件事。它证明的是「加入 Syllabus 的整体方案有效」，而非「upsampling 权重单独贡献了 +5.2 分」。
:::

### 3.6 第五步：中文文字渲染——把数据覆盖细到**单个字形**

论文给出的经验阈值：**模型对未见汉字几乎无法渲染，每个字至少约 300 次训练曝光才能稳定生成**。

做法是**按字统计曝光量再定向补充**，而不是无目标地堆中文海报：

1. 构造一个**频率不同**的汉字子集做消融；
2. 统计每个字在训练中的出现次数与生成准确率。

![Boogu Fig.27：横轴为按训练出现频次从高到低排列的汉字，纵轴为该字的生成准确率。曝光 ≥300 次的区间准确率稳定（频次标注 302 / 303 / 305 / 308），而 12 / 19 / 21 / 38 / 55 / 57 / 59 / 59 / 71 / 72 / 73 次的区间准确率快速衰减。](/boogu-fig27-char-frequency.png)

![Boogu Fig.28：SFT 前（左两格）对稀见字「挞」渲染失败；在把每个字的曝光补到 300 次以上的合成数据上做 SFT 后（右两格）稳定正确渲染。](/boogu-fig28-rare-char-fix.jpg)

资源受限时，作者的替代方案是**优先覆盖《现代汉语常用字》3,500 字，每字至少 300 次曝光**。

| SFT 优化 | LongTextBench (ZH) ↑ |
| :--- | ---: |
| Before | 0.9055 |
| After | **0.9538** |

::: warning 「300 次」不是普适定律
这是**本文模型、这套数据与该字体分布下的经验值**，不是适用于所有模型/字体的通用常数。它真正可复用的是**方法论**：把「能力覆盖」的统计粒度从「场景类别」下推到「视觉符号」。
:::

### 3.7 数据侧的一个量化阈值：身份记忆（§3.2.3）

为了回答「一个新概念需要被看多少次才能记住」，作者做了一个极干净的实验：

| 项 | 设置 |
| :--- | :--- |
| 数据 | 志愿者 **170 张**日常照片 |
| Caption | 每图 4 个变体（中/英 × 短/长） |
| 过采样后 | **11,628** 张（中英各 5,814），混入 **12.6M** SFT 集，约 **0.13%** |
| 训练 | 128 卡，global batch **1024**，常数 LR **1e-4**，warmup **500** 步，随机打散 |
| 结论 | 约 **4,500 次语言匹配曝光**才能稳健记住一个**完全未见过的真人身份** |

![Boogu Fig.29：同一中文 prompt 下不同 checkpoint 的生成。上排前四张为 1k / 2k / 3k / 5k 步，下排前三张为 6k / 8k / 9k 步（每 1k 步≈1,000 次曝光，其中中英各约 500）；右侧为真人在约 70kg / 75kg / 80kg 三个时期的训练原图。论文提示：flow matching 模型倾向于学到这些变化的**平均表征**。](/boogu-fig29-identity-memorize.jpg)

::: info 这个实验的双重价值
① 给「记忆一个新概念所需的曝光量」提供了可操作的数；② 顺带暴露 flow matching 的一个已知特性——**同主体跨状态样本会被平均化**，这在肖像一致性上既是优点（稳）也是缺点（丢细节）。
:::

---

## 4. 预训练与时间步采样（附录 B.5 + §3.2.3）

### 4.1 三阶段训练配置（Table 11）

| 阶段 | 数据 | Max Instruction Tokens | Global BS | Micro BS | Epochs | LR | Warmup |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **512²** | 开源 187M | 256 | 1280 | 10 | 1 | 1e-4 | 1200 |
| **1024²** | 开源 60M + Syllabus 47M | 512 | 2048 | 8 | 2 | 3e-5 | 800 |
| **2048²** | Syllabus 47M | 944 | 1024 | 2 | 1 | 3e-5 | 800 |
| 2048² TI2I | 11M T2I + 11M TI2I | 944 | 448 | 1 | 2 | 3e-5 | 800 |

共同设置：Max Output/Input Pixels 分别为 262,144 / 1,048,576 / 4,194,304（前三阶段），Max Side Length 800 / 2048 / 4096；超出阈值的图在预处理时**动态降采样**，等于天然支持 mixed-resolution 训练。优化器 Adam(0.9, 0.95)、weight decay 0.01、**Max Grad Norm 1.0**、Constant-with-Warmup、warmup init LR 1e-6；Instruct. Dropout Prob. **0.1**、Ref. Img. Dropout **0.01**；bf16 + gradient checkpointing；分片 HYBRID_SHARD_ZERO2 → HYBRID_SHARD。

::: tip 一条清晰的数据课程策略
**低分辨率先学广泛视觉概念，高分辨率逐渐提高高质量/复杂能力数据比重。** 512² 全开源 → 1024² 开源 60M 掺 Syllabus 47M → 2048² 全部 Syllabus。256 / 512 / 944 是**上限**，不代表每条 caption 都这么长。
:::

### 4.2 Flow Matching 基线约定

Boogu 的时间方向是 **$t=0$ 为纯噪声、$t=1$ 为干净图像**：

$$
x_t=(1-t)\,\epsilon+t\,x_0,\qquad \epsilon\sim\mathcal N(0,I)
$$

$$
\mathcal L_{\text{FM}}=\mathbb E_{t,\epsilon,x_0,c}\Big[\big\|v_\theta(x_t,t,c)-v^\*_t\big\|_2^2\Big]
$$

### 4.3 Dynamic Time Shifting 与「过度挤压」

标准 logit-normal 采样：

$$
u\sim\mathcal N(0,1),\qquad t=\operatorname{Sigmoid}(u)=\frac{1}{1+e^{-u}}
$$

按 latent token 数 $n_T$ 线性插值出偏移量：

$$
\mu(n_T)=k\,n_T+b,\qquad k=\frac{y_2-y_1}{x_2-x_1},\quad b=y_1-k\,x_1
$$

（参考点取 $(x_1,y_1)=(256,0.5)$、$(x_2,y_2)=(4096,1.15)$，故 $k=1.6927\times10^{-4}$，$b\approx0.4567$。）

$$
\boxed{\ t_{\text{shift}}=\tau\big(t;\mu(n_T),\sigma\big)=\frac{t^{\sigma}}{t^{\sigma}+e^{\mu(n_T)}(1-t)^{\sigma}}\ },\qquad \sigma>0
$$

关键洞察——在 **logit 空间它只是一次仿射变换**：

$$
\operatorname{logit}(t_{\text{shift}})=\sigma\cdot\operatorname{logit}(t)-\mu(n_T)
$$

取 $\sigma=1$ 就退化为**纯平移 $-\mu(n_T)$**。而 $n_T\propto H\times W$：分辨率从 512→1024→2048，latent token 数大致按 $1\to4\to16$ 增长，于是 $\mu$ 近似二次增长，把时间步分布**不断推向纯噪声端**。

| 分辨率 | $n_T$（示例） | $\mu(n_T)$ | 原始 $t=0.5$ 偏移后 | 论文报告的中位数 $P_{0.5}$ |
| :--- | ---: | ---: | ---: | ---: |
| 512² | 1,024 | 0.630 | **0.348** | — |
| 1024² | 4,096 | 1.150 | **0.240** | 0.24 |
| 2048² | 16,384 | 3.230 | **0.038** | 0.04 |

![Boogu Fig.31：$\sigma=1$ 下 1K（4096 tokens）与 2K（16384 tokens）的偏移后时间步密度。1K 的中位数 $P_{0.5}=0.24$、Top-30% 阈值 $P_{0.7}=0.35$；2K 坍缩到 $P_{0.5}=0.04$、$P_{0.7}=0.06$——这就是 **over-squeezing**。](/boogu-fig31-timestep-dist.png)

::: danger 常见误读修正
第 4 列是**本笔记按论文公式对示例 token 数复算**的结果（非论文原表）；论文直接给出的是第 5 列的分位数。若直接照抄网络流传的「2048² → 0.043」，那是把不同 $n_T$ 假设混用了。
:::

**症状**：2K 原生训练时收敛速率明显变慢，且训练早中期生成结果出现**可见噪声残留**。$\mu$ 过大 = 训练时信噪比被过度压低。

### 4.4 Rectified Dynamic Time Shifting：把有效 token 数**夹住**

论文先讨论了两个替代方案并否决：

| 方案 | 问题 |
| :--- | :--- |
| 直接把线性斜率压平（调 $(x_1,y_1),(x_2,y_2)$） | 会让 mixed-resolution 训练里**共存的 512² 图偏移不足** |
| 换成非线性 $\tilde\mu(n_T)=\alpha\log n_T+\beta$（取 $\alpha=\tfrac12,\ \beta=-\log 20$，以抵消 $n_T$ 相对边长的平方增长） | $(\alpha,\beta)$ 本身又是优化问题，单调曲线仍可能与各分辨率的最优偏移不匹配 |

最终方案极其简单——**夹住有效 token 数**：

$$
\boxed{\ \tilde\mu(n_T)=\mu\big(\min(n_T,n_{\text{cap}})\big)=\min\big(\mu(n_T),\mu(n_{\text{cap}})\big)\ },\qquad n_{\text{cap}}=4096
$$

即：**超过 1K 分辨率后不再增加任何时间步偏移**。2K 时 $\tilde\mu(16384)=\mu(4096)=1.15$，原始 $t=0.5$ 仍落在 $0.24$，而不是 $0.038$。

**为什么不需要解那个「复杂问题」**：用分位数刻画分布即可闭式写出

$$
P_q=Q_{t_{\text{shift}}}\big(q;\mu(n_T),\sigma\big)=\operatorname{Sigmoid}\Big(\sigma\,\Phi^{-1}(q)-\mu(n_T)\Big),\qquad \Phi(x)=\int_{-\infty}^{x}\frac{e^{-s^2/2}}{\sqrt{2\pi}}ds
$$

论文坦承：$\{P_q\}$ 曲线的斜率与走势原则上可以纯数学分析，但相当繁琐；**分段构造直接绕开了这个复杂度**，继承 1K 上已调好的偏移行为并无缝扩展到 2K，同时消除 over-squeezing 造成的「收敛慢 + 早期噪声」。

![Boogu Fig.32：$t_{\text{shift}}$ 分位数随 latent token 数变化。实线为原始线性 $\mu(n_T)$（中位数从 1K 的 0.24 掉到 2K 的 0.04，Top-30% 从 0.35 掉到 0.06）；虚线为 Rectified 方案，把有效 token 夹在 4096，分位数在此之后保持 1K 水平不再下滑。](/boogu-fig32-timestep-quantile.png)

### 4.5 对应实现

```python
import torch

# 论文采用的两个参考点
X1, Y1 = 256, 0.5
X2, Y2 = 4096, 1.15
K = (Y2 - Y1) / (X2 - X1)
B = Y1 - K * X1


def sample_timestep(n_tokens, batch_size, rectified=True, sigma=1.0):
    # logit-normal 基础采样
    u = torch.randn(batch_size)
    t = torch.sigmoid(u)

    # 核心：夹住「用于计算偏移强度」的 token 数
    eff = min(n_tokens, X2) if rectified else n_tokens
    mu = K * eff + B

    # logit 空间平移 −mu，等价于论文的 τ(t; mu, sigma)（sigma=1）
    t_shifted = torch.sigmoid(sigma * torch.logit(t) - mu)
    return t_shifted
```

::: warning 三个必须讲清的边界
1. **它调的是训练时间步采样分布，不是推理步数**。推理仍可用 30 步 sampler，也不意味着推理固定在某个时间步。
2. **它不改 Flow Matching 目标函数**，只改 $\mathcal L_{\text{FM}}$ 中 $t$ 的采样分布。
3. **它不是「高分辨率不需要高噪声训练」**——高分辨率仍会采到高噪声 $t$，只是不再让概率过度集中在近纯噪声区。
4. 论文明确表示**不声称 $\mu(\cdot)$ 是最优解**，「如何为所有分辨率找到最佳偏移函数」被留作开放问题——4096 不是放之四海皆准的阈值。
:::

### 4.6 与既有做法的关系（阅读补充）

| 技术 | 做法 | 与分辨率相关 |
| :--- | :--- | :--- |
| Uniform Sampling | $t\sim U(0,1)$ | 否 |
| **Logit-Normal**（源自 SD3 §3.1） | $t=\operatorname{Sigmoid}(u),\ u\sim\mathcal N(0,1)$，更偏向中间区域 | 通常否 |
| **Dynamic Time Shifting** | 按 latent token 数偏移 $t$ | 是 |
| **Boogu 的 Rectified Dynamic Shifting** | 偏移强度随分辨率增加，但**超过阈值即封顶** | 是 |

工业界普遍在用 Logit-Normal；HF 的 `FlowMatchEulerDiscreteScheduler` 也支持 `use_dynamic_shifting` + `base_shift=0.5 / max_shift=1.15 / base_image_seq_len=256 / max_image_seq_len=4096` 这一组参数。但要注意：**那个 Scheduler 主要用于推理侧调度，不能据此推断某项目训练阶段也用了 dynamic logit-normal**——训练与推理必须分开看。

---

## 5. Agentic 推理：Rewriter / Skill / Router / Reflection（§3.1.2、§3.1.4、§3.2.1）

### 5.1 Rewriter 的定位：Translator, Not an Enhancer

现有开源模型普遍挂一个 prompt rewriter，但作者指出两类毛病：① 输出**过长**，塞进冗余或幻觉细节、偏离原意并抬高推理成本；② **单次前向**，没有对歧义或组合指令做推理就落盘。

Boogu 把它当作**推理过程**而非一次性变换，并给出一条极关键的设计原则：

> **一个设计良好的 rewriter 应当以下界为恒等变换（lower-bounded by the identity transformation）**——当用户 prompt 已经清晰、规格明确、无冲突时，应当**基本原样输出**。

::: info 原则 vs 保证
论文把这写成设计原则（"rewriting can only help and never hurt"），**不是数学证明的性能保证**——真实系统仍会误改写。
:::

对照（前半为原文构造的说明性例子）：

| 输入 | 不合适的改写 | 符合原则的处理 |
| :--- | :--- | :--- |
| `A red apple on a wooden table.` | `A luxurious golden apple sitting on an antique marble table, surrounded by flowers, magical particles, cinematic lighting...` | **原样保留**——红苹果不能变金苹果，木桌不能变大理石 |

### 5.2 五类能力定向的 PE Skill

作者不是用一个通用 System Prompt 打天下，而是对易错能力设计专门 Skill（Fig.18b 的分类轴）：

| Skill | 解决的问题 | Rewriter 的动作 |
| :--- | :--- | :--- |
| **Reasoning** | 隐含知识与计算 | 先推理出确定答案再描述（"the animal that lays eggs and produces milk" → 鸭嘴兽；"draw the result of 7×8" → 56） |
| **Counting** | 数量不准 | **逐个枚举实例并给出位置**，把抽象约束「5」转成 5 个空间约束 |
| **Infographic** | 信息组织混乱 | 先规划模块、标题、内容、版式，再交给 DiT |
| **Scene-Text** | 拼写与文字布局 | 固定确切字符串与放置位置、字号、层级 |
| **NSFW** | 不安全请求 | 识别并 sanitize，充当**可解释的第一道内容审核** |

![Boogu Fig.17：自动 prompt 改写对 T2I 的影响。左列是原始 prompt `Newton's Laws of Motion Poster` 的生成结果，中间列是改写后 prompt 的结果，右侧列出两条 prompt。改写把同一意图展开为：深海军蓝黑板底纹背景、顶部居中大号白色无衬线标题 `NEWTON'S LAWS OF MOTION` + 细白下划线、下方三个纵向分区（各以细白线分隔）——左区「Law 1: Inertia」配红色书本插图 + 向右 `Constant Velocity` 箭头 + 向左 `Friction` 箭头 + 释义文本；中区「Law 2: F = ma」配蓝色购物车被手推动 + 向前的 `Force (F)` 红箭头 + 向上的 `Acceleration (a)` 箭头 + 释义；右区「Law 3: Action & Reaction」配火箭向上发射、火焰向下 + 向上 `Action` 箭头 + 向下 `Reaction` 箭头 + 释义；底部居中出处 `Sir Isaac Newton Philosophiae Naturalis Principia Mathematica - 1687`；并附 `clean, academic, symmetrical`、`warm museum-style lighting`、`realistic paper grain`、`8K resolution`、`shallow depth of field`、`Studio lighting, neutral color temperature, no glare` 等渲染指令。](/boogu-fig17-prompt-rewrite.png)

### 5.3 Rewriter 消融（Fig.18）

![Boogu Fig.18：(a) Scaling Rewriter——Qwen3.5 系列 0.8B/2B/4B/9B/27B/122B/397B 的 advantage score 为 0.71 / 0.98 / 1.31 / 1.53 / 1.80 / 1.59 / 2.16，122B 与 397B 为 MoE，122B 低于 27B（可能与 MoE 架构或路由行为有关）。(b) Capability-specific Skills——Counting 2.21、Infographic 32.33*（超轴范围）、Reasoning 1.97、NSFW 0.40、Scene Text 5.93。](/boogu-fig18-rewriter-ablation.png)

优势分定义为

$$
S_{\text{adv}}=\frac{N_{\text{win}}+N^{\text{good}}_{\text{tie}}}{N_{\text{lose}}+N^{\text{bad}}_{\text{tie}}}>1 \Rightarrow \text{优于 baseline}
$$

三个读数：

1. **总体趋势**：更大的 thinking 模型带来更大改写收益，但**参数量不严格单调**（122B < 27B）。
2. **Infographic 与 Scene-Text 收益最突出**（32.33* 与 5.93）。
3. **NSFW 的 0.40 < 1 是设计使然**：不安全 prompt 被有意 sanitize，字面满足度下降但安全性提升。

::: warning Skill 到底是什么，没有完全交代
结合 §3.1.2 与附录 B.2.1，可以确定的是：Reasoner 接收经 chat template 包装的用户输入 + 对应 System Prompt，然后分析、分解、改写。**论文没有说明这些 Skill 是否由专门的 SFT/RL 训练获得**，也**未公开触发分类器、完整 Prompt 模板与路由代码**——不能理解为「训练了五个专用 LoRA」。作者原话只是 "capability-specific PE skills"。
:::

### 5.4 系统层面的两个关键结论

**① 不同生成模型需要不同的 rewriter。** 两个原因：模型注入人类偏好的轴不同（有的偏写实、有的偏插画或美化），同一 prompt 会触发不同默认行为；模型训练时的 caption 策略不同，期待的输入表述风格也不同。**为模型 A 调的 rewriter 会给模型 B 产生分布外输入，反而拉低质量。** 直接推论：**用同一 prompt 比较两个模型往往不公平**，有意义的比较应让每个模型配自己的 prompt，在其**预期工作点**上评测。

**② 强 backbone 的潜力高度依赖 system prompt。** 结构差的 system prompt 是瓶颈，会压制模型潜力；校准好的则像催化剂。作者因此主张**改写 prompt 与模型要协同设计**。

### 5.5 系统级收益的量化（Table 1，中文 prompt）

| 模型 | Creativity | Overall |
| :--- | ---: | ---: |
| Boogu-Image-0.1-Base | 48.62 | 50.96 |
| **Boogu-Image-0.1-Base-Thinking** | **56.74**（+8.12） | **53.57**（+2.61） |

::: danger 别把这 +8.12 归给某一个 Prompt 模板
这是**同一系列模型 Base vs Base-Thinking 的系统级对照**（Thinking = 打开 agentic 改写/推理链路），不是单一模板的受控实验。英文表（Table 2）同样稳定：Base 47.52/51.00 → Base-Thinking 56.24/53.73。
:::

### 5.6 Model Router：给对的任务用对的模型

论文的类比很贴切：**人类画家画一幅画的时间，从简单速写到复杂构图天差地别**。

两个量化事实：GPT-Image-2 生成「a beautiful girl」这种极简 prompt，耗时可达 Z-Image-Turbo 的 **100×**；而在 Boogu 内部，许多情况下 **Turbo 与 Base 的输出几乎无法区分，推理成本却相差 50–100 倍**。

因此用 agent 按估计难度路由：简单请求走快 Turbo，复杂请求才动用重的 Base。

![Boogu Fig.19：推理时间与质量的权衡阶梯——Raw Model → + Prompt Enhance → + Best-of-N + Reflection → Agentic Model，质量随推理时间单调上升（图中 Numerous Strategies 省略）。](/boogu-fig19-inference-tradeoff.png)

::: warning Router 规则未公开，也没有「长度=难度」的证据
论文**没有披露复杂度特征、阈值与路由代码**。而且直觉上「200 词的简单风景」未必比「10 个词但要求精确数学公式/精确计数」更难——所以复杂度判据应看语义推理、空间关系、文字准确性与视觉结构，**不能只看 token 数**。这属于可尝试的工程方案，而非论文已验证的规则。
:::

### 5.7 评测协议的两个方法论警告（§3.2.1）

- **评测必须把推理成本计入**：统一多模态系统的理解与生成组件都贡献端到端性能，任一都无法孤立评测。论文指出实践中多数模型只报质量指标，延迟对齐困难——因为闭源系统常有额外生产约束（NSFW 过滤等），公平延迟对比本身困难。
- **A/B 只对单一 baseline 做对比会误导**（Fig.20）：模型 A 打败一个 baseline，不代表它超出了开源能力前沿（open-source capability envelope）。应在多维能力坐标上把 A 放到整个开源前沿里比。

---

## 6. RL：克制使用，保护生成分布（§3.2.3）

### 6.1 立场：分发生布多样性是通用基模的核心需求

论文的论证链：

1. 用户期待的不只是主流审美主体，也包括**偏离主流审美规范**的主体 → 输出分布的宽度是可用性前提。
2. 常见做法是用 RL 提升美学与稳定性，但**重度 RL 会把输出分布压向窄域**。
3. 以 Seedream 为例：人像效果惊艳（人人漂亮年轻、符合主流偏好），但**生成偏离该审美的主体变得相当困难甚至不可行**。**抬高平均质量的对齐，同时收窄了可达分布。**

![Boogu Fig.30：prompt「A sixty-year-old Chinese woman」下 (a) Seedream-4.5、(b) Seedream-5.0-lite、(c) Boogu-Image-0.1-Turbo 的生成。前两者把人画成精致妆容、理想化美化后的形象，Boogu 给出更自然写实的普通样貌。](/boogu-fig30-rl-diversity.jpg)

::: danger 这不是因果消融
Fig.30 是**定性案例**：两个 Seedream 版本与 Boogu 的训练数据、RL 配置、模型规模都不可控地不同。作者自己也强调「这种 RL 驱动的分布塑形**不是缺陷，而是合理的产品策略**，只是与我们最大化多样性的目标相冲突」。
:::

### 6.2 作者的做法

- **刻意避免重度美学 RL**，改为**把人类偏好放进理解系统，而不是不可逆地烘进生成器**，让风格（包括人脸属性）在推理时可控，生成器底层多样性得以保留。
- **但不放弃 RL**：只用在**不损害多样性**的定向问题上，且论文明确点名它们带来真实收益——**减少解剖学伪影、改进文字渲染**。
- 成本理由也写得直白：RL 需要 reward model 推理、反复采样与大量额外优化步，在有限预算下要**只用在不可替代的地方**。

| 能力 | 作者态度 | 理由 |
| :--- | :--- | :--- |
| Aesthetic | 避免大规模强化 | 易压缩风格与外貌多样性 |
| Anatomy | **定向使用 RL** | 多余手指、肢体错位通常不是用户想要的 |
| Text Rendering | **定向使用 RL** | 文字拼写与字形准确性相对易验证 |

### 6.3 RL 细节公开程度：明显低于 Mage-Flow / SeFi

::: danger 不要脑补 Boogu 的 RL recipe
论文明确说用 RL 改善 anatomy 与 text rendering，但**没有公开足以复现的算法与配置**：

| 我们关心的 | Boogu 是否说明 |
| :--- | :--- |
| 为什么不做大规模美学 RL | ✅ 明确解释 |
| 主要优化哪些能力 | ✅ Anatomy、Text Rendering |
| GRPO 还是 Diffusion-NFT | ❌ 未明确 |
| Reward Model 型号 | ❌ 未公开 |
| RL Prompt 来源 / 数量 / 长度 | ❌ 未公开 |
| 每 prompt rollout 次数 | ❌ 未公开 |
| 是否过滤低 reward 方差组 | ❌ 未公开 |
| 是否混合 SFT loss | ❌ 未公开 |
| Batch size / LR / steps | ❌ 未公开 |

所以**不能**因为它用了 RL 就默认 DiffusionNFT，**也不能**照搬 SeFi 的 12 rollouts 或 Mage-Flow 的配置。想学 NFT 的损失与 rollout 实现，去读 DiffusionNFT 原论文；要学多能力 reward 配比，去读 [Mage-Flow](/domains/vision/mage-flow) 与 [SeFi-Image](/domains/vision/image-rl-posttraining/sefi-image-rl)。
:::

### 6.4 Base / RL / Turbo 三阶段不要混为一谈

```text
Base Model  ──大规模 Flow Matching 训练（208.62M 图，≈$400K）
     │
     ├──► 定向 RL（改善文字与人体结构；实现细节未充分披露）
     │
     └──► Turbo 蒸馏：Decoupled-DMD，生成加速至约 4 步
```

::: warning Turbo 的蒸馏算法来源
论文正文**没有描述** Turbo 的蒸馏方法；「Decoupled-DMD / 四步」来自官方 HuggingFace 模型卡。论文也**未充分披露 RL 检查点与蒸馏初始化之间的先后依赖**，上图是概念上的阶段划分，不是已确认的官方执行顺序。
:::

---

## 7. Boosted Orthogonal Guidance（附录 C）：把 CFG 的更新方向当**矩阵**处理

> 这一节直接对应你关注的「生成偏亮、色彩过艳 / 过饱和」问题。BOG 是 training-free 的推理期方法，**不需要任何训练**。

![Boogu Fig.33：四个 prompt 的成对对比——上排标准 CFG、下排 BOG（每列同一 prompt）。BOG 在四个 prompt 上都表现出更强的写实感与更细的纹理，同时保持内容一致。](/boogu-fig33-bog-vs-cfg.jpg)

### 7.1 动机：CFG 放大 alignment 会牺牲 appearance

CFG 的问题是**过度放大 guidance 会用 appearance 退化换 alignment**，表现为**过饱和、色调被压平、过度平滑、视觉上不合理的伪影**。

BOG 的灵感来自 APG 的关键洞见：**guidance 更新可以分解为与模型预测 flow 平行（parallel）的分量与正交（orthogonal）的分量**。BOG 强调**保信息的正交分量**，以避免 color blow-up 与 texture collapse。

真正的差异点在于：BOG **拒绝把 DiT 预测视作扁平化的高维向量**——它把每步预测张量当作**与图像内在空间结构对齐的 2D 矩阵**（呼应 Moun-style 等尊重矩阵结构的优化器设计）。

::: info 为什么「当成矩阵」有实质意义
标准做法（vector-norm rescale，或再加一个 mean-centering）**只改变向量场中的幅度**：更新方向与其空间组合基本不动。BOG 的矩阵归一化**鼓励更高的有效秩（effective rank）**，从而在各空间维度上保留更多有信息的变化，避免过度平滑的更新。
:::

### 7.2 三个组件

**① Rolling-Sum Momentum**（轨迹平滑）

$$
\Delta D^{(i)}=\text{RollSumUpdate}(\Delta D,i)=\eta\,\Delta D^{(i)}+\rho\,\Delta D^{(i-1)},\qquad \Delta D^{(0)}=\mathbf 0
$$

默认 $\eta=0.9,\ \rho=0.1$。注意这与 APG 的「反向动量」（$\rho<0$）**相反**——APG 主张把模型推离此前 CFG 更新方向以更专注当前更新，但作者实测 $\rho<0$ 会让 $\Delta D^{(i)}$ 在向量场中**跨迭代方向抖动**。

**② Matrix Normalization（MNorm）**

令 $D\in\mathbb R^{m\times n}$（不失一般性设 $m\ge n$），做 SVD $D=U\Sigma V^{\top}$，逐奇异值做重整：

$$
\sigma_i'=\operatorname{norm}(\sigma_i,i),\qquad \operatorname{MNorm}(D)=U\,\operatorname{diag}\big(\operatorname{norm}(\sigma_1,1),\dots,\operatorname{norm}(\sigma_n,n)\big)V^{\top}
$$

论文给出三种 `norm` 选项：$\operatorname{Sigmoid}(\sigma_i)$、$\lambda\cdot \dfrac{e^{\sigma_i/\tau}}{\sum_j e^{\sigma_j/\tau}}$、以及 **$1$**。

取 $\operatorname{norm}(\sigma_i,i)=1$ 时，MNorm 等价于**最优半正交逼近**：

$$
\boxed{\ \operatorname{MNorm}(D)=\arg\min_{O\in\mathbb R^{m\times n},\;O^{\top}O=I_n}\ \|O-D\|_F\ }
$$

最小值为

$$
\min_{O^{\top}O=I_n}\|O-D\|_F^2=\|D\|_F^2+n-2\sum_{i=1}^{n}\sigma_i
$$

**③ Parallel-Orthogonal Decomposition（POD）**

以列向量为例做 Gram–Schmidt 式分解（行同理）：

$$
\Delta D^{\parallel}_{[:,j]}=\operatorname{Parall}\big(\Delta D_{[:,j]},D^{(c)}_{[:,j]}\big)=\frac{\Delta D_{[:,j]}^{\top}D^{(c)}_{[:,j]}}{\|D^{(c)}_{[:,j]}\|_2^2}\cdot D^{(c)}_{[:,j]},\qquad \Delta D^{\perp}_{[:,j]}=\Delta D_{[:,j]}-\Delta D^{\parallel}_{[:,j]}
$$

把列分解与行分解合并：

$$
\Delta D_{\text{pod}}=\mathrm{POD}\big(\Delta D,D^{(c)}\big)=\lambda_c\big(\Delta D^{\perp}_{\text{col}}+\mu\,\Delta D^{\parallel}_{\text{col}}\big)+\lambda_r\big(\Delta D^{\perp}_{\text{row}}+\mu\,\Delta D^{\parallel}_{\text{row}}\big)
$$

$$
\lambda_c=\frac{W}{H+W+\zeta},\qquad \lambda_r=\frac{H}{H+W+\zeta}
$$

实践上取 $\mu\le 1$（如 0、0.1、0.25）以在 $\omega$ 较大时抑制过饱和。**$\mu$ 就是「平行分量保留多少」的旋钮**——设 0 即完全丢弃与条件预测平行的分量。

### 7.3 完整单步算法

$$
D_{\text{CFG}}(x_t,t,c)=D_\theta(x_t,t,c_\varnothing)+\omega\big(D_\theta(x_t,t,c)-D_\theta(x_t,t,c_\varnothing)\big)=D^{(c)}+(\omega-1)\Delta D
$$

```text
Algorithm 1  Single Step of Boosted Orthogonal Guidance (BOG)
Input : 推理步 i、当前 latent x_t、条件 c、无条件态 c_∅、guidance scale ω
 1: t ← scheduler(i)
 2: ΔD^(c)_(i) ← D_θ(x_t, t, c);   ΔD^(c∅)_(i) ← D_θ(x_t, t, c_∅)
 3: ΔD_(i) ← ΔD^(c)_(i) − ΔD^(c∅)_(i)
 4: ΔD̂_(i) ← MNorm( RollSumUpdate(ΔD_(i), i) )        # 动量 + 矩阵归一化
 5: ΔD̃_(i) ← POD( ΔD̂_(i), ΔD^(c)_(i) )               # 平行-正交分解
 6: D_BOG_(i) ← ΔD^(c)_(i) + (ω − 1) · ΔD̃_(i)
 7: return D_BOG_(i)
```

**不需要显式 SVD**——$\operatorname{norm}=1$ 的情形可用 Newton–Schulz 迭代估计（论文 Code 1，批处理 GEMM）：

```python
def _newtonschulz5_batched(G, steps: int = 5, eps: float = 1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_ndim = G.ndim
    if orig_ndim == 2:
        G3 = G.unsqueeze(0); out_shape = None
    elif orig_ndim == 3:
        G3 = G; out_shape = None
    elif orig_ndim == 4:
        B, C, H, W = G.shape
        G3 = G.reshape(B * C, H, W); out_shape = (B, C, H, W)
    else:
        raise ValueError(f"Expected 2D/3D/4D tensor, got ndim={orig_ndim}")

    H, W = G3.shape[-2], G3.shape[-1]
    X = G3.to(torch.bfloat16)
    nrm = torch.linalg.norm(X, ord="fro", dim=(-2, -1))
    X = X / (nrm.unsqueeze(-1).unsqueeze(-1) + eps)

    transposed = False
    if H > W:
        X = X.transpose(-2, -1); transposed = True

    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        Bm = b * A + c * (A @ A)
        X = a * X + (Bm @ X)

    if transposed:
        X = X.transpose(-2, -1)
    if orig_ndim == 2:
        return X.squeeze(0)
    if out_shape is not None:
        return X.reshape(out_shape)
    return X
```

### 7.4 BOG 的限制与作者给的对策

::: danger BOG 不是免费的午餐
论文自陈两点副作用：**提高结构畸变概率**、**降低文字渲染成功率**。因此作者把 BOG 当作**可选滤镜（selectable filter）**，并引入 **BOG Interval $\Delta_{\text{BOG}}$**：每隔 $\Delta_{\text{BOG}}$ 个扩散步用 BOG，其余步用标准 CFG（例：10 步去噪、$\Delta_{\text{BOG}}=2$ → 偶数步 BOG、奇数步标准 CFG；$\Delta_{\text{BOG}}=1$ 即全程 BOG）。**推荐默认值 $\Delta_{\text{BOG}}=2$**，能显著缓解伪影，代价是 BOG 风格的摄影纹理略有折损。
:::

::: warning 一步消融就能验证的开放问题
保持 2K 模型、训练数据与训练步数完全一致，分别试 **无 Shift / 原始 Dynamic Shift / Rectified Shift**，观察训练 Loss（建议按时间步分桶统计）、采样残留噪声与生成细节；再用固定 seed 对比 **CFG / BOG / BOG($\Delta_{\text{BOG}}=2$)**，同时看泛锐度/饱和度与文字可读率。这能直观判断收益究竟来自哪里。
:::

---

## 8. 局限与开放问题（§4 Conclusion）

作者自陈的局限，逐条都很值得抄进自己的 checklist：

| 维度 | 局限 |
| :--- | :--- |
| 世界知识 | 在需要大量常识与领域知识的任务上（艺术风格、地标、公众人物、商业产品）仍落后领先闭源系统，且**这个差距很难可靠度量**，作者预期真实差距大于报告分数 |
| 文字渲染 | **目前只优化中英两种语言** |
| 人体结构 | 多人交互、重度遮挡、异常视角仍会出现手/肢体畸变；使用开源 FLUX.1 VAE，其**重建误差进一步限制了小脸、肢体、文字等细粒度细节的上限**，原生 2K 只能部分缓解 |

作者给出的三个方向：更透明忠实的评测（静态 benchmark 已饱和且误导）、突破开源数据瓶颈（语义噪声 + 认知深度不足）、更强的 agentic 生成（更紧的推理-生成回路、自验证、工具使用）。

---

## 9. 横向对比：Boogu 与本库已读的几篇

| 维度 | [Mage-Flow](/domains/vision/mage-flow) | [SeFi-Image](/domains/vision/image-rl-posttraining/sefi-image-rl) | [Qwen-Image-2.0](/domains/vision/qwen-image-2) | [Z-Image](/domains/vision/z-image) | **Boogu-Image** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 条件编码器 | Qwen 系文本编码器 | 语义-纹理双潜变量 | Qwen3-VL | Qwen3-4B + PE | **Qwen3-VL-8B 冻结 + 32B 级 Reasoner 解耦** |
| 数据哲学 | 十级硬过滤优先 | 三原则 caption（粒度） | capability taxonomy + flywheel | 主动挖掘闭环 | **缺陷标注而非删除 + Syllabus 权重重采样** |
| 数据量 | 10B→1.3B | 450M | — | 内部版权数据 | **208.62M（开源 187M + 21.62M）** |
| 时间步采样 | — | — | — | — | **Rectified Dynamic Time Shifting（2K 封顶）** |
| 美学 RL | 使用 | 使用定向能力优化 | T2I 三 reward | DPO+GRPO | **刻意避免大规模使用** |
| RL 算法 | Diffusion-NFT | Diffusion-NFT（12 rollout/prompt） | GRPO | DPO+GRPO | 未披露 |
| 少步加速 | 4-step D-DMD + adversarial guidance | — | — | 8 NFE D-DMD | 约 4 步 D-DMD（据模型卡） |
| 推理期 | — | — | PE + Hybrid CFG | PE-aware SFT | **Rewriter(Translator) + Skill + Router + Reflection + BOG** |
| 成本锚点 | — | 约 Z-Image 10–20% | — | 314K H800·h ≈ \$628K | **≈\$400K** |

::: tip 一句话总结差异
Boogu 是这批工作里**唯一把「理解」当成一等公民、并明确写出「宁可少做 RL 也要保住生成分布多样性」**的技术报告；它对数据缺陷、汉字曝光量、timestep 过度偏移这三处细节的处理，是最值得直接搬进自己 pipeline 的部分。
:::

---

## 10. 工程 Checklist：从这篇论文能直接搬走的东西

1. **先测 encoder 再扩 DiT**：固定 DiT，扫 encoder 规模；若曲线未饱和，优先升级条件编码器（本文 1B DiT 上 1.7B→14B 带来 +4.4 GenEval）。
2. **Rewriter 做三组对照**：Raw Prompt / 通用 Rewriter / 能力定向 Rewriter，**同 seed、同步数**下比复杂语义、文字准确率、人类偏好——否则分不清收益来自新增有效信息还是「prompt 变长」。
3. **SFT caption 分布与 rewriter 风格必须对齐**：训练用简洁客观摄影 caption，推理却狂写华丽长描述 = 分布外条件。
4. **缺陷数据：留 + 显式标注 + 控比例**。把「欠曝/噪点/运动模糊/水印/过曝」写成 caption 里的显式 token，让它们变成可控能力而不是默认画质。
5. **按符号粒度统计覆盖**：不止统计「中文海报有多少张」，还统计「每个汉字被曝光多少次」。经验阈值：约 300 次/字；预算有限就先覆盖 3,500 常用字。
6. **高分辨率训练要查时间步分桶 Loss**：如果中低噪声区 Loss 明显高于高噪声区，几乎一定是 dynamic shifting 在高分辨率下过度偏移。最小改法：把有效 token 数夹在 4096。
7. **RL 前先问「这个能力会收窄分布吗」**：anatomy / 拼写类可以做；审美偏好放进理解系统与推理期控制，别烘进 DiT。
8. **过饱和别只靠加大美学 RM 修**：先看 CFG 的 $\omega$ 与平行分量 $\mu$，再考虑 BOG（配 $\Delta_{\text{BOG}}=2$）。
9. **不要相信饱和 benchmark**：论文给了自建 Arena 的完整协议（1,200 双语 prompt × 盲测对战 × Bradley–Terry Elo），且与 LMArena Spearman ρ = 1.000——这是可以直接照抄的评测工程模板。
10. **成本锚点要对齐口径**：208.62M 图 / ≈\$400K 指的是**训练算力**，不含数据获取、存储与授权成本。

---

## 附：本笔记图源索引

| 图 | 论文位置 | 说明 |
| :--- | :--- | :--- |
| Fig.1 | §1 | Arena Elo + 推理时间-质量权衡（右图示意） |
| Fig.5 | §2.1 | 公开 benchmark 与人类偏好秩反转 |
| Fig.16 | §3.1.1 | 一个「和」字改变成语结构 |
| Fig.17 | §3.1.2 | 牛顿定律海报的自动改写 |
| Fig.18 | §3.1.2 | (a) Scaling Rewriter (b) 能力定向 Skill |
| Fig.19 | §3.2.1 | 推理时间-质量权衡阶梯 |
| Fig.21 | §3.2.2 | 世界地标搜索热度长尾 |
| Fig.22 | §3.2.2 | 四类缺陷图 + 显式标注 caption |
| Fig.23 | §3.2.2 | 开源数据 vs Syllabus 定性对比 |
| Fig.24 | §3.2.2 | 中英等价 prompt 的文化偏置 |
| Fig.25 / Fig.26 | §3.2.2 | Syllabus 宏观 / 微观分布 |
| Fig.27 / Fig.28 | §3.2.2 | 单字曝光量曲线 / 稀见字定向曝光修复 |
| Fig.29 | §3.2.3 | 身份记忆的曝光量阈值实验 |
| Fig.30 | §3.2.3 | 「六十岁中国女性」的美学 RL 分布收窄 |
| Fig.31 / Fig.32 | §3.2.3 | 偏移后时间步分布 / 分位数 vs token 数（Rectified） |
| Fig.33 | §3.2.3 + 附录 C | BOG vs 标准 CFG |
| Fig.34 | 附录 B.2.1 | Instruction Reasoner 工作流 |
| Fig.35 / Fig.36 / Fig.37 | 附录 B.2.2 / B.3 / B.6 | Encoder+DiT 流水线 / Dual-Stream 层 / 轻量层 |