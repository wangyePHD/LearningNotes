# Fine-T2I: 6M 规模开源全维度文生图精调数据集与过滤工程

> **标签**：`Vision` `Dataset` `Fine-T2I` `SFT` `Data Engineering` `Diffusion` `Autoregressive`  
> **更新时间**：2026-09-23  
> **参考来源**：[Fine-T2I: An Open, Large-Scale, and Diverse Dataset for High-Quality T2I Fine-Tuning](https://arxiv.org/abs/2602.09439) (arXiv:2602.09439, 东北大学 Xu Ma, Yitian Zhang, Qihua Dong, Yun Fu) · [HuggingFace Dataset](https://huggingface.co/datasets/ma-xu/fine-t2i) · [HuggingFace Space 预览](https://huggingface.co/spaces/ma-xu/fine-t2i-explore)

---

## 1. 问题背景与开源社区痛点

在当前的文生图（T2I）领域，模型架构（DiT、Flow Matching、自回归生成）、训练机制与推理技巧几乎在学术界与工业界完全公开共享。然而，**领先的文生图能力却依然高度垄断在少数工业巨头手中**（如 Nano Banana Pro、Seedream 4.0、Qwen-Image 等）。

造成这一持续鸿沟的核心症结在于**高质量对齐精调数据（Instruction-Aligned Fine-Tuning Data）的严重封锁**：

1. **商业版权高墙与成本阻碍**：工业级精修图像获取成本极高（商用版权图常需 \$10+/张），且带有严苛的版权保护，无法公开再分发；
2. **现有开源微调数据集的普遍缺陷**：
   - **低分辨率**：LAION-Art、LAION-Aesthetic、Pick-a-Pic、T2I-2M 等早期数据集分辨率通常 $\le 1024 \times 1024$，无法支撑现代模型原生的高清输出；
   - **文本-图像弱对齐与噪音**：互联网 Alt-text 严重失真，缺乏长难句与细粒度属性绑定；
   - **缺乏结构化分布设计**：未经过精细的风格、类别、任务规划，盲目抓取造成分布坍缩；
   - **模板死板**：依赖“a photo of ...”等固定句式，与真实用户复杂多变甚至含糊的 Prompt 严重脱节。

**Fine-T2I 的核心使命**：构建一套**大规模（>6M）、全高清（多数分辨率 $\ge 1\text{K}$）、全维度分布设计、完全开源开放商用协议**的文生图精调基石，填补开源社区从预训练迈向工业级精调的数据断层。

::: tip SeFi-Image 的渊源
在 SeFi 论文 §2.2 中，团队针对 9M 持续训练（Continual Training, CT）精心挑选的开源主干数据，正是本篇 **Fine-T2I**。SeFi 借助 Fine-T2I 极具挑战性的复杂长 prompt 与多样化排版，成功将基础模型的一对一文字映射迁移至真实多模态图文生成。
:::

---

## 2. Fine-T2I 规格体系与数据拓扑

Fine-T2I 总规模达 **630 余万对（约 2TB 磁盘占用）**，由 **6.15M 合成数据子集** 与 **16.8 万真实摄影大师子集** 互补构成。

```
                              ┌──────────────────────────────────────────────┐
                              │            Fine-T2I (约 6.31M / 2TB)          │
                              └──────────────────────┬───────────────────────┘
                                                     │
                     ┌───────────────────────────────┴───────────────────────────────┐
                     ▼                                                               ▼
        ┌─────────────────────────┐                                     ┌─────────────────────────┐
        │   合成图像集 (Synthetic) │                                     │  摄影师精选真实图 (Real) │
        │    6,145,693 对 (1.9TB)  │                                     │     168,424 对 (258GB)  │
        └────────────┬────────────┘                                     └────────────┬────────────┘
                     │                                                               │
     ┌───────────────┼───────────────┬───────────────┐                               │
     ▼               ▼               ▼               ▼                               ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐             ┌─────────────────────────┐
│  PE-AR  │     │  PE-AS  │     │  PO-AR  │     │  PO-AS  │             │   Pexels / Pixabay /    │
│ 1.62M 对│     │ 1.54M 对│     │ 1.69M 对│     │ 1.31M 对│             │      Unsplash-Lite      │
│ 增强长词│     │ 增强长词│     │ 原始短词│     │ 原始短词│             │  审美分 ≥ 6.5           │
│ 随机宽高│     │ 正方形  │     │ 随机宽高│     │ 正方形  │             │  Qwen2.5-VL-7B 双模重标 │
└─────────┘     └─────────┘     └─────────┘     └─────────┘             └─────────────────────────┘
```

### 2.1 四组合成子集矩阵

为了同时兼容学术基准（偏好正方形）与现代多长宽比自由生成需求，并将“Prompt 增强”从单纯的推理技巧内化为模型训练先验，合成集被划分为 $2 \times 2$ 象限：

| 子集代码 | Prompt 策略 | 画面长宽比与分辨率 | 最终样本数 | 磁盘体积 | 核心定位 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PE-AR** | **Enhanced (增强长描述)** | **Random AR (预设多比例随机)** | 1,615,592 | 476 GB | 工业级长 prompt + 自由画幅鲁棒性 |
| **PE-AS** | **Enhanced (增强长描述)** | **Square (多尺度正方形)** | 1,538,253 | 517 GB | 高信息密度长 prompt + 方图基准 |
| **PO-AR** | **Original (原始短描述)** | **Random AR (预设多比例随机)** | 1,686,498 | 479 GB | 贴合终端用户极简输入 + 自由画幅 |
| **PO-AS** | **Original (原始短描述)** | **Square (多尺度正方形)** | 1,305,350 | 436 GB | 传统短 prompt + 方形评测基准 |
| **合计** | - | - | **6,145,693** | **1,908 GB** | - |

### 2.2 摄影师精选真实图子集 (Curated Real-Image Set)

纯合成图容易导致模型在微调时沉溺于合成渲染器的特定审美风格或塑料感纹理。Fine-T2I 补充了来自开源创作者摄影平台的真实高保真图：

| 数据来源平台 | 初始下载量 | 严苛质检保留量 | 保留率 | 存储体积 | 核心特性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pexels** | 233,342 | 117,389 | 50.3% | 192.6 GB | 专业摄影师构图、真实自然光影、人像与街景 |
| **Pixabay** | 163,524 | 32,654 | 20.0% | 9.7 GB | 高清矢量与自然景致 |
| **Unsplash-Lite** | 24,997 | 18,381 | 73.6% | 56.2 GB | 高端画报、静物、微距与材质特写 |
| **真实集合计** | **421,863** | **168,424** | **39.9%** | **258.5 GB** | **Aesthetic 分数集中在 $[6.5, 7.5]$ 顶尖区间** |

---

## 3. 核心制作流水线与数据工程拆解（>95% 淘汰率）

Fine-T2I 最具工程价值之处在于其**“从 44.8M 原始候选池清洗淘汰至 6.15M，最终淘汰率高达 >95%”**的严密流水线。

```
 [阶段 1: 结构化 Prompt 采样]  44.8M 原始 Prompt (LLaMA-3-8B, 32类 x 11风格 x 10任务 x 5模板)
                │
                ▼
 [阶段 2: 语义层次化去重]      5.55M 留存 (all-MiniLM-L6-v2 嵌入, 128组分治, 余弦相似度 > 0.8 剔除 ~87.6%)
                │
                ▼
 [阶段 3: 安全与属性对齐校验]  5.16M 留存 (LLaMA-Guard-3 审查 + Qwen3-VL-8B-Instruct 属性自洽检验)
                │
                ▼
 [阶段 4: 双版本 Prompt 增强]  同步生成 Original (短) 与 PromptEnhancer CoT (长) 双版本
                │
                ▼
 [阶段 5: SOTA 多模型高清合成] Z-Image / FLUX2 (1-3 张候选择优, 覆盖 512 到 2560 宽全画幅)
                │
                ▼
 [阶段 6: 严苛双重过滤与质检]  6.15M 终版 (Aesthetic V2.5 > 5.5 + Qwen-VL-Thinking 思维链二值零容忍审计)
```

### 3.1 阶段 1：LLM 结构化加权采样生成（44.8M 原始池）

为了杜绝普通抓取数据的分布偏向，作者基于 Image Arena 真实用户调用分布与 Qwen-Image / Seedream 报告，设计了**加权分类学（Weighted Taxonomy）**：
- **生成模型**：LLaMA-3-8B-Instruct；
- **采样参数**：设置较高温度 $T=1.4$、较低截断 $p=0.8$，激发长尾多样性；抛弃低于 5 个单词的无效短句；
- **5 种 Prompt 句式模板**：
  1. 经典开头型：`a photo of ...`, `an image of ...`
  2. 用户祈使型：`Please help generate ...`, `Could you provide an image of ...`
  3. 连贯叙述型：完整单句或长复句；
  4. 句词混排型：核心句 + 补充后缀标签（如 `8K resolution, photorealistic, natural`）；
  5. 孤立关键词型：简单实体词组串联（如 `dog, flying, over sea`）。
- **总产出**：生成 44,800,567 条受控 Prompt。

### 3.2 阶段 2：语义两级层次化去重（淘汰率 ~87.6%）

LLM 在多 GPU 大规模生成时存在严重的“模式坍缩”和模板重复问题。如果使用传统的 MinHash-LSH，基于词法 n-gram 容易误杀语法结构相同但关键实体不同的样本。

- **算法方案**：基于语义嵌入进行层次化分治去重。
- **特征编码**：使用 `all-MiniLM-L6-v2` 将 Prompt 映射为 384 维向量；
- **两级分治架构**：
  1. 将 44.8M 向量划分为 128 个聚类分桶，在桶内并行两两计算余弦相似度；
  2. 跨桶汇总进行二次全局过滤；
  3. **相似度阈值**：设定余弦相似度 $\ge 0.8$ 即判定为重复。
- **过滤结果**：从 44,800,567 条剧烈精简至 5,555,147 条，剔除了近 90% 的冗余复读。

### 3.3 阶段 3：安全门禁与属性自洽校验

1. **安全与合规过滤**：使用 `LLaMA-Guard-3-8B` 对文本进行安全审计，直接剔除涉及暴力、涉黄、隐私和违法内容；过滤词数 $>150$ 的畸形冗长提示词，保留 5,497,062 条；
2. **属性对齐校验（Attribute Consistency Check）**：
   - 痛点：LLM 在遵循复杂指令时，可能在被要求输出“UI/UX 设计”时写成了一只真实猫咪；
   - 方案：引入 `Qwen3-VL-8B-Instruct` 作为独立属性审判员，输入 prompt 与指定的目标类别/风格，判断文本是否真正达标，未通过者剔除；最终锁定 **5,158,969 条** 高洁净 Prompt。

### 3.4 阶段 4：Prompt 增强与 CoT 改写

- 借鉴推理期 Prompt Rewriting 策略，调用微调后的 `PromptEnhancer`，通过思维链（Chain-of-Thought）将原始短描述扩写为细节丰富、光影构图与材质明确的长 Prompt；
- **保留双轨对照**：同一概念同时沉淀 Original 与 Enhanced 两个版本，使模型既能被充分的信息密度喂养，又具备理解真实用户极简短词的泛化力。

### 3.5 阶段 5：SOTA 双模型高质量图像合成

- **主力生成底座**：采用 **Z-Image**（单流 DiT 高效基模）为主力，辅以 **FLUX2** 生成高质量样本；
- **多候选择优**：每个 Prompt 生成 1~3 张候选图，通过 `Aesthetic Predictor V2.5` 自动挑选得分最高的一张进入后续管道；
- **多画幅分辨率池**（覆盖 512px 到 2560px）：
  - **正方形 (1:1)**：$512^2, 768^2, 1024^2, 1536^2, 2048^2, 2560^2$；
  - **横屏 (Landscape)**：4:3 ($2048 \times 1536$), 3:2 ($1536 \times 1024$), 16:9 ($2560 \times 1440, 2048 \times 1152$), 5:4；
  - **竖屏 (Portrait)**：3:4 ($1536 \times 2048$), 2:3 ($1024 \times 1536$), 9:16 ($1440 \times 2560$), 4:5。

### 3.6 阶段 6：思维链 VLM 二值“零容忍”审计与审美截断

即使选用顶级模型，合成图仍会出现“多指、不可读乱码、实体缺失、物理悬浮”等生成瑕疵。常规评分器（如 HPSv2、HPSv3）在细粒度错误检测上过于宽容。

- **审美硬门槛**：`Aesthetic Predictor V2.5 > 5.5`（高画质基线）；
- **VLM 严格二值死锁审计（Strict Visual Quality Auditor）**：
  - 调用具备思考模式的推理级 VLM，设定严苛的二值（True/False）审计提示词（论文 Figure 12）：

::: details Qwen-VL 思维链审计系统 Prompt（点击展开）
```text
System Prompt:
You are a Strict Visual Quality Auditor. Your sole task is to perform a binary pass/fail audit on text-image pairs based on absolute prompt alignment and technical flawlessness.

Evaluation Criteria:
I. Semantic & Text-Image Alignment
- Object Completeness: Every entity, person, or item required in the prompt must be clearly visible and identifiable. Missing any requested element results in an immediate 'False'.
- Quantity Precision: The number of objects in the image must exactly match the count specified in the text.
- Text Rendering (OCR): Any text, words, or letters requested in the prompt must be rendered with 100% spelling accuracy, correct font style (if specified), and zero character distortion.
- Attribute & Color Consistency: All specified colors, materials, sizes, and specific properties of objects must be strictly followed.
- Spatial & Relational Logic: Objects must be in the exact positions described. Actions/interactions between objects must be logical.
- Negative Constraints: If the prompt specifies what not to include, the image must not contain those elements.

II. Technical & Aesthetic Quality
- Anatomical Integrity: Zero tolerance for "AI hallucinations" in biological structures. This includes extra/missing fingers, limbs, distorted faces, unnatural joints, or merged body parts, etc.
- Physical & Geometric Logic: Objects must follow the laws of physics (unless the prompt says otherwise). No floating objects, impossible perspectives, unrealistic interactions, or "melting" textures where surfaces bleed into each other.
- Image Artifacts: Zero tolerance for unintended blurring, watermarks, signature-like scribbles, garbled characters, distorted text, etc.
- Style & Medium Fidelity: The image must perfectly embody the requested style. If the style is inconsistent across the frame, it is a failure.

Decision Logic:
- Output 'True' ONLY if the image is a perfect realization of the prompt with zero technical defects.
- Output 'False' if there is even one minor discrepancy.
Constraint: Think step-by-step. Output ONLY the word 'True' or 'False'.
```
:::

- **过滤威力**：在此阶段**再次刷掉了约 70% 的生成图文对**，最终淬炼出 6,145,693 对无结构残缺、拼写严格对齐的极品图文对。

---

## 4. Fine-T2I 数据分布多维透视

### 4.1 核心大类分布（覆盖真实需求重心）

| 大类 (Category) | 占比 (%) | 细分亮点覆盖 |
| :--- | :--- | :--- |
| **People (人物与人像)** | **37.9%** | 人像肖像 (87.0万)、复杂活动 (59.7万)、各年龄段 (24.2万)、微表情与动作、体育运动 |
| **Nature (自然与场景)** | **27.8%** | 实体物品 (47.2万)、美食静物 (25.7万)、城市风光 (24.6万)、动植物生态、室内外环境 |
| **Text Rendering (文字渲染)** | **17.4%** | 简短文字 (40.1万)、场景文字 (28.4万)、**挑战性长文本 (21.7万)**、艺术手写体、招牌 |
| **Design (设计与界面)** | **10.6%** | 艺术排版 (23.6万)、海报设计 (10.6万)、**UI/UX 软件界面 (10.4万)**、信息图表与 Slide (6.9万) |
| **Rare Cases (长尾罕见场景)** | **6.3%** | 罕见概念组合、特殊透视、抽象空间想象（防概念遗忘） |

### 4.2 细粒度任务与组合能力（Task Composition）

在 Fine-T2I 中，除了通用单点描述，有超过 **36.9%** 的样本涉及双重复杂属性复合：
- 单任务样本：推理空间关系 (48.9万)、相对位置判定 (38.1万)、色彩绑定 (32.6万)、精准计数 (29.4万)；
- 复合双任务：色彩+推理 (17.6万)、位置+推理 (18.6万)、计数+推理 (13.3万)、色彩+位置 (15.1万) 等。

---

## 5. 实验验证：跨模型家族的性能质变

为了严谨验证 Fine-T2I 作为通用微调集的有效性，论文选取了并未经过工业界内部闭源高审美数据浸泡的两个代表性底座进行微调测试：
- **扩散架构代表**：`SD-XL`（LoRA 微调，学习率 $1 \times 10^{-4}$，训练约 1 epoch）；
- **自回归架构代表**：`LlamaGen`（全参数微调，学习率 $3 \times 10^{-5}$，训练约 1 epoch）。

### 5.1 人工双盲偏好评测（Image Arena 500 条实战 Prompt）

由于 GenEval 等自动评估基准 Prompt 过于固定，作者从 Artificial Analysis Image Arena 采样了 500 条真实世界高难度用户 Prompt 进行人工盲测：

| 评估基模型 | 微调数据集 | 视觉画质 (Visual Quality) 胜率 | 图文对齐 (Text Alignment) 胜率 |
| :--- | :--- | :--- | :--- |
| **LlamaGen (自回归)** | **Fine-T2I** | **80.7% (绝对碾压)** | **65.3%** |
| **SD-XL (扩散)** | **Fine-T2I** | **显著胜出（画面更干净、去塑料感）** | **显著胜出** |

### 5.2 对比经典开源 SFT 数据集（同条件竞赛）

在以 LlamaGen 为基模的同条件对比实验中，Fine-T2I 迎战主流微调集：

| 训练数据集 | 数据规模 | 局限与盲点 | 盲测综合胜率表现 |
| :--- | :--- | :--- | :--- |
| **T2I-2M** | 200万 | 画质参差不齐，低分辨率较多，存在明显肢体畸形 | 极低（视觉与对齐均落后） |
| **BLIP3o-60k** | 6万 | **训练 Loss 最低，但生成效果垫底**：由于样本背景过于单调简单，模型在训练时学到了走捷径（Shortcut Learning），生成真实场景时泛化崩溃 | 居中，不及 Fine-T2I |
| **Fine-T2I (本文)** | **630万** | 高分辨率、多长宽比、真实摄影与合成结合，无结构幻觉 | **断层第一 (大幅领先)** |

---

## 6. 核心工程启示与避坑经验 (Lessons Learned)

论文 Appendix B 极其真诚地披露了大规模文生图数据工程中的真实困境与经验，具备极高的实操借鉴价值：

::: warning 1. LLM 批量生成 Prompt 的隐形冗余陷阱
- **现象**：即使给不同的 GPU 赋予完全随机的种子，LLM（LLaMA-3）在批量生成 Prompt 时依然存在严重的“思维定势”，生成的句式和实体组合高度趋同；
- **教训**：提高采样温度只能部分缓解，必须在后端配备**基于语义向量（如 all-MiniLM）的硬聚类去重**，否则约 90% 的算力都会浪费在渲染内容实质相同的重复图像上。
:::

::: warning 2. 多重属性约束的自相矛盾
- 当提示工程同时要求“字数少于 10 词”、“包含 UI 界面设计”、“体现传统民间文化”、“执行色彩+计数任务”时，LLM 往往顾此失彼，生成不自洽的残缺 Prompt；
- **策略**：属性标签应作为下游检索与分析的弱监督元数据（Soft Metadata），不能作为硬逻辑盲信。
:::

::: tip 3. 审美指标（Aesthetic Score）与人类审美漂移
- 早期文生图倾向于将“色彩极度浓烈、高对比度、CG 渲染风”评为高美学分；
- **当代人类偏好**已经迅速转向“自然、真实摄影、柔和光影、无 AI 味”；
- 纯粹依赖单点标量美学打分模型极易筛选出过度饱和的塑料感图片，**必须引入摄影师真实作品（如 Pexels/Unsplash）作为真实审美的稳压锚点**。
:::

---

## 7. 知识卡片小结

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Fine-T2I 核心要素速记卡片                         │
├────────────────────────────────────────────────────────────────────────┤
│ • 规模体量：6.15M 合成数据 + 16.8万摄影师真实图 ≈ 6.31M 对 / 2TB       │
│ • 生成底座：Z-Image (单流 DiT) + FLUX2，覆盖 512 到 2560 全画幅       │
│ • 淘汰比率：从 44.8M 原始提示词到 6.15M 终版，总淘汰率 > 95%           │
│ • 核心去重：all-MiniLM-L6-v2 384维嵌入，128组分级，余弦阈值 0.8        │
│ • 核心质检：Aesthetic V2.5 > 5.5 + Qwen-VL-Thinking 零容忍二值审计     │
│ • 标注创新：Original (短) + PromptEnhancer (长) 双版本镜像配对        │
│ • 核心定位：填补开源模型与商业闭源模型在 SFT/CT 阶段的高清数据鸿沟    │
└────────────────────────────────────────────────────────────────────────┘
```
