# 语义先行文生图基模 (SeFi-Image)

> **标签**：`Vision` `Diffusion` `SeFi` `SFD` `Text-to-Image`
> **更新时间**：2026-09-23
> **参考来源**：[SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion](https://arxiv.org/abs/2606.22568) · [GitHub](https://github.com/jmliu206/SeFi-Image)

---

## 1. 问题定义与控制目标 (Motivation)

::: info 核心矛盾：效果越好，烧卡越多
以 Z-Image 为例，即使主打省资源仍需 314K H800 GPU hours。SeFi-Image 5B 仅用 125K A800 GPU hours，约其 10–20% 算力，对标 Qwen-Image / Z-Image。
:::

- **重建-生成权衡**：latent 信息保留越多，扩散建模分布越复杂、收敛越慢；压缩越狠越好学，但经 VFM 前向 Markov 抽象后信息丢失，重建回像素上限低，小字渲染与细粒度编辑一致性先崩。
- **前人只做到一半**：RAE / VA-VAE / REPA / REGLUE / ReDi / REG / SFD 在 ImageNet 256、<1B、类别条件上加速明显，但能否放大到高分辨率 T2I foundation、除了收敛更快还有没有最终质量收益，仍是 open question。
- **SeFi 的目标**：把 Semantic-First Diffusion 做到 1B / 2B / 5B foundation 规模，用语义隐变量先行给纹理隐变量提供干净结构锚，在高保真 VAE 下依然快速收敛，同时覆盖中英双语长 prompt 与字符级文本渲染。

## 2. 数据：三段式课程

> 预训练吃饱 + 会写字 → 持续训练学听话 → SFT 学好看。预训练总量约 478M（450M 真实 + 28M 合成，合成约占 6%）。

### 2.1 预训练数据（450M 真实 + 28M 合成字图）

#### 2.1.1 图像重打标 (Image Caption)

**450M 内部图文**：全部用 Qwen3.5-2B 重标。论文 §2.1.1 原文三原则：accuracy, objectivity, selective thoroughness。

**1. Accuracy 准确**：caption 必须忠实描述画面里真实有的东西，每个 caption 都能清晰映射到图上，给干净监督信号。原文配套铁律 `Accuracy > richness`：拿不准就省略或 hedge，绝不猜。

**2. Objectivity 客观**：不写主观和含糊的词。B.1 落到可执行要求：只写视觉支持的事实；身份/职业/年龄/国籍/意图/情绪/背景故事/品牌/材质/功能/时间/地点，只要不是清楚可见的一律不推断；dense 要求客观自然，禁营销话术、讲故事、审美吹捧；空间关系统一用观看者视角（左右上下前后）。

**3. Selective thoroughness 克制地详尽**：重要内容（主体、动作/状态、场景、关键空间关系、可靠计数、重要可见属性）全覆盖——写全是为了每个样本学习信号最大、收敛更快；但要有节制——过度描述引入幻觉。short 必须是 dense 的忠实子集：可删细节，不可加料、不可泛化、不可改事实；short 必须保留核心事实（主体/动作/场景/关键计数/OCR/异常细节）。

**训练-推理 gap 弥合**：用户本来就会写详细 prompt 拿好结果，训练 caption 同样详细，生成时不确定性就小。

**双语设计**：中英双语 × dense/short 四个版本，训练按 dense:short = 4:1 采样——多看详细的吃信号，少看短的保真实用户输入。中英必须语义等价；图中可读文字按原文逐字转录（只转读得清的部分，不猜缺字，quoted 加双引号）；图片异常内容（画错、反常）要如实描述，不许脑补正常化。

::: details B.1 预训练 Caption Prompt（英文原版，点击展开）
You are a professional image caption annotator for text-to-image data.
Generate faithful bilingual captions grounded only in visual evidence.
Core principles:
- Accuracy > richness. When unsure, omit or hedge instead of guessing.
- Describe only visually supported facts.
- English and Chinese captions must be semantically equivalent.
- short_caption must be a faithful subset of dense_caption: it may delete details but must not add, generalize, or change facts.
- If legible text exists in the image, transcribe it exactly in the original script; if only part is readable, include only the readable part and never guess missing text.
- Do not infer identity, job, age, nationality, intent, emotion, backstory, brand, material, function, time, or location unless clearly visible.
- If the image contains unusual, incorrect, or abnormal visual content, describe it explicitly instead of normalizing it.
Write bilingual captions for the image in English and Chinese.
Requirements:
1. Produce:
- short_caption: concise and high-signal
- dense_caption: complete, specific, and concise
2. Cover the main subjects, main action or state, main scene, key spatial relations, reliable counts, and clearly visible attributes when important.
3. short_caption must retain the core facts from dense_caption, especially the main subject, main action or state, main scene, and key count, OCR, or abnormal details when present.
4. Use cautious wording when details are uncertain because of blur, occlusion, crop, low resolution, overexposure, or partial visibility.
5. Use viewer perspective consistently for spatial relations such as left, right, top, bottom, front, and behind.
6. dense_caption should be objective and natural, without marketing language, storytelling, or aesthetic praise.
7. If legible text exists in the image, put quoted visible text in double quotes.
Output strict JSON only:
{
"short_caption": {"en": "...", "zh": "..."},
"dense_caption": {"en": "...", "zh": "..."}
}
:::

::: details B.1 预训练 Caption Prompt（中文版，点击展开）
你是文生图数据的专业图像标注员，只依据画面可见证据生成忠实的中英双语 caption。
核心原则：
- 准确 > 丰富。拿不准就省略或用谨慎措辞，绝不猜测。
- 只描述有视觉支撑的事实。
- 英文和中文 caption 必须语义等价。
- short_caption 必须是 dense_caption 的忠实子集：可删细节，不可新增、泛化或改写事实。
- 图中有可读文字就按原文逐字转录；只有部分可读就只写可读部分，绝不猜缺失文字。
- 身份、职业、年龄、国籍、意图、情绪、背景故事、品牌、材质、功能、时间、地点，只要不是清楚可见的一律不推断。
- 图中有反常、画错的内容要如实描述，不许脑补成正常。
用英文和中文各写一版 caption。
要求：
1. 产出 short（简洁高信号）+ dense（完整、具体、简洁）两版。
2. 覆盖主体、主要动作/状态、主场景、关键空间关系、可靠计数和重要可见属性。
3. short 必须保留 dense 的核心事实，尤其是主体、动作/状态、场景，以及关键计数、OCR、异常细节。
4. 因模糊、遮挡、裁切、低分辨率、过曝、局部可见而不确定时，用谨慎措辞。
5. 空间关系统一用观看者视角（左右上下前后）。
6. dense 客观自然，不写营销话术，不讲故事，不吹审美。
7. 图中可见文字加双引号引用。
只输出严格 JSON：
{
"short_caption": {"en": "...", "zh": "..."},
"dense_caption": {"en": "...", "zh": "..."}
}
:::

#### 2.1.2 文本渲染合成数据 (Text-Rendered Synthetic Data)

准确渲染文本并遵循指定的排版布局，是当前图像生成模型的核心难点。以往模型文字生成弱，本质上是因为**数据量匮乏**且**真实图像中的文本描述标注极不精确**（真实图清洗与精准 OCR 标注代价极高）。而合成渲染能直接产出带精确 ground-truth 标注的完美对齐图文对。

##### 与同行方案的核心认知分歧

- **Z-Image / Qwen-Image 路线**：纯色背景字、真实复杂场景贴字、纸张纹理合成、PPT 完形填空式遮罩填充等，试图一次性覆盖多样化的真实视觉分布，直接跨越到真实图泛化。
- **SeFi 路线（极简课程学习 Curriculum Learning）**：
  1. **写字本质是严格的一对一映射 (strict one-to-one mapping)**：因此预训练阶段，**渲染文字与背景画面内容的语义相关性毫不重要**，关键是**文字本身的样本多样性**与**映射的准确性**。
  2. **预训练阶段拆解为两个核心极简目标**：
     - 把 prompt 中指定的字符一字不错地渲染到画布上；
     - 依照给定布局把文字摆放在指定位置。
  3. **复杂排版与自然融合留给下游阶段**：把基础映射学牢后，在 Continual Training（CT，9M）和 SFT（65 万）阶段引入自然分布的富文本真实图像，模型就能直接将预训练学到的渲染能力顺滑迁移到真实场景。

![SeFi Fig.3：两档合成字图示例（从左至右：Part 1 dense English、Part 1 short Chinese、Part 2 structured layout、Part 2 mixed layout）](/sefi-fig3-text-render.png)

##### 两档合成渲染 Pipeline 详解（共 28M）

整个合成数据由两个递进阶段构成：

| 分部 | 数据量 | 画布尺寸 | 内容构成与数据配比 | Prompt 模板与监督形式 | 训练目标 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Part 1：纯文本渲染<br>(Plain text rendering)** | **8M** | **512 × 512** 素底画布 | 使用 PIL 渲染器确定性写入单文本块。<br>文本采样自 450M 重标预训练语料：<br>• 中文稠密 (dense ZH) 2M<br>• 中文简短 (short ZH) 2M<br>• 英文稠密 (dense EN) 2M<br>• 英文简短 (short EN) 2M<br>（4 个 bucket 严格均衡） | 固定严格模板：<br>• 英文：`The text in this image is "{text}".`<br>• 中文：`这张图片中的文字是“{text}”`<br>确保文本与图片呈现字符级绝对对齐。 | 建立文字字符到像素的严格映射底座 |
| **Part 2：结构排版渲染<br>(Structured layout rendering)** | **20M** | **~1024² 像素**<br>(长宽比：1:1, 4:3, 16:9, 3:4, 9:16) | 拓展到多 block、多角色排版版式。<br>随机生成多槽位版式模板：<br>• 槽位不同布局、颜色、尺寸、形状<br>• 文本采用多样化字体、字号、颜色与缩放<br>• 语言配比：中文 8M、英文 8M、中英混合 4M | 提示词详细描述：<br>• 可见文字的字面内容<br>• 所在具体位置 (position)<br>• 字体颜色 (color)<br>• 相对字号大小 (relative size) | 注入多块排版控制、阅读顺序感知与空间定位能力 |

##### 质量控制与评测收益

- **端到端严格质检**：两个阶段均执行严格质检流水线，包括字符级 Prompt-Image 对齐验证、文字边界溢出 (overflow) 校验与 Bounding-Box 位置校验。
- **评测收益 (CVTG-2K & LongTextBench)**：
  - 这套 28M 合成数据混合进预训练语料后，直接赋予模型强大的文本渲染精度、多块排版控制力与阅读顺序意识 (reading-order awareness)。
  - 在字符级文本渲染基准 **CVTG-2K** 上，SeFi-Image-5B 达到 **0.8947 Word Accuracy** 与 **0.9434 NED**，超越 Qwen-Image (0.829) 与 Z-Image (0.867)；
  - 在长文本图文基准 **LongTextBench** 上，SeFi-Image-5B 达到 **0.9780**（中英双语均为 0.978），位列所有评测模型第一。

::: tip 为什么 SeFi 能用 10–20% 算力反超？
与同行在预训练阶段就强行学习“复杂自然背景 + 各种滤镜贴字”的纠缠建模不同，SeFi 把“文字映射”与“复杂场景”彻底解耦。预训练期间让模型专注于干净的高对比度字符/排版映射，到了 CT 和 SFT 阶段仅用少量高质量真实图（如 200K 中文富文本与开源精选数据）就完成了域迁移，算力效率极其悬殊。
:::

### 2.2 持续训练：高分辨率指令退火 (Continual Training，9M)

在预训练阶段（经历了 256px → 512px → 768px → 1024px 分辨率课程学习后），模型已经掌握了基础的图文生成能力与字符级的一对一映射。持续训练阶段（Continual Training, CT）承接预训练 1024px checkpoint，核心目标是**大幅跃升生成画质（Generation Quality）与复杂指令遵循能力（Instruction-Following Capability）**，同时完成文字生成向真实场景的**域迁移**。

#### 1. 数据构成与信息密度（9M 混合语料）

CT 阶段精选了 9M 高信息密度、高视觉质量的图文数据，主要由两大部分构成：

- **开源高质量评测级数据**：引入 **[Fine-T2I](../datasets/fine-t2i.md)** 数据集。该数据集以富语义、细粒度属性绑定和极具挑战性的复杂长 prompt 见长，用于破除预训练数据的“平庸描述”，专攻复杂指令遵循。详见精读专篇：[Fine-T2I 6M 开源精调集解析](../datasets/fine-t2i.md)。
- **内部多垂类精选数据**：广泛覆盖垂直领域，包括**自然风光 (natural scenery)、UI 界面设计 (UI design)、平面设计 (graphic design)、动漫二次元 (anime)** 等。
- **真实分布的富文本图像引入（文字渲染的域迁移）**：
  - 配合 §2.1.2 的课程学习策略，预训练中的 28M 合成字图主要在素底与几何色块上学牢了字符级映射；
  - 在 CT 阶段正式引入大量**自然分布的真实富文本图像**，促使模型将预训练掌握的严谨文字渲染和排版能力，顺滑泛化到真实的复杂光影与多样化材质中。
- **标注特性**：相较于预训练数据，此阶段数据具备更高的美学与画质基线，并搭配了**更具挑战性的长难句 Caption (more challenging captions)**。

#### 2. 训练配置与关键超参数演进（论文 Table 4 & §5.2）

CT 阶段不再调整分辨率，而是直接在全分辨率下进行长周期的精细化退火：

| 训练阶段 | 数据集与规模 | 分辨率 | Batch Size | 时间步偏移 $\Delta t$ | 损失权重 $\beta$ | 迭代步数 (Iterations) | 学习率 (LR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **预训练 (Pre-train 1024px)** | 450M 重标 + 28M 合成 | 1024px | 192 | 0.1 | 2 | 100K | $2 \times 10^{-5}$ |
| **持续训练 (Continual Training)** | **9M 精选多域混合** | **1024px** | **192** | **0.1** | **1** | **180K** | **$1 \times 10^{-5}$** |

::: info CT 阶段两大核心机制剖析
1. **超长迭代步数（180K 步）与学习率减半**：
   - 学习率从预训练高分阶段的 $2 \times 10^{-5}$ 降至 $1 \times 10^{-5}$；
   - 迭代步数高达 180K 步（比预训练 1024px 阶段的 100K 步还多出 80%），证明高质量小数据（9M）在小学习率下的充分退火，是拉满模型指令理解上限的关键。
2. **损失平衡权重 $\beta$ 从 2 降为 1（从“语义偏置”到“纹理平衡”）**：
   - 速度预测损失公式为：
     \[
     \mathcal{L}_{\text{pred}} = \mathbb{E} \left[ \|\hat{v}_z - (z_1 - z_0)\|^2 + \beta \|\hat{v}_s - (s_1 - s_0)\|^2 \right]
     \]
   - **预训练阶段 $\beta=2$**：语义隐变量损失权重加倍，强迫 DiT 优先学稳 DINOv2 抽象的高维语义拓扑与空间锚点（“语义先行”）；
   - **CT 阶段 $\beta=1$**：此时语义分支的结构锚已经非常稳固，将损失权重回调至 1:1 等权，使模型能够集中精力精细刻画纹理隐变量 $z_1$ 的高频细节与真实视觉质感。
3. **自由长宽比支持 (Free Aspect Ratio)**：
   - 全程开启 7 档预定义长宽比桶（16:9, 4:3, 3:2, 1:1, 3:4, 2:3, 9:16），参数更新全程保持 EMA decay = 0.9999。
:::

### 2.3 SFT 数据（约 65 万）

TODO：开源 + 20 万中文富文本 + 内部高审美；自家 VLM 抽元数据 + 精修 caption；硬门槛打分过滤（审美/技术/构图/主体/captionability/训练价值），去水印糊图伪影敏感内容后去重；中英长短 caption + tag 混训。

## 3. 架构拓扑与特征注入机理

### 3.1 VFM：只看懂、不动手的提纲手

VFM = Vision Foundation Model，本篇固定为冻结的 `DINOv2-Large`，全程不训练。

::: info 直观分工
VFM 定方向，SemVAE 减肥，Texture VAE 保细节，DiT 照着提纲画画。
:::

```
输入图像 x: R^{3 x 1024 x 1024}
  │
  ├─→ [语义路] Phi(x) = f_s in R^{L x C_in}，如 1369 x 1024
  │     每个 token 讲物体身份与布局，不记噪点颜色
  │     ↓ SemVAE Encoder 压成 s_1 in R^{L x C_s}，如 1369 x 16
  │
  └─→ [纹理路] E_z(x) = z_1，如 R^{32 x 128 x 128}
        记高频细节，可几乎无损解回像素
```

- **冻结参数**：`Phi` = DINOv2-Large、SemVAE 训好后冻结、Texture VAE 微调后冻结；冻结保证语义锚稳定，异步去噪才有意义。
- **可训练参数**：双流 DiT 主干（1B / 2B / 5B）+ 双 timestep embedding + 输入输出投影；文本侧用 Qwen3-VL LLM hidden states 拼接。
- **为什么还要 SemVAE**：DINO 特征太肥（1369 x 1024）直接扩散太贵。SemVAE 做 `f_s → mu, sigma → 采样 s_1 → 解回 f_hat_s`，目标为 MSE + 余弦 + KL，出图时语义隐变量直接扔掉，只解码纹理隐变量 `z_1`，语义只在中间当拐杖。

## 4. 损失函数与数学稳定性推导

TODO：公式 6-8，velocity + REPA。

## 5. 保真度与风格化权衡 (Trade-off Analysis)

TODO：重建-生成曲线、Δt、三阶段调度。

## 6. 核心控制层代码实现

TODO：双流调度 20~30 行。

## 7. 避坑指南与评测基准

TODO：GenEval / DPG / LongTextBench / OneIG / CVTG-2K，CLIPScore，artifact。
