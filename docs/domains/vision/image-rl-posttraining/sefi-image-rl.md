# 语义先行文生图基模 (SeFi-Image)

> **标签**：`Vision` `Diffusion` `SeFi` `SFD` `Text-to-Image`
> **更新时间**：2026-09-26
> **参考来源**：[SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion](https://arxiv.org/abs/2606.22568) · [GitHub](https://github.com/jmliu206/SeFi-Image)

---

::: warning 前置知识
本文 §3 的 SFD 机制（复合隐空间、双时间步 $\Delta t$、三阶段掩码调度、REPA 重解码）全部来自 **Semantic-First Diffusion (CVPR 2026)**，本文只做 T2I 化改造。**未读前置篇请先看**：[语义先行扩散范式 (SFD)](../sfd-semantic-first-diffusion.md)。
:::

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

### 2.3 监督微调：高审美收敛与多粒度指令对齐 (Supervised Fine-Tuning，650K)

在预训练和持续训练（CT）奠定了坚实的生成底座与指令理解能力之后，监督微调阶段（Supervised Fine-Tuning, SFT）的目标是**收敛模型输出分布（narrow the output distribution），拉满画面的视觉美学上限，并实现精确的细粒度指令遵循**。相比前序阶段，SFT 采取了最严苛的质量红线。

#### 1. 数据构成与严苛标准（约 650K 样本）

SFT 阶段精选了约 65 万对极致质量的图文数据，来源构成如下：

- **高质量开源精选数据**；
- **20 万中文富文本图像 (200K Chinese text-rich images)**：专门强化复杂中文版式、汉字书法与场景字体的精细渲染；
- **内部高审美样本集 (High-aesthetic samples)**：大幅提升构图、光影与艺术品味。
- **质量红线**：每个样本必须同时满足**顶级审美质感 (strong aesthetics)、清晰稳定的构图 (clear composition) 以及明确聚焦的主体 (well-defined subject)**。

#### 2. 自研专有 VLM 二阶段标注与精修流水线 (Appendix B.2)

由于 SFT 对 Caption 的准确性要求远高于预训练，团队构建了基于自研 VLM 的**“提取元数据 $\to$ 深度精修”**两阶段自动化打标工程：

```
[原始高质图像]
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 1：元数据抽取 Prompt (Metadata Extraction)                         │
│ • 分类识别 (11类 Enum：landscape, portrait, art, poster, ui, slide 等) │
│ • 标签抽取 (tags_en / tags_zh：风格/类型/主体/布局，每语种 6~16 个)      │
│ • 安全与水印：NSFW / 暴力 / 侵入性水印判断                              │
│ • OCR 结构化提取：位置 (position)、样式 (style)、精确字面 (visible_text)│
│ • 初版四路 Caption：short_en, short_zh, long_en, long_zh                │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ 输出结构化 Context JSON
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 2：带上下文的深度精修 Prompt (Detailed Caption Refinement)         │
│ • 事实真理源：以原图为绝对基准 (Source of Truth)，Context 仅供辅助参考  │
│ • 严禁质检废话 (Forbidden)：严禁写“无水印、无模糊、太小看不清”等汇报话术│
│ • 稳定阅读顺序：海报/UI/图表按“先背景层、再上层设计”，从上至下从左至右 │
│ • 字符级绝对继承：visible_text 中所有 OCR 字符必须逐字带双引号写入     │
└─────────────────────────────────────────────────────────────────────────┘
```

::: details B.2 SFT 两阶段标注 Prompt 核心原则（点击展开）
- **OCR 完整性压倒一切**：若存在可读文字，字符级转录完整性优先于简短度；短 caption 末尾必须以 `Visible text: "t1"; "t2"` 完整列出所有字符，不漏掉任何次级按钮或小字。
- **拒绝脑补推断**：禁止推测未明显可见的身份、职业、年龄、国籍、意图、情绪、品牌、材质与时间地点。
- **异常内容如实描述**：画面若存在反常、画错或残缺部分，客观据实记录，不得自我合理化纠正。
:::

#### 3. 八维 VLM 评分与硬过滤门禁 (Hard Filtering Gate, Figure 4)

打标完成后，系统独立调用 VLM 对图像进行多维度离散评分（1~5 分），并施加严格的硬门禁过滤：

| 评估维度 | 评分性质 | 维度内涵与判定依据 | 硬门禁规则 |
| :--- | :--- | :--- | :--- |
| **Aesthetics (美学)** | 正向核心 | 视觉吸引力、光影质感、色彩和谐度 | 必须达到最低门槛 |
| **Technical Quality (技术画质)** | 正向核心 | 边缘锐利度 (sharpness)、曝光度、压缩伪影、分辨率与渲染稳定性 | 必须达到最低门槛 |
| **Composition (构图)** | 正向核心 | 镜头视角、主体留白、几何透视与平衡感 | 必须达到最低门槛 |
| **Subject Clarity (主体清晰度)** | 正向核心 | 主体目标是否清晰明确、无混乱背景干扰 | 必须达到最低门槛 |
| **Captionability (可打标性)** | 正向核心 | **画面可见内容能否被客观、忠实地描述**（杜绝过度抽象与混乱无序） | **必须达到最低门槛** |
| **Training Value (综合训练价值)** | 正向核心 | 综合画质、语义深度、文字渲染与数据适配度的全局可用性 | 必须达到最低门槛 |
| **Artifacts (生成伪影)** | 负向抑制 | 肢体多指畸形、AI 融化物、浮空错位（得分越低代表问题越少） | 高分即淘汰 |
| **Political Sensitivity (政治敏感)**| 负向抑制 | 政治人物、敏感符号与意识形态（得分越低代表问题越少） | 高分即淘汰 |

::: warning SFT 一票否决硬淘汰规则
- 绝对剔除：涉黄、重度暴力血腥、可见水印/LOGO、严重模糊破损、明显生成伪影、政治敏感；
- **文字专属淘汰**：画面中若出现 **`unreadable key text`（关键核心文字不可读或乱码）**，直接整条废弃；
- 类别标记准入：仅保留打标为核心训练样本 (core)、多样性补充 (diversity)、文字排版 (text-layout) 或风格样例 (style) 的样本，剔除一切边缘与低价值样本，最终严格去重后导出。
:::

#### 4. SFT 训练配置与长文本上下文扩展（论文 Table 4 & §5.3）

SFT 阶段聚焦于在全分辨率下实现精炼对齐：

| 参数项 | SFT 阶段配置 | 对比前序阶段的变化与工程意图 |
| :--- | :--- | :--- |
| **数据规模** | **~650K** | 极致淬炼的小样本高质量集（占预训练规模的 ~0.14%） |
| **训练分辨率** | **1024px** | 保持与 CT 一致的全分辨率基准 |
| **Batch Size** | **192** | 维持全局大批次稳定梯度 |
| **时间步偏移 $\Delta t$** | **0.1** | 维持异步解耦的语义先行架构先验 |
| **损失平衡权重 $\beta$** | **1** | 语义与纹理损失等权平衡刻画 |
| **迭代步数 (Iterations)** | **10K 步** | 短周期快速收敛（仅需 1 万步即可完成高审美与对齐收敛，防止过拟合） |
| **学习率 (LR)** | **$1 \times 10^{-5}$** | 保持精细微调的低学习率 |
| **文本编码器 Context Length** | **$512 \to 1024$ (翻倍)** | **大幅扩展上下文长度**，从容容纳信息密集的超长提示词与复杂排版指令 |
| **文本形式配比** | **多粒度混训** | 包含中/英文 Dense Caption、Short Caption 与 Tags，赋予模型对从单短词到长难段落的鲁棒响应力 |

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

三阶段总目标（论文 Eq.6–8）：

$$
\mathcal{L}_{\text{pred}}=\mathbb{E}\Big[\big\|\hat{v}_z-(z_1-z_0)\big\|_2^2+\beta\big\|\hat{v}_s-(s_1-s_0)\big\|_2^2\Big]
$$

$$
\mathcal{L}_{\text{REPA}}(\psi,\phi)=-\mathbb{E}\big[\mathcal{L}_{\text{sim}}(y^*,\,h_\phi(h_t))\big],\qquad h_t=f_\psi([s_{t_s},z_{t_z}],[t_s,t_z])
$$

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{pred}}+\lambda\,\mathcal{L}_{\text{REPA}}
$$

::: info REPA 目标对齐的是「同一份」DINOv2 特征
$y^*=f(x_1)$ 既是 REPA 的监督目标，**又是 SemVAE 的输入**。因此 $\mathcal{L}_{\text{REPA}}$ 可读作「把噪声语义隐变量 $s_{t_s}$ 解码回干净语义表征」——比原版 REPA 的「从零蒸馏分析」更易优化，故对齐深度只需取第 2 层（原文消融 depth 2 最优，depth 8 次之）。
:::

::: warning $\beta$ 的课程调度是 SFD 原文没有的
| 阶段 | $\beta$ | 意图 |
| :--- | :--- | :--- |
| 预训练 | **2** | 语义权重加倍，强迫 DiT 先学稳 DINOv2 抽象的结构锚 |
| CT / SFT | **1** | 结构已稳，降至 1:1 让模型精刻画纹理高频细节 |

$\beta$ 过大会压制纹理学习（原文消融：$\beta$=8 时 FID 3.96 vs $\beta$=2 时 3.03）。
:::

## 5. 保真度与风格化权衡 (Trade-off Analysis)

### 5.1 异步调度在 $(t_s,t_z)$ 平面上的形状

$$
t_s\sim\mathcal U(0,1+\Delta t),\qquad t_z=\max(0,\,t_s-\Delta t),\qquad t_s\leftarrow\min(t_s,1)
$$

::: danger 两个钳位各管一件事，且顺序不可换
- `max(0,·)`：$t_s<\Delta t$ 时纹理锁死在 $t_z=0$（纯噪声）$\Rightarrow$ Stage I 只动语义。
- `min(·,1)`：$t=1$ 已是干净，语义没有「更干净」$\Rightarrow$ Stage III 语义钉死。
- **$t_z$ 必须用未钳位的 $t_s$ 算**。若先 `min` 再减 $\Delta t$，$t_z$ 上限被压到 $1-\Delta t$，**纹理永远画不完**。
- 采样上界取 $1+\Delta t$ 而非 1：否则 $t_z$ 到不了 1。
:::

```
t_z
 1.0 |                              ● (1,1) 两路皆净
     |                              │  ③ 垂直 Stage III
1-Δt |                  ● (1,1-Δt)   │     语义钉死，纹理收尾
     |              ╱                │
 Δt |      ● (Δt,0)                 │  ② 对角 Stage II
     |      │                       │     斜率恒 1，偏移恒 Δt
 0.0 |●─────┘  ① 水平 Stage I       │     语义初始化，纹理纯噪声
     +------------------------------+----→ t_s
      0        Δt                  1.0
```

| 阶段 | $t_s$ | $t_z$ | 掩码 $(M_s,M_z)$ | 行为 |
| :--- | :--- | :--- | :--- | :--- |
| I 语义初始化 | $[0,\Delta t)$ | $0$ | $(1,0)$ | 只画蓝图 |
| II 异步生成 | $[\Delta t,1]$ | $[0,1-\Delta t)$ | $(1,1)$ | 边画边描，恒定领先 |
| III 纹理收尾 | $1$ | $[1-\Delta t,1]$ | $(0,1)$ | 精修细节 |

$$
\hat v=[M_s\odot\hat v_s,\ M_z\odot\hat v_z],\qquad M_s\in\{0,1\}^{B\times C_s\times H\times W},\ M_z\in\{0,1\}^{B\times C_z\times H\times W}
$$

::: tip 「不增加推理步数」的技巧
时间范围从 $[0,1]$ 拉长到 $[0,1+\Delta t]$（Stage III 需要），但**同比放大步长间隔**，总步数不变。完成后**只解码 $z_1$**，$s_1$ 丢弃。
:::

### 5.2 $\Delta t$ 随分辨率递减

| 分辨率 | 256px | 512px | 768px | 1024px |
| :--- | :--- | :--- | :--- | :--- |
| $\Delta t$ | 0.2 | 0.2 | **0.1** | **0.1** |

原文未给该调度消融（属工程观察）。SFD 原文在 256px 上最优值为 0.3。**4 步 DMD2 蒸馏时同样保留 $\Delta t=0.1$ 的领先规则**，以免破坏三阶段结构。

### 5.3 重建-生成：SFD 让你敢 aggressively 微调 VAE

高保真 latent 分布更复杂、扩散更难收敛；压缩狠则重建上限低。SFD 的作用是**额外提供条件**：

$$
\text{更丰富的条件} \;\Longrightarrow\; \text{纹理隐变量待建模分布更窄} \;\Longrightarrow\; \text{更易生成}
$$

所以纹理 VAE 可以直接往重建质量上堆（本篇用微调 FLUX.2 VAE）：

| VAE (Kodak) | PSNR↑ | SSIM↑ | LPIPS↓ |
| :--- | :--- | :--- | :--- |
| SD1.5 | 26.66 | 0.7294 | 0.1452 |
| FLUX.1 | 32.37 | 0.9063 | 0.0554 |
| FLUX.2 | 33.18 | 0.9194 | 0.0442 |
| **FLUX.2-finetuned (本篇)** | **36.40** | **0.9565** | **0.0235** |

| VAE (OmniDoc-TokenBench, 3042 样本) | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | NED↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RAE-DINOv2-B | 14.32 | 0.3261 | 0.2290 | 18.21 | 0.0392 |
| VAVAE | 17.50 | 0.6905 | 0.0974 | 4.45 | 0.3488 |
| HunyuanImage-3.0 | 22.66 | 0.8672 | 0.0650 | 3.49 | 0.7753 |
| Wan2.2 | 21.67 | 0.8577 | 0.0525 | 3.05 | 0.8310 |
| Qwen-Image-VAE-2.0-f16c128 | 30.45 | 0.9706 | 0.0167 | 0.79 | 0.9617 |
| **FLUX.2-finetuned (本篇)** | **30.91** | **0.9718** | **0.0133** | **0.46** | **0.9648** |

::: warning 这张表就是「重建-生成权衡」的全部证据
纯 VFM 表征路线（RAE）PSNR 仅 14.32、NED 0.0392——小字直接崩；语义增强但纠缠（VA-VAE）NED 只 0.3488。**只有「独立语义旁路 + 高保真纹理 VAE」才能同时拿下 0.9648 NED 和 0.46 FID。** SeFi 全部字号渲染收益（CVTG-2K / LongTextBench 第一）都建立在这张表上。
:::

## 6. 核心控制层代码实现

双流异步调度核心（对应论文 Eq.9–10，训练侧 Eq.2–4）：

```python
import torch

def sample_train_timesteps(bs: int, dt: float, device="cuda"):
    """双时间步采样。顺序铁律：先算 t_z，再钳位 t_s。"""
    u = torch.rand(bs, device=device) * (1.0 + dt)   # 扩展区间
    t_z = torch.clamp(u - dt, min=0.0)               # 纹理滞后，锁死 >= 0
    t_s = torch.clamp(u, max=1.0)                    # 语义截断，锁死 <= 1
    return t_s, t_z


def sfd_masks(t: float, dt: float):
    """三阶段掩码可化简为两个阈值比较，无需显式分支。"""
    #  t <  dt        -> (1, 0)  Stage I   语义初始化
    #  dt <= t <  1   -> (1, 1)  Stage II  异步生成
    #  1  <= t <=1+dt -> (0, 1)  Stage III 纹理收尾
    return (t < 1.0), (t >= dt)


@torch.no_grad()
def sfd_sample(v_theta, s_shape, z_shape, dt=0.1, n_steps=50, device="cuda", **cond):
    """步数不变，仅把时间范围拉长到 1+dt。"""
    s, z = torch.randn(s_shape, device=device), torch.randn(z_shape, device=device)
    grid = torch.linspace(0.0, 1.0 + dt, n_steps + 1, device=device)

    for i in range(n_steps):
        t, t_n = grid[i].item(), grid[i + 1].item()
        t_s, t_z = min(t, 1.0), max(0.0, t - dt)
        M_s, M_z = sfd_masks(t, dt)
        step = t_n - t
        v_s, v_z = v_theta(torch.cat([s, z], dim=-1), [t_s, t_z], **cond)
        s = s + step * M_s * v_s      # Stage III 后语义冻结
        z = z + step * M_z * v_z      # Stage I 期间纹理不动

    return texture_vae.decode(z)     # 只解码纹理隐变量
```

## 7. 避坑指南与评测基准

### 7.1 主结果（SeFi-Image-5B）

| 基准 | 5B | 最强对手 | 判定 |
| :--- | :--- | :--- | :--- |
| GenEval Overall | **0.88** | Qwen-Image 0.85 / Z-Image 0.84 | ✅ 胜（1B 即 0.87 打平 Qwen-Image） |
| DPG-Bench Overall | 87.27 | Qwen-Image 88.32 / Z-Image 88.14 | ❌ 略逊 |
| LongTextBench Avg | **0.978** | JoyAI-Image 0.963 / Qwen-Image-2512 0.960 | ✅ 第一 |
| CVTG-2K NED / Word Acc. | **0.943 / 0.895** | JoyAI-Image 0.937 / 0.874 | ✅ 双项第一 |
| CVTG-2K CLIPScore | 0.816 | Qwen-Image 0.802 | ✅ |
| OneIG-EN Overall | **0.5606** | Z-Image 0.5460 / Qwen-Image 0.5390 | ✅ 第一 |
| OneIG-ZH Overall | **0.5379** | Z-Image 略低 | ✅ |

::: tip 读表要点
**长文本 + 字符级渲染 + 双语指令是 SFD 的主战场**（LongTextBench / CVTG-2K / OneIG 全部第一），因为语义分支提供结构骨架，擅长组织信息密集的长 prompt。**弱项是 DPG 的 Global 维度（88.24，全场最低）**，且 1B/2B 的长文本能力断崖（0.855 / 0.847）——长文本理解强依赖模型容量。
:::

### 7.2 RL 后训练增益（5B w/ vs w/o，见 [专题笔记 §C.2](./rl-comparison-2026.md)）

| 基准 | w/o RL | w/ RL | Δ |
| :--- | :--- | :--- | :--- |
| GenEval Overall | 0.87 | 0.88 | +0.01 |
| LongTextBench Avg | 0.9665 | **0.9780** | **+0.0115** |
| OneIG-ZH Overall | 0.5335 | **0.5379** | +0.0044 |
| OneIG-EN Overall | 0.5541 | **0.5606** | +0.0065 |
| DPG-Bench Overall | 87.45 | 87.27 | −0.18 |

RL 主要补文字渲染与指令遵循，**组合能力基本持平、DPG 略降**。

### 7.3 Turbo（4 步 DMD2 蒸馏）

| | GenEval | DPG | LongTextBench | 差距 |
| :--- | :--- | :--- | :--- | :--- |
| 5B full-step | 0.88 | 87.45 | 0.967 | — |
| 5B-Turbo | 0.87 | 86.45 | 0.922 | 1–4 分 |

组合任务掉分最小（语义分支在反向过程早期就锁定高层结构），**文字密集任务掉分最多**（字符渲染需要中间去噪步）。

### 7.4 五个避坑要点

1. **不要先钳位再算 $t_z$**（见 §5.1），纹理永远画不完。
2. **不要照搬 SFD 的 $\Delta t=0.3$**。本篇 1024px 用 0.1，且随分辨率递减。
3. **不要把 $\beta$ 一直锁在 2**。CT/SFT 必须降到 1，否则纹理细节被压制。
4. **不要用纯语义表征当纹理 latent**。表中 RAE 的 NED 0.0392 是前车之鉴。
5. **4 步蒸馏必须保留 $\Delta t$ 领先规则**，否则三阶段结构被压平。

::: warning 论文层面的信息缺口
- **RL 阶段的 prompt 池完全未披露**（来源、总量、生成方式均无）。原文仅说明「按可评估性筛选 + 每条带 capability tag + 400 组 × 12 候选」，对比 DiffusionNFT 直接用 FlowGRPO 的 GenEval/OCR train split。
- **纹理 VAE 微调收了重建收益，但 DPG Global 维度反而最低（88.24）**——高保真 latent 可能带来审美/整体观感上的轻微钝化，论文未解释。
- 消融仅在 50M 内部数据、256px、32×A800 条件下做过（Fig.9/10），**$\Delta t$ 与 $\beta$ 本身没有消融**。
:::

