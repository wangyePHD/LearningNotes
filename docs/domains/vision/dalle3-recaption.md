# 用更好的 Caption 改进图像生成：DALL-E 3 的 Recaption 范式

> **标签**：`Vision` `Diffusion` `Caption` `Recaption` `Prompt Following`
> **更新时间**：2026-09-23
> **参考来源**：[Improving Image Generation with Better Captions](https://cdn.openai.com/papers/dall-e-3.pdf)（Betker et al., OpenAI, 2023，DALL-E 3 系统报告）

![recaption 与效率演进 lineage](/sefi-caption-lineage.svg)

---

## 1. 问题定义与控制目标

当时的文生图模型有一个顽疾：**follow 不住详细 prompt**，经常丢词、混淆语义。论文的诊断直指训练数据：大规模图文对的 caption 来自人类随手写的 alt-text，只讲主体、不讲背景和常识关系，常见缺失有四类——① 背景物件（如厨房水槽、人行道 stop 牌）② 位置与数量 ③ 颜色尺寸等常识细节 ④ 图中文字；更糟的是网上 caption 经常是错的（广告、meme 混进 alt-text）。

::: info 核心论点
这些缺陷都可以用合成 caption 解决：在高度描述性的生成 caption 上训练，prompt following 稳定提升。这就是 recaption 范式的开山实验。
:::

## 2. 架构拓扑与特征注入机理

两段式流水线，captioner 与生成模型解耦：

- **可训练参数（一）图像 captioner**：语言模型 + 冻结 CLIP 图像编码器 $F(i)$ 做条件（像素太多，直接条件代价太大，CLIP 给压缩表示）；
- **可训练参数（二）文生图模型**：在重标后的数据集上从头训练，标准扩散目标；
- **冻结参数**：CLIP 图像编码器（只当特征提取器）；
- **推理外挂**：GPT-4 做 caption upsampling（见第 4 节），与 Lens 的 reasoner 同构。

## 3. 损失函数与数学稳定性推导

Captioner 先按下式做语言建模（$t = [t_1, \dots, t_n]$，$\Theta$ 为待优化参数）：

$$ \mathcal{L}(t) = \sum_j \log P(t_j \mid t_{j-k}, \dots, t_{j-1}; \Theta) $$

再把 CLIP 图像特征 $F(i)$ 和隐变量 $z_j$ 接进来变成条件 captioner（仿 Yu et al. 2022a，与 CLIP 目标联合预训练）：

$$ \mathcal{L}(t, i) = \sum_j \log P(t_j \mid t_{j-k}, \dots, t_{j-1}; z_j; F(i); \Theta) $$

::: info 证明细节
Base captioner 训出来是个好 captioner，但和人写 caption 一个毛病：不爱讲细节。所以又做了两轮微调（见第 5 节）：第一轮只描述主体的短 caption 数据 → SSC；第二轮长而详尽（环境、背景、图中文字、风格、色彩）→ DSC。全量 T2I 数据用微调后的 captioner 重标一遍。
:::

## 4. 描述性与噪声的权衡：混合比例 (Trade-off Analysis)

- **caption 类型对照**：GT-only / 95% SSC / 95% DSC 三个模型。用 GT caption 和 DSC 分别评：合成 caption 训的模型在 GT 上略优、在合成 caption 上明显优——**用合成 caption 没有坏处**；合成 caption 上评测曲线方差小得多（recaption 近似平均操作），且所有模型净 CLIP 更高（合成 caption 与图绑定更紧）。
- **混合比例**：DSC 占 65% / 80% / 90% / 95% 四档，65% 全面落后被 drop，比例越高 CLIP 越好。
- **代价与解法**：高比例合成 caption 会让模型适应长描述分布，短 prompt 采样会出分布。用 GPT-4 把用户短 prompt upsample 成详细描述即可（附录 C 的 prompt），还能顺手消解复杂关系。这是今天所有系统 reasoner/prompt 改写的祖师爷。

## 5. 核心控制层代码实现

```python
# Recaption 两阶段流水线（论文 §2.1.1 伪代码）
def build_captioner(base_lm, clip, ssc_data, dsc_data):
    model = joint_pretrain(base_lm, clip)      # Eq(2): CLIP 条件 + LM 联合目标
    model = finetune(model, ssc_data)          # 第一轮：只讲主体 -> SSC captioner
    model = finetune(model, model) if False else finetune(joint_pretrain(base_lm, clip), dsc_data)
    return model                               # 第二轮：长详尽描述 -> DSC captioner

def recaption_dataset(images, dsc_captioner, blend=0.95):
    out = []
    for img in images:
        synth = dsc_captioner.describe(img)    # 对每张图生成 DSC
        cap = synth if rand() < blend else img.alt_text
        out.append((img, cap))                 # 95% 合成 + 5% 原生
    return out

def sample_with_upsampling(user_prompt, t2i):
    detailed = gpt4_upsample(user_prompt)      # 短 prompt -> 长详细描述
    return t2i.generate(detailed)              # 在训练分布内采样
```

## 6. 避坑指南、评测基准与范式演进

- **评测**：自动评测（CLIP，GT caption 与合成 caption 双口径）+ 人工评测（prompt following / coherence / aesthetics 三维，附录有人评界面）；论文只讲 caption 带来的提升，**不含 DALL-E 3 训练实现细节**。
- **DALL-E 3 的能力点**：空间感知、文本渲染、特异性（specificity）、安全去偏， ironic 的 pizza 例（quarter-sized pizza vs pizza-sized quarter）就靠 upsampling 消歧。
- **范式演进**：DALL-E 3（2023，SSC→DSC）→ Lens（2026，GPT-4.1 平均 109 词×800M，dense-only 最优）→ SeFi（2026，Qwen3.5-2B×450M）。下面把 SeFi 的 caption 三原则按论文原文级展开，它就是这条线的当代形态。

### 6.1 SeFi 的 Caption 三原则（论文 §2.1.1 原文级详细）

> 出处：SeFi-Image §2.1.1，450M 内部图全用 Qwen3.5-2B 重标。原文三原则：accuracy, objectivity, selective thoroughness。

**1. Accuracy 准确**：caption 必须忠实描述画面里真实有的东西，每个 caption 都能清晰映射到图上，给干净监督信号。原文还配套一条铁律 `Accuracy > richness`：拿不准就省略或hedge，绝不猜。

**2. Objectivity 客观**：不写主观和含糊的词。B.1 落到 7 条可执行要求：只写视觉支持的事实；身份/职业/年龄/国籍/意图/情绪/背景故事/品牌/材质/功能/时间/地点，只要不是清楚可见的一律不推断；dense 要求客观自然，禁营销话术、讲故事、审美吹捧；空间关系统一用观看者视角（左右上下前后）。

**3. Selective thoroughness 克制地详尽**：重要内容（主体、动作/状态、场景、关键空间关系、可靠计数、重要可见属性）全覆盖——写全是为了每个样本学习信号最大、收敛更快；但要有节制——过度描述引入幻觉。short 必须是 dense 的忠实子集：可删细节，不可加料、不可泛化、不可改事实；short 必须保留核心事实（主体/动作/场景/关键计数/OCR/异常细节）。

**训练-推理 gap 弥合**：用户本来就会写详细 prompt 拿好结果，训练 caption 同样详细，生成时不确定性就小。这正是 DALL-E 3 §3.5 upsampling 的镜像：那边是推理时把短变长，这边是训练时直接喂长。

**双语设计**：中英双语 × dense/short，训练按 dense:short = 4:1 采样——多看详细的吃信号，少看短的保真实用户输入。中英必须语义等价；图中可读文字按原文逐字转录（只转读得清的部分，不猜缺字， quoted 加双引号）；图片异常内容（画错、反常）要如实描述，不许脑补正常化。

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
