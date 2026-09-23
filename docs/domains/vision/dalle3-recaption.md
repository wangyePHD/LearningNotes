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
- **范式演进**：DALL-E 3（2023，SSC→DSC）→ Lens（2026，GPT-4.1 平均 109 词×800M，dense-only 最优）→ SeFi（2026，Qwen3.5-2B×450M）。SeFi 就是这条线的当代形态，三原则原文级详细 + B.1 提示词中英可折叠版已收录在 SeFi 笔记 §2.1，买一送一不用两边维护。

完整原文级展开（含 B.1 提示词中英可折叠版）见 SeFi 笔记 §2.1：[语义先行文生图基模 → 2.1 预训练数据](/domains/vision/image-rl-posttraining/sefi-image-rl)。
