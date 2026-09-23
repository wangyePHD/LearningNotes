# Lens：训练效率重思考与 3.8B 小模型的越级 (Lens Efficiency)

> **标签**：`Vision` `Diffusion` `Efficiency` `Dense Caption` `Lens`
> **更新时间**：2026-09-23
> **参考来源**：[Lens: Rethinking Training Efficiency for Foundational Text-to-Image Models](https://arxiv.org/abs/2605.21573) · [GitHub](https://github.com/microsoft/Lens) · [HuggingFace](https://huggingface.co/microsoft/Lens)

![recaption 与效率演进 lineage](/sefi-caption-lineage.svg)

---

## 1. 问题定义与控制目标

基座文生图越做越大（Z-Image 6B、Qwen-Image 20B、Hunyuan-Image-3.0 MoE 80B），Z-Image 光预训练就要约 314K H800 GPU hours。Lens 只用 3.8B 参数、约 Z-Image 19.3% 算力，对标甚至超过 6B+ 模型，回答一个问题：**同样的质量能不能少烧卡**。

::: info 效率三因素（论文原话级）
训练效率 = 模型大小（每步成本）× 每 batch 数据信息密度（每步学到多少）× 收敛速度（要多少步）。只压参数不够，后两者才是被忽视的大头。
:::

- **算力口径**：Lens 192K A100 hours（BF16 312 TFLOPS）vs Z-Image 314K H800 hours（989.5 TFLOPS），按峰值 TFLOPS 归一化；recaption 预处理不计（一次处理可复用）。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：FLUX.2 VAE、GPT-OSS 语言编码器（20B MoE、激活 3B、24 层）；预训练只优化 diffusion transformer。
- **可训练参数**：48 个 MMDiT block 的去噪主干（3.8B），RMSNorm + RoPE（image 侧）。
- **文本注入**：取 GPT-OSS 第 4/12/18/24 层特征沿通道拼接 + 线性 adapter 对齐到 latent 维度；双分支分别处理图文再融合。
- **Reasoner（独立外挂）**：默认 GPT-5.5，可换开源模型；把含糊用户输入改写成训练 caption 分布的详细 prompt。用 GPT-OSS 当 reasoner 时零额外显存。

## 3. 损失函数与数学稳定性推导

预训练标准 flow-matching MSE，VAE 与文本编码器冻结：

$$ \mathcal{L} = \mathbb{E}_{z_0, z_1, t}\left[\| \hat{v}_\theta(z_t, t, c) - (z_1 - z_0) \|^2\right], \quad z_t = (1-t)z_0 + t z_1 $$

其中 $z_1 = \mathrm{VAE}(x)$，$c$ 为 GPT-OSS 多层拼接文本特征，$t$ 用 logit-normal 采样（512² 时 $\mu = 1.06$；混分辨率时 $\mu(n)$ 从 $n = 256$ 的 1.0 线性插值到 $n = 4096$ 的 1.3）。

后训练用 DiffusionNFT + GPT-4.1-mini 按 rubric 打 reward（见第 6 节 RL 数据设计）。

## 4. 数据信息密度与收敛权衡 (Trade-off Analysis)

**文本侧密度**：Lens-800M，800M 图全用 GPT-4.1 打长英文 caption，平均约 109 词；图里原文字保留原文。动机有三：① 网页 alt-text 又短又错，噪声监督浪费容量；② 用户本来就写长 prompt，训练对齐推理分布；③ 实测 dense-only 训练效果最好（Lens-Toy 消融：1.2B 主干 + Qwen3-0.6B encoder，在 Lens-130M 上 Brief / Detailed / Mixed 三档，GenEval 上 Detailed 胜）。

**图像侧密度**：batch 内混 3 种面积（$512^2/768^2/1024^2$）× 9 种长宽比 = 27 个 bucket，多分辨率学全局到局部、多比例学构图。副产品是推理泛化：没见过的比例（如 5:4、6:7）和 1440² 都能画，省掉昂贵的高分辨率训练。

**收敛侧选择**：VAE 不看 rFID、直接在 Lens-130M 上做 T2I 实测，FLUX.2 VAE 生成最好且收敛最快；强语言编码器同时带来英文训练→中法等多语言泛化，省掉多语言图文数据。

::: warning 避坑要点
VAE 选型别信 rFID 和 ImageNet 类条件生成这种代理指标，直接在 T2I 管线上实测；RL prompt 必须覆盖预训练分布（Lens-RL-8K），否则某些输入类型会退化。
:::

## 5. 核心控制层代码实现

```python
# 混合分辨率 bucket 采样：3 面积 x 9 长宽比 = 27 桶，按面积配 batch 以拉平 wall-clock
BASE_AREAS = [512**2, 768**2, 1024**2]
ASPECTS = ["1:2", "9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9", "2:1"]
BATCH_PER_BASE = {512**2: 24, 768**2: 10, 1024**2: 6}  # 高分辨率桶算量大，batch 调小

def build_buckets():
    buckets = {}
    for area in BASE_AREAS:
        for ar in ASPECTS:
            h, w = resolve_hw(area, ar)   # 如 512^2 + 1:2 -> 352x704
            buckets[(area, ar)] = (h, w)
    assert len(buckets) == 27
    return buckets

def mu_for_tokens(n):  # logit-normal 参数随 token 数插值
    return 1.0 + (1.3 - 1.0) * (n - 256) / (4096 - 256)

# 文本特征：GPT-OSS 4 层拼接 + adapter
text_feat = concat([gpt_oss.layers[i](prompt) for i in (4, 12, 18, 24)], dim=-1)
text_feat = linear_adapter(text_feat)  # 对齐 image latent 维度
```

## 6. 避坑指南与评测基准

- **预训练配方**：512² 先训 400K iters（128×A100，batch 3072，lr 2e-4 常数，AdamW 0.9/0.999，bf16 + grad ckpt，clip 1.0）→ 混分辨率继续 400K iters（lr 1e-4）。
- **RL（Lens-RL-8K，8406 条）**：分类驱动构造（Human/Object/Animal/Plant/Scene/Food/Event/Fictional/Text/UI 十大类→细类→具体 item），每条随机抽 1–4 个描述维度（属性/空间/计数/交互/颜色）让 GPT-4.1 组 prompt；每 prompt 10 条 sample-aware rubric + 1 条全局 rubric；每步 48 对 × 24 图，训 180 步（64×A100）。消融：全集 > 1/2 > 1/4，去掉 text 类 prompt 则 CVTG 文本项掉。
- **成绩（Table 2，20 步）**：OneIG EN/ZH 0.557/0.525、GenEval 0.930、LongText 0.937、CVTG 0.869/0.951/0.814，多项开源最佳；Turbo 4 步基本不掉。
- **推理**：默认 reasoner + 20 步 CFG 5.0；H100 上 1024² 3.15s，Turbo 0.84s。
