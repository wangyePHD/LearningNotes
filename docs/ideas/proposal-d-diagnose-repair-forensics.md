# Proposal D：多轮AI编辑的数字 ripple 能否用检测-诊断-修复闭环统一治愈？

> **标签**：`Forensics` `AIGC Detection` `Artifact Correction` `Agentic`
> **记录时间**：2026-09-23
> **状态**：🔬 [理论猜想]，待小数据验证后转🧪
> **目标期刊**：Information Fusion / IEEE TIP（中科院一区取证/修复口）
> **算力预算**：1x 4090，检测头 + 修复LoRA约1~2周

---

## 1. 灵感触发与背景

* `2608.28302 FUSED`：取证+语义稀疏MoE，联合输出图像分+像素mask，跨生成器评测。
* `2609.02640`：real/全合成/局部篡改三分类+分割，ACM MM Workshop。
* `2609.16832`：局部水印55变换benchmark，几何/生成式编辑最致命。
* `2609.11317 Mi-Ripple`：多轮引用编辑的格纹ripple，频域notch+结构平滑+cleaned再生，残差0.08~0.44。
* `GenShield`：检测+修复统一自回归，VCoT diagnose-then-repair + curriculum + STOP准则。

缺口：检测只给分不修，修复只修不解释，多轮退化无成对数据。

## 2. 核心猜想 Hypothesis

设 $I_a$ 为异常图，$I_c$ 为干净目标。若A（global-local重构差可训练-free定位 + VLM生成结构化缺陷描述 $T_{diag}$ ）成立，通过引入B（指令引导修复预热 + VCoT自纠多轮 + 终止态学习），则C（检测AUC/AP + 修复LPIPS/MUSIQ双SOTA且跨生成器泛化）在不破坏D（干净图零改动）前提下成立。

$$ \Delta = \|R_{full}(I) - R_{patch}(I)\| - \gamma \cdot \text{complexity}(I) $$

$$ I^{k+1} = G(I^{k}, \hat{T}^{k}_{diag}), \quad \hat{T}^{k}_{diag} \sim P_{diag}(I^{k}, Q) $$

$$ \mathcal{L} = \mathcal{L}_{FM} + \lambda \mathcal{L}_{AR}, \quad (Q,I_c) \to (T_{stop}, I_c) $$

::: tip 为什么是新赛道
检测-修复闭环 + 可解释终止，比纯二分类检测好发，且工业风控刚需。
:::

## 3. 方法设计

1. **GLARE式定位**：共享autoencoder整图vs切块重构差 + 语义复杂度校准，得可疑mask，无需训练。
2. **Stage1 指令修复**：$(I_a, T_{diag}) \to I_c$ 强监督建先验。
3. **Stage2 VCoT自纠**：$(Q, I_a) \to (\hat{T}_{diag}, I')$ 迭代，$I_c$ 作终止态学STOP。
4. **数据引擎**：用高级编辑器按缺陷描述合成artifact-restored大对，补空白。

## 4. 实验计划

检测：GenImage + OpenSDID跨生成器，报AUC/AP + IoU定位。修复：自造多轮编辑ripple集 + GenShield-Set思路，报LPIPS/MUSIQ/MANIQA + GPT-4o人评。必须报：检测开/关对修复增益，VCoT轮数-质量曲线，干净图误修率。

基线：GLARE、FUSED、Mi-Ripple、Qwen-Image-Edit直修、GPT-Image。

::: warning 证伪线
若训练-free定位在文档/人像上误检>15%，需回退到轻量监督头，不硬撑。
:::

## 5. 潜在坑点

* 合成缺陷分布与真实多轮分布gap：混合真实多轮编辑 + 合成，各半。
* 多轮自纠易过编辑：STOP阈值调优 + 干净图负样本。
* 与GenShield区分：本提案聚焦多轮ripple/细粒度伪影 + 开放benchmark，不只通用artifact。

## 6. 参考

* FUSED 2608.28302， 2609.02640， Watermark Benchmark 2609.16832， Mi-Ripple 2609.11317， GLARE ECCV26， GenShield
