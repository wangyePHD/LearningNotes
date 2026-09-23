# Proposal C：小物体尺度保真虚拟试穿能否用scale token + 维度奖励一次解决错位与刷分？

> **标签**：`Virtual Try-On` `Fashion` `Mask-free` `Reward`
> **记录时间**：2026-09-23
> **状态**：🧪 [值得单卡跑个玩具 Demo]
> **目标期刊**：Pattern Recognition / Engineering Applications of AI / IEEE TMM（中科院一区）
> **算力预算**：1x A100，JewelTry复现 + JVTO子集扩展约2~3周

---

## 1. 灵感触发与背景

衣服试穿已卷，珠宝/眼镜/手表还是蓝海：

* `2609.16626 JewelTry`：mask-free + scale adapter编码真实尺寸 + 单向条件注意力 + JVTO-Bench四类三元组。
* `WearWow ECCV26`：原生2K多衣，Adaptive Token Packing + Multi-dim Reward。
* `2608.29804 DAT`：7维保真评价轮廓/颜色/领口袖型/装饰/纹理/细节/logo，8B超GPT-5.5。
* `2609.13259 TryOnReward`：foveated一致性reward + pairwise/margin防RFT reward hacking，自建100K。
* `2609.18510 DiT-Garment / 2608.26714 LiveVVT / 2608.30450 FlowVVTON`：mask-free视频/实时流是趋势。

矛盾：FID/SSIM测不出“花纹对了但大小错了”，通用VLM reward会走全局捷径。

## 2. 核心猜想 Hypothesis

设饰品真实尺寸为 $s \in \mathbb{R}^3$（长/宽/厚），人体解剖先验为 $h$。若A（将 $s$ 编码为scale token与图像token拼接做in-context学习 + 单向注意力保几何）成立，通过引入B（7维foveated reward自适应加权RFT），则C（视觉保真+尺度精度+背景保持三优）在不破坏D（mask-free泛化）前提下成立。

$$ e_s = \text{MLP}(s) \in \mathbb{R}^{d}, \quad X = [e_s; z_{ref}; z_{person}] $$

$$ R = \sum_{k=1}^{7} w_k(t) \cdot r_k, \quad w_k \propto \text{欠优化程度} $$

::: tip 期刊喜欢的点
真实尺度误差mm级 + 可解释维度分 + 电商落地故事，比纯FID好讲10倍。
:::

## 3. 方法设计

1. **Scale adapter**：产品尺寸归一化后MLP成token，学饰品-人体相对比例。
2. **单向条件注意力**：饰品只自注意力，噪声可看饰品，反向阻断，保高频刻字/纹理，加attention refinement loss。
3. **维度reward RFT**：复用DAT思路训7头reward，foveation校准 grounding到相关区域，RFT时自适应聚合欠优化维度，防刷分。
4. 基座：JoyAI-Image-Edit / FLUX.1 Kontext + LoRA，mask-free端到端。

## 4. 实验计划

数据：VITON-HD/DressCode保通用性 + JVTO-Bench + 自采2000珠宝/眼镜2K三元组。

指标：SSIM/LPIPS/FID + 尺度误差 + DAT 7维Balanced Acc/SROCC/PLCC + 人类偏好A/B。必须报：有/无scale token的尺度误差对比，有/无foveation的reward hacking对比。

基线：OOTDiffusion、IDM-VTON、JewelTry、Oxygen-TryOn、FLUX.2。

::: info 最小Demo
先跑戒指/耳钉单类500对，验证scale token使尺度误差降>20%即继续。
:::

## 5. 潜在坑点

* 真实尺寸标注贵：先用相对比例+合成数据warm-start。
* 小物体像素占比<2%，attention易忽略：foveated loss加权 + 高分裁剪训练。
* 与JewelTry区分：本提案加多件2K + 维度RFT闭环，不只单件前向。

## 6. 参考

* JewelTry 2609.16626， WearWow， DAT 2608.29804， TryOnReward 2609.13259， Oxygen-TryOn
