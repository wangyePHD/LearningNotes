# 小物体试穿：先对准大小，再谈像不像 (Scale-Aware Try-On)

> **标签**：`Vision` `Virtual Try-On` `Fashion` `Reward`  
> **更新时间**：2026-09-23  
> **参考来源**：JewelTry 2609.16626 · DAT 2608.29804 · TryOnReward 2609.13259

---

## 1. 问题定义与控制目标

任务：戒指、耳钉、眼镜、手表上身。不是衣服，难在三点：
1. 小，像素占比常<2%，attention直接忽略；
2. 硬，刻字、花纹错一点就假；
3. 位置大小错一点就假，FID/SSIM根本测不出“花纹对但戴大了”。

现有通用VLM打分走全局捷径，背景好看就给高分，饰品错了不管。要攻的是尺度对准，不是单纯好看。

## 2. 架构拓扑与特征注入机理

* **Locked Backbone**：JoyAI-Image-Edit或FLUX.1 Kontext，冻结。
* **Trainable**：LoRA + scale MLP + 7头reward，总量小。

做法两步：

1. 真实尺寸编码。饰品长宽厚 $s \in \mathbb{R}^3$ 归一化后过MLP成一个token：
$$e_s = \text{MLP}(s) \in \mathbb{R}^{d}, \quad X = [e_s; z_{ref}; z_{person}]$$
模型in-context学饰品相对人体的比例，没这个token，大小全靠猜。

2. 单向注意力。饰品只做自注意力，噪声可以看饰品，反向阻断，刻字纹理不被污染。加attention refinement loss把注意力压到饰品区。

mask-free端到端，不依赖parsing/pose，遮挡靠模型自己推理。

## 3. 损失函数与数学稳定性推导

前向损失就是正常扩散损失加注意力约束。RFT阶段用维度奖励：

$$ R = \sum_{k=1}^{7} w_k \cdot r_k $$

7维照搬DAT：轮廓、颜色、领口袖型、装饰结构、材质纹理、细节、logo/刻字。每维foveation grounding到对应区域打分，不看整图。

权重 $w_k$ 按欠优化程度自适应给，练得差的分头多练，防止刷平均分。reward训练用pairwise + margin，同时学相对排序和绝对分。

::: info 为什么能防刷分
通用reward看整图，模型把背景做漂亮就能混高分。分维+foveation后，刻字错一笔对应头直接打低分，混不过去。
:::

## 4. 保真度与尺度精度权衡

报两类数，缺一不可：

* 像不像：SSIM/LPIPS/FID；
* 大小对不对：尺度误差，长宽相对误差百分比。

经验线：加scale token后尺度误差应降>20%，不降就别往下做。RFT后7维最低分项应上涨，平均分涨但最低分不涨就是刷分，没用。

## 5. 核心控制层代码实现

```python
# s: [B,3] 真实尺寸, z_ref: 饰品token, z_p: 人像token
e_s = scale_mlp(norm(s))          # [B,1,d]
x = torch.cat([e_s, z_ref, z_p], dim=1)
# 单向：ref只能看自己+scale
bias = block_C_to_N(x)            # 同Proposal B
h = attn(x, bias)
# RFT
r = [head_k(fovea_crop(h, box_k)) for k in range(7)]
R = (w * stack(r)).sum()          # w按欠优化自适应
```

小物体训练加高分裁剪，loss按面积反比加权，不然梯度被背景淹没。

## 6. 避坑指南与评测基准

数据：VITON-HD/DressCode保通用，JVTO-Bench四类必跑，自采2000张珠宝眼镜2K三元组做差异化。

基线：OOTDiffusion、IDM-VTON、JewelTry、Oxygen-TryOn。必须报两张消融表：有/无scale token的尺度误差，有/无foveation的最低分变化。

::: warning 避坑要点
1. 真实尺寸标注贵，先用相对比例+合成数据warm-start，别一上来就采3D；
2. <2%像素任务，batch里必须保证饰品裁剪占比，不然attention学不到；
3. 和JewelTry区分：它只做单件前向，本方案加多件2K + RFT闭环。
:::
