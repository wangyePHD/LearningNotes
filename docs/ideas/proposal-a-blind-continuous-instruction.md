# 盲复原统一指令：用退化图自身代替文本提示 (Blind Continuous Instruction)

> **标签**：`Vision` `Image Editing` `All-in-One Restoration` `LoRA`  
> **更新时间**：2026-09-23  
> **参考来源**：ImIR 2609.25267 · AcFlow 2609.10723 · Qwen-Image-Edit

---

## 1. 问题定义与控制目标

任务：一张模型同时处理去雨、去雾、去噪、去模糊、低光、JPEG六种退化，推理时不给退化标签。

现有做法的问题很具体：用文本做条件，比如`“remove rain”`，三个毛病：
1. 要给每种退化手写prompt，换个数据就得重调；
2. 文本是离散的，给不出“雨有多大、图有多暗”；
3. 盲测时直接崩，ImIR里报道文本版去雨从32.5dB掉到17.5dB。

要攻的点：保布局不难，难的是让模型知道修什么、修多强，且不知道退化类型时也不崩。

## 2. 架构拓扑与特征注入机理

只动两处，其余全冻。

* **Locked Backbone**：Qwen-Image-Edit整体冻结，含Qwen2.5-VL编码器 $E_{vl}$ 和MMDiT去噪器，参数占比约99%。
* **Trainable**：LoRA rank=16/64 + 一个2层MLP映射器 $M_{\phi}$，<5M参数，占比<1%。

走法分两路：

1. 结构路：退化图 $y$ 走VAE进DiT，保证位置、边缘不动。
2. 指令路：不走文本，text prompt留空，直接用图算指令：
$$c = M_{\phi}(E_{vl}(y)) \in \mathbb{R}^{d}$$

训练时拿干净图的 $E_{vl}(x)$ 当老师，推理时只有 $y$。任务标签 $\tau$ 可选，用FiLM做偏置；盲版本直接把 $\tau$ 合成一个，不输入标签。

强度控制很直接，做线性插值：
$$c(\alpha) = (1-\alpha)E_{vl}(y) + \alpha c, \quad \alpha \in [0,1]$$

$\alpha=0$约等于不修，$\alpha=1$全力度修。低光这种目标不唯一的任务，扫一遍 $\alpha$ 就出一族结果。

## 3. 损失函数与数学稳定性推导

总损失两项：

$$ \mathcal{L}_{total} = \mathcal{L}_{FM}(x, y, c) + \lambda \|c - E_{vl}(x)\|_2^2 $$

* $\mathcal{L}_{FM}$：标准flow matching去噪损失，学 $y \to x$；
* 第二项：指令对齐，把预测的 $c$ 往干净 embedding 拉，$\lambda$ 取0.1量级先跑。

::: info 为什么第二项必要
没有它，$M_{\phi}$ 会偷懒输出全零向量，全靠LoRA硬记，盲测必崩。加了它，mapper必须闭合退化-干净的embedding差，消融时把这一项去掉看PSNR掉多少即可验证。
:::

## 4. 保真度与修复强度权衡

核心就一条曲线：$\alpha$ 从0扫到1。

* $\alpha$ 小：PSNR高、LPIPS低，但低光看起来还是暗；
* $\alpha$ 大：视觉变亮干净，但可能过曝、细节 hallucination。

低光、去雾重点报这条曲线，去雨、去噪报单点即可。盲版本和带标签版本PSNR差应<0.5dB，文本基线差是15dB，这是关键对比。

## 5. 核心控制层代码实现

```python
# E_vl: frozen Qwen2.5-VL, M: 2-layer MLP + FiLM
with torch.no_grad():
    e_deg = E_vl(y)          # [B, L, d], 退化图特征
    e_clean = E_vl(x)        # [B, L, d], 仅训练用
c = M(e_deg, tau=None)       # [B, L, d], 预测的干净指令
alpha = 1.0
c_run = (1-alpha)*e_deg + alpha*c
loss_fm = flow_matching_loss(dit(y, c_run), x)
loss_align = F.mse_loss(c, e_clean.detach())
loss = loss_fm + 0.1 * loss_align
loss.backward()  # 只更新 M + LoRA
```

推理时把`E_vl(x)`那行删掉，调`alpha`即可。

## 6. 避坑指南与评测基准

先跑两个任务验证，别一次铺六个：去雨Rain100L + 低光LOL，单卡<5h。如果这两项超文本LoRA 1dB以上，再铺RESIDE、BSD68、GoPro、LIVE1。

基线：同backbone的Text-LoRA、Edit2Restore、ImIR复现。指标：PSNR/SSIM/LPIPS为主，CLIP-T/DINO-I看编辑保持，加20张人评。

::: warning 避坑要点
1. VLM本身怕噪声，mapper易学shortcut，必须做oracle实验：直接喂 $E_{vl}(x)$ 看上限，mapper应达到上限的90%以上；
2. 和ImIR撞车风险：差异点放在连续 $\alpha$ 族解和盲编辑统一上，不要只报平均PSNR；
3. 低光插值可能非单调，先画 $\alpha$ 曲线确认再写论文。
:::
