# 非对称保持编辑：先保住背景，再做指令修改 (Asymmetric Preservation Editing)

> **标签**：`Vision` `Image Editing` `FLUX Kontext` `Attention`  
> **更新时间**：2026-09-23  
> **参考来源**：RealFit 2609.25881 · SR-Edit 2609.02504 · IABEdit 2609.12691

---

## 1. 问题定义与控制目标

任务：指令编辑，比如“把猫换成狗”，改的地方要改对，不该动的地方一点不能动。

现在模型的两个具体毛病：
1. 改不全、 spills 到背景，PIE-Bench上背景PSNR和编辑区CLIP很难双高；
2. 现有mask矫正本身是heuristic，mask一错，矫正反而引入新伪影。

要攻的点：条件分支（原图特征）不能被噪声分支污染。RealFit在试衣上已证明对称注意力是主因，编辑同理。

## 2. 架构拓扑与特征注入机理

只动注意力mask和调制，其余全冻。

* **Locked Backbone**：FLUX.1 Kontext-dev或Qwen-Image-Edit，MMDiT全冻结。
* **Trainable**：LoRA + 一个轻量aligner，推理时aligner直接扔掉，零额外开销。

记噪声token为 $z_n$，条件token为 $z_c$，拼成 $[z_n; z_c]$。标准做法是全连，本方案改成：

$$ \text{Attn}_{mask} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} $$

含义：噪声可以看条件，条件只能看自己，看不到噪声。高频细节就不会被随机性带偏。

调制也拆开：噪声分支跟随当前步 $t$，条件分支固定在一个最优值 $t^{\star}$，信号全程不衰减。条件KV算一次后面全复用，省约75%计算。

mask提纯：每 $k$ 步从模型自己的预测里取edit/non-edit区，只对非编辑区做对齐矫正，不破坏原采样轨迹。

## 3. 损失函数与数学稳定性推导

$$ \mathcal{L}_{total} = \mathcal{L}_{FM} + \beta \|A_{align}(G) - F_{vl}(x_{gt})\|_2^2 $$

* $\mathcal{L}_{FM}$：正常去噪损失；
* 第二项：冻结VLM提GT的空间描述子 $F_{vl}$，可训aligner从生成图复现，残差当梯度，教模型改哪、保哪。

::: info 为什么推理零开销
$F_{vl}$ 和 $A_{align}$ 只在训练出现，推理只剩LoRA + mask， latency 反而因KV-cache下降。
:::

## 4. 保真度与指令遵循权衡

两条线：保真（PSNR/SSIM/LPIPS/DINO-I）vs 遵循（CLIP-T/编辑区CLIP）。

单向流开太死会保守：背景满分，但该改的没改。做法是扫 $t^{\star}$ 和阻断时机，只在早步全阻断、晚步放开，看CLIP-T掉不掉。红线：CLIP-T掉>2%而PSNR涨<0.5dB就判定过保守，回退。

## 5. 核心控制层代码实现

```python
# z_n: [B, N, d], z_c: [B, M, d]
x = torch.cat([z_n, z_c], dim=1)
# block C->N: 条件看不到噪声
bias = torch.zeros(B, 1, N+M, N+M, device=x.device)
bias[:, :, N:, :N] = -1e9
h = joint_attn(x, bias)  # N->N, N->C, C->C 保留

# 调制解耦：噪声跟 t，条件固定 t_star
z_n = modulate(z_n, t)
z_c = modulate(z_c, t_star)  # 常数
# 条件KV只算一次，后续复用
if step == 0: cache_KV(z_c)
```

## 6. 避坑指南与评测基准

数据集：PIE-Bench九类必报，MagicBrush、EmuEdit抽查。指标：背景PSNR/SSIM/LPIPS + 编辑区CLIP + 整图CLIP-T。

基线：P2P、MasaCtrl、PnP、FlowEdit、SR-Edit。消融四项：UIF / DTM / 自提纯 / VLM loss，缺一不可。

::: warning 避坑要点
1. $t^{\star}$ 太大欠表达，风格化任务先扫 $t^{\star}$；
2. 自提纯阈值敏感，报阈值鲁棒曲线，别只给最优点；
3. 和RealFit区分写清楚：它做试衣，本方案做通用指令编辑。
:::
