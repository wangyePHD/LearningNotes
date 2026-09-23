# ERNIE-Image 轻量 DPO 与多教师蒸馏 (百度 8B)

> **标签**：`Vision` `RL` `DPO` `Distillation` `Aesthetics`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2605.25347](https://arxiv.org/abs/2605.25347)

---

## 1. 问题定义与控制目标

8B 开源模型补齐指令遵循、文字渲染、美学三短板。RL 策略是"轻量"：base + reward 够强时 **few-step DPO** 即达标，hacking 最小；再用 MT-DMD 防蒸馏 capability drift。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：DPO 参考模型、FLUX.2 VAE、 Ministral-3B 文本编码器（降显存）；
- **可训练参数**：8B 单流 DiT；美学模型 ERNIE-Image-Aes（ArtiMuse-8B 微调）独立训练；
- **注入机理**：Flow Matching 速度场 L2 误差差做隐式奖励；蒸馏时专家委员会按 $(x_t, \sigma, c, \mathcal{O})$ 动态路由。

## 3. 损失函数与数学稳定性推导

$$
\mathrm{Diff} = \|v_{\mathrm{pol}}(x_t^{\mathrm{win}}) - v_t^{\mathrm{win}}\|^2 - \|v_{\mathrm{pol}}(x_t^{\mathrm{lose}}) - v_t^{\mathrm{lose}}\|^2
$$
$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}[\log\sigma(-\beta(\mathrm{Diff}_{\mathrm{policy}} - \mathrm{Diff}_{\mathrm{ref}}))], \quad \beta = 0.05
$$

**双 Anchor**（L2 无上界会被刷大 rejected 误差 → representation collapse）：
$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{DPO}} + 0.35\,\mathbb{E}[\ell^{\mathrm{win}}] + 0.15\,\mathbb{E}[\ell^{\mathrm{lose}}]
$$
连 lose 侧也锚（0.15），防 rejected 流形崩坏——可直接抄的配方。

MT-DMD：$K$ 个域专家（文字/艺术/布局…），门控 $\mathcal{W}_k(x_t,\sigma,c,\mathcal{O})$ 做非对称梯度拓扑（DM 问艺术专家要全局风格，CA 问文字专家要局部拼写），沿去噪轨迹换手（高噪布局→低噪纹理）。

::: info 美学标注协议
pairwise + 瑞士轮（ELO 太贵），单标注员独立完整 tournament；ERIA-1K 开源，SRCC 0.74 vs 次优 0.45。奖励模型先治标注偏置，RL 效果上游在数据。
:::

## 4. 效果权衡

GenEval 0.89（开源最高档），LongText 0.973；DPO 步数宜少不宜多；Turbo 8 NFE。

## 5. 核心控制层代码实现

```python
# Flow Matching DPO + 双 anchor（ERNIE 配方）
def ernie_dpo_loss(v_pol_w, v_pol_l, v_ref_w, v_ref_l, v_t_w, v_t_l, beta=0.05):
    def l2(a, b): return (a - b).pow(2).mean()
    diff_pol = l2(v_pol_w, v_t_w) - l2(v_pol_l, v_t_l)
    diff_ref = l2(v_ref_w, v_t_w) - l2(v_ref_l, v_t_l)
    l_dpo = -torch.log(torch.sigmoid(-beta * (diff_pol - diff_ref)))
    return l_dpo + 0.35 * l2(v_pol_w, v_t_w) + 0.15 * l2(v_pol_l, v_t_l)
```

## 6. 避坑指南与评测基准

- **评测**：GenEval / LongText / OneIG + ERIA-1K 美学 + 人评 7 维；
- **要点**：DPO 步数少 + 第三方质量分双轨监控（呼应 GDRO）；蒸馏走多教师 + 动态路由，与 Swift OPD 同趋势——2026 共识 = 分专家、再统一。
