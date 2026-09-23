# GDRO 组级直接奖励优化 (Group-level Direct Reward Optimization)

> **标签**：`Vision` `RL` `GDRO` `Reward Hacking` `Offline`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2601.02036](https://arxiv.org/abs/2601.02036) · 港大 + CUHK + 通义 · [代码待查]

---

## 1. 问题定义与控制目标

把 LLM 的组级奖励成功搬到 rectified flow T2I，但在线 RL 与 flow 模型三处不匹配：**采样贵**（rollout 主导训练）、**要随机 sampler**（flow 给定初噪即确定，ODE→SDE 近似有域外风险）、**reward hacking**（Flow-GRPO 把 OCR 分刷高、字放大占满图、细节崩坏）。目标：组级奖励 + 全离线 + sampler 无关 + 抗 hacking。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：参考模型 $v_{\mathrm{ref}}$、显式奖励模型（OCR / GenEval 打分器）；
- **可训练参数**：DiT 速度场（论文用 FLUX.1-dev + LoRA rank 32 + EMA）；
- **注入机理**：隐式奖励只由加噪图 + 两模型速度预测差算出，无需 rollout、无需随机 sampler，任意 timestep 可算：
$$
s_\theta(x,t) = -\beta\,\mathbb{E}_{t,v}\left[\|v - v_\theta(C)\|_2^2 - \|v - v_{\mathrm{ref}}(C)\|_2^2\right]
$$
其中 $C = (x_t(c), t, c)$，$\beta = T\beta_{\mathrm{KL}}$。

## 3. 损失函数与数学稳定性推导

同 prompt 一组图 $(x_1,\dots,x_k)$，显式奖励 softmax 成目标分布 $Q$，隐式奖励 softmax 成 $P_\theta$，Top-1 交叉熵：
$$
\mathcal{L}_{\mathrm{top\text{-}1}} = \log\sum_j e^{s_\theta(x_j,t)} - \sum_i q(i,\tau)\, s_\theta(x_i,t)
$$
$\tau$ 控制尖锐度。推广到全组排序（Plackett-Luce 视角，逐位对剩余集合求和）：
$$
\mathcal{L}_{\mathrm{GDRO}} = \sum_{i=1}^{k-1}\left(\log\sum_{m=i}^k e^{s_\theta(x_m,t)} - \sum_{j=i}^k q_i(j,\tau)\, s_\theta(x_j,t)\right)
$$

::: info 退化关系
$\tau \to 0$ 退化为纯排序目标；$k=2, \tau \to 0$ 退化为 DPO。GDRO 是 DPO 的组级、显式奖励泛化。
:::

Top-1 稳定项（拉开 margin 时 top-1 似然也在掉，伤画质）：
$$
\mathcal{L}_{\mathrm{final}} = \mathcal{L}_{\mathrm{GDRO}} + \gamma\, \|v - v_\theta(x_t(c),t,c)\|_2^2 \cdot M_{\mathrm{top1}}
$$

防 hacking 评测——corrected score（UnifiedReward 三维：alignment / coherence / style）：
$$
r_{\mathrm{corrected}} = r(\hat{u} - 3) + 0.2
$$
Flow-GRPO 原始分涨、修正分 100 GPU-h 后掉头；GDRO 双涨且峰值最高。

## 4. 对齐—效率—保真权衡

- 同分效率：OCR **2×**（29.6 vs 59.7 GPU-h），GenEval **3.7×**（68.4 vs 250 GPU-h）；
- DPO（k=2）：OCR 不稳、GenEval 直接 collapse——组级是稳定性关键；
- $\beta$ 按任务调：OCR $\beta{=}12$ 稳，GenEval $\beta{=}6$（改布局需更大分布偏移）；组大小 k=2 崩、k=4/6/8 稳。

::: warning 避坑要点
只看原始奖励必被 hacking 骗；$\beta$ 太小 collapse，太大欠优化；离线法无在线探索，需要主动寻优的奖励场景不适用。
:::

## 5. 核心控制层代码实现

```python
# 隐式奖励：只需加噪图 + 两模型速度预测（离线、可任意t）
def implicit_reward(v_theta, v_ref, v_target, beta):
    return -beta * ((v_theta - v_target).pow(2).mean()
                    - (v_ref - v_target).pow(2).mean())

# GDRO Top-1 CE：Q 来自显式奖励 softmax(tau)，P 来自隐式奖励 softmax
def gdro_top1_loss(s_theta, rewards, tau=0.05):
    Q = torch.softmax(rewards / tau, dim=0)
    logP = torch.log_softmax(s_theta, dim=0)
    return -(Q * logP).sum()  # CE(Q, P_theta)

# corrected score 双轨评测
def corrected_score(r, u_hat):
    return r * (u_hat - 3) + 0.2
```

## 6. 避坑指南与评测基准

- **评测**：OCR 精度 + GenEval + UnifiedReward 三维 + 用户研究；上线 corrected score 双轨；
- **超参起点**：k=6，$\tau{=}0.05$，OCR $\beta{=}12,\gamma{=}0.5$ / GenEval $\beta{=}6,\gamma{=}1.0$，lr 3e-4，预生成每 prompt 16 图；
- **局限**：纯离线无探索；corrected score 只是 hacking 趋势代理，非精确度量。
