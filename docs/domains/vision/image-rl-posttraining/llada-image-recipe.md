# LLaDA-Image 全开源配方：无 RLHF 也 SOTA (Inclusion AI)

> **标签**：`Vision` `Distillation` `TwinFlow` `Open Recipe`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2609.03796](https://arxiv.org/abs/2609.03796) · [GitHub](https://github.com/inclusionAI/LLaDA-Image)

---

## 1. 问题定义与控制目标

从零训练强图像生成器、全开源可复现。关键问题：**不用 RLHF 能否达到 SOTA？** 答案是能——real-data-dominant SFT + checkpoint merging + TwinFlow 蒸馏即全部（全文 grep 无 GRPO/DPO/奖励模型；RL 仅蒸馏后 "TBSM tuning" 一句）。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：dLLM VLM（LLaDA 2.0 Mini + SigLIP-VQ），CoT SFT 后全程冻结；编辑时参考图 bypass VLM，直注 DiT 双通道（语义 SigLIP-VQ + 像素 clean VAE latent）；
- **可训练参数**：6B 单流 DiT + RQA + Transformer connector，全 DiT 用无参数 RMSNorm + Muon 优化器；
- **数据**：220M（98% 真实，90%+ image-only 先验），SFT 真实占比 >70%。

## 3. 损失函数与数学稳定性推导

TwinFlow（DMD2 改进）：一个共享 DiT 扮演双角色，$+t$ 做 generator、$-t$ 做 fake-score estimator，无需单独 fake net：
$$
\hat{x} = z - F_\theta(z, +1, h_{\mathrm{cond}}), \quad \tilde{x}_t = (1-t)\,\mathrm{sg}(\hat{x}) + tz'
$$
$$
\mathcal{L}_{\mathrm{fake}} = \mathbb{E}\|F_\theta(\tilde{x}_t, -t) - (z' - \mathrm{sg}(\hat{x}))\|^2, \quad \mathcal{L}_{\mathrm{DMD}} = \mathbb{E}[w(t)\langle \mathrm{sg}(s_{\mathrm{fake}} - s_{\mathrm{real}}), \hat{x}_t\rangle]
$$

::: info 工程三件套
双头输出（推理只留 DMD head，零开销）+ four-step backward simulation + generator:fake = 1:2（DMD2 为 1:5）。Turbo 2–4 步。
:::

理解侧 CoT SFT：block diffusion（b=32），mask ratio $\rho = \cos(r\pi/2)$，batch 双重消费（mask + complement 各一步），Gen:Und:Text = 9:9:2。

## 4. 数据配比权衡

real-data-dominant 收敛早期慢于合成重配方，但终点真实感更强、天花板更高——RL 换的是后训练效率，数据配比可部分替代对齐。Qwen-Image-Bench 英/中 53.53/53.38，开源双轨 SOTA。

## 5. 核心控制层代码实现

```python
# TwinFlow signed-time：同一 DiT，符号即角色
x_hat = z - F_theta(z, +1, h_cond)          # generator: noise -> data
x_tilde = (1 - t) * x_hat.detach() + t * z2  # re-noise
L_fake = ((F_theta(x_tilde, -t, h_cond) - (z2 - x_hat.detach())) ** 2).mean()
# DMD: s_real = frozen F_base(+t), s_fake = F_theta(-t)，只更 generator
```

## 6. 避坑指南与评测基准

- **评测**：Qwen-Image-Bench / LongText / CVTG-2K / GEdit；理解侧 18 集验证 CoT SFT（MMBench 中英 +4.6/+5.0）；
- **启示**：RL 非必选项，但代价是数据与算力前置；TBSM 蒸馏后轻量 tuning（2607.18198）是开放问题；
- 与 SeFi 对照：同走"强 SFT + 蒸馏"，SeFi 另加在线 NFT 精修文字。
