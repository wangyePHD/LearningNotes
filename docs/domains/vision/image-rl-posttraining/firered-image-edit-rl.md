# FireRed-Image-Edit RL 落地全链路

> **标签**：`Vision` `RL` `DPO` `DiffusionNFT` `Image Editing`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2602.13344](https://arxiv.org/abs/2602.13344) · [GitHub](https://github.com/FireRedTeam/FireRed-Image-Edit)

---

## 1. 问题定义与控制目标

SFT 后的编辑模型指令遵循不稳、美学不足、文本渲染崩坏。RL 要解决：标准 Diffusion-DPO 的 double degradation（正负样本都被推离流形）、成对数据稀疏、人像身份漂移。管线：`Pretrain 300K → CT 65K → SFT 5K → DPO 5K → NFT 500`。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：old-policy（NFT 的 $v_{\mathrm{old}}$ 锚）、VLM 评测模型（Qwen3-VL-8B SFT 版，只打分）；
- **可训练参数**：MMDiT 主干（Qwen 系初始化）；
- **注入机理**：DPO 阶段偏好对梯度；NFT 阶段连续标量 $r$ 加权去噪；Consistency Loss 经共享人脸 backbone 注入语义级身份梯度。

## 3. 损失函数与数学稳定性推导

非对称 DPO + PSR（$\omega{>}1$ 放大 Win 项，$\lambda$ 锚 chosen 重建）：
$$
\mathcal{L} = -\mathbb{E}\left[\log\sigma\left(\beta\,[\mathrm{LoseDiff} - \omega\,\mathrm{WinDiff}]\right) - \lambda \mathcal{L}_w^\theta\right]
$$

DiffusionNFT（单样本连续 $r$，无 value、无 advantage 归一化）：
$$
\mathcal{L}_{\mathrm{NFT}} = \mathbb{E}\left[r\|v_\theta^+ - v\|^2 + (1-r)\|v_\theta^- - v\|^2\right]
$$
$$
v_\theta^+ = (1-\beta)v_{\mathrm{old}} + \beta v_\theta, \quad v_\theta^- = (1+\beta)v_{\mathrm{old}} - \beta v_\theta
$$

Logit 加权 ensemble reward（治稀疏整数分）：
$$
R(x,y) = \frac{1}{K}\sum_k \sum_{v} v \cdot P(v|x,y,c_k), \quad P = \mathrm{softmax}(z)
$$

Layout-Aware OCR 奖励（治"大字刷分"）：
$$
R = w_{\mathrm{text}}\left(1 - \frac{d}{\max|s_{\mathrm{tgt}}|}\right) + w_{\mathrm{layout}}\,\mathrm{Gate}\cdot\left(\frac{|s_{\mathrm{pred}}|}{|s_{\mathrm{tgt}}|}\sum_i e^{-d_i}e^{-\Delta s_i}\right)
$$

一致性损失（coarse-to-fine：高噪做语义锚，低噪只修像素）：
$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{mse}} + \lambda_{\mathrm{id}}(\sigma)\mathcal{L}_{\mathrm{id}}, \quad \lambda_{\mathrm{id}} = \eta\sigma^2\;(\sigma < 0.9)
$$
$\hat{x}_0 = x_t - \sigma_t v_t$ 经可微 ROI align 进共享人脸 backbone 算余弦距离，多人平均。

::: info 数据侧
Mix-Policy 偏好对：正样本来自外部专家分支（破自采样天花板），负来自 base+VLM 过滤，~60K 对零人工；NFT 用 DPO checkpoint 离线挖 semi-hard（均值尚可、方差大）再在线精调。
:::

## 4. 训练配比权衡

| 阶段 | 数据 | Instructive 占比 | 分辨率 |
| :--- | :--- | :--- | :--- |
| Pretrain | 100M + 5M 合成 | 5% | 384–512 |
| CT | 5M + 10M 合成 | 20% | 512–1024 |
| SFT | 50K + 50K | 45% | 1024 |
| DPO | 10K + 50K | 60% | 1024 |
| NFT | 20K 在线 | 60% | 1024 |

全阶段 AdamW + clip 1 + 分层 timestep 采样 + logit-normal 加权 + EMA。效果：ImgEdit / GEdit / REDEdit 开源 SOTA。

## 5. 核心控制层代码实现

```python
# 非对称 DPO：Win/Lose diff 非等权 + chosen 锚
def asymmetric_dpo_loss(Lw_t, Lw_r, Ll_t, Ll_r, beta, omega=2.0, lam=0.1):
    win_diff = Lw_t - Lw_r
    lose_diff = Ll_t - Ll_r
    return -torch.log(torch.sigmoid(beta * (lose_diff - omega * win_diff))) + lam * Lw_t

# NFT：连续标量加权正负分支
def nft_loss(v_theta, v_old, v_target, r, beta=0.5):
    v_pos = (1 - beta) * v_old + beta * v_theta
    v_neg = (1 + beta) * v_old - beta * v_theta
    return r * (v_pos - v_target).pow(2).mean() + (1 - r) * (v_neg - v_target).pow(2).mean()

# 一致性损失：单步估计 + 噪声条件权重
def consistency_loss(x_t, v_t, sigma_t, x_gt, face_backbone, align_T, eta=1.0):
    x_hat0 = x_t - sigma_t * v_t
    lam_id = eta * sigma_t**2 if sigma_t < 0.9 else 0.0
    l_id = 1 - F.cosine_similarity(face_backbone(align_T(x_hat0)),
                                   face_backbone(align_T(x_gt))).mean()
    return lam_id * l_id  # 另加 MSE 项
```

## 6. 避坑指南与评测基准

- **评测**：REDEdit-Bench（15 类）+ ImgEdit + GEdit + 人评（对齐/一致/真实/美学）；
- **三铁律**：DPO 必加 chosen 锚；OCR 类奖励必加位置/尺度/门控第二项；在线只做短程（500 步）+ 离线 semi-hard 挖掘；
- **稳定三件套**：分层 timestep 采样、logit-normal 加权、EMA，RL 阶段缺一不可。
