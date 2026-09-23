# SeFi-Image 在线 RL：双潜适配与能力标签 (Semantic-First Diffusion)

> **标签**：`Vision` `RL` `DiffusionNFT` `SeFi` `Text Rendering`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2606.22568](https://arxiv.org/abs/2606.22568) · [GitHub](https://github.com/jmliu206/SeFi-Image)

---

## 1. 问题定义与控制目标

5B 模型、125K A800-h（约 Z-Image 10–20% 算力）达到 Qwen/Z-Image 级别，RL 负责最后 sharpen：prompt following、画质、artifact 压制、文字渲染。纯**在线** RL（DiffusionNFT），基座为 5B（1B/2B 未做 RL）。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：DINOv2-Large 语义分支、双 VAE、old-policy checkpoint（本 batch 生成者）；
- **可训练参数**：SFD 双流 DiT（语义潜先行 $\Delta t$、纹理随后）；
- **注入机理**：与标准 NFT 唯一区别——target 为双流 concat $z_{\mathrm{comp}} = [z_{\mathrm{sem}}, z_{\mathrm{tex}}]$（生成样本经双 VAE 重编码），SFD 异步调度保持不变；RL 只重塑 reward→loss 映射，不改变生成动力学。

## 3. 损失函数与数学稳定性推导

在线循环：$\pi_i \to$ 生成（400 组 × 12 张 = 4800 图）$\to$ 全量打分 $\to$ 过滤 $\to$ 训练 $\to \pi_{i+1}$。

附录 C 四步：组内标准化 $A_i = (r_i - \mathrm{mean})/\sigma$ → 映射 $\rho_i = \mathrm{clip}(\mathrm{clip}(A_i)/2A_{\max} + 1/2)$ → 构造 $v^+ = \beta v_\theta + (1-\beta)v_{\mathrm{old}}$、$v^- = (1+\beta)v_{\mathrm{old}} - \beta v_\theta$ → $\mathcal{L} = \rho\mathcal{L}^+ + (1-\rho)\mathcal{L}^-$。

::: info 两个便宜 trick
低离散度 prompt 组直接丢（无偏好信号）；prompt 带能力标签（spatial / text-rendering / artifact-control），按标签在相关维度打分，防"好看但语义错"被强化。
:::

## 4. RL 能力边界（5B ablation）

| 基准 | w/o → w/ RL | 结论 |
| :--- | :--- | :--- |
| LongTextBench | 0.9665 → 0.9780 | 主收益 +1.15pt |
| CVTG-2K WordAcc | 0.8783 → 0.8947 | +1.64pt |
| GenEval | 0.87 → 0.88 | Position +3pt |
| DPG | 87.45 → 87.27 | 基本持平 |
| OneIG EN/ZH | +0.0065/+0.0044 | Text/Reasoning/Diversity 涨，Align 微降 |

**RL 是精修器不是发动机**：文字/遵循涨，构图不动。

::: warning 避坑要点
$\beta$、$A_{\max}$、学习率、总轮数**均未披露**——复现 checklist：先扫 $\beta \in [0.3, 0.7]$、组大小 M=12 起步；RL 只在 5B 验证，小模型性价比需单独验证。
:::

## 5. 核心控制层代码实现

```python
# SeFi 在线 NFT：双潜 target + 离散度过滤 + 能力标签
def sefi_rl_iteration(policy, prompt_groups, M=12):
    batch = []
    for g in prompt_groups:                       # 每组带 capability tags
        imgs = [policy.generate(g.prompt) for _ in range(M)]
        rewards = reward_model.score(imgs, dims=g.tags)  # 按标签维度打分
        if rewards.std() < DISPERSION_THRESH:
            continue                              # 低离散度组丢弃
        batch += select_top_bottom(imgs, rewards) # top 正梯度/bottom 隐式负
    z_comp = concat_vae_encode(batch)             # 双 VAE 重编码
    return nft_loss_on_dual_latent(policy, z_comp, batch.rewards)
```

## 6. 避坑指南与评测基准

- **评测**：LongText / CVTG-2K / GenEval / DPG / OneIG，RL 前后必须同表并列（SeFi Tables 11–15 范本）；
- **SFD 根因**：语义潜先行提供干净结构锚，纹理生成负担小——这才是小算力 + RL 有效的前提；
- 与 GDRO 对照：同组级奖励路线，GDRO 离线、SeFi 在线，按算力/探索需求二选一。
