# 语义先行文生图基模 (SeFi-Image)

> **标签**：`Vision` `Diffusion` `SeFi` `SFD` `Text-to-Image`
> **更新时间**：2026-09-23
> **参考来源**：[SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion](https://arxiv.org/abs/2606.22568) · [GitHub](https://github.com/jmliu206/SeFi-Image)

---

## 1. 问题定义与控制目标 (Motivation)

::: info 核心矛盾：效果越好，烧卡越多
以 Z-Image 为例，即使主打省资源仍需 314K H800 GPU hours。SeFi-Image 5B 仅用 125K A800 GPU hours，约其 10–20% 算力，对标 Qwen-Image / Z-Image。
:::

- **重建-生成权衡**：latent 信息保留越多，扩散建模分布越复杂、收敛越慢；压缩越狠越好学，但经 VFM 前向 Markov 抽象后信息丢失，重建回像素上限低，小字渲染与细粒度编辑一致性先崩。
- **前人只做到一半**：RAE / VA-VAE / REPA / REGLUE / ReDi / REG / SFD 在 ImageNet 256、<1B、类别条件上加速明显，但能否放大到高分辨率 T2I foundation、除了收敛更快还有没有最终质量收益，仍是 open question。
- **SeFi 的目标**：把 Semantic-First Diffusion 做到 1B / 2B / 5B foundation 规模，用语义隐变量先行给纹理隐变量提供干净结构锚，在高保真 VAE 下依然快速收敛，同时覆盖中英双语长 prompt 与字符级文本渲染。

## 2. 架构拓扑与特征注入机理

### 2.1 VFM：只看懂、不动手的提纲手

VFM = Vision Foundation Model，本篇固定为冻结的 `DINOv2-Large`，全程不训练。

::: info 直观分工
VFM 定方向，SemVAE 减肥，Texture VAE 保细节，DiT 照着提纲画画。
:::

```
输入图像 x: R^{3 x 1024 x 1024}
  │
  ├─→ [语义路] Phi(x) = f_s in R^{L x C_in}，如 1369 x 1024
  │     每个 token 讲物体身份与布局，不记噪点颜色
  │     ↓ SemVAE Encoder 压成 s_1 in R^{L x C_s}，如 1369 x 16
  │
  └─→ [纹理路] E_z(x) = z_1，如 R^{32 x 128 x 128}
        记高频细节，可几乎无损解回像素
```

- **冻结参数**：`Phi` = DINOv2-Large、SemVAE 训好后冻结、Texture VAE 微调后冻结；冻结保证语义锚稳定，异步去噪才有意义。
- **可训练参数**：双流 DiT 主干（1B / 2B / 5B）+ 双 timestep embedding + 输入输出投影；文本侧用 Qwen3-VL LLM hidden states 拼接。
- **为什么还要 SemVAE**：DINO 特征太肥（1369 x 1024）直接扩散太贵。SemVAE 做 `f_s → mu, sigma → 采样 s_1 → 解回 f_hat_s`，目标为 MSE + 余弦 + KL，出图时语义隐变量直接扔掉，只解码纹理隐变量 `z_1`，语义只在中间当拐杖。

## 3. 损失函数与数学稳定性推导

TODO：公式 6-8，velocity + REPA。

## 4. 保真度与风格化权衡 (Trade-off Analysis)

TODO：重建-生成曲线、Δt、三阶段调度。

## 5. 核心控制层代码实现

TODO：双流调度 20~30 行。

## 6. 避坑指南与评测基准

TODO：GenEval / DPG / LongTextBench / OneIG / CVTG-2K，CLIPScore，artifact。
