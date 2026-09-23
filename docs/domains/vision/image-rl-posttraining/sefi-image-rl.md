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

## 2. 数据：三段式课程

> 预训练吃饱 + 会写字 → 持续训练学听话 → SFT 学好看。预训练总量约 478M（450M 真实 + 28M 合成，合成约占 6%）。

### 2.1 预训练数据（450M 真实 + 28M 合成字图）

**450M 内部图文**：全部用 Qwen3.5-2B 重打 caption，三原则为准、客观、克制详细（防幻觉）。双语（中/英）× 长短（dense/short）四个版本，训练时 dense:short = 4:1——多看详细的学信号快，少看短的保推理不偏。

**28M 合成字图**：课程学习思想，预训练只要求写对、放对，不管语义搭不搭。

- *Part1 8M 纯文本块*：PIL 在 512×512 素底上确定性渲染，文案从 450M 语料采样，中密/中短/英密/英短各 2M。prompt 固定模板英文 `The text in this image is "..."` / 中文 `这张图片中的文字是“...”`，字符级对齐。
- *Part2 20M 结构排版*：多 block、多角色版式，长宽比 1:1 / 4:3 / 16:9 / 3:4 / 9:16（约 1024² 像素），随机槽位、颜色、字号、字体，中 8M、英 8M、中英混合 4M。prompt 描述文字内容 + 位置 + 颜色 + 相对大小。
- *质检*：字符级图文对齐校验、溢出与 bbox 检查。

::: tip 为什么这么设计
与 Z/Qwen 搞复杂场景合成不同，SeFi 认为写字本质是严格一对一映射，预训练先把映射学牢，逼真排版留给 CT/SFT 迁移。这就是 CVTG-2K 能反超的根因。
:::

### 2.2 持续训练数据（9M）

TODO：Fine-T2I + 内部多域（风景/UI/平面/二次元），更高质量更难 caption，主提 instruction-following。

### 2.3 SFT 数据（约 65 万）

TODO：开源 + 20 万中文富文本 + 内部高审美；自家 VLM 抽元数据 + 精修 caption；硬门槛打分过滤（审美/技术/构图/主体/captionability/训练价值），去水印糊图伪影敏感内容后去重；中英长短 caption + tag 混训。

## 3. 架构拓扑与特征注入机理

### 3.1 VFM：只看懂、不动手的提纲手

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

## 4. 损失函数与数学稳定性推导

TODO：公式 6-8，velocity + REPA。

## 5. 保真度与风格化权衡 (Trade-off Analysis)

TODO：重建-生成曲线、Δt、三阶段调度。

## 6. 核心控制层代码实现

TODO：双流调度 20~30 行。

## 7. 避坑指南与评测基准

TODO：GenEval / DPG / LongTextBench / OneIG / CVTG-2K，CLIPScore，artifact。
