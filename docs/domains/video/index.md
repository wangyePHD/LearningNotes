# 🎬 视频模型 (Video Intelligence)

本模块聚焦视频生成、时空建模、视频基础模型与多模态视频理解。

## 核心研究路线与笔记

| 篇目 | 主题 / 核心机制 | 关联模型 / 论文 | 状态 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **VideoDeltaNet on H3 (VDN)** | 双轨混合注意力、自适应正规方程与 8 卡 6.9 秒实时视频生成 | MiniMax H3, SGLang | 🟢 已完结 | [阅读精简笔记 →](./video-deltanet-h3.md) |
| **时空注意力机制** | 分解时空注意力 (Spatial-Temporal Factorized Attention) | TimeSformer, Video-DiT | 🟢 已完结 | [阅读笔记 →](./spatio-temporal-attention.md) |
| **Video DiT 架构演进** | 3D VAE 压缩 + 连续流匹配 + 运动一致性约束 | Sora, Wan2.1, CogVideoX | 🟢 已完结 | [阅读笔记 →](./video-diffusion-dit.md) |

::: tip 领域学习建议
从图像到视频的核心跳跃在于：**时间维度的连续性与物理一致性**。重点关注 3D VAE 的时空压缩比（如 $4 \times 8 \times 8$）以及注意力计算复杂度的降维技巧。
:::
