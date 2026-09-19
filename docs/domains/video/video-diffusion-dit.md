# Video DiT 架构演进与时空物理一致性

> **标签**：`Video Gen` `DiT` `Diffusion` `3D VAE`  
> **更新时间**：2026-09-19

---

## 1. 经典架构范式演进

从最早期的 UNet 级联插帧，到当今主流的 DiT (Diffusion Transformer) 架构，视频生成经历了三个核心阶段：

1. **2D UNet + Temporal Convolution/Attention 插帧**（如 AnimateDiff, SVD）
   - 痛点：时间感知受限于预训练 2D 先验，长视频容易发生剧烈形变与抖动。
2. **纯 3D VAE + 3D DiT (全注意力架构)**（如 Sora, Wan2.1, HunyuanVideo）
   - 将视频在潜空间直接看作 3D 体素 Token（$T_{\text{latent}} \times H_{\text{latent}} \times W_{\text{latent}}$），直接建模物理动力学。

---

## 2. 3D 因果时空自编码器 (3D Causal VAE)

视频压缩的关键是不能让未来的帧泄露给过去（保持因果性），否则在自回归或流式生成时会出现时序冲突：

$$
z = \text{Encoder}_{\text{3D}}(x), \quad x \in \mathbb{R}^{B \times C \times T \times H \times W}
$$
通常的时间压缩比（Temporal Compression Ratio）为 $4 \times$，空间压缩比为 $8 \times$。即原本 16 帧、分辨率 $720 \times 1280$ 的视频，在 Latent 空间中仅需表示为：
$$
4 \times 90 \times 160
$$
大大缓解了 DiT 主干网络的序列长度压力。
