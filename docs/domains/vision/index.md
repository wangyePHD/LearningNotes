# 🖼️ 计算机视觉与图像 (CV & Image)

本模块记录图像生成、分割、可控生成、底层视觉表征与基础卷积/注意力网络。

## 核心篇目

| 篇目 | 主题 / 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **单流扩散基模 Z-Image** | S3-DiT 6.15B 单流主干 + 数据四模块闭环，SFT 三件套 → D-DMD/DMDR 蒸馏 8 NFE → DPO+GRPO，全流程 314K H800·h；附「写实感来源」讨论 | 🟢 已完结 | [阅读笔记 →](./z-image.md) |
| **轻量统一多模态 DeepGen 1.0** | 3B VLM + 2B DiT = 5B，SCB 多层桥接 + think tokens，三阶段训练（对齐预训练 → 联合 SFT → MR-GRPO）；RL = 多奖励解耦归一化 + velocity KL + auxiliary SFT loss + 噪声保持随机采样，仅 1,500 steps，**RL 数据无 Edit**（RISE 13.3 → 10.8） | 🔄 精读中 | [阅读笔记 →](./deepgen.md) |
| **原生分辨率高效基模 Mage-Flow** | 微软 4B，Mage-VAE（one-step diffusion tokenizer，>10× 降本）+ Native-Resolution MMDiT + 栈级 CUDA 融合（MFU 13.88%→29.28%）；Data 10B→1.3B / Edit 90M→45M，Diffusion-NFT（Edit:Gen=4:1）→ 4-step D-DMD + adversarial guidance | 📋 目录已建 | [阅读笔记 →](./mage-flow.md) |
| **语义先行扩散范式 (SFD)** | 复合语义-纹理隐空间 + 固定偏移 $\Delta t$ 异步去噪三阶段，ImageNet FID 1.04 / 收敛快 100× | 🟢 已完结 | [阅读笔记 →](./sfd-semantic-first-diffusion.md) |
| **语义先行文生图基模 (SeFi-Image)** | SFD 语义隐变量先行 + 450M 三原则 caption，5B 约 Z-Image 10–20% 算力 | 🟢 精读中 | [阅读笔记 →](./image-rl-posttraining/sefi-image-rl.md) |
| **Caption 与效率专题** | DALL-E 3 recaption 范式与 Lens 效率方法论，两篇精读 | 🟢 已完结 | [进入专题 →](./caption-efficiency/) |
| **图像 AIGC 数据集专题** | 预训练/持续训练/SFT/评测集全生命周期大盘，收录 Fine-T2I 6M 数据工程精读 | 🟢 持续建设中 | [进入专题 →](./datasets/) |
