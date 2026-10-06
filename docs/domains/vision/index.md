# 🖼️ 计算机视觉与图像 (CV & Image)

本模块记录图像生成、分割、可控生成、底层视觉表征与基础卷积/注意力网络。

## 核心篇目

| 篇目 | 主题 / 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **单流扩散基模 Z-Image** | S3-DiT 6.15B 单流主干 + 数据四模块闭环，SFT 三件套 → D-DMD/DMDR 蒸馏 8 NFE → DPO+GRPO，全流程 314K H800·h；附「写实感来源」讨论 | 🟢 已完结 | [阅读笔记 →](./z-image.md) |
| **轻量统一多模态 DeepGen 1.0** | 3B VLM + 2B DiT = 5B，SCB 多层桥接 + think tokens，三阶段训练（对齐预训练 → 联合 SFT → MR-GRPO）；RL = 多奖励解耦归一化 + velocity KL + auxiliary SFT loss + 噪声保持随机采样，仅 1,500 steps，**RL 数据无 Edit**（RISE 13.3 → 10.8） | 🔄 精读中 | [阅读笔记 →](./deepgen.md) |
| **原生分辨率高效基模 Mage-Flow** | 微软 4B，Mage-VAE（one-step diffusion tokenizer，>10× 降本）+ Native-Resolution MMDiT + 栈级 CUDA 融合（MFU 13.88%→29.28%）；Data 10B→1.3B / Edit 90M→45M，Diffusion-NFT（Edit:Gen=4:1）→ 4-step D-DMD + adversarial guidance | 📋 目录已建 | [阅读笔记 →](./mage-flow.md) |
| **统一生编基模 Qwen-Image-2.0** | **Data 五小节 + §3.3 + §4.1 + §4.2 已精读**。Qwen3-VL + MMDiT 联合条件-目标建模。**Data**：① Capability-driven data taxonomy（Fig.5 暴露 9 类单图编辑 + 3 类多图编辑）；② Task-specific supervision（四类 Caption 按 task type 路由）；③ Stage-aware curriculum（六阶段 Fig.6 Sankey）；④ Error-attribution Flywheel（Fig.7 三轨路由）。**Prompt Enhancer**：从精细标注**反向随机退化**造真实短 prompt，退化的逆过程即 CoT，训练 (短 prompt, CoT, 精细标注) 三元组；SFT 后用 GRPO 对齐下游成图质量（冻结生成器）。**Training**：700K→250K→10K，Resolution ↑ / Edit 10%→30% / LR 1e-4→2e-5→1e-5。**RLHF**：T2I 三 reward + Editing 两 reward，先 scale calibration 再动态权重；**Hybrid CFG 只省 policy update 的 backward，不省 rollout** | 🔄 精读中（§3.1–3.2 / §4.3 待续） | [阅读笔记 →](./qwen-image-2.md) |
| **图像基模训练 Playbook v1.0** | 从五篇技术报告蒸馏出的**方法论字典**：Data Engine → Pretrain/SFT → Generation/Edit → RL → Evaluation → 实验系统 → 工业流程；每条经验标 **A/B/C 证据等级**，配三张诊断表 | 🟢 已完结 | [查阅手册 →](./training-playbook.md) |
| **语义先行扩散范式 (SFD)** | 复合语义-纹理隐空间 + 固定偏移 $\Delta t$ 异步去噪三阶段，ImageNet FID 1.04 / 收敛快 100× | 🟢 已完结 | [阅读笔记 →](./sfd-semantic-first-diffusion.md) |
| **语义先行文生图基模 (SeFi-Image)** | SFD 语义隐变量先行 + 450M 三原则 caption，5B 约 Z-Image 10–20% 算力 | 🟢 精读中 | [阅读笔记 →](./image-rl-posttraining/sefi-image-rl.md) |
| **Sol-RL：FP4 Explore, BF16 Train** | NVIDIA 的扩散 RL 高效 rollout scaling：NVFP4 探索 96 个 candidate → 只按 reward 排序挑 top-12+bottom-12 的 seed → BF16 重跑 24 张做 GRPO update。**「FP4 不需要生成得准，只需要挑得准」**；$N_{\rm explore}\gg K_{\rm train}$；pipeline 加速 2.4×、收敛加速最高 4.64× | 🔄 精读中（§1–3） | [阅读笔记 →](./image-rl-posttraining/sol-rl.md) |
| **Caption 与效率专题** | DALL-E 3 recaption 范式与 Lens 效率方法论，两篇精读 | 🟢 已完结 | [进入专题 →](./caption-efficiency/) |
| **图像 AIGC 数据集专题** | 预训练/持续训练/SFT/评测集全生命周期大盘，收录 Fine-T2I 6M 数据工程精读 | 🟢 持续建设中 | [进入专题 →](./datasets/) |
