# Proposal B：能否用非对称注意力流保住条件分支，实现先保持后编辑的高保真指令编辑？

> **标签**：`Image Editing` `FLUX Kontext` `Attention` `Preservation`
> **记录时间**：2026-09-23
> **状态**：🧪 [值得单卡跑个玩具 Demo]
> **目标期刊**：IEEE TIP / TCSVT / Neural Networks（中科院一区）
> **算力预算**：1x A100，LoRA微调 + PIE-Bench评测约1~2周

---

## 1. 灵感触发与背景

* `2609.25881 RealFit`：对称joint-attention让噪声污染确定性条件，提CAE/IIF诊断，单向流UIF + 解耦调制DTM省75%时间。
* `2609.02504 SR-Edit`：从自身预测迭代提纯mask，矫正不破坏采样动力学。
* `2609.12691 IABEdit`：冻结VLM残差作梯度教生成器，MagicBrush上+3.49 DINO-I。
* `2609.20633 RefineEdit`：PIE-Bench九类背景保持SOTA。

共识：编辑失败=定位不准+保持分支被污染。现有mask heuristic本身引入伪影。

## 2. 核心猜想 Hypothesis

设噪声token为 $z_n$，条件token为 $z_c$。标准双向注意力同时存在 $N\!\to\!C$ 与 $C\!\to\!N$ 两条路，后者引入随机性使 $p(z_c|t)$ 漂移。

若A（阻断 $C\!\to\!N$ ，只保留 $N\!\to\!N, N\!\to\!C, C\!\to\!C$ ，且条件分支调制固定为 $t^{\star}$ ）成立，通过引入B（冻结VLM语义残差loss + 自提纯mask矫正），则C（PSNR/SSIM/LPIPS/DINO-I双优且推理加速）在不破坏D（指令遵循CLIP-T）前提下成立。

$$ \text{Attn}_{mask} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \quad \text{on} \quad [z_n; z_c] $$

$$ \mathcal{L} = \mathcal{L}_{FM} + \beta \|A_{align}(G(z_n,z_c)) - F_{vl}(x_{gt})\|_2^2 $$

其中 $F_{vl}$ 为冻结VLM空间描述子，$A_{align}$ 为可训轻量aligner，推理时丢弃。

::: tip 卖点
训练时有VLM监督，推理零开销；条件KV-cache一次计算全程复用，RegionCache式加速可直接报。
:::

## 3. 方法设计

1. **UIF**：结构化mask阻断条件看噪声，保高频细节。
2. **DTM**：噪声分支跟随 $t$ ，条件分支固定 $t^{\star}$ 大调制，防信号衰减。
3. **Self-refine**：每 $k$ 步从模型预测轻量后处理得 $M_t$ ，非编辑区做动力学对齐矫正。
4. 只训LoRA + aligner，主干FLUX.1 Kontext-dev / Qwen-Image-Edit冻结。

## 4. 实验计划

基准：PIE-Bench 9类、MagicBrush、EmuEdit、CompBench。用PSNR/SSIM/LPIPS/MSE/DINO-I/CLIP-T/CLIP-I全套，重点报background preservation + edited-region CLIP双优。

基线：P2P、MasaCtrl、PnP、FlowEdit、SR-Edit、IABEdit复现。消融：UIF / DTM / refine / VLM loss四项。加速比：KV-cache前后latency/显存。

::: warning 证伪线
若CLIP-T掉>2%而保真只涨<0.5dB，说明单向流过保守，需放宽 $t^{\star}$ 或只在早步阻断。
:::

## 5. 潜在坑点

* 固定调制可能在极端风格化时欠表达：做 $t^{\star}$ 扫描。
* 自提纯mask阈值敏感：报阈值鲁棒性曲线。
* 与RealFit区分：RealFit做试衣，本提案做通用指令编辑+ VLM梯度，场景不同。

## 6. 参考

* RealFit 2609.25881， SR-Edit 2609.02504， IABEdit 2609.12691， RefineEdit 2609.20633
