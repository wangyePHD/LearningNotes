# 💡 科研 Idea 灵感池 (Idea Sandbox)

> 这里是专属的科研脑洞自留地。记录那些天马行空、甚至有点“奇奇怪怪”的算法直觉与跨模态猜想，不需要一开始就严谨，重点在于捕获那一瞬间的灵感火花。

---

## 脑洞状态速查

- 💡 **[纯脑洞 / 观察]**：一个直觉现象，还没想通底层机理。
- 🔬 **[理论猜想]**：感觉在数学公式或动力学上有戏，准备找时间推导。
- 🧪 **[值得单卡跑个玩具 Demo]**：轻量验证可行性。
- ⏸️ **[暂时搁置]**：受限于当前算力或理论瓶颈，先存个档。

---

## 🌟 脑洞列表

### 1. 能否用纯 RL (GRPO) 约束视频时序平滑度，省去昂贵的 3D 时空注意力？
- **记录时间**：2026-09-19
- **状态**：🧪 [值得单卡跑个玩具 Demo] · **标签**：`#Video Gen` `#GRPO` `#Efficiency`
- **灵感触发**：
  看 DeepSeek-R1 纯强化学习在无 Critic 下通过组内相对优势（GRPO）涌现慢思考能力。目前视频 DiT 为了物理时序一致性，硬扛了昂贵的 $O(T \cdot S^2)$ 3D 时空注意力计算。
- **核心猜想 (Hypothesis)**：
  如果让 2D DiT 生成多帧候选片段，设计一个纯基于光流一致性（Optical Flow Error）或相邻帧余弦差分的无参奖励函数 $R_{\text{smooth}}$，利用 GRPO 进行组内相对强化学习对齐。
- **潜在坑点**：
  RL 优化视觉生成可能会迅速卡入局部最优（比如生成完全静止的死帧欺骗光流奖励）。需要加多样性惩罚项。

---

### 2. Style Transfer 中 3D Latent 空间跨时间步特征互换的物理真实性
- **记录时间**：2026-09-15
- **状态**：🔬 [理论猜想] · **标签**：`#Style Transfer` `#3D VAE` `#Video`
- **灵感触发**：
  在图像风格迁移（如 SigStyle / OmniStyle）中，Attention Swapping 是保留结构的关键。但迁移到视频时，跨帧的风格往往出现闪烁（Flickering）。
- **核心猜想 (Hypothesis)**：
  3D Causal VAE 的潜空间本身已经具备时序因果压缩。如果把风格特征只注入 3D VAE 的第一帧与全局低频通道，让高频运动流场自由传播，能否以零训练代价实现完全无闪烁的视频风格化？

---

### 3. Flow Matching 最优传输路径上的“速度场曲率”是否代表语义突变点？
- **记录时间**：2026-09-10
- **状态**：💡 [纯脑洞 / 观察] · **标签**：`#Flow Matching` `#Math` `#Interpretability`
- **灵感触发**：
  CFM 默认假设数据和噪声之间是直线插值 $x_t = (1-t)x_0 + tx_1$。但在模型实际预测的速度场 $v_\theta(x_t, t)$ 中，中间步长（如 $t=0.4 \sim 0.6$）处梯度的二阶导数值特别大。
- **核心猜想 (Hypothesis)**：
  速度场的加速度 $\|\frac{d v_\theta}{d t}\|$ 峰值出现的时间点，是不是刚好对应语义轮廓从高斯噪声中“结晶涌现”的临界相位？如果在这个时间区间自适应加密欧拉步长（Adaptive Steps），其他区间稀疏步长，能否实现 3 步无损生成？

---

### 4. 能否用图像衍生连续指令替代文本提示，从而省去盲复原的任务标签与强度手调？ [详见 Proposal A →](./proposal-a-blind-continuous-instruction.md)
- **记录时间**：2026-09-23
- **状态**：🧪 [值得单卡跑个玩具 Demo] · **标签**：`#Image Editing` `#All-in-One Restoration` `#Hypothesis`
- **灵感触发**：
  受ImIR 2609.25267单卡3h六任务与AcFlow连续强度控制启发，文本prompt离散粗糙，降质图本身才是最精准指令。
- **核心猜想 (Hypothesis)**：
  若轻量mapper能闭合降质-干净VLM embedding gap，通过引入连续插值$c(\alpha)$与单LoRA共享，则盲复原PSNR可超文本基线且强度可调。
- **潜在坑点与证伪路径**：
  VLM对噪声敏感易学shortcut；先跑去雨+低光两任务超1dB再铺开，详见Proposal A第5节。

---

### 5. 能否阻断条件分支看噪声的反向流，从而在不损指令遵循下保住高频细节？ [详见 Proposal B →](./proposal-b-asymmetric-preservation-editing.md)
- **记录时间**：2026-09-23
- **状态**：🧪 [值得单卡跑个玩具 Demo] · **标签**：`#Image Editing` `#Attention` `#Hypothesis`
- **灵感触发**：
  受RealFit非对称流、SR-Edit自提纯、IABEdit VLM残差梯度启发，对称joint-attention是保真崩坏主因。
- **核心猜想 (Hypothesis)**：
  若阻断$C\to N$并固定条件调制$t^{\star}$，通过引入VLM残差loss+动力学对齐矫正，则PIE-Bench背景与CLIP双优。
- **潜在坑点与证伪路径**：
  过保守致CLIP-T掉>2%则只在早步阻断；阈值敏感需扫$t^{\star}$曲线，详见Proposal B第5节。

---

### 6. 小物体试穿能否用scale token一次解决大小错位，再用维度奖励治住刷分？ [详见 Proposal C →](./proposal-c-scale-aware-tryon.md)
- **记录时间**：2026-09-23
- **状态**：🧪 [值得单卡跑个玩具 Demo] · **标签**：`#Virtual Try-On` `#Fashion` `#Hypothesis`
- **灵感触发**：
  受JewelTry scale adapter、DAT 7维评价、TryOnReward foveated RFT启发，FID测不出尺度错，通用reward走捷径。
- **核心猜想 (Hypothesis)**：
  若真实尺寸编码为scale token做in-context学习，通过引入7维自适应加权RFT，则保真+尺度+背景三优且保持mask-free。
- **潜在坑点与证伪路径**：
  尺寸标注贵先用合成warm-start；小物体<2%像素需foveated加权，先跑戒指500对降20%误差再扩，详见Proposal C。

---

### 7. 多轮AI编辑的ripple能否用检测-诊断-修复闭环统一治愈而不误伤干净图？ [详见 Proposal D →](./proposal-d-diagnose-repair-forensics.md)
- **记录时间**：2026-09-23
- **状态**：🔬 [理论猜想] · **标签**：`#Forensics` `#AIGC Detection` `#Hypothesis`
- **灵感触发**：
  受FUSED检测定位统一、GLARE训练-free重构差、Mi-Ripple频域修复、GenShield VCoT启发，检测不修、修复不解释。
- **核心猜想 (Hypothesis)**：
  若global-local重构差+ VLM缺陷描述成立，通过引入指令预热+VCoT自纠+STOP学习，则检测与修复双SOTA且干净图零改动。
- **潜在坑点与证伪路径**：
  合成gap大需半真半合成；误检>15%回退监督头；过编辑靠STOP阈值约束，详见Proposal D。

---

## ✍️ 新增 Idea 极简模板 (复制即用)

```markdown
### N. [你的脑洞标题]
- **记录时间**：YYYY-MM-DD
- **状态**：💡 [纯脑洞] / 🔬 [理论猜想] / 🧪 [写个Demo] / ⏸️ [暂时搁置]
- **标签**：`#领域` `#关键词`
- **灵感触发**：看到什么、想到什么、被什么启发？
- **核心猜想 (Hypothesis)**：如果这样做，理论上会发生什么？
- **潜在坑点**：最担心崩在哪里？
```
