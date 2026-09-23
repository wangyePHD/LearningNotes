# Proposal A：能否用图像衍生连续指令替代文本提示，实现盲复原与盲编辑的单Adapter统一？

> **标签**：`Image Editing` `All-in-One Restoration` `Qwen-Image-Edit` `LoRA`
> **记录时间**：2026-09-23
> **状态**：🧪 [值得单卡跑个玩具 Demo]
> **目标期刊**：IEEE TMM / Pattern Recognition / Expert Systems with Applications（中科院一区）
> **算力预算**：1x 4090/A100，约3~10小时训练 + 1周评测

---

## 1. 灵感触发与背景

过去一个月arXiv高度收敛到一个结论：冻住大编辑模型，只训小模块。

* `2609.25267 ImIR`：冻住Qwen-Image-Edit，只训一个LoRA + 轻量token mapper，把降质图VLM embedding矫正为干净指令，6任务单卡3h，task-agnostic不掉点，低光21.3 vs 16.3dB碾压文本prompt。
* `2609.10723 AcFlow`：冻住DiT，学concept条件速度场做风格强度连续控制。
* `Edit2Restore / RealRestorer`：文本prompt做复原，需要手写prompt、离散、不可调。

矛盾点：文本是粗糙、离散、全局的，无法表达“这张图到底有多脏、要修多强”。而降质图本身就是最精准的指令。

## 2. 核心猜想 Hypothesis

若 $y$ 为降质图，$x$ 为干净目标，$E_{vl}(\cdot) \in \mathbb{R}^{d}$ 为Qwen2.5-VL编码器，则存在轻量映射 $M_{\phi}$ 使得：

$$c = M_{\phi}(E_{vl}(y), \tau) \approx E_{vl}(x)$$

其中 $\tau$ 为可选任务槽（FiLM调制），$c$ 直接作为Qwen-Image-Edit的指令条件（text prompt留空）。缩放 $c(\alpha) = (1-\alpha)E_{vl}(y) + \alpha c$ 可得连续家族解，$\alpha \in [0,1]$ 控制修复强度。

形式化：若A（mapper能闭合降质-干净embedding gap）成立，通过引入B（连续插值+单LoRA共享），则C（6任务PSNR/SSIM超文本基线，且盲任务不崩）在不破坏D（原编辑能力）前提下成立。

::: tip 为什么是一区故事
文本vs图像指令的matched comparison + 盲任务 + 可控族解，三个卖点都是期刊喜欢的“机制解释+实用价值”。
:::

## 3. 方法设计

**双通道输入：**
* 结构通道：$y$ 经VAE进DiT保布局。
* 语义通道：$c = M_{\phi}(E_{vl}(y))$，2层MLP + FiLM，参数<5M。

**训练：**
$$ \mathcal{L} = \mathcal{L}_{FM}(x, y, c) + \lambda \|c - E_{vl}(x)\|_2^2 $$

第一项为flow matching去噪损失，第二项为指令对齐损失。主干冻结，只训LoRA rank=16/64 + mapper。

**盲版本：** 将 $\tau$ 收缩为单一共享槽，全局上下文分支保留退化线索，推理时无需退化标签。

**可控版本：** 推理时 $c(\alpha)$ 插值，低光/去雾等非唯一目标任务可扫 $\alpha$ 出多解。

## 4. 实验计划

数据集：去雨Rain100L、去雾RESIDE、去噪BSD68、去模糊GoPro、低光LOL、JPEG LIVE1 + MagicBrush子集测编辑保持。

基线：Text-LoRA同 backbone、Edit2Restore、专有小模型、ImIR复现。

指标：PSNR/SSIM/LPIPS/DINO-I/CLIP-T + GPT-4o/人评自然度。必须报：task-aware vs task-agnostic落差表（文本版会崩17.5dB，图像版应<0.5dB），$\alpha$ 扫描曲线。

::: info 最小验证集
先跑去雨+低光两任务，单卡<5h，若PSNR超文本1dB以上即证伪通过，全量再铺开。
:::

## 5. 潜在坑点与证伪路径

* VLM embedding本身对噪声敏感，mapper可能学到shortcut：加oracle clean embedding上界消融。
* 缩放插值可能线性外推失效：先在低光上验证单调性，再推广。
* 审稿人会问与ImIR区别：差异化在连续族解+盲编辑统一+多任务权重分析，标题避重。

## 6. 参考

* ImIR 2609.25267， AcFlow 2609.10723， Edit2Restore， RealRestorer， Qwen-Image-Edit
