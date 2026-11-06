---
type: paper
tags: ['Distillation', 'Diffusion', 'Flow Matching', 'T2I', 'Video Gen', 'Controllability']
status: reading
summary: '作者提出 PDM：分开对齐正分支和 CFG 方向。'
models: ['SD3.5-M', 'FLUX.2-klein-4B', 'Wan2.1-VACE-1.3B']
---

# 论文笔记：Rethinking CFG in On-Policy Diffusion Distillation

2026-11-06

## 一句话

蒸馏时只对齐 CFG 合成后的速度，正负两个分支的误差可以互相抵消。如果老师的负分支有学生拿不到的信息，学生会把这部分信息藏进「正 − 负」的差值里，靠 γ 放大来补。训练用的 γ 下看着对，换个 γ 就坏。作者提出 PDM：分开对齐正分支和 CFG 方向。

- 论文：Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation，arXiv 2607.24731，2026-07
- 作者：Bingnan Li 等，Qwen Business Unit of Alibaba
- 论文 PDF（本地）：`/gpfs/wy/Projects/SeFi-Image-5B/DPO/refs/Rethinking-CFG-OPD_paper.pdf`
- 代码（本地）：`/gpfs/wy/Projects/SeFi-Image-5B/DPO/refs/Rethinking-CFG-OPD-main.zip`
- 关键词：OPD、CFG、负分支不对称（NBA）、PDM

![总览图：左 Naive 的问题，右 PDM 的做法和对 γ 的鲁棒性](/rethinking-cfg-opd-teaser.jpg)

*总览图。左半：Naive 只约束合成后的速度，e₊ 和 e₋ 可以互相抵消，学生在 (e₊, e₋) 平面上落在一条直线上的任意点都满足 loss = 0，这就是 NBA。右半：PDM 分开约束正分支预测和 CFG 方向 d，右下角是 FID / FVD 随测试 γ 的变化，Naive（红）在 γ = 1 时飙到 78.20 / 507.70，PDM（绿）基本不动。*

## 背景

**OPD（on-policy 蒸馏）**：学生自己从噪声往下采样，在它自己走到的每个状态 x_t 上让老师给出速度，学生去逼近。

**CFG**：每一步算两个分支，正分支 v⁺（带条件）和负分支 v⁻（空条件），再合成（论文 Eq. 3）：

$$\tilde{v} = \gamma v^{+} + (1-\gamma) v^{-} = v^{-} + \gamma\,(v^{+} - v^{-})$$

**Naive 做法**：只对齐合成后的速度。记两个分支的误差为（论文 Eq. 4）：

$$e_{+} = v^{+}_{T} - v^{+}_{S}, \qquad e_{-} = v^{-}_{T} - v^{-}_{S}$$

则 Naive loss 为（论文 Eq. 5，省略时间权重 w(t)）：

$$\mathcal{L}_{\text{naive}} = \left\| \gamma\, e_{+} + (1-\gamma)\, e_{-} \right\|_2^2$$

## 问题为什么会出现

Naive loss 等于 0 只要求（论文 Eq. 6、7）：

$$\gamma\, e_{+} + (1-\gamma)\, e_{-} = 0 \quad\Longleftrightarrow\quad e_{+} = \frac{\gamma-1}{\gamma}\, e_{-}$$

满足这个条件的非零 (e₊, e₋) 有无穷多组。论文把这叫「分支层面不可辨识」。

数字例子（γ = 5，速度取一维，合成 = 5v⁺ − 4v⁻）：

- 老师：v⁺ = 2.0，v⁻ = 1.0，合成 6.0，γ = 1 时输出 v⁺ = 2.0
- 学生 A（两个分支都对）：v⁺ = 2.0，v⁻ = 1.0，合成 6.0，γ = 1 时输出 2.0
- 学生 B（两个分支都错）：v⁺ = 1.6，v⁻ = 0.5，合成 6.0，γ = 1 时输出 1.6（错）

学生 B 的 Naive loss 也是 0，可推理时一换 γ 就错。

**学生为什么会走到 B 那样的解**：看梯度。合成公式是 γv⁺ − (γ−1)v⁻，当学生合成结果偏小时：

- v⁺ 被往上推，系数是 γ
- v⁻ 被往下推，系数是 γ−1（γ=5 时为 4）

两个分支被推向相反的方向。如果学生缺的信息正分支学不来，它就会把 v⁻ 推偏，把缺口写进差值 d = v⁺ − v⁻，靠 γ 放大来补上。结果是 e₊ 下降、e₋ 上升，论文称为 **NBA（Negative Branch Asymmetry，负分支不对称）**。（这一段梯度推理是我自己的理解，论文给的是 Eq. 6、7 和训练曲线。）

这也解释了为什么问题在训练 γ 下看不出来：信息只在「γ × 差值」这个组合里是对的。γ 变了放大量就不对；γ = 1 时差值完全不参与计算，信息全部丢失。

## 什么时候会出现

判断标准只有一个：**老师负分支用到的条件，学生拿不拿得到**。拿得到（共享）时 e₊ 和 e₋ 会一起下降，Naive 就够用；拿不到（特权）时就会出现 NBA。三个实验的条件设置，论文原文如下：

文字渲染（共享）和参考图风格（特权），论文 Eq. 13、14：

![论文 Eq. 13、14：文字渲染和参考图风格的正负分支条件](/rethinking-cfg-opd-eq13_14_setup_image.png)

视频 dense→sparse（特权），论文 Eq. 15：

![论文 Eq. 15：视频实验的正负分支条件](/rethinking-cfg-opd-eq15_setup_video.png)

视频这一组最好理解：老师的两个分支都知道稠密姿态，学生的两个分支都只有稀疏关键帧。缺的姿态信息只能靠把 v⁻ 推偏、用差值放大来补，于是 e₋ 越训越大。

论文 Figure 2 的训练曲线对应这两种情况：文字渲染下两条误差曲线一起下降；参考图风格下，Naive 和只对齐正分支的方法，都是 ‖e₊‖ 下降、‖e₋‖ 上升。

![训练过程中两个分支误差范数的变化](/rethinking-cfg-opd-loss_curve.jpg)

*两个分支的误差范数随训练步数的变化（蓝 ‖e₊‖，橙 ‖e₋‖）。上排文字渲染（共享负条件）：三种方法两条线都一起降。下排参考图风格（特权负分支）：naive 和 positive only 的橙线持续上升，这就是 NBA；PDM 的橙线被压住，不再增长。*

![参考图风格生成，不同 γ 下的对比](/rethinking-cfg-opd-NBA_style.jpg)

*NBA 在推理时的代价：参考图风格任务里，naive 学生在 γ = 1 时最明显地偏离参考风格，PDM 在各个 γ 下都贴近老师。*

## 方法：PDM（Positive-Direction Matching）

PDM 在合成之前就做监督，分两项：一项对齐正分支，一项对齐 CFG 方向 d = v⁺ − v⁻（论文 Eq. 9、10）：

$$\mathcal{L}_{\text{PDM}} = \left\| e_{+} \right\|_2^2 + \lambda \left\| d_T - d_S \right\|_2^2 = \left\| e_{+} \right\|_2^2 + \lambda \left\| e_{+} - e_{-} \right\|_2^2$$

- 第一项强制 v⁺ 本身是对的，信息只能放进正分支。
- 第二项保证在不同 γ 下的行为跟老师一致。
- loss 为 0 时只能 e₊ = e₋ = 0，不存在互相抵消的解。
- 取值：视频 λ = 1，文字渲染 λ = 2。

对照方法 **IBM**：两个分支各自对齐，并去掉 Naive 展开式里的交叉项（论文 Eq. 11）：

$$\mathcal{L}_{\text{IBM}} = \gamma^2 \left\| e_{+} \right\|_2^2 + (\gamma-1)^2 \left\| e_{-} \right\|_2^2$$

伪代码：

```python
v_T_pos, v_T_neg = teacher(x_t, t, c_T_pos), teacher(x_t, t, c_T_neg)
v_S_pos, v_S_neg = student(x_t, t, c_S_pos), student(x_t, t, c_S_neg)
e_pos, e_neg = v_T_pos - v_S_pos, v_T_neg - v_S_neg

loss_naive = (g*e_pos + (1-g)*e_neg).pow(2).mean()
loss_ibm   = g**2*e_pos.pow(2).mean() + (g-1)**2*e_neg.pow(2).mean()
loss_pdm   = e_pos.pow(2).mean() + lam*(e_pos - e_neg).pow(2).mean()
```

## 实验结果

在训练用的 γ 下，PDM 只比 Naive 略好；真正的差距出现在换 γ 之后。以下表格均为论文原表截图。

**Table 1：文字渲染（共享负条件）。** Naive 和 PDM 基本没有差别，γ = 1 时的下降来自老师本身对 CFG 的依赖。

![论文 Table 1](/rethinking-cfg-opd-table1_ocr_gamma.png)

**Table 2：视频 dense→sparse 主结果。** 600 个测试片段，联合训练 pose / depth / scribble，γ = 5。

![论文 Table 2](/rethinking-cfg-opd-table2_main_results.png)

*Pose 控制定性结果（γ = 5）：*

![Pose 控制的定性结果](/rethinking-cfg-opd-qual_res_pose.jpg)

*Depth 控制定性结果（γ = 5）：*

![Depth 控制的定性结果](/rethinking-cfg-opd-qual_res_depth.jpg)

*Scribble 控制定性结果（γ = 5）：*

![Scribble 控制的定性结果](/rethinking-cfg-opd-qual_res_scribble.jpg)

**Table 3：换 γ 测试（pose-only，训练 γ = 5）。** 这张表是论文的核心证据：Naive 在 γ = 1 时 MPJPE 从 4.62 涨到 8.98、FVD 从 67.44 涨到 507.70，PDM 基本不动。

![论文 Table 3](/rethinking-cfg-opd-table3_gamma_generalization.png)

![两段视频，同一模型在三个 γ 下的关键帧](/rethinking-cfg-opd-NBA_pose.jpg)

*Naive 与 PDM 在 CFG scale = 1 / 3 / 5 下的对比（关键帧 KF0 / 20 / 40 / 60，两段视频）。Naive 在 scale = 1 时画面糊成一团，PDM 三个 scale 下都稳定。*

**Table 4：方向项权重 λ 的消融。** λ = 0 时 PDM 退化为只对齐正分支；作者选 λ = 1。

![论文 Table 4](/rethinking-cfg-opd-table4_lambda.png)

**Table 5：监督步数 K 的消融。** 作者选 K = 8。

![论文 Table 5](/rethinking-cfg-opd-table5_K.png)

## 官方代码要点

三个子目录对应三个实验，都用 `loss_mode` 在 `naive / ibm / pdm` 之间切换。我只读了代码，没有跑。

- `DOPSD-flux/`：FLUX 参考图风格。loss 在 `train_dopsd.py:882-893`，与论文公式一致，最好读。
- `Dense2Sparse/`：Wan-VACE 视频（主实验）。loss 在 `train_opd.py:1374-1466`，方向项里的 v⁺ 被 detach，见下。
- `DiffusionOPD/`：SD3.5 文字渲染。loss 在 `scripts/train_sd3_opd.py:1515-1535`，与论文公式一致，但对齐的是 SDE 步均值。

**视频版的 detach**：正负分支分两次前向、分两次 backward（省显存），方向项写成 `g_s = vs_c_val - vs_u`，其中 `vs_c_val` 已 detach。所以 v⁺ 只受 ‖e₊‖² 的梯度，v⁻ 只受方向项的梯度。论文公式里方向项也会把梯度传给 v⁺。论文 Table 2、3 的数字是否就是用这个版本跑出来的，我没法从代码确认。

**视频版的 rollout**：从纯噪声出发，调度器设 50 步（`K_schedule=50`），只训练前 8 步（`K_rollout=8`）。下一步的 latent 用学生自己合成的速度推进，γ = 5。

**SD3 版的尺度**：对齐 Flow-GRPO 的 `prev_sample_mean`，权重 1/(2σ²)。代码注释提醒：γ = 4.5 时 IBM 的量级约是 Naive 的 32 倍，会和梯度裁剪相互影响，三种 loss 不在同一尺度上。

## 对我们的启示

开始一个 CFG 蒸馏任务前，先问一句：老师负分支有没有学生拿不到的条件（参考图、稠密控制、更长的 prompt 等）？

- **没有**：属于共享情况，Naive 够用。文字渲染实验里 PDM 和 Naive 的 OCR 只差 94.38 vs 94.11。
- **有**：用 PDM（或至少 IBM），并且要在多个 γ 下评测，特别是 γ = 1。
- **诊断手段**：训练时分别记录 ‖e₊‖ 和 ‖e₋‖。如果 e₋ 持续上升，就是 NBA。官方 FLUX 版已经记录 `mse_pos` / `mse_neg`，可以直接照着做。

dpo04 / dpo06（SDEdit-OPD）：如果师生都用同一个 prompt 加空文本或同一个 negative prompt，就属于共享情况。两个实验里负分支实际怎么构造，还没核对。
