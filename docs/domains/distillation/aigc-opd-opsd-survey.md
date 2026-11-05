---
type: survey
tags: ['Distillation', 'Image Editing', 'Video Gen', 'T2I']
status: done
summary: '2026 年，OPD/OPSD 已经从视频少步蒸馏，发展成图像和视频的后训练工具；最值得关注的是多专家能力合并和特权条件自蒸馏，不只是生成加速。'
models: ['Any-OPD', 'D-OPSD', 'MT-OPSD', 'OPSD-V', 'Flow-OPD', 'DiffusionOPD']
---

# AIGC 图像与视频生成、编辑中的 OPD / OPSD 进展

**调研时间范围：2026-01-01—2026-11-05（含）**  
**调研模型：`openai-codex/gpt-6.1-sol`**  
**组织方式：图像生成、图像编辑、视频生成/编辑、独立证据审计四路并行，汇总后交叉核验。**

> 日期主要采用 arXiv 首次提交日期，重要修订另行说明；它不等于会议发表、产品上线或代码发布的日期。性能数字为作者报告，本次没有独立运行论文训练或评测。部分全文、权重直连受限，未核实之处单独标明。“有仓库”“有权重链接”和“完整可复现”不混为一谈。本调研不宣称穷尽全部全年资料。

## 摘要

**2026 年，OPD/OPSD 已经从视频少步蒸馏，发展成图像和视频的后训练工具；最值得关注的是多专家能力合并和特权条件自蒸馏，不只是生成加速。**

主要进展可以概括为：

1. **多专家整合**：把构图、OCR、美学、编辑、音频等专长教师合并到一个学生。
2. **特权条件内化**：教师额外看到目标图、增强 prompt、干净编辑源图或真实视频历史，学生在推理时不再需要这些信息。
3. **学生误差修正**：训练状态和历史来自学生，直接处理多轮编辑、长视频生成中的分布偏移。
4. **联合能力与效率优化**：不再先能力合并、再独立压缩步数，而是联合训练可部署的少步学生。
5. **异构教师桥接**：跨 VAE、架构与噪声调度迁移能力，不能继续假设教师和学生处在同一 latent 坐标系。

现有证据不支持“普遍优于 RL”“不会遗忘”“必然超越教师”或“视觉 OPD/OPSD 已形成完整统一技术栈”等强结论。

---

## 一、术语与方法边界

### 1. OPD：在学生走到的状态上教学生

OPD 通常指 **On-Policy Distillation**。典型流程：

```text
学生自己生成轨迹
      ↓
取学生访问的 noisy latent / token prefix / 视频历史
      ↓
教师在这些状态上提供监督
      ↓
更新学生
```

它与“教师先生成一批图片，再拿来 SFT”不同：**监督状态是否来自当前学生，是核心区别。**

对于共享 latent 空间的 flow/diffusion 模型，一种代表性目标为：

$$
\mathcal L_{\mathrm{OPD}}
=
\mathbb E_{x_t\sim d_\theta}
\left[
 w(t)\left\|
 v_\theta(x_t,t,c)
 -\operatorname{sg}[v_T(x_t,t,c_T)]
 \right\|^2
\right].
$$

其中，$d_\theta$ 是学生诱导的状态分布，$\operatorname{sg}$ 表示停止梯度。这只是代表性形式，不能概括所有 OPD：实际还存在 PPO、DMD、黑盒输出监督等实现。

**命名注意：**DiffusionOPD 的标题使用 On-Policy Distillation，但摘要也使用了 Online Policy Distillation。应保留这一原文用词差异，而不是据此把整个领域改名为普通在线学习。

### 2. OPSD：同源教师通过额外信息获得优势

OPSD 通常指 **On-Policy Self-Distillation**。教师可以是同模型的冻结副本、EMA 副本或经过门控晋升的学生检查点。

| 学生条件 | 教师额外条件 | 代表工作 |
|---|---|---|
| 文本 | 目标参考图 | D-OPSD |
| 原始 prompt | 增强后的 prompt | PE-OPSD |
| 多轮编辑退化后的源图 | 原始干净源图 | MT-OPSD |
| 自生成视频历史 | 更干净的真实视频历史 | OPSD-V |

**Self 不等于没有外部信息、没有奖励模型，或始终只保留一份参数。**例如 DiffusionOPSD 使用奖励梯度构造正负目标，与特权上下文教师不同。

### 3. 必须区分的近邻

- 离线教师图片 SFT，不自动属于 OPD。
- 普通 DPO/GRPO，不自动属于 OPD/OPSD。
- 普通 consistency、DMD 或 few-step 蒸馏，必须核查学生分布和教师监督方式，不能仅凭“加速”纳入。
- Self-Forcing 通常让历史来自学生，结合视频级分布匹配；不必然等于逐去噪状态上的 velocity matching。
- Hybrid-policy 从教师轨迹锚点出发，再由学生局部 rollout；不能包装为全程纯学生分布。
- 直接速度匹配通常要求共同状态坐标和时间定义；异构 VAE、噪声调度需要桥接。
- 确定性 ODE 情形的 L2 转移匹配 surrogate，不等于任意确定性轨迹之间都有有限 KL。

### 4. 起源与背景

OPD 并非 2025 年 Thinking Machines 博客首次提出。更早相关来源包括：

- [DAgger](https://arxiv.org/abs/1011.0686)：学习者诱导的状态分布上查询专家，是思想前驱，不等于今天的完整 OPD 实现。
- [MiniLLM](https://arxiv.org/abs/2306.08543v1)，2023-06-14。
- [GKD](https://arxiv.org/abs/2306.13649v1)，2023-06-23；早期标题为 Generalized Knowledge Distillation for Auto-regressive Sequence Models，后改为 On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes。
- [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734)，2026-01-26，是本轮追到的明确 OPSD 命名来源之一；实验是数学推理，不是视觉生成。不能据此宣称绝对历史首次。
- [Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942)，2026-02-04，也定义相关 OPSD 机制，但实现与损失并非完全相同。

---

## 二、图像生成：从多专家合并到四步部署

### 1. 主要工作时间线

| 首发日期 | 工作 | 核心进展 |
|---|---|---|
| 05-06 | [D-OPSD](https://arxiv.org/abs/2605.05204) | 目标参考图构造更强 self-teacher，持续微调已少步蒸馏模型 |
| 05-08 | [Flow-OPD](https://arxiv.org/abs/2605.08063) | 构图、OCR、偏好等专家能力合并 |
| 05-14 | [DiffusionOPD](https://arxiv.org/abs/2605.15055) | 连续转移 KL/均值匹配视角下的 diffusion OPD |
| 06-25 | [DanceOPD](https://arxiv.org/abs/2606.27377) | 生成、局部编辑、全局编辑能力场整合；减少教师查询 |
| 08-04 | [Any-OPD](https://arxiv.org/abs/2608.03316) | 跨 VAE、架构、噪声调度的异构模型蒸馏 |
| 08-05 | [Poly-OPD](https://arxiv.org/abs/2608.04349) | 多异构教师，通过像素与表征桥接迁移能力 |
| 08-05 | [STEP-OPD](https://arxiv.org/abs/2608.04887) | 教师目标外推与内部表征动态对齐 |
| 08-07 | [FlowErase-OPD](https://arxiv.org/abs/2608.07620) | 多概念擦除与生成能力保留 |
| 08-10 | [DreOPD](https://arxiv.org/abs/2608.09233) | 退化 reference 加强外推目标 |
| 08-25 | [DiffusionOPSD](https://arxiv.org/abs/2608.24646) | 奖励梯度构造正负目标，有限步自蒸馏 |
| 09-26 | [CT-OPD](https://arxiv.org/abs/2609.32781) | 统一离散模型中，使用学生轨迹派生 mask 监督 |
| 09-29 | [PE-OPSD](https://arxiv.org/abs/2609.36638) | 将 prompt enhancement 内化进生成器 |
| 10-13 | [SMD](https://arxiv.org/abs/2610.18472) | 回归式 OPD 实现 4-NFE，覆盖图像、编辑与视频 |
| 10-25 | [PhaseOPD](https://arxiv.org/abs/2610.34191) | 自适应选择教师查询时刻，并提前结束 rollout |
| 11-03 | [StepX-OPD](https://arxiv.org/abs/2611.04386) | 联合多教师能力合并与四步生成 |

本轮未核实到 1—4 月符合本综述严格范围的新增图像生成核心工作；**这不等于证明没有。**

### 2. 路线 A：多教师能力合并

代表：Flow-OPD、DiffusionOPD、DanceOPD。

```text
基础模型
  ├─ 构图专家
  ├─ OCR 专家
  └─ 偏好/美学专家
          ↓
学生自己的生成轨迹 + 按任务查询教师
          ↓
一个多能力学生
```

- **Flow-OPD**：包含 cold-start、PPO-clipped 更新和 Manifold Anchor Regularization，不能简化成纯速度 MSE。首次 05-08，重要修订 v5 为 05-24。
- **DiffusionOPD**：学生自身 rollout，多任务教师提供连续转移 KL/均值匹配信号，覆盖 SDE 和 ODE。首次 05-14，v2 为 09-11。
- **DanceOPD**：单样本 hard-route 一个冻结能力场，在低噪声学生状态做单次教师查询和局部速度 MSE；扩展到生成与编辑。首次 06-25，v3 为 08-15。

| 工作 | 作者报告结果 | 审计注意 |
|---|---|---|
| Flow-OPD | 论文 v5 GenEval/OCR 为 0.93/0.93 | 项目页为 0.92/0.94，版本不一致；不可挑最好数字拼接 |
| DiffusionOPD | v2 主结果 GenEval/OCR 为 0.96/0.94 | 需要前置专家训练，不能只计算蒸馏阶段成本 |
| DanceOPD | 生成+编辑配置 GEditBench 5.347 | README 明确未发布论文教师 LoRA 和学生 checkpoint |

代码与项目：

- [Flow-OPD 项目](https://costaliya.github.io/Flow-OPD/) · [代码](https://github.com/CostaliyA/Flow-OPD)
- [DiffusionOPD 代码](https://github.com/ali-vilab/DiffusionOPD)
- [DanceOPD 项目](https://danceopd.github.io/) · [代码](https://github.com/worldbench/DanceOPD)

**判断：**方法路线已明确，但完整复现仍依赖教师、训练数据、权重与评测配置的公开程度。

### 3. 路线 B：特权条件自蒸馏

#### D-OPSD

完整标题：[D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models](https://arxiv.org/abs/2605.05204)。

- 学生仅使用文本条件。
- 同源教师额外获得目标图像信息。
- 在学生走到的相同状态上匹配预测。
- 研究个性化、风格/领域适配，以及维持少步模型的可训练性。
- 实验覆盖 Z-Image-Turbo 6B、FLUX.2-klein 4B。

**负结果不能省略：**FLUX.2-klein 的一个动漫适配实验中，FID 45.23→38.29，但 GenEval 0.816→0.730。因此不能称作无遗忘持续学习。

论文报告约为普通 SFT 的 4× FLOPs、2× 每迭代时间；部分大规模实验使用内部动漫数据，限制完整复现。

[项目](https://vvvvvjdy.github.io/d-opsd/) · [代码](https://github.com/vvvvvjdy/D-OPSD)

#### PE-OPSD

完整标题：[PE-OPSD: Internalizing Prompt Enhancement into Flow-matching Models via On-Policy Self-Distillation](https://arxiv.org/abs/2609.36638)。

```text
原始 prompt → 学生
增强 prompt → 同源教师
```

在学生状态上做监督，推理时去掉额外 prompt enhancement。论文表中 SD3.5-M GenEval 0.628→0.797，Z-Image 0.650→0.846。

**开放限制：**核查时[仓库](https://github.com/sleepy1231/PE-OPSD)只有 README，不能写成训练代码已公开。

### 4. 路线 C：异构教师桥接

最相关工作：[Any-OPD](https://arxiv.org/abs/2608.03316)、[Poly-OPD](https://arxiv.org/abs/2608.04349)。

不同模型可能具有不同 VAE、latent 维度/尺度、时间和噪声定义、架构，因此不能直接对两个 velocity 做相减。

- **Any-OPD**：把教师视为黑盒 sampler，通过冻结、模型无关的视觉表征连接教师和学生；匹配连续噪声水平而非原始 step index，并使用 anchoring 缓解域差异。
- **Poly-OPD**：学生生成图像被教师重新编码和精修，再在 DINOv2 空间监督；注意力 LoRA 共享、部分 adapter 教师专属，用于控制跨教师干扰。

本轮核实到原始摘要的机制与实验声明，**未达到 DiffusionOPD 等工作的正文及代码深审程度**。

### 5. 路线 D：改进目标，而不只是复制教师

#### STEP-OPD

[论文](https://arxiv.org/abs/2608.04887)使用外推目标：

$$
v_{\mathrm{target}}=v_T+\alpha_k(v_T-v_B),
$$

并匹配相邻 blocks 的内部表征变化。论文同表中，DiffusionOPD→STEP-OPD：GenEval 0.927→0.961，OCR 0.941→0.946。外推系数并非越大越好。

[项目](https://stepopd.github.io/) · [代码](https://github.com/LiangrunFlora/STEP-OPD)

#### DreOPD

[论文](https://arxiv.org/abs/2608.09233)用轻度退化 reference 加强 teacher-reference 对比，将目标转为闭式 velocity 回归。单任务报告 GenEval teacher 0.9470→student 0.9710，但并非全部指标超过教师。

[项目](https://sleepy1231.github.io/DreOPD/) · [代码](https://github.com/sleepy1231/DreOPD)

#### DiffusionOPSD

[On-Policy Self-Distillation in Diffusion Models](https://arxiv.org/abs/2608.24646)使用冻结 behavior policy 和可微奖励构造 detached 正负 clean-output targets，有限步拟合后刷新行为策略。

它不同于“额外参考图教师”：**没有独立特权图像教师，但需要可微奖励信号。**

[项目](https://diffusionopsd.github.io/) · [关联仓库](https://github.com/worldbench/DiffusionOPSD) · [v2 博客](https://diffusionopsd.github.io/blog/v2/)

审计注意：

- 关联仓库自称 external implementation，不能无条件称已核实官方实现。
- 不少结果使用各奖励专用 checkpoint，不能描述成一个通用模型横扫全部指标。
- 网页/配方“v2”不等于已确认的 arXiv v2。
- 博客披露预处理与计时口径修正，不能把新数字回填进早期论文。

### 6. 路线 E：能力合并与少步部署联合训练

#### SMD

完整标题：[Simplifying and Improving On-Policy Distillation for Few-Step Diffusion](https://arxiv.org/abs/2610.18472)，首次 10-13；方法名 Simplified Matching Distillation。

- on-policy velocity regression。
- 训练期 residual proxy：Pivot，用于分离局部修正与 interval-mean prediction。
- 排斥较弱教师预测。
- 不需要在线 score estimation 或对抗训练。
- 摘要覆盖图像生成、编辑、视频和联合音视频，最大到 33B。

本轮核实了原始摘要，**尚未完成完整实验表和代码审计**。

#### PhaseOPD

[Timing Matters Twice: When to Query and When to Skip in Flow-Matching On-Policy Distillation](https://arxiv.org/abs/2610.34191)，首次 10-25。

通过 predicted-clean latent 的变化识别生成阶段，按教师/任务选择 query 时刻，并在所需查询完成后跳过 rollout 后缀。同文比较 DanceOPD→PhaseOPD，训练时间 275.67→161.89 分钟，减少 41.3%；但与其他方法比较时并非各指标都最优。

#### StepX-OPD

[StepX-OPD: Rethinking Multi-Teacher On-Policy Distillation in Diffusion Models](https://arxiv.org/abs/2611.04386)，首次 **11-03**。

将多教师能力合并与四步生成联合优化，避免先能力蒸馏、再单独减少步数造成能力损失。

经补充核查的 SD3.5-M 同表数据：

| 方法 | NFE | GenEval | OCR | 归一化平均分 |
|---|---:|---:|---:|---:|
| Flow-OPD | 80 | 0.905 | 0.885 | 0.8788 |
| DiffusionOPD | 80 | 0.945 | 0.930 | 0.9022 |
| DiffusionOPD + 后续步数蒸馏 | 4 | 0.838 | 0.833 | 0.8403 |
| StepX-OPD | 4 | 0.944 | 0.899 | 0.8901 |

**不能据此说四步全面超过 DiffusionOPD：**平均分仍低于同表多步 DiffusionOPD。价值是较顺序式“能力合并→压步数”保留更多能力。

[代码](https://github.com/WinterCCC/StepX-OPD)核查到 SD3.5 特定变体，不能据此认定全部骨干和论文变体均已公开。

### 7. 统一离散模型的边界项

- [CT-OPD](https://arxiv.org/abs/2609.32781)，09-26：学生 rollout 提供未解析位置 mask，但丢弃学生 provisional token 值，重建教师 context/targets。应归为**轨迹 mask 派生监督**，不等于保留原始学生状态查询教师分布。同文 MMaDA GenEval 55.45→61.69，而 matched random-mask 对照为 60.06，真正由轨迹 mask 带来的增量应单独看。
- [NLD-Uni](https://arxiv.org/abs/2610.36466)，10-27：见下节编辑部分。其 T2I 分支没有 OPD teacher，不能把文生图成绩归因于 T2I OPD。

本轮未核实到纯 AR 图像 token 生成学生上、严格 OPD 且有生成质量实证的独立 2026 核心工作；统一模型的理解分支不应被误算作 AR 图像生成实证。

---

## 三、图像编辑：多轮稳定性是最直接的新突破

### 1. MT-OPSD

[On-Policy Self-Distillation for Multi-Turn Image Editing](https://arxiv.org/abs/2609.35611)，首次 **09-28**；Liangbing Zhao、Le Zhuo、Mohamed Elhoseiny。

核心问题：每次编辑都把上一轮生成结果作为输入，误差和非目标区域损伤逐轮积累。

方法：

- 学生条件图：反复执行 identity editing 得到的自生成退化图。
- 教师条件图：原始干净图。
- 两者在相同学生 noisy latent 上做速度监督。
- 加入 identity 分支与教师晋升门控。

约 2,000 个源图/指令对，不使用目标编辑图；LME-Bench 为 100 个十轮会话。

| 编辑骨干 | 十轮成功率 SR@10，前→后 |
|---|---:|
| Qwen-Image-Edit | 0.03→0.44 |
| FireRed-Image-Edit | 0.15→0.52 |
| FLUX.2-klein-base | 0.12→0.38 |

关键限制与消融：

1. Qwen 单轮 ImgEdit 4.51→4.49，不是全部指标提升。
2. 将速度匹配替换为伪目标 SFT，SR@10 仅 0.03，支持监督形式的重要性。
3. 不需要目标编辑图，不等于没有外部监督成本：门控与评测使用 GPT-4o。
4. 训练与异步门控均需要额外资源。

[仓库](https://github.com/liangbingzhao/MT-OPSD)经补充 API 核查，确有 `train.py`、`inference.py`、`mtopsd/`、`gate_worker.py` 等代码，不是仅 README。

### 2. 其他编辑进展

| 日期 | 工作 | 分类与注意 |
|---|---|---|
| 05-25 | [CollectionLoRA](https://arxiv.org/abs/2605.25378) | 多效果编辑 LoRA 与少步能力合并；原始论文最多 50 种效果，不要将后续 README 的规模回填进原始论文 |
| 06-25 | [Qwen-Image-2.0-RL Technical Report](https://arxiv.org/abs/2606.27608) | 生成、编辑分别 RL，再用学生轨迹上的 OPD 合并 |
| 09-10 | [SenseNova-U1.5](https://arxiv.org/abs/2609.11929) | 美学、文字、信息图、编辑专家的 multi-expert OPD |
| 10-19 | [CheckDistill](https://arxiv.org/abs/2610.26740) | 编辑奖励模型的 OPSD；随后用奖励做编辑器 RL，不是编辑器自身 OPSD |
| 10-27 | [NLD-Uni](https://arxiv.org/abs/2610.36466) | 学生输出被教师精修，再 SFT；作者承认不是传统严格 on-policy |
| 11-04 | [TailSteer](https://arxiv.org/abs/2611.06741) | 多轮编辑的 on-policy 轨迹保持近邻方法，不直接归为 OPSD |

#### CollectionLoRA

[代码](https://github.com/Qwen-Applications/CollectionLoRA)存在。原始表中，50-in-1 FM+Lightning→CollectionLoRA：EditReward 0.929→1.052，BCR 0.217→0.087，NFE 均为 8；与单效果教师比较时部分保真指标下降。它是效果压缩与加速的交叉路线，不是 MT-OPSD 式特权自蒸馏。

#### Qwen 与 SenseNova 技术报告

- Qwen 报告整体编辑 Arena Elo 1256→1349；但这是 RL+OPD 完整流水线的收益，不能全部归因于 OPD。
- SenseNova-U1.5 使用美学、文字、信息图、编辑四类教师；模型/代码入口存在，但“将开放”不等于完整 RL+OPD 管线已交付。

[SenseNova 项目](https://unify.light-ai.top/) · [代码](https://github.com/OpenSenseNova/SenseNova-U1)

#### CheckDistill

学生奖励模型生成评分轨迹，EMA 教师额外获得 checklist、逐项判断与理由，进行分布蒸馏。使用其奖励进行后续 RL，编辑器 ImgEdit 4.29→4.46。应明确这条间接链：

```text
奖励模型 OPSD → 更好的奖励 → 编辑器 RL
```

而不是“编辑器做 OPSD 后提升”。

#### NLD-Uni

[论文 §3.4](https://arxiv.org/html/2610.36466)明确说其编辑分支不是传统意义的 on-policy：学生图像经教师 VAE 编码、加噪、重新去噪，再对 refined output 做标准 SFT。

表4 ImgEdit：Base 3.81、offline KD 3.90、其方法 4.17。T2I 没有 OPD teacher，不能把其文生图指标归因于 OPD。

[代码](https://github.com/NVlabs/Nemotron-Labs-Diffusion-Uni)

---

## 四、视频生成与编辑

### 1. 路线 A：Self-Forcing 谱系

2025 的 [Self Forcing](https://arxiv.org/abs/2506.08009) 是背景，不计入 2026 新增。

| 日期 | 工作 | 核心贡献 |
|---|---|---|
| 02-02 | [Causal Forcing](https://arxiv.org/abs/2602.02214) | 因果初始化与 asymmetric DMD，改进实时 AR 视频蒸馏 |
| 03-12 | [OmniForcing](https://arxiv.org/abs/2603.11647) | 联合音视频因果蒸馏，Joint Self-Forcing 处理学生历史 |
| 03-26 | [ShotStream](https://arxiv.org/abs/2603.25746) | 镜头内→镜头间两阶段 self-forcing，多镜头流式生成 |
| 04-28 | [Mutual Forcing](https://arxiv.org/abs/2604.25819) | 同模型 Few/Multi 模式互教，结合 DMD 与 ShortCut |
| 05-14 | [Causal Forcing++](https://arxiv.org/abs/2605.15141) | 改善可扩展初始化，降低少步 AR 视频蒸馏成本 |
| 05-22 | [One-Forcing](https://arxiv.org/abs/2605.23458) | DMD+GAN，稳定单步自回归视频生成 |

这类工作的 on-policy 往往体现在训练历史来自学生，不一定采用逐去噪状态的教师速度 MSE。

| 工作 | 原始实验例子 | 比较边界 |
|---|---|---|
| Causal Forcing | 同文 chunk-wise 消融 VBench 82.00→84.04 | 不能与其他论文不同协议数字混比 |
| OmniForcing | 首块延迟 197s→0.7s，但 FVD 125.4→137.2、FAD 4.6→5.7 | 不能描述为无损加速 |
| ShotStream | 两阶段训练提高跨镜头主体一致性 0.604→0.654 | 内部数据和公开参考实现存在分布差距 |
| Mutual Forcing | 30 FPS@192×336；12 FPS@480×768；3.5 FPS@704×1280 | “实时”依赖分辨率，不能笼统宣称高分辨率实时 |
| One-Forcing | 同表单步 VBench 83.76，单步 Self-Forcing 77.18 | 包含 GAN 真实视频训练等额外因素，评测采样协议也需锁定 |

代码：

- [Causal Forcing / ++](https://github.com/thu-ml/Causal-Forcing)
- [ShotStream](https://github.com/KlingAIResearch/ShotStream)
- [Mutual Forcing](https://github.com/HVision-NKU/MutualForcing)
- [One-Forcing](https://github.com/Aurora-edu/One-Forcing)

### 2. 路线 B：OPSD-V，四步不变，修正长视频漂移

[OPSD-V: On-Policy Self-Distillation for Post-Training Few-Step Autoregressive Video Generators](https://arxiv.org/abs/2607.08766)，首次 **07-09**。

- 学生保持原四步推理与自身 KV 历史。
- 教师使用更干净的真实长视频旧历史，同时保留最近模型生成的 cache chunk。
- 教师在相同学生去噪状态提供密集监督。
- **经补充原文核验：底座冻结，学生 LoRA 可训练，教师 LoRA 是学生的 EMA 副本。**

| 底座 | 质量分数 | 动态分数 |
|---|---|---|
| LongLive | 0.8138→0.8242 | 0.5012→0.5890 |
| Self-Forcing | 0.8259→0.8389 | 0.6218→0.6570 |

语义分数略降，不是全面无代价提升。训练数据及部分内部评测限制独立复现；用户研究规模较小，不宜过度外推。

[项目](https://meigen-ai.github.io/OPSD-V) · [代码](https://github.com/MeiGen-AI/OPSD-V)

### 3. 路线 C：任意步数、特权控制与多模态合并

| 日期 | 工作 | 核心进展与边界 |
|---|---|---|
| 05-13 | [AnyFlow](https://arxiv.org/abs/2605.13724) | on-policy flow-map 蒸馏，支持任意步数、causal/bidirectional、T2V/I2V/V2V |
| 08-13 | [HPSD](https://arxiv.org/abs/2608.13205) | 高质量首帧+增强文本教师；采用 hybrid-policy，不是纯学生起点 rollout |
| 09-29 | [LIFT](https://arxiv.org/abs/2609.38146) | 稠密未来布局教师→稀疏布局/相机条件学生，属于可控 I2V |
| 11-02 | [OmniOPD](https://arxiv.org/abs/2611.02803) | 音频与视频 RL 专家合并，重点是模态监督平衡，不是压缩采样步数 |

#### AnyFlow

因果 14B T2V 在 4/32 NFE 报告 84.05/84.41；I2V 4 NFE 为 87.87。验证的是任意步数质量缩放，NFE 下降不能直接解释为等倍 wall-clock 加速。

[代码](https://github.com/NVlabs/AnyFlow)

#### HPSD

从教师轨迹 anchor 出发，再由学生局部 rollout。其失败分析尤其重要：高质量首帧教师和纯文本学生条件不同，naive OPD 会遇到 condition-state mismatch。

Wan 实验中 VideoAlign 0.5335→1.8753，但 VBench 动态 0.560→0.422，说明奖励与美学提升不等于运动增强。

[代码](https://github.com/Bujiazi/HPSD)

#### LIFT

Dense-to-sparse SFT→OPSD：FVD 110.53→99.35，mIoU 0.47→0.51。适应阶段 sample updates 128K→8K，但前面共同的 camera/dense-layout 训练不能从总成本里删掉。

[代码](https://github.com/jsxzs/LIFT)

#### OmniOPD

在学生联合音视频轨迹上接受模态专家监督；音频教师可看到更干净视觉条件，并引入 multi-noise sampling、可靠性加权。

同文 Vanilla OPD→OmniOPD：VQ 3.174→3.347、AQ 5.342→5.449、DeSync 0.583→0.457。但音频专家 AQ 为 5.576，不能称完整保留所有教师指标。

[代码](https://github.com/fanzh03/OmniOPD)核查到推理、评分、LoRA 合并等内容，未确认完整 OPD 训练入口。底座 LTX-2/LTX-2.3 命名存在差异，复现需锁定 checkpoint。

### 4. 视频编辑：已有明确进展，但覆盖不如图像编辑完整

- [RTR-DiT](https://arxiv.org/abs/2604.13509)，04-15：文字/参考图引导的流式视频风格化，学生 self-rollout、DMD 与对抗后训练。
- [RealCam](https://arxiv.org/abs/2605.06051)，05-07：相机控制的视频新视角重渲染，双向教师→因果学生。

注意速度口径：RTR-DiT 某表中处理 5 秒视频用时 0.12 分钟，即 7.2 秒，并不是严格端到端实时证据；RealCam 的 0.72 秒是流式响应/首块延迟，不是整段视频耗时。

**本轮尚未核实到通用视频对象增删、局部替换、inpainting 方向中，具有同等完整“学生分布+教师目标+原表+代码”证据链的方法集合。这不是证明不存在。**

---

## 五、技术报告与博客

| 来源 | 价值与边界 |
|---|---|
| [Thinking Machines：On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 学生分布监督的直观解释；2025-10-27，属于背景，实验是 LLM |
| [Qwen-Image-2.0-RL Technical Report](https://arxiv.org/abs/2606.27608) | OPD 如何嵌入实际生成/编辑后训练管线 |
| [SenseNova-U1.5 报告](https://arxiv.org/abs/2609.11929) | 统一模型的多专家 OPD；注意整体收益与独立消融的区别 |
| [DiffusionOPSD 项目](https://diffusionopsd.github.io/)与[v2 博客](https://diffusionopsd.github.io/blog/v2/) | 训练配方、奖励预处理与计时变化；更新数字不能回填早期论文 |
| [DanceOPD README](https://github.com/worldbench/DanceOPD) | 明确披露哪些论文教师/学生权重没有发布 |

2025 背景还包括 [Self-Forcing++](https://arxiv.org/abs/2510.02283) 与 [LiveTalk](https://arxiv.org/abs/2512.23576)。后者首发 2025-12-29，不能因 2026 报道而改算 2026 新论文。

---

## 六、其他真实线索与排除项

### 1. 扩展线索：核实到原始来源，但未达到相同深审程度

| 工作 | 方向 |
|---|---|
| [AFD](https://arxiv.org/abs/2605.26105)，05-25 | 黑盒教师成品视频监督的 on-policy adversarial flow distillation |
| [FlowCTS](https://arxiv.org/abs/2607.24522)，07-27 | 同学生状态比较后续轨迹，研究时间监督 |
| [Self-OPD](https://arxiv.org/abs/2608.26872)，08-27 | 不依赖任务专家教师，但仍使用奖励和随机探索 |
| [MILD](https://arxiv.org/abs/2609.34371)，09-28 | 图像专家→视频生成器，需要异构 latent connector 与运动保持 |
| [PACC](https://arxiv.org/abs/2609.36364)，09-28 | 固定视频生成器，只训练历史压缩器，不能称生成器权重得到更新 |
| [FlowMap-OPD](https://arxiv.org/abs/2609.37851)，09-29 | 少步 flow-map 蒸馏 |
| [GFD-OPD](https://arxiv.org/abs/2609.39692)，09-30 | 跨规模、CFG 误差等问题 |
| [D-OPCD](https://arxiv.org/abs/2610.07250)，10-05 | 将 agent 改进提示词内化，但不代表完整吸收候选选择/验证收益 |
| [FocusOPD](https://arxiv.org/abs/2610.21250)，10-15 | OPD 监督目标扩展 |
| [History Forcing](https://arxiv.org/abs/2611.02782)，11-02 | 历史条件残差修正教师目标，关注长视频场景一致性 |

这些条目不是未加区分的成熟推荐清单；仍需逐项补正文、消融、代码与资源审计。

### 2. 容易误纳入的理解/推理工作

| 来源 | 为什么不算本综述核心生成实证 |
|---|---|
| [Video-OPD](https://arxiv.org/abs/2602.02994) | Temporal Video Grounding，不是视频生成 |
| [Vision-OPD](https://arxiv.org/abs/2605.18740) | 视觉理解/推理 |
| [Visual-OPSD](https://arxiv.org/abs/2606.18974) | 将 visual thoughts 的推理收益内化，不等于生成质量提升 |
| [OPD-V](https://arxiv.org/abs/2608.05131) | 多模态推理；尤其 OPD-V 不等于 OPSD-V |
| [Video-OPSD](https://arxiv.org/abs/2608.27065) | Video-LLM 理解/推理，不是 OPSD-V 长视频生成 |
| [dOPSD](https://arxiv.org/abs/2607.04428) | diffusion 指文本去噪模型 |
| [FA-OPD](https://arxiv.org/abs/2605.27095) | 机器人动作策略，不是 AIGC 图像/视频 |
| [Flow3D-OPD](https://arxiv.org/abs/2609.07137) | 3D 几何生成，属于相邻方向 |

---

## 七、对当前 SeFi / MOSS / Adv-GRPO 工作的启示

### 1. MOSS 合成图片数据不是 OPD

当前已经完成的流程：

```text
caption → MOSS 生成固定图片 → 新 JSONL
```

这是**离线合成数据**。后续用它做 SFT 或偏好训练，不会自动成为 OPD。要成为 OPD，需要让 SeFi 当前学生生成的轨迹或输出参与教师监督构造。

### 2. MOSS→SeFi 优先研究异构桥接

不能默认两者共享 VAE、latent 坐标和噪声时间定义，不能直接照搬同底座 SD3.5 专家的 velocity matching。

优先阅读：**Any-OPD、Poly-OPD**。需要先明确教师可查询的接口、学生状态与教师状态的映射，以及监督是在像素、特征还是 latent 空间完成。

### 3. 多奖励能力整合

建议路径：

```text
DiffusionOPD → Flow-OPD → STEP-OPD / DreOPD → StepX-OPD
```

- 前两者理解教师训练与多能力整合。
- 中间理解监督目标改进。
- StepX-OPD 面向少步部署，避免顺序式压步数的能力损失。

### 4. 真实图分布迁移与 reward hacking

D-OPSD 的参考图像特权信息值得研究，但**不能直接声称它解决了 Adv-GRPO 的 reward hacking**。

- Adv-GRPO：学习和更新奖励/判别器。
- Privileged OPSD：改变教师信息与学生获得监督的方式。

二者能否结合，需要控制实验，不是名称层面的拼接。

---

## 八、优先阅读顺序

| 优先级 | 工作 | 决策价值 |
|---|---|---|
| 1 | DiffusionOPD | 与当前 flow/SD3.5 多奖励后训练最接近 |
| 2 | D-OPSD | 视觉特权条件自蒸馏的直接入口 |
| 3 | Any-OPD / Poly-OPD | 对应 MOSS→SeFi 的异构教师问题 |
| 4 | MT-OPSD | 多轮编辑稳定性有明确实验和消融 |
| 5 | StepX-OPD | 能力整合与四步部署联合优化 |
| 6 | OPSD-V | 历史分布、cache 与去噪状态监督 |
| 7 | HPSD | 理解特权条件迁移的失败模式 |
| 8 | SMD | 跨图像/编辑/视频的统一回归式少步路线，仍需进一步深审 |

---

## 九、证据限制与复现审计要点

### 已确认的口径问题

1. **版本混用：**Flow-OPD 论文和项目页部分数字不一致；DiffusionOPSD 网页 v2 与 arXiv 版本不是同一概念。
2. **整体收益不等于 OPD 因果收益：**Qwen、SenseNova 等同时修改数据、RL、架构与后训练组件。
3. **开源承诺不等于交付：**PE-OPSD 占位仓库、OmniOPD 推理代码、SenseNova SFT 代码不能替代完整 OPD 训练管线。
4. **速度口径：**steps、NFE、首块延迟、吞吐和整段 wall-clock 必须区分；CFG 也影响 NFE。
5. **质量权衡：**适配 FID、奖励或动态分数上升，可能伴随构图、语义、相机精度下降。
6. **归一化指标不可跨论文直接拼榜：**同名 GenEval/OCR/PickScore 可能受 teacher、seed、预处理、sampler 和归一化方式影响。
7. **权重链接未等于实测下载：**本轮未逐项下载核验论文权重及许可证。

### 检索方式

使用 arXiv 搜索/API、摘要页版本史、论文 HTML、作者项目页、GitHub README/目录及部分训练源码；关键词包括完整词组、OPD/OPSD 缩写，以及 image/video/editing/diffusion/flow、self-forcing 等组合，并追踪参考文献。

网络检索使用直连，取消代理变量；未运行训练、未占 GPU、未修改训练环境。部分搜索引擎直连失败、arXiv 查询出现限流，部分论文 HTML/PDF 不可读，因此不能宣称穷尽全部未索引报告与博客。

来源可信度排序：

```text
原论文方法/实验 + 可读实现
    > 原始摘要/作者技术报告
    > 作者项目页/博客
    > 二手解读
```

作者论文与作者项目页互相链接，不构成独立复现。

## 最终判断

**截至 2026-11-05，视觉 OPD/OPSD 已有真实且有价值的进展，但尚不是统一、普适、完整开源的技术栈。最实质的变化，是从“模仿教师、降低步数”，走向“整合专家、内化额外条件、修正学生自身误差，并兼顾部署效率”。**
