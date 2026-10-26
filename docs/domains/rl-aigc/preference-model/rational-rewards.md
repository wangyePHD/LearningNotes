---
type: paper
tags: ['Reward Model', 'Reasoning', 'Data-centric', 'Distillation', 'RL']
status: reading
summary: '**不设计新 RL 算法，只改 Reward Model 的数据生产方式**。Teacher Qwen3-VL-32B-Instruct 在**已知偏好标签**的条件下解释「为什么 A 更好」→ 隐藏标签做一致性过滤（80K 原始对中约 72% 通过，剩 57.6K）→ **Pointwise Projection** 把成对理由投影成单图评价 → SFT 蒸馏到 **Qwen3-VL-8B** Student。输出四维评分 + 文字理由（Text Faithfulness / Image Faithfulness / Physical & Visual Quality / Text Rendering）。UniGenBench++ Overall：FLUX.1-dev 60.97→70.34、SD3.5-Medium 60.71→70.56、Qwen-Image 78.36→82.60。**Reasoning 文本不直接进入 NFT 梯度，只有数值评分进入**。原始数据复用 HPDv3+RapidData 50K T2I 与 EditReward 30K Editing，50K 的抽样规则未披露；Table 12 的 55K Pointwise 样本与「每对两条」的 72K 口径不一致；HF 公开集 348,189 条与论文 80K 对不同口径。**Part 3**：Phase 2 判据只是「理由能否推回标签」而非「理由是否真实」（前后共用同一 Teacher 存在共享偏差）；Phase 3 把相对偏好转成绝对评分，附录 B.2 自承 Calibration Drift / Context Dependence 未解；消融显示同 Teacher 直接蒸馏 57.4% vs PARROT 64.2%。**Part 4**：无 BT Loss 无 GRPO，只一次 Token-level SFT；官方 LLaMA-Factory 配置为 **Full Fine-tuning**（Vision Tower/Projector 冻结、ZeRO-3、BF16、LR 1e-5、3 epoch、seq 32768、img max px 802816），与后续 DiffusionNFT 的 LoRA Rank 64 是两回事；分数以文本逐 Token 预测无独立 Scalar Head；仓库六个数据集标识与论文 80K→57.6K 口径不对应。**Part 5**：固定四维评分（非动态 Rubric），**适用维度等权平均**、N/A 不入分母，1–4 只是锚点可出浮点；组内归一化 R~=0.5+0.5·clip((R-R̄_group)/Z_c,-1,1) 但 **Z_c 计算方式未说明**；Temperature 0.1 非确定性，解析失败示例给 0.0 有被误当低质量图的风险。**Part 6**：接入 DiffusionNFT 而非新算法，训练前按 R̄_initial<3.0 筛困难 Prompt；论文 16 rollout / 16×A100 / 512² / LoRA 64-128 / 过滤阈值 0.9 与 0.05，但**公开代码只有 8 rollout**，且 global_std=True 时 valid_mask 的 stds 是**批次**标准差而非组内标准差；代码里两个 beta（config.beta=1.0 做 NFT 组合 vs config.train.beta=1e-4 做参考正则）与 Table 11 口径不一致。'
---

# RationalRewards：让奖励模型学会「说出理由」

> **arXiv**：[2604.11626](https://arxiv.org/html/2604.11626)（2026 年 4 月）
> **标题**：*RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time*
> **模型**：Teacher Qwen3-VL-32B-Instruct → Student **Qwen3-VL-Instruct-8B**
> **原图**：`Papers/reward-models/RationalRewards.pdf`
> **精读进度**：**Part 1–6 完结**
> **Part 1**：Abstract / §1 / §2 整体框架与核心创新 → §1–§6
> **Part 2**：§3 Training Data / 附录 F.1 → §7–§12
> **Part 3**：§2.1 + 附录 B、C、F，Figure 4 → §13–§18
> **Part 4**：SFT 训练 + 官方开源配置 → §19–§25
> **Part 5**：附录 D、E.1 + Algorithm 1 → §26–§31
> **Part 6**：附录 E.1 RL Fine-Tuning Setup / Table 11 / Algorithm 1 → §32–§39
> **子专题**：[🏅 偏好模型 (Preference Model)](./index.md)

---

## 1. 论文要解决什么问题

这篇论文要解决一个问题：<mark class="hl-key">**传统 Reward Model 只输出一个分数，却没有明确告诉我们图片为什么好、为什么不好。能不能把人类偏好背后的评价理由也学习出来，让 Reward 不仅能打分，还能指导图像生成模型优化？**</mark>

作者提出了 **RationalRewards（8B）**，以及负责生产训练数据的 **PARROT 方法**。

![RationalRewards Figure 1：Train-Time Scaling 与 Test-Time Prompt Tuning 两条应用路径的主结果。(a) ImgEdit-Bench Overall 与 (b) GEdit-Bench-EN Overall 的条形图，GPT-Image-1 分别为 4.20 / 7.53，同族模型经 +RL 与 +PT 后逐级上升（Qwen 家族 +RL 4.43、+PT 8.33）；(c) UniGenBench++ 的极坐标图按 Text / Logic / Relation / Action / Attrib. / Comp. 六个能力轴展开，Qwen-Image [RL] 与 FLUX.1-dev [RL] 相对原始版本在各轴全面外扩，最外圈标注 82.2 / 82.1 / 87.8 / 78.8。](/rational-fig1-train-test-overview.png)

---

## 2. 它和传统 Reward Model 的区别

假设 Prompt 是：

> 一只红色的猫站在蓝色汽车左侧，背景是一座雪山，写实摄影风格。

**传统 Reward Model**：

$$
(\text{Prompt},\ \text{Image}) \rightarrow 0.75
$$

**RationalRewards** 则先分析图片，再给出多维评分。

::: warning 下面是一个帮助理解的假设示例
<mark class="hl-key">**不是论文中的原始样本。**</mark>
:::

RationalRewards 的单图评价（输入：Prompt + 生成图片）：

| 维度 | 评分与理由 |
| :--- | :--- |
| **Text Faithfulness** | 3.0 / 4：<mark class="hl-trick">**猫和汽车都正确，但左右关系错误**</mark> |
| **Physical Quality** | 3.5 / 4：结构自然，光影基本合理 |
| **Text Rendering** | <mark class="hl-key">**不适用：Prompt 没有要求文字**</mark> |

最终 Reward = 适用维度均值：

$$
\frac{3.0 + 3.5}{2} = \boxed{3.25 \ /\ 4}
$$

::: tip 它最大的变化是什么
<mark class="hl-key">**不是单纯把一个数字拆成几个数字，而是模型必须生成文字评价理由，再输出对应的分数。**</mark>
:::

这里需要特别强调：<mark class="hl-key">**论文并没有证明生成了 Reasoning 就一定更准确**</mark>。作者是通过偏好数据监督和对照实验，<mark class="hl-trick">**尝试让 Reasoning 真正与人类偏好保持一致**</mark>。

---

## 3. 核心创新：PARROT 如何训练这个模型？

这也是最值得研究的部分。

<mark class="hl-key">**作者没有从零请人工写几万条评价理由，而是从已有的 Pairwise Preference Data 入手。**</mark>

$$
\begin{aligned}
&\textbf{原始偏好数据} \\
&\qquad (\text{Prompt} + \text{Image A} + \text{Image B} + \text{已知 A 优于 B})
\end{aligned}
$$

$$
\boxed{
\begin{array}{l}
\textbf{Phase 1：Preference-Anchored Rationale} \\
\qquad \text{Qwen3-VL-32B 看图、看 Prompt、看偏好标签，解释 A 为什么更好} \\[6pt]
\textbf{Phase 2：Consistency Filtering} \\
\qquad \text{隐藏偏好标签，检查生成的理由能否支持原始偏好结果} \\[6pt]
\textbf{Phase 3：Pointwise Projection} \\
\qquad \text{借助通过过滤的比较理由，为单张图片生成独立评价和评分} \\[6pt]
\textbf{SFT：Qwen3-VL-8B} \\
\qquad \text{同时学习 Pairwise 和 Pointwise 评价}
\end{array}
}
$$

![RationalRewards Figure 4：PARROT 的三阶段流水线。顶部是统一的目标函数——在观测偏好上最大化数据似然，其证据下界写作 $\log P(y|x) \ge \mathbb{E}_{z\sim q_\phi}[\log P_\theta(y|x,z)] - D_{KL}(q_\phi(z|x,y)\,\|\,P_\theta(z|x))$。① Rationale Generation (Hindsight)：Teacher VLM 接收 User Request 与 Generation A&B (X)，并额外接收人工偏好标签 (Y)，输出推断出的理由 $q_\phi(z|x,y)$——任务从开放式评价塌缩为定向解释。② Causal Consistency Check：只用 (X, Z) 让模型 Re-predict preference，与人工标签比对做一致性过滤。③ Student SFT (Foresight)：最小化 $D_{KL}(q_\phi(z|x,y)\|P_\theta(z|x))$，让 Student VLM 在**看不到偏好标签**的条件下仅凭 (X) 输出 Scoring Rationales (Z) 与分数。](/rational-fig4-parrot-pipeline.png)

这套方法叫 **PARROT（Preference-Anchored Rationalization）**。

### Phase 1 最值得注意的地方

<mark class="hl-key">**Teacher 生成理由时，已经知道哪张图片更好。**</mark>

比如偏好标签为 $A \succ B$，Teacher 的任务不再是自己判断 A、B 谁好，而是解释：

> 已知 A 比 B 更好，请根据图片内容解释原因。

<mark class="hl-trick">**这样可以降低 Teacher 自己判断错误而给出相反答案的风险。**</mark>

::: danger 但这立刻带来一个新问题
<mark class="hl-key">**既然答案提前告诉 Teacher，它会不会为了迎合答案而编造理由？**</mark>
:::

这就是 Phase 2 需要解决的问题。<mark class="hl-key">**作者对生成理由进行一致性检查，只有检查通过的样本才能继续用于训练。**</mark>

::: warning 这种过滤能验证什么、不能验证什么
<mark class="hl-trick">**这种过滤只能验证理由与标签的预测一致性，不能完全证明每条理由都具有真实的视觉依据。**</mark>
:::

论文最终使用<mark class="hl-key">**约 80K 原始 Pairwise 数据，其中约 72% 通过一致性过滤**</mark>。

---

## 4. 为什么还要把 Pairwise 转成 Pointwise？

<mark class="hl-key">**这是整个方法非常关键的一步。**</mark>

训练数据最初是：

$$
(\text{Prompt},\ I_A,\ I_B,\ A \succ B)
$$

但用于 DiffusionNFT 时，我们实际上需要：

$$
R(\text{Prompt},\ I_A)
$$

<mark class="hl-trick">**即每张 Rollout 图片都能单独获得评分，而不是每次都必须找另一张图片进行比较。**</mark>

因此，作者提出 **Pointwise Projection**：<mark class="hl-key">**让 Teacher 单独评价每张图，同时可以参考已经验证过的 Pairwise Reasoning，最终产生独立的多维评分和评价理由**</mark>。

最终 RationalRewards 可以使用两个接口：

| 接口 | 输入 | 输出 |
| :--- | :--- | :--- |
| **Pairwise** | Prompt + Image A + Image B | 比较理由、偏好判断 |
| **Pointwise** | Prompt + Image | 单图评价理由、多维分数 |

![RationalRewards Figure 5：RationalRewards 给出的真实 Pointwise 评分示例。左侧是图像编辑（Instruction: Change the background to a deserted road.），Flux-Kontext-Dev 给出 Text Faithfulness 4.0 / Image Faithfulness 4.0 / Phys-Visual 3.6 / Text Rendering 4.0，而 Ovis-u1 与 Bagel-Think 的 Text Faithfulness 掉到 1.0；右侧是纯文生图（an architectural blueprint displaying a simple house design…），两个 Flux 1[dev] 输出的 Image Faithfulness 均为 N/A——即该维度对纯文生图任务不适用，暴露在分数里而不参与平均。](/rational-fig5-pointwise-examples.jpg)

::: tip 这里与 TRM 有一个明显区别
<mark class="hl-key">**RationalRewards 主要学习有评价标准约束的多维 Reasoning 和评分；TRM 则明确要求每个案例先动态生成一套 Rubric，再逐项检查并评分。**</mark>

另外，<mark class="hl-trick">**RationalRewards 的训练主体是 SFT，论文并没有像 TRM 一样再对 Reward Model 本身进行 PD-GRPO。**</mark>
:::

---

## 5. 训练好 RationalRewards 后，如何用于图像生成？

作者提出两个方向，对应论文 Figure 2。

![RationalRewards Figure 2：两种使用路径。(a) Training Time RL——Prompt（可带参考图）经 Image Generator 生成图片（火焰图标表示可训练），RationalRewards 输出 Rationale 与四维评分（Text Faithfulness 2.2 / Image Faithfulness 2.5 / Physical and Visual Quality 2.0 / Text Rendering 2.0），合成 Reward 1–4 后经 RL Update 回传到 Image Generator。(b) Test Time Prompt Tuning——同样的前向过程，但 Image Generator 带雪花图标表示**冻结**，Rationale 被写成包含 Refined Prompt 的修改建议（"The original instruction 'Make it anime-style' is too vague…" → "Convert the image into a shonen anime style …"），再把 Refined Prompt 回灌给 Prompt Tuning 分支重新生成，全程不更新生成器参数。](/rational-fig2-reasoning-reward-model.png)

### 方向 A：Train-Time Scaling — DiffusionNFT

$$
\begin{aligned}
&\textbf{图像基模} \rightarrow \textbf{每条 Prompt 生成一组图片} \\
&\quad \downarrow \\
&\textbf{RationalRewards} \rightarrow \textbf{多维评分} \rightarrow \textbf{合成 Scalar Reward} \\
&\quad \downarrow \\
&\textbf{DiffusionNFT Loss} \rightarrow \textbf{更新图像基模}
\end{aligned}
$$

### 方向 B：Test-Time Scaling — Prompt Tuning

$$
\begin{aligned}
&\textbf{Prompt} \rightarrow \textbf{图像生成器} \rightarrow \textbf{Image} \\
&\quad \downarrow \\
&\textbf{RationalRewards 找到图片缺陷并提出修改建议} \\
&\quad \downarrow \\
&\textbf{改写 Prompt} \rightarrow \textbf{重新生成图片}
\end{aligned}
$$

<mark class="hl-key">**第一个方向是最关注的：用 RationalRewards 替代现有的 Reward Model，直接进行 DiffusionNFT 后训练。**</mark>

<mark class="hl-trick">**第二个方向无需更新生成模型的参数，只需要针对当前生成图片分析问题、修改 Prompt、再次生成。**</mark>

### 一个必须区分的细节

<mark class="hl-key">**Reasoning 文本本身没有直接参与 DiffusionNFT 的梯度计算。真正进入 NFT 训练的是 Reasoning 所对应的数值评分。**</mark>

<mark class="hl-trick">**Reasoning 的作用主要体现在 Reward Model 学到了怎样评价图片。**</mark>

### UniGenBench++ Overall 结果

| 模型 | 原始模型 | RationalRewards + NFT |
| :--- | ---: | ---: |
| **FLUX.1-dev** | 60.97 | **70.34** |
| **SD3.5-Medium** | 60.71 | **70.56** |
| **Qwen-Image** | 78.36 | **82.60** |

::: warning 这些是积极结果，但需要限定解读
<mark class="hl-trick">**仍然主要集中在一个 T2I 生成能力 Benchmark 上，不能据此确认所有人类偏好和美学质量都同步提升。**</mark>
:::

---

## 6. Part 1 最需要理解的三个结论

::: tip 首先，RationalRewards 的核心不是设计新的 RL 算法
<mark class="hl-key">**而是提出一种新的 Reward Model 训练数据生产方式。**</mark>

它<mark class="hl-trick">**用已有 Pairwise 偏好监督，生产可以训练生成式 Reward Model 的 Reasoning 数据**</mark>。
:::

::: tip 其次，它与 TRM 的技术路线不同
- **TRM**：主要依靠 Gemini + 人工标注结构化 SFT 数据，随后通过 **PD-GRPO** 增强评分；
- **RationalRewards**：利用<mark class="hl-key">**已知人类偏好标签约束 Teacher 产生评价理由，再用 SFT 蒸馏到 8B Student**</mark>。
:::

::: tip 最后，RationalRewards 的价值有两部分
<mark class="hl-key">**既能作为 DiffusionNFT 的数值 Reward，也能在测试阶段利用文字评价指导 Prompt 修正。**</mark>
:::

**下一部分：Part 2 — 原始偏好数据究竟怎么构建？** 我们会直接拆解 50K T2I + 30K Editing 原始数据的来源、图片对的组成、Preference Label 的获取方式、不同质量图片的分布，以及作者究竟公开了多少过滤细节。

::: danger 需要深究的地方
如果 Teacher 已经知道偏好答案，那么 PARROT 生成的 Reasoning<mark class="hl-key">**到底是在发现图片质量差异，还是在事后寻找理由解释已有答案？**</mark>

这是理解 RationalRewards 可靠性的核心问题，也是 Part 3 会重点分析的内容。
:::

---

## 7. 原始数据规模与来源

先给出一个重要结论：<mark class="hl-key">**RationalRewards 没有从零采集 80K 对图片并重新组织大规模人工标注，而是复用了已有的偏好数据集。真正新增的是将偏好标签转换为 Reasoning 监督的 PARROT 流程。**</mark>

论文 §3 和附录 F.1 给出了以下数据组成：

| 来源 | 任务 | 原始 Pairwise 数据 | PARROT 一致性过滤后 |
| :--- | :--- | ---: | ---: |
| **HPDv3 + RapidData** | 文生图 T2I | 50,000 对 | 约 36,000 对 |
| **EditReward** | 图像编辑 | 30,000 对 | 约 21,600 对 |
| **总计** | 两个任务 | **80,000 对** | **约 57,600 对** |

::: warning 三个容易混淆的细节
**首先**，<mark class="hl-key">**50K 是 HPDv3 与 RapidData 合计的文生图偏好对数，不是两个数据集各 50K。论文没有进一步披露各自占多少。**</mark>

**其次**，这里的 80K 是 <mark class="hl-key">**80K 对比较样本，不是 80K 张互不重复的图片**</mark>。一对包含两张候选图片，<mark class="hl-trick">**同一张图片也可能出现在不同的比较对中**</mark>。

**最后**，<mark class="hl-key">**57.6K 是通过后续 PARROT 一致性过滤留下的 Pairwise 数量，还不能直接等同于最终 SFT 样本数量**，因为作者又进行了 Pointwise Projection。</mark>
:::

---

## 8. 每条原始数据具体是什么？

假设有这样一条文生图 Prompt：

> 一只红色的猫坐在蓝色汽车的左侧，背景是雪山。

对同一条 Prompt，我们拿到了两张候选图片：

| 候选 | 角色 | 特征 |
| :--- | :--- | :--- |
| **Image A** | **Winner** | <mark class="hl-trick">**符合颜色、主体和位置要求**</mark> |
| **Image B** | **Loser** | <mark class="hl-trick">**猫的位置不符合要求**</mark> |

::: warning 以上只是说明数据结构的示意图片
<mark class="hl-key">**并非论文的原始训练样本。**</mark>
:::

对应的原始数据可以抽象为：

```json
{
  "prompt": "A red cat sits to the left of a blue car...",
  "image_a": "image_a.png",
  "image_b": "image_b.png",
  "preferred": "A"
}
```

即：

$$
\boxed{
D_{\mathrm{raw}} = \{(p,\ I_A,\ I_B,\ y)\}
}
$$

其中 $y$ 只有一个核心含义：<mark class="hl-key">**在这两张图之间，已知哪一张更符合偏好**</mark>。

<mark class="hl-trick">**这批原始数据没有作者新标注的完整 Reasoning、各维度解释或 1–4 分的绝对评分。这些内容需要后续 PARROT 来生成。**</mark>

对于图像编辑，数据结构则多一个 Source Image 和 Editing Instruction：

$$
D_{\mathrm{edit}} = \{(I_{\mathrm{src}},\ p_{\mathrm{edit}},\ I_A,\ I_B,\ y)\}
$$

::: tip 也就是说，RationalRewards 的原始偏好数据监督来源与 TRM 很不一样
- **TRM** 的 SFT 主要从 <mark class="hl-trick">**Gemini 辅助生成的结构化评价案例**</mark> 开始；
- **RationalRewards** 从 <mark class="hl-key">**已经存在的 Pairwise 偏好标签**</mark> 开始，再补充可解释的评价理由。

<mark class="hl-trick">**这里最重要的并非图片对长什么样，而是原始偏好标签本身已经提供了一个可靠性锚点：Teacher 后续生成理由时，至少有一个需要解释的已知比较结果。**</mark>
:::

---

## 9. 偏好标签是怎么来的？是不是全部由人工标注？

这里要区分<mark class="hl-key">**原始数据集的标注方式**</mark>和<mark class="hl-key">**RationalRewards 作者自己的数据生产方式**</mark>。。

以 HPDv3 为例，它本身是一个大规模人类偏好数据集，<mark class="hl-trick">**原始项目包含约 117 万次 Pairwise 比较**，覆盖 FLUX.1-dev、SD3、SDXL、Midjourney、Kolors 等不同来源的图像</mark>。

其公开数据除了 Winner / Loser，还可能包含<mark class="hl-key">**投票分布、标注置信度以及图片的生成模型来源**</mark>。例如，HPDv3 官方公开的数据结构中有：

```json
{
    "prompt": "...",
    "path1": "preferred_image.jpg",
    "path2": "dispreferred_image.jpg",
    "choice_dist": [12, 7],
    "confidence": 0.9999907,
    "model1": "flux",
    "model2": "infinity"
}
```

::: warning 这里容易读错
`choice_dist` 是<mark class="hl-key">**标注投票分布**</mark>，而 `confidence` 是<mark class="hl-trick">**数据集自身提供的偏好置信度，并不是 RationalRewards 新训练出来的分数**</mark>。
:::

::: danger 但 RationalRewards 论文没有具体说明如何从 HPDv3 和 RapidData 中抽出那 50K 对数据
它<mark class="hl-key">**没有明确给出是否依据置信度筛选、是否优先选择最新生成模型、是否对各模型来源进行等量采样，以及 Easy / Hard Pair 的实际比例**</mark>。

这几个细节对复现很重要，<mark class="hl-trick">**不能因为原始 HPDv3 提供了 `confidence`，就认定 RationalRewards 一定用它进行了过滤**</mark>。
:::

---

## 10. 原始数据经过了什么处理？

作者真正明确披露的核心过滤，是 <mark class="hl-key">**PARROT Phase 2 的一致性过滤，而非传统的图像质量过滤**</mark>。

$$
\begin{aligned}
&\textbf{80K 原始偏好对} \\
&\qquad (\text{Prompt / Source Image / A / B / Preference}) \\
&\quad \downarrow\ \textbf{Teacher 根据偏好标签生成 Reasoning} \\
&\quad \downarrow\ \textbf{遮蔽偏好标签，检查 Reasoning 能否推回正确偏好} \\
&\quad \downarrow\ \textbf{保留约 72\%} \\
&\textbf{57.6K Pairs} \\
&\quad \downarrow\ \textbf{进一步生成 Pointwise Reasoning + Score}
\end{aligned}
$$

过滤主要排除三种失败案例：

| 失败类型 | 说明 |
| :--- | :--- |
| **描述了不存在的东西** | <mark class="hl-trick">**Teacher 描述图片中不存在的东西**</mark> |
| **理由支持了错误图片** | <mark class="hl-key">**生成的理由反而支持错误图片**</mark> |
| **理由过于模糊** | <mark class="hl-trick">**理由过于模糊、无法区分两张图片**</mark> |

![RationalRewards Figure 4 的数据侧读法：① Teacher 在已知偏好标签 (Y) 的条件下生成 $q_\phi(z|x,y)$；② 遮蔽标签后仅凭 (X, Z) Re-predict preference 做一致性检查，被拒绝的样本（约 28%）不进入训练；③ 存活的 pairwise 理由再经 Pointwise Projection 展开为单图 Scoring Rationales 与分数，喂给 8B Student 做 SFT。80K → 57.6K → Pointwise 的两次口径切换都发生在这一条链上。](/rational-fig4-parrot-pipeline.png)

::: warning 两个必须标出的口径问题
**第一**，<mark class="hl-key">**72% 的通过率不等于 72% 的理由经过了事实核验。**</mark>一致性过滤检查的是<mark class="hl-trick">**理由能否预测原偏好，不能彻底排除言之成理但视觉依据有误的解释**</mark>。

**第二**，附录 Table 12 说 36K 文生图 Pair 最终约有 <mark class="hl-key">**55K 条 Pointwise 样本**</mark>，但同一表的说明又说<mark class="hl-key">**每对应该投影为两条 Pointwise 数据**</mark>。

按后者计算应为约 72K 条，<mark class="hl-trick">**两个数字不完全一致。论文没有解释这个差额，所以现阶段不能确定是否还有额外筛选**</mark>。
:::

---

## 11. 训练数据实际上已经公开了吗？

是的，官方已发布：

> **RationalRewards-SFTData**（Hugging Face）
> [huggingface.co/datasets/TIGER-Lab/RationalRewards-SFTData](https://huggingface.co/datasets/TIGER-Lab/RationalRewards-SFTData)

数据集页面显示 <mark class="hl-key">**348,189 条记录**</mark>，字段包括：

```text
messages
images
index
tag
model
```

::: danger 这里存在明显的统计口径差异
而论文报告的 <mark class="hl-key">**80K 对是原始偏好数据量。这两个数字明显不属于同一个统计口径**</mark>，<mark class="hl-trick">**公开数据集中包含不同 SFT 轨迹与格式，不能直接把 348K 理解为 348K 对独立人工偏好**</mark>。

<mark class="hl-key">**具体展开和重复关系需要进一步检查数据内容。**</mark>
:::

---

## 12. Part 2 总结

这篇论文的数据路线，可以先记住三个关键数字：

$$
\boxed{
80\text{K 原始偏好对}
\;\rightarrow\;
57.6\text{K 过滤后偏好对}
\;\rightarrow\;
\text{Pairwise + Pointwise SFT 数据}
}
$$

::: danger 从数据研究的角度，有两个值得警惕的地方
**首先**，<mark class="hl-key">**作者没有充分披露 50K 文生图偏好对的抽样规则**</mark>。

对于 Reward Model 来说，<mark class="hl-trick">**50K 条数据究竟来自旧模型还是强基模、难例占多少，可能直接影响它对后续 RL Rollout 的排序能力**</mark>。

**其次**，<mark class="hl-key">**PARROT 一致性过滤主要提高了 Reasoning 与偏好标签的一致性，并不自动保证偏好标签本身无噪声，也不保证 Reasoning 的每一条视觉描述都真实**</mark>。
:::

**下一部分 Part 3 是整篇论文最核心的部分：PARROT。** 我们会进一步拆解 Qwen3-VL-32B 如何利用已知偏好标签生成 Reasoning、为什么要遮蔽答案重新验证、Pointwise Projection 到底怎样生成 1–4 分，以及直接蒸馏和 PARROT 有什么本质区别。

---

# Part 3：PARROT —— 如何把 Pairwise 偏好数据变成 Reasoning Reward 数据

> **对应原文**：§2.1、Figure 4、附录 B、C、F

这是 RationalRewards 最核心的创新。思路其实很简单：<mark class="hl-key">**我们已经知道两张图片谁更好，但不知道为什么更好。PARROT 让一个大 VLM 根据已知偏好生成理由，再检查理由是否能支持原始偏好，最后把这些理由转成单图评分数据，训练一个 8B Reward Model。**</mark>

$$
\begin{aligned}
&\textbf{已知} \quad I_A \succ I_B \\
&\quad \downarrow \\
&\textbf{① Phase 1} \quad \text{Teacher 围绕已知结果寻找视觉证据} \\
&\quad \downarrow \\
&\textbf{② Phase 2} \quad \text{遮蔽标签，检查理由能否推回原偏好} \\
&\quad \downarrow \\
&\textbf{③ Phase 3} \quad \text{把比较理由投影成单图绝对评分} \\
&\quad \downarrow \\
&\textbf{SFT} \quad \text{8B Student 同时学会 Pairwise 与 Pointwise}
\end{aligned}
$$

![RationalRewards Figure 4：PARROT 三阶段全景。顶部是统一目标 $\log P(y|x) \ge \mathbb{E}_{z\sim q_\phi}[\log P_\theta(y|x,z)] - D_{KL}(q_\phi(z|x,y)\|P_\theta(z|x))$，把「理由 $z$」当作潜变量。① Rationale Generation (Hindsight)：Teacher VLM 拿到 User Request 与 Generation A&B (X) 并**额外拿到人工偏好标签 (Y)**，输出 $q_\phi(z|x,y)$。② Causal Consistency Check：只给 (X, Z) 让模型 Re-predict preference，与人工标签比对——这就是 Phase 2 的判据。③ Student SFT (Foresight)：Student VLM 只凭 (X) 输出 Scoring Rationales (Z) 与分数，因为训练时已经**看不到偏好标签**。](/rational-fig4-parrot-pipeline.png)

---

## 13. Phase 1：Preference-Anchored Rationale Generation

假设有一条文生图 Prompt：

> 一只红色的猫站在蓝色汽车左边，背景是一座雪山。

现有偏好数据告诉我们：

$$
I_A \succ I_B
$$

也就是人类更喜欢图片 A。

<mark class="hl-key">**作者没有让 Teacher 自己判断 A 和 B 谁更好，而是直接把偏好答案告诉 Teacher。**</mark>

具体输入：

| 输入 | 内容 |
| :--- | :--- |
| Prompt | 一只红色的猫站在蓝色汽车左边…… |
| Image A | 第一张候选图 |
| Image B | 第二张候选图 |
| Preference Label | <mark class="hl-trick">**A 优于 B**</mark> |
| Teacher | Qwen3-VL-32B-Instruct |

然后要求 Teacher 给出评价理由。

::: warning 下面这个 Pairwise Rationale 是帮助理解的假设示例
<mark class="hl-key">**不是论文中的真实训练样本。**</mark>
:::

| 维度 | Teacher 输出 |
| :--- | :--- |
| **Text Faithfulness** | 图片 A 正确呈现红色猫、蓝色汽车和雪山，且猫位于汽车左侧。图片 B 虽然包含这些主体，但猫位于汽车右侧。 |
| **Physical Quality** | 两张图片整体结构自然，但图片 B 的汽车阴影存在轻微异常。 |
| **Overall** | 图片 A 在 Prompt 遵循和视觉质量上优于图片 B，因此 A 更好。 |

作者把这种方式称为 **Preference Anchoring（偏好锚定）**。

::: tip 它的优势是什么
<mark class="hl-key">**Teacher 不需要先解决「谁更好」的问题，只需要围绕已知结果寻找具体视觉证据**</mark>。

相比让 Teacher 完全自由地评价两张图，<mark class="hl-trick">**可以减少与现有人类偏好标签冲突的情况**</mark>。
:::

::: danger 但这里有个明显风险
<mark class="hl-key">**Teacher 已经知道 A 更好，会不会为了证明 A 更好而编造理由？**</mark>所以作者设计了第二阶段。
:::

---

## 14. Phase 2：Predictive Consistency Filtering

<mark class="hl-key">**这个阶段的想法很有意思：把偏好答案藏起来，看 Teacher 能不能根据前面生成的理由重新判断出 A 更好。**</mark>

$$
\begin{aligned}
&\textbf{原始偏好} \quad A \succ B \\
&\quad \downarrow\ \textbf{Teacher 生成理由 } Z \\
&\quad \downarrow\ \textbf{（Teacher 知道 A 优于 B）} \\
&\quad \downarrow\ \textbf{再次调用 Teacher，提供比较案例与理由 Z，但隐藏原始偏好标签} \\
&\quad \downarrow \\
&\textbf{预测偏好是否仍然为 A 优于 B？} \\
&\quad \bullet\ \textbf{一致} \Rightarrow \text{保留} \\
&\quad \bullet\ \textbf{不一致} \Rightarrow \text{过滤}
\end{aligned}
$$

论文公式 (2) 就是一个操作：

$$
\mathcal{C}(x, y, z) = \mathbb{I}\Bigl[ \arg\max_{y'} P_{\mathrm{Teacher}}(y' \mid x, z) = y \Bigr]
$$

其中 $x$ 是 Prompt 和两张图片，$y$ 是真实偏好，$z$ 是生成的理由。

<mark class="hl-trick">**这里不需要计算什么特殊的视觉损失函数，核心就是再次调用 Teacher 做一次偏好预测，再与原始标签比较。**</mark>

论文中约 72% 的 Reasoning 通过过滤：

| 数据 | 数量 |
| :--- | ---: |
| 原始偏好对 | 80K |
| **保留** | **约 57.6K** |
| **过滤** | **约 22.4K** |

作者总结的失败类型包括：

| 失败类型 | 说明 |
| :--- | :--- |
| 视觉幻觉 | <mark class="hl-trick">**描述了图片里不存在的东西**</mark> |
| 忽略真实偏好 | <mark class="hl-key">**理由反而支持了错误图片**</mark> |
| 过于笼统 | <mark class="hl-trick">**无法区分两张图片**</mark> |

::: danger 这里有一个很大的局限
<mark class="hl-key">**Phase 2 验证的是 Reasoning 与 Preference Label 是否一致，并不是独立验证 Reasoning 是否真实。**</mark>

例如，Teacher 错误地认为图片 B 有严重畸形，实际上 B 没有。<mark class="hl-trick">**只要这个错误理由能够支持 A 优于 B，那么一致性过滤仍可能放行。**</mark>

而且<mark class="hl-key">**前后两次都使用 Qwen3-VL-32B-Instruct，本身存在共享偏差的问题**</mark>。

所以<mark class="hl-trick">**不要把这一步理解为「验证所有 Reasoning 都正确」。更准确地说，它是一次 Preference Consistency Check**</mark>。
:::

---

## 15. Phase 3：Pointwise Projection

<mark class="hl-key">**这是最重要的一步，因为前面生成的都是 Pairwise Reasoning，但我们最终想训练的是一个可以对单张图片独立评分的 Reward Model。**</mark>

作者的做法是：<mark class="hl-key">**把刚才已经验证的 Pairwise Reasoning 当作参考提示，让 Teacher 分别评价图片 A 和 B。**</mark>

例如，对图片 A 的输入是：

```text
Prompt: 一只红色的猫站在蓝色汽车左边……

Image: A

Reference Hint:
在先前比较中，图片 A 的空间关系正确，
且图片整体结构自然。

请独立评价这张图片，并给出各维度评分。
```

Teacher 得到的输出可以类似：

| 维度 | Image A | Image B |
| :--- | ---: | ---: |
| Text Faithfulness | 4.0 | 2.5 |
| Physical and Visual Quality | 3.5 | 3.0 |
| Text Rendering | N/A | N/A |

::: warning 以上为帮助理解的假设分数
<mark class="hl-key">**文生图没有源图，因此 Image Faithfulness 同样是 N/A。**</mark>
:::

![RationalRewards Figure 5：真实的 Pointwise 投影结果。左侧编辑任务（Change the background to a deserted road.）中 Flux-Kontext-Dev 得 4.0 / 4.0 / 3.6 / 4.0，Ovis-u1 与 Bagel-Think 的 Text Faithfulness 只有 1.0；右侧纯文生图（architectural blueprint…）两个输出的 Image Faithfulness 都是 N/A。可以看到「相对判断」被改写成「绝对判断」，而 Image Faithfulness 是否参与平均完全由任务类型决定。](/rational-fig5-pointwise-examples.jpg)

### 你会发现两个变化

::: tip 变化一：评价标准从相对偏好变成了绝对评分
<mark class="hl-key">**不再问 A 是否比 B 好，而是问这张图片本身质量如何。**</mark>
:::

::: tip 变化二：每个维度有明确的 1–4 分评价标准
<mark class="hl-key">**允许 2.5、3.2 这样的浮点分数**</mark>：

| 维度 | 衡量什么 |
| :--- | :--- |
| **Text Faithfulness** | 指令遵循 |
| **Physical and Visual Quality** | 物理合理性及视觉缺陷 |
| **Text Rendering** | 文字渲染 |

<mark class="hl-key">**对于文生图，最终主要使用三个维度；图像编辑则还需要 Image Faithfulness，用来评价对原图未编辑内容的保留。**</mark>
:::

到这里，一条原始 Pairwise 数据就有机会产生两条 Pointwise 数据，分别描述 A 和 B 的质量。

::: danger 但数量对不上
论文附录 Table 12 存在一个未解释的数量差异：<mark class="hl-key">**约 36K 条过滤后的 T2I Pair，实际列出的 Pointwise 样本约 55K 条，而不是理论上的 72K 条**</mark>。

因此<mark class="hl-trick">**不能认定所有 Pair 都完整产生并保留了两条 Pointwise 数据**</mark>。
:::

---

## 16. 最后怎样训练 Student？

经过 PARROT，我们得到两种训练监督：

| 数据类型 | Student 的输入 | Student 学习输出 |
| :--- | :--- | :--- |
| **Pairwise** | Prompt + Image A + Image B | 比较理由 + 哪张更好 |
| **Pointwise** | Prompt + 单张 Image | 各维度 Reasoning + 1–4 分数 |

然后使用 <mark class="hl-key">Qwen3-VL-8B 做 SFT，学习这些结构化文本输出**</mark>。

::: danger 特别注意
<mark class="hl-key">**Student 训练和实际推理时，不会获得原始 Preference Label 作为提示。**</mark>
:::

我们可以把它理解为：

$$
\boxed{
\text{Pairwise Preference}
\xrightarrow{\text{PARROT}}
\text{Reasoning Supervision}
\xrightarrow{\text{SFT}}
\text{RationalRewards-8B}
}
$$

::: tip 变分解释 ≠ 实际优化目标
作者还提供了变分推断的 ELBO 理论解释，<mark class="hl-key">**但工程上并不是先显式计算一个复杂 KL Loss、再单独优化 Teacher**</mark>。

最终 Student 使用的<mark class="hl-trick">**核心仍然是标准的 SFT Token Prediction Loss**</mark>。
:::

具体的 LoRA、Batch Size、Epoch、数据格式和训练脚本，我们留到 Part 4。

---

## 17. 这套数据生产方式真正值得思考的地方

<mark class="hl-key">**PARROT 与 TRM 的数据生产有一个本质不同：**</mark>

| | 监督数据的来源 |
| :--- | :--- |
| **TRM** | <mark class="hl-trick">**让 Gemini 直接生成高质量评价数据**</mark> |
| **PARROT** | <mark class="hl-key">**利用已有的真实偏好标签，约束 Teacher 生成解释，再把相对比较转换成单图评价**</mark> |

这比单纯让 Teacher 自由生成评分更有依据，<mark class="hl-trick">**因为至少有一个外部 Preference Label 作为监督锚点**</mark>。

::: tip 论文做了很关键的消融
直接使用 Qwen3-VL-32B-Instruct 生成监督数据、蒸馏相同的 8B Backbone，在 MMRB2-T2I 上达到 **57.4%**；采用 PARROT 则达到 **64.2%**。

<mark class="hl-key">**这提供了证据，说明改进不仅仅来自「使用更大的 Teacher」。**</mark>
:::

::: danger 但还有两个尚未完全解决的问题
**一是已知答案后生成的理由可能存在事后合理化**；

**二是从 Pairwise 转换成 Pointwise 后，相对偏好正确，并不意味着绝对评分已经校准准确**。

<mark class="hl-key">**论文附录 B.2 自己也承认了 Calibration Drift 和 Context Dependence 这两类风险。**</mark>
:::

---

## 18. Part 3 最终结论

<mark class="hl-key">**PARROT 的真正创新不是训练出了一个会说 Reasoning 的 VLM，而是建立了「已有偏好标签 → 有依据的比较理由 → 单图评分监督」的数据生产链路。**</mark>

$$
\underbrace{\text{已有偏好标签}}_{\text{可靠性锚点}}
\;\longrightarrow\;
\underbrace{\text{有依据的比较理由}}_{\text{Teacher 定向解释}}
\;\longrightarrow\;
\underbrace{\text{单图评分监督}}_{\text{可蒸馏、可用于 RL}}
$$

::: tip 但这条链路的每一环都不是「验证正确性」
- **Phase 1** 是<mark class="hl-trick">**事后合理化风险最高的一环**</mark>；
- **Phase 2** 只能筛掉<mark class="hl-trick">**推不回标签的理由**</mark>，筛不掉<mark class="hl-trick">**视觉依据为假但恰好支持标签的理由**</mark>；
- **Phase 3** 把相对正确<mark class="hl-trick">**悄悄转成绝对分数，而这一步的校准漂移论文自己承认没有解决**</mark>。

<mark class="hl-key">**换句话说，PARROT 把「验证理由是否正确」这个难题，降级成了「验证理由是否一致」这个容易得多但也更弱的问题。**</mark>
:::

**下一部分 Part 4：** 我们就进入真正的 Reward Model 训练——Qwen3-VL-8B 的 SFT 数据格式、Pairwise 与 Pointwise 如何混合、训练 Loss、LoRA 和实际开源代码。

---

# Part 4：RationalRewards-8B 的 SFT 训练

> **对应原文**：SFT 目标 + 官方开源训练配置（LLaMA-Factory）

这一部分最重要的结论是：<mark class="hl-key">**RationalRewards 没有使用 Bradley–Terry Loss，也没有对 Reward Model 本身进行 GRPO。它主要通过一次 SFT，让 Qwen3-VL-8B 学会输出评价理由、偏好判断和多维评分。**</mark>

而且查官方训练代码时发现了一个重要细节：<mark class="hl-trick">**仓库当前公开的 SFT 配置默认使用 Full Fine-tuning，而不是 LoRA。这一点要和后续图像基模使用 LoRA + DiffusionNFT 区分开。**</mark>

---

## 19. 先搞清楚到底训练了什么

作者使用两类数据共同训练同一个 Qwen3-VL-8B：

$$
\begin{aligned}
&\textbf{Qwen3-VL-8B-Instruct} \\
&\quad \bullet\ \textbf{Pairwise SFT} \\
&\qquad (\text{Prompt} + \text{图片 A/B}) \Rightarrow \text{学习比较理由、偏好判断} \\
&\quad \bullet\ \textbf{Pointwise SFT} \\
&\qquad (\text{Prompt} + \text{单张图片}) \Rightarrow \text{学习评价理由、各维度评分} \\
&\quad \downarrow\ \textbf{联合 SFT：Token-level Cross Entropy} \\
&\quad \downarrow \\
&\textbf{RationalRewards-8B} \\
&\qquad \text{既能比较两张图片，又能对单张图片独立评分}
\end{aligned}
$$

::: danger 这里有一个关键区别
<mark class="hl-key">**训练用的 Pairwise 标签，不是 RL 阶段的 Reward。它只是用于制作模型应该生成的答案。**</mark>

真正用于后续 DiffusionNFT 的，<mark class="hl-trick">**是训练完成后 Pointwise 接口输出的数值评分**</mark>。
:::

---

## 20. 两种数据到底怎么送入模型？

继续用前面的例子：

> Prompt：一只红色的猫站在蓝色汽车左侧。

现有偏好标签告诉我们，Image A 比 Image B 更好。经过 PARROT，Teacher 已经为两种样本生成了训练答案。

### Pairwise SFT：学习比较图片

**Student 输入**：

```text
User Prompt: A red cat stands to the left of a blue car.

Image A: <image>
Image B: <image>

Compare these images and explain their relative quality.
```

**Student 学习的目标输出**：

```text
Text Faithfulness:
Image A correctly places the cat to the left.
Image B has an incorrect spatial relationship.

Physical Quality:
Both images are reasonably natural.

Overall Preference: A is better than B.
```

::: warning 这个格式是简化示例
<mark class="hl-key">**并非官方 JSON 原文**</mark>。重点是：<mark class="hl-trick">**Student 输入不包含「A 更好」这个答案，而训练目标中包含正确的比较理由与偏好结果**</mark>。
:::

### Pointwise SFT：学习单图评分

**Student 输入**：

```text
User Prompt: A red cat stands to the left of a blue car.

Generated Image: <image>

Evaluate this image independently.
```

**Student 学习的目标输出**：

```text
Text Faithfulness:
The cat, car and relative positions match the prompt.
Score: 4.0

Physical and Visual Quality:
Natural lighting, with minor visual imperfections.
Score: 3.5

Text Rendering: N/A

Suggested improvement:
Improve fine-grained surface details.
```

同样是示意格式。<mark class="hl-key">**实际训练数据采用包含 `messages` 和 `images` 字段的 ShareGPT 风格多模态对话格式。**</mark>

::: tip 这两种数据最关键的区别
<mark class="hl-key">**Pairwise 让模型学会区分相对质量，Pointwise 让模型能够脱离比较对象，独立给一张图片评分。**</mark>
:::

论文明确提出联合训练：

$$
\mathcal{D}_{\mathrm{train}} = \mathcal{D}_{\mathrm{pair}} \cup \mathcal{D}_{\mathrm{point}}
$$

<mark class="hl-trick">**公开的 LLaMA-Factory 配置也确实在同一个 `dataset` 字段中列出了 Pairwise 和 Pointwise 数据集。**</mark>

---

## 21. Loss 怎么计算？需不需要 BT Loss？

<mark class="hl-key">**不需要。**</mark>

论文给出的核心目标是：

$$
\boxed{
\mathcal{L}_{\mathrm{SFT}} = -\mathbb{E}_{(x,y,z)\sim\mathcal{D}_{\mathrm{train}}} \log P_{\theta}(z, y \mid x)
}
$$

这里 $x$ 是输入图片及任务条件，$z$ 是评价理由，$y$ 是偏好判断或对应的评分结果。

具体到语言模型实现，本质上就是<mark class="hl-key">**逐 Token 的 Cross-Entropy**</mark>：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_{\theta}\bigl(a_t \mid x, a_{<t}\bigr)
$$

其中 $a_t$ 是目标回答中的 Token。

例如 Teacher 的目标回答是 `Image A is better because ...`，训练时就通过 <mark class="hl-trick">**Teacher Forcing**</mark>，让 Student 学会逐 Token 生成这句话，<mark class="hl-trick">**包括评价理由和最终结论**</mark>。

::: warning 分数也是以文本形式被预测的
<mark class="hl-key">**不需要单独增加一个 Scalar Reward Head。**</mark>这和 HPSv3、Arena 中的回归式 Preference RM 是两种不同架构。
:::

::: tip 再强调一次
论文虽通过 ELBO 解释 PARROT，但<mark class="hl-key">**实际 Student SFT 阶段并没有要求另行实现一个 KL Loss。它使用的就是上述标准 SFT 目标。**</mark>
:::

---

## 22. 官方开源的 SFT 训练配置

<mark class="hl-key">**这里直接以作者 GitHub 当前公开的配置为准。**</mark>

源文件：[`rationalrewards_full_sft.yaml`](https://github.com/TIGER-AI-Lab/RationalRewards/blob/main/rationalrewards_sft/examples/train_full/rationalrewards_full_sft.yaml)

| 参数 | 官方代码中的配置 |
| :--- | :--- |
| **Backbone** | Qwen3-VL-8B-Instruct |
| **训练框架** | LLaMA-Factory |
| **Fine-tuning** | <mark class="hl-trick">**Full Fine-tuning**</mark> |
| **Vision Tower** | Frozen |
| **Multimodal Projector** | Frozen |
| **DeepSpeed** | ZeRO-3 |
| **Precision** | BF16 |
| **Learning Rate** | `1e-5` |
| **Epoch** | `3` |
| **Per-device Batch Size** | `4` |
| **Gradient Accumulation** | `4` |
| **LR Scheduler** | Cosine |
| **Warmup Ratio** | `0.05` |
| **Max Sequence Length** | `32768` |
| **Image Max Pixels** | `802816` |
| **GPU 数量** | <mark class="hl-key">**配置未明确指定**</mark> |

::: warning 值得特别提醒的是
<mark class="hl-key">**这些是仓库当前提供的训练配置，论文正文没有完整列出 Reward Model SFT 的全部超参数。因此不能保证公开配置与作者最终实验使用的配置在每个细节上完全一致。**</mark>

而且，代码中的 `finetuning_type: full` 表示默认执行全参数微调，<mark class="hl-trick">**LoRA 相关配置只是被注释的可选项**</mark>，<mark class="hl-key">**不能把后续 DiffusionNFT 的 LoRA Rank 64 误当成 Reward Model 的 SFT 参数**</mark>。
:::

如果只想复现 SFT，官方入口是：

```bash
bash rationalrewards_sft/run_sft.sh
```

<mark class="hl-trick">**运行前需要配置模型路径、数据目录、缓存和输出目录。**</mark>

---

## 23. 还有一个需要认真核对的数据规模问题

<mark class="hl-key">**仓库的数据注册表与论文所描述的原始样本统计口径并不完全对应。**</mark>

论文描述的是约 <mark class="hl-trick">**80K 原始 Pairwise 数据，过滤后约 57.6K 对**</mark>；但公开 SFT 配置包括：

```text
t2i_pairwisev4all_90k
editreward_pairwisev4_42k
editpica_pairwisev4_33k
```

等**六个数据集标识**，而 Hugging Face 发布的 SFT 数据集还显示<mark class="hl-key">**约 348K 条记录**</mark>。

::: danger 这对复现很重要
<mark class="hl-key">**这些数据名称、记录数量与论文中的原始 Pair 数量不在同一统计层级，可能涉及数据扩展、多种训练格式或不同实验版本。**</mark>

<mark class="hl-trick">**目前无法仅凭公开说明准确还原每种样本的实际混合比例，也不能简单把文件名中的 `90k` 当成最终训练样本数。**</mark>

特别是可能关心的 <mark class="hl-key">**Pairwise 和 Pointwise 的训练比例**</mark>——<mark class="hl-trick">**论文没有给出清晰的比例消融**</mark>。
:::

---

## 24. SFT 到底带来了什么效果？

论文有一组非常有参考价值的对照：<mark class="hl-key">**用同样的 8B Backbone 和同等数据量，比较普通 Teacher 蒸馏与 PARROT 蒸馏**</mark>。

| 训练方法 | MMRB2-T2I | GenAI-Bench T2I |
| :--- | ---: | ---: |
| 直接蒸馏 Qwen3-VL-32B | 57.4% | 59.3% |
| **PARROT → Qwen3-VL-8B** | **64.2%** | **69.8%** |
| **差值** | **+6.8** | **+10.5** |

::: tip 这组结果说明什么
在作者的实验条件下，<mark class="hl-key">**不是简单使用更强的 Teacher 就能得到好的 Reward Model；Teacher 生产什么样的监督数据同样重要**</mark>。
:::

::: danger 但它没有单独证明两件事
1. <mark class="hl-key">**Pairwise SFT 与 Pointwise SFT 分别贡献多少**</mark>；
2. <mark class="hl-trick">**不能仅据此断言「生成 Reasoning 本身就能避免 Reward Hacking」**</mark>——那是论文另一组实验的结论。
:::

---

## 25. Part 4 总结

RationalRewards 的 Reward Model 训练，可以浓缩为：

$$
\boxed{
\begin{gathered}
\textbf{Pairwise Preference} \\
\downarrow\ \textbf{PARROT} \\
\textbf{Pairwise + Pointwise Reasoning Data} \\
\downarrow\ \textbf{SFT Cross-Entropy} \\
\textbf{Qwen3-VL-8B Reward Model}
\end{gathered}
}
$$

<mark class="hl-key">**整套方法没有专门设计复杂的 Reward Loss，真正有技术含量的地方在于 PARROT 制作出来的 SFT 监督，而不是 SFT 算法本身。**</mark>

::: danger 这里还有两个容易踩的坑
1. <mark class="hl-key">**Reward Model 的 SFT 是 Full FT，不是 LoRA**</mark>——而后续图像基模的 DiffusionNFT 才是 LoRA Rank 64，两套配置的 rank 含义完全不同；
2. <mark class="hl-trick">**分数是以文本形式被逐 Token 预测的，没有独立回归头**</mark>，因此它天然没有 BT/回归式 RM 的标量校准保证。
:::

::: tip 还有一个值得保留的研究疑问
Pairwise 监督明明只规定 $A$ 比 $B$ 好，<mark class="hl-key">**经过 Pointwise Projection 后，两张图各自得到的 1–4 分到底有多准确？**</mark>

这就涉及 <mark class="hl-trick">**Reward Calibration 和在线 RL 的有效性**</mark>。

**下一部分 Part 5：Reward 推理与打分。** 我们会重点研究四个评分维度的完整 1–4 分定义、如何处理 N/A、不同维度怎么融合，以及为什么这种数值设计可能直接影响 DiffusionNFT 的训练结果。

---

# Part 5：RationalRewards 如何给图片打分，并转换成 DiffusionNFT Reward？

> **对应原文**：附录 D（评分标准）、附录 E.1（Reward 聚合）、Algorithm 1

这一部分直接关系到后训练。已知 RationalRewards 如何制作数据、如何 SFT，现在需要回答：<mark class="hl-key">**给定一条 Prompt 和一张 Rollout 图片，RationalRewards 究竟输出什么？这些分数怎样变成 DiffusionNFT 能使用的 Reward？**</mark>

先说结论：<mark class="hl-key">**RationalRewards 不是直接预测一个 Scalar，也不是像 TRM 那样先生成动态 Rubric。它使用固定的多维评分标准，由 VLM 生成评价理由和各维度分数，再通过程序合成为单个 Reward。**</mark>

---

## 26. 模型的输入输出

对于文生图，输入非常简单：

$$
(\text{Prompt},\ \text{Generated Image}) \longrightarrow \text{RationalRewards-8B}
$$

输出包含三个评价维度：

| 维度 | 评价内容 | 分数 |
| :--- | :--- | :--- |
| **Text Faithfulness** | 主体、属性、数量、动作、关系是否符合 Prompt | 1–4 |
| **Physical and Visual Quality** | 物理合理性、光照、纹理、透视、人体结构、伪影 | 1–4 |
| **Text Rendering** | 文字是否正确、清晰、完整 | 1–4 或 N/A |

对于图像编辑，<mark class="hl-key">**另外增加 Image Faithfulness，判断非编辑区域和原图信息是否得到保留**</mark>。

::: warning 两个容易搞混的边界
- <mark class="hl-key">**文生图没有 Image Faithfulness 这一评分维度**</mark>；
- <mark class="hl-trick">**这和 TRM 的三个维度也不完全相同，RationalRewards 没有一个独立的 Aesthetics 分数**</mark>。
:::

### 真实推理格式

官方 GitHub 的 Pointwise 推理模板采用如下结构，下面只保留关键字段：

```text
# User Request Analysis
...

# Detailed Judgement

1. Text Faithfulness:
## Justification: ...
## Score: 3.5

2. Physical and Visual Quality:
## Justification: ...
## Score: 3.0

3. Text Rendering:
## Justification: ...
## Score: N/A

# Summary:
...

# User Request Refinement:
## Refinement Comments: ...
## Refined Request: ...
```

<mark class="hl-key">**这里的分数是举例；格式来自官方 README。**</mark>模型甚至可以同时提出改进 Prompt 的建议，但 <mark class="hl-trick">**NFT 训练实际只使用解析得到的数值评分**</mark>。

---

## 27. 每个维度具体怎么评分？

作者在附录 D 给出了完整评分标准：

| 分数 | Text Faithfulness | Physical and Visual Quality | Text Rendering |
| :--- | :--- | :--- | :--- |
| **4** | <mark class="hl-key">**核心要求完全符合，没有多余或错误内容**</mark> | <mark class="hl-key">**光影、结构、几何自然，没有明显伪影**</mark> | <mark class="hl-key">**文字准确清晰，融入画面**</mark> |
| **3** | 主要要求满足，少量细节错误 | 有轻微的纹理、光照等问题 | 少量拼写、大小写问题 |
| **2** | 部分主体、属性或关系错误 | 明显透视、阴影、结构或视觉缺陷 | 明显错字或文字变形 |
| **1** | 核心要求缺失或被违背 | 严重畸形、不可能的几何结构或伪影 | 文字无法识别或完全缺失 |

::: danger 这里存在一个很关键的设计
<mark class="hl-key">**1、2、3、4 是评分锚点，而不是只能输出四个离散值。**</mark>

模型实际可以输出 <mark class="hl-trick">**3.2、3.6、3.85 等浮点分数**</mark>。

官方 Prompt 规定了两个步骤：<mark class="hl-key">**先根据整体表现选择一个整数锚点，再通过具体视觉问题进行加减调整**</mark>。

例如图片整体不错，但人物手指稍微变形，那么 Physical Quality 不一定直接从 4 掉到 3，<mark class="hl-trick">**可以得到 3.6**</mark>。

这样做的目的就是<mark class="hl-key">**提高相似质量图片之间的评分分辨率**</mark>。

::: warning 但要区分两件事
<mark class="hl-trick">**论文没有充分证明模型能稳定地区分 3.6 和 3.7。这是评分分辨率与实际评分准确性之间的区别。**</mark>
:::

---

## 28. 用一个例子理解完整评分过程

假设 Prompt 是：

> 一只橘猫站在蓝色汽车左侧，背景是雪山，写实摄影风格。

同一个生成模型 Rollout 两张图片。

| | Image A | Image B |
| :--- | ---: | ---: |
| **Text Faithfulness** | 3.8 | 3.2 |
| **Physical Quality** | 3.6 | 3.0 |
| **Text Rendering** | N/A | N/A |
| **Raw Reward** | **3.70** | **3.10** |

<mark class="hl-key">**为什么 Text Rendering 是 N/A？因为原始 Prompt 没有要求生成文字。**</mark>

### Reward 如何融合？

<mark class="hl-key">**论文 Appendix E.1 中明确规定：所有适用维度等权平均，N/A 不参与计算。**</mark>

$$
\boxed{
R_{\mathrm{raw}}(p, x) = \frac{1}{\bigl|\mathcal{D}_{\mathrm{active}}\bigr|} \sum_{d \in \mathcal{D}_{\mathrm{active}}} s_d(p, x)
}
$$

例如 Image A：

$$
R_A = \frac{3.8 + 3.6}{2} = 3.70
$$

Image B：

$$
R_B = \frac{3.2 + 3.0}{2} = 3.10
$$

如果 Prompt 明确要求文字，例如在海报中写出 `"HELLO"`，那么 Text Rendering 就需要参与评分，<mark class="hl-trick">**分母相应变成 3**</mark>。

::: tip 所以作者没有训练一个额外的网络融合这些 Reward
<mark class="hl-key">**也没有使用自适应权重，而是直接采用 Equal-weight Average。**</mark>
:::

---

## 29. 原始 Reward 如何进入 DiffusionNFT？

<mark class="hl-key">**这里容易混淆两个归一化步骤。**</mark>

### 第一步：把评分维度转换为 Raw Reward

刚才 Image A 得到 $3.70$。

官方公开的推理解析示例还会将 1–4 分转换到 0–1：

$$
R_{01} = \frac{R_{\mathrm{raw}} - 1}{3}
$$

因此：

$$
R_A^{01} = 0.90, \qquad R_B^{01} = 0.70
$$

::: danger 但要注意归因
<mark class="hl-key">**这一步是官方 README 推理解析示例里的实现。论文 Appendix E.1 的算法直接从多维均值出发，没有明确说明训练代码是否采用完全相同的预归一化。**</mark>
:::

### 第二步：在 DiffusionNFT 中进行组内归一化

论文 Algorithm 1 第 7 行给出的形式是：

$$
\boxed{
\widetilde{R}_i = 0.5 + 0.5\,\operatorname{clip}\left(\frac{R_i - \bar{R}_{\mathrm{group}}}{Z_c},\ -1,\ 1\right)
}
$$

其中 $\bar{R}_{\mathrm{group}}$ 是<mark class="hl-key">**同一 Prompt 下 Rollout 的平均 Reward**</mark>；$Z_c$ 是论文使用的归一化尺度。

::: danger 这里有一个必须保留的空白
<mark class="hl-key">**附录没有在此公式附近充分说明 $Z_c$ 的具体计算方式，因此不能直接认定它就是组内标准差。**</mark>
:::

归一化后的 $\widetilde{R}_i \in [0,1]$，才作为 NFT 正负方向的权重。

$$
\begin{aligned}
&\textbf{Prompt} \rightarrow \textbf{16 张 Rollout} \\
&\quad \downarrow \\
&\textbf{RationalRewards-8B 分别评分} \\
&\qquad \text{Faithfulness / Physical Quality / Text Rendering} \\
&\quad \downarrow \\
&\textbf{过滤 N/A} \rightarrow \textbf{适用维度等权平均} \\
&\quad \downarrow \\
&\textbf{Group Normalization} \rightarrow [0, 1] \\
&\quad \downarrow \\
&\textbf{DiffusionNFT Loss} \rightarrow \textbf{更新图像生成模型}
\end{aligned}
$$

![RationalRewards Figure 2(a)：RationalRewards 到 RL 的完整回路。Image Generator 生成图片 → RationalRewards 输出 Rationale（四维带 Justification 与 Score）→ 合成为 Reward 1–4 → 通过 RL Update 箭头回传到 Image Generator。图中明确画出了 Score 与 Reward 之间存在一次「合成」步骤，而不是模型直接吐出一个标量。](/rational-fig2-reasoning-reward-model.png)

::: tip 这里必须记住
<mark class="hl-key">**Reasoning 主要用于帮助 Reward Model 做出评分，而不是作为文本直接参与 DiffusionNFT 的反向传播。**</mark>

论文 Algorithm 1 中，真正进入 NFT Loss 的是归一化数值 Reward：

$$
\mathcal{L}_{\mathrm{NFT}} = \widetilde{R}_i \lVert v_{\theta}^{+} - v \rVert_2^2 + (1 - \widetilde{R}_i)\lVert v_{\theta}^{-} - v \rVert_2^2
$$

<mark class="hl-trick">**更具体的 $v_{\theta}^{+}$、$v_{\theta}^{-}$ 如何构建，Part 6 再拆。**</mark>
:::

---

## 30. 这种评分设计有三个值得注意的问题

::: danger 首先，等权平均不一定合理
例如一张图的 Faithfulness 是 2.0、Physical Quality 是 4.0，最终 Reward 是 3.0；另一张图两个维度均为 3.0，<mark class="hl-trick">**Reward 也是 3.0。它们在 NFT 中得到相同 Raw Reward，但错误类型完全不同**</mark>。

若某个核心 Prompt 要求不可妥协，<mark class="hl-key">**等权平均可能掩盖严重错误**</mark>。
:::

::: danger 其次，Physical and Visual Quality 并不等价于真正的人类审美偏好
它的重点是物理合理性、结构和可见缺陷。<mark class="hl-trick">**虽然官方推理模板也提及构图和视觉质量，但没有单独的美学偏好轴**</mark>。

因此<mark class="hl-key">**自然质感、低饱和度、真实摄影感，未必能被充分优化**</mark>。
:::

::: danger 最后，生成式评分可能不稳定
官方示例使用 Temperature 0.1，<mark class="hl-trick">**虽然较低，但仍不等于完全确定性的 Scalar Head**</mark>。相同图片多次评价，可能出现不同 Reasoning 或小幅分数波动。

此外，<mark class="hl-key">**解析失败时官方 README 示例会给出 0.0 的归一化分数；真正用于训练时需要确认异常样本怎样处理，避免把解析失败错误地当作低质量图片**</mark>。
:::

::: warning 范围声明
这三点是<mark class="hl-key">**从算法设计与公开实现提出的风险，不是论文通过实验已经证明存在的失效现象**</mark>。
:::

---

## 31. Part 5 总结

RationalRewards 的 Reward 设计可以概括为：

$$
\boxed{
\text{Prompt + Image}
\rightarrow \text{Reasoning + Dimension Scores}
\rightarrow \text{Equal-weight Average}
\rightarrow \text{Group Normalization}
\rightarrow \text{NFT}
}
$$

<mark class="hl-key">**它的技术创新主要发生在前半段：让 VLM 学会依据视觉证据进行多维评价，而不是设计了新的 Reward 融合算法。**</mark>

::: tip 值得重点关注的疑问
RationalRewards 的改进究竟来自<mark class="hl-key">**更准确的 Reasoning**</mark>，还是来自<mark class="hl-key">**多维评分及其组合方式**</mark>？

<mark class="hl-trick">**论文没有通过完全解耦的实验充分回答这一点**</mark>。
:::

::: info 把它和 Arena-RL 放在一起看
两者在「最终 Reward 如何进入 NFT」这一步的处理非常接近——<mark class="hl-key">**都是组内去均值 + clip 到 $[0,1]$**</mark>：

| | RationalRewards | Arena-RL |
| :--- | :--- | :--- |
| **分组统计量** | 同 Prompt 的 $K$ 张 rollout 均值 | 同 Prompt 的 $K$ 张 rollout 均值 |
| **缩放常数** | $Z_c$（<mark class="hl-trick">**计算方式未说明**</mark>） | $A_{\max}=5$（<mark class="hl-trick">**明确是截断缩放常数**</mark>） |
| **多奖励合成** | <mark class="hl-trick">**适用维度等权平均**</mark> | <mark class="hl-key">**逐轴 z-score 后等权相加**</mark> |

<mark class="hl-key">**差别恰恰落在「等权平均」和「逐轴标准化」上——这正是 Arena-RL §64 论证过会出问题的地方**</mark>：等权平均没有解决量纲与话语权解耦（这里是 1–4 同量纲所以风险较小），但也没有逐轴标准化带来的分布对齐。
:::

**下一部分 Part 6：** 我们就进入实际训练——作者如何筛选 RL Prompt、16 张 Rollout 怎么组织、为什么选择 16 张 A100、NFT 的正负速度场怎样计算，以及论文中的 Mean / Std 过滤阈值究竟是怎么用的。

---

# Part 6：RationalRewards 如何进行 DiffusionNFT 后训练？

> **对应原文**：附录 E.1 RL Fine-Tuning Setup、Table 11、Algorithm 1 + 官方开源训练代码

最重要的结论是：<mark class="hl-key">**RationalRewards 没有提出新的图像 RL 算法，而是把训练好的 8B Reward Model 接入 DiffusionNFT，通过在线生成、评分、归一化和 Flow Matching Loss 更新图像生成模型。**</mark>

核对官方代码还发现一个复现时必须注意的差异：<mark class="hl-trick">**论文 Table 11 写每个 Prompt 生成 16 张图片，但当前公开的 FLUX-Kontext 和 Qwen-Image-Edit 配置都设置为 8 张。因此不能直接认为下载代码后运行的就是论文表格中的完整实验配置。**</mark>

---

## 32. RL Prompt 如何构建？这里有一个容易忽略的筛选步骤

<mark class="hl-key">**作者没有简单地从数据集中随机选取所有 Prompt，而是从 HPDv3 和 EditReward 中选择当前生成模型比较不擅长的任务。**</mark>

具体来说，先让基础生成模型对候选 Prompt 生成图片，再用 RationalRewards 评分，<mark class="hl-trick">**优先保留初始生成的平均分数低于 3.0（原始 1–4 分制）的 Prompt**</mark>：

$$
\boxed{
\bar{R}_{\mathrm{initial}}(p) < 3.0
}
$$

例如，一条 Prompt 的初始图片评分均值为 2.6，那么可能进入 RL 训练池；另一条平均为 3.7，则不在这一筛选条件之内。

::: tip 这样做的目的
让训练资源<mark class="hl-key">**集中在当前模型还有改善空间的 Prompt 上，而不是反复训练模型已经做得很好的任务**</mark>。
:::

::: danger 但论文没有披露
<mark class="hl-key">**候选 Prompt 总数、每条 Prompt 初始生成多少张图片，以及最终筛选后的训练池有多大**</mark>。

因此<mark class="hl-trick">**不能把这个流程还原成一套完全确定的数据生产参数**</mark>。
:::

::: tip 这一步特别值得关注
<mark class="hl-key">**它相当于在训练前做了一轮基于 Reward 的数据选择，而不是只在 Rollout 之后使用 Reward。**</mark>
:::

---

## 33. 一次训练迭代具体在做什么？

论文 Algorithm 1 可以还原成下面这个流程：

$$
\begin{aligned}
&\textbf{Step 1} \quad \text{从 RL Prompt 池采样} \\
&\quad \downarrow \\
&\textbf{Step 2} \quad \text{当前模型 Rollout（论文配置：每条 Prompt 生成 } K = 16 \text{ 张图片）} \\
&\quad \downarrow \\
&\textbf{Step 3} \quad \text{RationalRewards 打分（每张图片的适用维度等权平均 → Scalar）} \\
&\quad \downarrow \\
&\textbf{Step 4} \quad \text{Reward 归一化（组内去均值、按尺度归一化，得到 NFT 权重）} \\
&\quad \downarrow \\
&\textbf{Step 5} \quad \text{Forward Flow Matching（Clean Latent + Noise → NFT 正负速度场损失）} \\
&\quad \downarrow \\
&\textbf{Step 6} \quad \text{更新 LoRA，并通过 EMA 更新采样模型}
\end{aligned}
$$

<mark class="hl-key">**和 FlowGRPO 不同，DiffusionNFT 在训练阶段重新对生成的 Clean Latent 加噪，计算 Flow Matching 目标，不依赖反向采样轨迹的 Log Probability 来更新策略。**</mark>

::: tip 一个容易忽略的工程细节
官方训练代码<mark class="hl-key">**把 Reward 调用提交给线程池，先收集 Future，后面统一等待结果，并将不同 GPU 的评分汇总后计算 Advantage**</mark>。

这相当于<mark class="hl-trick">**让图片生成与部分 Reward 计算能够重叠，减少等待时间**</mark>。
:::

---

## 34. Reward 如何转换成 NFT 的训练权重？

假设同一条 Prompt 生成四张图片，RationalRewards 给出这些综合评分：

| 图片 | Raw Reward（1–4） | 相对组均值 |
| :--- | ---: | ---: |
| A | 3.6 | +0.6 |
| B | 3.2 | +0.2 |
| C | 2.8 | $-0.2$ |
| D | 2.4 | $-0.6$ |
| **均值** | **3.0** | **0** |

<mark class="hl-key">**以上是演示算法的假设数据，不是论文真实结果。**</mark>

论文 Algorithm 1 的归一化形式为：

$$
\widetilde{r}_i = 0.5 + 0.5\,\operatorname{clip}\left(\frac{r_i - \bar{r}_{\mathrm{group}}}{Z_c},\ -1,\ 1\right)
$$

因此，<mark class="hl-trick">**得分高于组均值的图片，NFT 权重趋向大于 0.5；低于组均值的图片则趋向小于 0.5**</mark>。

::: tip 与 Arena-RL 的相似之处
<mark class="hl-key">**都是通过相对评分将一组图片转成优化权重。**</mark>而进一步查当前公开的 `PerPromptStatTracker`，其默认配置<mark class="hl-trick">**使用 Prompt 级均值与当前收集批次的 Global Standard Deviation，而不是每条 Prompt 各自的标准差**</mark>——<mark class="hl-key">**这与 Arena-RL 公式 (9) 的 Group Mean + Batch Std 非常接近**</mark>。
:::

---

## 35. NFT 到底怎么更新图像模型？

这是 Part 6 最重要的公式。

对于一张 Rollout 图片的 Clean Latent $x_0$，随机采样时间 $t$ 和高斯噪声 $\epsilon$，构造：

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon
$$

对应的 Flow Matching 目标速度：

$$
v^{*} = \alpha'_t x_0 + \sigma'_t \epsilon
$$

NFT 再构造两个隐式速度场：

$$
v^{+} = (1 - \beta)\, v_{\mathrm{old}} + \beta\, v_{\theta}
$$

$$
v^{-} = (1 + \beta)\, v_{\mathrm{old}} - \beta\, v_{\theta}
$$

其中 <mark class="hl-key">$v_{\mathrm{old}}$ 是采样策略的冻结快照，$v_{\theta}$ 是当前正在更新的模型**</mark>。

最终 Loss：

$$
\boxed{
\mathcal{L}_{\mathrm{NFT}} = \widetilde{r}\lVert v^{+} - v^{*}\rVert^2 + (1 - \widetilde{r})\lVert v^{-} - v^{*}\rVert^2
}
$$

::: tip 理解起来并不复杂
<mark class="hl-key">**高分图片增大第一项的权重，引导当前模型更接近这些图片；低分图片增大第二项的权重，通过隐式负方向减少模型对它们的偏好**</mark>。
:::

::: danger 但有两个必须点明的细节
**第一**，<mark class="hl-trick">**这里的两项 Loss 并不是做两次独立 backward。它们组成同一个总损失，再统一反向传播更新当前模型的 LoRA 参数**</mark>。

**第二**，<mark class="hl-key">**代码中还额外实现了预测误差的自适应权重和参考模型正则项，这比论文 Algorithm 1 展示的简化公式更多一些**</mark>。
:::

---

## 36. 两个过滤阈值：Mean 0.9 和 Std 0.05 到底是什么意思？

这是这一部分最值得研究的工程 Trick。

论文 Table 11 给出：

$$
\text{Mean Threshold} = 0.9, \qquad \text{Std Threshold} = 0.05
$$

<mark class="hl-key">**它们不是最开始筛选 RL Prompt 的平均分小于 3.0，而是在线 Rollout 时的质量过滤参数。**</mark>

从算法意图来看，主要想避免两种不适合继续训练的情况：

| 情况 | 例子 | 理由 |
| :--- | :--- | :--- |
| **整组图片已经很好** | Reward 均值达到 0.95 | <mark class="hl-trick">**可能没有太多提升空间**</mark> |
| **同组图片分数太接近** | 标准差只有 0.01 | <mark class="hl-trick">**难以产生可靠的组内排序信号**</mark> |

对于在线 Rollout，一个合理的过滤条件可以写成：

$$
\boxed{
\operatorname{Keep}(p) = \bigl[\mu_p < 0.9\bigr] \land \bigl[\sigma_p > 0.05\bigr]
}
$$

| Prompt | Reward Mean | Reward Std | 按该规则 |
| :--- | ---: | ---: | :---: |
| A | 0.65 | 0.12 | **保留** |
| B | 0.95 | 0.10 | 过滤 |
| C | 0.65 | 0.02 | 过滤 |
| D | 0.94 | 0.01 | 过滤 |

::: warning 但注意
<mark class="hl-key">**这是对阈值设计目的的解释，不能直接视为作者全部代码都严格按该公式实现。**</mark>
:::

### 核对实际实现后发现一个很重要的差异

<mark class="hl-trick">**当前 `PerPromptStatTracker` 确实会计算每条 Prompt 的 `prompt_std` 和 `prompt_mean`，并依据「低方差且高均值」记录需要跳过的 Prompt；但同一份代码启用 `global_std=True` 时，返回给训练循环的 `stds` 是整个批次的标准差。**</mark>

训练循环又使用：

```python
valid_mask = (stds > 0.05) & (means < 0.9)
```

因此，<mark class="hl-key">**当前代码里的 `valid_mask` 并不严格等价于每条 Prompt 的组内方差过滤；当使用 Global Std 时，`stds > 0.05` 比较的是全局尺度**</mark>。

另有 `banned_prompts` 机制负责处理低组内方差且高均值的 Prompt。

![RationalRewards Figure 11：四种配置的训练曲线对照（4 行 × 3 列：Train reward / Reward Std / Eval reward）。从上到下分别是 Qwen-Edit w/ EditReward（scalar RM）、Qwen-Image w/ MultiReward（scalar RM）、Qwen-Edit w/ RationalRewards、Qwen-Image w/ RationalRewards。注意中间列 Reward Std：MultiReward 那行的方差长期贴着 0.05–0.06 波动，正好压在 0.05 阈值附近；两个 RationalRewards 配置的方差分别稳定在约 0.075–0.2 与 0.05–0.15 区间。最上行的 Eval reward 在 step≈200 处出现明显下跌后回升，对应训练中期的退化。](/rational-fig11-training-curves.png)

::: tip 从 Figure 11 能读出的一个细节
中间列的 Reward Std 曲线<mark class="hl-trick">**直接可视化了 0.05 这个阈值在起作用**</mark>：

- <mark class="hl-key">**MultiReward 那一行的方差长期贴在 $0.05 \sim 0.06$ 附近徘徊**</mark>——<mark class="hl-trick">**也就是说，如果按全局标准差过滤，这个配置会有大量 step 处在阈值边缘，训练是否真的被 `valid_mask` 大量筛掉，取决于实现是按 batch 还是按 prompt 统计**</mark>；
- 两个 RationalRewards 配置的方差则分别稳定在明显更高的区间，<mark class="hl-key">**过滤阈值对它们的约束弱得多**</mark>。
:::

### 所以复现时不要只复制阈值

<mark class="hl-key">**建议同时检查三件事：**</mark>

| 检查项 | 为什么重要 |
| :--- | :--- |
| **标准差到底按 Group 计算，还是按 Batch 计算** | <mark class="hl-trick">**这决定了 `stds > 0.05` 到底在过滤什么**</mark> |
| **被过滤的 Prompt 是完全跳过后续 Rollout，还是仅在当前更新中将 Loss Mask 为 0** | <mark class="hl-key">**两者对数据分布的影响完全不同**</mark> |
| **低均值、低方差的 Prompt 应不应该保留** | <mark class="hl-trick">**因为可能代表当前模型始终无法完成的困难任务**</mark> |

::: danger 特别是第三点
<mark class="hl-key">**低方差不一定说明任务没有训练价值，也可能说明所有生成结果都失败了。把这些 Prompt 全部删除可能造成困难任务长期学不到。**</mark>
:::

---

## 37. 作者具体用了多少卡？训练参数是什么？

论文附录 Table 11 的配置：

| 参数 | 论文报告 |
| :--- | :--- |
| **图像 RL 算法** | DiffusionNFT |
| **训练 GPU** | 8 × A100-80GB |
| **Reward GPU** | 8 × A100-80GB |
| **总 GPU** | **16 × A100-80GB** |
| **每 Prompt Rollout** | **16** |
| **训练分辨率** | 512 × 512 |
| **训练采样步数** | 15 |
| **测试采样步数** | 20 |
| **Noise Level** | 0.7 |
| **Learning Rate** | 2e-4 |
| **LoRA Rank / Alpha** | 64 / 128 |
| **Per-GPU Batch Size** | 8 |
| **Mean / Std 过滤阈值** | 0.9 / 0.05 |
| **β（表格称 Guidance Strength）** | 1e-4 |

### 但这里有一个很有价值的论文与代码一致性问题

| 配置项 | 当前代码 |
| :--- | :--- |
| **每 Prompt Rollout** | <mark class="hl-trick">**8 张（不是论文的 16）**</mark> |
| **FLUX-Kontext 采样步数** | 15 |
| **Qwen-Image-Edit 采样步数** | 10 |
| **每轮 Prompt Group 数量** | 64 |
| **每轮生成数量** | 64 × 8 = 512 |
| **训练 GPU** | 8 张 |

配置来源：[`flux_kontext_nft.py`](https://github.com/TIGER-AI-Lab/RationalRewards/blob/main/diffusion_rl_training/config/flux_kontext_nft.py)、[`qwen_edit_nft.py`](https://github.com/TIGER-AI-Lab/RationalRewards/blob/main/diffusion_rl_training/config/qwen_edit_nft.py)。

::: warning 范围声明
<mark class="hl-key">**这里只描述这两个公开配置，不代表论文所有 T2I 实验都采用相同设置。**</mark>
:::

::: danger 代码里实际上有两个不同的 beta
- <mark class="hl-key">`config.beta = 1.0`</mark>：用于 <mark class="hl-trick">**NFT 正负速度场的组合**</mark>；
- <mark class="hl-key">`config.train.beta = 0.0001`</mark>：用于训练代码中的<mark class="hl-trick">**参考模型正则项**</mark>。

因此，<mark class="hl-key">**论文表格里标为 Guidance Strength 的 `1e-4` 与当前代码的 NFT `config.beta = 1.0` 不能直接对应**</mark>。这属于<mark class="hl-trick">**复现前需要进一步核实的参数口径问题**</mark>。
:::

::: danger 还有一个复现入口的缺口
官方公开的 `diffusion_rl_training/` README <mark class="hl-key">**目前主要提供 FLUX-Kontext 和 Qwen-Image-Edit 两种启动目标，并未提供论文三个 T2I 基模各自同等完整的启动入口**</mark>。

所以虽然论文报告了 FLUX.1-dev、SD3.5-Medium、Qwen-Image 的结果，<mark class="hl-trick">**但直接运行公开脚本还不能保证完整复现这三组文生图实验**</mark>。
:::

---

## 38. 这套后训练究竟有什么值得学习？

<mark class="hl-key">**这篇论文的 DiffusionNFT 部分并没有特别新的训练算法，更值得学习的是它把三个环节结合起来了：**</mark>

::: tip 第一，训练前按初始 Reward 选择当前基模表现较弱的 Prompt
原始均值低于 3.0，<mark class="hl-trick">**把训练资源集中到困难任务**</mark>。
:::

::: tip 第二，在线 Rollout 后再用平均 Reward 和标准差控制哪些样本参与更新
减少高分饱和和缺乏区分度的样本对梯度的影响。<mark class="hl-trick">**不过具体过滤实现还存在前面提到的代码口径问题**</mark>。
:::

::: tip 第三，用 RationalRewards 的多维评价替换传统 Scalar RM
<mark class="hl-key">**仍然保持 DiffusionNFT 的整体算法不变**</mark>。因此论文中相对于 MultiReward 和通用 Qwen3-VL Judge 的结果可以为 Reward 设计提供参考，<mark class="hl-trick">**但不能完全排除数据选择、训练配置等因素的影响**</mark>。
:::

::: danger 一个自己设计图像 RL 系统时值得特别关注的潜在矛盾
一方面，<mark class="hl-key">**作者优先挑选基础模型平均评分较低的困难 Prompt**</mark>；另一方面，<mark class="hl-key">**在线训练又可能因为 Reward 标准差过低而减少这些困难 Prompt 的贡献**</mark>。

如果一个 Prompt 的所有图片都很差，且分数都接近 0.2，那么<mark class="hl-trick">**它到底应该被过滤，还是应该通过其他方式产生更强的训练信号？这是个值得独立验证的问题**</mark>。
:::

---

## 39. Part 6 总结

$$
\boxed{
\text{困难 Prompt 筛选}
\rightarrow \text{在线 Rollout}
\rightarrow \text{RationalRewards}
\rightarrow \text{组相对 Reward}
\rightarrow \text{DiffusionNFT + LoRA}
}
$$

这一部分最需要记住的数字是：<mark class="hl-key">**论文 16 张 Rollout、8 张训练 A100 + 8 张 Reward A100、512 分辨率、LoRA 64/128，以及 0.9 / 0.05 的在线过滤阈值**</mark>。

::: danger 但如果要直接复现
<mark class="hl-key">**应该以实际训练入口和具体代码为准，不能直接照抄论文 Table 11**</mark>。
:::

::: info 把 Part 5 §29 与本节对照看
RationalRewards 与 Arena-RL 的 NFT 接入路径<mark class="hl-key">**在优化器层面几乎完全一致**</mark>——都是 DiffusionNFT 的 $v^{\pm}$ 构造与两项加权损失。

真正的差别只有一处，但很关键：

| | RationalRewards | Arena-RL |
| :--- | :--- | :--- |
| **优化器** | DiffusionNFT（相同） | DiffusionNFT（相同） |
| **Reward 来源** | <mark class="hl-trick">**一个多维生成式 RM**</mark> | <mark class="hl-key">**偏好 RM + Faithfulness + Constraint + Anti-Hack 四路**</mark> |
| **多维合成** | <mark class="hl-trick">**适用维度等权平均**</mark> | <mark class="hl-key">**逐轴 z-score 后等权相加**</mark> |
| **组内缩放** | <mark class="hl-trick">**$Z_c$，计算方式未说明**</mark> | <mark class="hl-key">**$A_{\max}=5$，明确为截断常数**</mark> |

<mark class="hl-key">**也就是说，这两篇论文唯一真正可比的接口就是「一个标量 Reward 怎么进 NFT」——而这一步两边几乎是同一套。**</mark>
:::

**下一部分 Part 7：实验和消融。** 我们将重点看 RationalRewards 相比 MultiReward、通用 32B Judge 和直接蒸馏 Teacher 的优势究竟来自哪里，并结合 Reward Hacking 曲线判断：<mark class="hl-trick">**这些提升是 Reward 本身更可靠，还是只是更适合 UniGenBench++ 的评测方式**</mark>。
---

## 40. 本笔记与 Reward Hacking 子专题的关系

本篇 **Part 5、Part 6** 的内容（生成式评分的非确定性、解析失败给 0.0 的风险、训练中期的 reward 曲线退化、低方差 rollout 过滤的副作用）已在新建立的 [🚨 RL Reward Hacking 与分布偏移](../reward-hacking/) 里建立索引。

| 本笔记位置 | 归入该专题的哪个问题 |
| :--- | :--- |
| §28、§30 | 攻击面：Temperature 0.1 的非确定性；解析失败示例给 0.0 有被误当低质量图的风险 |
| §36 | 分布侧：两个过滤阈值与「低方差过滤可能把困难任务永久排除」 |
| §36 / Fig.11 | 训练中期的 eval reward 明显下跌后回升（对应退化再恢复） |
