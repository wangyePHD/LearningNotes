# RationalRewards：让奖励模型学会「说出理由」

> **标签**：`Preference Model` `Reasoning Reward` `Pointwise Projection` `Pairwise` `Consistency Filtering` `SFT Distillation` `Qwen3-VL`
> **更新时间**：2026-10-12
> **arXiv**：[2604.11626](https://arxiv.org/html/2604.11626)（2026 年 4 月）
> **标题**：*RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time*
> **模型**：Teacher Qwen3-VL-32B-Instruct → Student **Qwen3-VL-Instruct-8B**
> **原图**：`Papers/RetionalReward.pdf`
> **精读进度**：**Part 1–3 完结**
> **Part 1**：Abstract / §1 / §2 整体框架与核心创新 → §1–§6
> **Part 2**：§3 Training Data / 附录 F.1 → §7–§12
> **Part 3**：§2.1 + 附录 B、C、F，Figure 4 → §13–§18
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