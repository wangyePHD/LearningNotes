# 🏅 偏好模型 (Preference Model)

> **标签**：`Preference Model` `Reasoning Reward` `BRADLEY-TERRY` `Pointwise` `Pairwise` `SFT Distillation`
> **更新时间**：2026-10-12

本子专题研究 **RL-AIGC 里「谁来打分」这一环**：Reward Model 的结构（scalar vs reasoning-based）、训练数据从哪来、如何把 pairwise 偏好监督变成可蒸馏的监督信号，以及打分接口本身（pointwise / pairwise）如何影响它能否接入 RL。

上游见 [🎲 AIGC 强化学习后训练](../index.md)。注意本子专题与 Arena-RL 笔记里的 Preference RM 部分**不是重复**：Arena-RL 关心的是「偏好 RM 在整套复合奖励里怎么用」，本子专题关心的是「偏好 RM 本身怎么造」。

---

## 收录标准

| 判据 | 说明 |
| :--- | :--- |
| ✅ 收录 | Reward Model 的**结构设计**（是否输出 reasoning、多维分数、可解释性） |
| ✅ 收录 | 偏好监督数据 → 可训练 RM 监督信号的**转换与蒸馏管线** |
| ✅ 收录 | RM 的评测（与人类偏好一致性、与通用 VLM Judge 的区分） |
| ✅ 收录 | RM 作为 RL Reward 时的**接口约定**（pointwise vs pairwise、分数合成方式） |
| ❌ 不收 | 只讨论「如何把已有 RM 组合进多奖励」（归 Arena-RL 笔记） |
| ❌ 不收 | 只讨论 RL 算法本身（归 [图像基模 RL 后训练](../../vision/image-rl-posttraining/) 算法篇） |

---

## 论文篇

| 篇目 | 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **RationalRewards：让奖励模型学会「说出理由」** | **不设计新 RL 算法，只改 Reward Model 的数据生产方式**。Teacher (Qwen3-VL-32B-Instruct) 在**已知偏好标签**的条件下解释「为什么 A 更好」→ 隐藏标签做**一致性过滤**（约 72% 通过）→ **Pointwise Projection** 把成对理由投影成单图评价 → SFT 蒸馏到 **Qwen3-VL-8B** Student。输出四维评分 + 文字理由（Text Faithfulness / Image Faithfulness / Physical & Visual Quality / Text Rendering）。FLUX.1-dev 60.97→70.34、SD3.5-Medium 60.71→70.56、Qwen-Image 78.36→82.60（UniGenBench++ Overall）。**Part 3**：PARROT 三阶段全拆 —— Phase 2 的判据只是「理由能否推回标签」而非「理由是否真实」（前后两次共用同一 Teacher 存在共享偏差）；Phase 3 把相对偏好转成绝对评分，论文附录 B.2 自承 Calibration Drift / Context Dependence 未解；消融显示同 Teacher 直接蒸馏仅 57.4%、PARROT 达 64.2%。**Part 4**：**无 BT Loss、无 GRPO**，只有一次 Token-level SFT；官方 `rationalrewards_full_sft.yaml` 用 **LLaMA-Factory + Full Fine-tuning**（Vision Tower / Projector 冻结、ZeRO-3、BF16、LR 1e-5、3 epoch、seq 32768、img max px 802816）——**与后续 DiffusionNFT 的 LoRA Rank 64 是两回事**；分数以文本逐 Token 预测，**无独立 Scalar Head**；仓库六个数据集标识（`t2i_pairwisev4all_90k` 等）与论文 80K→57.6K 口径不对应，Pairwise:Pointwise 混合比例论文未给消融。**Part 5**：固定四维评分（非动态 Rubric）、**适用维度等权平均**（N/A 不入分母）、1–4 只是锚点可出浮点；`$\widetilde R=0.5+0.5\,\mathrm{clip}((R-\bar R_{group})/Z_c,-1,1)$` 但 **$Z_c$ 的计算方式未说明**；分数无独立 Scalar Head 且 Temperature 0.1 非确定性，**解析失败示例给 0.0 有被误当低质量图的风险** | 🔄 Part 1–5 完结 | [阅读笔记 →](./rational-rewards.md) |

---

## 待补主线（阅读路线图）

1. **Part 2：原始偏好数据究竟怎么构建** —— 50K T2I（HPDv3 / RapidData）+ 30K Editing（EditReward）的来源、图片对组成、Preference Label 获取方式、不同质量图片的分布、过滤细节的披露程度
2. **Part 3：事后合理化的根本问题** —— Teacher 已知答案时，生成的 Reasoning 究竟是在**发现**质量差异，还是在**事后寻找理由**解释已有答案
3. **与 Arena-RL 的 Reward 体系对照** —— 偏好 RM / Rubric Judge / Graded OCR 三类评分器的接口差异与组合可行性