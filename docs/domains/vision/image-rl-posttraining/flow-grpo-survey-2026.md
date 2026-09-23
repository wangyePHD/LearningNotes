# Flow-GRPO 综述精读 (Advances in GRPO for Generation Models)

> **标签**：`Vision` `RL` `GRPO` `Flow Matching` `Survey`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2603.06623](https://arxiv.org/abs/2603.06623) · 200+ 篇文献，7 方法维度 × 9 任务扩展

---

## 1. 问题定义与控制目标

Flow Matching / Diffusion 模型预训练只学数据分布，输出与人类偏好、任务目标（文字渲染、美学、指令遵循）不对齐。GRPO（LLM 侧：组内相对优势、无 critic）是候选对齐算法，但迁移到视觉生成有三座原生大山：

- **采样贵**：一张图需数十至上百步去噪，online rollout 主导训练时间；
- **无随机性**：rectified flow 给定初噪即确定（ODE），RL 缺探索；
- **奖励稀疏**：奖励只在终点（整图质量分），早期定结构、晚期修细节的步骤被均摊同一 advantage。

Flow-GRPO（首篇）：ODE→SDE 变换引入随机性 + 去噪过程建模为 MDP + denoising-shrinkage 降采样成本，GenEval 文字渲染 63%→95%。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：奖励模型（PickScore / HPS / OCR / VLM evaluator）全程冻结，只提供标量信号；
- **可训练参数**：速度场 / 噪声预测网络全量（或 LoRA），policy 即去噪策略；
- **注入机理**：奖励不进网络前向，只进梯度——经 advantage 加权改变去噪方向的更新量；Euphonium 例外，把过程奖励梯度直接注入 SDE drift：
$$
dx_t = [v_\theta(x_t,t) + \alpha \nabla_{x_t} R_t(x_t)] dt + \sigma(t) dW_t
$$

## 3. 损失函数与数学稳定性推导

GRPO 组相对优势（无 value 网络）：
$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})}, \quad \mathcal{L} = -\mathbb{E}\left[\sum_i \min(\rho_i \hat{A}_i, \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i)\right]
$$

Flow 侧 MDP формулировка：state $s_t=(x_t, c, t)$，action 为 SDE 随机项诱导的去噪方向，单步对数似然：
$$
\log \pi_\theta(a_t|s_t) = -\frac{\|a_t - v_\theta(x_t,t)\Delta t\|^2}{2\sigma^2(t)\Delta t} + \mathrm{const}
$$

::: info 稳定性要点
组内归一化消掉 prompt 难度差异带来的奖励尺度漂移，这是 GRPO 比 PPO 省 critic 又稳的根因；方差完全来自组内相对比较，因此组大小与组内多样性是第一超参。
:::

DenseGRPO 把终点稀疏奖励拆成步级增益（单步 ODE 预测干净图打分）：
$$
\Delta r_t = R(\hat{x}_1^{(t)}) - R(\hat{x}_1^{(t-1)})
$$
并在 $|\Delta r_t|$ 大的关键步自适应加大随机性——探索预算花在刀刃上。

## 4. 对齐—多样性—效率权衡

| 维度 | 代表方法 | 效果 |
| :--- | :--- | :--- |
| 稀疏→稠密 | DenseGRPO / SuperFlow / VGPO / Euphonium | GenEval 0.71→0.74，收敛 1.66× |
| 信用分配 | TreeGRPO / BranchGRPO / PCPO | 2.4× 加速，对齐 +16% |
| 采样效率 | MixGRPO-Flash / DiffusionNFT / AWM / DGPO | 省时 20–25× |
| 多样性 | DiverseGRPO / OSCAR / DRIFT | 抗 mode collapse |
| 防 hacking | GRPO-Guard / GARDO / GDRO | 见 A2 笔记 |
| ODE vs SDE | Neighbor GRPO（纯 ODE 对比） | 去随机性依赖 |

::: warning 避坑要点
终点奖励均摊到所有步会稀释信号：早期结构步与晚期细节步拿相同梯度权重，这是 Flow-GRPO 原版低效的根因；任何改进先问"信用分到步了吗"。
:::

## 5. 核心控制层代码实现

TreeGRPO 式步级优势（同父节点兄弟轨迹对比，控制公共前缀混杂）：
```python
# r_children: 当前分支子树终点奖励均值；r_siblings: 同层兄弟均值
def tree_advantage(r_children, r_siblings, eps=1e-6):
    mu = r_siblings.mean()
    sigma = r_siblings.std() + eps
    return (r_children.mean() - mu) / sigma  # 仅反映分支点决策质量

# DiffusionNFT 式加权（单样本连续标量，无需成对/分组）
def nft_loss(v_theta, v_old, v_target, r, beta=0.5):
    v_pos = (1 - beta) * v_old + beta * v_theta
    v_neg = (1 + beta) * v_old - beta * v_theta
    l_pos = (v_pos - v_target).pow(2).mean()
    l_neg = (v_neg - v_target).pow(2).mean()
    return r * l_pos + (1 - r) * l_neg  # r in [0,1]
```

## 6. 避坑指南与评测基准

- **评测**：PickScore / HPS v2.1 / GenEval / ImageReward；RL 必须双轨：原始奖励 + 独立质量分（见 GDRO corrected score），否则被 hacking 骗；
- **选型**：要效率看 §3.3（NFT/AWM/DGPO 省 20×+）；要归因精度看 §3.2（树/分支）；要抗 hack 看 §3.5（GDRO）；
- 本知识库后续：A2 GDRO、FireRed/Qwen/Swift/SeFi 均为此综述中的具体实例。
