# DeepSeek-R1 推理涌现与强化学习机制

> **标签**：`LLM` `Reasoning` `RL` `GRPO`  
> **更新时间**：2026-09-19

---

## 1. 核心突破：纯 RL 激发慢思考能力 (System 2)

DeepSeek-R1-Zero 证明了：**无需前期大规模人工 SFT 数据微调，仅靠强化学习奖励信号（如数学题目准确率、编译器执行反馈），大模型就能自发涌现长思维链（Chain of Thought）、顿悟（Aha moment）与自我验证反思。**

---

## 2. GRPO (Group Relative Policy Optimization) 数学目标

传统 PPO 需要额外维护一个规模庞大的 Critic（Value）模型来预估状态基线，显存开销巨大。  
GRPO 直接从当前策略中对同一个 Prompt $q$ 采样一组输出 $\{o_1, o_2, \dots, o_G\}$，利用组内归一化奖励作为优势估计（Advantage）：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q), \\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} \hat{A}_i, \, \text{clip}\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]
$$

其中第 $i$ 个回答的标准化优势值定义为：
$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}
$$

::: tip 关键优势
1. **显存减少近 50%**：无需加载 Critic 网络权重与优化器状态。
2. **训练稳定性高**：基于相对排序而非绝对分数，抵抗奖励漂移。
:::
