# Qwen-Image-2.0 RLHF 统一对齐 (阿里)

> **标签**：`Vision` `RL` `GRPO` `Image Editing` `Prompt Enhancer`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2605.10730](https://arxiv.org/abs/2605.10730)

---

## 1. 问题定义与控制目标

单模型统一 T2I 生成 + 指令编辑，RL 要同时推质感、图文对齐、人像、编辑遵循、未改区域一致性——单奖励必跷跷板。两处 RL：① Prompt Enhancer 的 GRPO；② 主模型多奖励 GRPO。管线：`Pretrain 700K（9:1）→ Continual 250K（7:3）→ SFT 10K → RLHF → 少步蒸馏`。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：Qwen3-VL 条件编码器、RL 时的图像生成器（PE 训练时冻结 DiT，只更新 PE）；
- **可训练参数**：MMDiT 主干；PE（Qwen3.5-9B 初始化）独立优化；
- **注入机理**：5 个任务专用奖励模型各管一维，校准到可比尺度后动态调权重；PE 侧奖励来自"改写→冻结 DiT 真渲染→打分"的端到端回路。

## 3. 损失函数与数学稳定性推导

Adapted GRPO + **CFG 杂交策略**：rollout 开 CFG（高质量候选→可靠奖励），优化目标去掉 unconditional 分支（省算力）——采样要质量、训练要效率的折中，已成行业惯例（Swift-Image 同构）。

PE 的 GRPO 融合（免手调 scale 的通用技巧）：每个奖励在其 GRPO 组内标准化，转 relative advantages 后融合，不加权 raw scores：
$$
\hat{A}^{(k)}_i = \frac{r^{(k)}_i - \mu^{(k)}}{\sigma^{(k)}}, \quad \hat{A}_i = \sum_k w_k \hat{A}^{(k)}_i
$$

::: info PE 数据构造
"逆向退化"：fine 标注 → LLM 按类别随机退化成口语短 prompt，逆过程即 CoT；SFT 后用 GRPO 对齐下游成图质量，奖励 = MLLM 视觉一致 + MLLM 美学 + 规则文本约束。
:::

## 4. 多奖励权衡

- T2I：美学 / 图文对齐 / 人像（解剖/比例/肤发）；
- TI2I：指令遵循 / 视觉一致性（几何/拓扑/语义守恒）；
- 动态三调：奖励权重 + 跨任务 prompt 分布 + 奖励校准，防单维度过优化。
- 另有 Data Flywheel：坏例按归因自动路由 RL / 预训练补数据 / prompt 工程三轨，RL 与数据闭环联动。

## 5. 核心控制层代码实现

```python
# 多奖励组内标准化融合（PE-GRPO / 主模型通用）
def fused_advantage(reward_dict, weights):
    advs = {}
    for k, r in reward_dict.items():  # r: 组内 G 个候选的原始分
        advs[k] = (r - r.mean()) / (r.std() + 1e-6)
    return sum(weights[k] * advs[k] for k in reward_dict)

# CFG 杂交：rollout 用 guided，优化只算 conditional 分支
# rollout: x ~ p_cfg(x|c)          # 高质量候选
# loss: 只对 conditional 分支求梯度，unconditional 分支 detach/跳过
```

## 6. 避坑指南与评测基准

- **评测**：LMArena + 人评（生成/编辑双轨）；RL 前后定性对比必须同 seed；
- **要点**：统一模型 RL 标配多奖励 + 动态权重；PE 与生成器解耦、各自 RL——prompt 工程从玄学变优化问题；
- Qwen-Image-2.1（7B，ModelScope）延续同范式，细节待公开。
