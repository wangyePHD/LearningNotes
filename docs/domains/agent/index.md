# 🤖 智能体系统与工具编排 (Agent Systems)

本模块从**零框架、零云端 API** 的最小实现出发，把「智能体 (Agent)」还原为一个可机械审计的确定性系统：**循环 (Loop) + 状态 (State) + 闭集动作空间 (Closed Action Space) + 契约校验 (Contract Validation)**。

核心立场：**智能体不是模型，也不是一种性格，而是一段 Python 控制流。模型只负责在闭集上做选择；可靠性来自外壳的约束，而不来自更巧妙的提示词。**

- 母本仓库：[pguso/agents-from-scratch](https://github.com/pguso/agents-from-scratch)（MIT，12 课渐进式，llama.cpp 本地推理）
- 配套仓库：JavaScript 版 [pguso/ai-agents-from-scratch](https://github.com/pguso/ai-agents-from-scratch)、产品工程版 [pguso/ai-product-from-scratch](https://github.com/pguso/ai-product-from-scratch)

## 核心篇目

| 篇目 | 主题 / 核心机制 | 状态 | 链接 |
| :--- | :--- | :--- | :--- |
| **智能体循环的形式化** | Agent = 受控马尔可夫链；闭集约束 argmax、状态转移方程、终止条件、上下文 $O(T^2)$ 预算与显存带宽延迟模型 | 🟢 已完结 | [阅读笔记 →](./agent-loop-formalization.md) |
| **结构化输出即可靠性契约** | 合法率 $q^n$ 指数衰减、$k$ 次重试残余失败 $(1-p)^k$、贪心解码的相关失败陷阱、语法约束解码 $P_{\mathcal{L}}=1$、闭集投影降级 | 🟢 已完结 | [阅读笔记 →](./structured-output-contract.md) |
| **规划作为数据与 AoT 依赖图** | 计划即偏序程序、Kahn $O(\lvert V\rvert+\lvert E\rvert)$ 拓扑执行、关键路径与并行上界、误差累积 $1-(1-\epsilon)^{N}$、DAG 校验与静默部分失败 | 🟢 已完结 | [阅读笔记 →](./planning-aot-dependency-graph.md) |

## 学习路线图（对齐仓库 12 课）

仓库的 12 课是「一次只加一个概念」的渐进式课程，本专区的笔记按**主题聚类**而非按课号顺序，便于形成机制层面的整体认知。

| 阶段 | 仓库课号 | 能力 | 本专区对应笔记 | 认知锚点 |
| :--- | :--- | :--- | :--- | :--- |
| 地基 | 01–03 | 文本对话 → 角色 → JSON 契约 | [结构化输出契约](./structured-output-contract.md) §1–3 | 概率输出如何被约束成可靠组件 |
| 行动 | 04–06 | 决策路由 → 工具 → 循环 | [智能体循环的形式化](./agent-loop-formalization.md) | 闭集 argmax 与状态转移 |
| 智能 | 07 | 短长期记忆 | [智能体循环的形式化](./agent-loop-formalization.md) §4 | 上下文预算的二次增长陷阱 |
| 规划 | 08–10 | 计划 → 原子动作 → AoT 依赖图 | [规划作为数据与 AoT](./planning-aot-dependency-graph.md) | 偏序、拓扑序、关键路径 |
| 可观测 | 11–12 | golden 回归评测 + span 遥测 | 两篇笔记的 §6 + 本页下方速查 | 缺陷发现提前量 |

## 本地运行速查

```bash
# 1. 依赖（本地推理，无需任何 API key）
pip install llama-cpp-python

# 2. 放置 GGUF 权重（Q4_K_M 约 5GB，对应约 5GB RAM）
#    huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF
#    -> models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

# 3. 自检 + 跑通全部 12 课示例
python setup_check.py
python complete_example.py
```

::: warning 硬件预期
CPU 推理下单次生成延迟量级为 **10–30 秒**。这是**正常现象**而非配置错误：自回归解码是显存/内存带宽受限（memory-bound），延迟近似为

$$
t_{\text{decode}} \approx \frac{G \cdot W}{\text{BW}}
$$

其中 $G$ 为生成 token 数、$W$ 为权重字节数、$\text{BW}$ 为内存带宽。详见 [CUDA 显存层级与 Roofline 模型 →](../../infra/cuda-memory-hierarchy.md)。
:::

## 关键机制速查

| 机制 | 一句话本质 | 定量锚点 |
| :--- | :--- | :--- |
| Agent Loop | 观察 → 决策 → 动作的重复，带状态与终止 | $C(T)=\sum_{t=1}^{T}(P_t+G_t)$ |
| 闭集动作 | 模型只**选择**不**生成**动作名 | $a^\star=\arg\max_{a\in\mathcal{A}}\pi_\theta(a\mid s_t)$ |
| 结构化输出 | 概率采样被投影到 JSON 文法 $\mathcal{L}$ | $P_{\mathcal{L}}\approx q^{n}$ |
| 重试 | 把低概率组件变成可用组件 | $P_{\text{fail}}=(1-p)^{k}$ |
| 记忆 | 跨步持久化的显式数据 | $P_t=O(t)\Rightarrow C=O(T^{2})$ |
| AoT 图 | 带依赖偏序的原子动作集合 | Kahn $O(\lvert V\rvert+\lvert E\rvert)$ |
| Golden Eval | 与 prompt 同管的回归测试 | 改 prompt 必跑回归 |
| Span 遥测 | trace → span 两级结构化日志 | 含 `duration_ms` / `error` |

## 与本库其他领域的交叉点

| 交叉主题 | 智能体侧 | 本库既有笔记 |
| :--- | :--- | :--- |
| 误差累积 | 多步智能体 $1-(1-\epsilon)^{N}$ vs 原子动作过细分解 | [Flow Matching 与连续流生成动力学 →](../../foundations/flow-matching-derivation.md) |
| 上下文压缩 | KV Cache 分页 / 记忆淘汰策略 | [vLLM PagedAttention →](../../infra/vllm-paged-attention.md) |
| 记忆容量 | 闭集检索 vs 联想记忆的容量极限 | [Softmax vs 线性注意力容量 →](../../foundations/linear-attention-capacity.md) |
| 后训练 | 用 RL 奖励训练智能体策略 | [DeepSeek-R1 与 GRPO →](../llm/deepseek-r1-reasoning.md) |
| 本地部署 | GGUF 量化权重与带宽约束 | [CUDA 显存层级与 Roofline →](../../infra/cuda-memory-hierarchy.md) |

::: tip 交叉启发
**多步智能体的误差累积与扩散 / Flow 采样链的误差累积是同构的。** 二者都是「$N$ 步串联的随机过程，每步引入小误差 $\epsilon$，最终误差 $\approx 1-(1-\epsilon)^{N}$」。差别只在**可否回溯**：

- 扩散/流匹配的每一步有**闭式解或 ODE 求解器**可以回溯重算；
- 智能体的每一步是**不可逆的外部副作用**（发邮件、删文件、改数据库），无法重算，只能靠**事前闭集约束 + 事后 golden 回归**来补偿。

这提示了一条可迁移到可控视频生成的研究类比：**长时序视频 DiT 的去噪链本质上是「无回滚能力的智能体循环」**，任何一步的时序不一致都会被后续步放大且不可修复，因此「盲复原」类工作的价值正在于把不可逆链条前置为可验证的显式映射。
:::
