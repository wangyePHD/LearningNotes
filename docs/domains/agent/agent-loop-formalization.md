# 智能体循环的形式化：Agent = 受控马尔可夫链 + 闭集动作投影

> **标签**：`Agent` `Agent Loop` `State` `Tool Use` `Infra` `Complexity`
> **更新时间**：2026-09-27
> **参考来源**：[pguso/agents-from-scratch](https://github.com/pguso/agents-from-scratch) · `agent/agent.py:303-329` · `agent/state.py:21-55` · [vLLM PagedAttention](../../infra/vllm-paged-attention.md)

---

## 1. 问题定义：把「智能体」还原为控制系统

### 1.1 为什么必须形式化

「智能体」这个词在工程语境里被严重过载。要消解歧义，唯一可靠的方式是**放弃拟人化描述**，把它写成一个五元组：

$$
\mathcal{A}_{\text{agent}} = \left( \mathcal{S}, \, \mathcal{A}, \, \pi, \, f, \, \mathcal{T} \right)
$$

| 符号 | 含义 | 在母本仓库中的具体载体 |
| :--- | :--- | :--- |
| $\mathcal{S}$ | 状态空间（智能体可自省的全部量） | `AgentState`：`steps` / `done` / `current_plan` / `last_action`（`agent/state.py:21-26`） |
| $\mathcal{A}$ | **闭集**动作空间，$\mathcal{A}=\{a_1,\dots,a_K\}$ | 提示词里枚举的 `analyze, research, summarize, answer, done`（`agent/agent.py:277`） |
| $\pi$ | 决策策略，由 LLM 实现 | `agent_step()` 中的一次 `llm.generate(prompt, temperature=0.0)` |
| $f$ | 状态转移函数 | `AgentState.increment_step()` / `mark_done()` |
| $\mathcal{T}$ | 终止条件集合 | `while not done and steps < max_steps`（`agent/agent.py:317`） |

::: danger 术语铁律
智能体**不思考**。它执行的是：把当前状态 $s_t$ 序列化成文本 → 前向一次 LLM → 解析出闭集内的一个 $a_t$ → 修改状态 → 重复。所谓「智能」完全来自 $\mathcal{A}$ 的设计质量与 $\pi$ 的命中率，与意识无关。
:::

### 1.2 Agent Loop 的最小判据：循环 + 状态

单轮 LLM 调用（Lesson 01）与 Agent（Lesson 06+）的唯一结构性差异是**两个成分的有无**：

| 组件 | 单轮 chatbot | Agent |
| :--- | :--- | :--- |
| 循环 | ✗ 一次生成即返回 | ✓ 迭代 $t=1,\dots,T$ |
| 状态 | ✗ 无（历史隐含在上下文里） | ✓ 显式 Python 对象，可读可改可断言 |
| 终止 | 隐式（生成 EOS） | 显式（$\mathcal{T}$） |

::: tip 关键机制
**没有显式状态的循环是伪 Agent。** 若状态只存在于对话历史中，智能体无法回答三个基本问题：进行到第几步了、上一步干了什么、是否该停。`AgentState` 把这三个问题变成三个可 `assert` 的字段——这就是「explicit over implicit」的全部含义。
:::

---

## 2. 核心算法：闭集约束 argmax 与状态转移方程

### 2.1 决策规则是约束 argmax，不是自由生成

LLM 的原始输出是 token 序列 $y \sim p_\theta(y \mid x)$。但 `agent_step()` 并不接受任何动作名——它只接受落在 $\mathcal{A}$ 内的字符串。因此真实决策规则是**在闭集上的投影 argmax**：

$$
a_t^\star \;=\; \arg\max_{a \in \mathcal{A}} \; \pi_\theta\!\left( a \mid \phi(s_t, x_t) \right)
$$

其中 $\phi(\cdot)$ 是状态到提示词的序列化函数（`agent/agent.py:275`：把 `steps` / `done` 直接写进提示词）。

配套的两道**后验校验**缺一不可：

$$
\text{valid}(a_t) \;=\; \mathbb{1}\!\left[ a_t \in \mathcal{A} \right] \cdot \mathbb{1}\!\left[ \text{JSON-parse}(y_t) \neq \emptyset \right]
$$

$$
\boxed{\;a_t \leftarrow \mathcal{P}_{\mathcal{A}}\!\left(\text{raw}(y_t)\right)\;}
$$

其中 $\mathcal{P}_{\mathcal{A}}$ 是**投影算子**（projection onto the action set）：

$$
\mathcal{P}_{\mathcal{A}}(y) = \begin{cases}
y, & \text{parse}(y) \in \mathcal{A} \\
\texttt{None} \;\to\; \text{retry}, & \text{otherwise}
\end{cases}
$$

对应代码 `agent/agent.py:295-301`：只有 `parsed and "action" in parsed` 才接受，否则重试至 3 次后返回 `None`，由 `run_loop` 撞上 `else: break` 主动终止（`agent/agent.py:326-327`）。

::: info 公式速查：为什么闭集是可靠性的来源
设模型对 $\mathcal{A}$ 中正确动作的最高概率为 $p_{\text{top}}$，且次高为 $p_2$。
- 自由生成时，「恰好输出合法动作名且参数正确」的概率 $P_{\text{gen}}$ 需同时满足**词法**与**语义**约束；
- 闭集选择时，只要 $p_{\text{top}}$ 占优（典型 LLM 在明确枚举下 $p_{\text{top}} > 0.8$），错误**至多退化为选错而非格式崩**。

即：闭集把「语法错误」这一整类失效转化为「语义选择」这一类更易被评测捕获的失效。这是**失效模式降维**。
:::

### 2.2 状态转移方程

一轮迭代的完整动力学：

$$
\begin{aligned}
s_t &\xrightarrow{\;\phi\;} x_t && \text{(状态序列化：} \phi(s_t) \to \text{prompt} \text{)}\\
y_t &\sim p_\theta(\cdot \mid x_t) && \text{(LLM 前向解码，} y_t \in \Sigma^{G_t} \text{)}\\
a_t &= \mathcal{P}_{\mathcal{A}}\!\left(\text{parse}(y_t)\right) && \text{(闭集投影)}\\
s_{t+1} &= f(s_t, a_t) && \text{(状态转移：} \text{steps} \mathrel{+}= 1,\; a_t = \texttt{done} \Rightarrow \texttt{done} \leftarrow \texttt{True} \text{)}
\end{aligned}
$$

终止条件（三者任一成立即停）：

$$
\mathcal{T} = \left\{ \; a_t = \texttt{done} \;\middle|\; t \geq T_{\max} \;\middle|\; a_t = \texttt{None} \;\right\}
$$

::: warning 「不前进的循环」
`run_loop` 的退出只依赖**计数器**（`steps < max_steps`）与**模型自报**（`action == "done"`），并不依赖**进展**。若模型连续返回同一个 `analyze`，循环会安静地耗尽预算并返回一串重复动作——不抛异常、不打印警告。

**检测手段**：记录动作序列 $a_1,\dots,a_T$，若 $\text{uniq}(a_{1:T}) = 1$ 且 $T > 3$，判定为震荡（oscillation）。
**修复策略**：在 $f$ 中维护 `last_action`，若 $a_t = a_{t-1}$ 则强制 $T_{\text{effective}} \leftarrow T_{\text{effective}} - 1$ 或注入扰动提示。
:::

---

## 3. 成本、复杂度与上下文预算

### 3.1 Token 预算展开

单步 token 开销 = 提示词 $P_t$ + 生成长 $G_t$：

$$
C(T) = \sum_{t=1}^{T} \left( P_t + G_t \right)
$$

三个提示词构成项在母本仓库中分别是：system prompt（固定 $P_{\text{sys}}$，约 40 token）、状态行（$O(1)$）、动作枚举（$O(K)$）。**注意 Lesson 06 的 `agent_step()` 并没有把历史动作注入提示词**——这是它上下文便宜的原因，也是它「每步近似独立」的根源。

### 3.2 全量记忆注入导致 $O(T^2)$ 二次增长

一旦接入 Lesson 07 的记忆，`run_with_memory()` 每轮调用 `self.memory.get_all()` 并把**全部**条目拼进提示词（`agent/agent.py:347-353`）：

$$
P_t = P_{\text{sys}} + \sum_{i=1}^{t} \left| m_i \right| \quad \Longrightarrow \quad P_t = O(t)
$$

$$
\boxed{\;C(T) = \sum_{t=1}^{T} O(t) = O\!\left(T^{2}\right)\;}
$$

这是**自回归上下文的标准二次税**。由此得到上下文窗口 $n_{\text{ctx}}$ 下的硬性步数上界（取记忆条目均长 $\bar{m}$ 字符）：

$$
T_{\max} \;\leq\; \left\lfloor \frac{n_{\text{ctx}} - P_{\text{sys}} - |x|}{\bar{m}} \right\rfloor
$$

以 $n_{\text{ctx}} = 2048$、$\bar{m} = 20$ 为例：$T_{\max} \approx 100$，但 **KV Cache 与 prefix 复用带来的显存成本同时按 $O(T)$ 增长**，且每轮重算全部 prefix（无 prefix caching 时）。

::: tip 交叉启发
**$O(T^2)$ 的 token 二次税与 KV Cache 分页是同一个问题的两面。** 母本仓库用「全量注入」换取了最简单的正确性，代价是二阶成本。工业界的三条出路恰好构成本专区的学习目标：

| 策略 | 复杂度 | 可追溯来源 |
| :--- | :--- | :--- |
| 滑动窗口（只留最近 $w$ 条） | $C(T) = O(T \cdot w)$ | `Memory.get_recent(n)`（`agent/memory.py:41-51`）已备但未用 |
| 摘要压缩（summarize） | $C(T) = O(T \cdot \bar{m}_{\text{sum}})$ | 需外加一次 LLM 调用 |
| 检索式记忆（只取 top-$k$） | $C(T) = O(T \cdot k)$ | `Memory.search()` 是朴素子串匹配，需换 embedding |

KV Cache 的分页化同理：见 [vLLM PagedAttention 与高并发吞吐优化 →](../../infra/vllm-paged-attention.md)。
:::

### 3.3 单步延迟：自回归解码是带宽受限的

在 CPU 本地推理（llama.cpp）下，一次生成的延迟几乎全部来自权重搬运：

$$
t_{\text{step}} \approx \underbrace{\frac{G_t \cdot W}{\text{BW}_{\text{HBM}}}_{\text{memory-bound}} + \underbrace{G_t \cdot \frac{2N_{\text{flop}}}{\text{FLOPS}}}_{\text{compute-bound，量级小} \cdot 10^{-2}}
$$

- $W$：GGUF 权重字节数（Llama-3-8B Q4\_K\_M $\approx 4.9$ GB）；
- $G_t$：生成 token 数（`LocalLLM` 默认 `max_tokens=512`，`shared/llm.py:29`）；
- 由于 $W \gg$ 激活值，Roofline 上工作点落在 memory-bound 区，比值 $\approx 10^{-1} \sim 10^{-2}$。

于是**总时延近似线性于 $T$**：

$$
T_{\text{total}} \approx T \cdot \frac{G \cdot W}{\text{BW}} \xrightarrow[\;G=512,\;W=4.9\text{GB},\;\text{BW}\approx 20\text{GB/s (CPU)}\;]{} \approx T \times 125\ \text{s}
$$

::: warning 本地 12 课的教学节奏
CPU 上单步可达 **分钟级**。这导致「跑一遍看效果」的反馈循环极慢，**必须依赖 golden 回归（Lesson 11）而非肉眼试跑**来推进学习。数值上：$512 \times 4.9 / 20 \approx 125$ 秒/step，与 QUICKSTART 中「10–30 秒」的差异来自生成长度（30 秒对应 $\sim 100$ token）。
:::

---

## 4. 失效模式与防御机制

| # | 失效模式 | 根因 | 检测 | 修复 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 格式非法 | 自由文本采样落在 JSON 文法外 | `parse() is None` | 重试 $k$ 次 / 语法约束解码（见 [结构化输出契约 →](./structured-output-contract.md)） |
| 2 | 幻觉工具名 | $a_t \notin \mathcal{A}$ 仍被执行 | 白名单查表 | `execute_tool` 先查 `tools` 字典，否则 `raise ValueError`（`agent/tools.py:80-83`） |
| 3 | 参数越界 | `**arguments` 直传，无 schema 校验 | 类型/范围断言 | JSON Schema `required` + 类型收窄；除零返回 `inf` 而非异常（`agent/tools.py:27`） |
| 4 | 循环不终止 | $\mathcal{T}$ 只看计数器与自报 done | `steps` 触顶 | 硬性 $T_{\max}$ + 早停（已实现） |
| 5 | 动作震荡 | 无进展检测 | $\text{uniq}(a_{1:T})=1$ | 记录 `last_action`，重复即降预算 |
| 6 | 上下文溢出 | $P_t = O(t)$ 二次增长 | $P_t + G_t > n_{\text{ctx}}$ | 滑窗 / 摘要 / 检索；`max_tokens` 收紧 |
| 7 | 静默部分失败 | 失败即 `break`，无日志 | **无**（`agent/agent.py:326`） | 引入 span 遥测记录失败原因 |

::: danger 勿照搬
母本仓库是**教学用实现**，`AgentState` 只有 4 个字段，`execute_tool` 只有 1 个工具，`agent_telemetry.jsonl` 是空文件。不要把它的 `Agent` 类当作可扩展基类直接用于生产——正确的做法是理解其**约束外壳**，然后替换 $\pi$（换更强模型）与 $\mathcal{A}$（换真实工具集）。
:::

---

## 5. 关键实现：带终止诊断的循环

在母本仓库 `run_loop`（`agent/agent.py:303-329`）基础上补齐震荡检测、进展校验与失败留痕的最小实现：

```python
from dataclasses import dataclass, field

@dataclass
class LoopTrace:
    actions: list[str] = field(default_factory=list)
    stop_reason: str = "unknown"
    steps: int = 0

def run_loop(agent, user_input: str, max_steps: int = 5,
             oscillation_limit: int = 3) -> LoopTrace:
    agent.state.reset()
    trace = LoopTrace()

    while not agent.state.done and agent.state.steps < max_steps:
        action = agent.agent_step(user_input)          # 闭集投影后的结果

        if action is None:
            trace.stop_reason = "parse_or_action_invalid"
            break                                      # 母本仓库此处静默 break

        name = action["action"]
        trace.actions.append(name)
        trace.steps += 1

        if trace.actions[-oscillation_limit:] == [name] * oscillation_limit:
            trace.stop_reason = "oscillation_detected"
            break

        if name == "done":
            agent.state.mark_done()
            trace.stop_reason = "model_signalled_done"
    else:
        trace.stop_reason = "max_steps_reached"        # while-else：正常耗尽

    return trace
```

关键改进三处：`while-else` 区分「自然耗尽」与「异常中断」；`oscillation_limit` 把「无进展」变成显式事件；`stop_reason` 让评测（Lesson 11）能断言终止行为而不只是看最终答案。

---

## 6. 评测与可观测性

### 6.1 循环级指标

| 指标 | 定义 | 母本仓库是否支持 |
| :--- | :--- | :--- |
| 格式合法率 | $\frac{\#\{\text{parse 成功}\}}{\#\{\text{调用}\}}$ | ✗ 无统计（仅重试） |
| 平均步数 | $\bar{T} = \frac{1}{N}\sum_{i=1}^{N} T_i$ | ✗ |
| 终止合规率 | $\frac{\#\{a_T = \texttt{done} \wedge T < T_{\max}\}}{N}$ | ✗ |
| 震荡率 | $\frac{\#\{\text{uniq}(a_{1:T})=1\}}{N}$ | ✗ |
| 工具成功率 | $1 - \frac{\text{tool\_failures}}{\text{tool\_calls}}$ | ✓ `Metrics.tool_success_rate`（`agent/telemetry.py:66-68`） |
| 重试率 | $\frac{\text{llm\_retries}}{\text{llm\_calls}}$ | ✓ `Metrics.llm_retries`（`agent/telemetry.py:171-172`） |

::: info 公式速查
$$
\bar{T} = \frac{1}{N}\sum_{i=1}^{N} T_i, \qquad
\text{终止合规率} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[a_{T_i} = \texttt{done} \wedge T_i < T_{\max}\right]
$$

若每步「正确推进」概率为 $p$（其余为原地踏步），则 $T$ 服从截断几何分布 $T \sim \text{Geom}(p) \big|_{T \leq T_{\max}}$：

$$
\mathbb{E}[T] = \frac{1 - p^{T_{\max}}}{1 - p}, \qquad \text{方差} \approx \frac{1-p}{p^{2}}
$$

$p = 0.6, T_{\max}=5$ 时 $\mathbb{E}[T] \approx 2.36$。$p$ 越低步数分布越重尾，**成本方差越大**——这正是评测必须报告步数分布而非只报平均值的原因。
:::

### 6.2 Span 两级遥测

母本 `telemetry.py` 已实现 `trace → span` 两级结构。把它挂到循环上（`@traced` 装饰器，`agent/telemetry.py:312-354`）即可得到：

```text
trace 7f3a91c2  (一次 run_loop)
├── span llm_call  duration=118_402 ms  attempt=1  success=true   prompt_len=612
├── span llm_call  duration=97_118 ms   attempt=1  success=true   prompt_len=612
├── span llm_call  duration=104_770 ms   attempt=2  success=true   prompt_len=612  ← 重试
└── span decision  selected=done  choices=[analyze, research, summarize, answer, done]
```

`attempt > 1` 的 span 计数即 `llm_retries`，是**格式合法率的在线估计量**（$\hat{p} = 1 - \frac{\text{llm\_failures}}{\text{llm\_calls}}$）。

### 6.3 Golden 回归：CPU 推理下的必需品

由于 §3.3 的分钟级延迟，母本把 `evals/golden_datasets.py` 设计为 4 组固定用例（结构化输出 4、工具调用 5、决策 4、记忆 3）+ 4 组边界（空输入 / 超长输入 / unicode / 输入含 JSON）。原则写在文件头：

> Golden dataset 与 prompt 同版本管理；**golden case 失败即 agent 坏，不是测试坏**。

这与本库视频 / 视觉域的「基模 + 固定 seed + 固定 benchmark」纪律同源：**任何 prompt 改动都必须用固定集回归**，否则无法区分「改进」与「过拟合」。

---

## 7. 与本库其他领域的交叉点

| 交叉 | 智能体侧表述 | 本库笔记 |
| :--- | :--- | :--- |
| 误差累积同构 | $P_{\text{no err}} = (1-\epsilon)^{T}$，不可回滚 | [Flow Matching 动力学 →](../../foundations/flow-matching-derivation.md) |
| 记忆容量 | 检索式记忆 top-$k$ vs 联想记忆 $e^{d}$ 容量 | [线性注意力容量极限 →](../../foundations/linear-attention-capacity.md) |
| 延迟本质 | 解码 $O(G \cdot W / \text{BW})$，memory-bound | [CUDA Roofline 模型 →](../../infra/cuda-memory-hierarchy.md) |
| 上下文管理 | KV Cache 分页 = 记忆分页 | [vLLM PagedAttention →](../../infra/vllm-paged-attention.md) |
| 策略后训练 | 用奖励优化 $\pi_\theta$，使 $a^\star$ 命中率上升 | [DeepSeek-R1 与 GRPO →](../llm/deepseek-r1-reasoning.md) |

::: tip 交叉启发（可证伪实验设想）
**命题**：在长时序可控视频生成中，引入「动作级闭集约束 + 逐段 golden 回归」，可把长视频的时序崩坏率（temporal inconsistency）在同等算力下降低，机理与智能体的「不可回滚链条前置校验」一致。

**最小证伪实验**：以 VideoDeltaNet (VDN-H3) 为被测生成器（见 [VideoDeltaNet H3 实时架构剖析 →](../video/video-deltanet-h3.md)），固定 seed 与 prompt 集，把去噪/滚动预测链切分为 $T$ 段，每段出口施加「与上一段重叠区域的一致性硬约束」（即闭集投影到合法运动集合），测量 VBench 风格时序一致性分数随 $T$ 的斜率。若加约束后崩坏率不随 $T$ 增长，命题成立；否则说明崩坏源自单段生成质量而非链式累积，命题被证伪。
:::
