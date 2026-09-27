# 结构化输出即可靠性契约：$q^n$ 指数衰减、相关重试陷阱与语法约束解码

> **标签**：`Agent` `Structured Output` `Tool Use` `Reliability` `Probability` `Math Derivation`
> **更新时间**：2026-09-27
> **参考来源**：[pguso/agents-from-scratch](https://github.com/pguso/agents-from-scratch) · `agent/agent.py:111-146` · `shared/utils.py:27-110` · `agent/tools.py:36-59`

---

## 1. 问题定义：概率组件 → 可靠组件

LLM 是**概率性**组件：一次调用返回 $y \sim p_\theta(y \mid x)$。而下游代码需要的是**确定性**契约：合法 JSON、字段齐全、枚举取值合法、工具名在白名单内。两者之间的鸿沟正是本节要量化的对象。

形式化地，设期望的合法输出集合为文法 (grammar) $\mathcal{L}$：

$$
P_{\mathcal{L}} \;=\; \Pr_{y \sim p_\theta(\cdot \mid x)} \left[ y \in \mathcal{L} \right] \;=\; \sum_{y \in \mathcal{L}} p_\theta(y \mid x)
$$

智能体可用性要求 $P_{\mathcal{L}} \to 1$。**关键在于 $P_{\mathcal{L}}$ 随 schema 规模指数衰减**，这决定了所有工程手段（重试、约束解码、校验）的取舍。

::: danger 术语铁律
「让模型输出 JSON」不是约束，是**祈使**。真正的约束是把解码器的可行集替换为 $\mathcal{L}$（§4），此时 $P_{\mathcal{L}} \equiv 1$，无需任何重试。
:::

---

## 2. 核心算法与数学

### 2.1 token 级乘积 → $q^n$ 指数衰减

自回归解码下 $\Pr[y \in \mathcal{L}]$ 精确分解为 token 级条件概率之积：

$$
\Pr[y \in \mathcal{L}] = \prod_{i=1}^{n} \Pr\!\left[ y_i \text{ 是第 } i \text{ 个合法 token} \;\middle|\; y_{<i} \right]
$$

设每个位置上「选中合法 token」的概率为常数 $q$（合法集合唯一确定时的理想化），则：

$$
\boxed{\;P_{\mathcal{L}} \approx q^{\,n}\;}
$$

其中 $n$ 为输出 token 数（受 `max_tokens` 与 schema 复杂度共同决定）。

**定量后果**：

| $q$ | $n=10$ | $n=20$ | $n=30$ | $n=50$ |
| :--- | :--- | :--- | :--- | :--- |
| 0.99 | 0.904 | 0.818 | 0.740 | 0.605 |
| 0.98 | 0.817 | 0.668 | 0.545 | 0.364 |
| 0.95 | 0.599 | 0.358 | 0.215 | 0.077 |
| 0.90 | 0.349 | 0.122 | 0.042 | 0.005 |

::: info 公式速查：schema 长度的代价
$n=30$、$q=0.98$ 时 $P_{\mathcal{L}} \approx 0.545$——**近一半的调用会格式崩**。这解释了为什么 `generate_structured()` 必须要 3 次重试（`agent/agent.py:139`），也解释了 golden dataset 里为什么把 schema 写成**多行 + 附 Example**（`evals/golden_datasets.py:26-31`）：

```python
"schema": """{
  "topic": "the topic name as a string",
  "difficulty": "beginner" or "intermediate" or "advanced"
}

Example: {"topic": "machine learning", "difficulty": "intermediate"}""",
```

few-shot 示例把隐式文法 $q$ 抬高，代价是 $P_t$ 增加约 40–60 token——**用 $O(1)$ 的上下文换指数级的合法性提升**，是本领域性价比最高的单一优化。
:::

### 2.2 $k$ 次重试的残余失败率

$k$ 次**独立**重试后仍全败的概率：

$$
P_{\text{fail}}^{(k)} = \left( 1 - P_{\mathcal{L}} \right)^{k}
$$

$$
\boxed{\;\text{有效可用率} = 1 - (1 - p)^{k}\;}
$$

| $p = P_{\mathcal{L}}$ | $k=1$ | $k=2$ | $k=3$ | $k=5$ |
| :--- | :--- | :--- | :--- | :--- |
| 0.30 | 0.300 | 0.510 | 0.657 | 0.832 |
| 0.50 | 0.500 | 0.750 | 0.875 | 0.969 |
| 0.70 | 0.700 | 0.910 | 0.973 | 0.998 |
| 0.85 | 0.850 | 0.978 | 0.997 | 0.9999 |

**边际收益递减极快**：$p=0.5$ 时 $k$ 从 1 到 3 提升 37.5 个百分点，从 3 到 5 只再提升 9.4 个点，而成本翻倍（母本仓库的循环是 $k=3$ 硬编码，见 `agent/agent.py:139` / `:186` / `:232` / `:291`）。

**期望成本**：每次调用开销 $\approx c$（$c$ 含 prefill + decode，见 [智能体循环的形式化 →](./agent-loop-formalization.md) §3.3），则

$$
\mathbb{E}[\text{cost}] = c \cdot \mathbb{E}[k] = c \sum_{j=1}^{k_{\max}} P(\text{需要第 } j \text{ 次}) = c \sum_{j=1}^{k_{\max}} (1-p)^{j-1}
$$

$p = 0.5$、$k_{\max}=3$ 时 $\mathbb{E}[k] = 1.75$，即**平均每 4 次成功要付 7 次调用的钱**。在 CPU 本地推理下（单次 $c \approx 125$ s），这是分钟级的直接放大。

### 2.3 陷阱：贪心解码使重试相关失败

母本仓库的重试全部在 `temperature=0.0` 下进行（`agent/agent.py:140, 187, 233, 292`；`agent/planner.py:41, 87, 131`）。而 `temperature=0` 即贪心解码 $\arg\max$，是**确定性函数**：

$$
y^{(1)} = y^{(2)} = \cdots = y^{(k)} = \arg\max_y p_\theta(y \mid x) \quad\Longrightarrow\quad P_{\text{fail}}^{(k)} = 1 - p
$$

$$
\boxed{\;\text{贪心重试的有效可用率} = p \text{（与 } k \text{ 无关）}\;}
$$

这**完全抵消**了重试机制的价值。若失败源于「模型在该上下文下就是会把 `}` 写成 `]`」，则重试 3 次与 1 次结果完全相同。

::: danger 勿照搬：相关失败 (Correlated Failure)
`for attempt in range(3): llm.generate(prompt, temperature=0.0)` 是**无效重试**。它只在模型带随机性（`seed=-1` 但贪心路径确定）之外的场景下有意义。

**正确修法（三选一，成本递增）**：

| 方案 | 机制 | $P_{\text{fail}}$ |
| :--- | :--- | :--- |
| ① 重试时升温 | 首次 $T=0$，重试 $T \in [0.3, 0.7]$ | 近似独立，$(1-p)^k$ 成立 |
| ② 错误反馈 | 把 `parse error` 回灌提示词：`Previous attempt failed: {err}. Fix and output ONLY valid JSON.` | 定向修复，$p' \gg p$ |
| ③ 语法约束解码 | GBNF / JSON Schema 约束 logits mask | $P_{\mathcal{L}} \equiv 1$ |
:::

### 2.4 语义层：闭集枚举与多数投票

格式合法 ≠ 内容正确。`decide()` 额外做了一次**枚举成员校验**（`agent/agent.py:190-193`）：

$$
a^\star = \arg\max_{a \in \mathcal{A}} \pi_\theta(a \mid x), \qquad \mathcal{A} = \{\texttt{answer\_question}, \texttt{summarize\_text}, \texttt{translate}\}
$$

若 $p_\text{top}$ 与 $p_2$ 差距过小，闭集 argmax 也会错。提升手段是**多次采样的多数投票（self-consistency）**：

$$
\hat{a} = \arg\max_{a \in \mathcal{A}} \sum_{i=1}^{m} \mathbb{1}\left[ a^{(i)} = a \right], \quad a^{(i)} \sim p_\theta(\cdot \mid x), \; T > 0
$$

$$
P_{\text{corr}}^{\text{majority}} = \sum_{a \in \mathcal{A}} \int \left( \sum_{i} \pi_\theta(a \mid x, z_i) \right)^{m} d\text{vol} \;\ge\; \max_a \pi_\theta(a \mid x)
$$

::: info 公式速查：两种「可靠性」必须分开度量
1. **格式可靠率** $P_{\mathcal{L}}$ —— 能否解析成契约对象。本笔记 §2.1–2.3 讨论的全部内容。
2. **决策准确率** $P_{\text{acc}}$ —— 选的动作/内容是否正确。需要 golden label，属于 Lesson 11 的范畴。

母本仓库把它们混在 `AgentEval` 的四个 suite 中（`agent/evals.py:77` / `:138` / `:212` / `:265`），但断言强度不同：`test_structured_output` 检查「能否解析 + 字段齐全」是**硬断言**，`test_tool_calls` 检查「工具名 + 部分参数」是**半硬断言**。**报告时必须分别给出两个数字**，否则「格式 100% 通过」会掩盖「决策 60% 正确」。
:::

---

## 3. 成本与上下文预算

结构化输出相比自由文本的净成本：

$$
\Delta c = \underbrace{G_{\text{json}} - G_{\text{free}}}_{\text{生成变长：} \approx +40\%} + \underbrace{(k-1) \cdot c \cdot (1 - p)}_{\text{重试期望}} + \underbrace{c_{\text{parse}}}_{\approx 0,\ \text{可忽略}}
$$

以 $p = 0.85$（few-shot 后）为基准，$k=3$ 的期望尝试次数为

$$
\mathbb{E}[k] = 1 + (1-p) + (1-p)^{2} = 1 + 0.15 + 0.0225 = 1.17
$$

| 方案 | 生成长度 $G$ 增幅 | 额外调用期望 | 端到端可用率 | 相对 few-shot |
| :--- | :--- | :--- | :--- | :--- |
| 自由文本 + 启发式抽取 | $+0\%$ | $0$ | $\approx 0.40$ | $0.47\times$ |
| JSON 提示（无示例） | $+40\%$ | $0$ | $\approx 0.55$ | $0.65\times$ |
| + few-shot 示例 | $+55\%$ | $0$ | $\approx 0.85$ | $1.00\times$（基准） |
| + $k=3$ 重试（**贪心 $T{=}0$**） | $+55\%$ | $0.17$ | $\approx 0.85$ | $\mathbf{1.00\times}$（**零提升**） |
| + $k=3$ 重试（升温 $T{=}0.6$） | $+55\%$ | $0.17$ | $\approx 1 - 0.15^{3} = 0.997$ | $1.17\times$ |
| + 语法约束解码 (L3) | $+40\%$ | $0$ | $\mathbf{1.0000}$ | $1.18\times$（且零重试延迟） |

**贪心重试那一行是纯浪费**：付出 17% 的额外调用成本，可用率一位不变（§2.3）。而 L3 语法约束同时拿下最高可用率**与**最低成本。

::: tip 关键机制：优先买「确定性」而非「重试次数」
把预算从「$k$ 从 3 提到 5」挪到「启用 grammar 约束」，可用率从 $0.99$ 提到 $1.0$，且**消除了全部重试延迟**。llama.cpp 原生支持 GBNF 语法（`llama_cpp.Llama(..., grammar=...)`），无需换推理栈。
:::

---

## 4. 三层强制等级

| 层级 | 机制 | $P_{\mathcal{L}}$ | 母本仓库 | 工业实现 |
| :--- | :--- | :--- | :--- | :--- |
| L1 | 提示词祈使（"Respond with ONLY valid JSON"） | $\approx q^n$，$q$ 低 | ✓ `agent.py:126-136` | 普遍存在但不可靠 |
| L2 | L1 + 解析容错 + 校验 + 重试 | $\approx 1 - (1-p)^k$ | ✓ `shared/utils.py:27` + 3 次重试 | 常见工程默认 |
| L3 | **logits mask / grammar 约束解码** | $\equiv 1$ | ✗ | outlines / xgrammar / GBNF / OpenAI strict schema |

L2 的解析容错值得单独看：`extract_json_from_text()` 实现了**五级降级**（`shared/utils.py:39-110`）：

1. 直接 `json.loads`；
2. 剥 ```` ```json ```` 围栏；
3. 剥离 `JSON:` / `Response:` / `Answer:` / `Here's the JSON:` / `The JSON is:` 前缀；
4. 抽取首个 `{` 到末个 `}` 之间的子串；
5. 修补**奇数引号**（未闭合字符串）后重试。

::: warning 降级解析的隐性代价
第 4 步用 `text.find('{')` / `text.rfind('}')` 做**贪婪子串抽取**。若模型输出解释文字中含花括号（例如「集合 $A=\{1,2\}$」），抽取结果会是畸形 JSON 并被第 5 步进一步「修补」成**语义错误但语法合法**的对象。这比直接失败更危险——**它污染了下游状态而不报错**。

**检测手段**：在 `AgentEval.test_structured_output` 中追加断言——若走了降级路径（而非路径 1），在 `EvalResult` 中标记 `degraded=True` 并单独统计降级率。
:::

### 4.1 L3 的实现（llama.cpp GBNF）

```python
import json
from llama_cpp import Llama, LlamaGrammar

# llama.cpp 原生 GBNF 文法：把可行集硬约束为合法 JSON
JSON_GRAMMAR = LlamaGrammar.from_string(r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null")
object ::= "{" ws ( string ws ":" ws value ("," ws string ws ":" ws value)* )? ws "}"
array  ::= "[" ws ( value ("," ws value)* )? ws "]"
string ::= "\"" char* "\""
char   ::= [^"\\\x00-\x1F] | "\\" (["\\bfnrt/] | "u" hex hex hex hex)
hex    ::= [0-9a-fA-F]
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws     ::= [ \t\n]*
""")

def structured_call(llm: Llama, prompt: str,
                    schema: dict, k: int = 3) -> dict | None:
    """L3 结构化调用：语法约束保证 P_L == 1，重试只用于字段语义校验。"""
    for attempt in range(1, k + 1):
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=512,
            temperature=0.0 if attempt == 1 else 0.6,   # ① 首次贪心，重试升温
            grammar=JSON_GRAMMAR,                        # ② L3 语法约束
        )["choices"][0]["text"]
        obj = json.loads(out)                            # 必然成功：$P_L = 1$

        missing = [f for f in schema.get("required", []) if f not in obj]
        if not missing:
            return obj

        # ③ 错误反馈：把缺失字段回灌，让模型定向修复
        prompt += (f"\n\nYour previous output was missing fields: {missing}. "
                   f"Return the corrected JSON object only.")
    return None
```

三点设计说明：

1. `grammar=JSON_GRAMMAR` 在采样时对非法 token 置 $-\infty$，$\mathcal{L}$ 成为解码器的硬可行集；
2. 重试时 $T = 0.6$ 打破贪心路径的相关性；
3. 重试携带**具体的缺失字段名**（错误反馈），而非重复同一提示词——这是把 $P_{\text{fail}}$ 从 $(1-p)^k$ 压到远低于该值的关键。

---

## 5. 工具 schema：契约的第二道闸

工具调用的可靠性 = 格式合法 × 工具名合法 × 参数合法。母本仓库的 `get_tool_schema()`（`agent/tools.py:36-59`）给出了最小可用 schema：

```python
{
  "calculator": {
    "description": "Perform basic arithmetic operations",
    "parameters": {
      "a": {"type": "number", "description": "First number"},
      "b": {"type": "number", "description": "Second number"},
      "operation": {"type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"}
    },
    "required": ["a", "b"]
  }
}
```

三处关键设计：

1. **`enum` 收窄动作空间**——把 `operation` 从自由文本压到 4 个值，$\lvert\mathcal{A}\rvert = 4$，对应 [智能体循环的形式化 →](./agent-loop-formalization.md) §2.1 的闭集 argmax；
2. **`required` 显式声明**——`decide()`/`request_tool()` 用 `"x" in parsed` 做成员校验，正是 `required` 的手写版（`agent/evals.py:111`）；
3. **白名单执行**——`execute_tool()` 先查字典再 `**arguments` 直传（`agent/tools.py:76-83`）：

```python
if tool_name not in tools:
    raise ValueError(f"Unknown tool: {tool_name}")
return tools[tool_name](**arguments)
```

::: danger 勿照搬：`**arguments` 直传缺少类型收窄
`execute_tool` 未校验 `arguments` 的键集合与类型。若模型多返回一个键（如 `"operationn": "add"`），会直接 `TypeError`；若把 `a` 传成字符串 `"42"`，`x + y` 会隐式失败或产出 `float('inf')`（除零被 `agent/tools.py:27` 静默转为 `inf` 而非抛错——**又一个静默失败**）。

**修法**：在执行前用 `jsonschema` 校验，或显式收窄：

```python
def coerce_args(tool_name: str, arguments: dict) -> dict:
    spec = get_tool_schema()[tool_name]["parameters"]
    if set(arguments) - set(spec):
        raise ValueError(f"unexpected arguments: {set(arguments) - set(spec)}")
    out = {}
    for k, meta in spec.items():
        if k not in arguments:
            continue
        v = arguments[k]
        out[k] = float(v) if meta["type"] == "number" else str(v)
        if meta.get("enum") and out[k] not in meta["enum"]:
            raise ValueError(f"{k}={out[k]!r} not in enum {meta['enum']}")
    return out
```
:::

---

## 6. 评测与可观测性

### 6.1 必须分开报的两个数字

| 指标 | 公式 | 母本仓库 |
| :--- | :--- | :--- |
| 格式合法率 | $P_{\mathcal{L}} = 1 - \frac{\text{llm\_failures}}{\text{llm\_calls}}$ | ✓ 由 `Metrics` 统计（`agent/telemetry.py:61-63`） |
| 字段合规率 | $\frac{1}{N}\sum \mathbb{1}[\text{required} \subseteq \text{keys}(o_i)]$ | ✓ `test_structured_output` 检查 2（`agent/evals.py:111`） |
| 工具名准确率 | $\frac{1}{N}\sum \mathbb{1}[a_i = a_i^{\star}]$ | ✓ `test_tool_calls` 检查 2（`agent/evals.py:170`） |
| 参数准确率 | $\frac{1}{N}\sum \mathbb{1}[\forall k:\ \text{args}_i[k] = \text{args}_i^{\star}[k]]$ | ⚠️ 部分检查（`:182-193`），**存在 `continue` 逃逸 bug** |
| 降级解析率 | $\frac{\#\{\text{走路径 2–5}\}}{N}$ | ✗ 未统计 |

::: warning `test_tool_calls` 的 `continue` 逃逸
`agent/evals.py:182-193` 在参数不匹配时 `add_result(passed=False)` 后 `continue`，但这个 `continue` 只跳出**内层 `for` 键循环**，因此该 case 仍会继续执行到底部的 `add_result(passed=True)`——**同一条 case 被记为既失败又通过**，`passed` 与 `failed` 双双 +1，`pass_rate` 失真。
:::

### 6.2 Golden dataset 驱动的回归流程

`evals/golden_datasets.py` 定义了 4 组共 16 条用例 + 4 组边界条件。改动任何 prompt 后的强制流程：

$$
\text{改动 prompt} \;\to\; \text{跑 golden} \;\to\; \text{对比 } (P_{\mathcal{L}},\, P_{\text{acc}}) \;\to\; \text{diff prompt 版本}
$$

聚合 pass rate：

$$
\text{pass\_rate} = \frac{\sum_{s} \text{passed}_{s}}{\sum_{s} \text{total}_{s}}, \qquad s \in \{\text{structured}, \text{tool}, \text{decision}, \text{memory}\}
$$

| 阶段 | 无 golden 回归 | 有 golden 回归 |
| :--- | :--- | :--- |
| 缺陷发现时机 | 上线后用户投诉 | **提交前 CI** |
| 平均定位耗时（本地 CPU） | 数小时（分钟级推理 × 人工试跑） | 一次 `python complete_example.py` |
| 回归归因 | 几乎不可能 | golden 失败即定位到具体 prompt 字段 |

---

## 7. 与本库其他领域的交叉点

| 交叉 | 结构化输出侧 | 本库笔记 |
| :--- | :--- | :--- |
| **约束 vs 惩罚** | grammar mask 是**硬约束**，KL 惩罚是软约束 | [GRPO 的 KL 正则项 →](../llm/deepseek-r1-reasoning.md) |
| **指数衰减同构** | $q^n$ 格式合法率 vs 流匹配的误差累积 | [Flow Matching 动力学 →](../../foundations/flow-matching-derivation.md) |
| **多数投票 = 自洽采样** | $m$ 次采样投票 | [DeepSeek-R1 慢思考 →](../llm/deepseek-r1-reasoning.md) |
| **拒答/降级的显存代价** | 重试放大 decode 成本 | [CUDA Roofline →](../../infra/cuda-memory-hierarchy.md)、[vLLM 吞吐 →](../../infra/vllm-paged-attention.md) |

::: tip 交叉启发
**「语法约束解码」与「扩散模型的硬约束采样」在数学上是同一件事。** 二者都通过**修改采样过程的可行集**把概率质量重新归一化到合法子流形上：

- 约束解码：$p_\theta(y \mid x) \cdot \mathbb{1}[y \in \mathcal{L}]$，重归一化后 $P_{\mathcal{L}} = 1$；
- Classifier Guidance / 无分类器引导：$\tilde{p}(x_0 \mid c) \propto p(x_0)\,p(c \mid x_0)^{\omega}$，把样本投影到条件流形；
- ControlNet 类条件注入：直接在特征空间做投影 $f \leftarrow f + \alpha \cdot \Delta f_{\text{ctrl}}$。

三者的共同工程结论是：**约束应尽量在生成过程内部施加（$\to \tilde{p} = 1$），而非在生成后施加（$\to$ 需采样 $1/p$ 次做拒绝采样）**。母本仓库的 L2 层次本质上是**拒绝采样**，其代价 $\propto 1/p$ 正是 §2.2 中 $\mathbb{E}[k]$ 爆炸的来源。
:::
