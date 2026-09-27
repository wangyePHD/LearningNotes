# 规划作为数据与 AoT 依赖图：偏序、拓扑执行、关键路径与静默部分失败

> **标签**：`Agent` `Planning` `Atomic Action` `AoT` `DAG` `Complexity` `Tool Use`
> **更新时间**：2026-09-27
> **参考来源**：[pguso/agents-from-scratch](https://github.com/pguso/agents-from-scratch) · `agent/planner.py:96-206` · `lessons/10_atom_of_thought.md` · [Flow Matching 与连续流生成动力学](../../foundations/flow-matching-derivation.md)

---

## 1. 问题定义：计划是数据，不是想法

母本仓库 `PHILOSOPHY.md` 明确拒绝 ReAct，并把规划定义为**可检查、可修改的数据结构**：

> "Planning is data generation, not reasoning. Plans are inspectable, modifiable data structures."

这句话有精确的数学含义。**计划 = 一个偏序集 (poset)**，而**想法 = 不可观测的隐变量**。二者可组合性完全不同：

| 维度 | 计划作为数据 | 计划作为想法 |
| :--- | :--- | :--- |
| 形式 | $P = (V, \preceq)$，$V$ 动作集，$\preceq$ 偏序 | $\mathbb{E}_{\text{LLM}}[\text{CoT}] \mid x$，不可枚举 |
| 可验证性 | ✓ 可判 DAG / 环 / 悬空依赖 | ✗ 无法判定 |
| 可并行 | ✓ 反链内节点可并行 | ✗ 串行文本 |
| 可回滚 | ✓ 节点粒度 | ✗ 整链粒度 |
| 可评测 | ✓ golden 计划集 | ✗ 只能评最终答案 |

::: danger 术语铁律
「模型规划了」是一个不可检验的断言。可检验的版本是：「模型输出了一个通过结构校验的偏序 $P = (V,\preceq)$，$V$ 中 $N$ 个节点的 golden 匹配率为 $m$」。本笔记全部采用后者。
:::

---

## 2. 核心算法

### 2.1 三个阶段的递进（Lesson 08 → 09 → 10）

| 阶段 | 结构 | 形式化 | 母本实现 |
| :--- | :--- | :--- | :--- |
| L08 计划 | **全序**（有序列表） | $P = \langle s_1, s_2, \dots, s_K \rangle$，$s_{i+1} \prec s_i$ | `create_plan()` `planner.py:11-47` |
| L09 原子动作 | **单步细化** | $\phi : s \mapsto (a, \text{inputs})$ | `create_atomic_action()` `planner.py:50-93` |
| L10 AoT | **偏序**（DAG） | $G = (V, E)$，$E \subseteq V \times V$ | `create_aot_graph()` `planner.py:96-147` |

从全序到偏序的泛化是本节的数学核心：

$$
\text{L08: } V \cong [K] \text{ 上的全序} \;\subset\; \text{L10: } V \text{ 上的任意 DAG}
$$

### 2.2 原子性的形式定义

Lesson 09 要求一个动作是「原子的」，其可检验含义是**三条可分割性公理**：

$$
\boxed{
\begin{aligned}
&\textbf{(A1) 独立可验}\;\; \exists\, \text{validate}(a_i) : \text{bool} \text{ 仅依赖 } \text{inputs}_i \\
&\textbf{(A2) 独立可测}\;\; \exists\, \text{test}(a_i) \text{ 单独可断言} \\
&\textbf{(A3) 独立可滚}\;\; \exists\, \text{rollback}(a_i) : \mathcal{S} \to \mathcal{S}
\end{aligned}
}
$$

违反 A1 的动作（需要看后续步骤才能判断对错，如「写一段总结」）**不可验证**；违反 A3 的动作（不可逆副作用，如已发送的邮件）**不可回滚**。

### 2.3 原子动作的粒度博弈：误差累积

设一步执行成功概率为 $q$，一个高层步骤被拆成 $N$ 个原子动作后，全对概率：

$$
P_{\text{all ok}} = q^{\,N} \;\approx\; \left(1 - \epsilon\right)^{N} \;\approx\; e^{-\epsilon N}
$$

$$
\boxed{\;\text{失败概率} \;\approx\; 1 - e^{-\epsilon N}\;\approx\;\epsilon N \quad (\epsilon N \ll 1)\;}
$$

| 原子化粒度 $N$ | $\epsilon = 0.05$ 时全对率 $0.95^{N}$ | 全对率 | 评价 |
| :--- | :--- | :--- | :--- |
| 1（不拆） | $0.95$ | 95% | 不可验证（违反 A1） |
| 5 | $0.95^{5}$ | 77.4% | **甜点区** |
| 20 | $0.95^{20}$ | 35.8% | 误差累积主导 |
| 60 | $0.95^{60}$ | 4.6% | 必然失败 |

::: info 公式速查：可用的粒度判据
要使全对率 $\geq \rho$，需

$$
N \;\leq\; \frac{\ln \rho}{\ln q} \;\approx\; \frac{\ln \rho}{-\epsilon}
$$

取 $\rho = 0.8$（工程可接受下限）、$\epsilon = 0.05$：$N \leq \frac{\ln 0.8}{\ln 0.95} \approx 4.5$。**故原子动作粒度不应超过约 5 个可失败单元**——超过后，正确的兜底策略不是「继续跑」，而是**重新规划（replan）**。
:::

**这也解释了为什么 AoT 的并行能力是必需的**：若 $|V| = 20$、$q = 0.95$，串行全对率仅 35.8%，但若依赖图被组织成 4 条长度 5 的反链并全部并行，则每条链全对率 $0.95^5 = 77.4\%$——**并行使可行解从「几乎不可能」变成「多数情况可用」**。

### 2.4 拓扑执行

执行 DAG 的标准算法是 Kahn 算法。母本 `execute_graph()`（`agent/planner.py:150-206`）用了一个更朴素的**迭代松弛**版本：

$$
R^{(0)} = \emptyset, \qquad
R^{(k+1)} = R^{(k)} \cup \left\{ v \in V \setminus R^{(k)} \;\middle|\; \text{deps}(v) \subseteq R^{(k)} \right\}
$$

$$
\text{终止条件：} \; R^{(K)} = V \;\wedge\; K \leq 2\lvert V \rvert
$$

复杂度对比（$\lvert E \rvert$ = 依赖边数）：

| 算法 | 时间复杂度 | 空间 | 特性 |
| :--- | :--- | :--- | :--- |
| 母本迭代松弛 | $O(\lvert V\rvert \cdot \lvert E\rvert)$ | $O(\lvert V\rvert)$ | 天然并行友好（每轮一批） |
| Kahn（入度 + 队列） | $O(\lvert V\rvert + \lvert E\rvert)$ | $O(\lvert V\rvert + \lvert E\rvert)$ | 逐个出队 |
| Tarjan SCC | $O(\lvert V\rvert + \lvert E\rvert)$ | $O(\lvert V\rvert)$ | **能定位环上具体节点** |

::: warning 母本实现是「顺序执行器」，不是调度器
`execute_graph` 的内层 `for node in nodes` 是**串行**的，且 `executor_func` 只有一个同步 callable 参数。Lesson 10 声称「enables parallel execution of independent actions」，但代码中：

- 同一轮内可执行的多节点**不会被并发发射**；
- 失败节点被 `executed.add(node_id)` 标记为已完成（`planner.py:204`），**以避免死循环**——但代价是失败的后继节点仍会被执行（因为其 `depends_on` 已被满足）。

因此「AoT 支持并行」在母本仓库里是**数据结构层面的能力，而非已实现的能力**。
:::

---

## 3. 成本、复杂度与关键路径

### 3.1 关键路径与并行上界

设节点 $v$ 的执行时长为 $t_v$。串行总时长与并行下界：

$$
T_{\text{seq}} = \sum_{v \in V} t_v, \qquad
T_{\text{par}} \;\geq\; L_{cp} \;:=\; \max_{p \in \mathcal{P}} \sum_{v \in p} t_v
$$

其中 $\mathcal{P}$ 为从源点到汇点的所有路径集合。**关键路径是任何调度的下界**（Amdahl 定律的图论版本）：

$$
\text{speedup}_{\max} = \frac{T_{\text{seq}}}{L_{cp}} \;\leq\; \frac{\lvert V \rvert}{\text{最长链长度}}
$$

举例，$|V| = 20$ 的 AoT 图：

| 依赖结构 | 串行 $T_{\text{seq}}$ | $L_{cp}$ | 加速上界 | 母本实际加速 |
| :--- | :--- | :--- | :--- | :--- |
| 全链 $\left(1 \to 2 \to \cdots \to 20\right)$ | $20t$ | $20t$ | $1.0\times$ | $1.0\times$ ✓ |
| 4 条长度 5 的反链 | $20t$ | $5t$ | $\mathbf{4.0\times}$ | $1.0\times$ ✗ |
| 「研究 → 写作 → 校对」三层，每层多节点 | $\approx 20t$ | $3t$ | $\approx 6.7\times$ | $1.0\times$ ✗ |

::: tip 关键机制：AoT 的价值 = 分支并行度 − 调度成本
**AoT 真正的收益不在于「更聪明的计划」，而在于把 $T_{\text{seq}}$ 压向 $L_{cp}$。** 母本仓库把数据结构（依赖边）做对了，却没有调度器，因此只拿到了 $1.0\times$。补上并发执行即免费获得表中第 2、3 行的加速。
:::

### 3.2 Token 成本：一次规划 vs 逐步规划

| 方案 | LLM 调用数 | Token 量 | 能否并行 | 计划可检查 |
| :--- | :--- | :--- | :--- | :--- |
| 逐步 ReAct（每步问一次） | $T$ | $O(T \cdot P)$ | ✗ | ✗ |
| 一次性 AoT 规划 | $1$ | $O(P_{\text{plan}})$ | ✓ | ✓ |
| 静态 AoT（规划一次，全链执行） | $1$ | $O(P_{\text{plan}})$ | ✓ | ✓ |
| 动态 AoT（每节点重规划） | $\lvert V\rvert$ | $O(\lvert V\rvert \cdot P)$ | 部分 | ✓ |

静态计划的风险是**计划过期（stale plan）**。形式化：设计划 $P$ 在状态 $s$ 下有效当且仅当

$$
s \models \bigwedge_{v \in V} \text{pre}(v)
$$

环境转移使 $s$ 变为 $s'$ 后，失效概率 $P_{\text{stale}} = 1 - \gamma$（$\gamma$ 为转移保持前置条件成立的概率）。则期望重规划次数：

$$
\mathbb{E}[\text{replans}] = \lvert V\rvert \cdot (1 - \gamma)
$$

$\lvert V|=10$、$\gamma = 0.7$ 时 $\mathbb{E} \approx 3$ 次重规划，即静态计划的 LLM 调用数从 1 升到约 4——**接近逐步 ReAct 的成本，却保留了结构化的可检查性**。这是 AoT 优于 ReAct 的真正量化理由。

---

## 4. 失效模式与防御机制

| # | 失效模式 | 根因 | 检测 | 修复 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 悬空依赖 | `depends_on` 引用不存在的 id | 遍历边查 $\mathrm{id} \in V$ | 丢弃悬空边或退回重规划 |
| 2 | 有向环 | 模型生成自依赖 | Tarjan SCC / Kahn 未排完即有剩余 | 定位环上节点，重规划 |
| 3 | 计划为空 | `nodes` 为空列表 | $\lvert V \rvert = 0$ | 视为失败而非成功 |
| 4 | 节点重复 | 两个节点同 `id` | `len(set(ids)) != len(ids)` | 去重 / 报错 |
| 5 | 过细粒度 | $N > \ln\rho/\ln q$ | $N$ 统计 | 合并节点或提高 $q$（验证/重试） |
| 6 | 计划过期 | 环境漂移 | 节点执行时校验 $\text{pre}(v)$ | 局部重规划 |
| 7 | **静默部分失败** | 迭代上限 `2\lvert V\rvert` 耗尽后静默返回 | **无** | 必须显式检查 $\lvert\text{executed}\rvert = \lvert V\rvert$ |
| 8 | 失败后仍执行后继 | 失败节点被标 `executed` | 记录 `success=False` 并阻断后继 | 引入状态三值（未执行 / 成功 / 失败） |

::: danger 勿照搬：静默部分失败 (Silent Partial Failure)
`agent/planner.py:170` 设 `max_iterations = len(nodes) * 2`，循环耗尽后**直接 return results，不检查是否所有节点都被执行**。若图中有环或存在悬空依赖，函数会返回**一份不完整的结果列表，且没有任何异常**。调用方 `execute_aot_plan()`（`agent/agent.py:484-498`）也不做检查。

**这是全仓库最危险的一处设计**：它把「结构错误」转化为「静默的错误结果」。正确写法必须以不变式收尾：

$$
\text{post-condition:}\quad \left\lvert \text{executed} \right\rvert = \lvert V \rvert
$$

违反即抛出携带未执行节点列表的显式异常。
:::

---

## 5. 关键实现：带校验的 DAG 规划与调度

```python
from collections import defaultdict, deque
from dataclasses import dataclass, field
@dataclass
class Node:
    id: str
    action: str
    deps: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)


@dataclass
class ExecState:
    PENDING, RUNNING, OK, FAILED = "pending", "running", "ok", "failed"


def validate_graph(graph: dict) -> list[Node]:
    """结构校验：唯一 id / 悬空依赖 / 无环。任一失败即抛出，绝不静默返回部分结果。"""
    raw = graph.get("nodes")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"graph must contain a non-empty 'nodes' list, got {raw!r}")

    nodes = [Node(id=n["id"], action=n["action"], deps=list(n.get("depends_on", [])))
             for n in raw]

    if len({n.id for n in nodes}) != len(nodes):
        raise ValueError("duplicate node ids")

    ids = {n.id for n in nodes}
    for n in nodes:
        dangling = set(n.deps) - ids
        if dangling:
            raise ValueError(f"node {n.id} has dangling deps {dangling}")

    cycle = find_cycle(nodes)
    if cycle:
        raise ValueError(f"cycle detected: {' -> '.join(cycle)}")
    return nodes


def find_cycle(nodes) -> list[str] | None:
    """Tarjan SCC 的简化版：返回任一有向环上的节点序列，无环则返回 None。 $O(V+E)$"""
    edges = {n.id: list(n.deps) for n in nodes}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)

    for start in edges:
        if color[start] != WHITE:
            continue
        stack = [(start, iter(edges[start]))]
        path = [start]
        color[start] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for dep in it:
                if color[dep] == GRAY:                       # 回边 -> 环
                    return path[path.index(dep):] + [dep]
                if color[dep] == WHITE:
                    color[dep] = GRAY
                    path.append(dep)
                    stack.append((dep, iter(edges[dep])))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
                path.pop()
    return None


def execute_dag(nodes, executor, max_workers: int = 8) -> list[dict]:
    """Kahn 拓扑调度：同一反链内的就绪节点并发执行。$O(V+E)$ 调度开销。"""
    deps = {n.id: set(n.deps) for n in nodes}
    by_id = {n.id: n for n in nodes}
    state = {n.id: ExecState.PENDING for n in nodes}
    results: list[dict] = []

    ready = deque(n.id for n in nodes if not deps[n.id])
    running: dict[str, Future] = {}

    while ready or running:
        while ready and len(running) < max_workers:          # ← 母本仓库缺失的并行发射
            nid = ready.popleft()
            state[nid] = ExecState.RUNNING
            running[nid] = executor.submit(by_id[nid].action)

        for nid, fut in list(running.items()):
            if not fut.done():
                continue
            del running[nid]
            try:
                results.append({"node_id": nid, "action": by_id[nid].action,
                                "result": fut.result(), "success": True})
                state[nid] = ExecState.OK
                newly = [m for m in nodes
                         if state[m.id] == ExecState.PENDING and nid in deps[m.id]]
                for m in newly:
                    deps[m.id].discard(nid)                 # ← 原地改写入度，Kahn 核心
                ready.extend(m.id for m in newly if not deps[m.id])
            except Exception as e:                          # ← 失败阻断后继，而非标记已执行
                results.append({"node_id": nid, "action": by_id[nid].action,
                                "error": str(e), "success": False})
                state[nid] = ExecState.FAILED
                for m in nodes:                             # 传递性阻断
                    if nid in deps[m.id]:
                        state[m.id] = ExecState.FAILED
                        results.append({"node_id": m.id, "action": m.action,
                                        "error": f"upstream {nid} failed",
                                        "success": False})

    # 不变式：要么全执行，要么显式失败——绝不留静默的部分结果
    unfinished = [nid for nid, s in state.items()
                  if s in (ExecState.PENDING, ExecState.RUNNING)]
    if unfinished:
        raise RuntimeError(f"graph not fully executed, pending={unfinished}")
    return results
```

三处关键修正：**① 环检测前置**（$O(\lvert V\rvert+\lvert E\rvert)$，不进入执行）；**② 失败传递性阻断**（三值状态取代母本的「失败也标已执行」）；**③ 并发发射**（`max_workers` 内凑批执行同反链节点，把 $T$ 从 $T_{\text{seq}}$ 压向 $L_{cp}$）。

---

## 6. 评测与可观测性

### 6.1 计划级指标

| 指标 | 定义 | 母本仓库 |
| :--- | :--- | :--- |
| 图合法率 | $\frac{\#\{\text{通过 } \text{validate}\}}{\#\{\text{规划调用}\}}$ | ✗ 无统计（仅 3 次重试） |
| 节点匹配率 $m$ | $\frac{1}{N}\sum_i \mathbb{1}[\text{action}_i = \text{action}_i^{\star}]$ | ✗ |
| 边精确率 | $\frac{\lvert E \cap E^{\star}\rvert}{\lvert E \rvert}$ | ✗ |
| 执行完成率 | $\frac{\lvert\text{executed}\rvert}{\lvert V\rvert}$ | ✗（**且不检查**） |
| 关键路径比 $\rho$ | $L_{cp} / T_{\text{seq}}$ | ✗（越小并行潜力越大） |
| 并行加速比 | $T_{\text{seq}} / T_{\text{par}}$ | ✗（恒为 1） |

::: info 公式速查：计划质量的三维刻画
$$
\text{结构正确性} = \mathbb{1}[\text{DAG}] \cdot \mathbb{1}[E \subseteq V \times V], \qquad
\text{内容正确性} = m, \qquad
\text{并行潜力} = \frac{L_{cp}}{T_{\text{seq}}} \in (0, 1]
$$

三者**不可互相替代**：结构对但 $m$ 低 → 计划无用；$m$ 高但 $\rho \to 1$ → 并行无用。**评测必须三维分开报**，否则会误把「结构合法率高」当作「规划能力强」。
:::

### 6.2 把 golden dataset 扩展到计划层

母本 `golden_datasets.py` 只覆盖**单步能力**（结构化输出 / 工具 / 决策 / 记忆），**完全没有计划层 golden**。这是当前评测体系最大的空洞：Lesson 08–10 的全部能力（计划、原子化、依赖）处于**零覆盖**状态。

建议补充的计划层 golden 结构：

```python
AOT_GOLDEN = [
    {
        "goal": "Research and write an article about AI agents",
        "must_include_actions": ["research", "write"],   # 必含节点（内容正确性）
        "must_have_dag": True,                            # 结构正确性
        "max_parallelism": 2,                             # 至少 2 个反链宽度
        "forbid_actions": ["delete_repository"],          # 安全约束
    },
    # 边界：单节点图（退化情况）
    {"goal": "Say hello", "must_include_actions": ["answer"],
     "must_have_dag": True, "max_parallelism": 1,
     "forbid_actions": []},
    # 边界：不可行目标（应返回空图或拒绝，而非编造节点）
    {"goal": "Do something impossible", "must_include_actions": [],
     "must_have_dag": True, "max_parallelism": 1,
     "forbid_actions": ["write", "research"]},
]
```

第三条尤其重要：**测试智能体是否会为不可行目标编造计划**，这是「约束优于能力」的直接检验。

### 6.3 Span 级遥测的扩展

AoT 执行应在 span 上标注节点 id 与边，以还原依赖拓扑：

```text
trace 5b2e10ff  (execute_dag, |V|=6, |E|=5)
├── span aot_graph  nodes=6  edges=5  L_cp/t_seq=0.50  validate=ok
├── span plan        node=1  action=research   duration=98_331 ms
├── span plan        node=2  action=outline    duration=91_004 ms   (depends_on=1)
├── span plan        node=3  action=write      duration=142_776 ms  (depends_on=1)
├── span plan        node=4  action=write      duration=137_220 ms  (depends_on=2)  ← 并行
└── span plan        node=5  action=review     duration=44_118 ms   (depends_on=3,4)
```

有了 $t_v$ 与 $\preceq$ 的 span 记录，即可**离线计算 $L_{cp}$ 与 $T_{\text{par}}$**，把「并行潜力 $\rho$」从推测变成实测。

---

## 7. 与本库其他领域的交叉点

| 交叉 | AoT 侧 | 本库笔记 |
| :--- | :--- | :--- |
| **误差累积同构** | $1-(1-\epsilon)^{\lvert V\rvert}$ | [Flow Matching 采样链 →](../../foundations/flow-matching-derivation.md) |
| **无回滚链条** | 原子动作 A3 公理 vs 长视频时序崩坏 | [Video DiT 时序一致性 →](../video/video-diffusion-dit.md) |
| **偏序调度 = 序列并行** | 反链并发 vs 序列并行 (SP) 切分 | [Wan2.1 8 卡 SP 优化 →](../../projects/video-gen-reproduction.md) |
| **约束优于能力** | `forbid_actions` vs CFG 硬约束 | [Flow Matching 条件流 →](../../foundations/flow-matching-derivation.md) |
| **DAG 校验 = 一致性检查** | 悬空依赖 / 环检测 | [SFD 语义先行扩散 →](../vision/sfd-semantic-first-diffusion.md) |

::: tip 交叉启发
**「原子动作的三条公理」可以直接翻译成可控视频生成的编辑算子设计准则。** 母本仓库的 A1/A2/A3 恰好对应图像编辑中反复出现的三个痛点：

| AoT 公理 | 视频编辑对应 | 在本库 Idea 中的体现 |
| :--- | :--- | :--- |
| A1 独立可验 | 编辑后结果无需看后续操作即可判对错 | [Proposal A 盲连续指令复原 →](../../ideas/proposal-a-blind-continuous-instruction.md)：轻量 mapper 预测干净指令，使每条指令**独立可解码** |
| A2 独立可测 | 每个编辑算子有单算子 benchmark | [Proposal C 尺度保真试穿 →](../../ideas/proposal-c-scale-aware-tryon.md)：7 维 foveated reward 给出可分解的量化判据 |
| A3 独立可滚 | 编辑可逆 / 不破坏未编辑区域 | [Proposal B 非对称保持 →](../../ideas/proposal-b-asymmetric-preservation-editing.md)：阻断 C-to-N 单向流 + VLM 残差梯度 |

**更尖锐的类比**：$\epsilon N$ 的误差累积公式说明「把一个复杂指令拆成 $N$ 个原子编辑」在 $N$ 足够大时必然退化。这正是 [Proposal A](../../ideas/proposal-a-blind-continuous-instruction.md) 选择**单 LoRA 六任务盲复原**（把 $N$ 压到 1，靠 mapper 承担映射）而非逐任务串联编辑的定量理由——**与 §2.3 中 $N \leq \ln\rho/\ln q \approx 4.5$ 的判据完全一致**。
:::
