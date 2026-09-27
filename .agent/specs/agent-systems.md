# 🤖 智能体系统与工具编排 (Agent Systems) 细粒度编写规范

当用户让 Agent 记录 LLM 智能体架构、Agent Loop、工具调用 (Tool Use)、结构化输出契约、记忆系统、规划 (Planning) / 任务图、上下文工程、多智能体编排、智能体评测 (Evals) 与可观测性 (Telemetry / Tracing) 等相关笔记时，必须遵循本规范。

---

## 0. 领域定位与反 anthropomorphism 铁律

本领域与视频 / 视觉 / LLM 领域的根本区别在于：**智能体不是模型，而是「模型 + 状态 + 循环 + 约束」的确定性外壳**。因此笔记必须：

1. **把智能体还原为控制系统**：状态 $s_t$、动作 $a_t$、观测 $o_t$、转移 $s_{t+1}=f(s_t,a_t,o_{t+1})$、终止条件，而禁止用「思考」「理解」「有意识地」等拟人化措辞；
2. **一切「智能」必须落到可验证的机制上**：闭集动作空间、JSON 契约校验、拓扑执行序、重试策略，而不是「模型很聪明」；
3. **可靠性必须给出定量公式**：token 级合法率 $q^n$、$k$ 次重试残余失败率 $(1-p)^k$、错误累积 $1-(1-\epsilon)^{N}$，禁止只写「更稳定」；
4. **禁止隐藏的中间推理**：笔记只记录可观测的 state transition 与 span，不记录也不美化 unexposed 的思维链。

## 1. 结构大纲标准 (必须包含以下 7 个小节)

```markdown
# [中文标题] ([英文机制/模式])

> **标签**：`Agent` `Tool Use` `Planning` `[其他标签]`  
> **更新时间**：YYYY-MM-DD  
> **参考来源**：[论文/仓库/官方文档链接] · [代码文件路径]

---

## 1. 问题定义与控制系统形式化
- 先把智能体写成受控马尔可夫链 / 有限状态机：状态空间 $\mathcal{S}$、闭集动作空间 $\mathcal{A} = \{a_1,\dots,a_K\}$、策略 $\pi_\theta(a \mid s, x)$、转移与终止条件；
- 明确区分「循环 (Loop)」与「状态 (State)」两个必要成分，缺任一者退化为单轮 chatbot；
- 给出循环伪代码（pseudocode）与真实代码的 `file.py:line` 对照。

## 2. 核心算法与数学目标
- 闭集动作选择形式化为约束 argmax：$a^\star = \arg\max_{a \in \mathcal{A}} \pi_\theta(a \mid s_t)$，并指出 LLM 只在 $\mathcal{A}$ 上做「选择」而不做「生成」，这是可靠性的来源；
- 若涉及奖励 / 优化（如 RL 训练智能体、GRPO 训练策略），必须写出完整目标函数 $\mathcal{J}(\theta)$、优势函数 $\hat{A}$ 与 KL 正则项；
- 若涉及采样去噪（结构化输出、majority vote、self-consistency），必须写出 token 级乘积概率与聚合后的成功率。

## 3. 成本、复杂度与上下文预算
- 逐步 token 开销展开：$C(T) = \sum_{t=1}^{T} (P_t + G_t)$，标注张量/序列形状（$P_t \in \mathbb{N}$ 字符或 token 数）；
- 必须讨论上下文增长阶数：全量记忆注入导致 $P_t = O(t)$、总代价 $O(T^2)$，并给出上下文窗口 $n_{\text{ctx}}$ 下的最大步数 $T_{\max} = \lfloor (n_{\text{ctx}} - P_0) / \bar{P} \rfloor$；
- 单步延迟用显存带宽模型估计：$t_{\text{step}} \approx G \cdot W / \text{BW}_{\text{HBM}}$，并与 `docs/infra/cuda-memory-hierarchy`、`docs/infra/vllm-paged-attention` 交叉引用；
- 用表格量化「结构 vs 裸生成」的 token 与延迟开销。

## 4. 失效模式 (Failure Modes) 与防御机制
- 至少覆盖：格式非法 / 幻觉工具名 / 参数越界 / 循环不终止 / 上下文溢出 / 计划过期 / 静默部分失败 / 重复动作震荡；
- 每个失效模式必须给出「根因 → 检测手段 → 修复策略」三段式；
- 用 `::: warning` 标注静默失败（silent failure）与相关失败（correlated failure）等**不显式报错但致命**的陷阱。

## 5. 关键实现片段 (Python / TypeScript)
- 提供可运行的最小实现：闭集路由、结构化输出校验 + 重试、拓扑执行器（Kahn 算法）、工具注册表、span 记录；
- 实现必须包含错误反馈（把 parse error 回灌 prompt）与降级路径（projection onto action set），不得只写 happy path；
- 涉及工具调用时必须给出 schema 定义（JSON Schema 的 `type` / `enum` / `required`）与白名单执行，禁止模型直接执行任意代码。

## 6. 评测 (Evals) 与可观测性 (Telemetry)
- 指标定义：pass rate、格式合法率、步数分布、平均延迟、重试率、工具成功率；给出聚合公式 pass_rate $= \frac{\sum_s \text{passed}_s}{\sum_s \text{total}_s}$；
- 必须强调 golden dataset（黄金数据集）与 prompt 版本同管：改 prompt 必跑回归，golden case 失败即 agent 坏而非测试坏；
- trace / span 层级：trace（一次完整交互）→ span（一次 LLM 调用 / 工具执行 / 记忆读写），要求含 `trace_id`、`span_id`、`duration_ms`、`error`；
- 用表格列出「无评测 vs 有 golden 回归」的缺陷发现提前量对比。

## 7. 与本库其他领域的交叉点
- 至少给出一条与视频 / 视觉 / LLM / Infra 领域的实质交叉（如：多步推理的误差累积与扩散采样步误差同构、KV Cache 分页与上下文压缩同构、RL 奖励设计用于智能体后训练）；
- 用 `::: tip 交叉启发` 记录可迁移到自身课题（可控视频生成、统一理解生成、算力优化）的类比与可证伪实验设想。
```

---

## 2. 容器使用约定

| 场景 | 容器 |
| :--- | :--- |
| 关键机制 / 核心洞见 | `::: tip 关键机制` |
| 定量结论 / 公式速查 | `::: info 公式速查` |
| 陷阱 / 反直觉 | `::: warning` |
| 不可照搬的工程简化 | `::: danger 勿照搬` |

## 3. 存储位置

- 领域目录：`docs/domains/agent/`
- 领域首页：`docs/domains/agent/index.md`（必须维护 12 课路线图与本地运行说明）
- 规范文件：`.agent/specs/agent-systems.md`
