# 🤖 Agent 细粒度规则与自动维护规范

知识库内置了完善的 **Agent 细粒度领域规范体系（Agent Specifications）**。无论是 Claude Code、DeepSeek、Codex 还是 Cursor，进入仓库后都会自动读取并遵守各领域的写作要求，实现全自动建档与索引挂载。

---

## 1. 领域细粒度规范索引 (.agent/specs/)

针对算法研究的不同模态，在根目录 `.agent/specs/` 下均预置了严格的写作规范与结构契约：

| 领域模块 | 规范文件 | 强制包含核心内容 |
| :--- | :--- | :--- |
| **🎬 视频生成与理解** | `.agent/specs/video-generation.md` | 3D VAE 压缩比、潜空间张量维度映射、时空解耦注意力推导、浮点复杂度 $O(T \cdot S^2)$、Einsum 算子代码、VBench 评测指标 |
| **🖼️ 图像与风格定制** | `.agent/specs/vision-customization.md` | 冻结主干 vs 训练分支比例、零卷积/交叉注意力特征注入机理、前向恒等性与梯度推导、保真度 vs 风格化博弈曲线 |
| **💬 语言模型与推理** | `.agent/specs/llm-reasoning.md` | GRPO 组相对策略优化数学目标、无 Critic 显存削减公式、抗作弊奖励工程（Reward Hacking）、慢思考顿悟现象分析 |
| **🎙️ 多模态与特征对齐** | `.agent/specs/multimodal-alignment.md` | 模态间隙 (Modality Gap) 分析、超球面特征映射、对称 InfoNCE 损失数学推导、特征崩塌防范与零样本迁移 |
| **📐 通用理论基石** | `.agent/specs/math-foundations.md` | 严格假设前提、步步有据的代数/微积分推导、反向传播梯度稳定性、Softmax 饱和证明、数值 Monte-Carlo 验证代码 |
| **⚡ 算力与系统工程** | `.agent/specs/system-infra.md` | Roofline 算术强度定量判定（Memory-Bound vs Compute-Bound）、显存分级带宽延迟、Triton/CUDA 优化实现、实测吞吐提升表 |
| **💡 科研灵感池** | `.agent/specs/idea-pool.md` | 疑问句反直觉假说、灵感触发点、核心科学假设 (Hypothesis)、审稿人视角潜在坑点与最小证伪实验 |

---

## 2. 自动化 4 步闭环维护机制 (Autonomous Loop)

当用户告诉 Agent 一段知识后，Agent 会全自动执行完整的 4 步流水线，**彻底免除人工维护工作**：

```text
               用户输入："今天推了下 Wan2.1 里的时序因果注意力，核心是..."
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ ① 读取对应领域的 spec 规范  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ ② 撰写高密度 Markdown + LaTeX │
                        │   (严格按 6 节大纲与容器组织) │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ ③ 自动完成系统三处协同挂载 │
                        │   ├── 更新目录 index.md   │
                        │   ├── 注册侧边栏 config.mts│
                        │   └── 注入 TagMatrix 标签池│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
               Agent 汇报："已按照视频领域规范完成撰写，已同步挂载至侧边栏与标签池。"
```

---

## 3. 多 Agent 入口自动适配

系统在根目录下同时预置了各大主流 Agent 的入口配置文件：
- `CLAUDE.md`：适配 **Claude Code CLI**，进入工作区自动注入系统提示词；
- `.cursorrules`：适配 **Cursor / Windsurf / Copilot**，打开项目自动遵守规范；
- `AGENT_GUIDE.md`：通用规范入口，兼容 **DeepSeek、OpenAI Codex、OpenCode** 等。
