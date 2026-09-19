# 🤖 Agent 自动化协作指南

本指南定义了 AI 智能体（如 Claude Code、DeepSeek、Codex 等）如何在本项目中安全、规范地**检索、新建、更新笔记与维护全局索引**。

---

## 1. 知识库目录结构规范

系统按「**垂直研究领域**」与「**底层基石工程**」双重解耦：

- `docs/domains/video/`: 视频生成与理解、时空注意力、Video DiT、SVD/Wan/Sora 等。
- `docs/domains/vision/`: 图像生成、ControlNet、分类检测分割、底层视觉。
- `docs/domains/llm/`: 语言模型预训练、SFT、GRPO/RLHF、推理涌现、Agent 机制。
- `docs/domains/multimodal/`: 多模态表征对齐 (CLIP)、VLM、语音与音视频统一模型。
- `docs/foundations/`: 通用理论与架构基石（数学推导、Flow Matching/Diffusion 方程、Transformer/Mamba 通用骨干）。
- `docs/infra/`: 算力与系统工程（CUDA 算子优化、Triton、显存分级、vLLM 推理加速、NCCL 分布式通信）。
- `docs/projects/`: 个人代码复现、Benchmark 评测与工业踩坑经验。

---

## 2. 智能体工作流 (Agent Workflow)

当收到用户的学习总结、论文精读或实验记录任务时，Agent 应执行以下步骤：

```text
接收输入 (论文/公式/代码/对话) 
   ──> 确定对应目录 (例如视频相关放入 docs/domains/video/)
   ──> 提取核心 LaTeX 公式与结构化内容
   ──> 创建小写 kebab-case 命名的 Markdown 文件
   ──> 自动更新对应目录下的 index.md 索引表格
   ──> 自动将新文章追加到 docs/.vitepress/config.mts 的对应 sidebar 数组中
```

---

## 3. 数学公式与容器排版规范

- **行内公式**：使用单个美元符号 `$E = mc^2$`。
- **独立公式块**：使用双美元符号包裹：
  ```markdown
  $$
  \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1}\left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
  $$
  ```
- **推导提示块**：善用 VitePress 的容器语法：
  - `::: tip 核心结论`
  - `::: info 推导细节`
  - `::: warning 注意事项`
  - `::: danger 避坑指南`
