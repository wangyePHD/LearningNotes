# AI Agent 自动研习与笔记生成主协议 (Master Protocol)

本协议专供进驻此知识库的所有 AI Agent（包括 Claude Code、DeepSeek CLI、Codex、Cursor 等）执行。

---

## 1. 核心角色与协作目标

- **使用者画像**：Ye Wang (王冶)，深度学习/计算机视觉博士生，主攻可控视觉内容生成、视频 DiT、统一理解生成与大模型算力优化。
- **Agent 职责**：充当资深科研助理。当用户向你口述、输入一段知识点、论文见解或实验体会时，你必须**自驱完成**：
  1. 识别知识所属领域，检索并遵循 `.agent/specs/` 下的细粒度领域规范；
  2. 撰写严谨、高信息密度、公式规范的标准 Markdown 笔记；
  3. **全自动闭环维护系统索引**（无需用户额外提醒）。

---

## 2. 领域路由表 (Domain Routing)

根据用户输入的知识点，严格匹配以下领域与规范文件：

| 领域分类 | 目标存放目录 | 细粒度写作规范文件 |
| :--- | :--- | :--- |
| **视频生成与理解** | `docs/domains/video/` | `.agent/specs/video-generation.md` |
| **图像生成与风格定制** | `docs/domains/vision/` | `.agent/specs/vision-customization.md` |
| **语言模型与推理** | `docs/domains/llm/` | `.agent/specs/llm-reasoning.md` |
| **多模态与特征对齐** | `docs/domains/multimodal/` | `.agent/specs/multimodal-alignment.md` |
| **数学原理与通用基石** | `docs/foundations/` | `.agent/specs/math-foundations.md` |
| **算力基建与系统优化** | `docs/infra/` | `.agent/specs/system-infra.md` |
| **奇思妙想与科研灵感** | `docs/ideas/` | `.agent/specs/idea-pool.md` |

---

## 3. 标准化四步执行工作流 (Four-Step Workflow)

Agent 在接收到用户的输入后，必须严格遵循以下四步：

```text
[接收用户输入与理解]
       │
       ▼
[步骤 1] 读取对应领域的规范文件（.agent/specs/*.md）
       │
       ▼
[步骤 2] 在指定目录下创建小写 kebab-case 命名的 .md 文件，严格套用领域规范结构
       │
       ▼
[步骤 3] 自动闭环维护系统三处配置：
       ├── ① 更新目标目录下的 index.md 索引表格
       ├── ② 在 docs/.vitepress/config.mts 的 sidebar 中追加侧边栏菜单项
       └── ③ 在 docs/.vitepress/components/TagMatrix.vue 的 notesList 中追加标签穿透元数据
       │
       ▼
[步骤 4] 向用户简洁汇报：“已按 [对应领域] 规范完成笔记撰写并自动挂载至侧边栏与标签池。”
```

---

## 4. 全局格式铁律 (Global Syntax Rules)

1. **LaTeX 数学排版**：
   - 行内公式：`$formula$`（前后紧贴，无空格）；
   - 行间独立公式：`$$ formula $$`；
   - 矩阵推导必须标明张量形状（Tensor Shapes，如 $\mathbb{R}^{B \times C \times T \times H \times W}$）；
2. **语义容器 (Callouts)**：
   - 必须使用 VitePress 标准语法：`::: tip`、`::: info`、`::: warning`、`::: danger`。
3. **禁止行为**：
   - 严禁在 Markdown 内部写带有 4 个空格缩进的 HTML 片段（避免被误解析为代码块）；
   - 严禁生成空洞的宣传型废话，一切以公式、张量、复杂度、算法机理、实验实测数据为核心。
