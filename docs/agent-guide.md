# 🤖 Agent 自动化协作指南

本指南定义了 AI 智能体（如 Claude Code、DeepSeek、Codex 等）如何在本项目中安全、规范地**检索、新建、更新笔记与维护全局索引**。

---

## 1. 智能体工作流 (Agent Workflow)

当收到用户的学习总结、论文精读或实验记录任务时，Agent 应执行以下步骤：

```text
接收输入 (论文/公式/代码) 
   ──> 确定分类目录 (algorithms / projects / knowledge)
   ──> 提取核心 LaTeX 公式与结构化内容
   ──> 创建独立 Markdown 文件 (遵循 Frontmatter 规范)
   ──> 自动更新对应目录下的 index.md 索引表格
   ──> 自动将新文章追加到 .vitepress/config.mts 的 sidebar
```

---

## 2. 文件命名与 Frontmatter 规范

新建笔记时，请遵循以下规范：
1. **文件名**：一律使用小写英文单词加连字符，例如 `diffusion-dpm-solver.md`。
2. **头部元信息**：

```markdown
# 文章主标题

> **标签**：`标签1` `标签2`  
> **更新时间**：YYYY-MM-DD

---

## 1. 核心概述 / 动机
...
```

---

## 3. 数学公式排版规范

- **行内公式**：使用单个美元符号 `$E = mc^2$`，符号与内容间**不要**留多余空格。
- **独立公式块**：使用双美元符号包裹：
  ```markdown
  $$
  \mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
  $$
  ```
- **推导提示块**：善用 VitePress 的容器语法：
  - `::: tip 核心结论`
  - `::: info 推导细节`
  - `::: warning 注意事项`
  - `::: danger 避坑指南`

---

## 4. 自动注册到侧边栏 (`.vitepress/config.mts`)

创建新笔记后，Agent 需要在 `docs/.vitepress/config.mts` 中对应的 `sidebar` 数组中追加菜单项：

```typescript
{ text: '新笔记标题', link: '/algorithms/新笔记文件名' }
```
这样无需手动重启，本地热更新即可在侧边栏显示新篇目。
