# 📖 AlgoNotes 系统使用与协作指南

> 本指南旨在帮助你及 AI 助手（Claude Code / DeepSeek / Codex）最高效地使用、检索、撰写和维护这套个人数字研习系统。

---

## 1. 核心操作与交互快捷键

| 功能 | 快捷键 / 操作 | 说明 |
| :--- | :--- | :--- |
| **全站离线全文检索** | `Cmd + K` 或 `Ctrl + K` | 毫秒级模糊匹配任意公式变量、模型名称、代码片段及标签 |
| **目录收缩 / 展开** | 点击左侧分类标题 | 支持多级树形目录折叠与展开 |
| **本页大纲直达** | 点击右侧目录导航（TOC） | 自动跟踪页面滚动位置，快速定位二级/三级标题 |
| **深色 / 浅色模式切换** | 点击右上角太阳 / 月亮图标 | 针对公式与代码高亮优化的主题配色，夜间推导不伤眼 |
| **手机 / 平板端查阅** | 浏览器访问 GitHub Pages 链接 | 全站自适应移动端，单栏阅读无排版错乱 |

---

## 2. 日常研习与同步工作流

遵循标准的 **“线下集中读写，线上只读沉淀”** 闭环：

```bash
# 1. 进入本地知识库目录
cd ~/algorithm-notes

# 2. 启动本地热更新预览（编写时即时查看公式渲染）
npm run dev
# 浏览器打开 http://localhost:5173

# 3. 记录笔记或呼叫 Agent 完成推导后，提交并推送到 GitHub
git add .
git commit -m "feat(video): add Wan2.1 temporal DiT derivation"
git push
# 此时 GitHub Actions 会在后台自动完成编译，30 秒后在线网页自动更新！
```

---

## 3. 规范化笔记模板与 LaTeX 排版规范

系统对 LaTeX 公式和容器排版做了深度适配，新建笔记推荐遵循以下格式：

### 3.1 笔记头部元信息 (Frontmatter)

```markdown
# 笔记主标题

> **标签**：`Video Gen` `DiT` `Math Derivation`  
> **更新时间**：2026-09-19  
> **核心引用**：[Paper Link](https://arxiv.org/...) · [Code](https://github.com/...)

---
```

### 3.2 数学公式排版规范

- **行内公式**：使用单美元符号 `$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$`（前后紧贴，无冗余空格）。
- **独立公式块**：使用双美元符号包裹：
  ```latex
  $$
  \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1}\left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
  $$
  ```
- **复杂矩阵推导**：
  ```latex
  $$
  \mathbf{K} = \begin{bmatrix}
  k_{11} & k_{12} & \cdots & k_{1d} \\
  k_{21} & k_{22} & \cdots & k_{2d} \\
  \vdots & \vdots & \ddots & \vdots \\
  k_{n1} & k_{n2} & \cdots & k_{nd}
  \end{bmatrix}
  $$
  ```

### 3.3 语义容器 (Callouts)

用视觉区分推导细节、避坑经验与核心结论：

```markdown
::: tip 核心定理 / 结论
这里记录关键公式结论，例如方差收敛条件或训练最优超参。
:::

::: info 推导细节
这里展开详细的代数变形、期望方差拆解或连续积分证明。
:::

::: warning 注意事项
记录复现中容易忽略的细节（例如归一化方式、时序位置编码对齐）。
:::

::: danger 梯度爆炸 / OOM 避坑
记录显存溢出、数值下溢（如 Softmax 溢出）等严重踩坑经验。
:::
```

---

## 4. AI Agent (Claude Code / DeepSeek) 协作实战秘籍

你本地的 CLI Agent 可以作为你的“专职科研助理”，你可以直接在命令行对 Agent 说：

### 场景一：读论文并自动归档入库
> 🗣️ **对 Agent 说**：
> *“帮我精读这篇关于视频生成的最新论文 [链接/PDF]，提取其 3D 时空注意力的数学公式，整理成一篇标准格式笔记，放入 docs/domains/video/ 目录下，并更新该目录的 index.md 和侧边栏 config.mts。”*

### 场景二：推导补全与公式验证
> 🗣️ **对 Agent 说**：
> *“在 docs/foundations/flow-matching-derivation.md 里，帮我补全从边际速度场到条件流匹配的连续性方程证明细节，用 LaTeX 块写清楚。”*

### 场景三：实验日志与 Benchmark 生成
> 🗣️ **对 Agent 说**：
> *“根据我刚才在 8 卡 H800 上的测试日志，在 docs/projects/ 下新建一篇评测报告，对比 TP 与 SP 在不同序列长度下的显存占用和生成耗时表格。”*

---

## 5. 常见问题排查 (FAQ)

- **Q: 新增了一篇 Markdown，为什么左侧侧边栏没有出现？**  
  A: 请检查 `docs/.vitepress/config.mts` 对应的 `sidebar` 分支，只需追加一行 `{ text: '你的标题', link: '/对应路径' }` 即可。
- **Q: 公式渲染报错或未解析？**  
  A: 检查公式块前后是否空行隔开，反斜杠 `\` 是否有未闭合的括号匹配。
- **Q: 想要添加全新的研究模态（如 3D 具身智能）怎么办？**  
  A: 在 `docs/domains/` 下新建 `embodied/` 文件夹，然后在 `.vitepress/config.mts` 的 `nav` 和 `sidebar` 中各添加一项即可无缝集成。
