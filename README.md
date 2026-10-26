# AlgoNotes | 算法工程师笔记系统

一套专为算法研究与工程实践打造的个人数字花园系统。

- **存储**：纯文本 Markdown + LaTeX 公式（对 AI Agent 零负担，全平台通用）。
- **展示**：基于 VitePress，极速静态编译，高颜值代码高亮与 KaTeX 数学公式排版。
- **检索**：本地毫秒级离线全文分词搜索。
- **结构**：每篇笔记的分类信息写在 frontmatter（类型、标签、状态、摘要），侧边栏、索引表和标签矩阵构建时自动生成。
- **协同**：Agent 规则统一写在 `AGENTS.md`（`CLAUDE.md` 与 `.cursorrules` 指向它），Claude Code / DeepSeek / Codex 开箱即用。
- **部署**：配置 GitHub Actions，一键 `git push` 自动构建部署到 GitHub Pages。

## 本地运行

```bash
# 安装依赖
npm install

# 启动本地热更新预览
npm run dev

# 生产环境构建
npm run build
```
