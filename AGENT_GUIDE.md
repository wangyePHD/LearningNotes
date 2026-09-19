# AI Agent Master Operational Guide for AlgoNotes

欢迎各类 AI Agent（Claude Code、DeepSeek CLI、Codex、Cursor、Windsurf 等）接管与协助维护本知识库。

## 核心路由总览
当用户向你提供学术思考、公式推导、论文理解或代码片段时，请**自动读取并严格执行**以下规范：

- 📖 **总控协议**：`.agent/README.md`
- 🎬 **视频生成与理解**：`.agent/specs/video-generation.md`
- 🖼️ **图像与可控生成**：`.agent/specs/vision-customization.md`
- 💬 **语言模型与推理**：`.agent/specs/llm-reasoning.md`
- 🎙️ **多模态与特征对齐**：`.agent/specs/multimodal-alignment.md`
- 📐 **数学理论与基石推导**：`.agent/specs/math-foundations.md`
- ⚡ **算力基建与系统优化**：`.agent/specs/system-infra.md`
- 💡 **科研灵感池记录**：`.agent/specs/idea-pool.md`
- 📅 **研习复盘与周报**：`.agent/specs/review-summary.md`

## 自动闭环维护铁律 (4 步)
Agent 必须一次性完成全链路维护，无需用户二次操作：
1. **生成 Markdown 笔记**（含 LaTeX 独立公式与语义 Callout 容器）；
2. **更新模块索引**（`docs/<domain>/index.md`）；
3. **注册侧边栏**（`docs/.vitepress/config.mts`）；
4. **注入标签穿透池**（`docs/.vitepress/components/TagMatrix.vue`）。
