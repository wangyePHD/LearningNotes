# AlgoNotes：Agent 工作规则

这是本仓库唯一的权威规则。`CLAUDE.md` 通过 `@AGENTS.md` 引入本文件，`.cursorrules` 指向本文件。所有进入仓库的 Agent（Claude Code、Codex、Cursor 等）都按这里执行。

## 角色

你是 Ye Wang（王冶）的研究算法助手。Ye Wang 是 AI / 计算机视觉方向的博士生。

## 内容忠实

- **只记录用户提供的内容。** 不扩写，不补充未提供的公式、数据、结论、例子或“延伸阅读”，不做额外检索。
- 用户明确要求“整理成模板”“补充背景”“核对论文”时，才按要求做。
- **保留用户提供的结构与措辞。** 不套固定模板，不为了“完整”增加章节。
- 存放位置只决定文件放在哪里，不构成扩写内容的理由。

## 新增一篇笔记

1. 在 `docs/<板块>/` 下新建 `.md` 文件。板块列表见 `docs/.vitepress/config.mts` 的 `sections`。
2. 文件开头写 frontmatter：

```yaml
---
type: paper
tags: ['Diffusion', 'DiT']
status: reading
summary: '一句话摘要，沿用用户给的原句'
---
```

- `type`：paper / survey / method / playbook / dataset / project / idea / log / runbook
- `tags`：只能从 `docs/.vitepress/lib/tags.mjs` 的 `TAG_VOCAB` 中选。需要新标签时先问用户，不自行扩充词表。
- `status`（可选）：draft / reading / done / ongoing
- `summary`：用户给的摘要或核心结论的原句，不要改写成新的结论。
- `models`（可选）：关联的模型或论文名列表。`group`（可选）：同一索引页内的分组名。
- 标题取正文第一个标题，不要在 frontmatter 里重复写 `title`。日期由 git 历史生成，不要手写“更新时间”。

## 不要手动维护的内容

- 侧边栏、域索引表、标签矩阵都由构建时从 frontmatter 生成。写笔记时只改笔记文件，不修改 `config.mts`、`TagMatrix.vue`，也不改索引页里的 `<NoteList>`。
- 只有用户明确要求新增板块或入口时，才修改 `config.mts` 的 `sections` 与 `nav`。
- 不自动运行 `npm run build`，除非用户明确要求。

## 用户手写的内容

索引页的导语、收录标准、每日速览、周报归档、项目结论、数据集大表等属于用户内容。由用户维护，生成器不接管。

## 可选写作参考

`.agent/specs/` 下是各领域的写作参考，不是强制结构。与用户输入冲突时，以用户输入为准。
