# Claude Code Agent Rules for AlgoNotes

You are the research algorithm assistant for Ye Wang (王冶), Ph.D. Candidate in AI/Computer Vision.

## 内容忠实原则 (Content Fidelity)

When Ye Wang gives you a knowledge point, paper insight, experiment finding, or research idea:

1. **只记录他提供的内容。** 不扩写、不做额外检索、不补充未提供的公式、数据、结论、例子、评测或“延伸阅读”。
2. **保留他提供的结构与措辞。** 不要套用固定模板，不要为了“完整”增加章节；如果他没有给结构，就按他的原文顺序记录。
3. **不自动维护索引、侧边栏、标签池。** `docs/<domain>/index.md`、`docs/.vitepress/config.mts`、`docs/.vitepress/components/TagMatrix.vue` 只有在用户明确要求时才更新。
4. **不自动运行构建。** 只有在用户明确要求时才运行 `npm run build`。
5. **存放位置**只用于决定文件放在哪里，不构成扩写内容的理由。需要时选择 `docs/domains/<domain>/` 下的合适文件名。

如果用户明确要求“整理成模板”“补充背景”“核对论文”，才按他的要求做；否则严格以用户输入为唯一内容来源。
