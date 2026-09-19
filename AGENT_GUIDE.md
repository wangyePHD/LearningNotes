# AI Agent Operational Rules for AlgoNotes

You are an AI assistant managing the user's personal algorithm and research notes repository.

## Repository Layout
- `docs/algorithms/`: Math derivations, core theory, ML algorithms, paper reading notes.
- `docs/projects/`: Reproductions, benchmark experiments, project logs.
- `docs/knowledge/`: Low-level system knowledge, CUDA kernels, HPC, distributed training.
- `docs/.vitepress/config.mts`: Navigation and sidebar configuration.

## When asked to create or update notes:
1. **Always use Markdown + LaTeX**:
   - Inline math: `$formula$`
   - Block math: `$$ ... $$`
2. **Naming convention**:
   - Lowercase kebab-case: e.g., `flash-attention-v2.md` inside `docs/<category>/`.
3. **Format consistency**:
   - Title: `# Title`
   - Metadata block: `> **标签**：...  \n> **更新时间**：YYYY-MM-DD`
   - Use containers: `::: tip`, `::: info`, `::: warning`, `::: danger`
4. **Always maintain the index**:
   - Update `docs/<category>/index.md` table.
   - Update `docs/.vitepress/config.mts` sidebar so the new note appears in the web navigation!
