# AI Agent Operational Rules for AlgoNotes

You are an AI assistant managing the user's personal algorithm and research notes repository.

## Repository Layout
- `docs/domains/video/`: Video generation, temporal modeling, Spatio-Temporal attention, DiT.
- `docs/domains/vision/`: Image generation, ControlNet, CV tasks.
- `docs/domains/llm/`: LLM training, post-training, reasoning models (DeepSeek-R1), RLHF/GRPO.
- `docs/domains/multimodal/`: Multimodal alignment, CLIP, VLM, Audio.
- `docs/foundations/`: Core math, Flow Matching, Diffusion ODEs, Transformer backbone.
- `docs/infra/`: CUDA kernels, Triton, memory hierarchy, vLLM, DeepSpeed, Megatron.
- `docs/projects/`: Real-world reproductions, ablation studies, benchmark logs.
- `docs/.vitepress/config.mts`: Navigation and sidebar configuration.

## When asked to create or update notes:
1. **Always use Markdown + LaTeX**:
   - Inline math: `$formula$`
   - Block math: `$$ ... $$`
2. **Naming convention**:
   - Lowercase kebab-case: e.g., `wan2.1-video-architecture.md` inside appropriate directory.
3. **Format consistency**:
   - Title: `# Title`
   - Metadata block: `> **标签**：...  \n> **更新时间**：YYYY-MM-DD`
   - Use containers: `::: tip`, `::: info`, `::: warning`, `::: danger`
4. **Always maintain the index and sidebar**:
   - Update `index.md` table inside the corresponding directory.
   - Update `docs/.vitepress/config.mts` sidebar so the new note immediately appears in the navigation tree!
