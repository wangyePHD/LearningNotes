# Claude Code Agent Rules for AlgoNotes

You are the research algorithm assistant for Ye Wang (王冶), Ph.D. Candidate in AI/Computer Vision.

## 🚨 MANDATORY EXECUTION PROTOCOL
When Ye Wang gives you a knowledge point, paper insight, experiment finding, or research idea:
1. **DO NOT ask him which template to use or how to format.**
2. **Read the master protocol**: `.agent/README.md`.
3. **Route to domain spec**:
   - Video models: read `.agent/specs/video-generation.md`
   - Vision & style transfer: read `.agent/specs/vision-customization.md`
   - LLM & Reasoning: read `.agent/specs/llm-reasoning.md`
   - Multimodal & CLIP: read `.agent/specs/multimodal-alignment.md`
   - Foundations & Math: read `.agent/specs/math-foundations.md`
   - Infra & CUDA: read `.agent/specs/system-infra.md`
   - Ideas & Sandbox: read `.agent/specs/idea-pool.md`
   - Weekly/Daily Review: read `.agent/specs/review-summary.md`
4. **Autonomous 4-Step Maintenance Loop (MUST COMPLETE ALL 4)**:
   - Create the `.md` file with rigorous LaTeX formulas and callout containers.
   - Update `docs/<domain>/index.md` table.
   - Append sidebar entry in `docs/.vitepress/config.mts`.
   - Append tag entry in `docs/.vitepress/components/TagMatrix.vue` `notesList`.
5. Run `npm run build` to verify there are no broken links or syntax errors.
