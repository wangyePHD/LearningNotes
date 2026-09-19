# 🚀 实战与复现项目 (Projects & Benchmarks)

记录真实模型复现、开源代码深潜、训练踩坑与 Benchmark 评测。

## 项目列表

| 项目/实验名称 | 涉及技术栈 | 硬件环境 | 实验结论 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| **开源视频生成模型复现与优化** | Wan2.1, DiT, Diffusers, FSDP | 8x NVIDIA H800 | 显存与通讯瓶颈分析，推理时延压缩 35% | [查看报告 →](./video-gen-reproduction.md) |
| **LLM 7B 模型 LoRA 微调评测** | HuggingFace, PEFT, DeepSpeed | 1x NVIDIA A100 | 显存节约 63.7%，吞吐量提升 80% | [查看报告 →](./llm-finetuning-benchmark.md) |
