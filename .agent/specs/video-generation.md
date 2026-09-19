# 🎬 视频模型 (Video Intelligence) 细粒度编写规范

当用户让 Agent 记录视频生成、时序建模、时空注意力、Video DiT、Sora/Wan/Hunyuan 等相关笔记时，必须遵循本规范。

---

## 1. 结构大纲标准 (必须包含以下 6 个小节)

```markdown
# [中文标题] ([英文模型/算法简称])

> **标签**：`Video Gen` `DiT` `[其他细分标签]`  
> **更新时间**：YYYY-MM-DD  
> **参考来源**：[论文标题/链接] · [代码仓库]

---

## 1. 核心动机与时空建模难点
- 阐明该方案旨在解决视频领域的什么核心痛点（如时空一致性、运动伪影、因果泄漏、长序列计算爆炸）。

## 2. 潜空间与张量维度形式化定义
- 必须明确写出 VAE 压缩前后的张量维度映射：
  $$ x \in \mathbb{R}^{B \times 3 \times T \times H \times W} \xrightarrow{\text{3D VAE}} z \in \mathbb{R}^{B \times C \times t \times h \times w} $$
- 注明时间压缩比（如 $4\times$）与空间压缩比（如 $8\times$），以及每帧 Patch 划分后的 Token 总序列长度 $N = t \cdot h \cdot w$。

## 3. 时空主干网络与注意力机制推导
- 阐述时间与空间维度的解耦/联合方式（如 Spatial-Temporal Factorized Attention / Full 3D Attention）；
- 写出关键的注意力计算公式与时序位置编码（如 3D RoPE）形式。

## 4. 计算复杂度与显存推导
- 对比 Joint Attention 与该方法的浮点运算复杂度（FLOPs）：
  $$ \text{Complexity} = \mathcal{O}(\dots) $$
- 用 `::: danger 显存与通讯瓶颈` 容器指出大分辨率长序列下的 OOM 风险。

## 5. 核心算子 PyTorch / Einsum 实现片段
- 必须提供一段 20~40 行的纯 PyTorch / Einsum 核心注意力切分或前向逻辑代码，包含张量维度注释。

## 6. 复现避坑与评测考量
- 包含常用评测基准（如 VBench、FVD、时序运动平滑度指标）；
- 用 `::: tip 调优技巧` 总结显存节省（如 SP 序列并行、Activation Checkpointing）建议。
```

---

## 2. 语言与用词规范
- 涉及时间维度必须统一用 $T$ 或 $t$，空间维度用 $H \times W$ 或 $S$，特征通道用 $C$ 或 $d$；
- 避免模糊描述（如“占用很大”），必须量化为序列长度与显存关系。
