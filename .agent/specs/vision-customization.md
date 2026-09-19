# 🖼️ 图像与可控生成 (CV & Vision) 细粒度编写规范

当用户让 Agent 记录图像生成、风格迁移（Style Transfer）、结构可控（ControlNet）、个性化定制（Customization）等相关笔记时，必须遵循本规范。

---

## 1. 结构大纲标准 (必须包含以下 6 个小节)

```markdown
# [中文标题] ([英文方法名称])

> **标签**：`Vision` `Style Transfer` / `ControlNet` `[其他标签]`  
> **更新时间**：YYYY-MM-DD  
> **参考来源**：[论文标题/链接] · [代码仓库]

---

## 1. 问题定义与控制目标
- 阐明任务类型：是结构先验控制（边缘/姿态/深度），还是风格迁移、主体定制（Subject-driven）或局部编辑；
- 明确要攻克的核心难点（如避免内容漂移、过度平滑、身份一致性崩溃）。

## 2. 架构拓扑与特征注入机理
- 明确标注冻结参数（Locked Backbone）与可训练参数（Trainable Branch）的划分与参数量占比；
- 阐述先验注入的机制（如 Cross-Attention 注入、Zero-Conv 侧分支、Feature Swapping、LoRA/HyperNetwork 调制）。

## 3. 损失函数与数学稳定性推导
- 严格写出训练的总损失函数：
  $$ \mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{style}} \mathcal{L}_{\text{style}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}} $$
- 若涉及零卷积或特定层初始化，必须写出前向恒等性证明与梯度反向传播公式；
- 用 `::: info 证明细节` 展开公式推导。

## 4. 保真度与风格化权衡 (Trade-off Analysis)
- 深度分析“内容结构保真 (Content Preservation)”与“风格迁移强度 (Stylization Intensity)”的博弈曲线；
- 总结调节二者平衡的关键超参（如注入时间步区间 $t \in [t_{\text{start}}, t_{\text{end}}]$、注意力衰减因子）。

## 5. 核心控制层代码实现
- 提供 20~30 行关键控制模块（如零卷积注入、注意力交换算子）代码。

## 6. 避坑指南与评测基准
- 包含典型评测指标（如 CLIP-I 图像相似度、CLIP-T 文本一致性、LPIPS、ArtFID）；
- 用 `::: warning 避坑要点` 提示过拟合或生成伪影的常见触发场景。
```
