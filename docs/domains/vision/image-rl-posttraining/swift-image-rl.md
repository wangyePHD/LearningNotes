# Swift-Image 并行专家 RL 与多教师蒸馏 (6B 小算力范本)

> **标签**：`Vision` `RL` `DiffusionNFT` `Distillation` `MoE-RL`
> **更新时间**：2026-09-23
> **参考来源**：[arXiv:2608.20334](https://arxiv.org/abs/2608.20334)

---

## 1. 问题定义与控制目标

6B 小模型、243K GPU-h 预算下推统一生成+编辑极限。单 policy 混合异构目标 → 梯度干扰、跷跷板、长训 collapse。原则：**Specialize-Consolidate**（先分专家到天花板，再蒸馏统一）+ 推理-渲染解耦。流程：`SFT → 并行专家 RL → 多教师 OPD → 剪枝 3B + 少步蒸馏`，另有独立 PE（SFT+GRPO）。

## 2. 架构拓扑与特征注入机理

- **冻结参数**：各域专家（蒸馏时冻结做 teacher）、PE 训练时冻结 DiT；
- **可训练参数**：6B 单流 DiT；专家划分 1×T2I + 1×通用编辑（mixed，做 student backbone）+ N×欠优化子域专家；
- **注入机理**：图像侧全用 DiffusionNFT（不用 GRPO/DPO）；task-aware reward routing 按任务从奖励池组合，专家只见内聚 reward 分布。

## 3. 损失函数与数学稳定性推导

奖励系统（在线实时）：T2I 4 维（对齐/分类美学/视觉质量/风格一致）+ Edit 3 维（遵循/参考一致/质量）+ BT preference 辅助 + ArcFace + PP-OCRv6，域专家按子域定制 criteria。

**CFG 非对称**（与 Qwen-Image-2.0 同构）：rollout 开 CFG、优化关 CFG，guided 能力隐式蒸进 unguided policy。

多教师 OPD：student 初始化自 mixed-task RL 通才，冻结专家做 teacher，student 在线采样、按域路由蒸馏；复用三处（6B 统一、跨容量 6B→3B、few-step 统一 + few-step NFT 精修）。

::: info PE 的 GRPO
DiT 冻结；双视角奖励（改写文本 IF+KR / 真渲染 IF+VQ），**组内标准化后融合成 relative advantages**，不加权 raw scores。PE 外挂其他开源模型也涨分。
:::

## 4. 效果权衡

RL 后 overall 3.98→4.16，REDEdit 3.98→4.34；few-step Turbo 4.20 **反超** multi-step teacher 4.16；剪枝 3B 几乎无损。PE：常规编辑 +0.16/+0.17，T2I +~5pts。

::: warning 避坑要点
RL/OPD/PE 的 prompts 数、steps、LR 等超参**均未公开**，复现需自扫；reward VLM backbone 型号未披露，自建 rubric 时先做偏置审计（见 ERNIE 美学审计）。
:::

## 5. 核心控制层代码实现

```python
# 并行专家路由：按任务域选奖励组合，各专家独立优化
def route_expert(task_domain):
    if task_domain == 't2i':
        return T2I_EXPERT, T2I_REWARDS  # 4维
    elif task_domain in UNDEROPTIMIZED:
        return EDIT_EXPERTS[task_domain], EDIT_REWARDS[task_domain]
    return EDIT_GENERALIST, EDIT_REWARDS_BASE  # student backbone

# 多教师 OPD：student 在线采样 -> 域匹配 teacher 蒸馏
def opd_step(student, teachers, batch):
    x = student.sample_online(batch.prompts)       # on-policy
    t = route_expert(batch.domain)                  # 匹配 teacher
    return distill_loss(student(x), teachers[t](x).detach())
```

## 6. 避坑指南与评测基准

- **评测**：自建 PiGeneral / PiPractical / REDEdit 对照 + 跨模型 PE 外挂验证；
- ** checklist**：分专家前先确认单 policy 跷跷板实锤；OPD student 必须用通才初始化（防 specialist 偏科）；few-step 阶段再做一次 NFT 精修。
