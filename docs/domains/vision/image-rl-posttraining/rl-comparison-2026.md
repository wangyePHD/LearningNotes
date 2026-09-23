# 图像基模 RL 对比总表与演进主线 (2026-only)

> **标签**：`Vision` `RL` `Survey` `Comparison`
> **更新时间**：2026-09-23
> **参考来源**：本目录 A/B 系列笔记

---

## 1. 问题定义与控制目标

一页纸回答：2026 各家 RL 用什么算法、什么奖励、什么基模，以及六条演进主线。

## 2. 对比总表

| 模型/方法 | RL 算法 | 奖励 | 基模 | 关键创新 | 效果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GDRO | 组级直接优化（离线） | OCR/GenEval + corrected score | FLUX.1-dev + LoRA | 全离线、sampler 无关 | 同分 2–3.7× 效率 |
| FireRed | 非对称 DPO → NFT 在线 | VLM ensemble + Layout-OCR + 一致性 loss | MMDiT | ω/λ 双锚、semi-hard | 三基准开源 SOTA |
| Qwen-2.0 | GRPO + PE 的 GRPO | 5 任务奖励动态权重 | MMDiT + Qwen3-VL | CFG 杂交、Flywheel | 质感/一致性提升 |
| LLaDA | 无 RLHF（TwinFlow） | — | 6B DiT + dLLM | 阴性证据 | Bench 开源 SOTA |
| Swift | 并行专家 NFT → OPD | 4+3 维 + BT + 人脸 + OCR | 6B 单流 DiT | 分专家再统一 | 3.98→4.16 |
| ERNIE | DPO few-step → MT-DMD | 人审美 + 自研 Aes | 8B 单流 DiT | 双 anchor 0.35/0.15 | GenEval 0.89 |
| SeFi | 在线 NFT | 四维 + 能力标签 | 5B SFD 双潜 | 双潜 target、离散度过滤 | 文字 +1~1.6pt |
| i1 | 无 RL（recipe 对照） | — | 3B | 全开源基座 | 全开源 SOTA |

## 3. 六条演进主线

1. **DPO 必配锚**：FireRed、ERNIE、GDRO 三方收敛——diffusion DPO 不加 chosen 锚必 collapse；
2. **在线 RL 收敛到 NFT**：FireRed / Swift / SeFi 全用 NFT 系；GRPO 留守文本侧与自回归；
3. **奖励两铁律**：可验证奖励配"合理性"第二项防 hack；多奖励动态权重/组内标准化；
4. **分专家再统一**：Swift OPD、ERNIE MT-DMD、Qwen Flywheel——异构目标不硬塞单 policy；
5. **RL 是精修器**：SeFi 文字涨构图不动；LLaDA/i1 证明好数据可部分替代 RL；
6. **工程 checklist**：CFG 杂交、semi-hard/离散度过滤、few-step 短程、双轨评测、PE 解耦 RL。

::: warning 避坑要点
任何 RL 立项先回答：信用分到步了吗（稠密化）、奖励能被 hack 吗（第二项）、多目标打架吗（分专家）、评测双轨了吗（corrected score）。
:::

## 4. 待补

TreeGRPO / Mask-GRPO / f-divergence 原文、Nucleus-Image、Qwen-2.1、TBSM（2607.18198）。

## 5. 对比速查代码

```python
# 一句话选型
def pick_rl_route(has_online_budget, multi_objective, need_exploration):
    if multi_objective:
        return '并行专家 + OPD/MT-DMD（Swift/ERNIE）'
    if not has_online_budget:
        return 'GDRO 离线组优化'
    return 'DiffusionNFT 在线 + 双轨评测（FireRed/SeFi）'
```
