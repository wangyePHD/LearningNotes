---
type: paper
tags: ['Distillation', 'Flow Matching', 'T2I', 'GRPO', 'Fine-tuning']
status: reading
summary: 'Flow-OPD reformulates multi-task alignment via dense supervision on self-generated trajectories.'
models: ['Flow-OPD', 'SD3.5-Medium', 'Flow-GRPO']
---

# Flow-OPD：把多个同源专家的能力合进一个图像生成模型

- 论文：[Flow-OPD: On-Policy Distillation for Flow Matching Models](https://arxiv.org/abs/2605.08063)，本文核对 **arXiv v5**（2026-05-24）。
- 官方代码：[CostaliyA/Flow-OPD](https://github.com/CostaliyA/Flow-OPD)。本文读取用户提供的 `/gpfs/wy/Projects/SeFi-Image-5B/DPO/refs/Flow-OPD-main.zip`，ZIP 标注的提交为 `434ab16911fc580f51e22107151915846222eb6a`。
- 下文源码路径、行号均对应这份 ZIP。标为“原代码”的片段只摘取相关分支，不是独立可运行脚本；必要的换行与缩进调整不改变计算。伪代码单独标明。
- **核验边界**：读过论文与源码，没有运行训练或复现指标。全部实验表格及配图均从原始 PDF 截取，不手工重画结果表。

## 一、通俗理解：学生自己走，专家分工指路

### 1.1 它想解决什么？

希望一个文生图模型既能画对“三只猫、两只狗”的数量和位置，又能写对招牌上的文字，还要好看。

直接用几个奖励一起训，容易出现“跷跷板”：字写好了，构图退步；更好看了，数量又不对。最终图片的一个分数，只告诉模型结果好不好，没有直接告诉它每一步该怎么改。

**Flow-OPD 的做法：先分别练出专家，再让学生学专家“具体怎么画”，而不只是学“哪张图分数高”。**

![论文原始 teaser：Figure 1，多任务训练曲线与指标对比，PDF 第 2 页](/flow-opd-fig01-teaser.png)

*论文 Figure 1 原图截图。左图比较普通 GRPO、冷启动后继续 GRPO、Flow-OPD 的训练曲线；右图比较构图、文字和画质/偏好指标。它是结果总览，不是方法流程图。DeQA / PickScore 在图中经过归一化，不要和后面表格的原始分数直接比较。*

### 1.2 用“咖啡店招牌写 HELLO”理解一轮训练

1. **学生自己生成**：从噪声走向图片，保存途中还没画完的带噪 latent。latent 可以先理解成模型内部正在加工的画布。
2. **请 OCR 老师看同一份画布**：把学生当前的 latent、时间步和 prompt 交给老师，问“如果是你，这一步准备往哪里改？”
3. **学生学习这个方向**：比较双方预测，缩小差距。不是等到图片画完后，才给它一个 OCR 分数。
4. **另外加一根画质安全绳**：在各类训练数据上加入参考模型的约束，避免学会功能后整体画质跑偏。论文把它叫 MAR。

教学示意，非论文原图：

```text
SD3.5-M → 分别训练构图 / OCR / 偏好等专家
                         │
             SFT 或专家权重合并，初始化学生
                         │
                 学生自己采样，保存 xₜ
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
     学生预测这步怎么走        对应任务老师预测怎么走
            └────────────┬────────────┘
                         ▼
                    缩小预测差距
                ＋ 全数据上的 MAR 约束
                         ▼
                    更新学生 LoRA
```

**On-policy 在哪里？** 监督状态来自当前学生自己走过的轨迹，不是固定的老师成品图数据集。

**推理需要几个模型？** 只用学生，不需要把老师一起部署。

### 1.3 与 Any-OPD、NLD-Uni 编辑侧的根本差别

- **Flow-OPD**：主实验的老师和学生都来自 **SD3.5-Medium，约 2.5B**。师生使用兼容的 latent、调度和预测表示，可以在同一个中间状态直接比较更新方向。重点是**同源专家能力整合**，不是把大模型压成小模型。
- **Any-OPD**：例如 SD3.5 学生与 FLUX 老师，VAE 和内部空间不同，需要经图片、老师精修及 DINO 特征搭桥。
- **NLD-Uni 编辑侧**：离散 VQ 学生与连续 latent 老师，通过像素图像和精修目标连接。

这里说的是这些具体工作，不能泛化为“所有旧 OPD 方法都是异构”。

## 二、方法模块与核心源码

### 2.1 先获得单项专家：奖励负责练专长

论文 §6.1：GenEval、OCR、PickScore 使用 Flow-GRPO 的官方专家检查点；DeQA 老师另行训练，正文给出的 DeQA : PickScore 奖励比例为 4:6。

单项专家阶段的逻辑仍是：模型生成一组图片，任务评分区分好坏，再用 GRPO 式更新强化高奖励行为。**奖励模型负责评分，专家老师则是能预测去噪方向的生成模型，两者不是一回事。**

仓库 `scripts/train_sd3.py:1178–1206` 的 `task_only` 路径包含这类更新。原代码摘录，省略中间的其他 loss 分支：

```python
advantages = torch.clamp(
    sample["advantages"][:, j],
    -config.train.adv_clip_max,
    config.train.adv_clip_max,
)
```

```python
ratio = torch.exp(log_prob - sample["log_probs"][:, j])
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(
    ratio,
    1.0 - config.train.clip_range,
    1.0 + config.train.clip_range,
)
policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

- `sample["advantages"]`：来自最终图片任务奖励的优势值；不是每一步重新给图片评分。
- `ratio`：新模型与采样时旧模型对这一步转移给出的概率密度之比。
- `clip`：限制一次更新过大。

这是通用训练器中的相关逻辑，**不是已经核实完所有专家正式训练超参的独立复现配方**。后续 OPD 可以直接加载已有专家，不必从头再练。

### 2.2 冷启动：先让学生站到较好的起点

论文 §5.1 给两条路线：SFT 或 Model Merging。冷启动不是完整方法的终点，后面仍要做学生轨迹上的在线蒸馏。

#### A. Model Merging：合并专家 LoRA

`scripts/merge.py:66–79`，平均分支的原代码：

```python
merged = OrderedDict()
for key in all_weights[0].keys():
    tensors = [w[key] for w in all_weights]
    if merge_mode == "average":
        merged[key] = torch.stack(tensors).mean(dim=0)
```

它对每个同名 LoRA 张量取平均；脚本还支持求和、加权。`scripts/single_node/merge.sh` 示例合并 GenEval、OCR、PickScore 三份 LoRA。

**这不是把多个老师每一步的预测取平均，也不能直接当成完整模型参数增量平均的同义词。** 这里实际操作的是保存下来的 LoRA 张量。

学生在 `scripts/train_sd3_opd_mix.py:762–767` 加载冷启动结果：

```python
if config.train.lora_path:
    pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
    pipeline.transformer.set_adapter("default")
else:
    pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
```

因此，**需要显式设置 `config.train.lora_path`，不能以为多教师示例默认已经加载合并后的学生**。

#### B. SFT：把参考模型生成的结果作为学习目标

论文描述用专家样本建立初始学生。仓库 `scripts/train_sd3_sft.py` 提供参考 adapter 采样、终态 latent 再加噪训练的实现。

原代码分别位于 `:615–620`、`:815`、`:835–836`：

```python
if global_step>0 and global_step%config.train.ref_update_step==0:
    copy_learner_to_ref(transformer)
with autocast():
    with torch.no_grad():
        pipeline.transformer.set_adapter("ref")
```

```python
noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
```

```python
target = noise - model_input
fm_loss = ((model_pred.float() - target.float()) ** 2).mean(dim=(1, 2, 3))
```

- `ref`：负责采样的参考 adapter；采样终态 latent 被存为后面的 `model_input`。
- 给这个终态重新加噪，学生预测 flow 速度，目标是 `noise - model_input`。
- 完整脚本还包含基于奖励的样本选择和基模约束，不只是上面两行裸 FM loss。

**边界**：这里只确认脚本中的计算流程；默认配置如何装入各个正式专家、如何组成论文的多专家 SFT 冷启动，仍需另核对，不能把示例直接当论文完整冷启动配置。

### 2.3 On-policy 采样：保存学生真正走过的状态

入口为 `scripts/train_sd3_opd_mix.py:1220–1241`，内部调用 `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`。

该函数在 `:10` 标有 `@torch.no_grad()`，整个 rollout 不保留反向计算图。每一步用学生预测，再进行随机 SDE 转移，保存状态。以下为该文件 `:163–173` 的原代码：

```python
latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
    self.scheduler,
    noise_pred.float(),
    t.unsqueeze(0),
    latents.float(),
    noise_level=noise_level,
)

all_latents.append(latents)
all_log_probs.append(log_prob)
all_prev_latents_mean.append(prev_latents_mean)
```

实际随机转移在 `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py:62`：

```python
prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise
```

先算平均落点，再加随机噪声。代码按噪声水平 `sigma` 从大到小走，所以这里 `dt < 0`，噪声项使用 `sqrt(-dt)`；不要直接把它和论文“从噪声到数据”的时间符号混用。

采样轨迹最后拆成相邻状态对（主训练文件 `:1293–1295`）：

```python
"latents": latents[:, :-1],
"next_latents": latents[:, 1:],
"log_probs": log_probs,
```

**为什么不边生成边反传整条链？** 这里先把学生经历存下来，训练时重新计算当前一步的学生预测。它不需要像 Any-OPD 那样，为了最终图像特征 loss 带梯度重跑后半段去噪并穿过 VAE。

### 2.4 任务路由：数据集决定老师，不是语义分类器选老师

多教师示例是 `config/grpo.py:mix_opd_8gpu`：OCR 数据绑定 OCR LoRA，GenEval 数据绑定 GenEval LoRA。当前配置中 OCR 占 1 个 epoch，GenEval 占 3 个 epoch；PickScore 项被注释掉。

`scripts/train_sd3_opd_mix.py:872–882` 的原代码：

```python
def get_current_dataset_idx(epoch):
    """根据 epoch 返回当前应该训练的数据集索引"""
    total_epochs_in_cycle = sum(ds["epochs_per_cycle"] for ds in alternate_datasets)
    epoch_in_cycle = epoch % total_epochs_in_cycle

    cumulative_epochs = 0
    for idx, ds in enumerate(alternate_datasets):
        if epoch_in_cycle < cumulative_epochs + ds["epochs_per_cycle"]:
            return idx
        cumulative_epochs += ds["epochs_per_cycle"]
    return 0
```

进入当前数据集的训练后，在 `:1485–1486` 取出对应老师：

```python
if is_opd_mode and is_alternate_mode:
    current_ref_transformer = ref_transformers.get(current_ds_name)
```

因此，“硬路由”的源码含义很直接：**这一轮在训 OCR 数据，就找 OCR 老师；下一轮训 GenEval 数据，就找构图老师。不是每条 prompt 先跑一遍智能分类器。**

老师由同一 SD3.5 基模加载对应专家 LoRA，预测时位于 `torch.no_grad()` 中；老师不会跟着学生一起优化。

### 2.5 密集监督：同一份带噪画布，比较师生建议的下一步

#### A. 同状态、同时间、同条件

主训练文件 `:1523–1535` 为学生前向；`:1565–1577` 为老师前向。默认开启 CFG，二者均计算正、负分支再合成。

以下为老师前向的原代码：

```python
ref_noise_pred = ref_trans(
    hidden_states=torch.cat([sample["latents"][:, j]] * 2),
    timestep=torch.cat([sample["timesteps"][:, j]] * 2),
    encoder_hidden_states=embeds,
    pooled_projections=pooled_embeds,
    return_dict=False,
)[0]
ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
ref_noise_pred = (
    ref_noise_pred_uncond
    + config.sample.guidance_scale
    * (ref_noise_pred_text - ref_noise_pred_uncond)
)
```

- `sample["latents"][:, j]`：学生实际走到的第 `j` 个状态。
- `timestep`：对应噪声水平。
- `embeds` / `pooled_embeds`：同一组文本条件。
- `noise_pred` 名字沿用了扩散代码习惯；在这里承担的是 **flow 速度预测**，不能只看名字把它认作普通噪声回归。

这里对齐的是 **CFG 合成后的预测**，不是 Rethinking CFG 里的 PDM 分支级监督。

#### B. 从速度转为“平均落点”

师生预测分别送进同一个 `_sde_step()`，得到 `prev_sample_mean` 与 `prev_sample_mean_ref_lora`。均值的原计算位于 `sd3_sde_with_logprob.py:53`：

```python
prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
```

不要被长度吓到，核心只有一件事：**当前状态固定后，下一步的平均落点由预测速度决定。**

比较的是平均落点，而不是两个独立随机落点，这样不会把采样噪声本身当成学生的错误。

在论文 §5.1.1、Eq.10–12 中，共享协方差的高斯转移 KL 可以写成均值差平方，再写成带时间权重的速度差平方。论文 Eq.15 的训练目标为：

$$
\mathcal L_{\mathrm{OPD}}
=\mathbb E_{x_t\sim\rho_t^\theta}
\left[w(t)\left\|v_\theta(x_t,t,c)-\operatorname{sg}\big(v_T(x_t,t,c)\big)\right\|_2^2\right].
$$

- $x_t$ 来自学生自己的采样。
- $v_T$ 是本任务对应的老师预测。
- $\operatorname{sg}$ 表示老师只提供目标，不接收梯度。
- $w(t)$ 调整不同时间步的权重。

#### C. `gkd` 分支：把差距直接当 loss

`scripts/train_sd3_opd_mix.py:1729–1736`，原代码：

```python
elif reward_mode == "gkd":
    # GKD: directly use mean(kl_reward) as loss, no policy loss / ratio
    if prev_sample_mean_ref_lora is not None:
        kl_reward_gkd = ((prev_sample_mean - prev_sample_mean_ref_lora) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
    else:
        kl_reward_gkd = torch.zeros_like(prev_sample_mean.mean(dim=(1, 2, 3), keepdim=True))
    gkd_loss = torch.mean(kl_reward_gkd)
    policy_loss = torch.tensor(0.0, device=gkd_loss.device, dtype=gkd_loss.dtype)
```

对于 `[B, C, H, W]` 的 latent：`mean(dim=(1, 2, 3))` 把一份预测里的空间/通道差距汇总，然后再对 batch 求平均。

所以“密集”指**差距来自每一步整块 latent 的预测**；不是说最终 loss 不再是标量。这里最后仍然得到一个标量 loss。

**实现口径提醒**：源码使用维度平均，并除以 `2 * std_dev_t**2`；实际 SDE 转移方差还包含 `|dt|`。因此不能不核对归约和步长因子，就把这行逐项称为论文 Eq.10 的完整、未缩放 KL。这里确认的是实际计算，而不是替代码补公式。

### 2.6 MAR：再加一份全数据上的参考约束

任务老师回答“字怎样写对 / 数量怎样画对”；MAR 参考模型约束“学功能时别把整体生成行为带坏”。论文 Eq.16 使用审美老师，在所有任务数据上额外提供监督。

主训练文件 `:915–925`：配置了 `train.mar_lora` 才加载额外参考模型。原代码摘录：

```python
mar_transformer = None
if config.train.get("mar_lora") and config.train.mar_lora != "":
    mar_pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    mar_pipeline.transformer.requires_grad_(False)
    mar_pipeline.transformer.to(accelerator.device)
    mar_transformer = PeftModel.from_pretrained(mar_pipeline.transformer, config.train.mar_lora)
    mar_transformer.set_adapter("default")
    mar_transformer.eval()
```

若 `mar_lora` 留空，`:1633` 走 `with transformer.module.disable_adapter():`，用禁用学生 LoRA 后的基模预测作参考。**所以默认有基模约束，不等于默认已经启用论文的审美专家 MAR。**

`gkd` 路径的原损失计算位于 `:1745–1752`：

```python
if reward_mode == "gkd":
    loss = gkd_loss
    if config.train.beta > 0:
        kl_loss = ((prev_sample_mean - prev_sample_mean_ref_base) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
        kl_loss = torch.mean(kl_loss)
        loss = gkd_loss + config.train.beta * kl_loss
    else:
        kl_loss = torch.tensor(0.0, device=gkd_loss.device, dtype=gkd_loss.dtype)
```

`prev_sample_mean_ref_base` 这个变量名也容易误导：加载了 `mar_lora` 时，它装的是额外参考模型的均值；没有加载时，才是基模均值。

### 2.7 必须区分：示例默认是 `kl_only`，不是 `gkd`

`config/grpo.py:mix_opd_8gpu` 默认设置：

```python
config.train.beta = 0.04
config.train.kl_reward_level = "step_wise"
config.train.kl_scale = -1
config.train.reward_mode = "kl_only"  # no outcome reward, pure KL-based OPD
config.train.ema = True
```

这一分支不是直接令 `loss = gkd_loss`，而是把与老师的预测差距变成负的逐步优势值。原代码分别位于 `:1667–1690`、`:1754–1765`，摘录相关计算：

```python
kl_reward = ((prev_sample_mean - prev_sample_mean_ref_lora) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
```

```python
advantages = config.train.kl_scale * kl_reward
advantages = torch.clamp(
    advantages,
    -config.train.adv_clip_max,
    config.train.adv_clip_max,
)
```

```python
ratio = torch.exp(log_prob - sample["log_probs"][:, j])
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(
    ratio,
    1.0 - config.train.clip_range,
    1.0 + config.train.clip_range,
)
policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

可以把两条路径记成：

- **`gkd`**：老师差距直接进入回归 loss；对应前面讲论文时使用的直观主线。
- **`kl_only`**：老师差距形成负奖励/优势，仍通过概率比和 clipping 更新；这是当前示例的默认路径。

源码里 `kl_only` 的这份 `advantages` 没有显式 `detach()`，不能直接把它当成“固定优势值”的教科书 PPO 实现。本文不据此宣称两个分支梯度严格等价。

另外，即使选 `gkd`，这个通用训练器仍会生成 `log_probs`，并调用任务 reward 服务。**“回归 loss 不使用 PPO ratio”不等于“现有代码已经移除了所有 logprob 和 reward 计算开销”。**

### 2.8 更新什么？只更新学生的可训练参数

主训练文件 `:756–767` 的 LoRA 为 `r=32`、`lora_alpha=64`，加在 transformer 的注意力投影上；VAE 和三个文本编码器不训练。

`optimizer` 接收的是 `transformer_trainable_parameters`。在 `:1803–1809` 执行：

```python
accelerator.backward(loss)
if accelerator.sync_gradients:
    accelerator.clip_grad_norm_(
        transformer.parameters(), config.train.max_grad_norm
    )
optimizer.step()
optimizer.zero_grad()
```

**梯度路径**：当前一步的 loss → 学生这一轮前向 → 学生 LoRA。老师预测不反传，rollout 不反传，也不靠最终图片的 VAE/DINO 特征反传。

多教师示例用 10 步采样，`timestep_fraction=0.99`；代码用 `int(10 * 0.99)`，实际训练前 **9** 个时间位置，不是笼统的“10 步全监督”。每个 epoch 采样后再用这些轨迹训练，不能把每一次优化都描述成即时重新采样。

### 2.9 阅读源码后保留的边界

- 多教师示例当前只启用 OCR / GenEval；不能直接当成论文四项能力的完整正式配置。
- 冷启动 `lora_path`、审美参考 `mar_lora` 都需要显式指定。留空时的行为和论文最完整方案不同。
- 论文 Eq.15 的局部回归形式、代码 `gkd`、默认 `kl_only` 需要分别看，不能混写成同一条实现。
- DeQA 老师的奖励混合比例在正文 §6.1 写 DeQA : PickScore = 4:6，附录 A 又描述 1:1 的奖励相加口径；不能在未进一步核对时当成同一套明确配方。
- 论文对“严格等价”“消除梯度干扰”等论述，不等于本文已经做了数学证明或独立复现；这里能确认的是监督对象、数据流和代码计算路径。

## 三、结果、实验表格与消融

### 3.1 为什么多奖励混训不够：Figure 2 与 Table 1

![Figure 2：单奖励专家在其他任务上可能退化，PDF 第 4 页](/flow-opd-fig02-single-reward.png)

*Figure 2 原图。训练单项专家能提升专长，但其他任务的生成可能退化，这是作者希望通过能力整合缓解的问题。*

![Table 1：逐步加入奖励后的能力退化，PDF 第 4 页](/flow-opd-tab01-reward-mixing.png)

*Table 1 原表截图。重点看 GenEval：加入其他奖励后，构图指标逐步下降。它支持本论文实验设置中的多任务冲突，不等于证明所有多奖励方法都必然失败。*

### 3.2 主结果：Figure 3 与 Table 2

实验基座为 SD3.5-Medium。论文 §6.1 使用 4 节点、每节点 8 张 H800 训练，评估为单节点 8 张 H800；附录 A 给采样 10 步、评估 40 步、512 分辨率、LoRA r32 / α64，以及 group size 24、noise level 0.7、MAR 系数 β=0.02。

仓库多教师示例则是 8 卡、`num_image_per_prompt=16`、`beta=0.04`。不要与论文的正式设置混为一谈；附录 A 的约 50 小时训练描述紧接在 GRPO-mix 设置之后，也不要直接当作独立核实过的 Flow-OPD 耗时。

![Figure 3：多任务生成的定性对比，PDF 第 8 页](/flow-opd-fig03-main-qualitative.png)

*Figure 3 原图。列为基模、各项 GRPO 专家、GRPO-Mix、Merge、Ours；行覆盖文字、物体、数量、复杂指令等例子。Merge 不是终点，后续学生轨迹蒸馏仍带来变化。*

![Table 2：构图、文字、质量与偏好主结果，PDF 第 8 页](/flow-opd-tab02-main-results.png)

按这张 **v5 Table 2** 原表读：

- Ours (Merge) 的 GenEval / OCR 都是 **0.93**；GRPO-Mix 为 0.73 / 0.83。
- Avg 为 **0.9021**，GRPO-Mix 为 0.8165；Avg 是四项归一化指标的平均，不是单项指标，也不是人类胜率。
- **不能说所有单项都超过每个专家**：GenEval 专家是 0.94，GRPO-Mix 的 DeQA 是 4.33，PickScore 专家是 23.19；Ours (Merge) 对应为 0.93、4.31、23.05。
- 论文强调的收益是兼顾多项能力。Teacher-surpassing 的定性例子，不能改写成“学生在所有任务上必然超过最强老师”。

**版本口径**：摘要 / 仓库 README 给过 GenEval 0.92、OCR 0.94；v5 Table 2 写 0.93 / 0.93，Figure 4 又显示 0.92 / 0.94。这里保留原图原表，按各自出处解释，不自行把数字改成一致。

### 3.3 冷启动消融：Figure 4

![Figure 4：不冷启动、SFT 冷启动与 Merge 冷启动，PDF 第 9 页](/flow-opd-fig04-cold-start.png)

*Figure 4 原图。比较 Base、GRPO-Mix、无冷启动 OPD、SFT 冷启动、Merge 冷启动。两种冷启动都改善构图/文字，Merge 在图中的整体表现更好。图中 Merge 的 GenEval / OCR 为 92 / 94，与 Table 2 的 93 / 93 存在原文数值不一致。*

这张图回答的是：**OPD 之前先把学生初始化好，有没有用？** 不要把全部收益归给冷启动；Table 2 的 `Merge+GRPO-Mix` 与 `Ours (Merge)` 还比较了相同类型起点之后的不同训练方式。

### 3.4 泛化结果：Table 3

![Table 3：T2I-CompBench++ 结果，PDF 第 10 页](/flow-opd-tab03-compbench.png)

*Table 3 原表。覆盖颜色、形状、纹理、复杂组合、3D 空间、数量和非空间关系。重点对照 Cold Start、Cold Start+GRPO、Ours (Merge)：冷启动后继续 GRPO 并不保证每项继续进步；OPD 在表中取得更好的综合组合能力。*

例如 3D-Spatial：Cold Start 0.4249，Cold Start+GRPO 0.4017，Ours (Merge) 0.4565。这是论文中的泛化证据，不是新增奖励直接优化出来的同一项训练成绩。

### 3.5 MAR 消融：Figure 5 与 Table 4

![Figure 5：有无 MAR 的定性消融，PDF 第 9 页](/flow-opd-fig05-mar-ablation.png)

*Figure 5 原图。作者用“w.o KL Loss / w. KL Loss”比较去掉和加入 MAR 后的生成，展示功能学习之外的画面质量与结构差异。*

![Table 4：画质与偏好评估，包含去掉 MAR 的消融，PDF 第 10 页](/flow-opd-tab04-quality-mar.png)

*Table 4 原表。`w.o. MAR` 与 `Ours (Merge)` 的比较中，Aesthetic 从 5.89 到 6.23，QwenVL Score 从 3.82 到 4.05，其余列也改善。这支持 MAR 在该实验中的帮助，不是“不可能再出现 reward hacking”的保证。*

### 3.6 与 DiffusionNFT 对比：Table 5 与 Figure 10

![Table 5：Qwen-VL 评估下与 DiffusionNFT 的对比，PDF 第 15 页](/flow-opd-tab05-diffusionnft.png)

*Table 5 原表。作者使用的 Qwen-VL 评分为 3.74 vs 4.05；这是特定评估器下的作者报告，不是真人偏好胜率。评估 prompt 见论文 Figure 6，本文没有独立运行。*

![Figure 10：与 DiffusionNFT 的定性比较，PDF 第 17 页](/flow-opd-fig10-diffusionnft.png)

*Figure 10 原图。论文用这些例子讨论局部结构、纹理及指令遵循。不能仅凭这些示例就推断所有任务、预算和超参下都优于 DiffusionNFT。*

### 3.7 附录更多结果：Figure 7–9

![Figure 7：PickScore 测试集上的更多生成对比，PDF 第 16 页](/flow-opd-fig07-pickscore.png)

*Figure 7 原图，比较 SD3.5-M、GRPO-mix、Ours。*

![Figure 8：GenEval 测试集上的更多生成对比，PDF 第 16 页](/flow-opd-fig08-geneval.png)

*Figure 8 原图，展示属性、空间关系、数量等例子。*

![Figure 9：OCR 测试集上的更多生成对比，PDF 第 17 页](/flow-opd-fig09-ocr.png)

*Figure 9 原图，展示招牌、封面、场记板、机场屏幕等文字渲染。原文图注称“quantitative comparisons”，图中实际是生成样例，不是新增数值表。*

### 3.8 失败案例与适用边界

![Figure 11：专家错误可能传到学生，PDF 第 17 页](/flow-opd-fig11-failures.png)

*按论文附录 B.3 的失败案例段落解读：当专家给出语义错误的监督时，学生也可能继承错误。图中列标题是 Flow-GRPO-GenEval 与 Flow-OPD；原图图注仍写着“comparisons with DiffusionNFT”，属于原文图注与内容口径不一致，本文未改图。*

- **老师有上限**：密集监督也可能密集传递错误；不能保证学生自行修正所有老师失误。
- **主方法依赖同源、兼容的表示**：论文把师生架构同质性列为限制，不是现成的跨 VAE 异构蒸馏方案。
- **没有独立复现**：结果是作者报告，当前公开示例的 loss 模式和参考配置与论文完整描述并不完全相同。

一句话收束：**Flow-OPD 把“多奖励一起打分”换成“同源专家在学生自己的中间状态上分别教下一步”，再用 MAR 约束整体生成行为；源码里要分清直接回归的 `gkd` 与默认 PPO 式的 `kl_only`。**
