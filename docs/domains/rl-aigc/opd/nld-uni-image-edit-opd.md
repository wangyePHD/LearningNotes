---
type: paper
tags: ['Distillation', 'Diffusion', 'Image Editing', 'Unified Model', 'Flow Matching']
status: reading
summary: 'NLD-Uni 编辑侧 OPD 的代码级拆解：学生先出图，老师在学生结果上做 SDEdit（σ=0.9）精修，精修图当 GT 用普通 SFT loss 训一步；学生结果只当精修起点、不进 loss，论文承认这不是传统意义上的 on-policy。'
models: ['NLD-Uni', 'Qwen-Image-Edit-2511', 'Qwen-Image', 'SD3.5-M', 'Qwen-Image-2.1']
---

# NLD-Uni 图像编辑 OPD 笔记

- 论文：*Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation*，arXiv 2610.36466（主文 §3.4、§4.1；附录 A.4、B、C.4）
- 代码：`DPO/refs/Nemotron-Labs-Diffusion-Uni-main.zip` → `minimal_training/`（下文的行号都指这个目录里的文件）
- 截图取自论文 PDF

---

## 一、通俗的全局解释

### 1. 要解决什么问题

NLD-Uni 是一个**离散扩散**的统一模型。图像被切成 VQ token（码本大小 131072），生成图像就是"一格一格填 token"。
它想向 **Qwen-Image**（编辑模型）学图像编辑。但 Qwen-Image 是**连续 latent 上的 flow matching** 模型。
两边说的不是一种"语言"：学生每个位置输出"13 万个码各自的概率"，老师输出"一个连续向量该往哪个方向走"。
所以没法像文本 OPD 那样，直接让学生的概率去贴老师的概率（算 KL）。

### 2. 怎么解决：让老师"改学生的作业"

既然没法比概率，就**直接比图**：

```
原图 + 编辑指令（例："让他举起右手"）
      │
      ▼
① 学生先做一遍编辑              → 学生结果：手举起来了，但脸有点糊
      │
      ▼
② 老师在学生结果上 SDEdit：
   用老师的 VAE 编码 → 加 90% 噪声（只剩一个模糊的影子）
   → 以"原图 + 指令"为条件去噪     → 精修图：姿势还是学生的，脸清楚了，背景和原图一致
      │
      ▼
③ 精修图 → 学生的 VQ 编码 → 当 GT，用学生平时的 SFT loss 训一步
      │
      └── 下一步用更新后的学生再做编辑（闭环）
```

- **学生结果只当 SDEdit 的起点，不进 loss**。loss 里只有精修图。
- loss **就是普通 SFT**（论文原话："the canonical SFT loss on the teacher-refined samples"）。
- 论文承认这"不是传统意义上的 on-policy"。叫 on-policy 是因为**目标是从当前学生自己的输出改出来的**，学生变了，目标也跟着变。

### 3. 为什么不直接用老师自己编辑的结果当 GT

看论文图 6：第 2 列是学生结果，第 3 列是老师精修，第 4 列是老师自己从零编辑。

![图 6](/nld-uni-fig06-flow-refine.png)

1. **对得上**：精修图（第 3 列）的姿势、构图和学生结果（第 2 列）一致，只是细节更好、原图保持得更好。老师自己编辑的结果（第 4 列）连人的姿势都不一样，学生要学的跨度太大。
2. **老师犯错时能兜底**：看附录图 14 第 3 行，指令是"去掉中间的木桌"。老师自己编辑（第 4 列）把旁边两把椅子也删了，改过头了。精修（第 3 列）被限制在学生结果附近，桌子去掉了，椅子都还在，比老师自己的结果和学生的结果都好。

![图 14](/nld-uni-fig14-refinement-more.png)

### 4. 关键参数 σ（论文里叫 r）

σ 是加噪的比例：学生结果保留 1−σ，老师重画 σ。论文选 **0.9**，加噪很重。
但论文指出，噪声对结果的影响是高度非线性的：**即使 σ=0.9，精修图仍然和学生结果对得上**（上面两张图都是 σ=0.9 的效果）。
σ=1.0 时学生结果完全被噪声覆盖，就退化成离线蒸馏（结果见第四节）。

---

## 二、核心代码讲解

代码片段为了便于阅读做了删减，省略了设备、dtype 之类的处理。变量名和关键行与原文件一致，行号指原文件。
注释里的具体数字都按 **1024×1024 的图**来算。

一步训练的调用链：

```
train.py（编辑分支）
  ├─ ① Student.rollout          学生出图                     objectives.py:104
  ├─ ② HTTP → EditTeacher       老师精修（用到 matched_sigmas） teachers.py:87 / :16
  └─ ③ Student.image_loss       精修图做 SFT                  objectives.py:184
         ├─ corrupt              给 GT 加噪                    corruption.py:38
         ├─ image_condition_dropout  CFG 丢条件                loss_math.py:101
         └─ image_token_losses   算 loss                       loss_math.py:92
```

### 2.1 训练循环里的编辑分支：`train.py:93-122`

```python
if task.endswith('_opd'):                       # 当前轮到 'image_edit_opd' 这个任务
    wrapped.eval()                              # 出图时关掉 dropout 等训练行为
    with torch.no_grad():                       # 出图不需要梯度，省显存
        proposal = wrapped('rollout', sample, task)
        # proposal = 学生的编辑结果，一张 1024×1024 的 PIL 图片

    if rank == 0:                               # 只有 0 号卡去找老师（演示版 batch=1）
        source = pad_to_square_and_resize(sample.images[0], 1024)   # 原图补成正方形、缩到 1024
        result = json.loads(request(cfg['edit_teacher'], {           # 通过 HTTP 发给老师服务
            'prompt':   sample.prompt,          # 编辑指令，例："Make the person raise his right arm."
            'source':   encode_image(source),   # 原图（PNG 转成 base64 字符串）
            'proposal': encode_image(proposal), # 学生的编辑结果
            'seed':     seed}))                 # 老师加噪用的随机种子，保证可复现
        target = decode_image(result.pop('image'))   # 老师返回的精修图
        # 三张图存进 edit-attempt-001/ 之类的目录：source.png / student.png / teacher.png
        # 训练时可以直接打开看老师改了多少
    target = broadcast(target)                  # 0 号卡把精修图发给其他所有卡

wrapped.train()
loss, components = wrapped('loss', sample, task, target=target)  # 用精修图当 GT 算 SFT loss
loss.backward()
wrapped.clip_grad_norm_(1.0)                    # 梯度裁剪
optimizer.step()                                # 更新一次参数；下一步学生就是新的了
```

一句话：**出图 → 发给老师 → 拿回精修图 → 当 GT 训一步**。学生结果 `proposal` 除了发给老师，没有在其他地方用到。

### 2.2 学生出图：`objectives.py:104-109`

```python
if task == 'image_edit_opd':
    return m.image_edit(
        example.images[0],          # 原图
        example.prompt,             # 编辑指令
        resolution=1024,            # 出 1024×1024 的图
        steps=64,                   # 离散扩散走 64 步（每步把一部分 token 从"粗"改到"细"）
        seed=self.options['seed'],
        crop_to_source=False)
```

这就是学生平时推理用的编辑接口，设置和部署时一样。这样学生是在"推理时真的会出的图"上被纠正，而不是在理想数据上。

### 2.3 老师精修：`teachers.py:87-117`（`EditTeacher`）

```python
class EditTeacher:
    def __init__(self, path, strength=.9, steps=40):
        # path = Qwen/Qwen-Image-Edit-2511，模型摊在 4 张卡上
        # strength = σ = 0.9：加 90% 的噪声
        # steps = 40：完整调度是 40 步（下面会看到实际只走其中 33 步）

    def __call__(self, payload):
        source, proposal = decode_image(payload['source']), decode_image(payload['proposal'])

        # (a) 学生结果 → 老师的 latent
        pixels = pipe.image_processor.preprocess(proposal, ...)      # 像素值归一化到 [-1, 1]
        proposal_latent = pipe._encode_vae_image(pixels, ...)
        #   Qwen 的 VAE 把图缩小 8 倍：1024×1024×3 → 128×128×16
        latent = pack(proposal_latent)
        #   每 2×2 个位置打包成一个 token：128×128×16 → 4096 个 token × 64 维
        noise = pack(randn_tensor(...))     # 同样形状的高斯噪声，用 payload['seed'] 生成

        # (b) 加噪：这一行就是 SDEdit 的全部
        initial = (1 - 0.9) * latent + 0.9 * noise
        #   = 10% 学生结果 + 90% 噪声。学生结果只剩一个模糊的影子（大致构图和颜色）

        # (c) 算出从 σ=0.9 开始的调度（见 2.4）
        sigmas, effective = matched_sigmas(0.9, initial.shape[1], 40)   # initial.shape[1] = 4096

        # (d) 老师从 σ=0.9 去噪到 0
        result = pipe(
            image=[source],                     # 条件 1：原图（老师据此保持背景、身份不变）
            prompt=payload['prompt'],           # 条件 2：编辑指令
            negative_prompt=' ', true_cfg_scale=4.,  # CFG 强度 4：每步跑两次前向（有条件 / 无条件）
            sigmas=sigmas, num_inference_steps=len(sigmas),   # 33 步
            latents=initial,                    # ★ 起点是"加了噪的学生结果"，不是纯噪声
        ).images[0]
        return 精修图 + {strength: 0.9, initial_sigma: 0.9, teacher_steps: 33}
```

**和老师平时编辑相比，只改了两个参数**：`latents=initial`（起点从纯噪声换成加噪的学生结果）和 `sigmas=sigmas`（从 0.9 开始，而不是从 1 开始）。去噪循环本身一行没改。
每张图的成本：33 步 × 2 次前向（CFG）= 66 次 20B 级模型的前向。

### 2.4 让起点"正好是 0.9"：`teachers.py:16-30`（`matched_sigmas`）

**先说为什么需要它**：Qwen 的调度器会对传进去的 σ 偷偷做两次变换：

1. **动态 shift**：图越大，把 σ 往大的方向推得越多（大图需要更多时间处理高噪声）。1024 的图（4096 个 token）shift 系数约为 2.0。
2. **终点拉伸**：把整条调度拉长，让最后一步停在 0.02，而不是 0。

所以如果你天真地传"从 0.9 开始"，**实际起点是 0.946**，比你加的 0.9 噪声更大。调度器以为图里有 94.6% 的噪声，实际只有 90%，去噪就会出偏差。
这个函数做的事是：**倒推回去，算出"传什么值进去，变换之后正好落在 0.9"**。

```python
def matched_sigmas(strength, sequence_length, steps=40):   # strength=0.9, sequence_length=4096
    # 第 1 步：算出调度器会用的 shift 系数（和 Qwen 调度器里的公式一样）
    scale = math.exp(.5 + (sequence_length - 256) * (.9 - .5) / (8192 - 256))
    #   4096 个 token → scale ≈ 2.0
    raw = np.linspace(1., 1. / steps, steps)               # 正常 40 步：1.0, 0.975, 0.95, ..., 0.025
    shift = lambda x: scale * x / (1 + (scale - 1) * x)    # 调度器的变换 1
    stretch = (1 - shift(raw[-1])) / .98                   # 调度器的变换 2（让终点 = 0.02）

    # 第 2 步：倒推
    shifted_start = 1 - stretch * (1 - strength)           # 先撤销"拉伸"：0.9 → 0.903
    start = shifted_start / (scale - (scale - 1) * shifted_start)   # 再撤销"shift"：0.903 → 0.823
    #   意思是：传 0.823 进去，调度器变换后正好是 0.9

    # 第 3 步：拼出要传的序列 = 起点 + 原 40 步里所有比起点小的点
    selected = np.concatenate(([start], raw[raw < start - 1e-12]))
    #   [0.823, 0.800, 0.775, ..., 0.025]，共 33 个点 → 实际只走 33 步

    # 第 4 步：自检，模拟调度器的变换，确认起点 = 0.9、终点 = 0.02
    effective = 1 - (1 - shift(selected)) / stretch        # [0.900, 0.886, 0.869, ..., 0.068, 0.020]
    assert np.isclose(effective[0], strength) and np.isclose(effective[-1], .02)
    return selected.tolist(), effective.tolist()
```

| | 传进去的第一个值 | 调度器实际用的起点 |
|---|---|---|
| 天真做法 | 0.900 | **0.946**（和加的噪声对不上） |
| `matched_sigmas` | 0.823 | **0.900**（正好） |

diffusers 里 img2img 的 `strength` 参数也有同样的问题：它不等于真实的 σ。

### 2.5 精修图做 SFT：`objectives.py:184-235`（`image_loss`）

编辑任务和文生图共用这一个函数（`objectives.py:239`：`task in ('t2i_sft', 'image_edit_opd')`）。
编辑任务只多了两件事：**GT 换成精修图**，以及 **原图作为条件放进输入**。分 5 步看：

```python
def image_loss(self, example, target):          # target = 老师的精修图（PIL 图片）

    # ── (a) GT：精修图 → 离散 token ────────────────────────────
    codes, shape = m.encode_image_gen(preprocess(pad_to_square_and_resize(target, 1024)))
    #   用学生自己的 VQ 编码器，把图变成一格一格的码，每格是 0 ~ 131071 之间的一个整数
    codes, shape, structural = append_image_row_markers(codes, shape, ...)
    #   每行末尾插 2 个"换行"标记 token；structural 记下它们的位置，后面不加噪、不算 loss

    # ── (b) 条件：原图 → token → 放进 prompt ───────────────────────
    source_codes, source_shape = m.encode_image_gen(preprocess(原图))   # 原图也变成码
    source = m.encoder.call_gen_embedding(source_codes, source_shape, enc=True)   # 码 → 向量
    question = '<image> ' + '<SPECIAL_19>' * source.shape[1] + '\n ' + example.prompt + ' '
    #   先在文本里放 N 个占位符 <SPECIAL_19>（N = 原图 token 数），后面接编辑指令
    inputs[ids.eq(19)] = source.flatten(0, 1)
    #   再把这 N 个占位符的位置，换成原图的真实向量
    #   → 模型看到的输入：[系统提示][原图的 N 个向量][编辑指令][待生成的图]

    # ── (c) 给 GT 加噪（见下面的解释） ─────────────────────────────
    protected = structural | 原图 padding 区域          # 换行标记和补边区域不加噪
    state = corrupt(codes, tree, structural_mask=protected)
    inputs[generated] = call_gen_embedding(child_embeddings=embeddings(state, ...))
    #   "待生成的图"那部分输入 = 加了噪的 GT

    # ── (d) CFG dropout：随机丢掉一些条件（见下面的表） ────────────────
    drop, drop_route, cfg_mode = image_condition_dropout(ids, fixed | generated, ids.eq(19), rand())
    inputs = torch.where(drop[..., None], mask_token_embedding, inputs)   # 被丢的位置换成 [MASK]

    # ── (e) 前向 + 算 loss：只在被加噪的位置 ──────────────────────────
    hidden = m.encoder(inputs_embeds=inputs, ...)
    mask = state.supervised[0]                  # True = 这个位置被加了噪（需要模型猜）
    logits = m.encoder.gen_predictor(hidden[mask])  # 每个位置对 131072 个码的打分
    exact, grouped = image_token_losses(logits, codes[0, mask], groups, ...)
    loss = exact.mean() + .2 * grouped.mean()   # 精确 CE + 0.2 × 组 CE（见下面的例子）
```

#### (c) 加噪是怎么回事：沿语义树往上爬

普通的 masked diffusion 是把 token 直接换成 [MASK]。NLD-Uni 换成了"更粗"的 token：

- 131072 个码按相似度组织成一棵树。叶子是具体的码，比如"橙色毛发的某种纹理"；往上一层是相似码的代表，比如"橙色毛发"；再往上更粗，比如"暖色区域"；树根就是完全随机的码。
- 加噪时，每个位置**独立地往上爬若干层**，噪声越大爬得越高，爬到树根就相当于全是噪声。
- 模型的任务是"从粗的猜回细的"。这和 OPD 无关，是 NLD-Uni 学生本身的训练方式，叫"自我精修"离散扩散。

#### (d) CFG dropout 的四种情况（编辑任务）

| 概率 | 丢掉什么 | 模型看到的 | 作用 |
|---|---|---|---|
| 55% | 不丢 | 原图 + 指令 | 正常训练 |
| 10% | 指令 | 只有原图 | 推理时做"指令 CFG" |
| 30% | 原图 | 只有指令 | 推理时做"原图 CFG" |
| 5% | 都丢 | 什么条件都没有 | 无条件生成 |

#### (e) loss 举例：精确 CE + 0.2 × 组 CE

131072 个码被分成 16384 组，平均每组 8 个很像的码，大小不一，最大的一组有 192 个（代码里的 `assets/image_groups.pt`，我核对过）。
假设某个位置的 GT 是码 #5000，模型给 #5000 打了 30% 的概率，给同组的 #5001 打了 50%：

- 精确 CE = −ln(0.30) = **1.20**：只认 #5000，猜成 #5001 不算对。
- 组 CE = −ln(0.30 + 0.50) = **0.22**：只要猜到同一组就算对一大半。
- 这个位置的 loss = 1.20 + 0.2 × 0.22 = **1.25**

组 CE 的作用是：猜到一个"很像"的码时，惩罚轻一点。最后对所有被加噪的位置求平均。
对应论文公式 (26)：`L = −(1/t) · log p_θ(X̂0 | X̂t, V_in, p)`，其中 X̂0 是精修图的 token。附录 A.4 的 Algorithm 1 就是上面整个流程。

---

## 三、正式训练的设置（论文附录 B）

| 项 | 值 |
|---|---|
| 老师 | 论文写 Qwen-Image，公开代码里是 `Qwen/Qwen-Image-Edit-2511` |
| 训练阶段 | 第 3 阶段（统一 OPD 阶段），从第 2 阶段的 NLD-Uni-Base 开始 |
| 编辑 OPD 数据 | 从编辑数据集（ShareGPT4o-Image、GPT-Edit-1.5M、UniWorld-V1 的编辑子集、ScaleEdit-12M）里**随机抽 10 万条**，只用原图和指令，丢掉原来的 GT |
| 同阶段混合 | 视觉推理 OPD（老师是 Nemotron-3 Nano Omni），加上文生图 SFT（防止文生图能力退化） |
| 优化 | AdamW，lr 2e-6，β=(0.99, 0.999)，**10k 步** |
| 算力 | 64 张 H100，OPD 阶段用了 5 天（全部训练 26 天） |
| σ | 0.9 |
| 老师调度 | 完整调度 40 步，实际走 33 步；CFG 4，负向 prompt 为空格 |
| 学生出图 | 64 步，1024 |

文生图**没有做 OPD**。论文的理由是：基座在 GenEval 和 HPSv3 上已经比 Qwen-Image、Flux 这些候选老师更强。

---

## 四、结果对比（论文截图）

### 4.1 ImgEdit 主结果（论文表 4）

最下面三行（蓝底）是关键对比：基座、离线蒸馏（老师自己的编辑结果当 GT）、OPD（精修图当 GT）。

![表 4](/nld-uni-tab4-imgedit.png)

### 4.2 σ 消融（论文表 9）

![表 9](/nld-uni-tab9-noise-ratio.png)

### 4.3 怎么读这些数

1. **OPD 比基座高 +0.36**（3.81 → 4.17），**比离线蒸馏高 +0.27**（3.90 → 4.17）。离线蒸馏只比基座高 +0.09，说明收益大部分来自"在学生结果上精修"这个设计，而不是来自"有了老师"。
2. **σ=1.0 时是 3.90，和离线蒸馏的 3.90 一样**。这在意料之中：σ=1 时学生结果被噪声完全覆盖，精修就等于老师从零编辑。两个独立实验的数字对得上。
3. **σ 越大越好，但不能到 1**：0.5 → 0.8 → 0.9 依次是 4.06 → 4.09 → 4.17，到 1.0 掉到 3.90。效果最好的区间在"加噪很重，但还留一点学生结果"的位置。
4. **提升最大的是 Extract**（2.26 → 3.59，+1.33），甚至超过了老师（3.43）。这和"精修图有时比老师自己的结果更好"一致。
5. **Action 反而下降了**（4.19 → 4.06），离线蒸馏是 4.21。论文没有解释。可能是动作类编辑需要大幅改变姿态，而精修被锁在学生结果附近，学生做不出的大改动，老师也改不过来。这是我的推测，没有验证。
6. **离老师还差 0.10**（4.17 vs 4.27），差距主要在 Action（−0.63）和 Hybrid（−0.34）。

---

## 五、几点注意

- 这里的 **"on-policy"比较宽松**：loss 是 SFT，梯度不经过学生的采样过程。σ=0.9 时老师重画了大部分内容，学生只决定了大的构图。
- 论文写老师是 Qwen-Image，公开代码里是 Qwen-Image-Edit-2511，两者不一致，以代码为准。
- 公开代码只是**最小演示**：batch 1、每个任务 10 步、2 条假数据、lr 1e-6。论文的正式设置见第三节。
- 我们把它搬到文生图（SD3.5-M 学生、Qwen-Image-2.1 老师）的版本，见同目录的 `SDEdit-OPD_流程.md` 和 `DPO/exp/dpo04_sdedit_opd_qwen21_to_sd35m/`。
