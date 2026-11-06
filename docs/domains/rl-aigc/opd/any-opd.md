---
type: paper
tags: ['Distillation', 'Flow Matching', 'T2I']
status: reading
summary: '前半段和 NLD-Uni 一样：学生出图，老师 SDEdit 精修。区别在后半段——不把精修图当 GT 做 SFT，而是让学生带梯度重跑后半段去噪，生成的图与精修图在 DINOv2 特征上比相似度；训练分两个阶段，前面先做 400 步离线蒸馏（anchoring）。'
models: ['Any-OPD', 'SD3.5-Medium', 'FLUX.1-dev', 'Z-Image']
---

# Any-OPD 伪代码

- 论文：*Any-OPD: Heterogeneous On-Policy Distillation for Flow-Matching Models via Representation-Space Bridging*，arXiv 2608.03316
- 设置：学生 SD3.5-Medium（2.5B，LoRA），老师 FLUX.1-dev（12B），也试过 Z-Image。没有公开代码，伪代码按论文 §3.3、§3.4、§4.1 整理。

**一句话**：前半段和 NLD-Uni 一样，学生出图，老师 SDEdit 精修。区别在后半段：不把精修图当 GT 做 SFT，而是**让学生带梯度重跑后半段去噪，生成的图和精修图在 DINOv2 特征上比相似度**。训练分两个阶段，前面先做一段离线蒸馏。

记号：下标 0 = 纯噪声，N = 干净图。σ 随下标递减。S = 学生，T = 老师。

---

## 阶段 1：Anchoring（离线蒸馏，400 步）

```python
for step in range(400):
    c   = sample_prompt()                               # Pick-a-Pic 训练集
    x_T = T.decode(T.sample(c, steps=50))               # 老师从纯噪声出图（512 px）
    z0  = S.vae.encode(x_T)                             # ★ 用学生的 VAE 编码 → 学生 latent 空间里的 GT
    m   = randint(0, M - 1)                             # 从学生调度里随机挑一个噪声级（M = 10）
    eps = randn_like(z0)
    z_m = (1 - σS[m]) * z0 + σS[m] * eps
    loss = || v_θ(z_m, σS[m], c) - (eps - z0) ||²        # 标准 flow matching loss
    loss.backward(); opt.step()
```

作用：先把学生的整体分布拉到老师那边，第 2 阶段只需要修正单张图的误差。
论文消融：不做这一阶段，或者只做 100 步，第 2 阶段的效果都明显更差。

## 阶段 2：On-policy（800 步）

```python
N_S = N_T = 20                                          # 学生、老师都用 20 步 Euler

for step in range(800):
    c = sample_prompt()

    # 1. 先定 SDEdit 强度：老师"往回退几步"
    r   = randint(15, 20)                               # r ∈ {15..20}，论文消融里效果最好
    k   = N_T - r                                       # 老师从第 k 步开始去噪
    σ_T = σT[k]                                         # 老师那一步的真实 σ（r=20 → k=0 → σ=1，等于完全重画）

    # 2. 按 σ 的数值（不是步号）对齐学生：找学生里第一个"至少和 σ_T 一样噪"的位置
    j = max(i for i in range(N_S) if σS[i] >= σ_T)      # σS[j] ≥ σ_T > σS[j+1]

    # 3. 学生出图（不带梯度，确定性 Euler），顺手缓存第 j 步的状态
    with no_grad():
        z = randn(...)
        for i in range(N_S):
            if i == j: z_j = z.clone()
            z = z + (σS[i+1] - σS[i]) * v_θ(z, σS[i], c)
        x_S = S.vae.decode(z)                           # 学生的图（像素）

    # 4. 老师 SDEdit 精修（只当黑盒采样器，用它自己的 VAE）
    with no_grad():
        zT    = T.vae.encode(x_S)
        zT    = (1 - σ_T) * zT + σ_T * randn_like(zT)   # 加噪到 σ_T
        zT    = T.euler(zT, from_step=k, to_step=N_T, c=c)   # 老师走完 σ_T → 0
        x_ref = T.vae.decode(zT)                        # 精修图

    # 5. ★ 学生从 z_j 带梯度重跑后半段（每步 gradient checkpoint）
    z = z_j.detach()                                    # 从学生"实际走到的状态"出发 = on-policy 的关键
    for i in range(j, N_S):
        z = z + (σS[i+1] - σS[i]) * v_θ(z, σS[i], c)
    x_hat = S.vae.decode(z)                             # VAE 冻结，但梯度要穿过它

    # 6. 在第三方特征空间里比较（两个模型唯一的接触点）
    f = DINOv2_Base_CLS                                 # 冻结
    loss = 1 - cos(f(x_hat), f(x_ref))
    loss.backward(); opt.step()                         # 梯度：DINO → 学生 VAE 解码器 → 第 j..N_S 步去噪 → LoRA
```

---

## 关键设置

| 项 | 值 |
|---|---|
| 训练参数 | LoRA r32 / α64（只训学生 transformer），lr 1e-4，每卡 batch 4，512 px |
| 步数 | 阶段 1 训 400 步，阶段 2 训 800 步 |
| prompt | Pick-a-Pic 训练集，测试集做验证 |
| 采样 | 学生用确定性 Euler。加随机性（CPS，η ≥ 0.3）在 200–300 步后变差 |
| 老师步数 | N_T = 20。N_T = 10 时精修图会糊，学生也会学成糊的 |
| loss 消融 | latent MSE 很快就崩；LPIPS 先涨后跌；DINO CLS 最稳。CLS 加 patch 或多层特征没有更好 |

## 和 NLD-Uni 的区别

| | NLD-Uni | Any-OPD |
|---|---|---|
| 前置阶段 | 没有 | 先做 400 步离线蒸馏 |
| SDEdit 强度 | 固定 σ = 0.9 | 每步随机 r ∈ {15..20}（最大到 σ = 1） |
| 精修图怎么用 | 当 GT，做学生自己的 SFT | 学生带梯度重跑第 j 步之后的去噪，和精修图比 DINO 特征 |
| 梯度路径 | 和普通训练一样 | 穿过 DINO、学生 VAE 解码器和多步去噪，显存开销大 |
| 管得住什么 | 逐位置的细节（token / 像素级） | 全局语义和风格（DINO CLS 只是一个全局向量） |

---

## 结果（论文截图）

### 主结果（表 1）：FLUX.1-dev → SD3.5-M

![表 1](/any-opd-tab1-main.png)

- DrawBench 上 6 项指标全部提升，Aesthetic、ImageReward、PickScore 三项**超过了老师**。
- HPSv3 从 9.12 涨到 10.97，老师是 11.19。
- DPG（构图 / 对齐）基本不变：84.58 → 84.71。论文的解释是 FLUX 在这一项上本来就不比学生强。
- 评测在 1024 px、40 步，训练在 512 px。
- 注意：这里的 PickScore 是 DrawBench 上的 0.866 / 0.884，和 Adv-GRPO 在 PickScore 测试集上的 21.7x（×26 的写法）**不是同一个测试集**，不能直接比。

![图 3](/any-opd-fig3-qualitative.png)

### SDEdit 强度（表 2）

![表 2](/any-opd-tab2-strength.png)

- r ∈ [0, 10)（σ < 0.63）明显差，DPG 甚至掉到 81.21，比基座的 84.58 还低。
- r ∈ [10, 15) 和 [15, 20] 差不多，论文选了 [15, 20]。**和 NLD-Uni 的结论一样：要加很重的噪声。**

### 换老师（表 3）：Z-Image → SD3.5-M，其他完全不变

![表 3](/any-opd-tab3-zimage.png)

- 所有指标都涨了，包括 GenEval（+0.80）和 DPG（+0.95）。Z-Image 本身构图更强，学生也跟着学到了这一点。
- ImageReward 1.082 超过了老师的 0.958；HPSv3 9.75 还没追上老师的 9.96。

### loss 消融（图 5）

![图 5](/any-opd-fig5-loss-ablation.png)

- latent MSE 几步就崩；LPIPS 先涨后跌；DINO CLS 一直稳定，而且最好。
- 越依赖"逐位置对齐"的 loss 越差。原因是老师每次精修都会重画局部细节，位置对不上。

### 阶段 1 是否必要（图 6）

![图 6](/any-opd-fig6-anchoring.png)

- 不做 anchoring 时，ImageReward 一直在 1.0 左右，Aesthetic 不涨。
- 只做 100 步的版本，始终追不上做了 400 步的版本。**第 2 阶段的效果取决于第 1 阶段做得好不好。**

### 学生采样要不要加随机性（图 7）

![图 7](/any-opd-fig7-cps.png)

- 确定性 Euler（η = 0）在两个指标上都最好。η ≥ 0.3 时，200–300 步之后开始变差。
- 这和 RL 的经验相反：RL 需要采样随机性来探索，这里加了随机性，被纠正的轨迹就不是部署时会走的那条了。

### 老师步数（图 4）

![图 4](/any-opd-fig4-teacher-steps.png)

- 老师只走 10 步时，精修图是糊的，学生会把这种糊也学过去（第 2 行）。**老师精修的质量就是学生能学到的上限。**
