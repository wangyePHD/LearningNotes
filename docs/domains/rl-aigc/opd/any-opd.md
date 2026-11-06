---
type: paper
tags: ['Distillation', 'Flow Matching', 'T2I']
status: reading
summary: 'OPD 就是两个流程加一次比较：学生先画一张图；老师把这张图修一遍；学生再从自己中途走到的状态 `z_j` 重新画一次，并朝老师修好的样子靠。'
models: ['Any-OPD', 'SD3.5-Medium', 'FLUX.1-dev']
---

# Any-OPD：核心代码与讲解

- 论文：*Any-OPD: Heterogeneous On-Policy Distillation for Flow-Matching Models via Representation-Space Bridging*，arXiv 2608.03316
- 设置：学生 SD3.5-Medium（2.5B，LoRA），老师 FLUX.1-dev（12B）。
- 论文没有公开代码，下面是按论文 §3.3、§3.4、§4.1 的公式整理的伪代码。
- 配套动画：`/gpfs/wy/Projects/SeFi-Image-5B/DPO/refs/sdedit_opd_notes/Any-OPD_流程讲解.mp4`（2 分钟，无配音）

---

## 一句话

**OPD 就是两个流程加一次比较：学生先画一张图；老师把这张图修一遍；学生再从自己中途走到的状态 `z_j` 重新画一次，并朝老师修好的样子靠。**

老师和学生之间只传两样东西：**图片**，和一个数 **σ_T**。

## 动画截图里的执行顺序

1. 老师先定：抽 r，得到 k = 20 − r，再查老师的噪声表得到 σ_T。
2. 由 σ_T 在学生的噪声表里找到 j。
3. 学生从纯噪声出图，走到第 j 步、更新之前，clone 当前状态 `z_j`。
4. 老师拿学生画完的图 x_S，加**新噪声**到 σ_T，去噪得到 x_ref。
5. 学生从 `z_j` 带梯度重跑后半段，得到 x_hat。
6. x_hat 和 x_ref 在冻结的 DINOv2 特征里比较，loss 回传到学生的 LoRA。

---

## 核心代码

### 通用设置

```python
# 模型：S = 学生 SD3.5-Medium（只训 LoRA r32/α64），T = 老师 FLUX.1-dev（冻结）
# 数据：Pick-a-Pic 训练集的 prompt；分辨率 512；lr 1e-4；每卡 batch 4
# 约定：σ 下标 0 = 纯噪声(σ=1)，下标越大越干净，最后一项 σ=0

σS_M = S.schedule(M=10)        # 学生的 10 档噪声表（阶段 1 用）
σS   = S.schedule(N_S=20)      # 学生的 20 步噪声表（阶段 2 用）
σT   = T.schedule(N_T=20)      # 老师的 20 步噪声表
f    = DINOv2_Base_CLS         # 冻结

def euler(model, z, σ, c, i0, i1):          # 从第 i0 步走到第 i1 步
    for i in range(i0, i1):
        z = z + (σ[i+1] - σ[i]) * model(z, σ[i], c)
    return z
```

### 阶段 1：Anchoring（400 步）

```python
for step in range(400):
    c   = sample_prompt()                                 # ① 取 prompt
    x_T = T.vae.decode(T.sample(c, steps=50))             # ② 老师出图（50 步，512px，无梯度）
    z0  = S.vae.encode(x_T)                               # ③ ★ 用【学生的】VAE 编码老师的图
    m   = randint(0, M - 1)                               # ④ 随机挑一个噪声档（取不到 σ=0）
    eps = randn_like(z0)                                  # ⑤ 噪声
    z_m = (1 - σS_M[m]) * z0 + σS_M[m] * eps              # ⑥ 加噪（直线插值）
    loss = ((S.v_θ(z_m, σS_M[m], c) - (eps - z0)) ** 2).mean()   # ⑦ 标准 flow matching
    loss.backward(); opt.step()                           # ⑧ 只更新 LoRA
```

### 阶段 2：On-policy（800 步）

```python
for step in range(800):
    c = sample_prompt()                                          # ① prompt

    # ===== 准备：r → k → σ_T → j =====
    r   = randint(15, 20)                                        # ② 老师「退回多远」
    k   = N_T - r                                                #    老师从第 k 步开始去噪
    σ_T = σT[k]                                                  #    查老师的噪声表，得到一个数
    j   = max(i for i in range(N_S) if σS[i] >= σ_T)             # ③ 学生里 σ ≥ σ_T 的最后一步

    # ===== ④ 学生出图（无梯度），更新前存下第 j 步的状态 =====
    with no_grad():
        z = randn(...)
        for i in range(N_S):
            if i == j: z_j = z.clone()                           # ★ 走到第 j 步、还没更新之前存
            z = z + (σS[i+1] - σS[i]) * S.v_θ(z, σS[i], c)       #    确定性 Euler，不加随机性
        x_S = S.vae.decode(z)                                    #    学生的图（3×512×512）

    # ===== ⑤ 老师精修（SDEdit，无梯度，只用图片）=====
    with no_grad():
        zT    = T.vae.encode(x_S)                                # 学生的图 → 老师的 latent（只是格式转换）
        zT    = (1 - σ_T) * zT + σ_T * randn_like(zT)            # ★ 加【新抽的】噪声到 σ_T（和 z_j 无关）
        zT    = euler(T.v_φ, zT, σT, c, k, N_T)                  # 老师自己的表，走 r 步
        x_ref = T.vae.decode(zT)                                 # 精修图

    # ===== ⑥ 学生从 z_j 带梯度重跑后半段 =====
    z = z_j.detach()                                             # 起点 = 学生自己走到的状态
    for i in range(j, N_S):                                      # 共 N_S - j 步，每步 gradient checkpoint
        z = z + (σS[i+1] - σS[i]) * S.v_θ(z, σS[i], c)
    x_hat = S.vae.decode(z)                                      # VAE 冻结，但梯度要穿过它

    # ===== ⑦ 在 DINO 特征里比较 =====
    loss = 1 - cos(f(x_hat), f(x_ref))                           # x_ref 是常数，无梯度
    loss.backward(); opt.step()                                  # 梯度：DINO → 学生 VAE 解码器 → 第 j~19 步 → LoRA
```

对应论文公式：

$$z_m = (1-\sigma^S_m)\, z^{\star} + \sigma^S_m\, \epsilon, \qquad \mathcal{L}_{\text{WS}} = \left\| v_\theta(z_m, \sigma^S_m, c) - (\epsilon - z^{\star}) \right\|_2^2 \quad\text{（阶段 1，Eq. 4、5）}$$

$$j = \max\{\, i : \sigma^S_i \ge \sigma^T_k \,\}, \quad\text{即}\quad \sigma^S_j \ge \sigma^T_k > \sigma^S_{j+1} \quad\text{（Eq. 10）}$$

$$\mathcal{L}_{\text{OPD}} = 1 - \cos\!\big( f(\hat{x}_S),\, f(x_{\text{ref}}) \big) \quad\text{（阶段 2，Eq. 12）}$$

---

## 关键点讲解

### 1. j 由老师决定，z_j 是学生自己的 latent

- 老师先抽 r，得到 `k = N_T − r`，再查**老师的**噪声表得到 `σ_T`。
- 然后在**学生的**噪声表里按 σ 的**数值**找 j（不是按步号，因为两边噪声表不一样）。
- 学生出图时，走到第 j 步、更新之前 clone 当前 latent，就是 `z_j`。
- 所以：**j 是下标，由老师的 σ_T 决定；`z_j` 是学生自己的内容，出图时克隆**。两者不矛盾。
- 老师精修（第 ⑤ 步）的起点是 **x_S 的编码加新噪声**，用不到 `z_j`。

### 2. σ_T 怎么从 r 得到：查表

```
r = 17 → k = 20 − 17 = 3 → σ_T = σT[3] = 0.914
```

老师的噪声表下面是我按 diffusers 里 FLUX 的默认设置算的（动态 shift，512 px 时参数 μ = 0.63），**论文没给具体数值**，只用来理解：

- r = 20：k = 0，σ_T = 1.00（相当于完全重画）
- r = 19：k = 1，σ_T = 0.97
- r = 18：k = 2，σ_T = 0.94
- r = 17：k = 3，σ_T = 0.91
- r = 16：k = 4，σ_T = 0.88
- r = 15：k = 5，σ_T = 0.85
- r = 10：k = 10，σ_T = 0.65
- r = 9：k = 11，σ_T = 0.61

一个对照检验：论文说 r ∈ [0, 10) 对应 σ < 0.63，和上面吻合。r 越大，加的噪声越重。

### 3. 为什么 r 取 15 到 20

- r 小：老师几乎不改图，x_ref ≈ x_S，两者的 DINO 特征几乎一样，loss ≈ 0，没有信号。
- r = 20：完全重画，x_ref 和 x_S 没有关系，梯度很吵。
- 中间偏大（15 到 19）：保留学生的构图，只换执行细节，是想要的效果。

论文 Table 2（原表截图）：

![表 2：老师精修强度](/any-opd-tab2-strength.png)

低噪声那一档（[0, 10)）明显最差；后两档接近，论文选 [15, 20] 只是「综合最均衡」，15 这个边界没有深层原理。

### 4. j 的对齐：为什么从 j 开始，不是 j+1，也不是更早

以 σ_T = 0.914 为例（学生的 σ 表是我假设的示意值）：

- 学生第 4 步覆盖 (0.900, 0.923]，包含 0.914，所以 j = 4。
- **不从 j+1 开始**：第 5 步起的区间都在 0.900 以下，老师改动的范围是 0.914 以下，中间 0.900 到 0.914 只有第 4 步管，会漏掉。
- **不从更早开始**：第 0 到 3 步的噪声比 0.923 还重，决定的是整体构图，老师没改这部分，不需要这些步来承担，带梯度还更费显存。
- 论文叫它 tightest segment：能完整覆盖老师改动范围的最短一段。
- 取值规则是「噪声至少一样大的最后一步」，永远取噪声更大的那一侧，不是数值上最近的那个。

### 5. 学生重跑的目的：满足 OPD

- **on-policy 来自起点**：`z_j` 是学生自己出图时走到的状态，不是用「正确的图加噪」造的。
  - 阶段 1（off-policy）：`z_m` 由老师图的 `z0` 加噪得到，永远是「完美状态」。
  - 阶段 2（on-policy）：学生被纠正的，是它部署时真会走到的、带误差的状态。
- **重跑本身是工程手段**：④ 为了先得到完整的图，必须是 no_grad；所以后半段要再跑一遍才能记录梯度，只重跑 j 到 19 步是为了省显存。
- 因为权重没变、采样是确定性的，⑥ 算出的 x_hat 在数值上等于 ④ 的 x_S，区别只在有没有梯度。
- `z_j.detach()` 切断的是「梯度往 j 之前传」；20 步用的是同一套 LoRA 权重，所以这次 loss 不去追究 j 之前那段是怎么算出来的，但权重本身是共享的。

### 6. 阶段 1 的 `m = randint(0, M-1)`

- 每次训练只挑一个「加噪程度」，学生推理时要走过所有噪声程度，所以每个样本随机抽一个，训练下来各档都会覆盖到。
- 目标速度 `eps − z0` 与 m 无关，变的只是输入 `z_m`。
- 取不到 σ = 0（那时没有噪声可还原）。
- **论文没有解释为什么 M = 10，正文也没有对 M 做消融。** M 只决定有多少个不同的 σ 可以被抽到，M → ∞ 就是连续抽，我推测效果差不多，但没有验证。
- σ 表里最小的 σ 取决于调度器的实现，论文没写，不要把示意表当真。

---

## 论文结果（原表截图）

主结果（Table 1）：FLUX.1-dev → SD3.5-M

![表 1](/any-opd-tab1-main.png)

loss 消融（Figure 5）：latent MSE 很快崩，LPIPS 先涨后跌，DINO CLS 最稳。

![图 5](/any-opd-fig5-loss-ablation.png)

阶段 1 是否必要（Figure 6）：

![图 6](/any-opd-fig6-anchoring.png)

学生采样要不要加随机性（Figure 7）：确定性 Euler（η = 0）最好。

![图 7](/any-opd-fig7-cps.png)

老师步数（Figure 4）：老师只走 10 步时精修图是糊的，学生也会学成糊的。

![图 4](/any-opd-fig4-teacher-steps.png)

---

## 论文没说清、伪代码里省略的

- 学生 rollout、带梯度重跑、老师精修是否带 CFG（论文只列了学生 guidance 4.5、老师 3.5）。
- 一个 batch 的 4 张图是共用同一个 r，还是各抽各的。
- 图输入 DINO 前怎么缩放。
- 阶段 1 的老师图是在线生成还是预生成（它不依赖学生，可以预生成）。
- r = 20 时 σ_T = 1，精修图和学生的图基本无关，论文承认梯度方差大，没说是否特殊处理。
- 论文摘要和 Figure 1 写学生 PickScore 起点是 0.846，Table 1 写 0.866，论文前后不一致。
