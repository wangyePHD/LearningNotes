# 🧪 实时研习白板：VideoDeltaNet (VDN-H3) 攻坚课堂

> **当前攻坚主题**：实时视频生成前沿 —— VideoDeltaNet on MiniMax H3 (UC Berkeley / Impossible Inc / UT Austin)  
> **模式说明**：本页面为实时互动的「攻坚白板」，我们一边讨论，一边在此处实时渲染严谨的 LaTeX 公式、架构图解与推导细节。全部学透后，再整理归档为正式笔记。

---

## 1. 为什么全时空 Softmax 注意力是实时视频生成的死穴？

给定长视频潜空间输入张量：
$$
X \in \mathbb{R}^{B \times C \times T \times H \times W}
$$
展开为全局 Token 序列长度 $N = T \times H \times W$。全时空 Softmax 注意力计算复杂度为：
$$
\text{Complexity}_{\text{Dense}} = \mathcal{O}(N^2 \cdot C) = \mathcal{O}((T \cdot H \cdot W)^2 \cdot C)
$$

::: danger 算力死穴数据
在 Frontier 级别的视频全模态模型（如 MiniMax H3、Sora、Wan2.1）中：
- 生成 14.4 秒视频（约 345 帧），序列长度 $N$ 常常高达数十万；
- **全时空 Softmax 注意力的计算与显存搬运耗时，占到了整个模型运行时间的 85% 以上**！
- 单张旗舰 GPU 跑完 50 步去噪需要近 14 分钟，根本无法用于实时交互。
:::

---

## 2. 宏观解耦：双轨混合注意力 (Hybrid Attention) 架构

VDN-H3 并没有盲目将所有注意力替换为线性注意力（这会导致人脸变样、背景模糊、长程漂移），而是提出了**空间-时间互补的双轨架构**：

```text
                        当前待生成的第 t 帧 Query q_t
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   【分支 1：局部滑动窗口 Softmax】                 【分支 2：双向远距线性记忆 (VDA)】
   - 覆盖：前后各 1 个 Chunk (每 Chunk 5 帧)         - 覆盖：滑窗之外的远距离历史与未来
   - 特性：保留极高锐利度与短程稳定性               - 特性：O(N) 线性极速扫过
   - 增强：4-way 边界锚点 (首尾帧全局可见)           - 状态：前向状态 S→ 与后向状态 S←
             │                                               │
             ▼ (门控 σ 抑制过拟合)                           ▼ (RMSNorm + 门控 σ)
             └───────────────────────┬───────────────────────┘
                                     ▼
                             加权合并送入残差流
```

### 核心设计细节 1：为什么必须扣除局部滑窗的帧？（防止 Double-Counting）
时间轴被严格划分为三段不相交的区域：
$$
\underbrace{t_{\text{past}}}_{\text{前向线性记忆 } S_t^{\to}} \quad \cap \quad \underbrace{[t - \Delta, t + \Delta]}_{\text{局部 Softmax 滑窗}} \quad \cap \quad \underbrace{t_{\text{future}}}_{\text{后向线性记忆 } S_t^{\leftarrow}} = \emptyset
$$

若在线性记忆中不扣除局部窗口的帧，则这些关键帧在 Softmax 和 Linear 两个分支中被**重复计算了两次**，导致局部能量严重过载、背景闪烁崩溃。

### 核心设计细节 2：Prompt 文本条件的“0.5 + 0.5”对齐初始化
文本提示词必须全局指导前后向生成，作者通过 Delta 规则将 Prompt 写入初始状态，并各缩放 $\frac{1}{2}$：
$$
S_0^{\to} = S_0^{\leftarrow} = \frac{1}{2} S_{\text{text}}
$$
在最终将两路线性记忆与 Query $q$ 做读出求和时：
$$
(S_0^{\to} + S_0^{\leftarrow}) q = \left(\frac{1}{2} S_{\text{text}} + \frac{1}{2} S_{\text{text}}\right) q = S_{\text{text}} q
$$
不多不少，精确等价于 100% 原始文本条件强度！

---

## 3. 微观数学内核：从 Token 级 Delta 规则到「帧级正规方程」

这是本论文最精妙、最具理论价值的突破点。

### ① 传统 LLM 线性注意力的 Delta Rule（单 Token 更新）
在传统序列模型（如 GDN、Kimi Linear）中，记忆状态 $S_{t-1}$ 的更新是单 Token 迭代的：
1. **衰减旧记忆**：$\bar{S} = S_{t-1} \text{Diag}(\alpha)$，其中 $\alpha \in [0, 1]$ 为通道衰减门控；
2. **擦除旧关联并写入新关联**：
$$
S_t = \bar{S} + \beta (v - \bar{S} k) k^T
$$
其中 $\beta$ 为写入强度（相当于自适应学习率），$(v - \bar{S} k)$ 为当前记忆预测目标 $v$ 的误差残差。

### ② 视频领域的灾难：SANA-WM 的“大锅饭累加”
在视频中，一帧有 $U$ 个空间 Token（如 $32 \times 32 = 1024$ 个）。  
SANA-WM 试图将所有 Token 简单求和累加进记忆：
$$
S_{\text{SANA}} = \bar{S} + \sum_{u=1}^U \beta_u (v_u - \bar{S} \hat{k}_u) \hat{k}_u^T = \bar{S} \left(I - \frac{A}{U}\right) + \frac{B}{\sqrt{U}}
$$
其中加权键格拉姆矩阵（Key Gram Matrix）：
$$
A = K^T \text{Diag}(\beta) K = \sum_{u=1}^U \beta_u k_u k_u^T
$$

::: danger 蓝天 Token 的冗余累加灾难
- **问题所在**：所有 Token 都在盲目地与**同一个冻结状态 $\bar{S}$** 计算残差，彼此之间没有任何协商与正交化！
- **几何崩溃**：如果一帧画面里有一大片蓝天（几百个 Token 拥有几乎完全相同的 Key 向量 $k$），累加求和后，该方向的特征被暴增了几百倍，直接冲垮记忆矩阵！
- 为了防止爆炸，SANA-WM 被迫加上了全局惩罚除以 $\sqrt{U}$，但这又连带误伤了画面中那些宝贵且独立的稀有 Token（细节直接丢失）。
:::

### ③ VDN 的终极解法：帧内正规方程联合求解 (Joint Normal Equation)
VDN 不再做盲目的梯度累加，而是将整帧的所有 $U$ 个 Token 抽象为一个**多目标最小二乘拟合问题**：

$$
\min_S \quad \frac{1}{2} \|S - \bar{S}\|_F^2 + \frac{1}{2} \sum_{u=1}^U \beta_u \|S k_u - v_u\|_2^2
$$

- **第一项（近端正则项）**：要求新记忆 $S$ 不能偏离历史累积记忆 $\bar{S}$ 太远；
- **第二项（帧内拟合项）**：要求同一个共享状态 $S$ 能够**同时协调满足**这一帧内所有空间 Token 的映射关系 $S k_u \approx v_u$！

对矩阵 $S$ 求导令导数为 0：
$$
\frac{\partial}{\partial S} = (S - \bar{S}) + S \underbrace{\left(\sum_{u=1}^U \beta_u k_u k_u^T\right)}_{A} - \underbrace{\left(\sum_{u=1}^U \beta_u v_u k_u^T\right)}_{B} = 0
$$

整理得到优雅闭式解（Normal Equation）：
$$
S(I + A) = \bar{S} + B \implies \mathbf{S_t = (\bar{S} + B)(I + A)^{-1}}
$$

::: tip 关键定理：为什么 $(I + A)^{-1}$ 完美解决了蓝天冗余问题？
1. **自适应阻尼冗余方向**：
   矩阵 $A \succeq 0$（半正定）。当有很多蓝天 Token 共享相同方向时，对应的特征值 $\lambda_i(A)$ 极大，在转移矩阵 $(I + A)^{-1}$ 中，该方向的缩放系数为：
   $$
   \frac{1}{1 + \lambda_i(A)} \to 0
   $$
   系统**自动将高度重复冗余的方向强力压制**，绝不积累膨胀！
2. **充分保留独立方向**：
   对于画面中独立稀有的独特 Token，$\lambda_j(A) \approx 0$，缩放系数 $\frac{1}{1 + \lambda_j} \approx 1$，原汁原味无损保留！
3. **原生非扩张性（稳定性保证）**：
   对任意特征值 $\lambda_i \ge 0$，恒有 $\frac{1}{1 + \lambda_i} \in (0, 1]$，算子范数 $\|(I + A)^{-1}\|_2 \le 1$。
   **无需任何生硬的人工除以 $\sqrt{U}$ 缩放，数学上绝对不可能梯度爆炸！**
:::
