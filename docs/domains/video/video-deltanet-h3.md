# VideoDeltaNet on MiniMax H3 (VDN-H3)：超越播放速度的实时视频生成架构

> **标签**：`Video Gen` `DiT` `Hybrid Attention` `Linear Attention` `DMD2` `Real-Time`  
> **更新时间**：2026-09-19  
> **论文/代码**：[OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3) · [技术博客](https://openvdn.github.io/)

---

## 1. 核心定位与里程碑战绩

针对大模型全时空 Softmax 注意力占全模型 85% 耗时的瓶颈，VDN-H3 首次在工业级视频生成底座上实现了**“生成速度超越视频播放速度”**：

- **基准规格**：14.4 秒长视频、768P 高清分辨率、345 帧（约 35 万个全局 Token）；
- **实测耗时**：8× B200 上端到端仅需 **9.0 秒**（去噪 Denoising 仅耗时 **6.9 秒**，8 步去噪）；
- **画质表现**：在 T2V（文生视频）、I2V（图生视频）、FL2VA（首尾帧生视频）中，画质高度逼近原始 MiniMax H3 稠密底座，彻底终结了“线性注意力必糊”的魔咒。

---

## 2. 五大核心技术点解构 (做了什么 vs 解决了什么)

### ① 双轨混合架构 (Hybrid Attention)
- **做了什么**：将全时空稠密注意力拆解为互补的两轨：
  - **局部轨**：Chunk 对齐的滑动窗口 Softmax（$K=5$ 帧为一个因果单元，每帧只看前后相邻 Chunk）；
  - **长程轨**：双向线性状态扫描（前向 $S^\to$ 记历史，后向 $S^\leftarrow$ 记未来，严格抠除滑窗范围防重复计数）；
  - **定海神针**：引入 4-way 边界锚点（首尾帧与全序列全局互看）。
- **解决了什么**：打破全时空 $\mathcal{O}(N^2)$ 的算力墙，**单层 Attention 模块速度提升 2.65 倍**，且仅以 3.57% 的微小计算量锁死了长视频的主角身份，杜绝漂移。
- **源码入口**：`src/models/hybrid_attention.py` (`HybridAttention`)、`src/models/softmax_attention/window.py` (`window_bounds`, `window_softmax_reference`)。

---

### ② 帧级正规方程自适应更新 (Video Delta Attention)
- **做了什么**：推导出与分辨率 $U$ 解耦的带近端正则化的帧级正规方程闭式解：
  $$
  S_t = (\bar{S} + B)(I + A)^{-1}, \quad \text{其中 } A = K^T \text{Diag}(\beta) K
  $$
- **解决了什么**：彻底摒弃了上代方法（SANA-WM）粗暴除以 Token 数 $\frac{1}{U}$ 的缺陷。转移矩阵 $(I + A)^{-1}$ 沿各特征方向缩放因子为 $\frac{1}{1 + \lambda_i}$：
  - **大面积重复背景**（$\lambda_i \gg 1$）：自动强力阻尼，消除冗余堆叠；
  - **正交独立细节**（$\lambda_i \approx 0$）：缩放乘子为 1，全强度无损保留，消除了高频画面模糊。
- **源码入口**：`src/models/linear_attention/scan.py` (`frame_statistics`, `gather_linear_state`)、`src/models/linear_attention/branch.py` (`BidirectionalLinearBranch`)。

---

### ③ 冻结底座的三阶段微创训练法 (3-Stage Adaptation)
- **做了什么**：完全冻结原始 MiniMax H3 70GB+ 的庞大预训练底座，仅微调旁路：
  - **Stage A1**：逐层对齐（Layer-wise Alignment），单独校准每一层的线性分支；
  - **Stage A2**：端到端分支适配（End-to-End），串联所有层联合优化线性分支参数；
  - **Stage B**：全模型协同微调，仅训练 QKVO 上的轻量 LoRA 和线性分支，解冻 Softmax 门控。
- **解决了什么**：**省去了数百万美元的预训练烧卡成本**，使任何已有的大型视频开源底座都能以即插即用（Plug-and-play）的方式低成本完成架构改装。
- **源码入口**：`src/training/train_stage_a1.py`、`src/training/train_stage_a2.py`、`src/training/train_stage_b.py`。

---

### ④ DMD2 少步数分布匹配蒸馏 (8-Step Distillation)
- **做了什么**：基于社区 Turbo-LoRA 进一步应用 DMD2（Distribution Matching Distillation 2）进行无 GAN 损失的极速蒸馏训练，将常规去噪步数从 50 步压缩至 **8 步**。
- **解决了什么**：去噪迭代次数削减 **84%**，在保留高动态真实物理运动的同时，抹去了绝大多数低步数蒸馏容易出现的结构伪影。
- **源码入口**：`src/training/dmd.py`、`src/training/train_stage_dmd.py`。

---

### ⑤ 异构分支多卡切分 (5:3 Branch-Specialized Parallelism)
- **做了什么**：在 8 张 GPU 上做序列并行时，不让每张卡平均计算两路分支，而是通过非均匀 All-to-All 通信进行负载均衡：
  - **5 张 GPU**：专门处理计算密度较大的 Softmax 滑窗分支；
  - **3 张 GPU**：专门处理显存吞吐型的 VDA 线性记忆分支。
- **解决了什么**：彻底消除了异构计算下的多卡木桶效应（Straggler Problem），在标准 Ulysses 算法基础上将跨卡通信延迟再压缩 **13.3%**，达成了单步 0.88 秒的极致吞吐。
- **源码入口**：`src/inference/utils/ulysses.py`、`src/inference/utils/ulysses_runtime.py`。

---

## 3. 源码 Review 导航速查表 (Code Review Roadmap)

当你开始阅读和单步调试官方代码时，建议按以下核心文件链路推进：

| 模块类别 | 核心源文件路径 | 核心类 / 函数 / 变量 | 代码 Review 聚焦要点 |
| :--- | :--- | :--- | :--- |
| **混合调度中枢** | `src/models/hybrid_attention.py` | `HybridAttention.forward` | 查看两路分支如何分别过门控并相加：`softmax_gate(x)` 与 `output_gate(x)` |
| **滑窗与锚点掩码** | `src/models/softmax_attention/window.py` | `window_bounds`, `window_softmax_reference` | 重点看 `anchor_frames in ("rows", "both")` 如何实现首尾帧全局双向连通 |
| **线性分支前向** | `src/models/linear_attention/branch.py` | `BidirectionalLinearBranch` | 重点看第 54 行 `TEXT_STATE_SCALE = 0.5` 的文本 Prompt 对称平分初始化 |
| **正规方程求解** | `src/models/linear_attention/scan.py` | `frame_statistics`, `gather_linear_state` | 重点看 `A = K^T Diag(\beta) K` 与 `(I + A)^{-1}` 预条件矩阵的求解与应用 |
| **多卡异构通信** | `src/inference/utils/ulysses.py` | `infer_ulysses` | 查看 8 卡集群下 5 卡（Softmax）与 3 卡（Linear）非均匀 All-to-All 切分逻辑 |
