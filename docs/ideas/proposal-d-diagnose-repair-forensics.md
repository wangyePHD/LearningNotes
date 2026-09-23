# 多轮编辑修复：先定位哪坏了，再修，修好就停 (Diagnose-then-Repair)

> **标签**：`Vision` `Forensics` `AIGC Detection` `Agentic`  
> **更新时间**：2026-09-23  
> **参考来源**：FUSED 2608.28302 · GLARE ECCV26 · Mi-Ripple 2609.11317

---

## 1. 问题定义与控制目标

任务：图被AI改了好几轮后，出现格纹ripple、手指多一根、文字糊了。要同时干三件事：判是不是AI改的、指出哪块坏了、把它修好，干净图一张不能动。

现状是断的：检测只给分不修，修复只修不说为什么，多轮退化连成对数据都没有。水印benchmark 2609.16832也说了，几何和生成式编辑最致命。

要攻的是闭环，不是单点：定位要准，修复要稳，什么时候停要自动。

## 2. 架构拓扑与特征注入机理

分三段，全是小模块。

1. 定位，不训练。借GLARE思路，同一autoencoder跑两次：整图重构一次，切块重构一次。
$$\Delta = \|R_{full}(I) - R_{patch}(I)\| - \gamma \cdot \text{complexity}(I)$$
真图切不切差不多，AI图切掉全局上下文后误差跳变，$\Delta$ 大的就是可疑区。加语义复杂度校准，纹理复杂区不误报。

2. 诊断，VLM出文字。输入可疑区，输出结构化描述 $T_{diag}$，比如“右手食指多一节，格纹周期约8px”。人能看懂，修复模型能用。

3. 修复，分两阶段。Stage1强监督：$(I_a, T_{diag}) \to I_c$，先学会听话修。Stage2自纠：只给简单问句 $Q$，模型自己先诊后修：
$$I^{k+1} = G(I^{k}, \hat{T}^{k}_{diag})$$
干净图当终止态，学到输出`“无伪影，停止”`为止。

## 3. 损失函数与数学稳定性推导

$$ \mathcal{L} = \mathcal{L}_{FM} + \lambda \mathcal{L}_{AR}, \quad (Q,I_c) \to (T_{stop}, I_c) $$

* $\mathcal{L}_{FM}$：修复去噪；
* $\mathcal{L}_{AR}$：文本诊断和STOP判断，$\lambda$ 先取0.5。

::: info 为什么要学STOP
多轮自纠最容易过修，越修越假。把干净图喂回去强制学STOP，推理时模型说停就停，干净图误修率直接可测。
:::

## 4. 保真度与过修权衡

两条曲线必画：

* VCoT轮数-质量曲线：前2轮LPIPS/MUSIQ应明显涨，第3轮后走平，走平就停；
* 干净图误修率：干净图进去，像素改动量应<阈值，改多了就是过修。

检测和修复增益要拆开报：定位开/关，修复涨多少，证明定位真有用，不是摆设。

## 5. 核心控制层代码实现

```python
# 定位
r_full = AE(I)
r_patch = stitch([AE(p) for p in patches(I)])
delta = (r_full - r_patch).abs() - gamma*complexity(I)
mask = delta > thresh              # 可疑区，无需训练

# Stage1
T = VLM(I_a, mask)                 # 缺陷描述
I_fix = editor(I_a, T)             # 强监督修复

# Stage2 VCoT
for k in range(K):
    T_hat = diag_model(I_cur, "请修复此图")
    if "无伪影" in T_hat: break
    I_cur = editor(I_cur, T_hat)
```

## 6. 避坑指南与评测基准

检测：GenImage + OpenSDID跨生成器，报AUC/AP + IoU。修复：自造多轮ripple集，报LPIPS/MUSIQ/MANIQA + 20张人评。

基线：GLARE、FUSED、Mi-Ripple、Qwen-Image-Edit直修。

::: warning 避坑要点
1. 训练-free定位在文档、人像上误检>15%就别硬撑，回退轻量监督头；
2. 合成缺陷和真实多轮有gap，数据一半合成一半真实多轮，别全合成；
3. 和GenShield区分：它做通用artifact，本方案只咬多轮ripple和细粒度伪影，加开放benchmark。
:::
