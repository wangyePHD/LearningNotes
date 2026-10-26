---
type: method
tags: ['DiT', 'Efficiency']
summary: '保留图像原始长宽比，将不同尺寸的图像转换成变长 Tokens，通过 Packing + VarLen Attention 实现高效训练。'
---

### 知识点：如何高效训练不同分辨率的图像？（Mage-Flow / MOSS-Image）

**核心思想：** 保留图像原始长宽比，将不同尺寸的图像转换成变长 Tokens，通过 Packing + VarLen Attention 实现高效训练。

**1. Native Resolution（保留长宽比）**

例如，1920×1080 的图像可以等比例缩放为 1280×720，而不是强制拉伸为 1024×1024。通常需要限制像素面积，并对齐到模型要求的尺寸倍数。

**2. Variable-Length Tokens（变长图像序列）**

不同分辨率产生不同数量的 Tokens。例如 MOSS-Image：

- 1024×1024 → 4096 Tokens
- 1280×720 → 3600 Tokens

**3. Packing（样本打包）**

例如，样本 A 有 1000 Tokens，样本 B 有 2000 Tokens，将两者拼接成一条 3000 Tokens 的序列，减少 Padding 浪费。

**4. VarLen Attention（变长注意力）**

通过 `cu_seqlens=[0,1000,3000]` 记录样本边界，让 A 和 B 分别计算 Attention，避免不同样本相互干扰。

**5. Progressive Resolution（渐进分辨率训练）**

例如，先在 256 分辨率级别训练，再逐步提高到 512、1024 级别，逐渐增加训练复杂度。

### 核心伪代码

下面是原理性伪代码，并非两篇论文的原始实现：

```
# 1. 渐进分辨率训练
for resolution in [256, 512, 1024]:

    # 2. 保留长宽比，限制图像面积
    images = resize_keep_ratio(
        images,
        max_area=resolution**2,
        align=16
    )

    # 3. 转成变长图文 Tokens
    samples = []
    for image, caption in dataset:
        img_tokens = patchify(VAE.encode(image))
        txt_tokens = TextEncoder(caption)
        samples.append(concat(txt_tokens, img_tokens))

    # 4. Packing：按 Token 预算打包
    packs = pack_samples(
        samples,
        max_tokens=8576
    )

    for pack in packs:
        # 5. 记录每条样本的边界
        lengths = [len(x) for x in pack]
        cu_seqlens = [0] + cumsum(lengths)

        # 6. 拼接序列
        tokens = concat(pack)

        # 7. VarLen Attention：样本之间互不干扰
        output = DiT(
            tokens,
            cu_seqlens=cu_seqlens,
            attention="flash_attn_varlen"
        )

        # 8. 仅对图像 Tokens 计算生成损失
        loss = flow_matching_loss(output)
        loss.backward()
```

实际训练中还需要处理图像与文本的位置编码、独立噪声采样、有效 Token 的 Loss Mask 和梯度累积等细节。上面只展示核心思路。

### 对应论文

- **Mage-Flow：** 强调 Native Resolution Training，使用 Packing + FlashAttention VarLen 高效处理变长图像序列。
- **MOSS-Image：** 使用渐进分辨率训练，Stage 3 保留长宽比，采用 Multi-sample Packing；论文未明确披露底层 VarLen Attention Kernel。

**总结：** Native Resolution 解决不同尺寸图像怎么训练；Packing + VarLen Attention 解决怎么高效训练；Progressive Resolution 解决如何逐步提高训练分辨率。
