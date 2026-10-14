# 🎲 AIGC 强化学习后训练 (RL for AIGC) 细粒度编写规范

当用户口述 **RL 奖励体系设计 / RL Prompt 数据生产 / RL 训练配置 / RL 评测协议 / Reward Hacking 防御 / 偏好模型（Reward Model）训练与蒸馏** 相关知识时，必须遵循本规范。

本领域与 `vision-customization.md` 的分工：`vision` 关注**基模能力构建**（架构、VAE、蒸馏、数据引擎），本领域（RL-AIGC）关注**后训练阶段的优化器 + 奖励 + 数据 + 评测**闭环。算法侧（如 GRPO / DPO / NFT / Flow-GRPO）若以奖励体系为主要变量，落在本领域。

**子专题路由**：

| 用户内容 | 存放位置 |
| :--- | :--- |
| 奖励**体系**怎么组合（多奖励、Gating、Veto、归一化） | `docs/domains/rl-aigc/<论文>.md` |
| **Reward Model 本身**怎么造（结构设计、pairwise→pointwise、偏好标签转 Reasoning 监督、RM 评测） | `docs/domains/rl-aigc/preference-model/<论文>.md` |
| **优化过程里奖励与分布怎么被钻空子**（Goodhart 饱和、mode collapse、具体 hack 形态与检测器、惩罚项设计、信任域、rollout 过滤、分布塌缩与 off-policy 失配） | `docs/domains/rl-aigc/reward-hacking/<论文>.md` |

判别问句：

- **「打分器怎么设计/怎么训练」** → 偏好模型子专题
- **「多个打分器怎么一起用」** → 本层
- **「模型怎么利用奖励的漏洞 / 训练分布怎么漂」** → Reward Hacking 子专题

---

## 0. 分 Part 写入铁律（优先级最高，高于本文件其他一切规定）

**只写用户口述到的 Part 范围。**

- 即使论文 PDF 完整可被读取，**也不得因为能读到原文就擅自补写用户尚未提供的章节**；
- 用户给的是 Part 1，就**只写 Part 1**，结构按该 Part 的自然顺序切分（Abstract → Introduction → 方法概览 → Related Work → 研究者视角 → 小结）；
- 公式、表格、超参、消融数字**只能来自用户提供的笔记内容**；用户笔记中已指向的论文编号（如「§3.3.1 的公式 (3)–(9)」）只允许作为**精读提醒**保留，**不得自行搬运公式或数值内容**；
- 末尾**照抄**用户给出的「下一 Part 预告」；
- `精读进度` 行必须如实标注当前 Part，**禁止**写「已同步补全 §3/§4/附录」；
- 元信息头只填用户给过的字段（标题、arXiv、基模、日期、标签、精读进度），**不补作者、机构、页数、算力**。

---

## 1. 结构大纲标准（必须包含以下 8 个小节）

```markdown
# [中文标题] ([来源/机构])

> **标签**：`RL` `Reward System` `Rubric` `[其他标签]`
> **更新时间**：YYYY-MM-DD
> **arXiv**：[编号](链接)
> **涉及基模**：[论文涉及的 base model 列表]
> **精读进度**：**Part 1 完结**（Abstract / §1 Introduction / §2 Related Work）
> **专区**：[🎲 AIGC 强化学习后训练](./index.md)

## 1. 总体：作者究竟提出了什么
- Abstract 逻辑的完整还原；
- 两类奖励体系（Preference Reward / Rubric-based Reward）的对照表；
- 核心贡献的 `$$ \boxed{} $$` 抽象 + Coverage / Composition / Robustness 三关键词解释。

## 2. Introduction：为什么单一 Preference Reward 不够
- 概念性示例（**必须声明「非论文对照实验」**）；
- 被忽略的机制：标量压缩，用 `$$ R_{\mathrm{pref}} \approx f(\cdots) $$` 抽象；
- 论文给出的实验证据（Figure / Table 编号 + 用户笔记中出现的具体数值）。

## 3. Reward System 拓扑
- 用一条流程公式或表格列出各奖励的职责；
- 用表格区分 **presence-oriented** 与 **absence-oriented** 奖励，并给出「为什么不能无条件启用」的论证；
- 说明 Anti-Hack 为何不是加项（One-sided Veto）。

## 4. 为什么不直接加权相加
- 原始分数表（尺度 / 分布 / 含义各不相同）；
- 简单相加 `$$ R = R_1 + R_2 + R_3 $$` 的失效原因；
- 作者设计的机制对照表（机制 → 解决的问题 → 主要目的）；
- 用 `::: warning` 标注论文引言表述与正文公式的不一致，指明后续以哪个为准。

## 5. Introduction 里的实验与贡献声明
- Table 编号 + 快照日期 + 每个基模的 before/after 与增益；
- 作者的额外承诺（如公开数据子集）；
- 用 `::: danger` 写研究者视角的保留意见（评测平台依赖、同平台训练/评测的泛化区分）。

## 6. Related Work 三分类定位
- 第一类 RL 算法：表格列方向，并说明本文为何选择 DiffusionNFT；
- 第二类 Preference / Quality Reward：按「总体偏好模型 vs 可验证评分器」分组，列出本文对比的具体 RM；
- 第三类 Rubric Reward：给出 rubric 拆分的示例与自动生产相关工作表；
- 用 `::: warning` 澄清「本文不是最早提出 X，贡献偏系统集成」。

## 7. 从我们的角度：Part 1 最应该学到什么
- 用户口述的每一条认识单独成一个 `::: tip` / `::: danger` 容器；
- 保留用户自己给出的研究关联判断（如与 RL 数据生产的衔接）。

## 8. Part 1 小结与下一 Part 预告
- 一句话总结核心主张；
- 已覆盖范围；
- **照抄用户给出的下一 Part 预告原文**。
```

---

## 2. 本领域专属铁律

1. **数字必须来自用户提供的笔记**：Elo、win rate、样本量、百分比一律不得自行从 PDF 补齐；
2. **公式优先于文字**：用户笔记中若已指出论文表述冲突（如「引言说 $[0,1]$ vs §3.3.1 是逐维 z-score」），必须显式保留该冲突并指明后续以哪个为准，但**不得自行把公式内容写进笔记**；
3. **概念性示例必须声明**：用户笔记中标明「非论文对照实验」的内容，笔记中必须保留该免责说明；
4. **区分「reward 上升」与「真实质量上升」**：若用户笔记中已有相关论证，保留原文；无则不自行补充；
5. **与既有笔记互链**：涉及 NFT / GRPO / 多奖励归一化时，链回 `docs/domains/vision/image-rl-posttraining/` 下的对应笔记；
6. **用户口述的「未公开 / 必须留白」清单要单独成节**：例如数据池总量、保留率、切分算法、温度参数等，一律原样转录成「是否披露」对照表，**不得替作者补齐**。