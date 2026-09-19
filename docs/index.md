---
layout: page
title: 个人研习仪表盘
---

<div class="dashboard-container">

<!-- 1. 个人研习画像与工作台头部 -->
<div class="profile-header-card">
  <div class="profile-main">
    <div class="avatar-badge">🧠</div>
    <div class="profile-info">
      <div class="profile-name-row">
        <h1 class="user-name">Ye Wang (王冶)</h1>
        <span class="status-tag">🟢 在研 · Ph.D. Candidate</span>
      </div>
      <p class="profile-bio">
        聚焦<strong>可控视觉内容生成与编辑 (Controllable Visual Synthesis)</strong>、<strong>视频生成 (Video DiT)</strong>、<strong>统一理解与生成</strong>及<strong>大模型推理与算力优化</strong>。
      </p>
      <div class="profile-links">
        <a class="chip-link" href="https://wangyephd.github.io/" target="_blank">🌐 个人学术主页</a>
        <a class="chip-link" href="https://github.com/wangyephd" target="_blank">💻 GitHub (@wangyephd)</a>
        <a class="chip-link" href="https://github.com/StyleX-Research" target="_blank">🔬 StyleX-Research</a>
        <a class="chip-link" href="https://space.bilibili.com/1127990326" target="_blank">📺 PaperABC</a>
        <a class="chip-link highlight" href="/guide/">📖 系统使用指南</a>
      </div>
    </div>
  </div>
</div>

<!-- 2. GitHub 提交打卡与研习活跃度热力图 -->
<div class="dashboard-section">
  <div class="section-title-row">
    <h3>📈 研习打卡与代码提交热力图 (Commit Activity)</h3>
    <span class="sub-badge">GitHub 数据同步</span>
  </div>
  <div class="heatmap-wrapper">
    <div class="heatmap-img-box">
      <img 
        src="https://ghchart.rshah.org/2da44e/wangyephd" 
        alt="Ye Wang's GitHub Commit Heatmap" 
        loading="lazy"
      />
    </div>
    <div class="heatmap-footer">
      <div class="stat-pill">📦 本地优先存储 (Local-First)</div>
      <div class="stat-pill">📐 论文级 LaTeX 支持</div>
      <div class="stat-pill">⚡ Git 自动化 CI/CD</div>
      <div class="stat-pill">🤖 AI Agent 协同读写</div>
    </div>
  </div>
</div>

<!-- 3. 功能直达工作台 (Research Domains & Foundations) -->
<div class="dashboard-section">
  <div class="section-title-row">
    <h3>🎯 研究模态与知识库看板 (Quick Navigation)</h3>
    <span class="sub-badge">点击直达对应侧边栏知识树</span>
  </div>

  <div class="grid-cards">
    <a href="/domains/video/" class="nav-card">
      <div class="card-icon">🎬</div>
      <div class="card-body">
        <h4>视频模型 (Video Intelligence)</h4>
        <p>Spatio-Temporal Attention 时空复杂度推导、3D Causal VAE、Video DiT 架构演进与时空物理一致性。</p>
      </div>
    </a>

    <a href="/domains/vision/" class="nav-card">
      <div class="card-icon">🖼️</div>
      <div class="card-body">
        <h4>图像与视觉 (CV & Image)</h4>
        <p>ControlNet 空间结构可控生成原理解析、Zero-Conv 数学稳定性证明、风格迁移与定制生成。</p>
      </div>
    </a>

    <a href="/domains/llm/" class="nav-card">
      <div class="card-icon">💬</div>
      <div class="card-body">
        <h4>语言模型与推理 (LLM)</h4>
        <p>DeepSeek-R1 慢思考涌现机制、GRPO 组相对策略优化数学目标、强化学习自省与思维链。</p>
      </div>
    </a>

    <a href="/domains/multimodal/" class="nav-card">
      <div class="card-icon">🎙️</div>
      <div class="card-body">
        <h4>多模态与对齐 (Multimodal)</h4>
        <p>CLIP 跨模态双塔对比学习、InfoNCE 对称损失推导、统一理解与生成表征对齐。</p>
      </div>
    </a>

    <a href="/foundations/" class="nav-card">
      <div class="card-icon">📐</div>
      <div class="card-body">
        <h4>通用理论基石 (Foundations)</h4>
        <p>Transformer 点积注意力方差推导、Flow Matching 连续流生成动力学 ODE 求解。</p>
      </div>
    </a>

    <a href="/infra/" class="nav-card">
      <div class="card-icon">⚡</div>
      <div class="card-body">
        <h4>算力基建与系统 (Infra)</h4>
        <p>CUDA 显存层级带宽延迟分析、Roofline 瓶颈判定、vLLM PagedAttention 高并发吞吐优化。</p>
      </div>
    </a>

    <a href="/projects/" class="nav-card full-span">
      <div class="card-icon">🚀</div>
      <div class="card-body">
        <h4>实战项目与复现 (Projects & Benchmarks)</h4>
        <p>开源视频生成模型（Wan2.1）8 卡集群本地复现报告、LLM 7B 模型 LoRA 微调显存与吞吐实测。</p>
      </div>
    </a>
  </div>
</div>

<!-- 4. 系统实用操作备忘录 (Functional Reminders) -->
<div class="dashboard-section">
  <div class="section-title-row">
    <h3>💡 知识库使用与 AI 协作备忘</h3>
    <a href="/guide/" class="more-link">查看完整指南 →</a>
  </div>

  <div class="tips-container">
    <div class="tip-item">
      <span class="tip-badge search">快捷检索</span>
      <p>随时在任意页面按 <code>Cmd + K</code>（Windows 下 <code>Ctrl + K</code>）呼出全文搜索框，支持毫秒级匹配公式变量、算法名与代码段。</p>
    </div>
    <div class="tip-item">
      <span class="tip-badge agent">AI 协同</span>
      <p>呼叫本地 Agent（Claude Code / DeepSeek）读论文时，直接命令：<em>“精读这篇论文，提取 LaTeX 公式并存入 docs/domains/video/，同步更新侧边栏”</em>。</p>
    </div>
    <div class="tip-item">
      <span class="tip-badge sync">双端同步</span>
      <p>本地执行 <code>git push</code> 后，GitHub Actions 自动化流水线将在 30 秒内完成网页构建与更新，线上端只读防手滑。</p>
    </div>
  </div>
</div>

</div>

<style>
.dashboard-container {
  max-width: 1040px;
  margin: 0 auto;
  padding: 1.5rem 1rem 3rem 1rem;
}

/* 个人信息卡片 */
.profile-header-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  padding: 1.8rem;
  margin-bottom: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
}

.profile-main {
  display: flex;
  align-items: flex-start;
  gap: 1.5rem;
}

.avatar-badge {
  font-size: 3.2rem;
  background: var(--vp-c-bg-mute);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  width: 76px;
  height: 76px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.profile-name-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 0.4rem;
}

.user-name {
  font-size: 1.6rem;
  font-weight: 700;
  margin: 0;
  color: var(--vp-c-text-1);
}

.status-tag {
  font-size: 0.8rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  background: rgba(46, 160, 67, 0.15);
  color: #2ea043;
  font-weight: 600;
}

.profile-bio {
  font-size: 0.95rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin: 0.4rem 0 1rem 0;
}

.profile-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}

.chip-link {
  display: inline-flex;
  align-items: center;
  font-size: 0.85rem;
  padding: 0.35rem 0.75rem;
  border-radius: 8px;
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-1);
  text-decoration: none;
  border: 1px solid var(--vp-c-divider);
  transition: all 0.2s ease;
}

.chip-link:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  transform: translateY(-1px);
}

.chip-link.highlight {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-2);
  font-weight: 600;
}

/* 分区通用标题 */
.dashboard-section {
  margin-bottom: 2.2rem;
}

.section-title-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.section-title-row h3 {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 0;
  color: var(--vp-c-text-1);
}

.sub-badge {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
}

.more-link {
  font-size: 0.85rem;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

/* 热力图卡片 */
.heatmap-wrapper {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 1.5rem;
}

.heatmap-img-box {
  overflow-x: auto;
  padding: 1rem 0.5rem;
  display: flex;
  justify-content: center;
  background: #ffffff;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}

.heatmap-img-box img {
  max-width: 100%;
  height: auto;
  min-width: 680px;
  border-radius: 4px;
}

/* 深色模式下的 GitHub 热力图自适应转换 */
:root.dark .heatmap-img-box {
  background: #0d1117;
  border-color: #30363d;
}

:root.dark .heatmap-img-box img {
  filter: invert(0.87) hue-rotate(180deg) brightness(0.95) contrast(1.15);
}

.heatmap-footer {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  margin-top: 1.2rem;
  padding-top: 1rem;
  border-top: 1px dashed var(--vp-c-divider);
}

.stat-pill {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-mute);
  padding: 0.25rem 0.65rem;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
}

/* 功能网格卡片 */
.grid-cards {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}

@media (max-width: 768px) {
  .grid-cards {
    grid-template-columns: 1fr;
  }
}

.nav-card {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.2rem;
  text-decoration: none;
  transition: all 0.2s ease;
}

.nav-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
}

.nav-card.full-span {
  grid-column: 1 / -1;
}

.card-icon {
  font-size: 2rem;
  flex-shrink: 0;
}

.card-body h4 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 0.3rem 0;
  color: var(--vp-c-text-1);
}

.card-body p {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  line-height: 1.45;
  margin: 0;
}

/* 提示卡片容器 */
.tips-container {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.tip-item {
  display: flex;
  align-items: flex-start;
  gap: 0.8rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 0.9rem 1.1rem;
}

.tip-badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  white-space: nowrap;
  flex-shrink: 0;
}

.tip-badge.search {
  background: rgba(64, 158, 255, 0.15);
  color: #409eff;
}

.tip-badge.agent {
  background: rgba(103, 58, 183, 0.15);
  color: #9c27b0;
}

.tip-badge.sync {
  background: rgba(230, 162, 60, 0.15);
  color: #e6a23c;
}

.tip-item p {
  font-size: 0.88rem;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.5;
}

.tip-item code {
  font-size: 0.82rem;
  padding: 0.15rem 0.4rem;
  background: var(--vp-c-bg-mute);
  border-radius: 4px;
}
</style>
