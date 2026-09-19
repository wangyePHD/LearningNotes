<script setup>
import { ref, onMounted } from 'vue'
import { withBase } from 'vitepress'

const quote = ref('生活不可能像你想象得那么好，但也不会像你想象得那么糟。人的脆弱和坚强都超乎自己的想象。')
const quoteFrom = ref('莫泊桑《一生》')
const isLoadingQuote = ref(false)

const fallbackQuotes = [
  { text: '生活不可能像你想象得那么好，但也不会像你想象得那么糟。人的脆弱和坚强都超乎自己的想象。', from: '莫泊桑《一生》' },
  { text: '流水不争先，争的是滔滔不绝。', from: '老子' },
  { text: '真正重要的东西，用眼睛是看不见的，必须用心去感受。', from: '圣埃克苏佩里《小王子》' },
  { text: '没有礁石，就没有美丽的浪花；没有挫折，就没有壮丽的人生。', from: '罗曼·罗兰' },
  { text: '我们都在阴沟里，但仍有人仰望星空。', from: '王尔德' },
  { text: '凡是过往，皆为序章；凡是未来，皆有可期。', from: '莎士比亚' },
  { text: '不积跬步，无以至千里；不积小流，无以成江海。', from: '荀子《劝学》' },
  { text: '追光的人，终会万丈光芒。', from: '研习札记' },
  { text: '慢一点没关系，只要你一直在向前走。', from: '日常自勉' }
]

const fetchDailyQuote = async () => {
  isLoadingQuote.value = true
  try {
    const res = await fetch('https://v1.hitokoto.cn/?c=d&c=e&c=k&c=i', {
      headers: { 'Accept': 'application/json' }
    })
    if (res.ok) {
      const data = await res.json()
      quote.value = data.hitokoto
      const author = data.from_who ? `${data.from_who} · ` : ''
      quoteFrom.value = `${author}《${data.from}》`
    } else {
      throw new Error('API failed')
    }
  } catch (e) {
    const pick = fallbackQuotes[Math.floor(Math.random() * fallbackQuotes.length)]
    quote.value = pick.text
    quoteFrom.value = pick.from
  } finally {
    isLoadingQuote.value = false
  }
}

onMounted(() => {
  fetchDailyQuote()
})
</script>

<template>
  <div class="dashboard-container">
    <!-- 1. 个人学术与研习画像 -->
    <div class="profile-header-card">
      <div class="profile-main">
        <div class="avatar-box">
          <img :src="withBase('/avatar.jpeg')" alt="Ye Wang (王冶)" class="avatar-img" />
        </div>
        <div class="profile-info">
          <div class="profile-name-row">
            <h1 class="user-name">Ye Wang (王冶)</h1>
            <span class="status-tag">🟢 在研 · Ph.D. Candidate</span>
          </div>
          <p class="profile-bio">
            聚焦<strong>可控视觉内容生成与编辑 (Controllable Visual Synthesis)</strong>、<strong>视频生成 (Video DiT)</strong>、<strong>统一理解与生成</strong>及<strong>大模型推理与算力优化</strong>。
          </p>
          <div class="profile-links">
            <a class="chip-link" href="https://wangyephd.github.io/" target="_blank" rel="noopener">🌐 个人学术主页</a>
            <a class="chip-link" href="https://github.com/wangyephd" target="_blank" rel="noopener">💻 GitHub (@wangyephd)</a>
            <a class="chip-link" href="https://github.com/StyleX-Research" target="_blank" rel="noopener">🔬 StyleX-Research</a>
            <a class="chip-link" href="https://space.bilibili.com/1127990326" target="_blank" rel="noopener">📺 PaperABC</a>
            <a class="chip-link highlight" href="/tags/">🏷️ 标签穿透矩阵</a>
            <a class="chip-link highlight" href="/ideas/">💡 Idea 灵感池</a>
            <a class="chip-link" href="/guide/">📖 使用指南</a>
          </div>
        </div>
      </div>
    </div>

    <!-- 1.5 每日一句·心灵鸡汤与研习自勉 (单独一行，楷体/宋体典雅设计) -->
    <div class="daily-quote-card">
      <div class="quote-symbol left">“</div>
      <div class="quote-main">
        <p class="quote-text" :class="{ fading: isLoadingQuote }">
          {{ quote }}
        </p>
        <div class="quote-meta">
          <span class="quote-source">—— {{ quoteFrom }}</span>
          <button 
            class="refresh-quote-btn" 
            title="点击换一句灵感" 
            :disabled="isLoadingQuote"
            @click="fetchDailyQuote"
          >
            <span class="refresh-icon" :class="{ spin: isLoadingQuote }">↻</span>
            <span class="refresh-text">换一句</span>
          </button>
        </div>
      </div>
      <div class="quote-symbol right">”</div>
    </div>

    <!-- 2. GitHub 研习与代码提交热力图 (纯粹展示代码与笔记活跃度) -->
    <div class="dashboard-section">
      <div class="section-title-row">
        <div class="title-with-icon">
          <span class="icon">📈</span>
          <h3>研习打卡与代码提交热力图</h3>
        </div>
        <a class="sub-link" href="https://github.com/wangyephd" target="_blank" rel="noopener">GitHub: @wangyephd ↗</a>
      </div>
      <div class="heatmap-wrapper">
        <div class="heatmap-img-box">
          <img 
            src="https://ghchart.rshah.org/2da44e/wangyephd" 
            alt="Ye Wang's GitHub Commit Heatmap" 
            loading="lazy"
          />
        </div>
      </div>
    </div>

    <!-- 3. 研究模态与知识库看板 (Quick Navigation) -->
    <div class="dashboard-section">
      <div class="section-title-row">
        <div class="title-with-icon">
          <span class="icon">🎯</span>
          <h3>研究模态与知识库看板</h3>
        </div>
        <span class="sub-text">点击直达对应领域知识树</span>
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

        <a href="/projects/" class="nav-card">
          <div class="card-icon">🚀</div>
          <div class="card-body">
            <h4>实战项目与复现 (Projects)</h4>
            <p>开源视频生成模型（Wan2.1）8 卡集群本地复现报告、LLM 7B 模型 LoRA 微调显存与吞吐实测。</p>
          </div>
        </a>

        <a href="/tags/" class="nav-card">
          <div class="card-icon">🏷️</div>
          <div class="card-body">
            <h4>标签穿透矩阵 (Tag Matrix)</h4>
            <p>打破目录隔离，通过 #DiT、#Diffusion、#Attention 等多维技术标签穿透全库聚合联想。</p>
          </div>
        </a>

        <a href="/ideas/" class="nav-card full-span">
          <div class="card-icon">💡</div>
          <div class="card-body">
            <h4>科研 Idea 灵感池 (Idea Sandbox)</h4>
            <p>专属脑洞自留地，记录关于跨模态生成、RL 视频时序平滑度、Flow Matching 速度场曲率等奇思妙想。</p>
          </div>
        </a>
      </div>
    </div>

    <!-- 4. 系统功能与架构特性 (独立功能模块，不再混入热力图) -->
    <div class="dashboard-section">
      <div class="section-title-row">
        <div class="title-with-icon">
          <span class="icon">⚙️</span>
          <h3>系统功能与底层支撑特性</h3>
        </div>
        <span class="sub-text">工程级设计保障</span>
      </div>

      <div class="features-grid">
        <div class="feature-card">
          <div class="f-icon">📦</div>
          <div class="f-content">
            <h5>本地优先与开放格式</h5>
            <p>底层 100% 纯 Markdown 文本，绝无商业专有格式绑定，终身掌握自己的研习数据。</p>
          </div>
        </div>

        <div class="feature-card">
          <div class="f-icon">📐</div>
          <div class="f-content">
            <h5>论文级 LaTeX 渲染</h5>
            <p>原生集成 KaTeX/MathJax，矩阵、条件微分方程、期望方差推导达到 arXiv 论文级美感。</p>
          </div>
        </div>

        <div class="feature-card">
          <div class="f-icon">⚡</div>
          <div class="f-content">
            <h5>Git 版本控制与自动化</h5>
            <p>行级版本回滚追踪，告别传统网盘冲突副本；提交自动触发 GitHub Actions 编译上线。</p>
          </div>
        </div>

        <div class="feature-card">
          <div class="f-icon">🤖</div>
          <div class="f-content">
            <h5>AI Agent 原生协同</h5>
            <p>预置规范接口，Claude Code、DeepSeek 可直接读取推导、提取论文并自动打标挂载。</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 5. 系统操作与 AI 协作备忘 (提醒与操作指南) -->
    <div class="dashboard-section">
      <div class="section-title-row">
        <div class="title-with-icon">
          <span class="icon">💡</span>
          <h3>系统操作备忘与使用提示</h3>
        </div>
        <a href="/guide/" class="more-link">完整使用指南 →</a>
      </div>

      <div class="tips-container">
        <div class="tip-item">
          <span class="tip-badge search">快捷检索</span>
          <p>随时在任意页面按 <code>Cmd + K</code>（Windows 下 <code>Ctrl + K</code>）呼出全文搜索框，毫秒级模糊匹配公式变量、算法名与代码段。</p>
        </div>
        <div class="tip-item">
          <span class="tip-badge agent">AI 协同</span>
          <p>呼叫本地 Agent（Claude Code / DeepSeek）精读论文时，直接命令：<em>“精读这篇论文，提取 LaTeX 公式并存入 docs/domains/video/，同步更新侧边栏”</em>。</p>
        </div>
        <div class="tip-item">
          <span class="tip-badge sync">双端同步</span>
          <p>本地执行 <code>git push</code> 后，GitHub Actions 自动化流水线将在 30 秒内完成网页构建与更新，线上端只读防手滑。</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
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

.avatar-box {
  width: 82px;
  height: 82px;
  border-radius: 18px;
  overflow: hidden;
  border: 2px solid var(--vp-c-divider);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  flex-shrink: 0;
  background: var(--vp-c-bg-mute);
}

.avatar-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
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

/* 每日一句心灵鸡汤卡片 (典雅楷体/宋体排版) */
.daily-quote-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-left: 4px solid var(--vp-c-brand-1);
  border-radius: 12px;
  padding: 1.1rem 1.6rem;
  margin-bottom: 2rem;
  box-shadow: 0 2px 14px rgba(0, 0, 0, 0.02);
  position: relative;
  overflow: hidden;
}

.quote-symbol {
  font-family: "Georgia", "Songti SC", "SimSun", serif;
  font-size: 2.8rem;
  line-height: 1;
  color: var(--vp-c-brand-1);
  opacity: 0.25;
  user-select: none;
  font-weight: bold;
}

.quote-symbol.left {
  margin-right: 1.2rem;
  align-self: flex-start;
}

.quote-symbol.right {
  margin-left: 1.2rem;
  align-self: flex-end;
}

.quote-main {
  flex-grow: 1;
  text-align: center;
  padding: 0.2rem 0;
}

.quote-text {
  font-family: "Kaiti SC", "STKaiti", "KaiTi", "Songti SC", "SimSun", "Noto Serif SC", serif;
  font-size: 1.15rem;
  line-height: 1.8;
  letter-spacing: 0.04em;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem 0;
  transition: opacity 0.3s ease;
  font-weight: 500;
}

.quote-text.fading {
  opacity: 0.3;
}

.quote-meta {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1.2rem;
}

.quote-source {
  font-family: "Kaiti SC", "STKaiti", "KaiTi", "Songti SC", serif;
  font-size: 0.88rem;
  color: var(--vp-c-text-2);
  font-style: italic;
}

.refresh-quote-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.78rem;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-mute);
  border: 1px solid var(--vp-c-divider);
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.refresh-quote-btn:hover:not(:disabled) {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  transform: scale(1.03);
}

.refresh-icon {
  font-size: 0.95rem;
  display: inline-block;
  transition: transform 0.3s ease;
}

.refresh-icon.spin {
  animation: quote-spin 0.8s linear infinite;
}

@keyframes quote-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@media (max-width: 640px) {
  .quote-symbol {
    display: none;
  }
  .quote-text {
    font-size: 1.05rem;
  }
  .daily-quote-card {
    padding: 1rem;
  }
}

/* 分区通用标题 */
.dashboard-section {
  margin-bottom: 2.2rem;
}

.section-title-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.title-with-icon {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.title-with-icon .icon {
  font-size: 1.2rem;
}

.title-with-icon h3 {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 0;
  color: var(--vp-c-text-1);
}

.sub-link {
  font-size: 0.85rem;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.sub-link:hover {
  text-decoration: underline;
}

.sub-text {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
}

.more-link {
  font-size: 0.85rem;
  color: var(--vp-c-brand-1);
  text-decoration: none;
  font-weight: 500;
}

/* 热力图卡片：纯净独立 */
.heatmap-wrapper {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 1.2rem;
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

:root.dark .heatmap-img-box {
  background: #0d1117;
  border-color: #30363d;
}

:root.dark .heatmap-img-box img {
  filter: invert(0.87) hue-rotate(180deg) brightness(0.95) contrast(1.15);
}

/* 研究模态导航网格卡片 */
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

/* 独立系统功能支撑网格 */
.features-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
}

@media (max-width: 768px) {
  .features-grid {
    grid-template-columns: 1fr;
  }
}

.feature-card {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.1rem;
}

.f-icon {
  font-size: 1.5rem;
  flex-shrink: 0;
}

.f-content h5 {
  font-size: 0.92rem;
  font-weight: 600;
  margin: 0 0 0.25rem 0;
  color: var(--vp-c-text-1);
}

.f-content p {
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
  line-height: 1.45;
  margin: 0;
}

/* 提示备忘录 */
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
