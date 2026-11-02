<script setup>
import { ref, onMounted } from 'vue'
import { withBase } from 'vitepress'
import RepoHeatmap from './RepoHeatmap.vue'

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

const profileLinks = [
  { text: '个人学术主页', href: 'https://wangyephd.github.io/' },
  { text: 'GitHub (@wangyephd)', href: 'https://github.com/wangyephd' },
  { text: 'StyleX-Research', href: 'https://github.com/StyleX-Research' },
  { text: 'PaperABC', href: 'https://space.bilibili.com/1127990326' },
  { text: '标签穿透矩阵', href: '/tags/', highlight: true },
  { text: 'Idea 灵感池', href: '/ideas/', highlight: true },
  { text: '研习复盘周报', href: '/reviews/', highlight: true },
  { text: '使用指南', href: '/guide/' }
]

const researchAreas = [
  { title: '视频模型 (Video Intelligence)', href: '/domains/video/', description: 'Spatio-Temporal Attention 时空复杂度推导、3D Causal VAE、Video DiT 架构演进与时空物理一致性。' },
  { title: '图像与视觉 (CV & Image)', href: '/domains/vision/', description: 'ControlNet 空间结构可控生成原理解析、Zero-Conv 数学稳定性证明、风格迁移与定制生成。' },
  { title: '语言模型与推理 (LLM)', href: '/domains/llm/', description: 'DeepSeek-R1 慢思考涌现机制、GRPO 组相对策略优化数学目标、强化学习自省与思维链。' },
  { title: '多模态与对齐 (Multimodal)', href: '/domains/multimodal/', description: 'CLIP 跨模态双塔对比学习、InfoNCE 对称损失推导、统一理解与生成表征对齐。' },
  { title: 'AIGC 强化学习后训练 (RL for AIGC)', href: '/domains/rl-aigc/', description: '扩散/流模型后训练全闭环：Reward 覆盖率与鲁棒性、Rubric checklist 自动生产、one-sided veto 反 hacking、LoRA soup 权重集成、Prompt 过滤与评测协议。' },
  { title: '智能体系统与工具编排 (Agent)', href: '/domains/agent/', description: 'Agent Loop 形式化为受控马尔可夫链、结构化输出 q^n 可靠性契约、AoT 依赖图拓扑调度与关键路径分析。' },
  { title: '通用理论基石 (Foundations)', href: '/foundations/', description: 'Transformer 点积注意力方差推导、Flow Matching 连续流生成动力学 ODE 求解。' },
  { title: '算力基建与系统 (Infra)', href: '/infra/', description: 'CUDA 显存层级带宽延迟分析、Roofline 瓶颈判定、vLLM PagedAttention 高并发吞吐优化。' },
  { title: '实战项目与复现 (Projects)', href: '/projects/', description: '开源视频生成模型（Wan2.1）8 卡集群本地复现报告、LLM 7B 模型 LoRA 微调显存与吞吐实测。' },
  { title: '标签穿透矩阵 (Tag Matrix)', href: '/tags/', description: '打破目录隔离，通过 #DiT、#Diffusion、#Attention 等多维技术标签穿透全库聚合联想。' },
  { title: '科研 Idea 灵感池 (Idea Sandbox)', href: '/ideas/', description: '专属脑洞自留地，记录关于跨模态生成、RL 视频时序平滑度、Flow Matching 速度场曲率等奇思妙想。' },
  { title: '研习复盘时光机 (Weekly Reviews)', href: '/reviews/', description: '由 AI Agent 自动提炼归纳的每周/阶段性学术综述，串联跨模态公式推导与实验深度复盘。' }
]

const features = [
  { title: '本地优先与开放格式', description: '底层 100% 纯 Markdown 文本，绝无商业专有格式绑定，终身掌握自己的研习数据。' },
  { title: '论文级 LaTeX 渲染', description: '原生集成 KaTeX/MathJax，矩阵、条件微分方程、期望方差推导达到 arXiv 论文级美感。' },
  { title: 'Git 版本控制与自动化', description: '行级版本回滚追踪，告别传统网盘冲突副本；提交自动触发 GitHub Actions 编译上线。' },
  { title: 'AI Agent 原生协同', description: '预置规范接口，Claude Code、DeepSeek 可直接读取推导、提取论文并自动打标挂载。' }
]

const resolveHref = (href) => href.startsWith('/') ? withBase(href) : href
const isExternal = (href) => !href.startsWith('/')

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
  <main class="dashboard-container">
    <div class="hero-grid">
      <section class="profile-header-card">
        <div class="profile-main">
          <div class="avatar-box">
            <img :src="withBase('/avatar.jpeg')" alt="Ye Wang (王冶)" class="avatar-img" />
          </div>
          <div class="profile-info">
            <div class="profile-name-row">
              <h1 class="user-name">Ye Wang (王冶)</h1>
              <span class="status-tag"><span class="status-dot"></span>在研 · Ph.D. Candidate</span>
            </div>
            <p class="profile-bio">
              聚焦<strong>可控视觉内容生成与编辑 (Controllable Visual Synthesis)</strong>、<strong>视频生成 (Video DiT)</strong>、<strong>统一理解与生成</strong>及<strong>大模型推理与算力优化</strong>。
            </p>
            <div class="profile-links">
              <a
                v-for="link in profileLinks"
                :key="link.text"
                class="chip-link"
                :class="{ highlight: link.highlight }"
                :href="resolveHref(link.href)"
                :target="isExternal(link.href) ? '_blank' : undefined"
                :rel="isExternal(link.href) ? 'noopener' : undefined"
              >
                {{ link.text }}<span class="link-arrow" aria-hidden="true">↗</span>
              </a>
            </div>
            <div class="sync-status"><span class="sync-dot"></span>GitHub Pages 已同步上线</div>
          </div>
        </div>
      </section>

      <section class="daily-quote-card">
        <div class="quote-symbol" aria-hidden="true">“</div>
        <div class="quote-main">
          <p class="quote-text" :class="{ fading: isLoadingQuote }">{{ quote }}</p>
          <div class="quote-meta">
            <span class="quote-source">—— {{ quoteFrom }}</span>
            <button
              class="refresh-quote-btn"
              title="点击换一句灵感"
              aria-label="换一句灵感"
              :disabled="isLoadingQuote"
              @click="fetchDailyQuote"
            >
              <span class="refresh-icon" :class="{ spin: isLoadingQuote }">↻</span>
              <span>换一句</span>
            </button>
          </div>
        </div>
      </section>
    </div>

    <section class="dashboard-section activity-section">
      <RepoHeatmap />
    </section>

    <section class="dashboard-section">
      <div class="section-title-row">
        <h3>研究模态与知识库看板</h3>
        <span class="sub-text">点击直达对应领域知识树</span>
      </div>

      <div class="grid-cards">
        <a
          v-for="(area, index) in researchAreas"
          :key="area.href"
          :href="withBase(area.href)"
          class="nav-card"
        >
          <div class="card-topline">
            <span class="card-index">{{ String(index + 1).padStart(2, '0') }}</span>
            <span class="card-arrow" aria-hidden="true">↗</span>
          </div>
          <div class="card-body">
            <h4>{{ area.title }}</h4>
            <p>{{ area.description }}</p>
          </div>
        </a>
      </div>
    </section>

    <section class="dashboard-section supporting-section">
      <div class="section-title-row">
        <h3>系统功能与底层支撑特性</h3>
        <span class="sub-text">工程级设计保障</span>
      </div>

      <div class="features-grid">
        <div v-for="feature in features" :key="feature.title" class="feature-card">
          <h5>{{ feature.title }}</h5>
          <p>{{ feature.description }}</p>
        </div>
      </div>
    </section>

    <section class="dashboard-section tips-section">
      <div class="section-title-row">
        <h3>系统操作备忘与使用提示</h3>
        <a :href="withBase('/guide/')" class="more-link">完整使用指南 <span aria-hidden="true">↗</span></a>
      </div>

      <div class="tips-container">
        <div class="tip-item">
          <span class="tip-badge">快捷检索</span>
          <p>随时在任意页面按 <code>Cmd + K</code>（Windows 下 <code>Ctrl + K</code>）呼出全文搜索框，毫秒级模糊匹配公式变量、算法名与代码段。</p>
        </div>
        <div class="tip-item">
          <span class="tip-badge">AI 协同</span>
          <p>呼叫本地 Agent（Claude Code / DeepSeek）精读论文时，直接命令：<em>“精读这篇论文，提取 LaTeX 公式并存入 docs/domains/video/，同步更新侧边栏”</em>。</p>
        </div>
        <div class="tip-item">
          <span class="tip-badge">双端同步</span>
          <p>本地执行 <code>git push</code> 后，GitHub Actions 自动化流水线将在 30 秒内完成网页构建与更新，线上端只读防手滑。</p>
        </div>
      </div>
    </section>
  </main>
</template>

<style scoped>
.dashboard-container {
  --dashboard-display: "Iowan Old Style", "Palatino Linotype", "Noto Serif SC", "Songti SC", serif;
  max-width: 1180px;
  margin: 0 auto;
  padding: 2.5rem 2rem 5rem;
}

.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.75fr);
  gap: 3rem;
  align-items: stretch;
  padding-bottom: 3rem;
  border-bottom: 1px solid var(--vp-c-divider);
}

.profile-header-card {
  padding: 0.5rem 0;
}

.profile-main {
  display: flex;
  align-items: flex-start;
  gap: 1.5rem;
}

.avatar-box {
  width: 88px;
  height: 88px;
  border-radius: 22px;
  overflow: hidden;
  flex-shrink: 0;
  background: var(--vp-c-bg-mute);
  box-shadow: 0 10px 30px rgba(28, 30, 38, 0.1);
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
  gap: 0.9rem;
  flex-wrap: wrap;
  margin-bottom: 0.8rem;
}

.user-name {
  margin: 0;
  font-family: var(--dashboard-display);
  font-size: clamp(1.85rem, 3vw, 2.45rem);
  font-weight: 600;
  letter-spacing: -0.035em;
  line-height: 1.15;
  color: var(--vp-c-text-1);
}

.status-tag,
.sync-status {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}

.status-dot,
.sync-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #5b8d72;
}

.sync-status {
  margin-top: 1.1rem;
}

.sync-dot {
  background: var(--vp-c-brand-1);
}

.profile-bio {
  max-width: 660px;
  margin: 0 0 1.3rem;
  color: var(--vp-c-text-2);
  font-size: 0.92rem;
  line-height: 1.85;
}

.profile-bio strong {
  color: var(--vp-c-text-1);
  font-weight: 550;
}

.profile-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 1rem;
}

.chip-link {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  color: var(--vp-c-text-2);
  font-size: 0.76rem;
  text-decoration: none;
  transition: color 160ms ease;
}

.chip-link.highlight {
  color: var(--vp-c-brand-1);
}

.chip-link:hover {
  color: var(--vp-c-brand-1);
}

.link-arrow {
  opacity: 0;
  transform: translate(-2px, 2px);
  transition: opacity 160ms ease, transform 160ms ease;
}

.chip-link:hover .link-arrow {
  opacity: 1;
  transform: translate(0, 0);
}

.daily-quote-card {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 250px;
  padding: 1.6rem 1.7rem;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  border-radius: 18px;
  background: var(--vp-c-bg-soft);
}

.quote-symbol {
  height: 2.2rem;
  color: var(--vp-c-brand-1);
  font-family: Georgia, serif;
  font-size: 3.8rem;
  line-height: 1;
  opacity: 0.35;
}

.quote-text {
  margin: 0.8rem 0 1.4rem;
  color: var(--vp-c-text-1);
  font-family: var(--dashboard-display);
  font-size: 1.02rem;
  line-height: 1.9;
  transition: opacity 200ms ease;
}

.quote-text.fading {
  opacity: 0.35;
}

.quote-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
}

.quote-source {
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
}

.refresh-quote-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  flex-shrink: 0;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
  cursor: pointer;
}

.refresh-quote-btn:hover:not(:disabled) {
  color: var(--vp-c-brand-1);
}

.refresh-icon {
  display: inline-block;
}

.refresh-icon.spin {
  animation: quote-spin 0.8s linear infinite;
}

@keyframes quote-spin {
  to { transform: rotate(360deg); }
}

.dashboard-section {
  margin-top: 4rem;
}

.activity-section {
  margin-top: 2.5rem;
}

.section-title-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.4rem;
}

.section-title-row h3 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: 1.12rem;
  font-weight: 650;
  letter-spacing: -0.015em;
}

.sub-text {
  color: var(--vp-c-text-3);
  font-size: 0.75rem;
}

.grid-cards {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem;
}

.nav-card {
  display: flex;
  flex-direction: column;
  min-height: 225px;
  padding: 1.25rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg);
  text-decoration: none;
  transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
}

.nav-card:hover {
  border-color: var(--vp-c-brand-2);
  box-shadow: 0 12px 32px rgba(35, 38, 52, 0.07);
  transform: translateY(-3px);
}

.card-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 2.4rem;
}

.card-index {
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
  font-size: 0.7rem;
  letter-spacing: 0.08em;
}

.card-arrow {
  color: var(--vp-c-text-3);
  font-size: 1rem;
  transition: color 180ms ease, transform 180ms ease;
}

.nav-card:hover .card-arrow {
  color: var(--vp-c-brand-1);
  transform: translate(2px, -2px);
}

.card-body {
  margin-top: auto;
}

.card-body h4 {
  margin: 0 0 0.65rem;
  color: var(--vp-c-text-1);
  font-size: 0.98rem;
  font-weight: 620;
  line-height: 1.55;
}

.card-body p {
  display: -webkit-box;
  margin: 0;
  overflow: hidden;
  color: var(--vp-c-text-2);
  font-size: 0.78rem;
  line-height: 1.75;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0 2.5rem;
}

.feature-card {
  padding: 1.3rem 0;
  border-top: 1px solid var(--vp-c-divider);
}

.feature-card h5 {
  margin: 0 0 0.5rem;
  color: var(--vp-c-text-1);
  font-size: 0.9rem;
  font-weight: 620;
}

.feature-card p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.78rem;
  line-height: 1.75;
}

.more-link {
  color: var(--vp-c-brand-1);
  font-size: 0.78rem;
  text-decoration: none;
}

.tips-container {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem;
}

.tip-item {
  padding: 1.2rem;
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
}

.tip-badge {
  display: block;
  margin-bottom: 0.7rem;
  color: var(--vp-c-brand-1);
  font-size: 0.75rem;
  font-weight: 650;
}

.tip-item p {
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.76rem;
  line-height: 1.8;
}

.tip-item code {
  padding: 0.1rem 0.3rem;
  border-radius: 4px;
  background: var(--vp-code-bg);
  color: var(--vp-code-color);
  font-size: 0.75rem;
}

@media (max-width: 1024px) {
  .hero-grid {
    grid-template-columns: minmax(0, 1fr);
    gap: 2rem;
  }

  .profile-header-card,
  .profile-info,
  .daily-quote-card {
    min-width: 0;
  }

  .daily-quote-card {
    min-height: 0;
  }

  .grid-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .tips-container {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .dashboard-container {
    padding: 1.5rem 1rem 3rem;
  }

  .profile-main {
    flex-direction: column;
  }

  .avatar-box {
    width: 72px;
    height: 72px;
    border-radius: 18px;
  }

  .hero-grid {
    padding-bottom: 2rem;
  }

  .dashboard-section {
    margin-top: 3rem;
  }

  .section-title-row {
    align-items: flex-start;
  }

  .sub-text {
    display: none;
  }

  .grid-cards,
  .features-grid {
    grid-template-columns: 1fr;
  }

  .nav-card {
    min-height: 190px;
  }

  .card-topline {
    margin-bottom: 1.5rem;
  }

  .quote-meta {
    align-items: flex-start;
    flex-direction: column;
  }
}

@media (prefers-reduced-motion: reduce) {
  .nav-card,
  .card-arrow,
  .link-arrow,
  .quote-text {
    transition: none;
  }

  .refresh-icon.spin {
    animation: none;
  }
}
</style>
