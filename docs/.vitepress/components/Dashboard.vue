<script setup>
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as notes } from '../data/notes.data.ts'
import RepoHeatmap from './RepoHeatmap.vue'

const areas = [
  {
    title: '视频生成与理解',
    en: 'Video',
    href: '/domains/video/',
    dir: 'domains/video',
    description: '时空建模、Video DiT 与视频基础模型'
  },
  {
    title: '图像与计算机视觉',
    en: 'Vision',
    href: '/domains/vision/',
    dir: 'domains/vision',
    description: '可控生成、图像编辑与视觉表征'
  },
  {
    title: '语言模型与推理',
    en: 'Language',
    href: '/domains/llm/',
    dir: 'domains/llm',
    description: '推理模型、强化学习与思维链'
  },
  {
    title: '多模态与跨模态对齐',
    en: 'Multimodal',
    href: '/domains/multimodal/',
    dir: 'domains/multimodal',
    description: '对比学习、统一理解与生成'
  },
  {
    title: 'AIGC 强化学习后训练',
    en: 'RL for AIGC',
    href: '/domains/rl-aigc/',
    dir: 'domains/rl-aigc',
    description: '奖励体系、偏好模型与优化方法'
  },
  {
    title: '智能体与工具编排',
    en: 'Agent',
    href: '/domains/agent/',
    dir: 'domains/agent',
    description: 'Agent Loop、规划与结构化输出'
  },
  {
    title: '通用理论基石',
    en: 'Foundations',
    href: '/foundations/',
    dir: 'foundations',
    description: '注意力、生成动力学与数学推导'
  },
  {
    title: '算力与系统工程',
    en: 'Infrastructure',
    href: '/infra/',
    dir: 'infra',
    description: 'CUDA、推理系统与训练工程'
  }
]

const quickLinks = [
  { title: '每日 arXiv', description: '扫描与筛选', href: '/daily-arxiv/' },
  { title: '研习白板', description: '当前攻关', href: '/whiteboard/' },
  { title: '科研灵感', description: '假设与提案', href: '/ideas/' },
  { title: '研习复盘', description: '阶段性总结', href: '/reviews/' }
]

const profileLinks = [
  { text: '学术主页', href: 'https://wangyephd.github.io/' },
  { text: 'GitHub', href: 'https://github.com/wangyephd' },
  { text: 'StyleX Research', href: 'https://github.com/StyleX-Research' },
  { text: 'PaperABC', href: 'https://space.bilibili.com/1127990326' }
]

const recentNotes = computed(() => notes.slice(0, 6))
const countFor = dir => notes.filter(note => note.dir === dir || note.dir.startsWith(`${dir}/`)).length
const cleanTitle = title => title.replace(/^[^\p{L}\p{N}]+/u, '')
</script>

<template>
  <main class="workspace">
    <section class="workspace-hero">
      <div class="hero-copy">
        <p class="eyebrow">Ye Wang · Personal research library</p>
        <h1>AlgoNotes<span class="title-dot">.</span></h1>
        <p class="hero-intro">
          王冶的研究与工程知识库。聚焦可控视觉生成与编辑、视频生成、统一理解及大模型推理。
        </p>
        <div class="hero-actions">
          <a class="primary-action" :href="withBase('/domains/vision/')">进入研究笔记 <span aria-hidden="true">↗</span></a>
          <a class="text-action" :href="withBase('/tags/')">按标签浏览 <span aria-hidden="true">→</span></a>
        </div>
      </div>

      <div class="library-summary" aria-label="知识库概览">
        <div class="summary-item">
          <strong>{{ String(notes.length).padStart(2, '0') }}</strong>
          <span>篇笔记</span>
        </div>
        <div class="summary-item">
          <strong>{{ String(areas.length).padStart(2, '0') }}</strong>
          <span>个知识板块</span>
        </div>
        <p>Markdown · Git versioned · Local first</p>
      </div>
    </section>

    <nav class="quick-nav" aria-label="快捷入口">
      <span class="quick-nav-label">工作台</span>
      <a v-for="link in quickLinks" :key="link.href" :href="withBase(link.href)" class="quick-link">
        <span>
          <strong>{{ link.title }}</strong>
          <small>{{ link.description }}</small>
        </span>
        <span class="quick-arrow" aria-hidden="true">↗</span>
      </a>
    </nav>

    <section class="workspace-section">
      <div class="section-heading">
        <div>
          <p class="section-kicker">Library</p>
          <h2>研究知识库</h2>
        </div>
        <span class="section-note">按研究方向浏览</span>
      </div>

      <div class="area-grid">
        <a v-for="(area, index) in areas" :key="area.href" :href="withBase(area.href)" class="area-item">
          <div class="area-topline">
            <span class="area-number">{{ String(index + 1).padStart(2, '0') }}</span>
            <span class="area-count">{{ countFor(area.dir) }} 篇</span>
          </div>
          <h3>{{ area.title }}</h3>
          <p>{{ area.description }}</p>
          <span class="area-en">{{ area.en }}</span>
        </a>
      </div>
    </section>

    <section class="workspace-section recent-section">
      <div class="section-heading">
        <div>
          <p class="section-kicker">Recently updated</p>
          <h2>最近更新</h2>
        </div>
        <a class="section-link" :href="withBase('/tags/')">查看全部 {{ notes.length }} 篇 <span aria-hidden="true">→</span></a>
      </div>

      <div class="recent-list">
        <a v-for="note in recentNotes" :key="note.url" :href="withBase(note.url)" class="recent-item">
          <span class="recent-domain">{{ note.domain }}</span>
          <span class="recent-title">{{ cleanTitle(note.title) }}</span>
          <time :datetime="note.date">{{ note.date }}</time>
          <span class="recent-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
    </section>

    <section class="workspace-section activity-section">
      <RepoHeatmap />
    </section>

    <footer class="workspace-footer">
      <span>Ye Wang · 王冶</span>
      <div>
        <a v-for="link in profileLinks" :key="link.href" :href="link.href" target="_blank" rel="noopener">
          {{ link.text }} <span aria-hidden="true">↗</span>
        </a>
        <a :href="withBase('/projects/')">实战项目 <span aria-hidden="true">→</span></a>
        <a :href="withBase('/guide/')">使用指南 <span aria-hidden="true">→</span></a>
      </div>
    </footer>
  </main>
</template>

<style scoped>
.workspace {
  max-width: 1160px;
  margin: 0 auto;
  padding: 4.5rem 2rem 3rem;
}

.workspace-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 4rem;
  align-items: end;
  padding-bottom: 3.5rem;
}

.eyebrow,
.section-kicker {
  margin: 0 0 1rem;
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.hero-copy h1 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: clamp(3.4rem, 8vw, 6.5rem);
  font-weight: 650;
  letter-spacing: -0.075em;
  line-height: 0.95;
}

.title-dot {
  color: var(--vp-c-brand-1);
}

.hero-intro {
  max-width: 610px;
  margin: 1.8rem 0 0;
  color: var(--vp-c-text-2);
  font-size: 1rem;
  line-height: 1.85;
}

.hero-actions {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  margin-top: 2rem;
}

.primary-action,
.text-action {
  text-decoration: none;
}

.primary-action {
  display: inline-flex;
  align-items: center;
  gap: 1.8rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  background: var(--vp-c-text-1);
  color: var(--vp-c-bg);
  font-size: 0.8rem;
  font-weight: 600;
  transition: opacity 160ms ease;
}

.primary-action:hover {
  opacity: 0.82;
}

.text-action,
.section-link {
  color: var(--vp-c-text-2);
  font-size: 0.78rem;
}

.text-action:hover,
.section-link:hover {
  color: var(--vp-c-brand-1);
}

.library-summary {
  display: grid;
  grid-template-columns: repeat(2, minmax(92px, 1fr));
  gap: 1.5rem;
  min-width: 250px;
  padding-left: 2rem;
  border-left: 1px solid var(--vp-c-divider);
}

.summary-item {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.summary-item strong {
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 1.8rem;
  font-weight: 500;
  letter-spacing: -0.05em;
}

.summary-item span,
.library-summary p {
  color: var(--vp-c-text-3);
  font-size: 0.7rem;
}

.library-summary p {
  grid-column: 1 / -1;
  margin: 0.2rem 0 0;
  font-family: var(--vp-font-family-mono);
  line-height: 1.6;
}

.quick-nav {
  display: grid;
  grid-template-columns: 110px repeat(4, minmax(0, 1fr));
  border-top: 1px solid var(--vp-c-divider);
  border-bottom: 1px solid var(--vp-c-divider);
}

.quick-nav-label {
  display: flex;
  align-items: center;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
}

.quick-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  min-height: 82px;
  padding: 1rem 1.2rem;
  border-left: 1px solid var(--vp-c-divider);
  text-decoration: none;
  transition: background 160ms ease;
}

.quick-link:hover {
  background: var(--vp-c-bg-soft);
}

.quick-link > span:first-child {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.quick-link strong {
  color: var(--vp-c-text-1);
  font-size: 0.82rem;
  font-weight: 600;
}

.quick-link small {
  color: var(--vp-c-text-3);
  font-size: 0.68rem;
}

.quick-arrow {
  color: var(--vp-c-text-3);
  font-size: 0.8rem;
}

.workspace-section {
  margin-top: 5rem;
}

.section-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.6rem;
}

.section-kicker {
  margin-bottom: 0.55rem;
  color: var(--vp-c-text-3);
}

.section-heading h2 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: 1.45rem;
  font-weight: 650;
  letter-spacing: -0.035em;
}

.section-note {
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
}

.area-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border-top: 1px solid var(--vp-c-divider);
  border-left: 1px solid var(--vp-c-divider);
}

.area-item {
  position: relative;
  min-height: 190px;
  padding: 1.25rem;
  border-right: 1px solid var(--vp-c-divider);
  border-bottom: 1px solid var(--vp-c-divider);
  text-decoration: none;
  transition: background 160ms ease;
}

.area-item:hover {
  background: var(--vp-c-bg-soft);
}

.area-topline {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 2rem;
}

.area-number,
.area-count,
.area-en {
  font-family: var(--vp-font-family-mono);
  font-size: 0.64rem;
}

.area-number {
  color: var(--vp-c-brand-1);
}

.area-count,
.area-en {
  color: var(--vp-c-text-3);
}

.area-item h3 {
  margin: 0 0 0.6rem;
  color: var(--vp-c-text-1);
  font-size: 0.92rem;
  font-weight: 620;
  line-height: 1.5;
}

.area-item p {
  margin: 0 0 1.2rem;
  color: var(--vp-c-text-2);
  font-size: 0.72rem;
  line-height: 1.7;
}

.area-en {
  position: absolute;
  bottom: 1.1rem;
  left: 1.25rem;
  letter-spacing: 0.04em;
}

.recent-list {
  border-top: 1px solid var(--vp-c-divider);
}

.recent-item {
  display: grid;
  grid-template-columns: 150px minmax(0, 1fr) 92px 20px;
  gap: 1.5rem;
  align-items: center;
  min-height: 68px;
  padding: 0.85rem 0;
  border-bottom: 1px solid var(--vp-c-divider);
  text-decoration: none;
}

.recent-domain {
  overflow: hidden;
  color: var(--vp-c-text-3);
  font-size: 0.7rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.recent-title {
  overflow: hidden;
  color: var(--vp-c-text-1);
  font-size: 0.86rem;
  font-weight: 520;
  text-overflow: ellipsis;
  white-space: nowrap;
  transition: color 160ms ease;
}

.recent-item:hover .recent-title {
  color: var(--vp-c-brand-1);
}

.recent-item time {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.65rem;
}

.recent-arrow {
  color: var(--vp-c-text-3);
  font-size: 0.75rem;
  opacity: 0;
  transition: opacity 160ms ease;
}

.recent-item:hover .recent-arrow {
  opacity: 1;
}

.activity-section {
  margin-top: 4rem;
}

.workspace-footer {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 2rem;
  margin-top: 5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-3);
  font-size: 0.7rem;
}

.workspace-footer > div {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.7rem 1.3rem;
}

.workspace-footer a {
  color: var(--vp-c-text-2);
  text-decoration: none;
}

.workspace-footer a:hover {
  color: var(--vp-c-brand-1);
}

@media (max-width: 1024px) {
  .workspace {
    padding-top: 3rem;
  }

  .area-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .quick-nav {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .quick-nav-label {
    display: none;
  }

  .quick-link:first-of-type {
    border-left: 0;
  }
}

@media (max-width: 720px) {
  .workspace {
    padding: 2.5rem 1.25rem;
  }

  .workspace-hero {
    grid-template-columns: 1fr;
    gap: 2.5rem;
    padding-bottom: 2.5rem;
  }

  .hero-copy h1 {
    font-size: clamp(3.2rem, 16vw, 5rem);
  }

  .hero-intro {
    font-size: 0.9rem;
  }

  .library-summary {
    min-width: 0;
    padding: 1.5rem 0 0;
    border-top: 1px solid var(--vp-c-divider);
    border-left: 0;
  }

  .quick-nav {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .quick-link {
    min-height: 76px;
    padding: 0.9rem;
    border-bottom: 1px solid var(--vp-c-divider);
  }

  .quick-link:nth-of-type(odd) {
    border-left: 0;
  }

  .quick-link:nth-last-child(-n + 2) {
    border-bottom: 0;
  }

  .workspace-section {
    margin-top: 3.5rem;
  }

  .section-note {
    display: none;
  }

  .area-item {
    min-height: 176px;
    padding: 1rem;
  }

  .area-en {
    left: 1rem;
  }

  .recent-item {
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.45rem 1rem;
    padding: 1rem 0;
  }

  .recent-domain {
    grid-column: 1;
    grid-row: 1;
  }

  .recent-title {
    grid-column: 1 / -1;
    grid-row: 2;
    white-space: normal;
    line-height: 1.6;
  }

  .recent-item time {
    grid-column: 2;
    grid-row: 1;
  }

  .recent-arrow {
    display: none;
  }

  .workspace-footer {
    flex-direction: column;
  }

  .workspace-footer > div {
    justify-content: flex-start;
  }
}

@media (max-width: 480px) {
  .area-grid {
    grid-template-columns: 1fr;
  }

  .area-item {
    min-height: 160px;
  }

  .hero-actions {
    align-items: flex-start;
    flex-direction: column;
    gap: 1rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .primary-action,
  .quick-link,
  .area-item,
  .recent-title,
  .recent-arrow {
    transition: none;
  }
}
</style>
