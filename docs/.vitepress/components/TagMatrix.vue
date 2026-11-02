<script setup>
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { data as notes } from '../data/notes.data.ts'

const selectedTag = ref('')

const allTags = computed(() => {
  const counts = {}
  notes.forEach(note => {
    note.tags.forEach(tag => { counts[tag] = (counts[tag] || 0) + 1 })
  })
  return Object.keys(counts)
    .sort((a, b) => counts[b] - counts[a] || a.localeCompare(b))
    .map(name => ({ name, count: counts[name] }))
})

const filteredNotes = computed(() => selectedTag.value
  ? notes.filter(note => note.tags.includes(selectedTag.value))
  : notes
)
</script>

<template>
  <div class="tag-browser">
    <p class="tag-intro">按技术标签跨目录检索，找到不同研究领域之间的关联。</p>

    <section class="tag-filters" aria-label="标签筛选">
      <div class="filter-heading">
        <span>筛选标签</span>
        <span>{{ allTags.length }} 个标签 · {{ notes.length }} 篇笔记</span>
      </div>
      <div class="tags-cloud">
        <button class="tag-pill" :class="{ active: selectedTag === '' }" :aria-pressed="selectedTag === ''" @click="selectedTag = ''">
          全部 <span>{{ notes.length }}</span>
        </button>
        <button
          v-for="tag in allTags"
          :key="tag.name"
          class="tag-pill"
          :class="{ active: selectedTag === tag.name }"
          :aria-pressed="selectedTag === tag.name"
          @click="selectedTag = tag.name"
        >
          {{ tag.name }} <span>{{ tag.count }}</span>
        </button>
      </div>
    </section>

    <section class="tag-results" aria-live="polite">
      <div class="results-heading">
        <h2>{{ selectedTag ? selectedTag : '全部笔记' }}</h2>
        <span>{{ filteredNotes.length }} 篇</span>
      </div>

      <div class="result-list">
        <article v-for="note in filteredNotes" :key="note.url" class="result-item">
          <div class="result-meta">
            <span>{{ note.domain }}</span>
            <time :datetime="note.date">{{ note.date }}</time>
          </div>
          <h3><a :href="withBase(note.url)">{{ note.title }}</a></h3>
          <p>{{ note.summary }}</p>
          <div class="result-tags">
            <button
              v-for="tag in note.tags"
              :key="tag"
              :class="{ active: selectedTag === tag }"
              :aria-label="`筛选标签 ${tag}`"
              @click="selectedTag = tag"
            >
              #{{ tag }}
            </button>
          </div>
        </article>
      </div>
    </section>
  </div>
</template>

<style scoped>
.tag-browser {
  max-width: 960px;
  margin: 0 auto;
  padding: 0.5rem 0 3rem;
}

.tag-intro {
  max-width: 650px;
  margin: 0 0 2.5rem;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
  line-height: 1.8;
}

.tag-filters {
  padding: 1.3rem 0 1.5rem;
  border-top: 1px solid var(--vp-c-divider);
  border-bottom: 1px solid var(--vp-c-divider);
}

.filter-heading,
.results-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.filter-heading {
  margin-bottom: 1rem;
}

.filter-heading > span:first-child {
  color: var(--vp-c-text-1);
  font-size: 0.78rem;
  font-weight: 600;
}

.filter-heading > span:last-child,
.results-heading > span {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.65rem;
}

.tags-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.tag-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.34rem 0.6rem;
  border: 1px solid transparent;
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 0.7rem;
  cursor: pointer;
  transition: background 160ms ease, color 160ms ease;
}

.tag-pill span {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.6rem;
}

.tag-pill:hover {
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-1);
}

.tag-pill.active {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
  color: #fff;
}

.tag-pill.active span {
  color: inherit;
  opacity: 0.75;
}

.tag-results {
  margin-top: 3rem;
}

.results-heading {
  margin-bottom: 1.2rem;
}

.results-heading h2 {
  margin: 0;
  padding: 0;
  border: 0;
  font-size: 1.2rem;
  font-weight: 650;
  letter-spacing: -0.02em;
}

.result-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0 2.5rem;
  border-top: 1px solid var(--vp-c-divider);
}

.result-item {
  min-width: 0;
  padding: 1.4rem 0 1.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
}

.result-meta {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  color: var(--vp-c-text-3);
  font-size: 0.65rem;
}

.result-meta time {
  flex-shrink: 0;
  font-family: var(--vp-font-family-mono);
}

.result-item h3 {
  margin: 0.75rem 0 0.55rem;
  font-size: 0.94rem;
  font-weight: 620;
  line-height: 1.55;
}

.result-item h3 a {
  color: var(--vp-c-text-1);
  text-decoration: none;
}

.result-item h3 a:hover {
  color: var(--vp-c-brand-1);
}

.result-item > p {
  display: -webkit-box;
  margin: 0;
  overflow: hidden;
  color: var(--vp-c-text-2);
  font-size: 0.75rem;
  line-height: 1.75;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.result-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.75rem;
  margin-top: 0.9rem;
}

.result-tags button {
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--vp-c-text-3);
  font-size: 0.65rem;
  cursor: pointer;
}

.result-tags button:hover,
.result-tags button.active {
  color: var(--vp-c-brand-1);
}

@media (max-width: 720px) {
  .result-list {
    grid-template-columns: 1fr;
  }

  .tag-results {
    margin-top: 2rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .tag-pill {
    transition: none;
  }
}
</style>
