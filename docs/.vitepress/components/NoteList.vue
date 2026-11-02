<script setup>
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as notes } from '../data/notes.data.ts'

// 只列出当前目录的直接子笔记；子专题仍由各自索引页维护。
const props = defineProps({
  folder: { type: String, required: true },
  groups: { type: Array, default: () => [] }
})

const items = computed(() => notes.filter(note => note.dir === props.folder))
const sections = computed(() => {
  if (!props.groups.length) return [{ name: '', items: items.value }]
  return props.groups.map(name => ({
    name,
    items: items.value.filter(note => note.group === name)
  }))
})
</script>

<template>
  <div class="note-list">
    <p v-if="!items.length" class="note-list-empty">暂无笔记。</p>

    <section v-for="section in sections" :key="section.name || 'all'" class="note-group">
      <h3 v-if="section.name" class="note-group-title">{{ section.name }}</h3>
      <div v-if="section.items.length" class="note-rows">
        <a v-for="(note, index) in section.items" :key="note.url" :href="withBase(note.url)" class="note-row">
          <span class="note-index" aria-hidden="true">{{ String(index + 1).padStart(2, '0') }}</span>
          <span class="note-content">
            <span class="note-title">{{ note.title }}</span>
            <span v-if="note.models.length" class="note-models">{{ note.models.join(' · ') }}</span>
            <span class="note-summary">{{ note.summary }}</span>
          </span>
          <span class="note-aside">
            <span v-if="note.statusLabel" class="note-status" :data-status="note.status">{{ note.statusLabel }}</span>
            <time :datetime="note.date">{{ note.date }}</time>
          </span>
          <span class="note-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
    </section>
  </div>
</template>

<style scoped>
.note-list {
  margin: 1.5rem 0 2rem;
}

.note-group + .note-group {
  margin-top: 3rem;
}

.note-list :deep(.note-group-title) {
  margin: 0 0 1rem;
  padding: 0;
  border: 0;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}

.note-rows {
  border-top: 1px solid var(--vp-c-divider);
}

.note-row {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) 92px 16px;
  gap: 1rem;
  align-items: start;
  padding: 1.25rem 0;
  border-bottom: 1px solid var(--vp-c-divider);
  text-decoration: none;
}

.note-index {
  padding-top: 0.2rem;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.64rem;
}

.note-content {
  display: flex;
  min-width: 0;
  flex-direction: column;
}

.note-title {
  color: var(--vp-c-text-1);
  font-size: 0.93rem;
  font-weight: 600;
  line-height: 1.55;
  transition: color 160ms ease;
}

.note-row:hover .note-title {
  color: var(--vp-c-brand-1);
}

.note-models {
  margin-top: 0.35rem;
  color: var(--vp-c-text-3);
  font-size: 0.67rem;
  line-height: 1.6;
}

.note-summary {
  display: -webkit-box;
  margin-top: 0.55rem;
  overflow: hidden;
  color: var(--vp-c-text-2);
  font-size: 0.76rem;
  font-weight: 400;
  line-height: 1.75;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.note-aside {
  display: flex;
  align-items: flex-end;
  flex-direction: column;
  gap: 0.45rem;
  padding-top: 0.15rem;
  text-align: right;
}

.note-status,
.note-aside time {
  color: var(--vp-c-text-3);
  font-size: 0.64rem;
  white-space: nowrap;
}

.note-status[data-status='reading'],
.note-status[data-status='ongoing'] {
  color: var(--vp-c-brand-1);
}

.note-aside time {
  font-family: var(--vp-font-family-mono);
}

.note-arrow {
  padding-top: 0.2rem;
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
  opacity: 0;
  transition: opacity 160ms ease;
}

.note-row:hover .note-arrow {
  opacity: 1;
}

.note-list-empty {
  color: var(--vp-c-text-3);
  font-size: 0.85rem;
}

@media (max-width: 640px) {
  .note-row {
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.5rem 1rem;
    padding: 1.1rem 0;
  }

  .note-index,
  .note-arrow {
    display: none;
  }

  .note-content {
    grid-column: 1 / -1;
  }

  .note-aside {
    grid-column: 1 / -1;
    grid-row: 2;
    align-items: center;
    flex-direction: row;
    padding-top: 0;
    text-align: left;
  }

  .note-summary {
    -webkit-line-clamp: 3;
  }
}

@media (prefers-reduced-motion: reduce) {
  .note-title,
  .note-arrow {
    transition: none;
  }
}
</style>
