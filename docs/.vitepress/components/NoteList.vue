<script setup>
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as notes } from '../data/notes.data.ts'

// 列出某个目录下的笔记（只含直接子文件；子专题由各自的索引页列出）。
// groups 非空时按笔记的 group 字段分组，并按 groups 的顺序显示。
const props = defineProps({
  folder: { type: String, required: true },
  groups: { type: Array, default: () => [] }
})

const items = computed(() => notes.filter(n => n.dir === props.folder))

const sections = computed(() => {
  if (!props.groups.length) return [{ name: '', items: items.value }]
  return props.groups.map(name => ({
    name,
    items: items.value.filter(n => n.group === name)
  }))
})
</script>

<template>
  <div class="note-list">
    <p v-if="!items.length" class="note-list-empty">暂无笔记。</p>
    <template v-for="section in sections" :key="section.name">
      <h3 v-if="section.name">{{ section.name }}</h3>
      <table v-if="section.items.length">
        <thead>
          <tr>
            <th>篇目</th>
            <th>主题 / 核心机制</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="n in section.items" :key="n.url">
            <td>
              <a :href="withBase(n.url)"><strong>{{ n.title }}</strong></a>
              <div v-if="n.models.length" class="note-list-models">{{ n.models.join(' · ') }}</div>
            </td>
            <td class="note-list-summary" :title="n.summary">{{ n.summary }}</td>
            <td class="note-list-status">{{ n.statusLabel }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </div>
</template>

<style scoped>
.note-list {
  margin-top: 1.25rem;
}

.note-list :deep(h3) {
  margin-top: 2.4rem;
  font-size: 1rem;
}

.note-list :deep(table) {
  margin-top: 1rem;
}

.note-list :deep(th:first-child) {
  width: 34%;
}

.note-list :deep(th:last-child) {
  width: 76px;
}

.note-list :deep(td) {
  vertical-align: top;
}

.note-list :deep(a) {
  color: var(--vp-c-text-1);
  font-weight: 550;
}

.note-list :deep(a:hover) {
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.note-list-summary {
  display: -webkit-box;
  overflow: hidden;
  color: var(--vp-c-text-2);
  font-size: 0.82rem;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

.note-list-models {
  margin-top: 6px;
  color: var(--vp-c-text-3);
  font-size: 0.68rem;
}

.note-list-status {
  color: var(--vp-c-text-3);
  font-size: 0.72rem;
  white-space: nowrap;
}

.note-list-empty {
  color: var(--vp-c-text-3);
}
</style>
