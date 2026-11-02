<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'
import { TYPE_LABEL, STATUS_LABEL } from '../lib/tags.mjs'

// 笔记页顶部的分类信息（类型、状态、标签），来自 frontmatter。没有 type 的页面（索引页等）不显示。
const { frontmatter } = useData()
const meta = computed(() => frontmatter.value)
</script>

<template>
  <div v-if="meta.type" class="note-meta">
    <span class="note-meta-chip">{{ TYPE_LABEL[meta.type] ?? meta.type }}</span>
    <span v-if="meta.status" class="note-meta-chip">{{ STATUS_LABEL[meta.status] }}</span>
    <span v-for="tag in meta.tags" :key="tag" class="note-meta-chip note-meta-tag">#{{ tag }}</span>
  </div>
</template>

<style scoped>
.note-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  margin: 0 0 20px;
}
.note-meta-chip {
  padding: 0;
  border-radius: 0;
  background: transparent;
  color: var(--vp-c-text-3);
  font-size: 11px;
  font-weight: 550;
  letter-spacing: 0.03em;
}
.note-meta-tag {
  color: var(--vp-c-brand-1);
}
</style>
