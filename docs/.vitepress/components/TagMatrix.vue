<template>
  <div class="tag-matrix-container">
    <div class="matrix-header">
      <p class="matrix-desc">
        打破传统文件夹目录的物理隔离，通过<strong>多维技术标签</strong>穿透视频、视觉、LLM、通用基石与算力基建，发掘跨模态的技术交点与研究启发。
      </p>
    </div>

    <!-- 标签过滤器面板 -->
    <div class="tags-filter-panel">
      <div class="panel-top">
        <span class="filter-label">🏷️ 标签快速穿透 (共 {{ allTags.length }} 个标签, {{ notesList.length }} 篇笔记)</span>
        <button v-if="selectedTag" class="clear-btn" @click="selectedTag = ''">
          清除筛选 (查看全部)
        </button>
      </div>

      <div class="tags-cloud">
        <button 
          class="tag-pill" 
          :class="{ active: selectedTag === '' }"
          @click="selectedTag = ''"
        >
          全部 ({{ notesList.length }})
        </button>
        <button 
          v-for="tag in allTags" 
          :key="tag.name"
          class="tag-pill"
          :class="{ active: selectedTag === tag.name }"
          @click="selectedTag = tag.name"
        >
          #{{ tag.name }} <span class="tag-count">{{ tag.count }}</span>
        </button>
      </div>
    </div>

    <!-- 筛选结果列表 -->
    <div class="results-container">
      <div class="results-header">
        <h4>
          {{ selectedTag ? `包含标签 #${selectedTag} 的笔记` : '全部穿透检索结果' }}
          <span class="results-count">({{ filteredNotes.length }} 篇)</span>
        </h4>
      </div>

      <div class="notes-grid">
        <div v-for="note in filteredNotes" :key="note.link" class="note-card">
          <div class="note-card-top">
            <span class="domain-badge" :class="note.domainClass">{{ note.domain }}</span>
            <span class="note-date">{{ note.date }}</span>
          </div>

          <h3 class="note-title">
            <a :href="note.link">{{ note.title }}</a>
          </h3>

          <p class="note-summary">{{ note.summary }}</p>

          <div class="note-tags">
            <span 
              v-for="t in note.tags" 
              :key="t" 
              class="mini-tag"
              :class="{ highlighted: selectedTag === t }"
              @click.stop="selectedTag = t"
            >
              #{{ t }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const selectedTag = ref('')

// 知识库全局全模态笔记池
const notesList = ref([
  {
    title: '时空注意力机制 (Spatio-Temporal Attention) 数学推导',
    link: '/domains/video/spatio-temporal-attention',
    domain: '🎬 视频模型',
    domainClass: 'domain-video',
    date: '2026-09-19',
    summary: '分析全时空联合注意力 O(N^2) 显存爆炸原因，推导空间-时间因子分解 (Factorized Attention) 降维证明。',
    tags: ['Video Gen', 'Attention', 'Complexity', 'Math Derivation']
  },
  {
    title: 'Video DiT 架构演进与时空物理一致性',
    link: '/domains/video/video-diffusion-dit',
    domain: '🎬 视频模型',
    domainClass: 'domain-video',
    date: '2026-09-19',
    summary: '从 2D UNet 级联到 3D Causal VAE 与 DiT 主干的演变，解析连续潜空间压缩与物理一致性建模。',
    tags: ['Video Gen', 'DiT', 'Diffusion', '3D VAE']
  },
  {
    title: 'ControlNet 空间结构可控生成原理解析',
    link: '/domains/vision/diffusion-controlnet',
    domain: '🖼️ 图像视觉',
    domainClass: 'domain-vision',
    date: '2026-09-19',
    summary: '锁定主干与零卷积 (Zero Convolution) 结构，从数学上推导权重初始为 0 时前向恒等性与梯度传递有效性。',
    tags: ['Vision', 'Diffusion', 'ControlNet', 'Zero-Conv']
  },
  {
    title: 'DeepSeek-R1 推理涌现与强化学习机制',
    link: '/domains/llm/deepseek-r1-reasoning',
    domain: '💬 语言推理',
    domainClass: 'domain-llm',
    date: '2026-09-19',
    summary: '纯 RL 激发慢思考 System 2 与 Aha 顿悟，推导 GRPO 组相对策略优化数学目标及无 Critic 显存减半优势。',
    tags: ['LLM', 'Reasoning', 'RL', 'GRPO']
  },
  {
    title: 'CLIP 跨模态对比表征学习数学解析',
    link: '/domains/multimodal/clip-alignment',
    domain: '🎙️ 多模态',
    domainClass: 'domain-multi',
    date: '2026-09-19',
    summary: '双塔视觉与语言对齐机制，严格推导对称式交叉熵 (InfoNCE) 损失函数及超球面余弦相似度几何意义。',
    tags: ['Multimodal', 'CLIP', 'Contrastive Learning', 'InfoNCE']
  },
  {
    title: 'Transformer 缩放点积注意力数学推导',
    link: '/foundations/transformer-attention',
    domain: '📐 理论基石',
    domainClass: 'domain-foundations',
    date: '2026-09-19',
    summary: '从随机变量独立同分布统计期望与方差出发，严格证明为什么必须除以 sqrt(d_k) 以抵御 Softmax 梯度饱和。',
    tags: ['Attention', 'Transformer', 'Math Derivation']
  },
  {
    title: 'Flow Matching 与连续流生成动力学推导',
    link: '/foundations/flow-matching-derivation',
    domain: '📐 理论基石',
    domainClass: 'domain-foundations',
    date: '2026-09-19',
    summary: '从传统弯曲扩散轨迹到直线最优传输 (Optimal Transport)，推导条件流匹配 (CFM) 速度场回归损失。',
    tags: ['Generative Models', 'Flow Matching', 'Diffusion', 'Math Derivation']
  },
  {
    title: 'CUDA 显存层级与 Roofline 性能模型',
    link: '/infra/cuda-memory-hierarchy',
    domain: '⚡ 算力工程',
    domainClass: 'domain-infra',
    date: '2026-09-19',
    summary: 'Registers, SRAM 到 HBM 带宽延迟金字塔，利用算术强度判定 Memory-Bound 与 Compute-Bound 并指导算子融合。',
    tags: ['CUDA', 'Infra', 'Roofline', 'Memory Bound']
  },
  {
    title: 'vLLM PagedAttention 与高并发吞吐优化',
    link: '/infra/vllm-paged-attention',
    domain: '⚡ 算力工程',
    domainClass: 'domain-infra',
    date: '2026-09-19',
    summary: '借鉴 OS 虚拟内存分页解决自回归推理中 60%~80% 的 KV Cache 显存碎片，实现显存零浪费与多采样共享。',
    tags: ['LLM', 'Infra', 'vLLM', 'PagedAttention']
  },
  {
    title: '开源视频生成模型本地复现与评测报告',
    link: '/projects/video-gen-reproduction',
    domain: '🚀 实战复现',
    domainClass: 'domain-projects',
    date: '2026-09-19',
    summary: 'Wan2.1 14B 模型在 8 卡 H800 上的序列并行 (SP) 与算子优化评测，生成耗时缩减 35.3%。',
    tags: ['Video Gen', 'Benchmark', 'DiT', 'Infra']
  },
  {
    title: 'LLM 7B 模型 LoRA 微调与吞吐评测',
    link: '/projects/llm-finetuning-benchmark',
    domain: '🚀 实战复现',
    domainClass: 'domain-projects',
    date: '2026-09-19',
    summary: '单卡 A100 下对比 Full SFT 与 LoRA 的可训练参数、显存峰值（节约 63.7%）与吞吐量提升实测。',
    tags: ['LLM', 'LoRA', 'Benchmark']
  },
  {
    title: '2026 年第 38 周研习周报：视频时空生成与算力工程闭环',
    link: '/reviews/2026-W38',
    domain: '📅 研习复盘',
    domainClass: 'domain-multi',
    date: '2026-09-19',
    summary: '高维学术复盘：总结时空因子分解注意力降维证明、Wan2.1 8卡序列并行优化、及纯 RL 视频平滑度假说。',
    tags: ['Video Gen', 'Attention', 'CUDA', 'Flow Matching', 'GRPO', 'Weekly Review']
  }
])

// 提取所有标签与频次
const allTags = computed(() => {
  const map = {}
  notesList.value.forEach(item => {
    item.tags.forEach(t => {
      map[t] = (map[t] || 0) + 1
    })
  })
  return Object.keys(map)
    .sort((a, b) => map[b] - map[a])
    .map(key => ({ name: key, count: map[key] }))
})

// 筛选笔记
const filteredNotes = computed(() => {
  if (!selectedTag.value) return notesList.value
  return notesList.value.filter(item => item.tags.includes(selectedTag.value))
})
</script>

<style scoped>
.tag-matrix-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 1rem 0 3rem 0;
}

.matrix-header {
  margin-bottom: 1.5rem;
}

.matrix-desc {
  font-size: 0.95rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

/* 标签云面板 */
.tags-filter-panel {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 2rem;
}

.panel-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

.filter-label {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.clear-btn {
  font-size: 0.8rem;
  color: var(--vp-c-brand-1);
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}

.clear-btn:hover {
  text-decoration: underline;
}

.tags-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.tag-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.85rem;
  padding: 0.3rem 0.7rem;
  border-radius: 999px;
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-2);
  border: 1px solid var(--vp-c-divider);
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-pill:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  transform: translateY(-1px);
}

.tag-pill.active {
  background: var(--vp-c-brand-1);
  color: #ffffff;
  border-color: var(--vp-c-brand-1);
  font-weight: 600;
}

.tag-pill.active .tag-count {
  background: rgba(255, 255, 255, 0.25);
  color: #ffffff;
}

.tag-count {
  font-size: 0.72rem;
  padding: 0.1rem 0.35rem;
  border-radius: 999px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
}

/* 结果区 */
.results-header {
  margin-bottom: 1rem;
}

.results-header h4 {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
  color: var(--vp-c-text-1);
}

.results-count {
  font-size: 0.85rem;
  color: var(--vp-c-text-3);
  font-weight: normal;
}

.notes-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.1rem;
}

@media (max-width: 768px) {
  .notes-grid {
    grid-template-columns: 1fr;
  }
}

.note-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.2rem;
  display: flex;
  flex-direction: column;
  transition: all 0.2s ease;
}

.note-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
}

.note-card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.domain-badge {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 6px;
}

.domain-video { background: rgba(255, 112, 67, 0.15); color: #ff7043; }
.domain-vision { background: rgba(38, 166, 154, 0.15); color: #26a69a; }
.domain-llm { background: rgba(142, 68, 173, 0.15); color: #9c27b0; }
.domain-multi { background: rgba(66, 165, 245, 0.15); color: #1e88e5; }
.domain-foundations { background: rgba(255, 179, 0, 0.18); color: #f57f17; }
.domain-infra { background: rgba(76, 175, 80, 0.15); color: #43a047; }
.domain-projects { background: rgba(100, 116, 139, 0.15); color: #64748b; }

.note-date {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.note-title {
  font-size: 1rem;
  font-weight: 600;
  margin: 0.2rem 0 0.5rem 0;
  line-height: 1.4;
}

.note-title a {
  color: var(--vp-c-text-1);
  text-decoration: none;
}

.note-title a:hover {
  color: var(--vp-c-brand-1);
}

.note-summary {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  line-height: 1.5;
  margin: 0 0 0.8rem 0;
  flex-grow: 1;
}

.note-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.mini-tag {
  font-size: 0.72rem;
  padding: 0.12rem 0.4rem;
  border-radius: 4px;
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-3);
  cursor: pointer;
  transition: all 0.15s ease;
}

.mini-tag:hover {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.mini-tag.highlighted {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}
</style>
