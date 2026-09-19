<template>
  <div class="repo-heatmap-card" :class="`theme-${currentTheme}`">
    <div class="heatmap-header">
      <div class="header-left">
        <div class="title-row">
          <span class="pulse-dot"></span>
          <span class="heatmap-title">AlgoNotes 本地知识库迭代活跃度</span>
        </div>
        <span class="heatmap-subtitle">仅统计当前算法研习库提交，记录真实思考与代码推进历程</span>
      </div>

      <!-- 配色切换器 (告别千篇一律的 GitHub 绿色) -->
      <div class="theme-picker">
        <span class="picker-label">主题配色:</span>
        <button 
          v-for="t in themeOptions" 
          :key="t.key" 
          class="theme-btn" 
          :class="{ active: currentTheme === t.key }"
          :title="t.name"
          @click="currentTheme = t.key"
        >
          <span class="color-dot" :style="{ background: t.color }"></span>
          <span class="theme-name">{{ t.name }}</span>
        </button>
      </div>
    </div>

    <!-- 研习投入度核心指标 -->
    <div class="stats-row">
      <div class="stat-item">
        <span class="stat-num">{{ repoCommitData.totalCommits }}</span>
        <span class="stat-desc">知识迭代总次数</span>
      </div>
      <div class="stat-divider"></div>
      <div class="stat-item">
        <span class="stat-num">{{ repoCommitData.activeDays }}</span>
        <span class="stat-desc">深度研习天数</span>
      </div>
      <div class="stat-divider"></div>
      <div class="stat-item flex-grow">
        <span class="stat-num date-text">{{ repoCommitData.latestDate || '今日' }}</span>
        <span class="stat-desc truncate-text" :title="repoCommitData.latestMessage">
          最近推进: {{ repoCommitData.latestMessage || '系统构建' }}
        </span>
      </div>
    </div>

    <!-- 热力图网格主视窗 -->
    <div class="grid-scroll-box">
      <div class="heatmap-chart">
        <!-- 月份标尺 -->
        <div class="months-row">
          <span 
            v-for="(m, idx) in monthLabels" 
            :key="idx" 
            class="month-label"
            :style="{ gridColumnStart: m.col }"
          >
            {{ m.text }}
          </span>
        </div>

        <!-- 星期与格子阵列 -->
        <div class="days-container">
          <div class="weekday-labels">
            <span>一</span>
            <span>三</span>
            <span>五</span>
            <span>日</span>
          </div>

          <div class="weeks-grid">
            <div 
              v-for="(week, wIdx) in calendarWeeks" 
              :key="wIdx" 
              class="week-col"
            >
              <div 
                v-for="(day, dIdx) in week" 
                :key="dIdx" 
                class="day-cell"
                :class="[
                  `level-${day.level}`,
                  { 'is-today': day.isToday, 'is-future': day.isFuture }
                ]"
                @mouseenter="hoveredDay = day"
                @mouseleave="hoveredDay = null"
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 悬停详情气泡与图例 -->
    <div class="heatmap-footer">
      <div class="hover-info">
        <span v-if="hoveredDay && !hoveredDay.isFuture">
          📅 <strong>{{ hoveredDay.date }}</strong>：
          <span v-if="hoveredDay.count > 0" class="active-text">
            贡献了 <strong>{{ hoveredDay.count }}</strong> 次知识推导与代码迭代
          </span>
          <span v-else class="empty-text">当日无提交 (思考沉淀)</span>
        </span>
        <span v-else class="tip-placeholder">鼠标悬停在色块上方查看具体日期的研习记录</span>
      </div>

      <div class="legend-box">
        <span class="legend-text">轻度</span>
        <span class="legend-cell level-0"></span>
        <span class="legend-cell level-1"></span>
        <span class="legend-cell level-2"></span>
        <span class="legend-cell level-3"></span>
        <span class="legend-cell level-4"></span>
        <span class="legend-text">高能</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { data as repoCommitData } from '../data/commits.data.ts'

// 配色方案（支持极光紫绀、深海蔚蓝、熔岩暖金三种学术高质感配色）
const currentTheme = ref('indigo')

const themeOptions = [
  { key: 'indigo', name: '极光紫绀', color: '#6366f1' },
  { key: 'cyan', name: '深海蔚蓝', color: '#0284c7' },
  { key: 'amber', name: '熔岩暖金', color: '#f59e0b' }
]

const hoveredDay = ref(null)

// 生成过去 52 周（364 天）的日历网格
const calendarData = computed(() => {
  const weeks = []
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  // 找到 52 周前的周一
  const startDay = new Date(today)
  const currentDayOfWeek = (startDay.getDay() + 6) % 7 // 0=Mon, 6=Sun
  startDay.setDate(startDay.getDate() - (51 * 7 + currentDayOfWeek))

  const cursor = new Date(startDay)
  let currentWeek = []

  for (let i = 0; i < 52 * 7; i++) {
    const y = cursor.getFullYear()
    const m = String(cursor.getMonth() + 1).padStart(2, '0')
    const d = String(cursor.getDate()).padStart(2, '0')
    const dateStr = `${y}-${m}-${d}`

    const count = repoCommitData.commitCounts[dateStr] || 0
    let level = 0
    if (count >= 6) level = 4
    else if (count >= 3) level = 3
    else if (count >= 2) level = 2
    else if (count >= 1) level = 1

    const isToday = cursor.getTime() === today.getTime()
    const isFuture = cursor.getTime() > today.getTime()

    currentWeek.push({
      date: dateStr,
      count,
      level,
      isToday,
      isFuture,
      month: cursor.getMonth() + 1,
      dayNum: cursor.getDate()
    })

    if (currentWeek.length === 7) {
      weeks.push(currentWeek)
      currentWeek = []
    }

    cursor.setDate(cursor.getDate() + 1)
  }

  return weeks
})

const calendarWeeks = computed(() => calendarData.value)

// 提取月份标签与其所在的列数
const monthLabels = computed(() => {
  const labels = []
  let lastMonth = -1
  const monthNames = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

  calendarData.value.forEach((week, colIndex) => {
    const firstDay = week[0]
    if (firstDay && firstDay.month !== lastMonth) {
      labels.push({
        text: monthNames[firstDay.month - 1],
        col: colIndex + 1
      })
      lastMonth = firstDay.month
    }
  })

  return labels
})
</script>

<style scoped>
.repo-heatmap-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  padding: 1.4rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
  transition: all 0.3s ease;
}

/* 头部 */
.heatmap-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1.2rem;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #6366f1;
  box-shadow: 0 0 8px #6366f1;
  animation: pulse-glow 2s infinite ease-in-out;
}

.theme-cyan .pulse-dot {
  background: #0284c7;
  box-shadow: 0 0 8px #0284c7;
}

.theme-amber .pulse-dot {
  background: #f59e0b;
  box-shadow: 0 0 8px #f59e0b;
}

@keyframes pulse-glow {
  0%, 100% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.3); opacity: 1; }
}

.heatmap-title {
  font-size: 1.08rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.heatmap-subtitle {
  font-size: 0.82rem;
  color: var(--vp-c-text-3);
  display: block;
  margin-top: 0.25rem;
}

/* 配色选择器 */
.theme-picker {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  background: var(--vp-c-bg-mute);
  padding: 0.25rem 0.5rem;
  border-radius: 999px;
  border: 1px solid var(--vp-c-divider);
}

.picker-label {
  font-size: 0.72rem;
  color: var(--vp-c-text-3);
  margin-right: 0.2rem;
}

.theme-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.75rem;
  padding: 0.15rem 0.45rem;
  border-radius: 999px;
  background: transparent;
  border: 1px solid transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}

.theme-btn:hover {
  color: var(--vp-c-text-1);
}

.theme-btn.active {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-divider);
  color: var(--vp-c-text-1);
  font-weight: 600;
}

.color-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

/* 核心投入指标行 */
.stats-row {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  padding: 0.8rem 1rem;
  background: var(--vp-c-bg-mute);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  margin-bottom: 1.4rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
}

.stat-item.flex-grow {
  flex-grow: 1;
}

.stat-num {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}

.stat-num.date-text {
  font-size: 0.95rem;
}

.stat-desc {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  margin-top: 0.1rem;
}

.truncate-text {
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.stat-divider {
  width: 1px;
  height: 28px;
  background: var(--vp-c-divider);
}

/* 图表滚动容器 */
.grid-scroll-box {
  overflow-x: auto;
  padding-bottom: 0.5rem;
}

.heatmap-chart {
  min-width: 720px;
}

.months-row {
  display: grid;
  grid-template-columns: repeat(52, 11px);
  gap: 3px;
  margin-left: 24px;
  margin-bottom: 4px;
  height: 14px;
}

.month-label {
  font-size: 0.68rem;
  color: var(--vp-c-text-3);
  white-space: nowrap;
}

.days-container {
  display: flex;
  gap: 6px;
}

.weekday-labels {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 2px 0;
  font-size: 0.65rem;
  color: var(--vp-c-text-3);
  width: 18px;
  text-align: right;
  line-height: 1;
}

.weeks-grid {
  display: flex;
  gap: 3px;
}

.week-col {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.day-cell {
  width: 11px;
  height: 11px;
  border-radius: 2px;
  transition: transform 0.15s ease;
  cursor: pointer;
}

.day-cell:hover {
  transform: scale(1.35);
  z-index: 10;
  outline: 1.5px solid var(--vp-c-text-1);
}

.day-cell.is-future {
  opacity: 0.2;
  cursor: default;
}

.day-cell.is-today {
  outline: 1.5px solid var(--vp-c-brand-1);
  outline-offset: 1px;
}

/* ========================================================
   三套自研专属配色系统 (非 GitHub 绿)
   ======================================================== */

/* 1. 极光紫绀系 (Indigo / Violet) */
.theme-indigo .level-0 { background: #e2e8f0; }
.theme-indigo .level-1 { background: #c7d2fe; }
.theme-indigo .level-2 { background: #818cf8; }
.theme-indigo .level-3 { background: #6366f1; }
.theme-indigo .level-4 { background: #4338ca; box-shadow: 0 0 6px rgba(99, 102, 241, 0.4); }

:root.dark .theme-indigo .level-0 { background: #1e2230; }
:root.dark .theme-indigo .level-1 { background: #312e81; }
:root.dark .theme-indigo .level-2 { background: #4338ca; }
:root.dark .theme-indigo .level-3 { background: #6366f1; }
:root.dark .theme-indigo .level-4 { background: #818cf8; box-shadow: 0 0 6px rgba(129, 140, 248, 0.6); }

/* 2. 深海蔚蓝系 (Sapphire / Cyan) */
.theme-cyan .level-0 { background: #e2e8f0; }
.theme-cyan .level-1 { background: #bae6fd; }
.theme-cyan .level-2 { background: #38bdf8; }
.theme-cyan .level-3 { background: #0284c7; }
.theme-cyan .level-4 { background: #0369a1; box-shadow: 0 0 6px rgba(2, 132, 199, 0.4); }

:root.dark .theme-cyan .level-0 { background: #182230; }
:root.dark .theme-cyan .level-1 { background: #075985; }
:root.dark .theme-cyan .level-2 { background: #0284c7; }
:root.dark .theme-cyan .level-3 { background: #38bdf8; }
:root.dark .theme-cyan .level-4 { background: #7dd3fc; box-shadow: 0 0 6px rgba(56, 189, 248, 0.6); }

/* 3. 熔岩暖金系 (Lava Amber) */
.theme-amber .level-0 { background: #e2e8f0; }
.theme-amber .level-1 { background: #fde68a; }
.theme-amber .level-2 { background: #fbbf24; }
.theme-amber .level-3 { background: #f59e0b; }
.theme-amber .level-4 { background: #d97706; box-shadow: 0 0 6px rgba(245, 158, 11, 0.4); }

:root.dark .theme-amber .level-0 { background: #26201b; }
:root.dark .theme-amber .level-1 { background: #78350f; }
:root.dark .theme-amber .level-2 { background: #b45309; }
:root.dark .theme-amber .level-3 { background: #f59e0b; }
:root.dark .theme-amber .level-4 { background: #fbbf24; box-shadow: 0 0 6px rgba(251, 191, 36, 0.6); }

/* 底部图例与气泡 */
.heatmap-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 1rem;
  padding-top: 0.8rem;
  border-top: 1px dashed var(--vp-c-divider);
  flex-wrap: wrap;
  gap: 0.6rem;
}

.hover-info {
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
}

.active-text strong {
  color: var(--vp-c-brand-1);
}

.empty-text {
  color: var(--vp-c-text-3);
}

.tip-placeholder {
  font-size: 0.78rem;
  color: var(--vp-c-text-3);
}

.legend-box {
  display: flex;
  align-items: center;
  gap: 3px;
}

.legend-text {
  font-size: 0.72rem;
  color: var(--vp-c-text-3);
  margin: 0 4px;
}

.legend-cell {
  width: 10px;
  height: 10px;
  border-radius: 2px;
}
</style>
