<script setup>
import { computed, ref } from 'vue'
import { data as repoCommitData } from '../data/commits.data.ts'

const hoveredDay = ref(null)

// 过去 52 周，按周一至周日排列；日期与提交次数来自现有 Git 数据加载器。
const calendarWeeks = computed(() => {
  const weeks = []
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  const startDay = new Date(today)
  const currentDayOfWeek = (startDay.getDay() + 6) % 7
  startDay.setDate(startDay.getDate() - (51 * 7 + currentDayOfWeek))

  const cursor = new Date(startDay)
  let currentWeek = []
  for (let i = 0; i < 52 * 7; i++) {
    const year = cursor.getFullYear()
    const month = String(cursor.getMonth() + 1).padStart(2, '0')
    const day = String(cursor.getDate()).padStart(2, '0')
    const date = `${year}-${month}-${day}`
    const count = repoCommitData.commitCounts[date] || 0
    const level = count >= 6 ? 4 : count >= 3 ? 3 : count >= 2 ? 2 : count >= 1 ? 1 : 0

    currentWeek.push({
      date,
      count,
      level,
      isToday: cursor.getTime() === today.getTime(),
      isFuture: cursor.getTime() > today.getTime(),
      month: cursor.getMonth() + 1
    })
    if (currentWeek.length === 7) {
      weeks.push(currentWeek)
      currentWeek = []
    }
    cursor.setDate(cursor.getDate() + 1)
  }
  return weeks
})

const monthLabels = computed(() => {
  const labels = []
  let lastMonth = -1
  calendarWeeks.value.forEach((week, index) => {
    const month = week[0]?.month
    if (month && month !== lastMonth) {
      labels.push({ text: `${month}月`, col: index + 1 })
      lastMonth = month
    }
  })
  return labels
})

const dayLabel = day => `${day.date}，${day.count} 次提交`
</script>

<template>
  <section class="activity-panel" aria-labelledby="activity-title">
    <div class="activity-header">
      <div>
        <p class="activity-kicker">Activity</p>
        <h2 id="activity-title">研习活跃度</h2>
        <p>过去 52 周的知识库 Git 提交记录</p>
      </div>
      <div class="activity-stats">
        <div><strong>{{ repoCommitData.totalCommits }}</strong><span>总提交</span></div>
        <div><strong>{{ repoCommitData.activeDays }}</strong><span>活跃天数</span></div>
        <div><strong class="latest-date">{{ repoCommitData.latestDate || '—' }}</strong><span>最近更新</span></div>
      </div>
    </div>

    <div class="heatmap-scroll">
      <div class="heatmap-chart">
        <div class="months-row" aria-hidden="true">
          <span v-for="month in monthLabels" :key="`${month.col}-${month.text}`" :style="{ gridColumnStart: month.col }">
            {{ month.text }}
          </span>
        </div>
        <div class="days-container">
          <div class="weekday-labels" aria-hidden="true"><span>一</span><span>三</span><span>五</span><span>日</span></div>
          <div class="weeks-grid">
            <div v-for="(week, weekIndex) in calendarWeeks" :key="weekIndex" class="week-col">
              <div
                v-for="day in week"
                :key="day.date"
                class="day-cell"
                :class="[`level-${day.level}`, { 'is-today': day.isToday, 'is-future': day.isFuture }]"
                :title="dayLabel(day)"
                :aria-label="dayLabel(day)"
                @mouseenter="hoveredDay = day"
                @mouseleave="hoveredDay = null"
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="heatmap-footer">
      <span class="hover-detail">
        <template v-if="hoveredDay && !hoveredDay.isFuture">{{ hoveredDay.date }} · {{ hoveredDay.count }} 次提交</template>
        <template v-else>每一个色块对应一天</template>
      </span>
      <div class="heatmap-legend" aria-label="提交次数由少到多">
        <span>少</span>
        <i v-for="level in 5" :key="level" :class="`level-${level - 1}`"></i>
        <span>多</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.activity-panel {
  padding: 1.6rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
}

.activity-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2rem;
  margin-bottom: 2rem;
}

.activity-kicker {
  margin: 0 0 0.45rem !important;
  color: var(--vp-c-text-3) !important;
  font-family: var(--vp-font-family-mono);
  font-size: 0.64rem !important;
  font-weight: 600;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.activity-header h2 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: 1.15rem;
  font-weight: 650;
  letter-spacing: -0.025em;
}

.activity-header p {
  margin: 0.45rem 0 0;
  color: var(--vp-c-text-3);
  font-size: 0.68rem;
}

.activity-stats {
  display: flex;
  gap: 2rem;
}

.activity-stats > div {
  display: flex;
  align-items: flex-end;
  flex-direction: column;
  gap: 0.25rem;
}

.activity-stats strong {
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 1rem;
  font-weight: 500;
}

.activity-stats strong.latest-date {
  font-size: 0.72rem;
  line-height: 1.55;
}

.activity-stats span {
  color: var(--vp-c-text-3);
  font-size: 0.62rem;
}

.heatmap-scroll {
  overflow-x: auto;
  padding-bottom: 0.4rem;
}

.heatmap-chart {
  min-width: 746px;
}

.months-row {
  display: grid;
  grid-template-columns: repeat(52, 12px);
  gap: 3px;
  height: 16px;
  margin: 0 0 4px 25px;
}

.months-row span {
  color: var(--vp-c-text-3);
  font-size: 0.62rem;
  white-space: nowrap;
}

.days-container {
  display: flex;
  gap: 7px;
}

.weekday-labels {
  display: flex;
  width: 18px;
  flex-direction: column;
  justify-content: space-between;
  padding: 1px 0;
  color: var(--vp-c-text-3);
  font-size: 0.58rem;
  line-height: 1;
  text-align: right;
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

.day-cell,
.heatmap-legend i {
  width: 12px;
  height: 12px;
  border-radius: 2px;
  background: var(--vp-c-bg-soft);
}

.day-cell {
  transition: transform 120ms ease;
}

.day-cell:hover {
  outline: 1px solid var(--vp-c-text-2);
  outline-offset: 1px;
  transform: scale(1.15);
}

.day-cell.is-today {
  outline: 1px solid var(--vp-c-brand-1);
  outline-offset: 1px;
}

.day-cell.is-future {
  opacity: 0.25;
}

.level-0 { background: #edefeb; }
.level-1 { background: #c9dcd2; }
.level-2 { background: #91b5a3; }
.level-3 { background: #568773; }
.level-4 { background: #2f6251; }

:global(.dark) .level-0 { background: #252a27; }
:global(.dark) .level-1 { background: #30493d; }
:global(.dark) .level-2 { background: #426b57; }
:global(.dark) .level-3 { background: #64947c; }
:global(.dark) .level-4 { background: #9ac2ac; }

.heatmap-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 1rem;
  color: var(--vp-c-text-3);
  font-size: 0.62rem;
}

.hover-detail {
  min-height: 1rem;
}

.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 4px;
}

.heatmap-legend i {
  display: block;
  width: 10px;
  height: 10px;
}

.heatmap-legend span {
  margin: 0 3px;
}

@media (max-width: 720px) {
  .activity-panel {
    padding: 1.2rem;
  }

  .activity-header {
    align-items: flex-start;
    flex-direction: column;
    gap: 1.3rem;
    margin-bottom: 1.5rem;
  }

  .activity-stats {
    width: 100%;
    justify-content: space-between;
    gap: 1rem;
  }

  .activity-stats > div {
    align-items: flex-start;
  }
}

@media (prefers-reduced-motion: reduce) {
  .day-cell {
    transition: none;
  }
}
</style>
