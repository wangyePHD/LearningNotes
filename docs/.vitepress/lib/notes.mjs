// 从笔记的 frontmatter 生成站点数据。笔记列表、侧边栏、标签矩阵、索引表共用这一份数据。
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { basename, join, relative, resolve, sep } from 'node:path'
import { execSync } from 'node:child_process'
import { TAG_VOCAB, TYPE_LABEL, STATUS_LABEL } from './tags.mjs'

// VitePress 从仓库根目录启动（npm run dev / build），所以以 cwd 为基准找 docs/。
export const DOCS_ROOT = resolve(process.cwd(), 'docs')

if (!existsSync(join(DOCS_ROOT, 'index.md'))) {
  throw new Error(`找不到文档目录 ${DOCS_ROOT}，请从仓库根目录运行 VitePress`)
}

const SKIP_DIRS = new Set(['public', 'node_modules'])

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name.startsWith('.') || SKIP_DIRS.has(name)) continue
    const full = join(dir, name)
    if (statSync(full).isDirectory()) walk(full, out)
    else out.push(relative(DOCS_ROOT, full).split(sep).join('/'))
  }
  return out
}

// 只解析本站用到的 YAML 子集：key: 单引号字符串 / 裸词 / [a, b] 列表。
function parseValue(raw) {
  if (raw.startsWith("'") && raw.endsWith("'")) return raw.slice(1, -1).replace(/''/g, "'")
  if (raw.startsWith('[') && raw.endsWith(']')) {
    return raw.slice(1, -1).split(',').map(s => s.trim()).filter(Boolean).map(parseValue)
  }
  return raw
}

export function splitFrontmatter(text) {
  const m = text.match(/^---\n([\s\S]*?)\n---\n/)
  if (!m) return null
  const data = {}
  for (const line of m[1].split('\n')) {
    const kv = line.match(/^([A-Za-z][\w-]*):\s*(.*)$/)
    if (kv) data[kv[1]] = parseValue(kv[2].trim())
  }
  return { data, body: text.slice(m[0].length) }
}

// 标题取正文第一个标题（跳过代码块）。
function firstHeading(body) {
  let inFence = false
  for (const line of body.split('\n')) {
    if (line.startsWith('```')) inFence = !inFence
    if (inFence) continue
    const m = line.match(/^#{1,6} (.+?)\s*$/)
    if (m) return m[1]
  }
  return ''
}

// 摘要显示用：去掉粗体、行内代码和数学定界符。源文件里保留原文。
export function plainText(s) {
  return s.replace(/\*\*/g, '').replace(/`/g, '').replace(/\$/g, '').replace(/\s+/g, ' ').trim()
}

// 每个笔记的日期取最近一次 git 提交；没有提交记录时回退到文件修改时间。
function gitDates() {
  const dates = new Map()
  try {
    const log = execSync('git log --name-only --format=@@%cs', {
      cwd: DOCS_ROOT,
      encoding: 'utf8',
      maxBuffer: 1 << 26,
      stdio: ['ignore', 'pipe', 'ignore'],
    })
    let date = ''
    for (const line of log.split('\n')) {
      if (line.startsWith('@@')) date = line.slice(2)
      else if (line.trim() && !dates.has(line.trim())) dates.set(line.trim(), date)
    }
  } catch {
    // 不在 git 仓库里：下面回退到文件修改时间
  }
  return dates
}

const statDate = file => new Date(statSync(file).mtimeMs).toISOString().slice(0, 10)

// 目录 -> 站内显示的领域名与徽章样式。more specific 的前缀要排在前面。
const DOMAIN_RULES = [
  ['domains/video', '🎬 视频模型', 'domain-video'],
  ['domains/vision', '🖼️ 图像视觉', 'domain-vision'],
  ['domains/llm', '💬 LLM', 'domain-llm'],
  ['domains/multimodal', '🎙️ 多模态', 'domain-multi'],
  ['domains/rl-aigc/preference-model', '🏅 偏好模型', 'domain-pm'],
  ['domains/rl-aigc/reward-hacking', '🚨 Reward Hacking', 'domain-rh'],
  ['domains/rl-aigc', '🎲 AIGC RL', 'domain-rl'],
  ['domains/agent', '🤖 智能体', 'domain-agent'],
  ['foundations', '📐 通用基石', 'domain-foundations'],
  ['infra', '⚡ 算力工程', 'domain-infra'],
  ['projects', '🚀 实战项目', 'domain-projects'],
  ['ideas', '💡 Idea', 'domain-ideas'],
  ['daily-arxiv', '📥 每日 arXiv', 'domain-log'],
  ['reviews', '📅 复盘', 'domain-log'],
  ['whiteboard', '🧪 白板', 'domain-log'],
]

function domainOf(dir) {
  const rule = DOMAIN_RULES.find(([prefix]) => dir === prefix || dir.startsWith(`${prefix}/`))
  return rule
    ? { domain: rule[1], domainClass: rule[2] }
    : { domain: '📄 其他', domainClass: '' }
}

// 收集所有笔记（docs/ 下除 index.md 以外的 .md）。缺少 frontmatter 或字段不合法时直接报错。
export function collectNotes() {
  const dates = gitDates()
  const notes = []
  const errors = []
  for (const rel of walk(DOCS_ROOT)) {
    if (!rel.endsWith('.md') || basename(rel) === 'index.md') continue
    const file = join(DOCS_ROOT, rel)
    const parsed = splitFrontmatter(readFileSync(file, 'utf8'))
    if (!parsed) {
      errors.push(`${rel}：缺少 frontmatter`)
      continue
    }
    const { data, body } = parsed
    const problems = []
    if (!firstHeading(body)) problems.push('没有标题')
    if (!TYPE_LABEL[data.type]) problems.push(`type 不合法：${data.type}`)
    if (!Array.isArray(data.tags)) {
      problems.push('tags 必须是列表')
    } else {
      for (const tag of data.tags) {
        if (!TAG_VOCAB.includes(tag)) problems.push(`未登记的标签：${tag}`)
      }
      if (data.tags.length === 0 && data.type !== 'log') problems.push('tags 不能为空')
    }
    if (data.status !== undefined && !STATUS_LABEL[data.status]) {
      problems.push(`status 不合法：${data.status}`)
    }
    if (typeof data.summary !== 'string' || data.summary === '') problems.push('缺少 summary')
    if (problems.length) {
      errors.push(`${rel}：${problems.join('；')}`)
      continue
    }
    const dir = rel.split('/').slice(0, -1).join('/')
    notes.push({
      file: rel,
      url: `/${rel.replace(/\.md$/, '')}`,
      dir,
      title: firstHeading(body),
      type: data.type,
      tags: data.tags,
      status: data.status ?? '',
      statusLabel: STATUS_LABEL[data.status] ?? '',
      summary: plainText(data.summary),
      models: Array.isArray(data.models) ? data.models : [],
      group: data.group ?? '',
      date: dates.get(`docs/${rel}`) ?? statDate(file),
      ...domainOf(dir),
    })
  }
  if (errors.length) {
    throw new Error(`笔记 frontmatter 有误：\n  ${errors.join('\n  ')}`)
  }
  return notes.sort((a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title, 'zh-CN'))
}

// 侧边栏：每个板块 = 概览页 + 该目录下的笔记；subgroups 是子专题（各自有索引页）。
export function buildSidebar(sections, notes) {
  const byDate = (a, b) => b.date.localeCompare(a.date) || a.title.localeCompare(b.title, 'zh-CN')
  const toItem = n => ({ text: n.title, link: n.url })
  const sidebar = {}
  for (const sec of sections) {
    const direct = notes.filter(n => n.dir === sec.dir).sort(byDate)
    const groups = [
      { text: sec.title, items: [{ text: sec.overview, link: `/${sec.dir}/` }, ...direct.map(toItem)] },
    ]
    for (const [sub, text] of Object.entries(sec.subgroups ?? {})) {
      const subNotes = notes.filter(n => n.dir === `${sec.dir}/${sub}`).sort(byDate)
      groups.push({
        text,
        items: [{ text: '专题概览', link: `/${sec.dir}/${sub}/` }, ...subNotes.map(toItem)],
      })
    }
    sidebar[`/${sec.dir}/`] = groups
  }
  return sidebar
}
