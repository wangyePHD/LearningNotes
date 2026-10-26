import { defineConfig } from 'vitepress'
import { buildSidebar, collectNotes } from './lib/notes.mjs'

// 板块结构：侧边栏的标题与概览页文字。笔记条目由 frontmatter 自动生成，这里不用手动登记。
const sections = [
  { dir: 'domains/video', title: '🎬 视频生成与理解', overview: '领域概览与路线图' },
  {
    dir: 'domains/vision',
    title: '🖼️ 图像与计算机视觉',
    overview: '领域概览与路线图',
    subgroups: {
      'caption-efficiency': '📝 Caption 与效率专题',
      'datasets': '🗂️ AIGC 数据集专题',
      'image-rl-posttraining': '🖼️ 图像基模 RL 后训练',
    },
  },
  { dir: 'domains/llm', title: '💬 语言模型与推理思考', overview: '领域概览与路线图' },
  { dir: 'domains/multimodal', title: '🎙️ 多模态与跨模态对齐', overview: '领域概览与路线图' },
  {
    dir: 'domains/rl-aigc',
    title: '🎲 AIGC 强化学习后训练',
    overview: '专区概览与收录标准',
    subgroups: {
      'preference-model': '🏅 偏好模型子专题',
      'reward-hacking': '🚨 Reward Hacking 子专题',
    },
  },
  { dir: 'domains/agent', title: '🤖 智能体系统与工具编排', overview: '领域概览与 12 课路线图' },
  { dir: 'foundations', title: '📐 通用理论基石', overview: '基石模块概览' },
  { dir: 'infra', title: '⚡ 算力与系统工程', overview: '工程基建概览' },
  { dir: 'projects', title: '🚀 个人复现与落地项目', overview: '项目概览与看板' },
  { dir: 'ideas', title: '💡 科研灵感空间', overview: '科研 Idea 灵感池' },
  { dir: 'daily-arxiv', title: '📥 每日 arXiv 速览', overview: '每日速览索引与用法' },
  { dir: 'reviews', title: '📅 研习复盘与时光机', overview: '周报时光机总览' },
]

export default defineConfig({
  title: 'AlgoNotes',
  description: 'Ye Wang (王冶) 的全模态算法研究与工程知识库',
  lang: 'zh-CN',
  base: process.env.BASE_PATH || '/',

  // 核心：启用数学公式渲染 (LaTeX KaTeX / MathJax)
  markdown: {
    math: true,
    lineNumbers: true
  },

  themeConfig: {
    logo: '/avatar.jpeg',
    siteTitle: 'AlgoNotes | Ye Wang',

    // 全局本地离线毫秒级搜索
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索知识库...',
                buttonAriaLabel: '搜索知识库'
              },
              modal: {
                noResultsText: '未找到相关笔记',
                resetButtonTitle: '清除搜索条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          }
        }
      }
    },

    // 顶部导航栏（仪表盘通过左上角 logo 进入）
    nav: [
      {
        text: '📚 研究领域',
        items: [
          { text: '🎬 视频模型 (Video Intelligence)', link: '/domains/video/' },
          { text: '🖼️ 图像与视觉 (CV & Image)', link: '/domains/vision/' },
          { text: '💬 语言与推理 (LLM & Reasoning)', link: '/domains/llm/' },
          { text: '🎙️ 多模态与语音 (Multimodal)', link: '/domains/multimodal/' },
          { text: '🎲 AIGC 强化学习 (RL for AIGC)', link: '/domains/rl-aigc/' },
          { text: '🤖 智能体与工具编排 (Agent Systems)', link: '/domains/agent/' }
        ]
      },
      {
        text: '🧱 基石与工程',
        items: [
          { text: '📐 通用基石', link: '/foundations/' },
          { text: '⚡ 算力与工程', link: '/infra/' }
        ]
      },
      {
        text: '🗓️ 日志',
        items: [
          { text: '📥 每日 arXiv', link: '/daily-arxiv/' },
          { text: '📅 研习复盘', link: '/reviews/' },
          { text: '🧪 研习白板', link: '/whiteboard/' }
        ]
      },
      {
        text: '💡 灵感与项目',
        items: [
          { text: '💡 Idea 池', link: '/ideas/' },
          { text: '🚀 实战项目', link: '/projects/' }
        ]
      },
      { text: '🏷️ 知识标签', link: '/tags/' },
      { text: '📖 使用指南', link: '/guide/' }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/wangyephd' }
    ],

    // 侧边栏由笔记 frontmatter 生成（见 docs/.vitepress/lib/notes.mjs）
    sidebar: buildSidebar(sections, collectNotes()),

    outline: {
      level: [2, 3],
      label: '本页大纲'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    }
  }
})
