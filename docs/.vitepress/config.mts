import { defineConfig } from 'vitepress'
import { buildSidebar, collectNotes } from './lib/notes.mjs'

// 板块结构：侧边栏的标题与概览页文字。笔记条目由 frontmatter 自动生成，这里不用手动登记。
const sections = [
  { dir: 'domains/video', title: '视频生成与理解', overview: '板块概览' },
  {
    dir: 'domains/vision',
    title: '图像与计算机视觉',
    overview: '板块概览',
    subgroups: {
      'caption-efficiency': 'Caption 与效率',
      'datasets': 'AIGC 数据集',
      'image-rl-posttraining': '图像基模 RL 后训练',
    },
  },
  { dir: 'domains/llm', title: '语言模型与推理', overview: '板块概览' },
  { dir: 'domains/multimodal', title: '多模态与跨模态对齐', overview: '板块概览' },
  {
    dir: 'domains/rl-aigc',
    title: 'AIGC 强化学习后训练',
    overview: '板块概览与收录标准',
    subgroups: {
      'preference-model': '偏好模型',
      'reward-hacking': 'Reward Hacking',
      'dpo': 'DPO',
    },
  },
  { dir: 'domains/agent', title: '智能体与工具编排', overview: '板块概览与学习路线' },
  { dir: 'foundations', title: '通用理论基石', overview: '板块概览' },
  { dir: 'infra', title: '算力与系统工程', overview: '板块概览' },
  { dir: 'projects', title: '复现与落地项目', overview: '项目概览' },
  { dir: 'ideas', title: '科研灵感', overview: '灵感池概览' },
  { dir: 'daily-arxiv', title: '每日 arXiv', overview: '每日速览索引' },
  { dir: 'reviews', title: '研习复盘', overview: '复盘归档' },
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
    siteTitle: 'AlgoNotes',

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
        text: '研究领域',
        items: [
          { text: '视频生成与理解', link: '/domains/video/' },
          { text: '图像与计算机视觉', link: '/domains/vision/' },
          { text: '语言模型与推理', link: '/domains/llm/' },
          { text: '多模态与跨模态对齐', link: '/domains/multimodal/' },
          { text: 'AIGC 强化学习后训练', link: '/domains/rl-aigc/' },
          { text: '智能体与工具编排', link: '/domains/agent/' }
        ]
      },
      {
        text: '基石与工程',
        items: [
          { text: '通用理论基石', link: '/foundations/' },
          { text: '算力与系统工程', link: '/infra/' }
        ]
      },
      {
        text: '研习记录',
        items: [
          { text: '每日 arXiv', link: '/daily-arxiv/' },
          { text: '研习复盘', link: '/reviews/' },
          { text: '研习白板', link: '/whiteboard/' }
        ]
      },
      {
        text: '项目与灵感',
        items: [
          { text: '复现与落地项目', link: '/projects/' },
          { text: '科研灵感', link: '/ideas/' }
        ]
      },
      { text: '标签', link: '/tags/' }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/wangyephd' }
    ],

    // 侧边栏由笔记 frontmatter 生成（见 docs/.vitepress/lib/notes.mjs）
    sidebar: buildSidebar(sections, collectNotes()),

    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'medium'
      }
    }
  }
})
