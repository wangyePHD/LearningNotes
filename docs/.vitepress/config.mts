import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AlgoNotes',
  description: '算法工程师的数字花园与研究笔记系统',
  lang: 'zh-CN',
  // 如果部署到 https://<USERNAME>.github.io/<REPO>/，可根据环境变量自动处理 base
  base: process.env.BASE_PATH || '/',

  // 核心：启用数学公式渲染 (LaTeX KaTeX/MathJax)
  markdown: {
    math: true,
    lineNumbers: true
  },

  themeConfig: {
    logo: '🧠',
    siteTitle: 'AlgoNotes 算法知识库',

    // 全局本地全文搜索（离线即用，无需申请任何云端服务）
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索笔记...',
                buttonAriaLabel: '搜索笔记'
              },
              modal: {
                noResultsText: '未找到相关结果',
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

    // 顶部导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '📐 理论与公式推导', link: '/algorithms/' },
      { text: '🚀 项目与论文复现', link: '/projects/' },
      { text: '⚡ 算力与工程基建', link: '/knowledge/' },
      { text: '🤖 Agent 协作指南', link: '/agent-guide' }
    ],

    // 侧边栏分模块自适应
    sidebar: {
      '/algorithms/': [
        {
          text: '理论与推导',
          items: [
            { text: '模块概览', link: '/algorithms/' },
            { text: 'Transformer 注意力机制与推导', link: '/algorithms/attention-mechanism' },
            { text: '扩散模型 (Diffusion) 数学原理解析', link: '/algorithms/diffusion-derivation' }
          ]
        }
      ],
      '/projects/': [
        {
          text: '项目与实验',
          items: [
            { text: '模块概览', link: '/projects/' },
            { text: 'LLM 7B 模型 LoRA 微调与吞吐评测', link: '/projects/llm-finetuning-benchmark' }
          ]
        }
      ],
      '/knowledge/': [
        {
          text: '工程与基建',
          items: [
            { text: '模块概览', link: '/knowledge/' },
            { text: 'CUDA 算子优化与内存瓶颈分析', link: '/knowledge/cuda-optimizations' }
          ]
        }
      ]
    },

    // 页面辅助功能
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
