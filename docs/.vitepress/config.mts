import { defineConfig } from 'vitepress'

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
    logo: '🧠',
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

    // 顶部导航栏
    nav: [
      { text: '仪表盘', link: '/' },
      {
        text: '📚 研究领域',
        items: [
          { text: '🎬 视频模型 (Video Intelligence)', link: '/domains/video/' },
          { text: '🖼️ 图像与视觉 (CV & Image)', link: '/domains/vision/' },
          { text: '💬 语言与推理 (LLM & Reasoning)', link: '/domains/llm/' },
          { text: '🎙️ 多模态与语音 (Multimodal)', link: '/domains/multimodal/' }
        ]
      },
      { text: '📐 通用基石', link: '/foundations/' },
      { text: '⚡ 算力与工程', link: '/infra/' },
      { text: '🚀 实战项目', link: '/projects/' },
      { text: '🏷️ 知识标签', link: '/tags/' },
      { text: '💡 Idea 池', link: '/ideas/' },
      { text: '📅 研习复盘', link: '/reviews/' },
      { text: '🧪 研习白板', link: '/whiteboard/' },
      { text: '📖 使用指南', link: '/guide/' }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/wangyephd' }
    ],

    // 侧边栏联动
    sidebar: {
      // 0. 研习白板
      '/whiteboard/': [
        {
          text: '🧪 实时研习攻坚区',
          items: [
            { text: 'VideoDeltaNet (VDN-H3) 实时课堂', link: '/whiteboard/' }
          ]
        }
      ],

      // 0.1 使用指南
      '/guide/': [
        {
          text: '📖 知识库使用与维护',
          items: [
            { text: '系统使用与协作总览', link: '/guide/' },
            { text: 'Agent 自动化协作规范', link: '/agent-guide' }
          ]
        }
      ],

      // 0.2 研习复盘与周报
      '/reviews/': [
        {
          text: '📅 研习复盘与时光机',
          items: [
            { text: '周报时光机总览', link: '/reviews/' },
            { text: '2026年第38周复盘 (视频/算力)', link: '/reviews/2026-W38' }
          ]
        }
      ],

      // 0.5 科研 Idea 灵感池
      '/ideas/': [
        {
          text: '💡 科研灵感空间',
          items: [
            { text: '科研 Idea 灵感池', link: '/ideas/' }
          ]
        }
      ],

      // 1. 视频模型专属侧边栏
      '/domains/video/': [
        {
          text: '🎬 视频生成与理解',
          items: [
            { text: '领域概览与路线图', link: '/domains/video/' },
            { text: '时空注意力 (3D/Spatio-Temporal Attention)', link: '/domains/video/spatio-temporal-attention' },
            { text: 'Video DiT 架构与时空一致性', link: '/domains/video/video-diffusion-dit' }
          ]
        }
      ],

      // 2. 图像与视觉专属侧边栏
      '/domains/vision/': [
        {
          text: '🖼️ 图像与计算机视觉',
          items: [
            { text: '领域概览与路线图', link: '/domains/vision/' },
            { text: 'ControlNet 空间结构可控生成原理解析', link: '/domains/vision/diffusion-controlnet' }
          ]
        }
      ],

      // 3. 语言与推理专属侧边栏
      '/domains/llm/': [
        {
          text: '💬 语言模型与推理思考',
          items: [
            { text: '领域概览与路线图', link: '/domains/llm/' },
            { text: 'DeepSeek-R1 推理涌现与强化学习机制', link: '/domains/llm/deepseek-r1-reasoning' }
          ]
        }
      ],

      // 4. 多模态专属侧边栏
      '/domains/multimodal/': [
        {
          text: '🎙️ 多模态与跨模态对齐',
          items: [
            { text: '领域概览与路线图', link: '/domains/multimodal/' },
            { text: 'CLIP 跨模态对比表征学习数学解析', link: '/domains/multimodal/clip-alignment' }
          ]
        }
      ],

      // 5. 通用理论基石专属侧边栏
      '/foundations/': [
        {
          text: '📐 通用理论基石',
          items: [
            { text: '基石模块概览', link: '/foundations/' },
            { text: 'Transformer 缩放点积注意力与方差推导', link: '/foundations/transformer-attention' },
            { text: 'Softmax vs 线性注意力：记忆容量极限对比', link: '/foundations/linear-attention-capacity' },
            { text: 'Flow Matching 与连续生成动力学', link: '/foundations/flow-matching-derivation' }
          ]
        }
      ],

      // 6. 算力与工程专属侧边栏
      '/infra/': [
        {
          text: '⚡ 算力与系统工程',
          items: [
            { text: '工程基建概览', link: '/infra/' },
            { text: 'CUDA 显存层级与 Roofline 性能模型', link: '/infra/cuda-memory-hierarchy' },
            { text: 'vLLM PagedAttention 与吞吐优化', link: '/infra/vllm-paged-attention' }
          ]
        }
      ],

      // 7. 实战项目专属侧边栏
      '/projects/': [
        {
          text: '🚀 个人复现与落地项目',
          items: [
            { text: '项目概览与看板', link: '/projects/' },
            { text: 'LLM 7B 模型 LoRA 微调与吞吐评测', link: '/projects/llm-finetuning-benchmark' },
            { text: '开源视频生成模型本地复现与评测', link: '/projects/video-gen-reproduction' }
          ]
        }
      ]
    },

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
