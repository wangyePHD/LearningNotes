// 笔记 frontmatter 的受控词表。tags 只能从 TAG_VOCAB 里选，type / status 的取值也在这里定义。
// 新增标签要先在这里登记；Agent 不自行扩充词表。
export const TAG_VOCAB = [
  'Diffusion', 'DiT', 'Flow Matching', 'VAE', 'Representation Alignment', 'Unified Model', 'T2I',
  'Video Gen', 'Image Editing', 'Controllability', 'Caption', 'Data-centric', 'Prompt Rewriting',
  'Distillation', 'LLM', 'Reasoning', 'Agent', 'Tool Use', 'Planning', 'Structured Output',
  'Attention', 'Linear Attention', 'Complexity', 'Math Derivation', 'Multimodal', 'Contrastive Learning',
  'RL', 'GRPO', 'DPO', 'DiffusionNFT', 'Reward Model', 'Rubric', 'Reward Hacking', 'Fine-tuning',
  'Infra', 'Efficiency', 'Benchmark', 'Hypothesis',
]

export const TYPE_LABEL = {
  paper: '论文精读',
  survey: '综述',
  method: '方法与推导',
  playbook: '方法论手册',
  dataset: '数据集',
  project: '实战项目',
  idea: '灵感',
  log: '日志',
  runbook: '操作手册',
}

export const STATUS_LABEL = {
  draft: '📋 目录已建',
  reading: '🔄 精读中',
  done: '🟢 已完结',
  ongoing: '🟢 持续建设中',
}
