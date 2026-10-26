import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import './custom.css'
import NoteList from '../components/NoteList.vue'
import NoteMeta from '../components/NoteMeta.vue'

export default {
  extends: DefaultTheme,
  // 笔记页顶部显示分类信息（来自 frontmatter）
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'doc-before': () => h(NoteMeta)
    })
  },
  enhanceApp({ app }) {
    app.component('NoteList', NoteList)
  }
}
