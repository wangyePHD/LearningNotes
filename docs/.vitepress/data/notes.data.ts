import { collectNotes } from '../lib/notes.mjs'

export interface NoteEntry {
  file: string
  url: string
  dir: string
  title: string
  type: string
  tags: string[]
  status: string
  statusLabel: string
  summary: string
  models: string[]
  group: string
  date: string
  domain: string
  domainClass: string
}

declare const data: NoteEntry[]
export { data }

export default {
  load(): NoteEntry[] {
    return collectNotes()
  }
}
