import { execSync } from 'node:child_process'

export interface CommitStats {
  commitCounts: Record<string, number>
  commitDetails: Record<string, string[]>
  totalCommits: number
  activeDays: number
  latestDate: string
  latestMessage: string
}

declare const data: CommitStats
export { data }

export default {
  load(): CommitStats {
    try {
      const output = execSync('git log --pretty=format:"%ad|%s" --date=short', { encoding: 'utf-8' })
      const lines = output.split('\n').filter(Boolean)
      const commitCounts: Record<string, number> = {}
      const commitDetails: Record<string, string[]> = {}

      lines.forEach(line => {
        const firstPipe = line.indexOf('|')
        if (firstPipe === -1) return
        const date = line.slice(0, firstPipe).trim()
        const msg = line.slice(firstPipe + 1).trim()
        if (date) {
          commitCounts[date] = (commitCounts[date] || 0) + 1
          if (!commitDetails[date]) {
            commitDetails[date] = []
          }
          commitDetails[date].push(msg)
        }
      })

      const dates = Object.keys(commitCounts)
      const firstLine = lines[0] || ''
      const firstPipe = firstLine.indexOf('|')
      const latestDate = firstPipe !== -1 ? firstLine.slice(0, firstPipe).trim() : ''
      const latestMessage = firstPipe !== -1 ? firstLine.slice(firstPipe + 1).trim() : ''

      return {
        commitCounts,
        commitDetails,
        totalCommits: lines.length,
        activeDays: dates.length,
        latestDate,
        latestMessage
      }
    } catch {
      return {
        commitCounts: {},
        commitDetails: {},
        totalCommits: 0,
        activeDays: 0,
        latestDate: '',
        latestMessage: ''
      }
    }
  }
}
