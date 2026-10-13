#!/usr/bin/env bash
# 新建当天的 arXiv 速览文件（已存在则不覆盖）
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DATE="${1:-$(date +%F)}"
TARGET="$DIR/$DATE.md"

if [ -f "$TARGET" ]; then
  echo "已存在，未覆盖：$TARGET"
  exit 0
fi

sed "s/YYYY-MM-DD/$DATE/g" "$DIR/_template.md.txt" > "$TARGET"
echo "已创建：$TARGET"
echo "别忘了把 $DATE 插到 docs/daily-arxiv/index.md 表格最前面。"
