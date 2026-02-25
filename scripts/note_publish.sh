#!/bin/bash
# note_publish.sh - ワンコマンドでnote記事生成→自動投稿
# Usage:
#   ./scripts/note_publish.sh                  # 最新日のdaily記事
#   ./scripts/note_publish.sh weekly           # weekly記事
#   ./scripts/note_publish.sh daily 2026-02-17 # 日付指定
#   ./scripts/note_publish.sh setup            # 初回セットアップ

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

MODE="${1:-daily}"

# Setup mode
if [ "$MODE" = "setup" ]; then
    echo "=== note.com 初回セットアップ ==="
    /usr/bin/python3 "$SCRIPT_DIR/note_publisher.py" setup
    exit 0
fi

DATE_ARG=""
if [ -n "${2:-}" ]; then
    DATE_ARG="--date $2"
fi

echo "=== note記事生成: $MODE ==="
/usr/bin/python3 "$SCRIPT_DIR/note_article_generator.py" --mode "$MODE" --no-open --no-clipboard --quiet $DATE_ARG

echo "=== note自動投稿 ==="
/usr/bin/python3 "$SCRIPT_DIR/note_publisher.py" post "$MODE" $DATE_ARG

echo ""
echo "投稿完了。"
