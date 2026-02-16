#!/bin/bash
# X Bot JP - Cron Wrapper
# Runs x_bot_jp.py with proper environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/data/x_bot_jp"
LOG_FILE="$LOG_DIR/cron.log"

mkdir -p "$LOG_DIR"

# Load environment
export PATH="/usr/local/bin:/usr/bin:/bin:/Library/Developer/CommandLineTools/usr/bin:$PATH"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Load X API keys from .env or profile
# These should be set in .env:
#   X_API_KEY=...
#   X_API_SECRET=...
#   X_ACCESS_TOKEN=...
#   X_ACCESS_SECRET=...

# Determine mode based on day
MODE="${1:-daily}"
DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday
DAY_OF_MONTH=$(date +%d)

# Log rotation (keep last 10000 lines)
if [ -f "$LOG_FILE" ] && [ $(wc -l < "$LOG_FILE") -gt 10000 ]; then
    tail -5000 "$LOG_FILE" > "$LOG_FILE.tmp"
    mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

echo "========================================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Mode: $MODE" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run the bot
/usr/bin/python3 "$SCRIPT_DIR/x_bot_jp.py" --mode "$MODE" >> "$LOG_FILE" 2>&1

# Monday: also run weekly ranking
if [ "$DAY_OF_WEEK" -eq 1 ] && [ "$MODE" = "daily" ]; then
    echo "--- Running weekly ranking ---" >> "$LOG_FILE"
    /usr/bin/python3 "$SCRIPT_DIR/x_bot_jp.py" --mode weekly >> "$LOG_FILE" 2>&1
fi

# 1st of month: also run monthly report
if [ "$DAY_OF_MONTH" = "01" ] && [ "$MODE" = "daily" ]; then
    echo "--- Running monthly report ---" >> "$LOG_FILE"
    /usr/bin/python3 "$SCRIPT_DIR/x_bot_jp.py" --mode monthly >> "$LOG_FILE" 2>&1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - Complete" >> "$LOG_FILE"
