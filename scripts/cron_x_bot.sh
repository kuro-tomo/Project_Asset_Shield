#!/bin/bash
# X Bot JP - Cron Wrapper + note自動投稿 + 通知
# 各タスクの成否を記録し、失敗時にアラート、完了後にレポート送信

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/data/x_bot_jp"
LOG_FILE="$LOG_DIR/cron.log"
PY="/usr/bin/python3"
NOTIFIER="$SCRIPT_DIR/notifier.py"

mkdir -p "$LOG_DIR"

# Load environment
export PATH="/usr/local/bin:/usr/bin:/bin:/Library/Developer/CommandLineTools/usr/bin:$PATH"
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Helper: run task, record result, alert on failure
run_task() {
    local TASK_NAME="$1"
    shift
    echo "--- $TASK_NAME ---" >> "$LOG_FILE"
    if "$@" >> "$LOG_FILE" 2>&1; then
        $PY -c "
import sys; sys.path.insert(0,'$SCRIPT_DIR')
from notifier import record_result
record_result('$TASK_NAME', True)
" 2>/dev/null
        return 0
    else
        local EXIT_CODE=$?
        echo "  FAILED (exit $EXIT_CODE)" >> "$LOG_FILE"
        $PY -c "
import sys; sys.path.insert(0,'$SCRIPT_DIR')
from notifier import record_result, alert
record_result('$TASK_NAME', False, 'exit code $EXIT_CODE')
alert('$TASK_NAME 失敗', 'exit code: $EXIT_CODE\nログ: $LOG_FILE')
" 2>/dev/null
        return $EXIT_CODE
    fi
}

MODE="${1:-daily}"
DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday
DAY_OF_MONTH=$(date +%d)
MONTH_NUM=$(date +%-m)

# Log rotation (keep last 10000 lines)
if [ -f "$LOG_FILE" ] && [ $(wc -l < "$LOG_FILE") -gt 10000 ]; then
    tail -5000 "$LOG_FILE" > "$LOG_FILE.tmp"
    mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

echo "========================================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Mode: $MODE" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# === Daily tasks ===

# 1. X Bot JP tweet
run_task "X投稿($MODE)" $PY "$SCRIPT_DIR/x_bot_jp.py" --mode "$MODE"

# 2. note article generation (failure must not block publishing)
run_task "note記事生成($MODE)" $PY "$SCRIPT_DIR/note_article_generator.py" --mode "$MODE" --no-open --no-clipboard --quiet || true

# 3. note auto-publish
run_task "note投稿($MODE)" $PY "$SCRIPT_DIR/note_publisher.py" post "$MODE" || true

# === Backfill: catch up any missed daily articles ===
if [ "$MODE" = "daily" ]; then
    BACKFILL_DATES=$($PY -c "
import sqlite3, json
from pathlib import Path
ROOT = '$PROJECT_ROOT'
db = sqlite3.connect(f'{ROOT}/data/x_bot_jp/tracker.db')
dates = [r[0] for r in db.execute('SELECT DISTINCT disc_date FROM disclosures ORDER BY disc_date DESC LIMIT 10')]
db.close()
hp = Path(f'{ROOT}/data/note_publisher/publish_history.json')
done = set()
if hp.exists():
    for r in json.loads(hp.read_text()):
        if r.get('mode') == 'daily':
            done.add(r['date'])
for d in sorted(dates):
    if d not in done:
        print(d)
" 2>/dev/null)

    for BDATE in $BACKFILL_DATES; do
        echo "--- Backfill: $BDATE ---" >> "$LOG_FILE"
        run_task "note記事バックフィル($BDATE)" $PY "$SCRIPT_DIR/note_article_generator.py" --mode daily --date "$BDATE" --no-open --no-clipboard --quiet || true
        run_task "note投稿バックフィル($BDATE)" $PY "$SCRIPT_DIR/note_publisher.py" post daily --date "$BDATE" || true
    done
fi

# === Monday: weekly tasks ===
if [ "$DAY_OF_WEEK" -eq 1 ] && [ "$MODE" = "daily" ]; then
    run_task "X投稿(weekly)" $PY "$SCRIPT_DIR/x_bot_jp.py" --mode weekly || true
    run_task "note記事生成(weekly)" $PY "$SCRIPT_DIR/note_article_generator.py" --mode weekly --no-open --no-clipboard --quiet || true
    run_task "note投稿(weekly)" $PY "$SCRIPT_DIR/note_publisher.py" post weekly || true
fi

# === 1st of month: monthly report ===
if [ "$DAY_OF_MONTH" = "01" ] && [ "$MODE" = "daily" ]; then
    run_task "X投稿(monthly)" $PY "$SCRIPT_DIR/x_bot_jp.py" --mode monthly || true

    # Monthly note article (previous month)
    PREV_MONTH=$($PY -c "
from datetime import datetime, timedelta
d = datetime.now().replace(day=1) - timedelta(days=1)
print(f'{d.year} {d.month}')
" 2>/dev/null)
    PREV_YEAR=$(echo $PREV_MONTH | cut -d' ' -f1)
    PREV_MON=$(echo $PREV_MONTH | cut -d' ' -f2)
    run_task "note記事生成(monthly)" $PY "$SCRIPT_DIR/note_article_generator.py" --mode monthly --year "$PREV_YEAR" --month "$PREV_MON" --no-open --no-clipboard --quiet || true
    run_task "note投稿(monthly)" $PY "$SCRIPT_DIR/note_publisher.py" post monthly --date "${PREV_YEAR}-$(printf '%02d' $PREV_MON)" || true
fi

# === 1st of quarter: quarterly report (Jan/Apr/Jul/Oct) ===
if [ "$DAY_OF_MONTH" = "01" ] && [ "$MODE" = "daily" ]; then
    case $MONTH_NUM in
        1|4|7|10)
            PREV_Q=$($PY -c "
from datetime import datetime, timedelta
now = datetime.now()
q = (now.month - 1) // 3  # current quarter 0-indexed
if q == 0:
    print(f'{now.year - 1} 4')
else:
    print(f'{now.year} {q}')
" 2>/dev/null)
            Q_YEAR=$(echo $PREV_Q | cut -d' ' -f1)
            Q_NUM=$(echo $PREV_Q | cut -d' ' -f2)
            run_task "note記事生成(quarterly)" $PY "$SCRIPT_DIR/note_article_generator.py" --mode quarterly --year "$Q_YEAR" --quarter "$Q_NUM" --no-open --no-clipboard --quiet || true
            run_task "note投稿(quarterly)" $PY "$SCRIPT_DIR/note_publisher.py" post quarterly --date "${Q_YEAR}-Q${Q_NUM}" || true
            ;;
    esac
fi

# === Jan 1st: yearly report ===
if [ "$DAY_OF_MONTH" = "01" ] && [ "$MONTH_NUM" = "1" ] && [ "$MODE" = "daily" ]; then
    PREV_YEAR=$(($(date +%Y) - 1))
    run_task "note記事生成(yearly)" $PY "$SCRIPT_DIR/note_article_generator.py" --mode yearly --year "$PREV_YEAR" --no-open --no-clipboard --quiet || true
    run_task "note投稿(yearly)" $PY "$SCRIPT_DIR/note_publisher.py" post yearly --date "$PREV_YEAR" || true
fi

# === Scheduled essay publishing (Tuesday) ===
if [ "$MODE" = "daily" ]; then
    ESSAY_SCHEDULE="$PROJECT_ROOT/data/note_articles/essay_schedule.json"
    if [ -f "$ESSAY_SCHEDULE" ]; then
        TODAY=$(date '+%Y-%m-%d')
        ESSAY_FILE=$($PY -c "
import json
with open('$ESSAY_SCHEDULE') as f:
    schedule = json.load(f)
for item in schedule:
    if item['date'] == '$TODAY' and not item['published']:
        print(item['file'])
        break
" 2>/dev/null)
        if [ -n "$ESSAY_FILE" ]; then
            ESSAY_PATH="$PROJECT_ROOT/data/note_articles/$ESSAY_FILE"
            if [ -f "$ESSAY_PATH" ]; then
                run_task "noteエッセイ投稿" $PY "$SCRIPT_DIR/note_publisher.py" post essay --file "$ESSAY_PATH" || true
                # Mark as published in schedule
                $PY -c "
import json
with open('$ESSAY_SCHEDULE') as f:
    schedule = json.load(f)
for item in schedule:
    if item['date'] == '$TODAY':
        item['published'] = True
with open('$ESSAY_SCHEDULE', 'w') as f:
    json.dump(schedule, f, ensure_ascii=False, indent=2)
" 2>/dev/null
            fi
        fi
    fi
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - Complete" >> "$LOG_FILE"

# === Daily report email ===
run_task "日次レポート送信" $PY "$NOTIFIER" report || true
