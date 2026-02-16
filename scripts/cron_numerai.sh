#!/bin/bash
# Asset Shield - Numerai Daily Submission (launchd)
# Both ASSET_SHIELD and MIKASA models
set -e

cd /Users/MBP/Desktop/Project_Asset_Shield
LOG="data/numerai/cron.log"
mkdir -p data/numerai

echo "" >> "$LOG"
echo "=== Numerai submission $(date '+%Y-%m-%d %H:%M:%S JST') ===" >> "$LOG"

# Load API keys from .env
set -a
source .env 2>/dev/null || true
set +a

# Fresh live data
rm -f data/numerai/live.parquet

# ASSET_SHIELD
echo "[1/2] ASSET_SHIELD..." >> "$LOG"
/usr/bin/python3 scripts/numerai_starter.py --mode submit --model-name asset_shield --ensemble >> "$LOG" 2>&1
RC1=$?

# MIKASA
echo "[2/2] MIKASA..." >> "$LOG"
/usr/bin/python3 scripts/numerai_starter.py --mode submit --model-name mikasa --ensemble --neutralize >> "$LOG" 2>&1
RC2=$?

if [ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; then
    echo "=== Both models submitted OK ===" >> "$LOG"
    osascript -e 'display notification "ASSET_SHIELD + MIKASA 提出完了" with title "Numerai" sound name "Glass"' 2>/dev/null || true
else
    echo "=== SUBMISSION FAILED (asset_shield=$RC1, mikasa=$RC2) ===" >> "$LOG"
    osascript -e 'display notification "Numerai提出失敗！ログを確認" with title "Numerai ERROR" sound name "Sosumi"' 2>/dev/null || true
fi
