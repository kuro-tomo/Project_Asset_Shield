#!/bin/bash
# Asset Shield V4.1 - Optimized Parallel Backtest
# Parameters: Max DD 20%, Vol_Low 18%, Cooldown 3 days

set -e

PROJECT_DIR="/Users/MBP/Desktop/Project_Asset_Shield"
OUTPUT_DIR="$PROJECT_DIR/output/v41_parallel"
VENV="$PROJECT_DIR/venv/bin/activate"

mkdir -p "$OUTPUT_DIR"
tmux kill-session -t v41backtest 2>/dev/null || true

echo "=============================================="
echo "ASSET SHIELD V4.1 - OPTIMIZED PARALLEL BACKTEST"
echo "=============================================="
echo "Max DD: 20% | Vol_Low: 18% | Cooldown: 3 days"
echo ""

tmux new-session -d -s v41backtest -n main
tmux split-window -h -t v41backtest
tmux split-window -v -t v41backtest:0.0
tmux split-window -v -t v41backtest:0.1

# Worker 1: 2008-2011 (GFC)
tmux send-keys -t v41backtest:0.0 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 1 --start 2008-01-01 --end 2011-12-31 --output $OUTPUT_DIR/segment_1.json 2>&1 | tee $OUTPUT_DIR/log_1.txt" Enter

# Worker 2: 2012-2015 (Abenomics)
tmux send-keys -t v41backtest:0.1 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 2 --start 2012-01-01 --end 2015-12-31 --output $OUTPUT_DIR/segment_2.json 2>&1 | tee $OUTPUT_DIR/log_2.txt" Enter

# Worker 3: 2016-2020 (COVID)
tmux send-keys -t v41backtest:0.2 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 3 --start 2016-01-01 --end 2020-12-31 --output $OUTPUT_DIR/segment_3.json 2>&1 | tee $OUTPUT_DIR/log_3.txt" Enter

# Worker 4: 2021-2026 (Modern)
tmux send-keys -t v41backtest:0.3 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 4 --start 2021-01-01 --end 2026-02-01 --output $OUTPUT_DIR/segment_4.json 2>&1 | tee $OUTPUT_DIR/log_4.txt" Enter

echo "V4.1 Workers launched in tmux session 'v41backtest'"
echo "Monitor: tmux attach -t v41backtest"
echo "Output: $OUTPUT_DIR/"
