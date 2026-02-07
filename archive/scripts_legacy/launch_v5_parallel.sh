#!/bin/bash
# Asset Shield V5.0 - Parallel Backtest (4 segments)

set -e

PROJECT_DIR="/Users/MBP/Desktop/Project_Asset_Shield"
OUTPUT_DIR="$PROJECT_DIR/output/v5_parallel"
VENV="$PROJECT_DIR/venv/bin/activate"

mkdir -p "$OUTPUT_DIR"
tmux kill-session -t v5backtest 2>/dev/null || true

echo "=============================================="
echo "ASSET SHIELD V5.0 - PARALLEL BACKTEST"
echo "=============================================="
echo "4 Segments: 2008-2012, 2012-2016, 2016-2020, 2020-2026"
echo ""

tmux new-session -d -s v5backtest -n main
tmux split-window -h -t v5backtest
tmux split-window -v -t v5backtest:0.0
tmux split-window -v -t v5backtest:0.1

# Worker 1: 2008-2012 (GFC + Recovery)
tmux send-keys -t v5backtest:0.0 "cd $PROJECT_DIR && source $VENV && python scripts/v5_worker.py --segment 1 --start 2008-05-01 --end 2012-12-31 --output $OUTPUT_DIR/segment_1.json 2>&1 | tee $OUTPUT_DIR/log_1.txt" Enter

# Worker 2: 2012-2016 (Abenomics)
tmux send-keys -t v5backtest:0.1 "cd $PROJECT_DIR && source $VENV && python scripts/v5_worker.py --segment 2 --start 2012-01-01 --end 2016-12-31 --output $OUTPUT_DIR/segment_2.json 2>&1 | tee $OUTPUT_DIR/log_2.txt" Enter

# Worker 3: 2016-2020 (Pre-COVID + COVID)
tmux send-keys -t v5backtest:0.2 "cd $PROJECT_DIR && source $VENV && python scripts/v5_worker.py --segment 3 --start 2016-01-01 --end 2020-12-31 --output $OUTPUT_DIR/segment_3.json 2>&1 | tee $OUTPUT_DIR/log_3.txt" Enter

# Worker 4: 2020-2026 (Post-COVID)
tmux send-keys -t v5backtest:0.3 "cd $PROJECT_DIR && source $VENV && python scripts/v5_worker.py --segment 4 --start 2020-01-01 --end 2026-02-01 --output $OUTPUT_DIR/segment_4.json 2>&1 | tee $OUTPUT_DIR/log_4.txt" Enter

echo "V5.0 Workers launched in tmux session 'v5backtest'"
echo "Monitor: tmux attach -t v5backtest"
echo "Output: $OUTPUT_DIR/"
