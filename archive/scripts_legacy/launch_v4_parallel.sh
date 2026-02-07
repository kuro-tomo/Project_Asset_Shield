#!/bin/bash
# Asset Shield V4.0 - Parallel Backtest Launcher
# Launches 4 tmux panes for 20-year backtest

set -e

PROJECT_DIR="/Users/MBP/Desktop/Project_Asset_Shield"
OUTPUT_DIR="$PROJECT_DIR/output/v4_parallel"
VENV="$PROJECT_DIR/venv/bin/activate"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Kill existing session if exists
tmux kill-session -t v4backtest 2>/dev/null || true

echo "=============================================="
echo "ASSET SHIELD V4.0 - PARALLEL BACKTEST"
echo "=============================================="
echo "Launching 4 workers for 20-year backtest..."
echo ""

# Create new tmux session with 4 panes
tmux new-session -d -s v4backtest -n main

# Split into 4 panes (2x2 grid)
tmux split-window -h -t v4backtest
tmux split-window -v -t v4backtest:0.0
tmux split-window -v -t v4backtest:0.1

# Run workers in each pane
# Pane 0: Segment 1 (2008-2011)
tmux send-keys -t v4backtest:0.0 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 1 --start 2008-01-01 --end 2011-12-31 --output $OUTPUT_DIR/segment_1.json 2>&1 | tee $OUTPUT_DIR/log_1.txt" Enter

# Pane 1: Segment 2 (2012-2015)
tmux send-keys -t v4backtest:0.1 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 2 --start 2012-01-01 --end 2015-12-31 --output $OUTPUT_DIR/segment_2.json 2>&1 | tee $OUTPUT_DIR/log_2.txt" Enter

# Pane 2: Segment 3 (2016-2020)
tmux send-keys -t v4backtest:0.2 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 3 --start 2016-01-01 --end 2020-12-31 --output $OUTPUT_DIR/segment_3.json 2>&1 | tee $OUTPUT_DIR/log_3.txt" Enter

# Pane 3: Segment 4 (2021-2026)
tmux send-keys -t v4backtest:0.3 "cd $PROJECT_DIR && source $VENV && python scripts/v4_worker.py --segment 4 --start 2021-01-01 --end 2026-02-01 --output $OUTPUT_DIR/segment_4.json 2>&1 | tee $OUTPUT_DIR/log_4.txt" Enter

echo "Workers launched in tmux session 'v4backtest'"
echo ""
echo "To monitor: tmux attach -t v4backtest"
echo "To check progress: ls -la $OUTPUT_DIR/"
echo ""
echo "Once all 4 segments complete, run:"
echo "  python scripts/v4_aggregate.py $OUTPUT_DIR"
