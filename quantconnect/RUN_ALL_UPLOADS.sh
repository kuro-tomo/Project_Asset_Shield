#!/bin/bash
# Asset Shield - 20 Stock Upload Script
# 17 new stocks to upload (3 already done: 72030, 67580, 83060)

echo "=== Asset Shield 20-Stock Upload ==="
echo "Execute each command, paste into QuantConnect, run backtest"
echo ""

STOCKS=(68570 80350 70110 70120 79740 65010 99840 83160 65260 95010 68610 60980 40630 72670 87660 94320 91040)

for i in "${!STOCKS[@]}"; do
    num=$((i + 1))
    code=${STOCKS[$i]}
    echo "[$num/17] Stock $code:"
    echo "  cat /Users/MBP/Desktop/Project_Asset_Shield/quantconnect/uploaders20/upload_${code}.py | pbcopy"
    echo ""
done

echo "[FINAL] 20-Stock Backtest:"
echo "  cat /Users/MBP/Desktop/Project_Asset_Shield/quantconnect/BACKTEST_20STOCKS.py | pbcopy"
echo ""
echo "=== Total: 17 uploads + 1 backtest ==="
