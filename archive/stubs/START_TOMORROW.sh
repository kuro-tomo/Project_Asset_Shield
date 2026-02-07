#!/bin/bash
# ========================================
# Asset Shield V3.2.0 - 明朝クイックスタート
# ========================================
# 実行: ./START_TOMORROW.sh
# ========================================

echo "========================================"
echo "Asset Shield V3.2.0 - QuantConnect投稿準備"
echo "========================================"
echo ""

# PATHにlean追加
export PATH="$PATH:/Users/MBP/Library/Python/3.9/bin"

# Step 0: 今日のデータ同期確認
echo "[Step 0] データ同期確認..."
cd /Users/MBP/Desktop/Project_Asset_Shield
python3 -c "
import sqlite3
conn = sqlite3.connect('data/jquants_cache.db')
result = conn.execute('SELECT MAX(date), COUNT(DISTINCT code) FROM daily_quotes').fetchone()
print(f'  最新日付: {result[0]}')
print(f'  銘柄数: {result[1]}')
conn.close()
"
echo ""

# Step 1: Parity Checkデータ確認
echo "[Step 1] Parity Checkデータ確認..."
echo "  CSVファイル:"
ls -la data/parity_check/*.csv | head -5
echo "  ..."
echo ""

# Step 2: QuantConnect ログイン
echo "[Step 2] QuantConnect準備..."
echo "  ブラウザで開く: https://www.quantconnect.com/terminal"
echo ""
echo "  または CLI ログイン:"
echo "  lean login"
echo ""

# Step 3: 次のアクション
echo "========================================"
echo "次のアクション (手動):"
echo "========================================"
echo ""
echo "1. QuantConnectにログイン"
echo "   open https://www.quantconnect.com/terminal"
echo ""
echo "2. 新規プロジェクト作成: 'AssetShieldJP'"
echo ""
echo "3. main.py をアップロード:"
echo "   cat quantconnect/AssetShieldJP/main.py | pbcopy"
echo "   (クリップボードにコピー済み)"
echo ""
echo "4. バックテスト実行 → OOS Sharpe >= 1.5 確認"
echo ""
echo "5. 全チェックPASS後のみ投稿"
echo ""
echo "========================================"
echo "詳細チェックリスト:"
echo "  cat QUANTCONNECT_SUBMISSION_CHECKLIST.md"
echo "========================================"
