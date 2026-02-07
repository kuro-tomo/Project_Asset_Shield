#!/usr/bin/env python3
"""
Asset Shield V3.2.0 - Data Parity Check
========================================

QuantConnect日本株データとローカルデータの完全一致を検証
対象: 10銘柄 (流動性上位から選択)

Author: Asset Shield Team
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'jquants_cache.db')

# 検証対象銘柄 (流動性上位10銘柄) - J-Quants 5桁形式
PARITY_STOCKS = [
    ("72030", "トヨタ自動車"),
    ("67580", "ソニーグループ"),
    ("83060", "三菱UFJ"),
    ("99840", "ソフトバンクグループ"),
    ("68610", "キーエンス"),
    ("94320", "NTT"),
    ("80350", "東京エレクトロン"),
    ("65010", "日立製作所"),
    ("79740", "任天堂"),
    ("40630", "信越化学"),
]


def extract_local_data():
    """ローカルDBから検証用データを抽出"""
    print("=" * 70)
    print("Phase 1: ローカルデータ抽出")
    print("=" * 70)

    conn = sqlite3.connect(CACHE_PATH)

    results = {}

    for code, name in PARITY_STOCKS:
        query = """
            SELECT date, open, high, low, close, volume, turnover, adjustment_close
            FROM daily_quotes
            WHERE code = ?
            AND date BETWEEN '2023-01-01' AND '2026-02-03'
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=[code])

        if df.empty:
            print(f"  ✗ {code} ({name}): データなし")
            continue

        # 調整済み終値を使用
        df['adj_close'] = df['adjustment_close'].fillna(df['close'])

        # 統計サマリー
        stats = {
            'records': len(df),
            'start_date': df['date'].min(),
            'end_date': df['date'].max(),
            'avg_close': df['adj_close'].mean(),
            'avg_volume': df['volume'].mean(),
            'avg_turnover': df['turnover'].mean(),
            'checksum': df['adj_close'].sum(),  # 簡易チェックサム
        }

        results[code] = {
            'name': name,
            'data': df,
            'stats': stats
        }

        print(f"  ✓ {code} ({name}): {stats['records']}件, "
              f"{stats['start_date']} → {stats['end_date']}")

    conn.close()

    return results


def generate_qc_comparison_data(local_data: dict):
    """QuantConnect比較用CSVを生成"""
    print("\n" + "=" * 70)
    print("Phase 2: QuantConnect比較用データ生成")
    print("=" * 70)

    output_dir = os.path.join(PROJECT_ROOT, 'data', 'parity_check')
    os.makedirs(output_dir, exist_ok=True)

    for code, info in local_data.items():
        df = info['data'].copy()

        # QuantConnect形式に変換 (OHLCV)
        qc_df = pd.DataFrame({
            'Date': pd.to_datetime(df['date']),
            'Open': df['open'],
            'High': df['high'],
            'Low': df['low'],
            'Close': df['adj_close'],
            'Volume': df['volume'].astype(int)
        })

        # CSV出力
        csv_path = os.path.join(output_dir, f'{code}_local.csv')
        qc_df.to_csv(csv_path, index=False)
        print(f"  → {csv_path}")

    return output_dir


def create_parity_report(local_data: dict):
    """Parityレポート生成"""
    print("\n" + "=" * 70)
    print("Phase 3: Parity Check Report")
    print("=" * 70)

    report = []
    report.append("# Asset Shield Data Parity Check Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## 検証対象銘柄 (10銘柄)")
    report.append("| Code | Name | Records | Start | End | Avg Close | Checksum |")
    report.append("|------|------|---------|-------|-----|-----------|----------|")

    total_checksum = 0

    for code, info in local_data.items():
        s = info['stats']
        report.append(
            f"| {code} | {info['name']} | {s['records']} | "
            f"{s['start_date']} | {s['end_date']} | "
            f"¥{s['avg_close']:,.0f} | {s['checksum']:,.0f} |"
        )
        total_checksum += s['checksum']

    report.append(f"\n**Total Checksum: {total_checksum:,.0f}**")

    report.append("\n## QuantConnect検証手順")
    report.append("""
1. QuantConnectにログイン
2. 新規アルゴリズムを作成
3. 以下のコードで同一銘柄のデータを取得
4. ローカルCSVと比較してChecksum一致を確認

```python
# QuantConnect Data Verification Code
from datetime import datetime
import numpy as np

class DataParityCheck(QCAlgorithm):

    STOCKS = ["7203", "6758", "8306", "9984", "6861",
              "9432", "8035", "6501", "7974", "4063"]

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10000000)

        # 日本株シンボル追加
        for code in self.STOCKS:
            # TSE銘柄コード形式: {code}.T または market指定
            symbol = self.AddEquity(f"{code}", Resolution.Daily, Market.Japan)
            self.Debug(f"Added: {code}")

    def OnData(self, data):
        pass

    def OnEndOfAlgorithm(self):
        for code in self.STOCKS:
            history = self.History(self.Symbol(f"{code}"),
                                   datetime(2023,1,1),
                                   datetime(2026,2,3),
                                   Resolution.Daily)

            if not history.empty:
                checksum = history['close'].sum()
                self.Debug(f"{code}: Records={len(history)}, Checksum={checksum:.0f}")
```
""")

    report.append("\n## 許容差異")
    report.append("""
- 終値差異: ±0.01% (株式分割調整による微差)
- 出来高差異: ±1% (データソースによる集計差)
- Checksum差異: ±0.1% で合格
""")

    # レポート保存
    report_path = os.path.join(PROJECT_ROOT, 'output', 'parity_check_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved: {report_path}")

    return '\n'.join(report)


def main():
    print("\n" + "=" * 70)
    print("Asset Shield V3.2.0 - Data Parity Check")
    print("=" * 70 + "\n")

    # 1. ローカルデータ抽出
    local_data = extract_local_data()

    if not local_data:
        print("ERROR: ローカルデータが見つかりません")
        return

    # 2. QC比較用CSV生成
    output_dir = generate_qc_comparison_data(local_data)

    # 3. レポート生成
    report = create_parity_report(local_data)

    print("\n" + "=" * 70)
    print("Parity Check準備完了")
    print("=" * 70)
    print(f"""
次のステップ:
1. QuantConnectにログイン
2. 上記レポートのコードを実行
3. Checksumを比較して一致確認
4. 差異が0.1%以内なら PASS

ローカルCSV: {output_dir}/
レポート: output/parity_check_report.md
""")


if __name__ == "__main__":
    main()
