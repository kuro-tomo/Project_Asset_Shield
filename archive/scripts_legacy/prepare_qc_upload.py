#!/usr/bin/env python3
"""
QuantConnect用カスタムデータ準備スクリプト
==========================================

J-QuantsデータをQuantConnect ObjectStore形式に変換

出力:
- CSVファイル (各銘柄)
- アップロード手順書
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'jquants_cache.db')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'quantconnect', 'custom_data')

# 対象銘柄 (流動性上位50)
TARGET_STOCKS = [
    "72030", "67580", "83060", "99840", "68610",
    "94320", "80350", "65010", "79740", "40630",
    "69020", "80580", "63670", "94330", "45020",
    "60980", "80010", "33820", "62730", "72670",
    "45030", "69540", "80310", "77510", "87660",
    "63260", "83160", "90220", "67020", "45680",
    "88020", "70110", "68570", "90200", "72690",
    "51080", "63010", "46610", "72700", "84110",
    "38610", "49010", "29140", "86040", "67520",
    "97350", "44520", "65940", "54010", "91010"
]


def prepare_custom_data():
    """カスタムデータCSVを準備"""
    print("=" * 70)
    print("QuantConnect カスタムデータ準備")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(CACHE_PATH)

    summary = []

    for code in TARGET_STOCKS:
        # データ取得
        query = """
            SELECT date, open, high, low, adjustment_close as close, volume
            FROM daily_quotes
            WHERE code = ?
            AND date >= '2008-01-01'
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=[code])

        if df.empty:
            print(f"  ✗ {code}: データなし")
            continue

        # NaN除去
        df = df.dropna()

        # QuantConnect形式でCSV出力
        # 形式: Date,Open,High,Low,Close,Volume (ヘッダーなし)
        csv_path = os.path.join(OUTPUT_DIR, f'{code}.csv')

        with open(csv_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(f"{row['date']},{row['open']},{row['high']},"
                       f"{row['low']},{row['close']},{int(row['volume'])}\n")

        summary.append({
            'code': code,
            'records': len(df),
            'start': df['date'].min(),
            'end': df['date'].max(),
            'path': csv_path
        })

        print(f"  ✓ {code}: {len(df)}件 ({df['date'].min()} → {df['date'].max()})")

    conn.close()

    # サマリー出力
    print("\n" + "=" * 70)
    print(f"完了: {len(summary)}/{len(TARGET_STOCKS)} 銘柄")
    print(f"出力先: {OUTPUT_DIR}")
    print("=" * 70)

    # アップロード手順書生成
    generate_upload_instructions(summary)

    return summary


def generate_upload_instructions(summary):
    """アップロード手順書を生成"""
    instructions = f"""# QuantConnect カスタムデータ アップロード手順

**生成日時:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**銘柄数:** {len(summary)}

---

## 方法1: ObjectStore経由 (推奨)

QuantConnect Webコンソールで以下を実行:

```python
# ObjectStoreにデータをアップロード
class DataUploader(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 2)

        # CSVデータをObjectStoreに保存
        stocks_data = {{
"""

    for item in summary[:10]:  # 最初の10銘柄のみ例示
        instructions += f'            "{item["code"]}": """...CSV data here...""",\n'

    instructions += """        }

        for code, data in stocks_data.items():
            self.ObjectStore.Save(f"japan/{code}.csv", data)
            self.Debug(f"Saved {code}")
```

## 方法2: GitHub経由

1. GitHubリポジトリを作成
2. CSVファイルをアップロード
3. Raw URLを使用してアクセス

```
https://raw.githubusercontent.com/YOUR_USERNAME/japan-stocks-data/main/72030.csv
```

## 方法3: Dropbox経由

1. CSVをDropboxにアップロード
2. 共有リンクを取得
3. `?dl=1` を追加してダイレクトリンクに変換

---

## アップロード対象ファイル

| Code | Records | Period | File |
|------|---------|--------|------|
"""

    for item in summary:
        instructions += f"| {item['code']} | {item['records']} | {item['start']} → {item['end']} | {os.path.basename(item['path'])} |\n"

    instructions += """

---

## 次のステップ

1. 上記いずれかの方法でデータをアップロード
2. `main_custom_data.py` の `GetSource` メソッドを更新
3. バックテスト実行
"""

    # 手順書保存
    instructions_path = os.path.join(OUTPUT_DIR, 'UPLOAD_INSTRUCTIONS.md')
    with open(instructions_path, 'w') as f:
        f.write(instructions)

    print(f"\n手順書: {instructions_path}")


if __name__ == "__main__":
    prepare_custom_data()
