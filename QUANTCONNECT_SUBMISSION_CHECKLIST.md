# QuantConnect バックテスト実行手順

**更新日時:** 2026-02-06
**ステータス:** ObjectStoreデータアップロード済み → バックテスト実行可能

---

## デプロイ手順

### 使用ファイル (全て32K文字以下)

```
quantconnect/UPLOAD_FUND_CHUNK1.py  ← Step 1a: ファンダメンタル (8銘柄, 25K chars)
quantconnect/UPLOAD_FUND_CHUNK2.py  ← Step 1b: ファンダメンタル (16銘柄, 27K chars)
quantconnect/UPLOAD_FUND_CHUNK3.py  ← Step 1c: ファンダメンタル (18銘柄, 27K chars)
quantconnect/UPLOAD_FUND_CHUNK4.py  ← Step 1d: ファンダメンタル (8銘柄, 15K chars)
quantconnect/ASSET_SHIELD_PRODUCTION.py  ← Step 2: 本番バックテスト (29K chars)
```

### Step 1: ファンダメンタルデータをObjectStoreにアップロード (4回)

各チャンクを順番に実行:

1. QuantConnect にログイン
2. 新規プロジェクト作成: **"FundUpload"**
3. `UPLOAD_FUND_CHUNK1.py` をペースト → バックテスト実行
4. 同じプロジェクトで内容を `UPLOAD_FUND_CHUNK2.py` に差し替え → 実行
5. `CHUNK3`, `CHUNK4` も同様に実行
6. 計50銘柄のファンダメンタルデータがObjectStoreに格納される

### Step 2: 本番アルゴリズムを実行

1. 新規プロジェクト作成: **"AssetShieldProduction"**
2. `ASSET_SHIELD_PRODUCTION.py` の全内容をコピーペースト
3. バックテスト実行

- 期間: 2008-05-07 ~ 2026-02-03 (自動設定済み)
- 初期資金: 10,000,000 JPY (自動設定済み)
- データ: ObjectStore `japan_stocks/*.csv` + `japan_fundamentals/*.csv` (50銘柄)

### Step 3: 結果確認

| チェック項目 | 基準 |
|-------------|------|
| OOS Sharpe (2021-2026) | >= 0.7 |
| Max Drawdown | <= 35% |
| トレード数 | > 0 |

---

## アルゴリズム仕様

| 項目 | 値 |
|------|-----|
| ユニバース | 50銘柄 (TOPIX大型株, ObjectStore CSV) |
| アルファ (5ファクター) | Value PBR (20%) + Quality ROE (15%) + Mom 12-1 (35%) + Short Mom (10%) + Low Vol (20%) |
| ファンダメンタル | **実データ** (J-Quants BPS/ROE/EPS, PIT準拠) |
| コストモデル | Commission 5bps + Spread f(ADT) + Market Impact (Almgren-Chriss) |
| レジーム | CRISIS / BEAR / NEUTRAL / BULL / SUPER_BULL (中央値ベース) |
| ポジション数 | 最大15銘柄 |
| リバランス | 63営業日ごと |
| ストップロス | -12% |
| テイクプロフィット | +35% |
| 最大保有期間 | 250営業日 |
| キルスイッチ | DD 18%で全ポジション清算、4段階回復 |
| セクター上限 | 30% |
| 最低ADT | 100M JPY |

### レジーム別エクスポージャー

| レジーム | 条件 | エクスポージャー |
|---------|------|----------------|
| CRISIS | Vol > 45% | 30% |
| BEAR | Vol > 25% & Trend < -3% | 60% |
| NEUTRAL | default | 100% |
| BULL | Trend > 3% | 110% |
| SUPER_BULL | Trend > 10% | 130% |

### Walk-Forward期間

| フェーズ | 期間 |
|---------|------|
| Training | 2008-05 ~ 2015-12 |
| Validation | 2016-01 ~ 2020-12 |
| OOS | 2021-01 ~ 2026-02 |

---

## ローカルバックテスト結果 (参考値)

| 指標 | 値 | 目標 |
|------|-----|------|
| Overall Sharpe | 1.09 | >= 0.8 |
| OOS Sharpe | 1.67 | >= 0.7 |
| Max DD | 12.93% | <= 35% |
| Win Rate | 57.11% | - |
| Profit Factor | 4.15 | - |

---

## ファイル構成

```
quantconnect/
├── ASSET_SHIELD_PRODUCTION.py        ← 本番アルゴリズム (5ファクター+コストモデル)
├── UPLOAD_FUND_CHUNK1~4.py           ← ファンダメンタルデータアップローダ (各32K以下)
├── UPLOAD_FUNDAMENTALS_READY.py      ← 一括版 (93K, QCには直接使えない)
├── UPLOAD_FUNDAMENTALS.py            ← アップローダテンプレート
├── ASSET_SHIELD_V57.py               ← V5.7 集中投資版 (旧版)
├── AssetShieldJP/
│   ├── main_objectstore.py           ← ObjectStore版 (旧版)
│   ├── main_objectstore_final.py     ← ObjectStore版 (旧版)
│   └── main_nikkei_futures.py        ← 先物版 (フォールバック)
├── custom_data/                      ← 50銘柄価格CSVデータ
└── fundamental_data/                 ← 50銘柄ファンダメンタルCSVデータ (BPS/ROE/EPS)
```

---

## トラブルシューティング

**データが読めない場合:**
- ObjectStoreに `japan_stocks/*.csv` が存在するか確認
- DataUploaderを再実行してデータを再アップロード

**バックテストがエラーで止まる場合:**
- QuantConnect Debugログを確認
- `Loaded X/50 symbols` のメッセージでデータ読み込み数をチェック

**Sharpeが低い場合:**
- パラメータ調整: `W_MOMENTUM`, `W_SHORT_MOM`, `W_LOW_VOL`
- `REBALANCE_DAYS` を 42 (月次) に変更
- `MAX_POSITIONS` を 10 に減らして集中投資
