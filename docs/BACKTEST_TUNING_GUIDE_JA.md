# バックテスト・チューニングガイド

## 概要

J-Quants APIを使用したバックテストでは、**チューニング（Training）を先に行い、その後に検証（Verification）を実行する**ワークフローが推奨されます。

## 推奨ワークフロー

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
│  Phase 1 (2006-2010): リーマンショック - 危機耐性テスト      │
│  Phase 2 (2011-2015): アベノミクス - 上昇相場追従テスト      │
│                                                             │
│  → Brain AIがパラメータを学習・最適化                       │
│  → brain_states_trained.json に保存                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  VERIFICATION PHASE                         │
│  Phase 3 (2016-2020): COVID - アウトオブサンプル検証        │
│  Phase 4 (2021-2026): 現代 - インフレ・金利上昇環境         │
│                                                             │
│  → 学習済みパラメータで未知データをテスト                   │
│  → 過学習していないか確認                                   │
└─────────────────────────────────────────────────────────────┘
```

## なぜチューニングを先に行うのか？

### 1. 過学習（Overfitting）の防止
- 全期間でパラメータを最適化すると、過去データに過剰適合する
- Training/Verification分離により、汎化性能を検証できる

### 2. 市場レジームへの適応
- Phase 1-2で危機・回復・上昇相場を経験
- Brain AIが適切なリスク管理パラメータを学習

### 3. 実運用への信頼性
- Phase 3-4は「未知のデータ」として扱う
- ここで良好な結果が出れば、実運用でも期待できる

## 使用方法

## 運用規約・制約の参照
バックテスト運用時に遵守すべき規約・制約（リスク管理、データ管理、監査方針など）は [`MASTER_CONSTITUTION.md`](../MASTER_CONSTITUTION.md) を一次情報とします。

### Step 1: J-Quants認証情報の設定

```bash
export JQUANTS_MAIL="your@email.com"
export JQUANTS_PASSWORD="your_password"
```

### Step 2: セットアップ確認（Dry Run）

```bash
PYTHONPATH=src python3 scripts/run_jquants_backtest.py --dry-run --mode full
```

### Step 3: トレーニング実行

```bash
PYTHONPATH=src python3 scripts/run_jquants_backtest.py --mode train --codes 7203,9984,6758
```

出力:
- [`output/brain_states_trained.json`](../output/brain_states_trained.json) - 学習済みパラメータ
- `output/backtest_TRAIN_*.json` - トレーニング結果

### Step 4: 検証実行

```bash
PYTHONPATH=src python3 scripts/run_jquants_backtest.py --mode verify --codes 7203,9984,6758
```

### Step 5: フル実行（Training + Verification）

```bash
PYTHONPATH=src python3 scripts/run_jquants_backtest.py --mode full --codes 7203,9984,6758
```

## Brain AIの学習メカニズム

### 学習パラメータ

| パラメータ | 初期値 | 説明 |
|-----------|--------|------|
| `adaptive_threshold` | 0.65 | エントリー信頼度閾値 |
| `risk_penalty` | 1.5 | ボラティリティペナルティ係数 |

### 学習ルール

```python
# 損失トレード後
if pnl < 0:
    adaptive_threshold += 0.10  # より慎重に
    risk_penalty += 0.3         # リスク回避強化

# 利益トレード後
if pnl > 0:
    adaptive_threshold -= 0.01  # 少し積極的に
```

### 期待される学習結果

- **Phase 1（リーマンショック）後**: threshold↑, risk_penalty↑
  - 危機時の損失から学習し、より保守的に
  
- **Phase 2（アベノミクス）後**: threshold↓
  - 上昇相場での利益から学習し、適度に積極的に

## API使用量の最適化

### キャッシュ活用

- 一度取得したデータは [`data/jquants_cache.db`](../data/jquants_cache.db) に保存
- 2回目以降はAPIコール不要

### プラン別制限

| プラン | リクエスト/分 | 日次上限 | 月額 |
|--------|--------------|---------|------|
| Free | 12 | 1,000 | ¥0 |
| Light | 60 | 10,000 | ¥1,980 |
| Standard | 300 | 100,000 | ¥9,900 |
| Premium | 600 | 1,000,000 | ¥49,500 |

### 推奨

1. **開発・テスト**: Freeプランで十分
2. **本番バックテスト**: Lightプラン推奨
3. **大規模スクリーニング**: Standard以上

## トラブルシューティング

### 「J-Quants credentials not found」

```bash
# 環境変数を設定
export JQUANTS_MAIL="your@email.com"
export JQUANTS_PASSWORD="your_password"
```

### 「Total Trades: 0」

- データが50日未満の場合、Brainは取引しない
- キャッシュに十分なデータがあるか確認

### 「Rate limit reached」

- Freeプランは12リクエスト/分
- 自動的に待機するが、大量データ取得時は時間がかかる

## 結果の解釈

### 成功基準

| Phase | 条件 |
|-------|------|
| Phase 1 | Max DD < 40%, Sharpe > -0.5 |
| Phase 2 | Return > 5%, Sharpe > 0.3 |
| Phase 3 | Sharpe > 0.2, Max DD < 35% |
| Phase 4 | Sharpe > 0.3, Return > 3% |

### 最終判定

- **全Phase合格**: 本番デプロイ可能
- **Training合格、Verification不合格**: 過学習の可能性、パラメータ調整必要
- **Training不合格**: 戦略自体の見直しが必要
