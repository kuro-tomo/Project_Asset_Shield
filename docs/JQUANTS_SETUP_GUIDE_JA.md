# J-Quants API セットアップガイド

## 概要

このガイドでは、Asset ShieldでJ-Quants APIを使用してバックテストを実行するための手順を説明します。

## 規約・制約の参照
J-Quants運用時の規約・制約（データ管理、セキュリティ、監査方針など）は [`MASTER_CONSTITUTION.md`](../MASTER_CONSTITUTION.md) を一次情報とします。

## 1. J-Quants APIプランの選択

### プラン比較表

| プラン | 月額料金 | API呼び出し/月 | レート制限 | 推奨用途 |
|--------|----------|----------------|------------|----------|
| **Free** | ¥0 | 1,000 | 12 req/min | 開発・テスト |
| **Light** | ¥1,980 | 10,000 | 60 req/min | 個人投資家 |
| **Standard** | ¥9,900 | 100,000 | 300 req/min | アクティブトレーダー |
| **Premium** | ¥49,500 | 1,000,000 | 600 req/min | 大規模運用・プラットフォーム実績 |

### 推奨プラン

- **開発・検証フェーズ**: Free または Light
- **本番バックテスト**: Standard 以上
- **リアルタイム運用**: Premium

### 20年バックテストに必要なAPI呼び出し数の見積もり

```
銘柄数 × 20年 × 約250営業日/年 ÷ 1回のAPI呼び出しで取得できる日数

例: 100銘柄の場合
- 日次データ: 100銘柄 × 1回 = 100 API呼び出し（キャッシュ後）
- 初回取得: 100銘柄 × 20年 ÷ 1年分/回 = 2,000 API呼び出し

→ Standard プラン以上を推奨
```

## 2. アカウント登録

### 手順

1. [J-Quants公式サイト](https://jpx-jquants.com/)にアクセス
2. 「新規登録」をクリック
3. メールアドレスとパスワードを設定
4. メール認証を完了
5. プランを選択（後からアップグレード可能）

### 必要な情報

- メールアドレス（認証用）
- パスワード（8文字以上、英数字混合推奨）
- 支払い情報（有料プランの場合）

## 3. 資格情報の設定

### 方法1: 環境変数（開発環境向け）

```bash
# ~/.zshrc または ~/.bashrc に追加
export JQUANTS_MAIL="your@email.com"
export JQUANTS_PASSWORD="your_password"

# 設定を反映
source ~/.zshrc
```

### 方法2: 資格情報ファイル（ローカル環境向け）

```bash
# ディレクトリ作成
mkdir -p ~/.jquants

# 資格情報ファイル作成
cat > ~/.jquants/credentials.json << EOF
{
    "mail_address": "your@email.com",
    "password": "your_password",
    "refresh_token": ""
}
EOF

# パーミッション設定（重要）
chmod 600 ~/.jquants/credentials.json
```

### 方法3: AWS Secrets Manager（本番環境向け）

```bash
# AWS CLIでシークレット作成
aws secretsmanager create-secret \
    --name "asset-shield/jquants-credentials" \
    --secret-string '{
        "mail_address": "your@email.com",
        "password": "your_password",
        "refresh_token": ""
    }'

# 環境変数でシークレット名を指定
export JQUANTS_SECRET_NAME="asset-shield/jquants-credentials"
```

## 4. バックテストの実行

### 事前確認（ドライラン）

```bash
# セットアップ確認
python scripts/run_jquants_backtest.py --dry-run

# 出力例:
# === DRY RUN - Checking Setup ===
# Provider Configuration:
#   Plan: free
#   Rate Limit: 12 req/min
#   Daily Limit: 1000 req/day
# Setup OK. Remove --dry-run to run backtest.
```

### バックテスト実行

```bash
# 基本実行（Free プラン、デフォルト銘柄）
python scripts/run_jquants_backtest.py

# カスタム設定
python scripts/run_jquants_backtest.py \
    --plan standard \
    --codes 7203,9984,6758,8306,9432 \
    --capital 100000000 \
    --credential-storage env

# 特定フェーズのみ実行
python scripts/run_jquants_backtest.py --phase phase_1_survival
```

### コマンドラインオプション

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `--plan` | J-Quantsプラン | free |
| `--codes` | 銘柄コード（カンマ区切り） | 7203,9984,6758 |
| `--capital` | 初期資本（円） | 100,000,000 |
| `--credential-storage` | 資格情報の取得元 | env |
| `--phase` | 実行するフェーズ | 全フェーズ |
| `--dry-run` | セットアップ確認のみ | - |

## 5. データキャッシュ

### キャッシュの仕組み

- 取得したデータは [`data/jquants_cache.db`](../data/jquants_cache.db) (SQLite) に保存
- 2回目以降はキャッシュからデータを取得（API呼び出し不要）
- キャッシュにより大幅なコスト削減が可能

### キャッシュ管理

```python
from src.shield.jquants_backtest_provider import DataCache

cache = DataCache("data/jquants_cache.db")

# キャッシュ統計
stats = cache.get_cache_stats()
print(f"キャッシュ済み: {stats['total_quotes']} レコード")
print(f"銘柄数: {stats['unique_stocks']}")
print(f"サイズ: {stats['cache_size_mb']:.2f} MB")

# API使用状況
usage = cache.get_api_usage_stats(days=30)
print(f"30日間のAPI呼び出し: {usage['total_calls']}")
```

### キャッシュのクリア

```bash
# キャッシュファイルを削除
rm data/jquants_cache.db
```

## 6. コスト監視

### 使用状況の確認

```python
from src.shield.jquants_backtest_provider import create_jquants_provider

provider = create_jquants_provider(plan="standard")
status = provider.get_status()

print(f"月間上限: {status['cost_estimate']['monthly_limit']} 回")
print(f"現在の使用: {status['cost_estimate']['current_usage']} 回")
print(f"残り: {status['cost_estimate']['remaining']} 回")
print(f"推定月額: ¥{status['cost_estimate']['estimated_cost_jpy']:,}")
```

### アラート設定（推奨）

```python
# 使用量が80%を超えたら警告
if status['cost_estimate']['current_usage'] > status['cost_estimate']['monthly_limit'] * 0.8:
    print("⚠️ 警告: API使用量が月間上限の80%を超えました")
```

## 7. セキュリティ考慮事項

### 資格情報の保護

- [ ] パスワードをコードにハードコードしない
- [ ] `.gitignore` に資格情報ファイルを追加
- [ ] 本番環境では AWS Secrets Manager を使用
- [ ] 定期的にパスワードをローテーション

### Burn Protocol との統合

資格情報は [`src/shield/security_governance.py`](../src/shield/security_governance.py) の Burn Protocol で管理されます：

```python
from src.shield.security_governance import BurnProtocol

burn = BurnProtocol()

# J-Quants資格情報を登録
burn.register_access("api_key", "jquants_credentials")

# 譲渡時にBurn実行
report = burn.execute_burn(dry_run=False)
```

## 8. トラブルシューティング

### 認証エラー

```
ERROR: J-Quants credentials not configured
```

**解決策**: 環境変数または資格情報ファイルを確認

```bash
# 環境変数確認
echo $JQUANTS_MAIL
echo $JQUANTS_PASSWORD
```

### レート制限エラー

```
ERROR: Daily API limit reached: 1000
```

**解決策**: 
1. 翌日まで待つ
2. プランをアップグレード
3. キャッシュを活用

### データ取得エラー

```
ERROR: J-Quants API request failed
```

**解決策**:
1. ネットワーク接続を確認
2. J-Quantsサービス状態を確認
3. 銘柄コードが正しいか確認

## 9. 推奨ワークフロー

### 開発フェーズ

1. Free プランで登録
2. 少数銘柄（3-5銘柄）でテスト
3. キャッシュを構築

### 検証フェーズ

1. Light/Standard プランにアップグレード
2. 対象銘柄全体のデータを取得
3. 全4フェーズのバックテストを実行

### 本番フェーズ

1. Premium プランを検討
2. AWS Secrets Manager で資格情報管理
3. 定期的なデータ更新スケジュールを設定

## 10. 参考リンク

- [J-Quants 公式サイト](https://jpx-jquants.com/)
- [J-Quants API ドキュメント](https://jpx-jquants.com/api-reference/)
- [JPX マーケットデータ](https://www.jpx.co.jp/markets/statistics-equities/)

## 11. 2026年1月30日 データ公開に伴う対応

2026年1月30日(金) 22:00より、J-Quants APIにて過去20年分を含む全データが取得可能になります。これに伴う特別な手順は以下の通りです。

### 11.1 初回データ取得手順
サーバー負荷集中を避けるため、以下のフェーズに分けて取得することを推奨します。

1.  **フェーズ1 (22:00 - 23:00):** 直近5年分 (2021-2026) の取得
    ```bash
    # Verificationフェーズ用のデータを優先確保
    python scripts/run_jquants_backtest.py --mode verify --plan standard
    ```
2.  **フェーズ2 (23:00 - ):** 残りの過去データ (2006-2020) の取得
    ```bash
    # Trainingフェーズ用のデータを取得
    python scripts/run_jquants_backtest.py --mode train --plan standard
    ```

### 11.2 注意点
*   **プラン確認:** 20年分のデータを効率よく取得するには `Standard` プラン以上が必須です。Free/Lightプランでは日次制限により完了まで数週間かかります。
*   **ディスク容量:** 全データキャッシュには約5GB-10GBの空き容量が必要です。
*   **APIエラー:** アクセス集中により `503 Service Unavailable` が発生する可能性があります。`JQuantsBacktestProvider` は自動リトライを行いますが、頻発する場合は時間を空けて実行してください。

---

## 付録: API エンドポイント一覧

| エンドポイント | 説明 | 使用例 |
|----------------|------|--------|
| `/token/auth_user` | 認証 | ログイン |
| `/token/auth_refresh` | トークン更新 | セッション維持 |
| `/listed/info` | 上場銘柄情報 | ユニバース取得 |
| `/prices/daily_quotes` | 日次株価 | バックテスト |
| `/trades` | 約定データ | マイクロストラクチャ分析 |
| `/orderbook` | 板情報 | 流動性分析 |
| `/markets/margin` | 信用残 | センチメント分析 |
| `/fins/statements` | 財務諸表 | ファンダメンタル分析 |
