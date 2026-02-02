# AWS移行および運用移管ガイド

## 1. 概要
本ドキュメントは、現在ローカル環境（MacBook Pro）で稼働している「Asset Shield」システムを、プラットフォーム実績の運用・継続稼働に伴いAWS環境へ移管するための手順書です。

## 2. システム構成
- **言語**: Python 3.9+
- **インフラ**: AWS (ECS Fargate, Aurora Serverless, Secrets Manager)
- **構成管理**: Terraform (Infrastructure as Code)

## 2.1 規約・制約
運用上の必須規約・制約（セキュリティ、データ管理、監査方針など）は [`MASTER_CONSTITUTION.md`](../MASTER_CONSTITUTION.md) を一次情報とします。

## 3. gRPCに関する技術的判断
現状のコードベース（Pythonモノリス）において、**gRPCへの改修は必須ではありません**。

- **現状**: REST/WebSocketおよび内部関数呼び出しで完結しており、正常に動作します。
- **将来**: マイクロサービス化（「Brain」と「Execution」の分離など）を進める段階で、既に用意されている `.proto` 定義を利用してgRPC化することが推奨されますが、初期移行のブロッカーにはなりません。
- **結論**: 提供するDockerコンテナをそのままAWS ECSにデプロイすることで、即座に運用開始可能です。

## 4. 移行ツールの使用方法
同梱の [`scripts/aws_migration_helper.py`](../scripts/aws_migration_helper.py) を使用することで、複雑なAWS構築手順を自動化できます。

### 事前準備
以下のツールがインストールされていることを確認してください（運用担当エンジニア向け）。
- Terraform
- Docker
- AWS CLI (認証済みであること)

### 手順
1. **初期化**:
   ```bash
   python scripts/aws_migration_helper.py init
   ```
   Terraformの初期化を行います。

2. **計画確認**:
   ```bash
   python scripts/aws_migration_helper.py plan
   ```
   構築されるリソースの一覧を表示します。ここでJ-QuantsのAPIキーの設定が求められる場合があります。

3. **デプロイ**:
   ```bash
   python scripts/aws_migration_helper.py deploy
   ```
   インフラの構築と、Dockerイメージのビルド・プッシュを自動で行います。

## 5. バックテストについて
J-QuantsのAPIキーが未設定の場合、バックテストは実行できません。
- **検証用**: `python scripts/run_jquants_backtest.py --dry-run` を実行することで、APIキー以外のシステム構成が正常であることを確認できます。
- **本番用**: APIキーを取得後、環境変数 `JQUANTS_API_KEY` を設定して実行してください。

---
*Asset Shield Deployment Team - 2026*
