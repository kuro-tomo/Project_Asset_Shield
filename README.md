# Project Asset Shield - 自己進化型システム

## 概要
Project Asset Shieldは、日本株市場に特化した自己進化型アルゴリズム取引システムです。本システムは、QuantConnect（Alpha Market/Streams）およびQuantiacsでの**実績構築（Track Record）と資本配分（Allocation）獲得**を事業目的とし、大規模な運用資産（AUM）に対応しながら、市場環境の変化に自律的に適応し、運用者の安全と知的財産を保護するように設計されています。

詳細な規約、設計原則、運用方針については、[MASTER_CONSTITUTION.md](MASTER_CONSTITUTION.md) を参照してください。収益戦略・実績形成については [`docs/STRATEGIC_ANALYSIS_JA.md`](docs/STRATEGIC_ANALYSIS_JA.md) を参照してください。

## ディレクトリ構造

```
Project_Asset_Shield/
├── MASTER_CONSTITUTION.md # 自己進化型システムの基本規約と理念
├── GENOME_MAP.md          # システム全体像と依存関係マップ
├── src/                   # コアロジック (現状維持)
├── docs/                  # 最新の運用ガイド、戦略、デューデリジェンス関連文書
├── archive/               # 旧版ドキュメント、生成済みレポート、計画書などの履歴
├── config/                # 設定ファイル
├── data/                  # ローカルデータストレージ
├── logs/                  # アプリケーションログ
├── infrastructure/        # DockerおよびTerraform定義
├── scripts/               # ユーティリティおよびメンテナンススクリプト
├── tests/                 # ユニットテストおよび結合テスト
└── manage.py              # CLIエントリーポイント
```

## インストール

1. リポジトリをクローン:
   ```bash
   git clone <repository_url>
   cd Project_Asset_Shield
   ```
2. 依存関係のインストール:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## 使用方法

[MASTER_CONSTITUTION.md](MASTER_CONSTITUTION.md) に記載されている運用ガイドを参照してください。

## ライセンス
Proprietary & Confidential. All rights reserved.
