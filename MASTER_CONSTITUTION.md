# MASTER_CONSTITUTION.md: 自己進化型システム「Asset Shield」憲章

## 1. はじめに

### 1.1 目的と理念
本憲章は、日本株市場を対象とした中長期AI運用システム「Asset Shield」が、自己進化を遂げながら持続的に価値を創造し、QuantConnect（Alpha Market/Streams）およびQuantiacsを通じた**グローバルな実績（Track Record）構築**と**資本配分（Allocation）の獲得**を最重要目的として達成するための根幹を定める。ここでの実績は、J-Quantsの20年データに基づくバックテストと、プラットフォーム上の公開実績・ランキングで検証可能な形で提示される。

### 1.2 自己進化型システムとしての基本原則
Asset Shieldは、以下の4つの柱を基盤として設計・運用される。

*   **300億円キャパシティ**: プラットフォーム上で上位ランクを狙うための**核心的優位性**として、流動性制約下でも安定したリスク調整後リターンを維持できる設計を徹底する。
*   **生存バイアス排除**: 過去データに過剰適合せず、未知の市場環境においても安定したパフォーマンスを発揮する汎用性と堅牢性を追求する。
*   **低回転率**: 取引コストと市場インパクトを抑制し、プラットフォーム評価指標（安定性・実現可能性）で上位を狙うための**核心的優位性**として、回転率を戦略的に低く保つ。
*   **自己増殖の安全弁**: システムの核となる知的財産と運用者の安全を確保するため、デッドマンスイッチを含むセキュリティプロトコルを実装する。

## 2. 運用規約

### 2.1 300億円キャパシティの維持
- **AUM目標**: 300億円（100億円 × 3社）の運用に耐えうる設計とする。
- **流動性チェック**: [`src/shield/alpha_model.py`](src/shield/alpha_model.py) のAlphaModelでは、個別銘柄の流動性を厳格に評価し、300億円規模の注文がマーケットに与える影響を最小化するAlmgren-Chrissモデルを適用する。具体的には、`data[symbol].Price * data[symbol].Volume < 3000000000` のような条件で流動性不足銘柄を排除する。
- **ポジションサイジング**: Kelly基準に基づき、リスク許容度と現在の市場環境に応じて動的にポジションサイズを調整する [`src/shield/money_management.py`](src/shield/money_management.py) を採用し、過度な集中投資を避ける。

### 2.2 生存バイアスの完全排除
- **バックテスト**: 20年間のマルチレジームストレステストを実施し、上場廃止銘柄を含む時系列データを正確に処理することで、過去データへの過剰最適化（オーバーフィッティング）およびルックアヘッドバイアスを排除する。[`src/shield/backtest_framework.py`](src/shield/backtest_framework.py) がこの機能を担う。
- **データ品質**: J-Quants APIからのデータ取得において、欠損値や誤った調整データに注意し、キャッシュシステム [`data/jquants_cache.db`](data/jquants_cache.db) を活用してデータの整合性を確保する。
- **学習と検証の分離**: バックテストはTraining Phase (2006-2015)でBrain AIがパラメータを学習し、Verification Phase (2016-2026)で学習済みパラメータを未知データに適用することで、汎化性能を厳しく検証する。

### 2.3 市場流動性への対応
- **執行アルゴリズム**: VWAP/TWAPおよびAlmgren-Chrissモデルを採用し、大規模な取引が市場に与える影響を最小化する。TSEの呼び値ルールに完全対応する。
- **レジーム検出**: [`src/shield/adaptive_core.py`](src/shield/adaptive_core.py) により、市場レジーム（危機、高ボラティリティ、通常、低ボラティリティ、トレンド）を動的に検出し、それに応じてリスク許容度やポジションスケーリングを調整する。

### 2.4 リスク管理とドローダウン抑制
- **最大ドローダウン**: 危機時においても最大ドローダウンを40%未満に抑制することを目標とする。Bootstrapモードではこれを緩和し、実績作りのためのリターンを優先する。
- **Sharpe Ratio**: 全てのフェーズで0.3以上を目標とし、リスク調整後リターンの最大化を図る。
- **パラメータ調整**: [`src/shield/brain.py`](src/shield/brain.py) において、損失トレード後に `adaptive_threshold` と `risk_penalty` を増加させ慎重な運用に、利益トレード後に `adaptive_threshold` を減少させ積極的に運用する学習ルールを持つ。

### 2.5 API利用とデータ管理
- **J-Quants API**: JPX公式のJ-Quants APIを主要データソースとし、安定したデータ供給を確保する。プラン選択は、開発・検証フェーズではFree/Light、本番バックテストではStandard以上、リアルタイム運用ではPremiumを推奨する。
- **資格情報**: J-Quantsの資格情報は、環境変数、資格情報ファイル、またはAWS Secrets Managerを用いて厳重に管理し、コードへのハードコードを避ける。`Burn Protocol` と連携し、譲渡時に安全に破棄される仕組みとする。

## 3. セキュリティとガバナンス

### 3.1 自己増殖の安全弁 (Silence Protocol)
- **デッドマンスイッチ**: 運用者からの定期的な「生存信号（Heartbeat）」が途絶えた場合、自動的にコアロジックを破壊し、証拠を隠滅するシステム [`scripts/silence_protocol.py`](scripts/silence_protocol.py) を実装する。
- **トリガー条件**: HEARTBEAT_TIMEOUT, PANIC_BUTTON, REMOTE_SIGNAL, CANARY_DEATH, INTRUSION_DETECTED, DURESS_CODEなど、複数のトリガー条件を設定する。
- **脅迫コード**: 物理的拘束下で入力された場合、表面上は正常動作しながらバックグラウンドで証拠隠滅を開始する特殊コードを備える。
- **セキュアワイプ**: DoD 5220.22-M 短縮版（3パス）を推奨し、ランダムデータで重要ファイルを複数回上書きすることで、データの復元を極めて困難にする。
- **デコイ生成**: 重要ファイルを削除する代わりに、無害なダミーコードで置き換え、不自然なファイル消失を隠蔽する。

### 3.2 技術的整合性証明と監査
- **Self-Validated Model**: 高額な外部ベンダー監査を回避するため、システム自身が生成する「監査証跡」と「完全性証明」を提示する。これには、全ソースコードのSHA-256ハッシュ値を含む `integrity_certificate.json` と、システム稼働ログの改ざん検知データベース `logs/audit.db` が含まれる。
- **法的適法性の主張**: 金商法および関連法規を遵守して設計されている旨の自己宣誓書を作成し、OSSライセンスの利用状況一覧を添付する。

### 3.3 PIIサニタイズと通信経路の匿名化
- **PII検出**: メールアドレス、電話番号、IPアドレス、パスポート番号、APIキー/パスワードなどの個人特定情報（PII）を自動検出・サニタイズする機能を備える。
- **匿名通信**: 商用VPN、Torネットワーク、自己管理VPSを組み合わせた多層的な匿名化レイヤーを推奨し、User-Agentのランダム化、リクエスト間隔のジッター、IPローテーションにより通信経路の匿名性を確保する。

### 3.4 緊急時対応フロー
- 通常の緊急停止（パニックコード）、物理的拘束下での静かな自壊（脅迫コード）、通信途絶による自動自壊、外部からの緊急停止（SIGUSR1）など、多岐にわたる緊急事態に対応するフローを定義する。

## 4. システムアーキテクチャと進化

### 4.1 モジュール構成と役割
- **shieldパッケージ**: ビジネスロジックの主要パッケージ。[`src/shield/adaptive_core.py`](src/shield/adaptive_core.py)（市場レジーム検出）、[`src/shield/brain.py`](src/shield/brain.py)（AI/MLモデル）、[`src/shield/execution_core.py`](src/shield/execution_core.py)（注文管理・執行）、[`src/shield/jquants_client.py`](src/shield/jquants_client.py)（J-Quants APIインターフェース）、[`src/shield/screener.py`](src/shield/screener.py)（ファンダメンタル分析）などのコアコンポーネントを含む。
- **Infrastructure**: DockerとTerraformによるAWS/GCPに依存しないインフラ定義。ワンクリックデプロイを可能とする。

### 4.2 自己修復・自己進化・自己増殖メカニズム
- **自己修復 (Self-Repair)**: [`src/shield/bio/repair.py`](src/shield/bio/repair.py) の `IntegrityMonitor` が、主要ファイルのSHA256ハッシュ値を格納した `dna_manifest.json` と比較し、改ざんや欠損を検知した場合、バックアップから自動修復する。
- **自己進化 (Self-Evolution)**: [`src/shield/bio/evolution.py`](src/shield/bio/evolution.py) の `EvolutionEngine` が、パフォーマンス（PnL, Win Rate, Drawdown）を監視し、停滞や危機時に `risk_penalty`, `adaptive_threshold` などのパラメータを調整・変異させることで、人間の介入なしにパフォーマンスを改善する。
- **自己増殖 (Self-Replication)**: [`src/shield/bio/replication.py`](src/shield/bio/replication.py) の `ReplicationManager` が、CPU負荷やエージェント数に応じて新しいPythonプロセスを起動し、処理能力と生存性を拡大する。エージェントの死活監視と自動復活機能を持つ。

### 4.3 パラメータチューニングと学習メカニズム
- **学習ルール**: Brain AIは損失トレード後に慎重さを増し、利益トレード後に積極性を増すようにパラメータを動的に調整する。
- **最適化目標**: 年率10%〜15%のリターン、Sharpe Ratio 1.2以上、Exposure 50%〜80%を目標とする。

### 4.4 開発標準と品質保証
- **コーディング標準**: Pythonを主言語とし、全てのソースコードコメントは英語、型ヒントを必須とし、モジュール性とテスト容易性を重視した設計を遵守する。
- **テスト**: ユニットテストおよび統合テストを徹底し、コード変更によるパフォーマンス劣化がないことを確認する。

## 5. 事業戦略と評価

### 5.1 収益モデルと評価額
- **フェーズ戦略**: Bootstrap（実績証明フェーズ）からExpansion（外部資金導入フェーズ）へ移行し、段階的にAUMを拡大する。
- **収益シミュレーション**: 300億円運用時、年間売上11.0億円、年間EBITDA 10.0億円、想定評価額50億円〜80億円を目指す。
- **ポテンシャル評価**: アグレッシブなパラメータチューニングにより、年間平均リターン20.13%、最大ドローダウン2.52%、Sharpe Ratio > 4.0の達成を目指し、評価額を$18M（約27億円）まで引き上げる。

### 5.2 プラットフォーム活用戦略
- **Phase 1（実績構築）**: QuantConnect（Alpha Market/Streams）およびQuantiacsでの公開トラックレコードを構築し、上位ランク獲得を目指す。
- **Phase 2（収益化）**: Alpha Marketの購読料とQuantiacsの運用報酬（パフォーマンス・フィー）によって継続収益を確立する。
- **Version-Qの位置づけ**: 適応コアロジック、学習エンジン、学習済みパラメータは競争優位の源泉として保持し、プラットフォーム評価基準に最適化する。

### 5.3 審査コスト最適化と公的証明力
- **Self-Validated Model**: システムが生成する監査証跡と完全性証明を提示し、「監査準備完了（Audit-Ready）」状態であることを訴求する。
- **公的証明力の二重化**: J-Quants 20年バックテスト（再現性・監査証跡）と、プラットフォーム上の公開トラックレコードを併用し、審査・ランキングで優位性を確立する。

### 5.4 匿名性確保戦略
- **ビジネス上の匿名性**: 公開情報は最小限とし、法務・税務上の代理人を窓口とすることで運用者の身元を保護する。
- **技術上の匿名性**: Silence Protocolの「デッドマンスイッチ」機能をセキュリティ要件として明示し、運用者の安全を担保する。

## 6. 付録

### 6.1 用語集
- **AUM**: Assets Under Management (運用資産額)
- **Sharpe Ratio**: リスク調整後リターンの指標。高いほど良い。
- **Max Drawdown**: ピークからの最大下落率。
- **Survivorship Bias**: 上場廃止銘柄をデータから除外することで発生するバイアス。
- **Almgren-Chriss Model**: 大口注文のマーケットインパクトを最小化するための最適執行モデル。
- **PII**: Personally Identifiable Information (個人特定情報)。
- **Dead Man's Switch**: 定期的な生存信号が途絶えた場合に、自動的にシステムを破壊する機能。
- **Decoy Generation**: 重要ファイルを無害なダミーコードに置き換えることで、不自然なファイル消失を隠蔽する機能。
- **VDR**: Virtual Data Room (仮想データルーム)。審査・監査資料の保管に使用される。
- **FIEA**: Financial Instruments and Exchange Act (金融商品取引法)。

### 6.2 参照ドキュメント
- [README.md](README.md)
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
- [PROJECT_HANDOVER.md](PROJECT_HANDOVER.md)
- [docs/AWS_MIGRATION_GUIDE_JA.md](docs/AWS_MIGRATION_GUIDE_JA.md)
- [docs/BACKTEST_TUNING_GUIDE_JA.md](docs/BACKTEST_TUNING_GUIDE_JA.md)
- [docs/BOOTSTRAP_EXECUTION_PLAN_JA.md](docs/BOOTSTRAP_EXECUTION_PLAN_JA.md)
- [docs/DD_COST_AVOIDANCE_STRATEGY_JA.md](docs/DD_COST_AVOIDANCE_STRATEGY_JA.md)
- [docs/JQUANTS_SETUP_GUIDE_JA.md](docs/JQUANTS_SETUP_GUIDE_JA.md)
- [docs/MA_NEGOTIATION_MATERIALS_JA.md](docs/MA_NEGOTIATION_MATERIALS_JA.md)
- [docs/OPERATOR_SAFETY_GUIDE_JA.md](docs/OPERATOR_SAFETY_GUIDE_JA.md)
- [docs/REVENUE_MODEL_2026.md](docs/REVENUE_MODEL_2026.md)
- [docs/STRATEGIC_ANALYSIS_JA.md](docs/STRATEGIC_ANALYSIS_JA.md)
- [plans/biological_core_design.md](plans/biological_core_design.md)
- [plans/naming_convention.md](plans/naming_convention.md)
- [plans/phase1_execution_plan.md](plans/phase1_execution_plan.md)
- [plans/valuation_report_20260127.md](plans/valuation_report_20260127.md)
- [plans/valuation_simulation_plan.md](plans/valuation_simulation_plan.md)
- [plans/valuation_strategy_2026.md](plans/valuation_strategy_2026.md)
- [output/ALPHA_MODEL_AUDIT_REPORT.md](output/ALPHA_MODEL_AUDIT_REPORT.md)
- [output/VALUATION_SIMULATION_REPORT.md](output/VALUATION_SIMULATION_REPORT.md)
- [output/ma_package/DD_COST_AVOIDANCE_STRATEGY.md](output/ma_package/DD_COST_AVOIDANCE_STRATEGY.md)
- [output/ma_package/NEGOTIATION_MATERIALS.md](output/ma_package/NEGOTIATION_MATERIALS.md)
- [output/ma_package/README.md](output/ma_package/README.md)
- [output/ma_package/VALUATION_SIMULATION_REPORT.md](output/ma_package/VALUATION_SIMULATION_REPORT.md)

---
**作成日**: 2026年2月2日
**バージョン**: 1.0.0
