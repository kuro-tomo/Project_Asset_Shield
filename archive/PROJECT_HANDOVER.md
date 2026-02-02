# Project "J-Alpha 300B" Transfer Document

## 1. プロジェクト背景 & 目的 (Context)
- **目的**: 日本株全銘柄を対象とした、AUM（運用資産）300億円規模に耐えうる中長期AI運用システムの構築と収益化。
- **ターゲット**: 
  1. QuantConnect Alpha Market (ライセンス販売 / サブスク収益)
  2. Quantiacs (機関投資家資金割り当て / 成功報酬収益)
- **主要データ**: J-Quants API (東証公式) の過去20年分データ。
- **制約条件**: コードの秘匿性維持（IP保護）、生存バイアスの完全排除、300億円時のマーケットインパクトの最小化。

## 2. 開発・運用リソース (Infrastructure)
- **Hardware**: MacBook Pro M4 Pro (48GB RAM / 1TB SSD)
- **AI Stack (Roo Code Configuration)**:
  - **Plan (Architect)**: Gemini 3 Pro (広大なコンテキスト窓による戦略立案)
  - **Code (Engineer)**: Claude 4.5 Opus (極めて高いコーディング精度とリファクタリング)
  - **Evaluation (Reviewer)**: GPT 5.2 Codex (厳格な論理検証とバックテスト統計の評価)

## 3. 実装パイプライン (Pipeline)

### Step 1: Data Engineering (Local Mac)
- J-QuantsデータをPandas/Polarsで処理し、LEANエンジン互換形式に変換。
- 特徴量エンジニアリング（20年分）を48GB RAMで並行処理し、シグナルをキャッシュ化。

### Step 2: Algorithm Implementation (Hybrid)
- **Core Logic**: Macローカルで秘匿。
- **Interface**: QuantConnect `AlphaModel` フレームワークへの準拠。
- **Connectivity**: GitHub Private Repoを介した、クラウドへのシグナル配信（シグナル・オンリー・デプロイ）。

### Step 3: Global Deployment
- QuantConnectクラウドでのインキュベーション（実力証明）。
- Quantiacsマーケットプレイスへのパラメータ最適化と申請。

## 4. 実行タイムライン (Roadmap)
- **Month 1**: システム移植 & ローカルLEANバックテスト完走 (Capacity $200M検証)。
- **Month 2-3**: QuantConnectでのライブデモ稼働 (フォワードテスト)。実績データの蓄積。
- **Month 4**: Alpha Market & Quantiacs への二重申請。
- **Month 5-6**: 収益化開始、日本法人(合同会社)設立による契約主体の法人化。

## 5. Roo Code への実装指示 (Instructions)

### A. Data Pre-processing (Claude 4.5 Opus 担当)
```python
# J-Quants CSVをQuantConnect LEAN形式へ高速変換する関数
import pandas as pd

def transform_jquants_to_lean(df):
    # 48GB RAMを活かしメモリ効率と速度を両立
    df['Time'] = pd.to_datetime(df['Date'])
    df = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df.set_index('Time').sort_index()
```

### B. Alpha Model Interface (GPT 5.2 Codex 担当)
```python
# QuantConnect Alpha Market審査基準に準拠したInsight生成
def Update(self, algorithm, data):
    insights = []
    for symbol in self.symbols:
        # 300億円のキャパシティを担保する流動性チェック
        if data[symbol].Price * data[symbol].Volume < 3000000000:
            continue
            
        # AI推論(手元のシグナルデータを参照)
        prediction = self.get_ai_signal(symbol)
        if prediction > self.threshold:
            insights.append(Insight.Price(symbol, timedelta(days=30), InsightDirection.Up))
    return insights
```

########################################################################

MECE確認事項 (Checklist)
[ ] Mutually Exclusive:
ローカル開発(Mac)とクラウド運用(QuantConnect)の責務は分離されているか？
法人化(事務)と開発(技術)のスケジュールに重複はないか？
[ ] Collectively Exhaustive:
税務(W-8BEN)、知財(GitHub Private)、キャパシティ(Market Impact)の全リスクを網羅しているか？
########################################################################
