# Asset Shield V2 売却用バリュエーション向上シミュレーション計画

**作成日:** 2026年1月28日
**目的:** パラメータチューニングにより、バックテスト上の年率リターンを10%〜15%に引き上げ、事業評価額（Valuation）を最大化する。

## 1. 現状の課題 (As-Is)
*   **資金稼働率 (Exposure):** 平均12%と非常に低い。
*   **年間リターン:** 2.7%〜3.1% (リスクフリーレート + α程度)。
*   **評価:** 安全性は極めて高いが、収益性が低く見え、買収意欲を刺激しにくい。

## 2. 目標設定 (To-Be)
売却交渉（M&A）において提示するための「ポテンシャル・モデル」を構築する。
*   **Target Annual Return:** **12.0% 以上**
*   **Target Sharpe Ratio:** **1.2 以上** (現状維持)
*   **Target Exposure:** **50% - 80%**

## 3. パラメータチューニング方針
「安全性重視（保守的）」から「収益性重視（積極的）」へシフトする。

### 3.1 Adaptive Core (`src/shield/adaptive_core.py`)
モデルのリスク許容度とポジションスケーリングを大幅に引き上げる。

| パラメータ | 現状値 | **変更値 (Aggressive)** | 理由 |
|:---|:---:|:---:|:---|
| `vol_target` | 0.15 (15%) | **0.25 (25%)** | 目標ボラティリティを引き上げ、ポジション縮小を防ぐ |
| `risk_multiplier` (NORMAL) | 1.0 | **1.5** | 平時のリスクテイク量を50%増やす |
| `position_scale` (NORMAL) | 1.0 | **1.2** | 基礎ポジションサイズを20%増やす |
| `entry_threshold` | 0.65 | **0.55** | エントリー基準を緩和し、取引機会を増やす |
| `stop_loss_pct` | 0.02 (2%) | **0.05 (5%)** | 浅い損切りを避け、トレンドフォローの勝率を上げる |

### 3.2 Strategy Config (`scripts/run_jquants_backtest.py` 内包ロジック)
資金管理ルールを変更し、キャッシュ余剰を減らす。

| パラメータ | 現状値 | **変更値 (Aggressive)** | 理由 |
|:---|:---:|:---:|:---|
| `position_size_pct` | 0.15 | **0.20** | 1銘柄あたり20%配分。5銘柄で最大100%のエクスポージャーを実現 |
| `take_profit` | 0.08 | **0.15** | 利確幅を広げ、大きなトレンドによる利益を最大化する |
| `max_holding_days` | 30 | **60** | 中期トレンドまでフォローできるように期間を延長 |

## 4. 実装・実行手順 (Code Modeへの指示)

1.  **専用スクリプト作成:** `scripts/run_valuation_simulation.py` を作成する。
    *   既存の `run_jquants_backtest.py` をベースにする。
    *   `BrainBacktestStrategy` を継承または修正し、上記のパラメータを外部から注入・上書きできるようにする。
    *   `AdaptiveCore` に一時的にパラメータを強制上書きする機能を持たせる（またはサブクラス化する）。

2.  **シミュレーション実行:**
    *   対象期間: 過去5年 (Phase 3 & 4: 2021-2026) を重点的に行う（直近の市場環境での証明が重要）。
    *   対象銘柄: 代表的なボラティリティのある銘柄群（例: 9984 ソフトバンクG, 6758 ソニー, 7203 トヨタ）。

3.  **結果出力:**
    *   `output/valuation_simulation_result.json` に結果を保存。
    *   特に「Annual Return」「Max Drawdown」「Sharpe Ratio」を記録する。

## 5. 評価額算出ロジック (Architect担当)
シミュレーション結果に基づき、以下の式で再評価を行う。

$$
\text{Valuation} = (\text{想定AUM} \times \text{Simulated Return} \times \text{Performance Fee \%}) \times \text{PER Multiple}
$$

*   想定AUM: $30M (約45億円)
*   Performance Fee: 20%
*   PER: 15倍
