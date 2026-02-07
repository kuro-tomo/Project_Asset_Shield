"""
Asset Shield V3.2.0 - Quantiacs Submission Format
==================================================

Quantiacs用に変換したバリュー・クオリティ戦略
日本株市場 (東証) 向け

Author: Asset Shield Team
Version: 3.2.0-QUANTIACS
"""

import xarray as xr
import numpy as np
import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qnstats


def load_data(period):
    """データ読み込み - 日本株"""
    # Quantiacsの日本株データを使用
    # 注: Quantiacsで日本株が利用可能か確認が必要
    # 利用不可の場合はFutures/Crypto/US Stocksに変更
    return qndata.stocks.load_ndx_data(min_date="2006-01-01")


def calculate_pbr_score(data):
    """PBRスコア計算 (低PBR = 高スコア)"""
    # Quantiacsのfundamentalデータを使用
    # book_value / price = 1/PBR
    close = data.sel(field="close")

    # 簡易版: 価格の逆数をPBR代理とする
    # 本番ではfundamentalデータを使用
    pbr_proxy = 1 / close.where(close > 0)

    # ランキング (高い = 良い)
    rank = pbr_proxy.rank("asset")
    score = rank / rank.max("asset")

    return score


def calculate_roe_score(data):
    """ROEスコア計算 (高ROE = 高スコア)"""
    close = data.sel(field="close")

    # モメンタム (60日リターン) をROE代理とする
    # 本番ではfundamentalデータを使用
    returns_60d = close / close.shift(time=60) - 1

    # ランキング
    rank = returns_60d.rank("asset")
    score = rank / rank.max("asset")

    return score


def calculate_trend_filter(data):
    """トレンドフィルター (60日MA)"""
    close = data.sel(field="close")

    # 市場平均
    market_avg = close.mean("asset")

    # 60日移動平均
    ma_60 = qnta.sma(market_avg, 60)

    # UP = 1, SIDEWAYS = 0.5, DOWN = 0
    trend = xr.where(market_avg > ma_60, 1.0,
             xr.where(market_avg > ma_60 * 0.95, 0.5, 0.0))

    return trend


def calculate_liquidity_filter(data, threshold_pct=0.2):
    """流動性フィルター (上位20%のみ)"""
    vol = data.sel(field="vol")
    close = data.sel(field="close")

    # 売買代金 = 出来高 × 終値
    turnover = vol * close

    # 60日平均
    avg_turnover = qnta.sma(turnover, 60)

    # 上位20%のみ通過
    threshold = avg_turnover.quantile(1 - threshold_pct, dim="asset")
    liquidity_ok = avg_turnover >= threshold

    return liquidity_ok.astype(float)


def strategy(data):
    """
    Asset Shield V3.2.0 メイン戦略

    ロジック:
    1. PBRスコア (低PBR優先)
    2. ROEスコア (高ROE優先)
    3. 複合スコア = 0.5*PBR + 0.5*ROE
    4. トレンドフィルター適用
    5. 流動性フィルター適用
    6. 上位20銘柄に均等配分
    """

    # スコア計算
    pbr_score = calculate_pbr_score(data)
    roe_score = calculate_roe_score(data)

    # 複合スコア
    composite = 0.5 * pbr_score + 0.5 * roe_score

    # フィルター
    trend = calculate_trend_filter(data)
    liquidity = calculate_liquidity_filter(data)

    # フィルター適用
    filtered_score = composite * liquidity

    # トレンドが弱い時はポジション縮小
    filtered_score = filtered_score * trend

    # 上位20銘柄を選択
    top_n = 20
    threshold = filtered_score.quantile(1 - top_n/filtered_score.sizes["asset"], dim="asset")

    # ウェイト計算 (均等配分)
    weights = xr.where(filtered_score >= threshold, 1.0, 0.0)

    # 正規化 (合計 = 1)
    weights_sum = weights.sum("asset")
    weights = weights / xr.where(weights_sum > 0, weights_sum, 1)

    # ロングオンリー、レバレッジなし
    weights = weights.clip(0, 1)

    return weights


def main():
    """メイン実行"""
    # データ読み込み
    data = load_data(period=365*10)  # 10年分

    # 戦略実行
    weights = strategy(data)

    # 結果検証
    weights = qnout.clean(weights, data, "stocks_nasdaq100")

    # 統計表示
    stats = qnstats.calc_stat(data, weights)
    print(stats.to_pandas())

    # 提出用出力
    qnout.write(weights)

    return weights


if __name__ == "__main__":
    main()
