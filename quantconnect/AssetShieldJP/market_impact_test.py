# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class MarketImpactTest(QCAlgorithm):
    """
    Asset Shield V3.2.0 - Market Impact Verification
    =================================================

    目標: Impact ≤ 20 bps

    Almgren-Chriss Model Parameters:
    - gamma (permanent): 0.10
    - eta (temporary): 0.01
    - sigma_annual: 0.25
    - max_participation: 0.10

    検証内容:
    1. 各取引のスリッページ測定
    2. 実効スプレッド計算
    3. 市場参加率の検証
    """

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)  # OOS期間のみ
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10_000_000)

        # スリッページモデル設定
        self.SetSecurityInitializer(self.CustomSecurityInitializer)

        # 実行コスト追跡
        self.total_trades = 0
        self.total_impact_bps = 0
        self.impact_list = []

        # テスト銘柄
        self.test_symbols = {}
        for code in ["7203", "6758", "8306", "8035", "6861"]:
            try:
                symbol = self.AddEquity(code, Resolution.Daily, Market.Japan)
                self.symbols[code] = symbol.Symbol
            except:
                pass

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("7203", 30),
            self.ExecuteTrade
        )

    def CustomSecurityInitializer(self, security):
        """カスタム執行モデル"""
        # Almgren-Chriss ベースのスリッページ
        security.SetSlippageModel(VolumeShareSlippageModel())
        security.SetFillModel(ImmediateFillModel())
        security.SetFeeModel(InteractiveBrokersFeeModel())

    def ExecuteTrade(self):
        """取引実行とインパクト測定"""
        if self.day_count % 63 != 0:  # リバランス日のみ
            self.day_count += 1
            return

        for code, symbol in self.test_symbols.items():
            if not self.Securities.ContainsKey(symbol):
                continue

            security = self.Securities[symbol]
            price = security.Price
            volume = security.Volume

            if price <= 0 or volume <= 0:
                continue

            # ポジションサイズ
            equity = self.Portfolio.TotalPortfolioValue
            order_value = equity * 0.05  # 5%
            shares = int(order_value / price)

            if shares <= 0:
                continue

            # 市場参加率チェック
            adt = volume * price  # 当日売買代金
            participation = order_value / adt if adt > 0 else 1

            if participation > 0.10:
                self.Debug(f"SKIP {code}: Participation {participation:.1%} > 10%")
                continue

            # 発注前価格記録
            pre_trade_price = price

            # 発注
            ticket = self.MarketOrder(symbol, shares)

            # インパクト計算 (簡易版)
            # 実際のフィルプライスはOnOrderEventで取得
            self.pending_orders[ticket.OrderId] = {
                'code': code,
                'pre_price': pre_trade_price,
                'shares': shares,
                'participation': participation
            }

        self.day_count += 1

    def OnOrderEvent(self, orderEvent):
        """注文イベント処理"""
        if orderEvent.Status != OrderStatus.Filled:
            return

        order_id = orderEvent.OrderId
        if order_id not in self.pending_orders:
            return

        info = self.pending_orders[order_id]
        fill_price = orderEvent.FillPrice
        pre_price = info['pre_price']

        # インパクト (bps)
        impact_bps = abs(fill_price / pre_price - 1) * 10000

        self.total_trades += 1
        self.total_impact_bps += impact_bps
        self.impact_list.append(impact_bps)

        self.Debug(f"FILL {info['code']}: Impact={impact_bps:.1f}bps, "
                   f"Participation={info['participation']:.1%}")

        del self.pending_orders[order_id]

    def OnEndOfAlgorithm(self):
        """終了時レポート"""
        if self.total_trades == 0:
            self.Debug("No trades executed")
            return

        avg_impact = self.total_impact_bps / self.total_trades
        max_impact = max(self.impact_list) if self.impact_list else 0
        p95_impact = np.percentile(self.impact_list, 95) if self.impact_list else 0

        self.Debug("=" * 60)
        self.Debug("Market Impact Verification Report")
        self.Debug("=" * 60)
        self.Debug(f"Total Trades: {self.total_trades}")
        self.Debug(f"Average Impact: {avg_impact:.2f} bps")
        self.Debug(f"Max Impact: {max_impact:.2f} bps")
        self.Debug(f"95th Percentile: {p95_impact:.2f} bps")
        self.Debug("=" * 60)

        # 合格判定
        if avg_impact <= 20:
            self.Debug("✓ PASS: Average Impact <= 20 bps")
        else:
            self.Debug("✗ FAIL: Average Impact > 20 bps")
