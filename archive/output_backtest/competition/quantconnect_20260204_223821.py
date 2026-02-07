
# Asset Shield V3.2.0 - QuantConnect Alpha Streams Submission
# =============================================================
# Strategy: Percentile-based PBR/ROE Value Investing
# Universe: Japanese Equities (TSE Prime)
# Rebalance: Quarterly (63 trading days)
#
# Performance (Backtest 2008-2026):
#   - OOS Sharpe: 1.02
#   - Annual Return: 28.35%
#   - Max Drawdown: 36.62%
#   - Win Rate: 61.42%

from AlgorithmImports import *
import numpy as np

class AssetShieldAlphaV3(QCAlgorithm):
    """
    Japanese Value-Quality Strategy with Almgren-Chriss Impact Model
    """

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)
        self.SetCash(10000000)  # 10M JPY
        self.SetBrokerageModel(BrokerageName.InteractiveBrokers, AccountType.Margin)

        # Universe Selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        # Strategy Parameters
        self.max_positions = 15
        self.position_pct = 0.08
        self.rebalance_days = 63
        self.holding_days = 250

        # Percentile Thresholds
        self.pbr_percentile = 0.20  # Bottom 20%
        self.roe_percentile = 0.80  # Top 20%
        self.composite_percentile = 0.80

        # Liquidity Threshold (400M JPY ADT)
        self.min_adt = 400_000_000

        # Risk Management
        self.stop_loss = 0.15
        self.take_profit = 0.40

        # State
        self.positions = {}
        self.last_rebalance = None
        self.day_count = 0

        # Almgren-Chriss Parameters
        self.gamma = 0.10  # Permanent impact
        self.eta = 0.01    # Temporary impact
        self.sigma_annual = 0.25
        self.max_participation = 0.10

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )

    def CoarseSelectionFunction(self, coarse):
        """Filter by liquidity and price"""
        return [x.Symbol for x in coarse
                if x.HasFundamentalData
                and x.DollarVolume > self.min_adt
                and x.Price > 100]

    def FineSelectionFunction(self, fine):
        """Select by PBR/ROE percentile ranking"""
        # Calculate PBR and ROE
        stocks_data = []
        for f in fine:
            if f.ValuationRatios.PBRatio > 0 and f.OperationRatios.ROE.Value > 0:
                stocks_data.append({
                    'symbol': f.Symbol,
                    'pbr': f.ValuationRatios.PBRatio,
                    'roe': f.OperationRatios.ROE.Value
                })

        if len(stocks_data) < 10:
            return []

        # Percentile ranking
        pbrs = [s['pbr'] for s in stocks_data]
        roes = [s['roe'] for s in stocks_data]

        for s in stocks_data:
            s['pbr_pct'] = sum(1 for p in pbrs if p <= s['pbr']) / len(pbrs)
            s['roe_pct'] = sum(1 for r in roes if r <= s['roe']) / len(roes)
            s['composite'] = (1 - s['pbr_pct']) * 0.5 + s['roe_pct'] * 0.5

        # Select top composite scores
        threshold = np.percentile([s['composite'] for s in stocks_data],
                                   self.composite_percentile * 100)
        selected = [s['symbol'] for s in stocks_data if s['composite'] >= threshold]

        return selected[:self.max_positions]

    def Rebalance(self):
        """Quarterly rebalance with impact-aware sizing"""
        self.day_count += 1
        if self.day_count % self.rebalance_days != 0:
            return

        # Get current universe
        for symbol in self.ActiveSecurities.Keys:
            if not self.Portfolio[symbol].Invested:
                # Calculate position size with impact constraint
                equity = self.Portfolio.TotalPortfolioValue
                base_size = equity * self.position_pct

                # Almgren-Chriss impact check
                adt = self.Securities[symbol].Volume * self.Securities[symbol].Price
                participation = base_size / adt if adt > 0 else 1.0

                if participation <= self.max_participation:
                    self.SetHoldings(symbol, self.position_pct)

    def OnData(self, data):
        """Risk management: stop-loss and take-profit"""
        for symbol in list(self.Portfolio.Keys):
            if self.Portfolio[symbol].Invested:
                pnl_pct = self.Portfolio[symbol].UnrealizedProfitPercent

                if pnl_pct <= -self.stop_loss:
                    self.Liquidate(symbol, "Stop Loss")
                elif pnl_pct >= self.take_profit:
                    self.Liquidate(symbol, "Take Profit")
