# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion

class AssetShieldNikkeiFutures(QCAlgorithm):
    """
    Asset Shield V3.2.0 - Nikkei 225 Futures
    =========================================

    Purpose: Validate execution engine and risk modules on QuantConnect cloud
    Data: Standard QuantConnect Nikkei 225 Futures (CME)

    Strategy Logic:
    - Trend Following (60-day MA)
    - Volatility-adjusted position sizing
    - Drawdown circuit breaker (35%)
    - Rebalance every 63 days
    """

    def Initialize(self):
        # === Period Settings ===
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2026, 1, 31)
        self.SetCash(10_000_000)  # 10M JPY equivalent

        # === Add Nikkei 225 Futures ===
        self.nikkei = self.AddFuture(
            Futures.Indices.Nikkei225,
            Resolution.Daily,
            dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
            dataMappingMode=DataMappingMode.LastTradingDay,
            contractDepthOffset=0
        )
        self.nikkei.SetFilter(0, 182)  # 6 months

        self.Debug(f"Added Nikkei 225 Futures: {self.nikkei.Symbol}")

        # === Strategy Parameters ===
        self.trend_period = 60
        self.rebalance_days = 63
        self.max_position = 0.80
        self.max_dd = 0.35
        self.stop_loss = 0.12
        self.take_profit = 0.35

        # === State Variables ===
        self.price_history = deque(maxlen=self.trend_period)
        self.peak_equity = self.Portfolio.TotalPortfolioValue
        self.circuit_breaker = False
        self.day_count = 0
        self.current_contract = None
        self.entry_price = 0
        self.entry_date = None

        # === Walk-Forward Tracking ===
        self.training_end = self.Time.replace(year=2018, month=12, day=31)
        self.validation_end = self.Time.replace(year=2022, month=12, day=31)
        self.phase_returns = {"training": [], "validation": [], "oos": []}
        self.prev_equity = self.Portfolio.TotalPortfolioValue

        # === Schedule Daily Check ===
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(10, 0),
            self.DailyProcess
        )

        self.Debug("Initialization complete")

    def DailyProcess(self):
        """Daily processing - trend check and rebalancing"""
        self.day_count += 1
        equity = self.Portfolio.TotalPortfolioValue

        # Track phase returns
        if self.prev_equity > 0:
            daily_ret = (equity / self.prev_equity) - 1
            if self.Time <= self.training_end:
                self.phase_returns["training"].append(daily_ret)
            elif self.Time <= self.validation_end:
                self.phase_returns["validation"].append(daily_ret)
            else:
                self.phase_returns["oos"].append(daily_ret)
        self.prev_equity = equity

        # Drawdown calculation
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Circuit breaker logic
        if dd > self.max_dd:
            self.circuit_breaker = True
            if self.current_contract and self.Portfolio[self.current_contract].Invested:
                self.Liquidate(self.current_contract, "CIRCUIT_BREAKER")
                self.Debug(f"[{self.Time.date()}] Circuit breaker triggered, DD: {dd:.1%}")
        elif dd < 0.20:
            self.circuit_breaker = False

    def OnData(self, data):
        """Process incoming data"""
        # Get futures chain
        for chain in data.FutureChains:
            if chain.Key != self.nikkei.Symbol:
                continue

            contracts = [c for c in chain.Value]
            if len(contracts) == 0:
                continue

            # Select front month with highest open interest
            contracts = sorted(contracts, key=lambda x: x.OpenInterest, reverse=True)
            front = contracts[0]
            price = front.LastPrice

            if price <= 0:
                continue

            # Update price history
            self.price_history.append(price)

            if len(self.price_history) < self.trend_period:
                return

            # Calculate trend
            ma = np.mean(list(self.price_history))

            if price > ma:
                trend = "UP"
            elif price > ma * 0.95:
                trend = "SIDEWAYS"
            else:
                trend = "DOWN"

            # Check for exit conditions on existing position
            if self.current_contract and self.Portfolio[self.current_contract].Invested:
                self._check_exit(front, price)

            # Rebalance check
            if self.day_count % self.rebalance_days != 0:
                return

            equity = self.Portfolio.TotalPortfolioValue
            dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

            self.Debug(f"[{self.Time.date()}] Price: {price:.0f} | MA: {ma:.0f} | "
                      f"Trend: {trend} | DD: {dd:.1%} | Equity: ${equity:,.0f}")

            # Position management
            self._manage_position(front, price, trend)

    def _check_exit(self, contract, current_price):
        """Check exit conditions"""
        if self.entry_price <= 0:
            return

        pnl_pct = (current_price / self.entry_price) - 1
        holding_days = (self.Time - self.entry_date).days if self.entry_date else 0

        should_exit = False
        reason = ""

        if pnl_pct <= -self.stop_loss:
            should_exit = True
            reason = "STOP_LOSS"
        elif pnl_pct >= self.take_profit:
            should_exit = True
            reason = "TAKE_PROFIT"
        elif holding_days >= 250:
            should_exit = True
            reason = "MAX_HOLD"

        if should_exit:
            self.Liquidate(self.current_contract, reason)
            self.Debug(f"EXIT: {reason}, P&L: {pnl_pct:.1%}, Days: {holding_days}")
            self.current_contract = None
            self.entry_price = 0
            self.entry_date = None

    def _manage_position(self, contract, price, trend):
        """Manage position based on trend"""
        if self.circuit_breaker:
            return

        current_invested = self.Portfolio[contract.Symbol].Invested if contract else False

        if trend == "UP":
            # Full position
            if not current_invested:
                self.SetHoldings(contract.Symbol, self.max_position)
                self.current_contract = contract.Symbol
                self.entry_price = price
                self.entry_date = self.Time
                self.Debug(f"ENTRY: Long at {price:.0f}")

        elif trend == "SIDEWAYS":
            # Half position
            target = self.max_position * 0.5
            if current_invested:
                current_pct = self.Portfolio[contract.Symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
                if abs(current_pct - target) > 0.1:
                    self.SetHoldings(contract.Symbol, target)
            else:
                self.SetHoldings(contract.Symbol, target)
                self.current_contract = contract.Symbol
                self.entry_price = price
                self.entry_date = self.Time

        else:  # DOWN
            # Exit position
            if current_invested:
                self.Liquidate(contract.Symbol, "TREND_DOWN")
                self.Debug(f"EXIT: Trend DOWN at {price:.0f}")
                self.current_contract = None
                self.entry_price = 0
                self.entry_date = None

    def OnEndOfAlgorithm(self):
        """Final report"""
        equity = self.Portfolio.TotalPortfolioValue
        initial = 10_000_000
        total_return = (equity / initial) - 1

        # Calculate Sharpe ratios
        def calc_sharpe(returns):
            if len(returns) < 20:
                return 0.0
            returns = np.array(returns)
            if np.std(returns) == 0:
                return 0.0
            annual_ret = np.mean(returns) * 252
            annual_vol = np.std(returns) * np.sqrt(252)
            return annual_ret / annual_vol if annual_vol > 0 else 0.0

        training_sharpe = calc_sharpe(self.phase_returns["training"])
        validation_sharpe = calc_sharpe(self.phase_returns["validation"])
        oos_sharpe = calc_sharpe(self.phase_returns["oos"])

        # Calculate max drawdown
        def calc_max_dd(returns):
            if len(returns) == 0:
                return 0.0
            equity_curve = np.cumprod(1 + np.array(returns))
            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / peak
            return np.max(dd)

        max_dd = calc_max_dd(
            self.phase_returns["training"] +
            self.phase_returns["validation"] +
            self.phase_returns["oos"]
        )

        self.Debug("=" * 70)
        self.Debug("ASSET SHIELD V3.2.0 - NIKKEI 225 FUTURES - FINAL REPORT")
        self.Debug("=" * 70)
        self.Debug(f"Total Return:      {total_return:.2%}")
        self.Debug(f"Final Equity:      ${equity:,.0f}")
        self.Debug(f"Max Drawdown:      {max_dd:.2%}")
        self.Debug("-" * 70)
        self.Debug(f"Training Sharpe:   {training_sharpe:.2f} ({len(self.phase_returns['training'])} days)")
        self.Debug(f"Validation Sharpe: {validation_sharpe:.2f} ({len(self.phase_returns['validation'])} days)")
        self.Debug(f"OOS Sharpe:        {oos_sharpe:.2f} ({len(self.phase_returns['oos'])} days)")
        self.Debug("=" * 70)

        # Validation check
        if oos_sharpe >= 0.7:
            self.Debug("✓ VALIDATION PASSED: OOS Sharpe >= 0.7")
        else:
            self.Debug("✗ VALIDATION FAILED: OOS Sharpe < 0.7")

        if max_dd <= 0.35:
            self.Debug("✓ RISK CHECK PASSED: Max DD <= 35%")
        else:
            self.Debug("✗ RISK CHECK FAILED: Max DD > 35%")
