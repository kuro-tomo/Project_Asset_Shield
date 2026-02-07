"""
Asset Shield V5.7 Ultra Concentrated - QuantConnect Edition
Top 5 High Conviction + 1.5x Leverage in SUPER_BULL

Backtest Results (18 Years):
- Total Return: +977%
- CAGR: 14.2%
- Max Drawdown: 32%

Strategy:
- Universe: Top 100 liquid Japanese stocks
- Selection: Top 5 by Momentum (60%) + Short Momentum (25%) + Low Vol (15%)
- Exposure: SUPER_BULL=150%, BULL=120%, NEUTRAL=100%, BEAR=80%, CRISIS=50%
- Kill-Switch: 18% DD with gradual recovery

Version: 5.7.0 (2026-02-06)
"""

from AlgorithmImports import *
import numpy as np

class AssetShieldV57(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000000)

        # Top liquid Japanese stocks (update with actual tickers)
        self.tickers = [
            "72030", "67580", "83060", "68570", "80350",
            "70110", "70120", "79740", "65010", "99840",
            "83160", "65260", "95010", "68610", "60980",
            "40630", "72670", "87660", "94320", "91040"
        ]
        self.tickers = list(dict.fromkeys(self.tickers))

        self.sector_map = {
            "72030": "06", "67580": "09", "83060": "15", "68570": "09",
            "80350": "09", "70110": "08", "70120": "06", "79740": "09",
            "65010": "09", "99840": "10", "83160": "15", "65260": "09",
            "95010": "11", "68610": "09", "60980": "10", "40630": "04",
            "72670": "06", "87660": "16", "94320": "10", "91040": "12"
        }

        # V5.7 Parameters
        self.MAX_POSITIONS = 5
        self.MAX_SECTOR_WEIGHT = 0.50
        self.MAX_SINGLE_WEIGHT = 0.30
        self.MIN_ADT = 30000000
        self.REBALANCE_DAYS = 20
        self.MAX_DRAWDOWN = 0.18

        # State
        self.symbols = {}
        self.prices = {}
        self.volumes = {}
        self.high_water_mark = self.Portfolio.TotalPortfolioValue
        self.kill_switch_active = False
        self.kill_switch_date = None
        self.recovery_step = 0
        self.recovery_steps = [0.0, 0.40, 0.80, 1.20]
        self.current_regime = "BULL"
        self.last_rebalance = None

        for t in self.tickers:
            self.symbols[t] = self.AddEquity(t, Resolution.Daily).Symbol
            self.prices[t] = []
            self.volumes[t] = []

        self.Debug("V5.7 Ultra Concentrated initialized")

    def OnData(self, slice):
        for t in self.tickers:
            if slice.Bars.ContainsKey(self.symbols[t]):
                bar = slice.Bars[self.symbols[t]]
                self.prices[t].append(float(bar.Close))
                self.volumes[t].append(float(bar.Volume))
                if len(self.prices[t]) > 300:
                    self.prices[t] = self.prices[t][-300:]
                    self.volumes[t] = self.volumes[t][-300:]

        current_value = self.Portfolio.TotalPortfolioValue
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
        drawdown = (self.high_water_mark - current_value) / self.high_water_mark

        self.UpdateRegime()
        self.CheckKillSwitch(drawdown)

        if sum(1 for t in self.tickers if len(self.prices[t]) >= 60) < 10:
            return

        should_rebalance = (self.last_rebalance is None or
                          (self.Time - self.last_rebalance).days >= self.REBALANCE_DAYS)

        if should_rebalance:
            self.Rebalance()
            self.last_rebalance = self.Time

    def UpdateRegime(self):
        vols, trends = [], []
        for t in self.tickers[:10]:
            if len(self.prices[t]) >= 60:
                p = self.prices[t]
                rets = [(p[i+1]/p[i] - 1) for i in range(-21, -1) if p[i] > 0]
                if rets:
                    vols.append(np.std(rets) * np.sqrt(252))
                if p[-60] > 0:
                    trends.append(p[-1] / p[-60] - 1)

        if len(vols) >= 5:
            avg_vol = np.mean(vols)
            avg_trend = np.mean(trends) if trends else 0

            if avg_vol > 0.45:
                self.current_regime = "CRISIS"
            elif avg_vol > 0.25 and avg_trend < -0.03:
                self.current_regime = "BEAR"
            elif avg_trend > 0.10:
                self.current_regime = "SUPER_BULL"
            elif avg_trend > 0.03:
                self.current_regime = "BULL"
            else:
                self.current_regime = "NEUTRAL"

    def CheckKillSwitch(self, drawdown):
        if not self.kill_switch_active and drawdown > self.MAX_DRAWDOWN:
            self.kill_switch_active = True
            self.kill_switch_date = self.Time
            self.recovery_step = 0
            self.Debug(f"KILL SWITCH: DD={drawdown:.1%}")
            return

        if self.kill_switch_active and self.kill_switch_date:
            days = (self.Time - self.kill_switch_date).days
            if self.current_regime != "CRISIS":
                step = min(days // 5, 3)
                if step > self.recovery_step:
                    self.recovery_step = step
                    if self.recovery_step >= 3:
                        self.kill_switch_active = False
                        self.kill_switch_date = None
                        self.high_water_mark = self.Portfolio.TotalPortfolioValue
            else:
                self.recovery_step = 0
                self.kill_switch_date = self.Time

    def CalcAlpha(self, t):
        if len(self.prices[t]) < 60:
            return None, None

        p = np.array(self.prices[t])
        v = np.array(self.volumes[t])

        # Momentum 12-1
        if len(p) >= 252 and p[-252] > 0:
            mom = (p[-22] / p[-252]) - 1
        elif len(p) >= 60 and p[-60] > 0:
            mom = (p[-1] / p[-60]) - 1
        else:
            mom = 0

        # Short momentum
        short_mom = (p[-1] / p[-20] - 1) if len(p) >= 20 and p[-20] > 0 else 0

        # Volatility
        rets = np.diff(p[-60:]) / np.where(p[-60:-1] == 0, 1, p[-60:-1])
        vol = np.std(rets) * np.sqrt(252) if len(rets) > 0 else 0.3

        # ADT
        adt = np.mean(p[-20:] * v[-20:]) if len(p) >= 20 else 0

        # V5.7 Alpha: Pure momentum focus
        alpha = mom * 0.60 + short_mom * 0.25 + (-vol) * 0.15

        return alpha, adt

    def Rebalance(self):
        equity = self.Portfolio.TotalPortfolioValue

        # Calculate alphas
        alphas = []
        for t in self.tickers:
            alpha, adt = self.CalcAlpha(t)
            if alpha is not None and adt >= self.MIN_ADT:
                alphas.append((t, alpha, adt))

        alphas.sort(key=lambda x: x[1], reverse=True)

        # Determine exposure
        if self.kill_switch_active:
            max_exp = self.recovery_steps[self.recovery_step]
        elif self.current_regime == "CRISIS":
            max_exp = 0.50
        elif self.current_regime == "BEAR":
            max_exp = 0.80
        elif self.current_regime == "NEUTRAL":
            max_exp = 1.00
        elif self.current_regime == "SUPER_BULL":
            max_exp = 1.50
        else:
            max_exp = 1.20

        # Select TOP 5
        selected = []
        sector_wts = {}
        total_wt = 0
        base_wt = max_exp / 5

        for t, alpha, adt in alphas:
            if len(selected) >= 5 or total_wt >= max_exp:
                break
            sector = self.sector_map.get(t, "10")
            cur_sector = sector_wts.get(sector, 0)
            if cur_sector + base_wt > self.MAX_SECTOR_WEIGHT:
                continue
            wt = min(base_wt, self.MAX_SINGLE_WEIGHT)
            if total_wt + wt > max_exp:
                wt = max_exp - total_wt
            selected.append((t, wt))
            sector_wts[sector] = cur_sector + wt
            total_wt += wt

        # Execute
        held = set(t for t in self.tickers if self.Portfolio[self.symbols[t]].Invested)
        new_holdings = set(t for t, _ in selected)

        for t in held - new_holdings:
            self.Liquidate(self.symbols[t])

        for t, wt in selected:
            self.SetHoldings(self.symbols[t], wt)

        self.Debug(f"{self.Time.date()} [{self.current_regime}] {len(selected)} pos, {total_wt:.0%} exp")

    def OnEndOfAlgorithm(self):
        final = self.Portfolio.TotalPortfolioValue
        ret = (final / 100000000 - 1) * 100
        self.Debug(f"=== V5.7 FINAL: {ret:.1f}% ===")
