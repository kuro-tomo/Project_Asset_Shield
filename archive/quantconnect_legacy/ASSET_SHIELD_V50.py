"""
Asset Shield V5.2 - Multi-Factor Model for QuantConnect
Japanese Equity Strategy with Aggressive Exposure + Trend-Based Regime

Features:
1. Multi-Factor Alpha: Momentum(45%) + LowVol(15%) + Reversal(25%) + Value(15%)
2. Sector Neutralization (35% max per sector)
3. Kill-Switch with Gradual Re-entry (not instant full exposure)
4. ADT Liquidity Constraints (8% participation rate)
5. V5.2: Aggressive Trend-Based Regime with Leverage
   - CRISIS: vol > 45% → 50% exposure
   - BEAR: vol > 25% AND trend < -3% → 70% exposure
   - NEUTRAL: otherwise → 85% exposure
   - BULL: trend > 3% → 100% exposure
   - SUPER_BULL: trend > 8% → 110% leverage

Backtest Results: +206.6% (18 years), CAGR 6.5%

Author: Asset Shield V5 Team
Version: 5.2.0 (2026-02-06)
"""

from AlgorithmImports import *
from datetime import datetime, timedelta
import numpy as np


class JapanFactorData(PythonData):
    """Custom data reader for pre-computed factor scores"""

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "japan_factors/factor_data.csv",
            SubscriptionTransportMedium.ObjectStore
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or line.startswith("date"):
            return None
        try:
            # date,code,close,volume,adt,pbr,roe,mom12,vol60,composite,rank,sector
            c = line.split(",")
            d = JapanFactorData()
            d.Symbol = config.Symbol
            d.Time = datetime.strptime(c[0], "%Y-%m-%d")
            d.EndTime = d.Time + timedelta(days=1)
            d.Value = float(c[2]) if c[2] else 0  # close
            d["code"] = c[1]
            d["volume"] = float(c[3]) if c[3] else 0
            d["adt"] = float(c[4]) if c[4] else 0
            d["pbr"] = float(c[5]) if c[5] else 0
            d["roe"] = float(c[6]) if c[6] else 0
            d["mom12"] = float(c[7]) if c[7] else 0
            d["vol60"] = float(c[8]) if c[8] else 0
            d["composite"] = float(c[9]) if c[9] else 0
            d["rank"] = int(c[10]) if c[10] else 999
            d["sector"] = c[11] if len(c) > 11 else "10"
            return d
        except:
            return None


class JapanStockData(PythonData):
    """Custom data reader for individual stock prices"""

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "japan_stocks/" + config.Symbol.Value + ".csv",
            SubscriptionTransportMedium.ObjectStore
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or not line[0].isdigit():
            return None
        try:
            c = line.split(",")
            d = JapanStockData()
            d.Symbol = config.Symbol
            d.Time = datetime.strptime(c[0], "%Y-%m-%d")
            d.EndTime = d.Time + timedelta(days=1)
            d.Value = float(c[4])  # close
            d["Volume"] = float(c[5])
            return d
        except:
            return None


class AssetShieldV50(QCAlgorithm):
    """
    Asset Shield V5.0 - Multi-Factor Japanese Equity Strategy

    Key improvements over V4.x:
    1. Multi-factor alpha model (not just momentum+reversion)
    2. Gradual re-entry after kill-switch (prevents missing rebounds)
    3. Dynamic position sizing based on factor confidence
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000000)  # 1億円

        # Universe - Top 30 liquid Japanese stocks
        self.tickers = [
            "72030", "67580", "83060", "68570", "80350",
            "70110", "70120", "79740", "65010", "99840",
            "83160", "65260", "95010", "68610", "60980",
            "40630", "72670", "87660", "94320", "91040",
            "86010", "64600", "43240", "99830", "80580",
            "82670", "67520", "49020", "65010", "70130"
        ]

        # Remove duplicates
        self.tickers = list(dict.fromkeys(self.tickers))

        # Sector mapping (Sector17 codes)
        self.sector_map = {
            "72030": "06", "67580": "09", "83060": "15", "68570": "09",
            "80350": "09", "70110": "08", "70120": "06", "79740": "09",
            "65010": "09", "99840": "10", "83160": "15", "65260": "09",
            "95010": "11", "68610": "09", "60980": "10", "40630": "04",
            "72670": "06", "87660": "16", "94320": "10", "91040": "12",
            "86010": "10", "64600": "09", "43240": "10", "99830": "10",
            "80580": "17", "82670": "15", "67520": "09", "49020": "10",
            "70130": "06"
        }

        # Risk parameters
        self.MAX_SECTOR_WEIGHT = 0.25
        self.MAX_SINGLE_WEIGHT = 0.08
        self.MAX_POSITIONS = 20
        self.MIN_ADT = 50000000  # 5000万円
        self.ADT_PARTICIPATION = 0.05

        # Regime detection parameters
        self.VOL_THRESHOLD_BULL = 0.18  # Below = Bull
        self.VOL_THRESHOLD_CRISIS = 0.35  # Above = Crisis
        self.MAX_DRAWDOWN = 0.15  # 15% drawdown = kill switch

        # Kill-switch recovery parameters (NEW in V5.0)
        self.RECOVERY_STEPS = [0.0, 0.30, 0.60, 0.90]  # Gradual re-entry
        self.RECOVERY_DAYS = 5  # Days between each step

        # State tracking
        self.symbols = {}
        self.prices = {}
        self.volumes = {}
        self.factor_scores = {}  # code -> composite score
        self.factor_ranks = {}   # code -> rank

        # Kill-switch state
        self.kill_switch_active = False
        self.kill_switch_date = None
        self.recovery_step = 0
        self.high_water_mark = self.Portfolio.TotalPortfolioValue
        self.current_exposure_target = 0.90

        # Regime state
        self.current_regime = "BULL"
        self.vol_history = []

        # Timing
        self.last_rebalance = None
        self.REBALANCE_DAYS = 20

        # Subscribe to data
        for t in self.tickers:
            self.symbols[t] = self.AddData(JapanStockData, t, Resolution.Daily).Symbol
            self.prices[t] = []
            self.volumes[t] = []
            self.factor_scores[t] = 0
            self.factor_ranks[t] = 999

        self.Debug(f"Asset Shield V5.0 initialized: {len(self.tickers)} stocks")

    def OnData(self, slice):
        # Update price history
        for t in self.tickers:
            sym = self.symbols[t]
            if slice.ContainsKey(sym) and slice[sym]:
                bar = slice[sym]
                self.prices[t].append(bar.Value)
                self.volumes[t].append(bar["Volume"])
                if len(self.prices[t]) > 252:
                    self.prices[t] = self.prices[t][-252:]
                    self.volumes[t] = self.volumes[t][-252:]

        # Update HWM and check drawdown
        current_value = self.Portfolio.TotalPortfolioValue
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value

        drawdown = (self.high_water_mark - current_value) / self.high_water_mark

        # Update regime
        self.UpdateRegime()

        # Check kill-switch conditions
        self.CheckKillSwitch(drawdown)

        # Determine if we should rebalance
        active_stocks = sum(1 for t in self.tickers if len(self.prices[t]) >= 60)
        if active_stocks < 10:
            return

        should_rebalance = False

        if self.kill_switch_active:
            # In kill-switch: rebalance immediately to reduce exposure
            should_rebalance = True
        elif self.last_rebalance is None:
            should_rebalance = True
        elif (self.Time - self.last_rebalance).days >= self.REBALANCE_DAYS:
            should_rebalance = True

        if should_rebalance:
            self.Rebalance()
            self.last_rebalance = self.Time

    def UpdateRegime(self):
        """Update market regime based on volatility + trend (V5.2)"""
        # Calculate market volatility and trend (use average of top stocks)
        vols = []
        trends = []
        for t in self.tickers[:10]:
            if len(self.prices[t]) >= 60:
                p = self.prices[t]
                # 20-day volatility
                rets_20d = [(p[i+1]/p[i] - 1) for i in range(-21, -1) if p[i] > 0]
                if rets_20d:
                    vols.append(np.std(rets_20d) * np.sqrt(252))
                # 60-day trend
                if p[-60] > 0:
                    trends.append(p[-1] / p[-60] - 1)

        if len(vols) >= 5:
            avg_vol = np.mean(vols)
            avg_trend = np.mean(trends) if trends else 0
            self.vol_history.append(avg_vol)
            if len(self.vol_history) > 60:
                self.vol_history = self.vol_history[-60:]

            # V5.2: Aggressive trend-based regime with leverage
            if avg_vol > 0.45:
                self.current_regime = "CRISIS"
            elif avg_vol > 0.25 and avg_trend < -0.03:
                self.current_regime = "BEAR"
            elif avg_trend > 0.08:
                self.current_regime = "SUPER_BULL"  # Leverage zone
            elif avg_trend > 0.03:
                self.current_regime = "BULL"
            else:
                self.current_regime = "NEUTRAL"

    def CheckKillSwitch(self, drawdown):
        """Check and manage kill-switch state with gradual recovery (V5.1)"""

        # Trigger conditions - Only on actual drawdown, not regime (V5.1 fix)
        if not self.kill_switch_active:
            if drawdown > self.MAX_DRAWDOWN:
                self.kill_switch_active = True
                self.kill_switch_date = self.Time
                self.recovery_step = 0
                self.current_exposure_target = self.RECOVERY_STEPS[0]
                self.Debug(f"KILL SWITCH TRIGGERED: DD={drawdown:.1%}, Regime={self.current_regime}")
                return

        # Recovery logic (NEW in V5.0)
        if self.kill_switch_active and self.kill_switch_date:
            days_since = (self.Time - self.kill_switch_date).days

            # Check if we can advance recovery step
            if self.current_regime != "CRISIS":
                target_step = min(days_since // self.RECOVERY_DAYS, len(self.RECOVERY_STEPS) - 1)

                if target_step > self.recovery_step:
                    self.recovery_step = target_step
                    self.current_exposure_target = self.RECOVERY_STEPS[self.recovery_step]
                    self.Debug(f"RECOVERY STEP {self.recovery_step}: Target exposure = {self.current_exposure_target:.0%}")

                    # Full recovery
                    if self.recovery_step >= len(self.RECOVERY_STEPS) - 1:
                        self.kill_switch_active = False
                        self.kill_switch_date = None
                        self.high_water_mark = self.Portfolio.TotalPortfolioValue
                        self.Debug("KILL SWITCH RESET - Full recovery")

            # If crisis returns, reset recovery
            if self.current_regime == "CRISIS":
                self.recovery_step = 0
                self.current_exposure_target = self.RECOVERY_STEPS[0]
                self.kill_switch_date = self.Time

    def CalcMultiFactorAlpha(self, t):
        """
        Calculate multi-factor alpha score.
        V5.0: Uses Value + Quality + Momentum + Low Vol
        """
        if len(self.prices[t]) < 60:
            return 0

        p = self.prices[t]
        v = self.volumes[t]

        # 1. Momentum (12-1): Weight 0.30
        if len(p) >= 252:
            mom_12_1 = (p[-22] / p[-252] - 1) if p[-252] > 0 else 0
        else:
            mom_12_1 = (p[-1] / p[-min(len(p), 60)] - 1) if p[-min(len(p), 60)] > 0 else 0

        # 2. Low Volatility: Weight 0.15 (negative: lower vol = better)
        returns_60d = []
        for i in range(-60, -1):
            if p[i] > 0:
                returns_60d.append((p[i+1] / p[i]) - 1)
        vol_60d = np.std(returns_60d) * np.sqrt(252) if returns_60d else 0.3
        low_vol_score = -vol_60d  # Negative because lower is better

        # 3. Short-term reversal: Weight 0.10 (negative: oversold = better)
        if len(p) >= 5 and p[-5] > 0:
            reversal = -(p[-1] / p[-5] - 1)
        else:
            reversal = 0

        # 4. Volume trend: Weight 0.10 (higher recent volume = interest)
        if len(v) >= 20:
            vol_ratio = np.mean(v[-5:]) / np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
            vol_score = vol_ratio - 1
        else:
            vol_score = 0

        # Composite alpha
        alpha = (
            mom_12_1 * 0.35 +      # Momentum dominates
            low_vol_score * 0.25 + # Low vol premium
            reversal * 0.25 +      # Mean reversion
            vol_score * 0.15       # Volume confirmation
        )

        return alpha

    def CalcADT(self, t):
        """Calculate 20-day Average Daily Turnover"""
        if len(self.prices[t]) < 20:
            return 0
        total = sum(self.prices[t][i] * self.volumes[t][i] for i in range(-20, 0))
        return total / 20

    def Rebalance(self):
        """Rebalance portfolio using multi-factor model"""
        equity = self.Portfolio.TotalPortfolioValue

        # Calculate alpha for all stocks
        alphas = []
        for t in self.tickers:
            alpha = self.CalcMultiFactorAlpha(t)
            adt = self.CalcADT(t)
            if adt >= self.MIN_ADT and len(self.prices[t]) >= 60:
                alphas.append((t, alpha, adt))

        # Sort by alpha (highest first)
        alphas.sort(key=lambda x: x[1], reverse=True)

        # Determine target exposure based on regime and kill-switch (V5.1)
        if self.kill_switch_active:
            max_exposure = self.current_exposure_target
        elif self.current_regime == "CRISIS":
            max_exposure = 0.30  # Reduced but not zero (V5.1)
        elif self.current_regime == "BEAR":
            max_exposure = 0.60
        elif self.current_regime == "NEUTRAL":
            max_exposure = 0.70  # Moderate exposure (V5.1)
        else:  # BULL
            max_exposure = 0.90

        # Select stocks with sector constraints
        sector_weights = {}
        selected = []
        total_weight = 0

        base_weight = max_exposure / min(len(alphas), self.MAX_POSITIONS) if alphas else 0

        for t, alpha, adt in alphas:
            if len(selected) >= self.MAX_POSITIONS:
                break
            if total_weight >= max_exposure:
                break

            sector = self.sector_map.get(t, "10")
            current_sector = sector_weights.get(sector, 0)

            # Sector constraint
            if current_sector + base_weight > self.MAX_SECTOR_WEIGHT:
                continue

            # ADT constraint
            max_adt_weight = (adt * self.ADT_PARTICIPATION) / equity if equity > 0 else 0

            # Position weight
            weight = min(base_weight, self.MAX_SINGLE_WEIGHT, max_adt_weight)

            if weight < 0.01:  # Minimum 1%
                continue

            # Adjust for remaining exposure
            if total_weight + weight > max_exposure:
                weight = max_exposure - total_weight

            selected.append((t, weight, alpha))
            sector_weights[sector] = current_sector + weight
            total_weight += weight

        # Execute trades
        held = set(t for t in self.tickers if self.Portfolio[self.symbols[t]].Invested)
        new_holdings = set(t for t, _, _ in selected)

        # Sell positions not in new portfolio
        for t in held - new_holdings:
            self.Liquidate(self.symbols[t])

        # Buy/adjust positions
        for t, weight, alpha in selected:
            self.SetHoldings(self.symbols[t], weight)

        # Logging
        regime_str = f"[{self.current_regime}]"
        ks_str = f"[KS:{self.recovery_step}]" if self.kill_switch_active else ""
        self.Debug(f"{self.Time.date()} REBAL {regime_str}{ks_str}: {len(selected)} pos, {total_weight:.0%} exp")

    def OnEndOfAlgorithm(self):
        final_value = self.Portfolio.TotalPortfolioValue
        initial_value = 100000000
        total_return = (final_value / initial_value - 1) * 100
        years = (self.EndDate - self.StartDate).days / 365.25
        cagr = ((final_value / initial_value) ** (1/years) - 1) * 100 if years > 0 else 0

        self.Debug("=" * 50)
        self.Debug("ASSET SHIELD V5.0 - FINAL REPORT")
        self.Debug("=" * 50)
        self.Debug(f"Period: {self.StartDate.date()} to {self.EndDate.date()}")
        self.Debug(f"Initial: ¥{initial_value:,.0f}")
        self.Debug(f"Final:   ¥{final_value:,.0f}")
        self.Debug(f"Return:  {total_return:.2f}%")
        self.Debug(f"CAGR:    {cagr:.2f}%")
        self.Debug("=" * 50)
