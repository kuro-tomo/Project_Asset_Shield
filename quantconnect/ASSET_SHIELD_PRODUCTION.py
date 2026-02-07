# region imports
from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import deque
# endregion


# =============================================================
# Custom Data Readers
# =============================================================

class JapanStockData(PythonData):
    """
    Price data from ObjectStore: japan_stocks/{code}.csv
    Format: Date,Open,High,Low,Close,Volume
    """

    def GetSource(self, config, date, isLiveMode):
        ticker = config.Symbol.Value
        return SubscriptionDataSource(
            f"japan_stocks/{ticker}.csv",
            SubscriptionTransportMedium.ObjectStore,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or not line[0].isdigit():
            return None
        try:
            cols = line.split(',')
            data = JapanStockData()
            data.Symbol = config.Symbol
            data.Time = datetime.strptime(cols[0], "%Y-%m-%d")
            data.EndTime = data.Time + timedelta(days=1)
            data.Value = float(cols[4])
            data["Open"] = float(cols[1])
            data["High"] = float(cols[2])
            data["Low"] = float(cols[3])
            data["Close"] = float(cols[4])
            data["Volume"] = int(float(cols[5]))
            return data
        except:
            return None


class JapanFundamentalData(PythonData):
    """
    Fundamental data from ObjectStore: japan_fundamentals/{code}.csv
    Format: disclosed_date,fiscal_quarter,bps,roe,eps
    Point-in-Time: Uses disclosed_date to prevent look-ahead bias.
    """

    def GetSource(self, config, date, isLiveMode):
        ticker = config.Symbol.Value.replace("_FUND", "")
        return SubscriptionDataSource(
            f"japan_fundamentals/{ticker}.csv",
            SubscriptionTransportMedium.ObjectStore,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or not line[0].isdigit():
            return None
        try:
            cols = line.split(',')
            data = JapanFundamentalData()
            data.Symbol = config.Symbol
            data.Time = datetime.strptime(cols[0], "%Y-%m-%d")
            data.EndTime = data.Time + timedelta(days=1)
            bps = float(cols[2])
            data.Value = bps
            data["BPS"] = bps
            data["ROE"] = float(cols[3])
            data["EPS"] = float(cols[4])
            data["Quarter"] = cols[1]
            return data
        except:
            return None


# =============================================================
# Transaction Cost Model
# =============================================================

class CostModel:
    """
    Institutional-grade transaction cost model for Japanese equities.

    Components:
    1. Commission:  5 bps per side (broker fee)
    2. Spread cost: f(ADT) - wider for less liquid stocks
    3. Market impact: Square-root model (Almgren-Chriss)
       impact = sigma_daily * sqrt(Q / ADV) * lambda

    Returns total one-way cost in decimal (e.g., 0.0030 = 30 bps).
    """

    COMMISSION_BPS = 5.0       # 5 bps per side
    IMPACT_LAMBDA = 0.10       # Market impact coefficient
    MIN_SPREAD_BPS = 2.0       # Floor for most liquid names
    MAX_SPREAD_BPS = 20.0      # Cap for illiquid names

    @staticmethod
    def estimate_spread_bps(adt_jpy):
        """Estimate bid-ask half-spread from ADT."""
        if adt_jpy <= 0:
            return CostModel.MAX_SPREAD_BPS
        # Empirical: spread ~ 1 / sqrt(ADT in billions)
        adt_b = adt_jpy / 1e9
        spread = 5.0 / max(np.sqrt(adt_b), 0.1)
        return np.clip(spread, CostModel.MIN_SPREAD_BPS, CostModel.MAX_SPREAD_BPS)

    @staticmethod
    def market_impact_bps(trade_value, adt_jpy, daily_vol):
        """
        Square-root market impact model.
        trade_value: JPY notional of the order
        adt_jpy:     20-day average daily turnover
        daily_vol:   Daily return volatility (not annualized)
        """
        if adt_jpy <= 0 or daily_vol <= 0:
            return 10.0  # Conservative fallback
        participation = trade_value / adt_jpy
        impact = daily_vol * np.sqrt(participation) * CostModel.IMPACT_LAMBDA
        return impact * 10000  # Convert to bps

    @staticmethod
    def total_cost_bps(trade_value, adt_jpy, daily_vol):
        """Total one-way cost in bps."""
        commission = CostModel.COMMISSION_BPS
        spread = CostModel.estimate_spread_bps(adt_jpy)
        impact = CostModel.market_impact_bps(trade_value, adt_jpy, daily_vol)
        return commission + spread + impact

    @staticmethod
    def total_cost_decimal(trade_value, adt_jpy, daily_vol):
        """Total one-way cost as decimal fraction."""
        return CostModel.total_cost_bps(trade_value, adt_jpy, daily_vol) / 10000


# =============================================================
# Main Algorithm
# =============================================================

class AssetShieldProduction(QCAlgorithm):
    """
    Asset Shield Production - Japanese Equities (ObjectStore)
    =========================================================
    Universe:  50 TOPIX Large Caps (ObjectStore CSV)
    Factors:   Value(PBR) + Quality(ROE) + Momentum(12-1) + ShortMom + LowVol
    Costs:     Commission + Spread + Market Impact (Almgren-Chriss)
    Regime:    5-level volatility + trend adaptive exposure
    Risk:      Kill-switch (18% DD), sector caps, stop-loss/take-profit

    Walk-Forward:
      Training:   2008-05 ~ 2015-12
      Validation: 2016-01 ~ 2020-12
      OOS:        2021-01 ~ 2026-02
    """

    # === 50-Stock Universe ===
    UNIVERSE = [
        "29140", "33820", "38610", "40630", "44520",
        "45020", "45030", "45680", "46610", "49010",
        "51080", "54010", "60980", "62730", "63010",
        "63260", "63670", "65010", "65940", "67020",
        "67520", "67580", "68570", "68610", "69020",
        "69540", "70110", "72030", "72670", "72690",
        "72700", "77510", "79740", "80010", "80310",
        "80350", "80580", "83060", "83160", "84110",
        "86040", "87660", "88020", "90200", "90220",
        "91010", "94320", "94330", "97350", "99840",
    ]

    SECTORS = {
        "29140": "01", "33820": "02", "38610": "03", "40630": "04",
        "44520": "03", "45020": "05", "45030": "05", "45680": "04",
        "46610": "03", "49010": "04", "51080": "06", "54010": "07",
        "60980": "10", "62730": "09", "63010": "09", "63260": "09",
        "63670": "09", "65010": "09", "65940": "09", "67020": "09",
        "67520": "09", "67580": "09", "68570": "09", "68610": "09",
        "69020": "06", "69540": "09", "70110": "08", "72030": "06",
        "72670": "06", "72690": "06", "72700": "06", "77510": "12",
        "79740": "09", "80010": "13", "80310": "13", "80350": "09",
        "80580": "14", "83060": "15", "83160": "15", "84110": "16",
        "86040": "16", "87660": "16", "88020": "17", "90200": "11",
        "90220": "11", "91010": "10", "94320": "10", "94330": "10",
        "97350": "13", "99840": "10",
    }

    def Initialize(self):
        # === Backtest Period ===
        self.SetStartDate(2008, 5, 7)
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10_000_000)

        # === Walk-Forward ===
        self.TRAINING_END = datetime(2015, 12, 31)
        self.VALIDATION_END = datetime(2020, 12, 31)

        # === Strategy Parameters ===
        self.MAX_POSITIONS = 15
        self.POSITION_PCT = 0.065
        self.MAX_POSITION_PCT = 0.10
        self.REBALANCE_DAYS = 63
        self.HOLDING_DAYS = 250
        self.STOP_LOSS = 0.12
        self.TAKE_PROFIT = 0.35
        self.MAX_DD_KILL = 0.18
        self.MAX_SECTOR_PCT = 0.30
        self.HISTORY_LEN = 300
        self.TREND_PERIOD = 60
        self.MIN_ADT = 100_000_000

        # === Factor Weights (5-factor) ===
        self.W_VALUE = 0.20       # PBR (lower = better)
        self.W_QUALITY = 0.15     # ROE (higher = better)
        self.W_MOMENTUM = 0.35    # 12-1 month
        self.W_SHORT_MOM = 0.10   # 20-day
        self.W_LOW_VOL = 0.20     # Lower vol = better

        # === Add Price Data ===
        self.symbols = {}
        self.price_hist = {}
        self.vol_hist = {}
        self.stock_data = {}

        for code in self.UNIVERSE:
            try:
                sec = self.AddData(JapanStockData, code, Resolution.Daily)
                self.symbols[code] = sec.Symbol
                self.price_hist[code] = deque(maxlen=self.HISTORY_LEN)
                self.vol_hist[code] = deque(maxlen=self.HISTORY_LEN)
                self.stock_data[code] = None
            except Exception as e:
                self.Error(f"Price data failed {code}: {e}")

        # === Add Fundamental Data ===
        self.fund_symbols = {}
        self.fundamentals = {}  # {code: {"bps": float, "roe": float, "eps": float}}

        for code in self.UNIVERSE:
            try:
                fund_ticker = f"{code}_FUND"
                sec = self.AddData(JapanFundamentalData, fund_ticker, Resolution.Daily)
                self.fund_symbols[code] = sec.Symbol
                self.fundamentals[code] = None
            except Exception as e:
                self.Error(f"Fundamental data failed {code}: {e}")

        if len(self.symbols) == 0:
            raise Exception("No symbols loaded - verify ObjectStore data")

        self.Debug(f"Loaded: {len(self.symbols)} price + {len(self.fund_symbols)} fundamental feeds")

        # === State ===
        self.positions = {}
        self.day_count = 0
        self.high_water_mark = self.Portfolio.TotalPortfolioValue
        self.kill_switch = False
        self.kill_switch_date = None
        self.recovery_step = 0
        self.current_regime = "NEUTRAL"
        self.trades = []
        self.total_cost_paid = 0.0  # Cumulative transaction costs

        # === Phase Tracking ===
        self.phase_returns = {"training": [], "validation": [], "oos": []}
        self.prev_equity = self.Portfolio.TotalPortfolioValue

        # === Schedule ===
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(10, 0),
            self.DailyRoutine
        )

    # =========================================================
    # Data Ingestion
    # =========================================================

    def OnData(self, data):
        # Price data
        for code, symbol in self.symbols.items():
            if data.ContainsKey(symbol):
                bar = data[symbol]
                if bar is not None and bar.Value > 0:
                    price = bar.Value
                    volume = bar["Volume"]
                    self.stock_data[code] = {
                        "price": price,
                        "volume": volume,
                        "time": bar.Time,
                    }
                    self.price_hist[code].append(price)
                    self.vol_hist[code].append(volume)

        # Fundamental data (PIT: only use after disclosed_date)
        for code, symbol in self.fund_symbols.items():
            if data.ContainsKey(symbol):
                bar = data[symbol]
                if bar is not None and bar.Value > 0:
                    self.fundamentals[code] = {
                        "bps": bar["BPS"],
                        "roe": bar["ROE"],
                        "eps": bar["EPS"],
                        "disclosed": bar.Time,
                    }

    # =========================================================
    # Daily Routine
    # =========================================================

    def DailyRoutine(self):
        self.day_count += 1
        equity = self.Portfolio.TotalPortfolioValue

        # Phase tracking
        if self.prev_equity > 0:
            ret = (equity / self.prev_equity) - 1
            if self.Time <= self.TRAINING_END:
                self.phase_returns["training"].append(ret)
            elif self.Time <= self.VALIDATION_END:
                self.phase_returns["validation"].append(ret)
            else:
                self.phase_returns["oos"].append(ret)
        self.prev_equity = equity

        # Drawdown & kill-switch
        if equity > self.high_water_mark:
            self.high_water_mark = equity
        dd = (self.high_water_mark - equity) / self.high_water_mark if self.high_water_mark > 0 else 0
        self._check_kill_switch(dd)

        # Regime
        self._update_regime()

        # Rebalance
        if self.day_count % self.REBALANCE_DAYS != 0:
            return

        active = sum(1 for d in self.stock_data.values() if d is not None)
        fund_active = sum(1 for f in self.fundamentals.values() if f is not None)
        self.Debug(
            f"[{self.Time.date()}] Eq=¥{equity:,.0f} DD={dd:.1%} "
            f"Regime={self.current_regime} Price={active} Fund={fund_active} "
            f"Pos={len(self.positions)} Costs=¥{self.total_cost_paid:,.0f}"
        )

        self._process_exits()

        if not self.kill_switch:
            self._process_entries()
        elif self.recovery_step > 0:
            self._process_entries(scale=self.recovery_step / 4.0)

    # =========================================================
    # Regime Detection (full universe, not [:20])
    # =========================================================

    def _update_regime(self):
        vols = []
        trends = []
        for code in self.symbols:
            prices = list(self.price_hist.get(code, []))
            if len(prices) < 60:
                continue
            p = np.array(prices)
            rets = np.diff(p[-22:]) / np.where(p[-22:-1] == 0, 1, p[-22:-1])
            if len(rets) > 0:
                vols.append(float(np.std(rets) * np.sqrt(252)))
            if p[-60] > 0:
                trends.append(float(p[-1] / p[-60] - 1))

        if len(vols) < 10:
            return

        avg_vol = np.median(vols)   # Median is more robust than mean
        avg_trend = np.median(trends) if trends else 0

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

    def _get_exposure_scale(self):
        return {
            "CRISIS": 0.30, "BEAR": 0.60, "NEUTRAL": 1.00,
            "BULL": 1.10, "SUPER_BULL": 1.30,
        }.get(self.current_regime, 1.00)

    # =========================================================
    # Kill-Switch
    # =========================================================

    def _check_kill_switch(self, dd):
        if not self.kill_switch and dd > self.MAX_DD_KILL:
            self.kill_switch = True
            self.kill_switch_date = self.Time
            self.recovery_step = 0
            self.Debug(f"KILL SWITCH ON: DD={dd:.1%}")
            for code in list(self.positions.keys()):
                if code in self.symbols:
                    self.Liquidate(self.symbols[code], "KILL_SWITCH")
            self.positions.clear()
            return

        if self.kill_switch and self.kill_switch_date:
            days = (self.Time - self.kill_switch_date).days
            if self.current_regime not in ("CRISIS", "BEAR"):
                step = min(days // 5, 4)
                if step > self.recovery_step:
                    self.recovery_step = step
                    self.Debug(f"Recovery step: {self.recovery_step}/4")
                if self.recovery_step >= 4:
                    self.kill_switch = False
                    self.kill_switch_date = None
                    self.high_water_mark = self.Portfolio.TotalPortfolioValue
                    self.Debug("KILL SWITCH OFF")
            else:
                self.recovery_step = 0
                self.kill_switch_date = self.Time

    # =========================================================
    # 5-Factor Alpha with Real Fundamentals
    # =========================================================

    def _calc_alpha(self, code):
        prices = list(self.price_hist.get(code, []))
        volumes = list(self.vol_hist.get(code, []))

        if len(prices) < 60:
            return None

        p = np.array(prices)
        v = np.array(volumes)

        # --- Price-based factors ---

        # Momentum 12-1 (skip last month to avoid reversal)
        if len(p) >= 252 and p[-252] > 0:
            mom = (p[-22] / p[-252]) - 1
        elif len(p) >= 120 and p[-120] > 0:
            mom = (p[-22] / p[-120]) - 1
        elif p[-60] > 0:
            mom = (p[-1] / p[-60]) - 1
        else:
            mom = 0.0

        # Short momentum (20-day)
        short_mom = (p[-1] / p[-22] - 1) if len(p) >= 22 and p[-22] > 0 else 0.0

        # Volatility (60-day annualized)
        rets = np.diff(p[-61:]) / np.where(p[-61:-1] == 0, 1, p[-61:-1])
        ann_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 5 else 0.30
        daily_vol = float(np.std(rets)) if len(rets) > 5 else 0.02

        # ADT
        adt = float(np.mean(p[-20:] * v[-20:])) if len(p) >= 20 and len(v) >= 20 else 0.0

        # --- Fundamental factors (PIT) ---

        fund = self.fundamentals.get(code)
        current_price = float(p[-1])

        if fund is not None and fund["bps"] > 0:
            pbr = current_price / fund["bps"]
            roe = fund["roe"]
        else:
            pbr = None
            roe = None

        return {
            "code": code,
            "momentum": mom,
            "short_mom": short_mom,
            "volatility": ann_vol,
            "daily_vol": daily_vol,
            "adt": adt,
            "price": current_price,
            "pbr": pbr,
            "roe": roe,
        }

    def _rank_candidates(self):
        alphas = []
        for code in self.symbols:
            result = self._calc_alpha(code)
            if result is None:
                continue
            if result["adt"] < self.MIN_ADT:
                continue
            if result["price"] <= 0:
                continue
            alphas.append(result)

        if len(alphas) < 5:
            return []

        # --- Cross-sectional z-scores ---

        def zscore(arr):
            s = np.std(arr)
            if s < 1e-8:
                return np.zeros_like(arr)
            return (arr - np.mean(arr)) / s

        moms = np.array([a["momentum"] for a in alphas])
        smoms = np.array([a["short_mom"] for a in alphas])
        vols = np.array([a["volatility"] for a in alphas])

        z_mom = zscore(moms)
        z_smom = zscore(smoms)
        z_vol = zscore(vols) * -1  # Lower vol = higher score

        # Fundamental z-scores (handle missing data)
        has_fund = [a for a in alphas if a["pbr"] is not None and a["roe"] is not None]
        fund_available = len(has_fund) >= 5

        if fund_available:
            pbrs = np.array([a["pbr"] if a["pbr"] is not None else np.nan for a in alphas])
            roes = np.array([a["roe"] if a["roe"] is not None else np.nan for a in alphas])

            # Fill NaN with cross-sectional median (neutral score)
            pbr_med = np.nanmedian(pbrs)
            roe_med = np.nanmedian(roes)
            pbrs = np.where(np.isnan(pbrs), pbr_med, pbrs)
            roes = np.where(np.isnan(roes), roe_med, roes)

            z_value = zscore(pbrs) * -1   # Lower PBR = higher value score
            z_quality = zscore(roes)       # Higher ROE = higher quality score
        else:
            z_value = np.zeros(len(alphas))
            z_quality = np.zeros(len(alphas))

        # --- Composite score ---

        for i, a in enumerate(alphas):
            if fund_available:
                a["score"] = (
                    z_value[i] * self.W_VALUE +
                    z_quality[i] * self.W_QUALITY +
                    z_mom[i] * self.W_MOMENTUM +
                    z_smom[i] * self.W_SHORT_MOM +
                    z_vol[i] * self.W_LOW_VOL
                )
                a["z_value"] = float(z_value[i])
                a["z_quality"] = float(z_quality[i])
            else:
                # Fallback: momentum-only when no fundamentals yet
                renorm = self.W_MOMENTUM + self.W_SHORT_MOM + self.W_LOW_VOL
                a["score"] = (
                    z_mom[i] * (self.W_MOMENTUM / renorm) +
                    z_smom[i] * (self.W_SHORT_MOM / renorm) +
                    z_vol[i] * (self.W_LOW_VOL / renorm)
                )
                a["z_value"] = 0.0
                a["z_quality"] = 0.0

            a["z_momentum"] = float(z_mom[i])
            a["z_lowvol"] = float(z_vol[i])

        alphas.sort(key=lambda x: x["score"], reverse=True)
        return alphas

    # =========================================================
    # Position Management with Cost-Aware Execution
    # =========================================================

    def _process_exits(self):
        for code in list(self.positions.keys()):
            if code not in self.symbols:
                continue
            symbol = self.symbols[code]
            if not self.Portfolio[symbol].Invested:
                del self.positions[code]
                continue

            stock = self.stock_data.get(code)
            if stock is None:
                continue

            price = stock["price"]
            pos = self.positions[code]
            pnl = (price / pos["entry_price"]) - 1 if pos["entry_price"] > 0 else 0
            days = (self.Time - pos["entry_date"]).days

            reason = None
            if days >= self.HOLDING_DAYS:
                reason = "MAX_HOLD"
            elif pnl <= -self.STOP_LOSS:
                reason = "STOP_LOSS"
            elif pnl >= self.TAKE_PROFIT:
                reason = "TAKE_PROFIT"
            elif self.current_regime == "CRISIS":
                reason = "CRISIS_EXIT"

            if reason:
                # Estimate exit cost
                trade_value = price * pos["shares"]
                exit_cost = CostModel.total_cost_decimal(
                    trade_value, pos.get("adt", self.MIN_ADT), pos.get("daily_vol", 0.02)
                )
                cost_jpy = trade_value * exit_cost
                self.total_cost_paid += cost_jpy

                self.Liquidate(symbol, reason)
                net_pnl = pnl - exit_cost - pos.get("entry_cost_pct", 0)
                self.trades.append({
                    "code": code,
                    "pnl_gross": pnl,
                    "pnl_net": net_pnl,
                    "cost_bps": (exit_cost + pos.get("entry_cost_pct", 0)) * 10000,
                    "reason": reason,
                })
                del self.positions[code]
                self.Debug(
                    f"EXIT {code}: {reason} gross={pnl:+.1%} net={net_pnl:+.1%} "
                    f"cost={cost_jpy:,.0f}"
                )

    def _process_entries(self, scale=1.0):
        exposure = self._get_exposure_scale() * scale
        max_pos = int(self.MAX_POSITIONS * exposure)
        available = max_pos - len(self.positions)
        if available <= 0:
            return

        candidates = self._rank_candidates()
        held = set(self.positions.keys())
        candidates = [c for c in candidates if c["code"] not in held]

        if not candidates:
            return

        equity = self.Portfolio.TotalPortfolioValue
        sector_weights = {}
        for code in self.positions:
            sec = self.SECTORS.get(code, "99")
            sector_weights[sec] = sector_weights.get(sec, 0) + self.POSITION_PCT

        for cand in candidates[:available]:
            code = cand["code"]
            sector = self.SECTORS.get(code, "99")

            cur_sector = sector_weights.get(sector, 0)
            if cur_sector + self.POSITION_PCT > self.MAX_SECTOR_PCT:
                continue

            price = cand["price"]
            adt = cand["adt"]
            daily_vol = cand["daily_vol"]

            pos_value = min(
                equity * self.POSITION_PCT * exposure,
                equity * self.MAX_POSITION_PCT,
                self.Portfolio.Cash * 0.85
            )

            if pos_value < 100_000:
                continue

            # Pre-trade cost estimate
            entry_cost = CostModel.total_cost_decimal(pos_value, adt, daily_vol)
            cost_jpy = pos_value * entry_cost

            # Reduce order size by expected cost
            net_value = pos_value - cost_jpy
            shares = int(net_value / price)
            if shares <= 0:
                continue

            self.total_cost_paid += cost_jpy

            self.MarketOrder(self.symbols[code], shares)
            self.positions[code] = {
                "entry_date": self.Time,
                "entry_price": price,
                "shares": shares,
                "entry_cost_pct": entry_cost,
                "adt": adt,
                "daily_vol": daily_vol,
            }
            sector_weights[sector] = cur_sector + self.POSITION_PCT
            self.Debug(
                f"ENTRY {code}: {shares} @ ¥{price:,.0f} "
                f"score={cand['score']:.2f} "
                f"PBR={cand['pbr']:.2f} ROE={cand['roe']:.1f}% "
                f"cost={CostModel.total_cost_bps(pos_value, adt, daily_vol):.1f}bps"
                if cand["pbr"] is not None and cand["roe"] is not None
                else f"ENTRY {code}: {shares} @ ¥{price:,.0f} score={cand['score']:.2f} (no fund)"
            )

    # =========================================================
    # Final Report
    # =========================================================

    def OnEndOfAlgorithm(self):
        equity = self.Portfolio.TotalPortfolioValue
        initial = 10_000_000
        total_return = (equity / initial) - 1
        years = (self.EndDate - self.StartDate).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        def sharpe(rets):
            if len(rets) < 20:
                return 0.0
            r = np.array(rets)
            s = np.std(r)
            if s < 1e-10:
                return 0.0
            return float((np.mean(r) * 252) / (s * np.sqrt(252)))

        def max_dd(rets):
            if not rets:
                return 0.0
            eq = np.cumprod(1 + np.array(rets))
            peak = np.maximum.accumulate(eq)
            return float(np.max((peak - eq) / peak))

        all_rets = (
            self.phase_returns["training"] +
            self.phase_returns["validation"] +
            self.phase_returns["oos"]
        )

        tr_sharpe = sharpe(self.phase_returns["training"])
        va_sharpe = sharpe(self.phase_returns["validation"])
        oos_sharpe = sharpe(self.phase_returns["oos"])
        overall_dd = max_dd(all_rets)

        wins_gross = len([t for t in self.trades if t["pnl_gross"] > 0])
        wins_net = len([t for t in self.trades if t["pnl_net"] > 0])
        n_trades = len(self.trades)
        win_rate_gross = wins_gross / n_trades if n_trades else 0
        win_rate_net = wins_net / n_trades if n_trades else 0
        avg_cost = np.mean([t["cost_bps"] for t in self.trades]) if self.trades else 0

        self.Debug("=" * 70)
        self.Debug("ASSET SHIELD PRODUCTION - FINAL REPORT")
        self.Debug("5-Factor: Value(PBR) + Quality(ROE) + Mom + ShortMom + LowVol")
        self.Debug("Cost Model: Commission + Spread + Market Impact")
        self.Debug("=" * 70)
        self.Debug(f"Total Return:         {total_return:+.2%}")
        self.Debug(f"CAGR:                 {cagr:.2%}")
        self.Debug(f"Final Equity:         ¥{equity:,.0f}")
        self.Debug(f"Max Drawdown:         {overall_dd:.2%}")
        self.Debug(f"Trades:               {n_trades}")
        self.Debug(f"Win Rate (gross):     {win_rate_gross:.1%}")
        self.Debug(f"Win Rate (net):       {win_rate_net:.1%}")
        self.Debug("-" * 70)
        self.Debug(f"Total Costs Paid:     ¥{self.total_cost_paid:,.0f}")
        self.Debug(f"Avg Cost/Trade:       {avg_cost:.1f} bps (round-trip)")
        self.Debug(f"Cost Drag on Return:  {self.total_cost_paid / initial:.2%}")
        self.Debug("-" * 70)
        self.Debug(f"Training Sharpe:      {tr_sharpe:.2f} ({len(self.phase_returns['training'])} days)")
        self.Debug(f"Validation Sharpe:    {va_sharpe:.2f} ({len(self.phase_returns['validation'])} days)")
        self.Debug(f"OOS Sharpe:           {oos_sharpe:.2f} ({len(self.phase_returns['oos'])} days)")
        self.Debug("=" * 70)

        # Factor contribution summary
        fund_count = sum(1 for f in self.fundamentals.values() if f is not None)
        self.Debug(f"Fundamental Coverage: {fund_count}/{len(self.UNIVERSE)} stocks")

        # Pass/Fail
        checks = [
            ("OOS Sharpe >= 0.7", oos_sharpe >= 0.7),
            ("Max DD <= 35%", overall_dd <= 0.35),
            ("Trades > 0", n_trades > 0),
            ("Win Rate (net) > 45%", win_rate_net > 0.45),
            ("Avg Cost < 50 bps", avg_cost < 50),
        ]
        for label, passed in checks:
            self.Debug(f"  [{'PASS' if passed else 'FAIL'}] {label}")

        if all(p for _, p in checks):
            self.Debug(">>> READY FOR ALPHA STREAMS SUBMISSION <<<")
        else:
            self.Debug(">>> FIX FAILURES BEFORE SUBMISSION <<<")
