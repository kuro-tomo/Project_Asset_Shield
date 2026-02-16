from AlgorithmImports import *
import numpy as np


class AssetShieldV8(QCAlgorithm):

    VERSION = "8.2"

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100_000)
        self.SetBenchmark("SPY")

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetSecurityInitializer(lambda s: s.SetSlippageModel(VolumeShareSlippageModel()))
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 2.0
        self.SetWarmup(30, Resolution.Daily)

        self.AddUniverse(self.CoarseFilter, self.FineFilter)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        self.MAX_POS = 10
        self.REBAL_DAYS = 21
        self.TRAIL_STOP = 0.08
        self.TAKE_PROFIT = 0.40
        self.MAX_DD_KILL = 0.15
        self.MAX_SECTOR_FRAC = 0.30
        self.LOOKBACK = 252

        self.W_MOM = 0.35
        self.W_SMOM = 0.10
        self.W_VOL = 0.15
        self.W_VAL = 0.20
        self.W_QUAL = 0.20

        self.active = set()
        self.sector_cache = {}
        self.fund_cache = {}
        self.trail_highs = {}

        self.hwm = 100_000
        self.kill = False
        self.kill_date = None
        self.recovery = 0
        self.regime = "NEUTRAL"
        self.trades = []
        self.day_count = self.REBAL_DAYS - 1 if self.LiveMode else 0

        self.IS_END = datetime(2017, 12, 31)
        self.phase_rets = {"IS": [], "OOS": []}
        self.prev_eq = 100_000
        self._cache_cleanup_counter = 0

        self.NOTIFY_EMAIL = "mikasa2564@gmail.com"
        self.prev_regime = "NEUTRAL"

        self.Debug(f"[INIT] V{self.VERSION} Live={self.LiveMode}")

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.DailyCheck
        )
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.BeforeMarketClose("SPY", 30),
            self.WeeklySummary
        )

    # ==========================================================
    # Safe Helpers
    # ==========================================================

    def _is_invested(self, symbol):
        try:
            return self.Portfolio[symbol].Invested
        except:
            return False

    def _safe_liquidate(self, symbol, tag=""):
        try:
            if self.Portfolio[symbol].Invested:
                self.Liquidate(symbol, tag)
        except:
            pass

    def _invested_symbols(self):
        result = []
        try:
            for s in list(self.Portfolio.Keys):
                try:
                    if self.Portfolio[s].Invested:
                        result.append(s)
                except:
                    continue
        except:
            pass
        return result

    # ==========================================================
    # Universe
    # ==========================================================

    def CoarseFilter(self, coarse):
        filtered = [
            x for x in coarse
            if x.HasFundamentalData and x.Price > 5 and x.DollarVolume > 5_000_000
        ]
        by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in by_volume[:150]]

    def FineFilter(self, fine):
        candidates = []
        for f in fine:
            try:
                pe = f.ValuationRatios.PERatio
                roe_val = f.OperationRatios.ROE.Value if f.OperationRatios.ROE else 0
                mktcap = f.MarketCap
                sector = f.AssetClassification.MorningstarSectorCode
            except:
                continue
            if pe <= 0 or pe > 200 or roe_val <= 0 or mktcap < 2_000_000_000:
                continue
            candidates.append(f)
            self.sector_cache[f.Symbol] = sector
            self.fund_cache[f.Symbol] = {"pe": pe, "roe": roe_val, "mktcap": mktcap}
        by_cap = sorted(candidates, key=lambda x: x.MarketCap, reverse=True)
        return [x.Symbol for x in by_cap[:100]]

    def OnSecuritiesChanged(self, changes):
        for s in changes.AddedSecurities:
            self.active.add(s.Symbol)
        for s in changes.RemovedSecurities:
            self.active.discard(s.Symbol)
            if s.Symbol != self.spy:
                self._safe_liquidate(s.Symbol, "UNIVERSE_EXIT")

    # ==========================================================
    # Regime
    # ==========================================================

    def _detect_regime(self):
        try:
            hist = self.History([self.spy], 60, Resolution.Daily)
        except:
            return
        if hist.empty or len(hist) < 40:
            return
        closes = hist["close"].values
        rets = np.diff(closes) / closes[:-1]
        vol = float(np.std(rets[-20:]) * np.sqrt(252))
        trend = float(closes[-1] / closes[0] - 1)

        old_regime = self.regime
        if vol > 0.40:
            self.regime = "CRISIS"
        elif vol > 0.25 and trend < -0.03:
            self.regime = "BEAR"
        elif trend > 0.10:
            self.regime = "SUPER_BULL"
        elif trend > 0.03:
            self.regime = "BULL"
        else:
            self.regime = "NEUTRAL"

        if self.regime != old_regime and self.regime in ("CRISIS", "BEAR"):
            eq = self.Portfolio.TotalPortfolioValue
            spy_w, sat_b = self._get_allocations()
            self._alert(
                f"REGIME -> {self.regime}",
                f"Regime changed: {old_regime} -> {self.regime}\n"
                f"Vol: {vol:.1%} Trend: {trend:+.1%}\nEquity: ${eq:,.0f}"
            )

    def _get_allocations(self):
        return {
            "CRISIS": (0.40, 0.00), "BEAR": (0.60, 0.15),
            "NEUTRAL": (0.80, 0.50), "BULL": (0.80, 0.55),
            "SUPER_BULL": (0.80, 0.60),
        }.get(self.regime, (0.80, 0.50))

    # ==========================================================
    # Daily Check — wrapped in top-level try/except
    # ==========================================================

    def DailyCheck(self):
        try:
            self._daily_check_impl()
        except BaseException as e:
            self.Debug(f"[CAUGHT] DailyCheck error: {e}")

    def _daily_check_impl(self):
        if self.IsWarmingUp:
            return
        self.day_count += 1
        eq = self.Portfolio.TotalPortfolioValue

        if self.prev_eq > 0 and not self.LiveMode:
            r = eq / self.prev_eq - 1
            phase = "IS" if self.Time <= self.IS_END else "OOS"
            self.phase_rets[phase].append(r)
        self.prev_eq = eq

        if self.LiveMode:
            self._cache_cleanup_counter += 1
            if self._cache_cleanup_counter % (self.REBAL_DAYS * 5) == 0:
                self._cleanup_caches()

        self.hwm = max(self.hwm, eq)
        dd = (self.hwm - eq) / self.hwm if self.hwm > 0 else 0
        self._kill_switch(dd)
        self._detect_regime()
        self._check_exits()

        if self.day_count % self.REBAL_DAYS != 0:
            return

        invested = self._invested_symbols()
        held = sum(1 for s in invested if s != self.spy)
        spy_w, factor_b = self._get_allocations()
        rebal_num = self.day_count // self.REBAL_DAYS
        if rebal_num % 2 == 0:
            self.Debug(f"[{self.Time.date()}] ${eq:,.0f} DD={dd:.1%} {self.regime} Pos={held}")

        if self.kill and self.recovery == 0:
            self.SetHoldings(self.spy, self._get_allocations()[0])
            return

        scale = self.recovery / 4.0 if self.kill else 1.0
        self._rebalance(scale)

    # ==========================================================
    # Kill-Switch
    # ==========================================================

    def _kill_switch(self, dd):
        if not self.kill and dd > self.MAX_DD_KILL:
            self.kill = True
            self.kill_date = self.Time
            self.recovery = 0
            sat_count = 0
            for symbol in self._invested_symbols():
                if symbol != self.spy:
                    self._safe_liquidate(symbol)
                    sat_count += 1
            self.trail_highs.clear()
            self.SetHoldings(self.spy, 0.40)
            self._alert("KILL-SWITCH ON",
                f"DD {dd:.1%} > {self.MAX_DD_KILL:.0%}. Liquidated {sat_count} sat positions.")
            return

        if self.kill and self.kill_date:
            days = (self.Time - self.kill_date).days
            if self.regime not in ("CRISIS", "BEAR"):
                step = min(days // 7, 4)
                if step > self.recovery:
                    self.recovery = step
                if self.recovery >= 4:
                    self.kill = False
                    self.kill_date = None
                    self.hwm = self.Portfolio.TotalPortfolioValue
                    self._alert("KILL-SWITCH OFF", "Full recovery complete.")
            else:
                self.recovery = 0
                self.kill_date = self.Time

    # ==========================================================
    # Trailing Stop & Take Profit
    # ==========================================================

    def _check_exits(self):
        for symbol in list(self.trail_highs.keys()):
            if symbol == self.spy:
                continue
            if not self._is_invested(symbol):
                self.trail_highs.pop(symbol, None)
                continue
            try:
                h = self.Portfolio[symbol]
                price = h.Price
            except:
                self.trail_highs.pop(symbol, None)
                continue
            if price <= 0:
                continue

            self.trail_highs[symbol] = max(self.trail_highs.get(symbol, price), price)
            peak = self.trail_highs[symbol]
            drop = (peak - price) / peak if peak > 0 else 0
            pnl = h.UnrealizedProfitPercent
            reason = None
            if drop > self.TRAIL_STOP:
                reason = "TRAIL"
            elif pnl > self.TAKE_PROFIT:
                reason = "TP"
            if reason:
                self._safe_liquidate(symbol, reason)
                self.trail_highs.pop(symbol, None)
                self.trades.append({"s": str(symbol), "pnl": float(pnl), "r": reason})

    # ==========================================================
    # Rebalance — with safe History access
    # ==========================================================

    def _rebalance(self, scale):
        spy_w, factor_budget = self._get_allocations()
        factor_budget *= scale
        self.SetHoldings(self.spy, spy_w)

        if factor_budget < 0.05:
            for symbol in self._invested_symbols():
                if symbol != self.spy:
                    self._safe_liquidate(symbol)
                    self.trail_highs.pop(symbol, None)
            return

        # Build safe symbol list
        syms = []
        for s in self.active:
            if s == self.spy:
                continue
            try:
                sec = self.Securities[s]
                if sec.IsTradable and sec.Price > 0:
                    syms.append(s)
            except:
                continue
        if len(syms) < 10:
            return

        # Safe History call
        try:
            hist = self.History(syms, self.LOOKBACK, Resolution.Daily)
        except:
            self.Debug("[WARN] History failed")
            return
        if hist.empty:
            return

        scores = []
        for symbol in syms:
            try:
                df = hist.loc[symbol]
                closes = df["close"].values
            except:
                continue
            if len(closes) < 60:
                continue

            if len(closes) >= 252 and closes[0] > 0:
                mom = closes[-22] / closes[0] - 1
            elif len(closes) >= 126 and closes[0] > 0:
                mom = closes[-22] / closes[0] - 1
            else:
                mom = (closes[-1] / closes[0] - 1) if closes[0] > 0 else 0

            smom = (closes[-1] / closes[-22] - 1) if len(closes) >= 22 and closes[-22] > 0 else 0
            rets = np.diff(closes[-60:]) / closes[-60:-1]
            vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 10 else 0.3
            daily_vol = float(np.std(rets)) if len(rets) > 10 else 0.02

            fc = self.fund_cache.get(symbol, {})
            pe, roe = fc.get("pe", 0), fc.get("roe", 0)
            if pe <= 0 or roe <= 0:
                continue

            scores.append({
                "symbol": symbol, "mom": mom, "smom": smom,
                "vol": vol, "daily_vol": daily_vol,
                "pe": pe, "roe": roe,
                "sector": self.sector_cache.get(symbol, 0),
            })

        if len(scores) < 10:
            return

        def z(arr):
            s = np.std(arr)
            return (arr - np.mean(arr)) / s if s > 1e-8 else np.zeros_like(arr)

        moms = z(np.array([s["mom"] for s in scores]))
        smoms = z(np.array([s["smom"] for s in scores]))
        vols = z(np.array([s["vol"] for s in scores])) * -1
        vals = z(np.array([s["pe"] for s in scores])) * -1
        quals = z(np.array([s["roe"] for s in scores]))

        for i, s in enumerate(scores):
            s["score"] = (moms[i]*self.W_MOM + smoms[i]*self.W_SMOM +
                         vols[i]*self.W_VOL + vals[i]*self.W_VAL + quals[i]*self.W_QUAL)

        scores.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        sec_count = {}
        max_per_sec = max(2, int(self.MAX_POS * self.MAX_SECTOR_FRAC))
        for s in scores:
            if len(selected) >= self.MAX_POS:
                break
            sec = s["sector"]
            if sec_count.get(sec, 0) >= max_per_sec:
                continue
            selected.append(s)
            sec_count[sec] = sec_count.get(sec, 0) + 1
        if not selected:
            return

        inv_vols = [1.0 / max(s["daily_vol"], 0.005) for s in selected]
        total_iv = sum(inv_vols)
        weights = {}
        for i, s in enumerate(selected):
            weights[s["symbol"]] = min((inv_vols[i] / total_iv) * factor_budget, 0.10)

        target_set = set(weights.keys())
        for symbol in self._invested_symbols():
            if symbol not in target_set and symbol != self.spy:
                self._safe_liquidate(symbol)
                self.trail_highs.pop(symbol, None)

        for symbol, w in weights.items():
            try:
                sec = self.Securities[symbol]
                if sec.Price <= 0 or not sec.IsTradable:
                    continue
            except:
                continue
            self.SetHoldings(symbol, w)
            if symbol not in self.trail_highs:
                self.trail_highs[symbol] = sec.Price

    # ==========================================================
    # Memory Management
    # ==========================================================

    def _cleanup_caches(self):
        keep = set(self.active)
        for s in self._invested_symbols():
            keep.add(s)
        ct = 0
        for cache in (self.sector_cache, self.fund_cache):
            stale = [k for k in cache if k not in keep]
            for k in stale:
                del cache[k]
            ct += len(stale)
        if len(self.trades) > 200:
            self.trades = self.trades[-200:]
        if ct > 0:
            self.Debug(f"[MEM] Purged {ct}")

    # ==========================================================
    # Notifications
    # ==========================================================

    def _alert(self, subject, body):
        tag = f"[AssetShield] {subject}"
        self.Debug(f"ALERT: {tag}")
        if self.NOTIFY_EMAIL and self.LiveMode:
            self.Notify.Email(self.NOTIFY_EMAIL, tag, body)

    def WeeklySummary(self):
        if self.IsWarmingUp:
            return
        eq = self.Portfolio.TotalPortfolioValue
        dd = (self.hwm - eq) / self.hwm if self.hwm > 0 else 0
        spy_w, sat_b = self._get_allocations()
        invested = self._invested_symbols()
        held = sum(1 for s in invested if s != self.spy)
        ret = eq / 100_000 - 1
        summary = (
            f"Weekly {self.Time.date()}\n{'='*40}\n"
            f"Equity: ${eq:,.0f} ({ret:+.1%})\nDD: {dd:.1%}\n"
            f"Regime: {self.regime}\nKill: {'ON' if self.kill else 'OFF'}\n"
            f"Pos: SPY + {held}\nAlloc: SPY {spy_w:.0%} + Sat {sat_b:.0%}"
        )
        if self.day_count % (self.REBAL_DAYS * 4) < self.REBAL_DAYS:
            self.Debug(f"[WEEKLY] ${eq:,.0f} {self.regime} Pos={held}")
        if self.NOTIFY_EMAIL and self.LiveMode:
            self.Notify.Email(self.NOTIFY_EMAIL, f"[AssetShield] Weekly {self.Time.date()}", summary)

    # ==========================================================
    # Final Report
    # ==========================================================

    def OnEndOfAlgorithm(self):
        eq = self.Portfolio.TotalPortfolioValue
        ret = eq / 100_000 - 1
        years = (self.EndDate - self.StartDate).days / 365.25
        cagr = (1 + ret) ** (1 / years) - 1 if years > 0 and ret > -1 else 0

        def sharpe(rets):
            if len(rets) < 20: return 0
            r = np.array(rets); s = np.std(r)
            return float(np.mean(r) * 252 / (s * np.sqrt(252))) if s > 1e-10 else 0

        def maxdd(rets):
            if not rets: return 0
            eq = np.cumprod(1 + np.array(rets))
            pk = np.maximum.accumulate(eq)
            return float(np.max((pk - eq) / pk))

        all_r = self.phase_rets["IS"] + self.phase_rets["OOS"]
        self.Debug("=" * 60)
        self.Debug(f"V{self.VERSION} Return: {ret:+.1%} CAGR: {cagr:.1%}")
        self.Debug(f"Sharpe: {sharpe(all_r):.2f} (IS={sharpe(self.phase_rets['IS']):.2f} OOS={sharpe(self.phase_rets['OOS']):.2f})")
        self.Debug(f"MaxDD: {maxdd(all_r):.1%}")
        self.Debug("=" * 60)
