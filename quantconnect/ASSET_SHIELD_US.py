from AlgorithmImports import *
import numpy as np


class AssetShieldV8(QCAlgorithm):
    """
    Asset Shield V8 - Core-Satellite
    =================================
    Core:  80% SPY (beta capture, never miss benchmark)
    Sat:   50% factor portfolio (V6 Pro alpha overlay)
    Total: ~130% in neutral, regime-adjusted 40-140%
    Kill-switch protects satellite only; SPY core always held.
    """

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

        # === Parameters ===
        self.MAX_POS = 10
        self.REBAL_DAYS = 21
        self.TRAIL_STOP = 0.08
        self.TAKE_PROFIT = 0.40
        self.MAX_DD_KILL = 0.15
        self.MAX_SECTOR_FRAC = 0.30
        self.LOOKBACK = 252

        # V6 Pro factor weights (proven best Sharpe)
        self.W_MOM = 0.35
        self.W_SMOM = 0.10
        self.W_VOL = 0.15
        self.W_VAL = 0.20
        self.W_QUAL = 0.20

        # === Caches ===
        self.active = set()
        self.sector_cache = {}
        self.fund_cache = {}
        self.trail_highs = {}

        # === State ===
        self.hwm = 100_000
        self.kill = False
        self.kill_date = None
        self.recovery = 0
        self.regime = "NEUTRAL"
        self.trades = []
        self.day_count = 0

        # Walk-forward
        self.IS_END = datetime(2017, 12, 31)
        self.phase_rets = {"IS": [], "OOS": []}
        self.prev_eq = 100_000

        # === Notifications ===
        self.NOTIFY_EMAIL = "mikasa2564@gmail.com"
        self.prev_regime = "NEUTRAL"

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.DailyCheck
        )
        # Weekly summary every Friday 30 min before close
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.BeforeMarketClose("SPY", 30),
            self.WeeklySummary
        )

    # ==========================================================
    # Universe Selection (same as V6 Pro)
    # ==========================================================

    def CoarseFilter(self, coarse):
        filtered = [
            x for x in coarse
            if x.HasFundamentalData
            and x.Price > 5
            and x.DollarVolume > 5_000_000
        ]
        by_volume = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in by_volume[:200]]

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

            if pe <= 0 or pe > 200:
                continue
            if roe_val <= 0:
                continue
            if mktcap < 2_000_000_000:
                continue

            candidates.append(f)
            self.sector_cache[f.Symbol] = sector
            self.fund_cache[f.Symbol] = {
                "pe": pe,
                "roe": roe_val,
                "mktcap": mktcap,
            }

        by_cap = sorted(candidates, key=lambda x: x.MarketCap, reverse=True)
        return [x.Symbol for x in by_cap[:100]]

    def OnSecuritiesChanged(self, changes):
        for s in changes.AddedSecurities:
            self.active.add(s.Symbol)
        for s in changes.RemovedSecurities:
            self.active.discard(s.Symbol)
            if self.Portfolio[s.Symbol].Invested and s.Symbol != self.spy:
                self.Liquidate(s.Symbol, "UNIVERSE_EXIT")

    # ==========================================================
    # Core-Satellite Exposure by Regime
    # ==========================================================

    def _detect_regime(self):
        hist = self.History([self.spy], 60, Resolution.Daily)
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

        # Notify on regime change to CRISIS or BEAR
        if self.regime != old_regime and self.regime in ("CRISIS", "BEAR"):
            eq = self.Portfolio.TotalPortfolioValue
            spy_w, sat_b = self._get_allocations()
            self._alert(
                f"REGIME -> {self.regime}",
                f"Regime changed: {old_regime} -> {self.regime}\n"
                f"SPY Vol(20d): {vol:.1%}  Trend(60d): {trend:+.1%}\n"
                f"Equity: ${eq:,.0f}\n"
                f"Allocation: SPY {spy_w:.0%} + Satellite {sat_b:.0%}"
            )

    def _get_allocations(self):
        """Returns (spy_weight, factor_budget) by regime."""
        return {
            "CRISIS":     (0.40, 0.00),
            "BEAR":       (0.60, 0.15),
            "NEUTRAL":    (0.80, 0.50),
            "BULL":       (0.80, 0.55),
            "SUPER_BULL": (0.80, 0.60),
        }.get(self.regime, (0.80, 0.50))

    # ==========================================================
    # Daily Check
    # ==========================================================

    def DailyCheck(self):
        if self.IsWarmingUp:
            return
        self.day_count += 1
        eq = self.Portfolio.TotalPortfolioValue

        if self.prev_eq > 0:
            r = eq / self.prev_eq - 1
            phase = "IS" if self.Time <= self.IS_END else "OOS"
            self.phase_rets[phase].append(r)
        self.prev_eq = eq

        self.hwm = max(self.hwm, eq)
        dd = (self.hwm - eq) / self.hwm if self.hwm > 0 else 0
        self._kill_switch(dd)
        self._detect_regime()
        self._check_exits()

        if self.day_count % self.REBAL_DAYS != 0:
            return

        held = sum(1 for s in self.active if self.Portfolio[s].Invested and s != self.spy)
        spy_w, factor_b = self._get_allocations()
        # Log every 2nd rebalance to reduce message count
        rebal_num = self.day_count // self.REBAL_DAYS
        if rebal_num % 2 == 0:
            self.Debug(
                f"[{self.Time.date()}] ${eq:,.0f} DD={dd:.1%} "
                f"Regime={self.regime} SPY={spy_w:.0%} Sat={factor_b:.0%} "
                f"Pos={held}"
            )

        if self.kill and self.recovery == 0:
            spy_w, _ = self._get_allocations()
            self.SetHoldings(self.spy, spy_w)
            return

        scale = self.recovery / 4.0 if self.kill else 1.0
        self._rebalance(scale)

    # ==========================================================
    # Kill-Switch (satellite only, SPY core stays)
    # ==========================================================

    def _kill_switch(self, dd):
        if not self.kill and dd > self.MAX_DD_KILL:
            self.kill = True
            self.kill_date = self.Time
            self.recovery = 0
            self.Debug(f"KILL ON: DD={dd:.1%} (satellite only)")

            # Liquidate satellite positions only, keep SPY
            sat_count = 0
            for symbol in list(self.Portfolio.Keys):
                if self.Portfolio[symbol].Invested and symbol != self.spy:
                    self.Liquidate(symbol)
                    sat_count += 1
            self.trail_highs.clear()

            # Maintain SPY core at crisis level
            self.SetHoldings(self.spy, 0.40)

            eq = self.Portfolio.TotalPortfolioValue
            self._alert(
                "KILL-SWITCH ON",
                f"Drawdown {dd:.1%} exceeded {self.MAX_DD_KILL:.0%} threshold.\n"
                f"Liquidated {sat_count} satellite positions.\n"
                f"SPY core maintained at 40%.\n"
                f"Equity: ${eq:,.0f}  Regime: {self.regime}\n"
                f"Recovery: 4-step re-entry over ~28 days when regime improves."
            )
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
                    self.Debug("KILL OFF")
                    eq = self.Portfolio.TotalPortfolioValue
                    self._alert(
                        "KILL-SWITCH OFF",
                        f"Full recovery complete. Resuming normal operations.\n"
                        f"Equity: ${eq:,.0f}  Regime: {self.regime}\n"
                        f"HWM reset to ${self.hwm:,.0f}"
                    )
            else:
                self.recovery = 0
                self.kill_date = self.Time

    # ==========================================================
    # Trailing Stop & Take Profit (satellite only)
    # ==========================================================

    def _check_exits(self):
        for symbol in list(self.trail_highs.keys()):
            if symbol == self.spy:
                continue

            h = self.Portfolio[symbol]
            if not h.Invested:
                self.trail_highs.pop(symbol, None)
                continue

            price = h.Price
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
                self.Liquidate(symbol, reason)
                self.trail_highs.pop(symbol, None)
                self.trades.append({"s": str(symbol), "pnl": float(pnl), "r": reason})
                # Only log take-profits and big losses to avoid rate limiting
                if reason == "TP" or pnl < -0.10:
                    ticker = str(symbol).split(" ")[0] if " " in str(symbol) else str(symbol)
                    self.Debug(f"{reason} {ticker} {pnl:+.1%}")

    # ==========================================================
    # Core-Satellite Rebalance
    # ==========================================================

    def _rebalance(self, scale):
        spy_w, factor_budget = self._get_allocations()
        factor_budget *= scale

        # --- CORE: Always hold SPY ---
        self.SetHoldings(self.spy, spy_w)

        # --- SATELLITE: Factor portfolio ---
        if factor_budget < 0.05:
            for symbol in list(self.Portfolio.Keys):
                if self.Portfolio[symbol].Invested and symbol != self.spy:
                    self.Liquidate(symbol)
                    self.trail_highs.pop(symbol, None)
            return

        syms = [s for s in self.active if s != self.spy]
        if len(syms) < 10:
            return

        hist = self.History(syms, self.LOOKBACK, Resolution.Daily)
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
            pe = fc.get("pe", 0)
            roe = fc.get("roe", 0)
            if pe <= 0 or roe <= 0:
                continue

            sector = self.sector_cache.get(symbol, 0)

            scores.append({
                "symbol": symbol,
                "mom": mom, "smom": smom,
                "vol": vol, "daily_vol": daily_vol,
                "pe": pe, "roe": roe,
                "sector": sector,
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
            s["score"] = (
                moms[i] * self.W_MOM +
                smoms[i] * self.W_SMOM +
                vols[i] * self.W_VOL +
                vals[i] * self.W_VAL +
                quals[i] * self.W_QUAL
            )

        scores.sort(key=lambda x: x["score"], reverse=True)

        n = self.MAX_POS
        selected = []
        sec_count = {}
        max_per_sec = max(2, int(n * self.MAX_SECTOR_FRAC))

        for s in scores:
            if len(selected) >= n:
                break
            sec = s["sector"]
            if sec_count.get(sec, 0) >= max_per_sec:
                continue
            selected.append(s)
            sec_count[sec] = sec_count.get(sec, 0) + 1

        if not selected:
            return

        # Inverse-vol sizing within factor_budget
        inv_vols = []
        for s in selected:
            iv = 1.0 / max(s["daily_vol"], 0.005)
            inv_vols.append(iv)

        total_iv = sum(inv_vols)
        weights = {}
        for i, s in enumerate(selected):
            w = (inv_vols[i] / total_iv) * factor_budget
            weights[s["symbol"]] = min(w, 0.10)  # cap 10% per satellite pos

        # Sell satellite positions not in target
        target_set = set(weights.keys())
        for symbol in list(self.Portfolio.Keys):
            if self.Portfolio[symbol].Invested and symbol not in target_set and symbol != self.spy:
                self.Liquidate(symbol)
                self.trail_highs.pop(symbol, None)

        for symbol, w in weights.items():
            if symbol not in self.Securities or self.Securities[symbol].Price <= 0:
                continue
            self.SetHoldings(symbol, w)
            if symbol not in self.trail_highs:
                self.trail_highs[symbol] = self.Securities[symbol].Price

        total_exp = spy_w + sum(weights.values())
        rebal_num = self.day_count // self.REBAL_DAYS
        if rebal_num % 2 == 0:
            names = [f"{str(s['symbol']).split(' ')[0]}({s['score']:.2f})" for s in selected[:6]]
            self.Debug(f"  -> SPY {spy_w:.0%} + {len(selected)} sat = {total_exp:.0%} total: {', '.join(names)}...")

    # ==========================================================
    # Notifications
    # ==========================================================

    def _alert(self, subject, body):
        """Send email notification for critical events."""
        tag = f"[AssetShield] {subject}"
        self.Debug(f"ALERT: {tag}")
        if self.NOTIFY_EMAIL and self.LiveMode:
            self.Notify.Email(self.NOTIFY_EMAIL, tag, body)

    def WeeklySummary(self):
        """Friday afternoon portfolio summary."""
        if self.IsWarmingUp:
            return
        eq = self.Portfolio.TotalPortfolioValue
        dd = (self.hwm - eq) / self.hwm if self.hwm > 0 else 0
        spy_w, sat_b = self._get_allocations()
        held = sum(1 for s in self.Portfolio.Keys if self.Portfolio[s].Invested and s != self.spy)
        ret = eq / 100_000 - 1

        summary = (
            f"Weekly Summary - {self.Time.date()}\n"
            f"{'='*40}\n"
            f"Equity:     ${eq:,.0f} ({ret:+.1%})\n"
            f"Drawdown:   {dd:.1%} (kill at {self.MAX_DD_KILL:.0%})\n"
            f"Regime:     {self.regime}\n"
            f"Kill-switch: {'ON (step ' + str(self.recovery) + '/4)' if self.kill else 'OFF'}\n"
            f"Positions:  SPY + {held} satellite\n"
            f"Allocation: SPY {spy_w:.0%} + Satellite {sat_b:.0%}\n"
        )

        # Only send email in live mode, always log for backtest verification
        if self.day_count % (self.REBAL_DAYS * 4) < self.REBAL_DAYS:
            self.Debug(f"[WEEKLY] ${eq:,.0f} DD={dd:.1%} {self.regime} Pos={held}")

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
            if len(rets) < 20:
                return 0
            r = np.array(rets)
            s = np.std(r)
            return float(np.mean(r) * 252 / (s * np.sqrt(252))) if s > 1e-10 else 0

        def maxdd(rets):
            if not rets:
                return 0
            eq = np.cumprod(1 + np.array(rets))
            pk = np.maximum.accumulate(eq)
            return float(np.max((pk - eq) / pk))

        all_r = self.phase_rets["IS"] + self.phase_rets["OOS"]
        is_sharpe = sharpe(self.phase_rets["IS"])
        oos_sharpe = sharpe(self.phase_rets["OOS"])
        full_sharpe = sharpe(all_r)
        full_dd = maxdd(all_r)

        n = len(self.trades)
        wins = len([t for t in self.trades if t["pnl"] > 0])

        self.Debug("=" * 65)
        self.Debug("ASSET SHIELD V8 - CORE-SATELLITE")
        self.Debug("SPY Core 80% + Factor Satellite 50% | Leveraged Alpha")
        self.Debug("=" * 65)
        self.Debug(f"Return:        {ret:+.1%}  CAGR: {cagr:.1%}")
        self.Debug(f"Equity:        ${eq:,.0f}")
        self.Debug(f"Max Drawdown:  {full_dd:.1%}")
        self.Debug(f"Sharpe (full): {full_sharpe:.2f}")
        self.Debug("-" * 65)
        self.Debug(f"IS Sharpe:     {is_sharpe:.2f} (2010-2017, {len(self.phase_rets['IS'])} days)")
        self.Debug(f"OOS Sharpe:    {oos_sharpe:.2f} (2018-2024, {len(self.phase_rets['OOS'])} days)")
        self.Debug("-" * 65)
        self.Debug(f"Trades:        {n}  Win: {wins}/{n}={wins/n:.0%}" if n else "No trades")
        self.Debug(f"Universe:      {len(self.active)} active stocks")
        self.Debug(f"Fund coverage: {len(self.fund_cache)} stocks with PE/ROE")
        self.Debug("=" * 65)

        checks = [
            ("OOS Sharpe > 0.5", oos_sharpe > 0.5),
            ("Max DD < 25%", full_dd < 0.25),
            ("CAGR > 12%", cagr > 0.12),
            ("Trades > 20", n > 20),
        ]
        for label, ok in checks:
            self.Debug(f"  [{'PASS' if ok else 'FAIL'}] {label}")
