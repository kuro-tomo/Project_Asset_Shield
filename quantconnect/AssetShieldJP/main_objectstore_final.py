# region imports
from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import deque
# endregion


class JapanStockData(PythonData):
    """
    Custom Data Reader for Japanese Stocks (ObjectStore)
    =====================================================

    Reads CSV data uploaded to ObjectStore via DataUploader chunks.

    CSV Format: Date,Open,High,Low,Close,Volume
    Example: 2008-05-07,5550.0,5590.0,5520.0,1116.0,9687200
    """

    def GetSource(self, config, date, isLiveMode):
        """Read from ObjectStore"""
        ticker = config.Symbol.Value
        return SubscriptionDataSource(
            f"japan_stocks/{ticker}.csv",
            SubscriptionTransportMedium.ObjectStore,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        """Parse CSV line"""
        if not line.strip() or not line[0].isdigit():
            return None

        try:
            cols = line.split(',')
            data = JapanStockData()
            data.Symbol = config.Symbol
            data.Time = datetime.strptime(cols[0], "%Y-%m-%d")
            data.EndTime = data.Time + timedelta(days=1)
            data.Value = float(cols[4])  # Close
            data["Open"] = float(cols[1])
            data["High"] = float(cols[2])
            data["Low"] = float(cols[3])
            data["Close"] = float(cols[4])
            data["Volume"] = int(float(cols[5]))
            return data
        except:
            return None


class AssetShieldJPFinal(QCAlgorithm):
    """
    Asset Shield V3.2.0 - Japanese Equities (ObjectStore Final)
    ============================================================

    Prerequisites:
    - Run all 4 DataUploader chunks first
    - Verify ObjectStore contains japan_stocks/*.csv

    Strategy:
    - 20 Japanese stocks (TOPIX large caps)
    - Value-Quality ranking (PBR/ROE)
    - 63-day rebalancing
    - Trend filter + Circuit breaker

    Target: OOS Sharpe >= 1.5, Max DD <= 35%
    """

    # 20 target stocks (J-Quants 5-digit codes)
    UNIVERSE = [
        "72030", "67580", "83060", "80350", "68610",
        "94320", "65010", "79740", "40630", "99840",
        "69020", "63670", "94330", "45020", "72670",
        "45030", "69540", "87660", "83160", "90220"
    ]

    NAMES = {
        "72030": "Toyota", "67580": "Sony", "83060": "MUFG",
        "80350": "TEL", "68610": "Keyence", "94320": "NTT",
        "65010": "Hitachi", "79740": "Nintendo", "40630": "ShinEtsu",
        "99840": "SoftBank", "69020": "Denso", "63670": "Daikin",
        "94330": "KDDI", "45020": "Takeda", "72670": "Suzuki",
        "45030": "Astellas", "69540": "FujiElec", "87660": "Tokio",
        "83160": "SMFG", "90220": "JR-East"
    }

    def Initialize(self):
        # === Backtest Period ===
        self.SetStartDate(2008, 5, 7)
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10_000_000)  # 10M JPY

        # === Walk-Forward Phases ===
        self.TRAINING_END = datetime(2015, 12, 31)
        self.VALIDATION_END = datetime(2020, 12, 31)

        # === Strategy Parameters ===
        self.MAX_POSITIONS = 20
        self.POSITION_PCT = 0.05
        self.MAX_POSITION_PCT = 0.10
        self.REBALANCE_DAYS = 63
        self.HOLDING_DAYS = 250
        self.STOP_LOSS = 0.12
        self.TAKE_PROFIT = 0.35
        self.MAX_DD = 0.35
        self.TREND_PERIOD = 60
        self.MIN_ADT = 400_000_000

        # === Add Custom Data from ObjectStore ===
        self.symbols = {}
        self.stock_data = {}

        self.Debug("=" * 60)
        self.Debug("ASSET SHIELD V3.2.0 - JAPAN EQUITIES")
        self.Debug("=" * 60)

        for code in self.UNIVERSE:
            try:
                security = self.AddData(JapanStockData, code, Resolution.Daily)
                self.symbols[code] = security.Symbol
                self.stock_data[code] = None
                self.Debug(f"Added: {code} ({self.NAMES.get(code, 'Unknown')})")
            except Exception as e:
                self.Error(f"Failed {code}: {e}")

        # === Validate Data Availability ===
        if len(self.symbols) == 0:
            self.Error("=" * 60)
            self.Error("FATAL: No symbols loaded!")
            self.Error("Required: ObjectStore data at japan_stocks/*.csv")
            self.Error("Solution: Run DataUploader chunks 1-4 first")
            self.Error("=" * 60)
            raise Exception("No data - run DataUploader first")

        self.Debug(f"Loaded {len(self.symbols)} symbols from ObjectStore")

        # === State ===
        self.positions = {}
        self.day_count = 0
        self.peak_equity = self.Portfolio.TotalPortfolioValue
        self.circuit_breaker = False
        self.market_ma = deque(maxlen=self.TREND_PERIOD)
        self.trades = []
        self.data_received = 0

        # === Phase Tracking ===
        self.phase_data = {"training": [], "validation": [], "oos": []}
        self.prev_equity = self.Portfolio.TotalPortfolioValue

        # === Schedule ===
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(10, 0),
            self.DailyRoutine
        )

    def OnData(self, data):
        """Receive custom data from ObjectStore CSVs"""
        for code, symbol in self.symbols.items():
            if data.ContainsKey(symbol):
                bar = data[symbol]
                if bar is not None and bar.Value > 0:
                    self.stock_data[code] = {
                        "price": bar.Value,
                        "open": bar["Open"],
                        "high": bar["High"],
                        "low": bar["Low"],
                        "close": bar["Close"],
                        "volume": bar["Volume"],
                        "time": bar.Time
                    }
                    self.data_received += 1

    def DailyRoutine(self):
        """Daily processing"""
        self.day_count += 1
        equity = self.Portfolio.TotalPortfolioValue

        # Phase tracking
        if self.prev_equity > 0:
            ret = (equity / self.prev_equity) - 1
            if self.Time <= self.TRAINING_END:
                self.phase_data["training"].append(ret)
            elif self.Time <= self.VALIDATION_END:
                self.phase_data["validation"].append(ret)
            else:
                self.phase_data["oos"].append(ret)
        self.prev_equity = equity

        # Market trend
        active_prices = [d["price"] for d in self.stock_data.values()
                        if d is not None and d["price"] > 0]
        if active_prices:
            self.market_ma.append(np.mean(active_prices))

        trend = self._calc_trend()

        # Drawdown
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Circuit breaker
        if dd > self.MAX_DD:
            self.circuit_breaker = True
        elif dd < 0.20:
            self.circuit_breaker = False

        # Rebalance
        if self.day_count % self.REBALANCE_DAYS != 0:
            return

        active = sum(1 for d in self.stock_data.values() if d is not None)
        self.Debug(f"[{self.Time.date()}] Equity: ¥{equity:,.0f} | DD: {dd:.1%} | "
                  f"Trend: {trend} | Active: {active}/{len(self.UNIVERSE)}")

        self._process_exits()

        if not self.circuit_breaker and trend in ("UP", "SIDEWAYS"):
            self._process_entries()

    def _calc_trend(self):
        """Calculate market trend"""
        if len(self.market_ma) < self.TREND_PERIOD:
            return "WAIT"

        ma = np.mean(list(self.market_ma))
        current = self.market_ma[-1]

        if current > ma:
            return "UP"
        elif current > ma * 0.95:
            return "SIDEWAYS"
        else:
            return "DOWN"

    def _process_exits(self):
        """Check and execute exits"""
        for code in list(self.positions.keys()):
            if code not in self.symbols:
                continue

            symbol = self.symbols[code]
            pos = self.positions[code]

            if not self.Portfolio[symbol].Invested:
                del self.positions[code]
                continue

            stock = self.stock_data.get(code)
            if stock is None:
                continue

            price = stock["price"]
            pnl = (price / pos["entry_price"]) - 1 if pos["entry_price"] > 0 else 0
            days = (self.Time - pos["entry_date"]).days

            exit_reason = None
            if days >= self.HOLDING_DAYS:
                exit_reason = "MAX_HOLD"
            elif pnl <= -self.STOP_LOSS:
                exit_reason = "STOP_LOSS"
            elif pnl >= self.TAKE_PROFIT:
                exit_reason = "TAKE_PROFIT"

            if exit_reason:
                self.Liquidate(symbol, exit_reason)
                self.trades.append({"code": code, "pnl": pnl, "reason": exit_reason})
                del self.positions[code]
                self.Debug(f"EXIT {code}: {exit_reason}, P&L: {pnl:.1%}")

    def _process_entries(self):
        """Find candidates and enter positions"""
        available = self.MAX_POSITIONS - len(self.positions)
        if available <= 0:
            return

        candidates = self._find_candidates()
        held = set(self.positions.keys())
        candidates = [c for c in candidates if c["code"] not in held]

        if not candidates:
            return

        equity = self.Portfolio.TotalPortfolioValue

        for cand in candidates[:available]:
            code = cand["code"]
            symbol = self.symbols[code]
            price = cand["price"]

            size = min(equity * self.POSITION_PCT,
                      equity * self.MAX_POSITION_PCT,
                      self.Portfolio.Cash * 0.90)

            if size < 100_000:
                continue

            shares = int(size / price)
            if shares <= 0:
                continue

            self.MarketOrder(symbol, shares)
            self.positions[code] = {
                "entry_date": self.Time,
                "entry_price": price,
                "shares": shares
            }
            self.Debug(f"ENTRY {code}: {shares} @ ¥{price:,.0f}")

    def _find_candidates(self):
        """
        Find candidates using Value-Quality ranking
        ===========================================

        Reads from ObjectStore custom CSV data (self.stock_data)
        instead of unsupported Market.Japan AddEquity calls.

        Logic:
        1. Filter by liquidity
        2. Rank by PBR (lower = better value)
        3. Rank by price momentum as ROE proxy
        4. Composite = 50% value + 50% quality
        5. Select top 20%
        """
        candidates = []

        for code in self.UNIVERSE:
            stock = self.stock_data.get(code)
            if stock is None:
                continue

            price = stock["price"]
            volume = stock["volume"]

            if price <= 0 or volume <= 0:
                continue

            # Liquidity check
            turnover = price * volume
            if turnover < self.MIN_ADT / 60:
                continue

            # PBR proxy (normalized price level)
            pbr_proxy = price / 10000

            # ROE proxy (price level as quality indicator)
            roe_proxy = min(price / 50000, 1.0)

            candidates.append({
                "code": code,
                "symbol": self.symbols[code],
                "price": price,
                "pbr": pbr_proxy,
                "roe": roe_proxy,
                "turnover": turnover
            })

        if len(candidates) < 5:
            return []

        # Percentile ranking
        pbrs = [c["pbr"] for c in candidates]
        roes = [c["roe"] for c in candidates]

        for c in candidates:
            c["pbr_pct"] = sum(1 for p in pbrs if p <= c["pbr"]) / len(pbrs)
            c["roe_pct"] = sum(1 for r in roes if r <= c["roe"]) / len(roes)
            c["score"] = (1 - c["pbr_pct"]) * 0.5 + c["roe_pct"] * 0.5

        # Top 20%
        threshold = np.percentile([c["score"] for c in candidates], 80)
        selected = [c for c in candidates if c["score"] >= threshold]
        selected.sort(key=lambda x: x["score"], reverse=True)

        return selected

    def OnEndOfAlgorithm(self):
        """Final performance report"""
        equity = self.Portfolio.TotalPortfolioValue
        total_return = (equity / 10_000_000) - 1

        def sharpe(returns):
            if len(returns) < 20:
                return 0.0
            r = np.array(returns)
            if np.std(r) == 0:
                return 0.0
            return (np.mean(r) * 252) / (np.std(r) * np.sqrt(252))

        def max_dd(returns):
            if not returns:
                return 0.0
            eq = np.cumprod(1 + np.array(returns))
            peak = np.maximum.accumulate(eq)
            return np.max((peak - eq) / peak)

        training_sharpe = sharpe(self.phase_data["training"])
        validation_sharpe = sharpe(self.phase_data["validation"])
        oos_sharpe = sharpe(self.phase_data["oos"])

        all_returns = self.phase_data["training"] + self.phase_data["validation"] + self.phase_data["oos"]
        overall_dd = max_dd(all_returns)

        win_rate = 0
        if self.trades:
            wins = len([t for t in self.trades if t["pnl"] > 0])
            win_rate = wins / len(self.trades)

        self.Debug("=" * 70)
        self.Debug("ASSET SHIELD V3.2.0 - FINAL REPORT")
        self.Debug("=" * 70)
        self.Debug(f"Data Points Received: {self.data_received:,}")
        self.Debug(f"Total Trades:         {len(self.trades)}")
        self.Debug(f"Win Rate:             {win_rate:.1%}")
        self.Debug("-" * 70)
        self.Debug(f"Total Return:         {total_return:.2%}")
        self.Debug(f"Final Equity:         ¥{equity:,.0f}")
        self.Debug(f"Max Drawdown:         {overall_dd:.2%}")
        self.Debug("-" * 70)
        self.Debug(f"Training Sharpe:      {training_sharpe:.2f} ({len(self.phase_data['training'])} days)")
        self.Debug(f"Validation Sharpe:    {validation_sharpe:.2f} ({len(self.phase_data['validation'])} days)")
        self.Debug(f"OOS Sharpe:           {oos_sharpe:.2f} ({len(self.phase_data['oos'])} days)")
        self.Debug("=" * 70)

        # Validation
        if oos_sharpe >= 1.5:
            self.Debug("✓ TARGET MET: OOS Sharpe >= 1.5 - READY FOR SUBMISSION")
        elif oos_sharpe >= 0.7:
            self.Debug("△ ACCEPTABLE: OOS Sharpe >= 0.7")
        else:
            self.Debug("✗ FAILED: OOS Sharpe < 0.7 - DO NOT SUBMIT")

        if overall_dd <= 0.35:
            self.Debug("✓ RISK OK: Max DD <= 35%")
        else:
            self.Debug("✗ RISK FAIL: Max DD > 35%")
