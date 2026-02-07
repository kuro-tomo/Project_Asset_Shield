# region imports
from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import deque
# endregion


class JapanStockData(PythonData):
    """
    Custom Data Class for Japanese Stocks
    =====================================

    Reads CSV data from ObjectStore in format:
    Date,Open,High,Low,Close,Volume
    2008-05-07,1798.0,1804.0,1787.5,1799.0,25995600
    """

    def GetSource(self, config, date, isLiveMode):
        """Define data source - ObjectStore"""
        ticker = config.Symbol.Value
        source = f"japan_stocks/{ticker}.csv"
        return SubscriptionDataSource(
            source,
            SubscriptionTransportMedium.ObjectStore,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        """Parse each CSV line"""
        if not line.strip():
            return None
        if not line[0].isdigit():
            return None  # Skip header

        data = JapanStockData()
        data.Symbol = config.Symbol

        try:
            cols = line.split(',')
            # Format: Date,Open,High,Low,Close,Volume
            data.Time = datetime.strptime(cols[0], "%Y-%m-%d")
            data.EndTime = data.Time + timedelta(days=1)

            data.Value = float(cols[4])  # Close price
            data["Open"] = float(cols[1])
            data["High"] = float(cols[2])
            data["Low"] = float(cols[3])
            data["Close"] = float(cols[4])
            data["Volume"] = int(float(cols[5]))

            # Calculate pseudo-fundamentals from price
            # (In production, would use separate fundamental data)
            data["PBR"] = data.Value / 1000  # Placeholder
            data["ROE"] = 0.10  # Placeholder

        except Exception as e:
            return None

        return data


class AssetShieldJPObjectStore(QCAlgorithm):
    """
    Asset Shield V3.2.0 - ObjectStore Custom Data Version
    ======================================================

    Full Japanese stock strategy using custom uploaded data.

    Prerequisites:
    1. Run DataUploader class first to upload CSV data
    2. Then run this class for backtest

    Strategy:
    - PBR/ROE ranking (Value-Quality)
    - 20 stock portfolio
    - 63-day rebalancing
    - Trend filter (60-day MA)
    - DD circuit breaker (35%)
    """

    # Target stocks (J-Quants 5-digit codes)
    STOCK_CODES = [
        "72030", "67580", "83060", "80350", "68610",
        "94320", "65010", "79740", "40630", "99840",
        "69020", "63670", "94330", "45020", "72670",
        "45030", "69540", "87660", "83160", "90220"
    ]

    # Stock name mapping for debugging
    STOCK_NAMES = {
        "72030": "Toyota", "67580": "Sony", "83060": "MUFG",
        "80350": "TEL", "68610": "Keyence", "94320": "NTT",
        "65010": "Hitachi", "79740": "Nintendo", "40630": "ShinEtsu",
        "99840": "SoftBank"
    }

    def Initialize(self):
        # === Period Settings ===
        self.SetStartDate(2008, 5, 7)
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10_000_000)  # 10M JPY

        # === Walk-Forward Periods ===
        self.training_end = datetime(2015, 12, 31)
        self.validation_end = datetime(2020, 12, 31)

        # === Strategy Parameters ===
        self.max_positions = 20
        self.position_pct = 0.05
        self.max_position_pct = 0.10
        self.rebalance_days = 63
        self.holding_days = 250
        self.stop_loss = 0.12
        self.take_profit = 0.35
        self.max_dd = 0.35
        self.trend_period = 60
        self.min_adt = 400_000_000  # 400M JPY

        # === Add Custom Data Symbols ===
        self.symbols = {}
        self.stock_data = {}  # Store latest data per stock

        for code in self.STOCK_CODES:
            try:
                symbol = self.AddData(JapanStockData, code, Resolution.Daily)
                self.symbols[code] = symbol.Symbol
                self.stock_data[code] = None
                self.Debug(f"Added: {code} ({self.STOCK_NAMES.get(code, 'Unknown')})")
            except Exception as e:
                self.Error(f"Failed to add {code}: {e}")

        # === Data Validation ===
        if len(self.symbols) == 0:
            self.Error("=" * 60)
            self.Error("CRITICAL ERROR: No symbols loaded!")
            self.Error("Required: Custom Japan Stock Data in ObjectStore")
            self.Error("Solution: Run DataUploader class first")
            self.Error("Path: japan_stocks/{code}.csv")
            self.Error("=" * 60)
            raise Exception("No data available - run DataUploader first")

        self.Debug(f"Successfully added {len(self.symbols)} symbols")

        # === State Variables ===
        self.positions_info = {}
        self.day_count = 0
        self.peak_equity = self.Portfolio.TotalPortfolioValue
        self.circuit_breaker = False
        self.market_prices = deque(maxlen=self.trend_period)
        self.trades = []
        self.data_points_received = 0

        # === Phase Tracking ===
        self.phase_equity = {"training": [], "validation": [], "oos": []}
        self.prev_equity = self.Portfolio.TotalPortfolioValue

        # === Schedule ===
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(10, 0),
            self.DailyProcess
        )

    def OnData(self, data):
        """Receive and store custom data"""
        for code, symbol in self.symbols.items():
            if data.ContainsKey(symbol):
                bar = data[symbol]
                if bar is not None:
                    self.stock_data[code] = {
                        "price": bar.Value,
                        "open": bar["Open"],
                        "high": bar["High"],
                        "low": bar["Low"],
                        "close": bar["Close"],
                        "volume": bar["Volume"],
                        "time": bar.Time
                    }
                    self.data_points_received += 1

    def DailyProcess(self):
        """Daily processing"""
        self.day_count += 1
        equity = self.Portfolio.TotalPortfolioValue

        # Phase tracking
        if self.prev_equity > 0:
            daily_ret = (equity / self.prev_equity) - 1
            if self.Time <= self.training_end:
                self.phase_equity["training"].append(daily_ret)
            elif self.Time <= self.validation_end:
                self.phase_equity["validation"].append(daily_ret)
            else:
                self.phase_equity["oos"].append(daily_ret)
        self.prev_equity = equity

        # Market average for trend
        prices = [d["price"] for d in self.stock_data.values()
                  if d is not None and d["price"] > 0]
        if prices:
            market_price = np.mean(prices)
            self.market_prices.append(market_price)

        # Trend calculation
        if len(self.market_prices) >= self.trend_period:
            ma = np.mean(list(self.market_prices))
            current = self.market_prices[-1] if self.market_prices else 0
            if current > ma:
                trend = "UP"
            elif current > ma * 0.95:
                trend = "SIDEWAYS"
            else:
                trend = "DOWN"
        else:
            trend = "WAIT"

        # Drawdown
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Circuit breaker
        if dd > self.max_dd:
            self.circuit_breaker = True
        elif dd < 0.20:
            self.circuit_breaker = False

        # Rebalance check
        if self.day_count % self.rebalance_days != 0:
            return

        active_stocks = sum(1 for d in self.stock_data.values() if d is not None)
        self.Debug(f"[{self.Time.date()}] Day {self.day_count} | "
                  f"Equity: ¥{equity:,.0f} | DD: {dd:.1%} | "
                  f"Trend: {trend} | Active: {active_stocks}")

        # Check exits
        self._check_exits()

        # Open new positions (if allowed)
        if not self.circuit_breaker and trend in ("UP", "SIDEWAYS"):
            self._open_positions()

    def _check_exits(self):
        """Check and execute exits"""
        for code in list(self.positions_info.keys()):
            if code not in self.symbols:
                continue

            symbol = self.symbols[code]
            info = self.positions_info[code]

            if not self.Portfolio[symbol].Invested:
                del self.positions_info[code]
                continue

            # Get current price from custom data
            stock_info = self.stock_data.get(code)
            if stock_info is None:
                continue

            current_price = stock_info["price"]
            entry_price = info["entry_price"]
            entry_date = info["entry_date"]

            holding_days = (self.Time - entry_date).days
            pnl_pct = (current_price / entry_price) - 1 if entry_price > 0 else 0

            should_exit = False
            reason = ""

            if holding_days >= self.holding_days:
                should_exit = True
                reason = "MAX_HOLD"
            elif pnl_pct <= -self.stop_loss:
                should_exit = True
                reason = "STOP_LOSS"
            elif pnl_pct >= self.take_profit:
                should_exit = True
                reason = "TAKE_PROFIT"

            if should_exit:
                self.Liquidate(symbol, reason)
                self.trades.append({
                    "code": code,
                    "entry": entry_date,
                    "exit": self.Time,
                    "pnl_pct": pnl_pct,
                    "reason": reason
                })
                del self.positions_info[code]
                self.Debug(f"EXIT {code}: {reason}, P&L: {pnl_pct:.1%}")

    def _open_positions(self):
        """Open new positions based on ranking"""
        current_count = len(self.positions_info)
        max_new = self.max_positions - current_count

        if max_new <= 0:
            return

        # Find candidates using custom data
        candidates = self._find_candidates()

        # Exclude already held
        held = set(self.positions_info.keys())
        candidates = [c for c in candidates if c["code"] not in held]

        if not candidates:
            return

        equity = self.Portfolio.TotalPortfolioValue

        for cand in candidates[:max_new]:
            code = cand["code"]
            symbol = self.symbols[code]
            price = cand["price"]

            # Position sizing
            pos_value = min(
                equity * self.position_pct,
                equity * self.max_position_pct,
                self.Portfolio.Cash * 0.90
            )

            if pos_value < 100_000:
                continue

            shares = int(pos_value / price)
            if shares <= 0:
                continue

            # Execute order
            self.MarketOrder(symbol, shares)

            self.positions_info[code] = {
                "entry_date": self.Time,
                "entry_price": price,
                "shares": shares
            }

            self.Debug(f"ENTRY {code}: {shares} shares @ ¥{price:,.0f}")

    def _find_candidates(self):
        """
        Find candidate stocks using PBR/ROE ranking
        ============================================

        Reads from custom CSV data stored in self.stock_data
        instead of standard AddEquity data.

        Ranking Logic:
        1. Filter by liquidity (volume * price > min_adt)
        2. Calculate PBR percentile (lower = better)
        3. Calculate ROE percentile (higher = better)
        4. Composite score = 0.5 * (1 - PBR_pct) + 0.5 * ROE_pct
        5. Select top 20%
        """
        candidates = []

        for code, symbol in self.symbols.items():
            # Get data from custom CSV source
            stock_info = self.stock_data.get(code)
            if stock_info is None:
                continue

            price = stock_info["price"]
            volume = stock_info["volume"]

            if price <= 0 or volume <= 0:
                continue

            # Liquidity filter (ADT proxy)
            turnover = price * volume
            if turnover < self.min_adt / 60:  # Daily vs 60-day avg
                continue

            # PBR proxy: inverse of price level (normalized)
            # Lower price relative to peers = potentially lower PBR
            pbr_proxy = price / 10000  # Normalize

            # ROE proxy: price momentum (60-day approximation)
            # In production, would use actual fundamental data
            roe_proxy = 0.10  # Placeholder - would calculate from fundamentals

            candidates.append({
                "code": code,
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "turnover": turnover,
                "pbr": pbr_proxy,
                "roe": roe_proxy
            })

        if len(candidates) < 5:
            return []

        # Calculate percentile rankings
        pbrs = [c["pbr"] for c in candidates]
        roes = [c["roe"] for c in candidates]

        for c in candidates:
            # PBR percentile (lower PBR = higher score)
            c["pbr_pct"] = sum(1 for p in pbrs if p <= c["pbr"]) / len(pbrs)
            # ROE percentile (higher ROE = higher score)
            c["roe_pct"] = sum(1 for r in roes if r <= c["roe"]) / len(roes)
            # Composite: value (low PBR) + quality (high ROE)
            c["composite"] = (1 - c["pbr_pct"]) * 0.5 + c["roe_pct"] * 0.5

        # Select top 20%
        threshold = np.percentile([c["composite"] for c in candidates], 80)
        selected = [c for c in candidates if c["composite"] >= threshold]
        selected.sort(key=lambda x: x["composite"], reverse=True)

        return selected

    def OnEndOfAlgorithm(self):
        """Final report with walk-forward analysis"""
        equity = self.Portfolio.TotalPortfolioValue
        initial = 10_000_000
        total_return = (equity / initial) - 1

        # Sharpe calculation
        def calc_sharpe(returns):
            if len(returns) < 20:
                return 0.0
            returns = np.array(returns)
            if np.std(returns) == 0:
                return 0.0
            annual_ret = np.mean(returns) * 252
            annual_vol = np.std(returns) * np.sqrt(252)
            return annual_ret / annual_vol if annual_vol > 0 else 0.0

        training_sharpe = calc_sharpe(self.phase_equity["training"])
        validation_sharpe = calc_sharpe(self.phase_equity["validation"])
        oos_sharpe = calc_sharpe(self.phase_equity["oos"])

        # Max DD
        all_returns = (self.phase_equity["training"] +
                      self.phase_equity["validation"] +
                      self.phase_equity["oos"])
        if all_returns:
            equity_curve = np.cumprod(1 + np.array(all_returns))
            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / peak
            max_dd = np.max(dd)
        else:
            max_dd = 0

        # Win rate
        if self.trades:
            wins = len([t for t in self.trades if t["pnl_pct"] > 0])
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0

        self.Debug("=" * 70)
        self.Debug("ASSET SHIELD V3.2.0 - JAPAN STOCKS (OBJECTSTORE)")
        self.Debug("=" * 70)
        self.Debug(f"Data Points Received: {self.data_points_received}")
        self.Debug(f"Total Trades:         {len(self.trades)}")
        self.Debug(f"Win Rate:             {win_rate:.1%}")
        self.Debug("-" * 70)
        self.Debug(f"Total Return:         {total_return:.2%}")
        self.Debug(f"Final Equity:         ¥{equity:,.0f}")
        self.Debug(f"Max Drawdown:         {max_dd:.2%}")
        self.Debug("-" * 70)
        self.Debug(f"Training Sharpe:      {training_sharpe:.2f}")
        self.Debug(f"Validation Sharpe:    {validation_sharpe:.2f}")
        self.Debug(f"OOS Sharpe:           {oos_sharpe:.2f}")
        self.Debug("=" * 70)

        # Validation
        if oos_sharpe >= 1.5:
            self.Debug("✓ OOS SHARPE >= 1.5 - READY FOR SUBMISSION")
        elif oos_sharpe >= 0.7:
            self.Debug("△ OOS SHARPE >= 0.7 - ACCEPTABLE")
        else:
            self.Debug("✗ OOS SHARPE < 0.7 - DO NOT SUBMIT")


# ============================================================
# DATA UPLOADER - RUN THIS FIRST
# ============================================================

class DataUploader(QCAlgorithm):
    """
    Data Uploader for ObjectStore
    =============================

    Run this BEFORE running AssetShieldJPObjectStore.

    Instructions:
    1. Copy CSV content from local files
    2. Paste into STOCK_DATA dictionary below
    3. Run this algorithm (short backtest)
    4. Data will be saved to ObjectStore
    5. Then run AssetShieldJPObjectStore
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 2)
        self.SetCash(1000)

        # ============================================================
        # PASTE YOUR CSV DATA HERE
        # Format: "code": """date,open,high,low,close,volume\n..."""
        # ============================================================

        STOCK_DATA = {
            # Example - replace with actual data from:
            # /Users/MBP/Desktop/Project_Asset_Shield/quantconnect/custom_data/

            "72030": """2008-05-07,5765.0,5915.0,5755.0,5913.0,18621000
2008-05-08,5920.0,5920.0,5720.0,5730.0,24562000
2008-05-09,5730.0,5780.0,5620.0,5700.0,18893000""",

            "67580": """2008-05-07,5070.0,5080.0,4910.0,4930.0,14891500
2008-05-08,4950.0,4990.0,4800.0,4820.0,20087800
2008-05-09,4820.0,4840.0,4730.0,4800.0,15093600""",

            # Add more stocks...
        }

        # Upload each stock's data
        uploaded = 0
        for code, csv_data in STOCK_DATA.items():
            if csv_data.strip():
                key = f"japan_stocks/{code}.csv"
                self.ObjectStore.Save(key, csv_data)
                lines = len(csv_data.strip().split('\n'))
                self.Debug(f"Uploaded: {key} ({lines} lines)")
                uploaded += 1

        self.Debug("=" * 60)
        self.Debug(f"Upload complete: {uploaded} stocks")
        self.Debug("Now run AssetShieldJPObjectStore for backtest")
        self.Debug("=" * 60)

    def OnData(self, data):
        pass
