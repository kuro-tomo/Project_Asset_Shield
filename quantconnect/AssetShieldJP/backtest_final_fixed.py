# region imports
from AlgorithmImports import *
from datetime import datetime, timedelta
# endregion


class JapanStockData(PythonData):
    """Custom data reader for ObjectStore CSVs"""

    def GetSource(self, config, date, isLiveMode):
        ticker = config.Symbol.Value
        return SubscriptionDataSource(
            f"japan_stocks/{ticker}.csv",
            SubscriptionTransportMedium.OBJECT_STORE  # Fixed: OBJECT_STORE not ObjectStore
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or not line[0].isdigit():
            return None
        try:
            # Format: Date,Open,High,Low,Close,Volume
            c = line.split(',')
            d = JapanStockData()
            d.Symbol = config.Symbol
            d.Time = datetime.strptime(c[0], "%Y-%m-%d")
            d.EndTime = d.Time + timedelta(days=1)
            d.Value = float(c[4])  # Close
            d["Open"] = float(c[1])
            d["High"] = float(c[2])
            d["Low"] = float(c[3])
            d["Volume"] = int(float(c[5]))
            return d
        except:
            return None


class AssetShieldBacktest(QCAlgorithm):
    """
    Asset Shield V3.2.0 - ObjectStore Backtest
    """

    TICKERS = ["7203", "6758", "8306"]

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(10000000)

        self.symbols = {}
        self.prices = {}
        self.data_count = 0
        self.last_rebalance = None

        self.Debug("=" * 50)
        self.Debug("Asset Shield V3.2.0 - Backtest Start")
        self.Debug("=" * 50)

        for ticker in self.TICKERS:
            try:
                security = self.AddData(JapanStockData, ticker, Resolution.DAILY)  # Fixed: DAILY not Daily
                self.symbols[ticker] = security.Symbol
                self.prices[ticker] = 0
                self.Debug(f"Added: {ticker}")
            except Exception as e:
                self.Error(f"Failed to add {ticker}: {e}")

        if len(self.symbols) == 0:
            raise Exception("No symbols loaded")

        self.Debug(f"Symbols loaded: {len(self.symbols)}")

    def OnData(self, slice):
        for ticker, symbol in self.symbols.items():
            if slice.ContainsKey(symbol):
                bar = slice[symbol]
                if bar and bar.Value > 0:
                    self.prices[ticker] = bar.Value
                    self.data_count += 1
                    self.Debug(f"[{self.Time.date()}] {ticker}: {bar.Value}")

        if self.data_count > 0:
            if self.last_rebalance is None or (self.Time - self.last_rebalance).days >= 30:
                self.Rebalance()
                self.last_rebalance = self.Time

    def Rebalance(self):
        equity = self.Portfolio.TotalPortfolioValue
        active = sum(1 for p in self.prices.values() if p > 0)
        self.Debug(f"REBALANCE | Equity: {equity} | Active: {active}")

        weight = 0.95 / len(self.TICKERS)
        for ticker, symbol in self.symbols.items():
            if self.prices[ticker] > 0:
                self.SetHoldings(symbol, weight)

    def OnEndOfAlgorithm(self):
        equity = self.Portfolio.TotalPortfolioValue
        ret = (equity / 10000000) - 1
        self.Debug("=" * 50)
        self.Debug(f"Data Points: {self.data_count}")
        self.Debug(f"Total Return: {ret:.2%}")
        self.Debug(f"Final Equity: {equity}")
        self.Debug("=" * 50)
