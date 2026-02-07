from AlgorithmImports import *
from datetime import datetime, timedelta
class JapanStockData(PythonData):
    def GetSource(self, config, date, isLiveMode):
        ticker = config.Symbol.Value
        return SubscriptionDataSource(f"japan_stocks/{ticker}.csv", SubscriptionTransportMedium.ObjectStore)
    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or not line[0].isdigit():
            return None
        try:
            c = line.split(',')
            d = JapanStockData()
            d.Symbol = config.Symbol
            d.Time = datetime.strptime(c[0], "%Y-%m-%d")
            d.EndTime = d.Time + timedelta(days=1)
            d.Value = float(c[4])
            d["Open"] = float(c[1])
            d["High"] = float(c[2])
            d["Low"] = float(c[3])
            d["Volume"] = int(float(c[5]))
            return d
        except:
            return None
class AssetShield3Stocks(QCAlgorithm):
    TICKERS = ["72030", "67580", "83060"]
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(10000000)
        self.symbols = {}
        self.prices = {}
        self.data_count = 0
        self.last_rebalance = None
        self.Debug("Asset Shield V3.2.0 - 3 Stock Test")
        for ticker in self.TICKERS:
            security = self.AddData(JapanStockData, ticker, Resolution.Daily)
            self.symbols[ticker] = security.Symbol
            self.prices[ticker] = 0
            self.Debug(f"Added: {ticker}")
    def OnData(self, slice):
        for ticker, symbol in self.symbols.items():
            if slice.ContainsKey(symbol):
                bar = slice[symbol]
                if bar and bar.Value > 0:
                    self.prices[ticker] = bar.Value
                    self.data_count += 1
        if self.data_count > 0 and (self.last_rebalance is None or (self.Time - self.last_rebalance).days >= 30):
            self.Rebalance()
            self.last_rebalance = self.Time
    def Rebalance(self):
        weight = 0.3
        for ticker, symbol in self.symbols.items():
            if self.prices[ticker] > 0:
                self.SetHoldings(symbol, weight)
        self.Debug(f"REBALANCE at {self.Time.date()}")
    def OnEndOfAlgorithm(self):
        equity = self.Portfolio.TotalPortfolioValue
        ret = (equity / 10000000) - 1
        self.Debug(f"Data Points: {self.data_count}")
        self.Debug(f"Return: {ret:.2%}")
        self.Debug(f"Final: {equity}")
