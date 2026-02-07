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
            return d
        except:
            return None
class AssetShield20(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(100000000)
        self.tickers = ["72030","67580","83060","68570","80350","70110","70120","79740","65010","99840","83160","65260","95010","68610","60980","40630","72670","87660","94320","91040"]
        self.symbols = {}
        self.prices = {}
        self.count = 0
        self.last_rebal = None
        for t in self.tickers:
            self.symbols[t] = self.AddData(JapanStockData, t, Resolution.Daily).Symbol
            self.prices[t] = 0
        self.Debug(f"Asset Shield 20-Stock: {len(self.tickers)} tickers")
    def OnData(self, slice):
        for t, sym in self.symbols.items():
            if slice.ContainsKey(sym) and slice[sym]:
                self.prices[t] = slice[sym].Value
                self.count += 1
        active = sum(1 for p in self.prices.values() if p > 0)
        if active >= 15 and (self.last_rebal is None or (self.Time - self.last_rebal).days >= 20):
            w = 0.9 / active
            for t, sym in self.symbols.items():
                if self.prices[t] > 0:
                    self.SetHoldings(sym, w)
            self.last_rebal = self.Time
            self.Debug(f"REBAL {self.Time.date()}: {active} stocks")
    def OnEndOfAlgorithm(self):
        ret = (self.Portfolio.TotalPortfolioValue / 100000000 - 1) * 100
        self.Debug(f"Points: {self.count}, Return: {ret:.2f}%")
