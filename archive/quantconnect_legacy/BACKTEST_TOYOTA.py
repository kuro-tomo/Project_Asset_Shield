from AlgorithmImports import *
from datetime import datetime, timedelta
class JapanStockData(PythonData):
    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource("japan_stocks/72030.csv", SubscriptionTransportMedium.ObjectStore)
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
class ToyotaBacktest(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(10000000)
        self.toyota = self.AddData(JapanStockData, "TOYOTA", Resolution.Daily).Symbol
        self.count = 0
        self.Debug("Toyota Backtest Start")
    def OnData(self, slice):
        if slice.ContainsKey(self.toyota):
            bar = slice[self.toyota]
            if bar and bar.Value > 0:
                self.count += 1
                if not self.Portfolio[self.toyota].Invested:
                    self.SetHoldings(self.toyota, 0.95)
                    self.Debug(f"BUY at {bar.Value}")
    def OnEndOfAlgorithm(self):
        self.Debug(f"Data points: {self.count}")
        self.Debug(f"Final: {self.Portfolio.TotalPortfolioValue}")
