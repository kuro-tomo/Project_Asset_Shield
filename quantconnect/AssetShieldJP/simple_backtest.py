from AlgorithmImports import *

class JapanStockData(PythonData):

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "japan_stocks/" + config.Symbol.Value + ".csv",
            SubscriptionTransportMedium.ObjectStore
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip():
            return None
        if not line[0].isdigit():
            return None

        index = JapanStockData()
        index.Symbol = config.Symbol

        data = line.split(",")
        index.Time = datetime.strptime(data[0], "%Y-%m-%d")
        index.Value = float(data[4])
        index["Open"] = float(data[1])
        index["High"] = float(data[2])
        index["Low"] = float(data[3])
        index["Close"] = float(data[4])
        index["Volume"] = float(data[5])

        return index


class SimpleBacktest(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetCash(10000000)

        self.tickers = ["7203", "6758", "8306"]
        self.symbols = {}
        self.dataCount = 0

        for ticker in self.tickers:
            self.symbols[ticker] = self.AddData(JapanStockData, ticker, Resolution.Daily).Symbol
            self.Debug("Added: " + ticker)

    def OnData(self, data):
        for ticker in self.tickers:
            symbol = self.symbols[ticker]
            if data.ContainsKey(symbol):
                bar = data[symbol]
                if bar is not None:
                    self.dataCount = self.dataCount + 1
                    self.Debug(str(self.Time.date()) + " " + ticker + ": " + str(bar.Value))

                    if not self.Portfolio[symbol].Invested:
                        self.SetHoldings(symbol, 0.3)

    def OnEndOfAlgorithm(self):
        self.Debug("=== FINAL RESULTS ===")
        self.Debug("Data Points: " + str(self.dataCount))
        self.Debug("Final Equity: " + str(self.Portfolio.TotalPortfolioValue))
