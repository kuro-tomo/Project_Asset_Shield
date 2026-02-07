from AlgorithmImports import *
from datetime import datetime, timedelta
class JapanStockData(PythonData):
    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource("japan_stocks/" + config.Symbol.Value + ".csv", SubscriptionTransportMedium.ObjectStore)
    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or not line[0].isdigit():
            return None
        try:
            c = line.split(",")
            d = JapanStockData()
            d.Symbol = config.Symbol
            d.Time = datetime.strptime(c[0], "%Y-%m-%d")
            d.EndTime = d.Time + timedelta(days=1)
            d.Value = float(c[4])
            d["Volume"] = float(c[5])
            return d
        except:
            return None
class AssetShieldLive(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.tickers = ["72030","67580","83060","68570","80350","70110","70120","79740","65010","99840","83160","65260","95010","68610","60980","40630","72670","87660","94320","91040"]
        self.sectors = ["06","09","15","09","09","08","06","09","09","10","15","09","11","09","10","04","06","16","10","12"]
        self.MAX_SECTOR = 0.25
        self.symbols = {}
        self.prices = {}
        self.volumes = {}
        self.sector_map = {}
        for i in range(len(self.tickers)):
            t = self.tickers[i]
            self.symbols[t] = self.AddData(JapanStockData, t, Resolution.Daily).Symbol
            self.prices[t] = []
            self.volumes[t] = []
            self.sector_map[t] = self.sectors[i]
        self.last_rebal = None
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(9, 30), self.DailyCheck)
        self.Debug("Asset Shield LIVE: " + str(len(self.tickers)) + " stocks")
    def DailyCheck(self):
        if self.last_rebal is None or (self.Time - self.last_rebal).days >= 20:
            active = sum(1 for t in self.tickers if len(self.prices[t]) >= 20)
            if active >= 15:
                self.Rebalance()
                self.last_rebal = self.Time
    def OnData(self, slice):
        for t in self.tickers:
            sym = self.symbols[t]
            if slice.ContainsKey(sym) and slice[sym]:
                bar = slice[sym]
                self.prices[t].append(bar.Value)
                self.volumes[t].append(bar["Volume"])
                if len(self.prices[t]) > 60:
                    self.prices[t] = self.prices[t][-60:]
                    self.volumes[t] = self.volumes[t][-60:]
    def CalcAlpha(self, t):
        if len(self.prices[t]) < 20:
            return 0
        p = self.prices[t]
        mom = (p[-1] / p[-20] - 1) if p[-20] > 0 else 0
        rev = -(p[-1] / p[-5] - 1) if len(p) >= 5 and p[-5] > 0 else 0
        return mom * 0.6 + rev * 0.4
    def CalcADT(self, t):
        if len(self.prices[t]) < 20:
            return 0
        total = 0
        for i in range(-20, 0):
            total += self.prices[t][i] * self.volumes[t][i]
        return total / 20
    def Rebalance(self):
        equity = self.Portfolio.TotalPortfolioValue
        alphas = []
        for t in self.tickers:
            a = self.CalcAlpha(t)
            adt = self.CalcADT(t)
            if adt > 0:
                alphas.append((t, a, adt))
        alphas.sort(key=lambda x: x[1], reverse=True)
        sector_exp = {}
        selected = []
        base_w = 0.9 / len(alphas) if alphas else 0
        for t, a, adt in alphas:
            sec = self.sector_map[t]
            cur = sector_exp.get(sec, 0)
            if cur + base_w <= self.MAX_SECTOR:
                max_w = (adt * 0.05) / equity if equity > 0 else 0
                w = min(base_w, max_w)
                if w > 0.01:
                    selected.append((t, w))
                    sector_exp[sec] = cur + w
        total_w = sum(w for _, w in selected)
        if total_w > 0:
            selected = [(t, w * 0.9 / total_w) for t, w in selected]
        held = [t for t in self.tickers if self.Portfolio[self.symbols[t]].Invested]
        new_t = [t for t, _ in selected]
        for t in held:
            if t not in new_t:
                self.Liquidate(self.symbols[t])
        for t, w in selected:
            self.SetHoldings(self.symbols[t], w)
        self.Log("LIVE REBAL " + str(self.Time.date()) + ": " + str(len(selected)) + " stocks, equity=" + str(int(equity)))
    def OnEndOfAlgorithm(self):
        self.Log("Asset Shield LIVE ended")
