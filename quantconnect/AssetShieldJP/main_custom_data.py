# region imports
from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
# endregion

class JapanEquityData(PythonData):
    """
    カスタムデータクラス: J-Quantsからの日本株データ
    QuantConnectにアップロードしたCSVを読み込む
    """

    def GetSource(self, config, date, isLiveMode):
        """データソースを指定"""
        # カスタムデータのURL/パス
        # ObjectStoreまたはDropbox/GitHub経由でCSVを提供
        ticker = config.Symbol.Value
        source = f"https://raw.githubusercontent.com/YOUR_REPO/japan_stocks/{ticker}.csv"

        return SubscriptionDataSource(
            source,
            SubscriptionTransportMedium.RemoteFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        """CSVの各行を解析"""
        if not (line.strip() and line[0].isdigit()):
            return None

        data = JapanEquityData()
        data.Symbol = config.Symbol

        try:
            cols = line.split(',')
            # 形式: Date,Open,High,Low,Close,Volume
            data.Time = datetime.strptime(cols[0], "%Y-%m-%d")
            data.Value = float(cols[4])  # Close
            data["Open"] = float(cols[1])
            data["High"] = float(cols[2])
            data["Low"] = float(cols[3])
            data["Close"] = float(cols[4])
            data["Volume"] = float(cols[5])
        except:
            return None

        return data


class AssetShieldJPCustomData(QCAlgorithm):
    """
    Asset Shield V3.2.0 - カスタムデータ版
    =========================================

    J-Quantsデータをカスタムデータとしてアップロード
    """

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)  # OOS期間
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10_000_000)

        # パラメータ
        self.max_positions = 20
        self.position_pct = 0.05
        self.rebalance_days = 63

        # カスタムデータ銘柄を追加
        self.symbols = {}
        self.stock_codes = ["72030", "67580", "83060", "80350", "68610",
                           "94320", "65010", "79740", "40630", "99840"]

        for code in self.stock_codes:
            symbol = self.AddData(JapanEquityData, code, Resolution.Daily)
            self.symbols[code] = symbol.Symbol
            self.Debug(f"Added custom data: {code}")

        # データ検証
        if len(self.symbols) == 0:
            raise Exception("ERROR: No symbols added. Check custom data URLs.")

        self.day_count = 0
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(9, 30),
            self.OnDailyRebalance
        )

    def OnDailyRebalance(self):
        """日次リバランス"""
        self.day_count += 1

        if self.day_count % self.rebalance_days != 0:
            return

        # 候補選定とポジション調整
        self._rebalance()

    def _rebalance(self):
        """リバランス実行"""
        candidates = []

        for code, symbol in self.symbols.items():
            if not self.Securities.ContainsKey(symbol):
                continue

            price = self.Securities[symbol].Price
            if price <= 0:
                continue

            candidates.append({'code': code, 'symbol': symbol, 'price': price})

        if len(candidates) == 0:
            self.Debug("No valid candidates")
            return

        # 均等配分
        weight = 1.0 / min(len(candidates), self.max_positions)

        for cand in candidates[:self.max_positions]:
            self.SetHoldings(cand['symbol'], weight)
            self.Debug(f"SET {cand['code']}: {weight:.1%}")

    def OnData(self, data):
        pass

    def OnEndOfAlgorithm(self):
        equity = self.Portfolio.TotalPortfolioValue
        self.Debug(f"Final Equity: ¥{equity:,.0f}")
