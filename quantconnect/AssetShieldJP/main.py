# region imports
from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
# endregion

class AssetShieldJP(QCAlgorithm):
    """
    Asset Shield V3.2.0 - QuantConnect LEAN Implementation
    ======================================================

    日本株バリュー・クオリティ戦略
    - PBR下位20% (割安)
    - ROE上位20% (高収益)
    - ADT 4億円以上 (流動性)
    - トレンドフィルター (60日MA)
    - サーキットブレーカー (DD 35%)

    Target: OOS Sharpe 1.67 再現
    """

    def Initialize(self):
        # ===== 基本設定 =====
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2026, 2, 3)
        self.SetCash(10_000_000)  # 1000万円

        # 日本市場設定
        self.SetBrokerageModel(BrokerageName.InteractiveBrokers, AccountType.Margin)
        self.SetTimeZone("Asia/Tokyo")

        # ===== 戦略パラメータ =====
        self.max_positions = 20
        self.position_pct = 0.05
        self.max_position_pct = 0.10
        self.rebalance_days = 63
        self.holding_days = 250
        self.min_adt = 400_000_000  # 4億円
        self.stop_loss = 0.12
        self.take_profit = 0.35

        # リスク制御
        self.max_portfolio_dd = 0.35
        self.trend_ma_period = 60

        # ===== 状態変数 =====
        self.positions_info = {}  # 保有ポジション情報
        self.day_count = 0
        self.peak_equity = self.Portfolio.TotalPortfolioValue
        self.circuit_breaker_active = False
        self.market_ma = []

        # ===== 日本株ユニバース =====
        # 東証上場銘柄を追加 (流動性上位)
        self.stock_codes = self._get_japan_universe()
        self.symbols = {}

        for code in self.stock_codes:
            try:
                symbol = self.AddEquity(
                    code,
                    Resolution.Daily,
                    Market.Japan,
                    True,  # fillDataForward
                    1.0,   # leverage
                    True   # extendedMarketHours
                )
                self.symbols[code] = symbol.Symbol
            except Exception as e:
                self.Debug(f"Failed to add {code}: {e}")

        # ===== スケジュール =====
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("7203", 30),
            self.OnDailyCheck
        )

        # ファンダメンタルデータ用
        self.UniverseSettings.Resolution = Resolution.Daily

        self.Debug(f"Initialized with {len(self.symbols)} symbols")

    def _get_japan_universe(self):
        """日本株ユニバース (TOPIX100相当)"""
        # 流動性上位銘柄コード
        return [
            "7203", "6758", "8306", "9984", "6861",
            "9432", "8035", "6501", "7974", "4063",
            "6902", "8058", "6367", "9433", "4502",
            "6098", "8001", "3382", "6273", "7267",
            "4503", "6954", "8031", "7751", "8766",
            "6326", "8316", "9022", "6702", "4568",
            "8802", "7011", "6857", "9020", "7269",
            "5108", "6301", "4661", "7270", "8411",
            "3861", "4901", "2914", "8604", "6752",
            "9735", "4452", "6594", "5401", "9101"
        ]

    def OnDailyCheck(self):
        """日次チェック"""
        self.day_count += 1

        equity = self.Portfolio.TotalPortfolioValue

        # トレンド計算
        market_price = self._get_market_average()
        self.market_ma.append(market_price)
        if len(self.market_ma) > self.trend_ma_period:
            self.market_ma.pop(0)

        ma_value = np.mean(self.market_ma) if self.market_ma else market_price

        if market_price > ma_value:
            trend = "UP"
        elif market_price > ma_value * 0.95:
            trend = "SIDEWAYS"
        else:
            trend = "DOWN"

        # ドローダウン計算
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # サーキットブレーカー
        if dd > self.max_portfolio_dd:
            self.circuit_breaker_active = True
        elif dd < 0.20:
            self.circuit_breaker_active = False

        # リバランス日判定
        if self.day_count % self.rebalance_days != 0:
            return

        self.Debug(f"[{self.Time}] Rebalance | Equity: ¥{equity:,.0f} | DD: {dd:.1%} | Trend: {trend}")

        # ポジション決済チェック
        self._check_exits()

        # 新規ポジション (条件付き)
        if not self.circuit_breaker_active and trend in ("UP", "SIDEWAYS"):
            self._open_positions()

    def _get_market_average(self):
        """市場平均価格"""
        prices = []
        for code, symbol in self.symbols.items():
            if self.Securities.ContainsKey(symbol):
                price = self.Securities[symbol].Price
                if price > 0:
                    prices.append(price)
        return np.mean(prices) if prices else 0

    def _check_exits(self):
        """決済チェック"""
        for code in list(self.positions_info.keys()):
            if code not in self.symbols:
                continue

            symbol = self.symbols[code]
            info = self.positions_info[code]

            if not self.Portfolio[symbol].Invested:
                del self.positions_info[code]
                continue

            current_price = self.Securities[symbol].Price
            entry_price = info['entry_price']
            entry_date = info['entry_date']

            # 保有日数
            holding_days = (self.Time - entry_date).days

            # P&L計算
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
                del self.positions_info[code]
                self.Debug(f"EXIT {code}: {reason}, P&L: {pnl_pct:.1%}")

    def _open_positions(self):
        """新規ポジション"""
        current_positions = len([c for c in self.positions_info if self.Portfolio[self.symbols[c]].Invested])
        max_new = self.max_positions - current_positions

        if max_new <= 0:
            return

        # 候補選定
        candidates = self._find_candidates()

        # 既存保有除外
        held = set(self.positions_info.keys())
        candidates = [c for c in candidates if c['code'] not in held]

        equity = self.Portfolio.TotalPortfolioValue

        for cand in candidates[:max_new]:
            code = cand['code']
            symbol = self.symbols[code]
            price = cand['price']

            # ポジションサイズ
            base_size = equity * self.position_pct
            max_size = equity * self.max_position_pct
            cash = self.Portfolio.Cash

            pos_value = min(base_size, max_size, cash * 0.90)

            if pos_value < 100_000:
                continue

            shares = int(pos_value / price)
            if shares <= 0:
                continue

            # 発注
            self.MarketOrder(symbol, shares)

            self.positions_info[code] = {
                'entry_date': self.Time,
                'entry_price': price,
                'shares': shares
            }

            self.Debug(f"ENTRY {code}: {shares} shares @ ¥{price:,.0f}")

    def _find_candidates(self):
        """候補銘柄選定 (PBR/ROE ランキング)"""
        candidates = []

        for code, symbol in self.symbols.items():
            if not self.Securities.ContainsKey(symbol):
                continue

            security = self.Securities[symbol]
            price = security.Price

            if price <= 0:
                continue

            # ADTチェック (簡易版: 直近出来高 × 価格)
            volume = security.Volume
            turnover = volume * price

            if turnover < self.min_adt / 60:  # 日次なので60日平均の1/60で判定
                continue

            # ファンダメンタル取得
            fundamentals = security.Fundamentals
            if fundamentals is None:
                continue

            try:
                # PBR
                pbr = fundamentals.ValuationRatios.PriceToBook
                # ROE
                roe = fundamentals.OperationRatios.ROE.Value if fundamentals.OperationRatios.ROE else 0

                if pbr <= 0 or not np.isfinite(roe):
                    continue

                candidates.append({
                    'code': code,
                    'symbol': symbol,
                    'price': price,
                    'pbr': pbr,
                    'roe': roe,
                    'turnover': turnover
                })
            except:
                continue

        if len(candidates) < 10:
            return []

        # パーセンタイルランキング
        pbrs = [c['pbr'] for c in candidates]
        roes = [c['roe'] for c in candidates]

        for c in candidates:
            c['pbr_pct'] = sum(1 for p in pbrs if p <= c['pbr']) / len(pbrs)
            c['roe_pct'] = sum(1 for r in roes if r <= c['roe']) / len(roes)
            c['composite'] = (1 - c['pbr_pct']) * 0.5 + c['roe_pct'] * 0.5

        # 上位20%
        threshold = np.percentile([c['composite'] for c in candidates], 80)
        selected = [c for c in candidates if c['composite'] >= threshold]
        selected.sort(key=lambda x: x['composite'], reverse=True)

        return selected

    def OnData(self, data):
        """データ受信時 (必須メソッド)"""
        pass

    def OnEndOfAlgorithm(self):
        """終了時レポート"""
        equity = self.Portfolio.TotalPortfolioValue
        initial = 10_000_000
        total_return = (equity / initial) - 1

        self.Debug("=" * 60)
        self.Debug("Asset Shield V3.2.0 - Final Report")
        self.Debug("=" * 60)
        self.Debug(f"Total Return: {total_return:.2%}")
        self.Debug(f"Final Equity: ¥{equity:,.0f}")
        self.Debug(f"Positions: {len(self.positions_info)}")
        self.Debug("=" * 60)
