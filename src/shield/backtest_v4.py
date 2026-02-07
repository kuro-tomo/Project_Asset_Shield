"""
Asset Shield V4.0 - Dominance Phase Backtest Engine
20-Year Historical Validation with HMM Regime Detection

Features:
1. HMM Market Regime Detection (Bull/Bear/Crisis)
2. Dynamic Risk Parameters based on regime
3. Kill-switch with correlation spike detection
4. Sector Neutralization (25% max per sector)
5. ADT Liquidity Guard (5% of 20-day ADT)
6. Comprehensive stress test reporting

Author: Asset Shield V4 Team
Version: 4.0.0 (2026-02-05)
"""

import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from shield.regime_hmm import (
    HMMRegimeDetector, KillSwitch, MarketRegime,
    RegimeState, RegimeRiskParams, REGIME_PARAMS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """V4.0 Backtest Configuration"""
    # Time period
    start_date: date = field(default_factory=lambda: date(2006, 1, 1))
    end_date: date = field(default_factory=lambda: date(2026, 2, 1))

    # Capital
    initial_capital: float = 100_000_000  # 1億円

    # Position constraints
    max_positions: int = 20
    min_positions: int = 10
    max_single_weight: float = 0.10
    max_sector_weight: float = 0.25

    # Liquidity constraints
    max_adt_participation: float = 0.05  # 5% of ADT
    min_adt: float = 100_000_000  # 1億円 minimum ADT

    # Regime parameters
    vol_threshold_low: float = 0.15
    vol_threshold_high: float = 0.35

    # Rebalance
    rebalance_frequency: int = 20  # days

    # Kill switch
    max_drawdown: float = 0.15
    correlation_threshold: float = 0.8


@dataclass
class DailyState:
    """Daily portfolio state"""
    date: date
    equity: float
    cash: float
    positions: Dict[str, float]  # code -> value
    weights: Dict[str, float]    # code -> weight
    regime: MarketRegime
    regime_prob: float
    volatility: float
    drawdown: float
    kill_switch_active: bool


@dataclass
class TradeRecord:
    """Trade execution record"""
    date: date
    code: str
    side: str  # BUY/SELL
    shares: int
    price: float
    value: float
    reason: str


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_date: date
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_monthly_return: float
    monthly_std: float
    best_month: float
    worst_month: float
    positive_months: int
    negative_months: int
    regime_stats: Dict[str, Any]


@dataclass
class StressTestResult:
    """Stress test period result"""
    period_name: str
    start_date: date
    end_date: date
    return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    regime_distribution: Dict[str, float]
    kill_switch_triggers: int
    description: str


class AssetShieldV4Backtest:
    """
    Asset Shield V4.0 Backtest Engine

    Implements full 20-year backtest with:
    - HMM regime detection
    - Dynamic risk management
    - Kill-switch protection
    - Sector neutralization
    - ADT liquidity constraints
    """

    # TSE Sector17 mapping (sector33 -> sector17)
    SECTOR33_TO_17 = {
        '0050': '01', '1050': '02', '2050': '03', '3050': '04', '3100': '04',
        '3150': '04', '3200': '04', '3250': '04', '3300': '04', '3350': '04',
        '3400': '04', '3450': '04', '3500': '05', '3550': '06', '3600': '08',
        '3650': '09', '3700': '06', '3750': '06', '3800': '09', '4050': '11',
        '5050': '12', '5100': '12', '5150': '12', '5200': '12', '5250': '10',
        '6050': '13', '6100': '14', '7050': '15', '7100': '16', '7150': '16',
        '7200': '16', '8050': '17', '9050': '10'
    }

    SECTOR17_NAMES = {
        '01': 'Foods', '02': 'Energy', '03': 'Construction',
        '04': 'Materials', '05': 'Pharma', '06': 'Auto',
        '07': 'Steel', '08': 'Machinery', '09': 'Electronics',
        '10': 'IT/Services', '11': 'Utilities', '12': 'Transport',
        '13': 'Wholesale', '14': 'Retail', '15': 'Banks',
        '16': 'Finance', '17': 'Real Estate'
    }

    def __init__(
        self,
        config: BacktestConfig,
        db_path: str = "/Users/MBP/Desktop/Project_Asset_Shield/data/jquants_cache.db"
    ):
        """Initialize V4.0 Backtest Engine"""
        self.config = config
        self.db_path = db_path

        # Initialize components
        self.regime_detector = HMMRegimeDetector(
            vol_threshold_low=config.vol_threshold_low,
            vol_threshold_high=config.vol_threshold_high
        )
        self.kill_switch = KillSwitch(
            max_drawdown=config.max_drawdown,
            correlation_threshold=config.correlation_threshold
        )

        # State tracking
        self.equity_curve: List[Tuple[date, float]] = []
        self.daily_states: List[DailyState] = []
        self.trades: List[TradeRecord] = []
        self.monthly_returns: List[Tuple[str, float]] = []

        # Data cache
        self._price_data: Dict[str, pd.DataFrame] = {}
        self._sector_map: Dict[str, str] = {}
        self._universe: List[str] = []

        logger.info(f"AssetShieldV4 initialized: {config.start_date} to {config.end_date}")

    def load_data(self) -> None:
        """Load historical data from database"""
        logger.info("Loading data from database...")

        conn = sqlite3.connect(self.db_path)

        # Load universe with sectors
        universe_df = pd.read_sql_query("""
            SELECT DISTINCT code, sector33_code
            FROM universe_snapshots
            WHERE is_active = 1
        """, conn)

        for _, row in universe_df.iterrows():
            code = row['code']
            sector33 = row['sector33_code']
            sector17 = self.SECTOR33_TO_17.get(sector33, '10')  # Default to IT/Services
            self._sector_map[code] = sector17

        self._universe = list(self._sector_map.keys())
        logger.info(f"Loaded {len(self._universe)} stocks in universe")

        # Load price data
        start_str = self.config.start_date.strftime('%Y-%m-%d')
        end_str = self.config.end_date.strftime('%Y-%m-%d')

        price_df = pd.read_sql_query(f"""
            SELECT code, date,
                   adjustment_open as open,
                   adjustment_high as high,
                   adjustment_low as low,
                   adjustment_close as close,
                   volume
            FROM daily_quotes
            WHERE date >= '{start_str}' AND date <= '{end_str}'
              AND adjustment_close > 0
            ORDER BY date
        """, conn)

        conn.close()

        # Organize by code
        for code in price_df['code'].unique():
            stock_data = price_df[price_df['code'] == code].copy()
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.set_index('date', inplace=True)
            stock_data = stock_data.sort_index()
            self._price_data[code] = stock_data

        logger.info(f"Loaded price data for {len(self._price_data)} stocks")

    def get_price(self, code: str, dt: date) -> Optional[float]:
        """Get closing price for a stock on a date"""
        if code not in self._price_data:
            return None
        df = self._price_data[code]
        ts = pd.Timestamp(dt)
        if ts in df.index:
            return df.loc[ts, 'close']
        # Find nearest previous date
        prior = df[df.index <= ts]
        if len(prior) > 0:
            return prior.iloc[-1]['close']
        return None

    def get_adt(self, code: str, dt: date, window: int = 20) -> float:
        """Calculate Average Daily Turnover"""
        if code not in self._price_data:
            return 0.0
        df = self._price_data[code]
        ts = pd.Timestamp(dt)
        prior = df[df.index <= ts].tail(window)
        if len(prior) < window // 2:
            return 0.0
        turnover = (prior['close'] * prior['volume']).mean()
        return turnover if not np.isnan(turnover) else 0.0

    def get_market_index(self, dt: date, lookback: int = 60) -> pd.Series:
        """Get market index (equal-weight of top stocks) for regime detection"""
        # Use top 50 stocks by ADT as market proxy
        ts = pd.Timestamp(dt)
        start_ts = ts - pd.Timedelta(days=lookback * 2)

        prices_list = []
        for code in list(self._price_data.keys())[:100]:  # Sample for speed
            df = self._price_data[code]
            subset = df[(df.index >= start_ts) & (df.index <= ts)]['close']
            if len(subset) >= lookback:
                prices_list.append(subset)

        if not prices_list:
            return pd.Series()

        # Equal-weight index
        combined = pd.concat(prices_list, axis=1)
        combined = combined.dropna(how='all')
        if len(combined) == 0:
            return pd.Series()

        # Normalize and average
        normalized = combined.div(combined.iloc[0])
        index = normalized.mean(axis=1)
        return index

    def calculate_alpha(self, code: str, dt: date) -> float:
        """Calculate alpha signal (momentum + mean reversion)"""
        if code not in self._price_data:
            return 0.0

        df = self._price_data[code]
        ts = pd.Timestamp(dt)
        prior = df[df.index <= ts].tail(60)

        if len(prior) < 20:
            return 0.0

        prices = prior['close'].values

        # 20-day momentum
        if len(prices) >= 20 and prices[-20] > 0:
            mom_20d = (prices[-1] / prices[-20]) - 1
        else:
            mom_20d = 0.0

        # 5-day mean reversion
        if len(prices) >= 5 and prices[-5] > 0:
            rev_5d = -((prices[-1] / prices[-5]) - 1)
        else:
            rev_5d = 0.0

        # Combined alpha
        alpha = mom_20d * 0.6 + rev_5d * 0.4
        return alpha

    def select_portfolio(
        self,
        dt: date,
        current_equity: float,
        risk_params: RegimeRiskParams
    ) -> List[Tuple[str, float]]:
        """
        Select portfolio with sector neutralization and ADT constraints.

        Returns:
            List of (code, target_weight) tuples
        """
        # Calculate alpha and ADT for all stocks
        candidates = []
        for code in self._universe:
            alpha = self.calculate_alpha(code, dt)
            adt = self.get_adt(code, dt)
            price = self.get_price(code, dt)

            if adt >= self.config.min_adt and price is not None and price > 0:
                sector = self._sector_map.get(code, '10')
                candidates.append({
                    'code': code,
                    'alpha': alpha,
                    'adt': adt,
                    'price': price,
                    'sector': sector
                })

        if not candidates:
            return []

        # Sort by alpha
        candidates.sort(key=lambda x: x['alpha'], reverse=True)

        # Apply constraints
        selected = []
        sector_weights = defaultdict(float)
        total_weight = 0.0

        max_exposure = risk_params.max_exposure
        max_single = min(risk_params.max_single_position, self.config.max_single_weight)
        max_sector = min(risk_params.max_sector_exposure, self.config.max_sector_weight)

        base_weight = max_exposure / self.config.max_positions

        for c in candidates:
            if len(selected) >= self.config.max_positions:
                break

            code = c['code']
            sector = c['sector']

            # Check sector constraint
            if sector_weights[sector] + base_weight > max_sector:
                continue

            # Check ADT constraint
            max_adt_value = c['adt'] * self.config.max_adt_participation
            max_adt_weight = max_adt_value / current_equity if current_equity > 0 else 0

            weight = min(base_weight, max_single, max_adt_weight)

            if weight < 0.01:  # Minimum 1% position
                continue

            if total_weight + weight > max_exposure:
                weight = max_exposure - total_weight

            selected.append((code, weight))
            sector_weights[sector] += weight
            total_weight += weight

        return selected

    def run_backtest(self) -> PerformanceMetrics:
        """Run full backtest"""
        logger.info("Starting V4.0 backtest...")

        # Load data if not already loaded
        if not self._price_data:
            self.load_data()

        # Initialize state
        equity = self.config.initial_capital
        cash = equity
        positions: Dict[str, int] = {}  # code -> shares
        position_values: Dict[str, float] = {}

        last_rebalance = None
        high_water_mark = equity

        # Get trading days
        all_dates = set()
        for df in self._price_data.values():
            all_dates.update(df.index.date)
        trading_days = sorted([d for d in all_dates
                              if self.config.start_date <= d <= self.config.end_date])

        logger.info(f"Trading days: {len(trading_days)}")

        # Main loop
        for i, current_date in enumerate(trading_days):
            # Update positions values
            position_values = {}
            for code, shares in positions.items():
                price = self.get_price(code, current_date)
                if price is not None:
                    position_values[code] = shares * price

            total_positions = sum(position_values.values())
            equity = cash + total_positions

            # Update high water mark
            if equity > high_water_mark:
                high_water_mark = equity

            drawdown = (high_water_mark - equity) / high_water_mark if high_water_mark > 0 else 0

            # Get market index for regime detection
            market_index = self.get_market_index(current_date)

            # Update regime
            regime_state = self.regime_detector.update(
                market_index,
                pd.Timestamp(current_date)
            )

            # Check kill switch
            kill_active, kill_reason = self.kill_switch.check_trigger(
                equity, pd.Timestamp(current_date), regime_state
            )

            # Get risk parameters
            risk_params = self.regime_detector.get_risk_params()

            # Rebalance logic
            should_rebalance = False
            if kill_active:
                # Emergency exit - liquidate all
                should_rebalance = True
                risk_params = REGIME_PARAMS[MarketRegime.CRISIS]
                logger.warning(f"{current_date}: Kill switch triggered - {kill_reason}")
            elif last_rebalance is None:
                should_rebalance = True
            elif (current_date - last_rebalance).days >= risk_params.rebalance_frequency:
                should_rebalance = True

            if should_rebalance:
                # Select new portfolio
                target_portfolio = self.select_portfolio(current_date, equity, risk_params)

                # Execute trades
                target_codes = set(code for code, _ in target_portfolio)
                current_codes = set(positions.keys())

                # Sell positions not in target
                for code in current_codes - target_codes:
                    if code in positions and positions[code] > 0:
                        price = self.get_price(code, current_date)
                        if price:
                            shares = positions[code]
                            value = shares * price
                            cash += value
                            self.trades.append(TradeRecord(
                                date=current_date, code=code, side='SELL',
                                shares=shares, price=price, value=value,
                                reason='Rebalance exit'
                            ))
                            del positions[code]

                # Buy new/adjust positions
                for code, target_weight in target_portfolio:
                    target_value = equity * target_weight
                    price = self.get_price(code, current_date)
                    if price is None or price <= 0:
                        continue

                    current_shares = positions.get(code, 0)
                    current_value = current_shares * price
                    delta_value = target_value - current_value

                    if abs(delta_value) < equity * 0.005:  # Skip small trades
                        continue

                    if delta_value > 0:
                        # Buy
                        shares_to_buy = int(delta_value / price)
                        if shares_to_buy > 0 and cash >= shares_to_buy * price:
                            cost = shares_to_buy * price
                            cash -= cost
                            positions[code] = current_shares + shares_to_buy
                            self.trades.append(TradeRecord(
                                date=current_date, code=code, side='BUY',
                                shares=shares_to_buy, price=price, value=cost,
                                reason='Rebalance entry'
                            ))
                    else:
                        # Sell partial
                        shares_to_sell = min(current_shares, int(abs(delta_value) / price))
                        if shares_to_sell > 0:
                            proceeds = shares_to_sell * price
                            cash += proceeds
                            positions[code] = current_shares - shares_to_sell
                            if positions[code] == 0:
                                del positions[code]
                            self.trades.append(TradeRecord(
                                date=current_date, code=code, side='SELL',
                                shares=shares_to_sell, price=price, value=proceeds,
                                reason='Rebalance reduce'
                            ))

                last_rebalance = current_date

            # Record daily state
            weights = {code: position_values.get(code, 0) / equity if equity > 0 else 0
                      for code in positions}

            state = DailyState(
                date=current_date,
                equity=equity,
                cash=cash,
                positions=position_values.copy(),
                weights=weights,
                regime=regime_state.regime,
                regime_prob=regime_state.probability,
                volatility=regime_state.volatility_20d,
                drawdown=drawdown,
                kill_switch_active=kill_active
            )
            self.daily_states.append(state)
            self.equity_curve.append((current_date, equity))

            # Progress logging
            if i % 500 == 0:
                logger.info(f"Progress: {current_date} | Equity: ¥{equity:,.0f} | Regime: {regime_state.regime.value}")

        # Calculate performance metrics
        metrics = self._calculate_metrics()
        return metrics

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            raise ValueError("No equity curve data")

        dates = [d for d, _ in self.equity_curve]
        equities = [e for _, e in self.equity_curve]

        initial = self.config.initial_capital
        final = equities[-1]

        # Total return
        total_return = (final / initial) - 1

        # CAGR
        years = (dates[-1] - dates[0]).days / 365.25
        cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0

        # Daily returns
        daily_returns = pd.Series(equities).pct_change().dropna()

        # Sharpe Ratio (assuming 0% risk-free rate for JPY)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (daily_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))
        else:
            sortino = 0.0

        # Max Drawdown
        peak = pd.Series(equities).expanding().max()
        drawdown = (peak - pd.Series(equities)) / peak
        max_dd = drawdown.max()
        max_dd_idx = drawdown.idxmax()
        max_dd_date = dates[max_dd_idx] if max_dd_idx < len(dates) else dates[-1]

        # Calmar Ratio
        calmar = cagr / max_dd if max_dd > 0 else 0

        # Monthly returns
        equity_df = pd.DataFrame({'date': dates, 'equity': equities})
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        monthly = equity_df['equity'].resample('M').last().pct_change().dropna()

        self.monthly_returns = [(str(d.date()), r) for d, r in monthly.items()]

        avg_monthly = monthly.mean()
        monthly_std = monthly.std()
        best_month = monthly.max()
        worst_month = monthly.min()
        positive_months = (monthly > 0).sum()
        negative_months = (monthly < 0).sum()

        # Win rate (trade-based)
        if self.trades:
            # Simple PnL estimation per trade is complex; use monthly as proxy
            win_rate = positive_months / len(monthly) if len(monthly) > 0 else 0
        else:
            win_rate = 0

        # Profit factor
        gains = monthly[monthly > 0].sum()
        losses = abs(monthly[monthly < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        # Regime stats
        regime_stats = self.regime_detector.get_regime_stats()

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_date=max_dd_date,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_monthly_return=avg_monthly,
            monthly_std=monthly_std,
            best_month=best_month,
            worst_month=worst_month,
            positive_months=positive_months,
            negative_months=negative_months,
            regime_stats=regime_stats
        )

    def run_stress_tests(self) -> List[StressTestResult]:
        """Run stress tests on specific historical periods"""
        stress_periods = [
            ("2008 GFC", date(2008, 1, 1), date(2009, 12, 31),
             "Global Financial Crisis - Lehman Shock"),
            ("2011 Earthquake", date(2011, 3, 1), date(2011, 12, 31),
             "Tohoku Earthquake and Fukushima"),
            ("2020 COVID", date(2020, 1, 1), date(2020, 12, 31),
             "COVID-19 Pandemic Crash and Recovery"),
            ("2022 Rate Hike", date(2022, 1, 1), date(2022, 12, 31),
             "Global Rate Hike Cycle"),
        ]

        results = []
        for name, start, end, desc in stress_periods:
            result = self._analyze_period(name, start, end, desc)
            if result:
                results.append(result)

        return results

    def _analyze_period(
        self,
        name: str,
        start: date,
        end: date,
        description: str
    ) -> Optional[StressTestResult]:
        """Analyze a specific period"""
        period_states = [s for s in self.daily_states if start <= s.date <= end]

        if len(period_states) < 10:
            return None

        equities = [s.equity for s in period_states]
        start_equity = equities[0]
        end_equity = equities[-1]

        # Return
        period_return = (end_equity / start_equity) - 1

        # Max drawdown in period
        peak = start_equity
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe for period
        daily_rets = pd.Series(equities).pct_change().dropna()
        if len(daily_rets) > 0 and daily_rets.std() > 0:
            sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252))
        else:
            sharpe = 0

        # Regime distribution
        regime_counts = defaultdict(int)
        for s in period_states:
            regime_counts[s.regime.value] += 1
        total = len(period_states)
        regime_dist = {k: v/total for k, v in regime_counts.items()}

        # Kill switch triggers
        kill_triggers = sum(1 for i in range(1, len(period_states))
                          if period_states[i].kill_switch_active and
                          not period_states[i-1].kill_switch_active)

        return StressTestResult(
            period_name=name,
            start_date=start,
            end_date=end,
            return_pct=period_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            regime_distribution=regime_dist,
            kill_switch_triggers=kill_triggers,
            description=description
        )

    def generate_report(self) -> str:
        """Generate comprehensive backtest report"""
        metrics = self._calculate_metrics()
        stress_results = self.run_stress_tests()

        report = []
        report.append("=" * 70)
        report.append("ASSET SHIELD V4.0 - DOMINANCE PHASE BACKTEST REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Period: {self.config.start_date} to {self.config.end_date}")
        report.append(f"Initial Capital: ¥{self.config.initial_capital:,.0f}")
        report.append(f"Final Equity: ¥{self.equity_curve[-1][1]:,.0f}")
        report.append("")

        report.append("-" * 70)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 70)
        report.append(f"Total Return:      {metrics.total_return:>12.2%}")
        report.append(f"CAGR:              {metrics.cagr:>12.2%}")
        report.append(f"Sharpe Ratio:      {metrics.sharpe_ratio:>12.2f}")
        report.append(f"Sortino Ratio:     {metrics.sortino_ratio:>12.2f}")
        report.append(f"Max Drawdown:      {metrics.max_drawdown:>12.2%}")
        report.append(f"Max DD Date:       {metrics.max_drawdown_date}")
        report.append(f"Calmar Ratio:      {metrics.calmar_ratio:>12.2f}")
        report.append(f"Win Rate:          {metrics.win_rate:>12.2%}")
        report.append(f"Profit Factor:     {metrics.profit_factor:>12.2f}")
        report.append("")

        report.append("-" * 70)
        report.append("MONTHLY STATISTICS")
        report.append("-" * 70)
        report.append(f"Avg Monthly Return: {metrics.avg_monthly_return:>11.2%}")
        report.append(f"Monthly Std Dev:    {metrics.monthly_std:>11.2%}")
        report.append(f"Best Month:         {metrics.best_month:>11.2%}")
        report.append(f"Worst Month:        {metrics.worst_month:>11.2%}")
        report.append(f"Positive Months:    {metrics.positive_months:>11}")
        report.append(f"Negative Months:    {metrics.negative_months:>11}")
        report.append("")

        report.append("-" * 70)
        report.append("REGIME STATISTICS")
        report.append("-" * 70)
        rs = metrics.regime_stats
        if rs:
            report.append(f"Total Days:        {rs.get('total_days', 0):>12}")
            report.append(f"Bull Days:         {rs.get('bull_days', 0):>12} ({rs.get('bull_pct', 0):.1%})")
            report.append(f"Bear Days:         {rs.get('bear_days', 0):>12} ({rs.get('bear_pct', 0):.1%})")
            report.append(f"Crisis Days:       {rs.get('crisis_days', 0):>12} ({rs.get('crisis_pct', 0):.1%})")
            report.append(f"Regime Changes:    {rs.get('regime_changes', 0):>12}")
            report.append(f"Avg Volatility:    {rs.get('avg_vol_20d', 0):>11.1%}")
        report.append("")

        report.append("-" * 70)
        report.append("STRESS TEST RESULTS")
        report.append("-" * 70)
        for sr in stress_results:
            report.append(f"\n{sr.period_name}: {sr.description}")
            report.append(f"  Period: {sr.start_date} to {sr.end_date}")
            report.append(f"  Return: {sr.return_pct:>10.2%}")
            report.append(f"  Max DD: {sr.max_drawdown:>10.2%}")
            report.append(f"  Sharpe: {sr.sharpe_ratio:>10.2f}")
            report.append(f"  Kill Switch Triggers: {sr.kill_switch_triggers}")
            regime_str = ", ".join([f"{k}:{v:.0%}" for k, v in sr.regime_distribution.items()])
            report.append(f"  Regimes: {regime_str}")

        report.append("")
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)


def run_v4_backtest():
    """Main entry point for V4.0 backtest"""
    config = BacktestConfig(
        start_date=date(2006, 1, 1),
        end_date=date(2026, 2, 1),
        initial_capital=100_000_000,
        max_positions=20,
        max_sector_weight=0.25,
        max_adt_participation=0.05,
        vol_threshold_low=0.15,
        vol_threshold_high=0.35,
        max_drawdown=0.15
    )

    engine = AssetShieldV4Backtest(config)
    engine.load_data()

    logger.info("Running V4.0 backtest...")
    metrics = engine.run_backtest()

    report = engine.generate_report()
    print(report)

    # Save report
    report_path = "/Users/MBP/Desktop/Project_Asset_Shield/output/v4_backtest_report.txt"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Save equity curve
    equity_df = pd.DataFrame(engine.equity_curve, columns=['date', 'equity'])
    equity_path = "/Users/MBP/Desktop/Project_Asset_Shield/output/v4_equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"Equity curve saved to {equity_path}")

    return metrics, engine


if __name__ == "__main__":
    run_v4_backtest()
