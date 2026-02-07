"""
Asset Shield V5.0 - Multi-Factor Backtest Engine
Japanese Equity Strategy with Kill-Switch Recovery

Features:
1. Multi-Factor Alpha: Value(PBR) + Quality(ROE) + Momentum + Low Vol
2. Gradual Kill-Switch Recovery (prevents missing rebounds)
3. Sector Neutralization
4. ADT Liquidity Constraints

Author: Asset Shield V5 Team
Version: 5.0.0 (2026-02-06)
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
from enum import Enum

from shield.factor_model import MultiFactorModel, FactorScores, FactorWeights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    CRISIS = "crisis"


@dataclass
class V5Config:
    """V5.0 Backtest Configuration"""
    start_date: date = field(default_factory=lambda: date(2015, 1, 1))
    end_date: date = field(default_factory=lambda: date(2026, 2, 1))
    initial_capital: float = 100_000_000

    # Position constraints
    max_positions: int = 25
    min_positions: int = 10
    max_single_weight: float = 0.08
    max_sector_weight: float = 0.25

    # Liquidity
    max_adt_participation: float = 0.05
    min_adt: float = 100_000_000

    # Regime thresholds
    vol_threshold_bull: float = 0.18
    vol_threshold_crisis: float = 0.35

    # Kill-switch
    max_drawdown: float = 0.15
    recovery_steps: List[float] = field(default_factory=lambda: [0.0, 0.30, 0.60, 0.90])
    recovery_days: int = 5

    # Rebalance
    rebalance_frequency: int = 20

    # Factor weights
    factor_weights: FactorWeights = field(default_factory=FactorWeights)


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
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
    total_trades: int
    regime_stats: Dict[str, Any]


class AssetShieldV5Backtest:
    """
    Asset Shield V5.0 Backtest Engine

    Key improvements over V4.x:
    1. Multi-factor alpha model (PBR + ROE + Momentum + Low Vol)
    2. Gradual re-entry after kill-switch
    3. Point-in-Time financial data
    """

    SECTOR33_TO_17 = {
        '0050': '01', '1050': '02', '2050': '03', '3050': '04', '3100': '04',
        '3150': '04', '3200': '04', '3250': '04', '3300': '04', '3350': '04',
        '3400': '04', '3450': '04', '3500': '05', '3550': '06', '3600': '08',
        '3650': '09', '3700': '06', '3750': '06', '3800': '09', '4050': '11',
        '5050': '12', '5100': '12', '5150': '12', '5200': '12', '5250': '10',
        '6050': '13', '6100': '14', '7050': '15', '7100': '16', '7150': '16',
        '7200': '16', '8050': '17', '9050': '10'
    }

    def __init__(self, config: V5Config):
        self.config = config
        self.factor_model = MultiFactorModel(weights=config.factor_weights)

        # State tracking
        self.equity_curve: List[Tuple[date, float]] = []
        self.daily_states: List[Dict] = []
        self.trades: List[Dict] = []
        self.monthly_returns: List[Tuple[str, float]] = []

        # Kill-switch state
        self.kill_switch_active = False
        self.kill_switch_date: Optional[date] = None
        self.recovery_step = 0
        self.current_exposure_target = 0.90
        self.high_water_mark = config.initial_capital

        # Regime
        self.current_regime = MarketRegime.BULL
        self.vol_history: List[float] = []
        self.regime_history: List[MarketRegime] = []

        # Price data cache
        self._price_data: Dict[str, pd.DataFrame] = {}

        logger.info(f"V5.0 Backtest initialized: {config.start_date} to {config.end_date}")

    def load_data(self) -> None:
        """Load data via factor model"""
        self.factor_model.load_data(self.config.start_date, self.config.end_date)
        self._price_data = self.factor_model._price_data
        logger.info(f"Loaded {len(self._price_data)} stocks")

    def get_price(self, code: str, dt: date) -> Optional[float]:
        """Get closing price"""
        if code not in self._price_data:
            return None
        df = self._price_data[code]
        ts = pd.Timestamp(dt)
        if ts in df.index:
            return df.loc[ts, 'close']
        prior = df[df.index <= ts]
        if len(prior) > 0:
            return prior.iloc[-1]['close']
        return None

    def update_regime(self, dt: date, scores: List[FactorScores]) -> MarketRegime:
        """Update market regime based on cross-sectional volatility"""
        if not scores:
            return self.current_regime

        # Use average volatility of top stocks
        vols = [s.vol_60d for s in scores[:30] if s.vol_60d is not None]
        if not vols:
            return self.current_regime

        avg_vol = np.mean(vols)
        self.vol_history.append(avg_vol)
        if len(self.vol_history) > 60:
            self.vol_history = self.vol_history[-60:]

        # Classify regime
        if avg_vol > self.config.vol_threshold_crisis:
            regime = MarketRegime.CRISIS
        elif avg_vol > self.config.vol_threshold_bull:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.BULL

        self.current_regime = regime
        self.regime_history.append(regime)
        return regime

    def check_kill_switch(self, dt: date, equity: float) -> float:
        """
        Check kill-switch and manage recovery.
        Returns target exposure (0.0 to 0.90)
        """
        # Update HWM
        if equity > self.high_water_mark:
            self.high_water_mark = equity

        drawdown = (self.high_water_mark - equity) / self.high_water_mark

        # Trigger conditions
        if not self.kill_switch_active:
            if drawdown > self.config.max_drawdown or self.current_regime == MarketRegime.CRISIS:
                self.kill_switch_active = True
                self.kill_switch_date = dt
                self.recovery_step = 0
                self.current_exposure_target = self.config.recovery_steps[0]
                logger.warning(f"{dt}: KILL SWITCH TRIGGERED (DD={drawdown:.1%}, Regime={self.current_regime.value})")
                return self.current_exposure_target

        # Recovery logic
        if self.kill_switch_active and self.kill_switch_date:
            days_since = (dt - self.kill_switch_date).days

            # Check if we can advance recovery
            if self.current_regime != MarketRegime.CRISIS:
                target_step = min(
                    days_since // self.config.recovery_days,
                    len(self.config.recovery_steps) - 1
                )

                if target_step > self.recovery_step:
                    self.recovery_step = target_step
                    self.current_exposure_target = self.config.recovery_steps[self.recovery_step]
                    logger.info(f"{dt}: Recovery step {self.recovery_step} -> {self.current_exposure_target:.0%}")

                    # Full recovery
                    if self.recovery_step >= len(self.config.recovery_steps) - 1:
                        self.kill_switch_active = False
                        self.kill_switch_date = None
                        self.high_water_mark = equity
                        logger.info(f"{dt}: KILL SWITCH RESET")

            # Crisis returns: reset recovery
            if self.current_regime == MarketRegime.CRISIS:
                self.recovery_step = 0
                self.current_exposure_target = self.config.recovery_steps[0]
                self.kill_switch_date = dt

        # Normal operation
        if not self.kill_switch_active:
            if self.current_regime == MarketRegime.CRISIS:
                return 0.0
            elif self.current_regime == MarketRegime.BEAR:
                return 0.60
            else:
                return 0.90

        return self.current_exposure_target

    def select_portfolio(
        self,
        scores: List[FactorScores],
        target_exposure: float,
        current_equity: float
    ) -> List[Tuple[str, float]]:
        """Select portfolio using multi-factor scores"""
        if not scores or target_exposure <= 0:
            return []

        selected = []
        sector_weights = defaultdict(float)
        total_weight = 0.0

        n_positions = min(len(scores), self.config.max_positions)
        base_weight = target_exposure / n_positions

        for score in scores:
            if len(selected) >= self.config.max_positions:
                break
            if total_weight >= target_exposure:
                break

            code = score.code
            sector = self.factor_model.get_sector(code)

            # Sector constraint
            if sector_weights[sector] + base_weight > self.config.max_sector_weight:
                continue

            # ADT constraint
            if score.adt_20d < self.config.min_adt:
                continue

            max_adt_weight = (score.adt_20d * self.config.max_adt_participation) / current_equity
            weight = min(base_weight, self.config.max_single_weight, max_adt_weight)

            if weight < 0.01:
                continue

            if total_weight + weight > target_exposure:
                weight = target_exposure - total_weight

            selected.append((code, weight))
            sector_weights[sector] += weight
            total_weight += weight

        return selected

    def run_backtest(self) -> PerformanceMetrics:
        """Run full V5.0 backtest"""
        logger.info("Starting V5.0 backtest...")

        if not self._price_data:
            self.load_data()

        # Get trading days
        all_dates = set()
        for df in self._price_data.values():
            all_dates.update(df.index.date)
        trading_days = sorted([d for d in all_dates
                              if self.config.start_date <= d <= self.config.end_date])

        logger.info(f"Trading days: {len(trading_days)}")

        # Initialize state
        equity = self.config.initial_capital
        cash = equity
        positions: Dict[str, int] = {}
        position_values: Dict[str, float] = {}

        last_rebalance = None
        last_factor_compute = None
        cached_scores: List[FactorScores] = []

        # Main loop
        for i, current_date in enumerate(trading_days):
            # Update position values
            position_values = {}
            for code, shares in positions.items():
                price = self.get_price(code, current_date)
                if price:
                    position_values[code] = shares * price

            total_positions = sum(position_values.values())
            equity = cash + total_positions

            # Compute factor scores periodically
            if last_factor_compute is None or (current_date - last_factor_compute).days >= self.config.rebalance_frequency:
                cached_scores = self.factor_model.compute_factor_scores(current_date)
                last_factor_compute = current_date

            # Update regime
            self.update_regime(current_date, cached_scores)

            # Check kill-switch
            target_exposure = self.check_kill_switch(current_date, equity)

            # Rebalance check
            should_rebalance = False
            if self.kill_switch_active and target_exposure == 0:
                should_rebalance = True
            elif last_rebalance is None:
                should_rebalance = True
            elif (current_date - last_rebalance).days >= self.config.rebalance_frequency:
                should_rebalance = True

            if should_rebalance and cached_scores:
                # Get target portfolio
                target_portfolio = self.select_portfolio(cached_scores, target_exposure, equity)
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
                            self.trades.append({
                                'date': current_date, 'code': code, 'side': 'SELL',
                                'shares': shares, 'price': price, 'value': value
                            })
                            del positions[code]

                # Buy/adjust positions
                for code, target_weight in target_portfolio:
                    target_value = equity * target_weight
                    price = self.get_price(code, current_date)
                    if not price or price <= 0:
                        continue

                    current_shares = positions.get(code, 0)
                    current_value = current_shares * price
                    delta_value = target_value - current_value

                    if abs(delta_value) < equity * 0.005:
                        continue

                    if delta_value > 0 and cash >= delta_value:
                        shares_to_buy = int(delta_value / price)
                        if shares_to_buy > 0:
                            cost = shares_to_buy * price
                            cash -= cost
                            positions[code] = current_shares + shares_to_buy
                            self.trades.append({
                                'date': current_date, 'code': code, 'side': 'BUY',
                                'shares': shares_to_buy, 'price': price, 'value': cost
                            })
                    elif delta_value < 0:
                        shares_to_sell = min(current_shares, int(abs(delta_value) / price))
                        if shares_to_sell > 0:
                            proceeds = shares_to_sell * price
                            cash += proceeds
                            positions[code] = current_shares - shares_to_sell
                            if positions[code] == 0:
                                del positions[code]
                            self.trades.append({
                                'date': current_date, 'code': code, 'side': 'SELL',
                                'shares': shares_to_sell, 'price': price, 'value': proceeds
                            })

                last_rebalance = current_date

            # Record state
            self.equity_curve.append((current_date, equity))
            self.daily_states.append({
                'date': current_date,
                'equity': equity,
                'regime': self.current_regime.value,
                'kill_switch': self.kill_switch_active,
                'exposure': target_exposure,
                'n_positions': len(positions)
            })

            # Progress
            if i % 500 == 0:
                logger.info(f"{current_date}: Equity=¥{equity:,.0f} Regime={self.current_regime.value} KS={self.kill_switch_active}")

        return self._calculate_metrics()

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics"""
        if not self.equity_curve:
            raise ValueError("No equity curve")

        dates = [d for d, _ in self.equity_curve]
        equities = [e for _, e in self.equity_curve]

        initial = self.config.initial_capital
        final = equities[-1]

        total_return = (final / initial) - 1
        years = (dates[-1] - dates[0]).days / 365.25
        cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0

        daily_returns = pd.Series(equities).pct_change().dropna()

        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (daily_returns.mean() * 252) / (downside.std() * np.sqrt(252))
        else:
            sortino = 0.0

        peak = pd.Series(equities).expanding().max()
        drawdown = (peak - pd.Series(equities)) / peak
        max_dd = drawdown.max()
        max_dd_idx = drawdown.idxmax()
        max_dd_date = dates[max_dd_idx] if max_dd_idx < len(dates) else dates[-1]

        calmar = cagr / max_dd if max_dd > 0 else 0

        # Monthly
        equity_df = pd.DataFrame({'date': dates, 'equity': equities})
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        monthly = equity_df['equity'].resample('M').last().pct_change().dropna()

        self.monthly_returns = [(str(d.date()), r) for d, r in monthly.items()]

        avg_monthly = monthly.mean() if len(monthly) > 0 else 0
        monthly_std = monthly.std() if len(monthly) > 0 else 0
        best_month = monthly.max() if len(monthly) > 0 else 0
        worst_month = monthly.min() if len(monthly) > 0 else 0
        positive_months = (monthly > 0).sum()
        negative_months = (monthly < 0).sum()
        win_rate = positive_months / len(monthly) if len(monthly) > 0 else 0

        gains = monthly[monthly > 0].sum()
        losses = abs(monthly[monthly < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        # Regime stats
        regime_counts = defaultdict(int)
        for r in self.regime_history:
            regime_counts[r.value] += 1
        total_days = len(self.regime_history)
        regime_stats = {
            'total_days': total_days,
            'bull_days': regime_counts['bull'],
            'bear_days': regime_counts['bear'],
            'crisis_days': regime_counts['crisis'],
            'bull_pct': regime_counts['bull'] / total_days if total_days > 0 else 0,
            'bear_pct': regime_counts['bear'] / total_days if total_days > 0 else 0,
            'crisis_pct': regime_counts['crisis'] / total_days if total_days > 0 else 0
        }

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
            total_trades=len(self.trades),
            regime_stats=regime_stats
        )

    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """Generate backtest report"""
        report = []
        report.append("=" * 70)
        report.append("ASSET SHIELD V5.0 - MULTI-FACTOR BACKTEST REPORT")
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
        report.append(f"Total Trades:      {metrics.total_trades:>12}")
        report.append("")
        report.append("-" * 70)
        report.append("REGIME STATISTICS")
        report.append("-" * 70)
        rs = metrics.regime_stats
        report.append(f"Bull Days:         {rs['bull_days']:>12} ({rs['bull_pct']:.1%})")
        report.append(f"Bear Days:         {rs['bear_days']:>12} ({rs['bear_pct']:.1%})")
        report.append(f"Crisis Days:       {rs['crisis_days']:>12} ({rs['crisis_pct']:.1%})")
        report.append("")
        report.append("=" * 70)
        return "\n".join(report)


def run_v5_backtest():
    """Main entry point"""
    config = V5Config(
        start_date=date(2015, 1, 1),
        end_date=date(2026, 2, 1),
        initial_capital=100_000_000,
        max_positions=25,
        max_drawdown=0.15,
        recovery_steps=[0.0, 0.30, 0.60, 0.90],
        recovery_days=5
    )

    engine = AssetShieldV5Backtest(config)
    engine.load_data()

    metrics = engine.run_backtest()
    report = engine.generate_report(metrics)
    print(report)

    # Save
    output_dir = Path("/Users/MBP/Desktop/Project_Asset_Shield/output/v5")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "V5_BACKTEST_REPORT.txt", 'w') as f:
        f.write(report)

    # Save equity curve
    equity_df = pd.DataFrame(engine.equity_curve, columns=['date', 'equity'])
    equity_df.to_csv(output_dir / "v5_equity_curve.csv", index=False)

    return metrics, engine


if __name__ == "__main__":
    run_v5_backtest()
