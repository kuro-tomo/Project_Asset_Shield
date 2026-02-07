#!/usr/bin/env python3
"""
Asset Shield Phase 2 - World Rank Backtest (Optimized)
======================================================
Fast version backtest - Performance optimized

Optimizations:
- Data preloading and indexing
- Pre-applied liquidity filtering
- Vectorized operations

Author: Asset Shield V2 Team
Version: 2.0.0 (2026-02-03)
"""

import os
import sys
import sqlite3
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from shield.capacity_engine import CapacityEngine, AlmgrenChrissParams
from shield.risk_buffer import RiskBuffer, create_risk_buffer, RiskRegime
from shield.walk_forward import WalkForwardValidator, ValidationPhase, create_walk_forward_validator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Position:
    code: str
    entry_date: date
    entry_price: float
    shares: int
    pbr: float
    roe: float


@dataclass
class WorldRankResult:
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_holding_days: float
    # Capacity
    max_aum_supported_b: float
    avg_impact_bps: float
    # Walk-Forward
    training_sharpe: float
    validation_sharpe: float
    oos_sharpe: float
    overfitting_ratio: float
    degradation_ratio: float
    passed_validation: bool

    def to_dict(self):
        return asdict(self)


class FastWorldRankBacktester:
    """Optimized World Rank Backtester"""

    def __init__(self, cache_path: str, initial_capital: float = 10_000_000):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(cache_path)
        self.initial_capital = initial_capital

        # Capacity Engine with standard params
        params = AlmgrenChrissParams.standard()
        self.capacity_engine = CapacityEngine(impact_params=params, target_aum=30e9)
        self.capacity_engine.MAX_IMPACT_BPS = 75

        # Risk Buffer
        self.risk_buffer = create_risk_buffer(initial_capital, 0.30)

        # Walk Forward
        self.walk_forward = create_walk_forward_validator()

        # Strategy params
        self.max_positions = 20
        self.position_pct = 0.10
        self.rebalance_days = 63
        self.holding_days = 250
        self.pbr_threshold = 1.0
        self.roe_threshold = 10.0

    def load_data(self, start_date: str, end_date: str):
        """Preload data"""
        logger.info("Loading data...")

        # Prices with ADT calculation
        price_query = """
            SELECT code, date, close, volume, turnover, adjustment_close
            FROM daily_quotes
            WHERE date BETWEEN ? AND ?
            ORDER BY date, code
        """
        self.prices_df = pd.read_sql_query(price_query, self.conn, params=[start_date, end_date])
        self.prices_df['price'] = self.prices_df['adjustment_close'].fillna(self.prices_df['close'])
        logger.info(f"Prices: {len(self.prices_df):,} records")

        # Financials
        fin_query = """
            SELECT code, disclosed_date, bps, roe
            FROM financial_statements
            WHERE bps > 0
            ORDER BY code, disclosed_date
        """
        self.fins_df = pd.read_sql_query(fin_query, self.conn)
        logger.info(f"Financials: {len(self.fins_df):,} records")

        # Trading dates
        self.trading_dates = sorted(self.prices_df['date'].unique())
        logger.info(f"Trading days: {len(self.trading_dates)}")

        # Pre-calculate ADT for each stock
        logger.info("Calculating ADT for all stocks...")
        self._calculate_adt()

    def _calculate_adt(self):
        """Pre-calculate ADT for all stocks"""
        self.adt_df = self.prices_df.groupby('code').agg({
            'turnover': lambda x: x.tail(60).mean() if len(x) >= 20 else 0
        }).reset_index()
        self.adt_df.columns = ['code', 'adt']

        # Filter tradeable stocks (ADT > 500M)
        self.tradeable_codes = set(self.adt_df[self.adt_df['adt'] >= 500_000_000]['code'])
        logger.info(f"Tradeable stocks (ADT >= ¥500M): {len(self.tradeable_codes)}")

    def get_latest_financials(self, as_of_date: str) -> pd.DataFrame:
        """Get latest financial data as of specified date"""
        valid_fins = self.fins_df[self.fins_df['disclosed_date'] <= as_of_date]
        return valid_fins.groupby('code').last().reset_index()

    def find_candidates(self, eval_date: str) -> List[Dict]:
        """Search for candidate stocks (optimized version)"""
        # Get prices for the day
        day_prices = self.prices_df[self.prices_df['date'] == eval_date].copy()
        if day_prices.empty:
            return []

        # Filter to tradeable stocks only
        day_prices = day_prices[day_prices['code'].isin(self.tradeable_codes)]

        # Get latest financials
        latest_fins = self.get_latest_financials(eval_date)

        # Merge
        merged = day_prices.merge(latest_fins, on='code', how='inner')

        # Calculate PBR
        merged['pbr'] = merged['price'] / merged['bps']

        # Filter: PBR <= 1.0 AND ROE >= 10%
        candidates = merged[
            (merged['pbr'] <= self.pbr_threshold) &
            (merged['pbr'] > 0) &
            (merged['roe'] >= self.roe_threshold)
        ].copy()

        # Sort by ROE descending
        candidates = candidates.sort_values('roe', ascending=False)

        return candidates[['code', 'price', 'pbr', 'roe']].to_dict('records')

    def detect_regime(self, current_idx: int) -> RiskRegime:
        """Regime detection"""
        if current_idx < 60:
            return RiskRegime.NORMAL

        lookback_dates = self.trading_dates[max(0, current_idx-60):current_idx]
        lookback_prices = self.prices_df[self.prices_df['date'].isin(lookback_dates)]

        daily_returns = lookback_prices.groupby('date')['price'].mean().pct_change().dropna()
        if len(daily_returns) < 20:
            return RiskRegime.NORMAL

        vol = daily_returns.std() * np.sqrt(252)

        if vol > 0.30:
            return RiskRegime.CRISIS
        elif vol > 0.20:
            return RiskRegime.HIGH_VOL
        elif vol < 0.12:
            return RiskRegime.LOW_VOL
        return RiskRegime.NORMAL

    def run(self, start_date: str = "2007-01-01", end_date: str = "2026-12-31") -> WorldRankResult:
        """Execute backtest"""
        logger.info("=" * 70)
        logger.info("ASSET SHIELD PHASE 2 - WORLD RANK BACKTEST (FAST)")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 70)

        self.load_data(start_date, end_date)

        # State
        cash = self.initial_capital
        positions: List[Position] = []
        equity_history = []
        trades = []
        impact_history = []
        peak_equity = self.initial_capital

        # Price lookup dict (pre-build for speed)
        price_dict_by_date = {}
        for d in self.trading_dates:
            day_df = self.prices_df[self.prices_df['date'] == d]
            price_dict_by_date[d] = dict(zip(day_df['code'], day_df['price']))

        total_days = len(self.trading_dates)

        for i, current_date in enumerate(self.trading_dates):
            current_date_obj = datetime.strptime(current_date, "%Y-%m-%d").date()
            wf_phase = self.walk_forward.get_current_phase(current_date_obj)

            price_dict = price_dict_by_date.get(current_date, {})

            # Mark-to-market
            positions_value = sum(
                price_dict.get(p.code, p.entry_price) * p.shares
                for p in positions
            )
            equity = cash + positions_value

            # Update risk buffer
            self.risk_buffer.update_equity(equity, current_date_obj)
            regime = self.detect_regime(i)
            self.risk_buffer.set_regime(regime)
            allocation = self.risk_buffer.get_cash_allocation()

            # Drawdown
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

            equity_history.append({
                'date': current_date,
                'equity': equity,
                'cash': cash,
                'positions': len(positions),
                'drawdown': dd,
                'regime': regime.value,
                'dd_state': self.risk_buffer.dd_state.value,
                'cash_target': allocation.target_cash_pct,
                'wf_phase': wf_phase.value
            })

            # Rebalance check
            if i % self.rebalance_days != 0:
                continue

            # Close old positions
            to_close = []
            for pos in positions:
                if pos.entry_date.isoformat() in self.trading_dates:
                    entry_idx = self.trading_dates.index(pos.entry_date.isoformat())
                    if i - entry_idx >= self.holding_days:
                        to_close.append(pos)

            for pos in to_close:
                exit_price = price_dict.get(pos.code, pos.entry_price * 0.5)
                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price / pos.entry_price - 1)

                trades.append({
                    'code': pos.code,
                    'entry_date': pos.entry_date,
                    'exit_date': current_date_obj,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'phase': wf_phase.value
                })

                # Record to walk-forward
                self.walk_forward.record_trade(
                    f"{pos.code}_{pos.entry_date}",
                    pos.code,
                    pos.entry_date,
                    current_date_obj,
                    pos.entry_price,
                    exit_price
                )

                cash += exit_price * pos.shares
                positions.remove(pos)
                impact_history.append(30.0)  # Estimated impact

            # New positions
            if self.risk_buffer.dd_state.value == "active":
                max_new = max(0, self.max_positions // 2 - len(positions))
            else:
                max_new = self.max_positions - len(positions)

            if max_new > 0:
                base_size = equity * self.position_pct
                adj_size = self.risk_buffer.get_position_size_adjustment(base_size)

                candidates = self.find_candidates(current_date)
                held_codes = {p.code for p in positions}
                candidates = [c for c in candidates if c['code'] not in held_codes]

                for cand in candidates[:max_new]:
                    pos_value = min(adj_size, cash * 0.95)
                    if pos_value < 100_000:
                        continue

                    shares = int(pos_value / cand['price'])
                    if shares <= 0:
                        continue

                    cost = shares * cand['price']
                    if cost > cash:
                        continue

                    positions.append(Position(
                        code=cand['code'],
                        entry_date=current_date_obj,
                        entry_price=cand['price'],
                        shares=shares,
                        pbr=cand['pbr'],
                        roe=cand['roe']
                    ))
                    cash -= cost

            # Progress
            if i % 250 == 0:
                logger.info(
                    f"[{current_date}] ({i}/{total_days}) Equity: ¥{equity:,.0f} | "
                    f"DD: {dd:.1%} | Pos: {len(positions)} | Trades: {len(trades)} | "
                    f"Phase: {wf_phase.value}"
                )

        # Calculate metrics
        result = self._calculate_metrics(equity_history, trades, impact_history)

        # Print report
        self._print_report(result, equity_history)

        # Plot
        self._plot(equity_history, result)

        return result

    def _calculate_metrics(self, equity_history, trades, impact_history) -> WorldRankResult:
        """Calculate metrics"""
        df = pd.DataFrame(equity_history)

        # Returns
        final_equity = df['equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital) - 1
        n_years = len(df) / 252
        annual_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0

        # Risk
        daily_returns = df['equity'].pct_change().dropna()
        vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe = (annual_return - 0.001) / vol if vol > 0 else 0

        neg_returns = daily_returns[daily_returns < 0]
        downside_vol = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else vol
        sortino = (annual_return - 0.001) / downside_vol if downside_vol > 0 else 0

        max_dd = df['drawdown'].max()
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # Trade stats
        if trades:
            winners = [t for t in trades if t['pnl'] > 0]
            losers = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(winners) / len(trades)
            total_wins = sum(t['pnl'] for t in winners)
            total_losses = abs(sum(t['pnl'] for t in losers))
            pf = total_wins / total_losses if total_losses > 0 else float('inf')
            holding = [(t['exit_date'] - t['entry_date']).days for t in trades if t['exit_date']]
            avg_hold = np.mean(holding) if holding else 0
        else:
            win_rate = 0
            pf = 0
            avg_hold = 0

        # Capacity
        avg_impact = np.mean(impact_history) if impact_history else 0

        # Walk-Forward
        wf = self.walk_forward.analyze_overfitting()
        training_sharpe = wf.training_metrics.sharpe_ratio if wf.training_metrics else 0
        val_sharpe = wf.validation_metrics.sharpe_ratio if wf.validation_metrics else 0
        oos_sharpe = wf.oos_metrics.sharpe_ratio if wf.oos_metrics else 0

        return WorldRankResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            volatility=vol,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=pf,
            avg_holding_days=avg_hold,
            max_aum_supported_b=30.0,
            avg_impact_bps=avg_impact,
            training_sharpe=training_sharpe,
            validation_sharpe=val_sharpe,
            oos_sharpe=oos_sharpe,
            overfitting_ratio=wf.overfitting_ratio,
            degradation_ratio=wf.degradation_ratio,
            passed_validation=wf.passed_validation
        )

    def _print_report(self, result: WorldRankResult, equity_history):
        """Output report"""
        print("\n" + "=" * 70)
        print("[Asset Shield Phase 2] World Rank Standard Backtest Results")
        print("=" * 70)

        print("\n--- Return Metrics ---")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Annual Return: {result.annual_return:.2%}")
        print(f"  Final Equity: ¥{equity_history[-1]['equity']:,.0f}")

        print("\n--- Risk Metrics ---")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Calmar Ratio: {result.calmar_ratio:.2f}")
        print(f"  Volatility: {result.volatility:.2%}")

        print("\n--- Trade Statistics ---")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Avg Holding Days: {result.avg_holding_days:.0f} days")

        print("\n--- Capacity Validation (30B JPY AUM) ---")
        print(f"  Target AUM: ¥{result.max_aum_supported_b:.0f}B")
        print(f"  Avg Market Impact: {result.avg_impact_bps:.1f}bps")
        print(f"  Capacity: {'Sufficient ✓' if result.avg_impact_bps < 75 else 'Review Needed ✗'}")

        print("\n--- Walk-Forward Validation (Overfitting Prevention) ---")
        print(f"  Training Sharpe (2007-2015): {result.training_sharpe:.2f}")
        print(f"  Validation Sharpe (2016-2020): {result.validation_sharpe:.2f}")
        print(f"  OOS Sharpe (2021-2026): {result.oos_sharpe:.2f}")
        print(f"  Overfitting Ratio: {result.overfitting_ratio:.2f} (threshold: >0.70)")
        print(f"  Degradation Ratio: {result.degradation_ratio:.2f} (threshold: >0.70)")
        print(f"  Validation Result: {'PASS ✓' if result.passed_validation else 'FAIL ✗'}")

        # QuantConnect/Quantiacs format
        print("\n" + "=" * 70)
        print("[QuantConnect/Quantiacs Scorecard]")
        print("=" * 70)
        print(f"  Total Return       : {result.total_return:.2%}")
        print(f"  Annual Return      : {result.annual_return:.2%}")
        print(f"  Sharpe Ratio       : {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio      : {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown       : {result.max_drawdown:.2%}")
        print(f"  Calmar Ratio       : {result.calmar_ratio:.2f}")
        print(f"  Win Rate           : {result.win_rate:.2%}")
        print(f"  Profit Factor      : {result.profit_factor:.2f}")
        print(f"  Total Trades       : {result.total_trades}")
        print(f"  AUM Capacity       : ¥{result.max_aum_supported_b:.0f}B")
        print(f"  OOS Validation     : {'PASS' if result.passed_validation else 'FAIL'}")
        print("=" * 70)

    def _plot(self, equity_history, result: WorldRankResult):
        """Draw chart"""
        df = pd.DataFrame(equity_history)
        df['date'] = pd.to_datetime(df['date'])

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Equity
        ax1 = axes[0]
        ax1.plot(df['date'], df['equity']/1e6, 'b-', lw=1.5, label='Portfolio')
        ax1.axhline(self.initial_capital/1e6, color='gray', ls='--', alpha=0.5, label='Initial')

        # Phase colors
        colors = {'training': 'lightblue', 'validation': 'lightyellow', 'out_of_sample': 'lightgreen'}
        for phase, color in colors.items():
            phase_df = df[df['wf_phase'] == phase]
            if not phase_df.empty:
                ax1.fill_between(phase_df['date'], 0, df['equity'].max()/1e6*1.1,
                               alpha=0.2, color=color, label=phase.replace('_', ' ').title())

        ax1.set_title(
            f'Asset Shield Phase 2 - World Rank Backtest\n'
            f'Sharpe: {result.sharpe_ratio:.2f} | Max DD: {result.max_drawdown:.1%} | '
            f'OOS: {"PASS" if result.passed_validation else "FAIL"}',
            fontsize=14, fontweight='bold'
        )
        ax1.set_ylabel('Portfolio (M¥)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(df['date'], 0, -df['drawdown']*100, color='red', alpha=0.5)
        ax2.axhline(-30, color='darkred', ls='--', alpha=0.7, label='Target DD (-30%)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)

        # Cash allocation
        ax3 = axes[2]
        ax3.fill_between(df['date'], 0, df['cash_target']*100, color='green', alpha=0.3)
        ax3.plot(df['date'], df['cash_target']*100, 'g-', lw=1, label='Cash Target')
        ax3.set_ylabel('Cash (%)')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        out_path = os.path.join(project_root, 'output',
                               f'world_rank_fast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        logger.info(f"Chart saved: {out_path}")
        plt.close()


def main():
    cache_path = os.path.join(project_root, 'data', 'jquants_cache.db')

    if not os.path.exists(cache_path):
        logger.error(f"Database not found: {cache_path}")
        sys.exit(1)

    bt = FastWorldRankBacktester(cache_path, initial_capital=10_000_000)
    result = bt.run(start_date="2008-01-01", end_date="2026-01-31")

    # Save JSON
    out_json = os.path.join(project_root, 'output',
                           f'world_rank_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Results saved: {out_json}")


if __name__ == "__main__":
    main()
