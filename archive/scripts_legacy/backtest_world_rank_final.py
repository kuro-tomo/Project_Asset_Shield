#!/usr/bin/env python3
"""
Asset Shield V3.2.0 - World Rank Final Edition
===============================================

Optimized for Overall Sharpe >= 1.0 through:
1. Crisis Detection & Regime-Based Position Sizing
2. Volatility-Adjusted Entry/Exit
3. Dynamic Stop-Loss Tightening
4. Cash Buffer During High-Vol Regimes

Target: Sharpe >= 1.0 (Overall), OOS Sharpe >= 0.7, Impact <= 20bps

Author: Asset Shield Team
Version: 3.2.0 Final
"""

import os
import sys
import sqlite3
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Crisis Detection Engine
# =============================================================================

class CrisisDetector:
    """
    Detects market crisis regimes using volatility and drawdown signals.

    Regimes:
    - CRISIS: VIX-equivalent > 30 or market DD > 15%
    - HIGH_VOL: VIX-equivalent 20-30
    - NORMAL: VIX-equivalent < 20
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.market_prices: List[float] = []
        self.peak = 0.0

    def update(self, market_return: float, market_price: float) -> str:
        """Update regime detection with new market data"""
        self.market_prices.append(market_price)
        self.peak = max(self.peak, market_price)

        # Calculate implied volatility (annualized)
        if len(self.market_prices) >= self.lookback:
            recent = self.market_prices[-self.lookback:]
            returns = np.diff(recent) / np.array(recent[:-1])
            vol = np.std(returns) * np.sqrt(252) * 100  # As percentage
        else:
            vol = 20.0  # Default

        # Market drawdown
        dd = (self.peak - market_price) / self.peak if self.peak > 0 else 0

        # Regime classification (calibrated for Japan market)
        if vol > 50 or dd > 0.25:
            return "CRISIS"
        elif vol > 35 or dd > 0.15:
            return "HIGH_VOL"
        else:
            return "NORMAL"

    def get_position_multiplier(self, regime: str) -> float:
        """Get position size multiplier based on regime"""
        return {
            "CRISIS": 0.40,    # 40% of normal size (was 25%)
            "HIGH_VOL": 0.70,  # 70% of normal size (was 50%)
            "NORMAL": 1.00     # Full size
        }.get(regime, 0.70)

    def get_stop_loss(self, regime: str, base_stop: float) -> float:
        """Tighten stop-loss during volatile periods"""
        multipliers = {
            "CRISIS": 0.5,    # Half the normal stop (tighter)
            "HIGH_VOL": 0.75,
            "NORMAL": 1.0
        }
        return base_stop * multipliers.get(regime, 1.0)


# =============================================================================
# Walk-Forward Validator
# =============================================================================

@dataclass
class PhaseResult:
    phase: str
    start_date: date
    end_date: date
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float


class WalkForwardOptimized:
    """Walk-forward validation with phase-based analysis"""

    PHASES = {
        "training": (date(2007, 1, 1), date(2015, 12, 31)),
        "validation": (date(2016, 1, 1), date(2020, 12, 31)),
        "out_of_sample": (date(2021, 1, 1), date(2026, 12, 31))
    }

    def __init__(self):
        self.equity_history: Dict[str, List[Tuple[date, float]]] = {p: [] for p in self.PHASES}
        self.trades: Dict[str, List[Dict]] = {p: [] for p in self.PHASES}

    def get_phase(self, d: date) -> str:
        for phase, (start, end) in self.PHASES.items():
            if start <= d <= end:
                return phase
        return "out_of_sample"

    def record(self, d: date, equity: float, trade: Optional[Dict] = None):
        phase = self.get_phase(d)
        self.equity_history[phase].append((d, equity))
        if trade:
            self.trades[phase].append(trade)

    def calculate_phase(self, phase: str) -> PhaseResult:
        eq_list = self.equity_history.get(phase, [])
        trades = self.trades.get(phase, [])

        if len(eq_list) < 10:
            return PhaseResult(phase, date(2007,1,1), date(2007,1,2), 0,0,0,0,0,0,0,0)

        dates, equity = zip(*eq_list)
        equity = np.array(equity, dtype=np.float64)

        start_d, end_d = dates[0], dates[-1]

        # Returns
        total_ret = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        years = max((end_d - start_d).days / 365.25, 0.1)
        annual_ret = (1 + total_ret) ** (1/years) - 1 if total_ret > -1 else 0

        # Daily returns
        daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        daily_ret = np.nan_to_num(daily_ret, 0)

        # Volatility & Sharpe
        vol = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 1 else 0.01
        sharpe = annual_ret / vol if vol > 0.01 else 0

        # Sortino
        neg_ret = daily_ret[daily_ret < 0]
        down_vol = np.std(neg_ret) * np.sqrt(252) if len(neg_ret) > 1 else vol
        sortino = annual_ret / down_vol if down_vol > 0.01 else sharpe

        # Max DD
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = np.max(dd)

        # Trade stats
        if trades:
            pnls = [t.get('pnl', 0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            pf = sum(wins) / sum(losses) if losses else (10 if wins else 0)
        else:
            win_rate, pf = 0, 0

        return PhaseResult(
            phase=phase, start_date=start_d, end_date=end_d,
            total_return=total_ret, annual_return=annual_ret,
            sharpe_ratio=sharpe, sortino_ratio=sortino,
            max_drawdown=max_dd, total_trades=len(trades),
            win_rate=win_rate, profit_factor=pf
        )


# =============================================================================
# Almgren-Chriss Impact Model
# =============================================================================

@dataclass
class ImpactResult:
    total_bps: float
    permanent_bps: float
    temporary_bps: float
    is_executable: bool


class AlmgrenChrissOptimized:
    """Optimized impact model with unified parameters"""

    def __init__(self):
        self.gamma = 0.10
        self.eta = 0.01
        self.sigma = 0.25 / np.sqrt(252)
        self.max_participation = 0.10
        self.spread_bps = 10.0

    def calculate(self, order_value: float, adt: float) -> ImpactResult:
        if adt <= 0 or order_value <= 0:
            return ImpactResult(999, 0, 0, False)

        participation = order_value / adt
        if participation > self.max_participation:
            return ImpactResult(999, 0, 0, False)

        perm = self.gamma * self.sigma * np.sqrt(participation) * 10000
        temp = self.eta * self.sigma * (participation ** 0.6) * 10000
        spread = self.spread_bps
        total = perm + temp + spread

        return ImpactResult(total, perm, temp, total < 50)


# =============================================================================
# World Rank Backtester
# =============================================================================

class WorldRankFinalBacktester:
    """
    Final optimized backtester targeting World Rank compliance.

    Key optimizations:
    1. Crisis regime detection with position scaling
    2. Volatility-adjusted stops
    3. Strict ADT filtering for impact control
    """

    def __init__(self, cache_path: str, initial_capital: float = 10_000_000):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=20000")
        self.initial_capital = initial_capital

        self.crisis_detector = CrisisDetector()
        self.walk_forward = WalkForwardOptimized()
        self.impact_model = AlmgrenChrissOptimized()

        # Optimized parameters for Sharpe maximization
        self.max_positions = 20
        self.base_position_pct = 0.05
        self.rebalance_days = 63
        self.holding_days = 250
        self.pbr_pct = 0.20
        self.roe_pct = 0.80
        self.composite_pct = 0.80
        self.min_adt = 400_000_000
        self.base_stop_loss = 0.12
        self.take_profit = 0.35

    def run(self, start_date: str = "2008-01-01", end_date: str = "2026-01-31") -> Dict:
        logger.info("=" * 70)
        logger.info("Asset Shield V3.2.0 - World Rank Final Edition")
        logger.info(f"Period: {start_date} → {end_date}")
        logger.info("=" * 70)

        # Load data
        logger.info("Loading data...")
        prices_df = pd.read_sql_query("""
            SELECT code, date, close, volume, turnover, adjustment_close
            FROM daily_quotes WHERE date BETWEEN ? AND ?
            ORDER BY date, code
        """, self.conn, params=[start_date, end_date])
        prices_df['price'] = prices_df['adjustment_close'].fillna(prices_df['close'])

        fins_df = pd.read_sql_query("""
            SELECT code, disclosed_date, bps, roe
            FROM financial_statements WHERE bps > 0
        """, self.conn)

        logger.info(f"Loaded {len(prices_df):,} price records")

        # Calculate market index for regime detection
        market_df = prices_df.groupby('date').agg({
            'price': 'mean',
            'turnover': 'sum'
        }).reset_index()
        market_index = dict(zip(market_df['date'], market_df['price']))

        # Build indices
        price_index = {d: dict(zip(g['code'], g['price']))
                       for d, g in prices_df.groupby('date')}
        turnover_index = {d: dict(zip(g['code'], g['turnover']))
                         for d, g in prices_df.groupby('date')}
        fins_index = {code: g.sort_values('disclosed_date')
                      for code, g in fins_df.groupby('code')}

        trading_dates = sorted(prices_df['date'].unique())

        # ADT calculation (point-in-time)
        logger.info("Computing point-in-time ADT...")
        adt_cache = {}

        # Initialize state
        cash = self.initial_capital
        positions = []
        equity_history = []
        all_trades = []
        impact_records = []
        peak_equity = self.initial_capital

        logger.info("Running backtest with crisis detection...")
        start_time = datetime.now()

        for i, current_date in enumerate(trading_dates):
            current_date_obj = datetime.strptime(current_date, "%Y-%m-%d").date()
            price_dict = price_index.get(current_date, {})
            turnover_dict = turnover_index.get(current_date, {})

            # Market regime detection
            market_price = market_index.get(current_date, 0)
            if i > 0:
                prev_price = market_index.get(trading_dates[i-1], market_price)
                market_ret = (market_price / prev_price - 1) if prev_price > 0 else 0
            else:
                market_ret = 0
            regime = self.crisis_detector.update(market_ret, market_price)

            # Position multiplier based on regime
            regime_mult = self.crisis_detector.get_position_multiplier(regime)
            dynamic_stop = self.crisis_detector.get_stop_loss(regime, self.base_stop_loss)

            # Portfolio valuation
            positions_value = sum(
                price_dict.get(p['code'], p['entry_price']) * p['shares']
                for p in positions
            )
            equity = cash + positions_value

            # Track metrics
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

            # Record for walk-forward
            self.walk_forward.record(current_date_obj, equity)

            equity_history.append({
                'date': current_date,
                'equity': equity,
                'regime': regime,
                'drawdown': dd
            })

            # Skip non-rebalance days
            if i % self.rebalance_days != 0:
                continue

            # Compute point-in-time ADT
            if current_date not in adt_cache:
                df_slice = prices_df[prices_df['date'] <= current_date]
                adt_df = df_slice.groupby('code')['turnover'].apply(
                    lambda x: x.tail(60).mean() if len(x) >= 20 else 0
                )
                adt_cache[current_date] = adt_df.to_dict()
            adt_map = adt_cache[current_date]

            # Close positions
            to_close = []
            for pos in positions:
                entry_idx = trading_dates.index(pos['entry_date']) if pos['entry_date'] in trading_dates else 0
                holding = i - entry_idx

                if holding >= self.holding_days:
                    to_close.append(pos)
                    continue

                if pos['code'] in price_dict:
                    pnl_pct = (price_dict[pos['code']] / pos['entry_price']) - 1

                    # Dynamic stop-loss (tighter in crisis)
                    if pnl_pct <= -dynamic_stop:
                        to_close.append(pos)
                        continue

                    if pnl_pct >= self.take_profit:
                        to_close.append(pos)
                        continue

            # Execute closes
            for pos in to_close:
                if pos['code'] not in price_dict:
                    continue

                exit_price = price_dict[pos['code']]
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                pnl_pct = (exit_price / pos['entry_price']) - 1

                trade = {
                    'code': pos['code'],
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'regime_at_exit': regime
                }
                all_trades.append(trade)
                self.walk_forward.record(current_date_obj, equity, trade)

                cash += exit_price * pos['shares']
                positions.remove(pos)

            # Open new positions (reduced in crisis)
            max_new = int((self.max_positions - len(positions)) * regime_mult)
            max_new = max(1, max_new)  # Always allow at least 1 new position

            if max_new > 0:  # Allow entries even in crisis (reduced size)
                # Find candidates
                candidates = self._find_candidates(
                    current_date, price_dict, fins_index, adt_map
                )

                held_codes = {p['code'] for p in positions}
                candidates = [c for c in candidates if c['code'] not in held_codes]

                # Position sizing with regime adjustment
                base_size = equity * self.base_position_pct * regime_mult

                for cand in candidates[:max_new]:
                    adt = adt_map.get(cand['code'], 0)
                    impact = self.impact_model.calculate(base_size, adt)

                    if not impact.is_executable:
                        continue

                    pos_value = min(base_size, cash * 0.95)
                    if pos_value < 100_000:
                        continue

                    shares = int(pos_value / cand['price'])
                    if shares <= 0:
                        continue

                    cost = shares * cand['price']
                    if cost > cash:
                        continue

                    positions.append({
                        'code': cand['code'],
                        'entry_date': current_date,
                        'entry_price': cand['price'],
                        'shares': shares,
                        'regime_at_entry': regime
                    })

                    impact_records.append(impact.total_bps)
                    cash -= cost

            # Progress
            if i % 500 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = i / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[{current_date}] {i}/{len(trading_dates)} | "
                    f"Equity: ¥{equity:,.0f} | DD: {dd:.1%} | "
                    f"Regime: {regime} | Pos: {len(positions)}"
                )

        logger.info(f"Backtest completed in {(datetime.now()-start_time).total_seconds():.1f}s")

        # Calculate results
        result = self._calculate_results(equity_history, all_trades, impact_records)

        # Print report
        self._print_report(result)

        # Save outputs
        self._save_outputs(result, equity_history)

        return result

    def _find_candidates(self, eval_date: str, price_dict: Dict,
                         fins_index: Dict, adt_map: Dict) -> List[Dict]:
        """Percentile-based candidate selection"""
        all_stocks = []

        tradeable = {c for c, a in adt_map.items() if a >= self.min_adt}

        for code in tradeable:
            if code not in price_dict:
                continue
            price = price_dict[code]
            if not price or price <= 0:
                continue

            if code not in fins_index:
                continue

            code_fins = fins_index[code]
            valid = code_fins[code_fins['disclosed_date'] <= eval_date]
            if valid.empty:
                continue

            latest = valid.iloc[-1]
            bps = latest['bps']
            roe = latest['roe']

            if bps <= 0 or not np.isfinite(roe):
                continue

            pbr = price / bps
            if pbr <= 0:
                continue

            all_stocks.append({
                'code': code,
                'price': price,
                'pbr': pbr,
                'roe': roe,
                'adt': adt_map.get(code, 0)
            })

        if len(all_stocks) < 10:
            return []

        # Percentile ranking
        pbrs = [s['pbr'] for s in all_stocks]
        roes = [s['roe'] for s in all_stocks]

        for s in all_stocks:
            s['pbr_pct'] = sum(1 for p in pbrs if p <= s['pbr']) / len(pbrs)
            s['roe_pct'] = sum(1 for r in roes if r <= s['roe']) / len(roes)
            s['composite'] = (1 - s['pbr_pct']) * 0.5 + s['roe_pct'] * 0.5

        # Select top composite
        threshold = np.percentile([s['composite'] for s in all_stocks],
                                   self.composite_pct * 100)

        candidates = [s for s in all_stocks if s['composite'] >= threshold]
        candidates.sort(key=lambda x: x['composite'], reverse=True)

        return candidates

    def _calculate_results(self, equity_history: List, trades: List,
                           impact_records: List) -> Dict:
        """Calculate comprehensive results with robust NaN handling"""
        equity = np.array([e['equity'] for e in equity_history], dtype=np.float64)
        equity = np.nan_to_num(equity, nan=self.initial_capital)

        # Overall metrics
        total_ret = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        total_ret = 0 if not np.isfinite(total_ret) else total_ret

        years = len(equity) / 252
        if years > 0 and total_ret > -1 and (1 + total_ret) > 0:
            annual_ret = (1 + total_ret) ** (1/years) - 1
        else:
            annual_ret = 0
        annual_ret = 0 if not np.isfinite(annual_ret) else annual_ret

        # Daily returns with NaN handling
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        daily_ret = np.nan_to_num(daily_ret, nan=0, posinf=0, neginf=0)

        # Volatility with safety
        if len(daily_ret) > 1:
            vol = float(np.std(daily_ret)) * np.sqrt(252)
        else:
            vol = 0.01
        vol = max(vol, 0.01)  # Minimum vol floor

        sharpe = annual_ret / vol if vol > 0.01 else 0
        sharpe = 0 if not np.isfinite(sharpe) else sharpe

        # Sortino
        neg_ret = daily_ret[daily_ret < 0]
        if len(neg_ret) > 1:
            down_vol = float(np.std(neg_ret)) * np.sqrt(252)
            down_vol = max(down_vol, 0.01)
        else:
            down_vol = vol
        sortino = annual_ret / down_vol if down_vol > 0.01 else sharpe
        sortino = 0 if not np.isfinite(sortino) else sortino

        # Max DD with safety
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = (peak - equity) / np.where(peak > 0, peak, 1)
        dd = np.nan_to_num(dd, nan=0, posinf=0, neginf=0)
        max_dd = float(np.max(dd))
        max_dd = 0 if not np.isfinite(max_dd) else max_dd

        calmar = annual_ret / max_dd if max_dd > 0.01 else 0
        calmar = 0 if not np.isfinite(calmar) else calmar

        # Trade stats
        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            win_rate = len(wins) / len(pnls)
            pf = sum(wins) / sum(losses) if losses else 10
            avg_hold = np.mean([
                (datetime.strptime(t['exit_date'], '%Y-%m-%d') -
                 datetime.strptime(t['entry_date'], '%Y-%m-%d')).days
                for t in trades
            ])
        else:
            win_rate, pf, avg_hold = 0, 0, 0

        avg_impact = np.mean(impact_records) if impact_records else 0

        # Walk-forward phases
        training = self.walk_forward.calculate_phase("training")
        validation = self.walk_forward.calculate_phase("validation")
        oos = self.walk_forward.calculate_phase("out_of_sample")

        return {
            'strategy': 'Asset Shield V3.2.0 World Rank Final',
            'total_return': total_ret,
            'annual_return': annual_ret,
            'final_equity': equity[-1],
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'volatility': vol,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': pf,
            'avg_holding_days': avg_hold,
            'avg_impact_bps': avg_impact,
            'walk_forward': {
                'training': asdict(training),
                'validation': asdict(validation),
                'out_of_sample': asdict(oos)
            }
        }

    def _print_report(self, result: Dict):
        """Print formatted report"""
        wf = result['walk_forward']
        training = wf['training']
        validation = wf['validation']
        oos = wf['out_of_sample']

        print("\n" + "=" * 70)
        print("Asset Shield V3.2.0 - World Rank Final Results")
        print("=" * 70)

        print(f"\nReturn: {result['total_return']*100:.2f}% ({result['annual_return']*100:.2f}% p.a.)")
        print(f"Final Equity: ¥{result['final_equity']:,.0f}")

        print(f"\nSharpe: {result['sharpe_ratio']:.2f} | Sortino: {result['sortino_ratio']:.2f}")
        print(f"Max DD: {result['max_drawdown']*100:.2f}% | Calmar: {result['calmar_ratio']:.2f}")

        print(f"\nTrades: {result['total_trades']} | Win Rate: {result['win_rate']*100:.2f}%")
        print(f"Profit Factor: {result['profit_factor']:.2f} | Avg Hold: {result['avg_holding_days']:.0f}d")
        print(f"Avg Impact: {result['avg_impact_bps']:.1f}bps")

        print(f"\nWalk-Forward:")
        print(f"  Training:   Sharpe {training['sharpe_ratio']:.2f}, {training['total_trades']} trades")
        print(f"  Validation: Sharpe {validation['sharpe_ratio']:.2f}, {validation['total_trades']} trades")
        print(f"  OOS:        Sharpe {oos['sharpe_ratio']:.2f}, {oos['total_trades']} trades")

        print("\n" + "=" * 70)
        print("World Rank Compliance")
        print("=" * 70)

        overall_pass = result['sharpe_ratio'] >= 1.0
        oos_pass = oos['sharpe_ratio'] >= 0.7
        impact_pass = result['avg_impact_bps'] <= 20

        print(f"  Overall Sharpe >= 1.0: {result['sharpe_ratio']:.2f} {'✓' if overall_pass else '✗'}")
        print(f"  OOS Sharpe >= 0.7:     {oos['sharpe_ratio']:.2f} {'✓' if oos_pass else '✗'}")
        print(f"  Avg Impact <= 20bps:   {result['avg_impact_bps']:.1f} {'✓' if impact_pass else '✗'}")

        all_pass = overall_pass and oos_pass and impact_pass
        print(f"\n  VERDICT: {'✓ WORLD RANK READY' if all_pass else '✗ OPTIMIZATION NEEDED'}")
        print("=" * 70)

    def _save_outputs(self, result: Dict, equity_history: List):
        """Save results and chart"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(PROJECT_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # JSON
        json_path = os.path.join(output_dir, f'world_rank_final_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results: {json_path}")

        # Chart
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        dates = [datetime.strptime(e['date'], '%Y-%m-%d') for e in equity_history]
        equity = [e['equity'] / 1_000_000 for e in equity_history]
        dd = [e['drawdown'] * 100 for e in equity_history]

        ax1 = axes[0]
        ax1.plot(dates, equity, 'b-', linewidth=1.5)
        ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(
            f"Asset Shield V3.2.0 World Rank Final\n"
            f"Sharpe: {result['sharpe_ratio']:.2f} | OOS: {result['walk_forward']['out_of_sample']['sharpe_ratio']:.2f}",
            fontsize=14, fontweight='bold'
        )
        ax1.set_ylabel('Portfolio (M JPY)')
        ax1.grid(True, alpha=0.3)

        # Color phases
        for phase, color in [('training', 'lightblue'), ('validation', 'lightyellow'), ('out_of_sample', 'lightgreen')]:
            start, end = WalkForwardOptimized.PHASES[phase]
            ax1.axvspan(start, end, alpha=0.2, color=color, label=phase.replace('_', ' ').title())
        ax1.legend(loc='upper left')

        ax2 = axes[1]
        ax2.fill_between(dates, 0, [-d for d in dd], color='red', alpha=0.5)
        ax2.axhline(y=-30, color='darkred', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        png_path = os.path.join(output_dir, f'world_rank_final_{timestamp}.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Chart: {png_path}")


def main():
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'jquants_cache.db')

    if not os.path.exists(cache_path):
        logger.error(f"Database not found: {cache_path}")
        sys.exit(1)

    bt = WorldRankFinalBacktester(cache_path, initial_capital=10_000_000)
    result = bt.run()

    return result


if __name__ == "__main__":
    main()
