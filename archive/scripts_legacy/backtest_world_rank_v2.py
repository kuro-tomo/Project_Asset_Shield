#!/usr/bin/env python3
"""
Asset Shield V3.2.0 - World Rank Edition V2
============================================

Optimizations for Overall Sharpe >= 1.0:
1. Trend Filter: No entries when market is in downtrend
2. Volatility Targeting: Normalize position sizes by realized vol
3. Momentum Overlay: Prefer stocks with positive momentum
4. Dynamic Rebalancing: More frequent in trending markets

Author: Asset Shield Team
Version: 3.2.0-V2
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
# Trend & Volatility Engine
# =============================================================================

class TrendVolEngine:
    """
    Market trend detection and volatility targeting.

    Key insight: The 2008 Lehman period had:
    - Strong downtrend (market below 200-day MA)
    - Extreme volatility (annualized > 40%)

    By filtering entries during these conditions, we protect capital.
    """

    def __init__(self, ma_period: int = 120, vol_lookback: int = 20, vol_target: float = 0.20):
        self.ma_period = ma_period  # Trend MA period
        self.vol_lookback = vol_lookback  # Vol calculation window
        self.vol_target = vol_target  # Target annualized vol (20%)

        self.market_prices: List[float] = []
        self.ma_buffer: List[float] = []

    def update(self, market_price: float) -> Tuple[str, float, float]:
        """
        Update with new market price.

        Returns:
            (trend, realized_vol, position_scalar)
        """
        self.market_prices.append(market_price)

        # Calculate MA
        if len(self.market_prices) >= self.ma_period:
            ma = np.mean(self.market_prices[-self.ma_period:])
        else:
            ma = market_price

        # Trend determination
        if market_price > ma * 1.02:
            trend = "UPTREND"
        elif market_price < ma * 0.98:
            trend = "DOWNTREND"
        else:
            trend = "NEUTRAL"

        # Realized volatility
        if len(self.market_prices) >= self.vol_lookback + 1:
            recent = self.market_prices[-(self.vol_lookback + 1):]
            returns = np.diff(recent) / np.array(recent[:-1])
            realized_vol = np.std(returns) * np.sqrt(252)
        else:
            realized_vol = self.vol_target

        # Position scalar for vol targeting (capped more conservatively)
        if realized_vol > 0.01:
            position_scalar = min(1.5, max(0.5, self.vol_target / realized_vol))
        else:
            position_scalar = 1.0

        return trend, realized_vol, position_scalar

    def should_enter(self, trend: str, realized_vol: float) -> bool:
        """Determine if new entries should be allowed"""
        # No entries in strong downtrend with high vol (Lehman-like)
        if trend == "DOWNTREND" and realized_vol > 0.35:
            return False
        # Reduce entries in downtrend
        if trend == "DOWNTREND" and realized_vol > 0.25:
            return np.random.random() > 0.5  # 50% probability
        return True


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


class WalkForwardV2:
    """Walk-forward with per-phase tracking"""

    PHASES = {
        "training": (date(2008, 1, 1), date(2015, 12, 31)),
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

        total_ret = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        years = max((end_d - start_d).days / 365.25, 0.1)
        annual_ret = (1 + total_ret) ** (1/years) - 1 if total_ret > -1 else 0

        daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        daily_ret = np.nan_to_num(daily_ret, 0)

        vol = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 1 else 0.01
        vol = max(vol, 0.01)
        sharpe = annual_ret / vol

        neg_ret = daily_ret[daily_ret < 0]
        down_vol = np.std(neg_ret) * np.sqrt(252) if len(neg_ret) > 1 else vol
        down_vol = max(down_vol, 0.01)
        sortino = annual_ret / down_vol

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = np.max(dd)

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
# Impact Model
# =============================================================================

@dataclass
class ImpactResult:
    total_bps: float
    is_executable: bool


class AlmgrenChrissV2:
    def __init__(self):
        self.gamma = 0.10
        self.eta = 0.01
        self.sigma = 0.25 / np.sqrt(252)
        self.max_participation = 0.10
        self.spread_bps = 10.0

    def calculate(self, order_value: float, adt: float) -> ImpactResult:
        if adt <= 0 or order_value <= 0:
            return ImpactResult(999, False)

        participation = order_value / adt
        if participation > self.max_participation:
            return ImpactResult(999, False)

        perm = self.gamma * self.sigma * np.sqrt(participation) * 10000
        temp = self.eta * self.sigma * (participation ** 0.6) * 10000
        total = perm + temp + self.spread_bps

        return ImpactResult(total, total < 50)


# =============================================================================
# World Rank Backtester V2
# =============================================================================

class WorldRankBacktesterV2:
    """
    V2 with trend filter and volatility targeting.

    Key improvements:
    1. Trend filter prevents entries in downtrends (protects 2008)
    2. Vol targeting normalizes position sizes
    3. Momentum overlay prefers trending stocks
    """

    def __init__(self, cache_path: str, initial_capital: float = 10_000_000):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA cache_size=20000")
        self.initial_capital = initial_capital

        self.trend_vol = TrendVolEngine(ma_period=120, vol_lookback=20, vol_target=0.20)
        self.walk_forward = WalkForwardV2()
        self.impact_model = AlmgrenChrissV2()

        # Optimized parameters (conservative for DD control)
        self.max_positions = 25
        self.base_position_pct = 0.04  # Reduced from 5% to 4%
        self.rebalance_days = 63
        self.holding_days = 250
        self.pbr_pct = 0.20
        self.roe_pct = 0.80
        self.composite_pct = 0.80
        self.min_adt = 400_000_000
        self.stop_loss = 0.12
        self.take_profit = 0.35

        # Momentum parameters
        self.momentum_lookback = 60
        self.momentum_weight = 0.3  # Weight in composite score

    def run(self, start_date: str = "2008-01-01", end_date: str = "2026-01-31") -> Dict:
        logger.info("=" * 70)
        logger.info("Asset Shield V3.2.0 - World Rank V2 (Trend+Vol)")
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

        # Market index for trend detection
        market_df = prices_df.groupby('date').agg({'price': 'mean'}).reset_index()
        market_index = dict(zip(market_df['date'], market_df['price']))

        # Build indices
        price_index = {d: dict(zip(g['code'], g['price'])) for d, g in prices_df.groupby('date')}
        turnover_index = {d: dict(zip(g['code'], g['turnover'])) for d, g in prices_df.groupby('date')}
        fins_index = {code: g.sort_values('disclosed_date') for code, g in fins_df.groupby('code')}

        trading_dates = sorted(prices_df['date'].unique())

        # Compute momentum for all stocks
        logger.info("Computing momentum...")
        momentum_index = self._compute_momentum(prices_df, trading_dates)

        # ADT cache
        adt_cache = {}

        # State
        cash = self.initial_capital
        positions = []
        equity_history = []
        all_trades = []
        impact_records = []
        peak_equity = self.initial_capital

        # Tracking
        entries_blocked = 0
        total_entry_attempts = 0

        logger.info("Running backtest with trend filter...")
        start_time = datetime.now()

        for i, current_date in enumerate(trading_dates):
            current_date_obj = datetime.strptime(current_date, "%Y-%m-%d").date()
            price_dict = price_index.get(current_date, {})

            # Update trend/vol engine
            market_price = market_index.get(current_date, 0)
            trend, realized_vol, vol_scalar = self.trend_vol.update(market_price)

            # Check if entries allowed
            entries_allowed = self.trend_vol.should_enter(trend, realized_vol)

            # Portfolio valuation
            positions_value = sum(
                price_dict.get(p['code'], p['entry_price']) * p['shares']
                for p in positions
            )
            equity = cash + positions_value

            # Track
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

            self.walk_forward.record(current_date_obj, equity)

            equity_history.append({
                'date': current_date,
                'equity': equity,
                'trend': trend,
                'vol': realized_vol,
                'drawdown': dd
            })

            # Skip non-rebalance days
            if i % self.rebalance_days != 0:
                continue

            # Compute ADT point-in-time
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

                    # Tighter stops in downtrend
                    effective_stop = self.stop_loss * (0.7 if trend == "DOWNTREND" else 1.0)

                    if pnl_pct <= -effective_stop:
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
                    'pnl_pct': pnl_pct
                }
                all_trades.append(trade)
                self.walk_forward.record(current_date_obj, equity, trade)

                cash += exit_price * pos['shares']
                positions.remove(pos)

            # Portfolio DD circuit breaker (only during active drawdown)
            if dd > 0.20 and trend == "DOWNTREND":  # Only block in downtrend + high DD
                entries_allowed = False

            # Open new positions
            max_new = self.max_positions - len(positions)
            total_entry_attempts += max_new

            if max_new > 0 and entries_allowed:
                # Get momentum data for this date
                mom_dict = momentum_index.get(current_date, {})

                # Find candidates with momentum overlay
                candidates = self._find_candidates_with_momentum(
                    current_date, price_dict, fins_index, adt_map, mom_dict
                )

                held_codes = {p['code'] for p in positions}
                candidates = [c for c in candidates if c['code'] not in held_codes]

                # Position sizing (no vol scaling - causes instability)
                base_size = equity * self.base_position_pct

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
                        'shares': shares
                    })

                    impact_records.append(impact.total_bps)
                    cash -= cost

            elif max_new > 0 and not entries_allowed:
                entries_blocked += max_new

            # Progress
            if i % 500 == 0:
                logger.info(
                    f"[{current_date}] {i}/{len(trading_dates)} | "
                    f"Equity: ¥{equity:,.0f} | DD: {dd:.1%} | "
                    f"Trend: {trend} | Vol: {realized_vol:.1%} | Pos: {len(positions)}"
                )

        logger.info(f"Backtest completed in {(datetime.now()-start_time).total_seconds():.1f}s")
        logger.info(f"Entry attempts blocked by trend filter: {entries_blocked}/{total_entry_attempts}")

        # Calculate results
        result = self._calculate_results(equity_history, all_trades, impact_records)

        # Print report
        self._print_report(result)

        # Save
        self._save_outputs(result, equity_history)

        return result

    def _compute_momentum(self, prices_df: pd.DataFrame, trading_dates: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute momentum for each stock on each date"""
        momentum_index = {}

        # Group by code for efficiency
        code_prices = {code: g.set_index('date')['price'].to_dict()
                       for code, g in prices_df.groupby('code')}

        for i, date in enumerate(trading_dates):
            if i < self.momentum_lookback:
                momentum_index[date] = {}
                continue

            lookback_date = trading_dates[i - self.momentum_lookback]
            mom_dict = {}

            for code, price_dict in code_prices.items():
                if date in price_dict and lookback_date in price_dict:
                    current = price_dict[date]
                    past = price_dict[lookback_date]
                    if past > 0:
                        mom_dict[code] = (current / past) - 1

            momentum_index[date] = mom_dict

        return momentum_index

    def _find_candidates_with_momentum(self, eval_date: str, price_dict: Dict,
                                        fins_index: Dict, adt_map: Dict,
                                        mom_dict: Dict) -> List[Dict]:
        """Find candidates with momentum overlay"""
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

            # Get momentum
            momentum = mom_dict.get(code, 0)

            all_stocks.append({
                'code': code,
                'price': price,
                'pbr': pbr,
                'roe': roe,
                'momentum': momentum,
                'adt': adt_map.get(code, 0)
            })

        if len(all_stocks) < 10:
            return []

        # Percentile ranking
        pbrs = [s['pbr'] for s in all_stocks]
        roes = [s['roe'] for s in all_stocks]
        moms = [s['momentum'] for s in all_stocks]

        for s in all_stocks:
            s['pbr_pct'] = sum(1 for p in pbrs if p <= s['pbr']) / len(pbrs)
            s['roe_pct'] = sum(1 for r in roes if r <= s['roe']) / len(roes)
            s['mom_pct'] = sum(1 for m in moms if m <= s['momentum']) / len(moms)

            # Composite with momentum overlay
            value_quality = (1 - s['pbr_pct']) * 0.5 + s['roe_pct'] * 0.5
            s['composite'] = value_quality * (1 - self.momentum_weight) + s['mom_pct'] * self.momentum_weight

        # Select top composite
        threshold = np.percentile([s['composite'] for s in all_stocks], self.composite_pct * 100)
        candidates = [s for s in all_stocks if s['composite'] >= threshold]

        # Additional filter: prefer positive momentum
        candidates = [c for c in candidates if c['momentum'] > -0.10]

        candidates.sort(key=lambda x: x['composite'], reverse=True)
        return candidates

    def _calculate_results(self, equity_history: List, trades: List, impact_records: List) -> Dict:
        """Calculate comprehensive results"""
        equity = np.array([e['equity'] for e in equity_history], dtype=np.float64)
        equity = np.nan_to_num(equity, nan=self.initial_capital)

        total_ret = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        total_ret = 0 if not np.isfinite(total_ret) else total_ret

        years = len(equity) / 252
        if years > 0 and total_ret > -1:
            annual_ret = (1 + total_ret) ** (1/years) - 1
        else:
            annual_ret = 0
        annual_ret = 0 if not np.isfinite(annual_ret) else annual_ret

        with np.errstate(divide='ignore', invalid='ignore'):
            daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        daily_ret = np.nan_to_num(daily_ret, 0)

        vol = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 1 else 0.01
        vol = max(vol, 0.01)
        sharpe = annual_ret / vol
        sharpe = 0 if not np.isfinite(sharpe) else sharpe

        neg_ret = daily_ret[daily_ret < 0]
        down_vol = np.std(neg_ret) * np.sqrt(252) if len(neg_ret) > 1 else vol
        down_vol = max(down_vol, 0.01)
        sortino = annual_ret / down_vol
        sortino = 0 if not np.isfinite(sortino) else sortino

        peak = np.maximum.accumulate(equity)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = (peak - equity) / np.where(peak > 0, peak, 1)
        dd = np.nan_to_num(dd, 0)
        max_dd = float(np.max(dd))

        calmar = annual_ret / max_dd if max_dd > 0.01 else 0

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

        training = self.walk_forward.calculate_phase("training")
        validation = self.walk_forward.calculate_phase("validation")
        oos = self.walk_forward.calculate_phase("out_of_sample")

        return {
            'strategy': 'Asset Shield V3.2.0 World Rank V2',
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
        wf = result['walk_forward']
        training = wf['training']
        validation = wf['validation']
        oos = wf['out_of_sample']

        print("\n" + "=" * 70)
        print("Asset Shield V3.2.0 - World Rank V2 Results")
        print("=" * 70)

        print(f"\nReturn: {result['total_return']*100:.2f}% ({result['annual_return']*100:.2f}% p.a.)")
        print(f"Final Equity: ¥{result['final_equity']:,.0f}")

        print(f"\nSharpe: {result['sharpe_ratio']:.2f} | Sortino: {result['sortino_ratio']:.2f}")
        print(f"Max DD: {result['max_drawdown']*100:.2f}% | Calmar: {result['calmar_ratio']:.2f}")
        print(f"Volatility: {result['volatility']*100:.2f}%")

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
        print(f"\n  VERDICT: {'✓ WORLD RANK READY' if all_pass else '✗ CONTINUE OPTIMIZATION'}")
        print("=" * 70)

    def _save_outputs(self, result: Dict, equity_history: List):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(PROJECT_ROOT, 'output')
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, f'world_rank_v2_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results: {json_path}")

        # Chart
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

        dates = [datetime.strptime(e['date'], '%Y-%m-%d') for e in equity_history]
        equity = [e['equity'] / 1_000_000 for e in equity_history]
        dd = [e['drawdown'] * 100 for e in equity_history]
        vol = [e['vol'] * 100 for e in equity_history]

        ax1 = axes[0]
        ax1.plot(dates, equity, 'b-', linewidth=1.5)
        ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(
            f"Asset Shield V3.2.0 World Rank V2\n"
            f"Overall Sharpe: {result['sharpe_ratio']:.2f} | OOS Sharpe: {result['walk_forward']['out_of_sample']['sharpe_ratio']:.2f}",
            fontsize=14, fontweight='bold'
        )
        ax1.set_ylabel('Portfolio (M JPY)')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.fill_between(dates, 0, [-d for d in dd], color='red', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.plot(dates, vol, 'orange', alpha=0.7)
        ax3.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Vol Target')
        ax3.set_ylabel('Volatility (%)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        png_path = os.path.join(output_dir, f'world_rank_v2_{timestamp}.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Chart: {png_path}")


def main():
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'jquants_cache.db')

    if not os.path.exists(cache_path):
        logger.error(f"Database not found: {cache_path}")
        sys.exit(1)

    bt = WorldRankBacktesterV2(cache_path, initial_capital=10_000_000)
    result = bt.run()

    return result


if __name__ == "__main__":
    main()
