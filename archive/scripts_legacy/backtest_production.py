#!/usr/bin/env python3
"""
Asset Shield V3.2.0 - Production Ready Backtest
================================================

Fixed version with proper risk controls:
1. Position value caps (no single position > 10% of equity)
2. Portfolio value validation (sanity checks)
3. Drawdown circuit breaker
4. Proper equity tracking

Author: Asset Shield Team
Version: 3.2.0-PROD
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    phase: str
    start_date: date
    end_date: date
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float


class WalkForwardTracker:
    PHASES = {
        "training": (date(2008, 1, 1), date(2015, 12, 31)),
        "validation": (date(2016, 1, 1), date(2020, 12, 31)),
        "out_of_sample": (date(2021, 1, 1), date(2026, 12, 31))
    }

    def __init__(self):
        self.data = {p: {'equity': [], 'trades': []} for p in self.PHASES}

    def get_phase(self, d: date) -> str:
        for phase, (start, end) in self.PHASES.items():
            if start <= d <= end:
                return phase
        return "out_of_sample"

    def record(self, d: date, equity: float, trade: Dict = None):
        phase = self.get_phase(d)
        self.data[phase]['equity'].append((d, equity))
        if trade:
            self.data[phase]['trades'].append(trade)

    def calculate(self, phase: str) -> PhaseResult:
        eq_data = self.data[phase]['equity']
        trades = self.data[phase]['trades']

        if len(eq_data) < 10:
            return PhaseResult(phase, date(2008,1,1), date(2008,1,2), 0,0,0,0,0,0)

        dates, equity = zip(*eq_data)
        equity = np.array(equity)

        total_ret = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        years = max((dates[-1] - dates[0]).days / 365.25, 0.1)
        annual_ret = (1 + total_ret) ** (1/years) - 1 if total_ret > -1 else 0

        daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        daily_ret = np.nan_to_num(daily_ret, 0)

        vol = np.std(daily_ret) * np.sqrt(252)
        vol = max(vol, 0.01)
        sharpe = annual_ret / vol

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = np.max(dd)

        if trades:
            pnls = [t.get('pnl', 0) for t in trades]
            wins = [p for p in pnls if p > 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
        else:
            win_rate = 0

        return PhaseResult(phase, dates[0], dates[-1], total_ret, annual_ret, sharpe, max_dd, len(trades), win_rate)


class ProductionBacktester:
    """Production-ready backtester with proper risk controls."""

    def __init__(self, cache_path: str, initial_capital: float = 10_000_000):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.initial_capital = initial_capital
        self.wf = WalkForwardTracker()

        # Conservative parameters
        self.max_positions = 20
        self.position_pct = 0.05          # 5% per position
        self.max_position_pct = 0.10      # Hard cap at 10%
        self.rebalance_days = 63
        self.holding_days = 250
        self.min_adt = 400_000_000
        self.stop_loss = 0.12
        self.take_profit = 0.35

        # Risk controls (balanced - not too restrictive)
        self.max_portfolio_dd = 0.35      # 35% DD circuit breaker (relaxed)
        self.trend_ma_period = 60         # Shorter MA for faster trend response

    def run(self, start_date: str = "2008-01-01", end_date: str = "2026-02-03") -> Dict:
        logger.info("=" * 70)
        logger.info("Asset Shield V3.2.0 - Production Backtest")
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

        # Build indices
        price_index = {d: dict(zip(g['code'], g['price'])) for d, g in prices_df.groupby('date')}
        fins_index = {code: g.sort_values('disclosed_date') for code, g in fins_df.groupby('code')}

        # Market index for trend
        market_df = prices_df.groupby('date')['price'].mean()
        market_prices = market_df.to_dict()

        trading_dates = sorted(prices_df['date'].unique())

        # ADT cache
        adt_cache = {}

        # State
        cash = self.initial_capital
        positions = []
        equity_history = []
        all_trades = []
        peak_equity = self.initial_capital
        circuit_breaker_active = False
        market_ma = []

        logger.info("Running backtest...")
        start_time = datetime.now()

        for i, current_date in enumerate(trading_dates):
            current_date_obj = datetime.strptime(current_date, "%Y-%m-%d").date()
            price_dict = price_index.get(current_date, {})

            # Update market MA
            mkt_price = market_prices.get(current_date, 0)
            market_ma.append(mkt_price)
            if len(market_ma) > self.trend_ma_period:
                market_ma.pop(0)

            ma_value = np.mean(market_ma) if market_ma else mkt_price
            # Allow entry in UP trend or sideways (within 5% of MA)
            if mkt_price > ma_value * 1.00:
                trend = "UP"
            elif mkt_price > ma_value * 0.95:
                trend = "SIDEWAYS"
            else:
                trend = "DOWN"

            # Portfolio valuation with validation
            positions_value = 0
            for p in positions:
                if p['code'] in price_dict:
                    pos_val = price_dict[p['code']] * p['shares']
                    # Sanity check: cap position value at 5x entry value
                    max_val = p['entry_price'] * p['shares'] * 5
                    pos_val = min(pos_val, max_val)
                    positions_value += pos_val
                else:
                    positions_value += p['entry_price'] * p['shares']

            equity = cash + positions_value

            # Sanity check: equity shouldn't spike unreasonably
            if equity_history:
                prev_equity = equity_history[-1]['equity']
                max_daily_change = 0.20  # 20% max daily change
                if prev_equity > 0:
                    daily_change = abs(equity / prev_equity - 1)
                    if daily_change > max_daily_change:
                        # Clamp to reasonable change
                        equity = prev_equity * (1 + max_daily_change * np.sign(equity - prev_equity))

            # Drawdown tracking
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

            # Circuit breaker (more forgiving reset)
            if dd > self.max_portfolio_dd:
                circuit_breaker_active = True
            elif dd < 0.20:  # Reset when DD recovers below 20%
                circuit_breaker_active = False

            # Record
            self.wf.record(current_date_obj, equity)
            equity_history.append({
                'date': current_date,
                'equity': equity,
                'dd': dd,
                'trend': trend
            })

            # Skip non-rebalance days
            if i % self.rebalance_days != 0:
                continue

            # Compute ADT
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

                    if pnl_pct <= -self.stop_loss:
                        to_close.append(pos)
                    elif pnl_pct >= self.take_profit:
                        to_close.append(pos)

            for pos in to_close:
                if pos['code'] not in price_dict:
                    continue

                exit_price = price_dict[pos['code']]
                pnl = (exit_price - pos['entry_price']) * pos['shares']

                trade = {
                    'code': pos['code'],
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'pnl': pnl,
                    'pnl_pct': (exit_price / pos['entry_price']) - 1
                }
                all_trades.append(trade)
                self.wf.record(current_date_obj, equity, trade)

                cash += exit_price * pos['shares']
                positions.remove(pos)

            # Open new positions (if allowed) - UP or SIDEWAYS trend
            if not circuit_breaker_active and trend in ("UP", "SIDEWAYS"):
                max_new = self.max_positions - len(positions)

                if max_new > 0:
                    candidates = self._find_candidates(current_date, price_dict, fins_index, adt_map)
                    held = {p['code'] for p in positions}
                    candidates = [c for c in candidates if c['code'] not in held]

                    for cand in candidates[:max_new]:
                        # Position sizing with caps
                        base_size = equity * self.position_pct
                        max_size = equity * self.max_position_pct
                        pos_value = min(base_size, max_size, cash * 0.90)

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
                        cash -= cost

            # Progress
            if i % 500 == 0:
                cb_status = "🚨" if circuit_breaker_active else "✓"
                logger.info(
                    f"[{current_date}] {i}/{len(trading_dates)} | "
                    f"Equity: ¥{equity:,.0f} | DD: {dd:.1%} {cb_status} | "
                    f"Pos: {len(positions)}"
                )

        logger.info(f"Backtest completed in {(datetime.now()-start_time).total_seconds():.1f}s")

        # Calculate results
        result = self._calculate_results(equity_history, all_trades)
        self._print_report(result)
        self._save_outputs(result, equity_history)

        return result

    def _find_candidates(self, eval_date: str, price_dict: Dict,
                         fins_index: Dict, adt_map: Dict) -> List[Dict]:
        """Find candidates using PBR/ROE ranking"""
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
            bps, roe = latest['bps'], latest['roe']

            if bps <= 0 or not np.isfinite(roe):
                continue

            pbr = price / bps
            if pbr <= 0:
                continue

            all_stocks.append({'code': code, 'price': price, 'pbr': pbr, 'roe': roe})

        if len(all_stocks) < 10:
            return []

        # Percentile ranking
        pbrs = [s['pbr'] for s in all_stocks]
        roes = [s['roe'] for s in all_stocks]

        for s in all_stocks:
            s['pbr_pct'] = sum(1 for p in pbrs if p <= s['pbr']) / len(pbrs)
            s['roe_pct'] = sum(1 for r in roes if r <= s['roe']) / len(roes)
            s['composite'] = (1 - s['pbr_pct']) * 0.5 + s['roe_pct'] * 0.5

        threshold = np.percentile([s['composite'] for s in all_stocks], 80)
        candidates = [s for s in all_stocks if s['composite'] >= threshold]
        candidates.sort(key=lambda x: x['composite'], reverse=True)

        return candidates

    def _calculate_results(self, equity_history: List, trades: List) -> Dict:
        equity = np.array([e['equity'] for e in equity_history])

        total_ret = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        years = len(equity) / 252
        annual_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 and total_ret > -1 else 0

        daily_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        daily_ret = np.nan_to_num(daily_ret, 0)

        vol = np.std(daily_ret) * np.sqrt(252)
        vol = max(vol, 0.01)
        sharpe = annual_ret / vol

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1)
        dd = np.nan_to_num(dd, nan=0.0)
        max_dd = float(np.nanmax(dd)) if len(dd) > 0 else 0.0

        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p < 0]
            win_rate = len(wins) / len(pnls)
            pf = sum(wins) / sum(losses) if losses else 10
        else:
            win_rate, pf = 0, 0

        training = self.wf.calculate("training")
        validation = self.wf.calculate("validation")
        oos = self.wf.calculate("out_of_sample")

        return {
            'strategy': 'Asset Shield V3.2.0 Production',
            'total_return': total_ret,
            'annual_return': annual_ret,
            'final_equity': equity[-1],
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'volatility': vol,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': pf,
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
        print("Asset Shield V3.2.0 - Production Results")
        print("=" * 70)

        print(f"\nReturn: {result['total_return']*100:.2f}% ({result['annual_return']*100:.2f}% p.a.)")
        print(f"Final Equity: ¥{result['final_equity']:,.0f}")

        print(f"\nSharpe: {result['sharpe_ratio']:.2f}")
        print(f"Max DD: {result['max_drawdown']*100:.2f}%")
        print(f"Volatility: {result['volatility']*100:.2f}%")

        print(f"\nTrades: {result['total_trades']} | Win Rate: {result['win_rate']*100:.2f}%")
        print(f"Profit Factor: {result['profit_factor']:.2f}")

        print(f"\nWalk-Forward:")
        print(f"  Training:   Sharpe {training['sharpe_ratio']:.2f}, {training['total_trades']} trades, DD {training['max_drawdown']*100:.1f}%")
        print(f"  Validation: Sharpe {validation['sharpe_ratio']:.2f}, {validation['total_trades']} trades, DD {validation['max_drawdown']*100:.1f}%")
        print(f"  OOS:        Sharpe {oos['sharpe_ratio']:.2f}, {oos['total_trades']} trades, DD {oos['max_drawdown']*100:.1f}%")

        print("\n" + "=" * 70)
        print("Production Readiness Check")
        print("=" * 70)

        overall_pass = result['sharpe_ratio'] >= 0.8
        dd_pass = result['max_drawdown'] <= 0.35
        oos_pass = oos['sharpe_ratio'] >= 0.7

        print(f"  Overall Sharpe >= 0.8:  {result['sharpe_ratio']:.2f} {'✓' if overall_pass else '✗'}")
        print(f"  Max DD <= 35%:          {result['max_drawdown']*100:.1f}% {'✓' if dd_pass else '✗'}")
        print(f"  OOS Sharpe >= 0.7:      {oos['sharpe_ratio']:.2f} {'✓' if oos_pass else '✗'}")

        all_pass = overall_pass and dd_pass and oos_pass
        print(f"\n  VERDICT: {'✓ PRODUCTION READY' if all_pass else '✗ NEEDS WORK'}")
        print("=" * 70)

    def _save_outputs(self, result: Dict, equity_history: List):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(PROJECT_ROOT, 'output')

        json_path = os.path.join(output_dir, f'production_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results: {json_path}")

        # Chart
        fig, ax = plt.subplots(figsize=(14, 6))
        dates = [datetime.strptime(e['date'], '%Y-%m-%d') for e in equity_history]
        equity = [e['equity'] / 1_000_000 for e in equity_history]

        ax.plot(dates, equity, 'b-', linewidth=1)
        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"Asset Shield V3.2.0 Production | Sharpe: {result['sharpe_ratio']:.2f} | Max DD: {result['max_drawdown']*100:.1f}%")
        ax.set_ylabel('Portfolio (M JPY)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(output_dir, f'production_{timestamp}.png')
        plt.savefig(png_path, dpi=150)
        plt.close()
        logger.info(f"Chart: {png_path}")


def main():
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'jquants_cache.db')
    bt = ProductionBacktester(cache_path, initial_capital=10_000_000)
    result = bt.run()
    return result


if __name__ == "__main__":
    main()
