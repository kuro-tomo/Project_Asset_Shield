#!/usr/bin/env python3
"""
Asset Shield V3 - Optimized Backtest Engine
============================================
M4 Pro High-Speed Processing Optimized Version

Optimizations:
- NumPy vectorization for batch operations
- Pre-allocated arrays for equity tracking
- Chunked DataFrame processing
- Minimized object creation in hot loops
- Efficient dictionary lookups with defaultdict

Fixes (V3.1.0):
- Point-in-Time ADT calculation (no future data leakage)
- Comprehensive NaN/Inf guards throughout all calculations
- Historical data lookback for ADT warmup period
- Reduced ADT threshold (400M -> 100M) for better coverage

Author: Asset Shield V3 Team
Version: 3.1.0 (2026-02-04)
"""

import os
import sys
import sqlite3
import logging
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import hashlib

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Optimized Data Structures
# =============================================================================

class Position:
    """Memory-efficient position tracking"""
    __slots__ = ('code', 'entry_date_idx', 'entry_price', 'shares', 'impact_bps', 'composite_score')

    def __init__(self, code: str, entry_date_idx: int, entry_price: float,
                 shares: int, impact_bps: float, composite_score: float):
        self.code = code
        self.entry_date_idx = entry_date_idx
        self.entry_price = entry_price
        self.shares = shares
        self.impact_bps = impact_bps
        self.composite_score = composite_score


@dataclass
class PhaseResult:
    """Walk-forward phase result"""
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
    trades: List[Dict] = field(default_factory=list)


# =============================================================================
# Almgren-Chriss (Vectorized)
# =============================================================================

class AlmgrenChrissOptimized:
    """Vectorized Almgren-Chriss market impact model"""

    __slots__ = ('gamma', 'eta', 'sigma', 'spread_bps', 'max_participation', '_calc_count')

    GAMMA = 0.10
    ETA = 0.01
    SIGMA = 0.25
    SPREAD_BPS = 10.0
    MAX_PARTICIPATION = 0.10

    def __init__(self):
        self.gamma = self.GAMMA
        self.eta = self.ETA
        self.sigma = self.SIGMA
        self.spread_bps = self.SPREAD_BPS
        self.max_participation = self.MAX_PARTICIPATION
        self._calc_count = 0

    def calculate_batch(self, order_values: np.ndarray, adts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized batch impact calculation"""
        self._calc_count += len(order_values)

        # Avoid division by zero
        safe_adts = np.where(adts > 0, adts, 1)
        participation = order_values / safe_adts

        # Permanent impact
        permanent_bps = self.gamma * self.sigma * np.sqrt(participation) * 10000

        # Temporary impact
        temporary_bps = self.eta * self.sigma * (participation ** 0.6) * 10000

        # Total
        total_bps = permanent_bps + temporary_bps + (self.spread_bps / 2)

        # Executable mask
        executable = (participation <= self.max_participation * 2) & (total_bps < 100) & (adts > 0)

        return total_bps, executable

    def calculate_single(self, order_value: float, adt: float) -> Tuple[float, bool]:
        """Single order impact calculation"""
        self._calc_count += 1

        if adt <= 0:
            return float('inf'), False

        participation = order_value / adt
        permanent = self.gamma * self.sigma * np.sqrt(participation) * 10000
        temporary = self.eta * self.sigma * (participation ** 0.6) * 10000
        total = permanent + temporary + (self.spread_bps / 2)
        executable = participation <= self.max_participation * 2 and total < 100

        return round(total, 4), executable


# =============================================================================
# Walk-Forward Optimized
# =============================================================================

class WalkForwardOptimized:
    """Optimized walk-forward validation"""

    PHASES = {
        "training": (date(2007, 1, 1), date(2015, 12, 31)),
        "validation": (date(2016, 1, 1), date(2020, 12, 31)),
        "out_of_sample": (date(2021, 1, 1), date(2026, 12, 31))
    }

    def __init__(self):
        self._trades = {p: [] for p in self.PHASES}
        self._equity = {p: [] for p in self.PHASES}
        self._phase_dates = self._precompute_phase_dates()

    def _precompute_phase_dates(self) -> Dict[str, Tuple[int, int]]:
        """Pre-compute phase date boundaries as ordinals for fast lookup"""
        return {
            phase: (start.toordinal(), end.toordinal())
            for phase, (start, end) in self.PHASES.items()
        }

    def get_phase_fast(self, date_ordinal: int) -> str:
        """Fast phase lookup using ordinals"""
        for phase, (start_ord, end_ord) in self._phase_dates.items():
            if start_ord <= date_ordinal <= end_ord:
                return phase
        return "out_of_sample"

    def record_trade(self, trade: Dict, phase: str):
        """Record trade to specific phase"""
        self._trades[phase].append(trade)

    def record_equity(self, equity: float, phase: str):
        """Record daily equity"""
        self._equity[phase].append(equity)

    def calculate_metrics(self, phase: str) -> PhaseResult:
        """Calculate metrics for a phase using vectorized operations with robust NaN handling"""
        trades = self._trades[phase]
        equity_raw = self._equity[phase]
        start_date, end_date = self.PHASES[phase]

        # Filter out NaN values from equity
        equity = np.array([e for e in equity_raw if e is not None and np.isfinite(e)], dtype=np.float64)

        if len(equity) < 2:
            return PhaseResult(
                phase=phase, start_date=start_date, end_date=end_date,
                total_return=0.0, annual_return=0.0, sharpe_ratio=0.0,
                sortino_ratio=0.0, max_drawdown=0.0, total_trades=len(trades),
                win_rate=0.0, profit_factor=0.0, trades=trades
            )

        # Filter trades with valid pnl
        valid_trades = [t for t in trades if t.get('pnl') is not None and np.isfinite(t.get('pnl', 0))]

        # Vectorized return calculation with safety checks
        if equity[0] > 0:
            total_return = (equity[-1] / equity[0]) - 1
        else:
            total_return = 0.0
        total_return = 0.0 if not np.isfinite(total_return) else total_return

        years = (end_date - start_date).days / 365.25
        if years > 0 and (1 + total_return) > 0:
            annual_return = (1 + total_return) ** (1/years) - 1
        else:
            annual_return = 0.0
        annual_return = 0.0 if not np.isfinite(annual_return) else annual_return

        # Vectorized daily returns with safety
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Volatility (active returns only) with DoF safety
        active_mask = daily_returns != 0
        active_count = int(np.sum(active_mask))
        if active_count > 1:
            vol = float(np.std(daily_returns[active_mask], ddof=1)) * np.sqrt(252)
        elif len(daily_returns) > 1:
            vol = float(np.std(daily_returns, ddof=1)) * np.sqrt(252)
        else:
            vol = 0.0

        vol = 0.0 if not np.isfinite(vol) or vol <= 0 else vol
        sharpe = (annual_return - 0.001) / vol if vol > 0.01 else 0.0
        sharpe = 0.0 if not np.isfinite(sharpe) else sharpe

        # Sortino (vectorized) with safety
        neg_mask = daily_returns < 0
        neg_count = int(np.sum(neg_mask))
        if neg_count > 1:
            downside_vol = float(np.std(daily_returns[neg_mask], ddof=1)) * np.sqrt(252)
            downside_vol = vol if not np.isfinite(downside_vol) or downside_vol <= 0 else downside_vol
            sortino = (annual_return - 0.001) / downside_vol if downside_vol > 0.01 else 0.0
        else:
            sortino = sharpe * 1.5 if sharpe > 0 else 0.0
        sortino = 0.0 if not np.isfinite(sortino) else sortino

        # Max drawdown (vectorized)
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (peak - equity) / np.where(peak > 0, peak, 1)
        drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
        max_dd = float(np.max(drawdown))
        max_dd = 0.0 if not np.isfinite(max_dd) else max_dd

        # Trade statistics with NaN filtering
        if valid_trades:
            pnls = np.array([t.get('pnl', 0) for t in valid_trades], dtype=np.float64)
            winners = pnls > 0
            win_rate = float(np.mean(winners))
            total_wins = float(np.sum(pnls[winners]))
            total_losses = float(np.abs(np.sum(pnls[~winners])))
            pf = total_wins / total_losses if total_losses > 0 else (10.0 if total_wins > 0 else 0.0)
        else:
            win_rate = 0.0
            pf = 0.0

        return PhaseResult(
            phase=phase, start_date=start_date, end_date=end_date,
            total_return=round(total_return, 6), annual_return=round(annual_return, 6),
            sharpe_ratio=round(sharpe, 4), sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 6), total_trades=len(trades),
            win_rate=round(win_rate, 4), profit_factor=round(pf, 4), trades=trades
        )

    def analyze(self) -> Dict:
        """Full walk-forward analysis with robust NaN handling"""
        results = {phase: self.calculate_metrics(phase) for phase in self.PHASES}

        training = results["training"]
        validation = results["validation"]
        oos = results["out_of_sample"]

        # Safe ratio calculations
        if training.sharpe_ratio > 0 and np.isfinite(training.sharpe_ratio) and np.isfinite(validation.sharpe_ratio):
            overfitting_ratio = validation.sharpe_ratio / training.sharpe_ratio
        else:
            overfitting_ratio = 1.0

        if validation.sharpe_ratio > 0 and np.isfinite(validation.sharpe_ratio) and np.isfinite(oos.sharpe_ratio):
            degradation_ratio = oos.sharpe_ratio / validation.sharpe_ratio
        else:
            degradation_ratio = 1.0 if oos.sharpe_ratio > 0 else 0.0

        # Ensure finite values
        overfitting_ratio = 1.0 if not np.isfinite(overfitting_ratio) else overfitting_ratio
        degradation_ratio = 0.0 if not np.isfinite(degradation_ratio) else degradation_ratio

        passed = overfitting_ratio >= 0.70 and degradation_ratio >= 0.70

        return {
            "training": asdict(training),
            "validation": asdict(validation),
            "out_of_sample": asdict(oos),
            "overfitting_ratio": round(overfitting_ratio, 4),
            "degradation_ratio": round(degradation_ratio, 4),
            "passed_validation": str(passed)
        }


# =============================================================================
# Risk Buffer Optimized
# =============================================================================

class RiskBufferOptimized:
    """Optimized risk management with pre-computed thresholds"""

    __slots__ = ('peak', 'dd_active', '_equity_buffer', '_buffer_idx', '_vol')

    REGIME_CASH = {"crisis": 0.70, "high_vol": 0.40, "normal": 0.20, "low_vol": 0.10}
    DD_TRIGGER = 0.20
    DD_RECOVERY = 0.10
    VOL_LOOKBACK = 20
    CRISIS_VOL = 0.45
    HIGH_VOL = 0.28
    LOW_VOL = 0.18

    def __init__(self, initial_equity: float):
        self.peak = initial_equity
        self.dd_active = False
        self._equity_buffer = np.zeros(self.VOL_LOOKBACK, dtype=np.float64)
        self._equity_buffer[:] = initial_equity
        self._buffer_idx = 0
        self._vol = 0.20

    def update(self, equity: float) -> Tuple[float, str, bool]:
        """Update risk state and return (cash_alloc, regime, dd_active)"""
        self.peak = max(self.peak, equity)
        dd = (self.peak - equity) / self.peak if self.peak > 0 else 0

        # Update circular buffer
        self._equity_buffer[self._buffer_idx] = equity
        self._buffer_idx = (self._buffer_idx + 1) % self.VOL_LOOKBACK

        # DD protection
        if not self.dd_active and dd >= self.DD_TRIGGER:
            self.dd_active = True
        elif self.dd_active and dd <= self.DD_RECOVERY:
            self.dd_active = False

        # Volatility regime
        if self._buffer_idx == 0:  # Full buffer
            returns = np.diff(self._equity_buffer) / self._equity_buffer[:-1]
            self._vol = float(np.nanstd(returns) * np.sqrt(252))

        if np.isnan(self._vol):
            regime = "normal"
        elif self._vol > self.CRISIS_VOL:
            regime = "crisis"
        elif self._vol > self.HIGH_VOL:
            regime = "high_vol"
        elif self._vol < self.LOW_VOL:
            regime = "low_vol"
        else:
            regime = "normal"

        base_cash = self.REGIME_CASH.get(regime, 0.20)
        cash_alloc = min(0.85, base_cash + 0.50) if self.dd_active else base_cash

        return cash_alloc, regime, self.dd_active

    def get_multiplier(self) -> float:
        return 0.5 if self.dd_active else 1.0


# =============================================================================
# Main Backtester (Optimized)
# =============================================================================

class BacktesterV3Optimized:
    """
    Asset Shield V3 - M4 Pro Optimized Backtester

    Optimizations:
    - Pre-indexed data structures
    - Vectorized candidate selection
    - Minimized loop overhead
    - Efficient memory allocation
    """

    def __init__(self, cache_path: str, initial_capital: float = 10_000_000):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(cache_path)
        self.initial_capital = initial_capital

        # Components
        self.almgren = AlmgrenChrissOptimized()
        self.walk_forward = WalkForwardOptimized()
        self.risk_buffer = RiskBufferOptimized(initial_capital)

        # Strategy parameters (V3)
        self.max_positions = 15
        self.position_pct = 0.08
        self.rebalance_days = 63
        self.holding_days = 250
        self.pbr_pct = 0.20
        self.roe_pct = 0.80
        self.composite_pct = 0.80
        self.min_adt = 100_000_000  # Reduced from 400M to 100M for better liquidity coverage
        self.stop_loss = -0.15
        self.take_profit = 0.40
        self.adt_lookback = 60  # Point-in-Time ADT lookback window

    def run(self, start_date: str = "2008-01-01", end_date: str = "2026-01-31") -> Dict:
        """Execute optimized backtest"""
        logger.info("=" * 70)
        logger.info("Asset Shield V3 - Optimized Backtest Engine")
        logger.info(f"Period: {start_date} -> {end_date}")
        logger.info("=" * 70)

        # Load data
        logger.info("Loading data...")
        t0 = datetime.now()
        prices_df, fins_df = self._load_data(start_date, end_date)
        logger.info(f"Data loaded: {len(prices_df):,} records in {(datetime.now()-t0).total_seconds():.1f}s")

        # Build indices
        logger.info("Building indices...")
        t0 = datetime.now()
        adt_map, tradeable = self._calc_adt(prices_df)
        price_idx = self._build_price_index(prices_df)
        fins_idx = self._build_fins_index(fins_df)

        # Only trade on dates in the specified period
        trading_dates = sorted([d for d in price_idx.keys() if d >= start_date and d <= end_date])
        date_to_idx = {d: i for i, d in enumerate(trading_dates)}
        logger.info(f"Indices built in {(datetime.now()-t0).total_seconds():.1f}s")
        logger.info(f"Trading days: {len(trading_dates)}, Tradeable stocks: {len(tradeable)}")

        # State
        cash = self.initial_capital
        positions: List[Position] = []
        trades = []
        impact_records = []
        trades_by_year = defaultdict(int)

        # Pre-allocate equity array
        n_days = len(trading_dates)
        equity_arr = np.zeros(n_days, dtype=np.float64)
        dd_arr = np.zeros(n_days, dtype=np.float64)

        logger.info("Running backtest...")
        t0 = datetime.now()
        peak = self.initial_capital

        for i, current_date in enumerate(trading_dates):
            date_obj = datetime.strptime(current_date, "%Y-%m-%d").date()
            date_ord = date_obj.toordinal()
            phase = self.walk_forward.get_phase_fast(date_ord)

            price_dict = price_idx.get(current_date, {})

            # Portfolio valuation
            pos_value = sum(
                price_dict.get(p.code, p.entry_price) * p.shares
                for p in positions
            )
            equity = cash + pos_value
            equity_arr[i] = equity

            # Track peak and DD
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            dd_arr[i] = dd

            # Risk buffer
            cash_alloc, regime, dd_active = self.risk_buffer.update(equity)

            # Walk-forward recording
            self.walk_forward.record_equity(equity, phase)

            # Rebalance check
            if i % self.rebalance_days != 0:
                continue

            # Close positions
            to_close = []
            for pos in positions:
                holding = i - pos.entry_date_idx

                if holding >= self.holding_days:
                    to_close.append(pos)
                    continue

                if pos.code in price_dict:
                    current_price = price_dict[pos.code]
                    # NaN guard for current price in stop-loss/take-profit check
                    if current_price is not None and np.isfinite(current_price) and current_price > 0 and pos.entry_price > 0:
                        pnl_pct = (current_price / pos.entry_price) - 1
                        if pnl_pct <= self.stop_loss or pnl_pct >= self.take_profit:
                            to_close.append(pos)

            for pos in to_close:
                if pos.code not in price_dict:
                    continue

                exit_price = price_dict[pos.code]
                # NaN guard for exit price
                if exit_price is None or not np.isfinite(exit_price) or exit_price <= 0:
                    # Use entry price as fallback (neutral exit)
                    exit_price = pos.entry_price

                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price / pos.entry_price) - 1 if pos.entry_price > 0 else 0.0
                # Ensure pnl values are finite
                pnl = 0.0 if not np.isfinite(pnl) else pnl
                pnl_pct = 0.0 if not np.isfinite(pnl_pct) else pnl_pct

                entry_date_str = trading_dates[pos.entry_date_idx]
                trade = {
                    'code': pos.code,
                    'entry_date': entry_date_str,
                    'exit_date': current_date,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'shares': pos.shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'impact_bps': pos.impact_bps,
                    'phase': phase
                }
                trades.append(trade)

                # Record to correct phase based on exit date
                exit_date_ord = date_obj.toordinal()
                correct_phase = self.walk_forward.get_phase_fast(exit_date_ord)
                self.walk_forward.record_trade(trade, correct_phase)
                trades_by_year[date_obj.year] += 1

                cash += exit_price * pos.shares
                positions.remove(pos)

            # New positions
            if dd_active:
                max_new = max(0, self.max_positions // 2 - len(positions))
            elif regime == "crisis":
                max_new = max(0, 2 - len(positions))
            elif regime == "high_vol":
                max_new = max(0, self.max_positions // 2 - len(positions))
            else:
                max_new = self.max_positions - len(positions)

            if max_new > 0:
                base_size = equity * self.position_pct
                adj_size = base_size * self.risk_buffer.get_multiplier()

                # Get Point-in-Time ADT for current date
                date_adt = adt_map.get(current_date, {})
                pit_tradeable = {c for c, a in date_adt.items() if a >= self.min_adt}

                candidates = self._find_candidates_fast(
                    current_date, price_dict, fins_idx, pit_tradeable, date_adt
                )

                held = {p.code for p in positions}
                candidates = [c for c in candidates if c['code'] not in held]

                for cand in candidates[:max_new]:
                    # Point-in-Time ADT lookup
                    date_adt = adt_map.get(current_date, {})
                    adt = date_adt.get(cand['code'], 0)
                    impact, executable = self.almgren.calculate_single(adj_size, adt)

                    if not executable:
                        continue

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
                        entry_date_idx=i,
                        entry_price=cand['price'],
                        shares=shares,
                        impact_bps=impact,
                        composite_score=cand['score']
                    ))
                    impact_records.append(impact)
                    cash -= cost

            # Progress
            if i % 500 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                speed = i / elapsed if elapsed > 0 else 0
                eta = (n_days - i) / speed if speed > 0 else 0
                logger.info(
                    f"[{current_date}] {i}/{n_days} ({i/n_days*100:.1f}%) | "
                    f"Equity: {equity/1e6:.1f}M | DD: {dd:.1%} | Pos: {len(positions)} | "
                    f"Speed: {speed:.0f} days/s | ETA: {eta/60:.1f}m"
                )

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f"Backtest completed in {elapsed:.1f}s ({n_days/elapsed:.0f} days/s)")

        # Calculate results
        result = self._calculate_results(equity_arr, dd_arr, trades, impact_records, trading_dates)
        result['trades_by_year'] = dict(trades_by_year)

        # Print report
        self._print_report(result)

        # Save chart
        self._save_chart(equity_arr, dd_arr, trading_dates, result)

        return result

    def _load_data(self, start_date: str, end_date: str):
        # Calculate lookback start date for ADT calculation (90 days before start)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        adt_lookback_start = (start_dt - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

        # Load prices with ADT lookback period
        prices = pd.read_sql_query(
            "SELECT code, date, close, turnover, adjustment_close FROM daily_quotes WHERE date BETWEEN ? AND ? ORDER BY date, code",
            self.conn, params=[adt_lookback_start, end_date]
        )
        prices['price'] = prices['adjustment_close'].fillna(prices['close'])
        # Fill NaN turnover with 0
        prices['turnover'] = prices['turnover'].fillna(0)
        # Remove rows with invalid prices
        prices = prices[prices['price'].notna() & (prices['price'] > 0)]

        # Mark which rows are in the actual trading period
        prices['in_period'] = prices['date'] >= start_date

        logger.info(f"Loaded data from {adt_lookback_start} to {end_date} ({len(prices)} rows, {prices['in_period'].sum()} in trading period)")

        fins = pd.read_sql_query(
            "SELECT code, disclosed_date, bps, roe FROM financial_statements WHERE bps > 0 ORDER BY code, disclosed_date",
            self.conn
        )
        return prices, fins

    def _calc_adt(self, df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], set]:
        """
        Point-in-Time ADT calculation - returns ADT by date and code.
        This ensures no future data leakage in liquidity assessment.
        """
        # Sort by code and date
        df_sorted = df.sort_values(['code', 'date'])

        # Calculate rolling ADT for each code (Point-in-Time)
        adt_by_date_code = {}
        all_codes = set()

        # Minimum days needed - use 3 for short periods, but prefer 10+
        min_days_required = 3

        for code, group in df_sorted.groupby('code'):
            group = group.sort_values('date')
            turnover_values = group['turnover'].fillna(0).values
            dates = group['date'].values

            # Rolling mean with lookback window
            for i, (d, tv) in enumerate(zip(dates, turnover_values)):
                start_idx = max(0, i - self.adt_lookback + 1)
                window = turnover_values[start_idx:i+1]

                # Use available data, minimum 3 days (or current day value as fallback)
                if len(window) >= min_days_required:
                    adt_val = float(np.mean(window))
                elif len(window) > 0:
                    # For very short periods, use available data with higher threshold
                    adt_val = float(np.mean(window))
                else:
                    adt_val = 0.0

                if d not in adt_by_date_code:
                    adt_by_date_code[d] = {}
                adt_by_date_code[d][code] = adt_val

                if adt_val >= self.min_adt:
                    all_codes.add(code)

        logger.info(f"PIT ADT computed: {len(adt_by_date_code)} dates, {len(all_codes)} tradeable codes")
        return adt_by_date_code, all_codes

    def _build_price_index(self, df: pd.DataFrame) -> Dict:
        return {d: dict(zip(g['code'], g['price'])) for d, g in df.groupby('date')}

    def _build_fins_index(self, df: pd.DataFrame) -> Dict:
        return {code: g.sort_values('disclosed_date') for code, g in df.groupby('code')}

    def _find_candidates_fast(self, eval_date, price_dict, fins_idx, tradeable, date_adt) -> List[Dict]:
        """Fast candidate selection using vectorized operations with PIT data"""
        stocks = []

        for code in tradeable:
            if code not in price_dict:
                continue
            price = price_dict[code]
            # NaN guard for price
            if price is None or not np.isfinite(price) or price <= 0:
                continue
            if code not in fins_idx:
                continue

            code_fins = fins_idx[code]
            valid = code_fins[code_fins['disclosed_date'] <= eval_date]
            if valid.empty:
                continue

            latest = valid.iloc[-1]
            bps = latest['bps']
            roe = latest['roe'] if pd.notna(latest['roe']) else 0

            # NaN guard for bps
            if bps is None or not np.isfinite(bps) or bps <= 0:
                continue

            pbr = price / bps
            # NaN guard for pbr
            if not np.isfinite(pbr):
                continue

            stocks.append({'code': code, 'price': price, 'pbr': pbr, 'roe': roe})

        if not stocks:
            return []

        # Vectorized percentile calculation
        pbrs = np.array([s['pbr'] for s in stocks])
        roes = np.array([s['roe'] for s in stocks])

        pbr_pct = np.argsort(np.argsort(pbrs)) / len(pbrs)
        roe_pct = np.argsort(np.argsort(roes)) / len(roes)

        scores = (1 - pbr_pct) * 0.5 + roe_pct * 0.5
        threshold = np.percentile(scores, self.composite_pct * 100)

        candidates = []
        for i, s in enumerate(stocks):
            if scores[i] >= threshold:
                candidates.append({'code': s['code'], 'price': s['price'], 'score': scores[i]})

        candidates.sort(key=lambda x: -x['score'])
        return candidates

    def _calculate_results(self, equity_arr, dd_arr, trades, impacts, dates) -> Dict:
        """Calculate results with robust NaN handling"""
        # Guard against empty or invalid equity array
        equity_arr = np.nan_to_num(equity_arr, nan=self.initial_capital, posinf=self.initial_capital, neginf=0)

        final = equity_arr[-1] if len(equity_arr) > 0 else self.initial_capital
        if final <= 0 or not np.isfinite(final):
            final = self.initial_capital

        total_ret = (final / self.initial_capital) - 1
        total_ret = 0.0 if not np.isfinite(total_ret) else total_ret

        n_years = len(equity_arr) / 252
        if n_years > 0 and (1 + total_ret) > 0:
            annual_ret = (1 + total_ret) ** (1/n_years) - 1
        else:
            annual_ret = 0.0
        annual_ret = 0.0 if not np.isfinite(annual_ret) else annual_ret

        # Safe daily return calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            daily_ret = np.diff(equity_arr) / np.where(equity_arr[:-1] > 0, equity_arr[:-1], 1)
        daily_ret = np.nan_to_num(daily_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # Volatility with DoF check
        if len(daily_ret) > 1:
            vol = float(np.std(daily_ret, ddof=1)) * np.sqrt(252)
        else:
            vol = 0.0
        vol = 0.0 if not np.isfinite(vol) or vol <= 0 else vol

        sharpe = (annual_ret - 0.001) / vol if vol > 0.001 else 0.0
        sharpe = 0.0 if not np.isfinite(sharpe) else sharpe

        neg = daily_ret[daily_ret < 0]
        if len(neg) > 1:
            down_vol = float(np.std(neg, ddof=1)) * np.sqrt(252)
        else:
            down_vol = vol
        down_vol = vol if not np.isfinite(down_vol) or down_vol <= 0 else down_vol

        sortino = (annual_ret - 0.001) / down_vol if down_vol > 0.001 else 0.0
        sortino = 0.0 if not np.isfinite(sortino) else sortino

        max_dd = float(np.nanmax(dd_arr)) if len(dd_arr) > 0 else 0.0
        max_dd = 0.0 if not np.isfinite(max_dd) else max_dd

        calmar = annual_ret / max_dd if max_dd > 0.001 else 0.0
        calmar = 0.0 if not np.isfinite(calmar) else calmar

        if trades:
            # Filter out trades with NaN pnl
            valid_trades = [t for t in trades if np.isfinite(t.get('pnl', 0))]
            if valid_trades:
                pnls = np.array([t['pnl'] for t in valid_trades])
                win_rate = float(np.mean(pnls > 0))
                wins = float(np.sum(pnls[pnls > 0]))
                losses = float(np.abs(np.sum(pnls[pnls <= 0])))
                pf = wins / losses if losses > 0 else (10.0 if wins > 0 else 0.0)
                avg_hold = np.mean([
                    (datetime.strptime(t['exit_date'], "%Y-%m-%d") -
                     datetime.strptime(t['entry_date'], "%Y-%m-%d")).days
                    for t in valid_trades
                ])
            else:
                win_rate = pf = avg_hold = 0.0
        else:
            win_rate = pf = avg_hold = 0.0

        avg_impact = float(np.mean(impacts)) if impacts else 0.0
        avg_impact = 0.0 if not np.isfinite(avg_impact) else avg_impact

        wf = self.walk_forward.analyze()

        # Final safety checks - replace any remaining NaN/Inf with 0
        def safe_round(val, decimals):
            if val is None or not np.isfinite(val):
                return 0.0
            return round(float(val), decimals)

        return {
            'strategy': 'Asset Shield V3 (Optimized)',
            'total_return': safe_round(total_ret, 6),
            'annual_return': safe_round(annual_ret, 6),
            'final_equity': safe_round(final, 0),
            'sharpe_ratio': safe_round(sharpe, 4),
            'sortino_ratio': safe_round(sortino, 4),
            'max_drawdown': safe_round(max_dd, 6),
            'calmar_ratio': safe_round(calmar, 4),
            'volatility': safe_round(vol, 6),
            'total_trades': len(trades),
            'win_rate': safe_round(win_rate, 4),
            'profit_factor': safe_round(pf, 4),
            'avg_holding_days': safe_round(avg_hold, 1),
            'avg_impact_bps': safe_round(avg_impact, 2),
            'walk_forward': wf,
            'almgren_calculations': self.almgren._calc_count
        }

    def _print_report(self, r: Dict):
        wf = r['walk_forward']
        print("\n" + "=" * 70)
        print("Asset Shield V3 - Optimized Backtest Results")
        print("=" * 70)
        print(f"\nReturn: {r['total_return']:.2%} ({r['annual_return']:.2%} p.a.)")
        print(f"Final Equity: ¥{r['final_equity']:,.0f}")
        print(f"\nSharpe: {r['sharpe_ratio']:.2f} | Sortino: {r['sortino_ratio']:.2f}")
        print(f"Max DD: {r['max_drawdown']:.2%} | Calmar: {r['calmar_ratio']:.2f}")
        print(f"\nTrades: {r['total_trades']} | Win Rate: {r['win_rate']:.2%}")
        print(f"Profit Factor: {r['profit_factor']:.2f} | Avg Hold: {r['avg_holding_days']:.0f}d")
        print(f"Avg Impact: {r['avg_impact_bps']:.1f}bps")
        print(f"\nWalk-Forward:")
        print(f"  Training:   Sharpe {wf['training']['sharpe_ratio']:.2f}, {wf['training']['total_trades']} trades")
        print(f"  Validation: Sharpe {wf['validation']['sharpe_ratio']:.2f}, {wf['validation']['total_trades']} trades")
        print(f"  OOS:        Sharpe {wf['out_of_sample']['sharpe_ratio']:.2f}, {wf['out_of_sample']['total_trades']} trades")
        print(f"\nWorld Rank Check:")
        print(f"  Sharpe >= 1.0: {r['sharpe_ratio']:.2f} {'✓' if r['sharpe_ratio'] >= 1.0 else '✗'}")
        print(f"  OOS >= 0.7:    {wf['out_of_sample']['sharpe_ratio']:.2f} {'✓' if wf['out_of_sample']['sharpe_ratio'] >= 0.7 else '✗'}")
        print(f"  Impact <= 20:  {r['avg_impact_bps']:.1f} {'✓' if r['avg_impact_bps'] <= 20 else '✗'}")
        print("=" * 70)

    def _save_chart(self, equity, dd, dates, result):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

        x = np.arange(len(dates))
        ax1.plot(x, equity/1e6, 'b-', lw=1)
        ax1.set_ylabel('Portfolio (M¥)')
        ax1.set_title(f"Asset Shield V3 | Sharpe: {result['sharpe_ratio']:.2f} | OOS: {result['walk_forward']['out_of_sample']['sharpe_ratio']:.2f}")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(x, 0, -dd*100, color='red', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_ylim(-50, 5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(PROJECT_ROOT, 'output', f'v3_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(out_path, dpi=120)
        plt.close()
        logger.info(f"Chart saved: {out_path}")


def main():
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'jquants_cache.db')
    if not os.path.exists(cache_path):
        logger.error(f"Database not found: {cache_path}")
        sys.exit(1)

    bt = BacktesterV3Optimized(cache_path)
    result = bt.run()

    out_json = os.path.join(PROJECT_ROOT, 'output', f'v3_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved: {out_json}")


if __name__ == "__main__":
    main()
