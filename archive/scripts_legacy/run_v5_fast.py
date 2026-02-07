#!/usr/bin/env python3
"""
Asset Shield V5.0 - Fast 18-Year Backtest (Optimized)
Uses top 50 liquid stocks only for speed
"""
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "/Users/MBP/Desktop/Project_Asset_Shield/data/jquants_cache.db"

@dataclass
class Config:
    start_date: date = date(2008, 5, 1)
    end_date: date = date(2026, 2, 1)
    initial_capital: float = 100_000_000
    max_positions: int = 20
    max_single_weight: float = 0.08
    max_sector_weight: float = 0.25
    min_adt: float = 100_000_000
    vol_threshold_bull: float = 0.18
    vol_threshold_crisis: float = 0.35
    max_drawdown: float = 0.15
    recovery_steps: List[float] = None
    recovery_days: int = 5
    rebalance_freq: int = 20

    def __post_init__(self):
        if self.recovery_steps is None:
            self.recovery_steps = [0.0, 0.30, 0.60, 0.90]


def get_liquid_stocks(conn, n=50):
    """Get top N liquid stocks by average turnover"""
    df = pd.read_sql_query("""
        SELECT code, AVG(adjustment_close * volume) as avg_turnover
        FROM daily_quotes
        WHERE date >= '2020-01-01'
        GROUP BY code
        HAVING avg_turnover > 100000000
        ORDER BY avg_turnover DESC
        LIMIT ?
    """, conn, params=(n,))
    return df['code'].tolist()


def get_sector_map(conn, codes):
    """Get sector mapping for codes"""
    placeholders = ','.join(['?' for _ in codes])
    df = pd.read_sql_query(f"""
        SELECT code, sector17_code FROM listed_info
        WHERE code IN ({placeholders})
    """, conn, params=codes)
    return dict(zip(df['code'], df['sector17_code'].fillna('10')))


def load_price_data(conn, codes, start_date, end_date):
    """Load price data for specific codes"""
    placeholders = ','.join(['?' for _ in codes])
    start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    df = pd.read_sql_query(f"""
        SELECT code, date, adjustment_close as close, adjustment_volume as volume
        FROM daily_quotes
        WHERE code IN ({placeholders})
          AND date >= ? AND date <= ?
          AND adjustment_close > 0
        ORDER BY code, date
    """, conn, params=(*codes, start_str, end_str))

    price_data = {}
    for code in codes:
        stock_df = df[df['code'] == code].copy()
        if len(stock_df) > 0:
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            stock_df.set_index('date', inplace=True)
            price_data[code] = stock_df
    return price_data


def load_financial_data(conn, codes, start_date, end_date):
    """Load financial data (PIT)"""
    placeholders = ','.join(['?' for _ in codes])
    start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    df = pd.read_sql_query(f"""
        SELECT code, disclosed_date, roe, bps, eps
        FROM financial_statements
        WHERE code IN ({placeholders})
          AND disclosed_date >= ? AND disclosed_date <= ?
          AND bps > 0
        ORDER BY code, disclosed_date
    """, conn, params=(*codes, start_str, end_str))

    fin_data = {}
    for code in codes:
        stock_df = df[df['code'] == code].copy()
        if len(stock_df) > 0:
            stock_df['disclosed_date'] = pd.to_datetime(stock_df['disclosed_date'])
            stock_df.set_index('disclosed_date', inplace=True)
            fin_data[code] = stock_df
    return fin_data


def calc_alpha(prices, volumes, fin_data, code, as_of):
    """Calculate multi-factor alpha"""
    if code not in prices:
        return None, None

    df = prices[code]
    ts = pd.Timestamp(as_of)
    avail = df[df.index <= ts]

    if len(avail) < 60:
        return None, None

    p = avail['close'].values
    v = avail['volume'].values
    close = p[-1]

    # ADT
    adt = np.mean(p[-20:] * v[-20:])

    # Momentum 12-1
    if len(p) >= 252:
        mom = (p[-22] / p[-252] - 1)
    elif len(p) >= 60:
        mom = (p[-1] / p[-60] - 1)
    else:
        mom = 0

    # Volatility
    rets = np.diff(p[-60:]) / p[-61:-1]
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 10 else 0.3

    # Reversal
    rev = -(p[-1] / p[-5] - 1) if len(p) >= 5 and p[-5] > 0 else 0

    # PBR (from financial data)
    pbr_score = 0
    if code in fin_data:
        fin_df = fin_data[code]
        fin_avail = fin_df[fin_df.index <= ts]
        if len(fin_avail) > 0:
            bps = fin_avail['bps'].iloc[-1]
            if bps > 0:
                pbr = close / bps
                pbr_score = -pbr  # Lower PBR = higher score

    # Composite
    alpha = mom * 0.35 + (-vol) * 0.20 + rev * 0.25 + pbr_score * 0.20

    return alpha, adt


def run_backtest():
    cfg = Config()

    logger.info("="*60)
    logger.info("ASSET SHIELD V5.0 - 18-YEAR BACKTEST")
    logger.info("="*60)

    conn = sqlite3.connect(DB_PATH)

    # Get liquid stocks
    logger.info("Getting liquid stocks...")
    codes = get_liquid_stocks(conn, 50)
    logger.info(f"Selected {len(codes)} stocks")

    # Load sector map
    sector_map = get_sector_map(conn, codes)

    # Load data
    logger.info("Loading price data...")
    prices = load_price_data(conn, codes, cfg.start_date, cfg.end_date)
    logger.info(f"Loaded {len(prices)} stocks with price data")

    logger.info("Loading financial data...")
    fin_data = load_financial_data(conn, codes, cfg.start_date, cfg.end_date)
    logger.info(f"Loaded {len(fin_data)} stocks with financial data")

    conn.close()

    # Get trading days
    all_dates = set()
    for df in prices.values():
        all_dates.update(df.index.date)
    trading_days = sorted([d for d in all_dates if cfg.start_date <= d <= cfg.end_date])
    logger.info(f"Trading days: {len(trading_days)}")

    # State
    equity = cfg.initial_capital
    cash = equity
    positions = {}  # code -> shares
    hwm = equity

    # Kill-switch
    ks_active = False
    ks_date = None
    ks_step = 0
    exposure_target = 0.90

    # Regime
    regime = "BULL"
    vol_hist = []

    # Tracking
    equity_curve = []
    last_rebal = None

    logger.info("Running backtest...")

    for i, dt in enumerate(trading_days):
        # Update positions
        pos_values = {}
        for code, shares in positions.items():
            if code in prices:
                df = prices[code]
                ts = pd.Timestamp(dt)
                if ts in df.index:
                    pos_values[code] = shares * df.loc[ts, 'close']
                elif len(df[df.index <= ts]) > 0:
                    pos_values[code] = shares * df[df.index <= ts].iloc[-1]['close']

        total_pos = sum(pos_values.values())
        equity = cash + total_pos

        # HWM and drawdown
        if equity > hwm:
            hwm = equity
        dd = (hwm - equity) / hwm

        # Regime (simple vol-based)
        vols = []
        for code in list(prices.keys())[:10]:
            df = prices[code]
            ts = pd.Timestamp(dt)
            avail = df[df.index <= ts]
            if len(avail) >= 21:
                ret = (avail['close'].iloc[-1] / avail['close'].iloc[-21] - 1)
                vols.append(abs(ret))

        if vols:
            avg_vol = np.mean(vols) * np.sqrt(252/20)
            vol_hist.append(avg_vol)
            if len(vol_hist) > 60:
                vol_hist = vol_hist[-60:]

            if avg_vol > cfg.vol_threshold_crisis:
                regime = "CRISIS"
            elif avg_vol > cfg.vol_threshold_bull:
                regime = "BEAR"
            else:
                regime = "BULL"

        # Kill-switch logic
        if not ks_active:
            if dd > cfg.max_drawdown or regime == "CRISIS":
                ks_active = True
                ks_date = dt
                ks_step = 0
                exposure_target = cfg.recovery_steps[0]
                logger.warning(f"{dt}: KILL SWITCH (DD={dd:.1%}, Regime={regime})")
        else:
            days_since = (dt - ks_date).days
            if regime != "CRISIS":
                target_step = min(days_since // cfg.recovery_days, len(cfg.recovery_steps) - 1)
                if target_step > ks_step:
                    ks_step = target_step
                    exposure_target = cfg.recovery_steps[ks_step]
                    if ks_step >= len(cfg.recovery_steps) - 1:
                        ks_active = False
                        hwm = equity
                        logger.info(f"{dt}: KILL SWITCH RESET")
            else:
                ks_step = 0
                exposure_target = cfg.recovery_steps[0]
                ks_date = dt

        # Determine max exposure
        if ks_active:
            max_exp = exposure_target
        elif regime == "CRISIS":
            max_exp = 0.0
        elif regime == "BEAR":
            max_exp = 0.60
        else:
            max_exp = 0.90

        # Rebalance
        should_rebal = False
        if ks_active and max_exp == 0:
            should_rebal = True
        elif last_rebal is None:
            should_rebal = True
        elif (dt - last_rebal).days >= cfg.rebalance_freq:
            should_rebal = True

        if should_rebal:
            # Calculate alphas
            alphas = []
            for code in prices.keys():
                alpha, adt = calc_alpha(prices, None, fin_data, code, dt)
                if alpha is not None and adt is not None and adt >= cfg.min_adt:
                    alphas.append((code, alpha, adt))

            alphas.sort(key=lambda x: x[1], reverse=True)

            # Select portfolio
            selected = []
            sector_wts = defaultdict(float)
            total_wt = 0
            base_wt = max_exp / min(len(alphas), cfg.max_positions) if alphas else 0

            for code, alpha, adt in alphas:
                if len(selected) >= cfg.max_positions or total_wt >= max_exp:
                    break

                sector = sector_map.get(code, '10')
                if sector_wts[sector] + base_wt > cfg.max_sector_weight:
                    continue

                max_adt_wt = (adt * 0.05) / equity if equity > 0 else 0
                wt = min(base_wt, cfg.max_single_weight, max_adt_wt)

                if wt < 0.01:
                    continue

                if total_wt + wt > max_exp:
                    wt = max_exp - total_wt

                selected.append((code, wt))
                sector_wts[sector] += wt
                total_wt += wt

            # Execute trades
            target_codes = set(c for c, _ in selected)
            current_codes = set(positions.keys())

            # Sell
            for code in current_codes - target_codes:
                if code in positions and positions[code] > 0:
                    if code in prices:
                        df = prices[code]
                        ts = pd.Timestamp(dt)
                        avail = df[df.index <= ts]
                        if len(avail) > 0:
                            price = avail.iloc[-1]['close']
                            cash += positions[code] * price
                            del positions[code]

            # Buy
            for code, target_wt in selected:
                target_val = equity * target_wt
                if code in prices:
                    df = prices[code]
                    ts = pd.Timestamp(dt)
                    avail = df[df.index <= ts]
                    if len(avail) > 0:
                        price = avail.iloc[-1]['close']
                        cur_shares = positions.get(code, 0)
                        cur_val = cur_shares * price
                        delta = target_val - cur_val

                        if delta > cash:
                            delta = cash

                        if delta > equity * 0.005:
                            shares_buy = int(delta / price)
                            if shares_buy > 0:
                                cash -= shares_buy * price
                                positions[code] = cur_shares + shares_buy
                        elif delta < -equity * 0.005 and cur_shares > 0:
                            shares_sell = min(cur_shares, int(abs(delta) / price))
                            if shares_sell > 0:
                                cash += shares_sell * price
                                positions[code] = cur_shares - shares_sell
                                if positions[code] == 0:
                                    del positions[code]

            last_rebal = dt

        equity_curve.append((dt, equity))

        if i % 500 == 0:
            logger.info(f"{dt}: Equity=¥{equity:,.0f} Regime={regime} DD={dd:.1%}")

    # Calculate metrics
    initial = cfg.initial_capital
    final = equity_curve[-1][1]
    total_ret = (final / initial) - 1
    years = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr = (final / initial) ** (1/years) - 1 if years > 0 else 0

    eqs = [e for _, e in equity_curve]
    daily_rets = pd.Series(eqs).pct_change().dropna()
    sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

    peak = pd.Series(eqs).expanding().max()
    drawdowns = (peak - pd.Series(eqs)) / peak
    max_dd = drawdowns.max()

    calmar = cagr / max_dd if max_dd > 0 else 0

    # Report
    print("\n" + "="*60)
    print("ASSET SHIELD V5.0 - BACKTEST RESULTS")
    print("="*60)
    print(f"Period:         {cfg.start_date} to {cfg.end_date}")
    print(f"Initial:        ¥{initial:,.0f}")
    print(f"Final:          ¥{final:,.0f}")
    print("-"*60)
    print(f"Total Return:   {total_ret:>12.2%}")
    print(f"CAGR:           {cagr:>12.2%}")
    print(f"Sharpe Ratio:   {sharpe:>12.2f}")
    print(f"Max Drawdown:   {max_dd:>12.2%}")
    print(f"Calmar Ratio:   {calmar:>12.2f}")
    print("="*60)

    # Save
    output_dir = Path("/Users/MBP/Desktop/Project_Asset_Shield/output/v5")
    output_dir.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq_df.to_csv(output_dir / "v5_equity_curve.csv", index=False)

    with open(output_dir / "V5_REPORT.txt", 'w') as f:
        f.write("ASSET SHIELD V5.0 - BACKTEST RESULTS\n")
        f.write(f"Period: {cfg.start_date} to {cfg.end_date}\n")
        f.write(f"Total Return: {total_ret:.2%}\n")
        f.write(f"CAGR: {cagr:.2%}\n")
        f.write(f"Sharpe: {sharpe:.2f}\n")
        f.write(f"Max DD: {max_dd:.2%}\n")
        f.write(f"Calmar: {calmar:.2f}\n")

    return total_ret, cagr, sharpe, max_dd


if __name__ == "__main__":
    run_backtest()
