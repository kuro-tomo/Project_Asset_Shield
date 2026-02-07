#!/usr/bin/env python3
"""V5.3 Full Universe Worker - Total War Edition
Scans ALL J-Quants stocks with ROE/PBR filters, selects Top 20 by Momentum
"""
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import json
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | W%(segment)s | %(message)s')

DB_PATH = "/Users/MBP/Desktop/Project_Asset_Shield/data/jquants_cache.db"

def get_all_stocks(conn):
    """Get ALL stocks with sufficient data"""
    df = pd.read_sql_query("""
        SELECT DISTINCT code FROM daily_quotes
        WHERE date >= '2008-01-01' AND adjustment_close > 0
    """, conn)
    return df['code'].tolist()

def get_sector_map(conn, codes):
    """Get sector mapping for all codes"""
    placeholders = ','.join(['?' for _ in codes])
    df = pd.read_sql_query(f"""
        SELECT code, sector17_code FROM listed_info
        WHERE code IN ({placeholders})
    """, conn, params=codes)
    return dict(zip(df['code'], df['sector17_code'].fillna('10')))

def load_price_data(conn, codes, start_date, end_date):
    """Load price data for all codes"""
    start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Load in batches to avoid memory issues
    price_data = {}
    batch_size = 500

    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        placeholders = ','.join(['?' for _ in batch])
        df = pd.read_sql_query(f"""
            SELECT code, date, adjustment_close as close, volume
            FROM daily_quotes
            WHERE code IN ({placeholders}) AND date >= ? AND date <= ? AND adjustment_close > 0
            ORDER BY code, date
        """, conn, params=(*batch, start_str, end_str))

        for code in batch:
            stock_df = df[df['code'] == code].copy()
            if len(stock_df) > 60:  # Need at least 60 days
                stock_df['date'] = pd.to_datetime(stock_df['date'])
                stock_df.set_index('date', inplace=True)
                price_data[code] = stock_df

    return price_data

def load_financial_data(conn, codes, start_date, end_date):
    """Load ROE and BPS data for all codes"""
    start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    fin_data = {}
    batch_size = 500

    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        placeholders = ','.join(['?' for _ in batch])
        df = pd.read_sql_query(f"""
            SELECT code, disclosed_date, roe, bps FROM financial_statements
            WHERE code IN ({placeholders}) AND disclosed_date >= ? AND disclosed_date <= ?
            ORDER BY code, disclosed_date
        """, conn, params=(*batch, start_str, end_str))

        for code in batch:
            stock_df = df[df['code'] == code].copy()
            if len(stock_df) > 0:
                stock_df['disclosed_date'] = pd.to_datetime(stock_df['disclosed_date'])
                stock_df.set_index('disclosed_date', inplace=True)
                fin_data[code] = stock_df

    return fin_data

def get_fundamentals(fin_data, code, as_of):
    """Get ROE and PBR as of date (Point-in-Time)"""
    if code not in fin_data:
        return None, None

    df = fin_data[code]
    ts = pd.Timestamp(as_of)
    avail = df[df.index <= ts]

    if len(avail) == 0:
        return None, None

    latest = avail.iloc[-1]
    roe = latest.get('roe')
    bps = latest.get('bps')

    return roe, bps

def calc_momentum(prices, code, as_of):
    """Calculate 12-1 momentum and ADT"""
    if code not in prices:
        return None, None

    df = prices[code]
    ts = pd.Timestamp(as_of)
    avail = df[df.index <= ts]

    if len(avail) < 252:
        return None, None

    p = avail['close'].values
    v = avail['volume'].values

    # Handle None values
    p = np.array([x if x is not None and x > 0 else 0 for x in p])
    v = np.array([x if x is not None else 0 for x in v])

    if p[-1] <= 0 or p[-252] <= 0:
        return None, None

    # 12-1 momentum (skip last month)
    mom = (p[-22] / p[-252]) - 1

    # ADT
    adt = np.mean(p[-20:] * v[-20:])

    return mom, adt

def run_segment(segment_id, start_date, end_date, output_path):
    logger = logging.LoggerAdapter(logging.getLogger(), {'segment': segment_id})
    logger.info(f"Starting FULL UNIVERSE: {start_date} to {end_date}")

    conn = sqlite3.connect(DB_PATH)

    # Get ALL stocks
    all_codes = get_all_stocks(conn)
    logger.info(f"Total universe: {len(all_codes)} stocks")

    sector_map = get_sector_map(conn, all_codes)
    logger.info("Sector map loaded")

    prices = load_price_data(conn, all_codes, start_date, end_date)
    logger.info(f"Price data loaded: {len(prices)} stocks with sufficient history")

    fin_data = load_financial_data(conn, all_codes, start_date, end_date)
    logger.info(f"Financial data loaded: {len(fin_data)} stocks")

    conn.close()

    # Get trading days
    all_dates = set()
    for df in prices.values():
        all_dates.update(df.index.date)
    trading_days = sorted([d for d in all_dates if start_date <= d <= end_date])
    logger.info(f"Trading days: {len(trading_days)}")

    # Config
    initial = 100_000_000
    equity = initial
    cash = equity
    positions = {}
    hwm = equity
    ks_active = False
    ks_date = None
    ks_step = 0
    exposure_target = 1.00
    recovery_steps = [0.0, 0.30, 0.60, 1.00]
    regime = "BULL"
    last_rebal = None
    equity_curve = []

    for i, dt in enumerate(trading_days):
        # Update positions value
        pos_values = {}
        for code, shares in positions.items():
            if code in prices:
                df = prices[code]
                ts = pd.Timestamp(dt)
                avail = df[df.index <= ts]
                if len(avail) > 0:
                    pos_values[code] = shares * avail.iloc[-1]['close']

        equity = cash + sum(pos_values.values())
        if equity > hwm:
            hwm = equity
        dd = (hwm - equity) / hwm

        # Regime detection (V5.1 trend-based) - after warmup
        if i >= 60:
            vols = []
            trends = []
            sample_codes = list(prices.keys())[:50]  # Sample 50 stocks
            for code in sample_codes:
                df = prices[code]
                ts = pd.Timestamp(dt)
                avail = df[df.index <= ts]
                if len(avail) >= 60:
                    rets_20d = avail['close'].pct_change().tail(20).dropna()
                    if len(rets_20d) > 0:
                        vols.append(rets_20d.std() * np.sqrt(252))
                    p_arr = avail['close'].values
                    if len(p_arr) >= 60 and p_arr[-60] > 0:
                        trends.append(p_arr[-1] / p_arr[-60] - 1)

            if len(vols) >= 5:
                avg_vol = np.mean(vols)
                avg_trend = np.mean(trends) if trends else 0

                if avg_vol > 0.40:
                    regime = "CRISIS"
                elif avg_vol > 0.22 and avg_trend < 0:
                    regime = "BEAR"
                elif avg_trend > 0.05:
                    regime = "BULL"
                else:
                    regime = "NEUTRAL"

        # Kill-switch (15% DD)
        if i >= 60 and not ks_active and dd > 0.15:
            ks_active = True
            ks_date = dt
            ks_step = 0
            exposure_target = recovery_steps[0]
            logger.warning(f"{dt}: KILL SWITCH (DD={dd:.1%})")

        # Recovery logic
        if ks_active and ks_date is not None:
            days_since = (dt - ks_date).days
            if regime != "CRISIS":
                target_step = min(days_since // 5, len(recovery_steps) - 1)
                if target_step > ks_step:
                    ks_step = target_step
                    exposure_target = recovery_steps[ks_step]
                    if ks_step >= len(recovery_steps) - 1:
                        ks_active = False
                        ks_date = None
                        hwm = equity
            else:
                ks_step = 0
                exposure_target = recovery_steps[0]
                ks_date = dt

        # Determine max exposure (V5.3: Max 100%, no leverage)
        if ks_active:
            max_exp = exposure_target
        elif i < 60:
            max_exp = 1.00
        elif regime == "CRISIS":
            max_exp = 0.30
        elif regime == "BEAR":
            max_exp = 0.60
        elif regime == "NEUTRAL":
            max_exp = 0.70
        else:  # BULL
            max_exp = 1.00

        # Rebalance every 20 days
        should_rebal = last_rebal is None or (dt - last_rebal).days >= 20
        if should_rebal and max_exp > 0:
            candidates = []

            # STEP 1: The Great Filter - scan ALL stocks
            for code in prices.keys():
                # Get fundamentals (ROE, BPS for PBR)
                roe, bps = get_fundamentals(fin_data, code, dt)

                # ROE filter: >= 5% (relaxed) or missing is OK
                if roe is not None and roe < 5:
                    continue

                # Get current price for PBR
                df = prices[code]
                ts = pd.Timestamp(dt)
                avail = df[df.index <= ts]
                if len(avail) < 252:
                    continue

                close = avail.iloc[-1]['close']
                if close <= 0 or bps is None or bps <= 0:
                    continue

                pbr = close / bps

                # PBR filter: <= 3.0 (relaxed to include quality growth)
                if pbr > 3.0:
                    continue

                # STEP 2: Calculate momentum
                mom, adt = calc_momentum(prices, code, dt)
                if mom is None or adt is None:
                    continue

                # ADT filter: at least 10M JPY
                if adt < 10_000_000:
                    continue

                # Composite score: Momentum (70%) + Quality (30%)
                roe_score = min(roe / 20, 1.0) if roe else 0.5  # Normalize ROE
                pbr_score = max(0, 1 - pbr / 5)  # Lower PBR = higher score
                quality = roe_score * 0.6 + pbr_score * 0.4
                composite = mom * 0.70 + quality * 0.30
                candidates.append((code, composite, adt, roe, pbr, mom))

            # Sort by composite score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)

            if i % 200 == 0:
                logger.info(f"{dt}: {len(candidates)} passed filters, regime={regime}")

            # STEP 3: Select Top 20 with sector constraints
            selected = []
            sector_wts = defaultdict(float)
            total_wt = 0
            base_wt = max_exp / 20  # Equal weight for 20 positions

            for code, composite, adt, roe, pbr, mom in candidates:
                if len(selected) >= 20 or total_wt >= max_exp:
                    break

                sector = sector_map.get(code, '10')

                # Sector cap: 25%
                if sector_wts[sector] + base_wt > 0.25:
                    continue

                # ADT constraint
                wt = min(base_wt, 0.08, (adt * 0.05) / equity if equity > 0 else 0)
                if wt < 0.01:
                    continue

                if total_wt + wt > max_exp:
                    wt = max_exp - total_wt

                selected.append((code, wt))
                sector_wts[sector] += wt
                total_wt += wt

            # Execute trades
            target_codes = set(c for c, _ in selected)

            # Sell positions not in target
            for code in list(positions.keys()):
                if code not in target_codes and code in prices:
                    df = prices[code]
                    ts = pd.Timestamp(dt)
                    avail = df[df.index <= ts]
                    if len(avail) > 0:
                        cash += positions[code] * avail.iloc[-1]['close']
                        del positions[code]

            # Buy/adjust positions
            for code, target_wt in selected:
                target_val = equity * target_wt
                if code in prices:
                    df = prices[code]
                    ts = pd.Timestamp(dt)
                    avail = df[df.index <= ts]
                    if len(avail) > 0:
                        price = avail.iloc[-1]['close']
                        cur_shares = positions.get(code, 0)
                        delta = target_val - cur_shares * price
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

        equity_curve.append((str(dt), equity))

        if i % 250 == 0:
            logger.info(f"{dt}: ¥{equity:,.0f} {regime} (DD={dd:.1%})")

    # Final metrics
    final = equity
    total_ret = (final / initial) - 1
    years = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr = (final / initial) ** (1/years) - 1 if years > 0 else 0

    eqs = [e for _, e in equity_curve]
    daily_rets = pd.Series(eqs).pct_change().dropna()
    sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0

    peak = pd.Series(eqs).expanding().max()
    max_dd = ((peak - pd.Series(eqs)) / peak).max()

    result = {
        "segment_id": segment_id,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "universe_size": len(prices),
        "metrics": {
            "total_return": float(total_ret),
            "cagr": float(cagr),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd)
        },
        "equity_curve": equity_curve[-10:]
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"DONE: Return={total_ret:.2%}, CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")
    print(f"\nSEGMENT {segment_id} COMPLETE")
    print(f"Universe: {len(prices)} stocks")
    print(f"Return: {total_ret:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max DD: {max_dd:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    run_segment(args.segment, date.fromisoformat(args.start), date.fromisoformat(args.end), args.output)
