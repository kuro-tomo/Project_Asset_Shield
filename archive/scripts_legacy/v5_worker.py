#!/usr/bin/env python3
"""V5.0 Parallel Worker - Single Segment"""
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

def get_liquid_stocks(conn, n=100):
    df = pd.read_sql_query("""
        SELECT code, AVG(adjustment_close * volume) as avg_turnover
        FROM daily_quotes WHERE date >= '2015-01-01'
        GROUP BY code HAVING avg_turnover > 50000000
        ORDER BY avg_turnover DESC LIMIT ?
    """, conn, params=(n,))
    return df['code'].tolist()

def get_sector_map(conn, codes):
    placeholders = ','.join(['?' for _ in codes])
    df = pd.read_sql_query(f"SELECT code, sector17_code FROM listed_info WHERE code IN ({placeholders})", conn, params=codes)
    return dict(zip(df['code'], df['sector17_code'].fillna('10')))

def load_price_data(conn, codes, start_date, end_date):
    placeholders = ','.join(['?' for _ in codes])
    start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    df = pd.read_sql_query(f"""
        SELECT code, date, adjustment_close as close, volume
        FROM daily_quotes WHERE code IN ({placeholders}) AND date >= ? AND date <= ? AND adjustment_close > 0
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
    placeholders = ','.join(['?' for _ in codes])
    start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    df = pd.read_sql_query(f"""
        SELECT code, disclosed_date, roe, bps FROM financial_statements
        WHERE code IN ({placeholders}) AND disclosed_date >= ? AND disclosed_date <= ? AND bps > 0
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

def calc_alpha(prices, fin_data, code, as_of):
    if code not in prices:
        return None, None
    df = prices[code]
    ts = pd.Timestamp(as_of)
    avail = df[df.index <= ts]
    if len(avail) < 60:
        return None, None
    p = avail['close'].values
    v = avail['volume'].values
    # Handle None values
    p = np.array([x if x is not None else 0 for x in p])
    v = np.array([x if x is not None else 0 for x in v])
    if p[-1] <= 0:
        return None, None
    close = p[-1]
    adt = np.mean(p[-20:] * v[-20:]) if len(p) >= 20 else 0
    mom = 0
    if len(p) >= 252 and p[-252] > 0:
        mom = (p[-22] / p[-252] - 1)
    elif len(p) >= 60 and p[-60] > 0:
        mom = (p[-1] / p[-60] - 1)
    rets = []
    if len(p) >= 60:
        prices_60 = p[-60:]
        denom = prices_60[:-1]
        denom = np.where(denom == 0, 1, denom)
        rets = np.diff(prices_60) / denom
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 10 else 0.3
    rev = -(p[-1] / p[-5] - 1) if len(p) >= 5 and p[-5] > 0 else 0
    pbr_score = 0
    if code in fin_data:
        fin_df = fin_data[code]
        fin_avail = fin_df[fin_df.index <= ts]
        if len(fin_avail) > 0:
            bps = fin_avail['bps'].iloc[-1]
            if bps > 0:
                pbr_score = -(close / bps)
    # V5.2: Momentum-focused alpha (best performer)
    alpha = mom * 0.45 + (-vol) * 0.15 + rev * 0.25 + pbr_score * 0.15
    return alpha, adt

def run_segment(segment_id, start_date, end_date, output_path):
    logger = logging.LoggerAdapter(logging.getLogger(), {'segment': segment_id})
    logger.info(f"Starting: {start_date} to {end_date}")

    conn = sqlite3.connect(DB_PATH)
    codes = get_liquid_stocks(conn, 100)
    sector_map = get_sector_map(conn, codes)
    prices = load_price_data(conn, codes, start_date, end_date)
    fin_data = load_financial_data(conn, codes, start_date, end_date)
    conn.close()

    logger.info(f"Loaded {len(prices)} stocks")

    all_dates = set()
    for df in prices.values():
        all_dates.update(df.index.date)
    trading_days = sorted([d for d in all_dates if start_date <= d <= end_date])

    # Config
    initial = 100_000_000
    equity = initial
    cash = equity
    positions = {}
    hwm = equity
    ks_active = False
    ks_date = None
    ks_step = 0
    exposure_target = 0.90
    recovery_steps = [0.0, 0.30, 0.60, 0.90]
    regime = "BULL"
    last_rebal = None
    equity_curve = []

    for i, dt in enumerate(trading_days):
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

        # Regime (only after warmup) - Volatility + Trend based
        if i >= 60:  # Warmup period
            vols = []
            trends = []
            for code in list(prices.keys())[:10]:
                df = prices[code]
                ts = pd.Timestamp(dt)
                avail = df[df.index <= ts]
                if len(avail) >= 60:
                    rets_20d = avail['close'].pct_change().tail(20).dropna()
                    if len(rets_20d) > 0:
                        vols.append(rets_20d.std() * np.sqrt(252))
                    # Add trend calculation (60-day price change)
                    prices_arr = avail['close'].values
                    if len(prices_arr) >= 60 and prices_arr[-60] > 0:
                        trend = (prices_arr[-1] / prices_arr[-60]) - 1
                        trends.append(trend)
            if len(vols) >= 5:
                avg_vol = np.mean(vols)
                avg_trend = np.mean(trends) if trends else 0
                if avg_vol > 0.45:
                    regime = "CRISIS"
                elif avg_vol > 0.25 and avg_trend < -0.03:
                    regime = "BEAR"  # High vol + clear downtrend only
                elif avg_trend > 0.08:
                    regime = "SUPER_BULL"  # Strong uptrend - use leverage
                elif avg_trend > 0.03:
                    regime = "BULL"  # Uptrend
                else:
                    regime = "NEUTRAL"  # Sideways

        # Kill-switch (only after warmup and only on drawdown)
        if i >= 60 and not ks_active and dd > 0.15:
            ks_active = True
            ks_date = dt
            ks_step = 0
            exposure_target = recovery_steps[0]
            logger.warning(f"{dt}: KILL SWITCH (DD={dd:.1%})")

        # Recovery logic (only if kill switch is active)
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
                        logger.info(f"{dt}: KILL SWITCH RESET")
            else:
                ks_step = 0
                exposure_target = recovery_steps[0]
                ks_date = dt

        # Determine max exposure (V5.2 - Best performer)
        if ks_active:
            max_exp = exposure_target
        elif i < 60:
            max_exp = 1.00
        elif regime == "CRISIS":
            max_exp = 0.50
        elif regime == "BEAR":
            max_exp = 0.70
        elif regime == "NEUTRAL":
            max_exp = 0.85
        elif regime == "SUPER_BULL":
            max_exp = 1.10
        else:  # BULL
            max_exp = 1.00

        # Rebalance
        should_rebal = last_rebal is None or (dt - last_rebal).days >= 20
        if should_rebal and max_exp > 0:
            alphas = []
            debug_count = 0
            for code in prices.keys():
                alpha, adt = calc_alpha(prices, fin_data, code, dt)
                if alpha is not None and adt is not None:
                    if adt >= 10_000_000:
                        alphas.append((code, alpha, adt))
                    elif i < 5 and debug_count < 3:
                        logger.info(f"  {code}: alpha={alpha:.3f} adt={adt:,.0f} (too low)")
                        debug_count += 1
                elif i < 5 and debug_count < 3:
                    df = prices.get(code)
                    avail_len = len(df[df.index <= pd.Timestamp(dt)]) if df is not None else 0
                    logger.info(f"  {code}: alpha=None avail_days={avail_len}")
                    debug_count += 1
            alphas.sort(key=lambda x: x[1], reverse=True)

            if i < 100:
                if len(alphas) > 0:
                    logger.info(f"{dt}: {len(alphas)} candidates, max_exp={max_exp:.0%}, top alpha={alphas[0][1]:.3f}")
                else:
                    logger.info(f"{dt}: NO candidates found")

            selected = []
            sector_wts = defaultdict(float)
            total_wt = 0
            base_wt = max_exp / min(len(alphas), 15) if alphas else 0  # Larger base weight

            for code, alpha, adt in alphas:
                if len(selected) >= 15 or total_wt >= max_exp:  # Fewer, larger positions
                    break
                sector = sector_map.get(code, '10')
                if sector_wts[sector] + base_wt > 0.35:  # Allow more sector concentration
                    continue
                wt = min(base_wt, 0.12, (adt * 0.08) / equity if equity > 0 else 0)  # Larger positions
                if wt < 0.01:
                    continue
                if total_wt + wt > max_exp:
                    wt = max_exp - total_wt
                selected.append((code, wt))
                sector_wts[sector] += wt
                total_wt += wt

            if i < 100:
                logger.info(f"{dt}: Selected {len(selected)} positions, total_wt={total_wt:.1%}")

            target_codes = set(c for c, _ in selected)
            for code in set(positions.keys()) - target_codes:
                if code in positions and code in prices:
                    df = prices[code]
                    ts = pd.Timestamp(dt)
                    avail = df[df.index <= ts]
                    if len(avail) > 0:
                        cash += positions[code] * avail.iloc[-1]['close']
                        del positions[code]

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
        if i % 200 == 0:
            logger.info(f"{dt}: ¥{equity:,.0f} {regime}")

    # Metrics
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
        "metrics": {
            "total_return": float(total_ret),
            "cagr": float(cagr),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd)
        },
        "equity_curve": equity_curve[-10:]  # Last 10 only for space
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"DONE: Return={total_ret:.2%}, Sharpe={sharpe:.2f}")
    print(f"\nSEGMENT {segment_id} COMPLETE")
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
