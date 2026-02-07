#!/usr/bin/env python3
"""V5.6 Concentrated - Top 10 High Conviction Positions"""
import sys, argparse, sqlite3, numpy as np, pandas as pd, json
from datetime import date, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | W%(segment)s | %(message)s')
DB_PATH = "/Users/MBP/Desktop/Project_Asset_Shield/data/jquants_cache.db"

def run_segment(segment_id, start_date, end_date, output_path):
    logger = logging.LoggerAdapter(logging.getLogger(), {'segment': segment_id})
    logger.info(f"V5.6 CONCENTRATED TOP10: {start_date} to {end_date}")

    conn = sqlite3.connect(DB_PATH)

    # Top 100 liquid stocks
    codes_df = pd.read_sql_query("""
        SELECT code, AVG(adjustment_close * volume) as avg_turnover
        FROM daily_quotes WHERE date >= '2015-01-01'
        GROUP BY code HAVING avg_turnover > 50000000
        ORDER BY avg_turnover DESC LIMIT 100
    """, conn)
    codes = codes_df['code'].tolist()

    sector_df = pd.read_sql_query("SELECT code, sector17_code FROM listed_info", conn)
    sector_map = dict(zip(sector_df['code'], sector_df['sector17_code'].fillna('10')))

    start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    ph = ','.join(['?' for _ in codes])
    price_df = pd.read_sql_query(f"""
        SELECT code, date, adjustment_close as close, volume
        FROM daily_quotes WHERE code IN ({ph}) AND date >= ? AND date <= ? AND adjustment_close > 0
    """, conn, params=(*codes, start_str, end_str))

    fin_df = pd.read_sql_query(f"""
        SELECT code, disclosed_date, roe, bps FROM financial_statements
        WHERE code IN ({ph}) AND disclosed_date >= ? AND disclosed_date <= ?
    """, conn, params=(*codes, start_str, end_str))
    conn.close()

    prices, fin_data = {}, {}
    for code in codes:
        sdf = price_df[price_df['code'] == code].copy()
        if len(sdf) > 60:
            sdf['date'] = pd.to_datetime(sdf['date'])
            sdf.set_index('date', inplace=True)
            prices[code] = sdf
        fdf = fin_df[fin_df['code'] == code].copy()
        if len(fdf) > 0:
            fdf['disclosed_date'] = pd.to_datetime(fdf['disclosed_date'])
            fdf.set_index('disclosed_date', inplace=True)
            fin_data[code] = fdf

    logger.info(f"Loaded {len(prices)} stocks")

    all_dates = set()
    for df in prices.values():
        all_dates.update(df.index.date)
    trading_days = sorted([d for d in all_dates if start_date <= d <= end_date])

    initial = 100_000_000
    equity, cash = initial, initial
    positions = {}
    hwm = equity
    ks_active, ks_date, ks_step = False, None, 0
    recovery_steps = [0.0, 0.30, 0.60, 1.00]
    regime = "BULL"
    last_rebal = None
    equity_curve = []

    for i, dt in enumerate(trading_days):
        ts = pd.Timestamp(dt)
        pos_val = 0
        for c, shares in positions.items():
            if c in prices:
                avail = prices[c][prices[c].index <= ts]
                if len(avail) > 0:
                    pos_val += shares * avail.iloc[-1]['close']
        equity = cash + pos_val
        if equity > hwm:
            hwm = equity
        dd = (hwm - equity) / hwm

        # Regime (V5.2 trend-based)
        if i >= 60:
            vols, trends = [], []
            for code in list(prices.keys())[:10]:
                avail = prices[code][prices[code].index <= ts]
                if len(avail) >= 60:
                    rets = avail['close'].pct_change().tail(20).dropna()
                    if len(rets) > 0:
                        vols.append(rets.std() * np.sqrt(252))
                    p = avail['close'].values
                    if p[-60] > 0:
                        trends.append(p[-1] / p[-60] - 1)
            if len(vols) >= 5:
                avg_vol, avg_trend = np.mean(vols), np.mean(trends) if trends else 0
                if avg_vol > 0.45:
                    regime = "CRISIS"
                elif avg_vol > 0.25 and avg_trend < -0.03:
                    regime = "BEAR"
                elif avg_trend > 0.08:
                    regime = "SUPER_BULL"
                elif avg_trend > 0.03:
                    regime = "BULL"
                else:
                    regime = "NEUTRAL"

        # Kill-switch
        if i >= 60 and not ks_active and dd > 0.15:
            ks_active, ks_date, ks_step = True, dt, 0
            logger.warning(f"{dt}: KILL SWITCH DD={dd:.1%}")

        if ks_active and ks_date:
            days = (dt - ks_date).days
            if regime != "CRISIS":
                step = min(days // 5, 3)
                if step > ks_step:
                    ks_step = step
                    if ks_step >= 3:
                        ks_active, ks_date, hwm = False, None, equity
            else:
                ks_step, ks_date = 0, dt

        # Max exposure (V5.2 aggressive + leverage)
        if ks_active:
            max_exp = recovery_steps[ks_step]
        elif i < 60:
            max_exp = 1.00
        elif regime == "CRISIS":
            max_exp = 0.50
        elif regime == "BEAR":
            max_exp = 0.70
        elif regime == "NEUTRAL":
            max_exp = 0.85
        elif regime == "SUPER_BULL":
            max_exp = 1.20  # Higher leverage for concentrated
        else:
            max_exp = 1.00

        # Rebalance every 20 days
        if (last_rebal is None or (dt - last_rebal).days >= 20) and max_exp > 0:
            candidates = []
            for code in prices:
                avail = prices[code][prices[code].index <= ts]
                if len(avail) < 60:
                    continue
                p = avail['close'].values
                v = avail['volume'].values
                p = np.array([x if x is not None else 0 for x in p])
                v = np.array([x if x is not None else 0 for x in v])
                if p[-1] <= 0:
                    continue

                # Momentum (12-1)
                mom = 0
                if len(p) >= 252 and p[-252] > 0:
                    mom = (p[-22] / p[-252] - 1)
                elif len(p) >= 60 and p[-60] > 0:
                    mom = (p[-1] / p[-60] - 1)

                # Volatility
                if len(p) >= 60:
                    rets = np.diff(p[-60:]) / np.where(p[-60:-1] == 0, 1, p[-60:-1])
                    vol = np.std(rets) * np.sqrt(252)
                else:
                    vol = 0.3

                # Reversal
                rev = -(p[-1] / p[-5] - 1) if len(p) >= 5 and p[-5] > 0 else 0

                # PBR score
                pbr_score = 0
                if code in fin_data:
                    fin_avail = fin_data[code][fin_data[code].index <= ts]
                    if len(fin_avail) > 0:
                        bps = fin_avail.iloc[-1].get('bps', 0)
                        if bps and bps > 0:
                            pbr_score = -(p[-1] / bps)

                # ADT
                adt = np.mean(p[-20:] * v[-20:]) if len(p) >= 20 else 0
                if adt < 10_000_000:
                    continue

                # V5.6: More momentum focused for concentration
                alpha = mom * 0.50 + (-vol) * 0.15 + rev * 0.20 + pbr_score * 0.15
                candidates.append((code, alpha, adt))

            candidates.sort(key=lambda x: x[1], reverse=True)

            # TOP 10 ONLY - High conviction
            selected = []
            sector_wts = defaultdict(float)
            total_wt = 0
            base_wt = max_exp / 10  # 10% per position at full exposure

            for code, alpha, adt in candidates:
                if len(selected) >= 10 or total_wt >= max_exp:
                    break
                sector = sector_map.get(code, '10')
                if sector_wts[sector] + base_wt > 0.35:  # Allow more sector concentration
                    continue
                wt = min(base_wt, 0.15, (adt * 0.08) / equity if equity > 0 else 0)
                if wt < 0.02:
                    continue
                if total_wt + wt > max_exp:
                    wt = max_exp - total_wt
                selected.append((code, wt))
                sector_wts[sector] += wt
                total_wt += wt

            if i < 50:
                logger.info(f"{dt}: Top10 selected, total_wt={total_wt:.1%}, {regime}")

            # Execute
            target = set(c for c, _ in selected)
            for code in list(positions.keys()):
                if code not in target and code in prices:
                    avail = prices[code][prices[code].index <= ts]
                    if len(avail) > 0:
                        cash += positions[code] * avail.iloc[-1]['close']
                        del positions[code]

            for code, twt in selected:
                tval = equity * twt
                avail = prices[code][prices[code].index <= ts]
                if len(avail) > 0:
                    price = avail.iloc[-1]['close']
                    cur = positions.get(code, 0)
                    delta = tval - cur * price
                    if delta > cash:
                        delta = cash
                    if delta > equity * 0.005:
                        buy = int(delta / price)
                        if buy > 0:
                            cash -= buy * price
                            positions[code] = cur + buy
                    elif delta < -equity * 0.005 and cur > 0:
                        sell = min(cur, int(abs(delta) / price))
                        if sell > 0:
                            cash += sell * price
                            positions[code] = cur - sell
                            if positions[code] == 0:
                                del positions[code]

            last_rebal = dt

        equity_curve.append((str(dt), equity))
        if i % 200 == 0:
            logger.info(f"{dt}: ¥{equity:,.0f} {regime} DD={dd:.1%}")

    final = equity
    ret = (final / initial) - 1
    years = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr = (final / initial) ** (1/years) - 1 if years > 0 else 0
    eqs = [e for _, e in equity_curve]
    rets = pd.Series(eqs).pct_change().dropna()
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    peak = pd.Series(eqs).expanding().max()
    max_dd = ((peak - pd.Series(eqs)) / peak).max()

    with open(output_path, 'w') as f:
        json.dump({"segment": segment_id, "return": ret, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}, f)

    logger.info(f"DONE: Return={ret:.2%}, CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")
    print(f"\n=== SEGMENT {segment_id} ===")
    print(f"Return: {ret:.2%}")
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
