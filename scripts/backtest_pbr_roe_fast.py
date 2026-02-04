#!/usr/bin/env python3
"""
Asset Shield - PBR/ROE Value Strategy Backtest (Optimized v3)
==============================================================
Fast version with robust NaN handling
"""

import os
import sys
import sqlite3
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    """Safely convert value to float"""
    try:
        result = float(val)
        if pd.isna(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def run_backtest(cache_path: str, start_date: str = "2008-01-01", end_date: str = "2025-12-31"):
    """Execute PBR/ROE strategy backtest (robust version)"""

    logger.info("=" * 70)
    logger.info("ASSET SHIELD - PBR/ROE VALUE STRATEGY BACKTEST")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("Criteria: PBR ≤ 1.0 AND ROE ≥ 10%")
    logger.info("=" * 70)

    conn = sqlite3.connect(cache_path)

    # Load financial data
    logger.info("Loading financial data...")
    fins_df = pd.read_sql_query("""
        SELECT code, disclosed_date, bps, roe
        FROM financial_statements
        WHERE bps IS NOT NULL AND bps > 0 AND roe IS NOT NULL
        ORDER BY code, disclosed_date
    """, conn)
    logger.info(f"Financial data: {len(fins_df):,} records, {fins_df['code'].nunique():,} stocks")

    # Load price data (valid prices only)
    logger.info("Loading price data...")
    price_df = pd.read_sql_query("""
        SELECT code, date,
               COALESCE(adjustment_close, close) as close
        FROM daily_quotes
        WHERE date BETWEEN ? AND ?
          AND COALESCE(adjustment_close, close) > 0
        ORDER BY date, code
    """, conn, params=[start_date, end_date])
    logger.info(f"Price data: {len(price_df):,} records, {price_df['code'].nunique():,} stocks")

    conn.close()

    # Date conversion
    price_df['date'] = pd.to_datetime(price_df['date'])
    fins_df['disclosed_date'] = pd.to_datetime(fins_df['disclosed_date'])

    # Pre-build price dictionary (optimization)
    logger.info("Building price lookup tables...")
    price_df['close'] = price_df['close'].astype(float)
    price_by_date = {date: group.set_index('code')['close'].to_dict()
                     for date, group in price_df.groupby('date')}

    # Rebalance dates (quarterly)
    trading_dates = sorted(price_df['date'].unique())
    rebalance_dates = [d for i, d in enumerate(trading_dates) if i % 63 == 0]
    logger.info(f"Rebalance dates: {len(rebalance_dates)}")

    # Index financial data by code and date
    fins_df = fins_df.sort_values(['code', 'disclosed_date'])
    latest_fins_by_code = {}

    # Backtest configuration
    initial_capital = 10_000_000.0
    max_positions = 20

    # Results tracking
    equity_history = []
    all_trades = []

    cash = initial_capital
    positions = {}

    for i, rebal_date in enumerate(rebalance_dates):
        # Get prices for the day
        day_prices = price_by_date.get(rebal_date, {})

        if not day_prices:
            continue

        # Mark-to-market portfolio
        portfolio_value = cash
        for code, pos in list(positions.items()):
            current_price = day_prices.get(code)
            if current_price and current_price > 0:
                portfolio_value += current_price * pos['shares']
            else:
                # Delisted: evaluate at 30% of last price (conservative)
                portfolio_value += pos['entry_price'] * pos['shares'] * 0.3

        # NaN check
        if pd.isna(portfolio_value) or portfolio_value <= 0:
            portfolio_value = cash

        equity_history.append({
            'date': rebal_date,
            'equity': portfolio_value,
            'positions': len(positions),
            'cash': cash
        })

        # === Rebalance Processing ===

        # 1. Close existing positions (held 180+ days)
        codes_to_close = []
        for code, pos in positions.items():
            holding_days = (rebal_date - pos['entry_date']).days
            if holding_days >= 180:
                codes_to_close.append(code)

        for code in codes_to_close:
            pos = positions[code]
            exit_price = day_prices.get(code)

            if not exit_price or exit_price <= 0:
                exit_price = pos['entry_price'] * 0.3  # Assumed delisting

            proceeds = exit_price * pos['shares']
            pnl_pct = (exit_price / pos['entry_price'] - 1) * 100 if pos['entry_price'] > 0 else 0

            all_trades.append({
                'code': code,
                'entry_date': pos['entry_date'],
                'exit_date': rebal_date,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pbr': pos['pbr'],
                'roe': pos['roe']
            })

            cash += proceeds
            del positions[code]

        # 2. Update financial data (latest before evaluation date)
        valid_fins = fins_df[fins_df['disclosed_date'] <= rebal_date]
        if len(valid_fins) > 0:
            latest_fins = valid_fins.groupby('code').last().reset_index()
        else:
            latest_fins = pd.DataFrame()

        # 3. Select new stocks
        if len(positions) < max_positions and len(latest_fins) > 0:
            candidates = []

            for _, fin_row in latest_fins.iterrows():
                code = fin_row['code']
                bps = safe_float(fin_row['bps'], 0)
                roe = safe_float(fin_row['roe'], 0)

                if code in positions or bps <= 0:
                    continue

                current_price = day_prices.get(code)
                if not current_price or current_price <= 0:
                    continue

                pbr = current_price / bps

                # Criteria: PBR <= 1.0 AND ROE >= 10%
                if pbr <= 1.0 and roe >= 10.0:
                    candidates.append({
                        'code': code,
                        'price': current_price,
                        'pbr': pbr,
                        'roe': roe
                    })

            # Sort by ROE
            candidates.sort(key=lambda x: x['roe'], reverse=True)

            # Purchase
            slots = max_positions - len(positions)
            if slots > 0 and candidates and cash > 100000:
                position_size = cash / slots

                for cand in candidates[:slots]:
                    if position_size < 50000:
                        break

                    price = safe_float(cand['price'], 0)
                    if price <= 0:
                        continue

                    shares = int(position_size / price)
                    if shares <= 0:
                        continue

                    cost = shares * price
                    if cost > cash:
                        continue

                    positions[cand['code']] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': rebal_date,
                        'pbr': cand['pbr'],
                        'roe': cand['roe']
                    }
                    cash -= cost

        # Progress display
        if i % 10 == 0:
            logger.info(f"[{rebal_date.strftime('%Y-%m-%d')}] Equity: ¥{portfolio_value:,.0f} | Positions: {len(positions)} | Trades: {len(all_trades)}")

    # === Results Summary ===
    if not equity_history:
        logger.error("No equity history generated")
        return None

    equity_df = pd.DataFrame(equity_history)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df.set_index('date', inplace=True)

    # Remove NaN
    equity_df['equity'] = equity_df['equity'].fillna(method='ffill').fillna(initial_capital)

    final_equity = safe_float(equity_df['equity'].iloc[-1], initial_capital)
    total_return = (final_equity / initial_capital - 1) * 100

    # Annual returns
    equity_df['year'] = equity_df.index.year
    annual_returns = {}
    for year in sorted(equity_df['year'].unique()):
        year_data = equity_df[equity_df['year'] == year]
        if len(year_data) >= 2:
            start_val = safe_float(year_data['equity'].iloc[0], initial_capital)
            end_val = safe_float(year_data['equity'].iloc[-1], initial_capital)
            if start_val > 0:
                annual_returns[year] = (end_val / start_val - 1) * 100

    # Maximum drawdown
    rolling_max = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100
    drawdown = drawdown.fillna(0)
    max_drawdown = drawdown.min()

    # Win rate
    if all_trades:
        wins = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        win_rate = wins / len(all_trades) * 100
    else:
        win_rate = 0

    # Sharpe ratio
    returns = equity_df['equity'].pct_change().dropna()
    returns = returns[~np.isinf(returns)]
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() * 4) / (returns.std() * np.sqrt(4))
    else:
        sharpe = 0

    # === Report output ===
    logger.info("\n" + "=" * 70)
    logger.info("🏆 BACKTEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Initial Capital: ¥{initial_capital:,.0f}")
    logger.info(f"Final Equity:    ¥{final_equity:,.0f}")
    logger.info(f"Total Return:    {total_return:+.2f}%")
    logger.info(f"Max Drawdown:    {max_drawdown:.2f}%")
    logger.info(f"Sharpe Ratio:    {sharpe:.2f}")
    logger.info(f"Win Rate:        {win_rate:.1f}%")
    logger.info(f"Total Trades:    {len(all_trades)}")

    logger.info("\n" + "-" * 40)
    logger.info("📊 ANNUAL RETURNS")
    logger.info("-" * 40)
    for year in sorted(annual_returns.keys()):
        ret = annual_returns.get(year, 0)
        if pd.isna(ret):
            ret = 0
        bar_len = min(int(abs(ret) / 5), 20)
        bar = "█" * bar_len if ret > 0 else "▓" * bar_len
        sign = "+" if ret >= 0 else ""
        logger.info(f"  {year}: {sign}{ret:6.2f}% {bar}")

    # Top/worst trades
    if all_trades:
        valid_trades = [t for t in all_trades if not pd.isna(t['pnl_pct'])]
        sorted_trades = sorted(valid_trades, key=lambda x: x['pnl_pct'], reverse=True)

        logger.info("\n" + "-" * 40)
        logger.info("🚀 TOP 5 TRADES")
        logger.info("-" * 40)
        for t in sorted_trades[:5]:
            logger.info(f"  {t['code']}: {t['pnl_pct']:+.1f}% (PBR:{t['pbr']:.2f}, ROE:{t['roe']:.1f}%)")

        logger.info("\n" + "-" * 40)
        logger.info("💀 WORST 5 TRADES")
        logger.info("-" * 40)
        for t in sorted_trades[-5:]:
            logger.info(f"  {t['code']}: {t['pnl_pct']:+.1f}% (PBR:{t['pbr']:.2f}, ROE:{t['roe']:.1f}%)")

    # === Create chart ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    ax1 = axes[0]
    ax1.plot(equity_df.index, equity_df['equity'] / 1_000_000, 'b-', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=initial_capital / 1_000_000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.fill_between(equity_df.index, initial_capital / 1_000_000, equity_df['equity'] / 1_000_000,
                     where=equity_df['equity'] >= initial_capital, alpha=0.3, color='green')
    ax1.fill_between(equity_df.index, initial_capital / 1_000_000, equity_df['equity'] / 1_000_000,
                     where=equity_df['equity'] < initial_capital, alpha=0.3, color='red')

    ax1.set_title(f'Asset Shield: PBR/ROE Value Strategy (2008-2025)\n'
                  f'Total Return: {total_return:+.1f}% | Max DD: {max_drawdown:.1f}% | Win Rate: {win_rate:.1f}%',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (Million JPY)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    ax2 = axes[1]
    ax2.fill_between(equity_df.index, 0, drawdown, color='red', alpha=0.5)
    ax2.axhline(y=max_drawdown, color='darkred', linestyle='--', alpha=0.7, label=f'Max DD: {max_drawdown:.1f}%')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()

    output_path = os.path.join(project_root, 'output', f'backtest_pbr_roe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n📊 Chart saved: {output_path}")
    plt.close()

    # Save JSON results
    import json
    result_data = {
        'strategy': 'PBR <= 1.0 AND ROE >= 10%',
        'period': f'{start_date} to {end_date}',
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'win_rate_pct': win_rate,
        'total_trades': len(all_trades),
        'annual_returns': {k: round(v, 2) for k, v in annual_returns.items()}
    }
    json_path = os.path.join(project_root, 'output', f'backtest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    logger.info(f"📄 Results saved: {json_path}")

    return result_data


if __name__ == "__main__":
    cache_path = os.path.join(project_root, 'data', 'jquants_cache.db')
    run_backtest(cache_path)
