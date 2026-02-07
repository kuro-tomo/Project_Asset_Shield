#!/usr/bin/env python3
"""
Asset Shield - PBR/ROE Value Strategy Backtest
===============================================
Undervalued quality stock strategy: PBR <= 1.0 AND ROE >= 10%

Strategy Logic:
1. After earnings release, extract stocks with PBR <= 1.0 AND ROE >= 10%
2. Purchase with equal amounts on next trading day
3. Sell and rebalance after 1 year (250 trading days) holding

Survivorship Bias Elimination:
- Include delisted stocks in calculations
- Calculate delisting as forced settlement at final trading price
"""

import os
import sys
import sqlite3
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Held position"""
    code: str
    entry_date: str
    entry_price: float
    shares: int
    pbr_at_entry: float
    roe_at_entry: float


@dataclass
class BacktestResult:
    """Backtest results"""
    total_return: float
    annual_returns: Dict[int, float]
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    equity_curve: pd.DataFrame


class PBRROEBacktester:
    """PBR/ROE Strategy Backtester"""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.conn = sqlite3.connect(cache_path)
        self.initial_capital = 10_000_000  # 10M JPY
        self.max_positions = 20  # Maximum number of positions
        self.holding_days = 250  # Holding period (trading days)

    def get_financial_data(self) -> pd.DataFrame:
        """Get financial data for PBR calculation"""
        query = """
            SELECT
                code,
                disclosed_date,
                fiscal_year,
                bps,
                roe,
                eps,
                net_sales,
                equity
            FROM financial_statements
            WHERE bps IS NOT NULL AND bps > 0
            ORDER BY code, disclosed_date
        """
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Financial data: {len(df):,} records, {df['code'].nunique():,} stocks")
        return df

    def get_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data"""
        query = """
            SELECT
                code,
                date,
                open,
                high,
                low,
                close,
                volume,
                adjustment_close
            FROM daily_quotes
            WHERE date BETWEEN ? AND ?
            ORDER BY date, code
        """
        df = pd.read_sql_query(query, self.conn, params=[start_date, end_date])
        logger.info(f"Price data: {len(df):,} records, {df['code'].nunique():,} stocks")
        return df

    def calculate_pbr(self, price: float, bps: float) -> Optional[float]:
        """Calculate PBR (Price-to-Book Ratio)"""
        if bps is None or bps <= 0:
            return None
        return price / bps

    def find_candidates(self, date: str, prices: pd.DataFrame, financials: pd.DataFrame) -> List[Dict]:
        """
        Extract stocks with PBR<=1.0 AND ROE>=10% as of specified date

        Args:
            date: Evaluation date
            prices: Price data
            financials: Financial data
        """
        candidates = []

        # Prices for the day
        day_prices = prices[prices['date'] == date].copy()
        if day_prices.empty:
            return candidates

        for _, price_row in day_prices.iterrows():
            code = price_row['code']
            close_price = price_row['adjustment_close'] or price_row['close']

            if close_price is None or close_price <= 0:
                continue

            # Get latest financial data (before evaluation date)
            code_fins = financials[
                (financials['code'] == code) &
                (financials['disclosed_date'] <= date)
            ]

            if code_fins.empty:
                continue

            latest_fin = code_fins.iloc[-1]
            bps = latest_fin['bps']
            roe = latest_fin['roe']

            if bps is None or bps <= 0:
                continue

            pbr = self.calculate_pbr(close_price, bps)

            # Criteria: PBR <= 1.0 AND ROE >= 10%
            if pbr is not None and pbr <= 1.0 and roe is not None and roe >= 10.0:
                candidates.append({
                    'code': code,
                    'date': date,
                    'price': close_price,
                    'bps': bps,
                    'pbr': pbr,
                    'roe': roe
                })

        return candidates

    def run_backtest(self, start_date: str = "2008-01-01", end_date: str = "2025-12-31") -> BacktestResult:
        """
        Execute backtest

        Args:
            start_date: Start date
            end_date: End date
        """
        logger.info("=" * 70)
        logger.info("PBR/ROE VALUE STRATEGY BACKTEST")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Criteria: PBR ≤ 1.0 AND ROE ≥ 10%")
        logger.info("=" * 70)

        # Load data
        financials = self.get_financial_data()
        prices = self.get_price_data(start_date, end_date)

        if financials.empty:
            logger.error("No financial data available")
            return None

        # Trading day list
        trading_dates = sorted(prices['date'].unique())
        logger.info(f"Trading days: {len(trading_dates)}")

        # Initialize portfolio
        cash = self.initial_capital
        positions: List[Position] = []
        equity_history = []
        trades = []
        annual_pnl = {}

        # Rebalance interval (quarterly)
        rebalance_interval = 63  # ~3 months

        for i, date in enumerate(trading_dates):
            # Mark-to-market current holdings
            day_prices = prices[prices['date'] == date]
            price_dict = dict(zip(day_prices['code'],
                                  day_prices['adjustment_close'].fillna(day_prices['close'])))

            portfolio_value = cash
            for pos in positions:
                current_price = price_dict.get(pos.code)
                if current_price:
                    portfolio_value += current_price * pos.shares
                else:
                    # Unknown price (delisting etc.) evaluated at last price
                    portfolio_value += pos.entry_price * pos.shares * 0.5  # 50% haircut assumed

            equity_history.append({
                'date': date,
                'equity': portfolio_value,
                'positions': len(positions),
                'cash': cash
            })

            # Annual tracking
            year = int(date[:4])
            if year not in annual_pnl:
                annual_pnl[year] = {'start': portfolio_value, 'end': portfolio_value}
            annual_pnl[year]['end'] = portfolio_value

            # Check if rebalance day
            if i % rebalance_interval != 0:
                continue

            # Close positions held 1+ years
            positions_to_close = []
            for pos in positions:
                days_held = trading_dates.index(date) - trading_dates.index(pos.entry_date) if pos.entry_date in trading_dates else self.holding_days + 1
                if days_held >= self.holding_days:
                    positions_to_close.append(pos)

            for pos in positions_to_close:
                current_price = price_dict.get(pos.code, pos.entry_price * 0.5)
                proceeds = current_price * pos.shares
                pnl = proceeds - (pos.entry_price * pos.shares)
                pnl_pct = (current_price / pos.entry_price - 1) * 100

                trades.append({
                    'code': pos.code,
                    'entry_date': pos.entry_date,
                    'exit_date': date,
                    'entry_price': pos.entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'pbr': pos.pbr_at_entry,
                    'roe': pos.roe_at_entry
                })

                cash += proceeds
                positions.remove(pos)

            # Select new stocks
            if len(positions) < self.max_positions:
                candidates = self.find_candidates(date, prices, financials)

                # Exclude already-held stocks
                held_codes = {p.code for p in positions}
                candidates = [c for c in candidates if c['code'] not in held_codes]

                # Sort by ROE descending
                candidates.sort(key=lambda x: x['roe'], reverse=True)

                # Purchase for available slots
                slots_available = self.max_positions - len(positions)
                position_size = cash / max(slots_available, 1)

                for candidate in candidates[:slots_available]:
                    if position_size < 100000:  # Minimum 100K JPY
                        break

                    shares = int(position_size / candidate['price'])
                    if shares <= 0:
                        continue

                    cost = shares * candidate['price']
                    if cost > cash:
                        continue

                    positions.append(Position(
                        code=candidate['code'],
                        entry_date=date,
                        entry_price=candidate['price'],
                        shares=shares,
                        pbr_at_entry=candidate['pbr'],
                        roe_at_entry=candidate['roe']
                    ))

                    cash -= cost

            # Progress display
            if i % 250 == 0:
                logger.info(f"[{date}] Equity: ¥{portfolio_value:,.0f}, Positions: {len(positions)}, Trades: {len(trades)}")

        # Results summary
        equity_df = pd.DataFrame(equity_history)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)

        # Return calculation
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1) * 100

        # Annual returns
        annual_returns = {}
        for year, data in annual_pnl.items():
            if data['start'] > 0:
                annual_returns[year] = (data['end'] / data['start'] - 1) * 100

        # Maximum drawdown
        rolling_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Sharpe ratio
        daily_returns = equity_df['equity'].pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0

        # Win rate
        if trades:
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = winning_trades / len(trades) * 100
        else:
            win_rate = 0

        result = BacktestResult(
            total_return=total_return,
            annual_returns=annual_returns,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(trades),
            equity_curve=equity_df
        )

        # Output report
        self._print_report(result, trades)
        self._plot_equity_curve(result.equity_curve, result.max_drawdown)

        return result

    def _print_report(self, result: BacktestResult, trades: List[Dict]):
        """Output results report"""
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Initial Capital: ¥{self.initial_capital:,}")
        logger.info(f"Final Equity: ¥{result.equity_curve['equity'].iloc[-1]:,.0f}")
        logger.info(f"Total Return: {result.total_return:.2f}%")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2f}%")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Win Rate: {result.win_rate:.1f}%")
        logger.info(f"Total Trades: {result.total_trades}")

        logger.info("\n" + "-" * 40)
        logger.info("ANNUAL RETURNS")
        logger.info("-" * 40)
        for year in sorted(result.annual_returns.keys()):
            ret = result.annual_returns[year]
            logger.info(f"  {year}: {ret:+.2f}%")

        # Top/bottom trades
        if trades:
            trades_sorted = sorted(trades, key=lambda x: x['pnl_pct'], reverse=True)
            logger.info("\n" + "-" * 40)
            logger.info("TOP 5 TRADES")
            logger.info("-" * 40)
            for t in trades_sorted[:5]:
                logger.info(f"  {t['code']}: {t['pnl_pct']:+.1f}% (PBR:{t['pbr']:.2f}, ROE:{t['roe']:.1f}%)")

            logger.info("\n" + "-" * 40)
            logger.info("WORST 5 TRADES")
            logger.info("-" * 40)
            for t in trades_sorted[-5:]:
                logger.info(f"  {t['code']}: {t['pnl_pct']:+.1f}% (PBR:{t['pbr']:.2f}, ROE:{t['roe']:.1f}%)")

    def _plot_equity_curve(self, equity_df: pd.DataFrame, max_dd: float):
        """Draw cumulative return curve"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Cumulative equity curve
        ax1 = axes[0]
        ax1.plot(equity_df.index, equity_df['equity'] / 1_000_000, 'b-', linewidth=1.5, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital / 1_000_000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(equity_df.index, self.initial_capital / 1_000_000, equity_df['equity'] / 1_000_000,
                         where=equity_df['equity'] >= self.initial_capital, alpha=0.3, color='green')
        ax1.fill_between(equity_df.index, self.initial_capital / 1_000_000, equity_df['equity'] / 1_000_000,
                         where=equity_df['equity'] < self.initial_capital, alpha=0.3, color='red')

        ax1.set_title('Asset Shield: PBR/ROE Value Strategy (2008-2025)\nPBR ≤ 1.0 AND ROE ≥ 10%', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Million JPY)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))

        # Drawdown
        ax2 = axes[1]
        rolling_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max * 100
        ax2.fill_between(equity_df.index, 0, drawdown, color='red', alpha=0.5)
        ax2.axhline(y=max_dd, color='darkred', linestyle='--', alpha=0.7, label=f'Max DD: {max_dd:.1f}%')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))

        plt.tight_layout()

        # Save
        output_path = os.path.join(project_root, 'output', f'backtest_pbr_roe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"\nChart saved to: {output_path}")
        plt.close()


def main():
    cache_path = os.path.join(project_root, 'data', 'jquants_cache.db')

    backtester = PBRROEBacktester(cache_path)
    result = backtester.run_backtest(
        start_date="2008-01-01",
        end_date="2025-12-31"
    )

    if result:
        # Save results as JSON
        import json
        output_data = {
            'total_return': result.total_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'annual_returns': result.annual_returns
        }
        output_path = os.path.join(project_root, 'output', f'backtest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
