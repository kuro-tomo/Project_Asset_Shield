#!/usr/bin/env python3
"""
Asset Shield V4.0 - Dominance Phase Backtest Runner
20-Year Historical Validation

Usage:
    python scripts/run_v4_backtest.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Run V4.0 backtest"""
    print("=" * 70)
    print("ASSET SHIELD V4.0 - DOMINANCE PHASE")
    print("20-Year Backtest with HMM Regime Detection")
    print("=" * 70)
    print()

    from shield.backtest_v4 import AssetShieldV4Backtest, BacktestConfig

    # Configuration
    config = BacktestConfig(
        start_date=date(2008, 1, 1),  # Start from 2008 (data availability)
        end_date=date(2026, 2, 1),
        initial_capital=100_000_000,  # 1億円
        max_positions=20,
        min_positions=10,
        max_single_weight=0.10,
        max_sector_weight=0.25,
        max_adt_participation=0.05,
        min_adt=100_000_000,
        vol_threshold_low=0.15,
        vol_threshold_high=0.35,
        rebalance_frequency=20,
        max_drawdown=0.15,
        correlation_threshold=0.8
    )

    print(f"Configuration:")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Initial Capital: ¥{config.initial_capital:,.0f}")
    print(f"  Max Positions: {config.max_positions}")
    print(f"  Max Sector: {config.max_sector_weight:.0%}")
    print(f"  Max ADT Participation: {config.max_adt_participation:.0%}")
    print(f"  Volatility Thresholds: Low={config.vol_threshold_low:.0%}, High={config.vol_threshold_high:.0%}")
    print()

    # Initialize engine
    logger.info("Initializing V4.0 backtest engine...")
    engine = AssetShieldV4Backtest(config)

    # Load data
    logger.info("Loading historical data...")
    engine.load_data()

    # Run backtest
    logger.info("Running backtest...")
    metrics = engine.run_backtest()

    # Generate report
    report = engine.generate_report()
    print()
    print(report)

    # Save outputs
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # Save report
    report_path = output_dir / "v4_backtest_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save equity curve
    import pandas as pd
    equity_df = pd.DataFrame(engine.equity_curve, columns=['date', 'equity'])
    equity_path = output_dir / "v4_equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    print(f"Equity curve saved to: {equity_path}")

    # Save monthly returns
    if engine.monthly_returns:
        monthly_df = pd.DataFrame(engine.monthly_returns, columns=['month', 'return'])
        monthly_path = output_dir / "v4_monthly_returns.csv"
        monthly_df.to_csv(monthly_path, index=False)
        print(f"Monthly returns saved to: {monthly_path}")

    # Save trades
    if engine.trades:
        trades_data = [{
            'date': t.date,
            'code': t.code,
            'side': t.side,
            'shares': t.shares,
            'price': t.price,
            'value': t.value,
            'reason': t.reason
        } for t in engine.trades]
        trades_df = pd.DataFrame(trades_data)
        trades_path = output_dir / "v4_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"Trades saved to: {trades_path}")

    print()
    print("=" * 70)
    print("V4.0 BACKTEST COMPLETE")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    main()
