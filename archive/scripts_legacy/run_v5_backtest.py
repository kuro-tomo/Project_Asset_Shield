#!/usr/bin/env python3
"""
Asset Shield V5.0 - 20-Year Backtest Runner
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import date
from shield.backtest_v5 import AssetShieldV5Backtest, V5Config

def main():
    print("=" * 70)
    print("ASSET SHIELD V5.0 - 20-YEAR BACKTEST")
    print("=" * 70)

    config = V5Config(
        start_date=date(2008, 1, 1),  # Data starts from 2008
        end_date=date(2026, 2, 1),
        initial_capital=100_000_000,
        max_positions=25,
        max_single_weight=0.08,
        max_sector_weight=0.25,
        min_adt=100_000_000,
        max_adt_participation=0.05,
        vol_threshold_bull=0.18,
        vol_threshold_crisis=0.35,
        max_drawdown=0.15,
        recovery_steps=[0.0, 0.30, 0.60, 0.90],
        recovery_days=5,
        rebalance_frequency=20
    )

    engine = AssetShieldV5Backtest(config)
    engine.load_data()

    print("\nRunning backtest...")
    metrics = engine.run_backtest()

    report = engine.generate_report(metrics)
    print(report)

    # Save results
    output_dir = Path("/Users/MBP/Desktop/Project_Asset_Shield/output/v5")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "V5_20YEAR_REPORT.txt", 'w') as f:
        f.write(report)

    import pandas as pd
    equity_df = pd.DataFrame(engine.equity_curve, columns=['date', 'equity'])
    equity_df.to_csv(output_dir / "v5_equity_curve.csv", index=False)

    print(f"\nResults saved to {output_dir}")

    return metrics

if __name__ == "__main__":
    main()
