#!/usr/bin/env python3
"""
Asset Shield V4.0 - Parallel Worker
Executes backtest for a single time segment
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import date

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def run_segment(segment_id: int, start_date: date, end_date: date, output_path: str):
    """Run backtest for a single segment"""
    from shield.backtest_v4 import AssetShieldV4Backtest, BacktestConfig

    logger.info(f"Worker {segment_id}: {start_date} to {end_date}")

    # V4.1 OPTIMIZED PARAMETERS
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100_000_000,
        max_positions=20,
        min_positions=10,
        max_single_weight=0.10,
        max_sector_weight=0.25,
        max_adt_participation=0.05,
        min_adt=100_000_000,
        vol_threshold_low=0.18,      # V4.1: 15% -> 18%
        vol_threshold_high=0.35,
        rebalance_frequency=20,
        max_drawdown=0.20,           # V4.1: 15% -> 20%
        correlation_threshold=0.8
    )

    engine = AssetShieldV4Backtest(config)
    engine.load_data()
    metrics = engine.run_backtest()
    stress_results = engine.run_stress_tests()

    # Convert numpy types to native Python types
    def to_native(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    # Prepare output
    result = {
        "segment_id": segment_id,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "metrics": {
            "total_return": to_native(metrics.total_return),
            "cagr": to_native(metrics.cagr),
            "sharpe_ratio": to_native(metrics.sharpe_ratio),
            "sortino_ratio": to_native(metrics.sortino_ratio),
            "max_drawdown": to_native(metrics.max_drawdown),
            "max_drawdown_date": str(metrics.max_drawdown_date),
            "calmar_ratio": to_native(metrics.calmar_ratio),
            "win_rate": to_native(metrics.win_rate),
            "profit_factor": to_native(metrics.profit_factor),
            "avg_monthly_return": to_native(metrics.avg_monthly_return),
            "monthly_std": to_native(metrics.monthly_std),
            "best_month": to_native(metrics.best_month),
            "worst_month": to_native(metrics.worst_month),
            "positive_months": int(metrics.positive_months),
            "negative_months": int(metrics.negative_months),
        },
        "regime_stats": {k: to_native(v) if not isinstance(v, str) else v for k, v in metrics.regime_stats.items()},
        "stress_tests": [{
            "period_name": sr.period_name,
            "return_pct": to_native(sr.return_pct),
            "max_drawdown": to_native(sr.max_drawdown),
            "sharpe_ratio": to_native(sr.sharpe_ratio),
            "kill_switch_triggers": int(sr.kill_switch_triggers),
            "regime_distribution": {k: to_native(v) for k, v in sr.regime_distribution.items()}
        } for sr in stress_results],
        "equity_curve": [(str(d), float(e)) for d, e in engine.equity_curve],
        "monthly_returns": [(str(m), float(r)) for m, r in engine.monthly_returns],
        "trade_count": int(len(engine.trades))
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Worker {segment_id} complete: Return={metrics.total_return:.2%}, Sharpe={metrics.sharpe_ratio:.2f}")
    logger.info(f"Results saved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=int, required=True, help="Segment ID (1-4)")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    run_segment(args.segment, start, end, args.output)
