#!/usr/bin/env python3
"""
Asset Shield V4.0 - Results Aggregator
Combines segment results into Grand Strategy Report
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def aggregate_results(output_dir: str):
    """Aggregate all segment results"""
    output_path = Path(output_dir)

    segments = []
    for i in range(1, 5):
        json_path = output_path / f"segment_{i}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                segments.append(json.load(f))

    if not segments:
        print("ERROR: No segment results found")
        return None

    print("=" * 70)
    print("ASSET SHIELD V4.0 - GRAND STRATEGY REPORT")
    print("20-Year Backtest Results (2006-2026)")
    print("=" * 70)
    print()

    # Combine equity curves
    all_equity = []
    for seg in segments:
        all_equity.extend(seg['equity_curve'])

    # Sort by date
    all_equity.sort(key=lambda x: x[0])

    # Calculate overall metrics
    initial = 100_000_000
    dates = [x[0] for x in all_equity]
    equities = [x[1] for x in all_equity]
    final = equities[-1]

    total_return = (final / initial) - 1
    years = (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days / 365.25
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0

    # Daily returns for Sharpe
    daily_returns = pd.Series(equities).pct_change().dropna()
    sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() * 252) / (downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

    # Max Drawdown
    peak = pd.Series(equities).expanding().max()
    drawdown = (peak - pd.Series(equities)) / peak
    max_dd = drawdown.max()

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0

    print("-" * 70)
    print("AGGREGATE PERFORMANCE (2006-2026)")
    print("-" * 70)
    print(f"Initial Capital:   ¥{initial:>15,.0f}")
    print(f"Final Equity:      ¥{final:>15,.0f}")
    print(f"Total Return:      {total_return:>15.2%}")
    print(f"CAGR:              {cagr:>15.2%}")
    print(f"Sharpe Ratio:      {sharpe:>15.2f}")
    print(f"Sortino Ratio:     {sortino:>15.2f}")
    print(f"Max Drawdown:      {max_dd:>15.2%}")
    print(f"Calmar Ratio:      {calmar:>15.2f}")
    print()

    print("-" * 70)
    print("SEGMENT BREAKDOWN")
    print("-" * 70)
    print(f"{'Segment':<20} {'Period':<25} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 70)

    segment_names = [
        "Phase 1: Survival",
        "Phase 2: Expansion",
        "Phase 3: OOS",
        "Phase 4: Modern"
    ]

    for i, seg in enumerate(segments):
        m = seg['metrics']
        name = segment_names[i] if i < len(segment_names) else f"Segment {i+1}"
        period = f"{seg['start_date']} to {seg['end_date']}"
        print(f"{name:<20} {period:<25} {m['total_return']:>9.2%} {m['sharpe_ratio']:>10.2f} {m['max_drawdown']:>9.2%}")

    print()
    print("-" * 70)
    print("STRESS TEST RESULTS")
    print("-" * 70)

    all_stress = []
    for seg in segments:
        all_stress.extend(seg.get('stress_tests', []))

    for st in all_stress:
        print(f"\n{st['period_name']}:")
        print(f"  Return: {st['return_pct']:>10.2%}")
        print(f"  Max DD: {st['max_drawdown']:>10.2%}")
        print(f"  Sharpe: {st['sharpe_ratio']:>10.2f}")
        print(f"  Kill Switch Triggers: {st['kill_switch_triggers']}")

    print()
    print("-" * 70)
    print("REGIME STATISTICS (Aggregate)")
    print("-" * 70)

    total_bull = sum(seg['regime_stats'].get('bull_days', 0) for seg in segments)
    total_bear = sum(seg['regime_stats'].get('bear_days', 0) for seg in segments)
    total_crisis = sum(seg['regime_stats'].get('crisis_days', 0) for seg in segments)
    total_days = total_bull + total_bear + total_crisis

    if total_days > 0:
        print(f"Bull Days:         {total_bull:>10} ({total_bull/total_days:>6.1%})")
        print(f"Bear Days:         {total_bear:>10} ({total_bear/total_days:>6.1%})")
        print(f"Crisis Days:       {total_crisis:>10} ({total_crisis/total_days:>6.1%})")

    total_trades = sum(seg.get('trade_count', 0) for seg in segments)
    print(f"Total Trades:      {total_trades:>10}")

    print()
    print("=" * 70)
    print("V4.0 DOMINANCE PHASE - VALIDATION COMPLETE")
    print("=" * 70)

    # Save combined equity curve
    equity_df = pd.DataFrame(all_equity, columns=['date', 'equity'])
    equity_df.to_csv(output_path / "v4_combined_equity.csv", index=False)
    print(f"\nCombined equity curve saved to: {output_path / 'v4_combined_equity.csv'}")

    # Save summary
    summary = {
        "period": "2006-2026",
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "total_trades": total_trades,
        "regime_distribution": {
            "bull_pct": total_bull / total_days if total_days > 0 else 0,
            "bear_pct": total_bear / total_days if total_days > 0 else 0,
            "crisis_pct": total_crisis / total_days if total_days > 0 else 0
        }
    }

    with open(output_path / "v4_grand_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_path / 'v4_grand_summary.json'}")

    return summary


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/MBP/Desktop/Project_Asset_Shield/output/v4_parallel"
    aggregate_results(output_dir)
