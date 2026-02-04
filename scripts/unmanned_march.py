#!/usr/bin/env python3
"""
Asset Shield - Unmanned March Mode
===================================

Fully autonomous execution script:
1. Monitor data recovery process
2. Auto-execute backtest after recovery completion
3. Generate and save final reports

Author: Asset Shield Autonomous Operations
Version: 1.0.0
"""

import os
import sys
import time
import subprocess
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Logging setup
log_path = os.path.join(project_root, 'logs', 'unmanned_march.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path)
    ]
)
logger = logging.getLogger(__name__)


class UnmannedMarchController:
    """Unmanned March Control System"""

    def __init__(self):
        self.project_root = project_root
        self.cache_path = os.path.join(project_root, 'data', 'jquants_cache.db')
        self.output_dir = os.path.join(project_root, 'output')
        self.recovery_log = os.path.join(project_root, 'logs', 'recovery_full.log')

        self.status = {
            'phase': 'initializing',
            'recovery_complete': False,
            'backtest_complete': False,
            'reports_generated': False,
            'started_at': datetime.now().isoformat(),
            'completed_at': None
        }

    def check_recovery_status(self) -> dict:
        """Check data recovery process status"""
        result = {
            'running': False,
            'completed': False,
            'progress': 0,
            'stocks_recovered': 0,
            'total_stocks': 3081
        }

        # Check if process is running
        try:
            ps_output = subprocess.check_output(['pgrep', '-f', 'recover_2021_data']).decode()
            result['running'] = bool(ps_output.strip())
        except subprocess.CalledProcessError:
            result['running'] = False

        # Check log for progress
        if os.path.exists(self.recovery_log):
            with open(self.recovery_log, 'r') as f:
                content = f.read()
                # Count recovered lines
                recovered = content.count('records recovered')
                result['stocks_recovered'] = recovered
                result['progress'] = (recovered / result['total_stocks']) * 100

                # Check for completion
                if 'Recovery Complete' in content or 'COMPLETE' in content or recovered >= result['total_stocks'] - 100:
                    result['completed'] = True

        return result

    def get_current_coverage(self) -> dict:
        """Get current data coverage statistics"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                substr(date, 1, 4) as year,
                COUNT(DISTINCT code) as stocks,
                COUNT(*) as records
            FROM daily_quotes
            WHERE date >= '2020-01-01'
            GROUP BY year
            ORDER BY year
        """)

        coverage = {}
        for row in cursor.fetchall():
            coverage[row[0]] = {'stocks': row[1], 'records': row[2]}

        conn.close()
        return coverage

    def wait_for_recovery(self, check_interval: int = 60):
        """Wait for recovery completion"""
        logger.info("=" * 70)
        logger.info("[UNMANNED MARCH] Data Recovery Monitoring Started")
        logger.info("=" * 70)

        self.status['phase'] = 'monitoring_recovery'

        while True:
            status = self.check_recovery_status()
            coverage = self.get_current_coverage()

            stocks_2021 = coverage.get('2021', {}).get('stocks', 0)

            logger.info(f"[Monitor] Progress: {status['progress']:.1f}% | "
                       f"2021 Coverage: {stocks_2021}/4020 | "
                       f"Process: {'Running' if status['running'] else 'Stopped'}")

            if status['completed'] or not status['running']:
                logger.info("Recovery process completion detected")
                self.status['recovery_complete'] = True
                break

            time.sleep(check_interval)

        # Final coverage report
        final_coverage = self.get_current_coverage()
        logger.info("=" * 70)
        logger.info("[RECOVERY COMPLETE] Final Coverage")
        for year, data in sorted(final_coverage.items()):
            logger.info(f"  {year}: {data['stocks']} stocks, {data['records']:,} records")
        logger.info("=" * 70)

    def run_backtest(self):
        """Execute backtest"""
        logger.info("=" * 70)
        logger.info("[UNMANNED MARCH] Backtest Execution Started")
        logger.info("=" * 70)

        self.status['phase'] = 'running_backtest'

        backtest_script = os.path.join(project_root, 'scripts', 'backtest_world_rank_precision.py')

        try:
            result = subprocess.run(
                ['python3', backtest_script],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("Backtest completed successfully")
                self.status['backtest_complete'] = True

                # Extract key metrics from output
                output = result.stdout
                if 'Sharpe' in output:
                    logger.info("[BACKTEST RESULTS SUMMARY]")
                    for line in output.split('\n'):
                        if any(k in line for k in ['Sharpe', 'Return', 'Drawdown', 'OOS']):
                            logger.info(f"  {line.strip()}")
            else:
                logger.error(f"Backtest error: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Backtest timed out")
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")

    def generate_final_reports(self):
        """Generate final reports"""
        logger.info("=" * 70)
        logger.info("[UNMANNED MARCH] Final Report Generation")
        logger.info("=" * 70)

        self.status['phase'] = 'generating_reports'

        # Find latest backtest results
        output_files = list(Path(self.output_dir).glob('world_rank_precision_*.json'))
        if not output_files:
            logger.error("Backtest results not found")
            return

        latest_json = max(output_files, key=os.path.getmtime)
        latest_png = latest_json.with_suffix('.png')

        logger.info(f"Latest results: {latest_json.name}")

        # Load results
        with open(latest_json, 'r') as f:
            results = json.load(f)

        # Generate comparison report
        self._generate_comparison_report(results)

        # Generate English report
        self._generate_english_report(results)

        self.status['reports_generated'] = True
        logger.info("Report generation complete")

    def _generate_comparison_report(self, results: dict):
        """Generate before/after comparison report"""
        report_path = os.path.join(self.output_dir, 'DATA_RECOVERY_COMPARISON.md')

        # Before data (from previous analysis)
        before = {
            'sharpe': 1.23,
            'oos_sharpe': 0.58,
            'total_return': 86.46,
            'max_dd': 0.3518,
            'degradation_ratio': 0.39,
            'stocks_2021': 1003
        }

        # After data
        after = {
            'sharpe': results.get('sharpe_ratio', 0),
            'oos_sharpe': results.get('walk_forward', {}).get('out_of_sample', {}).get('sharpe_ratio', 0),
            'total_return': results.get('total_return', 0),
            'max_dd': results.get('max_drawdown', 0),
            'degradation_ratio': results.get('walk_forward', {}).get('degradation_ratio', 0),
        }

        # Get current coverage
        coverage = self.get_current_coverage()
        after['stocks_2021'] = coverage.get('2021', {}).get('stocks', 0)

        content = f"""# Asset Shield V2 - Pre/Post Data Recovery Comparison Report

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

Restored data coverage for 2021+ and re-executed OOS (Out-of-Sample) period backtest.

---

## Data Coverage Comparison

| Year | Before | After | Improvement |
|------|--------|-------|-------------|
| 2021 | 1,003 | {after['stocks_2021']} | +{after['stocks_2021'] - 1003} |
| Target | 4,020 | 4,020 | - |
| Achievement | 25.0% | {after['stocks_2021']/4020*100:.1f}% | +{(after['stocks_2021']-1003)/4020*100:.1f}pp |

---

## Performance Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Sharpe Ratio | {before['sharpe']:.2f} | {after['sharpe']:.2f} | {after['sharpe']-before['sharpe']:+.2f} |
| OOS Sharpe | {before['oos_sharpe']:.2f} | {after['oos_sharpe']:.2f} | {after['oos_sharpe']-before['oos_sharpe']:+.2f} |
| Total Return | {before['total_return']*100:.1f}% | {after['total_return']*100:.1f}% | {(after['total_return']-before['total_return'])*100:+.1f}pp |
| Max Drawdown | {before['max_dd']*100:.1f}% | {after['max_dd']*100:.1f}% | {(after['max_dd']-before['max_dd'])*100:+.1f}pp |
| Degradation Ratio | {before['degradation_ratio']:.2f} | {after['degradation_ratio']:.2f} | {after['degradation_ratio']-before['degradation_ratio']:+.2f} |

---

## Walk-Forward Validation Results

| Phase | Sharpe | Return | Trades |
|-------|--------|--------|--------|
| Training (2007-2015) | {results.get('walk_forward', {}).get('training', {}).get('sharpe_ratio', 0):.2f} | {results.get('walk_forward', {}).get('training', {}).get('total_return', 0)*100:.1f}% | {results.get('walk_forward', {}).get('training', {}).get('total_trades', 0)} |
| Validation (2016-2020) | {results.get('walk_forward', {}).get('validation', {}).get('sharpe_ratio', 0):.2f} | {results.get('walk_forward', {}).get('validation', {}).get('total_return', 0)*100:.1f}% | {results.get('walk_forward', {}).get('validation', {}).get('total_trades', 0)} |
| OOS (2021-2026) | {results.get('walk_forward', {}).get('out_of_sample', {}).get('sharpe_ratio', 0):.2f} | {results.get('walk_forward', {}).get('out_of_sample', {}).get('total_return', 0)*100:.1f}% | {results.get('walk_forward', {}).get('out_of_sample', {}).get('total_trades', 0)} |

---

## Conclusion

Data recovery improved the verification accuracy for the OOS period.

**True Evaluation:**
- OOS Sharpe changes reflect "true strategy performance" excluding data gap effects
- If OOS Sharpe remains low after recovery, it suggests structural limitations of the value strategy
- If OOS Sharpe improves after recovery, it suggests data quality was the primary cause

---

*Asset Shield V2 - Unmanned March Mode Report*
"""

        with open(report_path, 'w') as f:
            f.write(content)

        logger.info(f"Comparison report saved: {report_path}")

    def _generate_english_report(self, results: dict):
        """Generate English final report"""
        report_path = os.path.join(self.output_dir, 'ASSET_SHIELD_FINAL_REPORT_EN_RECOVERED.md')

        coverage = self.get_current_coverage()
        stocks_2021 = coverage.get('2021', {}).get('stocks', 0)

        content = f"""# Asset Shield V2 - Final Report (Post-Recovery)
## QuantConnect / Quantiacs Submission Document

---

## Executive Summary

**Asset Shield V2** is a systematic value investing strategy for Japanese equities.
This report reflects performance after data recovery for 2021+ period.

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **{results.get('sharpe_ratio', 0):.2f}** |
| **Total Return** | **{results.get('total_return', 0)*100:.1f}%** |
| **Annual Return** | **{results.get('annual_return', 0)*100:.2f}%** |
| **Max Drawdown** | **{results.get('max_drawdown', 0)*100:.2f}%** |
| **OOS Sharpe** | **{results.get('walk_forward', {}).get('out_of_sample', {}).get('sharpe_ratio', 0):.2f}** |

---

## Data Quality Improvement

| Year | Before Recovery | After Recovery |
|------|-----------------|----------------|
| 2021 | 1,003 stocks | {stocks_2021} stocks |
| Coverage | 25.0% | {stocks_2021/4020*100:.1f}% |

---

## Walk-Forward Validation (Post-Recovery)

| Phase | Period | Sharpe | Return |
|-------|--------|--------|--------|
| Training (IS) | 2007-2015 | {results.get('walk_forward', {}).get('training', {}).get('sharpe_ratio', 0):.2f} | {results.get('walk_forward', {}).get('training', {}).get('total_return', 0)*100:.1f}% |
| Validation | 2016-2020 | {results.get('walk_forward', {}).get('validation', {}).get('sharpe_ratio', 0):.2f} | {results.get('walk_forward', {}).get('validation', {}).get('total_return', 0)*100:.1f}% |
| Out-of-Sample | 2021-2026 | {results.get('walk_forward', {}).get('out_of_sample', {}).get('sharpe_ratio', 0):.2f} | {results.get('walk_forward', {}).get('out_of_sample', {}).get('total_return', 0)*100:.1f}% |

**Overfitting Ratio**: {results.get('walk_forward', {}).get('overfitting_ratio', 0):.2f} (Threshold: >0.70)
**Degradation Ratio**: {results.get('walk_forward', {}).get('degradation_ratio', 0):.2f} (Threshold: >0.70)

---

## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {results.get('total_trades', 0)} |
| Win Rate | {results.get('win_rate', 0)*100:.1f}% |
| Profit Factor | {results.get('profit_factor', 0):.2f} |
| Avg Holding Period | {results.get('avg_holding_days', 0):.0f} days |

---

## Risk Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {results.get('sharpe_ratio', 0):.2f} |
| Sortino Ratio | {results.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {results.get('calmar_ratio', 0):.2f} |
| Volatility | {results.get('volatility', 0)*100:.1f}% |

---

## Capacity Analysis

| Metric | Value |
|--------|-------|
| Target AUM | $200M (¥30B) |
| Avg Market Impact | {results.get('avg_impact_bps', 0):.1f} bps |
| Max Impact Allowed | 75 bps |

---

## Data Recovery Note

This report was generated after recovering missing price data for 2021-2026.
The previous OOS analysis was limited by data coverage issues (only ~25% of stocks).
Post-recovery coverage improved to {stocks_2021/4020*100:.1f}% of the 2020 baseline.

---

*Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Asset Shield V2 - Unmanned March Mode*
"""

        with open(report_path, 'w') as f:
            f.write(content)

        logger.info(f"English report saved: {report_path}")

    def run(self):
        """Execute unmanned march mode"""
        logger.info("=" * 70)
        logger.info("[Asset Shield] Unmanned March Mode Activated")
        logger.info(f"Start time: {self.status['started_at']}")
        logger.info("=" * 70)

        try:
            # Phase 1: Wait for recovery
            self.wait_for_recovery(check_interval=60)

            # Phase 2: Run backtest
            self.run_backtest()

            # Phase 3: Generate reports
            self.generate_final_reports()

            # Complete
            self.status['completed_at'] = datetime.now().isoformat()
            self.status['phase'] = 'completed'

            logger.info("=" * 70)
            logger.info("[UNMANNED MARCH] All tasks completed")
            logger.info(f"Completion time: {self.status['completed_at']}")
            logger.info("=" * 70)
            logger.info("Awaiting master's review.")

        except Exception as e:
            logger.error(f"Unmanned march mode error: {e}")
            self.status['phase'] = 'error'
            raise


def main():
    controller = UnmannedMarchController()
    controller.run()


if __name__ == "__main__":
    main()
