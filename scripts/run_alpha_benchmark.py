#!/usr/bin/env python3
"""
Alpha Model Benchmark Runner
GPT 5.2 Codex Audit Implementation - Final Benchmark

Executes 20-year multi-regime verification using:
- Advanced Market Impact Model (Almgren-Chriss)
- Survivorship Bias Avoidance Logic
- J-Quants Historical Data

Usage:
    # Set credentials first
    export JQUANTS_MAIL="your@email.com"
    export JQUANTS_PASSWORD="your_password"
    
    # Run benchmark with Alpha Model
    python scripts/run_alpha_benchmark.py --mode full --codes 7203,9984,6758
    
    # Compare with baseline (Brain-only strategy)
    python scripts/run_alpha_benchmark.py --mode compare --codes 7203,9984,6758
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np

from shield.jquants_backtest_provider import (
    JQuantsBacktestProvider,
    create_jquants_provider,
    DataTransformer,
    NormalizedQuote
)
from shield.backtest_framework import (
    BacktestEngine,
    MultiPhaseBacktester,
    BacktestPhase,
    PHASE_CONFIGS,
    BacktestMetrics
)
from shield.alpha_model import (
    AlphaModel,
    AlphaModelBacktestStrategy,
    MarketImpactModel,
    SurvivorshipBiasHandler,
    DelistingReason
)
from shield.brain import ShieldBrain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Benchmark result container"""
    
    def __init__(self, name: str):
        self.name = name
        self.phases: Dict[str, Dict] = {}
        self.execution_time_seconds: float = 0
        self.total_trades: int = 0
        self.aggregate_metrics: Dict[str, float] = {}
        
    def add_phase_result(self, phase: str, metrics: BacktestMetrics, passed: bool):
        self.phases[phase] = {
            "passed": passed,
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor
        }
        self.total_trades += metrics.total_trades
    
    def calculate_aggregate(self):
        if not self.phases:
            return
            
        returns = [p["annualized_return"] for p in self.phases.values()]
        sharpes = [p["sharpe_ratio"] for p in self.phases.values()]
        drawdowns = [p["max_drawdown"] for p in self.phases.values()]
        
        self.aggregate_metrics = {
            "avg_annual_return": np.mean(returns),
            "avg_sharpe": np.mean(sharpes),
            "worst_drawdown": max(drawdowns),
            "consistency_score": sum(1 for p in self.phases.values() if p["passed"]) / len(self.phases),
            "total_trades": self.total_trades
        }
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "phases": self.phases,
            "aggregate": self.aggregate_metrics,
            "execution_time_seconds": self.execution_time_seconds
        }


def create_baseline_strategy(codes: List[str], training_mode: bool = True):
    """Create baseline Brain-only strategy for comparison"""
    
    class BaselineBrainStrategy:
        """Baseline strategy without Alpha Model enhancements"""
        
        def __init__(self, codes: List[str], training_mode: bool):
            self.codes = codes
            self.training_mode = training_mode
            self.brains: Dict[str, ShieldBrain] = {}
            for code in codes:
                self.brains[code] = ShieldBrain(target_id=f"baseline_{code}")
            
            self.price_history: Dict[str, List[float]] = defaultdict(list)
            self.lookback = 50
            self.max_positions = 5
            self.position_size_pct = 0.15
            self.take_profit = 0.08
            self.stop_loss = 0.03
            self.max_holding_days = 30
        
        def __call__(self, engine, current_date, prices, signals):
            # Update price history
            for code, price in prices.items():
                if code in self.codes:
                    self.price_history[code].append(price)
                    if len(self.price_history[code]) > 200:
                        self.price_history[code] = self.price_history[code][-200:]
            
            # Exit logic
            for code in list(engine.positions.keys()):
                if code not in prices:
                    continue
                
                price = prices[code]
                trade = engine.positions[code]
                trade.holding_days = (current_date - trade.entry_date).days
                pnl_pct = (price - trade.entry_price) / trade.entry_price
                
                exit_reason = None
                if pnl_pct >= self.take_profit:
                    exit_reason = "TAKE_PROFIT"
                elif pnl_pct <= -self.stop_loss:
                    exit_reason = "STOP_LOSS"
                elif trade.holding_days >= self.max_holding_days:
                    exit_reason = "TIME_EXIT"
                elif code in self.brains and len(self.price_history[code]) >= self.lookback:
                    confidence = self.brains[code].calculate_confidence(
                        self.price_history[code][-self.lookback:]
                    )
                    if confidence < -self.brains[code].get_threshold() * 0.5:
                        exit_reason = "BRAIN_EXIT"
                
                if exit_reason:
                    engine.close_position(code, price, exit_reason)
                    if self.training_mode and code in self.brains:
                        actual_pnl = (price - trade.entry_price) * trade.quantity
                        self.brains[code].record_trade_result(actual_pnl)
            
            # Entry logic
            if len(engine.positions) >= self.max_positions:
                return
            
            for code in self.codes:
                if code in engine.positions or code not in prices:
                    continue
                if len(self.price_history[code]) < self.lookback:
                    continue
                
                price = prices[code]
                brain = self.brains[code]
                confidence = brain.calculate_confidence(
                    self.price_history[code][-self.lookback:]
                )
                
                if confidence > brain.get_threshold():
                    position_value = engine.cash * self.position_size_pct
                    quantity = int(position_value / price)
                    
                    if quantity > 0 and position_value < engine.cash * 0.95:
                        engine.open_position(code, price, quantity)
    
    return BaselineBrainStrategy(codes, training_mode)


def run_benchmark(
    provider: JQuantsBacktestProvider,
    codes: List[str],
    strategy_name: str,
    strategy_factory,
    initial_capital: float = 100_000_000
) -> BenchmarkResult:
    """
    Run benchmark for a strategy.
    
    Args:
        provider: J-Quants data provider
        codes: Stock codes to trade
        strategy_name: Name for the benchmark
        strategy_factory: Function to create strategy
        initial_capital: Starting capital
        
    Returns:
        BenchmarkResult
    """
    result = BenchmarkResult(strategy_name)
    start_time = time.time()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Benchmark: {strategy_name}")
    logger.info(f"{'='*60}")
    
    # Create data provider
    data_provider = provider.create_backtest_data_provider(codes, use_adjusted=True)
    
    # Create strategy
    strategy = strategy_factory(codes, training_mode=True)
    
    # Initialize backtester
    backtester = MultiPhaseBacktester(
        strategy=strategy,
        data_provider=data_provider,
        initial_capital=initial_capital
    )
    
    # Run all phases
    for phase in BacktestPhase:
        config = PHASE_CONFIGS[phase]
        logger.info(f"\nPhase: {phase.value}")
        logger.info(f"Period: {config.start_date} to {config.end_date}")
        
        try:
            phase_result = backtester.run_phase(phase)
            result.add_phase_result(
                phase.value,
                phase_result.metrics,
                phase_result.passed
            )
            
            logger.info(f"  Return: {phase_result.metrics.total_return:.2%}")
            logger.info(f"  Sharpe: {phase_result.metrics.sharpe_ratio:.2f}")
            logger.info(f"  Max DD: {phase_result.metrics.max_drawdown:.2%}")
            logger.info(f"  Trades: {phase_result.metrics.total_trades}")
            logger.info(f"  Passed: {'✓' if phase_result.passed else '✗'}")
            
        except Exception as e:
            logger.error(f"Phase {phase.value} failed: {e}")
            result.phases[phase.value] = {"error": str(e)}
    
    result.execution_time_seconds = time.time() - start_time
    result.calculate_aggregate()
    
    return result


def run_comparison_benchmark(
    provider: JQuantsBacktestProvider,
    codes: List[str],
    initial_capital: float = 100_000_000
) -> Dict[str, Any]:
    """
    Run comparison benchmark between baseline and Alpha Model strategies.
    
    Args:
        provider: J-Quants data provider
        codes: Stock codes to trade
        initial_capital: Starting capital
        
    Returns:
        Comparison results
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARISON BENCHMARK")
    logger.info("Baseline (Brain-only) vs Alpha Model (Impact + Bias)")
    logger.info("="*60)
    
    # Run baseline benchmark
    baseline_result = run_benchmark(
        provider=provider,
        codes=codes,
        strategy_name="Baseline (Brain-only)",
        strategy_factory=create_baseline_strategy,
        initial_capital=initial_capital
    )
    
    # Run Alpha Model benchmark
    alpha_result = run_benchmark(
        provider=provider,
        codes=codes,
        strategy_name="Alpha Model (Impact + Bias)",
        strategy_factory=lambda c, t: AlphaModelBacktestStrategy(c, training_mode=t),
        initial_capital=initial_capital
    )
    
    # Calculate comparison metrics
    comparison = {
        "benchmark_id": f"COMPARE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "stocks": codes,
        "initial_capital": initial_capital,
        "baseline": baseline_result.to_dict(),
        "alpha_model": alpha_result.to_dict(),
        "comparison": {}
    }
    
    # Compare aggregate metrics
    if baseline_result.aggregate_metrics and alpha_result.aggregate_metrics:
        baseline_agg = baseline_result.aggregate_metrics
        alpha_agg = alpha_result.aggregate_metrics
        
        comparison["comparison"] = {
            "return_improvement": alpha_agg["avg_annual_return"] - baseline_agg["avg_annual_return"],
            "sharpe_improvement": alpha_agg["avg_sharpe"] - baseline_agg["avg_sharpe"],
            "drawdown_improvement": baseline_agg["worst_drawdown"] - alpha_agg["worst_drawdown"],
            "consistency_improvement": alpha_agg["consistency_score"] - baseline_agg["consistency_score"],
            "execution_time_ratio": alpha_result.execution_time_seconds / max(baseline_result.execution_time_seconds, 0.001)
        }
        
        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"Return Improvement: {comparison['comparison']['return_improvement']:.2%}")
        logger.info(f"Sharpe Improvement: {comparison['comparison']['sharpe_improvement']:.2f}")
        logger.info(f"Drawdown Improvement: {comparison['comparison']['drawdown_improvement']:.2%}")
        logger.info(f"Consistency Improvement: {comparison['comparison']['consistency_improvement']:.2%}")
        logger.info(f"Execution Time Ratio: {comparison['comparison']['execution_time_ratio']:.2f}x")
        
        # Performance degradation check
        if comparison['comparison']['return_improvement'] < -0.01:
            logger.warning("⚠️ PERFORMANCE DEGRADATION DETECTED: Return decreased")
        if comparison['comparison']['sharpe_improvement'] < -0.1:
            logger.warning("⚠️ PERFORMANCE DEGRADATION DETECTED: Sharpe decreased")
        
        if (comparison['comparison']['return_improvement'] >= -0.01 and 
            comparison['comparison']['sharpe_improvement'] >= -0.1):
            logger.info("✓ No significant performance degradation detected")
    
    return comparison


def run_full_benchmark(
    provider: JQuantsBacktestProvider,
    codes: List[str],
    initial_capital: float = 100_000_000
) -> Dict[str, Any]:
    """
    Run full benchmark with Alpha Model only.
    
    Args:
        provider: J-Quants data provider
        codes: Stock codes to trade
        initial_capital: Starting capital
        
    Returns:
        Benchmark results
    """
    logger.info("\n" + "="*60)
    logger.info("FULL ALPHA MODEL BENCHMARK")
    logger.info("20-Year Multi-Regime Verification")
    logger.info("="*60)
    
    result = run_benchmark(
        provider=provider,
        codes=codes,
        strategy_name="Alpha Model (Full)",
        strategy_factory=lambda c, t: AlphaModelBacktestStrategy(c, training_mode=t),
        initial_capital=initial_capital
    )
    
    # Get Alpha Model audit report
    alpha_model = AlphaModel()
    audit_report = alpha_model.get_audit_report()
    
    summary = {
        "benchmark_id": f"FULL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "stocks": codes,
        "initial_capital": initial_capital,
        "results": result.to_dict(),
        "alpha_model_audit": audit_report,
        "verification_status": "PASSED" if result.aggregate_metrics.get("consistency_score", 0) >= 0.75 else "NEEDS_REVIEW"
    }
    
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*60)
    logger.info(f"Average Annual Return: {result.aggregate_metrics.get('avg_annual_return', 0):.2%}")
    logger.info(f"Average Sharpe Ratio: {result.aggregate_metrics.get('avg_sharpe', 0):.2f}")
    logger.info(f"Worst Drawdown: {result.aggregate_metrics.get('worst_drawdown', 0):.2%}")
    logger.info(f"Consistency Score: {result.aggregate_metrics.get('consistency_score', 0):.0%}")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Execution Time: {result.execution_time_seconds:.1f}s")
    logger.info(f"Verification Status: {summary['verification_status']}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Alpha Model benchmark with J-Quants data"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "compare", "quick"],
        default="full",
        help="Benchmark mode: full (Alpha only), compare (vs baseline), quick (single phase)"
    )
    parser.add_argument(
        "--plan",
        choices=["free", "light", "standard", "premium"],
        default="free",
        help="J-Quants subscription plan"
    )
    parser.add_argument(
        "--codes",
        type=str,
        default="7203,9984,6758",
        help="Comma-separated stock codes"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000_000,
        help="Initial capital in JPY"
    )
    parser.add_argument(
        "--credential-storage",
        choices=["env", "file", "aws_secrets"],
        default="env",
        help="Credential storage type"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: output/benchmark_*.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check setup without running benchmark"
    )
    
    args = parser.parse_args()
    
    # Parse stock codes
    codes = [c.strip() for c in args.codes.split(",")]
    
    # Create provider
    logger.info("Initializing J-Quants provider...")
    provider = create_jquants_provider(
        plan=args.plan,
        credential_storage=args.credential_storage
    )
    
    if args.dry_run:
        logger.info("\n=== DRY RUN - Checking Setup ===")
        status = provider.get_status()
        
        logger.info(f"\nProvider Configuration:")
        logger.info(f"  Plan: {status['plan']}")
        logger.info(f"  Cache: {status['cache']['total_quotes']} quotes")
        
        logger.info(f"\nBenchmark Configuration:")
        logger.info(f"  Mode: {args.mode}")
        logger.info(f"  Stocks: {codes}")
        logger.info(f"  Capital: ¥{args.capital:,.0f}")
        
        # Test Alpha Model
        logger.info("\nTesting Alpha Model...")
        alpha_model = AlphaModel()
        alpha_model.set_market_data("7203", adv=5_000_000, volatility=0.25)
        
        impact = alpha_model.impact_model.estimate_total_impact(
            "7203", 10000, 2500, "BUY", "NORMAL"
        )
        logger.info(f"  Impact Model: OK (test impact = {impact.total_impact_bps:.1f}bps)")
        
        logger.info("\nSetup OK. Remove --dry-run to run benchmark.")
        return
    
    # Run benchmark
    if args.mode == "compare":
        results = run_comparison_benchmark(
            provider=provider,
            codes=codes,
            initial_capital=args.capital
        )
    elif args.mode == "quick":
        # Quick mode: single phase only
        logger.info("Quick mode: Running Phase 4 (Modern) only")
        result = run_benchmark(
            provider=provider,
            codes=codes,
            strategy_name="Alpha Model (Quick)",
            strategy_factory=lambda c, t: AlphaModelBacktestStrategy(c, training_mode=t),
            initial_capital=args.capital
        )
        results = {
            "benchmark_id": f"QUICK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "mode": "quick",
            "results": result.to_dict()
        }
    else:  # full
        results = run_full_benchmark(
            provider=provider,
            codes=codes,
            initial_capital=args.capital
        )
    
    # Save results
    output_path = args.output or f"output/benchmark_{results.get('benchmark_id', 'unknown')}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Final status
    if args.mode == "compare":
        improvement = results.get("comparison", {}).get("return_improvement", 0)
        if improvement >= 0:
            logger.info("\n✓ Alpha Model shows improvement over baseline")
        else:
            logger.warning("\n⚠️ Alpha Model shows degradation - review required")
    elif args.mode == "full":
        status = results.get("verification_status", "UNKNOWN")
        if status == "PASSED":
            logger.info("\n✓ Strategy VERIFIED - Ready for production")
        else:
            logger.warning(f"\n⚠️ Verification status: {status}")


if __name__ == "__main__":
    main()
