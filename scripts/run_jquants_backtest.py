#!/usr/bin/env python3
"""
J-Quants Backtest Runner with ShieldBrain Integration
Executes 20-year multi-regime verification using J-Quants data

Usage:
    # Set credentials first
    export JQUANTS_MAIL="your@email.com"
    export JQUANTS_PASSWORD="your_password"
    
    # Run training backtest (Phase 1 & 2) to tune Brain parameters
    python scripts/run_jquants_backtest.py --mode train --codes 7203,9984,6758
    
    # Run verification backtest (Phase 3 & 4) with tuned parameters
    python scripts/run_jquants_backtest.py --mode verify --codes 7203,9984,6758
    
    # Run full 4-phase backtest
    python scripts/run_jquants_backtest.py --mode full --codes 7203,9984,6758
"""

import os
import sys
import argparse
import logging
import json
from datetime import date, datetime
from typing import Dict, List, Any
from collections import defaultdict

# Add project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from shield.jquants_backtest_provider import (
    JQuantsBacktestProvider,
    create_jquants_provider
)
from shield.backtest_framework import (
    BacktestEngine,
    MultiPhaseBacktester,
    BacktestPhase,
    PHASE_CONFIGS
)
from shield.brain import ShieldBrain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrainBacktestStrategy:
    """
    Strategy wrapper that integrates ShieldBrain with the backtest framework.
    
    This class:
    - Maintains per-stock price history for Brain analysis
    - Uses Brain confidence scores for entry/exit decisions
    - Records trade results to enable Brain learning/evolution
    - Supports training mode (learning enabled) and verification mode (frozen params)
    """
    
    def __init__(self, codes: List[str], training_mode: bool = True):
        """
        Initialize Brain-based strategy.
        
        Args:
            codes: List of stock codes to trade
            training_mode: If True, Brain learns from trades. If False, parameters frozen.
        """
        self.codes = codes
        self.training_mode = training_mode
        
        # Create a Brain instance for each stock (specialized learning)
        self.brains: Dict[str, ShieldBrain] = {}
        for code in codes:
            brain_id = f"backtest_{code}"
            self.brains[code] = ShieldBrain(target_id=brain_id)
        
        # Track price history per stock for Brain analysis
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        
        # Track entry prices for PnL calculation
        self.entry_prices: Dict[str, float] = {}
        
        # Strategy parameters
        self.lookback = 50  # Days of history needed
        self.max_positions = 5
        self.position_size_pct = 0.15  # 15% of cash per position
        self.take_profit = 0.08  # 8% profit target
        self.stop_loss = 0.03  # 3% stop loss
        self.max_holding_days = 30
        
        logger.info(f"BrainBacktestStrategy initialized: {len(codes)} stocks, training={training_mode}")
    
    def __call__(
        self,
        engine: BacktestEngine,
        current_date: date,
        prices: Dict[str, float],
        signals: Dict
    ) -> None:
        """
        Execute strategy for current day.
        
        Args:
            engine: Backtest engine instance
            current_date: Current simulation date
            prices: Dict of {code: price} for current day
            signals: Additional signals (unused for now)
        """
        # Update price history
        for code, price in prices.items():
            if code in self.codes:
                self.price_history[code].append(price)
                # Keep only recent history to save memory
                if len(self.price_history[code]) > 200:
                    self.price_history[code] = self.price_history[code][-200:]
        
        # Process existing positions (exit logic)
        for code in list(engine.positions.keys()):
            if code not in prices:
                continue
            
            price = prices[code]
            trade = engine.positions[code]
            pnl_pct = (price - trade.entry_price) / trade.entry_price
            
            exit_reason = None
            
            # Take profit
            if pnl_pct >= self.take_profit:
                exit_reason = "TAKE_PROFIT"
            
            # Stop loss
            elif pnl_pct <= -self.stop_loss:
                exit_reason = "STOP_LOSS"
            
            # Time-based exit
            elif trade.holding_days >= self.max_holding_days:
                exit_reason = "TIME_EXIT"
            
            # Brain-based exit: confidence dropped significantly
            elif code in self.brains and len(self.price_history[code]) >= self.lookback:
                brain = self.brains[code]
                confidence = brain.calculate_confidence(self.price_history[code][-self.lookback:])
                threshold = brain.get_threshold()
                
                # Exit if confidence drops below negative threshold
                if confidence < -threshold * 0.5:
                    exit_reason = "BRAIN_EXIT"
            
            if exit_reason:
                engine.close_position(code, price, exit_reason)
                
                # Record result for Brain learning (training mode only)
                if self.training_mode and code in self.brains:
                    actual_pnl = (price - trade.entry_price) * trade.quantity
                    self.brains[code].record_trade_result(actual_pnl)
                    logger.debug(f"Brain learned from {code}: PnL={actual_pnl:.0f}, reason={exit_reason}")
        
        # Process potential entries (entry logic)
        if len(engine.positions) >= self.max_positions:
            return  # Max positions reached
        
        for code in self.codes:
            if code in engine.positions:
                continue  # Already have position
            
            if code not in prices:
                continue  # No price data
            
            if len(self.price_history[code]) < self.lookback:
                continue  # Not enough history
            
            price = prices[code]
            brain = self.brains[code]
            
            # Get Brain confidence
            recent_prices = self.price_history[code][-self.lookback:]
            confidence = brain.calculate_confidence(recent_prices)
            threshold = brain.get_threshold()
            
            # Entry condition: confidence above adaptive threshold
            if confidence > threshold:
                # Position sizing
                position_value = engine.cash * self.position_size_pct
                quantity = int(position_value / price)
                
                if quantity > 0 and position_value < engine.cash * 0.95:  # Keep 5% cash buffer
                    trade = engine.open_position(code, price, quantity)
                    if trade:
                        self.entry_prices[code] = price
                        logger.debug(
                            f"Brain entry {code}: price={price:.0f}, qty={quantity}, "
                            f"confidence={confidence:.3f}, threshold={threshold:.3f}"
                        )
    
    def get_brain_states(self) -> Dict[str, Dict]:
        """Get current state of all Brains for reporting."""
        states = {}
        for code, brain in self.brains.items():
            states[code] = {
                "adaptive_threshold": brain.adaptive_threshold,
                "risk_penalty": brain.risk_penalty,
                "lookback": brain.lookback
            }
        return states
    
    def save_brain_states(self, output_path: str) -> None:
        """Save Brain states to file for later verification runs."""
        states = self.get_brain_states()
        with open(output_path, 'w') as f:
            json.dump(states, f, indent=2)
        logger.info(f"Brain states saved to {output_path}")
    
    def load_brain_states(self, input_path: str) -> None:
        """Load Brain states from file (for verification mode)."""
        if not os.path.exists(input_path):
            logger.warning(f"Brain states file not found: {input_path}")
            return
        
        with open(input_path, 'r') as f:
            states = json.load(f)
        
        for code, state in states.items():
            if code in self.brains:
                self.brains[code].adaptive_threshold = state.get("adaptive_threshold", 0.65)
                self.brains[code].risk_penalty = state.get("risk_penalty", 1.5)
        
        logger.info(f"Brain states loaded from {input_path}")


def create_brain_strategy(codes: List[str], training_mode: bool = True) -> BrainBacktestStrategy:
    """
    Factory function to create Brain-based strategy.
    
    Args:
        codes: Stock codes to trade
        training_mode: Enable learning if True
        
    Returns:
        Configured BrainBacktestStrategy instance
    """
    return BrainBacktestStrategy(codes=codes, training_mode=training_mode)


def run_training_backtest(
    provider: JQuantsBacktestProvider,
    codes: List[str],
    initial_capital: float = 100_000_000
) -> Dict[str, Any]:
    """
    Run TRAINING backtest (Phase 1 & 2) to tune Brain parameters.
    
    This runs the strategy through historical crisis and expansion periods,
    allowing the Brain to learn and adapt its parameters.
    
    Args:
        provider: J-Quants data provider
        codes: List of stock codes to trade
        initial_capital: Starting capital in JPY
        
    Returns:
        Training results with tuned Brain states
    """
    logger.info("=" * 60)
    logger.info("TRAINING BACKTEST - Brain Parameter Tuning")
    logger.info("Phases: 1 (Lehman Shock) + 2 (Abenomics)")
    logger.info("=" * 60)
    
    # Check provider status
    status = provider.get_status()
    logger.info(f"Plan: {status['plan']}")
    logger.info(f"Cache: {status['cache']['total_quotes']} quotes cached")
    
    # Create data provider
    logger.info(f"\nLoading data for {len(codes)} stocks: {codes}")
    data_provider = provider.create_backtest_data_provider(codes, use_adjusted=True)
    
    # Create Brain-based strategy in TRAINING mode
    strategy = create_brain_strategy(codes, training_mode=True)
    
    # Initialize backtester
    backtester = MultiPhaseBacktester(
        strategy=strategy,
        data_provider=data_provider,
        initial_capital=initial_capital
    )
    
    # Run training phases (1 & 2)
    training_phases = [BacktestPhase.PHASE_1_SURVIVAL, BacktestPhase.PHASE_2_EXPANSION]
    results = {}
    
    for phase in training_phases:
        config = PHASE_CONFIGS[phase]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TRAINING Phase: {phase.value}")
        logger.info(f"Period: {config.start_date} to {config.end_date}")
        logger.info(f"Description: {config.description}")
        logger.info("=" * 60)
        
        try:
            result = backtester.run_phase(phase)
            
            results[phase.value] = {
                "passed": result.passed,
                "total_return": f"{result.metrics.total_return:.2%}",
                "annualized_return": f"{result.metrics.annualized_return:.2%}",
                "sharpe_ratio": f"{result.metrics.sharpe_ratio:.2f}",
                "max_drawdown": f"{result.metrics.max_drawdown:.2%}",
                "total_trades": result.metrics.total_trades,
                "win_rate": f"{result.metrics.win_rate:.2%}",
                "profit_factor": f"{result.metrics.profit_factor:.2f}",
                "notes": result.notes
            }
            
            logger.info(f"\nTraining Results:")
            logger.info(f"  Passed: {'✓' if result.passed else '✗'}")
            logger.info(f"  Total Return: {result.metrics.total_return:.2%}")
            logger.info(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
            logger.info(f"  Total Trades: {result.metrics.total_trades}")
            
        except Exception as e:
            logger.error(f"Training phase {phase.value} failed: {e}")
            results[phase.value] = {"error": str(e)}
    
    # Save tuned Brain states
    brain_states_path = "output/brain_states_trained.json"
    os.makedirs("output", exist_ok=True)
    strategy.save_brain_states(brain_states_path)
    
    # Log final Brain states
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - Brain States After Learning")
    logger.info("=" * 60)
    brain_states = strategy.get_brain_states()
    for code, state in brain_states.items():
        logger.info(f"  {code}: threshold={state['adaptive_threshold']:.3f}, risk_penalty={state['risk_penalty']:.2f}")
    
    summary = {
        "run_id": f"TRAIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "mode": "training",
        "stocks": codes,
        "initial_capital": f"¥{initial_capital:,.0f}",
        "phases": results,
        "brain_states": brain_states,
        "brain_states_file": brain_states_path
    }
    
    return summary


def run_verification_backtest(
    provider: JQuantsBacktestProvider,
    codes: List[str],
    initial_capital: float = 100_000_000,
    brain_states_path: str = "output/brain_states_trained.json"
) -> Dict[str, Any]:
    """
    Run VERIFICATION backtest (Phase 3 & 4) with frozen Brain parameters.
    
    This tests the tuned strategy on out-of-sample data to validate
    that the learned parameters generalize well.
    
    Args:
        provider: J-Quants data provider
        codes: List of stock codes to trade
        initial_capital: Starting capital in JPY
        brain_states_path: Path to trained Brain states
        
    Returns:
        Verification results
    """
    logger.info("=" * 60)
    logger.info("VERIFICATION BACKTEST - Out-of-Sample Testing")
    logger.info("Phases: 3 (COVID Shock) + 4 (Modern)")
    logger.info("=" * 60)
    
    # Check provider status
    status = provider.get_status()
    logger.info(f"Plan: {status['plan']}")
    logger.info(f"Cache: {status['cache']['total_quotes']} quotes cached")
    
    # Create data provider
    logger.info(f"\nLoading data for {len(codes)} stocks: {codes}")
    data_provider = provider.create_backtest_data_provider(codes, use_adjusted=True)
    
    # Create Brain-based strategy in VERIFICATION mode (no learning)
    strategy = create_brain_strategy(codes, training_mode=False)
    
    # Load trained Brain states
    strategy.load_brain_states(brain_states_path)
    
    # Log loaded Brain states
    logger.info("\nLoaded Brain States (Frozen):")
    brain_states = strategy.get_brain_states()
    for code, state in brain_states.items():
        logger.info(f"  {code}: threshold={state['adaptive_threshold']:.3f}, risk_penalty={state['risk_penalty']:.2f}")
    
    # Initialize backtester
    backtester = MultiPhaseBacktester(
        strategy=strategy,
        data_provider=data_provider,
        initial_capital=initial_capital
    )
    
    # Run verification phases (3 & 4)
    verification_phases = [BacktestPhase.PHASE_3_OOS, BacktestPhase.PHASE_4_MODERN]
    results = {}
    
    for phase in verification_phases:
        config = PHASE_CONFIGS[phase]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"VERIFICATION Phase: {phase.value}")
        logger.info(f"Period: {config.start_date} to {config.end_date}")
        logger.info(f"Description: {config.description}")
        logger.info("=" * 60)
        
        try:
            result = backtester.run_phase(phase)
            
            results[phase.value] = {
                "passed": result.passed,
                "total_return": f"{result.metrics.total_return:.2%}",
                "annualized_return": f"{result.metrics.annualized_return:.2%}",
                "sharpe_ratio": f"{result.metrics.sharpe_ratio:.2f}",
                "max_drawdown": f"{result.metrics.max_drawdown:.2%}",
                "total_trades": result.metrics.total_trades,
                "win_rate": f"{result.metrics.win_rate:.2%}",
                "profit_factor": f"{result.metrics.profit_factor:.2f}",
                "notes": result.notes
            }
            
            logger.info(f"\nVerification Results:")
            logger.info(f"  Passed: {'✓' if result.passed else '✗'}")
            logger.info(f"  Total Return: {result.metrics.total_return:.2%}")
            logger.info(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
            logger.info(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
            logger.info(f"  Total Trades: {result.metrics.total_trades}")
            
        except Exception as e:
            logger.error(f"Verification phase {phase.value} failed: {e}")
            results[phase.value] = {"error": str(e)}
    
    # Check if all verification phases passed
    all_passed = all(
        r.get("passed", False)
        for r in results.values()
        if "error" not in r
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"All Verification Phases Passed: {'✓' if all_passed else '✗'}")
    
    summary = {
        "run_id": f"VERIFY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "mode": "verification",
        "stocks": codes,
        "initial_capital": f"¥{initial_capital:,.0f}",
        "phases": results,
        "brain_states_used": brain_states,
        "all_passed": all_passed
    }
    
    return summary


def run_full_backtest(
    provider: JQuantsBacktestProvider,
    codes: List[str],
    initial_capital: float = 100_000_000
) -> Dict[str, Any]:
    """
    Run FULL 4-phase backtest: Training (1&2) then Verification (3&4).
    
    Args:
        provider: J-Quants data provider
        codes: List of stock codes to trade
        initial_capital: Starting capital in JPY
        
    Returns:
        Combined results from training and verification
    """
    logger.info("=" * 60)
    logger.info("FULL 4-PHASE BACKTEST")
    logger.info("Training (Phase 1&2) -> Verification (Phase 3&4)")
    logger.info("=" * 60)
    
    # Step 1: Training
    training_results = run_training_backtest(provider, codes, initial_capital)
    
    # Step 2: Verification with trained parameters
    verification_results = run_verification_backtest(
        provider, codes, initial_capital,
        brain_states_path=training_results["brain_states_file"]
    )
    
    # Combine results
    combined = {
        "run_id": f"FULL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "mode": "full",
        "stocks": codes,
        "initial_capital": f"¥{initial_capital:,.0f}",
        "training": training_results,
        "verification": verification_results,
        "final_verdict": verification_results.get("all_passed", False)
    }
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FULL BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Training Phases: {len(training_results.get('phases', {}))}")
    logger.info(f"Verification Phases: {len(verification_results.get('phases', {}))}")
    logger.info(f"Final Verdict: {'✓ PASSED' if combined['final_verdict'] else '✗ FAILED'}")
    
    # API usage
    final_status = provider.get_status()
    logger.info(f"\nAPI Usage Summary:")
    logger.info(f"  Total Calls: {final_status['api_usage']['total_calls']}")
    logger.info(f"  Cache Size: {final_status['cache']['cache_size_mb']:.2f} MB")
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Run J-Quants multi-phase backtest with ShieldBrain AI"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "verify", "full"],
        default="full",
        help="Backtest mode: train (Phase 1&2), verify (Phase 3&4), or full (all phases)"
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
        help="Comma-separated stock codes (e.g., 7203,9984,6758 for Toyota, SoftBank, Sony)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000_000,
        help="Initial capital in JPY (default: 100,000,000)"
    )
    parser.add_argument(
        "--credential-storage",
        choices=["env", "file", "aws_secrets"],
        default="env",
        help="Where to get J-Quants credentials from"
    )
    parser.add_argument(
        "--brain-states",
        type=str,
        default="output/brain_states_trained.json",
        help="Path to trained Brain states file (for verify mode)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check setup without running backtest"
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
        # Just check setup
        logger.info("\n=== DRY RUN - Checking Setup ===")
        status = provider.get_status()
        
        logger.info(f"\nProvider Configuration:")
        logger.info(f"  Plan: {status['plan']}")
        logger.info(f"  Rate Limit: {status['rate_limiter']['rpm_limit']} req/min")
        logger.info(f"  Daily Limit: {status['rate_limiter']['daily_limit']} req/day")
        
        logger.info(f"\nCache Status:")
        logger.info(f"  Cached Quotes: {status['cache']['total_quotes']}")
        logger.info(f"  Unique Stocks: {status['cache']['unique_stocks']}")
        logger.info(f"  Cache Size: {status['cache']['cache_size_mb']:.2f} MB")
        
        logger.info(f"\nCost Estimate:")
        logger.info(f"  Monthly Cost: ¥{status['cost_estimate']['estimated_cost_jpy']:,.0f}")
        logger.info(f"  Monthly Limit: {status['cost_estimate']['monthly_limit']} calls")
        logger.info(f"  Current Usage: {status['cost_estimate']['current_usage']} calls")
        
        logger.info(f"\nBacktest Configuration:")
        logger.info(f"  Mode: {args.mode}")
        logger.info(f"  Stocks: {codes}")
        logger.info(f"  Initial Capital: ¥{args.capital:,.0f}")
        
        if args.mode == "verify":
            logger.info(f"  Brain States: {args.brain_states}")
            if os.path.exists(args.brain_states):
                logger.info("    ✓ Brain states file found")
            else:
                logger.warning("    ✗ Brain states file NOT found - run training first!")
        
        logger.info("\nSetup OK. Remove --dry-run to run backtest.")
        return
    
    # Run backtest based on mode
    if args.mode == "train":
        results = run_training_backtest(
            provider=provider,
            codes=codes,
            initial_capital=args.capital
        )
    elif args.mode == "verify":
        results = run_verification_backtest(
            provider=provider,
            codes=codes,
            initial_capital=args.capital,
            brain_states_path=args.brain_states
        )
    else:  # full
        results = run_full_backtest(
            provider=provider,
            codes=codes,
            initial_capital=args.capital
        )
    
    # Save results
    output_path = f"output/backtest_{results['run_id']}.json"
    os.makedirs("output", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print final verdict for full mode
    if args.mode == "full":
        if results.get("final_verdict"):
            logger.info("\n" + "=" * 60)
            logger.info("🎉 STRATEGY VALIDATED - Ready for production deployment")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("⚠️  STRATEGY NEEDS IMPROVEMENT - Review results")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()
