"""
Walk-Forward Validation for Asset Shield Phase 2
IS/OOS Separation Framework for Overfitting Prevention

Implements:
- Strict In-Sample / Out-of-Sample separation
- Training period learning rules
- Overfitting detection metrics
- Parameter freezing after validation

Period Definition:
- Training: 2007-2015 (Parameter optimization, learning allowed)
- Validation: 2016-2020 (Verification, parameters frozen)
- Out-of-Sample: 2021-2026 (Final test, complete freeze)

Author: Asset Shield V2 Team
Version: 2.0.0 (2026-02-03)
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class ValidationPhase(Enum):
    """Walk-forward validation phases"""
    TRAINING = "training"           # 2007-2015: Learning allowed
    VALIDATION = "validation"       # 2016-2020: Frozen parameters
    OUT_OF_SAMPLE = "out_of_sample" # 2021-2026: Final test


@dataclass
class PhasePeriod:
    """Period definition for each validation phase"""
    phase: ValidationPhase
    start_date: date
    end_date: date
    description: str
    learning_allowed: bool

    def contains(self, check_date: date) -> bool:
        """Check if date falls within this period"""
        return self.start_date <= check_date <= self.end_date


@dataclass
class TradeRecord:
    """Individual trade record for analysis"""
    trade_id: str
    code: str
    entry_date: date
    exit_date: Optional[date]
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    pnl_pct: float
    is_winner: bool
    phase: ValidationPhase

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['entry_date'] = str(self.entry_date)
        result['exit_date'] = str(self.exit_date) if self.exit_date else None
        result['phase'] = self.phase.value
        return result


@dataclass
class PhaseMetrics:
    """Performance metrics for a validation phase"""
    phase: ValidationPhase
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_holding_days: float

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['phase'] = self.phase.value
        return result


@dataclass
class OverfittingAnalysis:
    """Overfitting detection analysis result"""
    overfitting_ratio: float        # validation_sharpe / training_sharpe
    degradation_ratio: float        # oos_sharpe / validation_sharpe
    is_overfitted: bool
    passed_validation: bool
    training_metrics: Optional[PhaseMetrics]
    validation_metrics: Optional[PhaseMetrics]
    oos_metrics: Optional[PhaseMetrics]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "overfitting_ratio": round(self.overfitting_ratio, 4),
            "degradation_ratio": round(self.degradation_ratio, 4),
            "is_overfitted": self.is_overfitted,
            "passed_validation": self.passed_validation,
            "training_metrics": self.training_metrics.to_dict() if self.training_metrics else None,
            "validation_metrics": self.validation_metrics.to_dict() if self.validation_metrics else None,
            "oos_metrics": self.oos_metrics.to_dict() if self.oos_metrics else None,
            "recommendations": self.recommendations
        }


@dataclass
class LearningState:
    """Adaptive learning state (training phase only)"""
    entry_threshold: float = 0.65
    risk_penalty: float = 1.5
    position_scale: float = 1.0
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.08
    lookback_period: int = 50

    # Learning rate parameters
    loss_learning_rate: float = 0.10    # Fast learning from losses
    win_learning_rate: float = 0.01     # Slow learning from wins

    def to_dict(self) -> Dict:
        return asdict(self)

    def freeze(self) -> Dict:
        """Return frozen copy of parameters"""
        return {
            "entry_threshold": self.entry_threshold,
            "risk_penalty": self.risk_penalty,
            "position_scale": self.position_scale,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "lookback_period": self.lookback_period
        }


# =============================================================================
# Phase Definitions
# =============================================================================

PHASE_DEFINITIONS = {
    ValidationPhase.TRAINING: PhasePeriod(
        phase=ValidationPhase.TRAINING,
        start_date=date(2007, 1, 1),
        end_date=date(2015, 12, 31),
        description="Training Phase: Parameter optimization and learning allowed",
        learning_allowed=True
    ),
    ValidationPhase.VALIDATION: PhasePeriod(
        phase=ValidationPhase.VALIDATION,
        start_date=date(2016, 1, 1),
        end_date=date(2020, 12, 31),
        description="Validation Phase: Parameters frozen, strategy verification",
        learning_allowed=False
    ),
    ValidationPhase.OUT_OF_SAMPLE: PhasePeriod(
        phase=ValidationPhase.OUT_OF_SAMPLE,
        start_date=date(2021, 1, 1),
        end_date=date(2026, 12, 31),
        description="Out-of-Sample Phase: Final test, complete parameter freeze",
        learning_allowed=False
    )
}


# =============================================================================
# Walk-Forward Validator
# =============================================================================

class WalkForwardValidator:
    """
    Walk-Forward Validation Framework

    Implements strict IS/OOS separation to prevent overfitting:
    - Training (2007-2015): Parameters can be optimized
    - Validation (2016-2020): Parameters frozen, strategy verified
    - Out-of-Sample (2021-2026): Final unbiased test

    Overfitting Detection:
    - overfitting_ratio = validation_sharpe / training_sharpe > 0.7
    - degradation_ratio = oos_sharpe / validation_sharpe > 0.7
    """

    # Overfitting detection thresholds
    MIN_OVERFITTING_RATIO = 0.70    # Validation/Training Sharpe
    MIN_DEGRADATION_RATIO = 0.70    # OOS/Validation Sharpe
    MIN_SHARPE_THRESHOLD = 0.50     # Minimum acceptable Sharpe

    # Learning boundaries
    MAX_ENTRY_THRESHOLD = 0.90
    MIN_ENTRY_THRESHOLD = 0.50
    MAX_RISK_PENALTY = 3.0
    MIN_RISK_PENALTY = 1.0

    # Trade window for learning
    LEARNING_WINDOW = 20            # Trades for win rate calculation

    def __init__(self):
        """Initialize Walk-Forward Validator"""
        self.phase_definitions = PHASE_DEFINITIONS

        # Trade tracking by phase
        self._trades: Dict[ValidationPhase, List[TradeRecord]] = {
            phase: [] for phase in ValidationPhase
        }

        # Learning state (training phase only)
        self.learning_state = LearningState()
        self._frozen_params: Optional[Dict] = None

        # Recent trade window for learning
        self._recent_trades: deque = deque(maxlen=self.LEARNING_WINDOW)

        # Phase metrics cache
        self._phase_metrics: Dict[ValidationPhase, PhaseMetrics] = {}

        logger.info("WalkForwardValidator initialized with strict IS/OOS separation")

    # -------------------------------------------------------------------------
    # Phase Detection
    # -------------------------------------------------------------------------

    def get_current_phase(self, current_date: date) -> ValidationPhase:
        """
        Determine which validation phase a date belongs to.

        Args:
            current_date: Date to check

        Returns:
            ValidationPhase enum
        """
        for phase, period in self.phase_definitions.items():
            if period.contains(current_date):
                return phase

        # Default to OOS if beyond defined periods
        if current_date > date(2026, 12, 31):
            return ValidationPhase.OUT_OF_SAMPLE

        # Before training period
        return ValidationPhase.TRAINING

    def is_learning_allowed(self, current_date: date) -> bool:
        """Check if learning is allowed for the current date"""
        phase = self.get_current_phase(current_date)
        return self.phase_definitions[phase].learning_allowed

    def get_phase_info(self, current_date: date) -> Dict[str, Any]:
        """Get detailed information about current phase"""
        phase = self.get_current_phase(current_date)
        period = self.phase_definitions[phase]

        return {
            "phase": phase.value,
            "description": period.description,
            "start_date": str(period.start_date),
            "end_date": str(period.end_date),
            "learning_allowed": period.learning_allowed,
            "current_date": str(current_date)
        }

    # -------------------------------------------------------------------------
    # Trade Recording
    # -------------------------------------------------------------------------

    def record_trade(
        self,
        trade_id: str,
        code: str,
        entry_date: date,
        exit_date: date,
        entry_price: float,
        exit_price: float,
        quantity: int = 1
    ) -> TradeRecord:
        """
        Record a completed trade.

        Args:
            trade_id: Unique trade identifier
            code: Stock code
            entry_date: Entry date
            exit_date: Exit date
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of shares

        Returns:
            TradeRecord with calculated PnL
        """
        # Calculate PnL
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price

        # Determine phase based on exit date
        phase = self.get_current_phase(exit_date)

        record = TradeRecord(
            trade_id=trade_id,
            code=code,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_winner=pnl > 0,
            phase=phase
        )

        # Store by phase
        self._trades[phase].append(record)

        # Track recent trades for learning
        self._recent_trades.append(record)

        # Apply learning if in training phase
        if phase == ValidationPhase.TRAINING:
            self._apply_learning(record)

        return record

    # -------------------------------------------------------------------------
    # Learning Rules (Training Phase Only)
    # -------------------------------------------------------------------------

    def _apply_learning(self, trade: TradeRecord) -> None:
        """
        Apply learning from trade result (Training phase only).

        Learning Rules:
        - Loss (win_rate < 40%): Fast learning - tighten entry, increase risk penalty
        - Win (win_rate > 60%): Slow learning - slightly relax entry
        """
        if len(self._recent_trades) < 10:
            return  # Need minimum trades for learning

        # Calculate recent win rate
        recent_wins = sum(1 for t in self._recent_trades if t.is_winner)
        win_rate = recent_wins / len(self._recent_trades)

        if win_rate < 0.40:
            # Fast learning from losses - tighten parameters
            self.learning_state.entry_threshold = min(
                self.MAX_ENTRY_THRESHOLD,
                self.learning_state.entry_threshold + self.learning_state.loss_learning_rate
            )
            self.learning_state.risk_penalty = min(
                self.MAX_RISK_PENALTY,
                self.learning_state.risk_penalty + 0.30
            )
            logger.debug(
                f"Learning (loss pattern): threshold={self.learning_state.entry_threshold:.2f}, "
                f"risk_penalty={self.learning_state.risk_penalty:.2f}"
            )

        elif win_rate > 0.60:
            # Slow learning from wins - slightly relax parameters
            self.learning_state.entry_threshold = max(
                self.MIN_ENTRY_THRESHOLD,
                self.learning_state.entry_threshold - self.learning_state.win_learning_rate
            )
            # Risk penalty decreases slowly
            self.learning_state.risk_penalty = max(
                self.MIN_RISK_PENALTY,
                self.learning_state.risk_penalty - 0.05
            )
            logger.debug(
                f"Learning (win pattern): threshold={self.learning_state.entry_threshold:.2f}, "
                f"risk_penalty={self.learning_state.risk_penalty:.2f}"
            )

    def apply_learning_manual(
        self,
        win_rate_20trades: float,
        current_date: date
    ) -> Dict[str, float]:
        """
        Manually apply learning rule (for external integration).

        Args:
            win_rate_20trades: Win rate over last 20 trades
            current_date: Current date (learning only if training phase)

        Returns:
            Updated learning parameters
        """
        if not self.is_learning_allowed(current_date):
            logger.warning(f"Learning not allowed in {self.get_current_phase(current_date).value} phase")
            return self.learning_state.freeze()

        if win_rate_20trades < 0.40:
            self.learning_state.entry_threshold = min(
                self.MAX_ENTRY_THRESHOLD,
                self.learning_state.entry_threshold + 0.10
            )
            self.learning_state.risk_penalty += 0.30
        elif win_rate_20trades > 0.60:
            self.learning_state.entry_threshold = max(
                self.MIN_ENTRY_THRESHOLD,
                self.learning_state.entry_threshold - 0.01
            )

        return self.learning_state.freeze()

    def freeze_parameters(self) -> Dict:
        """
        Freeze current learning state parameters.

        Called at end of training phase.

        Returns:
            Frozen parameter dictionary
        """
        self._frozen_params = self.learning_state.freeze()
        logger.info(f"Parameters frozen: {self._frozen_params}")
        return self._frozen_params

    def get_parameters(self, current_date: date) -> Dict:
        """
        Get parameters appropriate for current phase.

        Training: Returns current learning state
        Validation/OOS: Returns frozen parameters

        Args:
            current_date: Current date

        Returns:
            Parameter dictionary
        """
        phase = self.get_current_phase(current_date)

        if phase == ValidationPhase.TRAINING:
            return self.learning_state.freeze()

        # For validation/OOS, use frozen params
        if self._frozen_params is None:
            logger.warning("Parameters not frozen, using current state")
            return self.learning_state.freeze()

        return self._frozen_params

    # -------------------------------------------------------------------------
    # Metrics Calculation
    # -------------------------------------------------------------------------

    def calculate_phase_metrics(
        self,
        phase: ValidationPhase,
        equity_curve: Optional[List[float]] = None
    ) -> PhaseMetrics:
        """
        Calculate performance metrics for a validation phase.

        Args:
            phase: Phase to analyze
            equity_curve: Optional equity curve for return calculations

        Returns:
            PhaseMetrics with all performance measures
        """
        trades = self._trades[phase]

        if not trades:
            return PhaseMetrics(
                phase=phase,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_holding_days=0.0
            )

        # Trade-based metrics
        pnl_pcts = [t.pnl_pct for t in trades]
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        win_rate = len(winners) / len(trades) if trades else 0

        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calculate holding days
        holding_days = []
        for t in trades:
            if t.exit_date:
                days = (t.exit_date - t.entry_date).days
                holding_days.append(days)
        avg_holding = np.mean(holding_days) if holding_days else 0

        # Return metrics
        total_return = np.prod([1 + p for p in pnl_pcts]) - 1
        period = self.phase_definitions[phase]
        years = (period.end_date - period.start_date).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk metrics
        if len(pnl_pcts) > 1:
            # Assume daily compounding for Sharpe calculation
            returns_std = np.std(pnl_pcts)
            sharpe = (np.mean(pnl_pcts) / returns_std * np.sqrt(252)) if returns_std > 0 else 0

            # Sortino (downside deviation)
            negative_returns = [r for r in pnl_pcts if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                sortino = (np.mean(pnl_pcts) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            else:
                sortino = sharpe * 1.5  # No negative returns
        else:
            sharpe = 0
            sortino = 0

        # Max drawdown from equity curve or estimate from trades
        max_dd = self._estimate_max_drawdown(pnl_pcts)

        metrics = PhaseMetrics(
            phase=phase,
            total_return=round(total_return, 4),
            annualized_return=round(annualized_return, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 4),
            total_trades=len(trades),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            avg_holding_days=round(avg_holding, 1)
        )

        self._phase_metrics[phase] = metrics
        return metrics

    def _estimate_max_drawdown(self, returns: List[float]) -> float:
        """Estimate max drawdown from returns series"""
        if not returns:
            return 0.0

        cumulative = np.cumprod([1 + r for r in returns])
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak

        return float(np.max(drawdown))

    # -------------------------------------------------------------------------
    # Overfitting Detection
    # -------------------------------------------------------------------------

    def analyze_overfitting(self) -> OverfittingAnalysis:
        """
        Analyze potential overfitting across phases.

        Checks:
        - overfitting_ratio = validation_sharpe / training_sharpe > 0.7
        - degradation_ratio = oos_sharpe / validation_sharpe > 0.7

        Returns:
            OverfittingAnalysis with detailed results
        """
        # Calculate metrics for each phase
        training_metrics = self.calculate_phase_metrics(ValidationPhase.TRAINING)
        validation_metrics = self.calculate_phase_metrics(ValidationPhase.VALIDATION)
        oos_metrics = self.calculate_phase_metrics(ValidationPhase.OUT_OF_SAMPLE)

        recommendations = []

        # Calculate overfitting ratio
        if training_metrics.sharpe_ratio > 0:
            overfitting_ratio = validation_metrics.sharpe_ratio / training_metrics.sharpe_ratio
        else:
            overfitting_ratio = 1.0 if validation_metrics.sharpe_ratio >= 0 else 0.0

        # Calculate degradation ratio
        if validation_metrics.sharpe_ratio > 0:
            degradation_ratio = oos_metrics.sharpe_ratio / validation_metrics.sharpe_ratio
        else:
            degradation_ratio = 1.0 if oos_metrics.sharpe_ratio >= 0 else 0.0

        # Determine if overfitted
        is_overfitted = overfitting_ratio < self.MIN_OVERFITTING_RATIO

        # Determine if passed validation
        passed_validation = (
            overfitting_ratio >= self.MIN_OVERFITTING_RATIO and
            degradation_ratio >= self.MIN_DEGRADATION_RATIO and
            oos_metrics.sharpe_ratio >= self.MIN_SHARPE_THRESHOLD
        )

        # Generate recommendations
        if is_overfitted:
            recommendations.append(
                f"Overfitting detected: Validation Sharpe ({validation_metrics.sharpe_ratio:.2f}) "
                f"is {(1-overfitting_ratio)*100:.0f}% lower than Training ({training_metrics.sharpe_ratio:.2f})"
            )
            recommendations.append("Consider: Simplify strategy, increase regularization, or expand training data")

        if degradation_ratio < self.MIN_DEGRADATION_RATIO:
            recommendations.append(
                f"OOS degradation: OOS Sharpe ({oos_metrics.sharpe_ratio:.2f}) "
                f"is {(1-degradation_ratio)*100:.0f}% lower than Validation ({validation_metrics.sharpe_ratio:.2f})"
            )
            recommendations.append("Consider: Review market regime changes, check for look-ahead bias")

        if oos_metrics.sharpe_ratio < self.MIN_SHARPE_THRESHOLD:
            recommendations.append(
                f"OOS Sharpe below threshold: {oos_metrics.sharpe_ratio:.2f} < {self.MIN_SHARPE_THRESHOLD}"
            )

        if not recommendations:
            recommendations.append("Strategy passes all validation checks")

        return OverfittingAnalysis(
            overfitting_ratio=overfitting_ratio,
            degradation_ratio=degradation_ratio,
            is_overfitted=is_overfitted,
            passed_validation=passed_validation,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            oos_metrics=oos_metrics,
            recommendations=recommendations
        )

    # -------------------------------------------------------------------------
    # Summary & Reports
    # -------------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        analysis = self.analyze_overfitting()

        return {
            "training_trades": len(self._trades[ValidationPhase.TRAINING]),
            "validation_trades": len(self._trades[ValidationPhase.VALIDATION]),
            "oos_trades": len(self._trades[ValidationPhase.OUT_OF_SAMPLE]),
            "overfitting_ratio": f"{analysis.overfitting_ratio:.2f}",
            "degradation_ratio": f"{analysis.degradation_ratio:.2f}",
            "is_overfitted": analysis.is_overfitted,
            "passed_validation": analysis.passed_validation,
            "learning_state": self.learning_state.to_dict(),
            "frozen_params": self._frozen_params
        }

    def print_report(self) -> None:
        """Print detailed validation report"""
        analysis = self.analyze_overfitting()

        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION REPORT")
        print("=" * 70)

        for phase in ValidationPhase:
            metrics = self._phase_metrics.get(phase)
            if metrics:
                print(f"\n--- {phase.value.upper()} ---")
                print(f"  Total Return: {metrics.total_return:.2%}")
                print(f"  Annual Return: {metrics.annualized_return:.2%}")
                print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
                print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
                print(f"  Win Rate: {metrics.win_rate:.2%}")
                print(f"  Trades: {metrics.total_trades}")

        print("\n--- OVERFITTING ANALYSIS ---")
        print(f"  Overfitting Ratio: {analysis.overfitting_ratio:.2f} (threshold: {self.MIN_OVERFITTING_RATIO})")
        print(f"  Degradation Ratio: {analysis.degradation_ratio:.2f} (threshold: {self.MIN_DEGRADATION_RATIO})")
        print(f"  Is Overfitted: {'YES ✗' if analysis.is_overfitted else 'NO ✓'}")
        print(f"  Passed Validation: {'YES ✓' if analysis.passed_validation else 'NO ✗'}")

        print("\n--- RECOMMENDATIONS ---")
        for rec in analysis.recommendations:
            print(f"  • {rec}")

        print("\n" + "=" * 70)


# =============================================================================
# Utility Functions
# =============================================================================

def create_walk_forward_validator() -> WalkForwardValidator:
    """Factory function to create Walk-Forward Validator"""
    return WalkForwardValidator()


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Walk-Forward Validation Test - IS/OOS Separation")
    print("=" * 60)

    # Create validator
    validator = create_walk_forward_validator()

    # Test phase detection
    print("\n--- Phase Detection ---")
    test_dates = [
        date(2010, 6, 15),
        date(2015, 12, 31),
        date(2016, 1, 1),
        date(2019, 3, 15),
        date(2021, 1, 1),
        date(2024, 6, 30),
    ]

    for d in test_dates:
        info = validator.get_phase_info(d)
        print(f"  {d}: {info['phase']} (learning={info['learning_allowed']})")

    # Simulate trades across phases
    print("\n--- Simulating Trades ---")

    # Training phase trades
    np.random.seed(42)
    for i in range(50):
        entry = date(2010, 1, 1) + np.timedelta64(i * 30, 'D')
        exit = entry + np.timedelta64(30, 'D')
        entry_price = 1000 + np.random.normal(0, 50)
        # 55% win rate in training
        win = np.random.random() < 0.55
        exit_price = entry_price * (1.05 if win else 0.97)

        validator.record_trade(
            trade_id=f"TRAIN_{i}",
            code="7203",
            entry_date=entry.astype(date),
            exit_date=exit.astype(date),
            entry_price=entry_price,
            exit_price=exit_price
        )

    print(f"  Training trades recorded: {len(validator._trades[ValidationPhase.TRAINING])}")

    # Freeze parameters after training
    frozen = validator.freeze_parameters()
    print(f"  Frozen params: threshold={frozen['entry_threshold']:.2f}, risk_penalty={frozen['risk_penalty']:.2f}")

    # Validation phase trades
    for i in range(30):
        entry = date(2017, 1, 1) + np.timedelta64(i * 40, 'D')
        exit = entry + np.timedelta64(30, 'D')
        entry_price = 1100 + np.random.normal(0, 60)
        # 50% win rate in validation
        win = np.random.random() < 0.50
        exit_price = entry_price * (1.04 if win else 0.98)

        validator.record_trade(
            trade_id=f"VAL_{i}",
            code="7203",
            entry_date=entry.astype(date),
            exit_date=exit.astype(date),
            entry_price=entry_price,
            exit_price=exit_price
        )

    print(f"  Validation trades recorded: {len(validator._trades[ValidationPhase.VALIDATION])}")

    # OOS phase trades
    for i in range(20):
        entry = date(2022, 1, 1) + np.timedelta64(i * 50, 'D')
        exit = entry + np.timedelta64(30, 'D')
        entry_price = 1200 + np.random.normal(0, 70)
        # 48% win rate in OOS
        win = np.random.random() < 0.48
        exit_price = entry_price * (1.03 if win else 0.985)

        validator.record_trade(
            trade_id=f"OOS_{i}",
            code="7203",
            entry_date=entry.astype(date),
            exit_date=exit.astype(date),
            entry_price=entry_price,
            exit_price=exit_price
        )

    print(f"  OOS trades recorded: {len(validator._trades[ValidationPhase.OUT_OF_SAMPLE])}")

    # Print validation report
    validator.print_report()

    # Summary
    print("\n--- Summary ---")
    summary = validator.get_summary()
    print(f"  Passed Validation: {summary['passed_validation']}")
    print(f"  Is Overfitted: {summary['is_overfitted']}")

    print("\n" + "=" * 60)
    print("Walk-Forward Validation Test Complete")
    print("=" * 60)
