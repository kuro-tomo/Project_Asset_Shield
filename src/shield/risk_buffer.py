"""
Risk Buffer for Asset Shield Phase 2
Maximum Drawdown Suppression Below 30%

Implements:
- Two-Layer Protection System:
  - Layer 1: Regime-based cash allocation
  - Layer 2: Drawdown protection with hysteresis
- Emergency position reduction
- Whipsaw prevention through hysteresis

Author: Asset Shield V2 Team
Version: 2.0.0 (2026-02-03)
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class RiskRegime(Enum):
    """Market risk regime classification"""
    CRISIS = "crisis"           # Extreme volatility, correlation spike
    HIGH_VOL = "high_vol"       # High volatility period
    NORMAL = "normal"           # Standard market conditions
    LOW_VOL = "low_vol"         # Low volatility, calm markets


class DrawdownState(Enum):
    """Drawdown protection state machine"""
    NORMAL = "normal"               # No protection active
    WARNING = "warning"             # Approaching trigger (15-20% DD)
    PROTECTION_ACTIVE = "active"    # Protection triggered (>20% DD)
    RECOVERY = "recovery"           # Recovering but still cautious


@dataclass
class RiskState:
    """Current risk buffer state"""
    regime: RiskRegime
    regime_confidence: float
    drawdown_state: DrawdownState
    current_drawdown: float
    peak_equity: float
    cash_allocation: float          # Target cash percentage
    exposure_multiplier: float      # 1.0 = full exposure, 0.5 = 50%
    protection_trigger_date: Optional[date] = None
    days_in_protection: int = 0

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "regime_confidence": self.regime_confidence,
            "drawdown_state": self.drawdown_state.value,
            "current_drawdown": round(self.current_drawdown, 4),
            "peak_equity": self.peak_equity,
            "cash_allocation": round(self.cash_allocation, 4),
            "exposure_multiplier": round(self.exposure_multiplier, 4),
            "protection_trigger_date": str(self.protection_trigger_date) if self.protection_trigger_date else None,
            "days_in_protection": self.days_in_protection
        }


@dataclass
class AllocationRecommendation:
    """Cash allocation recommendation"""
    target_cash_pct: float          # Target cash percentage (0-1)
    target_equity_pct: float        # Target equity percentage (0-1)
    regime_contribution: float      # Cash from regime (0-1)
    dd_protection_contribution: float  # Additional cash from DD protection
    action: str                     # HOLD, REDUCE, INCREASE
    urgency: str                    # LOW, MEDIUM, HIGH, EMERGENCY
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Risk Buffer
# =============================================================================

class RiskBuffer:
    """
    Two-Layer Risk Protection System

    Layer 1: Regime-Based Cash Allocation
    - Adjusts cash allocation based on market regime
    - CRISIS: 70% cash, HIGH_VOL: 50%, NORMAL: 20%, LOW_VOL: 10%

    Layer 2: Drawdown Protection
    - Triggers at 20% drawdown
    - Reduces exposure by 50% when active
    - Recovers at 10% drawdown (hysteresis)

    Combined effect targets Max DD < 30%
    """

    # Layer 1: Regime-based cash allocation
    REGIME_CASH_ALLOCATION = {
        RiskRegime.CRISIS: 0.70,    # 70% cash in crisis
        RiskRegime.HIGH_VOL: 0.50,  # 50% cash in high volatility
        RiskRegime.NORMAL: 0.20,    # 20% cash normally
        RiskRegime.LOW_VOL: 0.10    # 10% cash in calm markets
    }

    # Layer 2: Drawdown protection thresholds
    DD_TRIGGER_THRESHOLD = 0.20     # Trigger at 20% drawdown
    DD_RECOVERY_THRESHOLD = 0.10    # Recover at 10% drawdown
    DD_WARNING_THRESHOLD = 0.15     # Warning at 15% drawdown
    DD_EXPOSURE_REDUCTION = 0.50    # Reduce exposure by 50% when triggered

    # Maximum allowable cash allocation
    MAX_CASH_ALLOCATION = 0.85      # Never more than 85% cash

    # Consecutive days confirmation for regime changes
    REGIME_CONFIRMATION_DAYS = 2

    def __init__(
        self,
        initial_equity: float = 100_000_000,
        max_target_dd: float = 0.30
    ):
        """
        Initialize Risk Buffer.

        Args:
            initial_equity: Starting equity value
            max_target_dd: Maximum target drawdown (default 30%)
        """
        self.initial_equity = initial_equity
        self.max_target_dd = max_target_dd

        # State tracking
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.current_drawdown = 0.0

        # Regime tracking
        self.current_regime = RiskRegime.NORMAL
        self.regime_confidence = 0.7
        self._regime_days = 0
        self._pending_regime: Optional[RiskRegime] = None

        # Drawdown protection state
        self.dd_state = DrawdownState.NORMAL
        self.protection_trigger_date: Optional[date] = None
        self.days_in_protection = 0

        # History for analysis
        self._state_history: List[RiskState] = []

        logger.info(
            f"RiskBuffer initialized: initial_equity=¥{initial_equity:,.0f}, "
            f"target_max_dd={max_target_dd:.0%}"
        )

    # -------------------------------------------------------------------------
    # Equity & Drawdown Tracking
    # -------------------------------------------------------------------------

    def update_equity(self, equity: float, current_date: Optional[date] = None) -> float:
        """
        Update equity and recalculate drawdown.

        Args:
            equity: Current portfolio equity
            current_date: Current date (for protection tracking)

        Returns:
            Current drawdown as decimal (0.15 = 15% drawdown)
        """
        self.current_equity = equity

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate drawdown
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Update drawdown protection state
        self._update_dd_state(current_date)

        return self.current_drawdown

    def _update_dd_state(self, current_date: Optional[date] = None) -> None:
        """Update drawdown protection state machine"""
        dd = self.current_drawdown
        prev_state = self.dd_state

        if self.dd_state == DrawdownState.NORMAL:
            if dd >= self.DD_TRIGGER_THRESHOLD:
                self.dd_state = DrawdownState.PROTECTION_ACTIVE
                self.protection_trigger_date = current_date
                self.days_in_protection = 0
                logger.warning(
                    f"DRAWDOWN PROTECTION TRIGGERED: DD={dd:.1%} >= {self.DD_TRIGGER_THRESHOLD:.0%}"
                )
            elif dd >= self.DD_WARNING_THRESHOLD:
                self.dd_state = DrawdownState.WARNING
                logger.info(f"Drawdown warning: {dd:.1%}")

        elif self.dd_state == DrawdownState.WARNING:
            if dd >= self.DD_TRIGGER_THRESHOLD:
                self.dd_state = DrawdownState.PROTECTION_ACTIVE
                self.protection_trigger_date = current_date
                self.days_in_protection = 0
                logger.warning(
                    f"DRAWDOWN PROTECTION TRIGGERED: DD={dd:.1%}"
                )
            elif dd < self.DD_WARNING_THRESHOLD * 0.7:  # Clear warning below ~10.5%
                self.dd_state = DrawdownState.NORMAL

        elif self.dd_state == DrawdownState.PROTECTION_ACTIVE:
            self.days_in_protection += 1

            if dd <= self.DD_RECOVERY_THRESHOLD:
                self.dd_state = DrawdownState.RECOVERY
                logger.info(
                    f"Entering recovery phase: DD={dd:.1%} <= {self.DD_RECOVERY_THRESHOLD:.0%}"
                )

        elif self.dd_state == DrawdownState.RECOVERY:
            if dd >= self.DD_TRIGGER_THRESHOLD:
                # Back to protection if drawdown worsens
                self.dd_state = DrawdownState.PROTECTION_ACTIVE
                logger.warning(f"Re-entering protection: DD={dd:.1%}")
            elif dd <= self.DD_RECOVERY_THRESHOLD * 0.5:  # Clear recovery at 5% DD
                self.dd_state = DrawdownState.NORMAL
                self.days_in_protection = 0
                self.protection_trigger_date = None
                logger.info("Drawdown protection fully cleared")

        if prev_state != self.dd_state:
            logger.info(f"DD State transition: {prev_state.value} -> {self.dd_state.value}")

    # -------------------------------------------------------------------------
    # Regime Management
    # -------------------------------------------------------------------------

    def set_regime(
        self,
        regime: RiskRegime,
        confidence: float = 0.7,
        force: bool = False
    ) -> None:
        """
        Set market regime with confirmation delay.

        Args:
            regime: New regime classification
            confidence: Confidence in regime (0-1)
            force: Skip confirmation period
        """
        if force or regime == self.current_regime:
            self.current_regime = regime
            self.regime_confidence = confidence
            self._pending_regime = None
            self._regime_days = 0
            return

        # Track pending regime change
        if regime == self._pending_regime:
            self._regime_days += 1
            if self._regime_days >= self.REGIME_CONFIRMATION_DAYS:
                logger.info(
                    f"Regime change confirmed: {self.current_regime.value} -> {regime.value}"
                )
                self.current_regime = regime
                self.regime_confidence = confidence
                self._pending_regime = None
                self._regime_days = 0
        else:
            self._pending_regime = regime
            self._regime_days = 1

    def classify_regime_from_volatility(
        self,
        volatility: float,
        avg_correlation: float = 0.5,
        momentum: float = 0.0
    ) -> RiskRegime:
        """
        Classify regime based on market metrics.

        Args:
            volatility: Annualized volatility
            avg_correlation: Average pairwise correlation
            momentum: Market momentum indicator

        Returns:
            Classified RiskRegime
        """
        # Crisis detection: high vol + high correlation
        if volatility > 0.30 and avg_correlation > 0.65:
            return RiskRegime.CRISIS

        # High volatility
        if volatility > 0.20:
            return RiskRegime.HIGH_VOL

        # Low volatility
        if volatility < 0.12:
            return RiskRegime.LOW_VOL

        return RiskRegime.NORMAL

    # -------------------------------------------------------------------------
    # Cash Allocation
    # -------------------------------------------------------------------------

    def get_cash_allocation(
        self,
        equity: Optional[float] = None,
        regime: Optional[RiskRegime] = None
    ) -> AllocationRecommendation:
        """
        Calculate target cash allocation based on regime and DD state.

        Args:
            equity: Current equity (uses stored if None)
            regime: Override regime (uses current if None)

        Returns:
            AllocationRecommendation with target allocations
        """
        if equity is not None:
            self.update_equity(equity)

        regime = regime or self.current_regime

        # Layer 1: Regime-based allocation
        regime_cash = self.REGIME_CASH_ALLOCATION.get(regime, 0.20)

        # Layer 2: Drawdown protection addition
        dd_cash = 0.0
        dd_reason = ""

        if self.dd_state == DrawdownState.PROTECTION_ACTIVE:
            # Add 50% to cash allocation when protection active
            dd_cash = self.DD_EXPOSURE_REDUCTION
            dd_reason = f"DD Protection Active (DD={self.current_drawdown:.1%})"
        elif self.dd_state == DrawdownState.RECOVERY:
            # Partial increase during recovery
            dd_cash = self.DD_EXPOSURE_REDUCTION * 0.5
            dd_reason = f"DD Recovery Phase (DD={self.current_drawdown:.1%})"
        elif self.dd_state == DrawdownState.WARNING:
            # Small increase during warning
            dd_cash = 0.10
            dd_reason = f"DD Warning (DD={self.current_drawdown:.1%})"

        # Combined allocation (capped)
        total_cash = min(regime_cash + dd_cash, self.MAX_CASH_ALLOCATION)
        total_equity = 1.0 - total_cash

        # Determine action and urgency
        if self.dd_state == DrawdownState.PROTECTION_ACTIVE:
            action = "REDUCE"
            urgency = "EMERGENCY"
        elif self.dd_state == DrawdownState.WARNING:
            action = "REDUCE"
            urgency = "HIGH"
        elif regime == RiskRegime.CRISIS:
            action = "REDUCE"
            urgency = "HIGH"
        elif regime == RiskRegime.HIGH_VOL:
            action = "REDUCE"
            urgency = "MEDIUM"
        else:
            action = "HOLD"
            urgency = "LOW"

        # Build reason string
        reason_parts = [f"Regime: {regime.value} -> {regime_cash:.0%} cash"]
        if dd_reason:
            reason_parts.append(dd_reason)
        reason = " | ".join(reason_parts)

        return AllocationRecommendation(
            target_cash_pct=round(total_cash, 4),
            target_equity_pct=round(total_equity, 4),
            regime_contribution=round(regime_cash, 4),
            dd_protection_contribution=round(dd_cash, 4),
            action=action,
            urgency=urgency,
            reason=reason
        )

    def get_exposure_multiplier(self) -> float:
        """
        Get current exposure multiplier (0-1).

        1.0 = full exposure allowed
        0.5 = reduce exposure by 50%
        """
        allocation = self.get_cash_allocation()
        return allocation.target_equity_pct

    def get_position_size_adjustment(
        self,
        base_size: float,
        risk_score: float = 1.0
    ) -> float:
        """
        Adjust position size based on current risk state.

        Args:
            base_size: Base position size
            risk_score: Individual position risk (0-1, higher = riskier)

        Returns:
            Adjusted position size
        """
        multiplier = self.get_exposure_multiplier()

        # Additional reduction for high-risk positions in protection mode
        if self.dd_state == DrawdownState.PROTECTION_ACTIVE and risk_score > 0.7:
            multiplier *= 0.5

        return base_size * multiplier

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def get_state(self) -> RiskState:
        """Get current risk buffer state"""
        return RiskState(
            regime=self.current_regime,
            regime_confidence=self.regime_confidence,
            drawdown_state=self.dd_state,
            current_drawdown=self.current_drawdown,
            peak_equity=self.peak_equity,
            cash_allocation=self.get_cash_allocation().target_cash_pct,
            exposure_multiplier=self.get_exposure_multiplier(),
            protection_trigger_date=self.protection_trigger_date,
            days_in_protection=self.days_in_protection
        )

    def record_state(self) -> None:
        """Record current state to history"""
        self._state_history.append(self.get_state())

    def get_state_history(self) -> List[Dict]:
        """Get history of risk states"""
        return [s.to_dict() for s in self._state_history]

    def reset_peak(self, new_peak: Optional[float] = None) -> None:
        """
        Reset peak equity (use with caution).

        Args:
            new_peak: New peak value (uses current equity if None)
        """
        self.peak_equity = new_peak if new_peak else self.current_equity
        self.current_drawdown = 0.0
        self.dd_state = DrawdownState.NORMAL
        self.days_in_protection = 0
        self.protection_trigger_date = None
        logger.info(f"Peak reset to ¥{self.peak_equity:,.0f}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of risk buffer status"""
        allocation = self.get_cash_allocation()

        return {
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": f"{self.current_drawdown:.2%}",
            "regime": self.current_regime.value,
            "dd_state": self.dd_state.value,
            "target_cash": f"{allocation.target_cash_pct:.0%}",
            "target_equity": f"{allocation.target_equity_pct:.0%}",
            "action": allocation.action,
            "urgency": allocation.urgency,
            "protection_active": self.dd_state == DrawdownState.PROTECTION_ACTIVE,
            "days_in_protection": self.days_in_protection
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_risk_buffer(
    initial_equity: float = 100_000_000,
    max_dd_target: float = 0.30
) -> RiskBuffer:
    """
    Factory function to create configured Risk Buffer.

    Args:
        initial_equity: Starting equity
        max_dd_target: Maximum target drawdown

    Returns:
        Configured RiskBuffer
    """
    return RiskBuffer(
        initial_equity=initial_equity,
        max_target_dd=max_dd_target
    )


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Risk Buffer Test - Two-Layer Protection System")
    print("=" * 60)

    # Create buffer
    buffer = create_risk_buffer(initial_equity=100_000_000)

    # Simulate equity curve with drawdown
    print("\n--- Simulating Equity Curve with Drawdown ---")

    test_scenarios = [
        (100_000_000, RiskRegime.NORMAL, "Starting point"),
        (105_000_000, RiskRegime.NORMAL, "5% gain"),
        (95_000_000, RiskRegime.NORMAL, "Back to -10% from peak"),
        (110_000_000, RiskRegime.LOW_VOL, "New peak +10%"),
        (99_000_000, RiskRegime.NORMAL, "10% drawdown from peak"),
        (94_000_000, RiskRegime.HIGH_VOL, "14.5% drawdown"),
        (88_000_000, RiskRegime.HIGH_VOL, "20% drawdown - TRIGGER"),
        (77_000_000, RiskRegime.CRISIS, "30% drawdown - CRISIS"),
        (85_000_000, RiskRegime.HIGH_VOL, "Recovery to 22.7% DD"),
        (99_000_000, RiskRegime.NORMAL, "Recovery to 10% DD - EXIT PROTECTION"),
        (105_000_000, RiskRegime.NORMAL, "Back near peak"),
        (115_000_000, RiskRegime.LOW_VOL, "New peak"),
    ]

    for equity, regime, description in test_scenarios:
        buffer.set_regime(regime, force=True)
        buffer.update_equity(equity)
        allocation = buffer.get_cash_allocation()

        print(f"\n{description}:")
        print(f"  Equity: ¥{equity:,.0f}")
        print(f"  Drawdown: {buffer.current_drawdown:.1%}")
        print(f"  Regime: {regime.value}")
        print(f"  DD State: {buffer.dd_state.value}")
        print(f"  Target Cash: {allocation.target_cash_pct:.0%} ({allocation.regime_contribution:.0%} regime + {allocation.dd_protection_contribution:.0%} DD)")
        print(f"  Action: {allocation.action} ({allocation.urgency})")
        buffer.record_state()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    summary = buffer.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Allocation table test
    print("\n--- Allocation Table by Regime ---")
    print(f"{'Regime':<12} {'Base Cash':<12} {'With 20% DD':<12}")
    print("-" * 36)

    test_buffer = create_risk_buffer()
    for regime in RiskRegime:
        test_buffer.set_regime(regime, force=True)

        # Normal state
        test_buffer.dd_state = DrawdownState.NORMAL
        test_buffer.current_drawdown = 0.05
        normal_alloc = test_buffer.get_cash_allocation()

        # With drawdown protection
        test_buffer.dd_state = DrawdownState.PROTECTION_ACTIVE
        test_buffer.current_drawdown = 0.22
        dd_alloc = test_buffer.get_cash_allocation()

        print(f"{regime.value:<12} {normal_alloc.target_cash_pct:.0%}{'':>8} {dd_alloc.target_cash_pct:.0%}")

    print("\n" + "=" * 60)
    print("Risk Buffer Test Complete")
    print("=" * 60)
