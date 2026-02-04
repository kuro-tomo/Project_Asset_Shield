"""
Capacity Engine for Asset Shield V3.2
30 Billion JPY AUM Feasibility Proof

Implements:
- ADT (Average Daily Turnover) calculation from DB turnover field
- Almgren-Chriss market impact model (SYNCHRONIZED with alpha_model.py)
- Portfolio capacity validation for institutional-scale AUM

V3.2.0: All impact parameters now sourced from UNIFIED_AC_PARAMS
        for 100% consistency across backtest and execution.

Author: Asset Shield V3 Team
Version: 3.2.0 (2026-02-04)
"""

import logging
import math
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import unified parameters from alpha_model
from shield.alpha_model import UNIFIED_AC_PARAMS, UnifiedACParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LiquidityMetrics:
    """Individual stock liquidity assessment result"""
    code: str
    adt_20d: float                    # 20-day average daily turnover (JPY)
    adt_60d: float                    # 60-day average daily turnover (JPY)
    max_position_value: float         # Maximum position value (JPY)
    max_shares: int                   # Maximum tradeable shares
    estimated_impact_bps: float       # Almgren-Chriss impact estimate
    participation_rate: float         # Fraction of daily volume
    is_tradeable: bool               # Meets all liquidity criteria
    capacity_grade: str               # A/B/C/D liquidity grade
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CapacityReport:
    """Portfolio-wide capacity validation report"""
    target_aum: float                      # Target AUM (JPY)
    achievable_aum: float                  # Actually achievable AUM
    aum_utilization: float                 # achievable / target
    total_candidates: int                  # Total stocks analyzed
    tradeable_count: int                   # Stocks passing liquidity filter
    aggregate_impact_bps: float            # Portfolio-weighted impact
    avg_participation_rate: float          # Average participation rate
    capacity_sufficient: bool              # Can support target AUM
    grade_distribution: Dict[str, int]     # Count by liquidity grade
    liquidity_metrics: List[LiquidityMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['liquidity_metrics'] = [m.to_dict() for m in self.liquidity_metrics]
        return result


@dataclass
class AlmgrenChrissParams:
    """
    Almgren-Chriss Market Impact Model Parameters - V3.2.0 SYNCHRONIZED

    Based on:
    - Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.
    - Conservative calibration for Japanese equity markets.

    V3.2.0: Now sources from UNIFIED_AC_PARAMS for 100% consistency
            across alpha_model.py, execution_core.py, and capacity_engine.py
    """
    gamma: float = field(default_factory=lambda: UNIFIED_AC_PARAMS.gamma)
    eta: float = field(default_factory=lambda: UNIFIED_AC_PARAMS.eta)
    sigma: float = field(default_factory=lambda: UNIFIED_AC_PARAMS.sigma_annual)
    spread_bps: float = field(default_factory=lambda: UNIFIED_AC_PARAMS.spread_bps)

    @classmethod
    def from_unified(cls, params: UnifiedACParams = None) -> 'AlmgrenChrissParams':
        """Create from unified parameters (recommended)"""
        p = params or UNIFIED_AC_PARAMS
        return cls(gamma=p.gamma, eta=p.eta, sigma=p.sigma_annual, spread_bps=p.spread_bps)

    @classmethod
    def conservative(cls) -> 'AlmgrenChrissParams':
        """Conservative parameters for impact estimation"""
        p = UnifiedACParams.conservative()
        return cls(gamma=p.gamma, eta=p.eta, sigma=p.sigma_annual, spread_bps=p.spread_bps)

    @classmethod
    def standard(cls) -> 'AlmgrenChrissParams':
        """Standard parameters for impact estimation (UNIFIED)"""
        return cls.from_unified(UNIFIED_AC_PARAMS)


# =============================================================================
# Capacity Engine
# =============================================================================

class CapacityEngine:
    """
    Capacity Engine for 30B JPY AUM Validation

    Validates that the strategy can be executed at institutional scale
    with acceptable market impact (< 50bps).

    Key Features:
    - ADT-based liquidity filtering (minimum 500M JPY daily turnover)
    - Almgren-Chriss impact modeling
    - Portfolio-wide capacity aggregation
    - Execution feasibility scoring

    Japanese Market Considerations:
    - TSE market structure
    - Tick size rules
    - Trading session patterns
    """

    # Capacity thresholds
    LIQUIDITY_THRESHOLD = 500_000_000      # Minimum ADT: 500M JPY
    MAX_PARTICIPATION = 0.10                # Maximum 10% of daily volume
    TARGET_AUM = 30_000_000_000            # Target AUM: 30B JPY
    MAX_IMPACT_BPS = 50                     # Maximum acceptable impact

    # Liquidity grading thresholds (ADT in JPY)
    GRADE_THRESHOLDS = {
        'A': 5_000_000_000,    # 50B+ JPY: Highest liquidity
        'B': 1_000_000_000,    # 10B-50B JPY: High liquidity
        'C': 500_000_000,      # 5B-10B JPY: Standard
        'D': 0                  # Below 5B JPY: Low liquidity
    }

    def __init__(
        self,
        impact_params: Optional[AlmgrenChrissParams] = None,
        target_aum: float = None
    ):
        """
        Initialize Capacity Engine.

        Args:
            impact_params: Almgren-Chriss parameters (uses conservative defaults)
            target_aum: Target AUM in JPY (default: 30B JPY)
        """
        self.params = impact_params or AlmgrenChrissParams.conservative()
        self.target_aum = target_aum or self.TARGET_AUM

        # Caches
        self._adt_cache: Dict[str, Dict[str, float]] = {}  # code -> {adt_20d, adt_60d}
        self._volatility_cache: Dict[str, float] = {}

        logger.info(
            f"CapacityEngine initialized: target_aum=JPY {self.target_aum/1e9:.1f}B, "
            f"min_adt=JPY {self.LIQUIDITY_THRESHOLD/1e8:.1f}00M"
        )

    # -------------------------------------------------------------------------
    # ADT Calculation
    # -------------------------------------------------------------------------

    def calculate_adt(
        self,
        turnover_history: List[float],
        lookback: int = 20
    ) -> float:
        """
        Calculate Average Daily Turnover from turnover history.

        Args:
            turnover_history: List of daily turnover values (JPY)
            lookback: Number of days for averaging (default: 20)

        Returns:
            ADT in JPY
        """
        if not turnover_history:
            return 0.0

        # Filter out None and zero values
        valid_turnover = [t for t in turnover_history if t and t > 0]

        if len(valid_turnover) < lookback // 2:
            logger.warning(f"Insufficient turnover data: {len(valid_turnover)}/{lookback}")
            return 0.0

        # Use recent data up to lookback period
        recent = valid_turnover[-lookback:] if len(valid_turnover) >= lookback else valid_turnover
        return float(np.mean(recent))

    def set_adt_data(
        self,
        code: str,
        turnover_history: List[float]
    ) -> None:
        """
        Set ADT data for a stock from turnover history.

        Args:
            code: Stock code
            turnover_history: List of daily turnover values
        """
        adt_20d = self.calculate_adt(turnover_history, lookback=20)
        adt_60d = self.calculate_adt(turnover_history, lookback=60)

        self._adt_cache[code] = {
            'adt_20d': adt_20d,
            'adt_60d': adt_60d
        }

    def set_volatility(self, code: str, volatility: float) -> None:
        """Set annualized volatility for a stock"""
        self._volatility_cache[code] = volatility

    def get_adt(self, code: str, lookback: int = 20) -> float:
        """Get cached ADT for a stock"""
        if code not in self._adt_cache:
            return 0.0

        if lookback <= 20:
            return self._adt_cache[code].get('adt_20d', 0.0)
        else:
            return self._adt_cache[code].get('adt_60d', 0.0)

    # -------------------------------------------------------------------------
    # Almgren-Chriss Impact Calculation
    # -------------------------------------------------------------------------

    def calculate_permanent_impact(
        self,
        order_value: float,
        adt: float,
        volatility: float = None
    ) -> float:
        """
        Calculate permanent price impact.

        Formula: I_perm = gamma * sigma * sqrt(order_value / ADT)

        Args:
            order_value: Order value in JPY
            adt: Average daily turnover in JPY
            volatility: Annualized volatility (uses default if None)

        Returns:
            Permanent impact in basis points
        """
        if adt <= 0:
            return float('inf')

        sigma = volatility or self.params.sigma
        participation = order_value / adt

        # Square root model for permanent impact
        impact = self.params.gamma * sigma * math.sqrt(participation)

        return impact * 10000  # Convert to bps

    def calculate_temporary_impact(
        self,
        order_value: float,
        adt: float,
        execution_days: float = 1.0,
        volatility: float = None
    ) -> float:
        """
        Calculate temporary price impact.

        Formula: I_temp = eta * sigma * (order_value / (ADT * T))^0.6

        Args:
            order_value: Order value in JPY
            adt: Average daily turnover in JPY
            execution_days: Execution horizon in days
            volatility: Annualized volatility

        Returns:
            Temporary impact in basis points
        """
        if adt <= 0 or execution_days <= 0:
            return float('inf')

        sigma = volatility or self.params.sigma
        participation_rate = order_value / (adt * execution_days)

        # Power law model for temporary impact
        impact = self.params.eta * sigma * (participation_rate ** 0.6)

        return impact * 10000  # Convert to bps

    def calculate_total_impact(
        self,
        order_value: float,
        adt: float,
        execution_days: float = 1.0,
        volatility: float = None
    ) -> float:
        """
        Calculate total market impact (permanent + temporary + spread).

        Args:
            order_value: Order value in JPY
            adt: Average daily turnover in JPY
            execution_days: Execution horizon in days
            volatility: Annualized volatility

        Returns:
            Total impact in basis points
        """
        perm = self.calculate_permanent_impact(order_value, adt, volatility)
        temp = self.calculate_temporary_impact(order_value, adt, execution_days, volatility)
        spread = self.params.spread_bps / 2  # One-way spread cost

        return perm + temp + spread

    # -------------------------------------------------------------------------
    # Stock Liquidity Assessment
    # -------------------------------------------------------------------------

    def get_liquidity_grade(self, adt: float) -> str:
        """
        Assign liquidity grade based on ADT.

        Grades:
            A: ADT >= 50B JPY (mega-cap liquidity)
            B: ADT >= 10B JPY (large-cap)
            C: ADT >= 5B JPY  (mid-cap, minimum)
            D: ADT < 5B JPY   (below threshold)
        """
        for grade, threshold in sorted(
            self.GRADE_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if adt >= threshold:
                return grade
        return 'D'

    def assess_stock_liquidity(
        self,
        code: str,
        target_position_value: float,
        price: float = None
    ) -> LiquidityMetrics:
        """
        Assess individual stock liquidity for capacity validation.

        Args:
            code: Stock code
            target_position_value: Target position value in JPY
            price: Current stock price (for share calculation)

        Returns:
            LiquidityMetrics with detailed assessment
        """
        warnings = []

        # Get ADT (fallback to 60-day if 20-day unavailable)
        adt_20d = self.get_adt(code, 20)
        adt_60d = self.get_adt(code, 60)

        # Use best available ADT
        adt = adt_20d if adt_20d > 0 else adt_60d

        if adt_20d == 0:
            if adt_60d > 0:
                warnings.append("Using 60-day ADT (20-day unavailable)")
            else:
                warnings.append("No ADT data available")

        # Calculate max position based on participation limit
        max_position = adt * self.MAX_PARTICIPATION if adt > 0 else 0

        # Calculate shares if price available
        max_shares = int(max_position / price) if price and price > 0 else 0

        # Calculate participation rate for target position
        participation = target_position_value / adt if adt > 0 else 1.0

        # Calculate impact
        volatility = self._volatility_cache.get(code, self.params.sigma)
        impact = self.calculate_total_impact(target_position_value, adt, 1.0, volatility)

        # Liquidity grade
        grade = self.get_liquidity_grade(adt)

        # Tradeable check
        is_tradeable = (
            adt >= self.LIQUIDITY_THRESHOLD and
            impact < self.MAX_IMPACT_BPS and
            participation <= self.MAX_PARTICIPATION
        )

        # Additional warnings
        if adt < self.LIQUIDITY_THRESHOLD:
            warnings.append(f"ADT below threshold: ¥{adt/1e9:.2f}B < ¥{self.LIQUIDITY_THRESHOLD/1e9:.2f}B")

        if impact > self.MAX_IMPACT_BPS:
            warnings.append(f"Impact exceeds limit: {impact:.1f}bps > {self.MAX_IMPACT_BPS}bps")

        if participation > self.MAX_PARTICIPATION:
            warnings.append(f"Participation rate too high: {participation:.1%} > {self.MAX_PARTICIPATION:.0%}")

        return LiquidityMetrics(
            code=code,
            adt_20d=adt_20d,
            adt_60d=adt_60d,
            max_position_value=max_position,
            max_shares=max_shares,
            estimated_impact_bps=round(impact, 2),
            participation_rate=round(participation, 4),
            is_tradeable=is_tradeable,
            capacity_grade=grade,
            warnings=warnings
        )

    # -------------------------------------------------------------------------
    # Portfolio Capacity Validation
    # -------------------------------------------------------------------------

    def validate_portfolio_capacity(
        self,
        candidates: List[str],
        prices: Dict[str, float],
        position_weights: Dict[str, float] = None
    ) -> CapacityReport:
        """
        Validate portfolio-wide capacity for target AUM.

        Args:
            candidates: List of stock codes to evaluate
            prices: Dict of {code: price}
            position_weights: Optional dict of {code: weight} (equal weight if None)

        Returns:
            CapacityReport with aggregate capacity assessment
        """
        # Default to equal weights
        if position_weights is None:
            equal_weight = 1.0 / len(candidates) if candidates else 0
            position_weights = {code: equal_weight for code in candidates}

        metrics_list = []
        tradeable_codes = []
        grade_dist = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        total_impact_weighted = 0.0
        total_participation_weighted = 0.0
        achievable_capacity = 0.0

        for code in candidates:
            weight = position_weights.get(code, 0)
            target_value = self.target_aum * weight
            price = prices.get(code, 0)

            # Assess liquidity
            metrics = self.assess_stock_liquidity(code, target_value, price)
            metrics_list.append(metrics)

            # Update grade distribution
            grade_dist[metrics.capacity_grade] = grade_dist.get(metrics.capacity_grade, 0) + 1

            if metrics.is_tradeable:
                tradeable_codes.append(code)

                # Aggregate impact (weighted by position size)
                total_impact_weighted += metrics.estimated_impact_bps * weight
                total_participation_weighted += metrics.participation_rate * weight

                # Achievable capacity = sum of max positions
                achievable_capacity += min(target_value, metrics.max_position_value)
            else:
                # Reduce achievable by untradeable amount
                achievable_capacity += metrics.max_position_value

        # Calculate aggregate metrics
        avg_participation = total_participation_weighted / len(tradeable_codes) if tradeable_codes else 0
        aum_utilization = achievable_capacity / self.target_aum if self.target_aum > 0 else 0

        # Capacity sufficient if we can achieve >= 80% of target with acceptable impact
        capacity_sufficient = (
            aum_utilization >= 0.80 and
            total_impact_weighted < self.MAX_IMPACT_BPS and
            len(tradeable_codes) >= 10  # Minimum diversification
        )

        report = CapacityReport(
            target_aum=self.target_aum,
            achievable_aum=achievable_capacity,
            aum_utilization=round(aum_utilization, 4),
            total_candidates=len(candidates),
            tradeable_count=len(tradeable_codes),
            aggregate_impact_bps=round(total_impact_weighted, 2),
            avg_participation_rate=round(avg_participation, 4),
            capacity_sufficient=capacity_sufficient,
            grade_distribution=grade_dist,
            liquidity_metrics=metrics_list
        )

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("CAPACITY VALIDATION REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Target AUM: ¥{self.target_aum/1e9:.1f}B")
        logger.info(f"Achievable AUM: ¥{achievable_capacity/1e9:.1f}B ({aum_utilization:.1%})")
        logger.info(f"Tradeable Stocks: {len(tradeable_codes)}/{len(candidates)}")
        logger.info(f"Aggregate Impact: {total_impact_weighted:.1f}bps")
        logger.info(f"Avg Participation: {avg_participation:.2%}")
        logger.info(f"Liquidity Grades: A={grade_dist['A']}, B={grade_dist['B']}, C={grade_dist['C']}, D={grade_dist['D']}")
        logger.info(f"Capacity Sufficient: {'YES ✓' if capacity_sufficient else 'NO ✗'}")
        logger.info(f"{'='*60}\n")

        return report

    def get_optimal_execution_schedule(
        self,
        code: str,
        order_value: float,
        max_days: int = 5,
        target_participation: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Calculate optimal execution schedule to minimize impact.

        Args:
            code: Stock code
            order_value: Total order value in JPY
            max_days: Maximum execution days
            target_participation: Target daily participation rate

        Returns:
            List of daily execution slices
        """
        adt = self.get_adt(code, 20)
        if adt <= 0:
            return [{"day": 1, "value": order_value, "participation": 1.0, "impact_bps": float('inf')}]

        schedule = []
        remaining = order_value

        for day in range(1, max_days + 1):
            if remaining <= 0:
                break

            # Daily slice based on target participation
            daily_target = min(adt * target_participation, remaining)
            remaining -= daily_target

            # Calculate impact for this slice
            impact = self.calculate_total_impact(daily_target, adt, 1.0)

            schedule.append({
                "day": day,
                "value": round(daily_target, 0),
                "participation": round(daily_target / adt, 4),
                "impact_bps": round(impact, 2)
            })

        # If remaining, add final slice
        if remaining > 0:
            impact = self.calculate_total_impact(remaining, adt, 1.0)
            schedule.append({
                "day": len(schedule) + 1,
                "value": round(remaining, 0),
                "participation": round(remaining / adt, 4),
                "impact_bps": round(impact, 2)
            })

        return schedule


# =============================================================================
# Utility Functions
# =============================================================================

def create_capacity_engine(
    target_aum_billions: float = 30.0,
    use_conservative_params: bool = True
) -> CapacityEngine:
    """
    Factory function to create configured Capacity Engine.

    Args:
        target_aum_billions: Target AUM in billions JPY
        use_conservative_params: Use conservative impact parameters

    Returns:
        Configured CapacityEngine
    """
    params = AlmgrenChrissParams.conservative() if use_conservative_params else AlmgrenChrissParams.standard()
    return CapacityEngine(
        impact_params=params,
        target_aum=target_aum_billions * 1e9
    )


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Capacity Engine Test - 30B JPY AUM Validation")
    print("=" * 60)

    # Create engine
    engine = create_capacity_engine(target_aum_billions=30.0)

    # Test data: Major Japanese stocks with estimated ADT
    test_stocks = {
        "7203": {"name": "Toyota", "adt": 80_000_000_000, "price": 2500},      # Toyota: 80B JPY
        "9984": {"name": "SoftBank", "adt": 50_000_000_000, "price": 8000},    # SBG: 50B JPY
        "6758": {"name": "Sony", "adt": 30_000_000_000, "price": 12000},       # Sony: 30B JPY
        "8306": {"name": "MUFG", "adt": 25_000_000_000, "price": 1200},        # MUFG: 25B JPY
        "6501": {"name": "Hitachi", "adt": 15_000_000_000, "price": 5000},     # Hitachi: 15B JPY
        "7267": {"name": "Honda", "adt": 12_000_000_000, "price": 1500},       # Honda: 12B JPY
        "4502": {"name": "Takeda", "adt": 8_000_000_000, "price": 4000},       # Takeda: 8B JPY
        "9432": {"name": "NTT", "adt": 6_000_000_000, "price": 170},           # NTT: 6B JPY
        "6902": {"name": "Denso", "adt": 5_000_000_000, "price": 2200},        # Denso: 5B JPY
        "2914": {"name": "JT", "adt": 4_000_000_000, "price": 3500}            # JT: 4B JPY
    }

    # Set ADT data
    for code, data in test_stocks.items():
        # Simulate turnover history
        turnover_history = [data["adt"] * np.random.uniform(0.8, 1.2) for _ in range(60)]
        engine.set_adt_data(code, turnover_history)

    # Test individual stock assessment
    print("\n--- Individual Stock Assessment ---")
    for code, data in list(test_stocks.items())[:3]:
        target_position = 3_000_000_000  # 3B JPY position
        metrics = engine.assess_stock_liquidity(code, target_position, data["price"])
        print(f"\n{code} ({data['name']}):")
        print(f"  ADT 20d: ¥{metrics.adt_20d/1e9:.1f}B")
        print(f"  Max Position: ¥{metrics.max_position_value/1e9:.1f}B")
        print(f"  Impact: {metrics.estimated_impact_bps:.1f}bps")
        print(f"  Grade: {metrics.capacity_grade}")
        print(f"  Tradeable: {'YES' if metrics.is_tradeable else 'NO'}")

    # Test portfolio capacity validation
    print("\n--- Portfolio Capacity Validation ---")
    candidates = list(test_stocks.keys())
    prices = {code: data["price"] for code, data in test_stocks.items()}

    report = engine.validate_portfolio_capacity(candidates, prices)

    print(f"\nCapacity Report Summary:")
    print(f"  Target AUM: ¥{report.target_aum/1e9:.0f}B")
    print(f"  Achievable: ¥{report.achievable_aum/1e9:.0f}B ({report.aum_utilization:.1%})")
    print(f"  Sufficient: {'YES ✓' if report.capacity_sufficient else 'NO ✗'}")

    # Test execution schedule
    print("\n--- Optimal Execution Schedule ---")
    schedule = engine.get_optimal_execution_schedule(
        "7203",
        order_value=5_000_000_000,  # 5B JPY order
        max_days=5
    )

    for slice in schedule:
        print(f"  Day {slice['day']}: ¥{slice['value']/1e9:.1f}B ({slice['participation']:.1%}), impact={slice['impact_bps']:.1f}bps")

    print("\n" + "=" * 60)
    print("Capacity Engine Test Complete")
    print("=" * 60)
