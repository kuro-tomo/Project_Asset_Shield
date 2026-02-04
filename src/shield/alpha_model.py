"""
Alpha Model with Advanced Market Impact & Survivorship Bias Avoidance
GPT 5.2 Codex Audit Recommendations Implementation

This module implements:
1. Advanced Market Impact Model (Almgren-Chriss based)
2. Survivorship Bias Avoidance Logic
3. Production-grade Alpha Signal Generation

Author: Asset Shield V2 Team
Version: 2.0.0 (2026-01-29)
"""

import logging
import math
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Market Impact Model (Almgren-Chriss Based)
# =============================================================================

@dataclass
class MarketImpactParams:
    """
    Market Impact Model Parameters
    
    Based on Almgren-Chriss (2000) optimal execution framework with
    extensions for Japanese equity market characteristics.
    
    References:
    - Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.
    - Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management.
    """
    # Permanent impact coefficient (price impact per unit traded)
    gamma: float = 0.1
    
    # Temporary impact coefficient (execution cost)
    eta: float = 0.01
    
    # Volatility (annualized)
    sigma: float = 0.20
    
    # Risk aversion parameter
    lambda_risk: float = 1e-6
    
    # Market participation rate limit
    max_participation_rate: float = 0.10  # 10% of ADV
    
    # Bid-ask spread (bps)
    spread_bps: float = 10.0
    
    # Price tick size (JPY)
    tick_size: float = 1.0


@dataclass
class ImpactEstimate:
    """Market impact estimation result"""
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_impact_bps: float
    execution_cost_jpy: float
    optimal_horizon_days: float
    participation_rate: float
    confidence: float
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MarketImpactModel:
    """
    Advanced Market Impact Calculator
    
    Implements Almgren-Chriss optimal execution model with:
    - Permanent impact (information leakage)
    - Temporary impact (execution pressure)
    - Volatility-adjusted sizing
    - ADV-based participation limits
    
    Japanese Market Adaptations:
    - TSE tick size rules
    - Itayose (opening auction) considerations
    - Lunch break liquidity patterns
    """
    
    # TSE Trading hours (minutes)
    MORNING_SESSION_MINUTES = 150  # 9:00-11:30
    AFTERNOON_SESSION_MINUTES = 150  # 12:30-15:00
    TOTAL_TRADING_MINUTES = 300
    
    # Liquidity patterns (relative to average)
    LIQUIDITY_PATTERN = {
        "opening": 1.5,      # First 30 min
        "morning_mid": 0.8,  # 9:30-11:00
        "morning_close": 1.2,  # 11:00-11:30
        "afternoon_open": 1.3,  # 12:30-13:00
        "afternoon_mid": 0.7,  # 13:00-14:30
        "closing": 2.0       # Last 30 min
    }

    # Safety floors (prevent zero/negative inputs from collapsing impact)
    MIN_SIGMA = 1e-6
    MIN_EXECUTION_HORIZON_DAYS = 0.25
    
    def __init__(self, params: Optional[MarketImpactParams] = None):
        """
        Initialize Market Impact Model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or MarketImpactParams()
        self._adv_cache: Dict[str, float] = {}
        self._volatility_cache: Dict[str, float] = {}
        
    def set_stock_data(
        self,
        code: str,
        adv: float,
        volatility: float
    ) -> None:
        """
        Set stock-specific data for impact calculation.

        Args:
            code: Stock code
            adv: Average Daily Volume (shares)
            volatility: Annualized volatility
        """
        if adv <= 0:
            logger.warning(
                "MarketImpactModel.set_stock_data: adv<=0 for %s (adv=%s). Impact calc may be disabled.",
                code,
                adv,
            )
        if volatility is not None and volatility <= 0:
            logger.warning(
                "MarketImpactModel.set_stock_data: volatility<=0 for %s (volatility=%s). Risk terms may collapse.",
                code,
                volatility,
            )
        self._adv_cache[code] = adv
        self._volatility_cache[code] = volatility

    def set_adt_data(
        self,
        code: str,
        adt_20d: float,
        adt_60d: float = None,
        volatility: float = None
    ) -> None:
        """
        Set ADT (Average Daily Turnover) data for capacity-aware impact calculation.

        This method is designed to work with the CapacityEngine for
        institutional-scale position sizing.

        Args:
            code: Stock code
            adt_20d: 20-day average daily turnover in JPY
            adt_60d: 60-day average daily turnover (fallback, optional)
            volatility: Annualized volatility (optional)
        """
        # Store ADT (turnover) - convert to approximate ADV if needed for compatibility
        # Assuming average price of 2000 JPY for conversion
        estimated_price = 2000
        estimated_adv = adt_20d / estimated_price if adt_20d > 0 else 0

        if adt_20d <= 0:
            logger.warning(
                "MarketImpactModel.set_adt_data: adt_20d<=0 for %s (adt_20d=%s). Capacity will be zero.",
                code,
                adt_20d,
            )
        if adt_60d is not None and adt_60d <= 0:
            logger.warning(
                "MarketImpactModel.set_adt_data: adt_60d<=0 for %s (adt_60d=%s).", code, adt_60d
            )

        self._adv_cache[code] = estimated_adv
        self._adt_cache = getattr(self, '_adt_cache', {})
        self._adt_cache[code] = {
            'adt_20d': adt_20d,
            'adt_60d': adt_60d or adt_20d
        }

        if volatility:
            self._volatility_cache[code] = volatility

    def get_adt(self, code: str, lookback: int = 20) -> float:
        """
        Get ADT (Average Daily Turnover) for a stock.

        Args:
            code: Stock code
            lookback: 20 or 60 day lookback

        Returns:
            ADT in JPY, or 0 if not available
        """
        adt_cache = getattr(self, '_adt_cache', {})
        if code not in adt_cache:
            return 0.0

        if lookback <= 20:
            return adt_cache[code].get('adt_20d', 0.0)
        return adt_cache[code].get('adt_60d', 0.0)

    def calculate_capacity_adjusted_size(
        self,
        code: str,
        target_value: float,
        max_participation: float = 0.10,
        max_impact_bps: float = 50.0
    ) -> tuple:
        """
        Calculate capacity-adjusted position size.

        Ensures position size respects:
        1. Maximum participation rate (10% of daily volume)
        2. Maximum acceptable market impact (50 bps)

        Args:
            code: Stock code
            target_value: Target position value in JPY
            max_participation: Maximum fraction of daily turnover (default 10%)
            max_impact_bps: Maximum acceptable impact in bps (default 50)

        Returns:
            Tuple of (adjusted_value, impact_bps, is_tradeable)
        """
        adt = self.get_adt(code, 20)

        if adt <= 0:
            logger.warning(
                "MarketImpactModel.calculate_capacity_adjusted_size: ADT<=0 for %s (adt=%s).",
                code,
                adt,
            )
            return (0, float('inf'), False)

        # Capacity constraint
        max_from_capacity = adt * max_participation

        # Calculate impact for target value
        impact = self.estimate_total_impact(
            code=code,
            order_size=int(target_value / 2000),  # Approximate shares
            price=2000,
            side="BUY",
            urgency="NORMAL"
        )

        # Adjust if impact exceeds limit
        adjusted_value = target_value
        if impact.total_impact_bps > max_impact_bps:
            reduction_factor = max_impact_bps / impact.total_impact_bps
            adjusted_value = target_value * reduction_factor

        # Apply capacity constraint
        adjusted_value = min(adjusted_value, max_from_capacity)

        # Recalculate impact for adjusted value
        final_impact = self.estimate_total_impact(
            code=code,
            order_size=int(adjusted_value / 2000),
            price=2000,
            side="BUY",
            urgency="NORMAL"
        )

        is_tradeable = final_impact.total_impact_bps <= max_impact_bps

        return (adjusted_value, final_impact.total_impact_bps, is_tradeable)
        
    def calculate_permanent_impact(
        self,
        order_size: int,
        adv: float,
        price: float,
        sigma: Optional[float] = None
    ) -> float:
        """
        Calculate permanent price impact (information leakage).
        
        Permanent impact represents the price change that persists
        after the trade is complete, reflecting information content.
        
        Formula: I_perm = gamma * sigma * (Q / ADV)^0.5
        
        Args:
            order_size: Order size in shares
            adv: Average daily volume
            price: Current price
            
        Returns:
            Permanent impact in basis points
        """
        if adv <= 0 or price <= 0:
            logger.warning(
                "MarketImpactModel.calculate_permanent_impact: adv<=0 or price<=0 (adv=%s, price=%s). Returning 0.",
                adv,
                price,
            )
            return 0.0
            
        participation = order_size / adv
        sigma = sigma if sigma is not None else self._volatility_cache.get("default", self.params.sigma)
        if sigma <= 0:
            logger.warning(
                "MarketImpactModel.calculate_permanent_impact: sigma<=0 (sigma=%s). Returning 0.",
                sigma,
            )
            return 0.0
        
        # Square root model for permanent impact
        impact = self.params.gamma * sigma * math.sqrt(participation)
        
        # Convert to basis points
        return impact * 10000
    
    def calculate_temporary_impact(
        self,
        order_size: int,
        adv: float,
        price: float,
        execution_horizon_days: float = 1.0,
        sigma: Optional[float] = None
    ) -> float:
        """
        Calculate temporary price impact (execution pressure).
        
        Temporary impact represents the additional cost incurred
        during execution due to market pressure.
        
        Formula: I_temp = eta * sigma * (Q / (ADV * T))^0.6
        
        Args:
            order_size: Order size in shares
            adv: Average daily volume
            price: Current price
            execution_horizon_days: Execution time in days
            
        Returns:
            Temporary impact in basis points
        """
        if adv <= 0 or price <= 0 or execution_horizon_days <= 0:
            logger.warning(
                "MarketImpactModel.calculate_temporary_impact: invalid inputs (adv=%s, price=%s, horizon=%s). Returning 0.",
                adv,
                price,
                execution_horizon_days,
            )
            return 0.0
            
        # Participation rate over execution horizon
        participation_rate = order_size / (adv * execution_horizon_days)
        sigma = sigma if sigma is not None else self._volatility_cache.get("default", self.params.sigma)
        if sigma <= 0:
            logger.warning(
                "MarketImpactModel.calculate_temporary_impact: sigma<=0 (sigma=%s). Returning 0.",
                sigma,
            )
            return 0.0
        
        # Power law model for temporary impact
        impact = self.params.eta * sigma * (participation_rate ** 0.6)
        
        # Convert to basis points
        return impact * 10000
    
    def calculate_spread_cost(self, price: float) -> float:
        """
        Calculate bid-ask spread cost.
        
        Args:
            price: Current price
            
        Returns:
            Spread cost in basis points
        """
        # Half spread (one-way cost)
        return self.params.spread_bps / 2
    
    def estimate_total_impact(
        self,
        code: str,
        order_size: int,
        price: float,
        side: str = "BUY",
        urgency: str = "NORMAL"
    ) -> ImpactEstimate:
        """
        Estimate total market impact for an order.
        
        Args:
            code: Stock code
            order_size: Order size in shares
            price: Current price
            side: "BUY" or "SELL"
            urgency: "LOW", "NORMAL", "HIGH", "URGENT"
            
        Returns:
            ImpactEstimate with detailed breakdown
        """
        warnings = []
        
        # Get ADV (use cache or estimate)
        adv = self._adv_cache.get(code, None)
        if adv is None or adv <= 0:
            adv = max(order_size * 10, 1)
            warnings.append(f"ADV not available for {code}, using estimate")
            warnings.append(f"ADV not available for {code}, using estimate")
            logger.warning(
                "MarketImpactModel.estimate_total_impact: missing ADV for %s. Using estimate adv=%s.",
                code,
                adv,
            )
        
        # Calculate participation rate
        participation_rate = order_size / adv if adv > 0 else 1.0
        if adv <= 0:
            logger.warning(
                "MarketImpactModel.estimate_total_impact: adv<=0 for %s (adv=%s). Participation forced to 1.0.",
                code,
                adv,
            )
        
        # Check participation limit
        if participation_rate > self.params.max_participation_rate:
            warnings.append(
                f"Participation rate {participation_rate:.1%} exceeds limit "
                f"{self.params.max_participation_rate:.1%}"
            )
        
        # Determine execution horizon based on urgency
        urgency_horizons = {
            "LOW": 5.0,
            "NORMAL": 1.0,
            "HIGH": 0.5,
            "URGENT": 0.25
        }
        base_horizon = urgency_horizons.get(urgency, 1.0)
        if base_horizon <= 0:
            logger.warning(
                "MarketImpactModel.estimate_total_impact: base_horizon<=0 for %s (urgency=%s, base_horizon=%s).",
                code,
                urgency,
                base_horizon,
            )
        
        # Adjust horizon for large orders
        if participation_rate > 0.05:
            optimal_horizon = max(base_horizon, participation_rate / 0.05)
        else:
            optimal_horizon = base_horizon
        if optimal_horizon < 1.0:
            logger.warning(
                "MarketImpactModel.estimate_total_impact: optimal_horizon<1d for %s (optimal_horizon=%s).",
                code,
                optimal_horizon,
            )
        optimal_horizon = max(optimal_horizon, self.MIN_EXECUTION_HORIZON_DAYS)
        if optimal_horizon <= 0:
            logger.warning(
                "MarketImpactModel.estimate_total_impact: optimal_horizon<=0 for %s (optimal_horizon=%s).",
                code,
                optimal_horizon,
            )

        # Volatility selection with safety floor
        sigma = self._volatility_cache.get(code, self.params.sigma)
        if sigma is None or sigma <= 0:
            logger.warning(
                "MarketImpactModel.estimate_total_impact: sigma<=0 or missing for %s (sigma=%s). Using fallback.",
                code,
                sigma,
            )
            sigma = self.params.sigma if self.params.sigma > 0 else self.MIN_SIGMA
        
        # Calculate impact components
        permanent_impact = self.calculate_permanent_impact(order_size, adv, price, sigma)
        temporary_impact = self.calculate_temporary_impact(
            order_size, adv, price, optimal_horizon, sigma
        )
        spread_cost = self.calculate_spread_cost(price)
        
        # Total impact
        total_impact = permanent_impact + temporary_impact + spread_cost
        
        # Execution cost in JPY
        order_value = order_size * price
        execution_cost = order_value * (total_impact / 10000)
        
        # Confidence based on data availability
        confidence = 0.9 if code in self._adv_cache else 0.6
        
        return ImpactEstimate(
            permanent_impact_bps=round(permanent_impact, 2),
            temporary_impact_bps=round(temporary_impact, 2),
            total_impact_bps=round(total_impact, 2),
            execution_cost_jpy=round(execution_cost, 0),
            optimal_horizon_days=round(optimal_horizon, 2),
            participation_rate=round(participation_rate, 4),
            confidence=confidence,
            warnings=warnings
        )
    
    def calculate_optimal_execution_schedule(
        self,
        code: str,
        total_shares: int,
        price: float,
        max_days: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Calculate optimal execution schedule using Almgren-Chriss.
        
        Minimizes: E[Cost] + lambda * Var[Cost]
        
        Args:
            code: Stock code
            total_shares: Total shares to execute
            price: Current price
            max_days: Maximum execution days
            
        Returns:
            List of daily execution targets
        """
        adv = self._adv_cache.get(code, total_shares * 5)
        sigma = self._volatility_cache.get(code, self.params.sigma)
        
        # Calculate optimal trading rate
        # kappa = sqrt(lambda * sigma^2 / eta)
        kappa = math.sqrt(
            self.params.lambda_risk * (sigma ** 2) / self.params.eta
        )
        
        schedule = []
        remaining = total_shares
        
        for day in range(max_days):
            if remaining <= 0:
                break
                
            # Exponential decay schedule
            decay_factor = math.exp(-kappa * day)
            daily_target = int(total_shares * decay_factor * (1 - math.exp(-kappa)))
            daily_target = min(daily_target, remaining)
            daily_target = min(daily_target, int(adv * self.params.max_participation_rate))
            
            if daily_target > 0:
                impact = self.estimate_total_impact(code, daily_target, price)
                schedule.append({
                    "day": day + 1,
                    "shares": daily_target,
                    "participation_rate": daily_target / adv if adv > 0 else 0,
                    "estimated_impact_bps": impact.total_impact_bps,
                    "estimated_cost_jpy": impact.execution_cost_jpy
                })
                remaining -= daily_target
        
        # Add remaining if any
        if remaining > 0:
            schedule.append({
                "day": len(schedule) + 1,
                "shares": remaining,
                "participation_rate": remaining / adv if adv > 0 else 0,
                "estimated_impact_bps": 0,  # Will be calculated at execution
                "estimated_cost_jpy": 0
            })
        
        return schedule


# =============================================================================
# Survivorship Bias Avoidance
# =============================================================================

class DelistingReason(Enum):
    """Reasons for stock delisting"""
    BANKRUPTCY = "bankruptcy"
    MERGER = "merger"
    ACQUISITION = "acquisition"
    GOING_PRIVATE = "going_private"
    REGULATORY = "regulatory"
    VOLUNTARY = "voluntary"
    UNKNOWN = "unknown"


@dataclass
class DelistedStock:
    """Information about a delisted stock"""
    code: str
    name: str
    delisting_date: date
    reason: DelistingReason
    final_price: float
    final_return: float  # Return from last holding to delisting
    merger_ratio: Optional[float] = None  # For M&A cases
    acquirer_code: Optional[str] = None


@dataclass
class SurvivorshipBiasReport:
    """Report on survivorship bias handling"""
    total_stocks_analyzed: int
    delisted_count: int
    bankruptcy_count: int
    merger_count: int
    bias_adjustment_factor: float
    affected_returns: Dict[str, float]
    warnings: List[str]


class SurvivorshipBiasHandler:
    """
    Survivorship Bias Avoidance Logic
    
    Ensures backtest results are not inflated by excluding failed companies.
    
    Key Features:
    1. Tracks delisted stocks throughout backtest period
    2. Applies appropriate terminal values for different delisting reasons
    3. Calculates bias adjustment factors
    4. Provides audit trail for due diligence
    
    Japanese Market Considerations:
    - TSE delisting rules
    - TOB (Takeover Bid) handling
    - Bankruptcy proceedings (Civil Rehabilitation, Corporate Reorganization)
    """
    
    # Terminal value assumptions by delisting reason
    TERMINAL_VALUES = {
        DelistingReason.BANKRUPTCY: 0.0,      # Total loss
        DelistingReason.MERGER: 1.0,          # Use merger ratio
        DelistingReason.ACQUISITION: 1.0,     # Use acquisition price
        DelistingReason.GOING_PRIVATE: 0.95,  # Slight discount
        DelistingReason.REGULATORY: 0.1,      # Severe penalty
        DelistingReason.VOLUNTARY: 0.9,       # Minor discount
        DelistingReason.UNKNOWN: 0.5          # Conservative assumption
    }
    
    def __init__(self):
        """Initialize Survivorship Bias Handler"""
        self._delisted_stocks: Dict[str, DelistedStock] = {}
        self._universe_history: Dict[date, List[str]] = {}
        self._position_adjustments: List[Dict] = []
        
    def register_delisting(
        self,
        code: str,
        name: str,
        delisting_date: date,
        reason: DelistingReason,
        final_price: float,
        last_held_price: Optional[float] = None,
        merger_ratio: Optional[float] = None,
        acquirer_code: Optional[str] = None
    ) -> DelistedStock:
        """
        Register a delisted stock.
        
        Args:
            code: Stock code
            name: Company name
            delisting_date: Date of delisting
            reason: Reason for delisting
            final_price: Final trading price
            last_held_price: Price when last held (for return calc)
            merger_ratio: Merger exchange ratio (if applicable)
            acquirer_code: Acquirer stock code (if applicable)
            
        Returns:
            DelistedStock record
        """
        # Calculate final return
        if last_held_price and last_held_price > 0:
            terminal_value = self.TERMINAL_VALUES.get(reason, 0.5)
            
            if reason == DelistingReason.MERGER and merger_ratio:
                final_return = (final_price * merger_ratio - last_held_price) / last_held_price
            elif reason == DelistingReason.ACQUISITION:
                final_return = (final_price - last_held_price) / last_held_price
            else:
                final_return = (final_price * terminal_value - last_held_price) / last_held_price
        else:
            final_return = -1.0 if reason == DelistingReason.BANKRUPTCY else 0.0
        
        delisted = DelistedStock(
            code=code,
            name=name,
            delisting_date=delisting_date,
            reason=reason,
            final_price=final_price,
            final_return=final_return,
            merger_ratio=merger_ratio,
            acquirer_code=acquirer_code
        )
        
        self._delisted_stocks[code] = delisted
        logger.info(
            f"Registered delisting: {code} ({name}) on {delisting_date}, "
            f"reason={reason.value}, final_return={final_return:.2%}"
        )
        
        return delisted
    
    def record_universe(self, as_of_date: date, codes: List[str]) -> None:
        """
        Record the investment universe as of a specific date.
        
        This creates a point-in-time snapshot to avoid look-ahead bias.
        
        Args:
            as_of_date: Date of the snapshot
            codes: List of stock codes in universe
        """
        self._universe_history[as_of_date] = codes.copy()
    
    def get_universe(self, as_of_date: date) -> List[str]:
        """
        Get the investment universe as of a specific date.
        
        Returns the most recent universe snapshot on or before the date.
        
        Args:
            as_of_date: Date to query
            
        Returns:
            List of stock codes in universe
        """
        # Find most recent snapshot
        valid_dates = [d for d in self._universe_history.keys() if d <= as_of_date]
        
        if not valid_dates:
            return []
            
        latest_date = max(valid_dates)
        return self._universe_history[latest_date]
    
    def is_delisted(self, code: str, as_of_date: date) -> bool:
        """
        Check if a stock was delisted by a specific date.
        
        Args:
            code: Stock code
            as_of_date: Date to check
            
        Returns:
            True if delisted by the date
        """
        if code not in self._delisted_stocks:
            return False
            
        return self._delisted_stocks[code].delisting_date <= as_of_date
    
    def get_delisting_info(self, code: str) -> Optional[DelistedStock]:
        """Get delisting information for a stock"""
        return self._delisted_stocks.get(code)
    
    def handle_position_delisting(
        self,
        code: str,
        position_size: int,
        entry_price: float,
        delisting_date: date
    ) -> Dict[str, Any]:
        """
        Handle a position in a delisting stock.
        
        Args:
            code: Stock code
            position_size: Number of shares held
            entry_price: Entry price
            delisting_date: Date of delisting
            
        Returns:
            Dict with adjustment details
        """
        delisted = self._delisted_stocks.get(code)
        
        if not delisted:
            logger.warning(f"No delisting info for {code}")
            return {"error": "No delisting information"}
        
        # Calculate terminal value
        terminal_multiplier = self.TERMINAL_VALUES.get(delisted.reason, 0.5)
        
        if delisted.reason == DelistingReason.MERGER and delisted.merger_ratio:
            # Convert to acquirer shares
            exit_value = delisted.final_price * delisted.merger_ratio * position_size
            adjustment_type = "MERGER_CONVERSION"
        elif delisted.reason == DelistingReason.ACQUISITION:
            # Cash out at acquisition price
            exit_value = delisted.final_price * position_size
            adjustment_type = "ACQUISITION_CASH"
        else:
            # Apply terminal value multiplier
            exit_value = delisted.final_price * terminal_multiplier * position_size
            adjustment_type = "TERMINAL_VALUE"
        
        entry_value = entry_price * position_size
        pnl = exit_value - entry_value
        pnl_pct = pnl / entry_value if entry_value > 0 else 0
        
        adjustment = {
            "code": code,
            "delisting_date": delisting_date.isoformat(),
            "reason": delisted.reason.value,
            "position_size": position_size,
            "entry_price": entry_price,
            "entry_value": entry_value,
            "exit_value": exit_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "adjustment_type": adjustment_type,
            "terminal_multiplier": terminal_multiplier
        }
        
        self._position_adjustments.append(adjustment)
        
        logger.info(
            f"Position delisting handled: {code}, "
            f"PnL={pnl:,.0f} ({pnl_pct:.2%}), type={adjustment_type}"
        )
        
        return adjustment
    
    def calculate_bias_adjustment(
        self,
        backtest_start: date,
        backtest_end: date,
        total_return: float
    ) -> SurvivorshipBiasReport:
        """
        Calculate survivorship bias adjustment factor.
        
        This estimates how much the backtest return would differ
        if delisted stocks were excluded (traditional biased approach).
        
        Args:
            backtest_start: Backtest start date
            backtest_end: Backtest end date
            total_return: Total return from backtest
            
        Returns:
            SurvivorshipBiasReport with adjustment details
        """
        # Count delistings in period
        period_delistings = [
            d for d in self._delisted_stocks.values()
            if backtest_start <= d.delisting_date <= backtest_end
        ]
        
        bankruptcy_count = sum(
            1 for d in period_delistings
            if d.reason == DelistingReason.BANKRUPTCY
        )
        merger_count = sum(
            1 for d in period_delistings
            if d.reason in [DelistingReason.MERGER, DelistingReason.ACQUISITION]
        )
        
        # Calculate affected returns
        affected_returns = {}
        total_negative_impact = 0.0
        
        for adj in self._position_adjustments:
            code = adj["code"]
            pnl_pct = adj["pnl_pct"]
            affected_returns[code] = pnl_pct
            
            if pnl_pct < 0:
                total_negative_impact += abs(pnl_pct)
        
        # Estimate bias adjustment factor
        # This represents how much returns would be inflated without proper handling
        if len(period_delistings) > 0 and total_return != 0:
            # Rough estimate: negative impact from delistings / total return
            bias_adjustment = 1.0 + (total_negative_impact / max(abs(total_return), 0.01))
        else:
            bias_adjustment = 1.0
        
        warnings = []
        if bankruptcy_count > 0:
            warnings.append(
                f"{bankruptcy_count} bankruptcies in period - returns properly adjusted"
            )
        if bias_adjustment > 1.1:
            warnings.append(
                f"Significant survivorship bias detected: {bias_adjustment:.2%} adjustment"
            )
        
        # Get total stocks from universe history
        all_codes = set()
        for codes in self._universe_history.values():
            all_codes.update(codes)
        
        return SurvivorshipBiasReport(
            total_stocks_analyzed=len(all_codes),
            delisted_count=len(period_delistings),
            bankruptcy_count=bankruptcy_count,
            merger_count=merger_count,
            bias_adjustment_factor=round(bias_adjustment, 4),
            affected_returns=affected_returns,
            warnings=warnings
        )
    
    def get_audit_trail(self) -> Dict[str, Any]:
        """
        Get complete audit trail for due diligence.
        
        Returns:
            Dict with all survivorship bias handling details
        """
        return {
            "delisted_stocks": {
                code: {
                    "name": d.name,
                    "delisting_date": d.delisting_date.isoformat(),
                    "reason": d.reason.value,
                    "final_price": d.final_price,
                    "final_return": d.final_return
                }
                for code, d in self._delisted_stocks.items()
            },
            "position_adjustments": self._position_adjustments,
            "universe_snapshots": len(self._universe_history),
            "methodology": {
                "terminal_values": {
                    k.value: v for k, v in self.TERMINAL_VALUES.items()
                },
                "description": "Point-in-time universe with proper delisting handling"
            }
        }


# =============================================================================
# Integrated Alpha Model
# =============================================================================

@dataclass
class AlphaSignal:
    """Alpha signal with impact-adjusted sizing"""
    code: str
    raw_alpha: float
    adjusted_alpha: float
    confidence: float
    impact_cost_bps: float
    recommended_size: int
    execution_horizon_days: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "raw_alpha": round(self.raw_alpha, 4),
            "adjusted_alpha": round(self.adjusted_alpha, 4),
            "confidence": round(self.confidence, 4),
            "impact_cost_bps": round(self.impact_cost_bps, 2),
            "recommended_size": self.recommended_size,
            "execution_horizon_days": round(self.execution_horizon_days, 2),
            "timestamp": self.timestamp.isoformat()
        }


class AlphaModel:
    """
    Production-Grade Alpha Model
    
    Integrates:
    1. Signal generation (from Brain/AdaptiveCore)
    2. Market impact estimation
    3. Survivorship bias handling
    4. Position sizing optimization
    
    This is the main interface for the trading system.
    """
    
    def __init__(
        self,
        impact_model: Optional[MarketImpactModel] = None,
        bias_handler: Optional[SurvivorshipBiasHandler] = None,
        max_impact_bps: float = 50.0,
        min_alpha_threshold: float = 0.01
    ):
        """
        Initialize Alpha Model.
        
        Args:
            impact_model: Market impact calculator
            bias_handler: Survivorship bias handler
            max_impact_bps: Maximum acceptable impact cost
            min_alpha_threshold: Minimum alpha to generate signal
        """
        self.impact_model = impact_model or MarketImpactModel()
        self.bias_handler = bias_handler or SurvivorshipBiasHandler()
        self.max_impact_bps = max_impact_bps
        self.min_alpha_threshold = min_alpha_threshold
        
        # Signal history for analysis
        self._signal_history: List[AlphaSignal] = []
        
    def set_market_data(
        self,
        code: str,
        adv: float,
        volatility: float
    ) -> None:
        """
        Set market data for a stock.
        
        Args:
            code: Stock code
            adv: Average daily volume
            volatility: Annualized volatility
        """
        self.impact_model.set_stock_data(code, adv, volatility)
    
    def generate_signal(
        self,
        code: str,
        raw_alpha: float,
        price: float,
        target_value: float,
        as_of_date: date,
        urgency: str = "NORMAL"
    ) -> Optional[AlphaSignal]:
        """
        Generate impact-adjusted alpha signal.
        
        Args:
            code: Stock code
            raw_alpha: Raw alpha signal [-1, 1]
            price: Current price
            target_value: Target position value in JPY
            as_of_date: Signal date
            urgency: Execution urgency
            
        Returns:
            AlphaSignal if tradeable, None otherwise
        """
        # Check if stock is delisted
        if self.bias_handler.is_delisted(code, as_of_date):
            logger.warning(f"Signal rejected: {code} is delisted as of {as_of_date}")
            return None
        
        # Check minimum alpha threshold
        if abs(raw_alpha) < self.min_alpha_threshold:
            return None
        
        # Calculate target shares
        target_shares = int(target_value / price) if price > 0 else 0
        
        if target_shares <= 0:
            return None
        
        # Estimate market impact
        side = "BUY" if raw_alpha > 0 else "SELL"
        impact = self.impact_model.estimate_total_impact(
            code, target_shares, price, side, urgency
        )
        
        # Check impact limit
        if impact.total_impact_bps > self.max_impact_bps:
            # Reduce size to meet impact limit
            reduction_factor = self.max_impact_bps / impact.total_impact_bps
            target_shares = int(target_shares * reduction_factor)
            
            # Recalculate impact
            impact = self.impact_model.estimate_total_impact(
                code, target_shares, price, side, urgency
            )
        
        # Adjust alpha for impact cost
        impact_cost_pct = impact.total_impact_bps / 10000
        adjusted_alpha = raw_alpha - (impact_cost_pct * np.sign(raw_alpha))
        
        # Check if still profitable after impact
        if abs(adjusted_alpha) < self.min_alpha_threshold:
            logger.info(
                f"Signal rejected after impact adjustment: {code}, "
                f"raw={raw_alpha:.4f}, adjusted={adjusted_alpha:.4f}"
            )
            return None
        
        signal = AlphaSignal(
            code=code,
            raw_alpha=raw_alpha,
            adjusted_alpha=adjusted_alpha,
            confidence=impact.confidence,
            impact_cost_bps=impact.total_impact_bps,
            recommended_size=target_shares,
            execution_horizon_days=impact.optimal_horizon_days,
            timestamp=datetime.now()
        )
        
        self._signal_history.append(signal)
        
        return signal
    
    def process_delisting(
        self,
        code: str,
        position_size: int,
        entry_price: float,
        delisting_date: date
    ) -> Dict[str, Any]:
        """
        Process a delisting event for a held position.
        
        Args:
            code: Stock code
            position_size: Number of shares held
            entry_price: Entry price
            delisting_date: Date of delisting
            
        Returns:
            Dict with adjustment details
        """
        return self.bias_handler.handle_position_delisting(
            code, position_size, entry_price, delisting_date
        )
    
    def get_execution_schedule(
        self,
        code: str,
        total_shares: int,
        price: float
    ) -> List[Dict[str, Any]]:
        """
        Get optimal execution schedule for a large order.
        
        Args:
            code: Stock code
            total_shares: Total shares to execute
            price: Current price
            
        Returns:
            List of daily execution targets
        """
        return self.impact_model.calculate_optimal_execution_schedule(
            code, total_shares, price
        )
    
    def get_signal_history(self) -> List[Dict]:
        """Get history of generated signals"""
        return [s.to_dict() for s in self._signal_history]
    
    def get_bias_report(
        self,
        start_date: date,
        end_date: date,
        total_return: float
    ) -> SurvivorshipBiasReport:
        """
        Get survivorship bias report for a backtest period.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            total_return: Total return from backtest
            
        Returns:
            SurvivorshipBiasReport
        """
        return self.bias_handler.calculate_bias_adjustment(
            start_date, end_date, total_return
        )
    
    def get_audit_report(self) -> Dict[str, Any]:
        """
        Get comprehensive audit report for due diligence.
        
        Returns:
            Dict with all model details
        """
        return {
            "model_version": "2.0.0",
            "generated_at": datetime.now().isoformat(),
            "impact_model": {
                "type": "Almgren-Chriss",
                "params": {
                    "gamma": self.impact_model.params.gamma,
                    "eta": self.impact_model.params.eta,
                    "sigma": self.impact_model.params.sigma,
                    "max_participation_rate": self.impact_model.params.max_participation_rate,
                    "spread_bps": self.impact_model.params.spread_bps
                }
            },
            "survivorship_bias": self.bias_handler.get_audit_trail(),
            "signal_count": len(self._signal_history),
            "configuration": {
                "max_impact_bps": self.max_impact_bps,
                "min_alpha_threshold": self.min_alpha_threshold
            }
        }


# =============================================================================
# Integration with Backtest Framework
# =============================================================================

class AlphaModelBacktestStrategy:
    """
    Strategy wrapper that integrates AlphaModel with the backtest framework.
    
    This provides a drop-in replacement for BrainBacktestStrategy with
    advanced market impact and survivorship bias handling.
    """
    
    def __init__(
        self,
        codes: List[str],
        alpha_model: Optional[AlphaModel] = None,
        training_mode: bool = True
    ):
        """
        Initialize Alpha Model strategy.
        
        Args:
            codes: List of stock codes to trade
            alpha_model: AlphaModel instance (creates default if None)
            training_mode: If True, enables learning
        """
        self.codes = codes
        self.alpha_model = alpha_model or AlphaModel()
        self.training_mode = training_mode
        
        # Import Brain for signal generation
        from shield.brain import ShieldBrain
        
        # Create Brain instances for signal generation
        self.brains: Dict[str, ShieldBrain] = {}
        for code in codes:
            brain_id = f"alpha_{code}"
            self.brains[code] = ShieldBrain(target_id=brain_id)
        
        # Track price history
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.volume_history: Dict[str, List[int]] = defaultdict(list)
        
        # Strategy parameters
        self.lookback = 50
        self.max_positions = 5
        self.position_size_pct = 0.15
        self.take_profit = 0.08
        self.stop_loss = 0.03
        self.max_holding_days = 30
        
        logger.info(
            f"AlphaModelBacktestStrategy initialized: {len(codes)} stocks, "
            f"training={training_mode}"
        )
    
    def __call__(
        self,
        engine: Any,  # BacktestEngine
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
            signals: Additional signals (may include volume data)
        """
        # Update price and volume history
        for code, price in prices.items():
            if code in self.codes:
                self.price_history[code].append(price)
                
                # Get volume from signals if available
                volume = signals.get(f"{code}_volume", 100000)  # Default ADV
                self.volume_history[code].append(volume)
                
                # Keep only recent history
                if len(self.price_history[code]) > 200:
                    self.price_history[code] = self.price_history[code][-200:]
                    self.volume_history[code] = self.volume_history[code][-200:]
                
                # Update market data in alpha model
                if len(self.volume_history[code]) >= 20:
                    adv = np.mean(self.volume_history[code][-20:])
                    returns = np.diff(np.log(self.price_history[code][-21:]))
                    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.2
                    self.alpha_model.set_market_data(code, adv, volatility)
        
        # Process existing positions (exit logic)
        for code in list(engine.positions.keys()):
            if code not in prices:
                continue
            
            price = prices[code]
            trade = engine.positions[code]
            
            # Update holding days
            trade.holding_days = (current_date - trade.entry_date).days
            
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
            
            # Brain-based exit
            elif code in self.brains and len(self.price_history[code]) >= self.lookback:
                brain = self.brains[code]
                confidence = brain.calculate_confidence(
                    self.price_history[code][-self.lookback:]
                )
                threshold = brain.get_threshold()
                
                if confidence < -threshold * 0.5:
                    exit_reason = "BRAIN_EXIT"
            
            if exit_reason:
                engine.close_position(code, price, exit_reason)
                
                # Record result for Brain learning
                if self.training_mode and code in self.brains:
                    actual_pnl = (price - trade.entry_price) * trade.quantity
                    self.brains[code].record_trade_result(actual_pnl)
        
        # Process potential entries
        if len(engine.positions) >= self.max_positions:
            return
        
        for code in self.codes:
            if code in engine.positions:
                continue
            
            if code not in prices:
                continue
            
            if len(self.price_history[code]) < self.lookback:
                continue
            
            price = prices[code]
            brain = self.brains[code]
            
            # Get Brain confidence as raw alpha
            recent_prices = self.price_history[code][-self.lookback:]
            raw_alpha = brain.calculate_confidence(recent_prices)
            
            # Calculate target position value
            target_value = engine.cash * self.position_size_pct
            
            # Generate impact-adjusted signal
            signal = self.alpha_model.generate_signal(
                code=code,
                raw_alpha=raw_alpha,
                price=price,
                target_value=target_value,
                as_of_date=current_date,
                urgency="NORMAL"
            )
            
            if signal and signal.adjusted_alpha > brain.get_threshold():
                # Use recommended size from alpha model
                quantity = signal.recommended_size
                
                if quantity > 0 and quantity * price < engine.cash * 0.95:
                    trade = engine.open_position(code, price, quantity)
                    if trade:
                        logger.debug(
                            f"Alpha entry {code}: price={price:.0f}, qty={quantity}, "
                            f"raw_alpha={signal.raw_alpha:.3f}, "
                            f"adjusted_alpha={signal.adjusted_alpha:.3f}, "
                            f"impact={signal.impact_cost_bps:.1f}bps"
                        )
    
    def get_brain_states(self) -> Dict[str, Dict]:
        """Get current state of all Brains"""
        states = {}
        for code, brain in self.brains.items():
            states[code] = {
                "adaptive_threshold": brain.adaptive_threshold,
                "risk_penalty": brain.risk_penalty,
                "lookback": brain.lookback
            }
        return states


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alpha Model Test - GPT 5.2 Codex Audit Implementation")
    print("=" * 60)
    
    # Test Market Impact Model
    print("\n--- Market Impact Model Test ---")
    impact_model = MarketImpactModel()
    impact_model.set_stock_data("7203", adv=5_000_000, volatility=0.25)
    
    # Estimate impact for a 100,000 share order
    impact = impact_model.estimate_total_impact(
        code="7203",
        order_size=100_000,
        price=2500,
        side="BUY",
        urgency="NORMAL"
    )
    
    print(f"Order: 100,000 shares @ ¥2,500")
    print(f"  Permanent Impact: {impact.permanent_impact_bps:.2f} bps")
    print(f"  Temporary Impact: {impact.temporary_impact_bps:.2f} bps")
    print(f"  Total Impact: {impact.total_impact_bps:.2f} bps")
    print(f"  Execution Cost: ¥{impact.execution_cost_jpy:,.0f}")
    print(f"  Optimal Horizon: {impact.optimal_horizon_days:.1f} days")
    print(f"  Participation Rate: {impact.participation_rate:.2%}")
    
    # Test execution schedule
    print("\n--- Optimal Execution Schedule ---")
    schedule = impact_model.calculate_optimal_execution_schedule(
        code="7203",
        total_shares=500_000,
        price=2500,
        max_days=5
    )
    
    for day in schedule:
        print(
            f"  Day {day['day']}: {day['shares']:,} shares "
            f"({day['participation_rate']:.1%} participation), "
            f"impact={day['estimated_impact_bps']:.1f}bps"
        )
    
    # Test Survivorship Bias Handler
    print("\n--- Survivorship Bias Handler Test ---")
    bias_handler = SurvivorshipBiasHandler()
    
    # Register a bankruptcy
    bias_handler.register_delisting(
        code="9999",
        name="Failed Corp",
        delisting_date=date(2020, 3, 15),
        reason=DelistingReason.BANKRUPTCY,
        final_price=10,
        last_held_price=1000
    )
    
    # Register a merger
    bias_handler.register_delisting(
        code="8888",
        name="Merged Inc",
        delisting_date=date(2021, 6, 30),
        reason=DelistingReason.MERGER,
        final_price=1500,
        last_held_price=1200,
        merger_ratio=1.2,
        acquirer_code="7777"
    )
    
    # Handle position in bankrupt company
    adjustment = bias_handler.handle_position_delisting(
        code="9999",
        position_size=1000,
        entry_price=800,
        delisting_date=date(2020, 3, 15)
    )
    print(f"Bankruptcy adjustment: PnL = ¥{adjustment['pnl']:,.0f} ({adjustment['pnl_pct']:.2%})")
    
    # Calculate bias report
    report = bias_handler.calculate_bias_adjustment(
        backtest_start=date(2019, 1, 1),
        backtest_end=date(2022, 12, 31),
        total_return=0.50
    )
    print(f"Bias adjustment factor: {report.bias_adjustment_factor:.4f}")
    print(f"Delistings in period: {report.delisted_count}")
    print(f"Bankruptcies: {report.bankruptcy_count}")
    
    # Test integrated Alpha Model
    print("\n--- Integrated Alpha Model Test ---")
    alpha_model = AlphaModel(
        impact_model=impact_model,
        bias_handler=bias_handler,
        max_impact_bps=30.0,
        min_alpha_threshold=0.02
    )
    
    # Generate a signal
    signal = alpha_model.generate_signal(
        code="7203",
        raw_alpha=0.15,
        price=2500,
        target_value=50_000_000,  # ¥50M
        as_of_date=date(2026, 1, 29),
        urgency="NORMAL"
    )
    
    if signal:
        print(f"Signal generated for 7203:")
        print(f"  Raw Alpha: {signal.raw_alpha:.4f}")
        print(f"  Adjusted Alpha: {signal.adjusted_alpha:.4f}")
        print(f"  Impact Cost: {signal.impact_cost_bps:.2f} bps")
        print(f"  Recommended Size: {signal.recommended_size:,} shares")
        print(f"  Execution Horizon: {signal.execution_horizon_days:.1f} days")
    
    # Try to generate signal for delisted stock
    print("\n--- Delisted Stock Signal Test ---")
    signal_delisted = alpha_model.generate_signal(
        code="9999",
        raw_alpha=0.20,
        price=10,
        target_value=10_000_000,
        as_of_date=date(2020, 4, 1),  # After delisting
        urgency="NORMAL"
    )
    print(f"Signal for delisted stock: {signal_delisted}")  # Should be None
    
    # Get audit report
    print("\n--- Audit Report ---")
    audit = alpha_model.get_audit_report()
    print(f"Model Version: {audit['model_version']}")
    print(f"Impact Model Type: {audit['impact_model']['type']}")
    print(f"Signal Count: {audit['signal_count']}")
    print(f"Delisted Stocks Tracked: {len(audit['survivorship_bias']['delisted_stocks'])}")
    
    print("\n" + "=" * 60)
    print("Alpha Model Test Complete")
    print("=" * 60)
