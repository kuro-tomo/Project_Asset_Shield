"""
Execution Core for Asset Shield V3.2
Layer 3: Execution (Stealth Algo)

Implements market impact minimization through:
- VWAP (Volume Weighted Average Price) execution
- TWAP (Time Weighted Average Price) execution
- Almgren-Chriss optimal execution model
- TSE tick size rules compliance

V3.2.0: SYNCHRONIZED Almgren-Chriss parameters with alpha_model.py
        All impact calculations now use UNIFIED_AC_PARAMS
"""

import logging
import math
from datetime import datetime, timedelta, time as dt_time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import unified parameters from alpha_model
from shield.alpha_model import UNIFIED_AC_PARAMS, UnifiedACParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategy types"""
    VWAP = "vwap"           # Volume Weighted Average Price
    TWAP = "twap"           # Time Weighted Average Price
    OPTIMAL = "optimal"     # Almgren-Chriss optimal execution
    AGGRESSIVE = "aggressive"  # Front-loaded execution
    PASSIVE = "passive"     # Back-loaded execution


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TickSizeRule:
    """TSE tick size rule definition"""
    price_from: float
    price_to: float
    tick_size: float


@dataclass
class SliceOrder:
    """Single slice of a parent order"""
    slice_id: int
    parent_id: str
    side: OrderSide
    target_quantity: int
    target_price: Optional[float]
    scheduled_time: datetime
    executed_quantity: int = 0
    executed_price: float = 0.0
    status: str = "pending"
    
    def to_dict(self) -> Dict:
        return {
            "slice_id": self.slice_id,
            "parent_id": self.parent_id,
            "side": self.side.value,
            "target_quantity": self.target_quantity,
            "target_price": self.target_price,
            "scheduled_time": self.scheduled_time.isoformat(),
            "executed_quantity": self.executed_quantity,
            "executed_price": self.executed_price,
            "status": self.status
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan for a parent order"""
    parent_id: str
    code: str
    side: OrderSide
    total_quantity: int
    strategy: ExecutionStrategy
    slices: List[SliceOrder] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_cost_bps: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "parent_id": self.parent_id,
            "code": self.code,
            "side": self.side.value,
            "total_quantity": self.total_quantity,
            "strategy": self.strategy.value,
            "slices": [s.to_dict() for s in self.slices],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "estimated_cost_bps": self.estimated_cost_bps
        }


class TSETickSizeManager:
    """
    TSE Tick Size Rules Manager

    Implements the tick size rules for Tokyo Stock Exchange.
    Tick sizes vary based on price levels and TOPIX membership.
    """
    
    # Standard tick size rules (as of 2024)
    STANDARD_RULES = [
        TickSizeRule(0, 1000, 1),
        TickSizeRule(1000, 3000, 1),
        TickSizeRule(3000, 5000, 5),
        TickSizeRule(5000, 10000, 10),
        TickSizeRule(10000, 30000, 10),
        TickSizeRule(30000, 50000, 50),
        TickSizeRule(50000, 100000, 100),
        TickSizeRule(100000, 300000, 100),
        TickSizeRule(300000, 500000, 500),
        TickSizeRule(500000, 1000000, 1000),
        TickSizeRule(1000000, 3000000, 1000),
        TickSizeRule(3000000, 5000000, 5000),
        TickSizeRule(5000000, 10000000, 10000),
        TickSizeRule(10000000, 30000000, 10000),
        TickSizeRule(30000000, 50000000, 50000),
        TickSizeRule(50000000, float('inf'), 100000),
    ]
    
    # TOPIX100 tick size rules (tighter spreads)
    TOPIX100_RULES = [
        TickSizeRule(0, 1000, 0.1),
        TickSizeRule(1000, 3000, 0.5),
        TickSizeRule(3000, 5000, 1),
        TickSizeRule(5000, 10000, 1),
        TickSizeRule(10000, 30000, 5),
        TickSizeRule(30000, 50000, 10),
        TickSizeRule(50000, 100000, 10),
        TickSizeRule(100000, 300000, 50),
        TickSizeRule(300000, 500000, 100),
        TickSizeRule(500000, 1000000, 100),
        TickSizeRule(1000000, 3000000, 500),
        TickSizeRule(3000000, 5000000, 1000),
        TickSizeRule(5000000, 10000000, 1000),
        TickSizeRule(10000000, 30000000, 5000),
        TickSizeRule(30000000, 50000000, 10000),
        TickSizeRule(50000000, float('inf'), 50000),
    ]
    
    def __init__(self, topix100_codes: Optional[List[str]] = None):
        self.topix100_codes = set(topix100_codes or [])
        
    def get_tick_size(self, code: str, price: float) -> float:
        """
        Get tick size for a given stock and price.
        
        Args:
            code: Stock code
            price: Current price
            
        Returns:
            Tick size in JPY
        """
        rules = self.TOPIX100_RULES if code in self.topix100_codes else self.STANDARD_RULES
        
        for rule in rules:
            if rule.price_from <= price < rule.price_to:
                return rule.tick_size
                
        return 1.0  # Default fallback
    
    def round_to_tick(self, code: str, price: float, direction: str = "nearest") -> float:
        """
        Round price to valid tick.
        
        Args:
            code: Stock code
            price: Price to round
            direction: "nearest", "up", or "down"
            
        Returns:
            Rounded price
        """
        tick = self.get_tick_size(code, price)
        
        if direction == "up":
            return math.ceil(price / tick) * tick
        elif direction == "down":
            return math.floor(price / tick) * tick
        else:
            return round(price / tick) * tick


class AlmgrenChrissModel:
    """
    Almgren-Chriss Optimal Execution Model - V3.2.0 SYNCHRONIZED

    Minimizes execution cost = market impact + timing risk

    V3.2.0: Parameters now synchronized with UNIFIED_AC_PARAMS from alpha_model.py
            All modules use identical impact coefficients for 100% consistency.

    Parameters (from UNIFIED_AC_PARAMS):
    - sigma: Daily volatility (converted from annualized)
    - eta: Temporary impact coefficient
    - gamma: Permanent impact coefficient
    - lambda_: Risk aversion parameter
    """

    def __init__(
        self,
        sigma: float = None,
        eta: float = None,
        gamma: float = None,
        lambda_: float = None,
        use_unified: bool = True
    ):
        """
        Initialize AC model with synchronized parameters.

        Args:
            sigma: Daily volatility (if None, uses UNIFIED_AC_PARAMS.sigma_daily)
            eta: Temporary impact (if None, uses UNIFIED_AC_PARAMS.eta)
            gamma: Permanent impact (if None, uses UNIFIED_AC_PARAMS.gamma)
            lambda_: Risk aversion (if None, uses UNIFIED_AC_PARAMS.lambda_risk)
            use_unified: If True, source defaults from UNIFIED_AC_PARAMS
        """
        if use_unified:
            # Use UNIFIED parameters as defaults for 100% consistency
            self.sigma = sigma if sigma is not None else UNIFIED_AC_PARAMS.sigma_daily
            self.eta = eta if eta is not None else UNIFIED_AC_PARAMS.eta
            self.gamma = gamma if gamma is not None else UNIFIED_AC_PARAMS.gamma
            self.lambda_ = lambda_ if lambda_ is not None else UNIFIED_AC_PARAMS.lambda_risk
        else:
            # Legacy defaults (deprecated)
            self.sigma = sigma if sigma is not None else 0.02
            self.eta = eta if eta is not None else 0.0001
            self.gamma = gamma if gamma is not None else 0.00005
            self.lambda_ = lambda_ if lambda_ is not None else 0.001

        logger.info(
            f"AlmgrenChrissModel initialized: sigma={self.sigma:.4f} (daily), "
            f"eta={self.eta:.4f}, gamma={self.gamma:.4f}, lambda={self.lambda_:.2e}"
        )
        
    def optimal_trajectory(
        self,
        total_shares: int,
        n_periods: int,
        daily_volume: int
    ) -> List[float]:
        """
        Calculate optimal execution trajectory.
        
        Args:
            total_shares: Total shares to execute
            n_periods: Number of time periods
            daily_volume: Average daily volume
            
        Returns:
            List of shares to execute in each period
        """
        X = total_shares
        T = n_periods
        
        # Participation rate
        participation = X / (daily_volume * T / 390)  # 390 minutes in trading day
        
        # Urgency parameter
        kappa = np.sqrt(self.lambda_ * self.sigma**2 / self.eta)
        
        # Optimal trajectory
        trajectory = []
        remaining = X
        
        for j in range(T):
            t = j / T
            # Exponential decay based on urgency
            if kappa * T > 0:
                x_j = X * (np.sinh(kappa * (T - j)) / np.sinh(kappa * T))
            else:
                x_j = X * (T - j) / T
                
            shares_this_period = remaining - x_j
            shares_this_period = max(0, min(remaining, shares_this_period))
            trajectory.append(int(shares_this_period))
            remaining -= shares_this_period
            
        # Ensure all shares are allocated
        if remaining > 0:
            trajectory[-1] += int(remaining)
            
        return trajectory
    
    def estimate_cost(
        self,
        total_shares: int,
        price: float,
        daily_volume: int,
        n_periods: int
    ) -> Tuple[float, float]:
        """
        Estimate execution cost in basis points.
        
        Returns:
            Tuple of (expected_cost_bps, cost_std_bps)
        """
        X = total_shares
        V = daily_volume
        
        # Participation rate
        participation = X / V
        
        # Permanent impact (linear)
        permanent_cost = self.gamma * X * price
        
        # Temporary impact (square root)
        temp_cost = self.eta * np.sqrt(X / n_periods) * price * n_periods
        
        # Total cost in bps
        total_cost = (permanent_cost + temp_cost) / (X * price) * 10000
        
        # Timing risk (volatility)
        timing_risk = self.sigma * np.sqrt(n_periods / 390) * 10000
        
        return round(total_cost, 2), round(timing_risk, 2)


class ExecutionCore:
    """
    Stealth Execution Algorithm Core
    
    Implements VWAP/TWAP execution with market impact minimization.
    """
    
    # TSE trading hours
    MORNING_START = dt_time(9, 0)
    MORNING_END = dt_time(11, 30)
    AFTERNOON_START = dt_time(12, 30)
    AFTERNOON_END = dt_time(15, 0)
    
    # Typical intraday volume distribution (percentage per 30-min bucket)
    VOLUME_PROFILE = {
        "09:00": 0.15,  # Opening auction + first 30 min
        "09:30": 0.10,
        "10:00": 0.08,
        "10:30": 0.07,
        "11:00": 0.08,
        "12:30": 0.12,  # Afternoon open
        "13:00": 0.08,
        "13:30": 0.07,
        "14:00": 0.08,
        "14:30": 0.17,  # Closing auction
    }
    
    def __init__(
        self,
        tick_manager: Optional[TSETickSizeManager] = None,
        ac_model: Optional[AlmgrenChrissModel] = None
    ):
        self.tick_manager = tick_manager or TSETickSizeManager()
        self.ac_model = ac_model or AlmgrenChrissModel()
        self._active_plans: Dict[str, ExecutionPlan] = {}
        
    def create_vwap_plan(
        self,
        code: str,
        side: OrderSide,
        total_quantity: int,
        current_price: float,
        daily_volume: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        n_slices: int = 10
    ) -> ExecutionPlan:
        """
        Create VWAP execution plan.
        
        Distributes order according to historical volume profile.
        
        Args:
            code: Stock code
            side: Buy or Sell
            total_quantity: Total shares to execute
            current_price: Current market price
            daily_volume: Average daily volume
            start_time: Execution start time
            end_time: Execution end time
            n_slices: Number of order slices
            
        Returns:
            ExecutionPlan with VWAP slices
        """
        parent_id = f"VWAP_{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Default to full trading day
        now = datetime.now()
        if start_time is None:
            start_time = now.replace(hour=9, minute=0, second=0)
        if end_time is None:
            end_time = now.replace(hour=15, minute=0, second=0)
            
        # Calculate time intervals
        total_minutes = (end_time - start_time).total_seconds() / 60
        interval_minutes = total_minutes / n_slices
        
        # Get volume weights for each slice
        slices = []
        remaining_qty = total_quantity
        
        for i in range(n_slices):
            slice_time = start_time + timedelta(minutes=i * interval_minutes)
            
            # Get volume weight for this time bucket
            time_key = f"{slice_time.hour:02d}:{(slice_time.minute // 30) * 30:02d}"
            volume_weight = self.VOLUME_PROFILE.get(time_key, 0.08)
            
            # Calculate slice quantity
            if i == n_slices - 1:
                slice_qty = remaining_qty
            else:
                slice_qty = int(total_quantity * volume_weight / sum(self.VOLUME_PROFILE.values()) * n_slices / 10)
                slice_qty = min(slice_qty, remaining_qty)
                
            remaining_qty -= slice_qty
            
            # Round price to valid tick
            target_price = self.tick_manager.round_to_tick(
                code, 
                current_price,
                "down" if side == OrderSide.BUY else "up"
            )
            
            slice_order = SliceOrder(
                slice_id=i,
                parent_id=parent_id,
                side=side,
                target_quantity=slice_qty,
                target_price=target_price,
                scheduled_time=slice_time
            )
            slices.append(slice_order)
        
        # Estimate execution cost
        cost_bps, _ = self.ac_model.estimate_cost(
            total_quantity, current_price, daily_volume, n_slices
        )
        
        plan = ExecutionPlan(
            parent_id=parent_id,
            code=code,
            side=side,
            total_quantity=total_quantity,
            strategy=ExecutionStrategy.VWAP,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            estimated_cost_bps=cost_bps
        )
        
        self._active_plans[parent_id] = plan
        logger.info(f"Created VWAP plan: {parent_id} for {total_quantity} shares")
        
        return plan
    
    def create_twap_plan(
        self,
        code: str,
        side: OrderSide,
        total_quantity: int,
        current_price: float,
        daily_volume: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        n_slices: int = 10
    ) -> ExecutionPlan:
        """
        Create TWAP execution plan.
        
        Distributes order evenly across time.
        
        Args:
            code: Stock code
            side: Buy or Sell
            total_quantity: Total shares to execute
            current_price: Current market price
            daily_volume: Average daily volume
            start_time: Execution start time
            end_time: Execution end time
            n_slices: Number of order slices
            
        Returns:
            ExecutionPlan with TWAP slices
        """
        parent_id = f"TWAP_{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        now = datetime.now()
        if start_time is None:
            start_time = now.replace(hour=9, minute=0, second=0)
        if end_time is None:
            end_time = now.replace(hour=15, minute=0, second=0)
            
        total_minutes = (end_time - start_time).total_seconds() / 60
        interval_minutes = total_minutes / n_slices
        
        # Equal distribution
        base_qty = total_quantity // n_slices
        remainder = total_quantity % n_slices
        
        slices = []
        for i in range(n_slices):
            slice_time = start_time + timedelta(minutes=i * interval_minutes)
            slice_qty = base_qty + (1 if i < remainder else 0)
            
            target_price = self.tick_manager.round_to_tick(
                code,
                current_price,
                "down" if side == OrderSide.BUY else "up"
            )
            
            slice_order = SliceOrder(
                slice_id=i,
                parent_id=parent_id,
                side=side,
                target_quantity=slice_qty,
                target_price=target_price,
                scheduled_time=slice_time
            )
            slices.append(slice_order)
        
        cost_bps, _ = self.ac_model.estimate_cost(
            total_quantity, current_price, daily_volume, n_slices
        )
        
        plan = ExecutionPlan(
            parent_id=parent_id,
            code=code,
            side=side,
            total_quantity=total_quantity,
            strategy=ExecutionStrategy.TWAP,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            estimated_cost_bps=cost_bps
        )
        
        self._active_plans[parent_id] = plan
        logger.info(f"Created TWAP plan: {parent_id} for {total_quantity} shares")
        
        return plan
    
    def create_optimal_plan(
        self,
        code: str,
        side: OrderSide,
        total_quantity: int,
        current_price: float,
        daily_volume: int,
        volatility: float,
        urgency: float = 0.5,
        n_slices: int = 10
    ) -> ExecutionPlan:
        """
        Create Almgren-Chriss optimal execution plan.
        
        Balances market impact vs timing risk based on urgency.
        
        Args:
            code: Stock code
            side: Buy or Sell
            total_quantity: Total shares to execute
            current_price: Current market price
            daily_volume: Average daily volume
            volatility: Current volatility estimate
            urgency: Urgency parameter [0, 1] (higher = faster execution)
            n_slices: Number of order slices
            
        Returns:
            ExecutionPlan with optimal trajectory
        """
        parent_id = f"OPT_{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Adjust model parameters based on urgency
        self.ac_model.lambda_ = 0.0001 + urgency * 0.01
        self.ac_model.sigma = volatility
        
        # Get optimal trajectory
        trajectory = self.ac_model.optimal_trajectory(
            total_quantity, n_slices, daily_volume
        )
        
        now = datetime.now()
        start_time = now.replace(hour=9, minute=0, second=0)
        end_time = now.replace(hour=15, minute=0, second=0)
        
        total_minutes = (end_time - start_time).total_seconds() / 60
        interval_minutes = total_minutes / n_slices
        
        slices = []
        for i, qty in enumerate(trajectory):
            slice_time = start_time + timedelta(minutes=i * interval_minutes)
            
            target_price = self.tick_manager.round_to_tick(
                code,
                current_price,
                "down" if side == OrderSide.BUY else "up"
            )
            
            slice_order = SliceOrder(
                slice_id=i,
                parent_id=parent_id,
                side=side,
                target_quantity=qty,
                target_price=target_price,
                scheduled_time=slice_time
            )
            slices.append(slice_order)
        
        cost_bps, risk_bps = self.ac_model.estimate_cost(
            total_quantity, current_price, daily_volume, n_slices
        )
        
        plan = ExecutionPlan(
            parent_id=parent_id,
            code=code,
            side=side,
            total_quantity=total_quantity,
            strategy=ExecutionStrategy.OPTIMAL,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            estimated_cost_bps=cost_bps
        )
        
        self._active_plans[parent_id] = plan
        logger.info(f"Created Optimal plan: {parent_id} for {total_quantity} shares")
        logger.info(f"Estimated cost: {cost_bps} bps, timing risk: {risk_bps} bps")
        
        return plan
    
    def get_next_slice(self, parent_id: str) -> Optional[SliceOrder]:
        """Get next pending slice for execution"""
        plan = self._active_plans.get(parent_id)
        if not plan:
            return None
            
        for slice_order in plan.slices:
            if slice_order.status == "pending":
                return slice_order
                
        return None
    
    def update_slice_execution(
        self,
        parent_id: str,
        slice_id: int,
        executed_quantity: int,
        executed_price: float
    ) -> bool:
        """Update slice with execution results"""
        plan = self._active_plans.get(parent_id)
        if not plan:
            return False
            
        for slice_order in plan.slices:
            if slice_order.slice_id == slice_id:
                slice_order.executed_quantity = executed_quantity
                slice_order.executed_price = executed_price
                slice_order.status = "filled" if executed_quantity >= slice_order.target_quantity else "partial"
                return True
                
        return False
    
    def get_execution_summary(self, parent_id: str) -> Optional[Dict]:
        """Get execution summary for a plan"""
        plan = self._active_plans.get(parent_id)
        if not plan:
            return None
            
        total_executed = sum(s.executed_quantity for s in plan.slices)
        total_value = sum(s.executed_quantity * s.executed_price for s in plan.slices)
        avg_price = total_value / total_executed if total_executed > 0 else 0
        
        return {
            "parent_id": parent_id,
            "code": plan.code,
            "strategy": plan.strategy.value,
            "target_quantity": plan.total_quantity,
            "executed_quantity": total_executed,
            "fill_rate": total_executed / plan.total_quantity if plan.total_quantity > 0 else 0,
            "average_price": round(avg_price, 2),
            "estimated_cost_bps": plan.estimated_cost_bps,
            "slices_completed": sum(1 for s in plan.slices if s.status == "filled"),
            "slices_total": len(plan.slices)
        }
    
    def cancel_plan(self, parent_id: str) -> bool:
        """Cancel an execution plan"""
        if parent_id in self._active_plans:
            plan = self._active_plans[parent_id]
            for slice_order in plan.slices:
                if slice_order.status == "pending":
                    slice_order.status = "cancelled"
            logger.info(f"Cancelled plan: {parent_id}")
            return True
        return False


if __name__ == "__main__":
    # Test execution core
    core = ExecutionCore()
    
    # Test VWAP plan
    print("=== VWAP Execution Plan ===")
    vwap_plan = core.create_vwap_plan(
        code="7203",
        side=OrderSide.BUY,
        total_quantity=100000,
        current_price=2500.0,
        daily_volume=5000000
    )
    
    print(f"Parent ID: {vwap_plan.parent_id}")
    print(f"Total Quantity: {vwap_plan.total_quantity:,}")
    print(f"Estimated Cost: {vwap_plan.estimated_cost_bps} bps")
    print(f"Number of Slices: {len(vwap_plan.slices)}")
    
    print("\nSlice Distribution:")
    for s in vwap_plan.slices[:5]:
        print(f"  {s.scheduled_time.strftime('%H:%M')}: {s.target_quantity:,} shares @ ¥{s.target_price:,.0f}")
    
    # Test Optimal plan
    print("\n=== Optimal Execution Plan ===")
    opt_plan = core.create_optimal_plan(
        code="7203",
        side=OrderSide.BUY,
        total_quantity=100000,
        current_price=2500.0,
        daily_volume=5000000,
        volatility=0.02,
        urgency=0.7
    )
    
    print(f"Parent ID: {opt_plan.parent_id}")
    print(f"Estimated Cost: {opt_plan.estimated_cost_bps} bps")
    
    print("\nOptimal Trajectory:")
    for s in opt_plan.slices[:5]:
        print(f"  {s.scheduled_time.strftime('%H:%M')}: {s.target_quantity:,} shares")
    
    # Test tick size
    print("\n=== Tick Size Examples ===")
    tick_mgr = TSETickSizeManager()
    test_prices = [500, 2500, 5000, 15000, 50000]
    for p in test_prices:
        tick = tick_mgr.get_tick_size("7203", p)
        print(f"  Price ¥{p:,}: Tick = ¥{tick}")
