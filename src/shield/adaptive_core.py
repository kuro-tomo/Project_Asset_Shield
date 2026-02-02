"""
Adaptive Core for Asset Shield V2
Layer 2: Signal Generation (Adaptive Core)

Implements self-regenerating algorithm that dynamically adapts to
market regime changes through:
- Real-time volatility monitoring
- Correlation matrix tracking
- Dynamic model re-calibration
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    CRISIS = "crisis"           # High vol, negative returns, correlation spike
    HIGH_VOL = "high_vol"       # High vol, mixed returns
    NORMAL = "normal"           # Average vol, trending
    LOW_VOL = "low_vol"         # Low vol, range-bound
    BULL_TREND = "bull_trend"   # Low-med vol, positive momentum
    BEAR_TREND = "bear_trend"   # Med-high vol, negative momentum


@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: float
    volatility: float
    correlation_avg: float
    momentum: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "volatility": self.volatility,
            "correlation_avg": self.correlation_avg,
            "momentum": self.momentum,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ModelParameters:
    """Adaptive model parameters"""
    # Risk parameters
    risk_multiplier: float = 1.0
    position_scale: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    
    # Signal thresholds
    entry_threshold: float = 0.65
    exit_threshold: float = 0.35
    
    # Volatility adjustments
    vol_target: float = 0.15  # 15% annualized target vol
    vol_lookback: int = 20
    
    # Correlation parameters
    correlation_threshold: float = 0.7
    decorrelation_bonus: float = 1.2
    
    # Regime-specific overrides
    regime_adjustments: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelParameters':
        return cls(**data)


class VolatilityTracker:
    """
    Real-time volatility monitoring with multiple timeframes.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 60]
        self._returns: deque = deque(maxlen=max(self.lookback_periods) + 1)
        self._prices: deque = deque(maxlen=max(self.lookback_periods) + 1)
        
    def add_price(self, price: float) -> None:
        """Add new price observation"""
        if self._prices:
            ret = np.log(price / self._prices[-1])
            self._returns.append(ret)
        self._prices.append(price)
        
    def get_volatility(self, lookback: int = 20) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            lookback: Number of periods for calculation
            
        Returns:
            Annualized volatility (assuming 252 trading days)
        """
        if len(self._returns) < lookback:
            return 0.0
            
        recent_returns = list(self._returns)[-lookback:]
        daily_vol = np.std(recent_returns)
        return daily_vol * np.sqrt(252)
    
    def get_volatility_regime(self) -> Tuple[str, float]:
        """
        Classify current volatility regime.
        
        Returns:
            Tuple of (regime_label, current_vol)
        """
        vol = self.get_volatility(20)
        
        if vol > 0.40:
            return "extreme", vol
        elif vol > 0.25:
            return "high", vol
        elif vol > 0.15:
            return "normal", vol
        else:
            return "low", vol
    
    def get_vol_of_vol(self, lookback: int = 20) -> float:
        """Calculate volatility of volatility (vol clustering indicator)"""
        if len(self._returns) < lookback * 2:
            return 0.0
            
        vols = []
        for i in range(lookback):
            start_idx = len(self._returns) - lookback - i
            end_idx = len(self._returns) - i
            if start_idx >= 0:
                period_returns = list(self._returns)[start_idx:end_idx]
                vols.append(np.std(period_returns))
                
        return np.std(vols) if vols else 0.0


class CorrelationMatrix:
    """
    Dynamic correlation matrix tracking for portfolio risk management.
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self._returns: Dict[str, deque] = {}
        
    def add_return(self, code: str, ret: float) -> None:
        """Add return observation for a stock"""
        if code not in self._returns:
            self._returns[code] = deque(maxlen=self.lookback)
        self._returns[code].append(ret)
        
    def get_correlation(self, code1: str, code2: str) -> float:
        """Calculate pairwise correlation"""
        if code1 not in self._returns or code2 not in self._returns:
            return 0.0
            
        r1 = list(self._returns[code1])
        r2 = list(self._returns[code2])
        
        min_len = min(len(r1), len(r2))
        if min_len < 10:
            return 0.0
            
        return np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
    
    def get_correlation_matrix(self, codes: List[str]) -> np.ndarray:
        """Calculate full correlation matrix for given codes"""
        n = len(codes)
        matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.get_correlation(codes[i], codes[j])
                matrix[i, j] = corr
                matrix[j, i] = corr
                
        return matrix
    
    def get_average_correlation(self, codes: List[str]) -> float:
        """Calculate average pairwise correlation"""
        if len(codes) < 2:
            return 0.0
            
        matrix = self.get_correlation_matrix(codes)
        # Extract upper triangle (excluding diagonal)
        upper_tri = matrix[np.triu_indices(len(codes), k=1)]
        return np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
    
    def detect_correlation_spike(self, codes: List[str], threshold: float = 0.8) -> bool:
        """Detect if correlations have spiked (crisis indicator)"""
        avg_corr = self.get_average_correlation(codes)
        return avg_corr > threshold


class AdaptiveCore:
    """
    Self-Regenerating Adaptive Algorithm Core
    
    Dynamically adjusts model parameters based on:
    1. Current market regime
    2. Real-time volatility
    3. Correlation dynamics
    4. Recent performance feedback
    """
    
    REGIME_PARAMS = {
        MarketRegime.CRISIS: {
            "risk_multiplier": 0.3,
            "position_scale": 0.5,
            "stop_loss_pct": 0.01,
            "entry_threshold": 0.85
        },
        MarketRegime.HIGH_VOL: {
            "risk_multiplier": 0.5,
            "position_scale": 0.7,
            "stop_loss_pct": 0.015,
            "entry_threshold": 0.75
        },
        MarketRegime.NORMAL: {
            "risk_multiplier": 1.0,
            "position_scale": 1.0,
            "stop_loss_pct": 0.02,
            "entry_threshold": 0.65
        },
        MarketRegime.LOW_VOL: {
            "risk_multiplier": 1.2,
            "position_scale": 1.1,
            "stop_loss_pct": 0.025,
            "entry_threshold": 0.60
        },
        MarketRegime.BULL_TREND: {
            "risk_multiplier": 1.1,
            "position_scale": 1.0,
            "stop_loss_pct": 0.02,
            "entry_threshold": 0.60
        },
        MarketRegime.BEAR_TREND: {
            "risk_multiplier": 0.6,
            "position_scale": 0.8,
            "stop_loss_pct": 0.015,
            "entry_threshold": 0.80
        }
    }
    
    def __init__(
        self,
        state_file: str = "adaptive_state.json",
        vol_target: float = 0.15
    ):
        """
        Initialize Adaptive Core.
        
        Args:
            state_file: Path to persist adaptive state
            vol_target: Target annualized volatility
        """
        self.state_file = state_file
        self.vol_target = vol_target
        
        # Initialize trackers
        self.vol_tracker = VolatilityTracker()
        self.corr_matrix = CorrelationMatrix()
        
        # Current state
        self.current_regime: Optional[RegimeState] = None
        self.parameters = ModelParameters(vol_target=vol_target)
        
        # Performance tracking for feedback loop
        self._trade_history: List[Dict] = []
        self._regime_history: List[RegimeState] = []
        
        # Load persisted state
        self._load_state()
        
    def _load_state(self) -> None:
        """Load persisted adaptive state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.parameters = ModelParameters.from_dict(data.get("parameters", {}))
                    logger.info(f"Loaded adaptive state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
                
    def _save_state(self) -> None:
        """Persist adaptive state"""
        try:
            data = {
                "parameters": self.parameters.to_dict(),
                "current_regime": self.current_regime.to_dict() if self.current_regime else None,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    def update_market_data(
        self,
        index_price: float,
        stock_returns: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update with new market data.
        
        Args:
            index_price: Market index price (e.g., TOPIX, Nikkei)
            stock_returns: Dict mapping stock codes to daily returns
        """
        # Update volatility tracker
        self.vol_tracker.add_price(index_price)
        
        # Update correlation matrix
        if stock_returns:
            for code, ret in stock_returns.items():
                self.corr_matrix.add_return(code, ret)
                
    def classify_regime(self, stock_codes: Optional[List[str]] = None) -> RegimeState:
        """
        Classify current market regime.
        
        Args:
            stock_codes: List of stock codes for correlation analysis
            
        Returns:
            RegimeState with current classification
        """
        # Get volatility metrics
        vol = self.vol_tracker.get_volatility(20)
        vol_regime, _ = self.vol_tracker.get_volatility_regime()
        
        # Get correlation metrics
        avg_corr = 0.0
        if stock_codes and len(stock_codes) >= 2:
            avg_corr = self.corr_matrix.get_average_correlation(stock_codes)
        
        # Calculate momentum (simple: recent returns)
        momentum = 0.0
        if len(self.vol_tracker._returns) >= 20:
            recent_returns = list(self.vol_tracker._returns)[-20:]
            momentum = sum(recent_returns)
        
        # Regime classification logic
        # Adjusted thresholds for earlier detection (2026 Standard)
        if vol > 0.30 and avg_corr > 0.65:
            regime = MarketRegime.CRISIS
            confidence = min(1.0, vol / 0.45 * avg_corr)
        elif vol > 0.20:
            regime = MarketRegime.HIGH_VOL
            confidence = min(1.0, vol / 0.30)
        elif vol < 0.12:
            regime = MarketRegime.LOW_VOL
            confidence = min(1.0, (0.15 - vol) / 0.10)
        elif momentum > 0.05:
            regime = MarketRegime.BULL_TREND
            confidence = min(1.0, momentum / 0.10)
        elif momentum < -0.05:
            regime = MarketRegime.BEAR_TREND
            confidence = min(1.0, abs(momentum) / 0.10)
        else:
            regime = MarketRegime.NORMAL
            confidence = 0.7
            
        state = RegimeState(
            regime=regime,
            confidence=round(confidence, 4),
            volatility=round(vol, 4),
            correlation_avg=round(avg_corr, 4),
            momentum=round(momentum, 4),
            timestamp=datetime.now()
        )
        
        self.current_regime = state
        self._regime_history.append(state)
        
        return state
    
    def recalibrate(self, stock_codes: Optional[List[str]] = None) -> ModelParameters:
        """
        Recalibrate model parameters based on current regime.
        
        Args:
            stock_codes: List of stock codes for analysis
            
        Returns:
            Updated ModelParameters
        """
        # Classify current regime
        regime_state = self.classify_regime(stock_codes)
        
        logger.info(f"Regime: {regime_state.regime.value} (confidence: {regime_state.confidence})")
        
        # Get regime-specific parameters
        regime_params = self.REGIME_PARAMS.get(regime_state.regime, {})
        
        # Apply regime adjustments with confidence weighting
        confidence = regime_state.confidence
        base_params = self.REGIME_PARAMS[MarketRegime.NORMAL]
        
        for key, regime_value in regime_params.items():
            base_value = base_params.get(key, getattr(self.parameters, key))
            # Blend between base and regime-specific based on confidence
            blended = base_value + (regime_value - base_value) * confidence
            setattr(self.parameters, key, round(blended, 4))
        
        # Volatility scaling
        current_vol = regime_state.volatility
        if current_vol > 0:
            vol_scale = self.vol_target / current_vol
            vol_scale = max(0.5, min(2.0, vol_scale))  # Clamp to reasonable range
            self.parameters.position_scale *= vol_scale
            self.parameters.position_scale = round(
                max(0.1, min(1.5, self.parameters.position_scale)), 4
            )
        
        # Save updated state
        self._save_state()
        
        return self.parameters
    
    def record_trade(self, trade_result: Dict) -> None:
        """
        Record trade result for performance feedback.
        
        Args:
            trade_result: Dict with trade details including PnL
        """
        trade_result["timestamp"] = datetime.now().isoformat()
        trade_result["regime"] = self.current_regime.regime.value if self.current_regime else "unknown"
        self._trade_history.append(trade_result)
        
        # Trigger feedback adjustment if enough trades
        if len(self._trade_history) >= 10:
            self._apply_performance_feedback()
            
    def _apply_performance_feedback(self) -> None:
        """Apply performance-based parameter adjustments"""
        recent_trades = self._trade_history[-20:]
        
        if not recent_trades:
            return
            
        # Calculate win rate and average PnL
        wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        win_rate = wins / len(recent_trades)
        avg_pnl = np.mean([t.get("pnl", 0) for t in recent_trades])
        
        # Adjust based on performance
        if win_rate < 0.4:
            # Poor performance: tighten risk
            self.parameters.entry_threshold = min(0.90, self.parameters.entry_threshold + 0.05)
            self.parameters.risk_multiplier = max(0.3, self.parameters.risk_multiplier - 0.1)
            logger.info("Performance feedback: Tightening risk parameters")
        elif win_rate > 0.6 and avg_pnl > 0:
            # Good performance: slightly loosen
            self.parameters.entry_threshold = max(0.50, self.parameters.entry_threshold - 0.02)
            self.parameters.risk_multiplier = min(1.2, self.parameters.risk_multiplier + 0.05)
            logger.info("Performance feedback: Optimizing parameters")
            
        self._save_state()
        
    def get_signal_adjustment(self, base_signal: float) -> float:
        """
        Adjust a base signal based on current regime.
        
        Args:
            base_signal: Raw signal value in [-1, 1]
            
        Returns:
            Adjusted signal value
        """
        if not self.current_regime:
            return base_signal
            
        # Apply risk multiplier
        adjusted = base_signal * self.parameters.risk_multiplier
        
        # Apply position scale
        adjusted *= self.parameters.position_scale
        
        # Clamp to valid range
        return max(-1.0, min(1.0, adjusted))
    
    def should_enter(self, signal_strength: float) -> bool:
        """Check if signal strength meets entry threshold"""
        return abs(signal_strength) >= self.parameters.entry_threshold
    
    def should_exit(self, signal_strength: float) -> bool:
        """Check if signal strength falls below exit threshold"""
        return abs(signal_strength) < self.parameters.exit_threshold
    
    def get_position_size(self, base_size: float, signal_strength: float) -> float:
        """
        Calculate position size with regime adjustments.
        
        Args:
            base_size: Base position size
            signal_strength: Signal strength [0, 1]
            
        Returns:
            Adjusted position size
        """
        adjusted_size = base_size * self.parameters.position_scale
        adjusted_size *= self.parameters.risk_multiplier
        adjusted_size *= signal_strength
        
        return round(adjusted_size, 2)
    
    def get_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calculate stop loss price based on current parameters"""
        if direction == "LONG":
            return entry_price * (1 - self.parameters.stop_loss_pct)
        else:
            return entry_price * (1 + self.parameters.stop_loss_pct)
            
    def get_take_profit(self, entry_price: float, direction: str) -> float:
        """Calculate take profit price based on current parameters"""
        if direction == "LONG":
            return entry_price * (1 + self.parameters.take_profit_pct)
        else:
            return entry_price * (1 - self.parameters.take_profit_pct)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current adaptive core status"""
        return {
            "regime": self.current_regime.to_dict() if self.current_regime else None,
            "parameters": self.parameters.to_dict(),
            "trade_count": len(self._trade_history),
            "regime_history_count": len(self._regime_history)
        }


if __name__ == "__main__":
    # Test adaptive core
    core = AdaptiveCore(state_file="test_adaptive_state.json")
    
    # Simulate market data
    np.random.seed(42)
    prices = [30000]  # Starting Nikkei price
    
    for i in range(100):
        # Random walk with drift
        ret = np.random.normal(0.0005, 0.015)
        prices.append(prices[-1] * (1 + ret))
        core.update_market_data(prices[-1])
    
    # Classify regime
    regime = core.classify_regime()
    print(f"=== Adaptive Core Status ===")
    print(f"Regime: {regime.regime.value}")
    print(f"Confidence: {regime.confidence}")
    print(f"Volatility: {regime.volatility:.2%}")
    print(f"Momentum: {regime.momentum:.4f}")
    
    # Recalibrate
    params = core.recalibrate()
    print(f"\n=== Recalibrated Parameters ===")
    print(f"Risk Multiplier: {params.risk_multiplier}")
    print(f"Position Scale: {params.position_scale}")
    print(f"Entry Threshold: {params.entry_threshold}")
    print(f"Stop Loss: {params.stop_loss_pct:.2%}")
    
    # Test signal adjustment
    base_signal = 0.7
    adjusted = core.get_signal_adjustment(base_signal)
    print(f"\n=== Signal Adjustment ===")
    print(f"Base Signal: {base_signal}")
    print(f"Adjusted Signal: {adjusted:.4f}")
    print(f"Should Enter: {core.should_enter(adjusted)}")
