"""
Asset Shield V4.0 - HMM Market Regime Detection
Dominance Phase Implementation

Hidden Markov Model for 3-state regime classification:
- BULL (Expansion): Low volatility, positive drift
- BEAR (Contraction): High volatility, negative drift
- CRISIS (Volatile): Extreme volatility, correlation spike

Author: Asset Shield V4 Team
Version: 4.0.0 (2026-02-05)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states"""
    BULL = "bull"           # Expansion - risk on
    BEAR = "bear"           # Contraction - risk off
    CRISIS = "crisis"       # Extreme volatility - exit all


@dataclass
class RegimeState:
    """Current regime detection state"""
    regime: MarketRegime
    probability: float
    volatility_20d: float
    volatility_60d: float
    drift_20d: float
    correlation_spike: bool
    timestamp: pd.Timestamp


@dataclass
class RegimeRiskParams:
    """Risk parameters for each regime"""
    max_exposure: float         # Maximum portfolio exposure
    max_single_position: float  # Maximum single position weight
    max_sector_exposure: float  # Maximum sector exposure
    rebalance_frequency: int    # Days between rebalances
    stop_loss_threshold: float  # Portfolio-level stop loss


# Default risk parameters per regime
REGIME_PARAMS = {
    MarketRegime.BULL: RegimeRiskParams(
        max_exposure=0.95,
        max_single_position=0.10,
        max_sector_exposure=0.25,
        rebalance_frequency=20,
        stop_loss_threshold=0.15
    ),
    MarketRegime.BEAR: RegimeRiskParams(
        max_exposure=0.60,
        max_single_position=0.08,
        max_sector_exposure=0.20,
        rebalance_frequency=10,
        stop_loss_threshold=0.10
    ),
    MarketRegime.CRISIS: RegimeRiskParams(
        max_exposure=0.0,  # Full cash
        max_single_position=0.0,
        max_sector_exposure=0.0,
        rebalance_frequency=1,
        stop_loss_threshold=0.05
    )
}


class HMMRegimeDetector:
    """
    Hidden Markov Model for Market Regime Detection

    Uses volatility and returns to classify market into 3 states:
    - BULL: Vol < vol_threshold_low, positive returns
    - BEAR: Vol > vol_threshold_low, negative returns
    - CRISIS: Vol > vol_threshold_high OR correlation spike
    """

    def __init__(
        self,
        vol_lookback_short: int = 20,
        vol_lookback_long: int = 60,
        vol_threshold_low: float = 0.15,   # Annualized
        vol_threshold_high: float = 0.35,  # Crisis threshold
        correlation_threshold: float = 0.8,
        min_history: int = 60
    ):
        """
        Initialize HMM Regime Detector.

        Args:
            vol_lookback_short: Short-term volatility window
            vol_lookback_long: Long-term volatility window
            vol_threshold_low: Low volatility threshold (annualized)
            vol_threshold_high: High volatility threshold (crisis)
            correlation_threshold: Correlation spike threshold
            min_history: Minimum data points needed
        """
        self.vol_lookback_short = vol_lookback_short
        self.vol_lookback_long = vol_lookback_long
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.correlation_threshold = correlation_threshold
        self.min_history = min_history

        # State tracking
        self.current_regime = MarketRegime.BULL
        self.regime_history: List[RegimeState] = []

        # Transition matrix (learned from data)
        self.transition_matrix = np.array([
            [0.95, 0.04, 0.01],  # From BULL
            [0.10, 0.85, 0.05],  # From BEAR
            [0.05, 0.15, 0.80]   # From CRISIS
        ])

        logger.info(
            f"HMMRegimeDetector initialized: "
            f"vol_low={vol_threshold_low:.0%}, vol_high={vol_threshold_high:.0%}"
        )

    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int
    ) -> float:
        """Calculate annualized volatility"""
        if len(returns) < window:
            return 0.0
        vol = returns.tail(window).std() * np.sqrt(252)
        return vol if not np.isnan(vol) else 0.0

    def calculate_drift(
        self,
        prices: pd.Series,
        window: int
    ) -> float:
        """Calculate price drift (annualized return)"""
        if len(prices) < window:
            return 0.0
        start_price = prices.iloc[-window]
        end_price = prices.iloc[-1]
        if start_price <= 0:
            return 0.0
        period_return = (end_price / start_price) - 1
        annualized = period_return * (252 / window)
        return annualized

    def detect_correlation_spike(
        self,
        market_returns: pd.Series,
        crypto_returns: Optional[pd.Series] = None,
        highbeta_returns: Optional[pd.Series] = None
    ) -> bool:
        """
        Detect correlation spike between crypto and high-beta stocks.
        Kill-switch trigger condition.
        """
        if crypto_returns is None or highbeta_returns is None:
            return False

        if len(market_returns) < 20:
            return False

        try:
            # Rolling correlation
            if len(crypto_returns) >= 20 and len(highbeta_returns) >= 20:
                corr = crypto_returns.tail(20).corr(highbeta_returns.tail(20))
                if not np.isnan(corr) and corr > self.correlation_threshold:
                    logger.warning(f"Correlation spike detected: {corr:.3f}")
                    return True
        except Exception as e:
            logger.debug(f"Correlation calculation error: {e}")

        return False

    def classify_regime(
        self,
        vol_short: float,
        vol_long: float,
        drift: float,
        correlation_spike: bool
    ) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on features.

        Returns:
            Tuple of (regime, probability)
        """
        # Crisis detection (highest priority)
        if correlation_spike or vol_short > self.vol_threshold_high:
            return MarketRegime.CRISIS, 0.95

        # Volatility regime detection
        vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

        # BULL: Low vol, positive drift
        if vol_short < self.vol_threshold_low and drift > 0:
            prob = min(0.95, 0.7 + 0.25 * (1 - vol_short / self.vol_threshold_low))
            return MarketRegime.BULL, prob

        # BEAR: High vol or negative drift
        if vol_short > self.vol_threshold_low or drift < -0.05:
            prob = min(0.90, 0.6 + 0.3 * (vol_short / self.vol_threshold_high))
            return MarketRegime.BEAR, prob

        # Default to current regime with lower confidence
        return self.current_regime, 0.5

    def update(
        self,
        market_prices: pd.Series,
        timestamp: pd.Timestamp,
        crypto_returns: Optional[pd.Series] = None,
        highbeta_returns: Optional[pd.Series] = None
    ) -> RegimeState:
        """
        Update regime detection with new data.

        Args:
            market_prices: Market index prices (e.g., TOPIX)
            timestamp: Current timestamp
            crypto_returns: Optional crypto returns for correlation
            highbeta_returns: Optional high-beta stock returns

        Returns:
            Current RegimeState
        """
        if len(market_prices) < self.min_history:
            # Not enough data - default to BULL
            state = RegimeState(
                regime=MarketRegime.BULL,
                probability=0.5,
                volatility_20d=0.0,
                volatility_60d=0.0,
                drift_20d=0.0,
                correlation_spike=False,
                timestamp=timestamp
            )
            return state

        # Calculate returns
        returns = market_prices.pct_change().dropna()

        # Calculate features
        vol_short = self.calculate_volatility(returns, self.vol_lookback_short)
        vol_long = self.calculate_volatility(returns, self.vol_lookback_long)
        drift = self.calculate_drift(market_prices, self.vol_lookback_short)

        # Check correlation spike (kill-switch)
        corr_spike = self.detect_correlation_spike(
            returns, crypto_returns, highbeta_returns
        )

        # Classify regime
        regime, probability = self.classify_regime(
            vol_short, vol_long, drift, corr_spike
        )

        # Apply transition smoothing (avoid regime flipping)
        if regime != self.current_regime:
            # Require high confidence to switch
            if probability < 0.7:
                regime = self.current_regime
                probability = 0.6

        self.current_regime = regime

        state = RegimeState(
            regime=regime,
            probability=probability,
            volatility_20d=vol_short,
            volatility_60d=vol_long,
            drift_20d=drift,
            correlation_spike=corr_spike,
            timestamp=timestamp
        )

        self.regime_history.append(state)

        return state

    def get_risk_params(self, regime: Optional[MarketRegime] = None) -> RegimeRiskParams:
        """Get risk parameters for current or specified regime"""
        if regime is None:
            regime = self.current_regime
        return REGIME_PARAMS[regime]

    def get_regime_stats(self) -> Dict[str, any]:
        """Get regime statistics from history"""
        if not self.regime_history:
            return {}

        regimes = [s.regime for s in self.regime_history]

        stats = {
            "total_days": len(regimes),
            "bull_days": sum(1 for r in regimes if r == MarketRegime.BULL),
            "bear_days": sum(1 for r in regimes if r == MarketRegime.BEAR),
            "crisis_days": sum(1 for r in regimes if r == MarketRegime.CRISIS),
            "regime_changes": sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1]),
            "current_regime": self.current_regime.value,
            "avg_vol_20d": np.mean([s.volatility_20d for s in self.regime_history]),
        }

        stats["bull_pct"] = stats["bull_days"] / stats["total_days"] if stats["total_days"] > 0 else 0
        stats["bear_pct"] = stats["bear_days"] / stats["total_days"] if stats["total_days"] > 0 else 0
        stats["crisis_pct"] = stats["crisis_days"] / stats["total_days"] if stats["total_days"] > 0 else 0

        return stats


class KillSwitch:
    """
    Emergency exit mechanism for crisis conditions.
    Triggers full de-leverage to cash.
    """

    def __init__(
        self,
        max_drawdown: float = 0.20,          # V4.1: 15% -> 20%
        correlation_threshold: float = 0.8,
        vol_spike_multiplier: float = 2.5,
        cooldown_days: int = 3               # V4.1: 5 -> 3 days
    ):
        """
        Initialize Kill Switch.

        Args:
            max_drawdown: Maximum drawdown before trigger
            correlation_threshold: Correlation spike threshold
            vol_spike_multiplier: Volatility spike detection multiplier
            cooldown_days: Days to wait after trigger before re-entry
        """
        self.max_drawdown = max_drawdown
        self.correlation_threshold = correlation_threshold
        self.vol_spike_multiplier = vol_spike_multiplier
        self.cooldown_days = cooldown_days

        self.triggered = False
        self.trigger_date: Optional[pd.Timestamp] = None
        self.trigger_reason: Optional[str] = None
        self.high_water_mark = 0.0

    def update_hwm(self, portfolio_value: float) -> None:
        """Update high water mark"""
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value

    def check_drawdown(self, portfolio_value: float) -> bool:
        """Check if drawdown exceeds threshold"""
        if self.high_water_mark <= 0:
            return False

        drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark
        return drawdown > self.max_drawdown

    def check_trigger(
        self,
        portfolio_value: float,
        current_date: pd.Timestamp,
        regime_state: RegimeState,
        vol_history: Optional[pd.Series] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if kill switch should be triggered.

        Returns:
            Tuple of (should_trigger, reason)
        """
        # Update high water mark
        self.update_hwm(portfolio_value)

        # Check cooldown
        if self.triggered and self.trigger_date is not None:
            days_since_trigger = (current_date - self.trigger_date).days
            if days_since_trigger < self.cooldown_days:
                return True, f"Cooldown ({days_since_trigger}/{self.cooldown_days} days)"

        # Check 1: Crisis regime
        if regime_state.regime == MarketRegime.CRISIS:
            self.triggered = True
            self.trigger_date = current_date
            self.trigger_reason = "Crisis regime detected"
            return True, self.trigger_reason

        # Check 2: Correlation spike
        if regime_state.correlation_spike:
            self.triggered = True
            self.trigger_date = current_date
            self.trigger_reason = "Correlation spike (crypto/high-beta)"
            return True, self.trigger_reason

        # Check 3: Max drawdown
        if self.check_drawdown(portfolio_value):
            self.triggered = True
            self.trigger_date = current_date
            drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark
            self.trigger_reason = f"Max drawdown exceeded ({drawdown:.1%})"
            return True, self.trigger_reason

        # Check 4: Volatility spike
        if vol_history is not None and len(vol_history) >= 60:
            current_vol = regime_state.volatility_20d
            avg_vol = vol_history.tail(60).mean()
            if current_vol > avg_vol * self.vol_spike_multiplier:
                self.triggered = True
                self.trigger_date = current_date
                self.trigger_reason = f"Volatility spike ({current_vol:.1%} vs avg {avg_vol:.1%})"
                return True, self.trigger_reason

        # No trigger - reset if previously triggered
        if self.triggered:
            self.triggered = False
            self.trigger_reason = None
            logger.info(f"Kill switch reset on {current_date}")

        return False, None

    def reset(self) -> None:
        """Force reset kill switch"""
        self.triggered = False
        self.trigger_date = None
        self.trigger_reason = None
        logger.info("Kill switch force reset")
