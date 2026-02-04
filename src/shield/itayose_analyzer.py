"""
Itayose Analyzer for Asset Shield V2
Layer 1: Data Ingestion (Microstructure) - Pre-Market Analysis

Analyzes pre-market order flow (8:00-9:00 JST) to extract:
- Order Flow Imbalance (OFI)
- Institutional demand/supply signals
- Opening price prediction confidence
"""

import logging
from datetime import datetime, time as dt_time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """TSE Market Phases"""
    PRE_OPEN = "pre_open"           # 8:00-9:00 (Itayose period)
    MORNING_SESSION = "morning"      # 9:00-11:30
    LUNCH_BREAK = "lunch"           # 11:30-12:30
    AFTERNOON_SESSION = "afternoon"  # 12:30-15:00
    AFTER_HOURS = "after_hours"     # 15:00-


@dataclass
class OrderBookSnapshot:
    """Single order book snapshot during Itayose"""
    timestamp: datetime
    code: str
    # Bid side (buy orders)
    bid_prices: List[float] = field(default_factory=list)
    bid_volumes: List[int] = field(default_factory=list)
    # Ask side (sell orders)
    ask_prices: List[float] = field(default_factory=list)
    ask_volumes: List[int] = field(default_factory=list)
    # Indicative price (expected opening price)
    indicative_price: Optional[float] = None
    indicative_volume: Optional[int] = None


@dataclass
class OFISignal:
    """Order Flow Imbalance Signal"""
    code: str
    timestamp: datetime
    ofi_value: float              # Positive = buy pressure, Negative = sell pressure
    ofi_normalized: float         # Normalized to [-1, 1]
    confidence: float             # Signal confidence [0, 1]
    institutional_flag: bool      # True if pattern suggests institutional activity
    predicted_direction: str      # "BULLISH", "BEARISH", "NEUTRAL"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ItayoseAnalyzer:
    """
    Itayose (Opening Auction) Pre-Market Analyzer
    
    Extracts institutional order flow signals from pre-market
    order book dynamics during the 8:00-9:00 JST window.
    
    Key Metrics:
    1. Order Flow Imbalance (OFI): Measures buy/sell pressure asymmetry
    2. Volume-Weighted Price Impact: Estimates opening price direction
    3. Institutional Detection: Identifies large block order patterns
    """
    
    # TSE Itayose window (JST)
    ITAYOSE_START = dt_time(8, 0, 0)
    ITAYOSE_END = dt_time(9, 0, 0)
    
    # Thresholds for institutional detection
    INSTITUTIONAL_VOLUME_THRESHOLD = 100000  # 100K+ shares
    LARGE_ORDER_RATIO = 0.05  # 5% of daily average volume
    
    def __init__(self, daily_avg_volume: Optional[Dict[str, int]] = None):
        """
        Initialize Itayose Analyzer.
        
        Args:
            daily_avg_volume: Dict mapping stock codes to average daily volume
        """
        self.daily_avg_volume = daily_avg_volume or {}
        self._snapshots: Dict[str, List[OrderBookSnapshot]] = {}
        
    def add_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add order book snapshot for analysis"""
        if snapshot.code not in self._snapshots:
            self._snapshots[snapshot.code] = []
        self._snapshots[snapshot.code].append(snapshot)
        
    def parse_jquants_orderbook(self, raw_data: List[Dict]) -> List[OrderBookSnapshot]:
        """
        Parse J-Quants orderbook API response into OrderBookSnapshot objects.
        
        Args:
            raw_data: Raw orderbook data from J-Quants API
            
        Returns:
            List of OrderBookSnapshot objects
        """
        snapshots = []
        
        for entry in raw_data:
            try:
                # Parse timestamp
                ts_str = entry.get("Time", "")
                if ts_str:
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.now()
                
                # Extract bid/ask levels (J-Quants provides up to 10 levels)
                bid_prices = []
                bid_volumes = []
                ask_prices = []
                ask_volumes = []
                
                for i in range(1, 11):
                    bp = entry.get(f"BidPrice{i}")
                    bv = entry.get(f"BidVolume{i}")
                    ap = entry.get(f"AskPrice{i}")
                    av = entry.get(f"AskVolume{i}")
                    
                    if bp is not None and bv is not None:
                        bid_prices.append(float(bp))
                        bid_volumes.append(int(bv))
                    if ap is not None and av is not None:
                        ask_prices.append(float(ap))
                        ask_volumes.append(int(av))
                
                snapshot = OrderBookSnapshot(
                    timestamp=timestamp,
                    code=entry.get("Code", ""),
                    bid_prices=bid_prices,
                    bid_volumes=bid_volumes,
                    ask_prices=ask_prices,
                    ask_volumes=ask_volumes,
                    indicative_price=entry.get("IndicativePrice"),
                    indicative_volume=entry.get("IndicativeVolume")
                )
                snapshots.append(snapshot)
                self.add_snapshot(snapshot)
                
            except Exception as e:
                logger.warning(f"Failed to parse orderbook entry: {e}")
                continue
                
        return snapshots
    
    def calculate_ofi(self, snapshots: List[OrderBookSnapshot]) -> float:
        """
        Calculate Order Flow Imbalance from sequential snapshots.
        
        OFI = Σ(ΔBid_Volume × I(ΔBid_Price ≥ 0)) - Σ(ΔAsk_Volume × I(ΔAsk_Price ≤ 0))
        
        Where I() is indicator function.
        
        Args:
            snapshots: List of sequential order book snapshots
            
        Returns:
            OFI value (positive = buy pressure, negative = sell pressure)
        """
        if len(snapshots) < 2:
            return 0.0
            
        ofi = 0.0
        
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            
            # Best bid changes
            if prev.bid_prices and curr.bid_prices:
                prev_best_bid = prev.bid_prices[0]
                curr_best_bid = curr.bid_prices[0]
                prev_bid_vol = prev.bid_volumes[0] if prev.bid_volumes else 0
                curr_bid_vol = curr.bid_volumes[0] if curr.bid_volumes else 0
                
                if curr_best_bid > prev_best_bid:
                    # Price improved, add new volume
                    ofi += curr_bid_vol
                elif curr_best_bid == prev_best_bid:
                    # Same price, add volume change
                    ofi += (curr_bid_vol - prev_bid_vol)
                else:
                    # Price dropped, subtract volume
                    ofi -= prev_bid_vol
            
            # Best ask changes
            if prev.ask_prices and curr.ask_prices:
                prev_best_ask = prev.ask_prices[0]
                curr_best_ask = curr.ask_prices[0]
                prev_ask_vol = prev.ask_volumes[0] if prev.ask_volumes else 0
                curr_ask_vol = curr.ask_volumes[0] if curr.ask_volumes else 0
                
                if curr_best_ask < prev_best_ask:
                    # Price improved (lower), subtract new volume
                    ofi -= curr_ask_vol
                elif curr_best_ask == prev_best_ask:
                    # Same price, subtract volume change
                    ofi -= (curr_ask_vol - prev_ask_vol)
                else:
                    # Price increased, add volume
                    ofi += prev_ask_vol
                    
        return ofi
    
    def calculate_volume_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """
        Calculate instantaneous volume imbalance at best levels.
        
        Returns:
            Imbalance ratio in [-1, 1] where positive = buy pressure
        """
        total_bid = sum(snapshot.bid_volumes) if snapshot.bid_volumes else 0
        total_ask = sum(snapshot.ask_volumes) if snapshot.ask_volumes else 0
        
        if total_bid + total_ask == 0:
            return 0.0
            
        return (total_bid - total_ask) / (total_bid + total_ask)
    
    def detect_institutional_activity(
        self, 
        snapshots: List[OrderBookSnapshot],
        code: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect potential institutional order activity patterns.
        
        Institutional signatures:
        1. Large block orders appearing/disappearing
        2. Consistent directional pressure
        3. Volume significantly above average
        
        Args:
            snapshots: Order book snapshots
            code: Stock code
            
        Returns:
            Tuple of (is_institutional, metadata)
        """
        if not snapshots:
            return False, {}
            
        metadata = {
            "large_orders_detected": 0,
            "max_single_order": 0,
            "directional_consistency": 0.0,
            "volume_ratio": 0.0
        }
        
        # Check for large orders
        daily_avg = self.daily_avg_volume.get(code, 1000000)
        threshold = max(self.INSTITUTIONAL_VOLUME_THRESHOLD, daily_avg * self.LARGE_ORDER_RATIO)
        
        imbalances = []
        for snap in snapshots:
            # Check bid side for large orders
            for vol in snap.bid_volumes:
                if vol >= threshold:
                    metadata["large_orders_detected"] += 1
                    metadata["max_single_order"] = max(metadata["max_single_order"], vol)
                    
            # Check ask side for large orders
            for vol in snap.ask_volumes:
                if vol >= threshold:
                    metadata["large_orders_detected"] += 1
                    metadata["max_single_order"] = max(metadata["max_single_order"], vol)
            
            imbalances.append(self.calculate_volume_imbalance(snap))
        
        # Calculate directional consistency
        if imbalances:
            positive_count = sum(1 for x in imbalances if x > 0)
            negative_count = sum(1 for x in imbalances if x < 0)
            total = len(imbalances)
            metadata["directional_consistency"] = max(positive_count, negative_count) / total
        
        # Volume ratio vs daily average
        total_volume = sum(
            sum(s.bid_volumes) + sum(s.ask_volumes) 
            for s in snapshots
        )
        metadata["volume_ratio"] = total_volume / daily_avg if daily_avg > 0 else 0
        
        # Institutional flag: large orders + consistent direction + high volume
        is_institutional = (
            metadata["large_orders_detected"] >= 3 and
            metadata["directional_consistency"] >= 0.7 and
            metadata["volume_ratio"] >= 0.1
        )
        
        return is_institutional, metadata
    
    def analyze(self, code: str) -> Optional[OFISignal]:
        """
        Perform complete Itayose analysis for a stock.
        
        Args:
            code: Stock code
            
        Returns:
            OFISignal with analysis results
        """
        snapshots = self._snapshots.get(code, [])
        
        if not snapshots:
            logger.warning(f"No snapshots available for {code}")
            return None
        
        # Sort by timestamp
        snapshots = sorted(snapshots, key=lambda x: x.timestamp)
        
        # Calculate OFI
        ofi_value = self.calculate_ofi(snapshots)
        
        # Normalize OFI to [-1, 1] using tanh
        ofi_normalized = np.tanh(ofi_value / 1000000)  # Scale factor for typical volumes
        
        # Detect institutional activity
        is_institutional, inst_metadata = self.detect_institutional_activity(snapshots, code)
        
        # Calculate confidence based on data quality
        confidence = min(1.0, len(snapshots) / 60)  # More snapshots = higher confidence
        if is_institutional:
            confidence = min(1.0, confidence * 1.2)  # Boost for institutional signal
        
        # Determine predicted direction
        if ofi_normalized > 0.3:
            direction = "BULLISH"
        elif ofi_normalized < -0.3:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        # Final volume imbalance from last snapshot
        final_imbalance = self.calculate_volume_imbalance(snapshots[-1])
        
        return OFISignal(
            code=code,
            timestamp=datetime.now(),
            ofi_value=ofi_value,
            ofi_normalized=round(ofi_normalized, 4),
            confidence=round(confidence, 4),
            institutional_flag=is_institutional,
            predicted_direction=direction,
            metadata={
                "snapshot_count": len(snapshots),
                "final_volume_imbalance": round(final_imbalance, 4),
                "institutional_details": inst_metadata,
                "indicative_price": snapshots[-1].indicative_price
            }
        )
    
    def analyze_batch(self, codes: List[str]) -> Dict[str, OFISignal]:
        """
        Analyze multiple stocks and return ranked signals.
        
        Args:
            codes: List of stock codes
            
        Returns:
            Dict mapping codes to OFISignal objects
        """
        results = {}
        
        for code in codes:
            signal = self.analyze(code)
            if signal:
                results[code] = signal
                
        return results
    
    def get_top_signals(
        self, 
        signals: Dict[str, OFISignal],
        direction: str = "BULLISH",
        top_n: int = 10
    ) -> List[OFISignal]:
        """
        Get top N signals by strength for a given direction.
        
        Args:
            signals: Dict of OFI signals
            direction: "BULLISH" or "BEARISH"
            top_n: Number of top signals to return
            
        Returns:
            List of top OFISignal objects sorted by strength
        """
        filtered = [
            s for s in signals.values() 
            if s.predicted_direction == direction
        ]
        
        # Sort by absolute OFI value * confidence
        sorted_signals = sorted(
            filtered,
            key=lambda x: abs(x.ofi_normalized) * x.confidence,
            reverse=True
        )
        
        return sorted_signals[:top_n]
    
    def clear_snapshots(self, code: Optional[str] = None) -> None:
        """Clear stored snapshots"""
        if code:
            self._snapshots.pop(code, None)
        else:
            self._snapshots.clear()


def create_mock_snapshots(code: str, n_snapshots: int = 30) -> List[OrderBookSnapshot]:
    """
    Create mock order book snapshots for testing.
    
    Args:
        code: Stock code
        n_snapshots: Number of snapshots to generate
        
    Returns:
        List of mock OrderBookSnapshot objects
    """
    snapshots = []
    base_price = 3500.0
    base_time = datetime.now().replace(hour=8, minute=0, second=0)
    
    for i in range(n_snapshots):
        # Simulate price drift with noise
        drift = np.random.normal(0.5, 2.0)  # Slight upward bias
        price = base_price + drift * i
        
        # Generate bid/ask levels
        bid_prices = [price - j * 1.0 for j in range(5)]
        ask_prices = [price + j * 1.0 for j in range(5)]
        
        # Generate volumes with some randomness
        bid_volumes = [int(np.random.exponential(50000)) for _ in range(5)]
        ask_volumes = [int(np.random.exponential(45000)) for _ in range(5)]
        
        snapshot = OrderBookSnapshot(
            timestamp=base_time.replace(minute=i * 2),
            code=code,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            indicative_price=price,
            indicative_volume=sum(bid_volumes[:3])
        )
        snapshots.append(snapshot)
        
    return snapshots


if __name__ == "__main__":
    # Test with mock data
    analyzer = ItayoseAnalyzer(daily_avg_volume={"7203": 5000000})
    
    # Generate mock snapshots
    mock_snapshots = create_mock_snapshots("7203", 30)
    for snap in mock_snapshots:
        analyzer.add_snapshot(snap)
    
    # Analyze
    signal = analyzer.analyze("7203")
    
    if signal:
        print(f"=== Itayose Analysis: {signal.code} ===")
        print(f"OFI Value: {signal.ofi_value:,.0f}")
        print(f"OFI Normalized: {signal.ofi_normalized}")
        print(f"Confidence: {signal.confidence}")
        print(f"Direction: {signal.predicted_direction}")
        print(f"Institutional: {signal.institutional_flag}")
        print(f"Metadata: {signal.metadata}")
