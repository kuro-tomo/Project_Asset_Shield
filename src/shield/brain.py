import math
import json
import os

class ShieldBrain:
    """
    Shield AI Core: Adaptive trading intelligence with volatility suppression.
    """
    def __init__(self, target_id="default"):
        # Unique memory file per asset for specialized learning
        self.memory_file = f"memory_{target_id}.json"
        self.lookback = 50
        self.adaptive_threshold = 0.65
        self.risk_penalty = 1.5
        self._load_memory()

    def _load_memory(self):
        """Loads specialized memory for the specific asset"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.adaptive_threshold = data.get("adaptive_threshold", 0.65)
                    self.risk_penalty = data.get("risk_penalty", 1.5)
            except: pass

    def _save_memory(self):
        """Persists specialized intelligence"""
        with open(self.memory_file, 'w') as f:
            json.dump({
                "adaptive_threshold": self.adaptive_threshold, 
                "risk_penalty": self.risk_penalty
            }, f)

    def calculate_indicators(self, prices):
        """Standard institutional-grade technical analysis"""
        if len(prices) < self.lookback: return 0.0, 50.0, 0.0
        sma = sum(prices) / len(prices)
        variance = sum((x - sma) ** 2 for x in prices) / len(prices)
        stdev = math.sqrt(variance)
        changes = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        gains = sum([x for x in changes[-14:] if x > 0])
        losses = sum([-x for x in changes[-14:] if x < 0])
        rsi = 100 - (100 / (1 + (gains/losses))) if losses > 0 else 100
        vol_ratio = stdev / (sma * 0.001) 
        return sma, rsi, vol_ratio

    def calculate_confidence(self, prices):
        """Generates trade confidence with volatility suppression"""
        sma, rsi, vol_ratio = self.calculate_indicators(prices)
        if sma == 0: return 0.0
        trend = (prices[-1] - sma) / sma
        score = trend * 20.0
        if rsi > 60 or rsi < 40: score *= 0.4
        confidence = score / (1.0 + (vol_ratio * self.risk_penalty))
        return round(float(max(min(confidence, 1.0), -1.0)), 4)

    def record_trade_result(self, pnl):
        """Evolves the specific brain based on asset performance"""
        if pnl < 0:
            self.adaptive_threshold = min(0.90, self.adaptive_threshold + 0.10)
            self.risk_penalty += 0.3
        else:
            self.adaptive_threshold = max(0.50, self.adaptive_threshold - 0.01)
            self.risk_penalty = max(1.0, self.risk_penalty - 0.05)
        self._save_memory()

    def get_threshold(self):
        return self.adaptive_threshold