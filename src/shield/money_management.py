import requests
import logging
try:
    from shield.strategy_config import ALLOCATION
    except ImportError:
        ALLOCATION = {"SAINT": 0.15, "GHOST": 0.85}
    
    # Fallback for screening params if import fails
    try:
        from shield.strategy_config import SCREENING_MODE, SCREENING_PARAMS
    except ImportError:
        SCREENING_MODE = "INSTITUTIONAL"
        SCREENING_PARAMS = {
            "INSTITUTIONAL": {"Z_SAFE": 3.0, "F_TARGET": 8, "PEG_MAX": 1.0},
            "BOOTSTRAP": {"Z_SAFE": 1.81, "F_TARGET": 6, "PEG_MAX": 2.0}
        }
    
    class MoneyManager:
    """
    JÖRMUNGANDR V9.1 Dual-Ledger Money Manager.
    Handles Kelly-based sizing and splits execution into Saint (Audit) and Ghost (Alpha) ledgers.
    """
    def __init__(self, win_rate=0.65, win_loss_ratio=2.0):
        self.W = win_rate
        self.R = win_loss_ratio

    def update_stats(self, win_rate: float, win_loss_ratio: float):
        """Update strategy performance stats dynamically"""
        self.W = win_rate
        self.R = win_loss_ratio
        logging.info(f"Money Manager Stats Updated: W={self.W:.2f}, R={self.R:.2f}")

    def get_realtime_fx(self):
        try:
            url = "https://open.er-api.com/v6/latest/USD"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("rates", {}).get("JPY")
        except Exception as e:
            logging.error(f"Primary FX fetch failed: {e}")
        return 150.0  # Emergency Floor

    def get_position_size(self, total_capital_usd, ticker_verdict):
        fx_rate = self.get_realtime_fx()
        
        # Kelly logic
        kelly_f = ((self.W * (self.R + 1)) - 1) / self.R
        safe_kelly = max(0.0, kelly_f * 0.5)

        multipliers = {
            "INSTITUTIONAL BUY": 1.2,
            "AGGRESSIVE BUY": 1.0,
            "NEUTRAL WATCH": 0.0
        }
        final_leverage = safe_kelly * multipliers.get(ticker_verdict, 0.0)

        # 1. 全体の配分を算出
        total_allocation_jpy = total_capital_usd * final_leverage * fx_rate

        # 2. 二重台帳への分割執行 (Dual-Ledger Split)
        saint_jpy = total_allocation_jpy * ALLOCATION["SAINT"]
        ghost_jpy = total_allocation_jpy * ALLOCATION["GHOST"]

        return {
            "leverage": round(final_leverage, 4),
            "fx_rate": round(fx_rate, 4),
            "total_jpy": round(total_allocation_jpy, 0),
            "ledgers": {
                "SAINT_PUBLIC": round(saint_jpy, 0),  # 表：15%
                "GHOST_PRIVATE": round(ghost_jpy, 0)  # 裏：85%
            },
            "status": "DUAL_LEDGER_READY"
        }
