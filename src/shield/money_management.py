"""
Money Management Module for Asset Shield V2
Dual-Ledger Kelly Criterion Position Sizing

Implements:
- Kelly-based position sizing with safety margin
- Dual-ledger execution split (Saint/Ghost)
- Real-time FX rate integration for USD/JPY conversion

Author: Asset Shield V2 Team
Version: 2.0.0 (2026-02-03)
"""

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
    Dual-Ledger Money Manager with Kelly Criterion.

    Handles Kelly-based sizing and splits execution into:
    - Saint (Public/Audit) ledger: 15% allocation
    - Ghost (Private/Alpha) ledger: 85% allocation
    """

    def __init__(self, win_rate=0.65, win_loss_ratio=2.0):
        """
        Initialize Money Manager.

        Args:
            win_rate: Historical win rate (default: 0.65)
            win_loss_ratio: Average win / average loss ratio (default: 2.0)
        """
        self.W = win_rate
        self.R = win_loss_ratio

    def update_stats(self, win_rate: float, win_loss_ratio: float):
        """Update strategy performance stats dynamically"""
        self.W = win_rate
        self.R = win_loss_ratio
        logging.info(f"Money Manager Stats Updated: W={self.W:.2f}, R={self.R:.2f}")

    def get_realtime_fx(self):
        """
        Fetch real-time USD/JPY exchange rate.

        Returns:
            float: Current USD/JPY rate, or 150.0 as fallback
        """
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
        """
        Calculate position size using Kelly Criterion with dual-ledger split.

        Args:
            total_capital_usd: Total capital in USD
            ticker_verdict: Screening verdict (INSTITUTIONAL BUY, AGGRESSIVE BUY, etc.)

        Returns:
            dict: Position sizing details including ledger split
        """
        fx_rate = self.get_realtime_fx()

        # Kelly Criterion: f* = (W(R+1) - 1) / R
        kelly_f = ((self.W * (self.R + 1)) - 1) / self.R
        safe_kelly = max(0.0, kelly_f * 0.5)  # Half-Kelly for safety

        multipliers = {
            "INSTITUTIONAL BUY": 1.2,
            "AGGRESSIVE BUY": 1.0,
            "NEUTRAL WATCH": 0.0
        }
        final_leverage = safe_kelly * multipliers.get(ticker_verdict, 0.0)

        # Step 1: Calculate total allocation
        total_allocation_jpy = total_capital_usd * final_leverage * fx_rate

        # Step 2: Dual-Ledger Split execution
        saint_jpy = total_allocation_jpy * ALLOCATION["SAINT"]
        ghost_jpy = total_allocation_jpy * ALLOCATION["GHOST"]

        return {
            "leverage": round(final_leverage, 4),
            "fx_rate": round(fx_rate, 4),
            "total_jpy": round(total_allocation_jpy, 0),
            "ledgers": {
                "SAINT_PUBLIC": round(saint_jpy, 0),    # Public ledger: 15%
                "GHOST_PRIVATE": round(ghost_jpy, 0)    # Private ledger: 85%
            },
            "status": "DUAL_LEDGER_READY"
        }
