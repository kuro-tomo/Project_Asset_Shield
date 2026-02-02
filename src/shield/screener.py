import pandas as pd
import numpy as np

try:
    from shield.strategy_config import SCREENING_MODE, SCREENING_PARAMS
except ImportError:
    SCREENING_MODE = "INSTITUTIONAL"
    SCREENING_PARAMS = {
        "INSTITUTIONAL": {"Z_SAFE": 3.0, "F_TARGET": 8, "PEG_MAX": 1.0, "ALLOW_AGGRESSIVE": False},
        "BOOTSTRAP": {"Z_SAFE": 1.81, "F_TARGET": 6, "PEG_MAX": 2.0, "ALLOW_AGGRESSIVE": True}
    }

class FinancialTrinity:
    """
    Financial Trinity Engine: Z-Score, F-Score, and Value Assessment.
    Core logic to eliminate bankruptcy risk and target 8/9 quality stocks.
    """
    def __init__(self):
        # Load configuration
        self.mode = SCREENING_MODE
        self.params = SCREENING_PARAMS.get(self.mode, SCREENING_PARAMS["INSTITUTIONAL"])
        
        # Thresholds defined for the Strategy
        self.Z_SAFE = self.params["Z_SAFE"]
        self.Z_DANGER = 1.81 # Constant
        self.F_TARGET = self.params["F_TARGET"]
        self.PEG_MAX = self.params.get("PEG_MAX", 1.0)
        self.ALLOW_AGGRESSIVE = self.params.get("ALLOW_AGGRESSIVE", False)

    def parse_jquants_financials(self, statements_data, market_cap=None):
        """
        Parse J-Quants financial statements API response into standardized format.
        
        Args:
            statements_data: Dictionary from J-Quants API (single quarter/year)
            market_cap: Current market capitalization (required for Z-Score)
        
        Returns:
            Dictionary with standardized keys
        """
        try:
            # Helper to safe float conversion
            def get_val(key, default=0.0):
                val = statements_data.get(key)
                if val is None or val == "":
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            # J-Quants keys (based on standard API response)
            total_assets = get_val("TotalAssets")
            current_assets = get_val("CurrentAssets")
            current_liabilities = get_val("CurrentLiabilities")
            total_liabilities = get_val("TotalLiabilities")
            net_sales = get_val("NetSales")
            operating_profit = get_val("OperatingProfit")
            net_income = get_val("Profit")  # Assuming Profit is Net Income
            retained_earnings = get_val("RetainedEarnings")
            operating_cash_flow = get_val("OperatingCashFlow")
            
            # Derived values
            working_capital = current_assets - current_liabilities
            
            return {
                "total_assets": total_assets,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "total_liabilities": total_liabilities,
                "working_capital": working_capital,
                "retained_earnings": retained_earnings,
                "ebit": operating_profit,  # Using Operating Profit as proxy for EBIT
                "market_cap": market_cap if market_cap else 0.0,
                "sales": net_sales,
                "net_income": net_income,
                "operating_cash_flow": operating_cash_flow,
                "long_term_debt": total_liabilities - current_liabilities,  # Approx
                "shares_outstanding": get_val("AverageNumberOfShares"), # Or IssuedShares
                "gross_profit": get_val("GrossProfit"),
            }
        except Exception as e:
            print(f"[Screener Error] Failed to parse J-Quants data: {e}")
            return {}

    def calculate_z_score(self, fin_data):
        """
        Altman Z-Score: Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
        Eliminating bankruptcy risk for institutional grade security.
        """
        try:
            # Ensure required keys exist and are non-zero where divisors
            if fin_data.get('total_assets', 0) == 0: return 0.0
            if fin_data.get('total_liabilities', 0) == 0: return 0.0

            # X1: Working Capital / Total Assets
            x1 = fin_data['working_capital'] / fin_data['total_assets']
            # X2: Retained Earnings / Total Assets
            x2 = fin_data['retained_earnings'] / fin_data['total_assets']
            # X3: EBIT / Total Assets
            x3 = fin_data['ebit'] / fin_data['total_assets']
            # X4: Market Cap / Total Liabilities
            x4 = fin_data['market_cap'] / fin_data['total_liabilities']
            # X5: Sales / Total Assets
            x5 = fin_data['sales'] / fin_data['total_assets']
            
            z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
            return round(z_score, 4)
        except Exception as e:
            # print(f"Z-Score Calc Error: {e}")
            return 0.0

    def calculate_f_score(self, current_fin, last_fin):
        """
        Piotroski F-Score: 9-point health check.
        Targeting stocks with a score of 8 or higher.
        """
        f_score = 0
        try:
            # 1. Net Income > 0
            if current_fin['net_income'] > 0: f_score += 1
            # 2. Return on Assets (ROA) > 0
            roa_now = current_fin['net_income'] / current_fin['total_assets']
            if roa_now > 0: f_score += 1
            # 3. Operating Cash Flow > 0
            if current_fin['operating_cash_flow'] > 0: f_score += 1
            # 4. Accruals (Cash Flow > Net Income)
            if current_fin['operating_cash_flow'] > current_fin['net_income']: f_score += 1
            # 5. Lower Leverage (Long-term Debt)
            if current_fin['long_term_debt'] < last_fin.get('long_term_debt', 9e15): f_score += 1
            # 6. Higher Liquidity (Current Ratio)
            curr_ratio_now = current_fin['current_assets'] / current_fin['current_liabilities']
            curr_ratio_last = last_fin['current_assets'] / last_fin['current_liabilities']
            if curr_ratio_now > curr_ratio_last: f_score += 1
            # 7. No New Shares (Dilution Check)
            if current_fin['shares_outstanding'] <= last_fin.get('shares_outstanding', 0): f_score += 1
            # 8. Higher Gross Margin
            gm_now = current_fin['gross_profit'] / current_fin['sales']
            gm_last = last_fin['gross_profit'] / last_fin['sales']
            if gm_now > gm_last: f_score += 1
            # 9. Higher Asset Turnover
            at_now = current_fin['sales'] / current_fin['total_assets']
            at_last = last_fin['sales'] / last_fin['total_assets']
            if at_now > at_last: f_score += 1
        except Exception as e:
            print(f"[Screener Error] F-Score calculation incomplete: {e}")
            
        return f_score

    def get_trinity_verdict(self, z, f, peg):
        """
        Synthesizing the Offensive Logic.
        """
        # Primary Institutional Check
        if z >= self.Z_SAFE and f >= self.F_TARGET and peg < self.PEG_MAX:
            return "INSTITUTIONAL BUY"
        
        # Bootstrap / Aggressive Logic
        if self.ALLOW_AGGRESSIVE:
            # Slightly relaxed: Lower F-Score allowed if Z-Score is very safe
            if z >= 3.0 and f >= (self.F_TARGET - 1):
                return "AGGRESSIVE BUY"
            # Recovery Play: High F-Score but Z-Score in grey zone
            if z >= 1.81 and f >= 8:
                return "AGGRESSIVE BUY"

        if z <= self.Z_DANGER:
            return "CRITICAL AVOID"
            
        return "NEUTRAL WATCH"