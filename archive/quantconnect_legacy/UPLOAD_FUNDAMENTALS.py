# region imports
from AlgorithmImports import *
# endregion


class FundamentalDataUploader(QCAlgorithm):
    """
    Upload fundamental data (BPS, ROE, EPS) to ObjectStore.

    Run this BEFORE running AssetShieldProduction.
    Output: japan_fundamentals/{code}.csv in ObjectStore

    Instructions:
    1. Copy this entire file into a new QuantConnect project
    2. Replace FUNDAMENTAL_DATA dict below with actual CSV content
       from quantconnect/fundamental_data/*.csv
    3. Run backtest (1-day) to upload data
    4. Then run ASSET_SHIELD_PRODUCTION.py
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 2)
        self.SetCash(1000)

        # ==========================================================
        # PASTE FUNDAMENTAL CSV DATA HERE
        # Format per stock: "code": """disclosed_date,quarter,bps,roe,eps\n..."""
        # Generate from: python3 scripts/export_fundamentals.py
        # ==========================================================

        FUNDAMENTAL_DATA = {
            # Example (replace with full data from fundamental_data/*.csv):
            #
            # "72030": """2008-08-07,1Q,3890.71,2.886296443380001,112.3
            # 2008-11-06,2Q,3804.08,4.137413691566155,156.92
            # 2009-02-06,3Q,3444.52,3.043806908001735,104.66
            # ...
            # """,
        }

        uploaded = 0
        for code, csv_data in FUNDAMENTAL_DATA.items():
            if csv_data.strip():
                key = f"japan_fundamentals/{code}.csv"
                self.ObjectStore.Save(key, csv_data.strip())
                lines = len(csv_data.strip().split('\n'))
                self.Debug(f"Uploaded: {key} ({lines} records)")
                uploaded += 1

        self.Debug("=" * 60)
        self.Debug(f"Fundamental upload complete: {uploaded} stocks")
        self.Debug("Now run AssetShieldProduction for backtest")
        self.Debug("=" * 60)

    def OnData(self, data):
        pass
