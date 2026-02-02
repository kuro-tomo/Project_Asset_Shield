import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Module imports
from shield.jquants_client import JQuantsClient
from shield.screener import FinancialTrinity
from shield.money_management import MoneyManager
from shield.sentiment import J_Sentiment
from shield.tracker import log_event
from shield.itayose_analyzer import ItayoseAnalyzer, OrderBookSnapshot
from shield.adaptive_core import AdaptiveCore, MarketRegime
from shield.execution_core import ExecutionCore, OrderSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AssetShield_Main")

class ProductionPipeline:
    def __init__(self):
        self.client = JQuantsClient()
        self.trinity = FinancialTrinity()
        self.money_manager = MoneyManager()
        self.sentiment = J_Sentiment()
        self.itayose = ItayoseAnalyzer()
        self.adaptive = AdaptiveCore()
        self.execution = ExecutionCore()
        
    def fetch_market_data(self, code: str):
        """Fetch all necessary data from J-Quants"""
        logger.info(f"Fetching market data for {code}...")
        
        # 1. Listed Info (to check if valid)
        info = self.client.get_listed_info(code)
        if not info:
            logger.error(f"Stock {code} not found.")
            return None
            
        # 2. Daily Quotes (for trend/volatility)
        # Fetch last 60 days
        end_date = datetime.now()
        start_date = end_date - __import__("datetime").timedelta(days=90)
        quotes = self.client.get_daily_quotes(
            code=code,
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d")
        )
        
        # 3. Financial Statements (most recent)
        statements = self.client.get_financial_statements(code)
        # Sort by date descending and take latest
        latest_statement = statements[-1] if statements else {}
        
        # 4. Order Book (Snapshot for Itayose/Execution)
        # Note: In real production, this would be a stream or high-freq poll
        now_str = datetime.now().strftime("%Y-%m-%d")
        orderbook = self.client.get_orderbook(code, now_str)
        
        return {
            "info": info[0],
            "quotes": quotes,
            "financials": latest_statement,
            "orderbook": orderbook
        }

    def run(self, ticker: str, capital_usd: float, dry_run: bool = False):
        print(f"🚀 Initializing TIR Alpha Pipeline for {ticker}...")
        
        # --- Step 1: Data Ingestion ---
        data = self.fetch_market_data(ticker)
        if not data:
            print(f"❌ Failed to fetch data for {ticker}")
            return None
            
        # Current Price (Approximate from last quote or orderbook)
        if data['quotes']:
            current_price = data['quotes'][-1]['Close']
        else:
            logger.warning("No quote data found, skipping.")
            return None

        # --- Step 2: Adaptive Core (Regime Detection) ---
        # Feed recent prices to volatility tracker
        for quote in data['quotes'][-60:]:
            self.adaptive.update_market_data(quote['Close'])
        
        regime_state = self.adaptive.classify_regime()
        print(f"🌍 Market Regime: {regime_state.regime.value} (Confidence: {regime_state.confidence:.2f})")
        
        # Recalibrate parameters
        model_params = self.adaptive.recalibrate()
        
        # --- Step 3: Financial Trinity (Fundamental) ---
        # Parse financials
        fin_data = self.trinity.parse_jquants_financials(
            data['financials'], 
            market_cap=current_price * float(data['financials'].get('AverageNumberOfShares', 0) or 0)
        )
        
        if fin_data:
            z_score = self.trinity.calculate_z_score(fin_data)
            # F-Score requires previous period, skipping for now or assume 0 if missing
            f_score = 9 # Placeholder for single period data limitation
            verdict = self.trinity.get_trinity_verdict(z=z_score, f=f_score, peg=0.8) # PEG hardcoded for now
            print(f"💎 Financial Trinity: Z={z_score:.2f}, F={f_score}, Verdict={verdict}")
        else:
            verdict = "NEUTRAL WATCH"
            print("⚠️ Financial data insufficient, defaulting to NEUTRAL")

        # --- Step 4: J-Sentiment (News Analysis) ---
        # In a real scenario, we would fetch news here. 
        # For now, we use the placeholder headlines or skip if API key missing.
        headlines = [f"Market analysis for {ticker}"] # Placeholder
        sentiment_res = self.sentiment.analyze(ticker, headlines)
        print(f"📊 Sentiment Score: {sentiment_res.get('score')} ({sentiment_res.get('logic')})")

        # --- Step 5: Itayose/Microstructure Analysis ---
        # If we had pre-market orderbook data
        if data['orderbook']:
            snapshots = self.itayose.parse_jquants_orderbook(data['orderbook'])
            ofi_signal = self.itayose.calculate_ofi(snapshots)
            print(f"🌊 Order Flow Imbalance: {ofi_signal:.2f}")
        
        # --- Step 6: Money Management ---
        # Adjust verdict based on sentiment
        if sentiment_res.get('score', 0.5) < 0.4:
            final_verdict = "STRATEGIC ACCUMULATE" # Contrarian? Or just caution.
        else:
            final_verdict = verdict
            
        allocation = self.money_manager.get_position_size(capital_usd, final_verdict)
        print(f"💰 Allocation: ¥{allocation['total_jpy']:,.0f} (Leverage: {allocation['leverage']}x)")

        # --- Step 7: Execution Planning ---
        if allocation['total_jpy'] > 0 and not dry_run:
            qty = int(allocation['total_jpy'] / current_price)
            # Round to 100 units (typical unit)
            qty = (qty // 100) * 100
            
            if qty > 0:
                # Create VWAP Plan
                daily_vol = int(data['quotes'][-1]['Volume']) if data['quotes'] else 1000000
                plan = self.execution.create_vwap_plan(
                    code=ticker,
                    side=OrderSide.BUY,
                    total_quantity=qty,
                    current_price=current_price,
                    daily_volume=daily_vol
                )
                print(f"⚙️ Execution Plan: {plan.strategy.value} for {qty:,} shares")
                print(f"   Estimated Cost: {plan.estimated_cost_bps:.2f} bps")
                
                execution_data = plan.to_dict()
            else:
                print("⚠️ Quantity too small to execute.")
                execution_data = {}
        else:
            print("🛑 Execution skipped (Dry Run or Zero Allocation)")
            execution_data = {}

        # --- Step 8: Audit Logging ---
        audit_payload = {
            "ticker": ticker,
            "regime": regime_state.to_dict(),
            "financials": {"verdict": verdict, "z_score": z_score if 'z_score' in locals() else 0},
            "sentiment": sentiment_res,
            "allocation": allocation,
            "execution": execution_data,
            "status": "EXECUTED" if execution_data else "SKIPPED"
        }
        
        tx_hash = log_event(ticker, "PRODUCTION_PIPELINE", audit_payload)
        print(f"🔒 Audit Trail Secured: {tx_hash}")
        
        return audit_payload

if __name__ == "__main__":
    try:
        pipeline = ProductionPipeline()
        
        # Target: Toyota (7203)
        # Note: Set JQUANTS_MAIL and JQUANTS_PASSWORD in env vars before running
        if not os.environ.get("JQUANTS_MAIL"):
            print("⚠️  WARNING: JQUANTS_MAIL not set. Using mock/dry-run mode behavior.")
            
        pipeline.run("7203", 100000, dry_run=True)
        
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
