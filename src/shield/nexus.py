import sqlite3
import json
import os
from shield.tracker import AuditLogger

class TokyoNexus:
    """
    The Nexus v3: Verdict Logger.
    Now saves the strategic verdict to the Audit DB for reporting.
    """
    def __init__(self, db_path="logs/audit.db"):
        self.db_path = db_path
        self.logger = AuditLogger()

    def _get_latest_log(self, ticker, log_type_key):
        if not os.path.exists(self.db_path): return None, None
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data_json, timestamp FROM audit_logs WHERE ticker = ? ORDER BY id DESC LIMIT 20", (ticker,))
        rows = cursor.fetchall()
        conn.close()
        for data_json, timestamp in rows:
            try:
                data = json.loads(data_json)
                if log_type_key in data: return data, timestamp
            except: continue
        return None, None

    def evaluate_ticker(self, ticker: str):
        print(f"--- 🔮 Tokyo Nexus: Synthesizing Strategy for {ticker} ---")

        screen_data, screen_time = self._get_latest_log(ticker, "z_score")
        sent_data, sent_time = self._get_latest_log(ticker, "score")

        if not screen_data or not sent_data:
            print("[Nexus] Error: Missing data.")
            return

        z_score = screen_data.get("z_score", 0)
        sent_score = sent_data.get("score", 0)
        
        verdict = "HOLD"
        confidence = "LOW"
        reason = []

        # --- LOGIC ENGINE ---
        if z_score < 0.8:
            verdict = "SELL / AVOID"
            confidence = "HIGH"
            reason.append("CRITICAL: Financial collapse imminent (Z < 0.8).")
        elif z_score > 2.9:
            reason.append("Financials are fortress-like (Safe).")
        else:
            reason.append(f"Financials are standard/leverages (Z={z_score:.2f}).")

        if sent_score > 10:
            reason.append("Market is EUPHORIC (High Momentum).")
        elif sent_score > 5:
            reason.append("Market is Bullish.")
        elif sent_score < -5:
            reason.append("Market is Bearish.")

        if verdict != "SELL / AVOID":
            if z_score > 2.5 and sent_score > 5:
                verdict = "STRONG BUY"
                confidence = "HIGH"
                reason.append("Perfect convergence of Safety and Growth.")
            elif z_score > 1.0 and sent_score > 12:
                verdict = "AGGRESSIVE BUY"
                confidence = "MEDIUM"
                reason.append("Riding the Momentum Wave despite average financials.")
            elif z_score > 2.0 and sent_score < -5:
                verdict = "WATCH (Value Trap?)"
                reason.append("Good company, bad sentiment.")

        print(f"\n[📊 Integrated Analysis]")
        print(f" Z-Score    : {z_score:.4f}")
        print(f" Sentiment  : {sent_score} (Signal: {sent_data.get('label')})")
        print("-" * 40)
        print(f" ⚖️ VERDICT : {verdict}")
        print(f" 📝 REASONING : {' '.join(reason)}")
        print("-" * 40)

        # --- SAVE VERDICT ---
        result_payload = {
            "nexus_verdict": verdict,
            "confidence": confidence,
            "reasoning": " ".join(reason),
            "inputs": {
                "z_score": z_score,
                "sentiment_score": sent_score
            }
        }
        self.logger.log_execution(ticker, result_payload)
        print("[Audit] Verdict logged to DB.")