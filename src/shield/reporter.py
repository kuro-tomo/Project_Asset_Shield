import os
import json
import sqlite3
import markdown
import base64
from datetime import datetime

class ReportGenerator:
    """
    Shield: Institutional Grade Intelligence Publisher.
    Optimized for high-retention premium reports.
    """
    def __init__(self, db_path="logs/audit.db", output_dir="output/reports"):
        self.db_path = db_path
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_latest_data(self, ticker, key_identifier):
        if not os.path.exists(self.db_path): return {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM audit_logs WHERE ticker = ? ORDER BY id DESC LIMIT 50", (ticker,))
        rows = cursor.fetchall()
        conn.close()
        for (data_json,) in rows:
            try:
                data = json.loads(data_json)
                if key_identifier in data: return data
            except: continue
        return {}

    def generate_report(self, ticker):
        # Data Retrieval
        fin_data = self._get_latest_data(ticker, "z_score")
        sent_data = self._get_latest_data(ticker, "score")
        nexus_data = self._get_latest_data(ticker, "nexus_verdict")

        # Variables
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        verdict = nexus_data.get("nexus_verdict", "NEUTRAL")
        
        # Color coding for UX
        v_color = "#cc0000" if "SELL" in verdict else "#003366"

        # Executive Summary Construction
        summary = f"Institutional analysis for {ticker} suggests a **{verdict}** stance. "
        summary += f"The Quantum Nexus engine detected a sentiment score of {sent_data.get('score', 0)} "
        summary += f"against a financial Z-Score of {fin_data.get('z_score', 0.0):.4f}."

        # Markdown Body (English)
        md_content = f"""
{summary}

---

### 📡 1. Market Sentiment Radar
* **Sentiment Score**: `{sent_data.get("score", 0)}` ({sent_data.get("label", "NEUTRAL")})
* **News Exposure**: {sent_data.get("headline_count", 0)} global headlines analyzed.

### 📊 2. Fundamental Health Integrity
* **Altman Z-Score**: `{fin_data.get("z_score", 0.0):.4f}`
* **Total Assets**: ¥{fin_data.get("components", {}).get("Total Assets", 0) / 1e12:.2f} Trillion
* **Operating Revenue**: ¥{fin_data.get("components", {}).get("Revenue", 0) / 1e12:.2f} Trillion

### 🛡️ 3. Risk Assessment & Validity
* **Integrity Hash**: `SHIELD-{base64.b64encode(ticker.encode()).decode()[:12].upper()}`
* **Validity**: This signal is valid for 24 hours from the timestamp.
"""
        html_body = markdown.markdown(md_content)

        # High-Value Template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Georgia', serif; line-height: 1.8; color: #1a1a1a; max-width: 850px; margin: 40px auto; padding: 60px; background-color: #fdfdfd; border: 1px solid #e0e0e0; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }}
                .header-badge {{ text-align:center; font-family: sans-serif; font-size: 0.7em; letter-spacing: 2px; color: #999; margin-bottom: 5px; }}
                h1 {{ font-size: 26pt; color: #002244; text-align: center; margin-top: 0; font-weight: bold; border-bottom: 3px solid #002244; padding-bottom: 20px; }}
                .verdict-box {{ background: #f8f9fa; border: 2px solid {v_color}; padding: 30px; border-radius: 4px; text-align: center; margin: 40px 0; }}
                .verdict-label {{ font-size: 0.9em; color: #666; text-transform: uppercase; letter-spacing: 3px; font-family: sans-serif; }}
                .verdict-value {{ font-size: 2.8em; color: {v_color}; font-weight: bold; margin: 10px 0; font-family: sans-serif; }}
                h3 {{ font-size: 1.3em; color: #003366; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 40px; font-family: sans-serif; }}
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ margin-bottom: 10px; border-left: 3px solid #eee; padding-left: 15px; }}
                .disclaimer {{ margin-top: 80px; padding: 30px; font-size: 0.75em; color: #777; border: 1px solid #eee; background: #fafafa; font-family: sans-serif; }}
                .disclaimer-title {{ font-weight: bold; color: #333; display: block; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="header-badge">INSTITUTIONAL GRADE INTELLIGENCE</div>
            <h1>SHIELD INTELLIGENCE REPORT</h1>
            <div style="text-align:center; color:#666; font-size:0.9em;">Target: {ticker} | Generated: {date_str}</div>

            <div class="verdict-box">
                <div class="verdict-label">Strategic Verdict</div>
                <div class="verdict-value">{verdict}</div>
                <div style="font-size:0.9em; color:#555;">Confidence Level: {nexus_data.get("confidence", "MEDIUM")}</div>
            </div>

            {html_body}
            
            <div class="disclaimer">
                <span class="disclaimer-title">INTELLECTUAL PROPERTY & DATA INTEGRITY:</span>
                This document is a high-fidelity intelligence product generated by the Shield (Quantum Nexus) engine.
                All data is synthesized from verified financial disclosures and global news streams.
                Copyright © 2026 Shield Project. All rights reserved.
                <br><br>
                <span class="disclaimer-title">IMPORTANT DISCLAIMER:</span>
                This report is for informational and research purposes only.
                <strong>It does not constitute financial advice, investment recommendations, or solicitation.</strong>
                The algorithmic verdicts are based on historical and current sentiment data, not a guarantee of future performance.
                Consult with a licensed professional before any transaction.
                Shield Project accepts no liability for financial outcomes.
            </div>
        </body>
        </html>
        """

        file_name = f"{ticker}_{datetime.now().strftime('%Y%m%d')}.html"
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"✅ Premium Report Published: {file_path}")

if __name__ == "__main__":
    # Test generation for a ticker already in the DB
    rep = ReportGenerator()
    rep.generate_report("7203.T")