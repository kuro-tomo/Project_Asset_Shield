#!/usr/bin/env python3
"""
Asset Shield V3 - Live Market Reconnaissance
=============================================
Real-time signal generation and data continuity verification.

Mission:
1. Fetch latest market data (Feb 4, 2026) from J-Quants API
2. Synchronize with existing 14.9M records in jquants_cache.db
3. Apply Asset Shield V3 Percentile-based Alpha Relaxation
4. Generate BUY/HOLD signals for next trading session
5. Detect signal drift vs OOS period

Author: Asset Shield V3 Reconnaissance Unit
Version: 3.0.0 (2026-02-04)
"""

import os
import sys
import sqlite3
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import time

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from shield.jquants_client import JQuantsClient

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'live_recon.log'))
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LiveSignal:
    """Live trading signal"""
    code: str
    name: str
    signal: str  # BUY, HOLD, SELL
    price: float
    pbr: float
    roe: float
    composite_score: float
    adt: float
    impact_bps: float
    confidence: float
    sector: str = ""


@dataclass
class DriftAnalysis:
    """Signal drift analysis result"""
    total_signals_today: int
    total_signals_oos_avg: float
    overlap_ratio: float
    new_entries: List[str]
    dropped_entries: List[str]
    drift_score: float  # 0-1, higher = more drift
    is_anomaly: bool


@dataclass
class SyncStatus:
    """Data synchronization status"""
    last_sync_date: str
    total_records: int
    records_added_today: int
    stocks_updated: int
    gaps_detected: List[str]
    is_current: bool


# =============================================================================
# Almgren-Chriss Impact Calculator (Inline for speed)
# =============================================================================

class QuickImpactCalc:
    """Lightweight Almgren-Chriss for live signal generation"""
    GAMMA = 0.10
    ETA = 0.01
    SIGMA = 0.25
    SPREAD_BPS = 10.0

    @staticmethod
    def calculate(order_value: float, adt: float) -> float:
        """Returns total impact in bps"""
        if adt <= 0:
            return float('inf')

        participation = order_value / adt
        permanent = QuickImpactCalc.GAMMA * QuickImpactCalc.SIGMA * np.sqrt(participation)
        temporary = QuickImpactCalc.ETA * QuickImpactCalc.SIGMA * (participation ** 0.6)
        spread = QuickImpactCalc.SPREAD_BPS / 2 / 10000

        total_bps = (permanent + temporary + spread) * 10000
        return round(total_bps, 2)


# =============================================================================
# Live Market Reconnaissance Engine
# =============================================================================

class LiveMarketRecon:
    """
    Live Market Reconnaissance Engine

    Fetches real-time data, syncs with cache, generates signals.
    """

    # Asset Shield V3 Parameters
    PBR_THRESHOLD = 1.0
    ROE_THRESHOLD = 10.0
    MIN_ADT = 500_000_000  # 500M JPY
    MAX_POSITIONS = 20
    POSITION_SIZE = 10_000_000  # 10M JPY per position

    # Percentile relaxation (V3 feature)
    PBR_PERCENTILE = 30  # Bottom 30% PBR
    ROE_PERCENTILE = 70  # Top 30% ROE

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.client = JQuantsClient()
        self.today = date(2026, 2, 4)
        self.today_str = self.today.strftime("%Y-%m-%d")

        # Results storage
        self.sync_status: Optional[SyncStatus] = None
        self.signals: List[LiveSignal] = []
        self.drift_analysis: Optional[DriftAnalysis] = None

        logger.info(f"LiveMarketRecon initialized for {self.today_str}")

    # -------------------------------------------------------------------------
    # Phase 1: Data Synchronization
    # -------------------------------------------------------------------------

    def sync_latest_data(self) -> SyncStatus:
        """Fetch and sync latest market data"""
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA SYNCHRONIZATION")
        logger.info("=" * 60)

        conn = sqlite3.connect(self.cache_path)

        # Check current state
        cursor = conn.execute("SELECT MAX(date), COUNT(*) FROM daily_quotes")
        last_date, total_records = cursor.fetchone()
        logger.info(f"Current cache: {total_records:,} records, last date: {last_date}")

        # Get distinct stocks
        cursor = conn.execute("SELECT DISTINCT code FROM daily_quotes")
        all_codes = [row[0] for row in cursor.fetchall()]
        logger.info(f"Total stocks in cache: {len(all_codes)}")

        # Determine sync range
        if last_date:
            last_date_obj = datetime.strptime(last_date, "%Y-%m-%d").date()
            sync_start = (last_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            sync_start = "2026-02-01"

        sync_end = self.today_str

        # Check if we need to sync
        if last_date == self.today_str:
            logger.info("Cache is already current. No sync needed.")
            self.sync_status = SyncStatus(
                last_sync_date=self.today_str,
                total_records=total_records,
                records_added_today=0,
                stocks_updated=0,
                gaps_detected=[],
                is_current=True
            )
            conn.close()
            return self.sync_status

        logger.info(f"Syncing data from {sync_start} to {sync_end}")

        # Fetch new data
        records_added = 0
        stocks_updated = 0
        gaps = []

        try:
            # Use date-based fetch for efficiency
            for fetch_date in pd.date_range(sync_start, sync_end):
                fetch_date_str = fetch_date.strftime("%Y-%m-%d")

                # Skip weekends
                if fetch_date.weekday() >= 5:
                    continue

                logger.info(f"Fetching data for {fetch_date_str}...")

                try:
                    # Rate limiting
                    time.sleep(0.5)

                    quotes = self.client.get_daily_quotes(from_date=fetch_date_str, to_date=fetch_date_str)

                    if quotes:
                        for q in quotes:
                            code = q.get('Code', q.get('code', ''))
                            if not code:
                                continue

                            # Normalize code to 5-digit
                            if len(code) == 4:
                                code = code + "0"

                            conn.execute("""
                                INSERT OR REPLACE INTO daily_quotes
                                (code, date, open, high, low, close, volume, turnover, adjustment_factor, adjustment_close)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                code,
                                fetch_date_str,
                                q.get('Open', q.get('open')),
                                q.get('High', q.get('high')),
                                q.get('Low', q.get('low')),
                                q.get('Close', q.get('close')),
                                q.get('Volume', q.get('volume')),
                                q.get('Turnover', q.get('turnover', 0)),
                                q.get('AdjustmentFactor', q.get('adjustment_factor', 1.0)),
                                q.get('AdjustmentClose', q.get('adjustment_close'))
                            ))
                            records_added += 1

                        stocks_updated += len(set(q.get('Code', q.get('code', '')) for q in quotes))
                        conn.commit()
                        logger.info(f"  Added {len(quotes)} records for {fetch_date_str}")
                    else:
                        gaps.append(fetch_date_str)
                        logger.warning(f"  No data returned for {fetch_date_str}")

                except Exception as e:
                    logger.error(f"  Error fetching {fetch_date_str}: {e}")
                    gaps.append(fetch_date_str)

        except Exception as e:
            logger.error(f"Sync failed: {e}")

        # Verify final state
        cursor = conn.execute("SELECT MAX(date), COUNT(*) FROM daily_quotes")
        final_date, final_records = cursor.fetchone()

        self.sync_status = SyncStatus(
            last_sync_date=final_date or self.today_str,
            total_records=final_records,
            records_added_today=records_added,
            stocks_updated=stocks_updated,
            gaps_detected=gaps,
            is_current=(final_date == self.today_str)
        )

        conn.close()

        logger.info(f"Sync complete: +{records_added:,} records, {stocks_updated} stocks updated")
        return self.sync_status

    # -------------------------------------------------------------------------
    # Phase 2: Signal Generation
    # -------------------------------------------------------------------------

    def generate_signals(self) -> List[LiveSignal]:
        """Generate live trading signals using Asset Shield V3 logic"""
        logger.info("=" * 60)
        logger.info("PHASE 2: SIGNAL GENERATION (Asset Shield V3)")
        logger.info("=" * 60)

        conn = sqlite3.connect(self.cache_path)

        # Get latest prices
        logger.info("Loading latest price data...")
        prices_df = pd.read_sql_query("""
            SELECT code, date, close, volume, turnover, adjustment_close
            FROM daily_quotes
            WHERE date >= date('2026-01-01')
            ORDER BY code, date
        """, conn)

        if prices_df.empty:
            logger.error("No recent price data available")
            conn.close()
            return []

        prices_df['price'] = prices_df['adjustment_close'].fillna(prices_df['close'])

        # Calculate ADT (60-day average)
        logger.info("Calculating ADT...")
        adt_df = prices_df.groupby('code').agg({
            'turnover': lambda x: x.tail(60).mean() if len(x) >= 20 else 0,
            'price': 'last',
            'date': 'max'
        }).reset_index()
        adt_df.columns = ['code', 'adt', 'price', 'last_date']

        # Filter by liquidity
        liquid_stocks = adt_df[adt_df['adt'] >= self.MIN_ADT].copy()
        logger.info(f"Liquidity filter: {len(liquid_stocks)} stocks with ADT >= 500M JPY")

        # Get financial data
        logger.info("Loading financial data...")
        fins_df = pd.read_sql_query("""
            SELECT code, disclosed_date, bps, roe, net_sales, net_income
            FROM financial_statements
            WHERE bps > 0
            ORDER BY code, disclosed_date DESC
        """, conn)

        # Get latest financials per stock
        latest_fins = fins_df.groupby('code').first().reset_index()

        # Merge price and financial data
        merged = liquid_stocks.merge(latest_fins[['code', 'bps', 'roe']], on='code', how='inner')

        # Calculate PBR
        merged['pbr'] = merged['price'] / merged['bps']
        merged = merged[merged['pbr'] > 0]

        logger.info(f"After merging financials: {len(merged)} candidates")

        # Asset Shield V3: Percentile-based Alpha Relaxation
        logger.info("Applying V3 Percentile-based Alpha Relaxation...")

        pbr_threshold = np.percentile(merged['pbr'].dropna(), self.PBR_PERCENTILE)
        roe_threshold = np.percentile(merged['roe'].dropna(), self.ROE_PERCENTILE)

        logger.info(f"  PBR threshold (P{self.PBR_PERCENTILE}): {pbr_threshold:.2f}")
        logger.info(f"  ROE threshold (P{self.ROE_PERCENTILE}): {roe_threshold:.1f}%")

        # Also apply absolute thresholds
        pbr_threshold = min(pbr_threshold, self.PBR_THRESHOLD)
        roe_threshold = max(roe_threshold, self.ROE_THRESHOLD)

        # Filter candidates
        candidates = merged[
            (merged['pbr'] <= pbr_threshold) &
            (merged['roe'] >= roe_threshold)
        ].copy()

        logger.info(f"Candidates after V3 filter: {len(candidates)}")

        # Calculate composite score
        candidates['pbr_score'] = 1 - (candidates['pbr'] / pbr_threshold)
        candidates['roe_score'] = (candidates['roe'] - roe_threshold) / (100 - roe_threshold)
        candidates['composite_score'] = (candidates['pbr_score'] * 0.5 + candidates['roe_score'] * 0.5)

        # Rank and select top candidates
        candidates = candidates.sort_values('composite_score', ascending=False)

        # Generate signals with impact check
        self.signals = []

        for _, row in candidates.head(self.MAX_POSITIONS * 2).iterrows():
            impact = QuickImpactCalc.calculate(self.POSITION_SIZE, row['adt'])

            # Skip if impact too high
            if impact > 50:
                continue

            signal = LiveSignal(
                code=row['code'],
                name=self._get_stock_name(conn, row['code']),
                signal="BUY" if row['composite_score'] > 0.5 else "HOLD",
                price=row['price'],
                pbr=round(row['pbr'], 3),
                roe=round(row['roe'], 2),
                composite_score=round(row['composite_score'], 4),
                adt=row['adt'],
                impact_bps=impact,
                confidence=min(0.95, 0.5 + row['composite_score'] * 0.5)
            )
            self.signals.append(signal)

            if len(self.signals) >= self.MAX_POSITIONS:
                break

        conn.close()

        logger.info(f"Generated {len(self.signals)} signals")
        for sig in self.signals[:5]:
            logger.info(f"  {sig.signal}: {sig.code} | PBR:{sig.pbr:.2f} ROE:{sig.roe:.1f}% | Impact:{sig.impact_bps:.1f}bps")

        return self.signals

    def _get_stock_name(self, conn, code: str) -> str:
        """Get stock name from cache or return code"""
        try:
            cursor = conn.execute(
                "SELECT name FROM stock_info WHERE code = ? LIMIT 1", (code,)
            )
            row = cursor.fetchone()
            return row[0] if row else code
        except:
            return code

    # -------------------------------------------------------------------------
    # Phase 3: Drift Analysis
    # -------------------------------------------------------------------------

    def analyze_drift(self) -> DriftAnalysis:
        """Compare live signals against OOS period to detect drift"""
        logger.info("=" * 60)
        logger.info("PHASE 3: SIGNAL DRIFT ANALYSIS")
        logger.info("=" * 60)

        conn = sqlite3.connect(self.cache_path)

        # Get historical signal patterns from OOS period (2021-2025)
        logger.info("Analyzing OOS period signal patterns...")

        # Load OOS period data
        oos_df = pd.read_sql_query("""
            SELECT dq.code, dq.date, dq.close, dq.turnover, fs.bps, fs.roe
            FROM daily_quotes dq
            LEFT JOIN financial_statements fs ON dq.code = fs.code
            WHERE dq.date BETWEEN '2021-01-01' AND '2025-12-31'
            AND fs.bps > 0
            ORDER BY dq.date
        """, conn)

        conn.close()

        if oos_df.empty:
            logger.warning("No OOS data available for drift analysis")
            self.drift_analysis = DriftAnalysis(
                total_signals_today=len(self.signals),
                total_signals_oos_avg=0,
                overlap_ratio=0,
                new_entries=[],
                dropped_entries=[],
                drift_score=1.0,
                is_anomaly=True
            )
            return self.drift_analysis

        # Calculate PBR for OOS period
        oos_df['pbr'] = oos_df['close'] / oos_df['bps']
        oos_df = oos_df[(oos_df['pbr'] > 0) & (oos_df['pbr'] < 10)]

        # Find stocks that would have qualified in OOS period
        oos_qualified = oos_df[
            (oos_df['pbr'] <= self.PBR_THRESHOLD) &
            (oos_df['roe'] >= self.ROE_THRESHOLD)
        ]['code'].unique()

        logger.info(f"OOS period qualified stocks: {len(oos_qualified)}")

        # Compare with today's signals
        today_codes = set(sig.code for sig in self.signals)
        oos_codes = set(oos_qualified)

        overlap = today_codes.intersection(oos_codes)
        new_entries = list(today_codes - oos_codes)
        dropped_entries = list(oos_codes - today_codes)[:20]  # Limit for readability

        overlap_ratio = len(overlap) / len(today_codes) if today_codes else 0

        # Calculate drift score
        # Low drift = high overlap, few new entries
        drift_score = 1 - overlap_ratio
        if len(new_entries) > len(self.signals) * 0.5:
            drift_score += 0.2

        drift_score = min(1.0, max(0.0, drift_score))
        is_anomaly = drift_score > 0.5 or len(new_entries) > 10

        self.drift_analysis = DriftAnalysis(
            total_signals_today=len(self.signals),
            total_signals_oos_avg=len(oos_qualified),
            overlap_ratio=round(overlap_ratio, 3),
            new_entries=new_entries[:10],
            dropped_entries=dropped_entries[:10],
            drift_score=round(drift_score, 3),
            is_anomaly=is_anomaly
        )

        logger.info(f"Drift score: {drift_score:.2f} | Anomaly: {is_anomaly}")
        logger.info(f"Overlap with OOS: {overlap_ratio:.1%}")
        logger.info(f"New entries: {len(new_entries)} | Dropped: {len(dropped_entries)}")

        return self.drift_analysis

    # -------------------------------------------------------------------------
    # Phase 4: Report Generation
    # -------------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate EVENING_RECON_REPORT.md"""
        logger.info("=" * 60)
        logger.info("PHASE 4: REPORT GENERATION")
        logger.info("=" * 60)

        report_path = os.path.join(PROJECT_ROOT, 'output', 'EVENING_RECON_REPORT.md')

        # Calculate summary stats
        buy_signals = [s for s in self.signals if s.signal == "BUY"]
        hold_signals = [s for s in self.signals if s.signal == "HOLD"]
        avg_impact = np.mean([s.impact_bps for s in self.signals]) if self.signals else 0
        avg_confidence = np.mean([s.confidence for s in self.signals]) if self.signals else 0

        content = f"""# Asset Shield V3 - Evening Reconnaissance Report
## Date: {self.today_str}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Reconnaissance Date | {self.today_str} |
| Total Active Signals | {len(self.signals)} |
| BUY Signals | {len(buy_signals)} |
| HOLD Signals | {len(hold_signals)} |
| Avg Market Impact | {avg_impact:.1f} bps |
| Avg Signal Confidence | {avg_confidence:.1%} |

---

## 1. Data Synchronization Status

| Metric | Status |
|--------|--------|
| Last Sync Date | {self.sync_status.last_sync_date if self.sync_status else 'N/A'} |
| Total Records | {self.sync_status.total_records:,} |
| Records Added Today | {self.sync_status.records_added_today:,} |
| Stocks Updated | {self.sync_status.stocks_updated} |
| Data Currency | {'CURRENT' if self.sync_status.is_current else 'STALE'} |

### Data Gaps Detected
{self._format_gaps()}

---

## 2. Active Signals for Next Trading Session

### BUY Signals (Immediate Action)

| Code | PBR | ROE | Score | ADT (B) | Impact | Confidence |
|------|-----|-----|-------|---------|--------|------------|
{self._format_signals(buy_signals)}

### HOLD Signals (Monitor)

| Code | PBR | ROE | Score | ADT (B) | Impact | Confidence |
|------|-----|-----|-------|---------|--------|------------|
{self._format_signals(hold_signals)}

---

## 3. Almgren-Chriss Impact Verification

### Parameters Applied
- Gamma (permanent): 0.10
- Eta (temporary): 0.01
- Sigma (volatility): 0.25
- Spread: 10 bps

### Impact Summary

| Metric | Value |
|--------|-------|
| Signals with Impact < 20bps | {len([s for s in self.signals if s.impact_bps < 20])} |
| Signals with Impact 20-50bps | {len([s for s in self.signals if 20 <= s.impact_bps < 50])} |
| Average Impact | {avg_impact:.2f} bps |
| Max Impact | {max([s.impact_bps for s in self.signals]) if self.signals else 0:.2f} bps |

### Liquidity Grade Distribution
{self._format_liquidity_grades()}

---

## 4. Signal Drift Analysis

| Metric | Value |
|--------|-------|
| Today's Signal Count | {self.drift_analysis.total_signals_today if self.drift_analysis else 0} |
| OOS Average Qualified | {self.drift_analysis.total_signals_oos_avg if self.drift_analysis else 0} |
| Overlap with OOS | {self.drift_analysis.overlap_ratio:.1%} |
| Drift Score | {self.drift_analysis.drift_score:.2f} |
| Anomaly Detected | {'YES - REVIEW REQUIRED' if self.drift_analysis and self.drift_analysis.is_anomaly else 'NO'} |

### New Entries (vs OOS)
{self._format_list(self.drift_analysis.new_entries if self.drift_analysis else [])}

### Dropped Entries (vs OOS)
{self._format_list(self.drift_analysis.dropped_entries if self.drift_analysis else [])}

---

## 5. Recommendations

{self._generate_recommendations()}

---

## Appendix: Strategy Parameters

| Parameter | Value |
|-----------|-------|
| PBR Threshold | {self.PBR_THRESHOLD} |
| ROE Threshold | {self.ROE_THRESHOLD}% |
| Min ADT | 500M JPY |
| Max Positions | {self.MAX_POSITIONS} |
| Position Size | 10M JPY |
| V3 PBR Percentile | {self.PBR_PERCENTILE} |
| V3 ROE Percentile | {self.ROE_PERCENTILE} |

---

*Asset Shield V3 - Autonomous Reconnaissance System*
*Report generated in silent mode*
"""

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(content)

        logger.info(f"Report saved: {report_path}")
        return report_path

    def _format_signals(self, signals: List[LiveSignal]) -> str:
        if not signals:
            return "| - | - | - | - | - | - | - |"

        lines = []
        for s in signals:
            lines.append(
                f"| {s.code} | {s.pbr:.2f} | {s.roe:.1f}% | {s.composite_score:.3f} | "
                f"¥{s.adt/1e9:.1f}B | {s.impact_bps:.1f}bps | {s.confidence:.0%} |"
            )
        return "\n".join(lines)

    def _format_gaps(self) -> str:
        if not self.sync_status or not self.sync_status.gaps_detected:
            return "None detected."
        return ", ".join(self.sync_status.gaps_detected[:5])

    def _format_list(self, items: List[str]) -> str:
        if not items:
            return "None"
        return ", ".join(items[:10])

    def _format_liquidity_grades(self) -> str:
        if not self.signals:
            return "No signals to grade."

        grades = {'A': 0, 'B': 0, 'C': 0}
        for s in self.signals:
            if s.adt >= 5_000_000_000:
                grades['A'] += 1
            elif s.adt >= 1_000_000_000:
                grades['B'] += 1
            else:
                grades['C'] += 1

        return f"Grade A (ADT>5B): {grades['A']} | Grade B (1-5B): {grades['B']} | Grade C (<1B): {grades['C']}"

    def _generate_recommendations(self) -> str:
        recs = []

        # Data status
        if self.sync_status and not self.sync_status.is_current:
            recs.append("⚠️ **DATA STALE**: Sync required before trading")
        else:
            recs.append("✓ Data is current")

        # Signal quality
        if len(self.signals) >= 15:
            recs.append("✓ Signal universe is healthy (15+ candidates)")
        elif len(self.signals) >= 10:
            recs.append("⚠️ Signal universe is adequate but limited")
        else:
            recs.append("⚠️ **LOW SIGNALS**: Consider relaxing filters")

        # Drift
        if self.drift_analysis:
            if self.drift_analysis.is_anomaly:
                recs.append("⚠️ **DRIFT DETECTED**: Manual review recommended")
            else:
                recs.append("✓ Signal drift within normal range")

        # Impact
        high_impact = [s for s in self.signals if s.impact_bps > 30]
        if high_impact:
            recs.append(f"⚠️ {len(high_impact)} signals with elevated impact (>30bps)")
        else:
            recs.append("✓ Market impact within acceptable bounds")

        return "\n".join(f"- {r}" for r in recs)

    # -------------------------------------------------------------------------
    # Main Execution
    # -------------------------------------------------------------------------

    def run(self):
        """Execute full reconnaissance mission"""
        logger.info("=" * 70)
        logger.info("ASSET SHIELD V3 - LIVE MARKET RECONNAISSANCE")
        logger.info(f"Mission Date: {self.today_str}")
        logger.info("=" * 70)

        start_time = datetime.now()

        # Phase 1: Sync
        self.sync_latest_data()

        # Phase 2: Signals
        self.generate_signals()

        # Phase 3: Drift
        self.analyze_drift()

        # Phase 4: Report
        report_path = self.generate_report()

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("=" * 70)
        logger.info("RECONNAISSANCE COMPLETE")
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 70)

        return {
            'sync_status': asdict(self.sync_status) if self.sync_status else None,
            'signals': [asdict(s) for s in self.signals],
            'drift': asdict(self.drift_analysis) if self.drift_analysis else None,
            'report_path': report_path
        }


# =============================================================================
# Main Entry
# =============================================================================

def main():
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'jquants_cache.db')

    if not os.path.exists(cache_path):
        logger.error(f"Cache database not found: {cache_path}")
        sys.exit(1)

    recon = LiveMarketRecon(cache_path)
    result = recon.run()

    # Save JSON results
    json_path = os.path.join(
        PROJECT_ROOT, 'output',
        f'recon_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"JSON results: {json_path}")


if __name__ == "__main__":
    main()
