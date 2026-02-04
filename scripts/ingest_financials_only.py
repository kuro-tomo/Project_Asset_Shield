#!/usr/bin/env python3
"""
Asset Shield - Financial Data Ingestion Script
===============================================
Script to fetch financial data (for PBR/ROE calculation) for all stocks
"""

import os
import sys
import sqlite3
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

load_dotenv(os.path.join(project_root, '.env'))

from shield.jquants_client import JQuantsClient
from shield.jquants_backtest_provider import RateLimiter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'financials_ingest.log'))
    ]
)
logger = logging.getLogger(__name__)


def normalize_stock_code(code: str) -> str:
    """Normalize to 5-digit J-Quants format"""
    code = str(code).strip()
    if len(code) == 4:
        return code + "0"
    return code


def save_statement(conn, stmt: Dict) -> bool:
    """Save a single financial statement to database"""
    def parse_float(val):
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    try:
        code = stmt.get("Code") or stmt.get("LocalCode", "")
        if not code:
            return False

        code = normalize_stock_code(code)

        # v1 field names (already converted by client)
        disclosed_date = stmt.get("DisclosedDate") or stmt.get("DiscDate")
        current_period_end = stmt.get("CurrentPeriodEndDate") or stmt.get("CurPerEn")
        fiscal_year = current_period_end[:4] if current_period_end else None
        fiscal_quarter = stmt.get("TypeOfCurrentPeriod") or stmt.get("CurPerType")

        # Financial metrics
        net_sales = parse_float(stmt.get("NetSales") or stmt.get("Sales"))
        operating_profit = parse_float(stmt.get("OperatingProfit") or stmt.get("OP"))
        ordinary_profit = parse_float(stmt.get("OrdinaryProfit") or stmt.get("OdP"))
        net_income = parse_float(stmt.get("Profit") or stmt.get("NP"))
        eps = parse_float(stmt.get("EarningsPerShare") or stmt.get("EPS"))
        total_assets = parse_float(stmt.get("TotalAssets") or stmt.get("TA"))
        equity = parse_float(stmt.get("Equity") or stmt.get("Eq"))
        equity_ratio = parse_float(stmt.get("EquityToAssetRatio") or stmt.get("EqAR"))
        bps = parse_float(stmt.get("BookValuePerShare") or stmt.get("BPS"))

        # Calculate ROE if possible: ROE = Net Income / Equity
        roe = None
        if net_income is not None and equity is not None and equity > 0:
            roe = (net_income / equity) * 100  # percentage

        conn.execute("""
            INSERT OR REPLACE INTO financial_statements
            (code, disclosed_date, fiscal_year, fiscal_quarter,
             net_sales, operating_profit, ordinary_profit, net_income,
             eps, total_assets, equity, equity_ratio, bps, roe, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            code, disclosed_date, fiscal_year, fiscal_quarter,
            net_sales, operating_profit, ordinary_profit, net_income,
            eps, total_assets, equity, equity_ratio, bps, roe,
            datetime.now().isoformat()
        ))
        return True
    except Exception as e:
        logger.warning(f"Failed to save statement: {e}")
        return False


def main():
    cache_path = os.path.join(project_root, 'data', 'jquants_cache.db')

    logger.info("=" * 70)
    logger.info("ASSET SHIELD - FINANCIAL DATA INGESTION")
    logger.info("=" * 70)

    # Initialize client
    client = JQuantsClient()
    rate_limiter = RateLimiter("premium")

    # Get all unique stock codes from daily_quotes
    with sqlite3.connect(cache_path) as conn:
        cursor = conn.execute("SELECT DISTINCT code FROM daily_quotes ORDER BY code")
        all_codes = [row[0] for row in cursor.fetchall()]

    logger.info(f"Total stocks to process: {len(all_codes)}")

    total_saved = 0
    failed_codes = []

    with sqlite3.connect(cache_path) as conn:
        for i, code in enumerate(all_codes):
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: [{i+1}/{len(all_codes)}] - Saved: {total_saved}")

            try:
                rate_limiter.wait_if_needed()

                statements = client.get_financial_statements(code=code)

                if statements:
                    saved_count = 0
                    for stmt in statements:
                        if save_statement(conn, stmt):
                            saved_count += 1
                    total_saved += saved_count

                    if saved_count > 0 and (i + 1) % 500 == 0:
                        conn.commit()
                        logger.info(f"[{i+1}/{len(all_codes)}] {code}: {saved_count} statements")

            except Exception as e:
                logger.error(f"Failed to fetch {code}: {e}")
                failed_codes.append(code)

        conn.commit()

    logger.info("=" * 70)
    logger.info("FINANCIAL DATA INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total statements saved: {total_saved:,}")
    logger.info(f"Failed codes: {len(failed_codes)}")

    # Verify
    with sqlite3.connect(cache_path) as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT code) as stocks,
                   SUM(CASE WHEN net_sales IS NOT NULL THEN 1 ELSE 0 END) as has_sales,
                   SUM(CASE WHEN bps IS NOT NULL THEN 1 ELSE 0 END) as has_bps,
                   SUM(CASE WHEN roe IS NOT NULL THEN 1 ELSE 0 END) as has_roe
            FROM financial_statements
        """)
        row = cursor.fetchone()
        logger.info(f"Verification:")
        logger.info(f"  Total records: {row[0]:,}")
        logger.info(f"  Unique stocks: {row[1]:,}")
        logger.info(f"  With Sales: {row[2]:,}")
        logger.info(f"  With BPS: {row[3]:,}")
        logger.info(f"  With ROE: {row[4]:,}")


if __name__ == "__main__":
    main()
