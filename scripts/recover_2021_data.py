#!/usr/bin/env python3
"""
Asset Shield - 2021+ Data Recovery Script
==========================================

Purpose: Restore data coverage from 1,000 stocks to 4,000 stocks for 2021+

Strategy:
1. Get all listed stocks from J-Quants API
2. Identify stocks with 2020 data but missing 2021+ data
3. High-speed data retrieval via parallel batch processing
4. Rate limiting in consideration of API limits (12 requests/min)

Author: Asset Shield Recovery Team
Version: 1.0.0
"""

import os
import sys
import sqlite3
import logging
import time
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from shield.jquants_client import JQuantsClient, JQuantsConfig

# Configure logging
log_path = os.path.join(project_root, 'logs', 'recovery_2021.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path)
    ]
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """API Rate Limiter - J-Quants limit: 12 requests/min"""

    def __init__(self, max_requests: int = 12, period_seconds: int = 60):
        self.max_requests = max_requests
        self.period = period_seconds
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [r for r in self.requests if now - r < self.period]

            if len(self.requests) >= self.max_requests:
                # Wait until oldest request expires
                sleep_time = self.period - (now - self.requests[0]) + 0.1
                if sleep_time > 0:
                    logger.debug(f"Rate limit: sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)

            self.requests.append(time.time())


class DataRecoveryEngine:
    """Data recovery engine for 2021+"""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.client = JQuantsClient()
        self.rate_limiter = RateLimiter(max_requests=10, period_seconds=60)  # Safety margin

        # Statistics
        self.stats = {
            'stocks_processed': 0,
            'records_fetched': 0,
            'errors': 0,
            'skipped': 0
        }

    def get_2020_stocks(self) -> Set[str]:
        """Get stocks with data in 2020"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT code FROM daily_quotes
            WHERE date BETWEEN '2020-01-01' AND '2020-12-31'
        """)
        stocks = {row[0] for row in cursor.fetchall()}
        conn.close()
        logger.info(f"2020 coverage: {len(stocks)} stocks")
        return stocks

    def get_2021_stocks(self) -> Set[str]:
        """Get stocks with data in 2021+"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT code FROM daily_quotes
            WHERE date >= '2021-01-01'
        """)
        stocks = {row[0] for row in cursor.fetchall()}
        conn.close()
        logger.info(f"2021+ coverage: {len(stocks)} stocks")
        return stocks

    def identify_missing_stocks(self) -> List[str]:
        """Identify missing stocks"""
        stocks_2020 = self.get_2020_stocks()
        stocks_2021 = self.get_2021_stocks()

        missing = stocks_2020 - stocks_2021
        logger.info(f"Missing stocks count: {len(missing)}")
        return list(missing)

    def fetch_stock_data(self, code: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch individual stock data"""
        self.rate_limiter.wait_if_needed()

        try:
            # Fetch data via J-Quants v2 API
            result = self.client.get_daily_quotes(
                code=code,
                from_date=start_date,
                to_date=end_date
            )

            if result:
                return result
            return []

        except Exception as e:
            logger.error(f"Error fetching {code}: {e}")
            return []

    def save_to_db(self, records: List[Dict], code: str):
        """Save data to database"""
        if not records:
            return 0

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        inserted = 0
        for rec in records:
            try:
                # Convert from J-Quants v2 format to DB format
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_quotes
                    (code, date, open, high, low, close, volume, turnover, adjustment_factor)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    code,
                    rec.get('Date', rec.get('date')),
                    rec.get('Open', rec.get('open')),
                    rec.get('High', rec.get('high')),
                    rec.get('Low', rec.get('low')),
                    rec.get('Close', rec.get('close')),
                    rec.get('Volume', rec.get('volume')),
                    rec.get('Turnover', rec.get('turnover', 0)),
                    rec.get('AdjustmentFactor', rec.get('adjustment_factor', 1.0))
                ))
                inserted += 1
            except Exception as e:
                logger.debug(f"Insert error for {code}: {e}")

        conn.commit()
        conn.close()
        return inserted

    def recover_stock(self, code: str, start_date: str = "2021-01-01",
                      end_date: str = "2026-02-03") -> int:
        """Recover data for a single stock"""
        records = self.fetch_stock_data(code, start_date, end_date)
        if records:
            saved = self.save_to_db(records, code)
            self.stats['records_fetched'] += saved
            return saved
        return 0

    def run_recovery(self, max_stocks: Optional[int] = None):
        """Execute recovery process"""
        logger.info("=" * 70)
        logger.info("Asset Shield - 2021+ Data Recovery Started")
        logger.info("=" * 70)

        # Identify missing stocks
        missing_stocks = self.identify_missing_stocks()

        if max_stocks:
            missing_stocks = missing_stocks[:max_stocks]

        total = len(missing_stocks)
        logger.info(f"Recovery target: {total} stocks")

        start_time = time.time()

        for i, code in enumerate(missing_stocks):
            try:
                records = self.recover_stock(code)
                self.stats['stocks_processed'] += 1

                if records > 0:
                    logger.info(f"[{i+1}/{total}] {code}: {records} records recovered")
                else:
                    self.stats['skipped'] += 1
                    logger.warning(f"[{i+1}/{total}] {code}: No data available")

            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"[{i+1}/{total}] {code}: Error - {e}")

            # Progress report every 50 stocks
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = self.stats['records_fetched'] / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / (self.stats['stocks_processed'] / elapsed) if self.stats['stocks_processed'] > 0 else 0
                logger.info(f"Progress: {i+1}/{total} | Records: {self.stats['records_fetched']:,} | Rate: {rate:.1f}/s | ETA: {eta/60:.1f}min")

        # Final report
        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info("Recovery Complete")
        logger.info(f"  Processed stocks: {self.stats['stocks_processed']}")
        logger.info(f"  Records fetched: {self.stats['records_fetched']:,}")
        logger.info(f"  Skipped: {self.stats['skipped']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Elapsed time: {elapsed/60:.1f} min")
        logger.info("=" * 70)

        return self.stats


def verify_coverage():
    """Verify coverage after recovery"""
    cache_path = os.path.join(project_root, 'data', 'jquants_cache.db')
    conn = sqlite3.connect(cache_path)

    query = """
    SELECT
        substr(date, 1, 4) as year,
        COUNT(DISTINCT code) as stocks,
        COUNT(*) as records
    FROM daily_quotes
    WHERE date >= '2020-01-01'
    GROUP BY year
    ORDER BY year
    """

    import pandas as pd
    df = pd.read_sql_query(query, conn)
    conn.close()

    print("\n=== Post-Recovery Data Coverage ===")
    print(df.to_string(index=False))

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='2021+ Data Recovery')
    parser.add_argument('--max-stocks', type=int, default=None,
                        help='Maximum stocks to process (for testing)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Run coverage verification only')
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                        help='Fetch start date')
    parser.add_argument('--end-date', type=str, default='2026-02-03',
                        help='Fetch end date')

    args = parser.parse_args()

    if args.verify_only:
        verify_coverage()
        return

    cache_path = os.path.join(project_root, 'data', 'jquants_cache.db')

    engine = DataRecoveryEngine(cache_path)
    engine.run_recovery(max_stocks=args.max_stocks)

    # Verify results
    verify_coverage()


if __name__ == "__main__":
    main()
