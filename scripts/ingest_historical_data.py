#!/usr/bin/env python3
"""
Asset Shield - Historical Data Ingestion Pipeline
==================================================

Comprehensive data ingestion system for 20-year survivorship-bias-free backtesting.

Features:
- Full universe fetch including DELISTED stocks (critical for survivorship bias elimination)
- 20-year daily OHLCV data (2006-2026)
- Financial statements for fundamental analysis
- Incremental ingestion with progress tracking
- Data integrity validation and reporting

Usage:
    # Load .env and run full ingestion
    python scripts/ingest_historical_data.py --mode full

    # Test with 10 years of major stocks
    python scripts/ingest_historical_data.py --mode test --years 10

    # Fetch only specific date range
    python scripts/ingest_historical_data.py --start 2016-01-01 --end 2026-01-31

    # Resume interrupted ingestion
    python scripts/ingest_historical_data.py --resume

Author: Asset Shield Development Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
import json
import sqlite3
import time
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict

# Load environment variables from .env
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

load_dotenv(os.path.join(project_root, '.env'))

from shield.jquants_client import JQuantsClient, JQuantsConfig
from shield.jquants_backtest_provider import (
    JQuantsBacktestProvider,
    DataCache,
    RateLimiter,
    DataTransformer,
    NormalizedQuote,
    DataValidationResult,
    create_jquants_provider
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'ingest.log'))
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

# Major Japanese stocks for testing (TOPIX Core 30 subset)
MAJOR_STOCKS = [
    "7203",  # Toyota Motor
    "9984",  # SoftBank Group
    "6758",  # Sony Group
    "8306",  # Mitsubishi UFJ
    "9432",  # NTT
    "6861",  # Keyence
    "6098",  # Recruit Holdings
    "4063",  # Shin-Etsu Chemical
    "8035",  # Tokyo Electron
    "6501",  # Hitachi
    "6902",  # Denso
    "7741",  # HOYA
    "4519",  # Chugai Pharmaceutical
    "6367",  # Daikin Industries
    "9433",  # KDDI
    "7267",  # Honda Motor
    "8316",  # Sumitomo Mitsui FG
    "4502",  # Takeda Pharmaceutical
    "6954",  # Fanuc
    "3382",  # Seven & i Holdings
]

# Phase date ranges for structured ingestion
# NOTE: J-Quants API data is available from 2007-01-04 onwards
PHASE_RANGES = {
    "phase_1_survival": {
        "start": "2007-01-04",  # J-Quants earliest available date
        "end": "2010-12-31",
        "description": "Lehman Shock & Recovery (Training)"
    },
    "phase_2_expansion": {
        "start": "2011-01-01",
        "end": "2015-12-31",
        "description": "Abenomics Era (Training)"
    },
    "phase_3_oos": {
        "start": "2016-01-01",
        "end": "2020-12-31",
        "description": "COVID Shock (Verification)"
    },
    "phase_4_modern": {
        "start": "2021-01-01",
        "end": "2026-12-31",
        "description": "Modern Era (Verification)"
    }
}


# ==============================================================================
# Code Normalization Utilities
# ==============================================================================

def normalize_stock_code(code: str) -> str:
    """
    Normalize stock code to J-Quants API v2 format (5-digit).

    J-Quants API v2 uses 5-digit codes where:
    - Ordinary shares end with '0' (e.g., 72030 for Toyota)
    - ETFs and REITs may use different suffixes

    Args:
        code: Stock code (4 or 5 digits)

    Returns:
        Normalized 5-digit code string
    """
    code = str(code).strip()

    # Already 5-digit code
    if len(code) == 5:
        return code

    # 4-digit code: append '0' for ordinary shares
    if len(code) == 4:
        return code + "0"

    # Handle other cases (log warning and return as-is)
    if len(code) < 4:
        logger.warning(f"Unusual stock code length ({len(code)}): {code}")
        return code.zfill(5)  # Pad with zeros

    return code


def denormalize_stock_code(code: str) -> str:
    """
    Convert 5-digit code back to 4-digit for display purposes.

    Args:
        code: 5-digit stock code

    Returns:
        4-digit code if ends with '0', otherwise original
    """
    code = str(code).strip()
    if len(code) == 5 and code.endswith('0'):
        return code[:-1]
    return code


def normalize_stock_codes(codes: List[str]) -> List[str]:
    """Normalize a list of stock codes to 5-digit format."""
    return [normalize_stock_code(code) for code in codes]


# ==============================================================================
# Data Models
# ==============================================================================

@dataclass
class IngestionProgress:
    """Track ingestion progress for resumability"""
    started_at: str
    last_updated: str
    mode: str
    total_stocks: int
    processed_stocks: int
    failed_stocks: List[str]
    current_phase: str
    phases_completed: List[str]
    total_records_ingested: int

    def save(self, path: str) -> None:
        """Save progress to file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> Optional["IngestionProgress"]:
        """Load progress from file"""
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.load(f)
            return cls(**data)


@dataclass
class IngestionReport:
    """Final ingestion report"""
    run_id: str
    started_at: str
    completed_at: str
    mode: str
    plan: str
    stocks_requested: int
    stocks_succeeded: int
    stocks_failed: int
    total_records: int
    date_range: Dict[str, str]
    phases_completed: List[str]
    validation_summary: Dict[str, Any]
    api_calls_made: int
    cache_size_mb: float
    errors: List[str]


# ==============================================================================
# Extended Database Schema
# ==============================================================================

def init_extended_schema(cache_path: str) -> None:
    """
    Initialize extended database schema for comprehensive data storage.
    Adds tables for:
    - Historical universe snapshots (for survivorship bias elimination)
    - Financial statements
    - Ingestion metadata
    """
    with sqlite3.connect(cache_path) as conn:
        # Historical universe snapshots - critical for survivorship bias
        conn.execute("""
            CREATE TABLE IF NOT EXISTS universe_snapshots (
                snapshot_date TEXT,
                code TEXT,
                company_name TEXT,
                market_code TEXT,
                sector33_code TEXT,
                listing_date TEXT,
                delisting_date TEXT,
                is_active INTEGER,
                cached_at TEXT,
                PRIMARY KEY (snapshot_date, code)
            )
        """)

        # Financial statements for fundamental analysis
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_statements (
                code TEXT,
                disclosed_date TEXT,
                fiscal_year TEXT,
                fiscal_quarter TEXT,
                net_sales REAL,
                operating_profit REAL,
                ordinary_profit REAL,
                net_income REAL,
                eps REAL,
                total_assets REAL,
                equity REAL,
                equity_ratio REAL,
                bps REAL,
                roe REAL,
                cached_at TEXT,
                PRIMARY KEY (code, disclosed_date, fiscal_quarter)
            )
        """)

        # Ingestion metadata for tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT,
                completed_at TEXT,
                mode TEXT,
                stocks_count INTEGER,
                records_count INTEGER,
                status TEXT,
                error_message TEXT
            )
        """)

        # Create indexes for efficient queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_universe_date
            ON universe_snapshots(snapshot_date)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_universe_code
            ON universe_snapshots(code)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fins_code_date
            ON financial_statements(code, disclosed_date)
        """)

        conn.commit()
        logger.info("Extended database schema initialized")


# ==============================================================================
# Universe Management (Survivorship Bias Elimination)
# ==============================================================================

class UniverseManager:
    """
    Manages the historical universe of stocks including delisted companies.
    Critical for eliminating survivorship bias in backtests.
    """

    def __init__(self, client: JQuantsClient, cache_path: str):
        self.client = client
        self.cache_path = cache_path

    def fetch_current_universe(self) -> List[Dict]:
        """Fetch current listed stocks from J-Quants API"""
        logger.info("Fetching current universe from J-Quants...")
        info = self.client.get_listed_info()
        if info:
            logger.info(f"Retrieved {len(info)} listed stocks")
        return info or []

    def fetch_historical_universe(self, target_date: str) -> List[Dict]:
        """
        Fetch universe snapshot for a specific historical date.
        This includes stocks that were listed at that time but may have since delisted.
        """
        logger.info(f"Fetching universe snapshot for {target_date}...")
        info = self.client.get_listed_info(date=target_date)
        if info:
            logger.info(f"Retrieved {len(info)} stocks for {target_date}")
        return info or []

    def save_universe_snapshot(self, snapshot_date: str, stocks: List[Dict]) -> int:
        """
        Save universe snapshot to database.

        Handles both J-Quants API v1 and v2 field names.
        v1: CompanyName, MarketCode, Sector33Code
        v2: CoName, Mkt, S33 (converted by client)
        """
        with sqlite3.connect(self.cache_path) as conn:
            saved = 0
            for stock in stocks:
                try:
                    # Get code and normalize to 5-digit format
                    code = stock.get("Code", "")
                    if not code:
                        continue

                    # Normalize code to 5-digit J-Quants v2 format
                    code = normalize_stock_code(code)

                    # Handle v1/v2 field name differences with fallbacks
                    company_name = (
                        stock.get("CompanyName") or
                        stock.get("CoName") or
                        ""
                    )
                    market_code = (
                        stock.get("MarketCode") or
                        stock.get("Mkt") or
                        ""
                    )
                    sector33_code = (
                        stock.get("Sector33Code") or
                        stock.get("S33") or
                        ""
                    )

                    # Date field in v2 is the snapshot date, not listing date
                    listing_date = stock.get("Date", snapshot_date)

                    # v2 API does not provide DelistingDate in eq-master endpoint
                    delisting_date = stock.get("DelistingDate")

                    conn.execute("""
                        INSERT OR REPLACE INTO universe_snapshots
                        (snapshot_date, code, company_name, market_code, sector33_code,
                         listing_date, delisting_date, is_active, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        snapshot_date,
                        code,
                        company_name,
                        market_code,
                        sector33_code,
                        listing_date,
                        delisting_date,
                        1 if not delisting_date else 0,
                        datetime.now().isoformat()
                    ))
                    saved += 1
                except Exception as e:
                    logger.warning(f"Failed to save stock {stock.get('Code')}: {e}")
            conn.commit()
        return saved

    def get_universe_for_date(self, target_date: str) -> List[str]:
        """
        Get list of stock codes that were listed on a specific date.
        Returns codes that were active (listed but not yet delisted) on that date.
        """
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT code FROM universe_snapshots
                WHERE snapshot_date <= ?
                AND (delisting_date IS NULL OR delisting_date > ?)
                ORDER BY code
            """, (target_date, target_date))
            return [row[0] for row in cursor.fetchall()]

    def get_all_historical_codes(self) -> Set[str]:
        """Get all stock codes that have ever been in the universe (for complete data fetch)"""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT DISTINCT code FROM universe_snapshots")
            return {row[0] for row in cursor.fetchall()}

    def build_historical_snapshots(
        self,
        start_year: int = 2007,  # J-Quants data starts from 2007
        end_year: int = 2026,
        rate_limiter: Optional[RateLimiter] = None
    ) -> int:
        """
        Build historical universe snapshots for each year.
        This captures which stocks were listed at different points in time.
        """
        total_saved = 0

        for year in range(start_year, end_year + 1):
            # Take snapshot at year start (January 1st)
            snapshot_date = f"{year}-01-01"

            logger.info(f"Building universe snapshot for {snapshot_date}...")

            if rate_limiter:
                rate_limiter.wait_if_needed()

            try:
                stocks = self.fetch_historical_universe(snapshot_date)
                if stocks:
                    saved = self.save_universe_snapshot(snapshot_date, stocks)
                    total_saved += saved
                    logger.info(f"  Saved {saved} stocks for {snapshot_date}")
                else:
                    logger.warning(f"  No data for {snapshot_date}")
            except Exception as e:
                logger.error(f"  Failed to fetch universe for {snapshot_date}: {e}")

        return total_saved


# ==============================================================================
# Financial Statements Ingestion
# ==============================================================================

class FinancialDataIngester:
    """Fetches and stores financial statements data"""

    def __init__(self, client: JQuantsClient, cache_path: str):
        self.client = client
        self.cache_path = cache_path

    def fetch_statements(self, code: str, date: Optional[str] = None) -> List[Dict]:
        """Fetch financial statements for a stock"""
        statements = self.client.get_financial_statements(code=code, date=date)
        return statements or []

    def save_statements(self, statements: List[Dict]) -> int:
        """Save financial statements to database"""
        with sqlite3.connect(self.cache_path) as conn:
            saved = 0
            for stmt in statements:
                try:
                    # Parse financial values (handle string/None values)
                    def parse_float(val):
                        if val is None or val == "":
                            return None
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return None

                    conn.execute("""
                        INSERT OR REPLACE INTO financial_statements
                        (code, disclosed_date, fiscal_year, fiscal_quarter,
                         net_sales, operating_profit, ordinary_profit, net_income,
                         eps, total_assets, equity, equity_ratio, bps, roe, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        stmt.get("Code") or stmt.get("LocalCode"),
                        stmt.get("DisclosedDate"),
                        stmt.get("CurrentPeriodEndDate", "")[:4] if stmt.get("CurrentPeriodEndDate") else None,
                        stmt.get("TypeOfCurrentPeriod"),
                        parse_float(stmt.get("NetSales")),
                        parse_float(stmt.get("OperatingProfit")),
                        parse_float(stmt.get("OrdinaryProfit")),
                        parse_float(stmt.get("Profit")),
                        parse_float(stmt.get("EarningsPerShare")),
                        parse_float(stmt.get("TotalAssets")),
                        parse_float(stmt.get("Equity")),
                        parse_float(stmt.get("EquityToAssetRatio")),
                        parse_float(stmt.get("BookValuePerShare")),
                        parse_float(stmt.get("ReturnOnEquity")),
                        datetime.now().isoformat()
                    ))
                    saved += 1
                except Exception as e:
                    logger.warning(f"Failed to save statement: {e}")
            conn.commit()
        return saved

    def ingest_for_stock(
        self,
        code: str,
        rate_limiter: Optional[RateLimiter] = None
    ) -> int:
        """Ingest all available financial statements for a stock"""
        if rate_limiter:
            rate_limiter.wait_if_needed()

        try:
            statements = self.fetch_statements(code)
            if statements:
                saved = self.save_statements(statements)
                return saved
            return 0
        except Exception as e:
            logger.error(f"Failed to fetch statements for {code}: {e}")
            return 0


# ==============================================================================
# Main Ingestion Pipeline
# ==============================================================================

class HistoricalDataPipeline:
    """
    Main orchestrator for historical data ingestion.
    Handles the complete workflow of fetching 20 years of data.
    """

    def __init__(
        self,
        plan: str = "premium",
        cache_path: str = "data/jquants_cache.db"
    ):
        self.plan = plan
        self.cache_path = os.path.join(project_root, cache_path)
        self.progress_path = os.path.join(project_root, "data", "ingestion_progress.json")

        # Initialize extended schema
        init_extended_schema(self.cache_path)

        # Create provider
        self.provider = create_jquants_provider(
            plan=plan,
            credential_storage="env"
        )

        # Initialize components
        self.rate_limiter = RateLimiter(plan)
        self.universe_manager = UniverseManager(
            self.provider.client,
            self.cache_path
        )
        self.financial_ingester = FinancialDataIngester(
            self.provider.client,
            self.cache_path
        )

        # Track run
        self.run_id = f"INGEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.errors: List[str] = []

    def verify_credentials(self) -> bool:
        """Verify J-Quants Premium credentials work"""
        logger.info("Verifying J-Quants credentials...")

        try:
            # Try to fetch a small piece of data
            info = self.provider.client.get_listed_info(code="7203")
            if info:
                logger.info(f"  Credentials verified. Test fetch: {info[0].get('CompanyName', 'Toyota')}")
                logger.info(f"  Plan: {self.plan}")
                logger.info(f"  Rate limit: {self.rate_limiter.limits['rpm']} req/min")
                return True
            else:
                logger.error("  Failed to fetch test data")
                return False
        except Exception as e:
            logger.error(f"  Credential verification failed: {e}")
            return False

    def ingest_daily_quotes(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        progress: Optional[IngestionProgress] = None
    ) -> Tuple[int, Dict[str, DataValidationResult]]:
        """
        Ingest daily OHLCV quotes for specified stocks and date range.

        Returns:
            Tuple of (total_records_ingested, validation_results_by_code)
        """
        total_records = 0
        validation_results: Dict[str, DataValidationResult] = {}

        logger.info(f"\n{'='*60}")
        logger.info(f"Ingesting daily quotes: {start_date} to {end_date}")
        logger.info(f"Stocks: {len(codes)}")
        logger.info(f"{'='*60}")

        for i, original_code in enumerate(codes):
            # Normalize code to 5-digit J-Quants v2 format
            code = normalize_stock_code(original_code)

            # Skip already processed if resuming
            if progress and code in progress.failed_stocks:
                continue

            logger.info(f"\n[{i+1}/{len(codes)}] Processing {code}...")

            try:
                self.rate_limiter.wait_if_needed()

                # Fetch quotes with normalized code
                quotes = self.provider.fetch_daily_quotes(
                    code=code,
                    from_date=start_date,
                    to_date=end_date,
                    use_cache=True  # Use cache for already-fetched data
                )

                if quotes:
                    # Normalize and validate
                    source = "api" if "Code" in quotes[0] else "cache"
                    normalized = DataTransformer.normalize_quotes(quotes, source)

                    # Validate data quality
                    validation = DataTransformer.validate_data(
                        normalized,
                        expected_start=datetime.strptime(start_date, "%Y-%m-%d").date(),
                        expected_end=datetime.strptime(end_date, "%Y-%m-%d").date()
                    )
                    validation_results[code] = validation

                    total_records += len(normalized)

                    logger.info(f"  Fetched {len(quotes)} records")
                    logger.info(f"  Valid: {validation.valid_records}, Invalid: {validation.invalid_records}")

                    if validation.warnings:
                        for w in validation.warnings[:3]:  # Show first 3 warnings
                            logger.warning(f"    {w}")
                else:
                    logger.warning(f"  No data available for {code}")
                    self.errors.append(f"No data for {code}")

            except Exception as e:
                logger.error(f"  Failed to process {code}: {e}")
                self.errors.append(f"Failed {code}: {str(e)}")
                if progress:
                    progress.failed_stocks.append(code)

            # Update progress
            if progress:
                progress.processed_stocks = i + 1
                progress.total_records_ingested = total_records
                progress.last_updated = datetime.now().isoformat()
                progress.save(self.progress_path)

        return total_records, validation_results

    def ingest_financial_data(
        self,
        codes: List[str]
    ) -> int:
        """Ingest financial statements for all stocks"""
        total_statements = 0

        logger.info(f"\n{'='*60}")
        logger.info("Ingesting financial statements")
        logger.info(f"{'='*60}")

        for i, code in enumerate(codes):
            logger.info(f"[{i+1}/{len(codes)}] Financial data for {code}...")

            try:
                saved = self.financial_ingester.ingest_for_stock(
                    code,
                    self.rate_limiter
                )
                total_statements += saved
                if saved > 0:
                    logger.info(f"  Saved {saved} statements")
            except Exception as e:
                logger.error(f"  Failed: {e}")

        return total_statements

    def run_phase_ingestion(
        self,
        phase_name: str,
        include_financials: bool = True
    ) -> IngestionReport:
        """
        Run ingestion for a single phase (for parallel execution).

        Args:
            phase_name: One of PHASE_RANGES keys (phase_1_survival, phase_2_expansion, etc.)
            include_financials: Whether to include financial statements
        """
        start_time = datetime.now()

        if phase_name not in PHASE_RANGES:
            raise ValueError(f"Unknown phase: {phase_name}. Valid: {list(PHASE_RANGES.keys())}")

        phase_config = PHASE_RANGES[phase_name]

        logger.info("\n" + "="*70)
        logger.info(f"ASSET SHIELD - PHASE INGESTION: {phase_name}")
        logger.info(f"Description: {phase_config['description']}")
        logger.info(f"Period: {phase_config['start']} to {phase_config['end']}")
        logger.info("="*70)

        # Verify credentials first
        if not self.verify_credentials():
            raise RuntimeError("Credential verification failed")

        # Extract years from phase dates
        start_year = int(phase_config['start'][:4])
        end_year = int(phase_config['end'][:4])

        # Step 1: Build historical universe for this phase's years
        logger.info(f"\n[STEP 1/4] Building universe snapshots for {start_year}-{end_year}...")
        universe_count = self.universe_manager.build_historical_snapshots(
            start_year=start_year,
            end_year=end_year,
            rate_limiter=self.rate_limiter
        )
        logger.info(f"  Universe records: {universe_count}")

        # Get all codes
        all_codes = list(self.universe_manager.get_all_historical_codes())
        if not all_codes:
            current = self.universe_manager.fetch_current_universe()
            all_codes = [s.get("Code") for s in current if s.get("Code")]

        logger.info(f"  Total stocks: {len(all_codes)}")

        # Step 2: Ingest daily quotes for this phase
        logger.info(f"\n[STEP 2/4] Ingesting daily OHLCV for {phase_name}...")

        records, validation = self.ingest_daily_quotes(
            codes=all_codes,
            start_date=phase_config['start'],
            end_date=phase_config['end']
        )

        # Step 3: Ingest financials (only for first phase to avoid duplicates)
        if include_financials and phase_name == "phase_1_survival":
            logger.info("\n[STEP 3/4] Ingesting financial statements...")
            statements_count = self.ingest_financial_data(all_codes)
            logger.info(f"  Statements: {statements_count}")
        else:
            logger.info(f"\n[STEP 3/4] Skipping financials (handled by phase_1 or disabled)")

        # Step 4: Report
        end_time = datetime.now()
        duration = end_time - start_time

        cache_stats = self.provider.cache.get_cache_stats()
        valid_stocks = sum(1 for v in validation.values() if v.is_valid)

        report = IngestionReport(
            run_id=f"{self.run_id}_{phase_name}",
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            mode=f"phase:{phase_name}",
            plan=self.plan,
            stocks_requested=len(all_codes),
            stocks_succeeded=valid_stocks,
            stocks_failed=len(all_codes) - valid_stocks,
            total_records=records,
            date_range={"start": phase_config['start'], "end": phase_config['end']},
            phases_completed=[phase_name],
            validation_summary={
                "valid_stocks": valid_stocks,
                "phase": phase_name
            },
            api_calls_made=self.rate_limiter.daily_count,
            cache_size_mb=cache_stats.get("cache_size_mb", 0),
            errors=self.errors
        )

        # Save phase report
        report_path = os.path.join(project_root, "output", f"ingestion_report_{self.run_id}_{phase_name}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        logger.info("\n" + "="*70)
        logger.info(f"PHASE {phase_name} COMPLETE")
        logger.info("="*70)
        logger.info(f"Duration: {duration}")
        logger.info(f"Records: {records:,}")
        logger.info(f"Report: {report_path}")

        return report

    def run_full_ingestion(
        self,
        start_year: int = 2007,  # J-Quants data starts from 2007
        end_year: int = 2026,
        include_financials: bool = True
    ) -> IngestionReport:
        """
        Run complete 20-year data ingestion.

        Steps:
        1. Build historical universe snapshots
        2. Ingest daily quotes for all phases
        3. Ingest financial statements
        4. Generate validation report
        """
        start_time = datetime.now()

        logger.info("\n" + "="*70)
        logger.info("ASSET SHIELD - FULL HISTORICAL DATA INGESTION")
        logger.info(f"Period: {start_year} to {end_year}")
        logger.info("="*70)

        # Verify credentials first
        if not self.verify_credentials():
            raise RuntimeError("Credential verification failed")

        # Step 1: Build historical universe
        logger.info("\n[STEP 1/4] Building historical universe snapshots...")
        universe_count = self.universe_manager.build_historical_snapshots(
            start_year=start_year,
            end_year=end_year,
            rate_limiter=self.rate_limiter
        )
        logger.info(f"  Total universe records: {universe_count}")

        # Get all codes to process
        all_codes = list(self.universe_manager.get_all_historical_codes())
        if not all_codes:
            # Fallback to current universe if no historical data
            current = self.universe_manager.fetch_current_universe()
            all_codes = [s.get("Code") for s in current if s.get("Code")]

        logger.info(f"  Total unique stocks to ingest: {len(all_codes)}")

        # Initialize progress tracking
        progress = IngestionProgress(
            started_at=start_time.isoformat(),
            last_updated=start_time.isoformat(),
            mode="full",
            total_stocks=len(all_codes),
            processed_stocks=0,
            failed_stocks=[],
            current_phase="",
            phases_completed=[],
            total_records_ingested=0
        )

        # Step 2: Ingest daily quotes by phase
        logger.info("\n[STEP 2/4] Ingesting daily OHLCV data by phase...")
        total_records = 0
        all_validation: Dict[str, DataValidationResult] = {}

        for phase_name, phase_config in PHASE_RANGES.items():
            progress.current_phase = phase_name
            progress.save(self.progress_path)

            logger.info(f"\n--- Phase: {phase_name} ---")
            logger.info(f"    {phase_config['description']}")
            logger.info(f"    Period: {phase_config['start']} to {phase_config['end']}")

            records, validation = self.ingest_daily_quotes(
                codes=all_codes,
                start_date=phase_config['start'],
                end_date=phase_config['end'],
                progress=progress
            )

            total_records += records
            all_validation.update(validation)
            progress.phases_completed.append(phase_name)
            progress.save(self.progress_path)

        # Step 3: Ingest financial statements
        if include_financials:
            logger.info("\n[STEP 3/4] Ingesting financial statements...")
            statements_count = self.ingest_financial_data(all_codes)
            logger.info(f"  Total financial statements: {statements_count}")
        else:
            logger.info("\n[STEP 3/4] Skipping financial statements (disabled)")
            statements_count = 0

        # Step 4: Generate report
        logger.info("\n[STEP 4/4] Generating validation report...")

        end_time = datetime.now()
        duration = end_time - start_time

        # Get cache stats
        cache_stats = self.provider.cache.get_cache_stats()
        api_stats = self.provider.cache.get_api_usage_stats(days=1)

        # Validation summary
        valid_stocks = sum(1 for v in all_validation.values() if v.is_valid)
        invalid_stocks = len(all_validation) - valid_stocks

        report = IngestionReport(
            run_id=self.run_id,
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            mode="full",
            plan=self.plan,
            stocks_requested=len(all_codes),
            stocks_succeeded=valid_stocks,
            stocks_failed=invalid_stocks,
            total_records=total_records,
            date_range={
                "start": f"{start_year}-01-01",
                "end": f"{end_year}-12-31"
            },
            phases_completed=list(PHASE_RANGES.keys()),
            validation_summary={
                "valid_stocks": valid_stocks,
                "invalid_stocks": invalid_stocks,
                "total_warnings": sum(len(v.warnings) for v in all_validation.values()),
                "stocks_with_gaps": sum(1 for v in all_validation.values() if v.missing_dates)
            },
            api_calls_made=api_stats.get("total_calls", 0),
            cache_size_mb=cache_stats.get("cache_size_mb", 0),
            errors=self.errors
        )

        # Save report
        report_path = os.path.join(project_root, "output", f"ingestion_report_{self.run_id}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("INGESTION COMPLETE")
        logger.info("="*70)
        logger.info(f"Duration: {duration}")
        logger.info(f"Total records: {total_records:,}")
        logger.info(f"Valid stocks: {valid_stocks} / {len(all_codes)}")
        logger.info(f"Cache size: {cache_stats.get('cache_size_mb', 0):.2f} MB")
        logger.info(f"API calls: {api_stats.get('total_calls', 0)}")
        logger.info(f"Report saved: {report_path}")

        return report

    def run_test_ingestion(
        self,
        years: int = 10,
        stocks: Optional[List[str]] = None
    ) -> IngestionReport:
        """
        Run test ingestion with limited scope.

        Args:
            years: Number of years to ingest (from current date backwards)
            stocks: Specific stocks to ingest (default: MAJOR_STOCKS)
        """
        start_time = datetime.now()

        test_stocks = stocks or MAJOR_STOCKS
        end_year = datetime.now().year
        start_year = end_year - years + 1

        logger.info("\n" + "="*70)
        logger.info("ASSET SHIELD - TEST INGESTION")
        logger.info(f"Period: {start_year} to {end_year} ({years} years)")
        logger.info(f"Stocks: {len(test_stocks)}")
        logger.info("="*70)

        # Verify credentials
        if not self.verify_credentials():
            raise RuntimeError("Credential verification failed")

        # Ingest quotes
        total_records = 0
        all_validation: Dict[str, DataValidationResult] = {}

        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        records, validation = self.ingest_daily_quotes(
            codes=test_stocks,
            start_date=start_date,
            end_date=end_date
        )

        total_records = records
        all_validation = validation

        end_time = datetime.now()
        duration = end_time - start_time

        # Get cache stats
        cache_stats = self.provider.cache.get_cache_stats()

        # Generate report
        valid_stocks = sum(1 for v in all_validation.values() if v.is_valid)

        report = IngestionReport(
            run_id=self.run_id,
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            mode="test",
            plan=self.plan,
            stocks_requested=len(test_stocks),
            stocks_succeeded=valid_stocks,
            stocks_failed=len(test_stocks) - valid_stocks,
            total_records=total_records,
            date_range={"start": start_date, "end": end_date},
            phases_completed=["test"],
            validation_summary={
                "valid_stocks": valid_stocks,
                "stocks_with_issues": len(test_stocks) - valid_stocks
            },
            api_calls_made=self.rate_limiter.daily_count,
            cache_size_mb=cache_stats.get("cache_size_mb", 0),
            errors=self.errors
        )

        # Print detailed validation for each stock
        logger.info("\n" + "="*70)
        logger.info("DATA INTEGRITY REPORT")
        logger.info("="*70)

        for code, v in sorted(all_validation.items()):
            status = "OK" if v.is_valid else "ISSUES"
            logger.info(f"\n{code}: [{status}]")
            logger.info(f"  Records: {v.valid_records:,} valid, {v.invalid_records} invalid")
            if v.missing_dates:
                logger.info(f"  Missing dates: {len(v.missing_dates)} (may be holidays)")
            if v.zero_volume_dates:
                logger.info(f"  Zero volume days: {len(v.zero_volume_dates)}")
            if v.price_anomalies:
                logger.info(f"  Price anomalies: {len(v.price_anomalies)}")
                for anomaly in v.price_anomalies[:3]:
                    logger.info(f"    - {anomaly}")

        logger.info("\n" + "="*70)
        logger.info("TEST INGESTION COMPLETE")
        logger.info("="*70)
        logger.info(f"Duration: {duration}")
        logger.info(f"Total records: {total_records:,}")
        logger.info(f"Valid stocks: {valid_stocks} / {len(test_stocks)}")
        logger.info(f"Cache size: {cache_stats.get('cache_size_mb', 0):.2f} MB")

        return report


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Asset Shield Historical Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with 10 years of major stocks
    python scripts/ingest_historical_data.py --mode test --years 10

    # Full 20-year ingestion
    python scripts/ingest_historical_data.py --mode full

    # Custom date range
    python scripts/ingest_historical_data.py --start 2016-01-01 --end 2026-01-31

    # Dry run to check setup
    python scripts/ingest_historical_data.py --dry-run
        """
    )

    parser.add_argument(
        "--mode",
        choices=["test", "full", "custom"],
        default="test",
        help="Ingestion mode: test (major stocks), full (all stocks), custom (specify range)"
    )

    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years for test mode (default: 10)"
    )

    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD) for custom mode"
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD) for custom mode"
    )

    parser.add_argument(
        "--stocks",
        type=str,
        help="Comma-separated stock codes (default: TOPIX Core 30 subset)"
    )

    parser.add_argument(
        "--plan",
        choices=["free", "light", "standard", "premium"],
        default="premium",
        help="J-Quants subscription plan"
    )

    parser.add_argument(
        "--no-financials",
        action="store_true",
        help="Skip financial statements ingestion"
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=list(PHASE_RANGES.keys()),
        help="Run specific phase only (for parallel execution)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check setup without running ingestion"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted ingestion"
    )

    args = parser.parse_args()

    # Check for .env
    env_path = os.path.join(project_root, '.env')
    if not os.path.exists(env_path):
        logger.error(f".env file not found at {env_path}")
        logger.error("Please create .env with JQUANTS_MAIL and JQUANTS_PASSWORD")
        sys.exit(1)

    # Load plan from environment if not specified
    plan = os.environ.get("JQUANTS_PLAN", args.plan)

    if args.dry_run:
        logger.info("=== DRY RUN - Checking Setup ===\n")

        logger.info("Environment:")
        logger.info(f"  JQUANTS_MAIL: {'SET' if os.environ.get('JQUANTS_MAIL') else 'NOT SET'}")
        logger.info(f"  JQUANTS_PASSWORD: {'SET' if os.environ.get('JQUANTS_PASSWORD') else 'NOT SET'}")
        logger.info(f"  Plan: {plan}")

        # Try to create pipeline and verify
        try:
            pipeline = HistoricalDataPipeline(plan=plan)
            if pipeline.verify_credentials():
                logger.info("\n  Credentials: VERIFIED")
            else:
                logger.error("\n  Credentials: FAILED")
                sys.exit(1)
        except Exception as e:
            logger.error(f"\n  Setup error: {e}")
            sys.exit(1)

        logger.info("\n  Setup OK. Remove --dry-run to run ingestion.")
        return

    # Parse stocks if provided
    stocks = None
    if args.stocks:
        stocks = [s.strip() for s in args.stocks.split(",")]

    # Create pipeline
    pipeline = HistoricalDataPipeline(plan=plan)

    try:
        # Single phase execution (for parallel tmux execution)
        if args.phase:
            logger.info(f"Running single phase: {args.phase}")
            report = pipeline.run_phase_ingestion(
                phase_name=args.phase,
                include_financials=not args.no_financials
            )
        elif args.mode == "test":
            report = pipeline.run_test_ingestion(
                years=args.years,
                stocks=stocks
            )
        elif args.mode == "full":
            report = pipeline.run_full_ingestion(
                include_financials=not args.no_financials
            )
        elif args.mode == "custom":
            if not args.start or not args.end:
                logger.error("Custom mode requires --start and --end dates")
                sys.exit(1)

            # Use test ingestion with custom dates
            test_stocks = stocks or MAJOR_STOCKS
            start_date = datetime.strptime(args.start, "%Y-%m-%d")
            end_date = datetime.strptime(args.end, "%Y-%m-%d")
            years = (end_date - start_date).days // 365 + 1

            report = pipeline.run_test_ingestion(
                years=years,
                stocks=test_stocks
            )

        # Save final report
        report_path = os.path.join(
            project_root, "output",
            f"ingestion_report_{report.run_id}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        logger.info(f"\nReport saved to: {report_path}")

    except KeyboardInterrupt:
        logger.warning("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nIngestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
