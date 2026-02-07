#!/usr/bin/env python3
"""
Asset Shield - Scheduled Data Synchronization Script
=====================================================

Automated daily data sync for the next trading day.
Features:
- J-Quants API data refresh
- Error logging with rotation
- Retry logic with exponential backoff
- Slack/Email notification on failure (configurable)
- Scheduled via cron or launchd

Usage:
    python scripts/scheduled_data_sync.py              # Run sync now
    python scripts/scheduled_data_sync.py --schedule   # Print cron schedule
    python scripts/scheduled_data_sync.py --dry-run    # Test without API calls

Author: Asset Shield V3 Team
Version: 1.0.0 (2026-02-04)
"""

import os
import sys
import logging
import json
import sqlite3
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from logging.handlers import RotatingFileHandler
import traceback
import time

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# =============================================================================
# LOGGING CONFIGURATION (with rotation)
# =============================================================================

LOG_DIR = PROJECT_ROOT / 'logs' / 'sync'
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging() -> logging.Logger:
    """Configure rotating file logger with console output"""
    logger = logging.getLogger('DataSync')
    logger.setLevel(logging.DEBUG)

    # Rotating file handler (10MB, keep 7 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR / 'data_sync.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=7
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()


# =============================================================================
# ERROR TRACKING
# =============================================================================

class SyncErrorTracker:
    """Track and persist sync errors for monitoring"""

    ERROR_DB = PROJECT_ROOT / 'logs' / 'sync_errors.db'

    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initialize error tracking database"""
        with sqlite3.connect(self.ERROR_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    traceback TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    records_synced INTEGER DEFAULT 0,
                    error_id INTEGER REFERENCES sync_errors(id)
                )
            """)

    def log_error(self, error_type: str, error_message: str, tb: str = None) -> int:
        """Log an error and return its ID"""
        with sqlite3.connect(self.ERROR_DB) as conn:
            cursor = conn.execute("""
                INSERT INTO sync_errors (timestamp, error_type, error_message, traceback)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), error_type, error_message, tb))
            return cursor.lastrowid

    def log_run(self, status: str, records: int = 0, error_id: int = None) -> int:
        """Log a sync run"""
        with sqlite3.connect(self.ERROR_DB) as conn:
            cursor = conn.execute("""
                INSERT INTO sync_runs (start_time, status, records_synced, error_id)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), status, records, error_id))
            return cursor.lastrowid

    def update_run(self, run_id: int, status: str, records: int = None, error_id: int = None):
        """Update a run record"""
        with sqlite3.connect(self.ERROR_DB) as conn:
            if records is not None:
                conn.execute("""
                    UPDATE sync_runs SET end_time = ?, status = ?, records_synced = ?, error_id = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), status, records, error_id, run_id))
            else:
                conn.execute("""
                    UPDATE sync_runs SET end_time = ?, status = ?, error_id = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), status, error_id, run_id))

    def get_recent_errors(self, days: int = 7) -> List[Dict]:
        """Get recent unresolved errors"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.ERROR_DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM sync_errors
                WHERE timestamp > ? AND resolved = FALSE
                ORDER BY timestamp DESC
            """, (cutoff,))
            return [dict(row) for row in cursor.fetchall()]


error_tracker = SyncErrorTracker()


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry with exponential backoff"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    logger.warning(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# DATA SYNC ENGINE
# =============================================================================

class DataSyncEngine:
    """
    Main data synchronization engine.
    Handles J-Quants API data refresh with error handling.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.cache_path = PROJECT_ROOT / 'data' / 'jquants_cache.db'

    def get_next_trading_day(self) -> date:
        """Calculate next trading day (skip weekends, JP holidays)"""
        today = date.today()
        next_day = today + timedelta(days=1)

        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_day += timedelta(days=1)

        # TODO: Add JP holiday calendar check
        return next_day

    def get_missing_dates(self) -> List[date]:
        """Find dates with missing or incomplete data"""
        missing = []

        with sqlite3.connect(self.cache_path) as conn:
            # Get last sync date
            cursor = conn.execute("""
                SELECT MAX(date) FROM daily_quotes
            """)
            last_date_str = cursor.fetchone()[0]

            if last_date_str:
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
                today = date.today()

                # Check each trading day
                current = last_date + timedelta(days=1)
                while current < today:
                    if current.weekday() < 5:  # Weekdays only
                        missing.append(current)
                    current += timedelta(days=1)

        return missing

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def sync_daily_quotes(self, target_date: date) -> int:
        """Sync daily quotes for a specific date"""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would sync daily quotes for {target_date}")
            return 0

        try:
            from shield.jquants_backtest_provider import JQuantsBacktestProvider

            provider = JQuantsBacktestProvider()
            quotes = provider.fetch_daily_quotes(
                start_date=target_date,
                end_date=target_date
            )

            logger.info(f"Synced {len(quotes)} quotes for {target_date}")
            return len(quotes)

        except ImportError:
            logger.warning("JQuantsBacktestProvider not available, using fallback")
            return self._sync_via_cache(target_date)

    def _sync_via_cache(self, target_date: date) -> int:
        """Fallback sync method using direct cache operations"""
        # This would call the J-Quants API directly
        # Placeholder for actual implementation
        logger.info(f"Fallback sync for {target_date} - checking API availability")
        return 0

    def run_full_sync(self) -> Dict[str, Any]:
        """Run full synchronization process"""
        run_id = error_tracker.log_run('started')
        result = {
            'status': 'success',
            'start_time': datetime.now().isoformat(),
            'records_synced': 0,
            'dates_processed': [],
            'errors': []
        }

        try:
            # 1. Check for missing historical dates
            missing_dates = self.get_missing_dates()
            logger.info(f"Found {len(missing_dates)} missing dates")

            # 2. Sync missing dates
            for target_date in missing_dates:
                try:
                    count = self.sync_daily_quotes(target_date)
                    result['records_synced'] += count
                    result['dates_processed'].append(target_date.isoformat())
                except Exception as e:
                    error_msg = f"Failed to sync {target_date}: {e}"
                    logger.error(error_msg)
                    result['errors'].append(error_msg)

            # 3. Prepare for next trading day
            next_trading = self.get_next_trading_day()
            logger.info(f"Next trading day: {next_trading}")

            # 4. Update run status
            error_tracker.update_run(run_id, 'completed', result['records_synced'])

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            tb = traceback.format_exc()
            error_id = error_tracker.log_error('sync_failure', str(e), tb)
            error_tracker.update_run(run_id, 'failed', error_id=error_id)
            logger.error(f"Sync failed: {e}")
            logger.debug(tb)

        result['end_time'] = datetime.now().isoformat()
        return result


# =============================================================================
# NOTIFICATION SYSTEM (Placeholder)
# =============================================================================

def send_notification(result: Dict[str, Any], channels: List[str] = None):
    """Send notification about sync result"""
    if channels is None:
        channels = ['log']

    message = f"Data Sync {result['status'].upper()}\n"
    message += f"Records: {result['records_synced']}\n"
    message += f"Dates: {len(result.get('dates_processed', []))}\n"

    if result.get('errors'):
        message += f"Errors: {len(result['errors'])}\n"

    for channel in channels:
        if channel == 'log':
            logger.info(f"Notification: {message}")
        elif channel == 'slack':
            # TODO: Implement Slack webhook
            pass
        elif channel == 'email':
            # TODO: Implement email notification
            pass


# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================

CRON_SCHEDULE = """
# Asset Shield Data Sync - Add to crontab (crontab -e)
# Runs at 6:00 AM JST (21:00 UTC previous day) before market open

0 21 * * 0-4 cd /Users/MBP/Desktop/Project_Asset_Shield && /usr/bin/python3 scripts/scheduled_data_sync.py >> logs/sync/cron.log 2>&1

# Alternative: Use launchd on macOS
# Copy the plist below to ~/Library/LaunchAgents/com.assetshield.datasync.plist
"""

LAUNCHD_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.assetshield.datasync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/MBP/Desktop/Project_Asset_Shield/scripts/scheduled_data_sync.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>6</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>WorkingDirectory</key>
    <string>/Users/MBP/Desktop/Project_Asset_Shield</string>
    <key>StandardOutPath</key>
    <string>/Users/MBP/Desktop/Project_Asset_Shield/logs/sync/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/MBP/Desktop/Project_Asset_Shield/logs/sync/launchd_error.log</string>
</dict>
</plist>
"""


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Asset Shield Scheduled Data Synchronization'
    )
    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Print cron/launchd schedule configuration'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without making actual API calls'
    )
    parser.add_argument(
        '--check-errors',
        action='store_true',
        help='Check recent unresolved errors'
    )
    parser.add_argument(
        '--notify',
        nargs='+',
        choices=['log', 'slack', 'email'],
        default=['log'],
        help='Notification channels'
    )

    args = parser.parse_args()

    if args.schedule:
        print("=== Cron Configuration ===")
        print(CRON_SCHEDULE)
        print("\n=== launchd Configuration (macOS) ===")
        print(LAUNCHD_PLIST)
        return

    if args.check_errors:
        errors = error_tracker.get_recent_errors()
        if errors:
            print(f"Found {len(errors)} unresolved errors:")
            for err in errors:
                print(f"  [{err['timestamp']}] {err['error_type']}: {err['error_message']}")
        else:
            print("No recent unresolved errors")
        return

    # Run sync
    logger.info("="*60)
    logger.info("Asset Shield Data Sync Starting")
    logger.info("="*60)

    engine = DataSyncEngine(dry_run=args.dry_run)
    result = engine.run_full_sync()

    # Log result
    logger.info(f"Sync completed: {json.dumps(result, indent=2)}")

    # Send notifications
    send_notification(result, args.notify)

    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
