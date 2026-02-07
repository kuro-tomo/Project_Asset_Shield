#!/usr/bin/env python3
"""
Generate QuantConnect DataUploader with Full CSV Data
=====================================================

This script generates a complete DataUploader class with all 20 stocks'
CSV data embedded, ready to copy-paste to QuantConnect.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'quantconnect', 'custom_data')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'quantconnect', 'AssetShieldJP', 'data_uploader_full.py')

# Target stocks
STOCK_CODES = [
    "72030", "67580", "83060", "80350", "68610",
    "94320", "65010", "79740", "40630", "99840",
    "69020", "63670", "94330", "45020", "72670",
    "45030", "69540", "87660", "83160", "90220"
]


def generate_uploader():
    """Generate full data uploader code"""
    print("Generating QuantConnect DataUploader...")

    # Header
    code = '''# region imports
from AlgorithmImports import *
# endregion

class DataUploaderFull(QCAlgorithm):
    """
    Asset Shield V3.2.0 - Full Data Uploader
    =========================================

    This uploads all 20 Japanese stock CSV data to ObjectStore.
    Run this FIRST, then run AssetShieldJPObjectStore.

    Generated: Auto-generated from local CSV files
    Stocks: 20
    Period: 2008-2026
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 2)
        self.SetCash(1000)

        # ============================================================
        # STOCK DATA (20 stocks, 2008-2026)
        # ============================================================

        STOCK_DATA = {
'''

    # Add each stock's data
    for stock_code in STOCK_CODES:
        csv_path = os.path.join(DATA_DIR, f'{stock_code}.csv')

        if not os.path.exists(csv_path):
            print(f"  ✗ {stock_code}: File not found")
            continue

        with open(csv_path, 'r') as f:
            csv_content = f.read().strip()

        lines = csv_content.split('\n')
        print(f"  ✓ {stock_code}: {len(lines)} lines")

        # Add to code (escape triple quotes if needed)
        code += f'            "{stock_code}": """{csv_content}""",\n\n'

    # Footer
    code += '''        }

        # Upload each stock's data to ObjectStore
        uploaded = 0
        total_lines = 0

        for code, csv_data in STOCK_DATA.items():
            if csv_data.strip():
                key = f"japan_stocks/{code}.csv"
                self.ObjectStore.Save(key, csv_data)
                lines = len(csv_data.strip().split('\\n'))
                total_lines += lines
                self.Debug(f"Uploaded: {key} ({lines} lines)")
                uploaded += 1

        self.Debug("=" * 60)
        self.Debug(f"Upload complete!")
        self.Debug(f"Stocks: {uploaded}")
        self.Debug(f"Total data points: {total_lines}")
        self.Debug("=" * 60)
        self.Debug("NEXT: Run AssetShieldJPObjectStore for backtest")

    def OnData(self, data):
        pass
'''

    # Write to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(code)

    print(f"\n✓ Generated: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # Also create a smaller test version with just 3 stocks
    generate_test_uploader()


def generate_test_uploader():
    """Generate smaller test uploader with 3 stocks"""
    test_output = os.path.join(PROJECT_ROOT, 'quantconnect', 'AssetShieldJP', 'data_uploader_test.py')

    code = '''# region imports
from AlgorithmImports import *
# endregion

class DataUploaderTest(QCAlgorithm):
    """
    Asset Shield V3.2.0 - Test Data Uploader (3 stocks)
    ===================================================

    Quick test with 3 stocks to verify ObjectStore works.
    """

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 2)
        self.SetCash(1000)

        STOCK_DATA = {
'''

    # Add just 3 stocks for testing
    test_stocks = ["72030", "67580", "83060"]

    for stock_code in test_stocks:
        csv_path = os.path.join(DATA_DIR, f'{stock_code}.csv')

        if not os.path.exists(csv_path):
            continue

        with open(csv_path, 'r') as f:
            # Only take first 100 lines for test
            lines = f.readlines()[:100]
            csv_content = ''.join(lines).strip()

        code += f'            "{stock_code}": """{csv_content}""",\n\n'

    code += '''        }

        for code, csv_data in STOCK_DATA.items():
            if csv_data.strip():
                key = f"japan_stocks/{code}.csv"
                self.ObjectStore.Save(key, csv_data)
                self.Debug(f"Uploaded: {key}")

        self.Debug("Test upload complete - 3 stocks")

    def OnData(self, data):
        pass
'''

    with open(test_output, 'w') as f:
        f.write(code)

    print(f"✓ Generated test version: {test_output}")


if __name__ == "__main__":
    generate_uploader()
