"""
Export fundamental data (BPS, ROE, EPS) from J-Quants SQLite cache
to CSV files for QuantConnect ObjectStore upload.

Output: quantconnect/fundamental_data/{code}.csv
Format: disclosed_date,fiscal_quarter,bps,roe,eps

Point-in-Time (PIT): disclosed_date is the actual publication date,
ensuring no look-ahead bias when used in backtests.
"""

import sqlite3
import os
import csv

DB_PATH = "/Users/MBP/Desktop/Project_Asset_Shield/data/jquants_cache.db"
OUTPUT_DIR = "/Users/MBP/Desktop/Project_Asset_Shield/quantconnect/fundamental_data"

UNIVERSE = [
    "29140", "33820", "38610", "40630", "44520",
    "45020", "45030", "45680", "46610", "49010",
    "51080", "54010", "60980", "62730", "63010",
    "63260", "63670", "65010", "65940", "67020",
    "67520", "67580", "68570", "68610", "69020",
    "69540", "70110", "72030", "72670", "72690",
    "72700", "77510", "79740", "80010", "80310",
    "80350", "80580", "83060", "83160", "84110",
    "86040", "87660", "88020", "90200", "90220",
    "91010", "94320", "94330", "97350", "99840",
]


def export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total_rows = 0
    for code in UNIVERSE:
        cursor.execute(
            """
            SELECT disclosed_date, fiscal_quarter, bps, roe, eps
            FROM financial_statements
            WHERE code = ?
              AND disclosed_date IS NOT NULL
              AND bps IS NOT NULL
              AND roe IS NOT NULL
            ORDER BY disclosed_date
            """,
            (code,),
        )
        rows = cursor.fetchall()

        if not rows:
            print(f"WARN: No data for {code}")
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{code}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        total_rows += len(rows)
        print(f"{code}: {len(rows)} records ({rows[0][0]} ~ {rows[-1][0]})")

    conn.close()
    print(f"\nExported {len(UNIVERSE)} stocks, {total_rows} total records")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    export()
