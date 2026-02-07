"""
Upload price CSV data to QuantConnect ObjectStore via REST API.

Usage:
  python3 scripts/upload_to_qc_objectstore.py

Requires:
  - QC_USER_ID and QC_API_TOKEN environment variables
  - Or enter interactively when prompted

Get your credentials from: https://www.quantconnect.com/account
"""

import os
import sys
import hashlib
import time
import json
import glob

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests

API_BASE = "https://www.quantconnect.com/api/v2"
PRICE_DIR = "/Users/MBP/Desktop/Project_Asset_Shield/quantconnect/custom_data"
FUND_DIR = "/Users/MBP/Desktop/Project_Asset_Shield/quantconnect/fundamental_data"


def get_auth():
    user_id = os.environ.get("QC_USER_ID", "")
    api_token = os.environ.get("QC_API_TOKEN", "")

    if not user_id:
        user_id = input("QuantConnect User ID: ").strip()
    if not api_token:
        api_token = input("QuantConnect API Token: ").strip()

    return user_id, api_token


def make_auth_headers(user_id, api_token):
    timestamp = str(int(time.time()))
    hash_bytes = hashlib.sha256(f"{api_token}:{timestamp}".encode()).hexdigest()
    return {
        "Timestamp": timestamp,
        "Authorization": f"Basic {hash_bytes}",
    }, user_id


def upload_object(user_id, api_token, key, data):
    """Upload string data to ObjectStore."""
    timestamp = str(int(time.time()))
    hash_hex = hashlib.sha256(f"{api_token}:{timestamp}".encode()).hexdigest()

    url = f"{API_BASE}/object/set"
    headers = {
        "Timestamp": timestamp,
    }

    resp = requests.post(
        url,
        headers=headers,
        auth=(user_id, hash_hex),
        json={
            "key": key,
            "data": data,
        }
    )
    return resp


def test_auth(user_id, api_token):
    """Test authentication."""
    timestamp = str(int(time.time()))
    hash_hex = hashlib.sha256(f"{api_token}:{timestamp}".encode()).hexdigest()

    resp = requests.get(
        f"{API_BASE}/authenticate",
        headers={"Timestamp": timestamp},
        auth=(user_id, hash_hex),
    )
    return resp.status_code == 200 and resp.json().get("success", False)


def main():
    print("=" * 60)
    print("QuantConnect ObjectStore Uploader")
    print("=" * 60)

    user_id, api_token = get_auth()

    print("\nTesting authentication...")
    if not test_auth(user_id, api_token):
        print("Authentication failed. Check your credentials.")
        print("Get them from: https://www.quantconnect.com/account")
        return

    print("Authentication OK\n")

    # Upload price data
    price_files = sorted(glob.glob(os.path.join(PRICE_DIR, "*.csv")))
    print(f"Price CSV files: {len(price_files)}")

    uploaded = 0
    failed = 0

    for csv_path in price_files:
        code = os.path.basename(csv_path).replace(".csv", "")
        key = f"japan_stocks/{code}.csv"

        with open(csv_path) as f:
            data = f.read()

        resp = upload_object(user_id, api_token, key, data)

        if resp.status_code == 200 and resp.json().get("success", False):
            lines = len(data.strip().split("\n"))
            print(f"  OK  {key} ({lines} rows, {len(data):,} chars)")
            uploaded += 1
        else:
            print(f"  FAIL {key}: {resp.status_code} {resp.text[:200]}")
            failed += 1

        time.sleep(0.3)  # Rate limiting

    print(f"\nPrice data: {uploaded} uploaded, {failed} failed")

    # Upload fundamental data
    fund_files = sorted(glob.glob(os.path.join(FUND_DIR, "*.csv")))
    print(f"\nFundamental CSV files: {len(fund_files)}")

    fund_uploaded = 0
    fund_failed = 0

    for csv_path in fund_files:
        code = os.path.basename(csv_path).replace(".csv", "")
        key = f"japan_fundamentals/{code}.csv"

        with open(csv_path) as f:
            data = f.read()

        resp = upload_object(user_id, api_token, key, data)

        if resp.status_code == 200 and resp.json().get("success", False):
            lines = len(data.strip().split("\n"))
            print(f"  OK  {key} ({lines} rows)")
            fund_uploaded += 1
        else:
            print(f"  FAIL {key}: {resp.status_code} {resp.text[:200]}")
            fund_failed += 1

        time.sleep(0.3)

    print(f"\nFundamental data: {fund_uploaded} uploaded, {fund_failed} failed")

    print("\n" + "=" * 60)
    print(f"TOTAL: {uploaded + fund_uploaded} uploaded, {failed + fund_failed} failed")
    if failed == 0 and fund_failed == 0:
        print("All data uploaded. Run ASSET_SHIELD_PRODUCTION.py on QC.")
    print("=" * 60)


if __name__ == "__main__":
    main()
