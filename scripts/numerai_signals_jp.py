#!/usr/bin/env python3
"""
Numerai Signals - Japanese Stocks Pipeline (J-Quants Premium)
==============================================================
J-Quants Premium data → feature engineering → LightGBM → Numerai Signals submission.

Monetizes the J-Quants Premium subscription (¥16,500/mo) via NMR rewards.

Features (Premium-exclusive marked with ★):
  - Price momentum (5d, 20d, 60d)
  - Volatility (20d realized)
  - Value (earnings yield via EPS/Price)
  - Quality (ROE = NP/Equity)
  ★ Margin ratio (long/short balance change)
  ★ Short-sell ratio by sector

Usage:
    # Full pipeline: fetch data, train, predict, submit
    python numerai_signals_jp.py --mode full

    # Fetch J-Quants data only
    python numerai_signals_jp.py --mode fetch

    # Train model only (requires fetched data)
    python numerai_signals_jp.py --mode train

    # Generate signals and submit (weekly cron)
    python numerai_signals_jp.py --mode submit

Author: Asset Shield Project
Date: 2026-02-13
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import httpx
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "numerai_signals_jp"
MODEL_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

# J-Quants API
JQUANTS_BASE = "https://api.jquants.com/v2"

# Feature lookback (trading days)
PRICE_LOOKBACK_DAYS = 400  # ~1.5 years calendar days for 250 trading days
FORWARD_RETURN_DAYS = 20   # Target: 20-day forward return

# LightGBM
LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 31,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

FEATURE_COLS = [
    "ret_5d", "ret_20d", "ret_60d",
    "vol_20d", "vol_60d",
    "turnover_20d",
    "earnings_yield", "roe", "bps_yield",
    "margin_ratio", "margin_ratio_chg",
    "sector_short_ratio",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signals_jp")

# ---------------------------------------------------------------------------
# J-Quants API Client
# ---------------------------------------------------------------------------
class JQuantsClient:
    """Minimal J-Quants v2 client using x-api-key auth."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"x-api-key": api_key}
        self.client = httpx.Client(timeout=60.0)

    def _get_all(self, path: str, params: dict, data_key: str = "data") -> list:
        """Fetch all pages from a paginated endpoint with retry on 429."""
        all_data = []
        while True:
            for attempt in range(5):
                resp = self.client.get(
                    f"{JQUANTS_BASE}{path}",
                    params=params,
                    headers=self.headers,
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt + 1
                    log.warning(f"  Rate limited, waiting {wait}s...")
                    import time; time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            else:
                resp.raise_for_status()  # Final attempt, let it fail
            body = resp.json()
            data = body.get(data_key, [])
            if isinstance(data, list):
                all_data.extend(data)
            pk = body.get("pagination_key")
            if not pk:
                break
            params["pagination_key"] = pk
        return all_data

    def get_listed(self, date: str = None) -> pd.DataFrame:
        params = {}
        if date:
            params["date"] = date
        data = self._get_all("/equities/master", params)
        return pd.DataFrame(data)

    def get_prices(self, date: str = None, code: str = None,
                   from_: str = None, to_: str = None) -> pd.DataFrame:
        params = {}
        if date:
            params["date"] = date
        if code:
            params["code"] = code
        if from_:
            params["from"] = from_
        if to_:
            params["to"] = to_
        data = self._get_all("/equities/bars/daily", params)
        return pd.DataFrame(data)

    def get_fins(self, date: str = None, code: str = None) -> pd.DataFrame:
        params = {}
        if date:
            params["date"] = date
        if code:
            params["code"] = code
        data = self._get_all("/fins/summary", params)
        return pd.DataFrame(data)

    def get_margin(self, date: str = None, code: str = None,
                   from_: str = None, to_: str = None) -> pd.DataFrame:
        params = {}
        if date:
            params["date"] = date
        if code:
            params["code"] = code
        if from_:
            params["from"] = from_
        if to_:
            params["to"] = to_
        data = self._get_all("/markets/margin-interest", params)
        return pd.DataFrame(data)

    def get_short_ratio(self, date: str = None,
                        from_: str = None, to_: str = None) -> pd.DataFrame:
        params = {}
        if date:
            params["date"] = date
        if from_:
            params["from"] = from_
        if to_:
            params["to"] = to_
        data = self._get_all("/markets/short-ratio", params)
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------
def fetch_all_data(jq: JQuantsClient, lookback_days: int = PRICE_LOOKBACK_DAYS):
    """Fetch all required data from J-Quants and cache locally."""
    for d in [DATA_DIR, MODEL_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    today = datetime.now()
    from_date = (today - timedelta(days=lookback_days)).strftime("%Y%m%d")
    to_date = today.strftime("%Y%m%d")

    # 1. Listed stocks (current)
    log.info("Fetching listed stocks...")
    listed = jq.get_listed()
    # Filter: ordinary shares (5th digit = 0), exclude ETFs etc.
    listed = listed[listed["Code"].str[-1] == "0"].copy()
    listed.to_parquet(CACHE_DIR / "listed.parquet", index=False)
    log.info(f"  Listed stocks: {len(listed)}")

    # 2. Daily prices (all stocks, full period) - by date to respect rate limits
    log.info(f"Fetching daily prices ({from_date} to {to_date})...")
    prices_cache = CACHE_DIR / "prices.parquet"
    if prices_cache.exists():
        existing = pd.read_parquet(prices_cache)
        last_date = existing["Date"].max().replace("-", "")
        if last_date >= to_date:
            log.info("  Prices cache is up to date.")
            prices = existing
        else:
            log.info(f"  Updating from {last_date}...")
            new_from = (datetime.strptime(last_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
            new_prices = _fetch_prices_by_date(jq, new_from, to_date)
            prices = pd.concat([existing, new_prices], ignore_index=True).drop_duplicates(
                subset=["Date", "Code"], keep="last"
            )
            prices.to_parquet(prices_cache, index=False)
    else:
        prices = _fetch_prices_by_date(jq, from_date, to_date)
        prices.to_parquet(prices_cache, index=False)
    log.info(f"  Price records: {len(prices)}")

    # 3. Financial data (latest for each stock) - concurrent fetch
    log.info("Fetching financial data (concurrent)...")
    fins_cache = CACHE_DIR / "fins.parquet"
    codes = listed["Code"].tolist()
    if fins_cache.exists():
        fins = pd.read_parquet(fins_cache)
        log.info(f"  Using cached fins: {len(fins)} records")
    else:
        fins = _fetch_fins_concurrent(jq, codes)
        if len(fins) > 0:
            fins.to_parquet(fins_cache, index=False)
    log.info(f"  Financial records: {len(fins)}")

    # 4. Margin data (weekly, last 3 months)
    log.info("Fetching margin data...")
    margin_from = (today - timedelta(days=90)).strftime("%Y%m%d")
    margin_cache = CACHE_DIR / "margin.parquet"
    if margin_cache.exists():
        margin = pd.read_parquet(margin_cache)
        log.info(f"  Using cached margin: {len(margin)} records")
    else:
        margin = _fetch_margin_by_date(jq, margin_from, to_date)
        if len(margin) > 0:
            margin.to_parquet(margin_cache, index=False)
    log.info(f"  Margin records: {len(margin)}")

    # 5. Short-sell ratio by sector (must query by date, not from/to without s33)
    log.info("Fetching short-sell ratio...")
    short_cache = CACHE_DIR / "short_ratio.parquet"
    if short_cache.exists():
        short = pd.read_parquet(short_cache)
        log.info(f"  Using cached short ratio: {len(short)} records")
    else:
        short_chunks = []
        cur = datetime.strptime(margin_from, "%Y%m%d")
        end_dt = datetime.strptime(to_date, "%Y%m%d")
        while cur <= end_dt:
            ds = cur.strftime("%Y%m%d")
            try:
                df = jq.get_short_ratio(date=ds)
                if len(df) > 0:
                    short_chunks.append(df)
            except Exception:
                pass
            cur += timedelta(days=1)
        short = pd.concat(short_chunks, ignore_index=True) if short_chunks else pd.DataFrame()
        if len(short) > 0:
            short.to_parquet(short_cache, index=False)
    log.info(f"  Short ratio records: {len(short)}")

    log.info("Data fetch complete.")
    return listed, prices, fins, margin, short


def _fetch_fins_concurrent(jq: JQuantsClient, codes: list, max_workers: int = 8) -> pd.DataFrame:
    """Fetch financial data for all stocks concurrently."""
    results = []
    done_count = 0
    total = len(codes)

    def _fetch_one(code):
        try:
            # Each thread needs its own client to avoid connection issues
            client = httpx.Client(timeout=60.0)
            params = {"code": code}
            all_data = []
            while True:
                resp = client.get(
                    f"{JQUANTS_BASE}/fins/summary",
                    params=params,
                    headers={"x-api-key": jq.api_key},
                )
                resp.raise_for_status()
                body = resp.json()
                data = body.get("data", [])
                if isinstance(data, list):
                    all_data.extend(data)
                pk = body.get("pagination_key")
                if not pk:
                    break
                params["pagination_key"] = pk
            client.close()
            return all_data
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, code): code for code in codes}
        for future in as_completed(futures):
            done_count += 1
            data = future.result()
            if data:
                results.extend(data)
            if done_count % 500 == 0:
                log.info(f"  Fins progress: {done_count}/{total}")

    return pd.DataFrame(results) if results else pd.DataFrame()


def _fetch_prices_by_date(jq: JQuantsClient, from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch prices date-by-date with retry on 429."""
    import time as _time
    chunks = []
    current = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(to_date, "%Y%m%d")
    total_days = (end - current).days + 1
    fetched = 0
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        for attempt in range(5):
            try:
                df = jq.get_prices(date=date_str)
                if len(df) > 0:
                    chunks.append(df)
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** attempt + 1
                    log.warning(f"  429 on {date_str}, wait {wait}s...")
                    _time.sleep(wait)
                elif e.response.status_code == 400:
                    break  # Non-trading day
                else:
                    log.warning(f"  Error {date_str}: {e}")
                    break
        fetched += 1
        if fetched % 30 == 0:
            log.info(f"  Price progress: {fetched}/{total_days} days")
        current += timedelta(days=1)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _fetch_margin_by_date(jq: JQuantsClient, from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch margin data by weekly dates."""
    chunks = []
    current = datetime.strptime(from_date, "%Y%m%d")
    end = datetime.strptime(to_date, "%Y%m%d")
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        try:
            df = jq.get_margin(date=date_str)
            if len(df) > 0:
                chunks.append(df)
        except httpx.HTTPStatusError:
            pass
        current += timedelta(days=7)  # Weekly data
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def build_features(listed: pd.DataFrame, prices: pd.DataFrame,
                   fins: pd.DataFrame, margin: pd.DataFrame,
                   short: pd.DataFrame) -> pd.DataFrame:
    """Build cross-sectional features for all stocks on the latest date."""
    log.info("Building features...")

    if len(prices) == 0:
        log.error("No price data available.")
        return pd.DataFrame()

    # Ensure types
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["AdjC"] = pd.to_numeric(prices["AdjC"], errors="coerce")
    prices["AdjVo"] = pd.to_numeric(prices["AdjVo"], errors="coerce")
    prices["Va"] = pd.to_numeric(prices["Va"], errors="coerce")

    # Sort
    prices = prices.sort_values(["Code", "Date"])

    # Pivot to wide format for vectorized computation
    pivot_close = prices.pivot_table(index="Date", columns="Code", values="AdjC")
    pivot_vol = prices.pivot_table(index="Date", columns="Code", values="AdjVo")
    pivot_va = prices.pivot_table(index="Date", columns="Code", values="Va")

    # --- Price features ---
    ret_5d = pivot_close.pct_change(5).iloc[-1]
    ret_20d = pivot_close.pct_change(20).iloc[-1]
    ret_60d = pivot_close.pct_change(60).iloc[-1]

    # --- Volatility ---
    daily_ret = pivot_close.pct_change()
    vol_20d = daily_ret.rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_60d = daily_ret.rolling(60).std().iloc[-1] * np.sqrt(252)

    # --- Turnover ---
    turnover_20d = pivot_va.rolling(20).mean().iloc[-1]

    features = pd.DataFrame({
        "Code": pivot_close.columns,
        "ret_5d": ret_5d.values,
        "ret_20d": ret_20d.values,
        "ret_60d": ret_60d.values,
        "vol_20d": vol_20d.values,
        "vol_60d": vol_60d.values,
        "turnover_20d": turnover_20d.values,
    })

    # --- Financial features ---
    if len(fins) > 0:
        fins_latest = _get_latest_fins(fins)
        latest_price = pivot_close.iloc[-1].to_dict()

        fin_features = []
        for _, row in fins_latest.iterrows():
            code = row["Code"]
            price = latest_price.get(code, np.nan)
            eps = pd.to_numeric(row.get("EPS"), errors="coerce")
            bps = pd.to_numeric(row.get("BPS"), errors="coerce")
            np_val = pd.to_numeric(row.get("NP"), errors="coerce")
            eq = pd.to_numeric(row.get("Eq"), errors="coerce")

            ey = eps / price if price and price > 0 and pd.notna(eps) else np.nan
            roe = np_val / eq if eq and eq > 0 and pd.notna(np_val) else np.nan
            bps_y = bps / price if price and price > 0 and pd.notna(bps) else np.nan

            fin_features.append({
                "Code": code,
                "earnings_yield": ey,
                "roe": roe,
                "bps_yield": bps_y,
            })

        fin_df = pd.DataFrame(fin_features)
        features = features.merge(fin_df, on="Code", how="left")
    else:
        features["earnings_yield"] = np.nan
        features["roe"] = np.nan
        features["bps_yield"] = np.nan

    # --- Margin features (★ Premium) ---
    if len(margin) > 0:
        margin["Date"] = pd.to_datetime(margin["Date"])
        margin["LongVol"] = pd.to_numeric(margin["LongVol"], errors="coerce")
        margin["ShrtVol"] = pd.to_numeric(margin["ShrtVol"], errors="coerce")

        margin_latest = margin.sort_values("Date").groupby("Code").tail(2)
        margin_feat = []
        for code, grp in margin_latest.groupby("Code"):
            grp = grp.sort_values("Date")
            long_v = grp["LongVol"].iloc[-1]
            short_v = grp["ShrtVol"].iloc[-1]
            ratio = long_v / short_v if short_v and short_v > 0 else np.nan

            if len(grp) >= 2:
                prev_long = grp["LongVol"].iloc[-2]
                prev_short = grp["ShrtVol"].iloc[-2]
                prev_ratio = prev_long / prev_short if prev_short and prev_short > 0 else np.nan
                ratio_chg = ratio - prev_ratio if pd.notna(ratio) and pd.notna(prev_ratio) else np.nan
            else:
                ratio_chg = np.nan

            margin_feat.append({
                "Code": code,
                "margin_ratio": ratio,
                "margin_ratio_chg": ratio_chg,
            })

        margin_df = pd.DataFrame(margin_feat)
        features = features.merge(margin_df, on="Code", how="left")
    else:
        features["margin_ratio"] = np.nan
        features["margin_ratio_chg"] = np.nan

    # --- Short-sell ratio by sector (★ Premium) ---
    if len(short) > 0 and len(listed) > 0:
        short["Date"] = pd.to_datetime(short["Date"])
        short["SellExShortVa"] = pd.to_numeric(short["SellExShortVa"], errors="coerce")
        short["ShrtWithResVa"] = pd.to_numeric(short["ShrtWithResVa"], errors="coerce")
        short["ShrtNoResVa"] = pd.to_numeric(short["ShrtNoResVa"], errors="coerce")

        short_latest = short.sort_values("Date").groupby("S33").last().reset_index()
        short_latest["total_sell"] = (
            short_latest["SellExShortVa"] +
            short_latest["ShrtWithResVa"] +
            short_latest["ShrtNoResVa"]
        )
        short_latest["sector_short_ratio"] = (
            (short_latest["ShrtWithResVa"] + short_latest["ShrtNoResVa"]) /
            short_latest["total_sell"]
        ).replace([np.inf, -np.inf], np.nan)

        sector_map = listed[["Code", "S33"]].drop_duplicates()
        features = features.merge(sector_map, on="Code", how="left")
        features = features.merge(
            short_latest[["S33", "sector_short_ratio"]],
            on="S33", how="left"
        )
        features.drop(columns=["S33"], inplace=True, errors="ignore")
    else:
        features["sector_short_ratio"] = np.nan

    # --- Rank-normalize all features to [0, 1] ---
    for col in FEATURE_COLS:
        if col in features.columns:
            features[col] = features[col].rank(pct=True)

    log.info(f"  Features built for {len(features)} stocks, {len(FEATURE_COLS)} features")
    return features


def _get_latest_fins(fins: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent financial record per stock."""
    fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")
    return fins.sort_values("DiscDate").groupby("Code").last().reset_index()


# ---------------------------------------------------------------------------
# Training Data Construction
# ---------------------------------------------------------------------------
def build_training_data(jq: JQuantsClient, listed: pd.DataFrame,
                        prices: pd.DataFrame, fins: pd.DataFrame,
                        margin: pd.DataFrame, short: pd.DataFrame) -> pd.DataFrame:
    """Build historical cross-sections with forward returns as targets."""
    log.info("Building training data (rolling cross-sections)...")

    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["AdjC"] = pd.to_numeric(prices["AdjC"], errors="coerce")
    prices = prices.sort_values(["Code", "Date"])

    pivot_close = prices.pivot_table(index="Date", columns="Code", values="AdjC")
    pivot_va = prices.pivot_table(index="Date", columns="Code", values="Va")

    # Forward returns (target)
    fwd_ret = pivot_close.pct_change(FORWARD_RETURN_DAYS).shift(-FORWARD_RETURN_DAYS)

    # Sample dates: every 5 trading days for efficiency
    dates = pivot_close.index[60:-FORWARD_RETURN_DAYS:5]

    # --- Pre-compute financial lookup (point-in-time) ---
    # Sorted by DiscDate so we can use asof merge per snapshot date
    fins_pit = pd.DataFrame()
    if len(fins) > 0:
        fins = fins.copy()
        fins["DiscDate"] = pd.to_datetime(fins["DiscDate"], errors="coerce")
        for c in ["EPS", "BPS", "NP", "Eq"]:
            if c in fins.columns:
                fins[c] = pd.to_numeric(fins[c], errors="coerce")
        # Keep only the latest disclosure per (Code, DiscDate)
        fins_pit = (fins.dropna(subset=["DiscDate"])
                    .sort_values("DiscDate")
                    .drop_duplicates(subset=["Code", "DiscDate"], keep="last"))
        log.info(f"  Fins PIT records: {len(fins_pit)}")

    # --- Pre-compute margin lookup (point-in-time) ---
    margin_pit = pd.DataFrame()
    if len(margin) > 0:
        margin = margin.copy()
        margin["Date"] = pd.to_datetime(margin["Date"])
        margin["LongVol"] = pd.to_numeric(margin["LongVol"], errors="coerce")
        margin["ShrtVol"] = pd.to_numeric(margin["ShrtVol"], errors="coerce")
        margin["margin_ratio"] = (margin["LongVol"] / margin["ShrtVol"]).replace(
            [np.inf, -np.inf], np.nan)
        margin = margin.sort_values(["Code", "Date"])
        margin["margin_ratio_prev"] = margin.groupby("Code")["margin_ratio"].shift(1)
        margin["margin_ratio_chg"] = margin["margin_ratio"] - margin["margin_ratio_prev"]
        margin_pit = margin[["Code", "Date", "margin_ratio", "margin_ratio_chg"]].copy()
        margin_pit = margin_pit.sort_values("Date")
        log.info(f"  Margin PIT records: {len(margin_pit)}")

    # --- Pre-compute short-sell ratio lookup (point-in-time) ---
    short_pit = pd.DataFrame()
    if len(short) > 0:
        short = short.copy()
        short["Date"] = pd.to_datetime(short["Date"])
        short["SellExShortVa"] = pd.to_numeric(short["SellExShortVa"], errors="coerce")
        short["ShrtWithResVa"] = pd.to_numeric(short["ShrtWithResVa"], errors="coerce")
        short["ShrtNoResVa"] = pd.to_numeric(short["ShrtNoResVa"], errors="coerce")
        short["total_sell"] = (short["SellExShortVa"] + short["ShrtWithResVa"]
                               + short["ShrtNoResVa"])
        short["sector_short_ratio"] = (
            (short["ShrtWithResVa"] + short["ShrtNoResVa"]) / short["total_sell"]
        ).replace([np.inf, -np.inf], np.nan)
        short_pit = short[["S33", "Date", "sector_short_ratio"]].copy()
        short_pit = short_pit.sort_values("Date")
        log.info(f"  Short PIT records: {len(short_pit)}")

    # --- Sector map (Code -> S33) for short-sell ratio ---
    sector_map = {}
    if len(listed) > 0 and "S33" in listed.columns:
        sector_map = listed.set_index("Code")["S33"].to_dict()

    all_rows = []
    daily_ret = pivot_close.pct_change()

    for dt in dates:
        idx = pivot_close.index.get_loc(dt)
        if idx < 60:
            continue

        snapshot = {}
        snapshot["ret_5d"] = pivot_close.iloc[idx] / pivot_close.iloc[idx - 5] - 1
        snapshot["ret_20d"] = pivot_close.iloc[idx] / pivot_close.iloc[idx - 20] - 1
        snapshot["ret_60d"] = pivot_close.iloc[idx] / pivot_close.iloc[idx - 60] - 1
        snapshot["vol_20d"] = daily_ret.iloc[idx - 20:idx].std() * np.sqrt(252)
        snapshot["vol_60d"] = daily_ret.iloc[idx - 60:idx].std() * np.sqrt(252)
        snapshot["turnover_20d"] = pivot_va.iloc[idx - 20:idx].mean()

        df_snap = pd.DataFrame(snapshot)
        df_snap["target"] = fwd_ret.loc[dt]
        df_snap["Date"] = dt
        df_snap["Code"] = df_snap.index
        df_snap = df_snap.reset_index(drop=True)

        # --- Merge financial features (point-in-time) ---
        if len(fins_pit) > 0:
            # Latest fins record per stock BEFORE this date
            fins_before = fins_pit[fins_pit["DiscDate"] <= dt]
            if len(fins_before) > 0:
                fins_latest = fins_before.groupby("Code").last().reset_index()
                # Current price for each stock at this snapshot
                price_at_dt = pivot_close.iloc[idx]
                fins_latest = fins_latest[fins_latest["Code"].isin(price_at_dt.index)]
                fins_latest["_price"] = fins_latest["Code"].map(price_at_dt)
                fins_latest["earnings_yield"] = np.where(
                    (fins_latest["_price"] > 0) & fins_latest["EPS"].notna(),
                    fins_latest["EPS"] / fins_latest["_price"], np.nan)
                fins_latest["roe"] = np.where(
                    (fins_latest["Eq"] > 0) & fins_latest["NP"].notna(),
                    fins_latest["NP"] / fins_latest["Eq"], np.nan)
                fins_latest["bps_yield"] = np.where(
                    (fins_latest["_price"] > 0) & fins_latest["BPS"].notna(),
                    fins_latest["BPS"] / fins_latest["_price"], np.nan)
                df_snap = df_snap.merge(
                    fins_latest[["Code", "earnings_yield", "roe", "bps_yield"]],
                    on="Code", how="left")

        # --- Merge margin features (point-in-time) ---
        if len(margin_pit) > 0:
            margin_before = margin_pit[margin_pit["Date"] <= dt]
            if len(margin_before) > 0:
                margin_latest = margin_before.groupby("Code").last().reset_index()
                df_snap = df_snap.merge(
                    margin_latest[["Code", "margin_ratio", "margin_ratio_chg"]],
                    on="Code", how="left")

        # --- Merge sector short-sell ratio (point-in-time) ---
        if len(short_pit) > 0 and sector_map:
            short_before = short_pit[short_pit["Date"] <= dt]
            if len(short_before) > 0:
                short_latest = short_before.groupby("S33").last().reset_index()
                # Map Code -> S33 -> sector_short_ratio
                df_snap["_s33"] = df_snap["Code"].map(sector_map)
                df_snap = df_snap.merge(
                    short_latest[["S33", "sector_short_ratio"]],
                    left_on="_s33", right_on="S33", how="left")
                df_snap.drop(columns=["_s33", "S33"], inplace=True, errors="ignore")

        # Rank-normalize features
        for col in FEATURE_COLS:
            if col in df_snap.columns:
                df_snap[col] = df_snap[col].rank(pct=True)

        # Rank-normalize target
        df_snap["target"] = df_snap["target"].rank(pct=True)

        all_rows.append(df_snap)

    if not all_rows:
        return pd.DataFrame()

    train_df = pd.concat(all_rows, ignore_index=True)
    train_df = train_df.dropna(subset=["target"])

    # Fill missing feature columns with neutral rank
    for col in FEATURE_COLS:
        if col not in train_df.columns:
            train_df[col] = 0.5

    log.info(f"  Training samples: {len(train_df)} ({len(dates)} dates)")
    return train_df


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def train_model(train_df: pd.DataFrame):
    """Train LightGBM model on historical cross-sections."""
    import lightgbm as lgb
    import cloudpickle

    log.info("Training LightGBM model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    X = train_df[available_features].fillna(0.5)
    y = train_df["target"]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    log.info(f"  Feature importance:\n{imp.to_string()}")

    # Save
    model_path = MODEL_DIR / "signals_jp_lgbm.pkl"
    with open(model_path, "wb") as f:
        cloudpickle.dump({"model": model, "features": available_features}, f)
    log.info(f"  Model saved: {model_path}")

    return model, available_features


# ---------------------------------------------------------------------------
# Signal Generation & Submission
# ---------------------------------------------------------------------------
def generate_and_submit(features: pd.DataFrame, model=None, model_features=None):
    """Generate signals and submit to Numerai Signals."""
    import cloudpickle
    from numerapi import SignalsAPI

    # Load model if not provided
    if model is None:
        model_path = MODEL_DIR / "signals_jp_lgbm.pkl"
        if not model_path.exists():
            log.error(f"Model not found: {model_path}. Run --mode train first.")
            return
        with open(model_path, "rb") as f:
            obj = cloudpickle.load(f)
            model = obj["model"]
            model_features = obj["features"]

    # Predict
    X = features[model_features].fillna(0.5)
    features["raw_signal"] = model.predict(X)

    # Normalize to (0, 1) exclusive using rank
    features["signal"] = features["raw_signal"].rank(pct=True)
    features["signal"] = features["signal"].clip(0.01, 0.99)

    # Map J-Quants code to bloomberg_ticker: "72030" -> "7203 JP"
    features["bloomberg_ticker"] = features["Code"].str[:4] + " JP"

    # Prepare submission
    submission = features[["bloomberg_ticker", "signal"]].dropna()
    submission = submission.drop_duplicates(subset=["bloomberg_ticker"], keep="first")

    log.info(f"  Submission: {len(submission)} stocks")
    log.info(f"  Signal stats: mean={submission['signal'].mean():.3f}, "
             f"std={submission['signal'].std():.3f}")

    # Save locally
    sub_path = DATA_DIR / f"submission_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    submission.to_csv(sub_path, index=False)
    log.info(f"  Saved: {sub_path}")

    # Submit to Numerai
    public_id = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    model_id = os.getenv("NUMERAI_SIGNALS_MODEL_ID", "")

    if not public_id or not secret_key:
        log.warning("NUMERAI_PUBLIC_ID / NUMERAI_SECRET_KEY not set. Skipping submission.")
        log.info(f"  To submit manually: napi.upload_predictions('{sub_path}')")
        return sub_path

    sapi = SignalsAPI(public_id, secret_key)

    if model_id:
        sapi.upload_predictions(str(sub_path), model_id=model_id)
    else:
        sapi.upload_predictions(str(sub_path))

    log.info("Submission uploaded to Numerai Signals!")
    return sub_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Numerai Signals JP (J-Quants Premium)")
    parser.add_argument("--mode", choices=["full", "fetch", "train", "submit"],
                        default="full", help="Pipeline mode")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("JQUANTS_API_KEY") or os.getenv("JQUANTS_X_API_KEY")
    if not api_key:
        log.error("JQUANTS_API_KEY not set in .env")
        sys.exit(1)

    jq = JQuantsClient(api_key)

    if args.mode in ("full", "fetch"):
        listed, prices, fins, margin, short = fetch_all_data(jq)
    else:
        # Load from cache
        listed = pd.read_parquet(CACHE_DIR / "listed.parquet")
        prices = pd.read_parquet(CACHE_DIR / "prices.parquet")
        fins_path = CACHE_DIR / "fins.parquet"
        fins = pd.read_parquet(fins_path) if fins_path.exists() else pd.DataFrame()
        margin_path = CACHE_DIR / "margin.parquet"
        margin = pd.read_parquet(margin_path) if margin_path.exists() else pd.DataFrame()
        short_path = CACHE_DIR / "short_ratio.parquet"
        short = pd.read_parquet(short_path) if short_path.exists() else pd.DataFrame()

    if args.mode in ("full", "train"):
        train_df = build_training_data(jq, listed, prices, fins, margin, short)
        if len(train_df) == 0:
            log.error("No training data. Check data fetch.")
            sys.exit(1)
        model, model_features = train_model(train_df)
    else:
        model, model_features = None, None

    if args.mode in ("full", "submit"):
        features = build_features(listed, prices, fins, margin, short)
        if len(features) == 0:
            log.error("No features built.")
            sys.exit(1)
        generate_and_submit(features, model, model_features)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
