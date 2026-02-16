#!/usr/bin/env python3
"""
X Bot JP - 機関投資家 空売り動向トラッカー
==========================================
J-Quants mkt-short-sale → X 自動投稿 → note.com 誘導

Features:
  1. Daily: 機関投資家の新規/増加/減少ポジション速報
  2. Daily: 機関空売り集中銘柄
  3. Weekly: 機関別空売り活動ランキング
  4. Monthly: 月次サマリー (note誘導)
  5. Charts: note記事用チャート生成

Usage:
  python x_bot_jp.py --mode daily      # 日次速報
  python x_bot_jp.py --mode weekly     # 週間ランキング
  python x_bot_jp.py --mode monthly    # 月次レポート
  python x_bot_jp.py --mode charts     # チャート生成のみ
  python x_bot_jp.py --mode daily --dry-run  # 投稿せずプレビュー

Author: Asset Shield Project (クオンツ軍師)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "x_bot_jp"
DB_PATH = DATA_DIR / "tracker.db"
CHART_DIR = DATA_DIR / "charts"
STOCK_MAP_PATH = DATA_DIR / "stock_map.json"
NOTE_URL = "https://note.com/quant_gunshi"

JQUANTS_BASE = "https://api.jquants.com/v2"
MAX_TWEET_LEN = 280
MAX_RETRIES = 5
RETRY_BASE_WAIT = 1.0

# Short seller name abbreviations (common hedge funds)
SS_ABBREV = {
    # English names (from API)
    "Goldman Sachs": "GS",
    "Morgan Stanley": "モルガンS",
    "JPMorgan": "JPモルガン",
    "Merrill Lynch": "メリルリンチ",
    "Barclays": "バークレイズ",
    "Citadel": "シタデル",
    "BlackRock": "ブラックロック",
    "Nomura": "野村",
    "Daiwa": "大和",
    "UBS": "UBS",
    "Credit Suisse": "CS",
    "Deutsche Bank": "ドイツ銀",
    "BNP Paribas": "BNP",
    "Societe Generale": "ソシエテ",
    "CLSA": "CLSA",
    "Millennium": "ミレニアム",
    "Two Sigma": "TwoSigma",
    "D.E. Shaw": "DEショウ",
    "AQR": "AQR",
    "Point72": "Point72",
    "Bridgewater": "ブリッジウォーター",
    "Citigroup": "シティ",
    "BofA": "BofA",
    "Susquehanna": "SIG",
    "Jane Street": "JaneSt",
    "Jump Trading": "Jump",
    # Japanese names (from API)
    "大和証券": "大和",
    "野村證券": "野村",
    "みずほ証券": "みずほ",
    "モルガン・スタンレーMUFG": "モルガンS",
    "三菱UFJモルガン": "三菱モルガン",
    "SMBC日興": "SMBC日興",
    "ベル投資": "ベル投資",
    "個人": "個人",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("x_bot_jp")

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
for candidate in [PROJECT_ROOT / ".env", Path.home() / ".env"]:
    if candidate.exists():
        load_dotenv(candidate)
        break


# ===================================================================
# J-Quants API Client
# ===================================================================
class JQuantsClient:
    """Minimal J-Quants v2 client with pagination and 429-retry."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("JQUANTS_API_KEY") or os.environ.get("JQUANTS_X_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("JQUANTS_API_KEY not configured")
        self._session = requests.Session()
        self._session.headers.update({"x-api-key": self.api_key, "Accept": "application/json"})

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{JQUANTS_BASE}{path}"
        wait = RETRY_BASE_WAIT
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, params=params, timeout=60)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", wait))
                    log.warning("Rate limited (429). Retry in %.1fs (%d/%d)", retry_after, attempt, MAX_RETRIES)
                    time.sleep(retry_after)
                    wait = min(wait * 2, 60)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError:
                raise
            except requests.exceptions.RequestException as exc:
                log.warning("Request error: %s (%d/%d)", exc, attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(wait)
                wait = min(wait * 2, 60)
        raise RuntimeError(f"Max retries exceeded for {path}")

    def _get_all(self, path: str, params: Optional[Dict] = None, data_key: str = "data") -> List[Dict]:
        params = dict(params or {})
        all_data: List[Dict] = []
        while True:
            body = self._get(path, params)
            chunk = body.get(data_key, [])
            if isinstance(chunk, list):
                all_data.extend(chunk)
            pag = body.get("pagination_key")
            if not pag:
                break
            params["pagination_key"] = pag
        return all_data

    def get_listed(self, date: Optional[str] = None) -> List[Dict]:
        params: Dict[str, str] = {}
        if date:
            params["date"] = date
        return self._get_all("/equities/master", params)

    def get_short_sale(
        self,
        disc_date: Optional[str] = None,
        disc_date_from: Optional[str] = None,
        disc_date_to: Optional[str] = None,
        code: Optional[str] = None,
    ) -> List[Dict]:
        """GET /markets/short-sale-report — institutional short positions."""
        params: Dict[str, str] = {}
        if disc_date:
            params["disc_date"] = disc_date
        if disc_date_from:
            params["disc_date_from"] = disc_date_from
        if disc_date_to:
            params["disc_date_to"] = disc_date_to
        if code:
            params["code"] = code
        return self._get_all("/markets/short-sale-report", params)

    def get_short_ratio(self, date: Optional[str] = None) -> List[Dict]:
        """GET /markets/short-ratio — sector short-sell ratio."""
        params: Dict[str, str] = {}
        if date:
            params["date"] = date
        return self._get_all("/markets/short-ratio", params)

    def get_margin_interest(self, date: Optional[str] = None,
                            code: Optional[str] = None) -> List[Dict]:
        """GET /markets/margin-interest — weekly margin trading balances."""
        params: Dict[str, str] = {}
        if date:
            params["date"] = date
        if code:
            params["code"] = code
        return self._get_all("/markets/margin-interest", params)

    def get_prices(self, date: Optional[str] = None, code: Optional[str] = None,
                   from_: Optional[str] = None, to_: Optional[str] = None) -> List[Dict]:
        """GET /equities/bars/daily — daily OHLCV."""
        params: Dict[str, str] = {}
        if date:
            params["date"] = date
        if code:
            params["code"] = code
        if from_:
            params["from"] = from_
        if to_:
            params["to"] = to_
        return self._get_all("/equities/bars/daily", params)


# ===================================================================
# SQLite Database
# ===================================================================
class TrackerDB:
    """SQLite for institutional short position tracking."""

    DDL = """
    CREATE TABLE IF NOT EXISTS disclosures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        disc_date TEXT NOT NULL,
        calc_date TEXT,
        code TEXT NOT NULL,
        company_name TEXT,
        ss_name TEXT NOT NULL,
        fund_name TEXT,
        ratio REAL NOT NULL,
        shares INTEGER,
        prev_ratio REAL,
        prev_date TEXT,
        change_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        post_type TEXT NOT NULL,
        tweet_text TEXT,
        tweet_id TEXT,
        payload TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_disc_date ON disclosures(disc_date);
    CREATE INDEX IF NOT EXISTS idx_disc_code ON disclosures(code);
    CREATE INDEX IF NOT EXISTS idx_disc_ss ON disclosures(ss_name);
    CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(date);
    CREATE INDEX IF NOT EXISTS idx_posts_type ON posts(post_type);

    CREATE TABLE IF NOT EXISTS verification (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_date TEXT NOT NULL,
        code TEXT NOT NULL,
        company_name TEXT,
        num_institutions INTEGER,
        signal_ratio REAL,
        price_at_signal REAL,
        price_5d REAL,
        price_20d REAL,
        ret_5d REAL,
        ret_20d REAL,
        verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_verif_date ON verification(signal_date);
    """

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(self.DDL)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def insert_disclosure(self, rec: Dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO disclosures (disc_date, calc_date, code, company_name, ss_name, "
            "fund_name, ratio, shares, prev_ratio, prev_date, change_type) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec["disc_date"], rec.get("calc_date"), rec["code"],
                rec.get("company_name"), rec["ss_name"], rec.get("fund_name"),
                rec["ratio"], rec.get("shares"), rec.get("prev_ratio"),
                rec.get("prev_date"), rec.get("change_type"),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def has_disclosures(self, disc_date: str) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM disclosures WHERE disc_date=?", (disc_date,)
        ).fetchone()
        return row[0] > 0

    def insert_post(self, date: str, post_type: str, tweet_text: str,
                    tweet_id: Optional[str], payload: Optional[Any] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO posts (date, post_type, tweet_text, tweet_id, payload) VALUES (?,?,?,?,?)",
            (date, post_type, tweet_text, tweet_id,
             json.dumps(payload, ensure_ascii=False) if payload else None),
        )
        self.conn.commit()
        return cur.lastrowid

    def has_post(self, date: str, post_type: str) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE date=? AND post_type=?", (date, post_type)
        ).fetchone()
        return row[0] > 0

    def get_disclosures_range(self, from_date: str, to_date: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM disclosures WHERE disc_date BETWEEN ? AND ? ORDER BY disc_date",
            (from_date, to_date),
        ).fetchall()

    def insert_verification(self, rec: Dict):
        self.conn.execute(
            "INSERT INTO verification (signal_date, code, company_name, num_institutions, "
            "signal_ratio, price_at_signal, price_5d, price_20d, ret_5d, ret_20d) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rec["signal_date"], rec["code"], rec.get("company_name"),
             rec.get("num_institutions"), rec.get("signal_ratio"),
             rec.get("price_at_signal"), rec.get("price_5d"), rec.get("price_20d"),
             rec.get("ret_5d"), rec.get("ret_20d")),
        )
        self.conn.commit()

    def has_verification(self, signal_date: str, code: str) -> bool:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM verification WHERE signal_date=? AND code=?",
            (signal_date, code),
        ).fetchone()
        return row[0] > 0

    def get_verification_stats(self, from_date: str, to_date: str) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT * FROM verification WHERE signal_date BETWEEN ? AND ?",
            (from_date, to_date),
        ).fetchall()
        if not rows:
            return {"count": 0}
        rets_5d = [r["ret_5d"] for r in rows if r["ret_5d"] is not None]
        rets_20d = [r["ret_20d"] for r in rows if r["ret_20d"] is not None]
        hit_5d = sum(1 for r in rets_5d if r < 0) / len(rets_5d) * 100 if rets_5d else 0
        hit_20d = sum(1 for r in rets_20d if r < 0) / len(rets_20d) * 100 if rets_20d else 0
        avg_5d = sum(rets_5d) / len(rets_5d) * 100 if rets_5d else 0
        avg_20d = sum(rets_20d) / len(rets_20d) * 100 if rets_20d else 0
        return {
            "count": len(rows),
            "hit_rate_5d": round(hit_5d, 1),
            "hit_rate_20d": round(hit_20d, 1),
            "avg_ret_5d": round(avg_5d, 2),
            "avg_ret_20d": round(avg_20d, 2),
        }

    def get_concentrated_dates(self, from_date: str, to_date: str) -> List[Dict]:
        """Get past concentrated short signals for verification."""
        rows = self.conn.execute(
            "SELECT disc_date, code, company_name, COUNT(DISTINCT ss_name) as n_inst, "
            "SUM(ratio) as total_ratio FROM disclosures "
            "WHERE disc_date BETWEEN ? AND ? "
            "GROUP BY disc_date, code HAVING n_inst >= 3 "
            "ORDER BY disc_date",
            (from_date, to_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_monthly_stats(self, year: int, month: int) -> Dict[str, Any]:
        prefix = f"{year}-{month:02d}"
        total_disclosures = self.conn.execute(
            "SELECT COUNT(*) FROM disclosures WHERE disc_date LIKE ?", (f"{prefix}%",)
        ).fetchone()[0]
        new_positions = self.conn.execute(
            "SELECT COUNT(*) FROM disclosures WHERE disc_date LIKE ? AND change_type='new'",
            (f"{prefix}%",)
        ).fetchone()[0]
        increases = self.conn.execute(
            "SELECT COUNT(*) FROM disclosures WHERE disc_date LIKE ? AND change_type='increase'",
            (f"{prefix}%",)
        ).fetchone()[0]
        decreases = self.conn.execute(
            "SELECT COUNT(*) FROM disclosures WHERE disc_date LIKE ? AND change_type='decrease'",
            (f"{prefix}%",)
        ).fetchone()[0]
        top_ss = self.conn.execute(
            "SELECT ss_name, COUNT(*) as cnt FROM disclosures WHERE disc_date LIKE ? "
            "GROUP BY ss_name ORDER BY cnt DESC LIMIT 5", (f"{prefix}%",)
        ).fetchall()
        total_posts = self.conn.execute(
            "SELECT COUNT(*) FROM posts WHERE date LIKE ?", (f"{prefix}%",)
        ).fetchone()[0]
        return {
            "year": year, "month": month,
            "total_disclosures": total_disclosures,
            "new_positions": new_positions,
            "increases": increases,
            "decreases": decreases,
            "top_institutions": [(r["ss_name"], r["cnt"]) for r in top_ss],
            "total_posts": total_posts,
        }


# ===================================================================
# Stock Name Map (code → company name)
# ===================================================================
def get_stock_name_map(client: JQuantsClient) -> Dict[str, str]:
    """Build stock code → company short name mapping. Cached to disk."""
    if STOCK_MAP_PATH.exists():
        age_hours = (time.time() - STOCK_MAP_PATH.stat().st_mtime) / 3600
        if age_hours < 24 * 7:
            with open(STOCK_MAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)

    log.info("Refreshing stock name map from eq-master...")
    listed = client.get_listed()
    mapping: Dict[str, str] = {}
    for rec in listed:
        code = rec.get("Code", "")
        name = rec.get("CoName") or rec.get("CompanyName", "")
        if code and name:
            mapping[code] = name

    STOCK_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STOCK_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    log.info("Stock map saved (%d stocks)", len(mapping))
    return mapping


# ===================================================================
# Short Seller Name Abbreviation
# ===================================================================
def shorten_ss_name(name: str, max_len: int = 10) -> str:
    """Abbreviate institutional short seller name for tweet display."""
    if not name:
        return "不明"
    name = _normalize_text(name)
    for key, abbr in SS_ABBREV.items():
        if key.lower() in name.lower():
            return abbr
    # Fallback: truncate
    short = name.split(",")[0].split("(")[0].strip()
    if len(short) > max_len:
        short = short[:max_len]
    return short


def _normalize_text(text: str) -> str:
    """Normalize full-width alphanumeric to half-width and clean whitespace."""
    # Full-width alphanumeric → half-width
    result = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:  # ！-～ → !-~
            result.append(chr(cp - 0xFEE0))
        elif ch == '\u3000':  # Ideographic space → regular space
            result.append(' ')
        else:
            result.append(ch)
    return "".join(result)


def shorten_company(name: str, max_len: int = 6) -> str:
    """Shorten company name for tweet display."""
    if not name:
        return "?"
    name = _normalize_text(name)
    # Remove common suffixes
    for suffix in ["株式会社", "(株)", "ホールディングス", "HD", "グループ"]:
        name = name.replace(suffix, "")
    name = name.strip()
    if len(name) > max_len:
        name = name[:max_len]
    return name


# ===================================================================
# Data Analysis
# ===================================================================
def fetch_and_categorize(client: JQuantsClient, disc_date: str,
                         stock_map: Dict[str, str]) -> List[Dict]:
    """Fetch disclosures for a date and categorize changes."""
    raw = client.get_short_sale(disc_date=disc_date)
    if not raw:
        return []

    results = []
    for rec in raw:
        ratio = float(rec.get("ShrtPosToSO", 0) or 0)
        prev_ratio = float(rec.get("PrevRptRatio", 0) or 0)
        prev_date = rec.get("PrevRptDate", "")
        code = rec.get("Code", "")
        ss_name = rec.get("SSName", "")
        fund_name = rec.get("FundName", "")

        # Determine change type
        if not prev_date or prev_ratio == 0:
            change_type = "new"
        elif ratio > prev_ratio:
            change_type = "increase"
        elif ratio < prev_ratio:
            change_type = "decrease"
        else:
            change_type = "unchanged"

        # Resolve company name from stock map
        code_4 = code[:4] if len(code) >= 4 else code
        company_name = stock_map.get(code, stock_map.get(code_4 + "0", code_4))

        results.append({
            "disc_date": disc_date,
            "calc_date": rec.get("CalcDate", ""),
            "code": code,
            "code_4": code_4,
            "company_name": company_name,
            "ss_name": ss_name,
            "ss_short": shorten_ss_name(ss_name),
            "fund_name": fund_name,
            "ratio": ratio,
            "shares": int(rec.get("ShrtPosShares", 0) or 0),
            "prev_ratio": prev_ratio,
            "prev_date": prev_date,
            "change_type": change_type,
        })

    return results


def find_concentrated_shorts(disclosures: List[Dict]) -> List[Dict]:
    """Find stocks with multiple institutions shorting them."""
    from collections import defaultdict
    by_stock: Dict[str, List[Dict]] = defaultdict(list)
    for d in disclosures:
        by_stock[d["code"]].append(d)

    concentrated = []
    for code, items in by_stock.items():
        if len(items) >= 2:
            total_ratio = sum(i["ratio"] for i in items)
            concentrated.append({
                "code": code,
                "code_4": items[0]["code_4"],
                "company_name": items[0]["company_name"],
                "count": len(items),
                "total_ratio": round(total_ratio, 2),
                "institutions": [i["ss_short"] for i in items],
            })
    concentrated.sort(key=lambda x: x["total_ratio"], reverse=True)
    return concentrated


# ===================================================================
# Verification & Analytics
# ===================================================================
def verify_past_signals(client: JQuantsClient, db: TrackerDB, lookback_days: int = 30):
    """Verify past concentrated short signals by checking subsequent price moves."""
    today = datetime.now()
    verify_from = (today - timedelta(days=lookback_days + 25)).strftime("%Y-%m-%d")
    verify_to = (today - timedelta(days=5)).strftime("%Y-%m-%d")

    signals = db.get_concentrated_dates(verify_from, verify_to)
    verified = 0
    for sig in signals:
        code = sig["code"]
        sig_date = sig["disc_date"]
        if db.has_verification(sig_date, code):
            continue
        try:
            prices = client.get_prices(
                code=code,
                from_=sig_date.replace("-", ""),
                to_=(today).strftime("%Y%m%d"),
            )
            if not prices:
                continue
            df = pd.DataFrame(prices)
            df["Date"] = pd.to_datetime(df["Date"])
            df["AdjC"] = pd.to_numeric(df["AdjC"], errors="coerce")
            df = df.sort_values("Date").dropna(subset=["AdjC"])
            if len(df) < 2:
                continue
            p0 = df["AdjC"].iloc[0]
            p5 = df["AdjC"].iloc[min(5, len(df) - 1)]
            p20 = df["AdjC"].iloc[min(20, len(df) - 1)] if len(df) > 5 else None
            ret_5d = (p5 / p0 - 1) if p0 > 0 else None
            ret_20d = (p20 / p0 - 1) if p20 and p0 > 0 else None
            db.insert_verification({
                "signal_date": sig_date, "code": code,
                "company_name": sig.get("company_name"),
                "num_institutions": sig.get("n_inst"),
                "signal_ratio": sig.get("total_ratio"),
                "price_at_signal": p0, "price_5d": p5, "price_20d": p20,
                "ret_5d": ret_5d, "ret_20d": ret_20d,
            })
            verified += 1
        except Exception as e:
            log.warning("Verify failed for %s %s: %s", code, sig_date, e)
    if verified:
        log.info("Verified %d signals", verified)
    return verified


def detect_anomalies(db: TrackerDB, disc_date: str, disclosures: List[Dict]) -> List[Dict]:
    """Detect stocks with unusual short position changes."""
    from collections import defaultdict
    # Get 30-day history for comparison
    dt = datetime.strptime(disc_date, "%Y-%m-%d")
    hist_from = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
    hist_rows = db.get_disclosures_range(hist_from, disc_date)

    # Build historical avg ratio per stock
    hist_ratios: Dict[str, List[float]] = defaultdict(list)
    for r in hist_rows:
        if r["disc_date"] != disc_date:
            hist_ratios[r["code"]].append(r["ratio"])

    anomalies = []
    for d in disclosures:
        code = d["code"]
        hist = hist_ratios.get(code, [])
        if len(hist) < 3:
            continue
        avg = sum(hist) / len(hist)
        std = (sum((x - avg) ** 2 for x in hist) / len(hist)) ** 0.5
        if std > 0 and abs(d["ratio"] - avg) > 2 * std:
            anomalies.append({
                **d,
                "avg_ratio": round(avg, 3),
                "std": round(std, 3),
                "z_score": round((d["ratio"] - avg) / std, 1),
            })
    anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    return anomalies[:5]


def calculate_sentiment(disclosures: List[Dict], db: TrackerDB,
                        disc_date: str) -> Dict[str, Any]:
    """Calculate overall market short-selling sentiment."""
    new_count = sum(1 for d in disclosures if d["change_type"] == "new")
    inc_count = sum(1 for d in disclosures if d["change_type"] == "increase")
    dec_count = sum(1 for d in disclosures if d["change_type"] == "decrease")
    total = len(disclosures)

    aggression = (new_count + inc_count) / total * 100 if total > 0 else 50

    # Compare to 7-day average
    dt = datetime.strptime(disc_date, "%Y-%m-%d")
    hist_from = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
    hist_rows = db.get_disclosures_range(hist_from, disc_date)
    hist_other = [r for r in hist_rows if r["disc_date"] != disc_date]
    if hist_other:
        hist_agg = sum(1 for r in hist_other if r["change_type"] in ("new", "increase"))
        hist_agg_pct = hist_agg / len(hist_other) * 100
        trend = aggression - hist_agg_pct
    else:
        hist_agg_pct = 50
        trend = 0

    if aggression > 55:
        label = "弱気"
        emoji = "\U0001f43b"
    elif aggression < 40:
        label = "強気"
        emoji = "\U0001f402"
    else:
        label = "中立"
        emoji = "\u2696\ufe0f"

    return {
        "aggression": round(aggression, 1),
        "trend": round(trend, 1),
        "label": label,
        "emoji": emoji,
        "new": new_count,
        "increase": inc_count,
        "decrease": dec_count,
        "total": total,
    }


# ===================================================================
# Tweet Formatting
# ===================================================================
def format_daily_changes(disclosures: List[Dict], date: str) -> str:
    """Format institutional short position changes tweet."""
    m, d = date[5:7].lstrip("0"), date[8:10].lstrip("0")

    new = [r for r in disclosures if r["change_type"] == "new"]
    inc = [r for r in disclosures if r["change_type"] == "increase"]
    dec = [r for r in disclosures if r["change_type"] == "decrease"]

    lines = [f"\U0001f3e6機関空売り速報({m}/{d})"]

    # Prioritize: new positions first, then largest increases
    shown = 0
    max_items = 5

    if new:
        lines.append("【新規】")
        for r in sorted(new, key=lambda x: x["ratio"], reverse=True)[:3]:
            lines.append(f"{r['ss_short']}→{shorten_company(r['company_name'])}({r['code_4']}) {r['ratio']:.2f}%")
            shown += 1
            if shown >= max_items:
                break

    if shown < max_items and inc:
        lines.append("【増加】")
        for r in sorted(inc, key=lambda x: x["ratio"] - x["prev_ratio"], reverse=True)[:2]:
            lines.append(f"{r['ss_short']}→{shorten_company(r['company_name'])}({r['code_4']}) {r['prev_ratio']:.2f}→{r['ratio']:.2f}%")
            shown += 1
            if shown >= max_items:
                break

    if shown < max_items and dec:
        lines.append("【減少】")
        for r in sorted(dec, key=lambda x: x["prev_ratio"] - x["ratio"], reverse=True)[:2]:
            lines.append(f"{r['ss_short']}→{shorten_company(r['company_name'])}({r['code_4']}) {r['prev_ratio']:.2f}→{r['ratio']:.2f}%")
            shown += 1
            if shown >= max_items:
                break

    total = len(disclosures)
    lines.append(f"計{total}件の開示")
    lines.append("#日本株 #機関空売り")

    text = "\n".join(lines)
    if len(text) > MAX_TWEET_LEN:
        text = text[:MAX_TWEET_LEN - 1] + "\u2026"
    return text


def format_concentrated_tweet(concentrated: List[Dict], date: str) -> str:
    """Format concentrated shorts tweet (stocks shorted by multiple institutions)."""
    m, d = date[5:7].lstrip("0"), date[8:10].lstrip("0")
    header = f"\U0001f534機関空売り集中銘柄({m}/{d})"
    footer = f"詳細→ {NOTE_URL}\n#日本株 #機関空売り"
    footer_len = len(footer) + 1  # +1 for newline before footer

    body_lines = []
    for c in concentrated[:3]:
        inst_str = "・".join(shorten_ss_name(i) for i in c["institutions"][:3])
        if len(c["institutions"]) > 3:
            inst_str += f"他{len(c['institutions']) - 3}社"
        entry = f"{shorten_company(c['company_name'])}({c['code_4']}) {c['count']}社 計{c['total_ratio']:.2f}%\n  {inst_str}"
        # Check if adding this entry would exceed limit
        candidate = header + "\n" + "\n".join(body_lines + [entry]) + "\n" + footer
        if len(candidate) > MAX_TWEET_LEN:
            break
        body_lines.append(entry)

    text = header + "\n" + "\n".join(body_lines) + "\n" + footer
    return text


def format_weekly_tweet(db: TrackerDB, end_date: str) -> str:
    """Format weekly institution ranking tweet."""
    dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = (dt - timedelta(days=6)).strftime("%Y-%m-%d")
    rows = db.get_disclosures_range(start_date, end_date)

    m1, d1 = start_date[5:7].lstrip("0"), start_date[8:10].lstrip("0")
    m2, d2 = end_date[5:7].lstrip("0"), end_date[8:10].lstrip("0")

    # Count by institution
    from collections import Counter
    ss_counts = Counter(r["ss_name"] for r in rows)
    top5 = ss_counts.most_common(5)

    # Count new vs decrease
    new_count = sum(1 for r in rows if r["change_type"] == "new")
    inc_count = sum(1 for r in rows if r["change_type"] == "increase")
    dec_count = sum(1 for r in rows if r["change_type"] == "decrease")

    lines = [f"\U0001f4ca週間機関空売りランキング({m1}/{d1}-{m2}/{d2})"]
    for i, (name, cnt) in enumerate(top5, 1):
        lines.append(f"{i}. {shorten_ss_name(name)} {cnt}件")
    lines.append(f"新規{new_count} 増加{inc_count} 減少{dec_count}")
    lines.append(f"詳細→ {NOTE_URL}")
    lines.append("#日本株 #機関空売り")

    text = "\n".join(lines)
    if len(text) > MAX_TWEET_LEN:
        text = text[:MAX_TWEET_LEN - 1] + "\u2026"
    return text


def format_verification_tweet(stats: Dict, end_date: str) -> str:
    """Format weekly verification tweet with hit rate."""
    m, d = end_date[5:7].lstrip("0"), end_date[8:10].lstrip("0")
    if stats["count"] == 0:
        return ""
    lines = [
        f"\u2705週次検証レポート({m}/{d})",
        f"機関集中銘柄{stats['count']}件の追跡結果:",
        f"5日的中率: {stats['hit_rate_5d']}% (平均{stats['avg_ret_5d']:+.1f}%)",
    ]
    if stats.get("hit_rate_20d"):
        lines.append(f"20日的中率: {stats['hit_rate_20d']}% (平均{stats['avg_ret_20d']:+.1f}%)")
    lines.append(f"詳細→ {NOTE_URL}")
    lines.append("#日本株 #機関空売り")
    text = "\n".join(lines)
    if len(text) > MAX_TWEET_LEN:
        text = text[:MAX_TWEET_LEN - 1] + "\u2026"
    return text


def format_anomaly_tweet(anomalies: List[Dict], date: str) -> str:
    """Format anomaly detection tweet."""
    if not anomalies:
        return ""
    m, d = date[5:7].lstrip("0"), date[8:10].lstrip("0")
    lines = [f"\u26a0\ufe0f空売り異常検知({m}/{d})"]
    for a in anomalies[:3]:
        direction = "\u2b06\ufe0f急増" if a["z_score"] > 0 else "\u2b07\ufe0f急減"
        lines.append(
            f"{direction} {shorten_company(a['company_name'])}({a['code_4']}) "
            f"{a['ratio']:.2f}% (平均{a['avg_ratio']:.2f}%)"
        )
    lines.append("#日本株 #機関空売り")
    text = "\n".join(lines)
    if len(text) > MAX_TWEET_LEN:
        text = text[:MAX_TWEET_LEN - 1] + "\u2026"
    return text


def format_sentiment_tweet(sentiment: Dict, date: str) -> str:
    """Format market sentiment barometer tweet."""
    m, d = date[5:7].lstrip("0"), date[8:10].lstrip("0")
    trend_arrow = "\u2197\ufe0f" if sentiment["trend"] > 2 else "\u2198\ufe0f" if sentiment["trend"] < -2 else "\u27a1\ufe0f"
    lines = [
        f"{sentiment['emoji']}空売りセンチメント({m}/{d}): {sentiment['label']}",
        f"攻勢度: {sentiment['aggression']:.0f}% {trend_arrow}",
        f"新規{sentiment['new']} 増加{sentiment['increase']} 減少{sentiment['decrease']}",
        f"計{sentiment['total']}件",
        "#日本株 #機関空売り",
    ]
    text = "\n".join(lines)
    if len(text) > MAX_TWEET_LEN:
        text = text[:MAX_TWEET_LEN - 1] + "\u2026"
    return text


def format_monthly_tweet(stats: Dict, verif_stats: Optional[Dict] = None) -> str:
    """Format monthly summary tweet with scorecard."""
    y, m = stats["year"], stats["month"]
    lines = [
        f"\U0001f4cb月次スコアカード({y}年{m}月)",
        f"開示数: {stats['total_disclosures']}件",
        f"新規: {stats['new_positions']} 増加: {stats['increases']} 減少: {stats['decreases']}",
    ]
    if verif_stats and verif_stats.get("count", 0) > 0:
        lines.append(f"集中銘柄的中率: {verif_stats['hit_rate_5d']}%(5日) {verif_stats['hit_rate_20d']}%(20日)")
        lines.append(f"平均α: {verif_stats['avg_ret_5d']:+.1f}%(5日) {verif_stats['avg_ret_20d']:+.1f}%(20日)")
    if stats["top_institutions"]:
        lines.append("最活発機関:")
        for name, cnt in stats["top_institutions"][:3]:
            lines.append(f"  {shorten_ss_name(name)} {cnt}件")
    lines.append(f"詳細分析→ {NOTE_URL}")
    lines.append("#日本株 #機関空売り")

    text = "\n".join(lines)
    if len(text) > MAX_TWEET_LEN:
        text = text[:MAX_TWEET_LEN - 1] + "\u2026"
    return text


# ===================================================================
# X (Twitter) Posting
# ===================================================================
def _credit_log_path() -> Path:
    """Path to cumulative credit usage log."""
    return DATA_DIR / "credit_usage.json"


def _update_credit_log(tweet_id: str, chars: int):
    """Track cumulative post count and estimated cost."""
    path = _credit_log_path()
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"total_posts": 0, "history": []}
    data["total_posts"] += 1
    data["history"].append({
        "tweet_id": tweet_id,
        "chars": chars,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # Keep last 500 entries
    data["history"] = data["history"][-500:]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    log.info("[CREDIT] Cumulative posts: %d (check dashboard for balance)", data["total_posts"])


def post_tweet(text: str, dry_run: bool = False) -> Optional[str]:
    """Post via tweepy v2 Client. Returns tweet_id or None."""
    if dry_run:
        log.info("[DRY-RUN] Would post (%d chars):\n%s", len(text), text)
        return None

    api_key = os.environ.get("X_API_KEY")
    api_secret = os.environ.get("X_API_SECRET")
    access_token = os.environ.get("X_ACCESS_TOKEN")
    access_secret = os.environ.get("X_ACCESS_SECRET")

    if not all([api_key, api_secret, access_token, access_secret]):
        log.warning("X API keys not set. Skipping post.")
        return None

    try:
        import tweepy
        client = tweepy.Client(
            consumer_key=api_key, consumer_secret=api_secret,
            access_token=access_token, access_token_secret=access_secret,
        )
        response = client.create_tweet(text=text)
        tweet_id = str(response.data["id"])
        log.info("Posted tweet %s (%d chars)", tweet_id, len(text))
        _update_credit_log(tweet_id, len(text))
        return tweet_id
    except Exception as exc:
        log.error("Failed to post tweet: %s", exc)
        return None


# ===================================================================
# Chart Generation (for note articles)
# ===================================================================
def generate_charts(disclosures: List[Dict], date: str,
                    client: Optional[JQuantsClient] = None):
    """Generate charts for note articles."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed. Skipping charts.")
        return

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    for font in ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "IPAGothic", "Noto Sans CJK JP"]:
        try:
            plt.rcParams["font.family"] = font
            break
        except Exception:
            continue

    if not disclosures:
        return

    from collections import Counter

    # 1. Institution activity bar chart
    ss_counts = Counter(d["ss_short"] for d in disclosures)
    top = ss_counts.most_common(15)
    if top:
        names, counts = zip(*reversed(top))
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#ff4444" if c >= 5 else "#ff8800" if c >= 3 else "#4488cc" for c in counts]
        ax.barh(names, counts, color=colors)
        ax.set_xlabel("開示件数")
        ax.set_title(f"機関投資家別 空売り開示件数 ({date})")
        plt.tight_layout()
        path = CHART_DIR / f"{date}_institution_bar.png"
        plt.savefig(path, dpi=150)
        plt.close()
        log.info("Saved: %s", path)

    # 2. Change type pie chart
    type_counts = Counter(d["change_type"] for d in disclosures)
    labels_map = {"new": "新規", "increase": "増加", "decrease": "減少", "unchanged": "変更なし"}
    labels = [labels_map.get(k, k) for k in type_counts.keys()]
    values = list(type_counts.values())
    colors_pie = {"new": "#ff4444", "increase": "#ff8800", "decrease": "#44aa44", "unchanged": "#999999"}
    pie_colors = [colors_pie.get(k, "#999999") for k in type_counts.keys()]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(values, labels=labels, colors=pie_colors, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"空売りポジション変動内訳 ({date})")
    plt.tight_layout()
    path = CHART_DIR / f"{date}_change_pie.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Saved: %s", path)

    # 3. Sector heatmap (if short-ratio data available)
    try:
        sector_data = client.get_short_ratio(date=date.replace("-", "")) if client else []
    except Exception:
        sector_data = []
    if sector_data:
        import numpy as np
        sdf = pd.DataFrame(sector_data)
        for col in ["SellExShortVa", "ShrtWithResVa", "ShrtNoResVa"]:
            if col in sdf.columns:
                sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
        if {"SellExShortVa", "ShrtWithResVa", "ShrtNoResVa", "S33"}.issubset(sdf.columns):
            sdf["total"] = sdf["SellExShortVa"] + sdf["ShrtWithResVa"] + sdf["ShrtNoResVa"]
            sdf["short_pct"] = ((sdf["ShrtWithResVa"] + sdf["ShrtNoResVa"]) / sdf["total"] * 100).replace(
                [float("inf"), float("-inf")], float("nan"))
            sdf = sdf.dropna(subset=["short_pct"]).sort_values("short_pct", ascending=True)
            if len(sdf) > 0:
                fig, ax = plt.subplots(figsize=(10, max(6, len(sdf) * 0.3)))
                colors_hm = ["#44aa44" if v < 30 else "#ff8800" if v < 50 else "#ff4444"
                             for v in sdf["short_pct"]]
                ax.barh(sdf["S33"].values, sdf["short_pct"].values, color=colors_hm)
                ax.set_xlabel("空売り比率 (%)")
                ax.set_title(f"セクター別 空売り比率 ({date})")
                plt.tight_layout()
                path = CHART_DIR / f"{date}_sector_heatmap.png"
                plt.savefig(path, dpi=150)
                plt.close()
                log.info("Saved: %s", path)

    # 4. Top shorted stocks
    stock_ratios: Dict[str, float] = {}
    for d in disclosures:
        key = f"{shorten_company(d['company_name'])}({d['code_4']})"
        stock_ratios[key] = stock_ratios.get(key, 0) + d["ratio"]
    top_stocks = sorted(stock_ratios.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_stocks:
        names, ratios = zip(*reversed(top_stocks))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, ratios, color="#cc4444")
        ax.set_xlabel("空売り比率合計 (%)")
        ax.set_title(f"機関空売り集中銘柄 ({date})")
        plt.tight_layout()
        path = CHART_DIR / f"{date}_stock_bar.png"
        plt.savefig(path, dpi=150)
        plt.close()
        log.info("Saved: %s", path)


# ===================================================================
# Trading Date Helper
# ===================================================================
def get_latest_disc_date(client: JQuantsClient, target: Optional[str] = None) -> Optional[str]:
    """Find the most recent date with short-sale disclosures."""
    dt = datetime.strptime(target, "%Y-%m-%d") if target else datetime.now()
    for offset in range(8):
        candidate = (dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            data = client.get_short_sale(disc_date=candidate)
            if data:
                log.info("Found disclosure data for %s (%d records)", candidate, len(data))
                return candidate
        except Exception:
            pass
    log.error("No disclosure data found within 7 days of %s", dt.strftime("%Y-%m-%d"))
    return None


# ===================================================================
# Main Pipelines
# ===================================================================
def run_daily(client: JQuantsClient, db: TrackerDB, dry_run: bool = False,
              target_date: Optional[str] = None):
    """Daily pipeline: disclosure changes + concentrated shorts."""
    date = get_latest_disc_date(client, target_date)
    if not date:
        log.error("Cannot determine disclosure date. Aborting.")
        return

    if db.has_post(date, "daily") and not dry_run:
        log.info("Already posted for %s. Skipping.", date)
        return

    stock_map = get_stock_name_map(client)
    disclosures = fetch_and_categorize(client, date, stock_map)
    if not disclosures:
        log.info("No disclosures for %s", date)
        return

    # Store disclosures
    if not db.has_disclosures(date):
        for d in disclosures:
            db.insert_disclosure(d)
        log.info("Stored %d disclosures for %s", len(disclosures), date)

    # --- Tweet 1: Daily changes ---
    daily_text = format_daily_changes(disclosures, date)
    daily_tid = post_tweet(daily_text, dry_run=dry_run)
    db.insert_post(date, "daily", daily_text, daily_tid,
                   {"total": len(disclosures),
                    "new": len([d for d in disclosures if d["change_type"] == "new"]),
                    "increase": len([d for d in disclosures if d["change_type"] == "increase"]),
                    "decrease": len([d for d in disclosures if d["change_type"] == "decrease"])})

    # --- Tweet 2: Concentrated shorts (if any) ---
    conc_tid = None
    concentrated = find_concentrated_shorts(disclosures)
    if concentrated:
        if daily_tid and not dry_run:
            log.info("Waiting 30s before next post...")
            time.sleep(30)
        conc_text = format_concentrated_tweet(concentrated, date)
        conc_tid = post_tweet(conc_text, dry_run=dry_run)
        db.insert_post(date, "concentrated", conc_text, conc_tid,
                       [{"code": c["code_4"], "count": c["count"],
                         "total_ratio": c["total_ratio"]} for c in concentrated[:5]])

    # --- Tweet 3: Sentiment barometer ---
    sentiment = calculate_sentiment(disclosures, db, date)
    last_tid = conc_tid or daily_tid
    if last_tid and not dry_run:
        time.sleep(30)
    sent_text = format_sentiment_tweet(sentiment, date)
    sent_tid = post_tweet(sent_text, dry_run=dry_run)
    db.insert_post(date, "sentiment", sent_text, sent_tid, sentiment)

    # --- Tweet 4: Anomaly detection (if any) ---
    anomalies = detect_anomalies(db, date, disclosures)
    if anomalies:
        if sent_tid and not dry_run:
            time.sleep(30)
        anom_text = format_anomaly_tweet(anomalies, date)
        if anom_text:
            anom_tid = post_tweet(anom_text, dry_run=dry_run)
            db.insert_post(date, "anomaly", anom_text, anom_tid,
                           [{"code": a["code_4"], "z": a["z_score"]} for a in anomalies])

    # --- Charts ---
    generate_charts(disclosures, date, client=client)


def run_weekly(client: JQuantsClient, db: TrackerDB, dry_run: bool = False,
               target_date: Optional[str] = None):
    """Weekly institution ranking + verification report."""
    date = get_latest_disc_date(client, target_date)
    if not date:
        return

    if db.has_post(date, "weekly") and not dry_run:
        log.info("Already posted weekly for %s. Skipping.", date)
        return

    # --- Tweet 1: Weekly ranking ---
    text = format_weekly_tweet(db, date)
    tid = post_tweet(text, dry_run=dry_run)
    db.insert_post(date, "weekly", text, tid)

    # --- Verify past signals ---
    verified = verify_past_signals(client, db)

    # --- Tweet 2: Verification report ---
    dt = datetime.strptime(date, "%Y-%m-%d")
    verif_from = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
    verif_stats = db.get_verification_stats(verif_from, date)
    if verif_stats["count"] > 0:
        if tid and not dry_run:
            time.sleep(30)
        verif_text = format_verification_tweet(verif_stats, date)
        if verif_text:
            verif_tid = post_tweet(verif_text, dry_run=dry_run)
            db.insert_post(date, "verification", verif_text, verif_tid, verif_stats)


def run_monthly(db: TrackerDB, dry_run: bool = False):
    """Monthly summary with verification scorecard."""
    now = datetime.now()
    if now.month == 1:
        year, month = now.year - 1, 12
    else:
        year, month = now.year, now.month - 1

    stats = db.get_monthly_stats(year, month)
    if stats["total_disclosures"] == 0:
        log.info("No disclosures for %04d-%02d. Skipping monthly.", year, month)
        return

    # Get verification stats for the month
    prefix = f"{year}-{month:02d}"
    verif_stats = db.get_verification_stats(f"{prefix}-01", f"{prefix}-31")

    text = format_monthly_tweet(stats, verif_stats=verif_stats)
    tid = post_tweet(text, dry_run=dry_run)
    db.insert_post(f"{year}-{month:02d}-01", "monthly", text, tid, stats)


def run_charts(client: JQuantsClient, db: TrackerDB, target_date: Optional[str] = None):
    """Generate charts only."""
    date = get_latest_disc_date(client, target_date)
    if not date:
        return
    stock_map = get_stock_name_map(client)
    disclosures = fetch_and_categorize(client, date, stock_map)
    if disclosures:
        generate_charts(disclosures, date)


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="X Bot JP - 機関投資家 空売り動向トラッカー",
    )
    parser.add_argument("--mode", choices=["daily", "weekly", "monthly", "charts", "all", "status"],
                        default="daily")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date", type=str, default=None, help="Override date (YYYY-MM-DD)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Status mode doesn't need J-Quants or DB
    if args.mode == "status":
        path = _credit_log_path()
        if path.exists():
            data = json.loads(path.read_text())
            print(f"=== X Bot Credit Status ===")
            print(f"Total posts: {data['total_posts']}")
            if data["history"]:
                last = data["history"][-1]
                print(f"Last post:   {last['timestamp']} (tweet {last['tweet_id']})")
            print(f"Log file:    {path}")
            print(f"Dashboard:   https://developer.x.com/en/portal/dashboard")
        else:
            print("No posts recorded yet.")
        return

    try:
        client = JQuantsClient()
    except RuntimeError:
        log.error("Cannot initialize J-Quants client. Exiting.")
        sys.exit(1)

    db = TrackerDB()

    try:
        if args.mode == "daily":
            run_daily(client, db, dry_run=args.dry_run, target_date=args.date)
        elif args.mode == "weekly":
            run_weekly(client, db, dry_run=args.dry_run, target_date=args.date)
        elif args.mode == "monthly":
            run_monthly(db, dry_run=args.dry_run)
        elif args.mode == "charts":
            run_charts(client, db, target_date=args.date)
        elif args.mode == "all":
            run_daily(client, db, dry_run=args.dry_run, target_date=args.date)
            run_weekly(client, db, dry_run=args.dry_run, target_date=args.date)
            run_monthly(db, dry_run=args.dry_run)
        log.info("Done.")
    except KeyboardInterrupt:
        log.info("Interrupted.")
    except Exception as exc:
        log.error("Unhandled error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
