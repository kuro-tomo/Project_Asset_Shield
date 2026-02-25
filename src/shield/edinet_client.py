"""
EDINET API Client for Asset Shield
====================================
金融庁 EDINET API v2 クライアント

提供機能:
  - 書類一覧取得 (documents list)
  - 書類ダウンロード (XBRL/PDF/CSV)
  - 大量保有報告書パーサー (bulk holding reports)
  - 有価証券報告書CSVパーサー (annual report fundamentals)
  - SQLiteキャッシュ (incremental)

環境変数:
  EDINET_API_KEY: Subscription-Key (必須)

Author: Asset Shield Project
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("edinet_client")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"

# 主要書類種別コード
DOCTYPE_ANNUAL_REPORT = "120"       # 有価証券報告書
DOCTYPE_QUARTERLY_REPORT = "140"    # 四半期報告書
DOCTYPE_SEMIANNUAL_REPORT = "160"   # 半期報告書
DOCTYPE_EXTRAORDINARY = "180"       # 臨時報告書
DOCTYPE_BULK_HOLDING = "350"        # 大量保有報告書
DOCTYPE_BULK_HOLDING_CHG = "360"    # 変更報告書(大量保有)
DOCTYPE_SHARE_BUYBACK = "220"       # 自己株券買付状況報告書

# ドキュメントタイプ (download)
DOC_TYPE_XBRL = 1
DOC_TYPE_PDF = 2
DOC_TYPE_ATTACHMENT = 3
DOC_TYPE_ENGLISH = 4
DOC_TYPE_CSV = 5

# レート制限
REQUEST_INTERVAL = 3.0  # 秒 (安全マージン)
RETRY_429_WAIT = 60     # 秒 (429エラー時)
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EDINETConfig:
    """EDINET API configuration."""
    api_key: str = ""
    base_url: str = EDINET_BASE
    cache_dir: Path = Path("data/edinet/cache")
    db_path: Path = Path("data/edinet/edinet.db")
    request_interval: float = REQUEST_INTERVAL


# ===========================================================================
# EDINET API Client
# ===========================================================================

class EDINETClient:
    """
    EDINET API v2 クライアント

    - ページネーション不要 (1日分の書類一覧を一括返却)
    - レート制限: 3秒間隔 (429対策)
    - SQLiteキャッシュ: 書類一覧 + ダウンロード済み書類
    """

    def __init__(self, config: Optional[EDINETConfig] = None,
                 project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.config = config or EDINETConfig(
            api_key=os.environ.get("EDINET_API_KEY", ""),
            cache_dir=self.project_root / "data" / "edinet" / "cache",
            db_path=self.project_root / "data" / "edinet" / "edinet.db",
        )
        if not self.config.api_key:
            log.warning("EDINET_API_KEY not set. API calls will fail.")

        self.session = requests.Session()
        self._last_request_time = 0.0

        # Ensure directories
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Init database
        self._init_db()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self):
        """Initialize SQLite tables for EDINET data."""
        con = sqlite3.connect(self.config.db_path)
        con.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id          TEXT PRIMARY KEY,
                edinet_code     TEXT,
                sec_code        TEXT,
                filer_name      TEXT,
                doc_type_code   TEXT,
                doc_description TEXT,
                submit_datetime TEXT,
                period_start    TEXT,
                period_end      TEXT,
                xbrl_flag       INTEGER,
                pdf_flag        INTEGER,
                csv_flag        INTEGER,
                fetch_date      TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_doc_sec ON documents(sec_code);
            CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type_code);
            CREATE INDEX IF NOT EXISTS idx_doc_fetch ON documents(fetch_date);

            CREATE TABLE IF NOT EXISTS bulk_holdings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id          TEXT,
                edinet_code     TEXT,
                sec_code        TEXT,
                issuer_name     TEXT,
                filer_name      TEXT,
                holding_ratio   REAL,
                prev_ratio      REAL,
                change_ratio    REAL,
                purpose         TEXT,
                submit_date     TEXT,
                report_type     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(doc_id)
            );
            CREATE INDEX IF NOT EXISTS idx_bh_sec ON bulk_holdings(sec_code);
            CREATE INDEX IF NOT EXISTS idx_bh_filer ON bulk_holdings(filer_name);
            CREATE INDEX IF NOT EXISTS idx_bh_date ON bulk_holdings(submit_date);

            CREATE TABLE IF NOT EXISTS fundamentals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id          TEXT,
                sec_code        TEXT,
                filer_name      TEXT,
                period_start    TEXT,
                period_end      TEXT,
                revenue         REAL,
                operating_profit REAL,
                ordinary_profit REAL,
                net_income      REAL,
                total_assets    REAL,
                net_assets      REAL,
                equity_ratio    REAL,
                cash_flow_ops   REAL,
                cash_flow_inv   REAL,
                cash_flow_fin   REAL,
                eps             REAL,
                bps             REAL,
                roe             REAL,
                submit_date     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(doc_id)
            );
            CREATE INDEX IF NOT EXISTS idx_fund_sec ON fundamentals(sec_code);
            CREATE INDEX IF NOT EXISTS idx_fund_period ON fundamentals(period_end);
        """)
        con.close()

    def _get_db(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.config.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _throttle(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.request_interval:
            time.sleep(self.config.request_interval - elapsed)

    # ------------------------------------------------------------------
    # Core API methods
    # ------------------------------------------------------------------

    def _api_get(self, url: str, params: Dict[str, Any],
                 stream: bool = False) -> requests.Response:
        """Execute GET request with throttling and retry on 429."""
        params["Subscription-Key"] = self.config.api_key

        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, stream=stream,
                                        timeout=120)
                self._last_request_time = time.time()

                if resp.status_code == 429:
                    wait = RETRY_429_WAIT * (attempt + 1)
                    log.warning(f"429 Too Many Requests. Waiting {wait}s ...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.RequestException as e:
                log.error(f"Request error (attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(self.config.request_interval * 2)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def get_document_list(self, date: str, doc_type: int = 2) -> Dict[str, Any]:
        """
        書類一覧API: 指定日の提出書類一覧を取得

        Args:
            date: YYYY-MM-DD 形式の日付
            doc_type: 1=メタデータのみ, 2=書類一覧+メタデータ

        Returns:
            {"metadata": {...}, "results": [...]}
        """
        url = f"{self.config.base_url}/documents.json"
        params = {"date": date, "type": doc_type}
        resp = self._api_get(url, params)
        return resp.json()

    def download_document(self, doc_id: str, doc_type: int = DOC_TYPE_CSV,
                          save: bool = True) -> bytes:
        """
        書類取得API: 指定書類をダウンロード

        Args:
            doc_id: 書類管理番号
            doc_type: 1=XBRL, 2=PDF, 3=添付, 4=英語, 5=CSV
            save: キャッシュディレクトリに保存するか

        Returns:
            bytes: ダウンロードしたファイル内容
        """
        url = f"{self.config.base_url}/documents/{doc_id}"
        params = {"type": doc_type}
        resp = self._api_get(url, params)
        content = resp.content

        if save:
            ext = "pdf" if doc_type == DOC_TYPE_PDF else "zip"
            path = self.config.cache_dir / f"{doc_id}_type{doc_type}.{ext}"
            path.write_bytes(content)
            log.info(f"Saved {path.name} ({len(content):,} bytes)")

        return content

    # ------------------------------------------------------------------
    # Bulk Holding Reports (大量保有報告書)
    # ------------------------------------------------------------------

    def fetch_bulk_holdings(self, date_from: str, date_to: str,
                            ) -> List[Dict[str, Any]]:
        """
        指定期間の大量保有報告書を取得・パース (CSV download方式)

        Args:
            date_from: 開始日 (YYYY-MM-DD)
            date_to: 終了日 (YYYY-MM-DD)

        Returns:
            List of parsed bulk holding records
        """
        results = []
        current = datetime.strptime(date_from, "%Y-%m-%d")
        end = datetime.strptime(date_to, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")

            # Check cache first
            con = self._get_db()
            cached = con.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE fetch_date = ?",
                (date_str,)
            ).fetchone()["cnt"]
            con.close()

            if cached == 0:
                try:
                    data = self.get_document_list(date_str)
                    docs = data.get("results") or []
                    self._store_documents(docs, date_str)
                    log.info(f"{date_str}: {len(docs)} documents fetched")
                except Exception as e:
                    log.error(f"{date_str}: fetch error: {e}")
                    current += timedelta(days=1)
                    continue

            # Extract bulk holding reports from DB
            con = self._get_db()
            rows = con.execute(
                """SELECT * FROM documents
                   WHERE fetch_date = ?
                   AND doc_type_code IN (?, ?)
                   AND csv_flag = 1""",
                (date_str, DOCTYPE_BULK_HOLDING, DOCTYPE_BULK_HOLDING_CHG)
            ).fetchall()
            con.close()

            # Check which doc_ids are already parsed
            if rows:
                doc_ids = [dict(r)["doc_id"] for r in rows]
                con = self._get_db()
                placeholders = ",".join("?" * len(doc_ids))
                already = {r["doc_id"] for r in con.execute(
                    f"SELECT doc_id FROM bulk_holdings WHERE doc_id IN ({placeholders})"
                    " AND issuer_name != '' AND issuer_name IS NOT NULL"
                    " AND holding_ratio IS NOT NULL",
                    doc_ids
                ).fetchall()}
                con.close()

                for row in rows:
                    doc = dict(row)
                    if doc["doc_id"] in already:
                        # Already properly parsed — load from DB
                        con = self._get_db()
                        cached_row = con.execute(
                            "SELECT * FROM bulk_holdings WHERE doc_id = ?",
                            (doc["doc_id"],)
                        ).fetchone()
                        con.close()
                        if cached_row:
                            results.append(dict(cached_row))
                        continue

                    # Download CSV and parse
                    parsed = self._parse_bulk_holding_csv(doc)
                    if parsed:
                        results.append(parsed)

            current += timedelta(days=1)

        # Store parsed holdings
        if results:
            self._store_bulk_holdings(results)

        return results

    def _store_documents(self, docs: List[Dict], fetch_date: str):
        """Store document list in SQLite."""
        con = self._get_db()
        for doc in docs:
            try:
                con.execute("""
                    INSERT OR IGNORE INTO documents
                    (doc_id, edinet_code, sec_code, filer_name,
                     doc_type_code, doc_description, submit_datetime,
                     period_start, period_end, xbrl_flag, pdf_flag,
                     csv_flag, fetch_date)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    doc.get("docID"),
                    doc.get("edinetCode"),
                    doc.get("secCode"),
                    doc.get("filerName"),
                    doc.get("docTypeCode"),
                    doc.get("docDescription"),
                    doc.get("submitDateTime"),
                    doc.get("periodStart"),
                    doc.get("periodEnd"),
                    int(doc.get("xbrlFlag") or 0),
                    int(doc.get("pdfFlag") or 0),
                    int(doc.get("csvFlag") or 0),
                    fetch_date,
                ))
            except Exception as e:
                log.debug(f"Document insert skip: {e}")
        con.commit()
        con.close()

    def _parse_bulk_holding_csv(self, doc: Dict) -> Optional[Dict[str, Any]]:
        """
        Download and parse bulk holding report CSV from EDINET.

        CSVのXBRLタクソノミから以下を抽出:
          - 発行者の名称 / 発行者の証券コード
          - 株券等保有割合 / 直前の報告書に記載された株券等保有割合
          - 保有目的
        """
        doc_id = doc.get("doc_id") or doc.get("docID") or ""
        filer = doc.get("filer_name") or doc.get("filerName") or ""
        submit = doc.get("submit_datetime") or doc.get("submitDateTime") or ""
        doc_type = doc.get("doc_type_code") or doc.get("docTypeCode") or ""
        edinet_code = doc.get("edinet_code") or doc.get("edinetCode") or ""

        try:
            content = self.download_document(doc_id, DOC_TYPE_CSV, save=False)
            zf = zipfile.ZipFile(io.BytesIO(content))
        except Exception as e:
            log.debug(f"CSV download/open failed for {doc_id}: {e}")
            return None

        csv_files = [f for f in zf.namelist()
                     if "XBRL_TO_CSV" in f and f.endswith(".csv")]
        if not csv_files:
            return None

        # Parse CSV: build key-value map from XBRL elements
        elements = {}
        for csv_name in csv_files:
            try:
                raw = zf.read(csv_name)
                try:
                    text = raw.decode("utf-16")
                except UnicodeDecodeError:
                    text = raw.decode("utf-8")

                for line in text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        item_name = parts[1].strip().strip('"')
                        val = parts[-1].strip().strip('"')
                        if val and val != "" and val != "－":
                            elements[item_name] = val
            except Exception as e:
                log.debug(f"CSV parse error {csv_name}: {e}")

        # Extract fields
        issuer_name = elements.get("発行者の名称", "")
        sec_code_raw = elements.get("発行者の証券コード", "")
        # Normalize sec_code: "9632" → "96320" (5-digit)
        sec_code = sec_code_raw.strip()
        if sec_code and len(sec_code) == 4:
            sec_code = sec_code + "0"

        # Holding ratio (decimal → percentage)
        holding_ratio = None
        ratio_str = elements.get("株券等保有割合", "")
        if ratio_str:
            try:
                val = float(ratio_str)
                holding_ratio = val * 100 if val < 1 else val
            except (ValueError, TypeError):
                pass

        # Previous ratio
        prev_ratio = None
        prev_str = elements.get("直前の報告書に記載された株券等保有割合", "")
        if prev_str:
            try:
                val = float(prev_str)
                prev_ratio = val * 100 if val < 1 else val
            except (ValueError, TypeError):
                pass

        change_ratio = None
        if holding_ratio is not None and prev_ratio is not None:
            change_ratio = holding_ratio - prev_ratio

        # Purpose
        purpose = elements.get("保有目的", "")
        if "純投資" in purpose:
            purpose = "純投資"
        elif "経営参加" in purpose:
            purpose = "経営参加"
        elif "政策投資" in purpose:
            purpose = "政策投資"

        # Report type from doc_type_code
        desc = doc.get("doc_description") or doc.get("docDescription") or ""
        if "訂正" in desc:
            report_type = "correction"
        elif doc_type == DOCTYPE_BULK_HOLDING:
            doc_title = elements.get("提出書類、表紙", "")
            report_type = "new" if "大量" in doc_title and "変更" not in doc_title else "change"
        else:
            report_type = "change"

        log.info(f"  Parsed {doc_id}: {issuer_name}({sec_code}) "
                 f"{holding_ratio}% filer={filer[:20]}")

        return {
            "doc_id": doc_id,
            "edinet_code": edinet_code,
            "sec_code": sec_code,
            "issuer_name": issuer_name,
            "filer_name": filer,
            "holding_ratio": holding_ratio,
            "prev_ratio": prev_ratio,
            "change_ratio": change_ratio,
            "purpose": purpose,
            "submit_date": submit[:10] if submit else "",
            "report_type": report_type,
        }

    def _store_bulk_holdings(self, holdings: List[Dict]):
        """Store parsed bulk holdings in SQLite (upsert)."""
        con = self._get_db()
        for h in holdings:
            try:
                con.execute("""
                    INSERT INTO bulk_holdings
                    (doc_id, edinet_code, sec_code, issuer_name, filer_name,
                     holding_ratio, prev_ratio, change_ratio, purpose,
                     submit_date, report_type)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        sec_code = excluded.sec_code,
                        issuer_name = excluded.issuer_name,
                        holding_ratio = excluded.holding_ratio,
                        prev_ratio = excluded.prev_ratio,
                        change_ratio = excluded.change_ratio,
                        purpose = excluded.purpose,
                        report_type = excluded.report_type
                """, (
                    h["doc_id"], h.get("edinet_code", ""), h.get("sec_code", ""),
                    h.get("issuer_name", ""), h.get("filer_name", ""),
                    h.get("holding_ratio"), h.get("prev_ratio"), h.get("change_ratio"),
                    h.get("purpose", ""), h.get("submit_date", ""), h.get("report_type", ""),
                ))
            except Exception as e:
                log.debug(f"Bulk holding insert skip: {e}")
        con.commit()
        con.close()

    # ------------------------------------------------------------------
    # Annual Report Fundamentals (有価証券報告書)
    # ------------------------------------------------------------------

    def fetch_fundamentals(self, date_from: str, date_to: str,
                           sec_codes: Optional[List[str]] = None,
                           ) -> pd.DataFrame:
        """
        指定期間の有価証券報告書からファンダメンタルデータを取得

        Args:
            date_from: 開始日
            date_to: 終了日
            sec_codes: フィルタ対象の証券コード (None=全て)

        Returns:
            DataFrame with fundamental data
        """
        # First fetch document list for the period
        current = datetime.strptime(date_from, "%Y-%m-%d")
        end = datetime.strptime(date_to, "%Y-%m-%d")
        annual_docs = []

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")

            # Check cache
            con = self._get_db()
            cached = con.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE fetch_date = ?",
                (date_str,)
            ).fetchone()["cnt"]
            con.close()

            if cached == 0:
                try:
                    data = self.get_document_list(date_str)
                    docs = data.get("results") or []
                    self._store_documents(docs, date_str)
                except Exception as e:
                    log.error(f"{date_str}: fetch error: {e}")
                    current += timedelta(days=1)
                    continue

            # Find annual/quarterly reports (only listed companies with sec_code)
            con = self._get_db()
            query = """
                SELECT * FROM documents
                WHERE fetch_date = ?
                AND doc_type_code IN (?, ?, ?)
                AND csv_flag = 1
                AND sec_code IS NOT NULL AND sec_code != ''
            """
            params: list = [date_str, DOCTYPE_ANNUAL_REPORT,
                            DOCTYPE_QUARTERLY_REPORT, DOCTYPE_SEMIANNUAL_REPORT]

            if sec_codes:
                placeholders = ",".join("?" * len(sec_codes))
                query += f" AND sec_code IN ({placeholders})"
                params.extend(sec_codes)

            rows = con.execute(query, params).fetchall()
            con.close()
            annual_docs.extend([dict(r) for r in rows])

            current += timedelta(days=1)

        # Download and parse CSV for each document
        parsed = []
        # Build set of already-complete doc_ids
        if annual_docs:
            con = self._get_db()
            all_ids = [d["doc_id"] for d in annual_docs]
            ph = ",".join("?" * len(all_ids))
            already = {r["doc_id"] for r in con.execute(
                f"SELECT doc_id FROM fundamentals WHERE doc_id IN ({ph})"
                " AND revenue IS NOT NULL",
                all_ids
            ).fetchall()}
            con.close()
        else:
            already = set()

        for doc in annual_docs:
            doc_id = doc["doc_id"]
            if doc_id in already:
                continue

            try:
                content = self.download_document(doc_id, DOC_TYPE_CSV)
                fund = self._parse_csv_fundamentals(content, doc)
                if fund:
                    parsed.append(fund)
                    log.info(f"  Fund: {doc.get('filer_name','')[:20]} "
                             f"rev={fund.get('revenue')} op={fund.get('operating_profit')}")
            except Exception as e:
                log.warning(f"Failed to parse {doc_id}: {e}")

        # Store parsed fundamentals
        if parsed:
            self._store_fundamentals(parsed)

        # Return all fundamentals for the period
        return self._load_fundamentals(date_from, date_to, sec_codes)

    def _parse_csv_fundamentals(self, zip_content: bytes,
                                doc: Dict) -> Optional[Dict[str, Any]]:
        """
        Parse XBRL_TO_CSV (type=5) zip content to extract financials.

        EDINET CSV: UTF-16, tab-separated, in XBRL_TO_CSV/ folder
        Key XBRL elements:
          jppfs_cor:Revenue / jppfs_cor:NetSales — 売上高
          jppfs_cor:OperatingProfit — 営業利益
          jppfs_cor:OrdinaryProfit — 経常利益
          jppfs_cor:ProfitLossAttributableToOwnersOfParent — 当期純利益
          jppfs_cor:TotalAssets — 総資産
          jppfs_cor:NetAssets — 純資産
          jppfs_cor:EquityToAssetRatio — 自己資本比率
          jppfs_cor:CashAndCashEquivalentsEOYPeriod — 現金及び預金
        """
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_content))
        except zipfile.BadZipFile:
            log.warning(f"Invalid zip for {doc.get('doc_id')}")
            return None

        # Find CSV files in XBRL_TO_CSV directory
        csv_files = [f for f in zf.namelist()
                     if "XBRL_TO_CSV" in f and f.endswith(".csv")]
        if not csv_files:
            return None

        # Parse all CSV files, collecting XBRL elements
        # Use both XBRL ID (parts[0]) and Japanese name (parts[1]) as keys
        elements = {}
        for csv_name in csv_files:
            try:
                raw = zf.read(csv_name)
                # Try UTF-16 first, fallback to UTF-8
                try:
                    text = raw.decode("utf-16")
                except UnicodeDecodeError:
                    text = raw.decode("utf-8")

                for line in text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        xbrl_id = parts[0].strip().strip('"')
                        item_name = parts[1].strip().strip('"') if len(parts) > 1 else ""
                        val = parts[-1].strip().strip('"')
                        if val and val != "":
                            elements[xbrl_id] = val
                            if item_name:
                                elements[item_name] = val
            except Exception as e:
                log.debug(f"CSV parse error {csv_name}: {e}")

        # Extract financial values
        def get_num(keys: List[str]) -> Optional[float]:
            for k in keys:
                for ek, ev in elements.items():
                    if k in ek:
                        try:
                            return float(ev.replace(",", ""))
                        except (ValueError, TypeError):
                            continue
            return None

        revenue = get_num(["Revenue", "NetSales", "売上高", "営業収益"])
        op_profit = get_num(["OperatingIncome", "OperatingProfit", "営業利益"])
        ord_profit = get_num(["OrdinaryProfit", "経常利益"])
        net_income = get_num([
            "ProfitLossAttributableToOwnersOfParent",
            "ProfitLoss",
            "当期純利益",
            "親会社株主に帰属する当期純利益",
        ])
        total_assets = get_num(["TotalAssets", "総資産"])
        net_assets = get_num(["NetAssets", "純資産"])
        equity_ratio = get_num(["EquityToAssetRatio", "自己資本比率"])
        cfo = get_num(["CashFlowsFromOperatingActivities", "営業活動による"])
        cfi = get_num(["CashFlowsFromInvestingActivities", "投資活動による"])
        cff = get_num(["CashFlowsFromFinancingActivities", "財務活動による"])
        eps = get_num(["BasicEarningsLossPerShare", "1株当たり当期純利益"])
        bps = get_num(["NetAssetsPerShare", "1株当たり純資産"])

        # Calculate ROE if possible
        roe = None
        if net_income and net_assets and net_assets > 0:
            roe = (net_income / net_assets) * 100

        return {
            "doc_id": doc["doc_id"],
            "sec_code": doc["sec_code"],
            "filer_name": doc["filer_name"],
            "period_start": doc.get("period_start") or "",
            "period_end": doc.get("period_end") or "",
            "revenue": revenue,
            "operating_profit": op_profit,
            "ordinary_profit": ord_profit,
            "net_income": net_income,
            "total_assets": total_assets,
            "net_assets": net_assets,
            "equity_ratio": equity_ratio,
            "cash_flow_ops": cfo,
            "cash_flow_inv": cfi,
            "cash_flow_fin": cff,
            "eps": eps,
            "bps": bps,
            "roe": roe,
            "submit_date": doc.get("submit_datetime", "")[:10],
        }

    def _store_fundamentals(self, records: List[Dict]):
        """Store parsed fundamentals in SQLite (upsert)."""
        con = self._get_db()
        for r in records:
            try:
                con.execute("""
                    INSERT INTO fundamentals
                    (doc_id, sec_code, filer_name, period_start, period_end,
                     revenue, operating_profit, ordinary_profit, net_income,
                     total_assets, net_assets, equity_ratio,
                     cash_flow_ops, cash_flow_inv, cash_flow_fin,
                     eps, bps, roe, submit_date)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        revenue = excluded.revenue,
                        operating_profit = excluded.operating_profit,
                        ordinary_profit = excluded.ordinary_profit,
                        net_income = excluded.net_income,
                        total_assets = excluded.total_assets,
                        net_assets = excluded.net_assets,
                        equity_ratio = excluded.equity_ratio,
                        cash_flow_ops = excluded.cash_flow_ops,
                        cash_flow_inv = excluded.cash_flow_inv,
                        cash_flow_fin = excluded.cash_flow_fin,
                        eps = excluded.eps, bps = excluded.bps,
                        roe = excluded.roe
                """, (
                    r["doc_id"], r.get("sec_code",""), r.get("filer_name",""),
                    r.get("period_start",""), r.get("period_end",""),
                    r.get("revenue"), r.get("operating_profit"), r.get("ordinary_profit"),
                    r.get("net_income"), r.get("total_assets"), r.get("net_assets"),
                    r.get("equity_ratio"), r.get("cash_flow_ops"), r.get("cash_flow_inv"),
                    r.get("cash_flow_fin"), r.get("eps"), r.get("bps"), r.get("roe"),
                    r.get("submit_date",""),
                ))
            except Exception as e:
                log.debug(f"Fundamentals insert skip: {e}")
        con.commit()
        con.close()

    def _load_fundamentals(self, date_from: str, date_to: str,
                           sec_codes: Optional[List[str]] = None,
                           ) -> pd.DataFrame:
        """Load fundamentals from SQLite as DataFrame."""
        con = self._get_db()
        query = """
            SELECT * FROM fundamentals
            WHERE submit_date BETWEEN ? AND ?
        """
        params: list = [date_from, date_to]
        if sec_codes:
            placeholders = ",".join("?" * len(sec_codes))
            query += f" AND sec_code IN ({placeholders})"
            params.extend(sec_codes)
        query += " ORDER BY sec_code, period_end"

        df = pd.read_sql_query(query, con, params=params)
        con.close()
        return df

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_bulk_holdings_df(self, date_from: str, date_to: str,
                             sec_codes: Optional[List[str]] = None,
                             ) -> pd.DataFrame:
        """
        大量保有報告書をDataFrameで取得 (キャッシュ済み + 新規取得)

        Args:
            date_from: 開始日
            date_to: 終了日
            sec_codes: フィルタ対象の証券コード

        Returns:
            DataFrame with bulk holding data
        """
        # Fetch new data (incremental)
        self.fetch_bulk_holdings(date_from, date_to)

        # Load from DB
        con = self._get_db()
        query = """
            SELECT * FROM bulk_holdings
            WHERE submit_date BETWEEN ? AND ?
        """
        params: list = [date_from, date_to]
        if sec_codes:
            placeholders = ",".join("?" * len(sec_codes))
            query += f" AND sec_code IN ({placeholders})"
            params.extend(sec_codes)
        query += " ORDER BY submit_date DESC, holding_ratio DESC"

        df = pd.read_sql_query(query, con, params=params)
        con.close()
        return df

    def get_recent_bulk_holdings(self, days: int = 7) -> pd.DataFrame:
        """直近N日間の大量保有報告書を取得"""
        today = datetime.now()
        date_from = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        date_to = today.strftime("%Y-%m-%d")
        return self.get_bulk_holdings_df(date_from, date_to)

    def get_latest_fundamentals(self,
                                sec_codes: Optional[List[str]] = None,
                                ) -> pd.DataFrame:
        """
        各銘柄の最新ファンダメンタルデータを取得

        Returns:
            DataFrame with latest fundamental per sec_code
        """
        con = self._get_db()
        query = """
            SELECT f.* FROM fundamentals f
            INNER JOIN (
                SELECT sec_code, MAX(period_end) as max_period
                FROM fundamentals
                GROUP BY sec_code
            ) latest ON f.sec_code = latest.sec_code
                    AND f.period_end = latest.max_period
        """
        if sec_codes:
            placeholders = ",".join("?" * len(sec_codes))
            query = f"""
                SELECT f.* FROM fundamentals f
                INNER JOIN (
                    SELECT sec_code, MAX(period_end) as max_period
                    FROM fundamentals
                    WHERE sec_code IN ({placeholders})
                    GROUP BY sec_code
                ) latest ON f.sec_code = latest.sec_code
                        AND f.period_end = latest.max_period
            """
            df = pd.read_sql_query(query, con, params=sec_codes)
        else:
            df = pd.read_sql_query(query, con)
        con.close()
        return df

    def get_fundamentals_history(self, sec_code: str,
                                 limit: int = 10) -> pd.DataFrame:
        """特定銘柄のファンダメンタル履歴を取得"""
        con = self._get_db()
        df = pd.read_sql_query(
            """SELECT * FROM fundamentals
               WHERE sec_code = ?
               ORDER BY period_end DESC
               LIMIT ?""",
            con, params=[sec_code, limit]
        )
        con.close()
        return df

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def compute_growth_metrics(self, sec_code: str) -> Optional[Dict[str, float]]:
        """
        直近2期分のファンダメンタルから成長率を算出

        Returns:
            dict with revenue_growth, op_margin_chg, roe_chg,
            asset_turnover_chg, cash_flow_yield
        """
        df = self.get_fundamentals_history(sec_code, limit=2)
        if len(df) < 2:
            return None

        latest = df.iloc[0]
        prev = df.iloc[1]

        def safe_growth(cur, prv):
            if prv and prv != 0 and cur is not None:
                return (cur - prv) / abs(prv)
            return None

        def safe_margin(profit, revenue):
            if revenue and revenue != 0 and profit is not None:
                return profit / revenue
            return None

        rev_growth = safe_growth(latest["revenue"], prev["revenue"])

        op_margin_cur = safe_margin(latest["operating_profit"], latest["revenue"])
        op_margin_prev = safe_margin(prev["operating_profit"], prev["revenue"])
        op_margin_chg = None
        if op_margin_cur is not None and op_margin_prev is not None:
            op_margin_chg = op_margin_cur - op_margin_prev

        roe_chg = None
        if latest["roe"] is not None and prev["roe"] is not None:
            roe_chg = latest["roe"] - prev["roe"]

        # Asset turnover = Revenue / Total Assets
        at_cur = safe_margin(latest["revenue"], latest["total_assets"])
        at_prev = safe_margin(prev["revenue"], prev["total_assets"])
        at_chg = None
        if at_cur is not None and at_prev is not None:
            at_chg = at_cur - at_prev

        # Cash flow yield = CFO / Total Assets
        cf_yield = safe_margin(latest["cash_flow_ops"], latest["total_assets"])

        return {
            "revenue_growth": rev_growth,
            "op_margin_chg": op_margin_chg,
            "roe_chg": roe_chg,
            "asset_turnover_chg": at_chg,
            "cash_flow_yield": cf_yield,
        }


# ===========================================================================
# Convenience functions
# ===========================================================================

def create_client(project_root: Optional[Path] = None) -> EDINETClient:
    """Create an EDINETClient with default configuration."""
    return EDINETClient(project_root=project_root)
