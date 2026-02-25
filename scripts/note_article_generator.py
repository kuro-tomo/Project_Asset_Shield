#!/usr/bin/env python3
"""
Note Article Generator - 定期購読マガジン向け記事自動生成
=========================================================
X Bot JP の tracker.db から日次データを読み取り、
note.com 定期購読マガジン用の記事（Markdown）を自動生成する。

Modes:
  daily     : 日次詳細レポート（有料コンテンツ）
  weekly    : 週間まとめ（有料コンテンツ）
  monthly   : 月間まとめ（有料コンテンツ）
  quarterly : 四半期まとめ（有料コンテンツ）
  yearly    : 年間まとめ（有料コンテンツ）

Output:
  1. Markdown → data/note_articles/YYYY-MM-DD_{mode}.md
  2. クリップボード → pbcopy
  3. note.com エディタ → 自動オープン
  4. macOS通知 → 記事準備完了

Usage:
  python note_article_generator.py --mode daily
  python note_article_generator.py --mode weekly
  python note_article_generator.py --mode daily --date 2026-02-17
  python note_article_generator.py --mode daily --no-open  # ブラウザ開かない
  python note_article_generator.py --mode monthly --year 2026 --month 1
  python note_article_generator.py --mode quarterly --year 2025 --quarter 4
  python note_article_generator.py --mode yearly --year 2025
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from shield.edinet_client import EDINETClient

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "x_bot_jp"
DB_PATH = DATA_DIR / "tracker.db"
ARTICLE_DIR = PROJECT_ROOT / "data" / "note_articles"
CHART_DIR = DATA_DIR / "charts"
NOTE_PROFILE = "https://note.com/quant_gunshi"
NOTE_NEW_URL = "https://note.com/notes/new"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("note_gen")

# Institution abbreviations (duplicated from x_bot_jp for standalone use)
SS_ABBREV = {
    "Goldman Sachs": "GS", "GOLDMAN SACHS": "GS",
    "Morgan Stanley": "モルガンS", "モルガン・スタンレー": "モルガンS",
    "JPMorgan": "JPモルガン",
    "Merrill Lynch": "メリルリンチ", "MERRILL LYNCH": "メリルリンチ",
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
    "大和証券": "大和", "野村證券": "野村",
    "みずほ証券": "みずほ", "SMBC日興": "SMBC日興",
}


def shorten_ss(name: str) -> str:
    if not name:
        return "不明"
    for key, abbr in SS_ABBREV.items():
        if key.lower() in name.lower():
            return abbr
    short = name.split(",")[0].split("(")[0].strip()
    return short[:12] if len(short) > 12 else short


def shorten_co(name: str) -> str:
    if not name:
        return "?"
    for s in ["株式会社", "(株)", "ホールディングス", "HD", "グループ"]:
        name = name.replace(s, "")
    return name.strip()[:10]


# ===================================================================
# Database Reader
# ===================================================================
class DBReader:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def get_available_dates(self, limit: int = 30) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT disc_date FROM disclosures ORDER BY disc_date DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [r["disc_date"] for r in rows]

    def get_disclosures(self, date: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM disclosures WHERE disc_date=? ORDER BY ratio DESC", (date,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_disclosures_range(self, from_date: str, to_date: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM disclosures WHERE disc_date BETWEEN ? AND ? ORDER BY disc_date, ratio DESC",
            (from_date, to_date)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, date: str) -> Dict[str, int]:
        row = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN change_type='new' THEN 1 ELSE 0 END) as new_cnt, "
            "SUM(CASE WHEN change_type='increase' THEN 1 ELSE 0 END) as inc_cnt, "
            "SUM(CASE WHEN change_type='decrease' THEN 1 ELSE 0 END) as dec_cnt, "
            "SUM(CASE WHEN change_type='unchanged' THEN 1 ELSE 0 END) as unch_cnt "
            "FROM disclosures WHERE disc_date=?", (date,)
        ).fetchone()
        return dict(row) if row else {}

    def get_top_institutions(self, date: str, limit: int = 15) -> List[Tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT ss_name, COUNT(*) as cnt FROM disclosures WHERE disc_date=? "
            "GROUP BY ss_name ORDER BY cnt DESC", (date,)
        ).fetchall()
        # Merge by abbreviated name (e.g. Nomura International + 野村證券 → 野村)
        merged: Dict[str, int] = {}
        for name, cnt in rows:
            key = shorten_ss(name)
            merged[key] = merged.get(key, 0) + cnt
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:limit]
        return ranked

    def get_concentrated(self, date: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT code, company_name, COUNT(DISTINCT ss_name) as n_inst, "
            "ROUND(SUM(ratio), 3) as total_ratio, "
            "GROUP_CONCAT(ss_name, '|') as institutions "
            "FROM disclosures WHERE disc_date=? "
            "GROUP BY code HAVING n_inst >= 2 "
            "ORDER BY total_ratio DESC", (date,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_top_stocks(self, date: str, limit: int = 20) -> List[dict]:
        rows = self.conn.execute(
            "SELECT code, company_name, ROUND(SUM(ratio), 3) as total_ratio, "
            "COUNT(DISTINCT ss_name) as n_inst "
            "FROM disclosures WHERE disc_date=? "
            "GROUP BY code ORDER BY total_ratio DESC LIMIT ?", (date, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_new_positions(self, date: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM disclosures WHERE disc_date=? AND change_type='new' "
            "ORDER BY ratio DESC", (date,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_big_changes(self, date: str, min_delta: float = 0.1) -> List[dict]:
        rows = self.conn.execute(
            "SELECT *, ROUND(ratio - prev_ratio, 3) as delta "
            "FROM disclosures WHERE disc_date=? AND prev_ratio IS NOT NULL "
            "AND ABS(ratio - prev_ratio) >= ? "
            "ORDER BY ABS(ratio - prev_ratio) DESC", (date, min_delta)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_prev_date_stats(self, date: str) -> Optional[Dict[str, int]]:
        dates = self.get_available_dates(30)
        for d in dates:
            if d < date:
                return self.get_stats(d)
        return None

    def get_weekly_data(self, end_date: str) -> Dict[str, Any]:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (dt - timedelta(days=6)).strftime("%Y-%m-%d")
        rows = self.get_disclosures_range(start_date, end_date)

        # Merge institutions by abbreviated name
        ss_raw = Counter(r["ss_name"] for r in rows)
        ss_merged: Dict[str, int] = {}
        for name, cnt in ss_raw.items():
            key = shorten_ss(name)
            ss_merged[key] = ss_merged.get(key, 0) + cnt
        top_inst = sorted(ss_merged.items(), key=lambda x: x[1], reverse=True)[:15]

        change_counts = Counter(r["change_type"] for r in rows)

        # Daily breakdown
        daily_counts = defaultdict(int)
        for r in rows:
            daily_counts[r["disc_date"]] += 1

        # Most targeted stocks
        stock_counts = Counter(r["code"] for r in rows)
        code_to_name = {}
        for r in rows:
            code_to_name[r["code"]] = r.get("company_name", r["code"])

        return {
            "start_date": start_date,
            "end_date": end_date,
            "total": len(rows),
            "top_institutions": top_inst,
            "change_counts": dict(change_counts),
            "daily_counts": dict(sorted(daily_counts.items())),
            "top_stocks": [(code, cnt, code_to_name.get(code, code))
                           for code, cnt in stock_counts.most_common(15)],
        }

    def get_bulk_holdings(self, date: str) -> List[dict]:
        """Get EDINET bulk holding reports for a specific date."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM edinet_bulk_holdings WHERE submit_date = ? "
                "ORDER BY holding_ratio DESC", (date,)
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []  # Table may not exist yet

    def get_bulk_holdings_range(self, from_date: str, to_date: str) -> List[dict]:
        """Get EDINET bulk holding reports for a date range."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM edinet_bulk_holdings "
                "WHERE submit_date BETWEEN ? AND ? "
                "ORDER BY submit_date DESC, holding_ratio DESC",
                (from_date, to_date)
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def get_verification_stats(self, from_date: str, to_date: str) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT * FROM verification WHERE signal_date BETWEEN ? AND ?",
            (from_date, to_date)
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

    def get_pnl_summary(self, from_date: str, to_date: str) -> Dict[str, Any]:
        """Get signal P&L summary for a period."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM signal_portfolio WHERE signal_date BETWEEN ? AND ? "
                "AND entry_price IS NOT NULL", (from_date, to_date)
            ).fetchall()
        except sqlite3.OperationalError:
            return {"count": 0}
        if not rows:
            return {"count": 0}

        type_labels = {"concentrated": "集中砲火", "new_position": "新規参入", "big_increase": "大幅増加"}
        by_type: Dict[str, list] = {k: [] for k in type_labels}
        for r in rows:
            st = r["signal_type"]
            if st in by_type:
                by_type[st].append(dict(r))

        result = {"count": len(rows), "by_type": {}}
        total_pnl = 0.0
        total_wins = 0
        total_losses = 0

        for stype, signals in by_type.items():
            if not signals:
                continue
            pnls = []
            for s in signals:
                p = s.get("pnl_5d") if s.get("status") != "open" else s.get("pnl_current")
                if p is not None:
                    pnls.append(p)
            if not pnls:
                continue
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p <= 0)
            cum_pnl = sum(pnls)
            total_pnl += cum_pnl
            total_wins += wins
            total_losses += losses
            result["by_type"][stype] = {
                "label": type_labels[stype],
                "count": len(signals),
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / len(pnls) * 100, 1) if pnls else 0,
                "cumulative_pnl": round(cum_pnl),
            }

        result["total_pnl"] = round(total_pnl)
        result["total_wins"] = total_wins
        result["total_losses"] = total_losses
        result["total_win_rate"] = round(total_wins / (total_wins + total_losses) * 100, 1) if (total_wins + total_losses) > 0 else 0
        return result

    def get_open_positions(self, limit: int = 10) -> List[dict]:
        """Get currently open signal positions."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM signal_portfolio WHERE status='open' AND entry_price IS NOT NULL "
                "ORDER BY pnl_current DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def get_closed_positions(self, from_date: str, to_date: str, limit: int = 10) -> List[dict]:
        """Get recently closed positions."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM signal_portfolio WHERE status='closed' "
                "AND signal_date BETWEEN ? AND ? AND entry_price IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT ?", (from_date, to_date, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def get_period_data(self, from_date: str, to_date: str) -> Dict[str, Any]:
        """Aggregate disclosure data for any period (month/quarter/year)."""
        rows = self.get_disclosures_range(from_date, to_date)
        if not rows:
            return {"total": 0, "from_date": from_date, "to_date": to_date}

        # Institution ranking (merged by abbreviation)
        ss_raw = Counter(r["ss_name"] for r in rows)
        ss_merged: Dict[str, int] = {}
        for name, cnt in ss_raw.items():
            key = shorten_ss(name)
            ss_merged[key] = ss_merged.get(key, 0) + cnt
        top_inst = sorted(ss_merged.items(), key=lambda x: x[1], reverse=True)[:20]

        # Change type counts
        change_counts = Counter(r["change_type"] for r in rows)

        # Daily breakdown
        daily_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            daily_counts[r["disc_date"]] += 1

        # Weekly breakdown
        weekly_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            dt = datetime.strptime(r["disc_date"], "%Y-%m-%d")
            week_start = (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")
            weekly_counts[week_start] += 1

        # Monthly breakdown
        monthly_counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            monthly_counts[r["disc_date"][:7]] += 1

        # Stock appearances
        stock_appearances = Counter(r["code"] for r in rows)
        code_to_name: Dict[str, str] = {}
        for r in rows:
            code_to_name[r["code"]] = r.get("company_name", r["code"])

        # Concentrated stocks: count days with 2+ institutions per stock
        date_code_insts: Dict[tuple, set] = defaultdict(set)
        for r in rows:
            date_code_insts[(r["disc_date"], r["code"])].add(r["ss_name"])
        concentrated_days: Dict[str, int] = defaultdict(int)
        for (d, code), insts in date_code_insts.items():
            if len(insts) >= 2:
                concentrated_days[code] += 1
        concentrated_stocks = sorted(concentrated_days.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "from_date": from_date,
            "to_date": to_date,
            "total": len(rows),
            "trading_days": len(daily_counts),
            "top_institutions": top_inst,
            "change_counts": dict(change_counts),
            "daily_counts": dict(sorted(daily_counts.items())),
            "weekly_counts": dict(sorted(weekly_counts.items())),
            "monthly_counts": dict(sorted(monthly_counts.items())),
            "top_stocks": [(code, cnt, code_to_name.get(code, code))
                           for code, cnt in stock_appearances.most_common(20)],
            "concentrated_stocks": [(code, days, code_to_name.get(code, code))
                                    for code, days in concentrated_stocks],
            "new_count": change_counts.get("new", 0),
            "increase_count": change_counts.get("increase", 0),
            "decrease_count": change_counts.get("decrease", 0),
        }


# ===================================================================
# Article Generators
# ===================================================================
def generate_daily_article(db: DBReader, date: str) -> str:
    """Generate premium daily article for note subscription magazine."""
    stats = db.get_stats(date)
    if not stats or stats.get("total", 0) == 0:
        return ""

    total = stats["total"]
    new_cnt = stats["new_cnt"]
    inc_cnt = stats["inc_cnt"]
    dec_cnt = stats["dec_cnt"]

    # Sentiment calculation
    aggression = (new_cnt + inc_cnt) / total * 100 if total > 0 else 50
    if aggression > 55:
        sentiment_label = "弱気（ベア）"
        sentiment_icon = "🐻"
    elif aggression < 40:
        sentiment_label = "強気（ブル）"
        sentiment_icon = "🐂"
    else:
        sentiment_label = "中立"
        sentiment_icon = "⚖️"

    # Previous day comparison
    prev_stats = db.get_prev_date_stats(date)
    if prev_stats and prev_stats.get("total", 0) > 0:
        delta_total = total - prev_stats["total"]
        delta_str = f"（前日比 {'+' if delta_total >= 0 else ''}{delta_total}件）"
    else:
        delta_str = ""

    top_inst = db.get_top_institutions(date)
    concentrated = db.get_concentrated(date)
    new_positions = db.get_new_positions(date)
    big_changes = db.get_big_changes(date)
    top_stocks = db.get_top_stocks(date)

    # Format date for display
    dt = datetime.strptime(date, "%Y-%m-%d")
    date_disp = f"{dt.year}年{dt.month}月{dt.day}日（{['月','火','水','木','金','土','日'][dt.weekday()]}）"

    lines = []

    # Title
    lines.append(f"# 機関空売り日報｜{dt.month}/{dt.day}")
    lines.append("")

    # Summary box
    lines.append(f"> **{date_disp}の機関投資家による空売りポジション開示を全件分析しました。**")
    lines.append("")

    # Overview
    lines.append("## 本日のサマリー")
    lines.append("")
    lines.append(f"- **開示件数**: {total}件{delta_str}")
    lines.append(f"- **新規ポジション**: {new_cnt}件")
    lines.append(f"- **増加**: {inc_cnt}件")
    lines.append(f"- **減少**: {dec_cnt}件")
    lines.append(f"- **センチメント**: {sentiment_icon} {sentiment_label}（攻勢度 {aggression:.0f}%）")
    lines.append("")

    # Institution ranking
    lines.append("## 機関別 空売り活動ランキング")
    lines.append("")
    lines.append("| 順位 | 機関名 | 件数 |")
    lines.append("|:---:|:---|:---:|")
    for i, (name, cnt) in enumerate(top_inst, 1):
        lines.append(f"| {i} | {name} | {cnt} |")
    lines.append("")

    if top_inst:
        lines.append(f"本日最も活発だったのは**{top_inst[0][0]}**（{top_inst[0][1]}件）。")
        if len(top_inst) >= 3:
            lines.append(f"続いて**{top_inst[1][0]}**（{top_inst[1][1]}件）、"
                         f"**{top_inst[2][0]}**（{top_inst[2][1]}件）。")
        lines.append("")

    # Concentrated shorts (premium content)
    if concentrated:
        lines.append("## 空売り集中銘柄")
        lines.append("")
        lines.append("**複数の機関投資家が同時にポジションを持つ銘柄**は、"
                      "機関の間で「下落する」というコンセンサスが形成されている可能性があり、要注目です。")
        lines.append("")
        lines.append("| 銘柄 | コード | 機関数 | 合計比率 | 参加機関 |")
        lines.append("|:---|:---:|:---:|:---:|:---|")
        for c in concentrated[:15]:
            co = shorten_co(c["company_name"])
            code4 = c["code"][:4] if len(c["code"]) >= 4 else c["code"]
            inst_list = c["institutions"].split("|") if c["institutions"] else []
            inst_str = "、".join(shorten_ss(i) for i in inst_list[:4])
            if len(inst_list) > 4:
                inst_str += f" 他{len(inst_list)-4}社"
            lines.append(f"| {co} | {code4} | {c['n_inst']} | {c['total_ratio']:.2f}% | {inst_str} |")
        lines.append("")

    # New positions (premium content)
    if new_positions:
        lines.append("## 新規ポジション一覧")
        lines.append("")
        lines.append("本日新たに空売りポジションが報告された銘柄です。新規参入は機関の新たな「売り」判断を意味します。")
        lines.append("")
        lines.append("| 機関 | 銘柄 | コード | 比率 |")
        lines.append("|:---|:---|:---:|:---:|")
        for p in new_positions[:20]:
            ss = shorten_ss(p["ss_name"])
            co = shorten_co(p["company_name"])
            code4 = p["code"][:4] if len(p["code"]) >= 4 else p["code"]
            lines.append(f"| {ss} | {co} | {code4} | {p['ratio']:.2f}% |")
        if len(new_positions) > 20:
            lines.append(f"")
            lines.append(f"*他 {len(new_positions) - 20}件*")
        lines.append("")

    # Big changes (premium content)
    if big_changes:
        lines.append("## 大幅変動ポジション")
        lines.append("")
        lines.append("前回比で大きく変動したポジション（±0.10%以上）です。")
        lines.append("")
        lines.append("| 機関 | 銘柄 | コード | 前回 → 今回 | 変動 |")
        lines.append("|:---|:---|:---:|:---:|:---:|")
        for c in big_changes[:15]:
            ss = shorten_ss(c["ss_name"])
            co = shorten_co(c["company_name"])
            code4 = c["code"][:4] if len(c["code"]) >= 4 else c["code"]
            delta = c["delta"]
            arrow = "🔺" if delta > 0 else "🔽"
            lines.append(f"| {ss} | {co} | {code4} | {c['prev_ratio']:.2f}% → {c['ratio']:.2f}% | {arrow}{abs(delta):.2f}% |")
        lines.append("")

    # Top shorted stocks
    if top_stocks:
        lines.append("## 空売り比率 上位銘柄")
        lines.append("")
        lines.append("機関全体の合計空売り比率が高い銘柄です。")
        lines.append("")
        lines.append("| 順位 | 銘柄 | コード | 合計比率 | 機関数 |")
        lines.append("|:---:|:---|:---:|:---:|:---:|")
        for i, s in enumerate(top_stocks[:15], 1):
            co = shorten_co(s["company_name"])
            code4 = s["code"][:4] if len(s["code"]) >= 4 else s["code"]
            lines.append(f"| {i} | {co} | {code4} | {s['total_ratio']:.2f}% | {s['n_inst']} |")
        lines.append("")

    # EDINET Bulk Holdings Section
    bulk = db.get_bulk_holdings(date)
    if bulk:
        lines.append("## EDINET 大量保有報告書")
        lines.append("")
        lines.append("金融庁EDINETに提出された大量保有報告書（5%ルール）です。"
                      "機関投資家の**株式保有比率の変動**がわかります。")
        lines.append("")

        new_bh = [h for h in bulk if h.get("report_type") == "new"]
        chg_bh = [h for h in bulk if h.get("report_type") == "change"]

        if new_bh:
            lines.append("### 新規大量保有")
            lines.append("")
            lines.append("| 報告者 | 対象企業 | コード | 保有比率 | 目的 |")
            lines.append("|:---|:---|:---:|:---:|:---|")
            for h in new_bh[:10]:
                filer = h.get("filer_name", "")[:15]
                issuer = shorten_co(h.get("issuer_name", ""))
                code4 = (h.get("sec_code") or "")[:4]
                ratio = h.get("holding_ratio")
                r_str = f"{ratio:.2f}%" if ratio else "―"
                purpose = h.get("purpose") or "―"
                lines.append(f"| {filer} | {issuer} | {code4} | {r_str} | {purpose} |")
            lines.append("")

        if chg_bh:
            lines.append("### 保有比率変更")
            lines.append("")
            lines.append("| 報告者 | 対象企業 | コード | 前回 → 今回 | 増減 |")
            lines.append("|:---|:---|:---:|:---:|:---:|")
            for h in chg_bh[:10]:
                filer = h.get("filer_name", "")[:15]
                issuer = shorten_co(h.get("issuer_name", ""))
                code4 = (h.get("sec_code") or "")[:4]
                ratio = h.get("holding_ratio")
                prev = h.get("prev_ratio")
                if ratio is not None and prev is not None:
                    arrow = "🔺" if ratio > prev else "🔽"
                    lines.append(f"| {filer} | {issuer} | {code4} | {prev:.2f}% → {ratio:.2f}% | {arrow}{abs(ratio-prev):.2f}% |")
                elif ratio is not None:
                    lines.append(f"| {filer} | {issuer} | {code4} | ― → {ratio:.2f}% | ― |")
            lines.append("")

        lines.append(f"*出典: 金融庁 EDINET（{len(bulk)}件）*")
        lines.append("")

    # Signal P&L section
    dt_30ago = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
    pnl = db.get_pnl_summary(dt_30ago, date)
    if pnl["count"] > 0:
        lines.append("## シグナル仮想損益")
        lines.append("")
        lines.append("当マガジンのシグナル（集中砲火・新規参入・大幅増加）を"
                      "仮想空売り（1シグナル100万円）した場合の損益です。")
        lines.append("")
        lines.append("| シグナル種別 | 勝敗 | 勝率 | 累計損益 |")
        lines.append("|:---|:---:|:---:|---:|")
        for stype_data in pnl["by_type"].values():
            pnl_val = stype_data["cumulative_pnl"]
            pnl_str = f"+{pnl_val:,}円" if pnl_val >= 0 else f"{pnl_val:,}円"
            lines.append(f"| {stype_data['label']} | "
                         f"{stype_data['wins']}勝{stype_data['losses']}敗 | "
                         f"{stype_data['win_rate']:.0f}% | {pnl_str} |")
        pnl_total = pnl["total_pnl"]
        pnl_total_str = f"+{pnl_total:,}円" if pnl_total >= 0 else f"{pnl_total:,}円"
        lines.append(f"| **合計** | **{pnl['total_wins']}勝{pnl['total_losses']}敗** | "
                     f"**{pnl['total_win_rate']:.0f}%** | **{pnl_total_str}** |")
        lines.append("")

        # Open positions
        open_pos = db.get_open_positions(5)
        if open_pos:
            lines.append("### 保有中ポジション")
            lines.append("")
            lines.append("| 銘柄 | コード | シグナル日 | エントリー | 現在値 | 含み損益 |")
            lines.append("|:---|:---:|:---:|:---:|:---:|---:|")
            for p in open_pos:
                co = shorten_co(p.get("company_name", ""))
                code4 = p["code"][:4] if len(p["code"]) >= 4 else p["code"]
                sig_dt = datetime.strptime(p["signal_date"], "%Y-%m-%d")
                entry = f"¥{p['entry_price']:,.0f}" if p.get("entry_price") else "―"
                curr = f"¥{p['current_price']:,.0f}" if p.get("current_price") else "―"
                pnl_c = p.get("pnl_current", 0)
                pnl_s = f"+{pnl_c:,}" if pnl_c >= 0 else f"{pnl_c:,}"
                lines.append(f"| {co} | {code4} | {sig_dt.month}/{sig_dt.day} | {entry} | {curr} | {pnl_s}円 |")
            lines.append("")

        lines.append("*※仮想損益です。貸株料・手数料は含みません。投資助言ではありません。*")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"**データソース**: J-Quants API（空売り）+ 金融庁 EDINET（大量保有）")
    lines.append(f"**分析**: クオンツ軍師（Asset Shield Project）")
    lines.append(f"**X（旧Twitter）**: 無料速報を毎朝配信中")
    lines.append("")
    lines.append("#機関空売り #大量保有 #日本株 #空売り分析 #機関投資家")

    return "\n".join(lines)


def generate_weekly_article(db: DBReader, end_date: str) -> str:
    """Generate premium weekly article for note subscription magazine."""
    data = db.get_weekly_data(end_date)
    if data["total"] == 0:
        return ""

    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    dt_start = datetime.strptime(data["start_date"], "%Y-%m-%d")

    lines = []

    lines.append(f"# 週間 機関空売りレポート｜{dt_start.month}/{dt_start.day}〜{dt_end.month}/{dt_end.day}")
    lines.append("")
    lines.append(f"> **{dt_start.month}/{dt_start.day}（{['月','火','水','木','金','土','日'][dt_start.weekday()]}）"
                 f"〜{dt_end.month}/{dt_end.day}（{['月','火','水','木','金','土','日'][dt_end.weekday()]}）の"
                 f"機関空売り動向をまとめました。**")
    lines.append("")

    # Weekly overview
    cc = data["change_counts"]
    lines.append("## 週間サマリー")
    lines.append("")
    lines.append(f"- **総開示件数**: {data['total']}件")
    lines.append(f"- **新規ポジション**: {cc.get('new', 0)}件")
    lines.append(f"- **増加**: {cc.get('increase', 0)}件")
    lines.append(f"- **減少**: {cc.get('decrease', 0)}件")
    lines.append("")

    # Daily breakdown
    lines.append("## 日別推移")
    lines.append("")
    lines.append("| 日付 | 開示件数 |")
    lines.append("|:---:|:---:|")
    for d, cnt in data["daily_counts"].items():
        dt_d = datetime.strptime(d, "%Y-%m-%d")
        lines.append(f"| {dt_d.month}/{dt_d.day}（{['月','火','水','木','金','土','日'][dt_d.weekday()]}） | {cnt} |")
    lines.append("")

    # Weekly institution ranking
    lines.append("## 週間 機関別活動ランキング")
    lines.append("")
    lines.append("| 順位 | 機関名 | 件数 |")
    lines.append("|:---:|:---|:---:|")
    for i, (name, cnt) in enumerate(data["top_institutions"], 1):
        lines.append(f"| {i} | {shorten_ss(name)} | {cnt} |")
    lines.append("")

    # Most targeted stocks
    lines.append("## 最も売られた銘柄 TOP15")
    lines.append("")
    lines.append("| 順位 | 銘柄 | コード | 開示回数 |")
    lines.append("|:---:|:---|:---:|:---:|")
    for i, (code, cnt, name) in enumerate(data["top_stocks"], 1):
        co = shorten_co(name)
        code4 = code[:4] if len(code) >= 4 else code
        lines.append(f"| {i} | {co} | {code4} | {cnt} |")
    lines.append("")

    # Verification stats (if available)
    verif = db.get_verification_stats(data["start_date"], end_date)
    if verif["count"] > 0:
        lines.append("## 過去シグナル検証")
        lines.append("")
        lines.append(f"集中銘柄シグナルの追跡結果（直近30日間の{verif['count']}件）：")
        lines.append("")
        lines.append(f"- **5日的中率**: {verif['hit_rate_5d']}%（平均リターン {verif['avg_ret_5d']:+.1f}%）")
        lines.append(f"- **20日的中率**: {verif['hit_rate_20d']}%（平均リターン {verif['avg_ret_20d']:+.1f}%）")
        lines.append("")

    # Signal P&L section
    pnl = db.get_pnl_summary(data["start_date"], end_date)
    if pnl["count"] > 0:
        lines.append("## シグナル仮想損益（週間）")
        lines.append("")
        lines.append("今週のシグナルに基づく仮想空売り成績（1シグナル100万円）：")
        lines.append("")
        lines.append("| シグナル種別 | 勝敗 | 勝率 | 累計損益 |")
        lines.append("|:---|:---:|:---:|---:|")
        for stype_data in pnl["by_type"].values():
            pnl_val = stype_data["cumulative_pnl"]
            pnl_str = f"+{pnl_val:,}円" if pnl_val >= 0 else f"{pnl_val:,}円"
            lines.append(f"| {stype_data['label']} | "
                         f"{stype_data['wins']}勝{stype_data['losses']}敗 | "
                         f"{stype_data['win_rate']:.0f}% | {pnl_str} |")
        pnl_total = pnl["total_pnl"]
        pnl_total_str = f"+{pnl_total:,}円" if pnl_total >= 0 else f"{pnl_total:,}円"
        lines.append(f"| **合計** | **{pnl['total_wins']}勝{pnl['total_losses']}敗** | "
                     f"**{pnl['total_win_rate']:.0f}%** | **{pnl_total_str}** |")
        lines.append("")
        lines.append("*※仮想損益です。貸株料・手数料は含みません。*")
        lines.append("")

    # Trend analysis
    aggression = (cc.get("new", 0) + cc.get("increase", 0)) / data["total"] * 100 if data["total"] > 0 else 50
    if aggression > 55:
        trend_text = "機関投資家は全体的に**弱気姿勢**を強めており、新規・増加ポジションが多い週でした。"
    elif aggression < 40:
        trend_text = "機関投資家は**ポジション縮小モード**に入っており、減少が優勢な週でした。相場の底打ちサインの可能性があります。"
    else:
        trend_text = "機関投資家の動きは**拮抗**しており、明確な方向感は見られませんでした。"
    lines.append("## 総括")
    lines.append("")
    lines.append(trend_text)
    lines.append("")

    # EDINET Bulk Holdings (weekly)
    bulk = db.get_bulk_holdings_range(data["start_date"], end_date)
    if bulk:
        lines.append("## EDINET 大量保有報告書（週間）")
        lines.append("")
        lines.append(f"今週EDINETに提出された大量保有報告書は**{len(bulk)}件**。")
        lines.append("")

        new_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "new")
        chg_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "change")
        lines.append(f"- **新規大量保有**: {new_cnt_bh}件")
        lines.append(f"- **保有比率変更**: {chg_cnt_bh}件")
        lines.append("")

        # Top filers
        filer_counts = Counter(h.get("filer_name", "不明") for h in bulk)
        if filer_counts:
            lines.append("### 週間 大量保有 報告者ランキング")
            lines.append("")
            lines.append("| 順位 | 報告者 | 件数 |")
            lines.append("|:---:|:---|:---:|")
            for i, (filer, cnt) in enumerate(filer_counts.most_common(10), 1):
                lines.append(f"| {i} | {filer[:15]} | {cnt} |")
            lines.append("")

        # Biggest holdings
        high_ratio = sorted(
            [h for h in bulk if h.get("holding_ratio")],
            key=lambda x: x["holding_ratio"], reverse=True
        )[:5]
        if high_ratio:
            lines.append("### 高保有比率 TOP5")
            lines.append("")
            lines.append("| 報告者 | 対象 | コード | 比率 |")
            lines.append("|:---|:---|:---:|:---:|")
            for h in high_ratio:
                filer = h.get("filer_name", "")[:15]
                issuer = shorten_co(h.get("issuer_name", ""))
                code4 = (h.get("sec_code") or "")[:4]
                lines.append(f"| {filer} | {issuer} | {code4} | {h['holding_ratio']:.2f}% |")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"**データソース**: J-Quants API（空売り）+ 金融庁 EDINET（大量保有）")
    lines.append(f"**分析**: クオンツ軍師（Asset Shield Project）")
    lines.append("")
    lines.append("#機関空売り #大量保有 #日本株 #週間レポート #機関投資家")

    return "\n".join(lines)


def generate_monthly_article(db: DBReader, year: int, month: int) -> str:
    """Generate monthly summary article for note subscription magazine."""
    _, last_day = calendar.monthrange(year, month)
    from_date = f"{year}-{month:02d}-01"
    to_date = f"{year}-{month:02d}-{last_day:02d}"
    data = db.get_period_data(from_date, to_date)

    if data["total"] == 0:
        return ""

    total = data["total"]
    trading_days = data["trading_days"]
    daily_avg = total / trading_days if trading_days > 0 else 0

    lines = []

    # Title
    lines.append(f"# 月間 機関空売りレポート｜{year}年{month}月")
    lines.append("")
    lines.append(f"> **{year}年{month}月の機関投資家による空売りポジション開示を"
                 f"全件分析しました。月間{total}件の開示データを集計・ランキング化しています。**")
    lines.append("")

    # Monthly summary
    lines.append("## 月間サマリー")
    lines.append("")
    lines.append(f"- **総開示件数**: {total}件")
    lines.append(f"- **新規ポジション**: {data['new_count']}件")
    lines.append(f"- **増加**: {data['increase_count']}件")
    lines.append(f"- **減少**: {data['decrease_count']}件")
    lines.append(f"- **営業日数**: {trading_days}日")
    lines.append(f"- **1日平均**: {daily_avg:.1f}件")
    lines.append("")

    # Weekly breakdown
    if data["weekly_counts"]:
        lines.append("## 週別推移")
        lines.append("")
        lines.append("| 週（月曜始まり） | 開示件数 |")
        lines.append("|:---|:---:|")
        for week_start, cnt in data["weekly_counts"].items():
            dt_w = datetime.strptime(week_start, "%Y-%m-%d")
            lines.append(f"| {dt_w.month}/{dt_w.day}〜 | {cnt} |")
        lines.append("")

    # Institution ranking TOP15
    lines.append("## 機関別 月間活動ランキング TOP15")
    lines.append("")
    lines.append("| 順位 | 機関名 | 件数 |")
    lines.append("|:---:|:---|:---:|")
    for i, (name, cnt) in enumerate(data["top_institutions"][:15], 1):
        lines.append(f"| {i} | {name} | {cnt} |")
    lines.append("")

    if data["top_institutions"]:
        lines.append(f"月間最も活発だったのは**{data['top_institutions'][0][0]}**"
                     f"（{data['top_institutions'][0][1]}件）。")
        if len(data["top_institutions"]) >= 3:
            lines.append(f"続いて**{data['top_institutions'][1][0]}**"
                         f"（{data['top_institutions'][1][1]}件）、"
                         f"**{data['top_institutions'][2][0]}**"
                         f"（{data['top_institutions'][2][1]}件）。")
        lines.append("")

    # Concentrated stocks TOP15
    if data["concentrated_stocks"]:
        lines.append("## 空売り集中銘柄 月間TOP15")
        lines.append("")
        lines.append("**複数機関が同日にポジションを持った日数**が多い銘柄です。"
                      "機関のコンセンサスが継続的に形成されている銘柄ほど注目度が高いと言えます。")
        lines.append("")
        lines.append("| 順位 | 銘柄 | コード | 集中日数 |")
        lines.append("|:---:|:---|:---:|:---:|")
        for i, (code, days, name) in enumerate(data["concentrated_stocks"][:15], 1):
            co = shorten_co(name)
            code4 = code[:4] if len(code) >= 4 else code
            lines.append(f"| {i} | {co} | {code4} | {days}日 |")
        lines.append("")

    # Most targeted stocks TOP15
    if data["top_stocks"]:
        lines.append("## 最も狙われた銘柄 TOP15")
        lines.append("")
        lines.append("| 順位 | 銘柄 | コード | 開示回数 |")
        lines.append("|:---:|:---|:---:|:---:|")
        for i, (code, cnt, name) in enumerate(data["top_stocks"][:15], 1):
            co = shorten_co(name)
            code4 = code[:4] if len(code) >= 4 else code
            lines.append(f"| {i} | {co} | {code4} | {cnt} |")
        lines.append("")

    # EDINET bulk holdings monthly
    bulk = db.get_bulk_holdings_range(from_date, to_date)
    if bulk:
        lines.append("## EDINET 大量保有 月間まとめ")
        lines.append("")
        new_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "new")
        chg_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "change")
        lines.append(f"月間でEDINETに提出された大量保有報告書は**{len(bulk)}件**。")
        lines.append("")
        lines.append(f"- **新規大量保有**: {new_cnt_bh}件")
        lines.append(f"- **保有比率変更**: {chg_cnt_bh}件")
        lines.append("")

        filer_counts = Counter(h.get("filer_name", "不明") for h in bulk)
        if filer_counts:
            lines.append("### 月間 大量保有 報告者ランキング")
            lines.append("")
            lines.append("| 順位 | 報告者 | 件数 |")
            lines.append("|:---:|:---|:---:|")
            for i, (filer, cnt) in enumerate(filer_counts.most_common(10), 1):
                lines.append(f"| {i} | {filer[:15]} | {cnt} |")
            lines.append("")

        high_ratio = sorted(
            [h for h in bulk if h.get("holding_ratio")],
            key=lambda x: x["holding_ratio"], reverse=True
        )[:5]
        if high_ratio:
            lines.append("### 高保有比率 TOP5")
            lines.append("")
            lines.append("| 報告者 | 対象 | コード | 比率 |")
            lines.append("|:---|:---|:---:|:---:|")
            for h in high_ratio:
                filer = h.get("filer_name", "")[:15]
                issuer = shorten_co(h.get("issuer_name", ""))
                code4 = (h.get("sec_code") or "")[:4]
                lines.append(f"| {filer} | {issuer} | {code4} | {h['holding_ratio']:.2f}% |")
            lines.append("")

    # Verification stats
    try:
        verif = db.get_verification_stats(from_date, to_date)
        if verif["count"] > 0:
            lines.append("## シグナル検証")
            lines.append("")
            lines.append(f"月間の集中銘柄シグナル追跡結果（{verif['count']}件）：")
            lines.append("")
            lines.append(f"- **5日的中率**: {verif['hit_rate_5d']}%（平均リターン {verif['avg_ret_5d']:+.1f}%）")
            lines.append(f"- **20日的中率**: {verif['hit_rate_20d']}%（平均リターン {verif['avg_ret_20d']:+.1f}%）")
            lines.append("")
    except Exception:
        pass

    # Signal P&L section
    pnl = db.get_pnl_summary(from_date, to_date)
    if pnl["count"] > 0:
        lines.append("## シグナル仮想損益（月間）")
        lines.append("")
        lines.append(f"今月のシグナル仮想空売り成績（1シグナル100万円×{pnl['count']}件）：")
        lines.append("")
        lines.append("| シグナル種別 | 勝敗 | 勝率 | 累計損益 |")
        lines.append("|:---|:---:|:---:|---:|")
        for stype_data in pnl["by_type"].values():
            pnl_val = stype_data["cumulative_pnl"]
            pnl_str = f"+{pnl_val:,}円" if pnl_val >= 0 else f"{pnl_val:,}円"
            lines.append(f"| {stype_data['label']} | "
                         f"{stype_data['wins']}勝{stype_data['losses']}敗 | "
                         f"{stype_data['win_rate']:.0f}% | {pnl_str} |")
        pnl_total = pnl["total_pnl"]
        pnl_total_str = f"+{pnl_total:,}円" if pnl_total >= 0 else f"{pnl_total:,}円"
        lines.append(f"| **合計** | **{pnl['total_wins']}勝{pnl['total_losses']}敗** | "
                     f"**{pnl['total_win_rate']:.0f}%** | **{pnl_total_str}** |")
        lines.append("")
        lines.append("*※仮想損益です。貸株料・手数料は含みません。投資助言ではありません。*")
        lines.append("")

    # Monthly wrap-up
    aggression = (data["new_count"] + data["increase_count"]) / total * 100 if total > 0 else 50
    lines.append("## 月間総括")
    lines.append("")
    if aggression > 55:
        lines.append(f"今月の機関投資家は全体的に**弱気姿勢**を強めました。"
                     f"新規・増加が全体の{aggression:.0f}%を占め、攻勢的な売り圧力が続きました。")
    elif aggression < 40:
        lines.append(f"今月の機関投資家は**ポジション縮小モード**でした。"
                     f"減少が優勢（攻勢度{aggression:.0f}%）で、空売りの巻き戻し（ショートカバー）が"
                     f"相場を下支えした可能性があります。")
    else:
        lines.append(f"今月の機関投資家の動きは**拮抗**していました（攻勢度{aggression:.0f}%）。"
                     f"新規・増加と減少がほぼ均衡し、明確な方向感は見られませんでした。")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"**データソース**: J-Quants API（空売り）+ 金融庁 EDINET（大量保有）")
    lines.append(f"**分析**: クオンツ軍師（Asset Shield Project）")
    lines.append("")
    lines.append("#機関空売り #月間レポート #日本株 #機関投資家")

    return "\n".join(lines)


def generate_quarterly_article(db: DBReader, year: int, quarter: int) -> str:
    """Generate quarterly summary article for note subscription magazine."""
    quarter_months = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
    start_month, end_month = quarter_months[quarter]
    _, last_day = calendar.monthrange(year, end_month)
    from_date = f"{year}-{start_month:02d}-01"
    to_date = f"{year}-{end_month:02d}-{last_day:02d}"
    data = db.get_period_data(from_date, to_date)

    if data["total"] == 0:
        return ""

    total = data["total"]
    trading_days = data["trading_days"]
    daily_avg = total / trading_days if trading_days > 0 else 0

    lines = []

    # Title
    lines.append(f"# 四半期 機関空売りレポート｜{year}年Q{quarter}（{start_month}月〜{end_month}月）")
    lines.append("")
    lines.append(f"> **{year}年第{quarter}四半期（{start_month}月〜{end_month}月）の機関投資家による"
                 f"空売りポジション開示を全件分析しました。四半期{total}件のデータを集計しています。**")
    lines.append("")

    # Quarterly summary
    lines.append("## 四半期サマリー")
    lines.append("")
    lines.append(f"- **総開示件数**: {total}件")
    lines.append(f"- **新規ポジション**: {data['new_count']}件")
    lines.append(f"- **増加**: {data['increase_count']}件")
    lines.append(f"- **減少**: {data['decrease_count']}件")
    lines.append(f"- **営業日数**: {trading_days}日")
    lines.append(f"- **1日平均**: {daily_avg:.1f}件")
    lines.append("")

    # Monthly breakdown
    if data["monthly_counts"]:
        lines.append("## 月別推移")
        lines.append("")
        lines.append("| 月 | 開示件数 |")
        lines.append("|:---:|:---:|")
        for ym, cnt in data["monthly_counts"].items():
            m = int(ym.split("-")[1])
            lines.append(f"| {m}月 | {cnt} |")
        lines.append("")

    # Institution ranking TOP15
    lines.append("## 機関別 四半期活動ランキング TOP15")
    lines.append("")
    lines.append("| 順位 | 機関名 | 件数 |")
    lines.append("|:---:|:---|:---:|")
    for i, (name, cnt) in enumerate(data["top_institutions"][:15], 1):
        lines.append(f"| {i} | {name} | {cnt} |")
    lines.append("")

    if data["top_institutions"]:
        lines.append(f"四半期で最も活発だったのは**{data['top_institutions'][0][0]}**"
                     f"（{data['top_institutions'][0][1]}件）。")
        if len(data["top_institutions"]) >= 3:
            lines.append(f"続いて**{data['top_institutions'][1][0]}**"
                         f"（{data['top_institutions'][1][1]}件）、"
                         f"**{data['top_institutions'][2][0]}**"
                         f"（{data['top_institutions'][2][1]}件）。")
        lines.append("")

    # Concentrated stocks TOP15
    if data["concentrated_stocks"]:
        lines.append("## 四半期 空売り集中銘柄 TOP15")
        lines.append("")
        lines.append("**複数機関が同日にポジションを持った日数**が多い銘柄です。"
                      "四半期を通じて機関の注目が集中した銘柄をランキングしました。")
        lines.append("")
        lines.append("| 順位 | 銘柄 | コード | 集中日数 |")
        lines.append("|:---:|:---|:---:|:---:|")
        for i, (code, days, name) in enumerate(data["concentrated_stocks"][:15], 1):
            co = shorten_co(name)
            code4 = code[:4] if len(code) >= 4 else code
            lines.append(f"| {i} | {co} | {code4} | {days}日 |")
        lines.append("")

    # Most targeted stocks TOP15
    if data["top_stocks"]:
        lines.append("## 最も狙われた銘柄 TOP15")
        lines.append("")
        lines.append("| 順位 | 銘柄 | コード | 開示回数 |")
        lines.append("|:---:|:---|:---:|:---:|")
        for i, (code, cnt, name) in enumerate(data["top_stocks"][:15], 1):
            co = shorten_co(name)
            code4 = code[:4] if len(code) >= 4 else code
            lines.append(f"| {i} | {co} | {code4} | {cnt} |")
        lines.append("")

    # EDINET bulk holdings quarterly
    bulk = db.get_bulk_holdings_range(from_date, to_date)
    if bulk:
        lines.append("## EDINET 大量保有 四半期まとめ")
        lines.append("")
        new_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "new")
        chg_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "change")
        lines.append(f"四半期でEDINETに提出された大量保有報告書は**{len(bulk)}件**。")
        lines.append("")
        lines.append(f"- **新規大量保有**: {new_cnt_bh}件")
        lines.append(f"- **保有比率変更**: {chg_cnt_bh}件")
        lines.append("")

        filer_counts = Counter(h.get("filer_name", "不明") for h in bulk)
        if filer_counts:
            lines.append("### 四半期 大量保有 報告者ランキング")
            lines.append("")
            lines.append("| 順位 | 報告者 | 件数 |")
            lines.append("|:---:|:---|:---:|")
            for i, (filer, cnt) in enumerate(filer_counts.most_common(10), 1):
                lines.append(f"| {i} | {filer[:15]} | {cnt} |")
            lines.append("")

    # Verification stats
    try:
        verif = db.get_verification_stats(from_date, to_date)
        if verif["count"] > 0:
            lines.append("## シグナル検証")
            lines.append("")
            lines.append(f"四半期の集中銘柄シグナル追跡結果（{verif['count']}件）：")
            lines.append("")
            lines.append(f"- **5日的中率**: {verif['hit_rate_5d']}%（平均リターン {verif['avg_ret_5d']:+.1f}%）")
            lines.append(f"- **20日的中率**: {verif['hit_rate_20d']}%（平均リターン {verif['avg_ret_20d']:+.1f}%）")
            lines.append("")
    except Exception:
        pass

    # Signal P&L section
    pnl = db.get_pnl_summary(from_date, to_date)
    if pnl["count"] > 0:
        lines.append("## シグナル仮想損益（四半期）")
        lines.append("")
        lines.append(f"四半期のシグナル仮想空売り成績（1シグナル100万円×{pnl['count']}件）：")
        lines.append("")
        lines.append("| シグナル種別 | 勝敗 | 勝率 | 累計損益 |")
        lines.append("|:---|:---:|:---:|---:|")
        for stype_data in pnl["by_type"].values():
            pnl_val = stype_data["cumulative_pnl"]
            pnl_str = f"+{pnl_val:,}円" if pnl_val >= 0 else f"{pnl_val:,}円"
            lines.append(f"| {stype_data['label']} | "
                         f"{stype_data['wins']}勝{stype_data['losses']}敗 | "
                         f"{stype_data['win_rate']:.0f}% | {pnl_str} |")
        pnl_total = pnl["total_pnl"]
        pnl_total_str = f"+{pnl_total:,}円" if pnl_total >= 0 else f"{pnl_total:,}円"
        lines.append(f"| **合計** | **{pnl['total_wins']}勝{pnl['total_losses']}敗** | "
                     f"**{pnl['total_win_rate']:.0f}%** | **{pnl_total_str}** |")
        lines.append("")
        lines.append("*※仮想損益です。貸株料・手数料は含みません。投資助言ではありません。*")
        lines.append("")

    # Quarterly trend analysis
    lines.append("## 四半期トレンド分析")
    lines.append("")
    if data["monthly_counts"]:
        monthly_agg_pcts = []
        for ym in sorted(data["monthly_counts"].keys()):
            m = int(ym.split("-")[1])
            # Get per-month aggression
            m_from = f"{year}-{m:02d}-01"
            _, m_last = calendar.monthrange(year, m)
            m_to = f"{year}-{m:02d}-{m_last:02d}"
            m_data = db.get_period_data(m_from, m_to)
            if m_data["total"] > 0:
                m_agg = (m_data["new_count"] + m_data["increase_count"]) / m_data["total"] * 100
            else:
                m_agg = 50.0
            monthly_agg_pcts.append((m, m_agg, m_data["total"]))

        lines.append("| 月 | 開示件数 | 攻勢度 | 判定 |")
        lines.append("|:---:|:---:|:---:|:---|")
        for m, agg, cnt in monthly_agg_pcts:
            if agg > 55:
                label = "弱気"
            elif agg < 40:
                label = "強気"
            else:
                label = "中立"
            lines.append(f"| {m}月 | {cnt} | {agg:.0f}% | {label} |")
        lines.append("")

        # Trend direction
        if len(monthly_agg_pcts) >= 2:
            first_agg = monthly_agg_pcts[0][1]
            last_agg = monthly_agg_pcts[-1][1]
            diff = last_agg - first_agg
            if diff > 5:
                lines.append(f"四半期を通じて攻勢度が上昇（{first_agg:.0f}% → {last_agg:.0f}%）しており、"
                             f"機関投資家の**弱気姿勢が強まるトレンド**が見られました。")
            elif diff < -5:
                lines.append(f"四半期を通じて攻勢度が低下（{first_agg:.0f}% → {last_agg:.0f}%）しており、"
                             f"機関投資家の**売り圧力が弱まるトレンド**が見られました。")
            else:
                lines.append(f"四半期を通じて攻勢度は安定（{first_agg:.0f}% → {last_agg:.0f}%）しており、"
                             f"機関投資家のスタンスに**大きな変化は見られませんでした**。")
            lines.append("")
    else:
        lines.append("データが不足しているため、トレンド分析は行えませんでした。")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"**データソース**: J-Quants API（空売り）+ 金融庁 EDINET（大量保有）")
    lines.append(f"**分析**: クオンツ軍師（Asset Shield Project）")
    lines.append("")
    lines.append("#機関空売り #四半期レポート #日本株 #機関投資家")

    return "\n".join(lines)


def generate_yearly_article(db: DBReader, year: int) -> str:
    """Generate yearly summary article for note subscription magazine."""
    from_date = f"{year}-01-01"
    to_date = f"{year}-12-31"
    data = db.get_period_data(from_date, to_date)

    if data["total"] == 0:
        return ""

    total = data["total"]
    trading_days = data["trading_days"]
    daily_avg = total / trading_days if trading_days > 0 else 0

    lines = []

    # Title
    lines.append(f"# 年間 機関空売りレポート｜{year}年")
    lines.append("")
    lines.append(f"> **{year}年の機関投資家による空売りポジション開示を全件分析しました。"
                 f"年間{total}件のデータを集計・ランキング化した完全版レポートです。**")
    lines.append("")

    # Yearly summary
    lines.append("## 年間サマリー")
    lines.append("")
    lines.append(f"- **総開示件数**: {total}件")
    lines.append(f"- **新規ポジション**: {data['new_count']}件")
    lines.append(f"- **増加**: {data['increase_count']}件")
    lines.append(f"- **減少**: {data['decrease_count']}件")
    lines.append(f"- **営業日数**: {trading_days}日")
    lines.append(f"- **1日平均**: {daily_avg:.1f}件")
    lines.append("")

    # Monthly breakdown (12 months)
    if data["monthly_counts"]:
        lines.append("## 月別推移")
        lines.append("")
        lines.append("| 月 | 開示件数 |")
        lines.append("|:---:|:---:|")
        for ym, cnt in data["monthly_counts"].items():
            m = int(ym.split("-")[1])
            lines.append(f"| {m}月 | {cnt} |")
        lines.append("")

    # Quarterly breakdown (computed from monthly data)
    if data["monthly_counts"]:
        lines.append("## 四半期別推移")
        lines.append("")
        lines.append("| 四半期 | 期間 | 開示件数 |")
        lines.append("|:---:|:---|:---:|")
        quarter_map = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
        for q, (sm, em) in quarter_map.items():
            q_total = 0
            for ym, cnt in data["monthly_counts"].items():
                m = int(ym.split("-")[1])
                if sm <= m <= em:
                    q_total += cnt
            if q_total > 0:
                lines.append(f"| Q{q} | {sm}月〜{em}月 | {q_total} |")
        lines.append("")

    # Institution ranking TOP20
    lines.append("## 機関別 年間活動ランキング TOP20")
    lines.append("")
    lines.append("| 順位 | 機関名 | 件数 |")
    lines.append("|:---:|:---|:---:|")
    for i, (name, cnt) in enumerate(data["top_institutions"][:20], 1):
        lines.append(f"| {i} | {name} | {cnt} |")
    lines.append("")

    if data["top_institutions"]:
        lines.append(f"年間で最も活発だったのは**{data['top_institutions'][0][0]}**"
                     f"（{data['top_institutions'][0][1]}件）。")
        if len(data["top_institutions"]) >= 3:
            lines.append(f"続いて**{data['top_institutions'][1][0]}**"
                         f"（{data['top_institutions'][1][1]}件）、"
                         f"**{data['top_institutions'][2][0]}**"
                         f"（{data['top_institutions'][2][1]}件）。")
        lines.append("")

    # Concentrated stocks TOP20
    if data["concentrated_stocks"]:
        lines.append("## 年間 空売り集中銘柄 TOP20")
        lines.append("")
        lines.append("**複数機関が同日にポジションを持った日数**が多い銘柄です。"
                      "年間を通じて機関から注目され続けた銘柄のランキングです。")
        lines.append("")
        lines.append("| 順位 | 銘柄 | コード | 集中日数 |")
        lines.append("|:---:|:---|:---:|:---:|")
        for i, (code, days, name) in enumerate(data["concentrated_stocks"][:20], 1):
            co = shorten_co(name)
            code4 = code[:4] if len(code) >= 4 else code
            lines.append(f"| {i} | {co} | {code4} | {days}日 |")
        lines.append("")

    # Most targeted stocks TOP20
    if data["top_stocks"]:
        lines.append("## 最も狙われた銘柄 TOP20")
        lines.append("")
        lines.append("| 順位 | 銘柄 | コード | 開示回数 |")
        lines.append("|:---:|:---|:---:|:---:|")
        for i, (code, cnt, name) in enumerate(data["top_stocks"][:20], 1):
            co = shorten_co(name)
            code4 = code[:4] if len(code) >= 4 else code
            lines.append(f"| {i} | {co} | {code4} | {cnt} |")
        lines.append("")

    # EDINET bulk holdings yearly
    bulk = db.get_bulk_holdings_range(from_date, to_date)
    if bulk:
        lines.append("## EDINET 大量保有 年間まとめ")
        lines.append("")
        new_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "new")
        chg_cnt_bh = sum(1 for h in bulk if h.get("report_type") == "change")
        lines.append(f"年間でEDINETに提出された大量保有報告書は**{len(bulk)}件**。")
        lines.append("")
        lines.append(f"- **新規大量保有**: {new_cnt_bh}件")
        lines.append(f"- **保有比率変更**: {chg_cnt_bh}件")
        lines.append("")

        filer_counts = Counter(h.get("filer_name", "不明") for h in bulk)
        if filer_counts:
            lines.append("### 年間 大量保有 報告者ランキング")
            lines.append("")
            lines.append("| 順位 | 報告者 | 件数 |")
            lines.append("|:---:|:---|:---:|")
            for i, (filer, cnt) in enumerate(filer_counts.most_common(10), 1):
                lines.append(f"| {i} | {filer[:15]} | {cnt} |")
            lines.append("")

    # Yearly verification stats
    try:
        verif = db.get_verification_stats(from_date, to_date)
        if verif["count"] > 0:
            lines.append("## 年間シグナル検証")
            lines.append("")
            lines.append(f"年間の集中銘柄シグナル追跡結果（{verif['count']}件）：")
            lines.append("")
            lines.append(f"- **5日的中率**: {verif['hit_rate_5d']}%（平均リターン {verif['avg_ret_5d']:+.1f}%）")
            lines.append(f"- **20日的中率**: {verif['hit_rate_20d']}%（平均リターン {verif['avg_ret_20d']:+.1f}%）")
            lines.append("")
    except Exception:
        pass

    # Signal P&L section
    pnl = db.get_pnl_summary(from_date, to_date)
    if pnl["count"] > 0:
        lines.append("## シグナル仮想損益（年間）")
        lines.append("")
        lines.append(f"年間のシグナル仮想空売り成績（1シグナル100万円×{pnl['count']}件）：")
        lines.append("")
        lines.append("| シグナル種別 | 勝敗 | 勝率 | 累計損益 |")
        lines.append("|:---|:---:|:---:|---:|")
        for stype_data in pnl["by_type"].values():
            pnl_val = stype_data["cumulative_pnl"]
            pnl_str = f"+{pnl_val:,}円" if pnl_val >= 0 else f"{pnl_val:,}円"
            lines.append(f"| {stype_data['label']} | "
                         f"{stype_data['wins']}勝{stype_data['losses']}敗 | "
                         f"{stype_data['win_rate']:.0f}% | {pnl_str} |")
        pnl_total = pnl["total_pnl"]
        pnl_total_str = f"+{pnl_total:,}円" if pnl_total >= 0 else f"{pnl_total:,}円"
        lines.append(f"| **合計** | **{pnl['total_wins']}勝{pnl['total_losses']}敗** | "
                     f"**{pnl['total_win_rate']:.0f}%** | **{pnl_total_str}** |")
        lines.append("")
        lines.append("*※仮想損益です。貸株料・手数料は含みません。投資助言ではありません。*")
        lines.append("")

    # Yearly wrap-up
    lines.append("## 年間総括")
    lines.append("")
    aggression = (data["new_count"] + data["increase_count"]) / total * 100 if total > 0 else 50

    # Monthly trend narrative
    if data["monthly_counts"] and len(data["monthly_counts"]) >= 2:
        months_sorted = sorted(data["monthly_counts"].items())
        peak_month = max(months_sorted, key=lambda x: x[1])
        trough_month = min(months_sorted, key=lambda x: x[1])
        peak_m = int(peak_month[0].split("-")[1])
        trough_m = int(trough_month[0].split("-")[1])
        lines.append(f"{year}年の機関空売り動向を総括します。年間{total}件の開示があり、"
                     f"1日平均{daily_avg:.1f}件でした。")
        lines.append("")
        lines.append(f"月別では、**{peak_m}月**が最多（{peak_month[1]}件）、"
                     f"**{trough_m}月**が最少（{trough_month[1]}件）でした。")
        lines.append("")

    if aggression > 55:
        lines.append(f"年間を通じた攻勢度は{aggression:.0f}%で、機関投資家は**弱気寄り**のスタンスでした。"
                     f"新規・増加ポジションが優勢で、売り圧力が強い一年でした。")
    elif aggression < 40:
        lines.append(f"年間を通じた攻勢度は{aggression:.0f}%で、機関投資家は**強気寄り**のスタンスでした。"
                     f"減少（ショートカバー）が優勢で、空売りの巻き戻しが目立つ一年でした。")
    else:
        lines.append(f"年間を通じた攻勢度は{aggression:.0f}%で、機関投資家のスタンスは**概ね中立**でした。"
                     f"新規・増加と減少がバランスし、方向感が定まらない一年でした。")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"**データソース**: J-Quants API（空売り）+ 金融庁 EDINET（大量保有）")
    lines.append(f"**分析**: クオンツ軍師（Asset Shield Project）")
    lines.append("")
    lines.append("#機関空売り #年間レポート #日本株 #機関投資家")

    return "\n".join(lines)


# ===================================================================
# Output & Workflow
# ===================================================================
def save_article(content: str, date: str, mode: str) -> Path:
    """Save article to file."""
    ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTICLE_DIR / f"{date}_{mode}.md"
    path.write_text(content, encoding="utf-8")
    log.info("Article saved: %s (%d chars)", path, len(content))
    return path


def copy_to_clipboard(content: str):
    """Copy article content to macOS clipboard."""
    try:
        proc = subprocess.run(
            ["pbcopy"], input=content.encode("utf-8"),
            timeout=5, check=True,
        )
        log.info("Copied to clipboard")
    except Exception as e:
        log.warning("Failed to copy to clipboard: %s", e)


def open_note_editor():
    """Open note.com new article page in default browser."""
    try:
        subprocess.run(["open", NOTE_NEW_URL], timeout=5)
        log.info("Opened note.com editor")
    except Exception as e:
        log.warning("Failed to open browser: %s", e)


def notify(title: str, message: str):
    """Send macOS notification."""
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}" sound name "Glass"'
        ], timeout=5)
    except Exception:
        pass


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Note Article Generator")
    parser.add_argument("--mode", choices=["daily", "weekly", "monthly", "quarterly", "yearly"],
                        default="daily")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: latest available.")
    parser.add_argument("--year", type=int, default=None,
                        help="Year for monthly/quarterly/yearly")
    parser.add_argument("--month", type=int, default=None,
                        help="Month for monthly (1-12)")
    parser.add_argument("--quarter", type=int, default=None,
                        help="Quarter for quarterly (1-4)")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open browser")
    parser.add_argument("--no-clipboard", action="store_true",
                        help="Don't copy to clipboard")
    parser.add_argument("--quiet", action="store_true",
                        help="No notification")
    args = parser.parse_args()

    if not DB_PATH.exists():
        log.error("tracker.db not found: %s", DB_PATH)
        sys.exit(1)

    db = DBReader()

    try:
        # Determine date / period
        now = datetime.now()

        if args.mode in ("daily", "weekly"):
            if args.date:
                date = args.date
            else:
                dates = db.get_available_dates(5)
                if not dates:
                    log.error("No data in tracker.db")
                    sys.exit(1)
                date = dates[0]
                log.info("Using latest date: %s", date)

        # Generate article
        if args.mode == "daily":
            content = generate_daily_article(db, date)
            save_date = date

        elif args.mode == "weekly":
            content = generate_weekly_article(db, date)
            save_date = date

        elif args.mode == "monthly":
            # Default to previous month
            if args.year and args.month:
                y, m = args.year, args.month
            else:
                prev = now.replace(day=1) - timedelta(days=1)
                y, m = prev.year, prev.month
            log.info("Monthly report: %d-%02d", y, m)
            content = generate_monthly_article(db, y, m)
            save_date = f"{y}-{m:02d}"

        elif args.mode == "quarterly":
            # Default to previous quarter
            if args.year and args.quarter:
                y, q = args.year, args.quarter
            else:
                cur_q = (now.month - 1) // 3 + 1
                if cur_q == 1:
                    y, q = now.year - 1, 4
                else:
                    y, q = now.year, cur_q - 1
            log.info("Quarterly report: %dQ%d", y, q)
            content = generate_quarterly_article(db, y, q)
            save_date = f"{y}-Q{q}"

        elif args.mode == "yearly":
            # Default to previous year
            y = args.year if args.year else now.year - 1
            log.info("Yearly report: %d", y)
            content = generate_yearly_article(db, y)
            save_date = f"{y}"

        else:
            content = ""
            save_date = ""

        if not content:
            log.error("No content generated for mode=%s", args.mode)
            sys.exit(1)

        # Save
        path = save_article(content, save_date, args.mode)

        # Clipboard
        if not args.no_clipboard:
            copy_to_clipboard(content)

        # Open editor
        if not args.no_open:
            open_note_editor()

        # Notify
        if not args.quiet:
            mode_jp_map = {
                "daily": "日報",
                "weekly": "週報",
                "monthly": "月報",
                "quarterly": "四半期報",
                "yearly": "年報",
            }
            mode_jp = mode_jp_map.get(args.mode, args.mode)
            notify(
                "note記事準備完了",
                f"{mode_jp}（{save_date}）をクリップボードにコピーしました。Cmd+Vで貼り付けてください。",
            )

        log.info("Done. Article: %s", path)

    finally:
        db.close()


if __name__ == "__main__":
    main()
