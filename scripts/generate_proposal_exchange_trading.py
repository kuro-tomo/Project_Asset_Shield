#!/usr/bin/env python3
"""Generate PDF proposal for Quantitative Python Developer – Exchange-Based Trading System."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser("~/Desktop/Upwork_Proposal_Exchange_Trading.pdf")
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_BOLD_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


class ProposalPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("gothic", size=8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"{self.page_no()}/{{nb}}", align="C")


def build_pdf():
    pdf = ProposalPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_font("gothic", "", FONT_PATH)
    pdf.add_font("gothic", "B", FONT_BOLD_PATH)
    pdf.add_page()

    LM = pdf.l_margin
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def section_header(text):
        pdf.set_fill_color(40, 60, 100)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("gothic", "B", 10)
        pdf.cell(0, 7, f"  {text}", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    def body(text, size=10, bold=False):
        pdf.set_x(LM)
        pdf.set_font("gothic", "B" if bold else "", size)
        pdf.multi_cell(W, 6, text, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def small_body(text, size=9):
        pdf.set_x(LM)
        pdf.set_font("gothic", "", size)
        pdf.multi_cell(W, 5, text, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    # === Page 1: Title + Cover Letter ===
    pdf.set_font("gothic", "B", 14)
    pdf.cell(0, 10, "Upwork Proposal", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(1)

    pdf.set_font("gothic", "B", 11)
    pdf.set_x(LM)
    pdf.multi_cell(W, 7,
        "Job: Quantitative Python Developer\n"
        "\u2013 Exchange-Based Trading System",
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Cover Letter
    section_header("Cover Letter (Copy & Paste)")

    cover = (
        "I build and operate exchange-connected trading systems as my core work. "
        "My current production system is a US equity Core-Satellite strategy on "
        "QuantConnect/LEAN with Interactive Brokers integration \u2014 8 generations "
        "of iteration, walk-forward validated across 15 years of data "
        "(IS\u00a0Sharpe\u00a01.10, OOS\u00a00.74), now in live paper trading.\n\n"
        "What maps directly to your scope:\n\n"
        "\u2022 API-driven execution: I've built complete pipelines from REST API "
        "data ingestion through signal generation to order execution. My system "
        "handles real-time market data, processes signals, and routes orders "
        "through broker APIs with full error handling and retry logic.\n\n"
        "\u2022 Event-driven architecture: My production system uses event-driven "
        "workflows \u2014 market data events trigger signal computation, which feeds "
        "a position manager that generates orders. State is managed cleanly "
        "with separation between signal logic and execution logic.\n\n"
        "\u2022 Risk controls in production: Kill-switches (drawdown-triggered "
        "liquidation), position limits, exposure caps, and 4-stage recovery "
        "protocols. These aren't theoretical \u2014 they're running in my live system.\n\n"
        "\u2022 Data infrastructure: SQLite storage, custom ETL pipelines, "
        "structured logging, and monitoring dashboards. I've also built "
        "fixed-width binary parsers and handled data quality at scale "
        "(1.77M records, 83 fields).\n\n"
        "I work independently, write clean Python with Git discipline, and "
        "prefer async text-based communication. Comfortable signing an NDA "
        "\u2014 I understand the sensitivity of proprietary trading systems.\n\n"
        "Rate: $75/hr. Available to start immediately."
    )
    body(cover, size=10)

    # === Page 2: Screening Questions ===
    pdf.add_page()

    pdf.set_font("gothic", "B", 14)
    pdf.cell(0, 10, "Screening Questions",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    # Q1
    section_header("Q1: Examples of Python-based systems you\u2019ve built for trading or quantitative data")

    q1 = (
        "1. Asset Shield V8 (current production): A US equity Core-Satellite "
        "strategy on QuantConnect/LEAN with IB integration. The system runs "
        "5-factor signal generation (momentum, mean-reversion, low-volatility, "
        "value PE, quality ROE), 5-state regime detection (VIX + moving-average "
        "based), automated kill-switch with staged recovery, and inverse-volatility "
        "position sizing. Walk-forward validated over 15 years (2010\u20132024) with "
        "strict IS/OOS separation. Currently in live paper trading.\n\n"
        "2. Quantitative data pipeline for Japanese markets: Built a complete "
        "ETL system \u2014 REST API ingestion, fixed-width binary parsing (374-byte "
        "records, cp932 encoding), SQLite storage (1.77M records across 83 fields), "
        "automated data quality checks (99%+ coverage), and signal generation "
        "for a multi-factor model.\n\n"
        "3. Numerai Tournament: ML pipeline with LightGBM, 705 engineered "
        "features, walk-forward quarterly retraining, probability calibration "
        "(isotonic regression), and automated weekly submission via API."
    )
    body(q1, size=10)
    pdf.ln(2)

    # Q2
    section_header("Q2: How would you architect a system where signal generation and execution are separate?")

    q2 = (
        "Three-layer separation with message passing:\n\n"
        "Signal Layer: Consumes market data events, computes factor scores and "
        "regime state, outputs target portfolio weights. Runs on its own schedule "
        "(e.g. daily close). No knowledge of order mechanics. Writes targets to "
        "a shared state store (SQLite/Redis).\n\n"
        "Execution Layer: Reads target weights, compares to current positions, "
        "generates orders. Handles order splitting, retry logic, partial fills, "
        "and exchange-specific constraints. Communicates with exchange APIs. "
        "Reports fill status back to the state store.\n\n"
        "Risk Layer (sits between): Validates every order before submission "
        "\u2014 position limits, exposure caps, drawdown checks, rate limits. "
        "Can block or reduce orders. Triggers kill-switch if thresholds are "
        "breached.\n\n"
        "All three communicate via well-defined interfaces (dataclasses/typed "
        "dicts), logged to structured storage. Each layer can be tested, "
        "restarted, or replaced independently. This is essentially how my "
        "current V8 system works."
    )
    body(q2, size=10)
    pdf.ln(2)

    # Q3
    section_header("Q3: Experience with live order handling and risk management")

    q3 = (
        "My production system implements:\n\n"
        "\u2022 Kill-switch: When portfolio drawdown exceeds 15%, all satellite "
        "positions are liquidated automatically. Core (SPY) is maintained. "
        "Recovery follows a 4-stage protocol before re-entering.\n\n"
        "\u2022 Position limits: Maximum allocation per holding is capped. "
        "Inverse-volatility sizing ensures volatile assets get smaller positions.\n\n"
        "\u2022 Exposure management: Total portfolio exposure monitored in real-time. "
        "Regime detection adjusts target exposure (crisis = reduce, bull = full).\n\n"
        "\u2022 Order validation: Every order is checked against risk parameters "
        "before submission. Logging captures all state transitions for "
        "post-trade analysis.\n\n"
        "\u2022 Monitoring: Email alerts for kill-switch triggers, regime changes, "
        "and weekly performance summaries. System uptime monitoring with "
        "automated notifications."
    )
    body(q3, size=10)
    pdf.ln(2)

    # Q4
    section_header("Q4: Comfortable working under NDA on proprietary systems?")

    q4 = (
        "Yes, fully comfortable. I build proprietary trading systems myself and "
        "understand the need for confidentiality around strategy logic, data "
        "sources, and execution infrastructure. I have no issue signing an NDA "
        "and working within its constraints."
    )
    body(q4, size=10)

    # === Page 3: Japanese translation ===
    pdf.add_page()

    pdf.set_font("gothic", "B", 14)
    pdf.cell(0, 10, "\u548c\u8a33\uff08\u4e0a\u69d8\u78ba\u8a8d\u7528\uff09",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    section_header("Cover Letter \u548c\u8a33")

    cover_ja = (
        "取引所接続型トレーディングシステムの構築・運用が本業です。"
        "現在の本番システムは、QuantConnect/LEAN上でIB連携した米国株コア・サテライト戦略"
        "（V8）で、8世代の反復開発を経て、15年分のデータでウォークフォワード検証済み"
        "（IS\u00a0Sharpe\u00a01.10、OOS\u00a00.74）。現在ライブペーパートレーディング中です。\n\n"
        "貴社スコープに直結するスキル：\n\n"
        "\u2022 API駆動の執行：REST APIデータ取得→シグナル生成→注文執行の完全パイプラインを構築済み。"
        "リアルタイム市場データ処理、ブローカーAPI経由の注文ルーティング、エラーハンドリング・リトライロジック完備。\n\n"
        "\u2022 イベント駆動アーキテクチャ：市場データイベント→シグナル計算→ポジションマネージャー→注文生成。"
        "シグナルロジックと執行ロジックを明確に分離。\n\n"
        "\u2022 本番リスク管理：キルスイッチ（DD発動で清算）、ポジション制限、エクスポージャーキャップ、"
        "4段階回復プロトコル。理論ではなく実稼働中。\n\n"
        "\u2022 データ基盤：SQLite、独自ETL、構造化ログ、監視ダッシュボード。"
        "固定長バイナリ解析（177万レコード、83フィールド）の実績あり。\n\n"
        "独立して作業し、Gitで規律あるPythonを書きます。NDA署名に問題ありません。\n\n"
        "時給：$75。即日稼働可能。"
    )
    body(cover_ja, size=9)
    pdf.ln(2)

    section_header("Q1 \u548c\u8a33\uff1a\u69cb\u7bc9\u3057\u305fPython\u30d9\u30fc\u30b9\u30b7\u30b9\u30c6\u30e0\u306e\u4f8b")

    q1_ja = (
        "1. Asset Shield V8（本番稼働中）：QC/LEAN上のIB連携米国株コア・サテライト戦略。"
        "5ファクターシグナル生成、5段階レジーム検出、段階的回復付きキルスイッチ、逆ボラサイジング。"
        "15年間のIS/OOS厳密分離ウォークフォワード検証済み。ライブペーパートレーディング中。\n\n"
        "2. 日本市場データパイプライン：REST API取得、固定長バイナリ解析（374バイト、cp932）、"
        "SQLite格納（177万レコード、83フィールド）、データ品質チェック（99%+カバレッジ）、"
        "マルチファクターモデル用シグナル生成。\n\n"
        "3. Numerai Tournament：LightGBM、705特徴量、四半期再学習ウォークフォワード、"
        "確率キャリブレーション、API自動提出。"
    )
    body(q1_ja, size=9)
    pdf.ln(2)

    section_header("Q2 \u548c\u8a33\uff1a\u30b7\u30b0\u30ca\u30eb\u3068\u57f7\u884c\u306e\u5206\u96e2\u8a2d\u8a08")

    q2_ja = (
        "メッセージパッシングによる3層分離：\n\n"
        "シグナル層：市場データイベントを消費し、ファクタースコアとレジーム状態を計算、"
        "目標ポートフォリオウェイトを出力。独自スケジュール（例：日次終値）で稼働。"
        "注文メカニクスの知識不要。共有ステートストア（SQLite/Redis）に書き込み。\n\n"
        "執行層：目標ウェイトを読み取り、現在ポジションと比較、注文を生成。"
        "注文分割、リトライロジック、部分約定、取引所固有の制約を処理。"
        "ステートストアに約定状況を報告。\n\n"
        "リスク層（両者の間に配置）：提出前に全注文を検証"
        "（ポジション制限、エクスポージャーキャップ、DD確認、レート制限）。"
        "注文をブロックまたは削減可能。閾値超過でキルスイッチ発動。\n\n"
        "3層は型定義されたインターフェースで通信し、構造化ストレージに記録。"
        "各層を独立してテスト・再起動・置換可能。現V8システムと本質的に同じ設計。"
    )
    body(q2_ja, size=9)
    pdf.ln(2)

    section_header("Q3 \u548c\u8a33\uff1a\u30e9\u30a4\u30d6\u6ce8\u6587\u30fb\u30ea\u30b9\u30af\u7ba1\u7406\u7d4c\u9a13")

    q3_ja = (
        "本番システムの実装：\n\n"
        "\u2022 キルスイッチ：DD 15%超過でサテライト全清算。コア（SPY）は維持。4段階回復プロトコル。\n\n"
        "\u2022 ポジション制限：銘柄別最大配分キャップ。逆ボラサイジングで高ボラ資産は小ポジション。\n\n"
        "\u2022 エクスポージャー管理：リアルタイム監視。レジームに応じて目標エクスポージャーを調整。\n\n"
        "\u2022 注文検証：全注文をリスクパラメータで事前チェック。状態遷移をログに記録。\n\n"
        "\u2022 監視：キルスイッチ発動・レジーム変更・週次パフォーマンスのメールアラート。"
    )
    body(q3_ja, size=9)
    pdf.ln(2)

    section_header("Q4 \u548c\u8a33\uff1aNDA\u306b\u3064\u3044\u3066")

    q4_ja = (
        "全く問題ありません。自身も独自のトレーディングシステムを構築しており、"
        "戦略ロジック・データソース・執行基盤の機密性の必要性を理解しています。"
        "NDA署名とその制約内での作業に支障ありません。"
    )
    body(q4_ja, size=9)

    # === Page 4: Instructions ===
    pdf.add_page()
    pdf.set_font("gothic", "B", 14)
    pdf.cell(0, 10, "\u5fdc\u52df\u624b\u9806",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    section_header("\u624b\u9806")

    steps = [
        "1. Upwork\u306b\u30ed\u30b0\u30a4\u30f3",
        "2. \u6848\u4ef6\u30da\u30fc\u30b8\u3092\u958b\u304d\u300cApply Now\u300d\u3092\u30af\u30ea\u30c3\u30af",
        "3. Cover Letter\u6b04\u306b1\u30da\u30fc\u30b8\u76ee\u306eCover Letter\u3092\u8cbc\u308a\u4ed8\u3051",
        "4. Q1\u6b04\u306b2\u30da\u30fc\u30b8\u76ee\u306eQ1\u56de\u7b54\u3092\u8cbc\u308a\u4ed8\u3051",
        "5. Q2\u6b04\u306b2\u30da\u30fc\u30b8\u76ee\u306eQ2\u56de\u7b54\u3092\u8cbc\u308a\u4ed8\u3051",
        "6. Q3\u6b04\u306b2\u30da\u30fc\u30b8\u76ee\u306eQ3\u56de\u7b54\u3092\u8cbc\u308a\u4ed8\u3051",
        "7. Q4\u6b04\u306b2\u30da\u30fc\u30b8\u76ee\u306eQ4\u56de\u7b54\u3092\u8cbc\u308a\u4ed8\u3051",
        "8. Hourly Rate: $75 \u3092\u78ba\u8a8d",
        "9. Schedule a rate increase: Every 6 months / 5%",
        "10. Boost: 0\u306b\u8a2d\u5b9a",
        "11. Submit Proposal \u3092\u30af\u30ea\u30c3\u30af",
    ]
    for step in steps:
        pdf.set_x(LM + 5)
        pdf.set_font("gothic", "", 11)
        pdf.multi_cell(W - 5, 7, step, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    section_header("\u6ce8\u610f\u4e8b\u9805")
    notes = [
        "\u2022 Cover Letter\u30fbQ1\u30fcQ4\u306f\u82f1\u6587\u306e\u307f\u8cbc\u308a\u4ed8\u3051\u3066\u304f\u3060\u3055\u3044\uff08\u548c\u8a33\u306f\u4e0a\u69d8\u78ba\u8a8d\u7528\uff09",
        "\u2022 Connects: 18\u679a\u6d88\u8cbb\uff08\u6b8b\u9ad8: 132\u679a\uff09",
        "\u2022 \u524d\u56de\u306e\u5fdc\u52df\u3068\u540c\u3058\u304f\u3001Policy Acknowledgment\u306b\u30c1\u30a7\u30c3\u30af\u3092\u5165\u308c\u308b\u3053\u3068",
    ]
    for note in notes:
        pdf.set_x(LM + 5)
        pdf.set_font("gothic", "", 10)
        pdf.multi_cell(W - 5, 6, note, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
