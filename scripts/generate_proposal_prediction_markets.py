#!/usr/bin/env python3
"""Generate PDF proposal for Quantitative Trading Systems Engineer - Prediction Markets."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser("~/Desktop/Upwork_Proposal_Prediction_Markets.pdf")
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

    # === Title ===
    pdf.set_font("gothic", "B", 14)
    pdf.cell(0, 10, "Upwork Proposal", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(1)

    pdf.set_font("gothic", "B", 11)
    pdf.set_x(LM)
    pdf.multi_cell(W, 7,
        "Job: Quantitative Trading Systems Engineer\n"
        "\u2013 Prediction Markets",
        align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # === URL ===
    section_header("Application URL")
    small_body(
        "https://www.upwork.com/freelance-jobs/apply/"
        "Quantitative-Trading-Systems-Engineer-Prediction-Markets"
        "_~(Job ID from Upwork)"
    )
    pdf.ln(2)

    # === Cover Letter ===
    section_header("Cover Letter (Copy & Paste)")

    cover = (
        "I've spent the past 4 years building systematic trading infrastructure "
        "from scratch. My current production system is a US equity Core-Satellite "
        "strategy running on QuantConnect with IB integration \u2014 8 generations "
        "of iteration, walk-forward validated across 15 years of data "
        "(IS\u00a0Sharpe\u00a01.10, OOS\u00a00.74), now in live paper trading.\n\n"
        "What maps directly to your needs:\n\n"
        "\u2022 Risk controls: I've implemented kill-switches (DD-triggered satellite "
        "liquidation), position limits, exposure caps, and 4-stage recovery "
        "protocols in production.\n\n"
        "\u2022 Strategy \u2192 execution pipeline: I translate quantitative concepts "
        "into executable code daily \u2014 factor models, regime detection, "
        "inverse-volatility sizing, trailing stops, rebalancing logic.\n\n"
        "\u2022 Backtesting infrastructure: Built complete walk-forward validation "
        "frameworks with IS/OOS splitting to prevent overfitting. Also competing "
        "in Numerai Tournament (LightGBM, 705 features).\n\n"
        "\u2022 Data pipelines: REST API ingestion, fixed-width/binary parsing, "
        "SQLite storage, signal generation \u2014 all built and maintained by me alone.\n\n"
        "I haven't worked with prediction markets specifically, but the core "
        "engineering is identical: ingest market data via API, generate signals, "
        "execute with risk controls, monitor live. I learn fast and work "
        "independently.\n\n"
        "I prefer async text-based communication \u2014 it produces better-documented "
        "decisions and fits trading system development well.\n\n"
        "Rate: $75/hr. Available to start immediately."
    )
    body(cover, size=10)
    pdf.ln(2)

    # === Question 1 ===
    section_header("Q1: Describe your recent experience with similar projects")

    q1 = (
        "Built and currently operate Asset Shield V8 \u2014 a systematic US equity "
        "Core-Satellite strategy on QuantConnect/LEAN with Interactive Brokers "
        "integration. The system runs 5-factor signal generation (momentum, "
        "mean-reversion, volatility, value, quality), 5-state regime detection, "
        "automated kill-switch with staged recovery, and inverse-volatility "
        "position sizing. Walk-forward validated over 15 years (2010-2024) with "
        "strict IS/OOS separation. Currently in live paper trading.\n\n"
        "Also built a complete data pipeline for Japanese horse racing prediction: "
        "fixed-width binary parsing (374-byte records, cp932), ETL to SQLite "
        "(1.77M records), ML walk-forward with quarterly retraining (LightGBM), "
        "Kelly-criterion bet sizing, and probability calibration. The project "
        "taught me to operate in markets with high transaction costs (25% takeout) "
        "and imperfect data \u2014 directly relevant to early-stage prediction markets."
    )
    body(q1, size=10)
    pdf.ln(2)

    # === Question 2 ===
    section_header("Q2: What tools do you use for data mining and visualization?")

    q2 = (
        "Data: Python (pandas, NumPy, SciPy), SQL/SQLite, REST API clients, "
        "custom ETL pipelines.\n"
        "ML: LightGBM, scikit-learn, statsmodels, isotonic/logistic calibration.\n"
        "Visualization: matplotlib, seaborn, custom PDF reporting.\n"
        "Platform: QuantConnect/LEAN Engine, Git.\n"
        "Monitoring: Custom logging, email alert systems, performance dashboards."
    )
    body(q2, size=10)

    # === New page for Japanese translation ===
    pdf.add_page()

    pdf.set_font("gothic", "B", 14)
    pdf.cell(0, 10, "\u548c\u8a33\uff08\u4e0a\u69d8\u78ba\u8a8d\u7528\uff09",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    # Cover Letter translation
    section_header("Cover Letter \u548c\u8a33")

    cover_ja = (
        "過去4年間、体系的トレーディング基盤をゼロから構築してきました。"
        "現在の本番システムは、QuantConnect上でIB連携した米国株コア・サテライト戦略（V8）で、"
        "8世代の反復開発を経て、15年分のデータでウォークフォワード検証済み"
        "（IS\u00a0Sharpe\u00a01.10、OOS\u00a00.74）。現在ライブペーパートレーディング中です。\n\n"
        "貴社の要件に直結するスキル：\n\n"
        "\u2022 リスク管理：キルスイッチ（DD発動でサテライト清算）、ポジション制限、"
        "エクスポージャーキャップ、4段階回復プロトコルを本番実装済み。\n\n"
        "\u2022 戦略→執行パイプライン：ファクターモデル、レジーム検出、逆ボラティリティサイジング、"
        "トレーリングストップ、リバランスロジックを日常的にコード化。\n\n"
        "\u2022 バックテスト基盤：IS/OOS分割によるウォークフォワード検証フレームワークを構築。"
        "Numerai Tournamentにも参戦中（LightGBM、705特徴量）。\n\n"
        "\u2022 データパイプライン：REST API取得、固定長/バイナリ解析、SQLite格納、"
        "シグナル生成——すべて一人で構築・保守。\n\n"
        "Prediction Market固有の経験はありませんが、核心技術は同一です：APIでデータ取得→"
        "シグナル生成→リスク管理付きで執行→ライブ監視。学習が速く、独立して作業します。\n\n"
        "効率のため、非同期テキストベースのコミュニケーションを好みます。\n\n"
        "時給：$75。即日稼働可能。"
    )
    body(cover_ja, size=9)
    pdf.ln(2)

    # Q1 translation
    section_header("Q1 \u548c\u8a33\uff1a\u985e\u4f3c\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306e\u7d4c\u9a13")

    q1_ja = (
        "Asset Shield V8を構築し現在運用中。QC/LEAN上のIB連携米国株コア・サテライト戦略。"
        "5ファクターシグナル生成（モメンタム、平均回帰、ボラティリティ、バリュー、クオリティ）、"
        "5段階レジーム検出、段階的回復付き自動キルスイッチ、逆ボラサイジング。"
        "15年間（2010-2024）のIS/OOS厳密分離ウォークフォワード検証済み。ライブペーパートレーディング中。\n\n"
        "加えて、日本競馬予測の完全データパイプラインを構築。固定長バイナリ解析（374バイト、cp932）、"
        "SQLiteへのETL（177万レコード）、四半期再学習のMLウォークフォワード（LightGBM）、"
        "ケリー基準ベットサイジング、確率キャリブレーション。高控除率（25%）と不完全データ下での"
        "運用経験は、初期段階のPrediction Marketに直接関連。"
    )
    body(q1_ja, size=9)
    pdf.ln(2)

    # Q2 translation
    section_header("Q2 \u548c\u8a33\uff1a\u30c7\u30fc\u30bf\u30de\u30a4\u30cb\u30f3\u30b0\u30fb\u53ef\u8996\u5316\u30c4\u30fc\u30eb")

    q2_ja = (
        "データ処理：Python（pandas, NumPy, SciPy）、SQL/SQLite、REST APIクライアント、独自ETL。\n"
        "機械学習：LightGBM、scikit-learn、statsmodels、isotonic/logisticキャリブレーション。\n"
        "可視化：matplotlib、seaborn、独自PDFレポート。\n"
        "プラットフォーム：QuantConnect/LEAN Engine、Git。\n"
        "監視：独自ログ、メールアラート、パフォーマンスダッシュボード。"
    )
    body(q2_ja, size=9)

    # === Page 3: Instructions ===
    pdf.add_page()
    pdf.set_font("gothic", "B", 14)
    pdf.cell(0, 10, "\u5fdc\u52df\u624b\u9806",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    section_header("\u624b\u9806")

    steps = [
        "1. Upwork\u306b\u30ed\u30b0\u30a4\u30f3",
        "2. \u300cBuy Connects\u300d\u304b\u308918 Connects\u4ee5\u4e0a\u3092\u8cfc\u5165\uff08$2.70\uff09",
        "3. \u6848\u4ef6\u30da\u30fc\u30b8\u3092\u958b\u304d\u300cApply Now\u300d\u3092\u30af\u30ea\u30c3\u30af",
        "4. Cover Letter\u6b04\u306b1\u30da\u30fc\u30b8\u76ee\u306eCover Letter\u3092\u8cbc\u308a\u4ed8\u3051",
        "5. Q1\u6b04\u306b2\u30da\u30fc\u30b8\u76ee\u306eQ1\u56de\u7b54\u3092\u8cbc\u308a\u4ed8\u3051",
        "6. Q2\u6b04\u306b2\u30da\u30fc\u30b8\u76ee\u306eQ2\u56de\u7b54\u3092\u8cbc\u308a\u4ed8\u3051",
        "7. Hourly Rate: $75 \u3092\u78ba\u8a8d",
        "8. Submit Proposal \u3092\u30af\u30ea\u30c3\u30af",
    ]
    for step in steps:
        pdf.set_x(LM + 5)
        pdf.set_font("gothic", "", 11)
        pdf.multi_cell(W - 5, 7, step, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    section_header("\u6ce8\u610f\u4e8b\u9805")
    notes = [
        "\u2022 Cover Letter\u306f\u82f1\u6587\u306e\u307f\u8cbc\u308a\u4ed8\u3051\u3066\u304f\u3060\u3055\u3044\uff08\u548c\u8a33\u306f\u4e0a\u69d8\u78ba\u8a8d\u7528\uff09",
        "\u2022 Q1\u30fbQ2\u3082\u82f1\u6587\u306e\u307f",
        "\u2022 URL\u306f\u6848\u4ef6\u30da\u30fc\u30b8\u304b\u3089\u76f4\u63a5Apply\u3057\u3066\u304f\u3060\u3055\u3044",
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
