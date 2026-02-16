#!/usr/bin/env python3
"""Generate a professional Japanese resume PDF for ProConnect."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser("~/Desktop/Resume_Quant_Engineer.pdf")

# Japanese font path (macOS built-in)
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_BOLD_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


class ResumePDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("gothic", size=8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"{self.page_no()}/{{nb}}", align="C")


def build_pdf():
    pdf = ResumePDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Register fonts
    pdf.add_font("gothic", "", FONT_PATH)
    pdf.add_font("gothic", "B", FONT_BOLD_PATH)

    pdf.add_page()

    # Title
    pdf.set_font("gothic", "B", 18)
    pdf.cell(0, 12, "職務経歴書", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(3)

    # Date
    pdf.set_font("gothic", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "2026年2月11日現在", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    def section_title(title):
        pdf.set_fill_color(40, 60, 100)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("gothic", "B", 11)
        pdf.cell(0, 8, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def body_text(text, size=10, bold=False):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("gothic", "B" if bold else "", size)
        pdf.multi_cell(0, 6, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullet(text, indent=8, size=10):
        pdf.set_font("gothic", "", size)
        pdf.set_x(pdf.l_margin + indent)
        w = pdf.w - pdf.l_margin - pdf.r_margin - indent
        pdf.multi_cell(w, 6, f"・{text}", align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # --- 職種 ---
    section_title("職種")
    body_text("クオンツエンジニア", 12, bold=True)
    pdf.ln(3)

    # --- 職務要約 ---
    section_title("職務要約")
    body_text(
        "米国株および日本株を対象としたアルゴリズムトレーディングシステムの設計・開発・運用を"
        "一貫して担当。マルチファクター戦略をベースに、8世代にわたる反復開発を通じて、"
        "15年バックテスト（2010〜2024年）で"
        "CAGR\u00a014.4%、Sharpe\u00a0Ratio\u00a00.92のCore-Satellite戦略を構築。"
        "ウォークフォワード検証（IS/OOS分割）による"
        "オーバーフィッティング防止、レジーム検出による動的資産配分、キルスイッチによる"
        "リスク管理を実装し、QuantConnect上で運用中。"
    )
    pdf.ln(3)

    # --- 専門領域 ---
    section_title("専門領域")
    bullets = [
        "マルチファクター株式戦略の設計・開発・運用（米国株・日本株）",
        "ウォークフォワード検証（IS/OOS分割）によるモデル評価・過学習防止",
        "レジーム検出（CRISIS/BEAR/NEUTRAL/BULL/SUPER_BULL）に基づく動的資産配分",
        "リスク管理機構の実装（キルスイッチ、ドローダウン制御、逆ボラティリティサイジング）",
        "金融データETLパイプライン構築（REST API → パーサー → SQLite → シグナル生成）",
        "機械学習による株式予測モデル構築（Numerai Tournament参戦中）",
    ]
    for b in bullets:
        bullet(b)
    pdf.ln(3)

    # --- 技術スタック ---
    section_title("技術スタック")

    skills = [
        ("言語", "Python（メイン）、SQL"),
        ("ライブラリ", "NumPy, pandas, scikit-learn, SciPy, LightGBM, statsmodels, matplotlib"),
        ("プラットフォーム", "QuantConnect（LEAN Engine）、Interactive Brokers API"),
        ("データベース", "SQLite、PostgreSQL"),
        ("データソース", "J-Quants API、Numerai Dataset（v5.0）"),
        ("バージョン管理", "Git"),
    ]
    for label, value in skills:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("gothic", "", 10)
        pdf.multi_cell(0, 6, f"  {label}：{value}", align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # --- 実績 ---
    section_title("主要実績")

    body_text("1. 米国株 Core-Satellite 戦略（V8）", 10, bold=True)
    sub_bullets = [
        "15年バックテスト（2010〜2024年）: 累積+649%, CAGR 14.4%, Sharpe 0.92",
        "ウォークフォワード検証: IS Sharpe 1.10 / OOS Sharpe 0.74",
        "最大ドローダウン: 25.7%（COVID-19ショック、構造上の想定内）",
        "5ファクター統合: Momentum 35%, Short-term Reversal 10%, Low Vol 15%, Value(PE) 20%, Quality(ROE) 20%",
        "レジーム検出による動的エクスポージャー調整（5段階）",
        "キルスイッチ: DD 15%到達で衛星ポジション自動清算",
        "QuantConnect上でPaper Trading運用中",
    ]
    for b in sub_bullets:
        bullet(b, indent=12, size=9)
    pdf.ln(2)

    body_text("2. 日本株データ基盤", 10, bold=True)
    sub_bullets2 = [
        "J-Quants APIを活用した5,000銘柄×18年分のETLパイプライン構築",
        "サバイバーシップバイアスフリーの4フェーズストレステスト",
        "SQLiteベースのデータウェアハウス（14.9M レコード）",
    ]
    for b in sub_bullets2:
        bullet(b, indent=12, size=9)
    pdf.ln(2)

    body_text("3. Numerai Tournament", 10, bold=True)
    sub_bullets4 = [
        "LightGBMによる株式予測モデルを構築、Round 1200より参戦",
        "705特徴量（medium feature set）、2,746,268サンプルで学習",
    ]
    for b in sub_bullets4:
        bullet(b, indent=12, size=9)
    pdf.ln(3)

    # --- 業界知識 ---
    section_title("業界知識")
    domain_bullets = [
        "米国株式市場（NYSE, NASDAQ, S&P 500）",
        "日本株式市場（JPX, 東証プライム）",
        "ファクター投資理論（Fama-French, Momentum, Low Volatility, Quality）",
        "ポートフォリオ構築理論（Core-Satellite, 逆ボラティリティ加重, Kelly基準）",
    ]
    for b in domain_bullets:
        bullet(b)
    pdf.ln(3)

    # --- 職歴 ---
    section_title("職歴")
    body_text("クオンツエンジニア / アルゴリズムストラテジスト（独立・フリーランス）", 10, bold=True)
    pdf.set_font("gothic", "", 9)
    pdf.cell(0, 6, "2021年 〜 現在　|　日本・リモート", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    body_text(
        "米国株・日本株を対象としたアルゴリズムトレーディングシステムの設計・開発・運用を"
        "すべて一人で担当。戦略設計からデータ取得、バックテスト、リスク管理、"
        "ライブ実行基盤の構築まで、エンドツーエンドで開発。"
        "8世代にわたる反復改善を経て、ウォークフォワード検証を通過する"
        "Core-Satellite戦略（V8）を完成。"
    )
    pdf.ln(3)

    # --- 希望条件 ---
    section_title("希望条件")
    conditions = [
        ("希望単価", "80〜100万円/月（経験・実績に応じて応相談）"),
        ("稼働形態", "フルリモート希望"),
        ("稼働日数", "週3〜5日（応相談）"),
        ("即日稼働", "可能"),
    ]
    for label, value in conditions:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("gothic", "B", 10)
        pdf.cell(30, 6, f"  {label}")
        pdf.set_font("gothic", "", 10)
        pdf.cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Output
    pdf.output(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
