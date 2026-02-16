#!/usr/bin/env python3
"""Generate PDF for note paid magazine introduction article."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser("~/Desktop/note_magazine_intro.pdf")
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_BOLD_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


class ArticlePDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("gothic", size=8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"{self.page_no()}/{{nb}}", align="C")


def build_pdf():
    pdf = ArticlePDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_font("gothic", "", FONT_PATH)
    pdf.add_font("gothic", "B", FONT_BOLD_PATH)
    pdf.add_page()

    LM = pdf.l_margin
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def title(text, size=16):
        pdf.set_x(LM)
        pdf.set_font("gothic", "B", size)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(W, 9, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    def heading(text, size=13):
        pdf.ln(3)
        pdf.set_x(LM)
        pdf.set_font("gothic", "B", size)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(W, 8, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def subheading(text, size=11):
        pdf.ln(2)
        pdf.set_x(LM)
        pdf.set_font("gothic", "B", size)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(W, 7, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def body(text, size=10):
        pdf.set_x(LM)
        pdf.set_font("gothic", "", size)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(W, 6, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def bold_body(text, size=10):
        pdf.set_x(LM)
        pdf.set_font("gothic", "B", size)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(W, 6, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def separator():
        pdf.ln(2)
        pdf.set_draw_color(180, 180, 180)
        y = pdf.get_y()
        pdf.line(LM + W * 0.3, y, LM + W * 0.7, y)
        pdf.ln(4)

    def bullet(text, size=10):
        pdf.set_x(LM + 5)
        pdf.set_font("gothic", "", size)
        pdf.cell(5, 6, "・")
        pdf.multi_cell(W - 10, 6, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def table_header(cols, widths):
        pdf.set_x(LM + 5)
        pdf.set_font("gothic", "B", 9)
        pdf.set_fill_color(40, 60, 100)
        pdf.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            pdf.cell(w, 7, f" {col}", fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

    def table_row(cells, widths, bold_cols=None):
        if bold_cols is None:
            bold_cols = set()
        pdf.set_x(LM + 5)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            pdf.set_font("gothic", "B" if i in bold_cols else "", 9)
            pdf.cell(w, 6, f" {cell}")
        pdf.ln()

    # ================================================================
    # Title
    # ================================================================
    title("「機関の手口」創刊のご挨拶")

    separator()

    # ================================================================
    # About
    # ================================================================
    heading("このマガジンについて")
    body("毎週、機関投資家の空売りデータを独自分析し、"
         "個人投資家が使える形でお届けするマガジンです。")
    body("データソースは金融商品取引法に基づく空売りポジション開示。"
         "毎日数百〜千件を超える膨大な開示情報を、自動収集・分析し、"
         "週次で重要ポイントを抽出します。")

    separator()

    # ================================================================
    # Contents
    # ================================================================
    heading("毎週の配信内容")

    subheading("1. 機関空売り集中銘柄レポート")
    body("複数の機関が同時にショートしている銘柄をランキング形式で掲載。"
         "新規参入・増加・撤退の動きも追跡します。")

    cols = ["銘柄", "機関数", "前週比", "主な機関"]
    ws = [25, 20, 20, 80]
    table_header(cols, ws)
    table_row(["A社", "5社", "+1", "GS, モルガンS, バークレイズ..."], ws, {1})
    table_row(["B社", "4社", "-2", "メリルリンチ, UBS..."], ws, {1})
    pdf.ln(2)

    pdf.set_x(LM)
    pdf.set_font("gothic", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(W, 5, "（上記はイメージです）")
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)

    subheading("2. ショートスクイーズ候補スクリーニング")
    body("空売り比率が高水準に達し、買い戻し圧力が高まっている銘柄をリストアップ。"
         "踏み上げの可能性がある銘柄を需給面から抽出します。")

    subheading("3. セクター別ポジション動向")
    body("機関がどのセクターに弱気か、どこからポジションを引き揚げているかを俯瞰。"
         "新規ショートが集中しているセクターは、機関の見通しを反映します。")

    separator()

    # ================================================================
    # Schedule
    # ================================================================
    heading("配信スケジュール")
    bullet("毎週土曜日に更新")
    bullet("前週の月〜金の開示データを集計・分析")

    separator()

    # ================================================================
    # CTA
    # ================================================================
    heading("第1号は今週土曜に配信")
    body("2026年2月第3週（2/17〜2/21）の機関空売り動向を分析した第1号を、"
         "2/22（土）に配信予定です。")
    body("まずは無料記事「機関投資家は何を空売りしているのか？」で、"
         "データの読み方と直近動向をご確認ください。")

    separator()

    # ================================================================
    # Disclaimer
    # ================================================================
    heading("注意事項")
    bullet("本マガジンは情報提供を目的としたものであり、"
           "特定の銘柄の売買を推奨するものではありません")
    bullet("投資判断は必ずご自身の責任で行ってください")

    separator()
    pdf.set_x(LM)
    pdf.set_font("gothic", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(W, 5,
        "クオンツ軍師 (@quant_gunshi) | データ駆動の株式分析",
        align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
