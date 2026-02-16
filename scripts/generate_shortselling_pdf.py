#!/usr/bin/env python3
"""Generate PDF of note.com short-selling article."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser("~/Desktop/note_shortselling_article.pdf")
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

    def bullet(text, size=10):
        pdf.set_x(LM + 5)
        pdf.set_font("gothic", "", size)
        pdf.cell(5, 6, "・")
        pdf.multi_cell(W - 10, 6, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    # ================================================================
    # Title
    # ================================================================
    title("機関投資家は何を空売りしているのか？")
    pdf.set_x(LM)
    pdf.set_font("gothic", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(W, 7, "— データで読む需給の裏側", align="L",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_text_color(0, 0, 0)

    body("個人投資家が見落としがちな情報がある。機関投資家の空売りポジションだ。")
    body("ゴールドマン・サックス、モルガン・スタンレー、バークレイズ — 彼らが「どの銘柄を」"
         "「どれだけ」空売りしているかは、実は毎日公開されている。"
         "この記事では、そのデータの読み方と、直近の動向を解説する。")

    separator()

    # ================================================================
    # Section 1
    # ================================================================
    heading("空売りポジション開示とは")
    body("金融商品取引法に基づき、発行済株式数の0.5%以上の空売りポジションを保有する場合、"
         "報告義務が発生する。0.2%以上でも内閣府令により報告が必要だ。")
    body("このデータは翌営業日に公開され、誰でも確認できる。しかし、生データは膨大で、"
         "毎日数百〜千件以上の開示がある。個人が手作業で追うのは現実的ではない。")
    bold_body("そこで当アカウントでは、このデータを自動取得・分析し、毎朝Xで速報配信している。")

    separator()

    # ================================================================
    # Section 2
    # ================================================================
    heading("直近データ：2026年2月第2週")

    subheading("開示件数の推移")

    cols = ["日付", "総開示数", "新規", "増加", "減少"]
    ws = [25, 25, 20, 20, 20]
    table_header(cols, ws)
    table_row(["2/13(木)", "813", "6", "262", "544"], ws, {1})
    table_row(["2/12(水)", "1,354", "13", "479", "862"], ws, {1})
    table_row(["2/10(月)", "956", "8", "466", "482"], ws, {1})
    table_row(["2/9(前週金)", "1,312", "11", "511", "790"], ws, {1})
    pdf.ln(3)

    body("2/12は1,354件と週内最多。減少が増加を上回る傾向が続いており、"
         "全体としてショートカバー（空売りの買い戻し）が優勢だ。")

    separator()

    # ================================================================
    # Section 3
    # ================================================================
    subheading("最も活発な機関投資家（2/13時点）")

    cols2 = ["順位", "機関名", "開示件数"]
    ws2 = [15, 70, 25]
    table_header(cols2, ws2)
    table_row(["1", "モルガン・スタンレーMUFG証券", "221件"], ws2, {2})
    table_row(["2", "バークレイズ・キャピタル", "180件"], ws2, {2})
    table_row(["3", "ゴールドマン・サックス", "177件"], ws2, {2})
    table_row(["4", "メリルリンチ", "118件"], ws2, {2})
    table_row(["5", "UBS", "46件"], ws2, {2})
    pdf.ln(3)

    bold_body("モルガン・スタンレーが圧倒的に活発で、全813件中221件（27%）を占める。")
    body("バークレイズとゴールドマンが僅差で続く。上位3社で全体の71%を占めており、"
         "日本株の空売り市場は少数の海外大手に集中していることがわかる。")

    separator()

    # ================================================================
    # Section 4
    # ================================================================
    subheading("機関空売り集中銘柄（2/13時点）")
    body("複数の機関が同時に空売りしている銘柄は、特に注目に値する。")

    cols3 = ["銘柄", "コード", "機関数", "合計比率", "主な機関"]
    ws3 = [28, 15, 15, 18, 80]
    table_header(cols3, ws3)
    table_row(["イメージワン", "2667", "5社", "0.18%", "大和,GS,モルガンS,バークレイズ"], ws3, {2, 3})
    table_row(["KLab", "3656", "4社", "0.12%", "モルガンS,バークレイズ,ベル投資,GS"], ws3, {2, 3})
    table_row(["大阪チタニウム", "5726", "5社", "0.12%", "バークレイズ,UBS,モルガンS,GS"], ws3, {2, 3})
    table_row(["サンリオ", "8136", "4社", "0.11%", "メリルリンチ,GS,モルガンS,UBS"], ws3, {2, 3})
    table_row(["SHIFT", "3697", "4社", "0.09%", "モルガンS,メリルリンチ,GS,UBS"], ws3, {2, 3})
    pdf.ln(3)

    bold_body("イメージ ワン（2667）は5社から空売りされており、週を通じて集中度トップを維持。")
    body("大阪チタニウム（5726）も前週から継続的に機関の関心を集めている。"
         "KLab（3656）は2/12に6社まで増加し、2/13は4社に減少。"
         "一部機関がポジションを縮小している可能性がある。")

    separator()

    # ================================================================
    # Section 5
    # ================================================================
    subheading("新規ポジション（2/13）")
    body("この日、新たに空売りポジションが開示された銘柄：")

    bullet("GS → フェローテック（6890） 0.02% — 半導体関連への新規ショート")
    bullet("UBS → ZETA（6031） 0.005%")
    bullet("バークレイズ → THE WHY HOW DO COMPANY（3823） 0.006%")
    bullet("モルガンS → アスリナ（3647） 0.006%")
    bullet("Jane Street → 夢みつけ隊（2673） 0.005%")
    pdf.ln(1)
    body("フェローテックへのGSの新規ショートは、半導体セクターへの慎重な見方を示唆する。")

    separator()

    # ================================================================
    # Section 6
    # ================================================================
    heading("このデータをどう活用するか")

    subheading("1. 逆張りシグナルとして")
    body("機関の空売りが集中している銘柄は、短期的にはさらなる下落圧力がかかる可能性がある。"
         "しかし空売り比率が十分に高まった銘柄はショートスクイーズ（踏み上げ）の候補にもなる。")

    subheading("2. ポジション解消のタイミング")
    body("「減少」が「増加」を大きく上回る日は、機関がポジションを巻き戻している。"
         "買い戻し圧力は株価の上昇要因だ。")

    subheading("3. セクター動向の先行指標")
    body("特定セクターに新規ショートが集中すれば、機関がそのセクターに弱気であることを意味する。"
         "個別銘柄のファンダメンタルズだけでなく、需給面からのシグナルとして活用できる。")

    separator()

    # ================================================================
    # Section 7
    # ================================================================
    heading("注意点")
    bullet("空売りポジション開示は0.2%以上が対象。それ以下のポジションは見えない")
    bullet("機関の空売りはヘッジ目的の場合も多く、必ずしも弱気を意味しない")
    bullet("データは前営業日時点。リアルタイムではない")
    bullet("本記事は情報提供を目的としたものであり、投資助言ではない")

    separator()

    # ================================================================
    # CTA
    # ================================================================
    heading("毎朝Xで速報配信中")
    body("当アカウント @quant_gunshi では、毎朝7:30に以下を自動配信している：")
    bullet("機関空売り速報 — 新規・増加・減少の主要変動")
    bullet("機関空売り集中銘柄 — 複数機関が同時にショートしている注目銘柄")
    pdf.ln(2)
    bold_body("フォローして需給の裏側を毎日チェックしよう。")
    body("https://x.com/quant_gunshi")

    separator()

    # ================================================================
    # Paid Magazine CTA
    # ================================================================
    heading("有料マガジン「機関空売りウィークリー」")
    body("本記事でお伝えした内容は、あくまで一週間分のスナップショットにすぎない。")
    body("有料マガジンでは、毎週以下を配信している：")
    bullet("機関空売り集中銘柄の週次レポート — 増減推移・新規参入・撤退の全容")
    bullet("ショートスクイーズ候補スクリーニング — 空売り比率が閾値を超えた銘柄リスト")
    bullet("セクター別ポジション動向 — 機関がどのセクターに強気/弱気かの俯瞰図")
    pdf.ln(2)
    bold_body("データは毎週更新。相場の需給を先読みしたい方はこちら。")

    separator()
    pdf.set_x(LM)
    pdf.set_font("gothic", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(W, 5,
        "クオンツ軍師 | データ駆動の株式分析",
        align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
