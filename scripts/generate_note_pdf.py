#!/usr/bin/env python3
"""Generate PDF preview of note.com article draft."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser("~/Desktop/note_article_draft_01.pdf")
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
        pdf.multi_cell(W, 9, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    def heading(text, size=13):
        pdf.ln(3)
        pdf.set_x(LM)
        pdf.set_font("gothic", "B", size)
        pdf.multi_cell(W, 8, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def subheading(text, size=11):
        pdf.ln(2)
        pdf.set_x(LM)
        pdf.set_font("gothic", "B", size)
        pdf.multi_cell(W, 7, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def body(text, size=10):
        pdf.set_x(LM)
        pdf.set_font("gothic", "", size)
        pdf.multi_cell(W, 6, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def bold_body(text, size=10):
        pdf.set_x(LM)
        pdf.set_font("gothic", "B", size)
        pdf.multi_cell(W, 6, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def separator():
        pdf.ln(2)
        pdf.set_draw_color(180, 180, 180)
        y = pdf.get_y()
        pdf.line(LM + W * 0.3, y, LM + W * 0.7, y)
        pdf.ln(4)

    def table_row(col1, col2, bold_val=False):
        pdf.set_x(LM + 10)
        pdf.set_font("gothic", "", 10)
        pdf.cell(60, 6, col1)
        pdf.set_font("gothic", "B" if bold_val else "", 10)
        pdf.cell(0, 6, col2, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # === Title ===
    title("S&P500に勝てなかったマルチファクターを捨てた話")

    bold_body("7回、戦略を壊しました。")

    body(
        "数ヶ月かけて作ったアルゴリズムが、検証した瞬間に崩壊する。"
        "その絶望を7回繰り返して、8代目でようやく「これだ」と思える構造に辿り着きました。"
    )
    body(
        "米国株のシステマティック運用を個人で開発しています。"
        "この記事では、具体的なパラメータや売買ロジックは公開しませんが、"
        "何を間違え、なぜ失敗し、どう考え方を変えたのか──その思考プロセスを書きます。"
    )

    separator()
    heading("なぜ「個人で」クオンツ戦略を作るのか")
    body(
        "機関投資家でもヘッジファンドでもない、一個人がアルゴリズム運用を開発する。"
        "正直、周囲に話しても「なんで？」という反応がほとんどです。"
    )
    body(
        "理由はシンプルで、自分の資産を、自分が納得できるロジックで運用したかったから。"
    )
    body(
        "感覚的な売買では、暴落のたびに狼狽し、上昇相場では根拠なく強気になる。"
        "その繰り返しから脱却するには、ルールベースで、統計的に検証された仕組みが必要だと考えました。"
    )

    separator()
    heading("V1〜V7：失敗の歴史")

    subheading("初期（V1〜V3）：「ファクターを増やせば勝てる」という幻想")
    body(
        "最初は学術論文で有名なファクター──モメンタム、バリュー、クオリティ──を"
        "片っ端から実装しました。バックテストは美しい右肩上がり。「これはいける」と確信しました。"
    )
    bold_body("しかし、ウォークフォワード検証をかけた瞬間、すべてが崩壊しました。")
    body(
        "イン・サンプル（学習期間）では素晴らしい成績なのに、アウト・オブ・サンプル（未知の期間）"
        "ではベンチマークに大きく劣後する。典型的なオーバーフィッティングです。"
    )
    body("ここで学んだ教訓：バックテストの数字は、それ単体では何の意味もない。")

    subheading("中期（V4〜V5）：パラメータ地獄")
    body(
        "次に「パラメータを最適化すれば安定する」と考えました。"
        "リバランス頻度、ファクターの閾値、ポジションサイズ──膨大な組み合わせを回し、"
        "最良のパラメータセットを追い求めました。"
    )
    body(
        "結果は散々でした。パラメータを変えるたびに成績が大きく変動する。"
        "パラメータに対してロバスト（頑健）でない戦略は、そもそも構造に欠陥がある"
        "ということに気づくまで、数ヶ月を浪費しました。"
    )

    subheading("転換点（V6）：構造的な壁に激突")
    body(
        "V6は、私の中で最も完成度が高いと思っていた戦略でした。"
        "複数ファクターを組み合わせたマルチファクター・ロングオンリー戦略。"
        "ウォークフォワードも通過し、いよいよ実運用かと意気込みました。"
    )
    bold_body("ところが、S&P\u00a0500に大幅に負けていたのです。")
    body(
        "ベンチマークの半分程度のリターンしか出せない。ファクター投資の理論は正しい。"
        "実装にもバグはない。それでもベンチマークに勝てない。"
    )
    body(
        "この瞬間、私は「パラメータの問題ではなく、"
        "アーキテクチャ（戦略構造）の問題だ」と理解しました。"
    )

    separator()
    heading("V8：構造で解く")
    body(
        "V6の敗因を分析して見えたのは、ファクター・ロングオンリー戦略の構造的限界でした。"
    )
    body(
        "どれだけ優れたファクターを組み合わせても、ロングオンリーでは市場のベータを"
        "十分に取り込めない。特に2010年代の米国株のような強い上昇相場では、"
        "銘柄選択のアルファだけでは市場についていけないのです。"
    )
    bold_body("答えは「コア・サテライト」という構造にありました。")
    body("・コア：市場全体のベータを確実に取り込むインデックス")
    body("・サテライト：マルチファクターモデルで超過リターンを狙う")
    body(
        "この二層構造により、「市場が上がれば恩恵を受け、かつファクターの力で超過リターンも狙える」"
        "という、V6では絶対に実現できなかった特性を手に入れました。"
    )
    body(
        "さらに、レジーム検知（市場環境の自動判定）とキルスイッチ（損失拡大時の自動防御機構）"
        "を組み込むことで、暴落局面での致命傷を防ぐ仕組みも実装しています。"
    )

    separator()
    heading("15年バックテスト（2010〜2024年）")
    body("戦略の詳細は非公開ですが、検証結果の概要は以下の通りです：")
    pdf.ln(2)

    # Table header
    pdf.set_x(LM + 10)
    pdf.set_font("gothic", "B", 10)
    pdf.set_fill_color(40, 60, 100)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(60, 7, "  指標", fill=True)
    pdf.cell(50, 7, "  数値", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)

    table_row("累積リターン", "+649%", True)
    table_row("年率リターン（CAGR）", "14.4%", True)
    table_row("シャープレシオ", "0.92", True)
    table_row("最大ドローダウン", "25.7%", True)
    table_row("IS\u00a0シャープレシオ", "1.10", True)
    table_row("OOS\u00a0シャープレシオ", "0.74", True)
    pdf.ln(3)

    body(
        "特に注目していただきたいのは、ISとOOSのシャープレシオの差です。"
    )
    body(
        "IS\u00a01.10に対してOOS\u00a00.74。確かに劣化はありますが、OOS単体でも0.74は実用に耐えうる水準です。"
        "V1〜V5では、この差が絶望的に大きかった。V8で初めて「ウォークフォワードを通過しても、"
        "まだ使える戦略」になりました。"
    )
    body(
        "最大ドローダウン25.7%はCOVID-19ショック（2020年3月）によるもので、"
        "コア部分がベンチマークと連動して下落した結果です。"
        "戦略のバグではなく構造上の想定内です。"
    )

    separator()
    heading("7世代の失敗から学んだこと")

    lessons = [
        ("1. バックテストを信じるな、ウォークフォワードを信じろ",
         "過去データに完璧にフィットする戦略は、未来には使えない。"),
        ("2. パラメータの最適化では構造の欠陥は治らない",
         "戦略がパラメータ変動に敏感なら、それはチューニングではなく設計の問題。"),
        ("3. ベンチマークに勝てない戦略に、存在価値はない",
         "S&P\u00a0500のインデックスファンドを買うだけで得られるリターンを超えられないなら、その戦略を動かす意味はない。"),
        ("4. ベータを軽視するな",
         "ファクター投資の文献はアルファの話ばかりだが、個人にとって最大のリターン源泉はベータ。まずベータを確保し、その上にアルファを積む。"),
        ("5. 防御機構はインフラ",
         "キルスイッチやドローダウン管理は「あったらいいな」ではなく、必須。生き残ることが最優先。"),
    ]
    for title_text, desc_text in lessons:
        bold_body(title_text)
        body(desc_text)

    separator()
    heading("次回")
    body(
        "「なぜファクター投資は論文通りにいかないのか」── "
        "アカデミアの理論と実装の間に横たわるギャップについて、"
        "失敗を具体例に掘り下げます。"
    )

    separator()
    pdf.set_x(LM)
    pdf.set_font("gothic", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(W, 5,
        "米国株システマティック運用を個人で開発・運用中。QuantConnect/LEAN + Interactive Brokers。",
        align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
