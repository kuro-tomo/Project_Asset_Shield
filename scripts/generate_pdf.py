#!/usr/bin/env python3
"""Asset Shield 3ヶ月マネタイズ戦略書 PDF生成"""

from fpdf import FPDF
import re

FONT_PATH_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_PATH_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_PATH_W8 = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"
OUTPUT = "/Users/MBP/Desktop/Asset_Shield_3Month_Monetization_Strategy.pdf"

# Colors
NAVY = (26, 42, 82)
DARK = (33, 33, 33)
GRAY = (100, 100, 100)
LIGHT_BG = (245, 247, 250)
WHITE = (255, 255, 255)
ACCENT = (0, 102, 204)
ROW_ALT = (240, 244, 250)
HEADER_BG = (26, 42, 82)
S_COLOR = (220, 38, 38)
A_COLOR = (234, 88, 12)
B_COLOR = (59, 130, 246)
C_COLOR = (156, 163, 175)


class StrategyPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font("HG", "", FONT_PATH_W3, )
        self.add_font("HGB", "", FONT_PATH_W6, )
        self.add_font("HGH", "", FONT_PATH_W8, )
        self.set_auto_page_break(True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("HG", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 6, "Asset Shield — 3ヶ月マネタイズ最終戦略書", align="L")
        self.ln(2)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("HG", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"- {self.page_no()} -", align="C")

    def cover_page(self):
        self.add_page()
        self.ln(50)
        self.set_font("HGH", "", 28)
        self.set_text_color(*NAVY)
        self.cell(0, 14, "Asset Shield", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_font("HGB", "", 20)
        self.cell(0, 12, "3ヶ月マネタイズ最終戦略書", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.5)
        self.line(70, self.get_y(), 140, self.get_y())
        self.ln(12)
        self.set_font("HG", "", 11)
        self.set_text_color(*DARK)
        for line in [
            "報告日: 2026年2月9日",
            "報告者: 老中（Claude Opus 4.6）",
            "宛先: 上様",
            "制約: 日本国内から3ヶ月以内に合法的にキャッシュフローを生む",
        ]:
            self.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(20)

        # Key metrics box
        x0 = 35
        w = 140
        self.set_fill_color(*LIGHT_BG)
        self.set_draw_color(*NAVY)
        self.rect(x0, self.get_y(), w, 48, style="DF")
        self.set_xy(x0, self.get_y() + 4)
        self.set_font("HGB", "", 11)
        self.set_text_color(*NAVY)
        self.cell(w, 7, "Asset Shield V8 Core-Satellite 主要指標", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_x(x0)
        self.ln(2)
        self.set_font("HG", "", 10)
        self.set_text_color(*DARK)
        metrics = [
            ("15年リターン: +649%", "CAGR: 14.4%"),
            ("Sharpe Ratio: 0.92", "Max DD: 25.7%"),
            ("IS Sharpe: 1.10", "OOS Sharpe: 0.74"),
        ]
        for left, right in metrics:
            self.set_x(x0 + 10)
            self.cell(60, 8, left)
            self.cell(60, 8, right, new_x="LMARGIN", new_y="NEXT")

    def section_title(self, text, level=1):
        self.ln(4)
        if level == 1:
            if self.get_y() > 40:
                self.add_page()
            self.set_font("HGH", "", 16)
            self.set_text_color(*NAVY)
            self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*NAVY)
            self.set_line_width(0.6)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("HGB", "", 13)
            self.set_text_color(*NAVY)
            self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font("HGB", "", 11)
            self.set_text_color(*DARK)
            self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font("HGB", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        x0 = self.l_margin
        self.set_x(x0)
        self.cell(6, 6, "・")
        self.multi_cell(180, 6, text)

    def render_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            col_widths = [190 / n] * n

        # Header
        self.set_font("HGB", "", 8.5)
        self.set_fill_color(*HEADER_BG)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("HG", "", 8.5)
        self.set_text_color(*DARK)
        for ri, row in enumerate(rows):
            if ri % 2 == 1:
                self.set_fill_color(*ROW_ALT)
                fill = True
            else:
                self.set_fill_color(*WHITE)
                fill = True
            max_h = 7
            for i, cell in enumerate(row):
                self.cell(col_widths[i], max_h, str(cell), border=1, fill=fill, align="C" if i > 0 else "L")
            self.ln()
        self.ln(3)

    def priority_badge(self, rank, text, revenue, speed, risk):
        colors = {"S": S_COLOR, "A": A_COLOR, "B": B_COLOR, "C": C_COLOR}
        c = colors.get(rank, GRAY)
        self.set_fill_color(*c)
        self.set_text_color(*WHITE)
        self.set_font("HGH", "", 10)
        self.cell(10, 7, rank, fill=True, align="C")
        self.set_fill_color(*WHITE)
        self.set_text_color(*DARK)
        self.set_font("HGB", "", 9.5)
        self.cell(55, 7, f"  {text}")
        self.set_font("HG", "", 9)
        self.cell(45, 7, revenue, align="C")
        self.cell(30, 7, speed, align="C")
        self.cell(30, 7, risk, align="C", new_x="LMARGIN", new_y="NEXT")


def build():
    pdf = StrategyPDF()

    # === COVER ===
    pdf.cover_page()

    # === 総括 ===
    pdf.section_title("総括：老中の結論")
    pdf.bold_text(
        "バイアウト（24ヶ月）を待たず、3ヶ月以内に収益を得る現実的手段を"
        "「収益期待値 × 実現速度 × 法的安全性」で精査した結果、以下の施策を同時並行で推奨いたします。"
    )
    pdf.ln(2)

    # Priority header
    pdf.set_font("HGB", "", 8.5)
    pdf.set_fill_color(*HEADER_BG)
    pdf.set_text_color(*WHITE)
    pdf.cell(10, 7, "級", fill=True, align="C")
    pdf.cell(55, 7, "  施策", fill=True)
    pdf.cell(45, 7, "3ヶ月期待収益", fill=True, align="C")
    pdf.cell(30, 7, "初収益まで", fill=True, align="C")
    pdf.cell(30, 7, "法的リスク", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.priority_badge("S", "フリーランス・クオンツコンサル", "240〜380万円", "2〜4週間", "ゼロ")
    pdf.priority_badge("A", "Numerai Signals提出", "3〜30万円", "5週間", "ゼロ")
    pdf.priority_badge("A", "note.com + Brain有料記事", "10〜80万円", "1〜2週間", "ゼロ")
    pdf.priority_badge("A", "Brain高単価コンテンツ", "10〜50万円", "2〜3週間", "ゼロ")
    pdf.priority_badge("B", "KDP電子書籍（権威付け）", "2〜10万円", "4〜6週間", "ゼロ")
    pdf.priority_badge("B", "SIGNATE/JPXコンペ参加", "0〜200万円", "コンペ次第", "ゼロ")
    pdf.priority_badge("C", "Quantiacs V8変換・提出", "0円（種まき）", "9〜12ヶ月", "ゼロ")

    pdf.ln(4)
    pdf.bold_text("3ヶ月合計の保守的見込み: 170〜700万円")

    # === 第1章 ===
    pdf.section_title("第1章：S級 — フリーランス・クオンツコンサルティング（本命）")

    pdf.section_title("なぜこれが最強か", 2)
    pdf.render_table(
        ["指標", "数値"],
        [
            ["金融系Pythonフリーランス月額平均", "80万円"],
            ["上位エージェント経由（ProConnect等）", "月額127万円"],
            ["3ヶ月稼働の期待収益", "240〜380万円"],
            ["初回入金まで", "最短2〜4週間"],
        ],
        [120, 70],
    )

    pdf.section_title("上様の市場価値", 2)
    pdf.body_text(
        "V8の実績（15年バックテスト、Sharpe 0.92、CAGR 14.4%）は、金融機関のクオンツチームが"
        "喉から手が出るほど欲しいスキルセットそのものである。"
    )
    pdf.bullet("需要: Python + 金融工学 + ファクターモデル + QuantConnect/IB実装")
    pdf.bullet("希少性: 日本国内でこのスキルセットを持つフリーランスは極めて少ない")
    pdf.bullet("法的安全性: 投資助言ではなくソフトウェア開発/コンサルなので金商法の対象外")

    pdf.section_title("具体的アクション（本日〜1週間）", 2)
    pdf.bold_text("1. フリーランスエージェント3社に同時登録")
    pdf.bullet("ProConnect — 金融特化、月額最高水準（127万円）")
    pdf.bullet("Findy Freelance — テック特化、金融案件豊富（98万円）")
    pdf.bullet("レバテックフリーランス — 案件数最大、金融系多数")
    pdf.ln(2)
    pdf.bold_text("2. ポートフォリオとして準備するもの")
    pdf.bullet("V8バックテスト結果のサマリー（Sharpe、CAGR、DD等。戦略ロジックは非開示）")
    pdf.bullet("Python/QuantConnect/IB実装の技術スタック一覧")
    pdf.bullet("ファクターモデル構築経験の概要（IP非開示で実績のみ）")
    pdf.ln(2)
    pdf.bold_text("3. 狙う案件の種類")
    pdf.bullet("証券会社/運用会社のクオンツシステム開発支援")
    pdf.bullet("ファクターモデル/リスクモデル構築")
    pdf.bullet("バックテストフレームワーク開発")
    pdf.bullet("週3〜4日稼働で月80〜100万円が現実的")

    pdf.section_title("IPの保護", 2)
    pdf.body_text(
        "コンサル先での成果物はコンサル先に帰属するが、V8の戦略ロジック自体は一切開示しない。"
        "提供するのは「クオンツ開発の技術力」であり「戦略そのもの」ではない。"
        "NDA締結は当然として、自身のIP（Asset Shield V8）には一切触れない契約とする。"
    )

    # === 第2章 ===
    pdf.section_title("第2章：A級 — Numerai Signals（不労所得パイプライン）")

    pdf.render_table(
        ["項目", "詳細"],
        [
            ["日本居住者参加", "可能（OFAC制裁対象外）"],
            ["KYC", "不要（ウォレットアドレスベース）"],
            ["初回ペイアウト", "アカウント開設から約5週間"],
            ["NMR価格", "約$9.25（≒1,400円）"],
            ["QC統合", "QuantConnect組み込みSignal Export機能で自動提出可能"],
        ],
        [60, 130],
    )

    pdf.section_title("V8→Numerai変換の最大課題", 2)
    pdf.bold_text("Barra中性化: Numerai Signalsは提出シグナルからMomentum・Value等の既知ファクターを中性化する。")
    pdf.body_text(
        "V8の5ファクター（Mom/Short Mom/Low Vol/Value PE/Quality ROE）は標準的Barraファクターと"
        "相関が高く、中性化後にアルファが大幅減衰するリスクがある。"
    )

    pdf.section_title("対策", 3)
    pdf.bullet("レジーム判定をシグナルの重み調整に活用 — CRISIS/BEAR/NEUTRAL/BULL/SUPER_BULLは独自ロジック")
    pdf.bullet("ファクターの相互作用項 — Mom × Quality、Value × Low Vol等の交差項を追加")
    pdf.bullet("時間軸の工夫 — Short Momの窓を調整し、Barraモメンタムとの相関を下げる")

    pdf.section_title("収益試算（保守的）", 2)
    pdf.render_table(
        ["ステーク額", "3ヶ月想定リターン", "3ヶ月収益"],
        [
            ["100 NMR (≒14万円)", "+15〜30%", "2〜4万円"],
            ["500 NMR (≒70万円)", "+15〜30%", "10〜20万円"],
            ["1000 NMR (≒140万円)", "+15〜30%", "20〜40万円"],
        ],
        [70, 60, 60],
    )

    pdf.section_title("税務上の重要注意", 3)
    pdf.bold_text("2026年中のNMR報酬は暗号資産として雑所得扱い → 最高税率55.945%")
    pdf.body_text(
        "2028年から申告分離課税20.315%に改正予定。"
        "初期は少額ステーク（100〜500 NMR）で検証し、2028年以降に本格化が合理的。"
    )

    # === 第3章 ===
    pdf.section_title("第3章：A級 — note.com + Brain 有料記事（ブランド構築 + 即金）")

    pdf.section_title("成功事例（日本のクオンツ）", 2)
    pdf.render_table(
        ["人物", "実績", "収益化手法"],
        [
            ["richmanbtc", "年間利益10億円級botter", "note有料記事 + Amazon書籍ベストセラー1位"],
            ["UKI", "累計利益1.7億円", "note有料記事が機関投資家からも高評価"],
            ["Quant College", "フォロワー4万人", "17テーマ以上の有料レクチャーシリーズ"],
        ],
        [35, 55, 100],
    )

    pdf.section_title("プラットフォーム二段構成", 2)
    pdf.render_table(
        ["項目", "note.com", "Brain"],
        [
            ["手数料", "15%", "12%"],
            ["紹介機能", "なし", "あり（アフィリエイト拡散）"],
            ["推奨価格帯", "980〜5,000円", "9,800〜19,800円"],
            ["用途", "入門・集客", "本編・高単価コンテンツ"],
        ],
        [50, 70, 70],
    )
    pdf.body_text("noteで入門記事（無料〜1,980円）→ Brainで本編（9,800〜19,800円）への導線を構築。")

    pdf.section_title("上様が書くべきコンテンツ（IP非開示）", 2)
    pdf.bold_text("シリーズ名案:「15年で+649%を実現した クオンツ戦略構築の実践知」")
    pdf.render_table(
        ["記事", "内容", "価格"],
        [
            ["第1回", "ファクター投資の基礎 — なぜマルチファクターか", "無料（集客）"],
            ["第2回", "バックテストの罠 — オーバーフィットを防ぐ技術", "980円"],
            ["第3回", "Walk-Forward検証の実装（IS/OOS分割）", "1,980円"],
            ["第4回", "レジーム判定とリスク管理 — Kill-switchの設計思想", "2,980円"],
            ["第5回", "Core-Satellite構造 — なぜ純粋ファクターは負けるか", "2,980円"],
            ["第6回", "QuantConnect実装入門", "1,980円"],
            ["マガジン", "全記事バンドル（Brain）", "9,800円"],
        ],
        [25, 115, 50],
    )

    pdf.section_title("法的注意事項（厳守）", 3)
    pdf.bullet("具体的銘柄の売買推奨は厳禁（金商法違反: 懲役5年/罰金500万円）")
    pdf.bullet("買い切り型のみ: noteメンバーシップ（月額課金）は投資助言業に該当する可能性 → 絶対に避ける")
    pdf.bullet("冒頭に「本コンテンツは投資教育目的であり、投資助言ではありません」の免責事項を明記")

    # === 第4章 ===
    pdf.section_title("第4章：明確に排除した施策とその理由")
    pdf.render_table(
        ["施策", "排除理由"],
        [
            ["有料シグナル配信", "金商法違反（懲役5年/罰金500万円）。登録に700万円+6ヶ月〜1年"],
            ["WorldQuant BRAIN", "日本は対象国外（確認済み）。IQC参加可だが継続報酬なし"],
            ["YouTube", "収益化まで5ヶ月+。3ヶ月では間に合わない"],
            ["独立ファンド設立", "初年度$500K〜$1.7M。3ヶ月の制約に合致しない"],
            ["GogoJungle EA販売", "主力がFX EA市場。米国株ファクターモデルとの親和性が低い"],
            ["Quantiacs (低優先)", "$1M配分でも年$10K。V8の構造的強みが移植で消失。活況度低下"],
        ],
        [55, 135],
    )

    # === 第5章 ===
    pdf.section_title("第5章：90日アクションプラン")

    pdf.section_title("Week 1（2/10〜2/16）— 全面展開開始", 2)
    pdf.render_table(
        ["曜日", "アクション"],
        [
            ["月", "フリーランスエージェント3社登録（ProConnect, Findy, レバテック）"],
            ["月", "note.comアカウント作成、第1回記事（無料）の執筆開始"],
            ["火", "Numeraiアカウント作成、NMR購入（取引所口座が必要）"],
            ["水", "X (Twitter) アカウント開設/整備、クオンツ系フォロー開始"],
            ["木", "note第1回記事公開"],
            ["金", "Numeraiデータ分析開始、V8シグナル変換パイプライン設計"],
            ["土日", "note第2回記事（有料）執筆 / Brain本編の構成"],
        ],
        [20, 170],
    )

    pdf.section_title("Week 2〜4（2/17〜3/9）— 収益の種を育てる", 2)
    pdf.bullet("フリーランス案件の面談・選定（初回稼働開始目標: 3月第2週）")
    pdf.bullet("note記事を週1本ペースで公開（第2〜5回）")
    pdf.bullet("Brain本編コンテンツ制作・リリース（9,800〜19,800円）")
    pdf.bullet("Numeraiシグナル変換パイプライン完成、ステークなし提出で検証")
    pdf.bullet("SIGNATE/Kaggle開催中コンペの確認・参加")
    pdf.bullet("KDP電子書籍の構成案作成・執筆開始")

    pdf.section_title("Month 2（3/10〜4/9）— 最初のキャッシュフロー", 2)
    pdf.bullet("フリーランス初月稼働 → 月末に80〜127万円の請求書発行")
    pdf.bullet("note記事の購読者数確認、マガジン化検討")
    pdf.bullet("Numerai少額ステーク開始（10〜50 NMR）")
    pdf.bullet("KDP電子書籍出版（48時間で販売開始可能）")

    pdf.section_title("Month 3（4/10〜5/9）— 収穫期", 2)
    pdf.bullet("フリーランス2ヶ月目入金（初月分80〜127万円が着金）")
    pdf.bullet("フリーランス2ヶ月目稼働 → 累計請求160〜254万円")
    pdf.bullet("note + Brain累計収益の確認・最適化")
    pdf.bullet("Numerai初回ペイアウト確認")

    # === 第6章 ===
    pdf.section_title("第6章：収益シミュレーション（3ヶ月累計）")

    pdf.section_title("保守的シナリオ", 2)
    pdf.render_table(
        ["施策", "3ヶ月収益"],
        [
            ["フリーランス（月80万×2ヶ月稼働+1ヶ月準備）", "160万円"],
            ["note.com + Brain有料記事", "10万円"],
            ["Numerai", "2万円"],
            ["合計", "172万円"],
        ],
        [130, 60],
    )

    pdf.section_title("中間シナリオ", 2)
    pdf.render_table(
        ["施策", "3ヶ月収益"],
        [
            ["フリーランス（月100万×2.5ヶ月）", "250万円"],
            ["note.com + Brain有料記事", "30万円"],
            ["Numerai", "10万円"],
            ["合計", "290万円"],
        ],
        [130, 60],
    )

    pdf.section_title("楽観シナリオ", 2)
    pdf.render_table(
        ["施策", "3ヶ月収益"],
        [
            ["フリーランス（月127万×3ヶ月）", "381万円"],
            ["note.com + Brain有料記事", "80万円"],
            ["Numerai", "30万円"],
            ["SIGNATE入賞", "100万円"],
            ["合計", "591万円"],
        ],
        [130, 60],
    )

    # === 結論 ===
    pdf.section_title("結論")
    pdf.ln(2)
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*NAVY)
    y0 = pdf.get_y()
    pdf.rect(15, y0, 180, 52, style="DF")
    pdf.set_xy(20, y0 + 4)
    pdf.set_font("HGB", "", 11)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(170, 7,
        "3ヶ月以内に日本国内で合法的にマネタイズする最善手は、"
        "フリーランス・クオンツコンサルティングを主軸に、"
        "Numerai + note.com/Brain有料記事を並行させる三刀流である。"
    )
    pdf.set_x(20)
    pdf.set_font("HG", "", 10)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(170, 6,
        "フリーランスは即座にキャッシュを生み、Numeraiは不労所得パイプラインの礎を築き、"
        "note.com/Brainはブランドと長期収益基盤を構築する。"
        "いずれもV8の戦略IPを開示する必要がなく、投資助言業登録も不要である。"
    )
    pdf.ln(8)

    pdf.body_text(
        "本戦略は、24ヶ月後のバイアウトに向けたトラックレコード構築（QCライブ運用）と完全に両立する。"
    )

    pdf.ln(10)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("HG", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "本報告書は Asset Shield 老中（Claude Opus 4.6）が全4班の調査結果を統合し作成", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "調査班: Numerai班 / Quantiacs班 / 国内合法マネタイズ班 / 日本クオンツ成功事例班", align="C")

    pdf.output(OUTPUT)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build()
