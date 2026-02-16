#!/usr/bin/env python3
"""Asset Shield 総合マネタイズ報告書 PDF生成（全章統合版）"""

from fpdf import FPDF

FONT_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_W8 = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"
OUTPUT = "/Users/MBP/Desktop/Asset_Shield_Full_Report.pdf"

NAVY = (26, 42, 82)
DARK = (33, 33, 33)
GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
ACCENT = (0, 102, 204)
LIGHT = (245, 247, 250)
LINE_C = (200, 200, 200)
RED = (180, 40, 40)
GREEN = (30, 120, 60)
WARN = (200, 120, 0)


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font("HG", "", FONT_W3)
        self.add_font("HGB", "", FONT_W6)
        self.add_font("HGH", "", FONT_W8)
        self.set_auto_page_break(True, margin=18)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("HG", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 5, "Asset Shield — 総合マネタイズ報告書", align="R")
        self.ln(3)
        self.set_draw_color(*LINE_C)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("HG", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 8, f"{self.page_no()} / {{nb}}", align="C")

    def chapter_title(self, num, title):
        self.ln(4)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font("HGH", "", 13)
        label = f"  第{num}章　{title}" if num else f"  {title}"
        self.cell(0, 10, label, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_text_color(*DARK)

    def section(self, title):
        self.ln(3)
        self.set_font("HGB", "", 11)
        self.set_text_color(*NAVY)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 90, self.get_y())
        self.ln(3)
        self.set_text_color(*DARK)

    def body(self, text):
        self.set_font("HG", "", 9)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bold_body(self, text):
        self.set_font("HGB", "", 9)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("HG", "", 9)
        self.set_text_color(*DARK)
        x0 = self.l_margin
        self.set_x(x0 + 4)
        self.cell(5, 5.5, "・")
        self.multi_cell(176, 5.5, text)

    def kv(self, key, val):
        self.set_font("HGB", "", 9)
        self.cell(50, 6, key)
        self.set_font("HG", "", 9)
        self.cell(0, 6, val, new_x="LMARGIN", new_y="NEXT")

    def warn_box(self, text):
        self.ln(2)
        self.set_fill_color(255, 245, 230)
        self.set_draw_color(*WARN)
        self.set_line_width(0.4)
        y0 = self.get_y()
        self.set_font("HGB", "", 9)
        self.set_text_color(*WARN)
        self.set_x(self.l_margin + 2)
        self.multi_cell(182, 5.5, f"⚠ {text}", fill=True)
        self.set_draw_color(*WARN)
        self.line(10, y0, 10, self.get_y())
        self.ln(2)
        self.set_text_color(*DARK)

    def red_box(self, text):
        self.ln(2)
        self.set_fill_color(255, 235, 235)
        self.set_draw_color(*RED)
        self.set_line_width(0.4)
        y0 = self.get_y()
        self.set_font("HGB", "", 9)
        self.set_text_color(*RED)
        self.set_x(self.l_margin + 2)
        self.multi_cell(182, 5.5, text, fill=True)
        self.line(10, y0, 10, self.get_y())
        self.ln(2)
        self.set_text_color(*DARK)

    def table_header(self, cols, widths):
        self.set_font("HGB", "", 8)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        for i, col in enumerate(cols):
            self.cell(widths[i], 7, col, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(*DARK)

    def table_row(self, cells, widths, bold_first=False):
        self.set_font("HG", "", 8)
        self.set_text_color(*DARK)
        for i, cell in enumerate(cells):
            if i == 0 and bold_first:
                self.set_font("HGB", "", 8)
            else:
                self.set_font("HG", "", 8)
            self.cell(widths[i], 6.5, cell, border=1, align="C" if i > 0 else "L")
        self.ln()


def build():
    pdf = ReportPDF()
    pdf.alias_nb_pages()

    # ===== COVER PAGE =====
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("HGH", "", 26)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 14, "Asset Shield", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("HGH", "", 18)
    pdf.cell(0, 12, "総合マネタイズ報告書", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("HGB", "", 11)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 8, "3ヶ月以内の国内合法マネタイズ戦略", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "希望単価最適化分析", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "レジュメ・LinkedIn戦略", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "収益構造の根本的再設計", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("HG", "", 9)
    pdf.cell(0, 6, "報告日: 2026年2月10日", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "報告者: 老中（Claude Opus 4.6）", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "宛先: 上様", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("HG", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "CONFIDENTIAL — 本報告書は営業秘密を含みます", align="C")

    # ===== TABLE OF CONTENTS =====
    pdf.add_page()
    pdf.chapter_title(0, "目次")
    toc = [
        ("第1章", "老中の根本的反省 — 労働売却 vs 資本収益"),
        ("第2章", "収益構造の再設計 — 上様の時間を拘束しない戦略"),
        ("第3章", "S級: V8ライブ運用（自己資本による完全自動収益）"),
        ("第4章", "A級: Numerai Signals（半自動・不労所得パイプライン）"),
        ("第5章", "A級: コンテンツ販売（note.com + Brain）"),
        ("第6章", "B級: コンペティション（SIGNATE / JPX / Kaggle）"),
        ("第7章", "C級: Quantiacs（長期種まき）"),
        ("第8章", "保険: フリーランスコンサル（週1〜2日限定）"),
        ("第9章", "希望単価の最適解（市場調査結果）"),
        ("第10章", "レジュメ・LinkedIn戦略"),
        ("第11章", "排除した施策と理由"),
        ("第12章", "90日アクションプラン（改訂版）"),
        ("第13章", "収益シミュレーション（改訂版）"),
        ("第14章", "上様へのご確認事項"),
    ]
    for num, title in toc:
        pdf.set_font("HGB", "", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(20, 7, num)
        pdf.set_font("HG", "", 9)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")

    # ===== CHAPTER 1: REFLECTION =====
    pdf.add_page()
    pdf.chapter_title(1, "老中の根本的反省 — 労働売却 vs 資本収益")

    pdf.section("過ちの認識")
    pdf.body(
        "上様が Asset Shield を築かれた本質的な目的は「自分の代わりに金を稼ぐ仕組み」の構築である。"
        "にもかかわらず、老中は初回報告でフリーランスコンサルティング（労働の時間売り）を"
        "S級＝最優先と位置付けた。これは本末転倒の重大な判断ミスである。"
    )

    pdf.section("収益構造の3類型")
    w = [50, 45, 45, 50]
    pdf.table_header(["区分", "上様の拘束時間", "スケーラビリティ", "V8との整合性"], w)
    pdf.table_row(["労働売却（コンサル）", "週3〜5日", "なし（時間の切り売り）", "低い（本業が遅れる）"], w)
    pdf.table_row(["IP収益（コンテンツ）", "執筆時のみ", "あり（売り切り型）", "中（知見を換金）"], w)
    pdf.table_row(["資本収益（ライブ運用）", "ゼロ（完全自動）", "資本に比例", "最高（V8の本懐）"], w)

    pdf.ln(2)
    pdf.bold_body("結論: 資本収益（V8ライブ運用）を最上位に据え、労働売却は最終手段の保険に格下げする。")

    # ===== CHAPTER 2: REDESIGN =====
    pdf.add_page()
    pdf.chapter_title(2, "収益構造の再設計 — 上様の時間を拘束しない戦略")

    pdf.section("改訂優先順位")
    w = [15, 60, 35, 40, 40]
    pdf.table_header(["優先度", "手段", "想定月収", "上様の稼働", "初収益まで"], w)
    pdf.table_row(["S", "V8ライブ運用（自己資金）", "資本次第", "ゼロ", "Live移行完了次第"], w, True)
    pdf.table_row(["A", "Numerai Signals", "2〜10万円", "週30分", "4週間"], w, True)
    pdf.table_row(["A", "Brain 買い切り記事", "5〜30万円", "執筆時のみ", "2週間"], w, True)
    pdf.table_row(["A", "note.com 有料記事", "1〜5万円", "同上", "1週間"], w, True)
    pdf.table_row(["B", "SIGNATE/JPX/Kaggle", "0〜200万円", "コンペ期間のみ", "コンペ次第"], w, True)
    pdf.table_row(["C", "Quantiacs", "0円（種まき）", "初期構築のみ", "9〜12ヶ月"], w, True)
    pdf.table_row(["保険", "コンサル（週1〜2日限定）", "40〜60万円", "週1〜2日", "2〜4週間"], w, True)

    pdf.ln(2)
    pdf.body(
        "コンサルは「保険」に格下げ。実行する場合でも週1〜2日に厳格に制限し、"
        "残りの時間はV8ライブ移行・Numerai提出・コンテンツ執筆に充てる。"
    )

    pdf.warn_box(
        "重要: S級（V8ライブ運用）の期待収益は投入自己資金の規模に完全依存する。"
        "この数値により全体の方針が根本的に変わるため、第14章で上様にご確認を仰ぐ。"
    )

    # ===== CHAPTER 3: V8 LIVE =====
    pdf.add_page()
    pdf.chapter_title(3, "S級: V8ライブ運用（自己資本による完全自動収益）")

    pdf.section("なぜこれが最上位か")
    pdf.bullet("上様の稼働時間: ゼロ（完全自動）")
    pdf.bullet("V8の本来の目的そのもの（自分の代わりに稼ぐ仕組み）")
    pdf.bullet("スケーラビリティ: 投入資本に比例（上限はマーケットインパクトのみ）")
    pdf.bullet("15年バックテスト検証済み: CAGR 14.4%, Sharpe 0.92, WF IS 1.10 / OOS 0.74")
    pdf.bullet("IPの完全保持（第三者への開示一切不要）")

    pdf.section("ライブ移行のステップ")
    pdf.kv("Step 1", "QuantConnect上でPaper Trading開始（現在のステータス）")
    pdf.kv("Step 2", "Paper Tradingで1〜3ヶ月間の動作検証（注文執行・レジーム判定・Kill-switch）")
    pdf.kv("Step 3", "Interactive Brokers口座開設・入金")
    pdf.kv("Step 4", "QC→IB接続によるLive移行")
    pdf.kv("Step 5", "初期は少額（全資金の20〜30%）で運用開始、段階的に増額")

    pdf.section("資金規模別の期待収益（年率ベース）")
    w = [40, 35, 35, 40, 40]
    pdf.table_header(["投入資金", "CAGR 14.4%", "年間期待収益", "月間期待収益", "3ヶ月期待収益"], w)
    pdf.table_row(["100万円", "14.4%", "14.4万円", "1.2万円", "3.6万円"], w)
    pdf.table_row(["500万円", "14.4%", "72万円", "6万円", "18万円"], w)
    pdf.table_row(["1,000万円", "14.4%", "144万円", "12万円", "36万円"], w)
    pdf.table_row(["3,000万円", "14.4%", "432万円", "36万円", "108万円"], w)
    pdf.table_row(["5,000万円", "14.4%", "720万円", "60万円", "180万円"], w)
    pdf.table_row(["1億円", "14.4%", "1,440万円", "120万円", "360万円"], w)

    pdf.ln(2)
    pdf.warn_box(
        "注意: 上記はバックテストCAGRに基づく期待値であり、ライブ運用では"
        "スリッページ・マーケットインパクト・実行コストにより2〜5%程度の減衰が一般的。"
        "また、Max DD 25.7%（主にCOVID級暴落時のSPY Core由来）への耐性が必要。"
    )

    pdf.section("リスク管理")
    pdf.bullet("Kill-switch: DD 15%でSatellite自動清算。SPY Core維持")
    pdf.bullet("初期投入は全資金の20〜30%に限定（残りは段階的増額）")
    pdf.bullet("Paper Trading期間中にLive環境固有の問題を洗い出し")
    pdf.bullet("Email通知システムで異常を即時検知")

    # ===== CHAPTER 4: NUMERAI =====
    pdf.add_page()
    pdf.chapter_title(4, "A級: Numerai Signals（半自動・不労所得パイプライン）")

    pdf.section("概要")
    w = [50, 140]
    pdf.table_header(["項目", "詳細"], w)
    pdf.table_row(["日本居住者参加", "可能（OFAC制裁対象外）"], w)
    pdf.table_row(["KYC", "不要（ウォレットアドレスベース）"], w)
    pdf.table_row(["初回ペイアウト", "アカウント開設から約5週間"], w)
    pdf.table_row(["NMR価格", "約$9.25（≒1,400円）"], w)
    pdf.table_row(["上様の稼働", "初期構築後は週30分（Signal提出の確認）"], w)
    pdf.table_row(["QC統合", "QuantConnect組み込みのSignal Export機能で自動提出可能"], w)

    pdf.section("V8→Numerai変換の課題")
    pdf.body(
        "Numerai SignalsはBarra中性化を適用し、Momentum・Value等の既知ファクターの効果を除去する。"
        "V8の5ファクターは標準的Barraファクターとの相関が高く、中性化後にアルファが大幅減衰するリスクがある。"
    )

    pdf.section("対策")
    pdf.bullet("レジーム判定をシグナルの重み調整に活用（独自ロジック、Barraに非存在）")
    pdf.bullet("ファクター交差項の追加（Mom × Quality、Value × Low Vol等）")
    pdf.bullet("Short Momの窓を調整し、標準Barraモメンタムとの相関を低減")

    pdf.section("収益試算")
    w = [50, 50, 45, 45]
    pdf.table_header(["ステーク額", "NMR換算", "3ヶ月想定リターン", "3ヶ月収益"], w)
    pdf.table_row(["14万円", "100 NMR", "+15〜30%", "2〜4万円"], w)
    pdf.table_row(["70万円", "500 NMR", "+15〜30%", "10〜20万円"], w)
    pdf.table_row(["140万円", "1000 NMR", "+15〜30%", "20〜40万円"], w)

    pdf.warn_box(
        "税務注意: 2026年中のNMR報酬は暗号資産として雑所得扱い → 最高税率55.945%。"
        "2028年から申告分離課税20.315%に改正予定。初期は少額ステーク（100〜500 NMR）で検証し、"
        "2028年以降に本格化が合理的。"
    )

    # ===== CHAPTER 5: CONTENT =====
    pdf.add_page()
    pdf.chapter_title(5, "A級: コンテンツ販売（note.com + Brain）")

    pdf.section("二段構成戦略")
    w = [40, 50, 50, 50]
    pdf.table_header(["項目", "note.com", "Brain", "使い分け"], w)
    pdf.table_row(["手数料", "15%", "12%", "Brainが有利"], w)
    pdf.table_row(["紹介機能", "なし", "あり（アフィリエイト拡散）", "Brain独自の強み"], w)
    pdf.table_row(["推奨価格帯", "無料〜2,980円", "9,800〜19,800円", "noteで集客→Brainで本編"], w)
    pdf.table_row(["上様の稼働", "1本3〜5時間", "1本5〜10時間", "執筆時のみ（不労収益化）"], w)

    pdf.section("コンテンツ構成案")
    pdf.body("シリーズ名: 「15年で+649%を実現した クオンツ戦略構築の実践知」")
    pdf.ln(1)
    w2 = [15, 80, 25, 30, 40]
    pdf.table_header(["回", "内容", "価格", "媒体", "法的安全性"], w2)
    pdf.table_row(["1", "ファクター投資の基礎 — なぜマルチファクターか", "無料", "note", "一般論→合法"], w2)
    pdf.table_row(["2", "バックテストの罠 — オーバーフィットを防ぐ技術", "980円", "note", "手法論→合法"], w2)
    pdf.table_row(["3", "Walk-Forward検証の実装（IS/OOS分割）", "1,980円", "note", "技術教育→合法"], w2)
    pdf.table_row(["4", "レジーム判定とリスク管理 — Kill-switchの設計思想", "2,980円", "note", "設計思想→合法"], w2)
    pdf.table_row(["5", "Core-Satellite構造 — なぜ純粋ファクターは負けるか", "2,980円", "note", "教訓→合法"], w2)
    pdf.table_row(["6", "QuantConnect実装入門", "1,980円", "note", "ツール解説→合法"], w2)
    pdf.table_row(["全編", "マガジン/バンドル", "9,800円", "Brain", "買い切り→合法"], w2)

    pdf.red_box(
        "厳禁: 具体的銘柄の売買推奨（金商法違反: 懲役5年/罰金500万円）。"
        "月額サブスクリプションでの投資情報提供は投資助言業に該当する可能性 → 買い切り型のみ。"
    )

    pdf.section("収益試算")
    w = [30, 35, 35, 45, 45]
    pdf.table_header(["シナリオ", "note購読者/月", "Brain購入者/月", "月収", "3ヶ月累計"], w)
    pdf.table_row(["保守的", "30人", "5人", "5〜8万円", "15〜25万円"], w)
    pdf.table_row(["中間", "100人", "15人", "15〜30万円", "45〜90万円"], w)
    pdf.table_row(["楽観的", "300人", "40人", "40〜80万円", "120〜240万円"], w)

    pdf.section("成功事例（日本のクオンツ）")
    pdf.bullet("richmanbtc — 年間利益10億円級botter。有料記事販売 + Amazon書籍ベストセラー1位")
    pdf.bullet("UKI — 累計利益1.7億円。「保存版 株のトレーディング手法まとめ」が機関投資家からも高評価")
    pdf.bullet("Quant College — フォロワー4万人。17テーマ以上の有料レクチャーシリーズ")

    # ===== CHAPTER 6: COMPETITIONS =====
    pdf.add_page()
    pdf.chapter_title(6, "B級: コンペティション（SIGNATE / JPX / Kaggle）")

    w = [60, 40, 45, 45]
    pdf.table_header(["コンペ", "賞金", "V8との親和性", "備考"], w)
    pdf.table_row(["JPXファンダメンタルズ（SIGNATE）", "50〜200万円", "極めて高い", "ファクターモデルそのもの"], w)
    pdf.table_row(["Jane Street（Kaggle）", "$120,000", "中", "スキル転用可"], w)
    pdf.table_row(["SIGNATE その他金融系", "10〜100万円", "高い", "開催次第"], w)

    pdf.ln(2)
    pdf.body("開催時期はコントロール不能。開催中のコンペがあれば即参加の方針。上様のファクターモデル構築力は上位入賞の十分な素地がある。")

    # ===== CHAPTER 7: QUANTIACS =====
    pdf.chapter_title(7, "C級: Quantiacs（長期種まき）")
    pdf.body(
        "配分報酬は純利益の10%。$1M配分でCAGR 14.4%の場合、年間$14,400（約216万円）。"
        "ただしV8の直接移植は不可（先物/NASDAQ-100のみ対応）。構造変換に工数がかかるうえ、"
        "配分獲得まで9〜12ヶ月を要する。3ヶ月以内の収益には貢献しないが、長期パイプラインとして種をまく。"
    )

    # ===== CHAPTER 8: CONSULTING =====
    pdf.add_page()
    pdf.chapter_title(8, "保険: フリーランスコンサル（週1〜2日限定）")

    pdf.section("格下げの理由")
    pdf.body(
        "コンサルティングは最も確実に3ヶ月以内に現金を生む手段である。しかしその本質は「上様の時間を他者に売る」"
        "労働売却であり、V8ライブ移行・Numerai提出・コンテンツ執筆に割く時間が消失する。"
        "短期の現金と引き換えに、本丸（資本収益・IP収益）の立ち上げが遅れる。"
    )

    pdf.section("実行する場合の厳格なルール")
    pdf.bullet("週1〜2日に限定（週3日以上は禁止）")
    pdf.bullet("残りの時間はV8ライブ移行・Numerai・コンテンツに充当")
    pdf.bullet("V8のIPは一切開示しない（NDA締結 + 戦略ロジック非開示の契約条件）")
    pdf.bullet("月額40〜60万円の収入を確保しつつ、本業を妨げない")

    pdf.section("プラットフォーム別の推奨単価")
    w = [45, 35, 45, 65]
    pdf.table_header(["プラットフォーム", "推奨単価", "マージン率", "特記事項"], w)
    pdf.table_row(["ProConnect", "120〜150万円/月", "旧8〜15%（現非公開）", "コンサル特化、金融DX案件"], w)
    pdf.table_row(["Findy Freelance", "100〜130万円/月", "0%（企業負担）", "手取り最大、技術寄り"], w)
    pdf.table_row(["レバテック", "90〜120万円/月", "非公開", "案件数最大"], w)
    pdf.table_row(["直接営業", "110〜140万円/月", "0%", "中間マージンなし"], w)

    # ===== CHAPTER 9: PRICING =====
    pdf.add_page()
    pdf.chapter_title(9, "希望単価の最適解（市場調査結果）")

    pdf.section("市場データ")
    w = [100, 90]
    pdf.table_header(["指標", "金額"], w)
    pdf.table_row(["フリーランスエンジニア平均", "74.6万円/月"], w)
    pdf.table_row(["金融業界案件平均", "102.1万円/月（業種別トップ）"], w)
    pdf.table_row(["Pythonフリーランス最高実績", "230万円/月"], w)
    pdf.table_row(["ProConnect平均", "170〜193万円/月"], w)
    pdf.table_row(["正社員クオンツ年収（日系）", "900〜1,200万円"], w)
    pdf.table_row(["正社員クオンツ年収（外資）", "2,000〜5,000万円+"], w)

    pdf.section("上様のスキルセット評価")
    w = [70, 120]
    pdf.table_header(["評価軸", "判定"], w)
    pdf.table_row(["スキル希少性", "ユニコーン級（QC×ファクター×フルスタック）"], w)
    pdf.table_row(["バックテスト実績", "15年/Sharpe 0.92/WF検証済 → プラス"], w)
    pdf.table_row(["ライブ運用実績", "Paper→Live移行中 → マイナス"], w)
    pdf.table_row(["コンサルファーム経歴", "なし → ProConnectでは不利"], w)
    pdf.table_row(["初回フリーランス", "プラットフォーム実績ゼロ → マイナス"], w)

    pdf.section("最適単価（結論）")
    pdf.bold_body("レジュメ記載: 月額 100〜150万円（税別・スコープにより応相談）")
    pdf.ln(1)
    pdf.bullet("下限100万 = 心理的壁の上。「100万以上の人材」と認知される")
    pdf.bullet("上限150万 = ProConnectのボリュームゾーンに食い込むアンカー")
    pdf.bullet("旧表記80〜130万は自己評価を毀損（Pythonエンジニア平均並み）→ 廃止")
    pdf.bullet("一度下げた基準値は二度と上がらない（フリーランス市場の鉄則）")

    pdf.section("交渉時のアンカリング戦術")
    pdf.body(
        "「クオンツエンジニアの正社員相場は外資系で年収2,000〜3,000万円、国内でも1,500万円以上が一般的です。"
        "フリーランスとしては月額150万円（年額1,800万円）を希望しておりますが、"
        "プロジェクトの規模・期間に応じて柔軟にご相談させていただければと存じます。」"
    )
    pdf.body(
        "構造: (1) 正社員相場2,000〜3,000万がアンカー → (2) 月150万=年1,800万が割安に見える → "
        "(3)「柔軟にご相談」で交渉余地を残す"
    )

    # ===== CHAPTER 10: RESUME & LINKEDIN =====
    pdf.add_page()
    pdf.chapter_title(10, "レジュメ・LinkedIn戦略")

    pdf.section("納品物一覧")
    w = [75, 30, 85]
    pdf.table_header(["ファイル", "言語", "用途"], w)
    pdf.table_row(["Resume_Quant_Consultant.pdf", "日本語", "ProConnect/Findy/レバテック"], w)
    pdf.table_row(["Resume_Quant_Consultant.docx", "日本語", "Google Docs編集用"], w)
    pdf.table_row(["Resume_Quant_Engineer_EN.docx", "英語", "外資ファンド/LinkedIn/海外リモート"], w)
    pdf.table_row(["LinkedIn_Profile_Draft.md", "英語", "LinkedInプロフィール（コピペ用）"], w)

    pdf.section("レジュメの設計方針")
    pdf.bullet("最終学歴: 削除（スキル・実績ベースのマッチングでは不要。書くと天井を設定される）")
    pdf.bullet("希望条件セクション: 削除（基本情報と重複。対応可能案件例のみ独立セクションに昇格）")
    pdf.bullet("戦略ロジック: 一切非開示。パフォーマンス数値（CAGR/Sharpe/DD）のみ記載")
    pdf.bullet("日英2本立て: 国内プラットフォーム用（日本語）+ 外資・海外リモート用（英語）")

    pdf.section("LinkedIn登録の必要性")
    pdf.body(
        "外資ヘッジファンド・運用会社のリクルーターは、LinkedIn上で"
        "\"quantitative engineer\" \"factor model\" \"QuantConnect\" 等のキーワードで候補者を検索する。"
        "プロフィールが存在しなければ、スキルセットがいかに希少であろうと"
        "彼らの目に触れる機会はゼロ。月150〜250万帯の外資案件に到達不能となる。"
    )
    pdf.bullet("登録コスト: 無料、10分で完了")
    pdf.bullet("Open to Work: リクルーター限定表示（公開には見えない）")
    pdf.bullet("プロフィール言語: 英語（日本の外資リクルーターも英語で検索する）")

    # ===== CHAPTER 11: EXCLUDED =====
    pdf.add_page()
    pdf.chapter_title(11, "排除した施策と理由")

    w = [50, 140]
    pdf.table_header(["施策", "排除理由"], w)
    pdf.table_row(["有料シグナル配信", "金商法違反（懲役5年/罰金500万円）。登録には700万円+6ヶ月〜1年"], w)
    pdf.table_row(["WorldQuant BRAIN", "日本は対象国外（確認済み）。IQC参加は可能だが継続報酬なし"], w)
    pdf.table_row(["YouTube", "収益化まで5ヶ月+。登録者1000人+再生4000時間必要。3ヶ月では不可"], w)
    pdf.table_row(["独立ファンド設立", "初年度$500K〜$1.7M。3ヶ月の制約に合致しない"], w)
    pdf.table_row(["GogoJungle", "FX EA市場が主力。米国株ファクターモデルとの親和性が低い"], w)
    pdf.table_row(["週3日+コンサル", "労働売却。V8ライブ移行を妨害。上様の時間を不当に拘束"], w)

    # ===== CHAPTER 12: ACTION PLAN =====
    pdf.add_page()
    pdf.chapter_title(12, "90日アクションプラン（改訂版）")

    pdf.section("Week 1（2/10〜2/16）— 基盤構築")
    pdf.bullet("V8 Paper Tradingの開始/確認（QC上）")
    pdf.bullet("Interactive Brokers口座開設申請")
    pdf.bullet("Numeraiアカウント作成、NMR購入（取引所口座が必要）")
    pdf.bullet("note.comアカウント作成、第1回記事（無料）の執筆開始")
    pdf.bullet("X (Twitter) アカウント開設/整備、クオンツ系フォロー開始")
    pdf.bullet("LinkedIn登録（LinkedIn_Profile_Draft.md をコピペ）")
    pdf.bullet("（保険として）フリーランスエージェント1〜2社に登録")

    pdf.section("Week 2〜4（2/17〜3/9）— パイプライン構築")
    pdf.bullet("V8 Paper Trading継続（注文執行・レジーム判定の動作検証）")
    pdf.bullet("Numeraiシグナル変換パイプライン構築、ステークなし提出で検証")
    pdf.bullet("note記事を週1本ペースで公開（第2〜4回）")
    pdf.bullet("Brain高単価コンテンツ（全編バンドル）の執筆")
    pdf.bullet("SIGNATE/Kaggle開催中コンペの確認・参加")
    pdf.bullet("（保険）コンサル案件の面談（週1〜2日限定案件のみ）")

    pdf.section("Month 2（3/10〜4/9）— 初期収益")
    pdf.bullet("IB口座開設完了 → 入金準備")
    pdf.bullet("V8 Paper Trading結果の検証（1ヶ月分の実績）")
    pdf.bullet("Numerai少額ステーク開始（10〜50 NMR）")
    pdf.bullet("note第5〜6回記事公開 + マガジン化")
    pdf.bullet("Brainコンテンツ公開（9,800円）")
    pdf.bullet("（保険）コンサル初月稼働 → 月末請求")

    pdf.section("Month 3（4/10〜5/9）— 収穫")
    pdf.bullet("V8 Live移行判断（Paper Trading結果が良好な場合）")
    pdf.bullet("Live運用開始（全資金の20〜30%で慎重に開始）")
    pdf.bullet("Numerai初回ペイアウト確認、ステーク増額判断")
    pdf.bullet("コンテンツ累計収益の確認・第2弾企画")
    pdf.bullet("（保険）コンサル2ヶ月目入金")

    # ===== CHAPTER 13: SIMULATION =====
    pdf.add_page()
    pdf.chapter_title(13, "収益シミュレーション（改訂版）")

    pdf.section("自己資金500万円の場合")
    pdf.body("（V8ライブ運用に500万円を投入する想定）")
    pdf.ln(1)
    w = [60, 40, 40, 50]
    pdf.table_header(["施策", "保守的", "中間", "楽観的"], w)
    pdf.table_row(["V8ライブ運用（月2以降）", "3万円", "10万円", "18万円"], w)
    pdf.table_row(["Numerai Signals", "2万円", "5万円", "15万円"], w)
    pdf.table_row(["note.com + Brain", "15万円", "45万円", "120万円"], w)
    pdf.table_row(["コンペ（不確定）", "0円", "0円", "100万円"], w)
    pdf.table_row(["コンサル保険（週1〜2日）", "80万円", "100万円", "120万円"], w)
    pdf.set_font("HGB", "", 8)
    pdf.set_fill_color(*LIGHT)
    pdf.cell(60, 7, "3ヶ月合計", border=1, fill=True)
    pdf.cell(40, 7, "100万円", border=1, fill=True, align="C")
    pdf.cell(40, 7, "160万円", border=1, fill=True, align="C")
    pdf.cell(50, 7, "373万円", border=1, fill=True, align="C")
    pdf.ln()

    pdf.ln(3)
    pdf.section("自己資金3,000万円の場合")
    pdf.body("（V8ライブ運用に3,000万円を投入する想定）")
    pdf.ln(1)
    pdf.table_header(["施策", "保守的", "中間", "楽観的"], w)
    pdf.table_row(["V8ライブ運用（月2以降）", "18万円", "60万円", "108万円"], w)
    pdf.table_row(["Numerai Signals", "2万円", "5万円", "15万円"], w)
    pdf.table_row(["note.com + Brain", "15万円", "45万円", "120万円"], w)
    pdf.table_row(["コンペ（不確定）", "0円", "0円", "100万円"], w)
    pdf.table_row(["コンサル保険（週1〜2日）", "0円", "0円", "0円"], w)
    pdf.set_font("HGB", "", 8)
    pdf.set_fill_color(*LIGHT)
    pdf.cell(60, 7, "3ヶ月合計", border=1, fill=True)
    pdf.cell(40, 7, "35万円", border=1, fill=True, align="C")
    pdf.cell(40, 7, "110万円", border=1, fill=True, align="C")
    pdf.cell(50, 7, "343万円", border=1, fill=True, align="C")
    pdf.ln()
    pdf.ln(2)
    pdf.body("※ 3,000万円規模ならコンサル保険は不要。V8ライブ運用のみで年間432万円（月36万円）の期待値。")

    # ===== CHAPTER 14: CONFIRMATION =====
    pdf.add_page()
    pdf.chapter_title(14, "上様へのご確認事項")

    pdf.body("本報告書の戦略を最終確定するにあたり、以下の1点を確認させていただきたく存じます。")
    pdf.ln(3)

    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.5)
    y0 = pdf.get_y()
    pdf.rect(12, y0, 186, 55)

    pdf.set_xy(16, y0 + 4)
    pdf.set_font("HGH", "", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, "V8ライブ運用に投じられる自己資金の規模")
    pdf.set_xy(16, y0 + 16)
    pdf.set_font("HG", "", 9)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(178, 5.5,
        "この数値により全体の方針が根本的に変わります:\n"
        "\n"
        "・100万円以下 → コンサル保険を週2日で実行。コンテンツ+Numeraiを並行\n"
        "・500万〜1,000万円 → コンサル保険は週1日。V8ライブ+コンテンツ中心\n"
        "・3,000万円以上 → コンサル不要。V8ライブ運用のみで月36万円+の期待値\n"
        "・1億円以上 → 完全不労。V8のみで月120万円+の期待値"
    )

    pdf.ln(10)
    pdf.set_draw_color(*LINE_C)
    pdf.set_line_width(0.2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("HG", "", 7.5)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "本報告書は Asset Shield 老中（Claude Opus 4.6）が全調査班の結果を統合し作成", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "調査班: Numerai班 / Quantiacs班 / 国内合法マネタイズ班 / 日本クオンツ成功事例班 / 単価調査班 / 価格心理学班", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "※ 戦略パフォーマンス指標はバックテスト結果であり、将来の成果を保証するものではありません", align="C")

    pdf.output(OUTPUT)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build()
