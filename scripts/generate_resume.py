#!/usr/bin/env python3
"""ProConnect用 職務経歴書 PDF生成"""

from fpdf import FPDF

FONT_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_W8 = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"
FONT_MIN = "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc"
OUTPUT = "/Users/MBP/Desktop/Resume_Quant_Consultant.pdf"

NAVY = (26, 42, 82)
DARK = (33, 33, 33)
GRAY = (100, 100, 100)
LIGHT = (245, 247, 250)
WHITE = (255, 255, 255)
ACCENT = (0, 102, 204)
LINE_C = (200, 200, 200)
HDR_BG = (26, 42, 82)


class ResumePDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font("HG", "", FONT_W3)
        self.add_font("HGB", "", FONT_W6)
        self.add_font("HGH", "", FONT_W8)
        self.add_font("MIN", "", FONT_MIN)
        self.set_auto_page_break(True, margin=18)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("HG", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 5, "職務経歴書", align="R")
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

    def section(self, title):
        self.ln(3)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font("HGB", "", 11)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(*DARK)

    def subsection(self, title):
        self.ln(2)
        self.set_font("HGB", "", 10)
        self.set_text_color(*NAVY)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*NAVY)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(2)
        self.set_text_color(*DARK)

    def kv(self, key, val):
        self.set_font("HGB", "", 9)
        self.cell(35, 6, key)
        self.set_font("HG", "", 9)
        self.cell(0, 6, val, new_x="LMARGIN", new_y="NEXT")

    def body(self, text):
        self.set_font("HG", "", 9)
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

    def skill_bar(self, name, level, years):
        """level: 1-5"""
        self.set_font("HG", "", 9)
        self.cell(50, 6, name)
        self.set_font("HGB", "", 9)
        bar = "■" * level + "□" * (5 - level)
        self.set_text_color(*NAVY)
        self.cell(30, 6, bar)
        self.set_text_color(*GRAY)
        self.set_font("HG", "", 8)
        self.cell(0, 6, years, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*DARK)

    def project_block(self, period, title, role, desc_lines, tech):
        self.ln(2)
        self.set_fill_color(*LIGHT)
        y0 = self.get_y()
        # Calculate height needed
        self.set_font("HGB", "", 9)
        self.set_text_color(*NAVY)
        self.cell(40, 6, period)
        self.set_font("HGB", "", 10)
        self.set_text_color(*DARK)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font("HG", "", 8.5)
        self.set_text_color(*GRAY)
        self.cell(0, 5, f"役割: {role}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*DARK)
        self.ln(1)
        for line in desc_lines:
            self.bullet(line)
        self.ln(1)
        self.set_font("HGB", "", 8)
        self.set_text_color(*ACCENT)
        self.cell(18, 5, "技術スタック:")
        self.set_font("HG", "", 8)
        self.multi_cell(168, 5, tech)
        self.set_text_color(*DARK)
        self.ln(1)
        # Draw subtle left border
        self.set_draw_color(*NAVY)
        self.set_line_width(0.8)
        self.line(10, y0, 10, self.get_y())


def build():
    pdf = ResumePDF()
    pdf.alias_nb_pages()

    # ===== PAGE 1 =====
    pdf.add_page()

    # Title
    pdf.set_font("HGH", "", 20)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 12, "職 務 経 歴 書", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("HG", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, f"作成日: 2026年2月10日", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # === 基本情報 ===
    pdf.section("基本情報")
    pdf.kv("氏名", "[要記入]")
    pdf.kv("生年月日", "[要記入]")
    pdf.kv("所在地", "[要記入]  （例: 東京都）")
    pdf.kv("稼働可能時期", "即日（2026年2月〜）")
    pdf.kv("稼働希望", "週3〜5日（リモート中心、常駐も可）")
    pdf.kv("希望単価", "月額 100〜150万円（税別・スコープにより応相談）")

    # === 職務要約 ===
    pdf.section("職務要約")
    pdf.body(
        "クオンツ・エンジニア / アルゴリズム・ストラテジスト。"
        "日米両市場を対象としたマルチファクター運用戦略の設計・実装・検証に従事。"
    )
    pdf.body(
        "【米国株】15年バックテスト（2010-2024）で CAGR 14.4%、Sharpe 0.92 を達成した "
        "Core-Satellite型戦略を独力で構築し、現在ライブ運用中。"
    )
    pdf.body(
        "【日本株】J-Quants全銘柄（約5,000銘柄・1,490万レコード）を対象とした "
        "18年間（2008-2026）の大規模バックテスト基盤を構築。"
        "リーマンショック・アベノミクス・COVID・金利正常化の4フェーズで "
        "サバイバーシップバイアスフリーのストレステストを実施。"
    )
    pdf.body(
        "Python / QuantConnect (LEAN) / Interactive Brokers を用いた"
        "フルスタックのクオンツ開発（データ取得 → ファクター算出 → バックテスト → "
        "ライブ実行 → リスク管理 → 通知）に精通。"
        "金融工学の理論とソフトウェアエンジニアリングの実務を両立できる稀少な人材。"
    )

    # === 技術スキル ===
    pdf.section("技術スキル")

    pdf.subsection("プログラミング言語")
    pdf.skill_bar("Python", 5, "10年+（NumPy/Pandas/SciPy/scikit-learn）")
    pdf.skill_bar("SQL", 4, "8年+（SQLite/PostgreSQL）")
    pdf.skill_bar("C# / .NET", 3, "3年+（QuantConnect LEAN）")
    pdf.skill_bar("JavaScript/TypeScript", 3, "3年+")

    pdf.subsection("クオンツ・金融専門")
    pdf.skill_bar("ファクターモデル設計", 5, "マルチファクター（5因子）、ランク正規化")
    pdf.skill_bar("バックテスト設計", 5, "Walk-Forward、IS/OOS分割、過学習防止")
    pdf.skill_bar("リスク管理", 5, "Kill-switch、逆ボラサイジング、DD制御")
    pdf.skill_bar("レジーム判定", 5, "マクロ指標ベースの相場局面分類")
    pdf.skill_bar("ポートフォリオ構築", 5, "Core-Satellite、セクター中立化")
    pdf.skill_bar("統計分析", 4, "仮説検定、ブートストラップ、モンテカルロ")

    pdf.subsection("プラットフォーム / インフラ")
    pdf.skill_bar("QuantConnect (LEAN)", 5, "アルゴ設計〜ライブ運用まで一貫対応")
    pdf.skill_bar("Interactive Brokers API", 4, "注文管理、ポジション管理、データ取得")
    pdf.skill_bar("Git / GitHub", 5, "バージョン管理、CI/CD")
    pdf.skill_bar("Linux / macOS", 4, "サーバー構築、スクリプト自動化")
    pdf.skill_bar("Docker", 3, "コンテナ化、再現可能環境")
    pdf.skill_bar("AWS / GCP", 3, "EC2/S3、Cloud Functions")

    pdf.subsection("データ基盤")
    pdf.skill_bar("金融データパイプライン", 5, "ETL設計、データ品質検証")
    pdf.skill_bar("SQLite / PostgreSQL", 4, "大規模時系列データ管理")
    pdf.skill_bar("J-Quants API", 4, "日本株データ取得・加工")
    pdf.skill_bar("REST API設計", 4, "認証、ページネーション、レート制御")

    # === プロジェクト実績 ===
    pdf.section("プロジェクト実績")

    pdf.project_block(
        "2023 〜 現在",
        "米国株 Core-Satellite 運用戦略（V8）",
        "設計・開発・運用（個人プロジェクト）",
        [
            "SPY 80% Core + 50% マルチファクター Satellite の 130% Core-Satellite 構造を設計",
            "5ファクターモデル（Momentum / Short-term Momentum / Low Volatility / Value / Quality）を構築",
            "マクロ指標ベースのレジーム判定（5段階）で動的にエクスポージャーを調整",
            "DD 15%で自動発動する Kill-switch を実装（Satellite清算、Core維持）",
            "15年バックテスト（2010-2024）: リターン +649%、CAGR 14.4%、Sharpe 0.92、Max DD 25.7%",
            "Walk-Forward検証: IS Sharpe 1.10 / OOS Sharpe 0.74（頑健性を確認）",
            "Email通知システム（Kill-switch発動、レジーム変更、週次サマリー）を実装",
            "QuantConnect上でペーパートレード → Interactive Brokers ライブ移行を推進中",
        ],
        "Python, QuantConnect (LEAN/C#), Interactive Brokers API, NumPy, Pandas, SciPy, SQLite",
    )

    pdf.project_block(
        "2022 〜 2023",
        "マルチファクター戦略 研究開発（V1〜V7）",
        "設計・開発・検証（個人プロジェクト）",
        [
            "米国株式のファクター投資戦略を7世代にわたり反復改善",
            "V6 Pro（純粋ファクター Long-Only）がSPYを下回る課題を特定 → Core-Satellite構造（V8）で解決",
            "パラメータチューニングだけでは構造的問題を解決できないことを実証",
            "逆ボラティリティ・サイジング、セクター分散制約、トレーリングストップ等のリスク管理手法を体系化",
            "41モジュール構成のPythonライブラリ（src/shield/）として知見をコード化",
        ],
        "Python, QuantConnect, pandas, scikit-learn, matplotlib, SQLite",
    )

    pdf.project_block(
        "2021 〜 現在",
        "日本株 全銘柄バックテスト基盤（18年・5,000銘柄）",
        "設計・開発・検証（個人プロジェクト）",
        [
            "J-Quants API の全銘柄データ（約5,000銘柄、1,490万レコード）を用いた大規模バックテスト基盤を構築",
            "サバイバーシップバイアスフリー: 上場廃止銘柄を含む全銘柄で検証可能な設計",
            "4フェーズ・ストレステスト: リーマンショック(2006-2010) → アベノミクス(2011-2015) → COVID(2016-2020) → 金利正常化(2021-2026)",
            "Walk-Forward検証フレームワーク（IS/Validation/OOS 3分割）を実装",
            "SQLiteデータウェアハウス（2.3GB）、データ品質カバレッジ99%+を確認",
            "41モジュール構成のPython量的分析ライブラリ（src/shield/）を構築",
        ],
        "Python, J-Quants API, SQLite, NumPy, Pandas, SciPy, scikit-learn, REST API",
    )

    pdf.project_block(
        "2024 〜 現在",
        "金融データパイプライン基盤構築",
        "設計・開発（個人プロジェクト）",
        [
            "J-Quants API を用いた日本株全銘柄のデータ取得・加工パイプラインを構築",
            "固定長バイナリデータ（374バイト、cp932）のパーサーを開発、83フィールドの自動抽出を実現",
            "SQLiteベースのデータウェアハウス設計（1,490万レコード規模）",
            "データ品質検証フレームワーク（カバレッジ率99%+を確認）を実装",
            "ETLパイプライン: ダウンローダー → パーサー → ETL → スキーマ → SQLite の自動化",
        ],
        "Python, SQLite, J-Quants API, REST API, cp932エンコーディング, ETL設計",
    )

    # === 自己PR ===
    pdf.section("強み・自己PR")

    pdf.subsection("1. 戦略設計から実運用まで一気通貫")
    pdf.body(
        "アイデア着想 → 数理モデル設計 → Pythonコード実装 → バックテスト検証 → "
        "リスク管理設計 → ライブ運用基盤構築 → モニタリング・通知の全工程を一人で完遂できます。"
        "外注や分業なしで、構想から本番稼働まで最短距離で到達可能です。"
    )

    pdf.subsection("2. 失敗から学ぶ反復改善力")
    pdf.body(
        "7世代の戦略改善を通じて「パラメータチューニングでは構造的問題は解決できない」"
        "「ファクター Long-Only はベータリスクに脆弱」等の本質的な教訓を獲得。"
        "V6の失敗（SPY下回り）を分析し、Core-Satellite構造（V8）で根本解決に至りました。"
        "この反復改善プロセスは、クライアントの既存戦略の改善にも直接応用できます。"
    )

    pdf.subsection("3. データ品質への執着")
    pdf.body(
        "固定長バイナリ（cp932/374バイト）の完全パーサー開発、83フィールドの自動検証、"
        "99%+のカバレッジ率確認など、データ品質に対して妥協しません。"
        "「ガベージイン・ガベージアウト」を防ぐデータパイプラインの設計・運用が可能です。"
    )

    pdf.subsection("4. 再現可能な研究プロセス")
    pdf.body(
        "Walk-Forward検証（IS/OOS分割）、決定論的バックテスト、Git管理による完全な再現性を確保。"
        "過学習・サバイバーシップバイアス・ルックアヘッドバイアス等の定量リサーチの落とし穴を熟知し、"
        "堅牢な検証フレームワークを構築できます。"
    )

    # === 対応可能な案件例 ===
    pdf.section("対応可能な案件例")
    pdf.bullet("クオンツ戦略の設計・開発・バックテスト支援")
    pdf.bullet("ファクターモデル / リスクモデルの構築")
    pdf.bullet("バックテストフレームワークの設計・開発")
    pdf.bullet("金融データパイプライン（ETL）の構築")
    pdf.bullet("QuantConnect / Interactive Brokers を用いた自動売買システム開発")
    pdf.bullet("既存クオンツ戦略のレビュー・改善提案")
    pdf.bullet("Pythonによる金融データ分析基盤の構築")

    pdf.ln(6)
    pdf.set_draw_color(*LINE_C)
    pdf.set_line_width(0.2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("HG", "", 7.5)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "※ 本経歴書に記載の戦略パフォーマンス指標はバックテスト結果であり、将来の成果を保証するものではありません。", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "※ 戦略の具体的ロジック・パラメータは営業秘密として非開示とさせていただきます。", align="C")

    pdf.output(OUTPUT)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build()
