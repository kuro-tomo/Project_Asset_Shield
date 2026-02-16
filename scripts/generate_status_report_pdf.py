#!/usr/bin/env python3
"""Asset Shield — 全体現況レポート PDF生成"""

from fpdf import FPDF
from datetime import datetime

FONT_PATH_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_PATH_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_PATH_W8 = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"
OUTPUT = "/Users/MBP/Desktop/Asset_Shield_Status_Report_20260215.pdf"

# Colors
NAVY = (26, 42, 82)
DARK = (33, 33, 33)
GRAY = (100, 100, 100)
LIGHT_BG = (245, 247, 250)
WHITE = (255, 255, 255)
ACCENT = (0, 102, 204)
ROW_ALT = (240, 244, 250)
GREEN = (22, 163, 74)
RED = (220, 38, 38)
ORANGE = (234, 88, 12)
YELLOW_BG = (255, 251, 235)


class StatusPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.add_font("HG", "", FONT_PATH_W3)
        self.add_font("HGB", "", FONT_PATH_W6)
        self.add_font("HGH", "", FONT_PATH_W8)
        self.set_auto_page_break(True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("HG", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 6, "Asset Shield — System Status Report  |  2026-02-15", align="L")
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
        self.ln(40)
        self.set_font("HGH", "", 30)
        self.set_text_color(*NAVY)
        self.cell(0, 14, "Asset Shield", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_font("HGB", "", 18)
        self.cell(0, 12, "全体現況レポート", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.5)
        self.line(70, self.get_y(), 140, self.get_y())
        self.ln(12)
        self.set_font("HG", "", 11)
        self.set_text_color(*DARK)
        for line in [
            "報告日: 2026年2月15日",
            "報告者: 老中（Claude Opus 4.6）",
            "宛先: 上様",
            "",
            "目的: プロジェクト全体の稼働状況・構成・課題の網羅的報告",
        ]:
            self.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")

        # Status summary box
        self.ln(16)
        x0, w, h = 30, 150, 50
        self.set_fill_color(*NAVY)
        self.rect(x0, self.get_y(), w, h, "F")
        y = self.get_y() + 8
        self.set_xy(x0, y)
        self.set_font("HGB", "", 13)
        self.set_text_color(*WHITE)
        self.cell(w, 8, "SYSTEM STATUS: OPERATIONAL", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x0, y + 14)
        self.set_font("HG", "", 10)
        self.cell(w, 7, "Numerai Tournament: Round 1204 提出済", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x0, y + 22)
        self.cell(w, 7, "Numerai Signals JP: 2/14 提出済", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_xy(x0, y + 30)
        self.cell(w, 7, "自動化: launchd + cron 二重構成 稼働中", align="C", new_x="LMARGIN", new_y="NEXT")

    def section_title(self, num, title):
        self.ln(6)
        self.set_font("HGH", "", 14)
        self.set_text_color(*NAVY)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.ln(2)
        self.set_font("HGB", "", 11)
        self.set_text_color(*DARK)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.cell(0, 6, f"  ・{text}", new_x="LMARGIN", new_y="NEXT")

    def status_badge(self, text, color):
        self.set_font("HGB", "", 8)
        self.set_text_color(*WHITE)
        tw = self.get_string_width(text) + 6
        self.set_fill_color(*color)
        x = self.get_x()
        y = self.get_y()
        self.rect(x, y, tw, 6, "F")
        self.set_xy(x + 1, y)
        self.cell(tw - 2, 6, text)
        self.set_xy(x + tw + 2, y)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (190) / len(headers)
            col_widths = [w] * len(headers)

        # Header
        self.set_font("HGB", "", 8.5)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
        self.ln()

        # Rows
        self.set_font("HG", "", 8.5)
        self.set_text_color(*DARK)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 1:
                self.set_fill_color(*ROW_ALT)
                fill = True
            else:
                self.set_fill_color(*WHITE)
                fill = True
            max_h = 7
            for i, cell in enumerate(row):
                self.cell(col_widths[i], max_h, str(cell), border=1, align="C" if i > 0 else "L", fill=fill)
            self.ln()

    def info_box(self, title, content, bg_color=LIGHT_BG):
        self.ln(2)
        x0 = 10
        w = 190
        self.set_fill_color(*bg_color)
        y_start = self.get_y()
        # Calculate height
        self.set_font("HG", "", 9)
        lines = content.split("\n")
        h = 10 + len(lines) * 6 + 4
        self.rect(x0, y_start, w, h, "F")
        self.set_xy(x0 + 4, y_start + 2)
        self.set_font("HGB", "", 9.5)
        self.set_text_color(*NAVY)
        self.cell(0, 6, title)
        self.set_xy(x0 + 4, y_start + 9)
        self.set_font("HG", "", 9)
        self.set_text_color(*DARK)
        for line in lines:
            self.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x0 + 4)
        self.set_y(y_start + h + 2)


def build_pdf():
    pdf = StatusPDF()

    # ========== Cover ==========
    pdf.cover_page()

    # ========== 1. Project Overview ==========
    pdf.add_page()
    pdf.section_title("1", "プロジェクト概要")

    pdf.body(
        "Asset Shield は日本株市場に特化した自己進化型アルゴリズム取引システムである。"
        "QuantConnect（Alpha Market / Streams）および Quantiacs での実績構築（Track Record）と"
        "資本配分（Allocation）獲得を事業目的とする。"
    )

    pdf.sub_title("設計思想 — 4つの柱")
    pillars = [
        ["300億円キャパシティ", "流動性制約下でも安定したリスク調整後リターンを維持。Almgren-Chrissモデル適用。"],
        ["生存バイアス排除", "上場廃止銘柄含む20年データでバックテスト。学習/検証フェーズを厳格分離。"],
        ["低回転率", "取引コスト・市場インパクト抑制。プラットフォーム評価指標での優位性確保。"],
        ["自己増殖の安全弁", "デッドマンスイッチ含むSilence Protocol。知的財産と運用者の安全を確保。"],
    ]
    pdf.table(["柱", "内容"], pillars, [40, 150])

    pdf.ln(4)
    pdf.sub_title("事業目標")
    pdf.table(
        ["指標", "目標値"],
        [
            ["AUM目標", "300億円（100億円 x 3社）"],
            ["年間売上（300億運用時）", "11.0億円"],
            ["年間EBITDA", "10.0億円"],
            ["想定評価額", "50億〜80億円"],
            ["年間平均リターン（アグレッシブ）", "20.13%"],
            ["Sharpe Ratio目標", "> 4.0"],
        ],
        [60, 130],
    )

    # ========== 2. Automated Execution Systems ==========
    pdf.add_page()
    pdf.section_title("2", "自動実行システム（3系統）")

    pdf.sub_title("2-1. Numerai Tournament（毎日提出）")
    pdf.body(
        "グローバル株式市場を対象とした Numerai トーナメントに毎日自動提出。"
        "LightGBM アンサンブル（7ターゲット）による予測を生成し、2モデルを運用。"
    )
    pdf.table(
        ["項目", "詳細"],
        [
            ["スケジュール", "毎日 10:00 JST（launchd + cron 二重構成）"],
            ["起床設定", "pmset wakepoweron 9:55 毎日"],
            ["モデル1", "asset_shield — アンサンブル7モデル（中立化なし）"],
            ["モデル2", "mikasa — アンサンブル7モデル + 中立化（proportion=0.6）"],
            ["データセット", "v5.2 'Faith II'（2026年1月〜）"],
            ["特徴量セット", "rain（666特徴量）"],
            ["学習済モデル", "lgbm_ensemble_standard_v5.2.pkl（21.5MB）"],
            ["最終提出", "2026-02-15 10:00 — Round 1204 両モデル提出成功"],
        ],
        [40, 150],
    )

    pdf.ln(2)
    pdf.info_box(
        "アンサンブル構成（7ターゲット）",
        "target, target_ender_20（ペイアウト対象）, target_jasper_20,\n"
        "target_cyrusd_20, target_ralph_20, target_victor_20, target_waldo_20",
    )

    pdf.ln(4)
    pdf.sub_title("2-2. Numerai Signals JP（毎週土曜提出）")
    pdf.body(
        "J-Quants API から日本株データを取得し、独自特徴量を構築。"
        "LightGBM で学習後、Numerai Signals に提出。日本株に特化したシグナル。"
    )
    pdf.table(
        ["項目", "詳細"],
        [
            ["スケジュール", "毎週土曜 10:00 JST（launchd + cron 二重構成）"],
            ["データソース", "J-Quants API（株価・財務・信用・空売り比率）"],
            ["対象銘柄", "約4,470銘柄（東証全銘柄）"],
            ["特徴量数", "12個"],
            ["学習期間", "直近38日のローリング学習"],
            ["最終提出", "2026-02-14 23:35 — 提出成功（26秒）"],
        ],
        [40, 150],
    )

    pdf.ln(2)
    pdf.info_box(
        "特徴量重要度（上位6）",
        "sector_short_ratio: 6796  |  vol_60d: 6613  |  turnover_20d: 6484\n"
        "ret_60d: 5669  |  vol_20d: 5653  |  ret_20d: 4634",
    )

    pdf.ln(4)
    pdf.sub_title("2-3. X Bot JP（平日自動投稿）")
    pdf.body(
        "日本株セクター別空売り分析を毎朝 X（旧Twitter）に自動投稿。"
        "月曜は週次検証、月初は月次スコアカードも生成。"
    )
    pdf.table(
        ["項目", "詳細"],
        [
            ["スケジュール", "平日 7:30 JST（cron のみ）"],
            ["内容", "セクター別空売り比率分析"],
            ["投稿DB", "data/x_bot_jp/posts.db（40KB）"],
            ["最終更新", "2026-02-13 22:36"],
        ],
        [40, 150],
    )

    # ========== 3. Execution Infrastructure ==========
    pdf.add_page()
    pdf.section_title("3", "実行インフラストラクチャ")

    pdf.sub_title("3-1. ローカル実行環境")
    pdf.table(
        ["項目", "値"],
        [
            ["ハードウェア", "MacBook Pro M4 Pro（Mac16,7）"],
            ["OS", "macOS Darwin 25.2.0"],
            ["Python", "3.9.6（/usr/bin/python3）"],
            ["プロジェクトパス", "/Users/MBP/Desktop/Project_Asset_Shield/"],
            ["Git管理", "有効"],
        ],
        [50, 140],
    )

    pdf.ln(4)
    pdf.sub_title("3-2. スケジューラ構成（二重化）")
    pdf.body(
        "macOS の cron と launchd を二重構成で運用。launchd はスリープ復帰後に"
        "未実行ジョブを自動補完するため、信頼性が高い。"
    )
    pdf.table(
        ["ジョブ", "cron", "launchd", "状態"],
        [
            ["Numerai Daily", "0 10 * * *", "com.assetshield.numerai", "両方登録済"],
            ["Signals JP", "0 10 * * 6", "com.assetshield.numerai-signals", "両方登録済"],
            ["X Bot JP", "30 7 * * 1-5", "（未登録）", "cron のみ"],
        ],
        [40, 45, 65, 40],
    )

    pdf.ln(4)
    pdf.sub_title("3-3. 電源管理（2026-02-15 修正済）")
    pdf.table(
        ["設定", "修正前", "修正後", "備考"],
        [
            ["sleep (AC)", "1分", "0（無効）", "夜間cron問題を解消"],
            ["standby (AC)", "有効", "無効", "ディープスタンバイ防止"],
            ["sleep (Battery)", "1分", "1分（据置）", "バッテリ保護"],
            ["displaysleep", "0（無効）", "0（据置）", ""],
            ["wakepoweron", "9:55毎日", "9:55毎日（据置）", "朝10:00 cron用"],
        ],
        [35, 35, 40, 80],
    )

    # ========== 4. Core Architecture ==========
    pdf.add_page()
    pdf.section_title("4", "コアアーキテクチャ（src/shield）")

    pdf.sub_title("4-1. モジュール構成")
    pdf.table(
        ["モジュール", "役割", "ファイル"],
        [
            ["Adaptive Core", "市場レジーム検出（危機/高ボラ/通常/低ボラ/トレンド）", "adaptive_core.py"],
            ["Brain AI", "AI/MLモデル。損益に応じたパラメータ自律調整", "brain.py"],
            ["Alpha Model", "アルファシグナル生成。流動性チェック内蔵", "alpha_model.py"],
            ["Execution Core", "注文管理・VWAP/TWAP執行", "execution_core.py"],
            ["Money Management", "Kelly基準ポジションサイジング", "money_management.py"],
            ["Screener", "ファンダメンタル分析（J-Quants連携）", "screener.py"],
            ["J-Quants Client", "J-Quants APIインターフェース", "jquants_client.py"],
            ["Nexus", "中央処理ハブ", "nexus.py"],
            ["Pipeline", "パイプライン管理", "pipeline.py"],
            ["Tracker", "監査ログ・イベント記録", "tracker.py"],
            ["Security Gov.", "セキュリティガバナンス", "security_governance.py"],
            ["Evolution", "自己進化エンジン", "evolution.py"],
        ],
        [35, 95, 60],
    )

    pdf.ln(4)
    pdf.sub_title("4-2. 自己進化機構（src/shield/bio）")
    pdf.table(
        ["機能", "内容", "ファイル"],
        [
            ["Self-Repair", "SHA256ハッシュ比較による改ざん検知・自動修復", "bio/repair.py"],
            ["Self-Evolution", "PnL/WinRate/DD監視によるパラメータ自律調整", "bio/evolution.py"],
            ["Self-Replication", "CPU負荷に応じたプロセス増殖・死活監視", "bio/replication.py"],
            ["BioCore", "上記3機能の統合オーケストレーション", "bio/core.py"],
        ],
        [35, 100, 55],
    )

    pdf.ln(4)
    pdf.sub_title("4-3. 学習パラメータ")
    pdf.table(
        ["パラメータ", "Standard", "Deep"],
        [
            ["n_estimators", "2,000", "20,000"],
            ["learning_rate", "0.01", "0.001"],
            ["max_depth", "5", "6"],
            ["num_leaves", "32", "64"],
            ["colsample_bytree", "0.1", "0.1"],
            ["subsample", "0.8", "0.8"],
            ["早期停止", "200 rounds", "200 rounds"],
            ["訓練時間目安", "約10分", "約2時間"],
        ],
        [50, 70, 70],
    )

    # ========== 5. Data Infrastructure ==========
    pdf.add_page()
    pdf.section_title("5", "データ基盤")

    pdf.sub_title("5-1. データソースと容量")
    pdf.table(
        ["データ", "ソース", "サイズ", "更新頻度"],
        [
            ["train.parquet", "Numerai v5.2", "2.57 GB", "週次"],
            ["validation.parquet", "Numerai v5.2", "4.13 GB", "週次"],
            ["live.parquet", "Numerai v5.2", "9.97 MB", "毎日"],
            ["features.json", "Numerai v5.2", "323 KB", "バージョン毎"],
            ["jquants_cache.db", "J-Quants API", "2.47 GB", "随時"],
            ["posts.db", "X Bot JP", "40 KB", "平日"],
        ],
        [45, 45, 45, 55],
    )

    pdf.ln(4)
    pdf.sub_title("5-2. 学習済みモデル")
    pdf.table(
        ["モデルファイル", "サイズ", "更新日"],
        [
            ["lgbm_ensemble_standard_v5.2.pkl", "21.5 MB", "2026-02-13"],
            ["lgbm_standard_v5.0.pkl", "6.9 MB", "2026-02-11"],
            ["lgbm-standard-v5.pkl", "6.9 MB", "2026-02-11"],
        ],
        [80, 50, 60],
    )

    pdf.ln(4)
    pdf.sub_title("5-3. QuantConnect関連")
    pdf.table(
        ["ファイル", "内容", "更新日"],
        [
            ["ASSET_SHIELD_PRODUCTION.py", "本番用アルゴリズム", "2026-02-06"],
            ["ASSET_SHIELD_US.py", "米国株版（最新）", "2026-02-14"],
            ["AssetShieldJP/", "日本株版QuantConnectアルゴ", "2026-02-05"],
            ["UPLOAD_FUND_CHUNK1-4.py", "ファンダメンタルデータ分割アップロード", "2026-02-06"],
            ["fundamental_data/", "QC用ファンダメンタルデータ（52ファイル）", "2026-02-06"],
        ],
        [65, 80, 45],
    )

    # ========== 6. Security & Governance ==========
    pdf.add_page()
    pdf.section_title("6", "セキュリティ・ガバナンス")

    pdf.sub_title("6-1. Silence Protocol（デッドマンスイッチ）")
    pdf.body(
        "運用者の生存信号（Heartbeat）が途絶えた場合、自動的にコアロジックを破壊し、"
        "証拠を隠滅するシステム。DoD 5220.22-M準拠のセキュアワイプ機能を実装。"
    )
    pdf.table(
        ["トリガー", "動作"],
        [
            ["HEARTBEAT_TIMEOUT", "生存信号途絶 → 自動自壊"],
            ["PANIC_BUTTON", "緊急停止コード入力 → 即時破壊"],
            ["DURESS_CODE", "脅迫コード → 表面正常動作、裏で証拠隠滅"],
            ["INTRUSION_DETECTED", "侵入検知 → 防御モード発動"],
            ["CANARY_DEATH", "カナリアファイル消失 → 自壊"],
            ["REMOTE_SIGNAL", "外部からの緊急停止（SIGUSR1）"],
        ],
        [55, 135],
    )

    pdf.ln(4)
    pdf.sub_title("6-2. 監査・完全性証明")
    pdf.table(
        ["機能", "内容"],
        [
            ["SHA-256ハッシュ", "全ソースコードの完全性証明（integrity_certificate.json）"],
            ["Audit DB", "改ざん検知データベース（logs/audit.db）"],
            ["Self-Validated Model", "外部監査不要の自己証明体系"],
            ["PII サニタイズ", "メール・電話・APIキー等の自動検出・除去"],
        ],
        [50, 140],
    )

    pdf.ln(4)
    pdf.sub_title("6-3. 通信セキュリティ")
    pdf.bullet("商用VPN + Tor + 自己管理VPSの多層匿名化推奨")
    pdf.bullet("User-Agentランダム化、リクエスト間隔ジッター、IPローテーション")
    pdf.bullet("資格情報は環境変数/.envで管理（コードへのハードコード禁止）")

    # ========== 7. QuantConnect Strategy ==========
    pdf.add_page()
    pdf.section_title("7", "QuantConnect 戦略展開")

    pdf.sub_title("7-1. 収益化ロードマップ")
    pdf.table(
        ["フェーズ", "内容", "目標"],
        [
            ["Phase 1: 実績構築", "Alpha Market/Streams でトラックレコード構築", "上位ランク獲得"],
            ["Phase 2: 収益化", "購読料 + パフォーマンスフィー", "継続収益確立"],
            ["Phase 3: 拡大", "外部資金導入、AUM拡大", "300億円運用"],
        ],
        [40, 90, 60],
    )

    pdf.ln(4)
    pdf.sub_title("7-2. 競争優位（Version-Q）")
    pdf.bullet("適応コアロジック: 市場レジーム自動検出・パラメータ自律調整")
    pdf.bullet("学習エンジン: Brain AI による損益連動型パラメータチューニング")
    pdf.bullet("300億円設計: 大規模AUMでの流動性制約対応（Almgren-Chriss）")
    pdf.bullet("低回転率: プラットフォーム評価基準（安定性・実現可能性）に最適化")

    # ========== 8. Scripts Inventory ==========
    pdf.ln(6)
    pdf.section_title("8", "スクリプト一覧")

    pdf.sub_title("8-1. 自動実行系")
    pdf.table(
        ["スクリプト", "内容"],
        [
            ["numerai_starter.py", "Numerai Tournament パイプライン（DL→学習→予測→提出）"],
            ["numerai_signals_jp.py", "Numerai Signals JP パイプライン（J-Quants→学習→提出）"],
            ["x_bot_jp.py", "X 自動投稿（セクター空売り分析）"],
            ["cron_numerai.sh", "Numerai 日次提出ラッパー"],
            ["cron_signals_jp.sh", "Signals JP 週次提出ラッパー"],
            ["cron_x_bot.sh", "X Bot 平日投稿ラッパー"],
        ],
        [55, 135],
    )

    pdf.ln(4)
    pdf.sub_title("8-2. データ・分析系")
    pdf.table(
        ["スクリプト", "内容"],
        [
            ["ingest_historical_data.py", "J-Quants 20年ヒストリカルデータ取込"],
            ["scheduled_data_sync.py", "定期データ同期"],
            ["live_market_recon.py", "ライブ市場偵察"],
            ["parity_check.py", "データ整合性チェック"],
            ["export_fundamentals.py", "ファンダメンタルデータ出力"],
            ["upload_to_qc_objectstore.py", "QuantConnect ObjectStore アップロード"],
        ],
        [60, 130],
    )

    pdf.ln(4)
    pdf.sub_title("8-3. セキュリティ・保守系")
    pdf.table(
        ["スクリプト", "内容"],
        [
            ["silence_protocol.py", "デッドマンスイッチ（Silence Protocol）"],
            ["repair_watcher.py", "ファイル改ざん検知・自動修復"],
            ["watcher.py", "システム監視"],
            ["unmanned_march.py", "無人運転モード"],
        ],
        [55, 135],
    )

    # ========== 9. Issues & Recommendations ==========
    pdf.add_page()
    pdf.section_title("9", "要対応事項・推奨改善")

    pdf.sub_title("9-1. 要対応（優先度: 高〜中）")

    pdf.info_box(
        "WARN: mikasa モデルの中立化で数値演算警告",
        "neutralize() 関数で divide by zero / overflow in matmul が発生。\n"
        "NaN含有特徴量の前処理改善が必要。提出自体は成功するが予測品質に影響の可能性。\n"
        "対策: 中立化前にNaN行を除外、または特徴量の標準化を追加。",
        YELLOW_BG,
    )

    pdf.ln(2)
    pdf.info_box(
        "WARN: X Bot JP が launchd 未登録",
        "cron のみで運用しているため、スリープ復帰時に取りこぼす可能性。\n"
        "対策: com.assetshield.xbot.plist を作成し launchd に登録。",
        YELLOW_BG,
    )

    pdf.ln(4)
    pdf.sub_title("9-2. 推奨改善（優先度: 低）")
    pdf.bullet("Numerai cron ジョブにAPIキーがハードコードされている → .env 参照に統一すべき")
    pdf.bullet("Python 3.9.6 は古い（macOS付属版）→ Python 3.12+ へのアップグレード推奨")
    pdf.bullet("urllib3 の OpenSSL 警告 → LibreSSL 互換問題。Homebrew Python で解消可能")
    pdf.bullet("train/validation データの再学習頻度を定義すべき（現在は手動）")
    pdf.bullet("QuantConnect デプロイの自動化（現在は手動アップロード）")

    pdf.ln(6)
    pdf.sub_title("9-3. 本日実施済み")
    pdf.table(
        ["対応", "内容", "結果"],
        [
            ["電源管理修正", "AC電源時 sleep=0, standby=0 に設定", "夜間スリープ問題を解消"],
        ],
        [50, 90, 50],
    )

    # ========== 10. Summary ==========
    pdf.ln(8)
    pdf.section_title("10", "総括")
    pdf.body(
        "Asset Shield は設計・実装・自動化の3層において概ね健全に稼働している。\n\n"
        "Numerai Tournament は毎日自動提出が確認でき、Signals JP も週次で正常動作。"
        "QuantConnect 向けアルゴリズムも日米両版が開発進行中である。"
        "本日の電源管理修正により夜間のcron不実行問題は解消済み。\n\n"
        "直近の優先事項は mikasa モデルの中立化警告への対処と、"
        "X Bot JP の launchd 登録による信頼性向上である。"
        "中長期的には Python バージョンアップと学習パイプラインの完全自動化が推奨される。"
    )

    # Save
    pdf.output(OUTPUT)
    print(f"PDF saved: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
