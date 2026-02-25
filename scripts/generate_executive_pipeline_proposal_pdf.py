#!/usr/bin/env python3
"""Generate PDF: Ultimate Executive Pipeline v2 Improvement Proposal."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser(
    "~/Desktop/Ultimate_Executive_Pipeline_v2_Proposal.pdf"
)
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_BOLD_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_HEAVY_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"


class PipelinePDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("gothic", size=7)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 10,
            f"Ultimate Executive Pipeline v2 Proposal  |  Page {self.page_no()}/{{nb}}",
            align="C",
        )


def section_header(pdf, text, w):
    """Dark banner section header."""
    pdf.ln(4)
    pdf.set_fill_color(30, 45, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("gothic", "B", 12)
    pdf.cell(w, 8, f"  {text}", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def sub_header(pdf, text, w):
    """Light sub-header."""
    pdf.set_fill_color(220, 230, 245)
    pdf.set_text_color(30, 45, 80)
    pdf.set_font("gothic", "B", 10)
    pdf.cell(w, 7, f"  {text}", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def body_text(pdf, text, w, size=9):
    """Standard body text."""
    pdf.set_font("gothic", "", size)
    pdf.multi_cell(w, 5.5, text, align="L",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def bold_body(pdf, text, w, size=9):
    pdf.set_font("gothic", "B", size)
    pdf.multi_cell(w, 5.5, text, align="L",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def bullet(pdf, items, w, indent=8):
    pdf.set_font("gothic", "", 9)
    for item in items:
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(w - indent, 5.5, f"・{item}", align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def draw_table(pdf, headers, rows, w, col_ratios=None):
    """Simple table with header row."""
    if col_ratios is None:
        col_ratios = [1.0 / len(headers)] * len(headers)
    col_widths = [w * r for r in col_ratios]
    lm = pdf.l_margin

    # Header
    pdf.set_fill_color(30, 45, 80)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("gothic", "B", 8)
    x = lm
    for i, h in enumerate(headers):
        pdf.set_x(x)
        pdf.cell(col_widths[i], 6, f" {h}", border=1, fill=True)
        x += col_widths[i]
    pdf.ln()

    # Rows
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("gothic", "", 8)
    for row in rows:
        x = lm
        max_h = 6
        # Calculate max height needed
        cell_lines = []
        for i, cell in enumerate(row):
            lines = pdf.multi_cell(
                col_widths[i] - 2, 5, cell,
                dry_run=True, output="LINES"
            )
            cell_lines.append(lines)
            h = len(lines) * 5 + 1
            if h > max_h:
                max_h = h

        y_start = pdf.get_y()
        for i, cell in enumerate(row):
            pdf.set_x(x)
            pdf.rect(x, y_start, col_widths[i], max_h)
            pdf.set_x(x + 1)
            pdf.set_y(y_start)
            pdf.multi_cell(
                col_widths[i] - 2, 5, cell,
                new_x=XPos.RIGHT, new_y=YPos.TOP
            )
            x += col_widths[i]
        pdf.set_y(y_start + max_h)
    pdf.ln(3)


# ── Flowchart drawing ──────────────────────────────────────────

def fc_box(pdf, x, y, w, h, text, fill_color=(240, 245, 255),
           border_color=(30, 45, 80), text_color=(0, 0, 0), font_size=7):
    """Draw a rounded-ish box with centered text."""
    pdf.set_fill_color(*fill_color)
    pdf.set_draw_color(*border_color)
    pdf.set_line_width(0.4)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_text_color(*text_color)
    pdf.set_font("gothic", "B", font_size)
    lines = text.split("\n")
    line_h = font_size * 0.4
    total_h = line_h * len(lines)
    start_y = y + (h - total_h) / 2
    for i, line in enumerate(lines):
        tw = pdf.get_string_width(line)
        pdf.set_xy(x + (w - tw) / 2, start_y + i * line_h)
        pdf.cell(tw, line_h, line)
    pdf.set_line_width(0.2)


def fc_arrow(pdf, x1, y1, x2, y2, color=(30, 45, 80)):
    """Draw arrow line with arrowhead."""
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.5)
    pdf.line(x1, y1, x2, y2)
    # arrowhead
    ah = 1.5
    pdf.line(x2 - ah, y2 - ah, x2, y2)
    pdf.line(x2 + ah, y2 - ah, x2, y2)
    pdf.set_line_width(0.2)


def fc_diamond(pdf, cx, cy, half_w, half_h, text,
               fill_color=(255, 240, 220), border_color=(180, 100, 30)):
    """Draw diamond (decision) shape."""
    pdf.set_fill_color(*fill_color)
    pdf.set_draw_color(*border_color)
    pdf.set_line_width(0.4)
    # Draw diamond as polygon
    points = [
        (cx, cy - half_h),
        (cx + half_w, cy),
        (cx, cy + half_h),
        (cx - half_w, cy),
    ]
    # Fill with lines approach
    for dy in range(-int(half_h), int(half_h) + 1):
        ratio = 1 - abs(dy) / half_h
        lx = cx - half_w * ratio
        rx = cx + half_w * ratio
        pdf.set_fill_color(*fill_color)
        pdf.line(lx, cy + dy, rx, cy + dy)
    # Border
    pdf.set_draw_color(*border_color)
    for i in range(4):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 4]
        pdf.line(x1, y1, x2, y2)
    # Text
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("gothic", "B", 6)
    tw = pdf.get_string_width(text)
    pdf.set_xy(cx - tw / 2, cy - 2)
    pdf.cell(tw, 4, text)
    pdf.set_line_width(0.2)


def fc_label(pdf, x, y, text, color=(100, 100, 100), size=6):
    """Small label near arrows."""
    pdf.set_text_color(*color)
    pdf.set_font("gothic", "", size)
    pdf.set_xy(x, y)
    pdf.cell(pdf.get_string_width(text), 3, text)


def draw_flowchart(pdf, w):
    """Draw the full pipeline flowchart."""
    pdf.add_page()
    section_header(pdf, "フローチャート: Ultimate Executive Pipeline v2", w)
    pdf.ln(2)

    # Dimensions
    bw = 90  # box width
    bh = 16  # box height
    cx = pdf.l_margin + w / 2  # center x
    bx = cx - bw / 2  # box left x
    gap = 5  # gap between boxes

    y = pdf.get_y() + 2

    # START
    fc_box(pdf, bx, y, bw, 10, "START: CLI Entry\n--idea / --idea-file / --step / --from",
           fill_color=(30, 45, 80), text_color=(255, 255, 255), font_size=7)
    y += 10
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # API Key Check
    fc_box(pdf, bx, y, bw, 12, "APIキー検証 (Anthropic + Tavily)\n不在 → ERROR終了 (mock禁止)",
           fill_color=(255, 230, 230), border_color=(180, 50, 50))
    y += 12
    fc_arrow(pdf, cx, y, cx, y + gap)
    fc_label(pdf, cx + 2, y, "OK", color=(0, 128, 0))
    y += gap

    # Checkpoint Check
    fc_box(pdf, bx, y, bw, 12,
           "チェックポイント検査\n既存output/ → スキップ / --from N → Step Nから再開")
    y += 12
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # Step 0
    fc_box(pdf, bx, y, bw, bh,
           "Step 0: The Scout (市場調査)\nTavily API → 競合・市場規模・トレンド\n→ 00_market_data.md",
           fill_color=(220, 240, 255))
    y += bh
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # Step 1
    fc_box(pdf, bx, y, bw, bh,
           "Step 1: The Visionary (孫正義)\n100倍構想拡大・パラダイムシフト\n→ 01_vision.md",
           fill_color=(255, 245, 220))
    y += bh
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # Step 2
    fc_box(pdf, bx, y, bw, bh,
           "Step 2: The Strategist (Peter Thiel)\n独占戦略・Secretの発見\n→ 02_strategy.md",
           fill_color=(220, 240, 220))
    y += bh
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # Step 2.5 (NEW)
    fc_box(pdf, bx, y, bw, bh,
           "Step 2.5: Devil's Advocate (マンガー)【新設】\n逆転思考による致命傷探索\n→ 02a_devils_advocate.md",
           fill_color=(255, 220, 220), border_color=(200, 50, 50))
    y += bh
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # FATAL Diamond
    dh = 10
    fc_diamond(pdf, cx, y + dh, 20, dh, "[FATAL] タグあり？")
    # Yes branch (left)
    left_x = cx - 20
    pdf.set_draw_color(200, 50, 50)
    pdf.set_line_width(0.5)
    pdf.line(left_x, y + dh, left_x - 15, y + dh)
    fc_label(pdf, left_x - 25, y + dh - 4, "Yes", color=(200, 50, 50))
    fc_label(pdf, left_x - 38, y + dh, "--strict時", color=(200, 50, 50))
    fc_label(pdf, left_x - 33, y + dh + 3.5, "→ 停止", color=(200, 50, 50))
    # No branch (down)
    fc_arrow(pdf, cx, y + dh * 2, cx, y + dh * 2 + gap)
    fc_label(pdf, cx + 2, y + dh * 2 - 2, "No", color=(0, 128, 0))
    pdf.set_line_width(0.2)

    y = y + dh * 2 + gap

    # Step 3a
    fc_box(pdf, bx, y, bw, bh,
           "Step 3a: The Artisan (Steve Jobs)【分離】\nUX設計・画面遷移・核心体験定義\n→ 03a_ux_design.md",
           fill_color=(240, 235, 255), border_color=(100, 60, 180))
    y += bh
    fc_arrow(pdf, cx, y, cx, y + gap)
    fc_label(pdf, cx + 3, y, "← 人間が確認・編集可", color=(100, 60, 180), size=5)
    y += gap

    # Step 3b
    fc_box(pdf, bx, y, bw, bh,
           "Step 3b: The Builder (Wozniak)【分離】\nプロトタイプコード生成\n→ 03b_product_and_code.md + prototype.py",
           fill_color=(240, 235, 255), border_color=(100, 60, 180))
    y += bh
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # Step 4
    fc_box(pdf, bx, y, bw, bh,
           "Step 4: The Scaler (Jeff Bezos)\nFlywheel設計・物流・顧客執着\n→ 04_scale.md",
           fill_color=(220, 240, 255))
    y += bh
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # Step 5
    fc_box(pdf, bx, y, bw, bh + 2,
           "Step 5: The Judge (Warren Buffett)【最適化】\nExecutive Summaryのみ入力 (必要時全文)\nMoat評価・Go/No-Go → 05_final_decision.md",
           fill_color=(255, 250, 220), border_color=(180, 150, 30))
    y += bh + 2
    fc_arrow(pdf, cx, y, cx, y + gap)
    y += gap

    # END
    fc_box(pdf, bx, y, bw, 12,
           "END: Go → 緑で祝辞+次のアクション\nNo-Go → 赤で理由+改善提案",
           fill_color=(30, 45, 80), text_color=(255, 255, 255))


def build_pdf():
    pdf = PipelinePDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_font("gothic", "", FONT_PATH)
    pdf.add_font("gothic", "B", FONT_BOLD_PATH)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    W = pdf.w - pdf.l_margin - pdf.r_margin

    # ── Title Page ──
    pdf.ln(30)
    pdf.set_font("gothic", "B", 24)
    pdf.cell(W, 14, "Ultimate Executive Pipeline", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("gothic", "B", 18)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(W, 10, "v2 改善提案書", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    pdf.set_draw_color(30, 45, 80)
    pdf.set_line_width(0.8)
    mid = pdf.l_margin + W / 2
    pdf.line(mid - 40, pdf.get_y(), mid + 40, pdf.get_y())
    pdf.ln(10)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("gothic", "", 11)
    pdf.cell(W, 8, "6段階AIエージェントによるビジネスアイデア評価パイプライン",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(W, 8, "原案の妥当性評価と改善提案",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)

    pdf.set_font("gothic", "", 9)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(W, 6, "Asset Shield Project  |  2026-02-16",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)

    # ── Page 2: 総評 ──
    pdf.add_page()
    section_header(pdf, "総評", W)
    bold_body(pdf, "構想は秀逸、されど実戦には幾つかの隘路あり", W, 10)
    body_text(pdf,
        "全体の設計思想――チェックポイント制による段階的AI処理パイプライン――は"
        "堅実にして実用的である。が、仔細に検分すれば以下の問題が浮かび上がる。",
        W)

    # ── 設計の優れた点 ──
    section_header(pdf, "一、設計の優れたる点", W)

    sub_header(pdf, "チェックポイント・再開機構", W)
    bullet(pdf, [
        "各段階をMarkdownファイルに永続化する設計は極めて妥当",
        "人間が中間ファイルを編集してから次段階に進める仕組みは、AI任せの暴走を防ぐ良き統治機構",
        "--step 引数による個別段階実行も、デバッグ・運用の両面で有用",
    ], W)

    sub_header(pdf, "段階的フィルタリング", W)
    bullet(pdf, [
        "「発散→収束」の流れ（Son → Thiel → Jobs → Bezos → Buffett）は論理的に整合",
        "各段階が前段階の出力のみに依存する線形パイプラインは、実装・デバッグが容易",
    ], W)

    # ── 問題点 ──
    section_header(pdf, "二、問題点・懸念事項", W)

    # Problem 1
    sub_header(pdf, "【致命的】Step 0: The Scout の設計が曖昧", W)
    body_text(pdf,
        "原案は「search toolがなければmockせよ」とするが、此処が最大の弱点である。"
        "実際のビジネス判断に使うならば：", W)
    bullet(pdf, [
        "Web検索API（Tavily, SerpAPI, Brave Search等）が必須だが、指定なし",
        "モック（偽データ）で進めれば、以降全段階が砂上の楼閣となる",
        "Anthropic APIだけでは2025年5月以降の市場データを持たぬ",
        "Step 0が機能せねば、全パイプラインの価値が半減する",
    ], W)

    # Problem 2
    sub_header(pdf, "【重大】Step 3: Coding Agent の野心過大", W)
    body_text(pdf,
        "「戦略文書からプロトタイプコード生成」を一発で求めるのは精度が極めて低い。", W)
    bullet(pdf, [
        "生成コードは動作検証されぬまま prototype.py に保存される。実行不能な成果物となる危険大",
        "UX設計とコード生成を一つのステップに詰め込むのは責務過多。分離すべし",
    ], W)

    # Problem 3
    sub_header(pdf, "【重大】ペルソナ・プロンプトの罠", W)
    body_text(pdf, "各「経営者ペルソナ」には構造的な問題がある：", W)

    draw_table(pdf,
        ["ペルソナ", "期待される効果", "実際の懸念"],
        [
            ["孫正義", "100倍の発想拡大",
             "荒唐無稽な妄想に暴走しやすい。「300年計画」と言い出す"],
            ["Peter Thiel", "独占戦略の発見",
             "何でも「競争は負け犬」と切り捨て、建設的提案が薄くなる"],
            ["Steve Jobs", "UXへの執着",
             "「美しいが実装不可能」な設計を生み出す"],
            ["Jeff Bezos", "フライホイール",
             "何でもAWSに載せろと言い出す危険"],
            ["Warren Buffett", "冷静な判断",
             "前段階が全て楽観的なら、最終段で否決されるだけ。時既に遅し"],
        ],
        W, col_ratios=[0.18, 0.25, 0.57])

    bold_body(pdf,
        "根本問題：楽観→楽観→楽観→楽観→否決、では工数が無駄になる。"
        "Buffettの批判的視点はStep 2の直後（早期）に入れるべき。", W)

    # Problem 4
    sub_header(pdf, "【中程度】コンテキスト窓の圧迫", W)
    bullet(pdf, [
        "Step 5（Buffett）は全5段階の出力を入力に取る",
        "各段階の出力が仮に2000-3000トークンとすると、Step 5の入力は15,000トークン超",
        "システムプロンプト込みで入力が膨大になり、応答品質が落ちる可能性あり",
    ], W)

    # Problem 5
    sub_header(pdf, "【軽微】CLI設計の不足", W)
    bullet(pdf, [
        "--idea を毎回引数で渡すのは長文アイデアには不便。--idea-file オプションも欲しい",
        "--step all と --step 0 の挙動が曖昧。途中失敗時の挙動が未定義",
    ], W)

    # ── 改善提言 ──
    section_header(pdf, "三、改善提言 詳説", W)

    # Improvement 1
    sub_header(pdf, "改善1: Step 0 検索機構の厳格化", W)
    bold_body(pdf, "問題: 「mockせよ」は砂上の楼閣を招く", W)
    body_text(pdf, "改善内容：", W)
    bullet(pdf, [
        "Tavily Search APIを第一候補とする（Anthropic公式のtool use例でも採用、月1000回無料）",
        "APIキー不在時はエラー終了。偽データで先に進ませぬ",
        "検索結果は「競合」「市場規模」「トレンド」の3軸で構造化して保存",
    ], W)
    body_text(pdf,
        "根拠：パイプラインの入口が腐れば出口も腐る。"
        "「Garbage In, Garbage Out」の鉄則。", W)

    # Improvement 2
    sub_header(pdf, "改善2: 早期関門（Devil's Advocate）の挿入", W)
    bold_body(pdf, "問題: 否定的検証がStep 5まで来ない", W)
    body_text(pdf, "改善内容：", W)
    bullet(pdf, [
        "Step 2（Thiel）の直後にStep 2.5: Devil's Advocate（チャーリー・マンガー）を挿入",
        "マンガーの「逆転の思考（Inversion）」で「この事業が失敗するなら何故か？」を徹底洗出",
        "致命的欠陥が発見された場合、ファイルに [FATAL] タグを付与",
        "[FATAL] タグがあれば次段階開始時に警告を表示し、--strict モードでは停止",
    ], W)
    body_text(pdf,
        "根拠：Buffettの盟友マンガーの口癖は「逆から考えろ」。"
        "早期に致命傷を発見すれば、Step 3以降の無駄な工数を省ける。", W)

    # Improvement 3
    sub_header(pdf, "改善3: Step 3 の責務分離", W)
    bold_body(pdf, "問題: UX設計とコード生成が混在", W)
    body_text(pdf, "改善内容：", W)
    bullet(pdf, [
        "Step 3a: The Artisan（Steve Jobs）→ UX設計・画面遷移・核心機能の定義のみ",
        "Step 3b: The Builder（Coding Agent / Wozniak）→ 3aの設計書に基づきコード生成",
        "3aと3bの間にチェックポイントを置き、UX設計を人間が確認・修正してからコード生成に進める",
    ], W)
    body_text(pdf,
        "根拠：Jobsは「何を作るか」の天才であってコードは書かぬ。"
        "Wozniakの領分を分けるべし。", W)

    # Improvement 4
    sub_header(pdf, "改善4: Step 5 のコンテキスト最適化", W)
    bold_body(pdf, "問題: 全段階全文投入でコンテキスト窓を圧迫", W)
    body_text(pdf, "改善内容：", W)
    bullet(pdf, [
        "各段階の出力ファイル末尾に ## Executive Summary セクション（5行以内）を必ず付与させる",
        "Step 5（Buffett）には各段階のExecutive Summaryのみを入力",
        "詳細が必要な段階があれば、Buffettが「追加情報要求」として指定 → 該当ファイル全文を二次入力",
    ], W)
    body_text(pdf,
        "根拠：Buffettは年次報告書を読む男。"
        "要約から入り、必要な箇所だけ深掘りするのが彼の流儀。", W)

    # Improvement 5
    sub_header(pdf, "改善5: CLI の強化", W)
    draw_table(pdf,
        ["項目", "現状", "改善後"],
        [
            ["アイデア入力", "--idea \"テキスト\" のみ",
             "--idea-file path も追加。長文対応"],
            ["段階実行", "--step 3 で単一実行",
             "--from 3 で「3以降全実行」を追加"],
            ["失敗時挙動", "未定義",
             "--strict で失敗即停止、デフォルトは警告続行"],
            ["出力先", "output/ 固定",
             "--output-dir で指定可能"],
        ],
        W, col_ratios=[0.18, 0.35, 0.47])

    # ── 原案との差分まとめ ──
    section_header(pdf, "四、原案 (v1) と改善案 (v2) の差分まとめ", W)

    draw_table(pdf,
        ["項目", "原案 (v1)", "改善案 (v2)"],
        [
            ["段階数", "6段階 (Step 0-5)",
             "8段階 (Step 0, 1, 2, 2.5, 3a, 3b, 4, 5)"],
            ["早期否定検証", "なし（Step 5まで放置）",
             "Step 2.5で致命傷を早期検出"],
            ["Step 3", "UX + コード一括",
             "設計(3a)とコード(3b)を分離"],
            ["Step 5 入力", "全文投入",
             "Executive Summary → 必要時のみ全文"],
            ["検索API不在時", "mock許容",
             "エラー終了"],
            ["CLI", "--step, --idea",
             "--from, --idea-file, --strict, --output-dir 追加"],
            ["FATAL分岐", "なし",
             "[FATAL]タグで早期警告・停止"],
        ],
        W, col_ratios=[0.2, 0.35, 0.45])

    # ── 費用対効果 ──
    section_header(pdf, "五、費用対効果", W)
    body_text(pdf,
        "Anthropic API（Claude Sonnet 4.5 想定）で全8段階を実行した場合：", W)
    bullet(pdf, [
        "入力: 概算 ~35,000トークン（全段階合計、v2は2段階増）",
        "出力: 概算 ~18,000トークン（全段階合計）",
        "1回実行あたり約 $0.25 - 0.50（Sonnet 4.5 価格）",
        "Opus 使用なら $2.00 - 4.00",
        "ビジネスアイデアの初期検討ツールとしては十分に安価",
    ], W)

    # ── 結論 ──
    section_header(pdf, "結論", W)
    pdf.ln(2)
    bold_body(pdf, "設計思想は7割方正しい。", W, 11)
    body_text(pdf,
        "チェックポイント制・再開機構・段階的フィルタリングは実用的である。", W, 10)
    pdf.ln(2)
    body_text(pdf,
        "されど、以下の三点は実戦投入前に必ず修正すべき隘路である：", W, 10)
    pdf.ln(1)
    pdf.set_font("gothic", "B", 10)
    for i, item in enumerate([
        "Step 0 の検索機構の曖昧さ → Tavily API 必須化で解決",
        "Step 3 の責務過多 → UX設計とコード生成の分離で解決",
        "批判的視点の配置が遅すぎる → Step 2.5 マンガー挿入で解決",
    ], 1):
        pdf.set_x(pdf.l_margin + 5)
        pdf.multi_cell(W - 5, 6, f"{i}. {item}",
                       align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Flowchart ──
    draw_flowchart(pdf, W)

    pdf.output(OUTPUT_PATH)
    print(f"PDF saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
