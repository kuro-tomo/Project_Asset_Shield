#!/usr/bin/env python3
"""Shopify MCP Server 設計書 PDF生成 — ツリー構成 & フローチャート"""

from fpdf import FPDF

FONT_PATH_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_PATH_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_PATH_W8 = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"
OUTPUT = "/Users/MBP/Desktop/Shopify_MCP_Server_Design.pdf"

# Colors
NAVY = (26, 42, 82)
DARK = (33, 33, 33)
GRAY = (100, 100, 100)
LIGHT_BG = (245, 247, 250)
WHITE = (255, 255, 255)
ACCENT = (0, 102, 204)
ROW_ALT = (240, 244, 250)
HEADER_BG = (26, 42, 82)
GREEN = (16, 120, 67)
ORANGE = (210, 105, 0)
TEAL = (0, 128, 128)
LIGHT_NAVY = (220, 228, 242)


class DesignPDF(FPDF):
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
        self.cell(0, 6, "Shopify MCP Server — Architecture Design Document", align="L")
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
        self.set_font("HGH", "", 28)
        self.set_text_color(*NAVY)
        self.cell(0, 14, "Shopify MCP Server", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_font("HGB", "", 18)
        self.cell(0, 12, "Architecture Design Document", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("HGB", "", 14)
        self.set_text_color(*ACCENT)
        self.cell(0, 10, "Project Tree & Flow Charts", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.5)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(12)
        self.set_font("HG", "", 11)
        self.set_text_color(*DARK)
        for line in [
            "Date: 2026-02-18",
            "Author: 老中 (Claude Opus 4.6)",
            "To: 上様",
            "Status: Design Phase — Implementation Pending",
        ]:
            self.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(16)

        # Summary box
        x0 = 25
        w = 160
        self.set_fill_color(*LIGHT_BG)
        self.set_draw_color(*NAVY)
        self.rect(x0, self.get_y(), w, 40, style="DF")
        self.set_xy(x0, self.get_y() + 4)
        self.set_font("HGB", "", 11)
        self.set_text_color(*NAVY)
        self.cell(w, 7, "概要", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_x(x0 + 8)
        self.ln(1)
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.set_x(x0 + 8)
        self.multi_cell(w - 16, 6,
            "熨斗アプリを母体とし、Shopify実APIに準拠したMCPサーバーを構築。\n"
            "AIエージェントがShopify店舗を操作可能にするDeveloper Kitとして\n"
            "海外開発者に販売する。日本式ギフト文化の知見が差別化の核。"
        )

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

    # ──────── Tree rendering ────────
    def tree_line(self, depth, connector, name, description="", is_dir=False):
        """Render one line of a directory tree."""
        indent = 12 + depth * 10
        self.set_x(indent)
        self.set_font("HG", "", 8.5)
        self.set_text_color(*GRAY)
        self.cell(12, 5.5, connector)

        if is_dir:
            self.set_font("HGB", "", 8.5)
            self.set_text_color(*NAVY)
        else:
            self.set_font("HG", "", 8.5)
            self.set_text_color(*DARK)
        self.cell(50, 5.5, name)

        if description:
            self.set_font("HG", "", 7.5)
            self.set_text_color(*GRAY)
            self.cell(0, 5.5, description, new_x="LMARGIN", new_y="NEXT")
        else:
            self.ln(5.5)

    # ──────── Flowchart box ────────
    def flow_box(self, x, y, w, h, text, bg_color=LIGHT_BG, text_color=NAVY, border_color=NAVY):
        """Draw a box for flowchart."""
        self.set_fill_color(*bg_color)
        self.set_draw_color(*border_color)
        self.set_line_width(0.4)
        self.rect(x, y, w, h, style="DF")
        self.set_xy(x, y + 1)
        self.set_font("HGB", "", 8)
        self.set_text_color(*text_color)
        self.multi_cell(w, 4.5, text, align="C")

    def flow_arrow_down(self, x, y1, y2):
        """Draw a downward arrow."""
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(x, y1, x, y2)
        # arrowhead
        self.line(x - 1.5, y2 - 2.5, x, y2)
        self.line(x + 1.5, y2 - 2.5, x, y2)

    def flow_arrow_right(self, x1, x2, y):
        """Draw a rightward arrow."""
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(x1, y, x2, y)
        self.line(x2 - 2.5, y - 1.5, x2, y)
        self.line(x2 - 2.5, y + 1.5, x2, y)

    def render_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            col_widths = [190 / n] * n
        self.set_font("HGB", "", 8.5)
        self.set_fill_color(*HEADER_BG)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("HG", "", 8.5)
        self.set_text_color(*DARK)
        for ri, row in enumerate(rows):
            if ri % 2 == 1:
                self.set_fill_color(*ROW_ALT)
            else:
                self.set_fill_color(*WHITE)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1, fill=True,
                          align="C" if i > 0 else "L")
            self.ln()
        self.ln(3)


def build():
    pdf = DesignPDF()

    # ============================================================
    # COVER PAGE
    # ============================================================
    pdf.cover_page()

    # ============================================================
    # PAGE 2: PROJECT TREE STRUCTURE
    # ============================================================
    pdf.section_title("1. Project Tree Structure")
    pdf.body_text(
        "以下はShopify MCP Serverプロジェクトの完全なディレクトリ構成である。"
        "熨斗アプリの資産を継承しつつ、MCP準拠のツール群とShopify API連携層を新設する。"
    )
    pdf.ln(2)

    # Tree background
    tree_y = pdf.get_y()
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*NAVY)
    pdf.rect(10, tree_y, 190, 155, style="DF")
    pdf.ln(3)

    # Root
    tree = [
        (0, "",   "shopify-mcp-server/",     "", True),
        (0, "├─", "package.json",            "Node.js 設定 / MCP SDK依存", False),
        (0, "├─", "tsconfig.json",           "TypeScript 設定", False),
        (0, "├─", "wrangler.toml",           "Cloudflare Workers Deploy設定", False),
        (0, "├─", ".env.example",            "環境変数テンプレート", False),
        (0, "├─", "README.md",              "Developer Kit ドキュメント", False),
        (0, "├─", "LICENSE",                 "MIT License", False),
        (0, "├─", "src/",                    "",  True),
        (1, "├─", "index.ts",               "MCP Server エントリポイント", False),
        (1, "├─", "server.ts",              "MCP Server 初期化 & ツール登録", False),
        (1, "├─", "tools/",                  "",  True),
        (2, "├─", "search-products.ts",      "商品検索 (Storefront API)", False),
        (2, "├─", "manage-cart.ts",          "カート操作 (CRUD)", False),
        (2, "├─", "apply-discount.ts",       "割引適用 (Discount Functions)", False),
        (2, "├─", "configure-noshi.ts",      "熨斗・ギフト設定 (差別化コア)", False),
        (2, "├─", "check-inventory.ts",      "在庫確認", False),
        (2, "├─", "create-checkout.ts",      "決済フロー開始 (Checkout API)", False),
        (2, "└─", "track-order.ts",          "注文追跡 (Admin API)", False),
        (1, "├─", "shopify/",                "",  True),
        (2, "├─", "client.ts",              "Shopify GraphQL クライアント", False),
        (2, "├─", "storefront.ts",          "Storefront API ラッパー", False),
        (2, "├─", "admin.ts",               "Admin API ラッパー", False),
        (2, "└─", "types.ts",               "Shopify API 型定義", False),
        (1, "├─", "middleware/",              "",  True),
        (2, "├─", "auth.ts",               "OAuth 2.0 + Session Token", False),
        (2, "├─", "rate-limiter.ts",        "Shopify API レート制限", False),
        (2, "└─", "validator.ts",           "入力検証 / 幻覚防止ガード", False),
        (1, "├─", "noshi/",                   "",  True),
        (2, "├─", "engine.ts",              "熨斗判定エンジン (文脈→種類)", False),
        (2, "├─", "templates.ts",           "熨斗テンプレート定義", False),
        (2, "└─", "i18n.ts",               "多言語ギフトガイド", False),
        (1, "└─", "utils/",                   "",  True),
        (2, "├─", "logger.ts",              "構造化ログ", False),
        (2, "└─", "errors.ts",              "エラーハンドリング", False),
        (0, "├─", "tests/",                  "",  True),
        (1, "├─", "tools/",                  "ツール単体テスト",  True),
        (1, "├─", "middleware/",              "ミドルウェアテスト",  True),
        (1, "├─", "noshi/",                  "熨斗エンジンテスト",  True),
        (1, "└─", "integration/",            "E2Eテスト (MSW)",  True),
        (0, "├─", "docs/",                   "",  True),
        (1, "├─", "setup-guide.md",          "初期設定ガイド", False),
        (1, "├─", "tool-reference.md",       "全ツールAPI仕様書", False),
        (1, "├─", "noshi-guide.md",          "熨斗機能ガイド (EN/JP)", False),
        (1, "└─", "deployment.md",           "Cloudflare Workers デプロイ手順", False),
        (0, "└─", "examples/",               "",  True),
        (1, "├─", "basic-search.ts",         "商品検索サンプル", False),
        (1, "├─", "gift-purchase.ts",        "ギフト購入フロー", False),
        (1, "└─", "claude-desktop.json",     "Claude Desktop 設定例", False),
    ]

    for depth, connector, name, desc, is_dir in tree:
        pdf.tree_line(depth, connector, name, desc, is_dir)

    pdf.ln(6)

    # ============================================================
    # PAGE 3: ARCHITECTURE FLOWCHART
    # ============================================================
    pdf.section_title("2. Architecture Flowchart — System Overview")
    pdf.body_text(
        "AIエージェントからShopify店舗APIまでの全体データフロー。"
        "MCP Serverが中間ゲートウェイとして認証・検証・レート制限を担う。"
    )
    pdf.ln(2)

    # Layer 1: AI Agent
    bw = 56  # box width
    bh = 12
    cx = 105  # center x

    y = pdf.get_y()
    pdf.flow_box(cx - bw/2, y, bw, bh, "AI Agent\n(Claude / GPT / Custom)", NAVY, WHITE)
    pdf.flow_arrow_down(cx, y + bh, y + bh + 8)

    # Layer 2: MCP Protocol
    y2 = y + bh + 8
    pdf.flow_box(cx - bw/2, y2, bw, 9, "MCP Protocol (stdio/SSE)", LIGHT_NAVY, NAVY)
    pdf.flow_arrow_down(cx, y2 + 9, y2 + 17)

    # Layer 3: MCP Server (big box)
    y3 = y2 + 17
    server_h = 58
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.6)
    pdf.rect(20, y3, 170, server_h, style="DF")
    pdf.set_xy(20, y3 + 1)
    pdf.set_font("HGH", "", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(170, 6, "Shopify MCP Server", align="C", new_x="LMARGIN", new_y="NEXT")

    # Middleware row
    mw_y = y3 + 9
    mw_bw = 48
    mw_bh = 10
    pdf.flow_box(25, mw_y, mw_bw, mw_bh, "Auth Middleware\nOAuth 2.0 + Session", (230, 240, 250), DARK, GRAY)
    pdf.flow_box(76, mw_y, mw_bw, mw_bh, "Rate Limiter\nShopify API Quota", (230, 240, 250), DARK, GRAY)
    pdf.flow_box(127, mw_y, mw_bw, mw_bh, "Validator\nInput Guard", (230, 240, 250), DARK, GRAY)

    # Tools row
    tool_y = mw_y + mw_bh + 4
    tw = 36
    th = 18
    tool_names = [
        "search\nproducts",
        "manage\ncart",
        "apply\ndiscount",
        "configure\nnoshi",
    ]
    tool_colors = [ACCENT, GREEN, ORANGE, (180, 30, 60)]
    for i, (tname, tcol) in enumerate(zip(tool_names, tool_colors)):
        tx = 24 + i * 40
        pdf.flow_box(tx, tool_y, tw, th, tname, tcol, WHITE, tcol)

    tool_y2 = tool_y
    more_tools = [
        "check\ninventory",
        "create\ncheckout",
        "track\norder",
    ]
    more_colors = [TEAL, NAVY, GRAY]

    # Second tool row (if space)
    tool_y2 = tool_y + th + 3
    for i, (tname, tcol) in enumerate(zip(more_tools, more_colors)):
        tx = 44 + i * 40
        pdf.flow_box(tx, tool_y2, tw, th - 4, tname, tcol, WHITE, tcol)

    # Arrow down from server
    y4 = y3 + server_h
    pdf.flow_arrow_down(cx, y4, y4 + 8)

    # Layer 4: Shopify APIs
    y5 = y4 + 8
    api_w = 52
    pdf.flow_box(25, y5, api_w, 12, "Storefront API\nGraphQL", GREEN, WHITE, GREEN)
    pdf.flow_box(80, y5, api_w, 12, "Admin API\nGraphQL", ACCENT, WHITE, ACCENT)
    pdf.flow_box(135, y5, api_w, 12, "Checkout API\nREST", ORANGE, WHITE, ORANGE)

    # Arrow down to store
    pdf.flow_arrow_down(cx, y5 + 12, y5 + 20)
    y6 = y5 + 20
    pdf.flow_box(cx - 35, y6, 70, 10, "Shopify Store", NAVY, WHITE)

    pdf.set_y(y6 + 16)

    # ============================================================
    # PAGE 4: REQUEST FLOW (Sequence)
    # ============================================================
    pdf.section_title("3. Request Flow — 商品検索からチェックアウトまで")
    pdf.body_text("AIエージェントがギフト購入を完了するまでの典型的なリクエストフロー。")
    pdf.ln(2)

    # Sequence diagram as table-like structure
    steps = [
        ("1", "Agent → MCP", "search_products", "「母の日 和菓子 3000円以内」で商品検索", ACCENT),
        ("2", "MCP → Shopify", "Storefront API", "GraphQL productSearch クエリ実行", GREEN),
        ("3", "Shopify → MCP", "Response", "商品リスト返却 (名前, 価格, 在庫, 画像)", GREEN),
        ("4", "MCP → Agent", "Tool Result", "構造化された商品データをJSON返却", ACCENT),
        ("5", "Agent → MCP", "configure_noshi", "蝶結び / 御祝 / 送り主名を指定", (180, 30, 60)),
        ("6", "MCP (内部)", "Noshi Engine", "文脈判定: 母の日→蝶結び確認, テンプレ選択", (180, 30, 60)),
        ("7", "Agent → MCP", "manage_cart", "商品+熨斗オプションをカートに追加", GREEN),
        ("8", "Agent → MCP", "apply_discount", "ギフトクーポン適用 (Discount Functions)", ORANGE),
        ("9", "Agent → MCP", "create_checkout", "Checkout API で決済URL生成", ORANGE),
        ("10", "MCP → Agent", "Checkout URL", "ユーザーに決済リンクを提示", ACCENT),
    ]

    # Render as styled list
    for step_no, route, tool, desc, color in steps:
        y = pdf.get_y()
        if y > 260:
            pdf.add_page()

        # Step number circle
        pdf.set_fill_color(*color)
        pdf.set_font("HGH", "", 8)
        pdf.set_text_color(*WHITE)
        r = 4
        cx_s = 16
        cy_s = pdf.get_y() + r
        pdf.ellipse(cx_s - r, cy_s - r, 2*r, 2*r, style="F")
        pdf.set_xy(cx_s - r, cy_s - 2.5)
        pdf.cell(2*r, 5, step_no, align="C")

        # Route
        pdf.set_xy(24, cy_s - 4)
        pdf.set_font("HGB", "", 8)
        pdf.set_text_color(*DARK)
        pdf.cell(30, 5, route)

        # Tool name
        pdf.set_font("HGB", "", 8)
        pdf.set_text_color(*color)
        pdf.cell(32, 5, tool)

        # Description
        pdf.set_font("HG", "", 8)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 5, desc, new_x="LMARGIN", new_y="NEXT")

        # Connector line
        if step_no != "10":
            pdf.set_draw_color(*GRAY)
            pdf.set_line_width(0.2)
            lx = 16
            pdf.line(lx, pdf.get_y(), lx, pdf.get_y() + 3)
            pdf.ln(3)
        else:
            pdf.ln(2)

    pdf.ln(4)

    # ============================================================
    # PAGE 5: NOSHI ENGINE FLOW
    # ============================================================
    pdf.section_title("4. Noshi Engine — 熨斗判定フローチャート")
    pdf.body_text(
        "configure_noshi ツールの内部処理。AIエージェントから受けたギフト文脈を解析し、"
        "適切な熨斗の種類・水引・表書きを自動判定する。これが本製品の差別化の核である。"
    )
    pdf.ln(2)

    y = pdf.get_y()
    bw2 = 50
    bh2 = 11

    # Start
    pdf.flow_box(cx - bw2/2, y, bw2, bh2, "Agent: configure_noshi\n(occasion, recipient)", ACCENT, WHITE, ACCENT)
    pdf.flow_arrow_down(cx, y + bh2, y + bh2 + 6)

    # Decision: occasion type
    dy = y + bh2 + 6
    dw = 54
    dh = 12
    # Diamond shape approximated as rotated square (using rect for simplicity with label)
    pdf.set_fill_color(*LIGHT_NAVY)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.4)
    pdf.rect(cx - dw/2, dy, dw, dh, style="DF")
    pdf.set_xy(cx - dw/2, dy + 2)
    pdf.set_font("HGB", "", 8)
    pdf.set_text_color(*NAVY)
    pdf.cell(dw, 4, "occasion 判定", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(cx - dw/2)
    pdf.set_font("HG", "", 7)
    pdf.cell(dw, 4, "(祝儀 / 不祝儀 / 御礼 / その他)", align="C")

    # Three branches
    branch_y = dy + dh + 6
    branches = [
        (35,  "祝儀 (慶事)",  "蝶結び\n(何度あっても良い祝事)\n出産・入学・母の日", GREEN),
        (105, "結婚",         "結切り\n(一度きりの祝事)\n結婚祝・快気祝", ORANGE),
        (170, "不祝儀",       "結切り(黒白)\n弔事・香典返し", GRAY),
    ]

    for bx, label, detail, color in branches:
        # Arrow down from decision
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.3)
        # horizontal line from center to branch
        if bx != 105:
            pdf.line(cx, dy + dh/2, bx, dy + dh/2)
            pdf.line(bx, dy + dh/2, bx, branch_y)
        else:
            pdf.flow_arrow_down(cx, dy + dh, branch_y)

        pdf.flow_box(bx - 22, branch_y, 44, 20, f"{label}\n{detail}", color, WHITE, color)

    # Final merge
    merge_y = branch_y + 24
    for bx in [35, 105, 170]:
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.3)
        pdf.line(bx, branch_y + 20, bx, merge_y)

    pdf.line(35, merge_y, 170, merge_y)
    pdf.line(105, merge_y, 105, merge_y + 5)

    final_y = merge_y + 5
    pdf.flow_box(cx - 30, final_y, 60, 11, "表書き + 名入れ 生成\nJSON Response 返却", NAVY, WHITE)

    pdf.set_y(final_y + 18)

    # ============================================================
    # PAGE 6: TECH STACK & PRICING
    # ============================================================
    pdf.section_title("5. Tech Stack")
    pdf.render_table(
        ["Layer", "Technology", "Reason"],
        [
            ["Runtime", "Node.js + TypeScript", "MCP SDK 第一級サポート"],
            ["MCP SDK", "@modelcontextprotocol/sdk", "Anthropic 公式"],
            ["Shopify", "@shopify/shopify-api", "公式 GraphQL SDK"],
            ["Deploy", "Cloudflare Workers", "低コスト / グローバルEdge / ARM"],
            ["Auth", "Shopify OAuth 2.0", "公式認証フロー"],
            ["DB", "Cloudflare D1 (SQLite)", "セッション・設定保存"],
            ["Test", "Vitest + MSW", "高速テスト + API Mock"],
            ["CI/CD", "GitHub Actions", "自動テスト + 自動デプロイ"],
        ],
        [35, 65, 90],
    )

    pdf.section_title("6. Pricing Model (Developer Kit)", level=1)
    pdf.render_table(
        ["Tier", "Price", "Contents"],
        [
            ["Free (OSS)", "$0", "MCP Server Core (検索・在庫のみ)"],
            ["Pro", "$29/mo", "全7ツール + 熨斗エンジン + サポート"],
            ["Enterprise", "$99/mo", "カスタムDiscount Logic + 専用サポート"],
            ["Kit (一括)", "$49-199", "テンプレート + ドキュメント + サンプル"],
        ],
        [35, 30, 125],
    )

    pdf.ln(4)
    pdf.section_title("7. Competitive Advantage — 他に無い差別化", level=1)
    pdf.ln(2)

    advantages = [
        ("熨斗エンジン", "日本式ギフト文化を理解するMCPサーバーは世界初。インバウンド需要に直結"),
        ("MCP準拠", "Claude Desktop / Cursor等から即座に利用可能。プロトコル標準に先行"),
        ("Shopify公式API", "架空仕様でなく実API。Shopify App Storeへの掲載も視野"),
        ("Edge Deploy", "Cloudflare Workers で世界中から低レイテンシ。サーバーレスで運用コストゼロに近い"),
        ("Developer Kit", "海外開発者に売れるパッケージ。ドキュメント英語、コード英語"),
    ]
    for title, desc in advantages:
        pdf.set_font("HGB", "", 9.5)
        pdf.set_text_color(*NAVY)
        pdf.cell(40, 6, title)
        pdf.set_font("HG", "", 9)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 6, desc, new_x="LMARGIN", new_y="NEXT")

    # === Footer ===
    pdf.ln(10)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("HG", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "本設計書は Asset Shield 老中 (Claude Opus 4.6) が作成",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "上様の号令をもって実装を開始する",
             align="C")

    pdf.output(OUTPUT)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build()
