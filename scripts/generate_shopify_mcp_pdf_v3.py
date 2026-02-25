#!/usr/bin/env python3
"""Shopify Agentic MCP Server V3 — 自動売買特化設計書 (熨斗削除)"""

from fpdf import FPDF

FONT_PATH_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_PATH_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_PATH_W8 = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"
OUTPUT = "/Users/MBP/Desktop/Shopify_Agentic_MCP_V3.pdf"

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
RED = (180, 30, 60)
GOOGLE_BLUE = (66, 133, 244)
SHOPIFY_GREEN = (0, 128, 96)
PURPLE = (103, 58, 183)
GOLD = (180, 140, 20)


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
        self.cell(0, 6, "Shopify Agentic MCP Gateway V3 — Autonomous Commerce Architecture", align="L")
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

    # ── Cover ──
    def cover_page(self):
        self.add_page()
        self.ln(32)
        self.set_font("HGH", "", 26)
        self.set_text_color(*NAVY)
        self.cell(0, 14, "Shopify Agentic MCP Gateway", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("HGB", "", 15)
        self.set_text_color(*DARK)
        self.cell(0, 10, "Autonomous Commerce Architecture V3", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("HGB", "", 11)
        self.set_text_color(*RED)
        self.cell(0, 8, "UCP / AP2 / MCP Tri-Protocol — Fee-Collecting Gateway", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.5)
        self.line(50, self.get_y(), 160, self.get_y())
        self.ln(10)
        self.set_font("HG", "", 10.5)
        self.set_text_color(*DARK)
        for line in [
            "Date: 2026-02-18",
            "Author: 老中 (Claude Opus 4.6)",
            "To: 上様",
            "Version: V3 (自動売買特化 — 熨斗削除)",
            "Status: Design Phase — Awaiting Go Order",
        ]:
            self.cell(0, 7.5, line, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(8)

        # Protocol badges
        x0 = 25
        w = 160
        badge_y = self.get_y()
        self.set_fill_color(*LIGHT_BG)
        self.set_draw_color(*NAVY)
        self.rect(x0, badge_y, w, 18, style="DF")
        badges = [
            ("UCP", SHOPIFY_GREEN, "Google + Shopify (80+ Partners)"),
            ("AP2", GOOGLE_BLUE, "Google (60+ Partners)"),
            ("MCP", PURPLE, "Anthropic"),
        ]
        bx = x0 + 6
        for name, color, org in badges:
            self.set_xy(bx, badge_y + 2)
            self.set_fill_color(*color)
            self.set_font("HGH", "", 10)
            self.set_text_color(*WHITE)
            self.cell(20, 6, name, fill=True, align="C")
            self.set_fill_color(*LIGHT_BG)
            self.set_font("HG", "", 7)
            self.set_text_color(*GRAY)
            self.set_xy(bx - 2, badge_y + 10)
            self.cell(52, 5, org, align="C")
            bx += 54

        self.set_y(badge_y + 22)

        # Mission statement box
        box_y = self.get_y()
        self.set_fill_color(*LIGHT_BG)
        self.set_draw_color(*NAVY)
        self.rect(x0, box_y, w, 38, style="DF")
        self.set_xy(x0, box_y + 3)
        self.set_font("HGH", "", 11)
        self.set_text_color(*NAVY)
        self.cell(w, 7, "Mission", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_x(x0 + 8)
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.set_x(x0 + 8)
        self.multi_cell(w - 16, 5.5,
            "世界中のAIエージェント (Gemini, ChatGPT, Copilot, Claude) と\n"
            "Shopify店舗の間に立つ「関所」を構築する。\n"
            "UCP/AP2/MCPの三規格に準拠し、通過する全取引から0.5%を自動徴収。\n"
            "発表からわずか1ヶ月 — 動く実装は世界にほぼ無い。先行者総取りの局面。"
        )

    # ── Helpers ──
    def section_title(self, text, level=1):
        self.ln(3)
        if level == 1:
            if self.get_y() > 40:
                self.add_page()
            self.set_font("HGH", "", 15)
            self.set_text_color(*NAVY)
            self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*NAVY)
            self.set_line_width(0.6)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
        elif level == 2:
            self.set_font("HGB", "", 12)
            self.set_text_color(*NAVY)
            self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font("HGB", "", 10)
            self.set_text_color(*DARK)
            self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_font("HGB", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.set_x(self.l_margin)
        self.cell(6, 5.5, "・")
        self.multi_cell(180, 5.5, text)

    def tree_line(self, depth, conn, name, desc="", is_dir=False):
        indent = 12 + depth * 9
        self.set_x(indent)
        self.set_font("HG", "", 8)
        self.set_text_color(*GRAY)
        self.cell(10, 5, conn)
        if is_dir:
            self.set_font("HGB", "", 8)
            self.set_text_color(*NAVY)
        else:
            self.set_font("HG", "", 8)
            self.set_text_color(*DARK)
        self.cell(44, 5, name)
        if desc:
            self.set_font("HG", "", 7)
            self.set_text_color(*GRAY)
            self.cell(0, 5, desc, new_x="LMARGIN", new_y="NEXT")
        else:
            self.ln(5)

    def flow_box(self, x, y, w, h, text, bg=LIGHT_BG, fg=NAVY, border=NAVY):
        self.set_fill_color(*bg)
        self.set_draw_color(*border)
        self.set_line_width(0.4)
        self.rect(x, y, w, h, style="DF")
        self.set_xy(x, y + 1)
        self.set_font("HGB", "", 7.5)
        self.set_text_color(*fg)
        self.multi_cell(w, 4, text, align="C")

    def flow_arrow_down(self, x, y1, y2):
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(x, y1, x, y2)
        self.line(x - 1.5, y2 - 2.5, x, y2)
        self.line(x + 1.5, y2 - 2.5, x, y2)

    def flow_arrow_right(self, x1, x2, y):
        self.set_draw_color(*NAVY)
        self.set_line_width(0.4)
        self.line(x1, y, x2, y)
        self.line(x2 - 2.5, y - 1.5, x2, y)
        self.line(x2 - 2.5, y + 1.5, x2, y)

    def render_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            col_widths = [190 / n] * n
        self.set_font("HGB", "", 8)
        self.set_fill_color(*HEADER_BG)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("HG", "", 8)
        self.set_text_color(*DARK)
        for ri, row in enumerate(rows):
            if ri % 2 == 1:
                self.set_fill_color(*ROW_ALT)
            else:
                self.set_fill_color(*WHITE)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6.5, str(cell), border=1, fill=True,
                          align="C" if i > 0 else "L")
            self.ln()
        self.ln(2)


def build():
    pdf = DesignPDF()

    # ================================================================
    # COVER
    # ================================================================
    pdf.cover_page()

    # ================================================================
    # 1. WHY NOW — 市場機会
    # ================================================================
    pdf.section_title("1. Why Now — なぜ今、関所を建てるのか")
    pdf.body_text(
        "2026年1月11日、Google + ShopifyがUCPを、GoogleがAP2を正式発表。"
        "Walmart, Target, Etsy, Wayfair等80社以上が参画を表明した。"
        "AIエージェントが人間に代わって買い物をする時代の幕が開いた。"
    )
    pdf.ln(1)

    pdf.render_table(
        ["Event", "Date", "Impact"],
        [
            ["AP2 発表 (Google + 60社)", "2025-09", "AIの財布の標準規格が定まった"],
            ["UCP 発表 (Google + Shopify + 80社)", "2026-01-11", "AI商取引の共通言語が定まった"],
            ["Shopify Agentic Storefronts", "2026-01", "Shopify管理画面からAIチャネル一括管理"],
            ["ChatGPT + ACP (OpenAI + Stripe)", "2026-01", "競合プロトコルも同時に始動"],
            ["本プロジェクト開始", "2026-02-18", "発表から38日。動く実装はほぼ無い"],
        ],
        [55, 25, 110],
    )

    pdf.bold_text("先行者利益の窓は狭い。今すぐ建てねば、他の者が関所を建てる。")

    # ================================================================
    # 2. TRI-PROTOCOL
    # ================================================================
    pdf.section_title("2. Tri-Protocol Architecture — 三つの武器")

    pdf.render_table(
        ["Protocol", "Developer", "Role in Our System"],
        [
            ["UCP", "Google + Shopify", "共通商取引言語: Discovery → Negotiation → Checkout"],
            ["AP2", "Google (60+ Partners)", "AIの財布: 暗号署名Mandate (Intent/Cart/Payment)"],
            ["MCP", "Anthropic", "AI↔Server接続: Claude Desktop等から直接呼出し"],
        ],
        [18, 45, 127],
    )

    pdf.section_title("UCP 3層構造", 2)
    pdf.render_table(
        ["Layer", "Role", "Scope"],
        [
            ["Shopping Service", "コア取引プリミティブ", "checkout session, line items, totals, status"],
            ["Capabilities", "機能領域 (独立版管理)", "Checkout, Catalog, Orders — 各々バージョニング"],
            ["Extensions", "ドメイン特化 (合成型)", "fulfillment, discount, ap2_mandate 等"],
        ],
        [32, 50, 108],
    )

    pdf.section_title("UCP Checkout State Machine", 2)
    y = pdf.get_y()
    states = [
        (18, "incomplete", "情報不足\nAPIで自動解決", ORANGE),
        (76, "requires_escalation", "人間の入力必要\ncontinue_url発行", RED),
        (140, "ready_for_complete", "全情報完備\nAgentが決済完了", GREEN),
    ]
    for sx, label, desc, color in states:
        pdf.flow_box(sx, y, 54, 15, f"{label}\n{desc}", color, WHITE, color)
    pdf.flow_arrow_right(72, 76, y + 7.5)
    pdf.flow_arrow_right(130, 140, y + 7.5)
    pdf.set_y(y + 20)

    pdf.section_title("AP2 Mandate署名チェーン", 2)
    pdf.render_table(
        ["Mandate", "Signer", "Purpose"],
        [
            ["Intent Mandate", "User (事前委任)", "上限金額・カテゴリ・有効期限をAIに委任"],
            ["Cart Mandate", "Merchant → User", "カート内容・価格・配送条件を暗号固定"],
            ["Payment Mandate", "User → Bank/PSP", "AI発の取引であることを決済網に証明"],
        ],
        [32, 38, 120],
    )

    pdf.section_title("UCP Transport Bindings", 2)
    pdf.render_table(
        ["Transport", "Format", "Use Case"],
        [
            ["REST", "OpenAPI 3.x", "Primary HTTP — 上様のGateway main binding"],
            ["MCP", "OpenRPC", "LLMツール公開 (Claude Desktop, Cursor等)"],
            ["A2A", "Agent Card", "Agent間通信 (Gemini ↔ Copilot)"],
            ["Embedded", "JSON-RPC 2.0", "ブラウザ埋込チェックアウト"],
        ],
        [28, 38, 124],
    )

    # ================================================================
    # 3. SYSTEM CONCEPT
    # ================================================================
    pdf.section_title("3. System Concept — 関所モデル")
    pdf.body_text(
        "上様のシステムは「関所」(Gateway) である。"
        "世界中のAIエージェントがShopify店舗で自律購買する際、上様の関所を通る。"
        "通過する全ての銭から通行料 (0.5%) を自動徴収する。"
    )

    pdf.render_table(
        ["構成要素", "役割", "所在"],
        [
            ["鍛冶場 (Dev)", "Claude Codeを副官とし、コードを錬成・テスト", "MBP M4 Pro (48GB)"],
            ["関所 (Gateway)", "AIの注文受付 → UCP/AP2処理 → Shopify連携", "AWS Lambda (Graviton)"],
            ["金庫 (Ledger)", "全取引の0.5%を自動回収・蓄積", "上様のWallet / DynamoDB"],
        ],
        [28, 82, 80],
    )

    pdf.section_title("収益の血流", 2)
    # Flow diagram: 4 steps horizontal
    y = pdf.get_y()
    flow_steps = [
        (12, "探索\nAIが商品検索\nscout_inventory", ACCENT),
        (60, "交渉\nUCP Capability\nnegotiate_terms", GREEN),
        (108, "決済\nAP2 Mandate Chain\nexecute_checkout", ORANGE),
        (156, "徴収\n0.5% Auto-Deduct\nfee_collector", GOLD),
    ]
    for fx, ftxt, fcol in flow_steps:
        pdf.flow_box(fx, y, 44, 18, ftxt, fcol, WHITE, fcol)
    for i in range(3):
        pdf.flow_arrow_right(12 + 44 + i * 48, 60 + i * 48, y + 9)
    pdf.set_y(y + 24)
    pdf.ln(2)

    # ================================================================
    # 4. PROJECT TREE
    # ================================================================
    pdf.section_title("4. Project Tree — 自動売買特化構成")
    pdf.body_text(
        "熨斗関連を全て削除。UCP/AP2/MCP の三層と、"
        "自動売買に必要な5つのMCPツールに集中した構成。"
    )
    pdf.ln(1)

    tree_y = pdf.get_y()
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*NAVY)
    pdf.rect(10, tree_y, 190, 162, style="DF")
    pdf.ln(2)

    tree = [
        (0, "",   "shopify-agentic-mcp/",       "", True),
        (0, "├─", "package.json",               "@modelcontextprotocol/sdk, @shopify/shopify-api, jose", False),
        (0, "├─", "tsconfig.json",              "TypeScript 5.x strict", False),
        (0, "├─", "serverless.yml",             "AWS Lambda ARM (Graviton3) + API Gateway", False),
        (0, "├─", ".env.example",               "SHOPIFY_API_KEY, AP2_SIGNING_KEY, FEE_WALLET", False),
        (0, "├─", "README.md",                  "Developer Kit documentation (EN)", False),
        (0, "├─", "LICENSE",                     "MIT", False),
        (0, "├─", "src/",                        "",  True),
        (1, "├─", "index.ts",                   "Lambda handler + MCP Server entrypoint", False),
        (1, "├─", "server.ts",                  "MCP Server init, tool registration", False),
        (1, "├─", "ucp/",                        "UCP Protocol Implementation",  True),
        (2, "├─", "profile.ts",                 "/.well-known/ucp — profile publisher", False),
        (2, "├─", "negotiate.ts",               "Capability intersection algorithm", False),
        (2, "├─", "checkout-session.ts",         "3-state machine (incomplete→escalation→ready)", False),
        (2, "├─", "shopping-service.ts",         "Core primitives: line items, totals, currency", False),
        (2, "├─", "discovery.ts",               "Catalog search + Schema.org JSON-LD", False),
        (2, "└─", "schemas/",                    "",  True),
        (3, "├─", "checkout.schema.json",        "dev.ucp.shopping.checkout", False),
        (3, "├─", "catalog.schema.json",         "dev.ucp.shopping.catalog", False),
        (3, "├─", "order.schema.json",           "dev.ucp.shopping.order", False),
        (3, "└─", "fulfillment.schema.json",     "dev.ucp.shopping.fulfillment", False),
        (1, "├─", "ap2/",                         "AP2 Mandate Implementation",  True),
        (2, "├─", "verifier.ts",                 "Intent/Cart/Payment Mandate crypto verify", False),
        (2, "├─", "signer.ts",                   "Cart Mandate signing (merchant-side)", False),
        (2, "├─", "mandate-store.ts",            "Mandate履歴 DynamoDB persistence", False),
        (2, "└─", "types.ts",                    "Mandate type definitions", False),
        (1, "├─", "tools/",                       "MCP Tools — Agent向け5本刀",  True),
        (2, "├─", "scout-inventory.ts",          "UCP Catalog検索 (Storefront API)", False),
        (2, "├─", "negotiate-terms.ts",          "UCP Capability Negotiation + discount", False),
        (2, "├─", "execute-checkout.ts",         "AP2 Mandate chain → complete checkout", False),
        (2, "├─", "manage-cart.ts",              "Cart CRUD (UCP line items)", False),
        (2, "└─", "track-order.ts",              "Order追跡 + webhook (Admin API)", False),
        (1, "├─", "shopify/",                     "Shopify API Client Layer",  True),
        (2, "├─", "client.ts",                   "GraphQL client (Storefront + Admin)", False),
        (2, "├─", "storefront.ts",               "Storefront API wrapper", False),
        (2, "├─", "admin.ts",                    "Admin API wrapper", False),
        (2, "└─", "types.ts",                    "Shopify API type definitions", False),
        (1, "├─", "middleware/",                   "",  True),
        (2, "├─", "auth.ts",                     "Shopify OAuth 2.0 + UCP Profile auth", False),
        (2, "├─", "rate-limiter.ts",             "Shopify API quota + burst protection", False),
        (2, "├─", "guardrail.ts",                "AI hallucination guard (price/inventory)", False),
        (2, "└─", "fee-collector.ts",            "0.5% auto-deduction per transaction", False),
        (1, "└─", "utils/",                        "",  True),
        (2, "├─", "logger.ts",                   "Structured logging (CloudWatch)", False),
        (2, "├─", "crypto.ts",                   "JWS/JWK signing (jose library)", False),
        (2, "└─", "errors.ts",                   "UCP error response builder", False),
        (0, "├─", "tests/",                       "",  True),
        (1, "├─", "ucp/",                         "UCP negotiate, checkout, discovery",  True),
        (1, "├─", "ap2/",                         "Mandate sign/verify/store",  True),
        (1, "├─", "tools/",                       "5 MCP tool unit tests",  True),
        (1, "├─", "middleware/",                   "Auth, rate-limit, guardrail, fee",  True),
        (1, "└─", "e2e/",                         "Full purchase flow (MSW mock)",  True),
        (0, "├─", "docs/",                        "",  True),
        (1, "├─", "architecture.md",              "System architecture overview", False),
        (1, "├─", "ucp-integration.md",           "UCP compliance guide", False),
        (1, "├─", "ap2-mandates.md",              "AP2 Mandate implementation guide", False),
        (1, "├─", "deployment.md",                "AWS Lambda deploy (SAM/CDK)", False),
        (1, "└─", "developer-kit.md",             "Quick start for Kit purchasers", False),
        (0, "└─", "examples/",                     "",  True),
        (1, "├─", "autonomous-purchase.ts",        "Full E2E: search → negotiate → checkout", False),
        (1, "├─", "mandate-flow.ts",               "AP2 Mandate sign/verify example", False),
        (1, "└─", "claude-desktop.json",           "Claude Desktop MCP config", False),
    ]

    for depth, conn, name, desc, is_dir in tree:
        pdf.tree_line(depth, conn, name, desc, is_dir)

    pdf.ln(4)

    # ================================================================
    # 5. ARCHITECTURE FLOWCHART
    # ================================================================
    pdf.section_title("5. Architecture Flowchart — 関所の全体図")
    pdf.body_text(
        "4種のAIエージェント → UCP Discovery → Transport選択 → 関所 (Gateway) → Shopify。"
        "関所内部でUCP処理・AP2検証・手数料徴収が全自動で行われる。"
    )
    pdf.ln(1)

    cx = 105
    y = pdf.get_y()

    # Layer 1: AI Agents
    agents = [
        (22, "Gemini\n(Google)", GOOGLE_BLUE),
        (67, "ChatGPT\n(OpenAI)", GREEN),
        (112, "Copilot\n(MS)", ACCENT),
        (157, "Claude\n(Anthropic)", PURPLE),
    ]
    for ax, aname, acolor in agents:
        pdf.flow_box(ax, y, 38, 11, aname, acolor, WHITE, acolor)

    # Converge
    y_conv = y + 14
    for ax in [41, 86, 131, 176]:
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.25)
        pdf.line(ax, y + 11, ax, y_conv)
    pdf.line(41, y_conv, 176, y_conv)
    pdf.line(cx, y_conv, cx, y_conv + 4)

    # Layer 2: UCP Discovery
    y2 = y_conv + 4
    pdf.flow_box(cx - 38, y2, 76, 8, "UCP Profile Discovery  /.well-known/ucp", LIGHT_NAVY, NAVY)
    pdf.flow_arrow_down(cx, y2 + 8, y2 + 13)

    # Layer 3: Transport
    y3 = y2 + 13
    transports = [
        (28, "REST (OpenAPI)", ACCENT),
        (78, "MCP (OpenRPC)", PURPLE),
        (128, "A2A (Agent Card)", TEAL),
    ]
    for tx, tname, tcol in transports:
        pdf.flow_box(tx, y3, 46, 8, tname, tcol, WHITE, tcol)
    y3b = y3 + 11
    for tc in [51, 101, 151]:
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.25)
        pdf.line(tc, y3 + 8, tc, y3b)
    pdf.line(51, y3b, 151, y3b)
    pdf.line(cx, y3b, cx, y3b + 3)

    # Layer 4: GATEWAY (main box)
    y4 = y3b + 3
    gw_h = 52
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.8)
    pdf.rect(14, y4, 182, gw_h, style="DF")
    pdf.set_xy(14, y4 + 1)
    pdf.set_font("HGH", "", 9.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(182, 5.5, "上様の関所 — Shopify Agentic MCP Gateway  (AWS Lambda / Graviton)", align="C")

    # Middleware row
    mw_y = y4 + 8
    mw_data = [
        ("Auth\nOAuth 2.0 + UCP", (220, 230, 245)),
        ("Rate Limiter\nShopify Quota", (220, 230, 245)),
        ("Guardrail\nHallucination防止", (220, 230, 245)),
        ("Fee Collector\n0.5% 自動徴収", (255, 235, 215)),
    ]
    for i, (mname, mcol) in enumerate(mw_data):
        pdf.flow_box(18 + i * 44, mw_y, 42, 10, mname, mcol, DARK, GRAY)

    # UCP + AP2 processing
    proc_y = mw_y + 13
    pdf.flow_box(18, proc_y, 86, 7, "UCP Shopping Service + Capabilities", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)
    pdf.flow_box(108, proc_y, 86, 7, "AP2 Mandate Verifier (Intent/Cart/Pay)", GOOGLE_BLUE, WHITE, GOOGLE_BLUE)

    # Tools row
    tool_y = proc_y + 10
    tools = [
        ("scout_inventory", ACCENT),
        ("negotiate_terms", GREEN),
        ("execute_checkout", ORANGE),
        ("manage_cart", TEAL),
        ("track_order", GRAY),
    ]
    tw = 34.5
    for i, (tname, tcol) in enumerate(tools):
        pdf.flow_box(18 + i * 36, tool_y, tw, 10, tname, tcol, WHITE, tcol)

    # Arrow out
    y5 = y4 + gw_h
    pdf.flow_arrow_down(cx, y5, y5 + 5)

    # Layer 5: Shopify APIs
    y6 = y5 + 5
    pdf.flow_box(18, y6, 58, 9, "Storefront API (GraphQL)", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)
    pdf.flow_box(78, y6, 54, 9, "Admin API (GraphQL)", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)
    pdf.flow_box(134, y6, 58, 9, "Checkout API (REST)", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)

    # Arrow to store
    pdf.flow_arrow_down(cx, y6 + 9, y6 + 15)
    y7 = y6 + 15
    pdf.flow_box(cx - 28, y7, 56, 8, "Shopify Store (Merchant)", NAVY, WHITE)

    # Fee arrow to wallet
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.5)
    fee_y = mw_y + 5
    pdf.line(196, fee_y, 202, fee_y)
    pdf.line(202, fee_y, 202, fee_y + 16)
    pdf.flow_box(196, fee_y + 16, 14, 11, "上様\nWallet", GOLD, WHITE, GOLD)

    pdf.set_y(y7 + 14)

    # ================================================================
    # 6. TRANSACTION FLOW (16-step)
    # ================================================================
    pdf.section_title("6. Transaction Flow — 自律購買の全16ステップ")
    pdf.body_text(
        "AIエージェントが商品を発見し、自律的に購入を完了するまでの完全なフロー。"
        "UCP State Machine と AP2 Mandate Chain が連動する。"
    )
    pdf.ln(1)

    steps = [
        ("1",  "Agent",      "UCP Discovery",       "/.well-known/ucp でProfile取得、Capability確認", ACCENT),
        ("2",  "Agent→GW",   "scout_inventory",     "UCP Catalog検索: キーワード/カテゴリ/価格帯", ACCENT),
        ("3",  "GW→Shopify", "Storefront API",      "GraphQL productSearch + JSON-LD enrichment", SHOPIFY_GREEN),
        ("4",  "GW→Agent",   "Search Result",       "商品リスト返却 (Schema.org準拠, 在庫・価格付き)", ACCENT),
        ("5",  "Agent→GW",   "negotiate_terms",     "UCP Capability Negotiation: 割引・配送・条件交渉", GREEN),
        ("6",  "GW (内部)",  "Intersection Algo",   "Agent/Merchant双方のCapabilityの積集合を算出", GREEN),
        ("7",  "Agent→GW",   "manage_cart",         "UCP Line Items構築: 商品+数量+オプション", TEAL),
        ("8",  "GW",         "Checkout: incomplete", "Checkout Session作成、配送先・決済手段を収集中", ORANGE),
        ("9",  "AP2",        "Intent Mandate",      "User事前署名: 上限金額/カテゴリ/有効期限を委任", GOOGLE_BLUE),
        ("10", "GW",         "Mandate Verify",      "Intent Mandate の暗号署名を検証 (JWS/JWK)", GOOGLE_BLUE),
        ("11", "GW",         "Cart Mandate Sign",   "Merchant側署名: カート内容・価格を暗号固定", GOOGLE_BLUE),
        ("12", "User",       "Cart Approve",        "User device署名: カートMandate承認", GOOGLE_BLUE),
        ("13", "GW",         "Checkout: ready",     "ready_for_complete — 全Mandate + 全情報完備", GREEN),
        ("14", "Agent→GW",   "execute_checkout",    "Payment Mandate発行→決済網→Shopify決済完了", NAVY),
        ("15", "GW (内部)",  "fee_collector",       "取引額の0.5%を自動計算、上様Walletへ送金", GOLD),
        ("16", "GW→Agent",   "Order Confirmed",     "注文番号+追跡情報+レシートを返却、完了", ACCENT),
    ]

    for sno, route, tool, desc, color in steps:
        y = pdf.get_y()
        if y > 265:
            pdf.add_page()

        pdf.set_fill_color(*color)
        pdf.set_font("HGH", "", 7)
        pdf.set_text_color(*WHITE)
        r = 3.5
        cx_s = 15
        cy_s = pdf.get_y() + r
        pdf.ellipse(cx_s - r, cy_s - r, 2*r, 2*r, style="F")
        pdf.set_xy(cx_s - r, cy_s - 2)
        pdf.cell(2*r, 4, sno, align="C")

        pdf.set_xy(22, cy_s - 3.5)
        pdf.set_font("HGB", "", 7.5)
        pdf.set_text_color(*DARK)
        pdf.cell(22, 4.5, route)
        pdf.set_font("HGB", "", 7.5)
        pdf.set_text_color(*color)
        pdf.cell(30, 4.5, tool)
        pdf.set_font("HG", "", 7.5)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 4.5, desc, new_x="LMARGIN", new_y="NEXT")

        if sno != "16":
            pdf.set_draw_color(*GRAY)
            pdf.set_line_width(0.15)
            pdf.line(15, pdf.get_y(), 15, pdf.get_y() + 2)
            pdf.ln(2)
        else:
            pdf.ln(1)

    pdf.ln(2)

    # ================================================================
    # 7. UCP PROFILE EXAMPLE
    # ================================================================
    pdf.section_title("7. UCP Profile — /.well-known/ucp")
    pdf.body_text(
        "関所が公開するUCPプロファイル。AIエージェントはこれを自動取得し、"
        "対応Capability・Transport・決済手段を判断して接続してくる。"
    )
    pdf.ln(1)

    json_y = pdf.get_y()
    pdf.set_fill_color(40, 44, 52)
    pdf.set_draw_color(*NAVY)
    pdf.rect(15, json_y, 180, 62, style="DF")
    pdf.set_xy(18, json_y + 2)
    pdf.set_font("HG", "", 7)
    pdf.set_text_color(180, 220, 180)
    lines = [
        '{',
        '  "version": "2026-01-11",',
        '  "services": [{',
        '    "type": "dev.ucp.shopping",',
        '    "transports": [',
        '      { "type": "rest", "endpoint": "https://gw.example.com/ucp/v1" },',
        '      { "type": "mcp",  "endpoint": "https://gw.example.com/mcp" },',
        '      { "type": "a2a",  "endpoint": "https://gw.example.com/a2a" }',
        '    ],',
        '    "capabilities": [',
        '      { "name": "dev.ucp.shopping.checkout",     "version": "2026-01-11" },',
        '      { "name": "dev.ucp.shopping.catalog",      "version": "2026-01-11" },',
        '      { "name": "dev.ucp.shopping.order",        "version": "2026-01-11" },',
        '      { "name": "dev.ucp.shopping.fulfillment",  "version": "2026-01-11" }',
        '    ],',
        '    "payment_handlers": [',
        '      { "id": "shopify_pay", "type": "processor_tokenizer" },',
        '      { "id": "ap2_mandate", "type": "mandate" }',
        '    ],',
        '    "signing_keys": { "jwk": "https://gw.example.com/.well-known/jwks.json" }',
        '  }]',
        '}',
    ]
    for jl in lines:
        pdf.set_x(18)
        pdf.cell(0, 2.8, jl, new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(json_y + 66)

    # ================================================================
    # 8. 5 MCP TOOLS — 五本刀
    # ================================================================
    pdf.section_title("8. Five MCP Tools — 五本刀")
    pdf.body_text(
        "関所が公開するMCPツールは5本。各ツールがUCP CapabilityまたはAP2 Mandateに対応する。"
        "AIエージェントはこの5本を組み合わせて自律購買を遂行する。"
    )
    pdf.ln(1)

    pdf.render_table(
        ["Tool Name", "UCP/AP2 Mapping", "Input", "Output"],
        [
            ["scout_inventory", "Catalog Capability", "keyword, category, price range", "Product[] (JSON-LD)"],
            ["negotiate_terms", "Checkout Capability", "cart, agent_profile", "Negotiated terms, discounts"],
            ["execute_checkout", "Checkout + AP2", "cart, mandates (3 types)", "Order confirmation"],
            ["manage_cart", "Shopping Service", "add/remove/update items", "Cart state (line items, totals)"],
            ["track_order", "Order Capability", "order_id", "Status, tracking, fulfillment"],
        ],
        [32, 36, 56, 66],
    )

    pdf.section_title("fee_collector (内部処理)", 3)
    pdf.body_text(
        "fee_collectorは外部公開ツールではなく、execute_checkout完了時に自動発火する内部ミドルウェア。"
        "取引額 × 0.5% を算出し、Payment Handler経由で上様のWalletへ送金する。"
        "全トランザクションはDynamoDB Ledgerに記録され、ダッシュボードで確認可能。"
    )

    # ================================================================
    # 9. GUARDRAIL MIDDLEWARE
    # ================================================================
    pdf.section_title("9. Guardrail Middleware — AI幻覚防止")
    pdf.body_text(
        "AIエージェントは価格・在庫を「幻覚」する可能性がある。"
        "関所は全リクエストを検証し、不正な値を弾くガードレールを備える。"
    )
    pdf.ln(1)

    pdf.render_table(
        ["Check", "Trigger", "Action"],
        [
            ["Price Hallucination", "Agentの提示価格 != Shopify実価格", "UCP error返却、Checkout拒否"],
            ["Inventory Ghost", "在庫0の商品をカートに追加", "在庫状況をリアルタイム検証、拒否"],
            ["Mandate Forgery", "AP2署名の検証失敗", "取引即座にブロック、ログ記録"],
            ["Rate Abuse", "同一Agent短時間大量リクエスト", "429返却、一時BAN"],
            ["Amount Exceed", "Intent Mandate上限額超過", "UCP requires_escalation返却"],
        ],
        [35, 55, 100],
    )

    # ================================================================
    # 10. TECH STACK
    # ================================================================
    pdf.section_title("10. Tech Stack")
    pdf.render_table(
        ["Layer", "Technology", "Reason"],
        [
            ["Runtime", "Node.js 22 + TypeScript 5.x", "UCP reference impl + MCP SDK対応"],
            ["MCP SDK", "@modelcontextprotocol/sdk", "Anthropic公式 (UCP MCP binding)"],
            ["Shopify", "@shopify/shopify-api", "公式GraphQL SDK"],
            ["UCP", "ucp.dev reference (TS)", "Google+Shopify公式リファレンス"],
            ["AP2", "github: google-agentic-commerce/AP2", "Google公式 Mandate SDK"],
            ["Crypto", "jose", "JWS/JWK/JWT (AP2署名)"],
            ["Deploy", "AWS Lambda (Graviton3 ARM)", "Serverless + 低コスト"],
            ["API GW", "AWS API Gateway (HTTP API)", "REST/WebSocket endpoint"],
            ["CDN", "CloudFront", "/.well-known/ucp + edge cache"],
            ["DB", "DynamoDB", "Session/Mandate履歴/Fee Ledger"],
            ["Cache", "ElastiCache Redis", "Catalog cache + rate limit state"],
            ["Test", "Vitest + MSW", "Unit + API mock"],
            ["CI/CD", "GitHub Actions → SAM deploy", "自動テスト + Lambda deploy"],
        ],
        [22, 62, 106],
    )

    # ================================================================
    # 11. REVENUE MODEL
    # ================================================================
    pdf.section_title("11. Revenue Model — 二刀流")

    pdf.section_title("A. 関所手数料 (Transaction Fee)", 2)
    pdf.render_table(
        ["Monthly GMV", "Fee (0.5%)", "Annual"],
        [
            ["$10,000 (立ち上げ期)", "$50/mo", "$600"],
            ["$100,000", "$500/mo", "$6,000"],
            ["$1,000,000", "$5,000/mo", "$60,000"],
            ["$10,000,000 (成熟期)", "$50,000/mo", "$600,000"],
        ],
        [65, 60, 65],
    )

    pdf.section_title("B. Developer Kit (一括 + SaaS)", 2)
    pdf.render_table(
        ["Tier", "Price", "Contents"],
        [
            ["Starter (OSS)", "Free", "scout_inventory only (集客・認知用)"],
            ["Pro", "$49/mo", "全5ツール + UCP/AP2完全対応"],
            ["Enterprise", "$199/mo", "Custom capabilities + SLA + Priority Support"],
            ["Kit (一括購入)", "$99-299", "Full source + docs + deploy guide + 6mo updates"],
        ],
        [35, 28, 127],
    )

    # ================================================================
    # 12. COMPETITIVE ANALYSIS
    # ================================================================
    pdf.section_title("12. Competitive Landscape")
    pdf.render_table(
        ["Player", "Protocol", "Fee Model", "Open Source", "Status"],
        [
            ["Shopify Storefront MCP", "MCP only", "None", "Yes", "Released"],
            ["Shopify Dev MCP", "MCP only", "None", "Yes", "Released"],
            ["OpenAI ACP + Stripe", "ACP", "Stripe fee", "Yes", "Released"],
            ["Coinbase x402", "HTTP 402", "On-chain", "Yes", "Early"],
            ["上様 Gateway", "UCP+AP2+MCP", "0.5% tx fee", "Freemium", "Building"],
        ],
        [40, 32, 32, 28, 22],
    )

    pdf.body_text(
        "Shopify公式MCPはUCP非準拠。OpenAI ACPはStripe決済のみでAP2非対応。"
        "UCP + AP2 + MCP の三位一体Gatewayは市場に存在しない。"
        "先行者として関所を建て、トランザクション手数料で自動収益を得る。"
    )

    # ================================================================
    # 13. RISK ANALYSIS
    # ================================================================
    pdf.section_title("13. Risk & Mitigation")
    pdf.render_table(
        ["Risk", "Probability", "Impact", "Mitigation"],
        [
            ["UCP仕様変更 (発表38日)", "High", "High", "version date管理 + 抽象化層で吸収"],
            ["Shopify公式がUCP GWを出す", "Medium", "High", "先行して顧客基盤構築、Kit販売で回収"],
            ["AP2 Mandate仕様変動", "Medium", "Medium", "GitHub SDK追従 + abstraction layer"],
            ["0.5%手数料の法的リスク", "Low", "Medium", "Payment Handler登録 or SaaS課金に転換"],
            ["GMV立ち上がり遅延", "Medium", "Medium", "Kit販売で固定収益確保"],
        ],
        [42, 22, 20, 106],
    )

    # ================================================================
    # 14. IMPLEMENTATION ROADMAP
    # ================================================================
    pdf.section_title("14. Implementation Roadmap")
    pdf.render_table(
        ["Phase", "Duration", "Deliverables"],
        [
            ["Phase 1: Core", "Week 1-2", "MCP Server scaffold + UCP profile + scout_inventory"],
            ["Phase 2: Checkout", "Week 3-4", "Checkout state machine + manage_cart + negotiate_terms"],
            ["Phase 3: AP2", "Week 5-6", "Mandate verifier/signer + execute_checkout"],
            ["Phase 4: Fee", "Week 7", "fee_collector + DynamoDB Ledger + dashboard"],
            ["Phase 5: Deploy", "Week 8", "AWS Lambda deploy + CloudFront + monitoring"],
            ["Phase 6: Kit", "Week 9-10", "Developer Kit packaging + docs + examples"],
            ["Phase 7: Launch", "Week 11-12", "Beta launch + first merchants + Kit sales"],
        ],
        [30, 28, 132],
    )

    # ================================================================
    # FOOTER
    # ================================================================
    pdf.ln(4)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("HG", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "本設計書は Asset Shield 老中 (Claude Opus 4.6) が UCP/AP2公式仕様を精査の上作成",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Sources: shopify.engineering/ucp | ucp.dev | github.com/google-agentic-commerce/AP2",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("HGH", "", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 6, "上様の号令をもって実装を開始する", align="C")

    pdf.output(OUTPUT)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build()
