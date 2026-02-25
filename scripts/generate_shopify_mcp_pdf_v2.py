#!/usr/bin/env python3
"""Shopify Agentic MCP Server 設計書 V2 (UCP/AP2準拠) — 全面改訂版"""

from fpdf import FPDF

FONT_PATH_W3 = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_PATH_W6 = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_PATH_W8 = "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc"
OUTPUT = "/Users/MBP/Desktop/Shopify_MCP_Server_Design_V2.pdf"

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
        self.cell(0, 6, "Shopify Agentic MCP Server V2 — UCP/AP2 Compliant Architecture", align="L")
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
        self.ln(30)
        self.set_font("HGH", "", 26)
        self.set_text_color(*NAVY)
        self.cell(0, 14, "Shopify Agentic MCP Server", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("HGB", "", 16)
        self.cell(0, 10, "Architecture Design Document V2", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("HGB", "", 12)
        self.set_text_color(*RED)
        self.cell(0, 8, "UCP / AP2 / MCP Tri-Protocol Compliant", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)
        self.set_draw_color(*NAVY)
        self.set_line_width(0.5)
        self.line(55, self.get_y(), 155, self.get_y())
        self.ln(10)
        self.set_font("HG", "", 11)
        self.set_text_color(*DARK)
        for line in [
            "Date: 2026-02-18 (Revised)",
            "Author: 老中 (Claude Opus 4.6)",
            "To: 上様",
            "Status: Design Phase — Awaiting Go Order",
        ]:
            self.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

        # Protocol badges
        x0 = 25
        w = 160
        badge_y = self.get_y()
        self.set_fill_color(*LIGHT_BG)
        self.set_draw_color(*NAVY)
        self.rect(x0, badge_y, w, 18, style="DF")
        badges = [
            ("UCP", SHOPIFY_GREEN, "Shopify + Google"),
            ("AP2", GOOGLE_BLUE, "Google (60+ Partners)"),
            ("MCP", PURPLE, "Anthropic"),
        ]
        bx = x0 + 8
        for name, color, org in badges:
            self.set_xy(bx, badge_y + 2)
            self.set_fill_color(*color)
            self.set_font("HGH", "", 10)
            self.set_text_color(*WHITE)
            self.cell(20, 6, name, fill=True, align="C")
            self.set_fill_color(*LIGHT_BG)
            self.set_font("HG", "", 7.5)
            self.set_text_color(*GRAY)
            self.set_xy(bx, badge_y + 9)
            self.cell(48, 5, org, align="C")
            bx += 52

        self.set_y(badge_y + 22)

        # Summary box
        self.set_fill_color(*LIGHT_BG)
        self.set_draw_color(*NAVY)
        box_y = self.get_y()
        self.rect(x0, box_y, w, 48, style="DF")
        self.set_xy(x0, box_y + 3)
        self.set_font("HGB", "", 11)
        self.set_text_color(*NAVY)
        self.cell(w, 7, "概要", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_x(x0 + 8)
        self.set_font("HG", "", 9.5)
        self.set_text_color(*DARK)
        self.set_x(x0 + 8)
        self.multi_cell(w - 16, 5.5,
            "熨斗アプリを母体とし、2026年1月発表のUCP (Universal Commerce Protocol) および\n"
            "AP2 (Agent Payment Protocol) に完全準拠したMCPサーバーを構築する。\n"
            "世界中のAIエージェント (Gemini, ChatGPT, Copilot, Claude) が上様のサーバーへ\n"
            "「買い出し」に来る関所として機能し、通過するトランザクションごとに手数料を徴収。\n"
            "日本式ギフト文化 (熨斗) のUCP Extensionが世界唯一の差別化要素。\n"
            "Developer Kitとして海外開発者に販売するパッケージも同時に構築する。"
        )

    def section_title(self, text, level=1):
        self.ln(4)
        if level == 1:
            if self.get_y() > 40:
                self.add_page()
            self.set_font("HGH", "", 15)
            self.set_text_color(*NAVY)
            self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*NAVY)
            self.set_line_width(0.6)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("HGB", "", 12)
            self.set_text_color(*NAVY)
            self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font("HGB", "", 10)
            self.set_text_color(*DARK)
            self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
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
        x0 = self.l_margin
        self.set_x(x0)
        self.cell(6, 5.5, "・")
        self.multi_cell(180, 5.5, text)

    def tree_line(self, depth, connector, name, description="", is_dir=False):
        indent = 12 + depth * 9
        self.set_x(indent)
        self.set_font("HG", "", 8)
        self.set_text_color(*GRAY)
        self.cell(10, 5, connector)
        if is_dir:
            self.set_font("HGB", "", 8)
            self.set_text_color(*NAVY)
        else:
            self.set_font("HG", "", 8)
            self.set_text_color(*DARK)
        self.cell(46, 5, name)
        if description:
            self.set_font("HG", "", 7)
            self.set_text_color(*GRAY)
            self.cell(0, 5, description, new_x="LMARGIN", new_y="NEXT")
        else:
            self.ln(5)

    def flow_box(self, x, y, w, h, text, bg_color=LIGHT_BG, text_color=NAVY, border_color=NAVY):
        self.set_fill_color(*bg_color)
        self.set_draw_color(*border_color)
        self.set_line_width(0.4)
        self.rect(x, y, w, h, style="DF")
        self.set_xy(x, y + 1)
        self.set_font("HGB", "", 7.5)
        self.set_text_color(*text_color)
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

    def protocol_badge(self, x, y, name, color):
        self.set_fill_color(*color)
        self.set_font("HGH", "", 7)
        self.set_text_color(*WHITE)
        self.set_xy(x, y)
        self.cell(16, 5, name, fill=True, align="C")


def build():
    pdf = DesignPDF()

    # ================================================================
    # COVER PAGE
    # ================================================================
    pdf.cover_page()

    # ================================================================
    # 1. TRI-PROTOCOL OVERVIEW
    # ================================================================
    pdf.section_title("1. Tri-Protocol Architecture — UCP / AP2 / MCP")
    pdf.body_text(
        "2026年1月、Google + Shopifyが共同でUCPを、GoogleがAP2を発表。"
        "Walmart, Target, Etsy, Wayfair等80社以上が参画。"
        "これにAnthropicのMCPを組み合わせた三位一体が、上様のシステムの骨格となる。"
    )
    pdf.ln(1)

    pdf.render_table(
        ["Protocol", "Developer", "Role", "Launched"],
        [
            ["UCP", "Google + Shopify", "AI専用の共通商取引言語 (Discovery/Negotiation/Checkout)", "2026-01"],
            ["AP2", "Google (60+ Partners)", "AIの財布: 暗号署名付きMandate (Intent/Cart/Payment)", "2025-09"],
            ["MCP", "Anthropic", "AI↔Server接続規格 (stdio/SSE). UCP公式バインディング", "2024"],
        ],
        [20, 45, 90, 35],
    )

    pdf.section_title("UCP 3層アーキテクチャ", 2)
    pdf.render_table(
        ["Layer", "Role", "Example"],
        [
            ["Shopping Service", "コア取引プリミティブ定義", "checkout session, line items, totals, status"],
            ["Capabilities", "機能領域 (独立バージョニング)", "Checkout, Orders, Catalog"],
            ["Extensions", "ドメイン特化スキーマ (合成)", "dev.ucp.shopping.fulfillment, .noshi (独自)"],
        ],
        [35, 55, 100],
    )

    pdf.section_title("AP2 Mandate Chain (暗号署名チェーン)", 2)
    pdf.render_table(
        ["Mandate Type", "Who Signs", "What It Proves"],
        [
            ["Intent Mandate", "User (事前)", "AIへの委任: 上限金額・条件・期間の制約"],
            ["Cart Mandate", "Merchant → User", "カート内容・価格・配送を改竄不能に固定"],
            ["Payment Mandate", "User → Payment Network", "AI発の取引であると銀行/決済網に証明"],
        ],
        [35, 40, 115],
    )

    pdf.section_title("UCP Checkout State Machine", 2)
    pdf.body_text("UCPはチェックアウトを3状態の有限状態機械で管理する:")
    y = pdf.get_y()
    states = [
        (20, "incomplete", "情報不足\nAPI解決を試行", ORANGE),
        (80, "requires_escalation", "人間の入力必要\ncontinue_url発行", RED),
        (140, "ready_for_complete", "全情報揃い\nAgentが決済完了", GREEN),
    ]
    for sx, label, desc, color in states:
        pdf.flow_box(sx, y, 52, 16, f"{label}\n{desc}", color, WHITE, color)
    # arrows
    pdf.flow_arrow_right(72, 80, y + 8)
    pdf.flow_arrow_right(132, 140, y + 8)
    pdf.set_y(y + 22)
    pdf.ln(2)

    pdf.section_title("UCP Transport Bindings", 2)
    pdf.render_table(
        ["Transport", "Format", "Use Case"],
        [
            ["REST", "OpenAPI 3.x", "Primary HTTP binding (上様のGateway)"],
            ["MCP", "OpenRPC", "LLMツール公開 (Claude Desktop等)"],
            ["A2A", "Agent Card", "Agent-to-Agent通信 (Gemini↔Copilot)"],
            ["Embedded", "OpenRPC / JSON-RPC 2.0", "ブラウザ埋込チェックアウト"],
        ],
        [30, 45, 115],
    )

    # ================================================================
    # 2. SYSTEM CONCEPT — 三要素
    # ================================================================
    pdf.section_title("2. System Concept — 関所モデル")
    pdf.body_text(
        "上様のシステムは「関所」(Gateway) である。"
        "世界中のAIエージェントがShopify店舗で買い物をする際、"
        "上様の関所を通らねばならぬ。通過する銭から通行料を徴収する。"
    )
    pdf.ln(1)

    pdf.render_table(
        ["要素", "役割", "所在"],
        [
            ["鍛冶場 (Dev)", "Claude Codeを副官とし、プログラムを錬成・テスト", "MBP (M4 Pro 48GB)"],
            ["関所 (Gateway)", "AIエージェントの注文を受け、UCP/AP2でShopifyへ繋ぐ", "AWS Lambda (ARM/Graviton)"],
            ["金庫 (Ledger)", "決済ごとの手数料を自動回収・蓄積", "上様のウォレット"],
        ],
        [30, 90, 70],
    )

    pdf.section_title("収益の血流 (Transaction Flow)", 2)
    pdf.bold_text("探索 → 交渉 → 決済 → 徴収")
    pdf.bullet("探索: 海外のAI (Gemini/ChatGPT/Copilot) が商品検索 → 上様のMCPサーバーへ問合せ")
    pdf.bullet("交渉: 上様のサーバーがUCP Capability Negotiationで在庫・価格・条件を自動交渉")
    pdf.bullet("決済: AP2 Mandateチェーンで暗号署名付き決済を実行")
    pdf.bullet("徴収: Payment Handler経由で通行料 (手数料) を上様の懐へ自動回収")

    # ================================================================
    # 3. PROJECT TREE (UCP Native)
    # ================================================================
    pdf.section_title("3. Project Tree — UCP/AP2 Native Structure")
    pdf.body_text(
        "前版からの主要変更: UCP Shopping Service/Capabilities/Extensions の3層を反映。"
        "AP2 Mandate検証層を新設。/.well-known/ucpによるProfile公開に対応。"
    )
    pdf.ln(1)

    tree_y = pdf.get_y()
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*NAVY)
    pdf.rect(10, tree_y, 190, 188, style="DF")
    pdf.ln(2)

    tree = [
        (0, "",   "shopify-agentic-mcp/",       "", True),
        (0, "├─", "package.json",               "dependencies: @modelcontextprotocol/sdk, @shopify/shopify-api", False),
        (0, "├─", "tsconfig.json",              "TypeScript strict mode", False),
        (0, "├─", "serverless.yml",             "AWS Lambda (ARM/Graviton) deploy config", False),
        (0, "├─", ".env.example",               "SHOPIFY_API_KEY, AP2_SIGNING_KEY, etc.", False),
        (0, "├─", "README.md",                  "Developer Kit documentation (EN)", False),
        (0, "├─", "src/",                        "",  True),
        (1, "├─", "index.ts",                   "Lambda handler + MCP Server entrypoint", False),
        (1, "├─", "server.ts",                  "MCP Server init, tool/resource registration", False),
        (1, "├─", "ucp/",                        "UCP Protocol Layer",  True),
        (2, "├─", "profile.ts",                 "/.well-known/ucp profile publisher", False),
        (2, "├─", "negotiate.ts",               "Capability intersection algorithm", False),
        (2, "├─", "checkout-session.ts",         "Checkout state machine (3-state)", False),
        (2, "├─", "shopping-service.ts",         "Core primitives (line items, totals)", False),
        (2, "└─", "schemas/",                    "JSON Schema definitions",  True),
        (3, "├─", "checkout.schema.json",        "dev.ucp.shopping.checkout", False),
        (3, "├─", "catalog.schema.json",         "dev.ucp.shopping.catalog", False),
        (3, "└─", "order.schema.json",           "dev.ucp.shopping.order", False),
        (1, "├─", "ap2/",                         "AP2 Mandate Layer",  True),
        (2, "├─", "mandate-verifier.ts",         "Intent/Cart/Payment Mandate cryptographic verify", False),
        (2, "├─", "mandate-signer.ts",           "Cart Mandate signing (merchant side)", False),
        (2, "└─", "types.ts",                    "AP2 Mandate type definitions", False),
        (1, "├─", "tools/",                       "MCP Tools (Agent-facing)",  True),
        (2, "├─", "scout-inventory.ts",          "UCP Catalog search (semantic + structured)", False),
        (2, "├─", "negotiate-terms.ts",          "UCP Capability Negotiation + discount", False),
        (2, "├─", "execute-checkout.ts",         "AP2 Mandate chain → checkout complete", False),
        (2, "├─", "configure-noshi.ts",          "Noshi Extension (差別化コア)", False),
        (2, "├─", "manage-cart.ts",              "Cart CRUD with UCP line items", False),
        (2, "├─", "track-order.ts",              "Order tracking + webhooks", False),
        (2, "└─", "check-inventory.ts",          "Real-time inventory check", False),
        (1, "├─", "extensions/",                  "UCP Extensions (domain-specific)",  True),
        (2, "├─", "noshi/",                       "dev.ucp.shopping.noshi (世界唯一)",  True),
        (3, "├─", "engine.ts",                   "Occasion→水引→表書き判定ロジック", False),
        (3, "├─", "templates.ts",                "熨斗テンプレート DB", False),
        (3, "├─", "i18n.ts",                     "EN/JP/ZH multi-language guide", False),
        (3, "└─", "schema.json",                 "dev.ucp.shopping.noshi extension schema", False),
        (2, "└─", "fulfillment/",                 "dev.ucp.shopping.fulfillment",  True),
        (3, "└─", "handler.ts",                  "Shipping/pickup/delivery window", False),
        (1, "├─", "middleware/",                   "",  True),
        (2, "├─", "auth.ts",                     "Shopify OAuth 2.0 + UCP Profile Auth", False),
        (2, "├─", "rate-limiter.ts",             "Shopify API quota + burst protection", False),
        (2, "├─", "guardrail.ts",                "AI hallucination guard (price/inventory)", False),
        (2, "└─", "fee-collector.ts",            "Transaction fee (0.5%) auto-deduction", False),
        (1, "└─", "utils/",                        "",  True),
        (2, "├─", "logger.ts",                   "Structured logging (CloudWatch)", False),
        (2, "├─", "crypto.ts",                   "JWS/JWK signing utilities", False),
        (2, "└─", "errors.ts",                   "UCP error response builder", False),
        (0, "├─", "tests/",                       "",  True),
        (1, "├─", "ucp/",                         "UCP layer tests",  True),
        (1, "├─", "ap2/",                         "AP2 mandate tests",  True),
        (1, "├─", "tools/",                       "MCP tool tests",  True),
        (1, "├─", "extensions/",                   "Extension tests (noshi)",  True),
        (1, "└─", "e2e/",                         "End-to-end (MSW mock)",  True),
        (0, "├─", "docs/",                        "",  True),
        (1, "├─", "ucp-integration.md",           "UCP準拠ガイド", False),
        (1, "├─", "ap2-mandates.md",              "AP2 Mandate実装ガイド", False),
        (1, "├─", "noshi-extension.md",           "熨斗Extension仕様 (EN/JP)", False),
        (1, "├─", "deployment.md",                "AWS Lambda deploy手順", False),
        (1, "└─", "developer-kit.md",             "Kit購入者向けクイックスタート", False),
        (0, "└─", "examples/",                     "",  True),
        (1, "├─", "agent-gift-purchase.ts",        "AIギフト購入 E2Eサンプル", False),
        (1, "├─", "mandate-flow.ts",               "AP2 Mandate署名サンプル", False),
        (1, "└─", "claude-desktop-config.json",    "Claude Desktop MCP設定例", False),
    ]

    for depth, connector, name, desc, is_dir in tree:
        pdf.tree_line(depth, connector, name, desc, is_dir)

    pdf.ln(4)

    # ================================================================
    # 4. ARCHITECTURE FLOWCHART (UCP Native)
    # ================================================================
    pdf.section_title("4. Architecture Flowchart — UCP/AP2/MCP Tri-Protocol")
    pdf.body_text(
        "世界中のAIエージェントから上様の関所 (Gateway) を経由し、"
        "Shopify店舗APIまでの完全なデータフロー。"
        "UCP Profile Discovery → Capability Negotiation → AP2 Mandate Chain → Checkout Complete。"
    )
    pdf.ln(1)

    cx = 105
    y = pdf.get_y()

    # Layer 1: AI Agents (multiple)
    agents = [
        (25, "Gemini\n(Google)", GOOGLE_BLUE),
        (70, "ChatGPT\n(OpenAI)", GREEN),
        (115, "Copilot\n(Microsoft)", ACCENT),
        (155, "Claude\n(Anthropic)", PURPLE),
    ]
    for ax, aname, acolor in agents:
        pdf.flow_box(ax, y, 38, 12, aname, acolor, WHITE, acolor)

    # Converge arrows
    y_conv = y + 15
    for ax in [44, 89, 134, 174]:
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.3)
        pdf.line(ax, y + 12, ax, y_conv)
    pdf.line(44, y_conv, 174, y_conv)
    pdf.line(cx, y_conv, cx, y_conv + 4)

    # Layer 2: UCP Profile Discovery
    y2 = y_conv + 4
    pdf.flow_box(cx - 35, y2, 70, 9, "UCP Profile Discovery (/.well-known/ucp)", LIGHT_NAVY, NAVY)
    pdf.flow_arrow_down(cx, y2 + 9, y2 + 15)

    # Layer 3: MCP / REST / A2A Transport
    y3 = y2 + 15
    transports = [
        (30, "REST\nOpenAPI 3.x", ACCENT),
        (80, "MCP\nOpenRPC", PURPLE),
        (130, "A2A\nAgent Card", TEAL),
    ]
    for tx, tname, tcol in transports:
        pdf.flow_box(tx, y3, 44, 10, tname, tcol, WHITE, tcol)
    # merge
    y3b = y3 + 13
    for tx_c in [52, 102, 152]:
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.3)
        pdf.line(tx_c, y3 + 10, tx_c, y3b)
    pdf.line(52, y3b, 152, y3b)
    pdf.line(cx, y3b, cx, y3b + 3)

    # Layer 4: THE GATEWAY (Big box)
    y4 = y3b + 3
    gw_h = 55
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.8)
    pdf.rect(15, y4, 180, gw_h, style="DF")
    pdf.set_xy(15, y4 + 1)
    pdf.set_font("HGH", "", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(180, 6, "上様の関所 — Shopify Agentic MCP Server (AWS Lambda)", align="C")

    # Middleware row
    mw_y = y4 + 9
    mw_items = [
        ("Auth\nOAuth 2.0 + UCP", (220, 230, 245)),
        ("Rate Limiter\nAPI Quota Guard", (220, 230, 245)),
        ("Guardrail\nHallucination Guard", (220, 230, 245)),
        ("Fee Collector\n0.5% Auto-Deduct", (255, 235, 220)),
    ]
    for i, (mname, mcol) in enumerate(mw_items):
        mx = 19 + i * 44
        pdf.flow_box(mx, mw_y, 40, 11, mname, mcol, DARK, GRAY)

    # UCP layer
    ucp_y = mw_y + 14
    pdf.set_xy(19, ucp_y)
    pdf.set_font("HGB", "", 7)
    pdf.set_text_color(*SHOPIFY_GREEN)
    pdf.cell(170, 4, "UCP Shopping Service + Capabilities + Extensions", align="C")

    # Tools row
    tool_y = ucp_y + 6
    tool_items = [
        ("scout\ninventory", ACCENT),
        ("negotiate\nterms", GREEN),
        ("execute\ncheckout", ORANGE),
        ("configure\nnoshi", RED),
    ]
    for i, (tname, tcol) in enumerate(tool_items):
        tx = 19 + i * 44
        pdf.flow_box(tx, tool_y, 40, 12, tname, tcol, WHITE, tcol)

    tool_y2 = tool_y + 14
    tool_items2 = [
        ("manage\ncart", GREEN),
        ("check\ninventory", TEAL),
        ("track\norder", GRAY),
    ]
    for i, (tname, tcol) in enumerate(tool_items2):
        tx = 40 + i * 44
        pdf.flow_box(tx, tool_y2, 40, 10, tname, tcol, WHITE, tcol)

    # AP2 mandate bar
    ap2_y = tool_y2 + 12
    pdf.flow_box(19, ap2_y, 172, 7, "AP2 Mandate Verifier — Intent / Cart / Payment cryptographic chain", GOOGLE_BLUE, WHITE, GOOGLE_BLUE)

    # Arrow out of gateway
    y5 = y4 + gw_h
    pdf.flow_arrow_down(cx, y5, y5 + 6)

    # Layer 5: Shopify APIs
    y6 = y5 + 6
    pdf.flow_box(20, y6, 55, 10, "Storefront API (GraphQL)", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)
    pdf.flow_box(78, y6, 55, 10, "Admin API (GraphQL)", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)
    pdf.flow_box(136, y6, 55, 10, "Checkout API (REST)", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)

    # Arrow to store
    pdf.flow_arrow_down(cx, y6 + 10, y6 + 16)
    y7 = y6 + 16
    pdf.flow_box(cx - 30, y7, 60, 9, "Shopify Store (Merchant)", NAVY, WHITE)

    # Wallet arrow (fee)
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.5)
    wallet_y = y4 + 14
    pdf.line(195, wallet_y, 202, wallet_y)
    pdf.line(202, wallet_y, 202, wallet_y + 20)
    pdf.flow_box(196, wallet_y + 20, 14, 12, "上様\nWallet", GOLD, WHITE, GOLD)

    pdf.set_y(y7 + 14)

    # ================================================================
    # 5. UCP CHECKOUT + AP2 MANDATE FLOW
    # ================================================================
    pdf.section_title("5. UCP Checkout + AP2 Mandate — 決済フローチャート")
    pdf.body_text(
        "AIエージェントがギフト購入を完了するまでの完全なフロー。"
        "UCP Checkout State Machine と AP2 Mandate Chain が連動する。"
    )
    pdf.ln(1)

    steps = [
        ("1",  "Agent",      "UCP Discovery",        "/.well-known/ucp でProfile取得, Capability確認", ACCENT),
        ("2",  "Agent→GW",   "scout_inventory",      "UCP Catalog検索: 「母の日 和菓子 3000円」", ACCENT),
        ("3",  "GW→Shopify", "Storefront API",       "GraphQL productSearch + semantic matching", SHOPIFY_GREEN),
        ("4",  "GW→Agent",   "UCP Response",         "商品リスト (JSON-LD, Schema.org準拠)", ACCENT),
        ("5",  "Agent→GW",   "negotiate_terms",      "UCP Capability Negotiation: 割引・配送・条件", GREEN),
        ("6",  "Agent→GW",   "configure_noshi",      "dev.ucp.shopping.noshi Extension: 蝶結び/御祝", RED),
        ("7",  "GW (内部)",  "Noshi Engine",         "文脈判定: 母の日→蝶結び, テンプレ選択, i18n", RED),
        ("8",  "Agent→GW",   "manage_cart",          "UCP Line Items構築 + 熨斗オプション追加", GREEN),
        ("9",  "GW",         "Status: incomplete",   "Checkout Session作成, 情報収集中", ORANGE),
        ("10", "AP2",        "Intent Mandate",       "User署名: 上限3000円, 和菓子カテゴリ, 24h有効", GOOGLE_BLUE),
        ("11", "GW",         "Cart Mandate Sign",    "Merchant側署名: カート内容・価格を暗号固定", GOOGLE_BLUE),
        ("12", "User",       "Cart Mandate Approve", "User署名: カート承認 (デバイス暗号署名)", GOOGLE_BLUE),
        ("13", "GW",         "Status: ready",        "ready_for_complete — 全Mandate揃う", GREEN),
        ("14", "Agent→GW",   "execute_checkout",     "Payment Mandate発行→決済網→支払完了", NAVY),
        ("15", "GW (内部)",  "Fee Collector",        "0.5% 自動徴収 → 上様Wallet", GOLD),
        ("16", "GW→Agent",   "Order Confirmation",   "注文番号 + 追跡情報 + 熨斗プレビュー返却", ACCENT),
    ]

    for step_no, route, tool, desc, color in steps:
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
        pdf.cell(2*r, 4, step_no, align="C")

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

        if step_no != "16":
            pdf.set_draw_color(*GRAY)
            pdf.set_line_width(0.15)
            pdf.line(15, pdf.get_y(), 15, pdf.get_y() + 2)
            pdf.ln(2)
        else:
            pdf.ln(1)

    pdf.ln(2)

    # ================================================================
    # 6. NOSHI ENGINE FLOW (UCP Extension)
    # ================================================================
    pdf.section_title("6. Noshi Engine — dev.ucp.shopping.noshi Extension")
    pdf.body_text(
        "UCP Extension として定義する世界唯一の熨斗判定エンジン。"
        "JSON Schema allOfで Checkout Capabilityに合成され、"
        "熨斗オプションがチェックアウトセッションの一部として流通する。"
    )
    pdf.ln(1)

    y = pdf.get_y()
    bw2 = 52

    # Start
    pdf.flow_box(cx - bw2/2, y, bw2, 11, "Agent: configure_noshi\n(occasion, recipient, sender)", ACCENT, WHITE, ACCENT)
    pdf.flow_arrow_down(cx, y + 11, y + 17)

    # UCP Extension validation
    vy = y + 17
    pdf.flow_box(cx - bw2/2, vy, bw2, 9, "UCP Extension Schema Validate\ndev.ucp.shopping.noshi", SHOPIFY_GREEN, WHITE, SHOPIFY_GREEN)
    pdf.flow_arrow_down(cx, vy + 9, vy + 14)

    # Decision
    dy = vy + 14
    dw = 56
    dh = 12
    pdf.set_fill_color(*LIGHT_NAVY)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.4)
    pdf.rect(cx - dw/2, dy, dw, dh, style="DF")
    pdf.set_xy(cx - dw/2, dy + 2)
    pdf.set_font("HGB", "", 8)
    pdf.set_text_color(*NAVY)
    pdf.cell(dw, 4, "occasion 分岐", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(cx - dw/2)
    pdf.set_font("HG", "", 6.5)
    pdf.cell(dw, 4, "(celebration / wedding / condolence / thanks / other)", align="C")

    branch_y = dy + dh + 8
    branches = [
        (30,  "Celebration",  "蝶結び (bow-tie)\n出産/入学/母の日/昇進\n何度あっても良い祝事", GREEN),
        (80,  "Wedding",      "結切り (fixed-knot)\n結婚祝/結婚内祝\n一度きりの祝事", ORANGE),
        (130, "Condolence",   "結切り 黒白\n(black-white)\n弔事/香典返し", GRAY),
        (175, "Thanks/Other", "花結び or 無地\n御礼/御中元/御歳暮\nカスタム対応", TEAL),
    ]

    for bx, label, detail, color in branches:
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.3)
        if bx == 80 or bx == 130:
            pdf.line(cx, dy + dh, bx, dy + dh)
            pdf.line(bx, dy + dh, bx, branch_y)
        else:
            pdf.line(cx, dy + dh/2, bx, dy + dh/2)
            pdf.line(bx, dy + dh/2, bx, branch_y)
        pdf.flow_box(bx - 21, branch_y, 42, 20, f"{label}\n{detail}", color, WHITE, color)

    merge_y = branch_y + 24
    for bx in [30, 80, 130, 175]:
        pdf.set_draw_color(*NAVY)
        pdf.set_line_width(0.3)
        pdf.line(bx, branch_y + 20, bx, merge_y)
    pdf.line(30, merge_y, 175, merge_y)
    pdf.line(cx, merge_y, cx, merge_y + 4)

    final_y = merge_y + 4
    pdf.flow_box(cx - 32, final_y, 64, 14, "表書き + 名入れ生成\ni18n (EN/JP/ZH) 対応\nUCP Checkout Sessionに合成", NAVY, WHITE)

    pdf.set_y(final_y + 20)

    # ================================================================
    # 7. UCP PROFILE (/.well-known/ucp)
    # ================================================================
    pdf.section_title("7. UCP Profile — /.well-known/ucp")
    pdf.body_text(
        "UCPではサーバーが /.well-known/ucp にプロファイルを公開する。"
        "AIエージェントはこのプロファイルを見て、対応Capability・Transport・決済手段を判断する。"
    )
    pdf.ln(1)

    # Simulated JSON
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*NAVY)
    json_y = pdf.get_y()
    pdf.rect(15, json_y, 180, 72, style="DF")
    pdf.set_xy(17, json_y + 2)
    pdf.set_font("HG", "", 7.5)
    pdf.set_text_color(*DARK)
    json_lines = [
        '{',
        '  "version": "2026-01-11",',
        '  "services": [{',
        '    "type": "dev.ucp.shopping",',
        '    "transports": [',
        '      { "type": "rest",     "endpoint": "https://api.example.com/ucp/v1" },',
        '      { "type": "mcp",      "endpoint": "https://api.example.com/mcp" },',
        '      { "type": "a2a",      "endpoint": "https://api.example.com/a2a" }',
        '    ],',
        '    "capabilities": [',
        '      { "name": "dev.ucp.shopping.checkout",   "version": "2026-01-11" },',
        '      { "name": "dev.ucp.shopping.catalog",    "version": "2026-01-11" },',
        '      { "name": "dev.ucp.shopping.order",      "version": "2026-01-11" },',
        '      { "name": "dev.ucp.shopping.noshi",      "version": "2026-02-18" }',
        '    ],',
        '    "payment_handlers": [',
        '      { "id": "shopify_payments", "type": "processor_tokenizer" },',
        '      { "id": "ap2_mandate",      "type": "mandate" }',
        '    ],',
        '    "signing_keys": { "jwk": "https://api.example.com/.well-known/jwks.json" }',
        '  }]',
        '}',
    ]
    for jl in json_lines:
        pdf.set_x(17)
        pdf.cell(0, 3.2, jl, new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(json_y + 76)

    # ================================================================
    # 8. TECH STACK
    # ================================================================
    pdf.section_title("8. Tech Stack — 2026 Edition")
    pdf.render_table(
        ["Layer", "Technology", "Reason"],
        [
            ["Runtime", "Node.js 22 + TypeScript 5.x", "MCP SDK + UCP Reference Impl対応"],
            ["MCP SDK", "@modelcontextprotocol/sdk", "Anthropic公式 (UCP MCP binding対応)"],
            ["Shopify", "@shopify/shopify-api", "公式GraphQL SDK (Storefront + Admin)"],
            ["UCP", "ucp.dev reference impl (TS)", "Google+Shopify公式リファレンス"],
            ["AP2", "AP2 GitHub SDK", "Google公式 (Mandate sign/verify)"],
            ["Crypto", "jose (JWS/JWK/JWT)", "AP2 Mandate暗号署名"],
            ["Deploy", "AWS Lambda (ARM Graviton3)", "低コスト + 上様指定のServerless"],
            ["CDN", "CloudFront", "グローバルEdge + /.well-known/ucp配信"],
            ["DB", "DynamoDB", "セッション・Mandate履歴・Fee Ledger"],
            ["Cache", "ElastiCache (Redis)", "UCP Profile + Catalog edge cache"],
            ["Test", "Vitest + MSW", "高速テスト + API Mock"],
            ["CI/CD", "GitHub Actions", "自動テスト + SAM/CDK deploy"],
        ],
        [25, 60, 105],
    )

    # ================================================================
    # 9. PRICING & REVENUE
    # ================================================================
    pdf.section_title("9. Revenue Model — 二刀流")

    pdf.section_title("A. トランザクション手数料 (関所モデル)", 2)
    pdf.render_table(
        ["Item", "Rate", "Note"],
        [
            ["AI Agent経由取引", "0.5%", "UCP checkout完了時に自動徴収"],
            ["想定月間GMV $100K", "$500/月", "年間 $6,000"],
            ["想定月間GMV $1M", "$5,000/月", "年間 $60,000"],
            ["想定月間GMV $10M", "$50,000/月", "年間 $600,000"],
        ],
        [55, 35, 100],
    )

    pdf.section_title("B. Developer Kit 販売", 2)
    pdf.render_table(
        ["Tier", "Price", "Contents"],
        [
            ["Starter (OSS)", "Free", "MCP Server Core (scout_inventory only)"],
            ["Pro", "$49/mo", "全7ツール + Noshi Extension + UCP/AP2完全対応"],
            ["Enterprise", "$199/mo", "Custom Extensions + Priority Support + SLA"],
            ["Kit (一括)", "$99-299", "Complete source + docs + deploy guide"],
        ],
        [35, 30, 125],
    )

    # ================================================================
    # 10. COMPETITIVE LANDSCAPE
    # ================================================================
    pdf.section_title("10. Competitive Landscape — 2026年2月時点")

    pdf.render_table(
        ["Competitor", "Protocol", "Noshi", "AP2", "Developer Kit"],
        [
            ["Shopify公式 Storefront MCP", "MCP only", "No", "No", "No"],
            ["Shopify公式 Dev MCP", "MCP only", "No", "No", "No"],
            ["OpenAI ACP (Stripe)", "ACP", "No", "No", "No"],
            ["上様のサーバー", "UCP+AP2+MCP", "Yes (唯一)", "Yes", "Yes"],
        ],
        [50, 30, 22, 22, 30],
    )

    pdf.body_text(
        "Shopify公式はStorefront MCP / Dev MCPを提供しているが、UCP準拠ではない。"
        "OpenAIはStripeとACP (Agentic Commerce Protocol) を発表したが、UCP/AP2とは別系統。"
        "UCP+AP2+MCPの三位一体かつNoshi Extensionを持つサーバーは市場に存在しない。"
    )

    # ================================================================
    # 11. RISK & MITIGATION
    # ================================================================
    pdf.section_title("11. Risk Analysis")
    pdf.render_table(
        ["Risk", "Impact", "Mitigation"],
        [
            ["UCP仕様変更 (発表1ヶ月)", "High", "version date管理 + 後方互換対応"],
            ["Shopify公式がUCP MCPを出す", "Medium", "Noshi Extensionは公式に無い差別化"],
            ["AP2 Mandate仕様変動", "Medium", "GitHub SDK追従 + abstraction layer"],
            ["0.5%手数料の規約リスク", "Medium", "Payment Handler登録 or SaaS課金に切替可"],
            ["競合参入 (ACP系)", "Low", "UCP+AP2+MCPの三位一体は参入障壁高い"],
        ],
        [50, 25, 115],
    )

    # ================================================================
    # FOOTER
    # ================================================================
    pdf.ln(6)
    pdf.set_draw_color(*NAVY)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("HG", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "本設計書は Asset Shield 老中 (Claude Opus 4.6) が UCP/AP2公式仕様を精査の上作成",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Sources: shopify.engineering/ucp | ucp.dev | AP2 GitHub | Google Cloud Blog",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "上様の号令をもって実装を開始する",
             align="C")

    pdf.output(OUTPUT)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build()
