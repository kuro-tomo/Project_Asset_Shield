#!/usr/bin/env python3
"""Generate PDF with Upwork proposal and application URL."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT_PATH = os.path.expanduser("~/Desktop/Upwork_Proposal_QC_Backtesting.pdf")
FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_BOLD_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


class ProposalPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("gothic", size=8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"{self.page_no()}/{{nb}}", align="C")


def build_pdf():
    pdf = ProposalPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_font("gothic", "", FONT_PATH)
    pdf.add_font("gothic", "B", FONT_BOLD_PATH)
    pdf.add_page()

    LM = pdf.l_margin
    W = pdf.w - pdf.l_margin - pdf.r_margin

    # Title
    pdf.set_font("gothic", "B", 16)
    pdf.cell(0, 10, "Upwork Proposal", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(2)

    # Job title
    pdf.set_font("gothic", "B", 12)
    pdf.set_x(LM)
    pdf.multi_cell(W, 7,
        "Job: Quant Trading Dev Expert for Backtesting\n"
        "via Python/C#/Others on QuantConnect",
        align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # URL section
    pdf.set_fill_color(40, 60, 100)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("gothic", "B", 10)
    pdf.cell(0, 7, "  Application URL", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("gothic", "", 9)
    pdf.set_x(LM)
    url = ("https://www.upwork.com/freelance-jobs/apply/"
           "Quant-Trading-Dev-Expert-for-Backtesting-via-Python"
           "-Others-QuantConnect_~015f8571127970e1d3/")
    pdf.multi_cell(W, 5, url, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # Proposal section
    pdf.set_fill_color(40, 60, 100)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("gothic", "B", 10)
    pdf.cell(0, 7, "  Proposal Text (Copy & Paste)", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    proposal = (
        "I build and run quantitative strategies on QuantConnect as my core work. "
        "My current production system is a US equity Core-Satellite strategy (V8) "
        "\u2014 15-year backtest returning +649% cumulative (CAGR 14.4%, Sharpe 0.92), "
        "walk-forward validated with IS Sharpe 1.10 and OOS 0.74. It's now in paper "
        "trading on QC with IB integration.\n\n"
        "I've iterated through 8 strategy generations, so I'm well past the "
        "\"first backtest\" stage. My daily workflow is exactly what you're describing: "
        "writing strategies in QuantConnect/LEAN (Python), integrating external data "
        "via REST APIs, and producing structured output for analysis. I've built "
        "complete ETL pipelines \u2014 API ingestion, parsing, SQLite storage, signal "
        "generation \u2014 and I'm comfortable working with tick-level data sources "
        "like Tardis.\n\n"
        "On the quantitative side: 5-factor models, regime detection, "
        "inverse-volatility sizing, kill-switches, and rigorous walk-forward "
        "validation to avoid overfitting. I also participate in the Numerai "
        "Tournament (LightGBM, 705 features).\n\n"
        "I prefer async text-based communication for efficiency \u2014 it produces "
        "better-documented decisions and fits quantitative work well.\n\n"
        "Rate: $75/hr. Available to start immediately."
    )

    pdf.set_font("gothic", "", 10)
    pdf.set_x(LM)
    pdf.multi_cell(W, 6, proposal, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # Notes
    pdf.set_draw_color(180, 180, 180)
    y = pdf.get_y()
    pdf.line(LM + W * 0.3, y, LM + W * 0.7, y)
    pdf.ln(4)

    pdf.set_font("gothic", "B", 10)
    pdf.set_x(LM)
    pdf.multi_cell(W, 6, "Notes", align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    pdf.set_font("gothic", "", 9)
    pdf.set_text_color(80, 80, 80)
    notes = [
        "1. Upwork\u306b\u30ed\u30b0\u30a4\u30f3\u3057\u3001\u4e0a\u8a18URL\u3092\u958b\u304f",
        "2. \"Apply Now\" \u3092\u30af\u30ea\u30c3\u30af",
        "3. Cover Letter\u6b04\u306bProposal Text\u3092\u8cbc\u308a\u4ed8\u3051",
        "4. Hourly Rate: $75 \u3092\u78ba\u8a8d",
        "5. Submit Proposal \u3092\u30af\u30ea\u30c3\u30af",
    ]
    for note in notes:
        pdf.set_x(LM + 5)
        pdf.multi_cell(W - 5, 6, note, align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
