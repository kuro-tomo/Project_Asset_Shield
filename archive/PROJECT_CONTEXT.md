# Project Context: Asset Shield V2 (Project JÖRMUNGANDR)

## 1. Mission & Timeline
* **Ultimate Goal:** Exit (M&A) with a valuation of $50M - $75M (USD) by Q2 2026.
* **Product Name:** Asset Shield V2 (Commercial) / Project JÖRMUNGANDR (Internal)
* **Core Strategy:** Microstructure-based adaptive quant infrastructure for Japanese equity market (TSE).
* **Key Deliverable:** White-labeled, standalone revenue package with tamper-proof track record.

## 2. Development Roadmap

### Phase 1: Development & Verification (Feb 2026)
* **Feb 1 - Feb 15:** J-Quants integration and Itayose analysis implementation
* **Feb 16 - Mar 10:** 20-year backtest and stress test completion
* **Mar 11 - Mar 20:** Technical whitepaper preparation

### Phase 2: Audit & Due Diligence (Mar - Apr 2026)
* **Mar 21 - Mar 31:** Pre-marketing and VDR (Virtual Data Room) setup
* **Apr 01 - Apr 10:** 10-day live track record publication (Auction Phase)

## 3. Technical Architecture

### Layer 1: Data Ingestion (Microstructure)
* **J-Quants Native:** Direct integration with JPX's official J-Quants API
* **Data Types:** Order book, trade execution, margin trading data
* **Itayose Analyzer:** Pre-market (8:00-9:00 JST) order flow imbalance extraction
* **Module:** [`modules/jquants_client.py`](modules/jquants_client.py), [`modules/itayose_analyzer.py`](modules/itayose_analyzer.py)

### Layer 2: Signal Generation (Adaptive Core)
* **Regime Detection:** Crisis, High-Vol, Normal, Low-Vol, Bull/Bear Trend
* **Dynamic Recalibration:** Real-time volatility and correlation monitoring
* **Self-Regenerating:** Performance feedback loop for parameter optimization
* **Module:** [`modules/adaptive_core.py`](modules/adaptive_core.py), [`modules/brain.py`](modules/brain.py)

### Layer 3: Execution (Stealth Algo)
* **VWAP/TWAP:** Volume/Time weighted execution strategies
* **Almgren-Chriss:** Optimal execution with market impact minimization
* **TSE Compliance:** Tick size rules and trading hour awareness
* **Module:** [`modules/execution_core.py`](modules/execution_core.py)

### Supporting Modules
* **Financial Trinity:** Z-Score, F-Score, PEG analysis ([`modules/screener.py`](modules/screener.py))
* **Money Management:** Kelly-based sizing with dual-ledger ([`modules/money_management.py`](modules/money_management.py))
* **Audit Trail:** SHA-256 hashed event logging ([`modules/tracker.py`](modules/tracker.py))

## 4. Technical Stack
* **Analysis/Strategy:** Python 3.11+ (NumPy, Polars, PyTorch)
* **Execution Core:** Rust (Planned - PyO3 ABI for HFT optimization)
* **Infrastructure:** Docker / Terraform (AWS/GCP Agnostic)
* **Database:** PostgreSQL (Aurora Serverless) for audit trail

## 5. Asset Decoupling Protocol

### Version-J (Transferable)
White-labeled production package for asset transfer:
* J-Quants client and data pipeline
* Itayose analyzer
* Execution core (VWAP/TWAP)
* Financial screener
* Money management
* Audit tracker
* Infrastructure (Docker/Terraform)

### Version-Q (IP Reserved)
Next-generation core retained by originator:
* Adaptive core logic
* Brain/Evolution engine
* Experimental model weights
* Learned parameters

**Module:** [`modules/asset_decoupling.py`](modules/asset_decoupling.py)

## 6. Quantitative Verification

### 20-Year Multi-Regime Stress Test
Survivorship-bias-free testing with delisted stocks included:

| Phase | Period | Description | Key Events |
|-------|--------|-------------|------------|
| 1 | 2006-2010 | Survival Test | Lehman Shock, GFC |
| 2 | 2011-2015 | Expansion Test | Abenomics, JPY depreciation |
| 3 | 2016-2020 | OOS Stability | COVID Shock, V-recovery |
| 4 | 2021-2026 | Modern Adaptation | Inflation, rate normalization |

**Module:** [`modules/backtest_framework.py`](modules/backtest_framework.py)

### Performance Requirements
* **AUM Capacity:** ¥30B without Sharpe degradation
* **Transaction Cost:** Almgren-Chriss model with TSE tick size rules
* **Max Drawdown:** <40% during crisis phases
* **Sharpe Ratio:** >0.3 across all phases

## 7. Security & Governance

### Burn Protocol
Automated removal of developer access post-transfer:
* SSH key revocation
* API credential rotation
* Git credential clearing
* Backdoor scanning and removal

### Integrity Hash
SHA-256 certification for delivery:
* Merkle root of all file hashes
* Tamper-proof verification
* Audit-ready documentation

**Module:** [`modules/security_governance.py`](modules/security_governance.py)

## 8. Infrastructure

### One-Click Deployment
* **Terraform:** VPC, ECS, RDS, Secrets Manager
* **Docker:** Multi-stage build, non-root user, health checks
* **Region:** ap-northeast-1 (Tokyo) primary

**Files:** [`infrastructure/Dockerfile`](infrastructure/Dockerfile), [`infrastructure/terraform/main.tf`](infrastructure/terraform/main.tf)

## 9. Coding Standards (CRITICAL)
* **Language:** Python is the primary language
* **Comments:** ALL source code comments must be in ENGLISH
* **Architecture:** Modular design favoring readability and testability
* **Type Safety:** Type hints required for all public functions
* **Documentation:** Docstrings for all modules and classes

## 10. Core Algorithms

### Financial Trinity
1. **Altman Z-Score:** Bankruptcy risk assessment
2. **Piotroski F-Score:** Financial health (target: 8/9)
3. **PEG Ratio:** Valuation relative to growth

### Microstructure Analysis
1. **Order Flow Imbalance (OFI):** Buy/sell pressure asymmetry
2. **Institutional Detection:** Large block order patterns
3. **Volume Profile:** Intraday volume distribution

### Adaptive Logic
1. **Regime Classification:** Market state detection
2. **Volatility Targeting:** Dynamic position scaling
3. **Correlation Monitoring:** Risk concentration alerts

## 11. File Structure

```
Project_Asset_Shield/
├── modules/
│   ├── jquants_client.py      # J-Quants API integration
│   ├── itayose_analyzer.py    # Pre-market analysis
│   ├── adaptive_core.py       # Dynamic recalibration
│   ├── execution_core.py      # VWAP/TWAP execution
│   ├── asset_decoupling.py    # Version-J/Q separation
│   ├── backtest_framework.py  # 20-year verification
│   ├── security_governance.py # Burn protocol & integrity
│   ├── screener.py            # Financial Trinity
│   ├── brain.py               # Learning core
│   ├── evolution.py           # Evolution engine
│   ├── money_management.py    # Position sizing
│   ├── tracker.py             # Audit logging
│   └── sentiment.py           # J-Sentiment analysis
├── infrastructure/
│   ├── Dockerfile             # Container definition
│   └── terraform/
│       └── main.tf            # AWS infrastructure
├── config/                    # Configuration files
├── dashboard/                 # Web dashboard
├── data/                      # Data storage
├── logs/                      # Audit logs
└── output/                    # Generated reports
```

## 12. Environment Variables

```bash
# J-Quants API
JQUANTS_MAIL=your_email@example.com
JQUANTS_PASSWORD=your_password
JQUANTS_REFRESH_TOKEN=optional_token

# Security
ASSET_SHIELD_KEY=encryption_master_key

# AI Services
GEMINI_API_KEY=your_gemini_key
```

---

## 13. Sales Strategy (Hybrid Approach)

### Primary: M&A of Version-J (Q2 2026)
* **Target Valuation:** $50M - $75M USD
* **Timeline:** Q2 2026 (April - June)
* **Package:** Version-J (Transferable Production System)

### Secondary: Version-Q API Business (Q3 2026+)
* **Model:** Signal-as-a-Service (SaaS)
* **Target ARR:** $10-20M (Year 3)
* **Products:**
  - Tier 1: Signal API ($10K-25K/month)
  - Tier 2: Execution API ($5K-15K/month)
  - Tier 3: Full Suite ($50K-100K/month)

### Target Buyers/Customers
* **M&A:** Japanese securities firms, global asset managers, Middle Eastern sovereign wealth funds
* **API:** Hedge funds, prop trading firms, family offices

### Due Diligence Requirements
* [ ] Third-party audit (Big4 or specialized firm)
* [ ] 30-day live track record
* [ ] Legal review (FIEA/Financial Instruments and Exchange Act)
* [ ] Technical whitepaper (English)
* [ ] VDR setup (Intralinks/Datasite)

## 14. Institutional Investor Readiness

### Completed
* ✅ 20-year multi-regime backtest (survivorship-bias-free)
* ✅ Institutional-grade execution algorithms (VWAP/TWAP/Almgren-Chriss)
* ✅ Tamper-proof audit trail (SHA-256, Merkle root)
* ✅ ¥30B AUM capacity without Sharpe degradation
* ✅ TSE tick size compliance

### In Progress
* ⏳ 10-day live track record (target: 30 days)
* ⏳ Technical whitepaper preparation

### Required
* ❌ Third-party audit
* ❌ Legal/regulatory review
* ❌ Sharia compliance screening (for Middle Eastern clients)

---
**Disclaimer:** This document is prepared for auditing purposes regarding the asset transfer of "Asset Shield". All intellectual property rights regarding "Version-Q" are reserved by the originator.

**Last Updated:** January 26, 2026
**Version:** 2.2.0-STRATEGY
