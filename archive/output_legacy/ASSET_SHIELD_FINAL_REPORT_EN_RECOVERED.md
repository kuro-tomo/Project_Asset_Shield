# Asset Shield V2 - Final Report (Post-Recovery)
## QuantConnect / Quantiacs Submission Document

---

## Executive Summary

**Asset Shield V2** is a systematic value investing strategy for Japanese equities.
This report reflects performance after data recovery for 2021+ period.

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **nan** |
| **Total Return** | **nan%** |
| **Annual Return** | **nan%** |
| **Max Drawdown** | **31.16%** |
| **OOS Sharpe** | **0.00** |

---

## Data Quality Improvement

| Year | Before Recovery | After Recovery |
|------|-----------------|----------------|
| 2021 | 1,003 stocks | 4000 stocks |
| Coverage | 25.0% | 99.5% |

---

## Walk-Forward Validation (Post-Recovery)

| Phase | Period | Sharpe | Return |
|-------|--------|--------|--------|
| Training (IS) | 2007-2015 | 0.86 | 472.9% |
| Validation | 2016-2020 | nan | nan% |
| Out-of-Sample | 2021-2026 | 0.00 | 0.0% |

**Overfitting Ratio**: nan (Threshold: >0.70)
**Degradation Ratio**: 1.00 (Threshold: >0.70)

---

## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | 119 |
| Win Rate | 61.3% |
| Profit Factor | 5.08 |
| Avg Holding Period | 354 days |

---

## Risk Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | nan |
| Sortino Ratio | nan |
| Calmar Ratio | nan |
| Volatility | 18.1% |

---

## Capacity Analysis

| Metric | Value |
|--------|-------|
| Target AUM | $200M (¥30B) |
| Avg Market Impact | 14.6 bps |
| Max Impact Allowed | 75 bps |

---

## Data Recovery Note

This report was generated after recovering missing price data for 2021-2026.
The previous OOS analysis was limited by data coverage issues (only ~25% of stocks).
Post-recovery coverage improved to 99.5% of the 2020 baseline.

---

*Report Generated: 2026-02-04 02:01:30*
*Asset Shield V2 - Unmanned March Mode*
