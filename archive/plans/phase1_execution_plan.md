# Phase 1 Execution Plan: Value Maximization & Verification

**Period:** Now - Feb 15, 2026
**Objective:** Confirm "Asset Shield V2" valuation (Target: $8M - $12M) through verified 20-year backtest performance.

## 1. Key Milestones

| Date | Milestone | Description |
|------|-----------|-------------|
| **1/28 - 1/30** | **Preparation** | Code freeze, dry-runs, environment verification. |
| **1/30 (Fri) 22:00** | **Data Release** | **J-Quants full historical data (20 years) becomes available.** Start ingestion immediately. |
| **1/31 - 2/05** | **Ingestion & Initial Training** | Complete data download (cache construction) and run initial `train` mode backtests. |
| **2/06 - 2/10** | **Parameter Tuning** | Refine `adaptive_core.py` and `brain.py` parameters to hit >15% CAGR while maintaining Sharpe > 1.3. |
| **2/11 - 2/15** | **Final Verification** | Run OOS (Out-of-Sample) verification and generate `INTEGRITY_CERTIFICATE.json`. |

## 2. Detailed Action Plan

### Step 1: Preparation (Now - 1/30)
- [ ] **Environment Check:** verify `scripts/run_jquants_backtest.py --dry-run` works without errors.
- [ ] **Credential Setup:** Ensure J-Quants credentials are securely configured (using `env` or `aws_secrets`).
- [ ] **Disk Space Check:** Ensure sufficient disk space for 20 years of daily data + microstructure samples (approx. 5-10GB for target universe).

### Step 2: Data Ingestion (1/30 22:00 - )
- **Strategy:** Prioritize "Training Phase" data (2006-2015) first to start learning process early.
- **Command:**
  ```bash
  # First pass: Get training data (Lehman Shock to Abenomics)
  python scripts/run_jquants_backtest.py --mode train --codes 7203,9984,6758 --plan standard
  ```
- **Note:** The script automatically caches data to `data/jquants_cache.db`.

### Step 3: Parameter Tuning (The "Brain" Training)
- **Goal:** Optimize `adaptive_threshold` and `risk_penalty` in `ShieldBrain` to hit >15% CAGR.
- **Process (Baseline):**
  1. Run `--mode train` (Phases 1 & 2).
  2. The `ShieldBrain` automatically adjusts:
     - Increases `risk_penalty` on losses (tighter risk).
     - Decreases `adaptive_threshold` on profits (more aggressive).
  3. Analyze `output/brain_states_trained.json`.
- **Process (Advanced - if Sharpe < 1.3):**
  1. Integrate `AdaptiveCore` (`src/shield/adaptive_core.py`) into `scripts/run_jquants_backtest.py`.
  2. Use `AdaptiveCore.get_position_size()` to dynamically reduce exposure during High Vol regimes.
  3. Re-run backtest with dynamic position sizing.

### Step 4: Verification (The "Audit" Run)
- **Goal:** Prove the system works on unseen data (COVID Shock + Modern Era).
- **Command:**
  ```bash
  # Verify using learned parameters
  python scripts/run_jquants_backtest.py --mode verify --brain-states output/brain_states_trained.json
  ```
- **Success Criteria:**
  - [ ] No bankruptcy (Survival) during COVID.
  - [ ] Recovery faster than TOPIX.
  - [ ] Sharpe Ratio > 1.0 in verification phase.

### Step 5: Output Generation
- Create the "Audit Package" for the M&A negotiation:
  - `output/ma_package/TECHNICAL_DD_REPORT.json` (Backtest stats)
  - `output/ma_package/INTEGRITY_CERTIFICATE.json` (Code & Data hash)

## 3. Risk Management
- **API Rate Limits:** `JQuantsBacktestProvider` handles rate limiting, but for 20 years of data, `Standard` plan or higher is recommended to avoid multi-day downloads.
- **Data Quality:** Watch for "0 volume" days or missing adjustments in older data. The `BacktestFramework` handles delisted stocks, so ensure the universe selection includes them for survivorship-bias-free testing.

## 4. Next Steps (Immediate)
1. Run dry-run to confirm system readiness.
2. Review `src/shield/brain.py` specifically for initial parameter ranges.
