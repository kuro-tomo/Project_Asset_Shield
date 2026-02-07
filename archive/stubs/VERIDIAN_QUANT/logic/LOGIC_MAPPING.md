# VERIDIAN QUANT - Logic Mapping Documentation

## Porting Manifest from Asset Shield V3.2.0

**Created:** 2026-02-04
**Source System:** Asset Shield (Japanese Equities)
**Target Domain:** Horse Racing Analytics (JRA, NAR, International)

---

## 1. Ported Components

### 1.1 Vectorized Date Parsing
| Asset Shield | VERIDIAN QUANT | Status |
|--------------|----------------|--------|
| `parse_dates_vectorized()` | `parse_dates_vectorized()` | Direct Port |
| `_parse_date_cached()` | `_parse_date_cached()` | Direct Port |

**Performance Guarantee:** 100x speedup vs iterative strptime preserved.

### 1.2 Data Transformation Pipeline
| Asset Shield Component | Racing Adaptation | Notes |
|-----------------------|-------------------|-------|
| `NormalizedQuote` | `RacingFeatureVector` | Unified schema for ML input |
| `DataTransformer.normalize_quotes_vectorized()` | `VectorizedFeaturePipeline.process_race_card_batch()` | Batch processing |
| `DataTransformer.validate_data()` | Integrated validation | NaN/Inf guards |

---

## 2. Domain-Specific Adaptations

### 2.1 Track Bias Tensor (NEW)
Replaces equity market sector analysis with track surface/condition effects.

**Dimensions:**
- Rail position advantage: `[inside, middle, outside]` → Shape (3,)
- Running style advantage: `[front, stalker, closer]` → Shape (3,)
- Distance band index: `[sprint, mile, intermediate, staying]` → Scalar 0-3

**Computation Method:**
```
rail_bias = (actual_win_rate - expected_win_rate) / expected_win_rate
```

### 2.2 Bloodline Tensor (NEW)
Hereditary performance indicators replacing fundamental financial metrics.

**Vector Components (8 dimensions each):**
1. Win Rate
2. Place Rate (Top 3)
3. Distance Index (normalized around 1600m)
4. Surface Index (turf preference)
5. Stamina Index (closing tendency)
6. Speed Index (early position)
7. Class Progression
8. Consistency (finish position stability)

**Cross Compatibility:**
```
cross_score = 0.5 * complementary_score + 0.5 * bms_alignment
```

### 2.3 Sectional Timing Engine (NEW)
Furlong-by-furlong pace analysis replacing intraday price momentum.

**Key Metrics:**
- `pace_figure`: Ragozin-style composite rating (100 base)
- `finish_speed`: Final furlong velocity (m/s)
- `energy_curve`: Deceleration pattern (2nd derivative)

---

## 3. Architecture Mapping

### Asset Shield → VERIDIAN QUANT Equivalences

| Equity Concept | Racing Equivalent |
|----------------|-------------------|
| Stock Code | Horse ID |
| Sector (TSE17) | Track + Surface + Distance Band |
| Market Cap | Earnings (lifetime prizemoney) |
| ADT (Avg Daily Turnover) | Average Field Size Faced |
| P/B Ratio | Class Rating / Speed Figure |
| ROE | Win Rate |
| Volatility (σ) | Finish Position Standard Deviation |
| Almgren-Chriss Impact | Field Strength Adjustment |
| Survivorship Bias | Include retired/deceased horses |

### Data Pipeline Flow
```
JRA/NAR API
    ↓
[Rate Limiter]
    ↓
[SQLite Cache] (racing_cache.db)
    ↓
[VectorizedFeaturePipeline]
    ├─ TrackBiasEngine
    ├─ BloodlineEngine
    └─ SectionalEngine
    ↓
[RacingFeatureVector] (64-dim tensor)
    ↓
[ML Model / Alpha Generation]
    ↓
[Kelly Criterion Sizing]
    ↓
[Execution (Bet Placement)]
```

---

## 4. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] SQLite schema for racing data
- [ ] JRA scraper / API client
- [ ] Basic track bias computation

### Phase 2: Feature Engineering (Week 3-4)
- [ ] Bloodline database population
- [ ] Sectional timing ingestion
- [ ] Feature vector validation

### Phase 3: Alpha Model (Week 5-6)
- [ ] Port AlphaModel structure from Asset Shield
- [ ] Racing-specific signal generation
- [ ] Backtest framework adaptation

### Phase 4: Production (Week 7-8)
- [ ] Real-time race card processing
- [ ] Automated bet sizing (Kelly)
- [ ] Performance monitoring

---

## 5. Key Differences from Equity System

| Aspect | Equity (Asset Shield) | Racing (VERIDIAN) |
|--------|----------------------|-------------------|
| Market Hours | 9:00-15:00 JST | Race-by-race |
| Position Sizing | 10% ADV limit | Kelly Criterion |
| Rebalancing | Daily | Per race |
| Impact Model | Almgren-Chriss | Field strength |
| Data Frequency | Daily OHLCV | Per race (weekly updates) |
| Universe | ~4000 stocks | ~50 runners/day |

---

## 6. File Structure

```
VERIDIAN_QUANT/
├── logic/
│   ├── vectorized_feature_engine.py   # Ported & adapted from Asset Shield
│   ├── LOGIC_MAPPING.md               # This document
│   ├── alpha_model.py                 # To be created
│   └── execution_core.py              # To be created
├── data_ingestion/
│   ├── jra_client.py                  # To be created
│   ├── nar_client.py                  # To be created
│   └── cache.py                       # To be created
├── backtest_results/
│   └── .gitkeep
└── production_ops/
    └── .gitkeep
```

---

## 7. Dependencies

### From Asset Shield (Verified Compatible)
- numpy >= 1.24.0
- pandas >= 2.0.0
- sqlite3 (built-in)
- dataclasses (built-in)

### Racing-Specific (To Add)
- beautifulsoup4 (JRA scraping)
- requests (API calls)
- scikit-learn (ML models)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-04
**Author:** Asset Shield Porting Team
