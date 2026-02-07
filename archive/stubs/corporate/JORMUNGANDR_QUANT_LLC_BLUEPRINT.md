# Jörmungandr Quant LLC - Structural Blueprint

## Corporate Structure & QuantConnect Alpha Streams Integration

**Document Version:** 1.0
**Created:** 2026-02-04
**Classification:** Administrative Draft

---

## 1. Entity Overview

### Name & Concept
**Jörmungandr Quant LLC**

*Jörmungandr (Old Norse: "huge monster") - The World Serpent of Norse mythology that encircles the entire world. Symbolizes:*
- **Comprehensive coverage:** All asset classes, global markets
- **Self-sustaining cycle:** Algorithmic systems that evolve autonomously
- **Interconnected systems:** Multi-strategy, multi-asset integration

### Entity Structure
```
Jörmungandr Quant LLC (Delaware)
│
├── Asset Shield Division
│   └── Japanese Equities (TSE Prime/Standard)
│
├── VERIDIAN QUANT Division
│   └── Horse Racing Analytics (JRA/NAR)
│
└── Market Scouting Division
    ├── Energy Futures (EEX Japan Power)
    └── Crypto Perpetuals (BTC/ETH/SOL)
```

---

## 2. QuantConnect Alpha Streams Requirements

### What is Alpha Streams?
QuantConnect's marketplace where algorithm authors can license trading signals to institutional investors. Algorithms run in LEAN format on QC infrastructure.

### LEAN Format Specifications

#### File Structure
```
JormungandrQuant/
├── main.py                 # Algorithm entry point
├── alpha/
│   ├── __init__.py
│   ├── asset_shield_alpha.py
│   ├── veridian_alpha.py
│   └── market_scout_alpha.py
├── execution/
│   ├── __init__.py
│   └── almgren_chriss.py
├── portfolio/
│   ├── __init__.py
│   └── sector_neutral.py
├── universe/
│   ├── __init__.py
│   └── tse_prime_filter.py
├── data/
│   └── custom_data_sources.py
└── config.json
```

#### Main Algorithm Template
```python
from AlgorithmImports import *

class JormungandrQuantAlgorithm(QCAlgorithm):
    """
    Jörmungandr Quant - Multi-Strategy Algorithm
    LEAN Format for QuantConnect Alpha Streams
    """

    def Initialize(self):
        # Basic settings
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000000)  # 100M JPY equivalent
        self.SetBrokerageModel(BrokerageName.InteractiveBrokers, AccountType.Margin)

        # Universe selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        # Alpha model
        self.SetAlpha(AssetShieldAlphaModel())

        # Portfolio construction
        self.SetPortfolioConstruction(SectorNeutralPortfolioModel())

        # Execution
        self.SetExecution(AlmgrenChrissExecutionModel())

        # Risk management
        self.SetRiskManagement(MaxDrawdownRiskModel(0.15))

    def CoarseSelectionFunction(self, coarse):
        """Filter universe by liquidity"""
        return [x.Symbol for x in coarse
                if x.HasFundamentalData
                and x.DollarVolume > 100000000]  # 100M JPY ADT

    def FineSelectionFunction(self, fine):
        """Filter by fundamentals"""
        return [x.Symbol for x in fine
                if x.ValuationRatios.PBRatio > 0
                and x.ValuationRatios.PBRatio < 3.0
                and x.OperationRatios.ROE.OneYear > 0.08]
```

#### Alpha Model Interface
```python
from AlgorithmImports import *

class AssetShieldAlphaModel(AlphaModel):
    """
    Asset Shield Alpha Model - Ported to LEAN Format
    """

    def __init__(self, lookback_days: int = 60, rebalance_frequency: int = 5):
        self.lookback_days = lookback_days
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance = None

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        """
        Generate alpha insights based on Asset Shield logic.

        Returns:
            List of Insight objects with direction, magnitude, confidence
        """
        insights = []

        # Check rebalance frequency
        if self.last_rebalance and (algorithm.Time - self.last_rebalance).days < self.rebalance_frequency:
            return insights

        self.last_rebalance = algorithm.Time

        # Get universe symbols
        symbols = [x.Symbol for x in algorithm.ActiveSecurities.Values]

        for symbol in symbols:
            # Compute alpha score (placeholder - implement Asset Shield logic)
            history = algorithm.History(symbol, self.lookback_days, Resolution.Daily)
            if history.empty:
                continue

            alpha_score = self._compute_alpha_score(history)

            if alpha_score > 0.5:
                insights.append(Insight.Price(
                    symbol,
                    timedelta(days=self.rebalance_frequency),
                    InsightDirection.Up,
                    magnitude=alpha_score,
                    confidence=min(0.9, alpha_score)
                ))
            elif alpha_score < -0.5:
                insights.append(Insight.Price(
                    symbol,
                    timedelta(days=self.rebalance_frequency),
                    InsightDirection.Down,
                    magnitude=abs(alpha_score),
                    confidence=min(0.9, abs(alpha_score))
                ))

        return insights

    def _compute_alpha_score(self, history: pd.DataFrame) -> float:
        """
        Compute composite alpha score.
        Port Asset Shield's multi-factor model here.
        """
        # Placeholder implementation
        returns = history['close'].pct_change().dropna()
        momentum = returns.mean() * 252  # Annualized
        volatility = returns.std() * np.sqrt(252)
        sharpe = momentum / (volatility + 1e-8)
        return np.tanh(sharpe)  # Normalize to [-1, 1]
```

---

## 3. Alpha Streams Submission Requirements

### Performance Criteria
| Metric | Minimum Requirement |
|--------|---------------------|
| Sharpe Ratio | > 1.0 |
| Sortino Ratio | > 1.5 |
| Max Drawdown | < 20% |
| Win Rate | > 50% |
| Track Record | > 6 months OOS |

### Documentation Requirements
1. **Strategy Overview:** 2-page summary of approach
2. **Backtest Report:** Full 10+ year backtest with statistics
3. **Risk Disclosure:** Detailed risk factors and limitations
4. **Capacity Analysis:** Maximum AUM before alpha decay

### Compliance Considerations
- **No high-frequency:** Minimum 1-day holding period
- **No illiquidity exploitation:** Must trade in liquid names
- **Auditable logic:** All signals must be explainable
- **No data leakage:** Strict point-in-time data validation

---

## 4. Revenue Model

### Alpha Streams Economics
```
Licensing Fee Structure:
- Base: 0.5% - 2.0% of AUM per annum
- Performance: 10% - 20% of alpha generated

Revenue Share:
- QuantConnect: 20% of licensing fees
- Algorithm Author: 80% of licensing fees

Example (100M USD AUM):
- Base Fee (1%): $1,000,000/year
- Author Share: $800,000/year
- QC Share: $200,000/year
```

### Multi-Division Revenue Projection
| Division | Strategy | Est. Capacity | Est. Annual Revenue |
|----------|----------|---------------|---------------------|
| Asset Shield | JP Equities | 30B JPY | $1.5M |
| VERIDIAN | Racing Alpha | N/A | $200K (direct) |
| Market Scout | Energy/Crypto | 10M USD | $100K |
| **Total** | | | **$1.8M** |

---

## 5. Operational Structure

### Team Roles (Future)
```
CEO/Lead Quant: Strategy development, risk oversight
CTO: Infrastructure, algorithm deployment
COO: Compliance, investor relations
```

### Technology Stack
```
Algorithm Development:
- Python 3.11+
- LEAN Engine (QuantConnect)
- pandas, numpy, scikit-learn

Data Infrastructure:
- SQLite (local development)
- PostgreSQL (production)
- AWS S3 (data lake)

Execution:
- Interactive Brokers API
- QuantConnect Cloud
```

### Compliance Framework
1. **SEC/CFTC:** Register as exempt reporting adviser if < $150M AUM
2. **Japan FSA:** May require registration for JP equities
3. **Data Privacy:** No PII in algorithms
4. **Audit Trail:** Full logging of all decisions

---

## 6. Integration Roadmap

### Phase 1: Foundation (Q1 2026)
- [ ] Delaware LLC formation
- [ ] EIN registration
- [ ] Bank account setup
- [ ] QuantConnect account verification

### Phase 2: Development (Q2 2026)
- [ ] Port Asset Shield to LEAN format
- [ ] Run 6-month paper trading
- [ ] Document strategy for submission
- [ ] Internal compliance review

### Phase 3: Launch (Q3 2026)
- [ ] Submit to Alpha Streams review
- [ ] Address reviewer feedback
- [ ] Go live on marketplace
- [ ] Monitor performance and AUM

### Phase 4: Expansion (Q4 2026+)
- [ ] Add VERIDIAN racing signals (if applicable)
- [ ] Add energy/crypto strategies
- [ ] Scale AUM and licensing

---

## 7. Risk Factors

### Business Risks
- **Alpha decay:** Strategies may lose edge over time
- **Competition:** Other quants with similar approaches
- **Platform dependency:** Reliance on QuantConnect

### Operational Risks
- **Single point of failure:** Key person risk
- **Technology failure:** System outages
- **Data quality:** Bad data leading to bad signals

### Regulatory Risks
- **Registration requirements:** May change
- **Cross-border complexity:** JP + US operations
- **Tax optimization:** Transfer pricing concerns

---

## 8. Appendix: LEAN Code Templates

### Sector Neutral Portfolio Model
```python
from AlgorithmImports import *

class SectorNeutralPortfolioModel(PortfolioConstructionModel):
    """
    Sector-neutral portfolio construction.
    Maximum 20% exposure per sector.
    """

    MAX_SECTOR_EXPOSURE = 0.20

    def CreateTargets(self, algorithm, insights):
        targets = []

        # Group insights by sector
        sector_groups = self._group_by_sector(algorithm, insights)

        # Apply sector caps
        for sector, sector_insights in sector_groups.items():
            sector_weight = min(self.MAX_SECTOR_EXPOSURE, len(sector_insights) * 0.02)
            per_stock_weight = sector_weight / len(sector_insights)

            for insight in sector_insights:
                direction = 1 if insight.Direction == InsightDirection.Up else -1
                targets.append(PortfolioTarget(
                    insight.Symbol,
                    direction * per_stock_weight
                ))

        return targets

    def _group_by_sector(self, algorithm, insights):
        groups = {}
        for insight in insights:
            security = algorithm.Securities[insight.Symbol]
            sector = security.Fundamentals.AssetClassification.MorningstarSectorCode
            if sector not in groups:
                groups[sector] = []
            groups[sector].append(insight)
        return groups
```

### Almgren-Chriss Execution Model
```python
from AlgorithmImports import *

class AlmgrenChrissExecutionModel(ExecutionModel):
    """
    Almgren-Chriss optimal execution.
    Minimizes market impact + timing risk.
    """

    GAMMA = 0.10  # Permanent impact
    ETA = 0.01    # Temporary impact
    MAX_PARTICIPATION = 0.10  # 10% of ADV

    def Execute(self, algorithm, targets):
        for target in targets:
            security = algorithm.Securities[target.Symbol]

            # Calculate order size respecting participation limit
            adv = security.Volume * security.Price
            max_order_value = adv * self.MAX_PARTICIPATION
            target_value = abs(target.Quantity) * security.Price

            actual_value = min(target_value, max_order_value)
            actual_shares = int(actual_value / security.Price)

            if actual_shares > 0:
                direction = 1 if target.Quantity > 0 else -1
                algorithm.MarketOrder(target.Symbol, direction * actual_shares)
```

---

**Document Status:** Draft
**Next Review:** 2026-02-11
**Confidentiality:** Internal Use Only
