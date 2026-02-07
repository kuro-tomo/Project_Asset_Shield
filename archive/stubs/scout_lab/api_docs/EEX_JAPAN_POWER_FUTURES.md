# EEX Japan Power Futures - API Documentation & Market Intelligence

## Data Reconnaissance Report
**Created:** 2026-02-04
**Target Market:** Japan Electric Power Exchange (JEPX) via EEX
**Focus Areas:** Liquidity, Tick Size, Data Accessibility

---

## 1. Market Overview

### Japan Electric Power Exchange (JEPX)
- **Established:** 2003
- **Products:** Day-ahead spot, Intraday, Futures
- **Trading Hours:** 24/7 for futures
- **Settlement:** Physical delivery (zonal)

### EEX Japan Power Futures
- **Underlying:** JEPX spot price (area average)
- **Contract Size:** 1 MW per delivery hour
- **Delivery Areas:** 9 zones (Hokkaido, Tohoku, Tokyo, Chubu, Hokuriku, Kansai, Chugoku, Shikoku, Kyushu)
- **Quote Currency:** JPY/kWh

---

## 2. Liquidity Analysis

### Key Metrics to Monitor
| Metric | Definition | Target Threshold |
|--------|------------|------------------|
| Open Interest | Total outstanding contracts | > 10,000 contracts |
| Daily Volume | Contracts traded per day | > 1,000 contracts |
| Bid-Ask Spread | Typical spread | < 0.5 JPY/kWh |
| Market Depth | Volume within 3 ticks | > 500 MW |

### Liquidity Concerns
- **Low liquidity hours:** 02:00-06:00 JST
- **Seasonal patterns:** Higher liquidity in summer (cooling demand)
- **Zone disparity:** Tokyo/Kansai most liquid, Hokkaido/Shikoku illiquid

---

## 3. Tick Size Specification

### EEX Japan Power Futures
| Contract | Tick Size | Tick Value |
|----------|-----------|------------|
| Monthly Base | 0.01 JPY/kWh | 7.44 JPY per contract |
| Monthly Peak | 0.01 JPY/kWh | 4.16 JPY per contract |
| Quarterly | 0.01 JPY/kWh | Variable |

### Calculation
```
Tick Value = Tick Size × Contract Size × Delivery Hours
Monthly Base = 0.01 × 1 MW × 744 hours = 7.44 JPY
```

---

## 4. Data API Options

### Option A: EEX Direct API
**Endpoint:** `https://api.eex.com/v1/market-data`
**Authentication:** API Key (institutional only)
**Rate Limit:** 1000 requests/hour
**Cost:** EUR 500/month minimum

**Available Data:**
- Real-time quotes (15-min delayed for non-members)
- Historical OHLC (daily/hourly)
- Open interest and volume
- Settlement prices

### Option B: JEPX Data Service
**Endpoint:** `https://www.jepx.org/market/spreadsheet.html`
**Authentication:** None (public)
**Format:** CSV/Excel downloads
**Lag:** T+1 for spot, T+0 EOD for summary

**Available Data:**
- Spot market prices (30-min intervals)
- System prices by area
- Trading volumes

### Option C: Third-Party Aggregators
| Provider | Coverage | Cost | API Quality |
|----------|----------|------|-------------|
| Refinitiv | Comprehensive | High | Excellent |
| Bloomberg | Comprehensive | Very High | Excellent |
| Quandl | Limited | Low | Moderate |
| OANDA | None | - | - |

---

## 5. M4 Pro Environment Compatibility

### Required Python Packages
```bash
# Core data handling
pip install pandas numpy

# API clients
pip install requests aiohttp

# Specialized energy data
pip install entsoe-py  # European, but pattern reference
pip install energyquantified  # If expanding to EU
```

### Environment Variables Needed
```bash
export EEX_API_KEY="your_api_key"
export EEX_API_SECRET="your_api_secret"
export JEPX_DATA_PATH="/path/to/jepx/downloads"
```

### Sample Data Fetcher (Skeleton)
```python
import requests
from datetime import datetime, timedelta

class EEXJapanClient:
    BASE_URL = "https://api.eex.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers['Authorization'] = f'Bearer {api_key}'

    def get_futures_quotes(self, area: str, contract_month: str):
        """Fetch futures quotes for specified area and contract"""
        endpoint = f"{self.BASE_URL}/japan-power/futures"
        params = {
            'area': area,
            'contract': contract_month,
            'granularity': 'hourly'
        }
        response = self.session.get(endpoint, params=params)
        return response.json()

    def get_historical_ohlc(self, area: str, start: datetime, end: datetime):
        """Fetch historical OHLC data"""
        endpoint = f"{self.BASE_URL}/japan-power/historical"
        params = {
            'area': area,
            'start': start.isoformat(),
            'end': end.isoformat()
        }
        response = self.session.get(endpoint, params=params)
        return response.json()
```

---

## 6. Trading Strategy Considerations

### Mean Reversion Opportunities
- **Basis trade:** Futures vs spot spread convergence
- **Area spread:** Tokyo-Kansai price differential
- **Seasonal:** Summer peak vs winter base load

### Momentum Signals
- **Fuel linkage:** LNG price correlation
- **Weather:** Temperature deviation impact
- **Demand forecast:** Industrial activity indicators

### Risk Factors
- **Regulatory:** METI price caps during crises
- **Grid constraints:** Interconnector capacity limits
- **Renewable intermittency:** Solar/wind forecast errors

---

## 7. Next Steps

### Immediate Actions
1. [ ] Register for EEX API trial access
2. [ ] Download historical JEPX spot data (2020-present)
3. [ ] Build liquidity profile for each delivery area
4. [ ] Prototype basis trading strategy

### Data Collection Checklist
- [ ] Spot prices (30-min, all 9 areas)
- [ ] Futures settlement prices (daily)
- [ ] Open interest time series
- [ ] Bid-ask snapshots (if available)
- [ ] Weather data overlay (JMA)
- [ ] LNG import price (customs data)

---

## 8. References

- [EEX Japan Power Products](https://www.eex.com/en/markets/power/japan)
- [JEPX Official](https://www.jepx.org/)
- [METI Electricity Statistics](https://www.enecho.meti.go.jp/)
- [OCCTO Grid Operations](https://www.occto.or.jp/)

---

**Document Status:** Draft
**Next Review:** 2026-02-11
