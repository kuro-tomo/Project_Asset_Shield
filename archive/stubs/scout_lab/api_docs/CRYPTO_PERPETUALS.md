# Crypto Perpetuals - API Documentation & Data Pipeline

## Data Reconnaissance Report
**Created:** 2026-02-04
**Target Products:** Perpetual Futures (BTC, ETH, SOL, etc.)
**Focus Areas:** Funding Rate History, Order Book Depth, CCXT Compatibility

---

## 1. Perpetual Futures Overview

### What Are Perpetuals?
- **Definition:** Futures contracts with no expiration date
- **Settlement:** Cash-settled (no physical delivery)
- **Funding Mechanism:** Periodic payments between longs/shorts to anchor to spot price
- **Leverage:** Up to 125x on major exchanges

### Key Exchanges (by Volume)
| Exchange | 24h Volume (BTC) | Funding Interval | API Quality |
|----------|------------------|------------------|-------------|
| Binance | ~$15B | 8 hours | Excellent |
| Bybit | ~$8B | 8 hours | Excellent |
| OKX | ~$6B | 8 hours | Good |
| dYdX | ~$2B | 1 hour | Good |
| GMX | ~$500M | Continuous | Moderate |

---

## 2. Funding Rate Mechanics

### Calculation (Binance Standard)
```
Funding Rate = Premium Index + clamp(Interest Rate - Premium Index, -0.05%, 0.05%)

Where:
- Premium Index = (Mark Price - Spot Index) / Spot Index
- Interest Rate = (Quote Interest - Base Interest) / Funding Interval
```

### Funding Rate Arbitrage Signals
| Condition | Signal | Action |
|-----------|--------|--------|
| Funding > +0.1% | Bearish bias | Short perp, Long spot |
| Funding < -0.1% | Bullish bias | Long perp, Short spot |
| Funding → 0 | Neutral | Close arb positions |

### Historical Patterns
- **Bull markets:** Sustained positive funding (longs pay shorts)
- **Bear markets:** Sustained negative funding (shorts pay longs)
- **Ranging:** Oscillating around zero

---

## 3. Order Book Depth Analysis

### Key Metrics
| Metric | Definition | Usage |
|--------|------------|-------|
| Bid Depth (5%) | Total bids within 5% of mid | Liquidity measure |
| Ask Depth (5%) | Total asks within 5% of mid | Liquidity measure |
| Imbalance Ratio | (Bids - Asks) / (Bids + Asks) | Short-term direction |
| Spread | Best Ask - Best Bid | Execution cost |

### Depth Levels to Monitor
```python
DEPTH_LEVELS = [
    0.001,  # 0.1% - Scalping
    0.005,  # 0.5% - Day trading
    0.010,  # 1.0% - Swing trading
    0.050,  # 5.0% - Position sizing
]
```

---

## 4. CCXT Integration

### Installation & Setup
```bash
pip install ccxt

# For async operations (recommended)
pip install ccxt[async]
```

### M4 Pro Environment Configuration
```bash
# Add to ~/.zshrc or ~/.bash_profile
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"

# Rate limit settings (CCXT auto-handles, but good to know)
export CCXT_RATE_LIMIT_ENABLED=true
```

### Sample CCXT Client
```python
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class CryptoDataPipeline:
    """
    CCXT-based data pipeline for crypto perpetuals.
    Optimized for M4 Pro with async operations.
    """

    SUPPORTED_EXCHANGES = ['binance', 'bybit', 'okx']

    def __init__(self, exchange_id: str = 'binance'):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Perpetual futures
            }
        })

    def fetch_funding_rate_history(
        self,
        symbol: str = 'BTC/USDT:USDT',
        since: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.

        Args:
            symbol: Unified CCXT symbol (e.g., 'BTC/USDT:USDT' for perp)
            since: Start datetime (default: 7 days ago)
            limit: Number of records

        Returns:
            DataFrame with funding rate history
        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)

        since_ts = int(since.timestamp() * 1000)

        # Binance-specific endpoint
        if self.exchange_id == 'binance':
            funding = self.exchange.fetch_funding_rate_history(
                symbol, since=since_ts, limit=limit
            )
        else:
            # Generic fallback
            funding = self.exchange.fetch_funding_rates([symbol])

        df = pd.DataFrame(funding)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['funding_rate'] = df['fundingRate']

        return df[['datetime', 'symbol', 'funding_rate']]

    def fetch_order_book(
        self,
        symbol: str = 'BTC/USDT:USDT',
        limit: int = 100
    ) -> Dict:
        """
        Fetch current order book with depth analysis.

        Args:
            symbol: Unified CCXT symbol
            limit: Depth levels

        Returns:
            Dict with order book and computed metrics
        """
        ob = self.exchange.fetch_order_book(symbol, limit=limit)

        # Compute depth metrics
        mid_price = (ob['bids'][0][0] + ob['asks'][0][0]) / 2

        def depth_within_pct(side: List, pct: float) -> float:
            threshold = mid_price * (1 - pct) if side == ob['bids'] else mid_price * (1 + pct)
            return sum(
                price * amount for price, amount in side
                if (price >= threshold if side == ob['bids'] else price <= threshold)
            )

        bid_depth_1pct = depth_within_pct(ob['bids'], 0.01)
        ask_depth_1pct = depth_within_pct(ob['asks'], 0.01)

        imbalance = (bid_depth_1pct - ask_depth_1pct) / (bid_depth_1pct + ask_depth_1pct + 1e-8)
        spread_bps = (ob['asks'][0][0] - ob['bids'][0][0]) / mid_price * 10000

        return {
            'symbol': symbol,
            'timestamp': ob['timestamp'],
            'mid_price': mid_price,
            'best_bid': ob['bids'][0][0],
            'best_ask': ob['asks'][0][0],
            'spread_bps': spread_bps,
            'bid_depth_1pct': bid_depth_1pct,
            'ask_depth_1pct': ask_depth_1pct,
            'imbalance': imbalance,
            'raw_bids': ob['bids'][:20],
            'raw_asks': ob['asks'][:20],
        }

    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT:USDT',
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for perpetual contract.

        Args:
            symbol: Unified CCXT symbol
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            since: Start datetime
            limit: Number of candles

        Returns:
            DataFrame with OHLCV data
        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=30)

        since_ts = int(since.timestamp() * 1000)

        ohlcv = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ts, limit=limit
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df


# Async version for high-frequency data collection
class AsyncCryptoDataPipeline:
    """Async version for concurrent data fetching across exchanges"""

    def __init__(self, exchange_ids: List[str] = None):
        if exchange_ids is None:
            exchange_ids = ['binance', 'bybit', 'okx']

        self.exchanges = {}
        for ex_id in exchange_ids:
            self.exchanges[ex_id] = getattr(ccxt_async, ex_id)({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })

    async def fetch_all_funding_rates(self, symbol: str) -> Dict[str, float]:
        """Fetch funding rates from all exchanges concurrently"""
        import asyncio

        async def fetch_single(ex_id: str, exchange):
            try:
                result = await exchange.fetch_funding_rate(symbol)
                return ex_id, result.get('fundingRate', 0)
            except Exception as e:
                return ex_id, None

        tasks = [
            fetch_single(ex_id, ex)
            for ex_id, ex in self.exchanges.items()
        ]
        results = await asyncio.gather(*tasks)

        return {ex_id: rate for ex_id, rate in results if rate is not None}

    async def close(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            await exchange.close()
```

---

## 5. Data Storage Schema

### SQLite Schema for Perpetuals
```sql
CREATE TABLE IF NOT EXISTS funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    mark_price REAL,
    index_price REAL,
    cached_at TEXT,
    UNIQUE(exchange, symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    mid_price REAL NOT NULL,
    spread_bps REAL,
    bid_depth_1pct REAL,
    ask_depth_1pct REAL,
    imbalance REAL,
    raw_data TEXT,  -- JSON blob
    cached_at TEXT
);

CREATE INDEX idx_funding_symbol_time ON funding_rates(symbol, timestamp);
CREATE INDEX idx_orderbook_symbol_time ON order_book_snapshots(symbol, timestamp);
```

---

## 6. Strategy Ideas

### Funding Rate Arbitrage
```
Long Basis: Long Spot + Short Perp (when funding positive)
Short Basis: Short Spot + Long Perp (when funding negative)

Expected Return = |Funding Rate| × 3 × 365 (annualized)
Risk: Funding rate reversal, liquidation on leverage side
```

### Order Flow Imbalance
```
Signal: Sustained imbalance > 0.3 suggests directional move
Entry: Market order in direction of imbalance
Exit: Imbalance returns to neutral or reverses
```

### Cross-Exchange Arbitrage
```
Condition: Funding rate divergence between exchanges > 0.05%
Action: Long low-funding exchange, Short high-funding exchange
```

---

## 7. M4 Pro Optimization Notes

### Performance Considerations
- **Async preferred:** Use `ccxt.async_support` for concurrent fetching
- **Rate limits:** Binance (1200 req/min), Bybit (600 req/min)
- **Local caching:** SQLite with WAL mode for fast writes
- **Memory:** Order book snapshots can grow large; limit depth

### Environment Check Script
```python
import sys
import platform
import ccxt

def check_environment():
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"CCXT Version: {ccxt.__version__}")
    print(f"Supported Exchanges: {len(ccxt.exchanges)}")

    # Test connection
    exchange = ccxt.binance({'options': {'defaultType': 'swap'}})
    markets = exchange.load_markets()
    perp_count = len([m for m in markets if markets[m]['swap']])
    print(f"Binance Perpetual Markets: {perp_count}")

if __name__ == '__main__':
    check_environment()
```

---

## 8. Next Steps

### Immediate Actions
1. [ ] Install CCXT and test connection to Binance/Bybit
2. [ ] Create API keys (read-only for data collection)
3. [ ] Initialize SQLite database with schema
4. [ ] Schedule hourly funding rate collection

### Data Collection Priority
1. Funding rate history (all major perps) - 1 year backfill
2. Hourly OHLCV for BTC/ETH/SOL perps - 1 year backfill
3. Order book snapshots (top 20 levels) - start real-time

---

## 9. References

- [CCXT Documentation](https://docs.ccxt.com/)
- [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)
- [Bybit V5 API](https://bybit-exchange.github.io/docs/)
- [Understanding Funding Rates](https://www.binance.com/en/support/faq/360033525031)

---

**Document Status:** Draft
**Next Review:** 2026-02-11
