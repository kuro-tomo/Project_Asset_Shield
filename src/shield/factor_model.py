"""
Asset Shield V5.0 - Multi-Factor Alpha Model
Japanese Equity Factor Model for Institutional Use

Factors:
1. Value (PBR): Lower Price-to-Book = Higher Score
2. Quality (ROE): Higher Return on Equity = Higher Score
3. Momentum (12-1): 12-month return excluding last month
4. Low Volatility: Lower 60-day volatility = Higher Score
5. Earnings Revision: Positive EPS change = Higher Score

Author: Asset Shield V5 Team
Version: 5.0.0 (2026-02-06)
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FactorScores:
    """Factor scores for a single stock on a single date"""
    code: str
    date: date
    close: float
    volume: float
    adt_20d: float

    # Raw factors
    pbr: Optional[float]
    roe: Optional[float]
    mom_12m: Optional[float]
    mom_1m: Optional[float]
    vol_60d: Optional[float]
    eps_change: Optional[float]

    # Z-scores (cross-sectional)
    z_value: float = 0.0
    z_quality: float = 0.0
    z_momentum: float = 0.0
    z_lowvol: float = 0.0
    z_revision: float = 0.0

    # Composite
    composite_score: float = 0.0
    rank: int = 0


@dataclass
class FactorWeights:
    """Factor weights for composite score calculation"""
    value: float = 0.25      # PBR (negative: lower is better)
    quality: float = 0.25    # ROE (positive: higher is better)
    momentum: float = 0.30   # 12-1 month momentum
    low_vol: float = 0.15    # Low volatility
    revision: float = 0.05   # Earnings revision

    def validate(self):
        total = self.value + self.quality + self.momentum + self.low_vol + self.revision
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"


class MultiFactorModel:
    """
    Multi-Factor Alpha Model for Japanese Equities

    Computes factor scores using Point-in-Time (PIT) data
    to avoid look-ahead bias.
    """

    def __init__(
        self,
        db_path: str = "/Users/MBP/Desktop/Project_Asset_Shield/data/jquants_cache.db",
        weights: Optional[FactorWeights] = None
    ):
        self.db_path = db_path
        self.weights = weights or FactorWeights()
        self.weights.validate()

        # Data caches
        self._price_data: Dict[str, pd.DataFrame] = {}
        self._financial_data: Dict[str, pd.DataFrame] = {}
        self._sector_map: Dict[str, str] = {}

        logger.info(f"MultiFactorModel initialized with weights: {self.weights}")

    def load_data(self, start_date: date, end_date: date) -> None:
        """Load all required data from database"""
        logger.info(f"Loading data from {start_date} to {end_date}")

        conn = sqlite3.connect(self.db_path)

        # Load sector mapping
        sector_df = pd.read_sql_query("""
            SELECT code, sector17_code, sector33_code
            FROM listed_info
            WHERE sector17_code IS NOT NULL
        """, conn)

        for _, row in sector_df.iterrows():
            self._sector_map[row['code']] = row['sector17_code']

        # Load price data
        start_str = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        price_df = pd.read_sql_query(f"""
            SELECT code, date,
                   adjustment_close as close,
                   adjustment_volume as volume
            FROM daily_quotes
            WHERE date >= '{start_str}' AND date <= '{end_str}'
              AND adjustment_close > 0
            ORDER BY code, date
        """, conn)

        # Organize price data by code
        for code in price_df['code'].unique():
            stock_data = price_df[price_df['code'] == code].copy()
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data.set_index('date', inplace=True)
            stock_data = stock_data.sort_index()
            self._price_data[code] = stock_data

        logger.info(f"Loaded price data for {len(self._price_data)} stocks")

        # Load financial data (PIT: use disclosed_date)
        fin_df = pd.read_sql_query(f"""
            SELECT code, disclosed_date, fiscal_year, fiscal_quarter,
                   roe, bps, eps, equity, total_assets
            FROM financial_statements
            WHERE disclosed_date >= '{start_str}' AND disclosed_date <= '{end_str}'
              AND bps > 0
            ORDER BY code, disclosed_date
        """, conn)

        # Organize financial data by code
        for code in fin_df['code'].unique():
            fin_data = fin_df[fin_df['code'] == code].copy()
            fin_data['disclosed_date'] = pd.to_datetime(fin_data['disclosed_date'])
            fin_data.set_index('disclosed_date', inplace=True)
            fin_data = fin_data.sort_index()
            self._financial_data[code] = fin_data

        logger.info(f"Loaded financial data for {len(self._financial_data)} stocks")

        conn.close()

    def get_pit_financials(self, code: str, as_of: date) -> Optional[Dict]:
        """
        Get Point-in-Time financial data as of a given date.
        Only uses data that was actually available at that time.
        """
        if code not in self._financial_data:
            return None

        fin_df = self._financial_data[code]
        as_of_ts = pd.Timestamp(as_of)

        # Get most recent disclosed data as of this date
        available = fin_df[fin_df.index <= as_of_ts]
        if len(available) == 0:
            return None

        latest = available.iloc[-1]

        # Get previous quarter for EPS change calculation
        prev_eps = None
        if len(available) >= 2:
            prev_eps = available.iloc[-2]['eps']

        return {
            'roe': latest['roe'],
            'bps': latest['bps'],
            'eps': latest['eps'],
            'prev_eps': prev_eps,
            'equity': latest['equity'],
            'total_assets': latest['total_assets']
        }

    def calculate_price_factors(
        self,
        code: str,
        as_of: date
    ) -> Dict[str, Optional[float]]:
        """Calculate price-based factors for a stock"""
        if code not in self._price_data:
            return {'mom_12m': None, 'mom_1m': None, 'vol_60d': None,
                    'close': None, 'volume': None, 'adt_20d': None}

        df = self._price_data[code]
        as_of_ts = pd.Timestamp(as_of)

        # Get data up to as_of date
        available = df[df.index <= as_of_ts]
        if len(available) < 60:
            return {'mom_12m': None, 'mom_1m': None, 'vol_60d': None,
                    'close': None, 'volume': None, 'adt_20d': None}

        close = available['close'].iloc[-1]
        volume = available['volume'].iloc[-1]

        # ADT (20-day average daily turnover)
        recent_20 = available.tail(20)
        adt_20d = (recent_20['close'] * recent_20['volume']).mean()

        # 60-day volatility (annualized)
        returns_60d = available['close'].pct_change().tail(60).dropna()
        vol_60d = returns_60d.std() * np.sqrt(252) if len(returns_60d) >= 30 else None

        # 1-month momentum
        if len(available) >= 22:
            mom_1m = (close / available['close'].iloc[-22]) - 1
        else:
            mom_1m = None

        # 12-month momentum (excluding last month) = 12-1 momentum
        if len(available) >= 252:
            price_12m_ago = available['close'].iloc[-252]
            price_1m_ago = available['close'].iloc[-22]
            mom_12m = (price_1m_ago / price_12m_ago) - 1
        else:
            mom_12m = None

        return {
            'close': close,
            'volume': volume,
            'adt_20d': adt_20d,
            'mom_12m': mom_12m,
            'mom_1m': mom_1m,
            'vol_60d': vol_60d
        }

    def compute_factor_scores(
        self,
        as_of: date,
        min_adt: float = 100_000_000  # 1億円
    ) -> List[FactorScores]:
        """
        Compute factor scores for all eligible stocks on a given date.
        """
        all_scores = []

        # Calculate raw factors for all stocks
        for code in self._price_data.keys():
            price_factors = self.calculate_price_factors(code, as_of)

            # Skip if insufficient price data or low liquidity
            if price_factors['close'] is None:
                continue
            if price_factors['adt_20d'] is None or price_factors['adt_20d'] < min_adt:
                continue

            fin_data = self.get_pit_financials(code, as_of)

            # Calculate PBR
            pbr = None
            if fin_data and fin_data['bps'] and fin_data['bps'] > 0:
                pbr = price_factors['close'] / fin_data['bps']

            # Calculate EPS change
            eps_change = None
            if fin_data and fin_data['eps'] and fin_data['prev_eps']:
                if fin_data['prev_eps'] != 0:
                    eps_change = (fin_data['eps'] - fin_data['prev_eps']) / abs(fin_data['prev_eps'])

            score = FactorScores(
                code=code,
                date=as_of,
                close=price_factors['close'],
                volume=price_factors['volume'],
                adt_20d=price_factors['adt_20d'],
                pbr=pbr,
                roe=fin_data['roe'] if fin_data else None,
                mom_12m=price_factors['mom_12m'],
                mom_1m=price_factors['mom_1m'],
                vol_60d=price_factors['vol_60d'],
                eps_change=eps_change
            )
            all_scores.append(score)

        if not all_scores:
            return []

        # Cross-sectional z-score normalization
        self._compute_z_scores(all_scores)

        # Compute composite scores
        for score in all_scores:
            score.composite_score = (
                self.weights.value * score.z_value +
                self.weights.quality * score.z_quality +
                self.weights.momentum * score.z_momentum +
                self.weights.low_vol * score.z_lowvol +
                self.weights.revision * score.z_revision
            )

        # Rank by composite score (higher = better)
        all_scores.sort(key=lambda x: x.composite_score, reverse=True)
        for i, score in enumerate(all_scores):
            score.rank = i + 1

        return all_scores

    def _compute_z_scores(self, scores: List[FactorScores]) -> None:
        """Compute cross-sectional z-scores for all factors"""

        # Value (PBR): NEGATIVE z-score (lower PBR = higher score)
        pbr_values = [s.pbr for s in scores if s.pbr is not None]
        if pbr_values:
            pbr_mean = np.mean(pbr_values)
            pbr_std = np.std(pbr_values)
            if pbr_std > 0:
                for s in scores:
                    if s.pbr is not None:
                        # Negative: lower PBR gets higher z-score
                        s.z_value = -(s.pbr - pbr_mean) / pbr_std
                        # Winsorize
                        s.z_value = np.clip(s.z_value, -3, 3)

        # Quality (ROE): POSITIVE z-score
        roe_values = [s.roe for s in scores if s.roe is not None]
        if roe_values:
            roe_mean = np.mean(roe_values)
            roe_std = np.std(roe_values)
            if roe_std > 0:
                for s in scores:
                    if s.roe is not None:
                        s.z_quality = (s.roe - roe_mean) / roe_std
                        s.z_quality = np.clip(s.z_quality, -3, 3)

        # Momentum (12-1): POSITIVE z-score
        mom_values = [s.mom_12m for s in scores if s.mom_12m is not None]
        if mom_values:
            mom_mean = np.mean(mom_values)
            mom_std = np.std(mom_values)
            if mom_std > 0:
                for s in scores:
                    if s.mom_12m is not None:
                        s.z_momentum = (s.mom_12m - mom_mean) / mom_std
                        s.z_momentum = np.clip(s.z_momentum, -3, 3)

        # Low Volatility: NEGATIVE z-score (lower vol = higher score)
        vol_values = [s.vol_60d for s in scores if s.vol_60d is not None]
        if vol_values:
            vol_mean = np.mean(vol_values)
            vol_std = np.std(vol_values)
            if vol_std > 0:
                for s in scores:
                    if s.vol_60d is not None:
                        s.z_lowvol = -(s.vol_60d - vol_mean) / vol_std
                        s.z_lowvol = np.clip(s.z_lowvol, -3, 3)

        # Earnings Revision: POSITIVE z-score
        rev_values = [s.eps_change for s in scores if s.eps_change is not None]
        if rev_values:
            rev_mean = np.mean(rev_values)
            rev_std = np.std(rev_values)
            if rev_std > 0:
                for s in scores:
                    if s.eps_change is not None:
                        s.z_revision = (s.eps_change - rev_mean) / rev_std
                        s.z_revision = np.clip(s.z_revision, -3, 3)

    def get_sector(self, code: str) -> str:
        """Get sector17 code for a stock"""
        return self._sector_map.get(code, '10')  # Default: IT/Services

    def get_top_stocks(
        self,
        scores: List[FactorScores],
        n: int = 30,
        max_sector_weight: float = 0.25,
        max_per_sector: int = 5
    ) -> List[FactorScores]:
        """
        Select top N stocks with sector neutralization.
        """
        selected = []
        sector_counts = {}

        for score in scores:
            if len(selected) >= n:
                break

            sector = self.get_sector(score.code)
            current_count = sector_counts.get(sector, 0)

            # Enforce sector limits
            if current_count >= max_per_sector:
                continue

            selected.append(score)
            sector_counts[sector] = current_count + 1

        return selected


def generate_factor_csv(
    start_date: date,
    end_date: date,
    output_path: str,
    rebalance_frequency: int = 20
) -> None:
    """
    Generate factor data CSV for QuantConnect.

    Output format:
    date,code,close,volume,adt,pbr,roe,mom12,vol60,composite,rank,sector
    """
    logger.info(f"Generating factor CSV from {start_date} to {end_date}")

    model = MultiFactorModel()
    model.load_data(start_date, end_date)

    # Get all trading days
    all_dates = set()
    for df in model._price_data.values():
        all_dates.update(df.index.date)

    trading_days = sorted([d for d in all_dates if start_date <= d <= end_date])

    records = []
    last_compute = None
    cached_scores = []

    for current_date in trading_days:
        # Recompute factor scores periodically
        if last_compute is None or (current_date - last_compute).days >= rebalance_frequency:
            cached_scores = model.compute_factor_scores(current_date)
            last_compute = current_date
            logger.info(f"{current_date}: Computed {len(cached_scores)} factor scores")

        # Output top 50 stocks each day
        for score in cached_scores[:50]:
            records.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'code': score.code,
                'close': round(score.close, 2),
                'volume': int(score.volume),
                'adt': int(score.adt_20d),
                'pbr': round(score.pbr, 3) if score.pbr else '',
                'roe': round(score.roe, 2) if score.roe else '',
                'mom12': round(score.mom_12m, 4) if score.mom_12m else '',
                'vol60': round(score.vol_60d, 4) if score.vol_60d else '',
                'composite': round(score.composite_score, 4),
                'rank': score.rank,
                'sector': model.get_sector(score.code)
            })

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate factor data for backtesting
    generate_factor_csv(
        start_date=date(2015, 1, 1),
        end_date=date(2026, 2, 1),
        output_path="/Users/MBP/Desktop/Project_Asset_Shield/output/factor_data.csv",
        rebalance_frequency=20
    )
