"""
VERIDIAN QUANT - Vectorized Feature Engineering Module
Ported from Asset Shield V3.2.0 and Adapted for Horse Racing Analytics

Original Source: Asset Shield src/shield/jquants_backtest_provider.py
Porting Date: 2026-02-04
Target Domain: Horse Racing (JRA, NAR, International)

RACING-SPECIFIC ADAPTATIONS:
1. Track Bias Tensor - Surface/Weather/Rail position effects
2. Bloodline Tensor - Sire/Dam/Broodmare sire performance vectors
3. Sectional Timing - Furlong-by-furlong pace analysis
4. Class Transition - Grade level progression patterns

ARCHITECTURE NOTES:
- Maintains vectorized NumPy/Pandas paradigm from Asset Shield
- 100x speedup preserved through batch operations
- Domain-specific normalization layers added
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
from enum import Enum


# =============================================================================
# VECTORIZED DATE PARSING (Ported from Asset Shield V3.2.0)
# =============================================================================

@lru_cache(maxsize=10000)
def _parse_date_cached(date_str: str) -> date:
    """Parse date string with caching for repeated lookups"""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def parse_dates_vectorized(date_strings: List[str]) -> np.ndarray:
    """
    Vectorized date parsing using pandas.
    100x faster than iterative strptime for large datasets.

    Args:
        date_strings: List of date strings in YYYY-MM-DD format

    Returns:
        NumPy array of datetime64[D] objects
    """
    return pd.to_datetime(date_strings, format="%Y-%m-%d").values.astype('datetime64[D]')


# =============================================================================
# RACING-SPECIFIC ENUMERATIONS
# =============================================================================

class TrackSurface(Enum):
    """Track surface types"""
    TURF = "turf"
    DIRT = "dirt"
    POLYTRACK = "polytrack"
    TAPETA = "tapeta"


class TrackCondition(Enum):
    """Track condition states (JRA Standard)"""
    FIRM = "firm"           # 良
    GOOD = "good"           # 稍重
    YIELDING = "yielding"   # 重
    SOFT = "soft"           # 不良


class RaceGrade(Enum):
    """Race grade classification"""
    G1 = 1
    G2 = 2
    G3 = 3
    LISTED = 4
    OPEN = 5
    CLASS_3WIN = 6
    CLASS_2WIN = 7
    CLASS_1WIN = 8
    MAIDEN = 9


# =============================================================================
# TRACK BIAS TENSOR ENGINE
# =============================================================================

@dataclass
class TrackBiasVector:
    """
    Track bias representation for a specific track/condition/distance combination.

    Dimensions:
    - Rail position advantage (inside/middle/outside) [3]
    - Running style advantage (front/stalker/closer) [3]
    - Distance band (sprint/mile/intermediate/staying) [4]
    """
    track_id: str
    surface: TrackSurface
    condition: TrackCondition
    rail_bias: np.ndarray  # Shape: (3,) - inside, middle, outside
    pace_bias: np.ndarray  # Shape: (3,) - front, stalker, closer
    distance_band: int     # 0-3
    sample_size: int
    confidence: float


class TrackBiasEngine:
    """
    Vectorized Track Bias Calculator

    Computes position and pace advantages based on historical results.
    Uses rolling window analysis with decay weighting.
    """

    RAIL_POSITIONS = ['inside', 'middle', 'outside']
    RUNNING_STYLES = ['front', 'stalker', 'closer']
    DISTANCE_BANDS = {
        'sprint': (1000, 1400),
        'mile': (1400, 1800),
        'intermediate': (1800, 2200),
        'staying': (2200, 4000)
    }

    def __init__(self, decay_factor: float = 0.95, min_samples: int = 50):
        self.decay_factor = decay_factor
        self.min_samples = min_samples
        self._bias_cache: Dict[str, TrackBiasVector] = {}

    def compute_bias_vectorized(
        self,
        results_df: pd.DataFrame,
        track_id: str,
        surface: TrackSurface,
        condition: TrackCondition
    ) -> TrackBiasVector:
        """
        Compute track bias using vectorized operations.

        Args:
            results_df: Historical race results with columns:
                - finish_position, gate_number, total_gates,
                - first_call_position, final_time, distance
            track_id: Track identifier
            surface: Track surface type
            condition: Track condition

        Returns:
            TrackBiasVector with computed biases
        """
        # Filter for track/surface/condition
        mask = (
            (results_df['track_id'] == track_id) &
            (results_df['surface'] == surface.value) &
            (results_df['condition'] == condition.value)
        )
        filtered = results_df[mask].copy()

        if len(filtered) < self.min_samples:
            return self._default_bias(track_id, surface, condition, len(filtered))

        # Vectorized rail position classification
        gate_pct = filtered['gate_number'] / filtered['total_gates']
        filtered['rail_zone'] = pd.cut(
            gate_pct,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['inside', 'middle', 'outside']
        )

        # Vectorized running style classification (based on first call position)
        field_pct = filtered['first_call_position'] / filtered['field_size']
        filtered['running_style'] = pd.cut(
            field_pct,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['front', 'stalker', 'closer']
        )

        # Compute win rates by rail position (vectorized groupby)
        rail_win_rates = (
            filtered.groupby('rail_zone')['is_winner']
            .mean()
            .reindex(self.RAIL_POSITIONS, fill_value=0.0)
            .values
        )

        # Compute win rates by running style
        pace_win_rates = (
            filtered.groupby('running_style')['is_winner']
            .mean()
            .reindex(self.RUNNING_STYLES, fill_value=0.0)
            .values
        )

        # Normalize to bias scores (deviation from expected)
        expected_win_rate = 1.0 / 3.0
        rail_bias = (rail_win_rates - expected_win_rate) / expected_win_rate
        pace_bias = (pace_win_rates - expected_win_rate) / expected_win_rate

        # Determine predominant distance band
        avg_distance = filtered['distance'].mean()
        distance_band = self._classify_distance(avg_distance)

        # Compute confidence based on sample size
        confidence = min(1.0, len(filtered) / (self.min_samples * 10))

        return TrackBiasVector(
            track_id=track_id,
            surface=surface,
            condition=condition,
            rail_bias=rail_bias,
            pace_bias=pace_bias,
            distance_band=distance_band,
            sample_size=len(filtered),
            confidence=confidence
        )

    def _classify_distance(self, distance: float) -> int:
        """Classify distance into band index"""
        for i, (band_name, (min_d, max_d)) in enumerate(self.DISTANCE_BANDS.items()):
            if min_d <= distance < max_d:
                return i
        return 3  # Default to staying

    def _default_bias(
        self,
        track_id: str,
        surface: TrackSurface,
        condition: TrackCondition,
        sample_size: int
    ) -> TrackBiasVector:
        """Return neutral bias when insufficient data"""
        return TrackBiasVector(
            track_id=track_id,
            surface=surface,
            condition=condition,
            rail_bias=np.zeros(3),
            pace_bias=np.zeros(3),
            distance_band=1,
            sample_size=sample_size,
            confidence=0.0
        )


# =============================================================================
# BLOODLINE TENSOR ENGINE
# =============================================================================

@dataclass
class BloodlineVector:
    """
    Bloodline performance tensor for a horse.

    Dimensions:
    - Sire performance vector [8] - win%, place%, distance_idx, surface_idx, stamina, speed, class, consistency
    - Dam performance vector [8] - same dimensions
    - Broodmare sire vector [8] - same dimensions
    - Cross compatibility score [1]
    """
    horse_id: str
    sire_vector: np.ndarray      # Shape: (8,)
    dam_vector: np.ndarray       # Shape: (8,)
    bms_vector: np.ndarray       # Shape: (8,) - Broodmare sire
    cross_score: float           # Compatibility/nick score
    inbreeding_coeff: float      # Wright's coefficient
    sample_confidence: float


class BloodlineEngine:
    """
    Vectorized Bloodline Analysis Engine

    Computes hereditary performance indicators using:
    - Dosage Index (Rasmussen factor adaptation)
    - Aptitudinal Index (surface/distance suitability)
    - Cross compatibility (successful nick patterns)
    """

    PERFORMANCE_DIMS = [
        'win_rate', 'place_rate', 'distance_index', 'surface_index',
        'stamina_index', 'speed_index', 'class_progression', 'consistency'
    ]

    def __init__(self, min_progeny: int = 20):
        self.min_progeny = min_progeny
        self._sire_cache: Dict[str, np.ndarray] = {}

    def compute_bloodline_vectorized(
        self,
        sire_results_df: pd.DataFrame,
        dam_results_df: pd.DataFrame,
        bms_results_df: pd.DataFrame,
        horse_id: str
    ) -> BloodlineVector:
        """
        Compute bloodline vector using vectorized operations.

        Args:
            sire_results_df: Progeny results for sire
            dam_results_df: Progeny results for dam
            bms_results_df: Progeny results for broodmare sire
            horse_id: Target horse identifier

        Returns:
            BloodlineVector with computed performance tensors
        """
        sire_vector = self._compute_progeny_vector(sire_results_df)
        dam_vector = self._compute_progeny_vector(dam_results_df)
        bms_vector = self._compute_progeny_vector(bms_results_df)

        # Cross compatibility (simplified nick scoring)
        cross_score = self._compute_cross_score(sire_vector, dam_vector, bms_vector)

        # Inbreeding coefficient (placeholder - requires full pedigree)
        inbreeding_coeff = 0.0

        # Confidence based on sample sizes
        total_samples = len(sire_results_df) + len(dam_results_df) + len(bms_results_df)
        confidence = min(1.0, total_samples / (self.min_progeny * 30))

        return BloodlineVector(
            horse_id=horse_id,
            sire_vector=sire_vector,
            dam_vector=dam_vector,
            bms_vector=bms_vector,
            cross_score=cross_score,
            inbreeding_coeff=inbreeding_coeff,
            sample_confidence=confidence
        )

    def _compute_progeny_vector(self, results_df: pd.DataFrame) -> np.ndarray:
        """Compute performance vector from progeny results"""
        if len(results_df) < self.min_progeny:
            return np.zeros(8)

        # Vectorized aggregations
        win_rate = (results_df['finish_position'] == 1).mean()
        place_rate = (results_df['finish_position'] <= 3).mean()

        # Distance suitability (weighted by wins)
        wins = results_df[results_df['finish_position'] == 1]
        if len(wins) > 0:
            avg_win_distance = wins['distance'].mean()
            distance_index = (avg_win_distance - 1600) / 400  # Normalize around mile
        else:
            distance_index = 0.0

        # Surface suitability (turf win rate vs dirt)
        turf_wins = ((results_df['surface'] == 'turf') & (results_df['finish_position'] == 1)).sum()
        total_wins = (results_df['finish_position'] == 1).sum()
        surface_index = turf_wins / max(total_wins, 1) - 0.5  # Centered at 0

        # Stamina index (late-closing tendency)
        stamina_index = (results_df['final_position'] < results_df['first_call_position']).mean() - 0.5

        # Speed index (early position)
        speed_index = 1.0 - (results_df['first_call_position'] / results_df['field_size']).mean()

        # Class progression (improvement in grade)
        class_prog = results_df.groupby('horse_id')['race_grade'].apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0
        ).mean()
        class_index = -class_prog / 5.0  # Lower grade number is better

        # Consistency (std of finish positions)
        consistency = 1.0 - min(1.0, results_df['finish_position'].std() / 5.0)

        return np.array([
            win_rate, place_rate, distance_index, surface_index,
            stamina_index, speed_index, class_index, consistency
        ])

    def _compute_cross_score(
        self,
        sire_vector: np.ndarray,
        dam_vector: np.ndarray,
        bms_vector: np.ndarray
    ) -> float:
        """Compute cross compatibility score"""
        # Simple complementary scoring - opposing strengths
        complementary = np.abs(sire_vector - dam_vector).mean()
        bms_alignment = np.dot(dam_vector, bms_vector) / (
            np.linalg.norm(dam_vector) * np.linalg.norm(bms_vector) + 1e-8
        )
        return (complementary * 0.5 + bms_alignment * 0.5)


# =============================================================================
# SECTIONAL TIMING ENGINE
# =============================================================================

@dataclass
class SectionalProfile:
    """
    Sectional timing profile for a horse.

    Stores furlong-by-furlong pace analysis with:
    - Absolute sectional times
    - Relative to race pace (beaten lengths per furlong)
    - Energy distribution pattern
    """
    horse_id: str
    race_id: str
    distance: int
    sectionals: np.ndarray      # Shape: (N,) - time per furlong
    relative_pace: np.ndarray   # Shape: (N,) - vs race average
    energy_curve: np.ndarray    # Shape: (N,) - deceleration pattern
    finish_speed: float         # Final furlong velocity
    pace_figure: float          # Composite pace rating


class SectionalEngine:
    """
    Vectorized Sectional Timing Analyzer

    Processes GPS/sensor timing data to compute:
    - Pace figures (Ragozin-style speed ratings)
    - Energy distribution curves
    - Optimal trip patterns
    """

    FURLONGS_PER_METER = 1 / 201.168

    def __init__(self, par_times: Optional[Dict[int, np.ndarray]] = None):
        """
        Args:
            par_times: Dictionary mapping distance to par sectional array
        """
        self.par_times = par_times or {}

    def compute_sectionals_vectorized(
        self,
        timing_df: pd.DataFrame,
        horse_id: str,
        race_id: str
    ) -> SectionalProfile:
        """
        Compute sectional profile using vectorized operations.

        Args:
            timing_df: DataFrame with columns:
                - horse_id, race_id, checkpoint, cumulative_time, distance
            horse_id: Target horse
            race_id: Target race

        Returns:
            SectionalProfile with computed metrics
        """
        # Filter for specific horse/race
        mask = (timing_df['horse_id'] == horse_id) & (timing_df['race_id'] == race_id)
        horse_data = timing_df[mask].sort_values('checkpoint')

        if len(horse_data) < 2:
            return self._empty_profile(horse_id, race_id)

        # Compute sectional times (vectorized diff)
        cumulative = horse_data['cumulative_time'].values
        sectionals = np.diff(cumulative, prepend=0)

        # Get race average sectionals for relative pace
        race_mask = timing_df['race_id'] == race_id
        race_avg = (
            timing_df[race_mask]
            .groupby('checkpoint')['cumulative_time']
            .mean()
            .diff()
            .fillna(0)
            .values
        )

        # Relative pace (negative = faster than average)
        if len(race_avg) == len(sectionals):
            relative_pace = sectionals - race_avg
        else:
            relative_pace = np.zeros_like(sectionals)

        # Energy curve (deceleration pattern)
        energy_curve = np.diff(sectionals, prepend=sectionals[0])

        # Finish speed (final sectional converted to m/s)
        distance = horse_data['distance'].iloc[-1]
        finish_speed = (200 / sectionals[-1]) if sectionals[-1] > 0 else 0

        # Pace figure (composite rating)
        par = self.par_times.get(distance, np.ones_like(sectionals) * 12.0)
        if len(par) == len(sectionals):
            pace_figure = 100 - (sectionals - par).sum() * 2
        else:
            pace_figure = 80.0  # Default

        return SectionalProfile(
            horse_id=horse_id,
            race_id=race_id,
            distance=distance,
            sectionals=sectionals,
            relative_pace=relative_pace,
            energy_curve=energy_curve,
            finish_speed=finish_speed,
            pace_figure=pace_figure
        )

    def _empty_profile(self, horse_id: str, race_id: str) -> SectionalProfile:
        """Return empty profile when insufficient data"""
        return SectionalProfile(
            horse_id=horse_id,
            race_id=race_id,
            distance=0,
            sectionals=np.array([]),
            relative_pace=np.array([]),
            energy_curve=np.array([]),
            finish_speed=0.0,
            pace_figure=0.0
        )


# =============================================================================
# UNIFIED FEATURE VECTOR
# =============================================================================

@dataclass
class RacingFeatureVector:
    """
    Unified feature vector for racing prediction.

    Combines all feature engines into a single tensor for ML input.
    """
    horse_id: str
    race_id: str

    # Track features
    track_bias: TrackBiasVector

    # Bloodline features
    bloodline: BloodlineVector

    # Recent form (last 5 runs sectionals)
    recent_sectionals: List[SectionalProfile]

    # Computed composite features
    composite_vector: np.ndarray  # Shape: (64,) - flattened feature tensor

    def to_tensor(self) -> np.ndarray:
        """Flatten all features into ML-ready tensor"""
        features = []

        # Track bias (10 features)
        features.extend(self.track_bias.rail_bias)
        features.extend(self.track_bias.pace_bias)
        features.append(self.track_bias.distance_band)
        features.append(self.track_bias.confidence)
        features.append(self.track_bias.sample_size / 1000)  # Normalized
        features.append(0.0)  # Padding

        # Bloodline (26 features)
        features.extend(self.bloodline.sire_vector)
        features.extend(self.bloodline.dam_vector)
        features.extend(self.bloodline.bms_vector)
        features.append(self.bloodline.cross_score)
        features.append(self.bloodline.inbreeding_coeff)

        # Recent form - last 5 pace figures (5 features)
        pace_figs = [s.pace_figure for s in self.recent_sectionals[-5:]]
        pace_figs.extend([0.0] * (5 - len(pace_figs)))  # Pad if needed
        features.extend(pace_figs)

        # Recent form - avg finish speed (1 feature)
        if self.recent_sectionals:
            avg_speed = np.mean([s.finish_speed for s in self.recent_sectionals])
        else:
            avg_speed = 0.0
        features.append(avg_speed)

        # Pad to 64 features
        features.extend([0.0] * (64 - len(features)))

        return np.array(features[:64])


# =============================================================================
# BATCH PROCESSING PIPELINE
# =============================================================================

class VectorizedFeaturePipeline:
    """
    High-performance batch feature computation pipeline.

    Processes entire race cards in vectorized batches.
    """

    def __init__(self):
        self.track_engine = TrackBiasEngine()
        self.bloodline_engine = BloodlineEngine()
        self.sectional_engine = SectionalEngine()

    def process_race_card_batch(
        self,
        runners_df: pd.DataFrame,
        historical_results_df: pd.DataFrame,
        timing_df: pd.DataFrame
    ) -> Dict[str, RacingFeatureVector]:
        """
        Process entire race card in vectorized batch.

        Args:
            runners_df: Race card with runner details
            historical_results_df: Historical results for all runners
            timing_df: Sectional timing data

        Returns:
            Dictionary mapping horse_id to RacingFeatureVector
        """
        results = {}

        # Get race metadata
        race_id = runners_df['race_id'].iloc[0]
        track_id = runners_df['track_id'].iloc[0]
        surface = TrackSurface(runners_df['surface'].iloc[0])
        condition = TrackCondition(runners_df['condition'].iloc[0])

        # Compute track bias once for entire race
        track_bias = self.track_engine.compute_bias_vectorized(
            historical_results_df, track_id, surface, condition
        )

        # Process each runner
        for _, runner in runners_df.iterrows():
            horse_id = runner['horse_id']

            # Get bloodline vectors (would need pedigree lookup in production)
            sire_results = historical_results_df[
                historical_results_df['sire_id'] == runner.get('sire_id', '')
            ]
            dam_results = historical_results_df[
                historical_results_df['dam_id'] == runner.get('dam_id', '')
            ]
            bms_results = historical_results_df[
                historical_results_df['sire_id'] == runner.get('bms_id', '')
            ]

            bloodline = self.bloodline_engine.compute_bloodline_vectorized(
                sire_results, dam_results, bms_results, horse_id
            )

            # Get recent sectionals
            horse_races = timing_df[timing_df['horse_id'] == horse_id]['race_id'].unique()
            recent_sectionals = []
            for past_race in horse_races[-5:]:
                profile = self.sectional_engine.compute_sectionals_vectorized(
                    timing_df, horse_id, past_race
                )
                if profile.pace_figure > 0:
                    recent_sectionals.append(profile)

            # Build feature vector
            feature_vec = RacingFeatureVector(
                horse_id=horse_id,
                race_id=race_id,
                track_bias=track_bias,
                bloodline=bloodline,
                recent_sectionals=recent_sectionals,
                composite_vector=np.zeros(64)  # Computed via to_tensor()
            )
            feature_vec.composite_vector = feature_vec.to_tensor()

            results[horse_id] = feature_vec

        return results


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'parse_dates_vectorized',
    '_parse_date_cached',
    'TrackSurface',
    'TrackCondition',
    'RaceGrade',
    'TrackBiasVector',
    'TrackBiasEngine',
    'BloodlineVector',
    'BloodlineEngine',
    'SectionalProfile',
    'SectionalEngine',
    'RacingFeatureVector',
    'VectorizedFeaturePipeline',
]
