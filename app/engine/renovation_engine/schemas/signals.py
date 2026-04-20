from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GapScoreResult:
    subject_price_per_sqft: float = 0.0
    avg_area_price_per_sqft: float = 0.0
    gap_pct: float = 0.0  # 0.30 == 30% undervalued vs area avg (positive = cheaper)
    gap_bucket: str = "LOW"  # HIGH / MEDIUM_HIGH / MEDIUM / LOW
    score_points: int = 0  # recommended points for use in 0–100 scoring


@dataclass(frozen=True)
class AgeDetectionResult:
    years_since_last_sale: Optional[int] = None
    permit_years_since_last: Optional[int] = None
    fixer_probability: str = "LOW"  # HIGH / MEDIUM_HIGH / LOW
    score_points: int = 0
    drivers: tuple[str, ...] = ()
