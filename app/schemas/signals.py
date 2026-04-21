from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GapScoreResult:
    subject_price_per_sqft: float = 0.0
    avg_area_price_per_sqft: float = 0.0
    gap_pct: float = 0.0
    gap_bucket: str = "LOW"
    score_points: int = 0


@dataclass(frozen=True)
class AgeDetectionResult:
    years_since_last_sale: Optional[int] = None
    permit_years_since_last: Optional[int] = None
    fixer_probability: str = "LOW"
    score_points: int = 0
    drivers: tuple[str, ...] = ()
