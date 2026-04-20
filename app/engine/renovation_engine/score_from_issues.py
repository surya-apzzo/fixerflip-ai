"""
Renovation scoring helpers.

Implements two key "fixer likelihood" signals:
1) Price-per-sqft gap score vs nearby comps average
2) Renovation age detection using year built + years since last sale + permit recency

These scores are intended to feed the Renovation / FlipScore engines.
"""

from __future__ import annotations

from typing import Optional

from app.engine.renovation_engine.schemas import AgeDetectionResult, GapScoreResult


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_gap_score(
    *,
    listing_price: float,
    living_area_sqft: float,
    avg_area_price_per_sqft: float,
) -> GapScoreResult:
    """
    Parameters:
    - listing_price: The price of the property
    - living_area_sqft: The square footage of the property
    - avg_area_price_per_sqft: The average price per square foot of the property in the area

    Returns:
    - GapScoreResult: A GapScoreResult object containing the subject price per square foot, average area price per square foot, gap percentage, gap bucket, and score points
    """
    if listing_price <= 0 or living_area_sqft <= 0 or avg_area_price_per_sqft <= 0:
        return GapScoreResult(
            subject_price_per_sqft=0.0,
            avg_area_price_per_sqft=max(avg_area_price_per_sqft, 0.0),
            gap_pct=0.0,
            gap_bucket="LOW",
            score_points=0,
        )

    subject_ppsf = listing_price / living_area_sqft
    gap = (avg_area_price_per_sqft - subject_ppsf) / avg_area_price_per_sqft
    gap = _clamp(gap, -1.0, 1.0)

    # Bucket + suggested points (tunable).
    if gap <= -0.10:
        bucket, pts = "OVERPRICED", -10
    elif gap > -0.10 and gap <= 0:
        bucket, pts = "OVERPRICED", -5
    elif gap >= 0.30:
        bucket, pts = "HIGH", 35
    elif gap >= 0.20:
        bucket, pts = "MEDIUM_HIGH", 25
    elif gap >= 0.10:
        bucket, pts = "MEDIUM", 15
    else:
        bucket, pts = "LOW", 5

    return GapScoreResult(
        subject_price_per_sqft=subject_ppsf,
        avg_area_price_per_sqft=avg_area_price_per_sqft,
        gap_pct=gap,
        gap_bucket=bucket,
        score_points=pts,
    )


def compute_renovation_age_detection(
    *,
    year_built: Optional[int],
    years_since_last_sale: Optional[int] = None,
    permit_years_since_last: Optional[int] = None,
) -> AgeDetectionResult:
    """Compute renovation age/fixer signal using sale recency + build + permits."""
    drivers: list[str] = []
    sale_recency_years = years_since_last_sale

    pts = _age_points_from_last_sale(sale_recency_years, drivers)
    pts = _apply_year_built_modifier(pts, year_built, drivers)
    pts = _apply_permit_modifier(pts, permit_years_since_last, drivers)

    pts = int(_clamp(float(pts), 0.0, 35.0))
    prob = _bucket_probability_from_points(pts)
    prob, drivers = _apply_strong_fixer_override(
        prob,
        year_built,
        sale_recency_years,
        drivers,
    )

    return AgeDetectionResult(
        years_since_last_sale=years_since_last_sale,
        permit_years_since_last=permit_years_since_last,
        fixer_probability=prob,
        score_points=pts,
        drivers=tuple(drivers),
    )


def _age_points_from_last_sale(
    sale_recency_years: Optional[int], drivers: list[str]
) -> int:
    if sale_recency_years is None:
        drivers.append("Last sale age unknown.")
        return 8
    if sale_recency_years >= 20:
        drivers.append(f"Long-term ownership (~{sale_recency_years}y since last sale).")
        return 25
    if sale_recency_years >= 15:
        drivers.append(f"Last sale >15y (~{sale_recency_years}y).")
        return 20
    if sale_recency_years >= 10:
        drivers.append(f"Last sale 10–15y (~{sale_recency_years}y).")
        return 12
    if sale_recency_years >= 3:
        drivers.append(f"Last sale within 3–10y (~{sale_recency_years}y).")
        return 5
    drivers.append(f"Recent sale (~{sale_recency_years}y) suggests recent flip.")
    return 0


def _apply_year_built_modifier(
    pts: int, year_built: Optional[int], drivers: list[str]
) -> int:
    if year_built is not None and year_built < 1990:
        drivers.append(f"Older build ({year_built}) increases fixer likelihood.")
        return pts + 5
    return pts


def _apply_permit_modifier(
    pts: int, permit_years_since_last: Optional[int], drivers: list[str]
) -> int:
    p = permit_years_since_last
    if p is None:
        return pts
    if p <= 5:
        drivers.append("Recent permits (<5y) suggest recent updates.")
        return pts - 8
    if p <= 10:
        drivers.append("Some permit activity (5–10y).")
        return pts - 3
    drivers.append("No recent permits (>10y).")
    return pts + 5


def _bucket_probability_from_points(pts: int) -> str:
    if pts >= 22:
        return "HIGH"
    if pts >= 12:
        return "MEDIUM_HIGH"
    return "LOW"


def _apply_strong_fixer_override(
    prob: str,
    year_built: Optional[int],
    years_since_last_sale: Optional[int],
    drivers: list[str],
) -> tuple[str, list[str]]:
    if (year_built is not None and year_built < 1990) and (
        years_since_last_sale is not None and years_since_last_sale >= 15
    ):
        drivers.append("Strong fixer signal: built <1990 and last sale >=15y.")
        return "HIGH", drivers
    return prob, drivers
