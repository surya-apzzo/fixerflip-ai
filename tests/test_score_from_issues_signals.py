from app.engine.renovation_engine.score_from_issues import (
    compute_gap_score,
    compute_renovation_age_detection,
)


def test_gap_score_bucket_and_points():
    result = compute_gap_score(
        listing_price=300000,
        living_area_sqft=1500,
        avg_area_price_per_sqft=300,
    )
    assert result.gap_bucket in {"HIGH", "MEDIUM_HIGH", "MEDIUM", "LOW", "OVERPRICED"}
    assert -10 <= result.score_points <= 35


def test_gap_score_overpriced_signal_is_negative():
    result = compute_gap_score(
        listing_price=480000,
        living_area_sqft=1200,
        avg_area_price_per_sqft=300,
    )
    assert result.gap_bucket == "OVERPRICED"
    assert result.score_points < 0


def test_age_detection_returns_probability_and_drivers():
    result = compute_renovation_age_detection(
        year_built=1980,
        years_since_last_sale=18,
        permit_years_since_last=15,
    )
    assert result.fixer_probability in {"HIGH", "MEDIUM_HIGH", "LOW"}
    assert result.score_points >= 0
    assert len(result.drivers) > 0
