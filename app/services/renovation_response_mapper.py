from __future__ import annotations

from app.schemas.responses.renovation import RenovationEstimateResponse


def _needs_systems_review(tokens: list[str]) -> bool:
    lowered = [t.strip().lower() for t in tokens if t and t.strip()]
    risk_markers = (
        "fire damage",
        "smoke damage",
        "electrical",
        "plumbing",
        "water damage",
        "mold",
        "structural",
    )
    return any(any(marker in token for marker in risk_markers) for token in lowered)


def to_production_renovation_response(
    estimate,
    *,
    renovated_image_url: str | None = None,
) -> RenovationEstimateResponse:
    work_items = list(estimate.suggested_work_items)
    if _needs_systems_review([*estimate.assumptions, *work_items]):
        existing = {item.strip().lower() for item in work_items}
        if "possible systems review" not in existing:
            work_items.append("possible systems review")

    return RenovationEstimateResponse(
        renovation_class=estimate.renovation_class,
        estimated_renovation_range=f"${estimate.minimum_cost:,} - ${estimate.maximum_cost:,}",
        estimated_timeline=f"{estimate.minimum_timeline_weeks}-{estimate.maximum_timeline_weeks} weeks",
        suggested_work_items=work_items,
        confidence_score=f"{estimate.confidence_score}%",
        explanation_summary=estimate.explanation_summary,
        renovated_image_url=renovated_image_url,
    )
