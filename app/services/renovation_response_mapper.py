from __future__ import annotations

from app.schemas.responses.renovation import RenovationEstimateResponse


def build_renovation_estimate_response(
    estimate,
    *,
    room_type: str = "unknown",
    renovated_image_url: str | None = None,
) -> RenovationEstimateResponse:
    """Build the renovation estimate response from the estimate."""
    rt = (room_type or "").strip() or "unknown"
    return RenovationEstimateResponse(
        renovation_class=estimate.renovation_class,
        estimated_renovation_range=f"${estimate.minimum_cost:,} - ${estimate.maximum_cost:,}",
        estimated_timeline=f"{estimate.minimum_timeline_weeks}-{estimate.maximum_timeline_weeks} weeks",
        suggested_work_items=list(estimate.suggested_work_items),
        confidence_score=f"{estimate.confidence_score}%",
        explanation_summary=estimate.explanation_summary,
        room_type=rt,
        renovated_image_url=renovated_image_url,
    )
