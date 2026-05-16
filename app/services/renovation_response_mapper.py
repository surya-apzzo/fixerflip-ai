from __future__ import annotations

from app.schemas.responses.renovation import RenovationEstimateResponse


def build_renovation_estimate_response(
    estimate,
    *,
    room_type: str = "unknown",
    condition_score: int,
    renovated_image_url: str | None = None,
    staged_source_image_url: str | None = None,
    warnings: list[str] | None = None,
) -> RenovationEstimateResponse:
    """Build the renovation estimate response from the estimate."""
    rt = (room_type or "").strip() or "unknown"
    preview_url = (renovated_image_url or "").strip() or None
    staged_url = (staged_source_image_url or "").strip() or None
    preview_available = bool(
        preview_url
        and preview_url != staged_url
        and "renovated-images/" in preview_url.lower()
    )
    return RenovationEstimateResponse(
        renovation_class=estimate.renovation_class,
        estimated_renovation_range=f"${estimate.minimum_cost:,} - ${estimate.maximum_cost:,}",
        estimated_timeline=f"{estimate.minimum_timeline_weeks}-{estimate.maximum_timeline_weeks} weeks",
        suggested_work_items=list(estimate.suggested_work_items),
        confidence_score=f"{estimate.confidence_score}%",
        explanation_summary=estimate.explanation_summary,
        room_type=rt,
        condition_score=condition_score,
        renovated_image_url=preview_url if preview_available else None,
        staged_source_image_url=staged_url,
        renovation_preview_available=preview_available,
        warnings=list(warnings or []),
    )
