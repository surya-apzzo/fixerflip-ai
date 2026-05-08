from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.engine.image_condition.services.aggregator import aggregate
from app.engine.image_condition.services.image_filter import classify_and_filter
from app.engine.image_condition.services.vision_scorer import score_from_images
from app.schemas.requests.property_condition import ConditionScoreRequest
from app.schemas.responses.property_condition import ConditionScoreResponse

router = APIRouter()


@router.post("/condition-score", response_model=ConditionScoreResponse)
async def condition_score(payload: ConditionScoreRequest) -> ConditionScoreResponse:
    filter_result = classify_and_filter(payload.image_urls)
    selected = filter_result["selected"]
    total_input = int(filter_result.get("total_input", len(payload.image_urls)))
    discarded_count = int(filter_result.get("discarded_count", 0))

    if total_input > 0 and not selected:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "errors": [
                    {
                        "field": "image_urls",
                        "message": "No usable property images found. Verify image URLs are valid and publicly accessible.",
                    }
                ],
                "meta": {
                    "total_input": total_input,
                    "images_discarded": discarded_count,
                },
            },
        )

    vision_result = await score_from_images(selected)
    final = aggregate(vision_result)

    return ConditionScoreResponse(
        property_id=payload.property_id,
        condition_score=final["condition_score"],
        grade=final["grade"],
        text_score=final["text_score"],
        vision_score=final["vision_score"],
        room_scores=final["room_scores"],
        positive_signals=final["positive_signals"],
        caution_signals=final["caution_signals"],
        red_flags=final["red_flags"],
        images_analyzed=len(selected),
        images_discarded=discarded_count,
        cost_usd=final["cost_usd"],
    )

