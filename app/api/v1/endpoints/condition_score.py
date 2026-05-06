from __future__ import annotations

from fastapi import APIRouter

from app.image_condition.services.aggregator import aggregate
from app.image_condition.services.image_filter import classify_and_filter
from app.image_condition.services.vision_scorer import score_from_images
from app.schemas.requests.property_condition import ConditionScoreRequest
from app.schemas.responses.property_condition import ConditionScoreResponse

router = APIRouter()


@router.post("/condition-score", response_model=ConditionScoreResponse)
async def condition_score(payload: ConditionScoreRequest) -> ConditionScoreResponse:
    filter_result = classify_and_filter(payload.image_urls)
    selected = filter_result["selected"]
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
        images_discarded=int(filter_result["discarded_count"]),
        cost_usd=final["cost_usd"],
    )

