from __future__ import annotations

from fastapi import APIRouter, HTTPException
from openai import APITimeoutError

from app.core.config import settings
from app.engine.image_condition.services.aggregator import aggregate
from app.engine.image_condition.services.image_filter import classify_and_filter
from app.engine.image_condition.services.vision_scorer import ImageDownloadError, score_from_images
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
                        "message": (
                            "No usable house/property photos found after filtering. "
                            "Floor plans, aerials, pools, and street views are excluded. "
                            "Verify image URLs are valid and publicly accessible."
                        ),
                    }
                ],
                "meta": {
                    "total_input": total_input,
                    "images_discarded": discarded_count,
                },
            },
        )

    try:
        vision_result = await score_from_images(selected)
    except ImageDownloadError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "IMAGE_DOWNLOAD_FAILED",
                "message": (
                    "Listing photos could not be downloaded from the server (HTTP 403 is common "
                    "for Cotality/CRMLS URLs). Re-host images on your own storage (S3/CDN) and "
                    "pass those URLs, or set IMAGE_DOWNLOAD_REFERER / IMAGE_DOWNLOAD_PROXY_TEMPLATE "
                    "if your MLS feed allows server access."
                ),
                "meta": {
                    "images_selected": exc.selected,
                    "images_prepared": exc.prepared,
                    "images_failed": exc.failed,
                },
            },
        ) from exc
    except APITimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "code": "VISION_TIMEOUT",
                "message": (
                    f"OpenAI vision timed out after {settings.OPENAI_CONDITION_SCORE_VISION_TIMEOUT_SECONDS}s. "
                    f"Try fewer images, lower CONDITION_SCORE_VISION_CHUNK_SIZE (current "
                    f"{settings.CONDITION_SCORE_VISION_CHUNK_SIZE}), or raise "
                    "OPENAI_CONDITION_SCORE_VISION_TIMEOUT_SECONDS."
                ),
                "meta": {
                    "images_selected": len(selected),
                    "chunk_size": settings.CONDITION_SCORE_VISION_CHUNK_SIZE,
                },
            },
        ) from exc
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
        images_analyzed=int(vision_result.get("images_prepared", len(final["room_scores"]))),
        images_discarded=discarded_count,
        cost_usd=final["cost_usd"],
    )

