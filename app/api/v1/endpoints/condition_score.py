from __future__ import annotations

from fastapi import APIRouter, HTTPException
from openai import APITimeoutError

from app.core.config import settings
from app.core.image_download import image_download_config_summary
from app.engine.image_condition.services.aggregator import aggregate
from app.engine.image_condition.services.image_filter import (
    classify_and_filter_urls,
    deduplicate_filtered_by_room_type,
)
from app.engine.image_condition.services.vision_scorer import ImageDownloadError, score_from_images
from app.schemas.requests.property_condition import ConditionScoreRequest
from app.schemas.responses.property_condition import ConditionScoreResponse

router = APIRouter()


@router.post("/condition-score", response_model=ConditionScoreResponse)
async def condition_score(payload: ConditionScoreRequest) -> ConditionScoreResponse:
    image_urls = [u.strip() for u in payload.image_urls if u and u.strip()]
    filter_result = classify_and_filter_urls(image_urls)
    selected = filter_result["selected"]
    total_input = int(filter_result.get("total_input", len(image_urls)))
    discarded_count = int(filter_result.get("discarded_count", 0))
    download_failures = int(filter_result.get("download_failures", 0))

    images_after_filter = len(selected)
    selected, images_deduplicated = deduplicate_filtered_by_room_type(selected)

    if total_input > 0 and not selected:
        if download_failures >= total_input:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "IMAGE_DOWNLOAD_FAILED",
                    "message": (
                        "Could not download listing photo URLs. For Cotality/Trestle "
                        "(api.cotality.com/trestle/Media/...) set TRESTLE_CLIENT_ID, "
                        "TRESTLE_CLIENT_SECRET, and TRESTLE_BASE_URL so downloads use OAuth Bearer "
                        "auth; otherwise re-host photos on your S3/CDN."
                    ),
                    "meta": {
                        "total_input": total_input,
                        "images_discarded": discarded_count,
                        "download_failures": download_failures,
                        "download_config": image_download_config_summary("condition_score"),
                    },
                },
            )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "errors": [
                    {
                        "field": "image_urls",
                        "message": (
                            "No usable house/property photos found after filtering. "
                            "Floor plans, aerials, pools, and street views are excluded, or CLIP could not "
                            "classify the photos as interior/exterior house shots."
                        ),
                    }
                ],
                "meta": {
                    "total_input": total_input,
                    "images_discarded": discarded_count,
                    "download_failures": download_failures,
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
                    "Listing photo bytes could not be prepared for vision after download. "
                    "Check that image_urls return valid image content from this server."
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
        images_after_filter=images_after_filter,
        images_deduplicated=images_deduplicated,
        cost_usd=final["cost_usd"],
    )
