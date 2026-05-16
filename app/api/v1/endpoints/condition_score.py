from __future__ import annotations

from fastapi import APIRouter, HTTPException
from openai import APITimeoutError

from app.core.config import settings
from app.core.image_download import image_download_config_summary
from app.engine.image_condition.services.aggregator import aggregate
from app.engine.image_condition.services.image_filter import (
    classify_and_filter_urls,
    clip_available,
    deduplicate_filtered_by_room_type,
)
from app.engine.image_condition.services.vision_scorer import ImageDownloadError, score_from_images
from app.schemas.requests.property_condition import ConditionScoreRequest
from app.schemas.responses.property_condition import ConditionScoreResponse

router = APIRouter()


@router.post("/condition-score", response_model=ConditionScoreResponse)
async def condition_score(payload: ConditionScoreRequest) -> ConditionScoreResponse:
    image_urls = [u.strip() for u in payload.image_urls if u and u.strip()]
    filter_result = classify_and_filter_urls(image_urls, property_id=payload.property_id)
    selected = filter_result["selected"]
    urls_received = int(filter_result.get("urls_received", len(image_urls)))
    urls_processed = int(filter_result.get("urls_processed", len(image_urls)))
    urls_truncated = int(filter_result.get("urls_truncated", 0))
    total_input = int(filter_result.get("total_input", urls_processed))
    discarded_count = int(filter_result.get("discarded_count", 0))
    download_failures = int(filter_result.get("download_failures", 0))
    waf_blocked = bool(filter_result.get("waf_blocked", False))
    clip_loaded = bool(filter_result.get("clip_available", clip_available()))

    images_after_filter = len(selected)
    selected, images_deduplicated = deduplicate_filtered_by_room_type(selected)
    images_after_dedupe = len(selected)

    if total_input > 0 and not selected:
        if waf_blocked and download_failures >= total_input:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "COTALITY_WAF_BLOCKED",
                    "message": (
                        "Cotality blocked listing photo and OAuth requests from this server's IP "
                        "(Incapsula WAF). Trestle credentials are set but the token endpoint returned 403 HTML. "
                        "Fix options: (1) pass image_urls on your STORAGE_PUBLIC_BASE_URL after uploading photos "
                        "from a network Cotality allows, (2) set TRESTLE_HTTP_PROXY to an allowed egress proxy, "
                        "(3) ask Cotality to whitelist your Railway/static egress IP."
                    ),
                    "meta": {
                        "total_input": total_input,
                        "images_discarded": discarded_count,
                        "download_failures": download_failures,
                        "download_config": image_download_config_summary("condition_score"),
                    },
                },
            )
        if download_failures >= total_input:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "IMAGE_DOWNLOAD_FAILED",
                    "message": (
                        "Could not download listing photo URLs. For Cotality/Trestle "
                        "(api.cotality.com/trestle/Media/...) set TRESTLE_CLIENT_ID, "
                        "TRESTLE_CLIENT_SECRET, and TRESTLE_BASE_URL, or use URLs on your S3/CDN "
                        "(STORAGE_PUBLIC_BASE_URL)."
                    ),
                    "meta": {
                        "total_input": total_input,
                        "images_discarded": discarded_count,
                        "download_failures": download_failures,
                        "download_config": image_download_config_summary("condition_score"),
                    },
                },
            )
        if images_after_filter > 0 and images_after_dedupe == 0:
            message = (
                "Photos passed download/filter but none were kept for vision. "
                "Redeploy the latest API build (dedupe fallback when CLIP is missing). "
                "Ensure S3 cache objects are real JPEGs, not HTML error pages."
            )
        elif not clip_loaded and images_after_filter == 0:
            message = (
                "No listing photos could be classified. CLIP is not installed on this server and "
                "cached/downloaded bytes may be invalid. Rebuild Docker with torch+CLIP, or warm S3 "
                "cache from local using the same STORAGE_* and property_id."
            )
        else:
            message = (
                "No usable house/property photos found after filtering. "
                "Floor plans, aerials, pools, and street views are excluded, or photos could not "
                "be downloaded/decoded."
            )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "errors": [{"field": "image_urls", "message": message}],
                "meta": {
                    "total_input": total_input,
                    "images_discarded": discarded_count,
                    "download_failures": download_failures,
                    "images_after_filter": images_after_filter,
                    "images_after_dedupe": images_after_dedupe,
                    "images_deduplicated": images_deduplicated,
                    "clip_available": clip_loaded,
                    "discard_invalid_bytes": int(filter_result.get("discard_invalid_bytes", 0)),
                    "discard_clip_non_property": int(filter_result.get("discard_clip_non_property", 0)),
                    "discard_clip_weak_house": int(filter_result.get("discard_clip_weak_house", 0)),
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
        urls_received=urls_received,
        urls_processed=urls_processed,
        urls_truncated=urls_truncated,
        cost_usd=final["cost_usd"],
    )
