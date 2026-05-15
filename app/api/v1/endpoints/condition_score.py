from __future__ import annotations

from fastapi import APIRouter, HTTPException
from openai import APITimeoutError

from app.core.config import settings
from app.engine.image_condition.services.aggregator import aggregate
from app.engine.image_condition.services.image_filter import classify_and_filter_inputs
from app.engine.image_condition.services.vision_image_payload import decode_image_base64_field
from app.engine.image_condition.services.vision_scorer import ImageDownloadError, score_from_images
from app.schemas.requests.property_condition import ConditionScoreRequest
from app.schemas.responses.property_condition import ConditionScoreResponse

router = APIRouter()


def _build_filter_inputs(payload: ConditionScoreRequest) -> list[tuple[str | None, bytes | None]]:
    if payload.images:
        items: list[tuple[str | None, bytes | None]] = []
        for img in payload.images:
            url = (img.url or "").strip() or None
            b64 = (img.base64 or "").strip()
            image_bytes = decode_image_base64_field(b64) if b64 else None
            items.append((url, image_bytes))
        return items

    return [(url.strip(), None) for url in payload.image_urls if url and url.strip()]


@router.post("/condition-score", response_model=ConditionScoreResponse)
async def condition_score(payload: ConditionScoreRequest) -> ConditionScoreResponse:
    filter_items = _build_filter_inputs(payload)
    filter_result = classify_and_filter_inputs(filter_items)
    selected = filter_result["selected"]
    total_input = int(filter_result.get("total_input", len(filter_items)))
    discarded_count = int(filter_result.get("discarded_count", 0))
    uses_embedded_bytes = any(b is not None for _u, b in filter_items)

    if total_input > 0 and not selected:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "errors": [
                    {
                        "field": "images" if uses_embedded_bytes else "image_urls",
                        "message": (
                            "No usable house/property photos found after filtering. "
                            "Floor plans, aerials, pools, and street views are excluded."
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
                    "Listing photo URLs could not be downloaded (Cotality/CRMLS often return HTTP 403). "
                    "Fix: (1) Send images[].base64 from your frontend (browser can load MLS photos), "
                    "(2) re-host on your S3/CDN and pass public URLs, or (3) remove IMAGE_DOWNLOAD_PROXY_TEMPLATE "
                    "(weserv.nl returns 404 for these URLs). Referer-only access rarely works for api.cotality.com."
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
