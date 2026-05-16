from fastapi import APIRouter, Body, HTTPException, Query

from app.engine.renovation_engine.vision_analysis import analyze_renovation_image_url
from app.schemas.requests.condition import ImageConditionRequest
from app.schemas.requests.renovation import RenovationEstimateRequest
from app.schemas.responses.condition import ImageConditionResult
from app.schemas.responses.renovation import RenovationEstimateResponse
from app.services.renovation_service import build_renovation_estimate

router = APIRouter(prefix="/renovation")


@router.post("/estimate", response_model=RenovationEstimateResponse)
async def renovation_estimate(payload: RenovationEstimateRequest) -> RenovationEstimateResponse:
    try:
        return await build_renovation_estimate(payload)
    except ValueError as exc:
        from app.services.listing_image_storage import extract_property_id_from_listing_url

        image_url = (payload.image_url or "").strip()
        effective_pid = (payload.property_id or "").strip() or extract_property_id_from_listing_url(
            image_url
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LISTING_IMAGE_STAGING_FAILED",
                "message": str(exc),
                "effective_property_id": effective_pid or None,
                "hint": (
                    "Railway cannot download Cotality URLs directly. "
                    "Send source_image_base64, pre-warm S3 from local, or use TRESTLE_HTTP_PROXY."
                ),
            },
        ) from exc



@router.post("/image-condition", response_model=ImageConditionResult)
async def image_condition(
    payload: ImageConditionRequest | None = Body(default=None),
    image_url: str | None = Query(default=None, description="Legacy query-param form (prefer JSON body)."),
) -> ImageConditionResult:
    resolved_url = (payload.image_url if payload is not None else "") or (image_url or "")
    resolved_url = resolved_url.strip()
    if not resolved_url:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "errors": [{"field": "image_url", "message": "image_url is required."}],
            },
        )
    return await analyze_renovation_image_url(resolved_url)
