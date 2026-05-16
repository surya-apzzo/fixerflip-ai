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
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LISTING_IMAGE_STAGING_FAILED",
                "message": str(exc),
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
