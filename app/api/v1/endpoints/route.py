from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile

from app.core.image_bytes import is_valid_image_bytes
from app.engine.renovation_engine.vision_analysis import analyze_renovation_image_url
from app.schemas.requests.condition import ImageConditionRequest
from app.schemas.requests.renovation import RenovationEstimateRequest
from app.schemas.responses.condition import ImageConditionResult
from app.schemas.responses.renovation import RenovationEstimateResponse
from app.schemas.responses.stage_image import StageListingImageResponse
from app.services.listing_image_storage import (
    extract_property_id_from_listing_url,
    resolve_effective_property_id,
    stage_listing_image_for_renovation_async,
    stage_listing_image_from_bytes,
)
from app.services.renovation_service import build_renovation_estimate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/renovation")

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _staging_http_exception(exc: ValueError, *, image_url: str, property_id: str) -> HTTPException:
    effective_pid = (property_id or "").strip() or extract_property_id_from_listing_url(image_url)
    return HTTPException(
        status_code=422,
        detail={
            "code": "LISTING_IMAGE_STAGING_FAILED",
            "message": str(exc),
            "effective_property_id": effective_pid or None,
            "hint": (
                "Upload the photo with POST /renovation/stage-image (multipart) or "
                "POST /renovation/estimate-with-image, then use staged_source_image_url. "
                "Railway cannot download Cotality URLs when Incapsula blocks the server."
            ),
        },
    )


@router.post("/stage-image", response_model=StageListingImageResponse)
async def stage_listing_image(
    image_url: str = Form(..., description="Original listing URL (cache key; e.g. Cotality Trestle URL)."),
    property_id: str = Form(default="", description="Listing id (optional; parsed from Cotality URL if empty)."),
    source_image: UploadFile = File(..., description="JPEG/PNG/WebP file from browser or app."),
) -> StageListingImageResponse:
    """
    Demo-friendly: upload a photo file so Railway never calls Cotality.

    Returns a bucket URL to pass as ``image_url`` on ``POST /renovation/estimate``.
    """
    raw = await source_image.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"Image too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
            },
        )
    try:
        staged = stage_listing_image_from_bytes(
            image_url.strip(),
            raw,
            property_id=property_id.strip(),
        )
    except ValueError as exc:
        raise _staging_http_exception(exc, image_url=image_url, property_id=property_id) from exc

    pid = resolve_effective_property_id(property_id, image_url, flow="renovation")
    return StageListingImageResponse(
        staged_source_image_url=staged.url,
        effective_property_id=pid,
        storage_key=staged.storage_key,
        source=staged.source,
    )


@router.post("/stage-image/json", response_model=StageListingImageResponse)
async def stage_listing_image_json(
    image_url: str = Body(...),
    property_id: str = Body(default=""),
    source_image_base64: str = Body(...),
) -> StageListingImageResponse:
    """Stage via JSON base64 (same as estimate ``source_image_base64``)."""
    try:
        staged = await stage_listing_image_for_renovation_async(
            image_url.strip(),
            property_id=property_id.strip(),
            source_image_base64=source_image_base64,
        )
    except ValueError as exc:
        raise _staging_http_exception(exc, image_url=image_url, property_id=property_id) from exc

    pid = resolve_effective_property_id(property_id, image_url, flow="renovation")
    return StageListingImageResponse(
        staged_source_image_url=staged.url,
        effective_property_id=pid,
        storage_key=staged.storage_key,
        source=staged.source,
    )


@router.post("/estimate-with-image", response_model=RenovationEstimateResponse)
async def renovation_estimate_with_image(
    payload: str = Form(
        ...,
        description="JSON string: RenovationEstimateRequest fields (omit source_image_base64).",
    ),
    source_image: UploadFile = File(..., description="Listing photo file (required on Railway + Cotality)."),
) -> RenovationEstimateResponse:
    """
    One-shot demo: upload photo + estimate JSON in one multipart request.

    Use when Cotality blocks Railway and you have the image file locally or from the app.
    """
    try:
        request = RenovationEstimateRequest.model_validate(json.loads(payload))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": f"Invalid payload JSON: {exc}"},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": str(exc)},
        ) from exc

    raw = await source_image.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "Image too large (max 20MB)."},
        )
    if not is_valid_image_bytes(raw):
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "Uploaded file is not a valid image."},
        )

    request = request.model_copy(
        update={"source_image_base64": base64.b64encode(raw).decode("ascii")}
    )
    try:
        return await build_renovation_estimate(request)
    except ValueError as exc:
        raise _staging_http_exception(
            exc,
            image_url=request.image_url,
            property_id=request.property_id,
        ) from exc


@router.post("/estimate", response_model=RenovationEstimateResponse)
async def renovation_estimate(payload: RenovationEstimateRequest) -> RenovationEstimateResponse:
    try:
        return await build_renovation_estimate(payload)
    except ValueError as exc:
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
                    "Use POST /renovation/estimate-with-image (multipart file) or "
                    "/renovation/stage-image, then pass staged_source_image_url as image_url."
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
