from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, Request, UploadFile

from app.core.image_bytes import is_valid_image_bytes
from app.core.image_download import requires_client_image_upload
from app.engine.renovation_engine.vision_analysis import analyze_renovation_image_url
from app.schemas.requests.condition import ImageConditionRequest
from app.schemas.requests.renovation import RenovationEstimateRequest
from app.schemas.responses.condition import ImageConditionResult
from app.schemas.responses.renovation import RenovationEstimateResponse
from app.schemas.responses.stage_image import StageListingImageResponse
from app.services.listing_image_storage import (
    extract_property_id_from_listing_url,
    get_cached_listing_bytes_any_flow,
    is_own_storage_url,
    listing_image_storage_configured,
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
                "Attach the listing photo: POST /renovation/estimate as multipart with fields "
                "'payload' (JSON) + 'source_image' (file). Railway cannot download Cotality or "
                "Cloudflare-protected CDN URLs without an uploaded file."
            ),
        },
    )


async def _read_uploaded_image(source_image: UploadFile) -> bytes:
    raw = await source_image.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"Image too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
            },
        )
    if not is_valid_image_bytes(raw):
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": "Uploaded file is not a valid image."},
        )
    return raw


async def _estimate_from_multipart(
    *,
    payload: str,
    source_image: UploadFile,
) -> RenovationEstimateResponse:
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

    raw = await _read_uploaded_image(source_image)
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


@router.post("/stage-image", response_model=StageListingImageResponse)
async def stage_listing_image(
    image_url: str = Form(..., description="Original listing URL (cache key; e.g. Cotality Trestle URL)."),
    property_id: str = Form(default="", description="Listing id (optional; parsed from Cotality URL if empty)."),
    source_image: UploadFile = File(..., description="JPEG/PNG/WebP file from browser or app."),
) -> StageListingImageResponse:
    """Upload a photo to S3 so Railway never calls Cotality/CDN directly."""
    raw = await _read_uploaded_image(source_image)
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
    """Stage via JSON base64."""
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
    payload: str = Form(..., description="JSON RenovationEstimateRequest (omit source_image_base64)."),
    source_image: UploadFile = File(..., description="Listing photo file."),
) -> RenovationEstimateResponse:
    """Alias for multipart estimate (same as POST /estimate with multipart)."""
    return await _estimate_from_multipart(payload=payload, source_image=source_image)


@router.post("/estimate", response_model=RenovationEstimateResponse)
async def renovation_estimate(request: Request) -> RenovationEstimateResponse:
    """
    Renovation estimate.

    **JSON** (``Content-Type: application/json``): standard body; requires ``source_image_base64``
    on Railway when ``image_url`` is Cotality/MLS CDN.

    **Multipart** (``Content-Type: multipart/form-data``): fields ``payload`` (JSON string) +
    ``source_image`` (file) — **use this for demos on Railway**.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        payload_field = form.get("payload")
        source_field = form.get("source_image")
        if payload_field is None or source_field is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "Multipart request requires form fields: payload (JSON), source_image (file).",
                },
            )
        if not hasattr(source_field, "read"):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "VALIDATION_ERROR",
                    "message": "source_image must be a file upload, not plain text.",
                },
            )
        payload_str = payload_field if isinstance(payload_field, str) else str(payload_field)
        return await _estimate_from_multipart(payload=payload_str, source_image=source_field)

    try:
        body = await request.json()
        payload = RenovationEstimateRequest.model_validate(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": f"Invalid JSON body: {exc}"},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "VALIDATION_ERROR", "message": str(exc)},
        ) from exc

    image_url = (payload.image_url or "").strip()
    needs_client_bytes = (
        bool(image_url)
        and not is_own_storage_url(image_url)
        and requires_client_image_upload(image_url)
        and not (payload.source_image_base64 or "").strip()
    )
    if needs_client_bytes and listing_image_storage_configured():
        pid = resolve_effective_property_id(payload.property_id, image_url, flow="renovation")
        if get_cached_listing_bytes_any_flow(pid, image_url) is not None:
            needs_client_bytes = False
    if needs_client_bytes:
        effective_pid = (
            (payload.property_id or "").strip()
            or extract_property_id_from_listing_url(image_url)
            or None
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LISTING_IMAGE_UPLOAD_REQUIRED",
                "message": (
                    "This MLS/Cotality image_url cannot be downloaded on the server. "
                    "The mobile app must send the photo as source_image_base64 (or image_base64) "
                    "in the same JSON body, not only image_url."
                ),
                "effective_property_id": effective_pid,
                "mobile_fix": (
                    "1) Load image_url on the device. 2) Encode JPEG/PNG as base64. "
                    "3) POST JSON with image_url + source_image_base64 + property_id."
                ),
                "hint": (
                    "Node BFF: forward source_image_base64 from the app to POST /api/v1/renovation/estimate. "
                    "Or use multipart: payload + source_image file."
                ),
            },
        )

    try:
        return await build_renovation_estimate(payload)
    except ValueError as exc:
        raise _staging_http_exception(
            exc,
            image_url=payload.image_url,
            property_id=payload.property_id,
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
