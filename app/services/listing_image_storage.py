"""Cache MLS listing photos in S3 when Cotality/Trestle blocks cloud/datacenter IPs."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import re
from dataclasses import dataclass

from app.core.config import settings
from app.core.image_bytes import guess_image_media_type, is_valid_image_bytes
from app.core.image_download import ImageDownloadFlow, download_listing_image_with_meta, image_download_config_summary
from app.core.trestle_auth import is_trestle_media_url, trestle_credentials_configured
from app.services.storage_service import (
    _make_s3_client,
    _require_storage_config,
    _upload_bytes_to_bucket,
    public_url_for_object_key,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ListingImageResolveResult:
    content: bytes | None
    source: str = "none"
    waf_blocked: bool = False
    cached_public_url: str | None = None


@dataclass(slots=True)
class ListingImageStageResult:
    """Listing photo staged on our bucket for renovation vision/edit."""

    url: str
    source: str
    storage_key: str | None = None


_TRESTLE_PROPERTY_ID_RE = re.compile(r"/Media/Property/PHOTO-Jpeg/(\d+)/", re.IGNORECASE)


def listing_image_storage_configured() -> bool:
    try:
        _require_storage_config()
        return True
    except ValueError:
        return False


def _cache_prefix(flow: ImageDownloadFlow) -> str:
    if flow == "renovation":
        return (settings.STORAGE_RENOVATION_LISTING_IMAGE_PREFIX or "renovation/listings").strip("/")
    return (settings.STORAGE_CONDITION_SCORE_IMAGE_PREFIX or "condition-score/listings").strip("/")


def extract_property_id_from_listing_url(url: str) -> str:
    """Best-effort MLS id from Cotality Trestle media paths."""
    match = _TRESTLE_PROPERTY_ID_RE.search(url or "")
    return match.group(1) if match else ""


def resolve_effective_property_id(property_id: str, source_url: str, *, flow: ImageDownloadFlow) -> str:
    explicit = re.sub(r"[^a-zA-Z0-9._-]+", "_", (property_id or "").strip())[:80]
    if explicit:
        return explicit
    from_url = extract_property_id_from_listing_url(source_url)
    if from_url:
        return from_url
    return "renovation" if flow == "renovation" else "unknown"


def _cache_key(property_id: str, source_url: str, *, flow: ImageDownloadFlow) -> str:
    digest = hashlib.sha256(source_url.strip().encode("utf-8")).hexdigest()[:32]
    safe_id = resolve_effective_property_id(property_id, source_url, flow=flow)
    return f"{_cache_prefix(flow)}/{safe_id}/{digest}.jpg"


def public_url_to_storage_key(url: str) -> str | None:
    cleaned = (url or "").strip()
    if not cleaned:
        return None
    public_base = (settings.STORAGE_PUBLIC_BASE_URL or "").strip().rstrip("/")
    if public_base and cleaned.startswith(public_base + "/"):
        return cleaned[len(public_base) + 1 :]
    endpoint = (settings.STORAGE_ENDPOINT_URL or "").strip().rstrip("/")
    bucket = (settings.STORAGE_BUCKET_NAME or "").strip()
    if endpoint and bucket:
        prefix = f"{endpoint}/{bucket}/"
        if cleaned.startswith(prefix):
            return cleaned[len(prefix) :]
    return None


def is_own_storage_url(url: str) -> bool:
    return public_url_to_storage_key(url) is not None


def _get_bytes_by_key(key: str) -> bytes | None:
    try:
        _require_storage_config()
    except ValueError:
        return None
    try:
        client = _make_s3_client()
        obj = client.get_object(Bucket=settings.STORAGE_BUCKET_NAME, Key=key)
        body = obj.get("Body")
        if body is None:
            return None
        data = body.read()
        if not is_valid_image_bytes(data):
            logger.warning("S3 object is not a valid image (key=%s, bytes=%s)", key, len(data or b""))
            return None
        return data
    except Exception as exc:
        logger.debug("S3 get_object failed for key=%s: %s", key, exc)
        return None


def get_cached_listing_bytes(property_id: str, source_url: str, *, flow: ImageDownloadFlow) -> bytes | None:
    return _get_bytes_by_key(_cache_key(property_id, source_url, flow=flow))


def get_cached_listing_bytes_any_flow(
    property_id: str,
    source_url: str,
) -> tuple[bytes, ImageDownloadFlow, str] | None:
    """S3 cache lookup across renovation + condition-score prefixes (same URL hash)."""
    for flow in ("renovation", "condition_score"):
        key = _cache_key(property_id, source_url, flow=flow)
        data = _get_bytes_by_key(key)
        if data is not None:
            return data, flow, key
    return None


def put_listing_image_cache(
    property_id: str,
    source_url: str,
    image_bytes: bytes,
    *,
    flow: ImageDownloadFlow,
) -> str | None:
    if not listing_image_storage_configured() or not image_bytes:
        return None
    key = _cache_key(property_id, source_url, flow=flow)
    try:
        _upload_bytes_to_bucket(image_bytes, key=key, content_type="image/jpeg")
        return public_url_for_object_key(key)
    except Exception as exc:
        logger.warning("Listing image S3 cache upload failed key=%s: %s", key, exc)
        return None


def _decode_source_image_base64(raw: str) -> bytes:
    cleaned = (raw or "").strip()
    if not cleaned:
        raise ValueError("source_image_base64 is empty.")
    if cleaned.startswith("data:") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    try:
        data = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("source_image_base64 is not valid base64.") from exc
    if not is_valid_image_bytes(data):
        raise ValueError("source_image_base64 is not a valid image.")
    return data


def stage_listing_image_for_renovation(
    source_url: str,
    *,
    property_id: str = "",
    source_image_base64: str = "",
) -> ListingImageStageResult:
    """
    Stage a listing photo on S3 and return a bucket URL for vision + image edit.

    Order: already on our bucket → S3 cache hit → client base64 upload → HTTP download + upload.
    """
    cleaned = (source_url or "").strip()
    if not cleaned and not (source_image_base64 or "").strip():
        raise ValueError("image_url or source_image_base64 is required.")

    if not listing_image_storage_configured():
        raise ValueError(
            "STORAGE_* is not configured. Set bucket credentials so listing photos can be staged "
            "before renovation (required when Cotality/Incapsula blocks Railway)."
        )

    cache_id = resolve_effective_property_id(property_id, cleaned, flow="renovation")

    if cleaned and is_own_storage_url(cleaned):
        key = public_url_to_storage_key(cleaned)
        return ListingImageStageResult(url=cleaned, source="already_staged", storage_key=key)

    if cleaned:
        cache_hit = get_cached_listing_bytes_any_flow(cache_id, cleaned)
        if cache_hit is not None:
            cached_bytes, hit_flow, hit_key = cache_hit
            if hit_flow == "renovation":
                cache_key = hit_key
                staged_url = public_url_for_object_key(cache_key)
                source = "s3_cache"
            else:
                staged_url = put_listing_image_cache(
                    cache_id, cleaned, cached_bytes, flow="renovation"
                )
                cache_key = _cache_key(cache_id, cleaned, flow="renovation")
                source = "s3_cache_shared"
                if not staged_url:
                    staged_url = public_url_for_object_key(hit_key)
                    cache_key = hit_key
                    source = "s3_cache"
            logger.info(
                "Renovation listing image S3 cache hit property_id=%s flow=%s key=%s",
                cache_id,
                hit_flow,
                cache_key,
            )
            return ListingImageStageResult(
                url=staged_url,
                source=source,
                storage_key=cache_key,
            )

    if (source_image_base64 or "").strip():
        if not cleaned:
            raise ValueError("image_url is required with source_image_base64 (used as cache key).")
        image_bytes = _decode_source_image_base64(source_image_base64)
        staged_url = put_listing_image_cache(cache_id, cleaned, image_bytes, flow="renovation")
        if not staged_url:
            raise ValueError("Failed to upload source_image_base64 to listing image cache.")
        cache_key = _cache_key(cache_id, cleaned, flow="renovation")
        logger.info(
            "Renovation listing image staged from client base64 property_id=%s key=%s bytes=%s",
            cache_id,
            cache_key,
            len(image_bytes),
        )
        return ListingImageStageResult(
            url=staged_url,
            source="client_base64",
            storage_key=cache_key,
        )

    if not cleaned:
        raise ValueError("image_url is required.")

    resolved = resolve_listing_image_bytes(cleaned, property_id=cache_id, flow="renovation")
    if resolved.content is not None and is_valid_image_bytes(resolved.content):
        cache_key = _cache_key(cache_id, cleaned, flow="renovation")
        staged_url = resolved.cached_public_url or public_url_for_object_key(cache_key)
        logger.info(
            "Renovation listing image staged after download property_id=%s source=%s key=%s",
            cache_id,
            resolved.source,
            cache_key,
        )
        return ListingImageStageResult(
            url=staged_url,
            source="uploaded" if resolved.source == "download" else resolved.source,
            storage_key=cache_key,
        )

    raise ValueError(
        renovation_listing_staging_error_message(
            resolved,
            image_url=cleaned,
            property_id=cache_id,
        )
    )


def renovation_listing_staging_error_message(
    resolved: ListingImageResolveResult,
    *,
    image_url: str,
    property_id: str,
) -> str:
    base = renovation_listing_download_error_message(resolved, image_url=image_url)
    pid = (property_id or "").strip() or extract_property_id_from_listing_url(image_url) or "unknown"
    public_base = (settings.STORAGE_PUBLIC_BASE_URL or "").strip() or "(set STORAGE_PUBLIC_BASE_URL)"
    return (
        f"{base} "
        f"Effective property_id={pid}. "
        f"No S3 cache found for this image_url yet. "
        f"On Railway, Cotality blocks server download — use one of: "
        f"(1) POST with source_image_base64 from your app, "
        f"(2) run once locally with the same STORAGE_* to warm cache at "
        f"renovation/listings/{pid}/<hash>.jpg, "
        f"(3) set TRESTLE_HTTP_PROXY, "
        f"(4) pass image_url already on {public_base}. "
        f"Referer/proxy env vars do not bypass Incapsula when the OAuth token endpoint is blocked."
    )


async def stage_listing_image_for_renovation_async(
    source_url: str,
    *,
    property_id: str = "",
    source_image_base64: str = "",
) -> ListingImageStageResult:
    return await asyncio.to_thread(
        stage_listing_image_for_renovation,
        source_url,
        property_id=property_id,
        source_image_base64=source_image_base64,
    )


def resolve_listing_image_bytes(
    source_url: str,
    *,
    property_id: str = "",
    flow: ImageDownloadFlow,
) -> ListingImageResolveResult:
    """
    Resolve bytes for one listing photo:
    1) STORAGE_PUBLIC_BASE_URL object (if URL is already on our bucket)
    2) S3 cache for (property_id, source_url) under flow-specific prefix
    3) HTTP download with Trestle Bearer when configured (no public proxy for Trestle URLs)
    4) On success, write S3 cache for future requests from blocked IPs
    """
    cleaned = (source_url or "").strip()
    if not cleaned:
        return ListingImageResolveResult(content=None, source="empty")

    cache_id = resolve_effective_property_id(property_id, cleaned, flow=flow)

    own_key = public_url_to_storage_key(cleaned)
    if own_key:
        data = _get_bytes_by_key(own_key)
        if data is not None:
            return ListingImageResolveResult(content=data, source="storage_url")

    if listing_image_storage_configured():
        cached = get_cached_listing_bytes(cache_id, cleaned, flow=flow)
        if cached is not None:
            logger.info(
                "%s listing image cache hit property_id=%s url=%s",
                flow,
                cache_id,
                cleaned[:120],
            )
            return ListingImageResolveResult(content=cached, source="s3_cache")

    outcome = download_listing_image_with_meta(cleaned, flow=flow)
    if outcome.content is not None and is_valid_image_bytes(outcome.content):
        cached_url = None
        if listing_image_storage_configured():
            cached_url = put_listing_image_cache(cache_id, cleaned, outcome.content, flow=flow)
        return ListingImageResolveResult(
            content=outcome.content,
            source="download",
            cached_public_url=cached_url,
        )

    return ListingImageResolveResult(
        content=None,
        source="download_failed",
        waf_blocked=outcome.waf_blocked,
    )


async def resolve_listing_image_bytes_async(
    source_url: str,
    *,
    property_id: str = "",
    flow: ImageDownloadFlow,
) -> ListingImageResolveResult:
    return await asyncio.to_thread(
        resolve_listing_image_bytes,
        source_url,
        property_id=property_id,
        flow=flow,
    )


def renovation_listing_download_error_message(
    resolved: ListingImageResolveResult,
    *,
    image_url: str,
) -> str:
    """User-facing hint when renovation vision/edit cannot load a Cotality/Trestle photo."""
    if resolved.waf_blocked or (
        is_trestle_media_url(image_url) and trestle_credentials_configured()
    ):
        return (
            "Cotality blocked listing photo download from this server (Incapsula WAF / no Trestle token). "
            "Send property_id so we can use S3-cached photos, set TRESTLE_HTTP_PROXY, use image_url on "
            f"{settings.STORAGE_PUBLIC_BASE_URL or 'your STORAGE_PUBLIC_BASE_URL'}, or ask Cotality to whitelist egress IP. "
            f"Config: {image_download_config_summary('renovation')}"
        )
    return (
        "Listing photo URL could not be downloaded for renovation. "
        "Use a public CDN URL, configure RENOVATION_IMAGE_DOWNLOAD_REFERER, or pass property_id after "
        "photos were cached from a network Cotality allows."
    )
