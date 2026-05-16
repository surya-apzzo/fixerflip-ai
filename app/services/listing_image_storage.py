"""Cache MLS listing photos in S3 when Cotality/Trestle blocks cloud/datacenter IPs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass

from app.core.config import settings
from app.core.image_bytes import guess_image_media_type, is_valid_image_bytes
from app.core.image_download import ImageDownloadFlow, download_listing_image_with_meta, image_download_config_summary
from app.core.trestle_auth import is_trestle_media_url, trestle_credentials_configured
from app.services.storage_service import _make_s3_client, _require_storage_config, _upload_bytes_to_bucket

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ListingImageResolveResult:
    content: bytes | None
    source: str = "none"
    waf_blocked: bool = False
    cached_public_url: str | None = None


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


def _normalize_property_id(property_id: str, *, flow: ImageDownloadFlow) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (property_id or "").strip())[:80]
    if safe:
        return safe
    return "renovation" if flow == "renovation" else "unknown"


def _cache_key(property_id: str, source_url: str, *, flow: ImageDownloadFlow) -> str:
    digest = hashlib.sha256(source_url.strip().encode("utf-8")).hexdigest()[:32]
    safe_id = _normalize_property_id(property_id, flow=flow)
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
        return _upload_bytes_to_bucket(image_bytes, key=key, content_type="image/jpeg")
    except Exception as exc:
        logger.warning("Listing image S3 cache upload failed key=%s: %s", key, exc)
        return None


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

    cache_id = _normalize_property_id(property_id, flow=flow)

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
