"""Cache listing photos in S3 so condition-score works when Cotality blocks cloud IPs."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from app.core.config import settings
from app.core.image_download import ImageDownloadFlow, download_listing_image_with_meta
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


def _cache_prefix() -> str:
    return (settings.STORAGE_CONDITION_SCORE_IMAGE_PREFIX or "condition-score/listings").strip("/")


def _cache_key(property_id: str, source_url: str) -> str:
    digest = hashlib.sha256(source_url.strip().encode("utf-8")).hexdigest()[:32]
    safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", (property_id or "unknown").strip())[:80] or "unknown"
    return f"{_cache_prefix()}/{safe_id}/{digest}.jpg"


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
        return data if data and len(data) > 256 else None
    except Exception as exc:
        logger.debug("S3 get_object failed for key=%s: %s", key, exc)
        return None


def get_cached_listing_bytes(property_id: str, source_url: str) -> bytes | None:
    return _get_bytes_by_key(_cache_key(property_id, source_url))


def put_listing_image_cache(property_id: str, source_url: str, image_bytes: bytes) -> str | None:
    if not listing_image_storage_configured() or not image_bytes:
        return None
    key = _cache_key(property_id, source_url)
    try:
        return _upload_bytes_to_bucket(image_bytes, key=key, content_type="image/jpeg")
    except Exception as exc:
        logger.warning("Listing image S3 cache upload failed key=%s: %s", key, exc)
        return None


def resolve_listing_image_bytes(
    source_url: str,
    *,
    property_id: str,
    flow: ImageDownloadFlow,
) -> ListingImageResolveResult:
    """
  Resolve bytes for one listing photo:
  1) Our STORAGE_PUBLIC_BASE_URL object
  2) S3 cache for (property_id, source_url)
  3) HTTP download (Trestle Bearer when configured)
  4) On success, write S3 cache for future Railway requests
    """
    cleaned = (source_url or "").strip()
    if not cleaned:
        return ListingImageResolveResult(content=None, source="empty")

    own_key = public_url_to_storage_key(cleaned)
    if own_key:
        data = _get_bytes_by_key(own_key)
        if data is not None:
            return ListingImageResolveResult(content=data, source="storage_url")
        logger.warning("Own storage URL could not be read from bucket key=%s", own_key)

    if listing_image_storage_configured() and property_id:
        cached = get_cached_listing_bytes(property_id, cleaned)
        if cached is not None:
            logger.info(
                "condition-score listing image cache hit property_id=%s url=%s",
                property_id,
                cleaned[:120],
            )
            return ListingImageResolveResult(content=cached, source="s3_cache")

    outcome = download_listing_image_with_meta(cleaned, flow=flow)
    if outcome.content is not None:
        cached_url = None
        if listing_image_storage_configured() and property_id:
            cached_url = put_listing_image_cache(property_id, cleaned, outcome.content)
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
