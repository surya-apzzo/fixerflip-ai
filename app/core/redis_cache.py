from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from app.core.config import settings

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None

_REDIS_CLIENT = None
_IMAGE_MEMORY_CACHE: dict[str, tuple[bytes, str, datetime]] = {}
_IMAGE_CACHE_HIT_COUNTS: dict[str, int] = {"redis": 0, "memory": 0, "file": 0, "network": 0}
logger = logging.getLogger(__name__)


def get_redis_client():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if not settings.REDIS_URL or redis is None:
        return None
    try:
        _REDIS_CLIENT = redis.Redis.from_url(settings.REDIS_URL)
        return _REDIS_CLIENT
    except Exception:
        return None


def get_bytes(key: str) -> Optional[bytes]:
    client = get_redis_client()
    if client is None:
        return None
    try:
        value = client.get(key)
    except Exception:
        return None
    return bytes(value) if value else None


def set_bytes(key: str, value: bytes, ttl_seconds: int) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, value)
    except Exception:
        return


def get_text(key: str) -> Optional[str]:
    raw = get_bytes(key)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except Exception:
        return None


def set_text(key: str, value: str, ttl_seconds: int) -> None:
    set_bytes(key, value.encode("utf-8"), ttl_seconds)


def get_image_download_cache(url: str) -> Optional[tuple[bytes, str]]:
    base = _image_download_key_base(url)
    content = get_bytes(f"{base}:content")
    if not content:
        return None
    media_type = get_text(f"{base}:media_type") or "image/png"
    return content, media_type


def set_image_download_cache(url: str, content: bytes, media_type: str, ttl_seconds: int) -> None:
    base = _image_download_key_base(url)
    set_bytes(f"{base}:content", content, ttl_seconds)
    set_text(f"{base}:media_type", media_type, ttl_seconds)


def _image_download_key_base(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return f"renovation:image_download:{digest}"


def get_cached_image_download(
    url: str,
    *,
    ttl_seconds: int,
    cache_dir: Path,
) -> Optional[tuple[bytes, str]]:
    now = datetime.now(timezone.utc)

    redis_hit = get_image_download_cache(url)
    if redis_hit is not None:
        _record_image_cache_source("redis", url)
        content, media_type = redis_hit
        _IMAGE_MEMORY_CACHE[url] = (
            content,
            media_type,
            now + timedelta(seconds=ttl_seconds),
        )
        return content, media_type

    memory_hit = _IMAGE_MEMORY_CACHE.get(url)
    if memory_hit:
        content, media_type, expires_at = memory_hit
        if expires_at > now:
            _record_image_cache_source("memory", url)
            return content, media_type
        _IMAGE_MEMORY_CACHE.pop(url, None)

    file_hit = _read_file_cache(url, now=now, cache_dir=cache_dir)
    if file_hit is not None:
        _record_image_cache_source("file", url)
        content, media_type = file_hit
        _IMAGE_MEMORY_CACHE[url] = (
            content,
            media_type,
            now + timedelta(seconds=ttl_seconds),
        )
        return content, media_type

    return None


def set_cached_image_download(
    url: str,
    *,
    content: bytes,
    media_type: str,
    ttl_seconds: int,
    cache_dir: Path,
) -> None:
    now = datetime.now(timezone.utc)
    _record_image_cache_source("network", url)
    _IMAGE_MEMORY_CACHE[url] = (
        content,
        media_type,
        now + timedelta(seconds=ttl_seconds),
    )
    set_image_download_cache(url, content, media_type, ttl_seconds)
    _write_file_cache(url, content=content, media_type=media_type, now=now, cache_dir=cache_dir)


def _cache_base_path(image_url: str, *, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    return cache_dir / key


def _read_file_cache(image_url: str, *, now: datetime, cache_dir: Path) -> Optional[tuple[bytes, str]]:
    base = _cache_base_path(image_url, cache_dir=cache_dir)
    bin_path = base.with_suffix(".bin")
    meta_path = base.with_suffix(".meta")
    if not (bin_path.exists() and meta_path.exists()):
        return None
    try:
        raw = meta_path.read_text(encoding="utf-8").strip().split("|", 1)
        expires_at = datetime.fromisoformat(raw[0])
        media_type = raw[1] if len(raw) == 2 and raw[1] else "image/png"
    except Exception:
        return None
    if expires_at <= now:
        return None
    try:
        return bin_path.read_bytes(), media_type
    except Exception:
        return None


def _write_file_cache(
    image_url: str,
    *,
    content: bytes,
    media_type: str,
    now: datetime,
    cache_dir: Path,
) -> None:
    base = _cache_base_path(image_url, cache_dir=cache_dir)
    bin_path = base.with_suffix(".bin")
    meta_path = base.with_suffix(".meta")
    expires_at = now + timedelta(seconds=settings.REDIS_CACHE_TTL_SECONDS)
    try:
        bin_path.write_bytes(content)
        meta_path.write_text(f"{expires_at.isoformat()}|{media_type}", encoding="utf-8")
    except Exception:
        return


def _record_image_cache_source(source: str, image_url: str) -> None:
    _IMAGE_CACHE_HIT_COUNTS[source] = _IMAGE_CACHE_HIT_COUNTS.get(source, 0) + 1
    logger.info(
        "image-download-cache source=%s url_hash=%s hits=%s",
        source,
        hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:12],
        _IMAGE_CACHE_HIT_COUNTS[source],
    )
