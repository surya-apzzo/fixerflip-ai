"""Trestle (Cotality) OAuth2 client-credentials token for authenticated media downloads."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import settings
from app.core.cotality_waf import is_incapsula_waf_response

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cached_token: str | None = None
_cached_expires_at: float = 0.0
_TOKEN_REFRESH_BUFFER_SECONDS = 120.0


def trestle_credentials_configured() -> bool:
    return bool(
        (settings.TRESTLE_CLIENT_ID or "").strip()
        and (settings.TRESTLE_CLIENT_SECRET or "").strip()
        and (settings.TRESTLE_BASE_URL or "").strip()
    )


def is_trestle_media_url(image_url: str) -> bool:
    """True for api.cotality.com /trestle/Media URLs that require Bearer auth."""
    parsed = urlparse((image_url or "").strip())
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower().replace("\\", "/")
    if "cotality.com" in host and "/trestle/" in path:
        return True
    if "/trestle/media" in path:
        return True
    return False


def trestle_token_url() -> str:
    base = (settings.TRESTLE_BASE_URL or "").strip().rstrip("/")
    path = (settings.TRESTLE_TOKEN_PATH or "/trestle/oidc/connect/token").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def clear_trestle_token_cache() -> None:
    global _cached_token, _cached_expires_at
    with _lock:
        _cached_token = None
        _cached_expires_at = 0.0


def _fetch_trestle_access_token() -> tuple[str, float]:
    token_url = trestle_token_url()
    data = {
        "client_id": settings.TRESTLE_CLIENT_ID.strip(),
        "client_secret": settings.TRESTLE_CLIENT_SECRET.strip(),
        "grant_type": "client_credentials",
        "scope": (settings.TRESTLE_TOKEN_SCOPE or "api").strip() or "api",
    }
    proxy = (settings.TRESTLE_HTTP_PROXY or "").strip() or None
    response = httpx.post(
        token_url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=30.0,
        proxy=proxy,
    )
    status_code = int(getattr(response, "status_code", 0) or 0)
    if status_code >= 400 and is_incapsula_waf_response(response):
        raise httpx.HTTPStatusError(
            "Cotality Incapsula WAF blocked Trestle token endpoint (datacenter IP?). "
            "Stage listing photos to your S3 bucket or set TRESTLE_HTTP_PROXY / ask Cotality to whitelist egress IP.",
            request=response.request,
            response=response,
        )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise ValueError("Trestle token response missing access_token.")
    expires_in = float(payload.get("expires_in") or 28800)
    expires_at = time.time() + max(60.0, expires_in - _TOKEN_REFRESH_BUFFER_SECONDS)
    return token, expires_at


def get_trestle_access_token(*, force_refresh: bool = False) -> str | None:
    """
    Return a cached Bearer token, or fetch via POST /trestle/oidc/connect/token.
    Returns None when Trestle credentials are not configured.
    """
    if not trestle_credentials_configured():
        return None

    global _cached_token, _cached_expires_at
    now = time.time()
    if not force_refresh:
        with _lock:
            if _cached_token and _cached_expires_at > now:
                return _cached_token

    try:
        token, expires_at = _fetch_trestle_access_token()
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and is_incapsula_waf_response(exc.response):
            logger.warning(
                "Trestle token blocked by Cotality Incapsula WAF url=%s (not invalid credentials). "
                "Use S3-cached listing URLs, TRESTLE_HTTP_PROXY, or Cotality IP whitelist.",
                trestle_token_url(),
            )
        else:
            logger.warning("Trestle token request failed url=%s error=%s", trestle_token_url(), exc)
        return None
    except Exception as exc:
        logger.warning(
            "Trestle token request failed url=%s error=%s",
            trestle_token_url(),
            exc,
        )
        return None

    with _lock:
        _cached_token = token
        _cached_expires_at = expires_at
    logger.info(
        "Trestle access token refreshed (expires_in ~%ss, cached until %s)",
        int(expires_at - now),
        int(expires_at),
    )
    return token
