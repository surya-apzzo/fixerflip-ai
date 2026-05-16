"""MLS/CDN listing image download helpers (separate referer/proxy per API flow)."""

from __future__ import annotations

import logging
from typing import Any, Literal
from urllib.parse import quote, urlparse

import httpx

from app.core.config import settings
from app.core.trestle_auth import (
    clear_trestle_token_cache,
    get_trestle_access_token,
    is_trestle_media_url,
    trestle_credentials_configured,
)

logger = logging.getLogger(__name__)

_RESPONSE_BODY_SNIPPET_LIMIT = 500

ImageDownloadFlow = Literal["renovation", "condition_score"]

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_BUILTIN_PROXY_TEMPLATES: tuple[str, ...] = (
    "https://images.weserv.nl/?url={url_no_scheme_encoded}",
    "https://wsrv.nl/?url={url_full_encoded}",
)

# Hosts that reject generic server clients unless Referer matches the portal.
_MLS_HOST_REFERERS: dict[str, str] = {
    "crmls.org": "https://www.crmls.org/",
    "mlslistings.com": "https://www.mlslistings.com/",
    "realty.dev": "https://www.realty.dev/",
    "cotality.com": "https://www.cotality.com/",
}

_FLOW_REFERER_ENV: dict[ImageDownloadFlow, str] = {
    "renovation": "RENOVATION_IMAGE_DOWNLOAD_REFERER",
    "condition_score": "CONDITION_SCORE_IMAGE_DOWNLOAD_REFERER",
}

_FLOW_PROXY_ENV: dict[ImageDownloadFlow, str] = {
    "renovation": "RENOVATION_IMAGE_DOWNLOAD_PROXY_TEMPLATE",
    "condition_score": "CONDITION_SCORE_IMAGE_DOWNLOAD_PROXY_TEMPLATE",
}


def _referer_for_image_host(host: str) -> str | None:
    normalized = (host or "").lower().removeprefix("www.")
    if not normalized:
        return None
    for domain, referer in _MLS_HOST_REFERERS.items():
        if normalized == domain or normalized.endswith(f".{domain}"):
            return referer
    return None


def _referer_for_image_url(image_url: str) -> str | None:
    """CRMLS assets on third-party CDNs (e.g. imagecdn.realty.dev) need crmls.org Referer."""
    parsed = urlparse(image_url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower().replace("\\", "/")
    if "crmls" in host or "/crmls/" in path or "mls_photos/crmls" in path:
        return _MLS_HOST_REFERERS["crmls.org"]
    if "cotality.com" in host or "/trestle/" in path:
        return _MLS_HOST_REFERERS["cotality.com"]
    return _referer_for_image_host(parsed.netloc)


def image_download_referer(flow: ImageDownloadFlow) -> str:
    if flow == "renovation":
        specific = (settings.RENOVATION_IMAGE_DOWNLOAD_REFERER or "").strip()
    else:
        specific = (settings.CONDITION_SCORE_IMAGE_DOWNLOAD_REFERER or "").strip()
    if specific:
        return specific
    return (settings.IMAGE_DOWNLOAD_REFERER or "").strip()


def image_download_proxy_template(flow: ImageDownloadFlow) -> str:
    if flow == "renovation":
        specific = (settings.RENOVATION_IMAGE_DOWNLOAD_PROXY_TEMPLATE or "").strip()
    else:
        specific = (settings.CONDITION_SCORE_IMAGE_DOWNLOAD_PROXY_TEMPLATE or "").strip()
    if specific:
        return specific
    return (settings.IMAGE_DOWNLOAD_PROXY_TEMPLATE or "").strip()


def image_download_config_summary(flow: ImageDownloadFlow) -> dict[str, Any]:
    """Non-secret snapshot for logs / API error meta (verify Railway env is loaded)."""
    referer = image_download_referer(flow)
    proxy_template = image_download_proxy_template(flow)
    return {
        "flow": flow,
        "referer_configured": bool(referer),
        "proxy_configured": bool(proxy_template),
        "referer_env": _FLOW_REFERER_ENV[flow],
        "proxy_env": _FLOW_PROXY_ENV[flow],
        "trestle_auth_configured": trestle_credentials_configured(),
    }


def build_image_download_headers(
    image_url: str,
    *,
    flow: ImageDownloadFlow,
    force_refresh_trestle_token: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": _BROWSER_USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    parsed = urlparse(image_url)
    # Host/path wins over env override so CRMLS referer does not break api.cotality.com URLs.
    mls_referer = _referer_for_image_url(image_url) if parsed.netloc else None
    override = image_download_referer(flow)
    if mls_referer:
        headers["Referer"] = mls_referer
        headers["Origin"] = mls_referer.rstrip("/")
    elif override:
        headers["Referer"] = override
    elif parsed.scheme and parsed.netloc:
        headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

    if is_trestle_media_url(image_url):
        token = get_trestle_access_token(force_refresh=force_refresh_trestle_token)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif trestle_credentials_configured():
            logger.warning(
                "Trestle media URL without Bearer token (token fetch failed): %s",
                image_url[:200],
            )
        else:
            logger.debug(
                "Trestle media URL without TRESTLE_CLIENT_ID/SECRET configured: %s",
                image_url[:200],
            )
    return headers


def _proxy_urls_from_template(template: str, image_url: str) -> list[str]:
    if not template:
        return []
    if "://" in image_url:
        raw = image_url.split("://", 1)[1]
    else:
        raw = image_url
    candidates: list[str] = []
    placeholder_values = {
        "url_no_scheme_encoded": (quote(raw, safe=""), quote(f"https://{raw}", safe="")),
        "url_full_encoded": (quote(image_url, safe=""),),
    }
    for placeholder, encoded_values in placeholder_values.items():
        token = "{" + placeholder + "}"
        if token not in template:
            continue
        for encoded_value in encoded_values:
            url = template.replace(token, encoded_value)
            if url not in candidates:
                candidates.append(url)
    if not candidates and "{url_no_scheme_encoded}" not in template and "{url_full_encoded}" not in template:
        candidates.append(template.replace("{url}", quote(image_url, safe="")))
    return candidates


def _response_body_snippet(
    response: httpx.Response | None,
    *,
    limit: int = _RESPONSE_BODY_SNIPPET_LIMIT,
) -> str:
    if response is None:
        return ""
    try:
        text = response.text or ""
    except Exception:
        text = (response.content or b"").decode("utf-8", errors="replace")
    return text[:limit]


def _response_headers_snapshot(response: httpx.Response | None) -> dict[str, str]:
    if response is None:
        return {}
    return dict(response.headers)


def _request_headers_for_log(headers: dict[str, str]) -> dict[str, str]:
    """Log sent headers; redact Bearer tokens."""
    out = dict(headers)
    if out.get("Authorization", "").startswith("Bearer "):
        out["Authorization"] = "Bearer ***"
    return out


def _log_image_download_http_failure(
    *,
    image_url: str,
    flow: ImageDownloadFlow,
    attempt: str,
    request_url: str,
    request_headers: dict[str, str],
    response: httpx.Response | None = None,
    exc: Exception | None = None,
) -> None:
    """
    Log status, response headers, and body snippet (e.g. AccessDenied, Forbidden XML).
    Use dev logs to compare with local when Cotality URLs fail only in one environment.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        response = exc.response

    status_code = response.status_code if response is not None else None
    if status_code is None and exc is not None:
        status_code = getattr(exc, "status_code", None)

    body_snippet = _response_body_snippet(response)
    response_headers = _response_headers_snapshot(response)

    logger.warning(
        "Image download HTTP failure | flow=%s attempt=%s status_code=%s "
        "listing_url=%s request_url=%s request_headers=%s response_headers=%s "
        "response_body_snippet=%s exc=%s",
        flow,
        attempt,
        status_code,
        image_url[:200],
        request_url[:200],
        _request_headers_for_log(request_headers),
        response_headers,
        body_snippet,
        repr(exc) if exc is not None else None,
    )


def build_proxy_image_urls(image_url: str, *, flow: ImageDownloadFlow) -> list[str]:
    templates: list[str] = []
    configured = image_download_proxy_template(flow)
    if configured:
        templates.append(configured)
    for builtin in _BUILTIN_PROXY_TEMPLATES:
        if builtin not in templates:
            templates.append(builtin)

    candidates: list[str] = []
    for template in templates:
        for url in _proxy_urls_from_template(template, image_url):
            if url not in candidates:
                candidates.append(url)
    return candidates


def _try_proxy_download(
    *,
    image_url: str,
    flow: ImageDownloadFlow,
    proxy_urls: list[str],
    timeout: float,
    headers: dict[str, str],
) -> bytes | None:
    for index, proxy_url in enumerate(proxy_urls, start=1):
        try:
            proxy_response = httpx.get(
                proxy_url,
                timeout=timeout,
                follow_redirects=True,
                headers=headers,
            )
            proxy_response.raise_for_status()
            content = proxy_response.content
            if content and len(content) > 256:
                return content
            _log_image_download_http_failure(
                image_url=image_url,
                flow=flow,
                attempt=f"proxy_{index}_too_small",
                request_url=proxy_url,
                request_headers=headers,
                response=proxy_response,
            )
        except httpx.HTTPStatusError as proxy_exc:
            _log_image_download_http_failure(
                image_url=image_url,
                flow=flow,
                attempt=f"proxy_{index}",
                request_url=proxy_url,
                request_headers=headers,
                exc=proxy_exc,
            )
        except Exception as proxy_exc:
            logger.warning(
                "Image download proxy error | flow=%s attempt=proxy_%s listing_url=%s "
                "request_url=%s request_headers=%s exc=%s",
                flow,
                index,
                image_url[:200],
                proxy_url[:200],
                _request_headers_for_log(headers),
                proxy_exc,
            )
    return None


def _download_direct(
    *,
    image_url: str,
    flow: ImageDownloadFlow,
    timeout: float,
    attempt: str,
    force_refresh_trestle_token: bool = False,
) -> tuple[bytes | None, int | None]:
    """GET listing URL; returns (content, http_status_on_failure)."""
    headers = build_image_download_headers(
        image_url,
        flow=flow,
        force_refresh_trestle_token=force_refresh_trestle_token,
    )
    try:
        response = httpx.get(
            image_url,
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        )
        response.raise_for_status()
        return response.content, None
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else None
        _log_image_download_http_failure(
            image_url=image_url,
            flow=flow,
            attempt=attempt,
            request_url=image_url,
            request_headers=headers,
            exc=exc,
        )
        return None, status
    except Exception as exc:
        logger.warning(
            "Skipping image download for URL %s: %s | request_headers=%s",
            image_url[:200],
            exc,
            _request_headers_for_log(headers),
        )
        return None, None


def download_listing_image_bytes(
    image_url: str,
    *,
    flow: ImageDownloadFlow,
    timeout: float = 20.0,
) -> bytes | None:
    """Download one listing photo; Trestle URLs use OAuth Bearer when configured."""
    cleaned = (image_url or "").strip()
    if not cleaned:
        return None

    trestle_url = is_trestle_media_url(cleaned)
    content, status = _download_direct(
        image_url=cleaned,
        flow=flow,
        timeout=timeout,
        attempt="direct",
    )
    if content is not None:
        return content

    if status == 401 and trestle_url and trestle_credentials_configured():
        clear_trestle_token_cache()
        content, status = _download_direct(
            image_url=cleaned,
            flow=flow,
            timeout=timeout,
            attempt="direct_token_refresh",
            force_refresh_trestle_token=True,
        )
        if content is not None:
            return content

    if status not in (401, 403):
        return None

    # Public proxies cannot forward Trestle Bearer auth to Cotality media URLs.
    if trestle_url and trestle_credentials_configured():
        logger.warning(
            "Skipping image download for Trestle media URL %s: HTTP %s after Bearer auth (%s)",
            cleaned[:200],
            status,
            image_download_config_summary(flow),
        )
        return None

    proxy_urls = build_proxy_image_urls(cleaned, flow=flow)
    headers = build_image_download_headers(cleaned, flow=flow)
    content = _try_proxy_download(
        image_url=cleaned,
        flow=flow,
        proxy_urls=proxy_urls,
        timeout=timeout,
        headers=headers,
    )
    if content is not None:
        return content
    logger.warning(
        "Skipping image download for URL %s: HTTP %s; direct + %s proxy URL(s) failed (%s)",
        cleaned[:200],
        status,
        len(proxy_urls),
        image_download_config_summary(flow),
    )
    return None
