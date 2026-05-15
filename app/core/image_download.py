"""MLS/CDN listing image download helpers (separate referer/proxy per API flow)."""

from __future__ import annotations

import logging
from typing import Any, Literal
from urllib.parse import quote, urlparse

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

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
    }


def build_image_download_headers(image_url: str, *, flow: ImageDownloadFlow) -> dict[str, str]:
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
    for proxy_url in proxy_urls:
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
        except Exception as proxy_exc:
            logger.debug("Proxy download failed for %s via %s: %s", image_url[:80], proxy_url[:80], proxy_exc)
    return None


def download_listing_image_bytes(
    image_url: str,
    *,
    flow: ImageDownloadFlow,
    timeout: float = 20.0,
) -> bytes | None:
    """Download one listing photo; try configured + built-in proxies on HTTP 401/403."""
    cleaned = (image_url or "").strip()
    if not cleaned:
        return None

    headers = build_image_download_headers(cleaned, flow=flow)
    try:
        response = httpx.get(cleaned, timeout=timeout, follow_redirects=True, headers=headers)
        response.raise_for_status()
        return response.content
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status not in (401, 403):
            logger.warning("Skipping image download for URL %s: HTTP %s", cleaned[:120], status)
            return None
        proxy_urls = build_proxy_image_urls(cleaned, flow=flow)
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
            cleaned[:120],
            status,
            len(proxy_urls),
            image_download_config_summary(flow),
        )
        return None
    except Exception as exc:
        logger.warning("Skipping image download for URL %s: %s", cleaned[:120], exc)
        return None
