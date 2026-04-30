from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core import redis_cache
from app.core.config import settings

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None
_BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_DEFAULT_TIMEOUT_SECONDS = 7.0


@dataclass(frozen=True)
class CostIndexFactors:
    labor_index: float | None = None
    time_factor: float | None = None
    location_factor: float | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient()
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client during app shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _normalize_zip_code(zip_code: str) -> str:
    raw = (zip_code or "").strip()
    match = re.match(r"^(\d{5})(?:-\d{4})?$", raw)
    return match.group(1) if match else ""


def _normalize_index_multiplier(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric <= 0:
        return None
    if numeric > 10:
        numeric = numeric / 100.0
    if 0.5 <= numeric <= 2.5:
        return round(numeric, 4)
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    nested = _as_mapping(data.get("data"))
    for key in keys:
        if key in nested:
            return nested[key]
    indices = _as_mapping(data.get("indices"))
    for key in keys:
        if key in indices:
            return indices[key]
    return None


def _extract_bls_latest_wage(data: dict[str, Any]) -> float | None:
    results = data.get("Results")
    if isinstance(results, list) and results:
        results = results[0]
    series = _as_mapping(results).get("series")
    if not isinstance(series, list) or not series:
        return None
    observations = _as_mapping(series[0]).get("data")
    if not isinstance(observations, list) or not observations:
        return None
    try:
        return float(observations[0]["value"])
    except (KeyError, TypeError, ValueError):
        return None


def _parse_rsmeans_location_factor(data: dict[str, Any]) -> float | None:
    location_raw = _first_present(
        data,
        (
            "locationFactor",
            "location_factor",
            "totalIndex",
            "total_index",
            "totalWeightedAverage",
            "total_weighted_average",
            "costIndex",
            "cost_index",
            "regionalFactor",
            "regional_factor",
            "location",
            "total",
        ),
    )
    return _normalize_index_multiplier(location_raw)


async def _fetch_bls_labor_index() -> float | None:
    series_id = settings.BLS_CONSTRUCTION_WAGE_SERIES_ID
    if not series_id:
        return None

    payload: dict[str, Any] = {"seriesid": [series_id]}
    if settings.BLS_API_KEY:
        payload["registrationkey"] = settings.BLS_API_KEY

    client = _get_http_client()
    for attempt in range(2):
        try:
            response = await client.post(_BLS_API_URL, json=payload, timeout=5.0)
            response.raise_for_status()
            parsed = response.json()
            if parsed.get("status") and parsed.get("status") != "REQUEST_SUCCEEDED":
                logger.warning("BLS request did not succeed: %s", parsed.get("message"))
                return None

            current_wage = _extract_bls_latest_wage(parsed)
            if current_wage is None:
                logger.warning("BLS response did not include latest wage data.")
                return None
            return _normalize_index_multiplier(current_wage / settings.BLS_BASE_WAGE)
        except Exception as exc:
            logger.warning("BLS labor index attempt %s failed: %s", attempt + 1, exc)
            if attempt == 1:
                return None
            await asyncio.sleep(0.2)
    return None


def _build_rsmeans_url(zip_code: str) -> str:
    base_url = settings.RSMEANS_BASE_URL.rstrip("/")
    if "{zip_code}" in base_url:
        return base_url.replace("{zip_code}", zip_code)
    return f"{base_url}/cost-indices/{zip_code}"


async def _fetch_rsmeans_location_factor(zip_code: str) -> float | None:
    if not settings.RSMEANS_API_KEY or not settings.RSMEANS_BASE_URL:
        return None

    url = _build_rsmeans_url(zip_code)
    headers = {
        "Authorization": f"Bearer {settings.RSMEANS_API_KEY}",
        "Accept": "application/json",
    }
    client = _get_http_client()
    for attempt in range(2):
        try:
            response = await client.get(url, headers=headers, timeout=_DEFAULT_TIMEOUT_SECONDS)
            response.raise_for_status()
            parsed = response.json()
            if not isinstance(parsed, dict):
                logger.warning("RSMeans response was not a JSON object.")
                return None
            location_factor = _parse_rsmeans_location_factor(parsed)
            if location_factor is None:
                logger.warning("RSMeans response did not contain a location factor field.")
            return location_factor
        except Exception as exc:
            logger.warning("RSMeans index attempt %s failed for ZIP %s: %s", attempt + 1, zip_code, exc)
            if attempt == 1:
                return None
            await asyncio.sleep(0.2)
    return None


def _cache_ttl_seconds() -> int:
    return settings.LOCATION_INDEX_CACHE_TTL_SECONDS


async def _get_cached_text(cache_key: str) -> str | None:
    try:
        return await asyncio.to_thread(redis_cache.get_text, cache_key)
    except Exception:
        return None


async def _set_cached_text(cache_key: str, value: str) -> None:
    try:
        await asyncio.to_thread(redis_cache.set_text, cache_key, value, _cache_ttl_seconds())
    except Exception:
        return


async def _fetch_bls_labor_index_cached() -> float | None:
    cache_key = f"renovation:indices:bls:{settings.BLS_CONSTRUCTION_WAGE_SERIES_ID}"
    cached = await _get_cached_text(cache_key)
    if cached:
        return _normalize_index_multiplier(cached)

    labor_index = await _fetch_bls_labor_index()
    if labor_index is not None:
        await _set_cached_text(cache_key, str(labor_index))
    return labor_index


async def _fetch_rsmeans_location_factor_cached(zip_code: str) -> float | None:
    cache_key = f"renovation:indices:rsmeans_location:{zip_code}"
    cached = await _get_cached_text(cache_key)
    if cached:
        return _normalize_index_multiplier(cached)

    location_factor = await _fetch_rsmeans_location_factor(zip_code)
    if location_factor is not None:
        await _set_cached_text(cache_key, json.dumps(location_factor))
    return location_factor


async def resolve_location_indices(zip_code: str) -> CostIndexFactors:
    """
    Resolve cost factors without cross-provider fallback.

    BLS supplies labor_index/time_factor as current_wage / base_wage.
    RSMeans/Gordian supplies the regional location_factor for the ZIP code.
    """
    normalized_zip = _normalize_zip_code(zip_code)
    if not normalized_zip:
        return CostIndexFactors()

    bls_task = asyncio.create_task(_fetch_bls_labor_index_cached())
    rsmeans_task = asyncio.create_task(_fetch_rsmeans_location_factor_cached(normalized_zip))
    bls_result, rsmeans_result = await asyncio.gather(
        bls_task,
        rsmeans_task,
        return_exceptions=True,
    )

    labor_index = None if isinstance(bls_result, Exception) else bls_result
    location_factor = None if isinstance(rsmeans_result, Exception) else rsmeans_result

    if isinstance(bls_result, Exception):
        logger.warning("BLS labor index task failed: %s", bls_result)
    if isinstance(rsmeans_result, Exception):
        logger.warning("RSMeans location factor task failed for ZIP %s: %s", normalized_zip, rsmeans_result)

    return CostIndexFactors(
        labor_index=labor_index,
        time_factor=labor_index,
        location_factor=location_factor,
    )
