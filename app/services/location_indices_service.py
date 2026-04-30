import asyncio
import logging
from typing import Tuple, Optional

import httpx
from app.core import redis_cache
from app.core.config import settings

logger = logging.getLogger(__name__)

# Cache the results for 30 days (30 * 24 * 60 * 60 seconds)
CACHE_TTL_SECONDS = 2592000

# Global HTTP client for connection pooling (initialized lazily to avoid event loop issues)
_http_client: Optional[httpx.AsyncClient] = None

def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient()
    return _http_client

async def close_http_client() -> None:
    """
    Call this function in your FastAPI @app.on_event("shutdown") 
    to cleanly close the connection pool and prevent memory leaks.
    """
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None

async def _fetch_from_bls() -> Optional[float]:
    """
    Real integration for BLS API using httpx.
    National average index, ignores zip code mapping unless manually provided later.
    """
    logger.info("Fetching national BLS data")
    
    api_key = getattr(settings, "BLS_API_KEY", None)
    if not api_key:
        logger.warning("BLS_API_KEY is missing in .env! Using fallback logic.")
        return 1.10
        
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    client = _get_http_client()
    
    for attempt in range(2):
        try:
            payload = {
                "seriesid": ["CEU2000000008"], 
                "registrationkey": api_key
            }
            # Shorter timeout: BLS shouldn't take more than 5s
            response = await client.post(url, json=payload, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            
            series = data.get("Results", {}).get("series", [])
            if not series or not series[0].get("data"):
                logger.warning("BLS response missing expected data structure. Keys: %s", list(data.keys()))
                return None
                
            raw_value = float(series[0]["data"][0]["value"])
            
            # CEU2000000008 returns average hourly earnings in dollars
            # We normalize it against a national base wage to get an index
            base_wage = getattr(settings, "BLS_BASE_WAGE", 34.50)
            return raw_value / base_wage
            
        except Exception as e:
            logger.warning("BLS attempt %s failed: %s", attempt + 1, e)
            if attempt == 1:
                logger.error("BLS API Error (Final): %s", e)
                return None
            await asyncio.sleep(0.2) # Jitter/delay before retry


async def _fetch_from_rsmeans(zip_code: str) -> Optional[float]:
    """
    Real integration for RSMeans API using httpx.
    """
    logger.info("Fetching RSMeans data for zip: %s", zip_code)
    
    api_key = getattr(settings, "RSMEANS_API_KEY", None)
    if not api_key:
        logger.warning("RSMEANS_API_KEY is missing in .env! Using fallback logic.")
        return 1.05 # Removed geographic bias from fallback to avoid inaccurate assumptions
        
    # Replace this URL with the exact endpoint provided by Gordian/RSMeans
    url = f"https://api.rsmeans.com/v1/cost-indices/{zip_code}"
    client = _get_http_client()
    
    for attempt in range(2):
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
            # RSMeans timeout set to 7s
            response = await client.get(url, headers=headers, timeout=7.0)
            response.raise_for_status()
            data = response.json()
            
            # Safely extract from RSMeans JSON. You must adjust this exact path
            if "materialIndex" in data:
                return float(data["materialIndex"])
            elif "costIndex" in data:
                return float(data["costIndex"])
            elif "data" in data and "costIndex" in data["data"]:
                return float(data["data"]["costIndex"])
                
            logger.warning("RSMeans response did not contain expected fields. Keys: %s", list(data.keys()))
            return None
            
        except Exception as e:
            logger.warning("RSMeans attempt %s failed for zip %s: %s", attempt + 1, zip_code, e)
            if attempt == 1:
                logger.error("RSMeans API Error (Final) for zip %s: %s", zip_code, e)
                return None
            await asyncio.sleep(0.2) # Jitter/delay before retry


async def _fetch_from_bls_cached() -> Optional[float]:
    cache_key = "renovation:indices:bls_national"
    try:
        cached_val = await asyncio.to_thread(redis_cache.get_text, cache_key)
        if cached_val:
            return float(cached_val)
    except Exception as e:
        logger.warning("Failed to parse cached BLS index: %s", e)

    val = await _fetch_from_bls()
    if val is not None:
        await asyncio.to_thread(redis_cache.set_text, cache_key, str(val), CACHE_TTL_SECONDS)
    return val


async def _fetch_from_rsmeans_cached(zip_code: str) -> Optional[float]:
    cache_key = f"renovation:indices:rsmeans:{zip_code}"
    try:
        cached_val = await asyncio.to_thread(redis_cache.get_text, cache_key)
        if cached_val:
            return float(cached_val)
    except Exception as e:
        logger.warning("Failed to parse cached RSMeans index for %s: %s", zip_code, e)

    val = await _fetch_from_rsmeans(zip_code)
    if val is not None:
        await asyncio.to_thread(redis_cache.set_text, cache_key, str(val), CACHE_TTL_SECONDS)
    return val


async def resolve_location_indices(zip_code: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Takes a zip code, checks individual caches, and if not found, calls BLS and RSMeans.
    Returns (labor_index, material_index).
    """
    if not zip_code or not zip_code.strip():
        return None, None
        
    zip_code = zip_code.strip()
    
    # Fetch in parallel. The individual cached functions handle their own caching logic independently.
    labor_task = asyncio.create_task(_fetch_from_bls_cached())
    material_task = asyncio.create_task(_fetch_from_rsmeans_cached(zip_code))
    
    results = await asyncio.gather(labor_task, material_task, return_exceptions=True)
    
    labor_idx = results[0] if not isinstance(results[0], Exception) else None
    material_idx = results[1] if not isinstance(results[1], Exception) else None
    
    if isinstance(results[0], Exception):
        logger.error("BLS parallel task failed: %s", results[0])
    if isinstance(results[1], Exception):
        logger.error("RSMeans parallel task failed for %s: %s", zip_code, results[1])
        
    return labor_idx, material_idx
