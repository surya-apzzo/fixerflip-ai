from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CostIndexFactors:
    labor_index: float | None = None
    time_factor: float | None = None
    location_factor: float | None = None


async def close_http_client() -> None:
    """No-op retained for startup/shutdown compatibility."""
    return None


async def resolve_location_indices(zip_code: str) -> CostIndexFactors:
    """External location index lookups removed; keep default admin indices."""
    if zip_code:
        logger.debug("Location index lookup disabled; using default indices for zip=%s", zip_code)
    return CostIndexFactors()
