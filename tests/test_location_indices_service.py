from __future__ import annotations

import app.services.location_indices_service as location_indices_service


async def test_resolve_location_indices_returns_default_factors_only():
    factors = await location_indices_service.resolve_location_indices("94103")
    assert factors.labor_index is None
    assert factors.time_factor is None
    assert factors.location_factor is None


async def test_close_http_client_is_noop():
    assert await location_indices_service.close_http_client() is None
