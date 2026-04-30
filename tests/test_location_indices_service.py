from __future__ import annotations

import app.services.location_indices_service as location_indices_service


def test_parse_rsmeans_location_factor_normalizes_raw_index_values():
    parsed = location_indices_service._parse_rsmeans_location_factor(
        {
            "data": {
                "totalWeightedAverage": 108,
            },
        }
    )

    assert parsed == 1.08


def test_extract_bls_latest_wage():
    wage = location_indices_service._extract_bls_latest_wage(
        {
            "Results": {
                "series": [
                    {
                        "data": [
                            {
                                "value": "38.64",
                            }
                        ]
                    }
                ]
            }
        }
    )

    assert wage == 38.64


async def test_resolve_location_indices_returns_bls_and_rsmeans_factors(monkeypatch):
    async def fake_bls_labor_index_cached():
        return 1.12

    async def fake_rsmeans_location_factor_cached(zip_code: str):
        assert zip_code == "94103"
        return 1.08

    monkeypatch.setattr(
        location_indices_service,
        "_fetch_bls_labor_index_cached",
        fake_bls_labor_index_cached,
    )
    monkeypatch.setattr(
        location_indices_service,
        "_fetch_rsmeans_location_factor_cached",
        fake_rsmeans_location_factor_cached,
    )

    factors = await location_indices_service.resolve_location_indices("94103-1234")

    assert factors.labor_index == 1.12
    assert factors.time_factor == 1.12
    assert factors.location_factor == 1.08


async def test_resolve_location_indices_does_not_cross_fallback(monkeypatch):
    async def fake_bls_labor_index_cached():
        return None

    async def fake_rsmeans_location_factor_cached(_zip_code: str):
        return 1.07

    monkeypatch.setattr(
        location_indices_service,
        "_fetch_bls_labor_index_cached",
        fake_bls_labor_index_cached,
    )
    monkeypatch.setattr(
        location_indices_service,
        "_fetch_rsmeans_location_factor_cached",
        fake_rsmeans_location_factor_cached,
    )

    factors = await location_indices_service.resolve_location_indices("94103")

    assert factors.labor_index is None
    assert factors.time_factor is None
    assert factors.location_factor == 1.07
