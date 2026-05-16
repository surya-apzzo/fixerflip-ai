"""Tests for Cotality Incapsula WAF detection."""

import httpx

from app.core.cotality_waf import is_incapsula_waf_response
from app.services import listing_image_storage as storage


def test_incapsula_html_detected() -> None:
    html = '<script src="/_Incapsula_Resource?SWJIYLWA=abc"></script>'
    response = httpx.Response(
        403,
        text=html,
        headers={"set-cookie": "visid_incap_3235047=abc; path=/"},
    )
    assert is_incapsula_waf_response(response) is True


def test_normal_403_not_flagged_as_incapsula() -> None:
    response = httpx.Response(403, text='{"error":"invalid_client"}')
    assert is_incapsula_waf_response(response) is False


def test_public_url_to_storage_key() -> None:
    from app.core.config import settings

    base = "https://cdn.example/bucket"
    settings.STORAGE_PUBLIC_BASE_URL = base
    key = storage.public_url_to_storage_key(f"{base}/condition-score/listings/p1/abc.jpg")
    assert key == "condition-score/listings/p1/abc.jpg"
