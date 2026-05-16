"""Unit tests for per-flow MLS image download settings."""

import httpx

from app.core import image_download as mod


def test_cotality_host_uses_cotality_referer_even_with_crmls_override(monkeypatch) -> None:
    monkeypatch.setattr(mod.settings, "CONDITION_SCORE_IMAGE_DOWNLOAD_REFERER", "https://www.crmls.org/")
    headers = mod.build_image_download_headers(
        "https://api.cotality.com/trestle/Media/Property/PHOTO-Jpeg/1156941869/2/x.jpg",
        flow="condition_score",
    )
    assert headers["Referer"] == "https://www.cotality.com/"


def test_crmls_path_on_realty_cdn_uses_crmls_referer(monkeypatch) -> None:
    monkeypatch.setattr(mod.settings, "RENOVATION_IMAGE_DOWNLOAD_REFERER", "")
    monkeypatch.setattr(mod.settings, "IMAGE_DOWNLOAD_REFERER", "")
    headers = mod.build_image_download_headers(
        "https://imagecdn.realty.dev/mls_photos/CRMLS/SB25134127/19.jpg?w=600",
        flow="renovation",
    )
    assert headers["Referer"] == "https://www.crmls.org/"


def test_flow_specific_referer_overrides_global(monkeypatch) -> None:
    monkeypatch.setattr(mod.settings, "IMAGE_DOWNLOAD_REFERER", "https://global.example/")
    monkeypatch.setattr(mod.settings, "RENOVATION_IMAGE_DOWNLOAD_REFERER", "https://reno.example/")
    monkeypatch.setattr(mod.settings, "CONDITION_SCORE_IMAGE_DOWNLOAD_REFERER", "https://score.example/")
    assert mod.image_download_referer("renovation") == "https://reno.example/"
    assert mod.image_download_referer("condition_score") == "https://score.example/"


def test_global_referer_fallback_when_flow_specific_empty(monkeypatch) -> None:
    monkeypatch.setattr(mod.settings, "IMAGE_DOWNLOAD_REFERER", "https://global.example/")
    monkeypatch.setattr(mod.settings, "RENOVATION_IMAGE_DOWNLOAD_REFERER", "")
    monkeypatch.setattr(mod.settings, "CONDITION_SCORE_IMAGE_DOWNLOAD_REFERER", "")
    assert mod.image_download_referer("renovation") == "https://global.example/"
    assert mod.image_download_referer("condition_score") == "https://global.example/"


def test_flow_specific_proxy_overrides_global(monkeypatch) -> None:
    monkeypatch.setattr(
        mod.settings, "IMAGE_DOWNLOAD_PROXY_TEMPLATE", "https://proxy.global/{url_full_encoded}"
    )
    monkeypatch.setattr(
        mod.settings,
        "RENOVATION_IMAGE_DOWNLOAD_PROXY_TEMPLATE",
        "https://proxy.reno/{url_full_encoded}",
    )
    monkeypatch.setattr(
        mod.settings,
        "CONDITION_SCORE_IMAGE_DOWNLOAD_PROXY_TEMPLATE",
        "https://proxy.score/{url_full_encoded}",
    )
    assert mod.image_download_proxy_template("renovation") == "https://proxy.reno/{url_full_encoded}"
    assert (
        mod.image_download_proxy_template("condition_score")
        == "https://proxy.score/{url_full_encoded}"
    )


def test_response_body_snippet_includes_s3_error_code() -> None:
    xml = (
        '<?xml version="1.0"?><Error><Code>AccessDenied</Code>'
        "<Message>Access Denied</Message></Error>"
    )
    response = httpx.Response(403, text=xml)
    snippet = mod._response_body_snippet(response, limit=500)
    assert "AccessDenied" in snippet
