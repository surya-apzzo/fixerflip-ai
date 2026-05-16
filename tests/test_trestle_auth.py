"""Tests for Trestle OAuth token helper."""

from unittest.mock import MagicMock, patch

from app.core import image_download as dl_mod
from app.core import trestle_auth as ta


def test_is_trestle_media_url() -> None:
    assert ta.is_trestle_media_url(
        "https://api.cotality.com/trestle/Media/Property/PHOTO-Jpeg/1156941869/3/x"
    )
    assert not ta.is_trestle_media_url("https://imagecdn.realty.dev/photo.jpg")


def test_trestle_token_cached(monkeypatch) -> None:
    monkeypatch.setattr(ta.settings, "TRESTLE_BASE_URL", "https://api.cotality.com")
    monkeypatch.setattr(ta.settings, "TRESTLE_CLIENT_ID", "client-id")
    monkeypatch.setattr(ta.settings, "TRESTLE_CLIENT_SECRET", "client-secret")
    ta.clear_trestle_token_cache()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"access_token": "tok-abc", "expires_in": 28800, "token_type": "Bearer"}

    with patch("app.core.trestle_auth.httpx.post", return_value=mock_resp) as post:
        first = ta.get_trestle_access_token()
        second = ta.get_trestle_access_token()

    assert first == "tok-abc"
    assert second == "tok-abc"
    assert post.call_count == 1
    call_kwargs = post.call_args.kwargs
    assert call_kwargs["data"]["grant_type"] == "client_credentials"
    assert call_kwargs["data"]["scope"] == "api"


def test_build_headers_includes_bearer_for_trestle_media(monkeypatch) -> None:
    monkeypatch.setattr(dl_mod.settings, "TRESTLE_BASE_URL", "https://api.cotality.com")
    monkeypatch.setattr(dl_mod.settings, "TRESTLE_CLIENT_ID", "client-id")
    monkeypatch.setattr(dl_mod.settings, "TRESTLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(dl_mod, "get_trestle_access_token", lambda **_: "signed-token")

    headers = dl_mod.build_image_download_headers(
        "https://api.cotality.com/trestle/Media/Property/PHOTO-Jpeg/1156941869/3/x",
        flow="condition_score",
    )
    assert headers["Authorization"] == "Bearer signed-token"
    assert headers["Referer"] == "https://www.cotality.com/"
