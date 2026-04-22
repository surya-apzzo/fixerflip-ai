from __future__ import annotations

import pytest

from app.engine.renovation_engine import image_edit_engine as engine


class _FakeImageItem:
    b64_json = "ZmFrZV9pbWFnZQ=="
    revised_prompt = "revised prompt"


class _FakeEditResponse:
    data = [_FakeImageItem()]


@pytest.mark.asyncio
async def test_edit_property_image_uses_high_input_fidelity_for_gpt_image(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_download_image(_: str) -> tuple[bytes, str]:
        return b"fake-image-bytes", "image/png"

    class _FakeImagesAPI:
        async def edit(self, **kwargs):
            captured["edit_kwargs"] = kwargs
            return _FakeEditResponse()

    class _FakeOpenAIClient:
        def __init__(self, api_key: str):
            assert api_key == "test-key"
            self.images = _FakeImagesAPI()

    monkeypatch.setattr(engine, "_download_image", _fake_download_image)
    monkeypatch.setattr(engine, "AsyncOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(engine.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(engine.settings, "OPENAI_IMAGE_EDIT_MODEL", "gpt-image-1")

    result = await engine.edit_property_image_from_url(
        image_url="https://example.com/property.png",
        instruction="Repair the damaged flooring and repaint the walls.",
    )

    edit_kwargs = captured["edit_kwargs"]
    assert edit_kwargs["input_fidelity"] == "high"
    assert "Non-negotiable structural preservation rules" in edit_kwargs["prompt"]
    assert "Do not invent new steps" in edit_kwargs["prompt"]
    assert result.image_base64 == "ZmFrZV9pbWFnZQ=="


@pytest.mark.asyncio
async def test_edit_property_image_skips_input_fidelity_for_non_gpt_image(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_download_image(_: str) -> tuple[bytes, str]:
        return b"fake-image-bytes", "image/png"

    class _FakeImagesAPI:
        async def edit(self, **kwargs):
            captured["edit_kwargs"] = kwargs
            return _FakeEditResponse()

    class _FakeOpenAIClient:
        def __init__(self, api_key: str):
            assert api_key == "test-key"
            self.images = _FakeImagesAPI()

    monkeypatch.setattr(engine, "_download_image", _fake_download_image)
    monkeypatch.setattr(engine, "AsyncOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(engine.settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(engine.settings, "OPENAI_IMAGE_EDIT_MODEL", "dall-e-2")

    await engine.edit_property_image_from_url(
        image_url="https://example.com/property.png",
        instruction="Repair the damaged flooring and repaint the walls.",
    )

    edit_kwargs = captured["edit_kwargs"]
    assert "input_fidelity" not in edit_kwargs
