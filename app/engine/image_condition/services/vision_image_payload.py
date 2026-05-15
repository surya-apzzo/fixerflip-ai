"""Prepare image bytes for OpenAI vision inside condition-score only (JPEG data URLs)."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO

from PIL import Image

from app.engine.renovation_engine.image_edit_engine import _download_source_image

_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
_MAX_SIDE = 1280
_JPEG_QUALITY = 82


def decode_image_base64_field(value: str) -> bytes:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Empty base64 image.")
    if raw.startswith("data:"):
        _header, _sep, payload = raw.partition(",")
        if not payload:
            raise ValueError("Invalid data URL: missing base64 payload.")
        raw = payload
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data.") from exc
    if not decoded:
        raise ValueError("Decoded image is empty.")
    if len(decoded) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Image exceeds size limit.")
    return decoded


def bytes_to_openai_vision_data_url(image_bytes: bytes) -> str:
    """Resize/re-encode to JPEG and return ``data:image/jpeg;base64,...``."""
    if len(image_bytes) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Image exceeds size limit.")

    with Image.open(BytesIO(image_bytes)) as im:
        im.load()
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > _MAX_SIDE:
            scale = _MAX_SIDE / float(max(w, h))
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        out = BytesIO()
        im.save(out, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        jpeg_bytes = out.getvalue()

    if len(jpeg_bytes) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Encoded image exceeds size limit.")

    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


async def image_url_to_openai_vision_data_url(image_url: str) -> str:
    """Download (MLS referer + optional proxy), then build a vision data URL."""
    url = (image_url or "").strip()
    if not url:
        raise ValueError("Empty image URL.")
    image_bytes, _media_type = await _download_source_image(url)
    return bytes_to_openai_vision_data_url(image_bytes)
