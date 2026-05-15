"""Prepare image bytes for OpenAI vision inside condition-score only (JPEG data URLs)."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from app.engine.renovation_engine.image_edit_engine import _download_source_image
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
# Smaller than renovation edits — fewer vision tokens, faster OpenAI calls.
_MAX_SIDE = 1280
_JPEG_QUALITY = 82


async def image_url_to_openai_vision_data_url(image_url: str) -> str:
    """
    Download the URL (MLS referer + optional proxy), re-encode as JPEG, return data URL.

    Uses the same download path as renovation edits so Cotality/CRMLS hosts get
  referer headers and IMAGE_DOWNLOAD_PROXY_TEMPLATE fallback on 403.
    """
    url = (image_url or "").strip()
    if not url:
        raise ValueError("Empty image URL.")

    image_bytes, _media_type = await _download_source_image(url)

    if len(image_bytes) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Image download exceeds size limit.")

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
