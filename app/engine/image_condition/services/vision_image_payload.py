"""Prepare image bytes for OpenAI vision (JPEG data URLs)."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
_MAX_SIDE = 1280
_JPEG_QUALITY = 82


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

