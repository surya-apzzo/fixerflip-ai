"""Prepare image bytes for OpenAI vision inside condition-score only (JPEG data URLs)."""

from __future__ import annotations

import base64
import logging
from io import BytesIO

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
_MAX_SIDE = 2048


async def image_url_to_openai_vision_data_url(image_url: str) -> str:
    """
    Download the URL, decode with Pillow, re-encode as JPEG, return ``data:image/jpeg;base64,...``.

    OpenAI rejects many remote URLs and mislabeled bodies; this path matches supported formats.
    """
    url = (image_url or "").strip()
    if not url:
        raise ValueError("Empty image URL.")

    async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": _BROWSER_USER_AGENT})
        response.raise_for_status()
        body = response.content
        ctype = (response.headers.get("content-type") or "").split(";")[0].strip().lower()

    if len(body) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Image download exceeds size limit.")
    if ctype and not ctype.startswith("image/") and ctype != "application/octet-stream":
        logger.warning("condition-score: unexpected content-type %s for %s", ctype, url[:100])

    with Image.open(BytesIO(body)) as im:
        im.load()
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > _MAX_SIDE:
            scale = _MAX_SIDE / float(max(w, h))
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        out = BytesIO()
        im.save(out, format="JPEG", quality=90, optimize=True)
        jpeg_bytes = out.getvalue()

    if len(jpeg_bytes) > _MAX_DOWNLOAD_BYTES:
        raise ValueError("Encoded image exceeds size limit.")

    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
