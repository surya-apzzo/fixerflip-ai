"""Validate raw bytes look like an image (not HTML/WAF error pages)."""

from __future__ import annotations


def is_valid_image_bytes(data: bytes | None) -> bool:
    if not data or len(data) < 256:
        return False
    sample = data[:800].lstrip().lower()
    if sample.startswith(b"<!doctype") or sample.startswith(b"<html") or b"<html" in sample:
        return False
    if b"_incapsula_resource" in sample:
        return False
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return True
    return False
