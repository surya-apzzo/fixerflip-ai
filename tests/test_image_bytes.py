"""Tests for image byte validation."""

from app.core.image_bytes import is_valid_image_bytes


def test_rejects_incapsula_html() -> None:
    html = b"<html><script src='/_Incapsula_Resource'></script></html>" + b"x" * 300
    assert is_valid_image_bytes(html) is False


def test_accepts_jpeg_magic() -> None:
    assert is_valid_image_bytes(b"\xff\xd8\xff" + b"x" * 300) is True
