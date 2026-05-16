"""Detect Cotality Imperva/Incapsula WAF blocks (common on cloud datacenter IPs)."""

from __future__ import annotations

import httpx


def is_incapsula_waf_response(response: httpx.Response | None) -> bool:
    if response is None:
        return False
    header_blob = " ".join(f"{k}:{v}" for k, v in response.headers.items()).lower()
    if "incap_ses_" in header_blob or "visid_incap" in header_blob:
        return True
    try:
        body = (response.text or "")[:2000].lower()
    except Exception:
        body = response.content[:2000].decode("utf-8", errors="replace").lower()
    return "_incapsula_resource" in body or "incapsula" in body
