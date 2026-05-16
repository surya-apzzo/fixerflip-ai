#!/usr/bin/env python3
"""
Pre-warm S3 listing cache from your laptop (Cotality often works on home IP).

Usage:
  python scripts/warm_renovation_cache.py \\
    --image-url "https://api.cotality.com/trestle/Media/Property/PHOTO-Jpeg/..." \\
    --property-id 1158744011

Requires .env with STORAGE_* and optional TRESTLE_* (for download).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.listing_image_storage import stage_listing_image_for_renovation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm renovation/listings S3 cache.")
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--property-id", default="")
    args = parser.parse_args()
    result = stage_listing_image_for_renovation(
        args.image_url.strip(),
        property_id=args.property_id.strip(),
    )
    print(f"OK source={result.source}")
    print(f"staged_source_image_url={result.url}")
    if result.storage_key:
        print(f"storage_key={result.storage_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
