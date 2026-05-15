from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

import httpx
from PIL import Image

from app.engine.renovation_engine.image_edit_engine import _build_image_download_headers

logger = logging.getLogger(__name__)

# Only these labels count toward condition score (actual house / unit photos).
HOUSE_ROOM_LABELS: tuple[str, ...] = (
    "kitchen interior",
    "bathroom interior",
    "living room interior",
    "bedroom interior",
    "basement interior",
    "exterior front of house",
)

ROOM_WEIGHTS: dict[str, float] = {
    "kitchen interior": 0.35,
    "bathroom interior": 0.25,
    "living room interior": 0.15,
    "bedroom interior": 0.1,
    "basement interior": 0.1,
    "exterior front of house": 0.05,
}

# Non-property / non-house images — never used for condition score.
NON_PROPERTY_LABELS: frozenset[str] = frozenset(
    {
        "floor plan diagram",
        "neighborhood street view",
        "aerial view",
        "swimming pool",
        "garage exterior",
        "driveway or parking lot",
        "backyard only no house",
        "map or satellite image",
        "document screenshot or logo",
        "community amenity or clubhouse",
    }
)

# Minimum CLIP probability on a house label to keep an ambiguous image.
_MIN_HOUSE_LABEL_PROBABILITY = 0.12

ALL_LABELS = [*HOUSE_ROOM_LABELS, *sorted(NON_PROPERTY_LABELS)]
HOUSE_LABEL_INDICES = [ALL_LABELS.index(label) for label in HOUSE_ROOM_LABELS]
NON_PROPERTY_LABEL_INDICES = {ALL_LABELS.index(label) for label in NON_PROPERTY_LABELS}

# Vision prompt uses short names; CLIP + weights use long labels.
_VISION_ROOM_TYPE_ALIASES: dict[str, str] = {
    "kitchen": "kitchen interior",
    "bathroom": "bathroom interior",
    "bath": "bathroom interior",
    "living room": "living room interior",
    "living": "living room interior",
    "livingroom": "living room interior",
    "bedroom": "bedroom interior",
    "bed": "bedroom interior",
    "basement": "basement interior",
    "exterior": "exterior front of house",
    "exterior front": "exterior front of house",
    "front of house": "exterior front of house",
    "house exterior": "exterior front of house",
}


def normalize_house_room_type(raw: str, *, clip_fallback: str = "") -> str | None:
    """Map vision/CLIP labels to canonical ``ROOM_WEIGHTS`` keys."""
    text = re.sub(r"\s+", " ", (raw or "").strip().lower()).replace("_", " ")
    if text in ROOM_WEIGHTS:
        return text
    if text in _VISION_ROOM_TYPE_ALIASES:
        return _VISION_ROOM_TYPE_ALIASES[text]
    clip = re.sub(r"\s+", " ", (clip_fallback or "").strip().lower())
    if clip in ROOM_WEIGHTS:
        return clip
    return None


_clip_model = None
_clip_preproc = None
_clip_tokenize = None
_clip_torch = None


@dataclass(slots=True)
class FilteredImage:
    image_url: str
    room_type: str
    confidence: float
    weight: float
    image_bytes: bytes | None = None


def _load_clip() -> bool:
    global _clip_model, _clip_preproc, _clip_tokenize, _clip_torch
    if _clip_model is not None:
        return True
    try:
        import clip  # type: ignore
        import torch  # type: ignore
    except Exception:
        logger.warning("CLIP dependencies unavailable; using URL heuristic home-image filter.")
        return False

    _clip_model, _clip_preproc = clip.load("ViT-B/32", device="cpu")
    _clip_tokenize = clip.tokenize
    _clip_torch = torch
    return True


def _heuristic_home_image(url: str) -> bool:
    """URL-only filter when CLIP is unavailable (less accurate than vision classification)."""
    lowered = (url or "").lower()
    discard_terms = (
        "floorplan",
        "floor-plan",
        "floor_plan",
        "siteplan",
        "site-plan",
        "plat",
        "survey",
        "map",
        "streetview",
        "street-view",
        "aerial",
        "drone",
        "satellite",
        "pool",
        "garage",
        "amenity",
        "clubhouse",
        "logo",
        "document",
        "screenshot",
        "thumbnail-doc",
    )
    return not any(term in lowered for term in discard_terms)


def _best_house_label(probs) -> tuple[str, float] | None:
    """Best-scoring house label, or None if nothing crosses the keep threshold."""
    best_idx = max(HOUSE_LABEL_INDICES, key=lambda i: float(probs[i].item()))
    confidence = float(probs[best_idx].item())
    if confidence < _MIN_HOUSE_LABEL_PROBABILITY:
        return None
    return ALL_LABELS[best_idx], confidence


def _is_non_property_image(probs) -> bool:
    """True when the strongest CLIP label is explicitly non-property."""
    best_idx = int(probs.argmax().item())
    return best_idx in NON_PROPERTY_LABEL_INDICES


def _download_image_bytes(image_url: str) -> bytes | None:
    try:
        headers = _build_image_download_headers(image_url)
        response = httpx.get(image_url, timeout=12.0, follow_redirects=True, headers=headers)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        logger.warning("Skipping image download for URL %s: %s", image_url, exc)
        return None


def _clip_classify_bytes(image_bytes: bytes, *, source: str) -> FilteredImage | None:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Invalid image bytes for %s: %s", source[:100], exc)
        return None

    if not _load_clip():
        if _heuristic_home_image(source):
            return FilteredImage(
                image_url=source,
                room_type="unknown",
                confidence=0.5,
                weight=0.1,
                image_bytes=image_bytes,
            )
        return None

    text_tokens = _clip_tokenize(ALL_LABELS)
    try:
        tensor = _clip_preproc(image).unsqueeze(0)
        with _clip_torch.no_grad():
            logits, _ = _clip_model(tensor, text_tokens)
            probs = logits.softmax(dim=-1)[0]
        if _is_non_property_image(probs):
            logger.debug("Discarded non-property image (CLIP): %s", source[:100])
            return None
        house_match = _best_house_label(probs)
        if house_match is None:
            logger.debug("Discarded image with no confident house label: %s", source[:100])
            return None
        room_type, confidence = house_match
        return FilteredImage(
            image_url=source,
            room_type=room_type,
            confidence=confidence,
            weight=ROOM_WEIGHTS[room_type],
            image_bytes=image_bytes,
        )
    except Exception as exc:
        logger.warning("CLIP classification skipped for %s: %s", source[:100], exc)
        return None


def classify_and_filter_inputs(
    items: list[tuple[str | None, bytes | None]],
) -> dict[str, object]:
    """
    Filter listing photos. Each item is (optional_url, optional_bytes).

    When bytes are provided (client/base64 path), the server does not download the URL.
    """
    if not items:
        return {"selected": [], "discarded_count": 0, "total_input": 0}

    selected: list[FilteredImage] = []
    discarded_count = 0
    download_failures = 0

    for url, image_bytes in items:
        source = (url or "").strip() or "embedded-image"
        if image_bytes is not None:
            row = _clip_classify_bytes(image_bytes, source=source)
            if row is None:
                discarded_count += 1
            else:
                selected.append(row)
            continue

        if not (url or "").strip():
            discarded_count += 1
            continue

        downloaded = _download_image_bytes(url.strip())
        if downloaded is None:
            download_failures += 1
            discarded_count += 1
            continue
        row = _clip_classify_bytes(downloaded, source=url.strip())
        if row is None:
            discarded_count += 1
        else:
            selected.append(row)

    return {
        "selected": selected,
        "discarded_count": discarded_count,
        "total_input": len(items),
        "download_failures": download_failures,
    }


def classify_and_filter(image_urls: list[str]) -> dict[str, object]:
    """
    Keep only house/property photos (kitchen, bath, living areas, basement, front exterior).

    Floor plans, aerials, pools, street views, maps, and similar images are discarded
    and never sent to vision scoring.
    """
    clean_urls = [url.strip() for url in image_urls if url and url.strip()]
    items = [(url, None) for url in clean_urls]
    return classify_and_filter_inputs(items)

