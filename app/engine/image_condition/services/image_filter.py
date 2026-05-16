from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

from PIL import Image

from app.core.image_download import download_listing_image_bytes

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

_HOUSE_ROOM_TYPES = frozenset(HOUSE_ROOM_LABELS)

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

# Vision prompt uses short names; map to canonical CLIP house labels.
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
    """Map vision/CLIP labels to canonical house room types."""
    text = re.sub(r"\s+", " ", (raw or "").strip().lower()).replace("_", " ")
    if text in _HOUSE_ROOM_TYPES:
        return text
    if text in _VISION_ROOM_TYPE_ALIASES:
        return _VISION_ROOM_TYPE_ALIASES[text]
    clip = re.sub(r"\s+", " ", (clip_fallback or "").strip().lower())
    if clip in _HOUSE_ROOM_TYPES:
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
            image_bytes=image_bytes,
        )
    except Exception as exc:
        logger.warning("CLIP classification skipped for %s: %s", source[:100], exc)
        return None


def deduplicate_filtered_by_room_type(
    images: list[FilteredImage],
) -> tuple[list[FilteredImage], int]:
    """
    Keep one photo per ``room_type`` (highest CLIP confidence) before vision scoring.

    Example: three ``bedroom interior`` URLs after filter → one bedroom is sent to OpenAI.
    """
    if not images:
        return [], 0

    best_by_room: dict[str, FilteredImage] = {}
    for img in images:
        room = (img.room_type or "unknown").strip() or "unknown"
        prev = best_by_room.get(room)
        if prev is None or float(img.confidence) > float(prev.confidence):
            best_by_room[room] = img

    label_order = {label: index for index, label in enumerate(HOUSE_ROOM_LABELS)}
    unique = sorted(
        best_by_room.values(),
        key=lambda row: (label_order.get(row.room_type, 999), -float(row.confidence)),
    )
    skipped = len(images) - len(unique)
    if skipped:
        logger.info(
            "condition-score dedupe: %s filtered image(s) -> %s unique room type(s) (%s duplicate room photos skipped)",
            len(images),
            len(unique),
            skipped,
        )
    return unique, skipped


def classify_and_filter_urls(image_urls: list[str]) -> dict[str, object]:
    """Download each URL, run CLIP, and keep house/property photos only."""
    if not image_urls:
        return {"selected": [], "discarded_count": 0, "total_input": 0, "download_failures": 0}

    selected: list[FilteredImage] = []
    discarded_count = 0
    download_failures = 0

    for url in image_urls:
        cleaned = (url or "").strip()
        if not cleaned:
            discarded_count += 1
            continue

        downloaded = download_listing_image_bytes(cleaned, flow="condition_score")
        if downloaded is None:
            download_failures += 1
            discarded_count += 1
            continue
        row = _clip_classify_bytes(downloaded, source=cleaned)
        if row is None:
            discarded_count += 1
        else:
            selected.append(row)

    return {
        "selected": selected,
        "discarded_count": discarded_count,
        "total_input": len(image_urls),
        "download_failures": download_failures,
    }

