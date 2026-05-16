from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

from PIL import Image

from app.core.image_bytes import is_valid_image_bytes
from app.services.listing_image_storage import resolve_listing_image_bytes

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

# One vision photo per room when the listing has them (dedupe uses ranked CLIP labels).
CONDITION_SCORE_ROOM_PRIORITY: tuple[str, ...] = (
    "kitchen interior",
    "bathroom interior",
    "living room interior",
    "bedroom interior",
    "exterior front of house",
    "basement interior",
)

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
_clip_load_attempted = False


@dataclass(slots=True)
class FilteredImage:
    image_url: str
    room_type: str
    confidence: float
    image_bytes: bytes | None = None
    # Other house labels for this photo (2nd/3rd best) so duplicates can fill different rooms.
    room_rankings: tuple[tuple[str, float], ...] = ()


def clip_available() -> bool:
    return _load_clip()


def _load_clip() -> bool:
    global _clip_model, _clip_preproc, _clip_tokenize, _clip_torch, _clip_load_attempted
    if _clip_model is not None:
        return True
    if _clip_load_attempted:
        return False
    _clip_load_attempted = True
    try:
        import clip  # type: ignore
        import torch  # type: ignore
    except Exception as exc:
        logger.warning(
            "CLIP dependencies unavailable (%s). Dedupe will keep up to %s photos for "
            "OpenAI vision room labeling; install torch+CLIP in the deploy image for best results.",
            exc,
            len(CONDITION_SCORE_ROOM_PRIORITY),
        )
        return False

    _clip_model, _clip_preproc = clip.load("ViT-B/32", device="cpu")
    _clip_tokenize = clip.tokenize
    _clip_torch = torch
    logger.info("CLIP ViT-B/32 loaded for condition-score room filtering.")
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


def _ranked_house_labels(probs) -> list[tuple[str, float]]:
    """House labels for one photo, highest softmax probability first."""
    pairs = [(ALL_LABELS[i], float(probs[i].item())) for i in HOUSE_LABEL_INDICES]
    pairs.sort(key=lambda row: -row[1])
    return [(label, conf) for label, conf in pairs if conf >= _MIN_HOUSE_LABEL_PROBABILITY]


def _best_house_label(probs) -> tuple[str, float] | None:
    """Best-scoring house label, or None if nothing crosses the keep threshold."""
    ranked = _ranked_house_labels(probs)
    if not ranked:
        return None
    return ranked[0]


def _is_non_property_image(probs) -> bool:
    """True when the strongest CLIP label is explicitly non-property."""
    best_idx = int(probs.argmax().item())
    return best_idx in NON_PROPERTY_LABEL_INDICES


def _clip_classify_bytes(image_bytes: bytes, *, source: str) -> FilteredImage | None:
    if not is_valid_image_bytes(image_bytes):
        logger.warning("Bytes are not a valid image (HTML/WAF page?) for %s", source[:100])
        return None
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
        rankings = _ranked_house_labels(probs)
        if not rankings:
            logger.debug("Discarded image with no confident house label: %s", source[:100])
            return None
        room_type, confidence = rankings[0]
        return FilteredImage(
            image_url=source,
            room_type=room_type,
            confidence=confidence,
            image_bytes=image_bytes,
            room_rankings=tuple(rankings),
        )
    except Exception as exc:
        logger.warning("CLIP classification skipped for %s: %s", source[:100], exc)
        return None


def deduplicate_filtered_by_room_type(
    images: list[FilteredImage],
) -> tuple[list[FilteredImage], int]:
    """
    Keep one photo per room (kitchen, bath, living, bedroom, exterior, basement).

    Uses each photo's ranked CLIP house labels so duplicate "best" labels (e.g. many
    exteriors) can still fill other rooms via 2nd/3rd choice labels.
    """
    if not images:
        return [], 0

    room_slots: dict[str, FilteredImage] = {}
    used_urls: set[str] = set()

    for img in sorted(images, key=lambda row: -float(row.confidence)):
        rankings = img.room_rankings or ((img.room_type, img.confidence),)
        assigned_room: str | None = None
        for room, _conf in rankings:
            if room not in _HOUSE_ROOM_TYPES or room in room_slots:
                continue
            room_slots[room] = img
            used_urls.add(img.image_url)
            assigned_room = room
            break
        if assigned_room:
            img.room_type = assigned_room

    priority = {label: index for index, label in enumerate(CONDITION_SCORE_ROOM_PRIORITY)}
    unique = sorted(
        room_slots.values(),
        key=lambda row: (priority.get(row.room_type, 999), -float(row.confidence)),
    )

    # CLIP missing on server: photos stay room_type=unknown → assign no slots above.
    # Keep best N unique photos; OpenAI vision prompt assigns kitchen/bath/etc.
    if not unique and images:
        cap = len(CONDITION_SCORE_ROOM_PRIORITY)
        seen_urls: set[str] = set()
        fallback: list[FilteredImage] = []
        for img in sorted(images, key=lambda row: -float(row.confidence)):
            if img.image_url in seen_urls:
                continue
            seen_urls.add(img.image_url)
            fallback.append(img)
            if len(fallback) >= cap:
                break
        unique = fallback
        used_urls = seen_urls
        logger.info(
            "condition-score dedupe (no CLIP): %s filtered -> %s photo(s) for vision labeling",
            len(images),
            len(unique),
        )

    skipped = len(images) - len(used_urls)
    if unique and (skipped or len(unique) > 1):
        logger.info(
            "condition-score dedupe: %s filtered -> %s room(s) [%s] (%s duplicate photos skipped)",
            len(images),
            len(unique),
            ", ".join(row.room_type for row in unique),
            skipped,
        )
    return unique, skipped


def classify_and_filter_urls(
    image_urls: list[str],
    *,
    property_id: str = "",
) -> dict[str, object]:
    """Download each URL (S3 cache / storage URL / HTTP), run CLIP, keep house photos only."""
    if not image_urls:
        return {
            "selected": [],
            "discarded_count": 0,
            "total_input": 0,
            "download_failures": 0,
            "waf_blocked": False,
            "clip_available": clip_available(),
        }

    selected: list[FilteredImage] = []
    discarded_count = 0
    download_failures = 0
    waf_blocked = False

    for url in image_urls:
        cleaned = (url or "").strip()
        if not cleaned:
            discarded_count += 1
            continue

        resolved = resolve_listing_image_bytes(
            cleaned,
            property_id=property_id,
            flow="condition_score",
        )
        if resolved.waf_blocked:
            waf_blocked = True
        downloaded = resolved.content
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
        "waf_blocked": waf_blocked,
        "clip_available": clip_available(),
    }

