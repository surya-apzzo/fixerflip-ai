from __future__ import annotations

import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from PIL import Image

from app.core.config import settings
from app.core.image_bytes import is_valid_image_bytes
from app.services.listing_image_storage import ListingImageResolveResult, resolve_listing_image_bytes

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

# Absolute floor: below this, CLIP has no meaningful house signal (do not fallback-keep).
_CLIP_HOUSE_SIGNAL_FLOOR = 0.02
# Drop only when a non-property label clearly beats the best house label.
_NON_PROPERTY_MIN_CONFIDENCE = 0.20
_NON_PROPERTY_MARGIN_OVER_HOUSE = 0.05


def _min_house_label_probability() -> float:
    return float(settings.CONDITION_SCORE_CLIP_MIN_HOUSE_PROB)


def _clip_min_photos_target() -> int:
    return max(1, int(settings.CONDITION_SCORE_CLIP_MIN_PHOTOS))

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


def _label_score(probs, index: int) -> float:
    return float(probs[index].item())


def _best_label_in_indices(probs, indices: list[int]) -> tuple[str, float]:
    best_idx = max(indices, key=lambda i: _label_score(probs, i))
    return ALL_LABELS[best_idx], _label_score(probs, best_idx)


def _all_ranked_house_labels(probs) -> list[tuple[str, float]]:
    """All house labels for one photo, highest softmax probability first."""
    pairs = [(ALL_LABELS[i], _label_score(probs, i)) for i in HOUSE_LABEL_INDICES]
    pairs.sort(key=lambda row: -row[1])
    return pairs


def _house_rankings_for_photo(probs) -> tuple[tuple[str, float], ...]:
    """Ranked house labels stored on FilteredImage (used by room dedupe)."""
    return tuple(
        (label, conf)
        for label, conf in _all_ranked_house_labels(probs)
        if conf >= _CLIP_HOUSE_SIGNAL_FLOOR
    )


def select_filtered_images(
    strict: list[FilteredImage],
    weak: list[FilteredImage],
    *,
    min_photos: int | None = None,
) -> list[FilteredImage]:
    """
    Keep strict CLIP passes; if too few, add best weak house-scored MLS photos.

    Real listing interiors often score 0.04–0.06 on house labels (16-way softmax) — still
  valid house photos, just below a high fixed threshold.
    """
    goal = min_photos if min_photos is not None else _clip_min_photos_target()
    goal = min(goal, len(strict) + len(weak))
    selected = list(strict)
    if len(selected) >= goal:
        return selected
    seen = {img.image_url for img in selected}
    for img in sorted(weak, key=lambda row: -float(row.confidence)):
        if len(selected) >= goal:
            break
        if img.image_url in seen:
            continue
        selected.append(img)
        seen.add(img.image_url)
    return selected


def _is_non_property_image(probs) -> bool:
    """
    True when a non-property label clearly beats every house label.

    Do not use global argmax over all 16 labels — with ViT-B/32 softmax, a pool or
    garage label can edge out kitchen/bedroom by 0.01 and wrongly discard MLS photos.
    """
    best_house_label, house_conf = _best_label_in_indices(probs, HOUSE_LABEL_INDICES)
    best_non_label, non_conf = _best_label_in_indices(probs, list(NON_PROPERTY_LABEL_INDICES))
    if non_conf >= _NON_PROPERTY_MIN_CONFIDENCE and non_conf > house_conf + _NON_PROPERTY_MARGIN_OVER_HOUSE:
        logger.debug(
            "CLIP non-property discard: %s=%.3f vs best house %s=%.3f",
            best_non_label,
            non_conf,
            best_house_label,
            house_conf,
        )
        return True
    return False


def _clip_classify_bytes(image_bytes: bytes, *, source: str) -> tuple[FilteredImage | None, str]:
    """
    Classify image bytes. Second value is discard reason when None:
    invalid_bytes | non_property | weak_house | clip_error | heuristic_reject
    """
    if not is_valid_image_bytes(image_bytes):
        logger.warning("Bytes are not a valid image (HTML/WAF page?) for %s", source[:100])
        return None, "invalid_bytes"
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Invalid image bytes for %s: %s", source[:100], exc)
        return None, "invalid_bytes"

    if not _load_clip():
        if _heuristic_home_image(source):
            return (
                FilteredImage(
                    image_url=source,
                    room_type="unknown",
                    confidence=0.5,
                    image_bytes=image_bytes,
                ),
                "",
            )
        return None, "heuristic_reject"

    text_tokens = _clip_tokenize(ALL_LABELS)
    try:
        tensor = _clip_preproc(image).unsqueeze(0)
        with _clip_torch.no_grad():
            logits, _ = _clip_model(tensor, text_tokens)
            probs = logits.softmax(dim=-1)[0]
        if _is_non_property_image(probs):
            return None, "non_property"
        all_rankings = _all_ranked_house_labels(probs)
        room_type, confidence = all_rankings[0]
        rankings = _house_rankings_for_photo(probs)
        if confidence < _CLIP_HOUSE_SIGNAL_FLOOR:
            logger.debug(
                "Discarded image with no house CLIP signal (%s=%.3f) for %s",
                room_type,
                confidence,
                source[:100],
            )
            return None, "weak_house"
        row = FilteredImage(
            image_url=source,
            room_type=room_type,
            confidence=confidence,
            image_bytes=image_bytes,
            room_rankings=rankings,
        )
        if confidence >= _min_house_label_probability():
            return row, ""
        return row, "weak_house"
    except Exception as exc:
        logger.warning("CLIP classification skipped for %s: %s", source[:100], exc)
        return None, "clip_error"


def sample_urls_evenly(urls: list[str], limit: int) -> list[str]:
    """Spread picks across the full listing (first, middle, last) when URLs exceed the cap."""
    if limit <= 0 or not urls:
        return []
    if len(urls) <= limit:
        return list(urls)
    if limit == 1:
        return [urls[0]]
    last = len(urls) - 1
    indices = sorted({int(round(i * last / (limit - 1))) for i in range(limit)})
    return [urls[i] for i in indices]


def prepare_condition_score_urls(image_urls: list[str]) -> tuple[list[str], int, int]:
    """
    Normalize URL list and apply CONDITION_SCORE_MAX_INPUT_URLS.

    Returns (urls_to_process, urls_received, urls_truncated).
    """
    cleaned = [u.strip() for u in image_urls if u and u.strip()]
    received = len(cleaned)
    cap = max(6, int(settings.CONDITION_SCORE_MAX_INPUT_URLS))
    if received <= cap:
        return cleaned, received, 0
    sampled = sample_urls_evenly(cleaned, cap)
    truncated = received - len(sampled)
    logger.info(
        "condition-score: %s image_urls received, processing %s (even sample; %s skipped)",
        received,
        len(sampled),
        truncated,
    )
    return sampled, received, truncated


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


def _resolve_listing_url(url: str, *, property_id: str) -> tuple[str, ListingImageResolveResult]:
    cleaned = (url or "").strip()
    return cleaned, resolve_listing_image_bytes(
        cleaned,
        property_id=property_id,
        flow="condition_score",
    )


def classify_and_filter_urls(
    image_urls: list[str],
    *,
    property_id: str = "",
) -> dict[str, object]:
    """Download each URL (parallel), run CLIP, keep house photos only."""
    urls_to_process, urls_received, urls_truncated = prepare_condition_score_urls(image_urls)
    if not urls_to_process:
        return {
            "selected": [],
            "discarded_count": 0,
            "total_input": 0,
            "urls_received": urls_received,
            "urls_processed": 0,
            "urls_truncated": urls_truncated,
            "download_failures": 0,
            "waf_blocked": False,
            "clip_available": clip_available(),
        }

    strict: list[FilteredImage] = []
    weak: list[FilteredImage] = []
    download_failures = 0
    waf_blocked = False
    discard_invalid_bytes = 0
    discard_clip_non_property = 0
    discard_clip_weak_house = 0
    workers = max(1, int(settings.CONDITION_SCORE_DOWNLOAD_CONCURRENCY))

    resolved_rows: list[tuple[str, ListingImageResolveResult]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_resolve_listing_url, url, property_id=property_id): url
            for url in urls_to_process
        }
        for future in as_completed(futures):
            try:
                resolved_rows.append(future.result())
            except Exception as exc:
                failed_url = futures[future]
                logger.warning("condition-score resolve failed for %s: %s", failed_url[:120], exc)
                resolved_rows.append(
                    (
                        failed_url,
                        ListingImageResolveResult(content=None, source="resolve_error"),
                    )
                )

    for cleaned, resolved in resolved_rows:
        if not cleaned:
            continue
        if resolved.waf_blocked:
            waf_blocked = True
        downloaded = resolved.content
        if downloaded is None:
            download_failures += 1
            continue
        row, discard_reason = _clip_classify_bytes(downloaded, source=cleaned)
        if row is None:
            if discard_reason == "invalid_bytes":
                discard_invalid_bytes += 1
            elif discard_reason == "non_property":
                discard_clip_non_property += 1
            elif discard_reason in ("weak_house", "clip_error", "heuristic_reject"):
                discard_clip_weak_house += 1
            continue
        if discard_reason == "weak_house":
            weak.append(row)
        else:
            strict.append(row)

    selected = select_filtered_images(strict, weak)
    fallback_added = max(0, len(selected) - len(strict))
    discard_clip_weak_house += max(0, len(weak) - fallback_added)
    discarded_count = len(urls_to_process) - len(selected)

    if urls_to_process:
        logger.info(
            "condition-score filter: %s urls -> %s kept (%s strict, %s fallback, min_house_prob=%.2f) | "
            "discarded: invalid=%s non_property=%s weak_house=%s download_fail=%s",
            len(urls_to_process),
            len(selected),
            len(strict),
            fallback_added,
            _min_house_label_probability(),
            discard_invalid_bytes,
            discard_clip_non_property,
            discard_clip_weak_house,
            download_failures,
        )

    if urls_to_process and not selected:
        logger.warning(
            "condition-score filter dropped all %s images (invalid_bytes=%s clip_non_property=%s "
            "clip_weak_house=%s download_failures=%s). Check Cotality bytes vs HTML cache.",
            len(urls_to_process),
            discard_invalid_bytes,
            discard_clip_non_property,
            discard_clip_weak_house,
            download_failures,
        )

    return {
        "selected": selected,
        "discarded_count": discarded_count,
        "clip_strict_count": len(strict),
        "clip_fallback_count": fallback_added,
        "total_input": len(urls_to_process),
        "urls_received": urls_received,
        "urls_processed": len(urls_to_process),
        "urls_truncated": urls_truncated,
        "download_failures": download_failures,
        "waf_blocked": waf_blocked,
        "clip_available": clip_available(),
        "discard_invalid_bytes": discard_invalid_bytes,
        "discard_clip_non_property": discard_clip_non_property,
        "discard_clip_weak_house": discard_clip_weak_house,
    }

