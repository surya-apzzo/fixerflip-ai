from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

ROOM_WEIGHTS: dict[str, float] = {
    "kitchen interior": 0.35,
    "bathroom interior": 0.25,
    "living room interior": 0.15,
    "bedroom interior": 0.1,
    "basement interior": 0.1,
    "exterior front of house": 0.05,
}

DISCARD_ROOMS: set[str] = {
    "floor plan diagram",
    "neighborhood street view",
    "aerial view",
    "swimming pool",
    "garage exterior",
}

ALL_LABELS = [*ROOM_WEIGHTS.keys(), *DISCARD_ROOMS]

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
    lowered = (url or "").lower()
    discard_terms = (
        "floorplan",
        "floor-plan",
        "map",
        "streetview",
        "aerial",
        "pool",
        "garage",
    )
    return not any(term in lowered for term in discard_terms)


def _download_image_bytes(image_url: str) -> bytes | None:
    try:
        response = httpx.get(image_url, timeout=12.0, follow_redirects=True)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        logger.warning("Skipping image download for URL %s: %s", image_url, exc)
        return None


def classify_and_filter(image_urls: list[str]) -> dict[str, object]:
    """
    Keep all home-related images after filtering.
    No fixed image cap is applied in this stage.
    """
    clean_urls = [url.strip() for url in image_urls if url and url.strip()]
    if not clean_urls:
        return {"selected": [], "discarded_count": 0, "total_input": 0}

    if not _load_clip():
        selected = [
            FilteredImage(
                image_url=url,
                room_type="unknown",
                confidence=0.5,
                weight=0.1,
            )
            for url in clean_urls
            if _heuristic_home_image(url)
        ]
        return {
            "selected": selected,
            "discarded_count": max(0, len(clean_urls) - len(selected)),
            "total_input": len(clean_urls),
        }

    text_tokens = _clip_tokenize(ALL_LABELS)
    selected: list[FilteredImage] = []
    discarded_count = 0

    for url in clean_urls:
        image_bytes = _download_image_bytes(url)
        if image_bytes is None:
            discarded_count += 1
            continue
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = _clip_preproc(image).unsqueeze(0)
            with _clip_torch.no_grad():
                logits, _ = _clip_model(tensor, text_tokens)
                probs = logits.softmax(dim=-1)[0]
            best_idx = int(probs.argmax().item())
            room_type = ALL_LABELS[best_idx]
            confidence = float(probs[best_idx].item())
            if room_type in DISCARD_ROOMS:
                discarded_count += 1
                continue
            selected.append(
                FilteredImage(
                    image_url=url,
                    room_type=room_type,
                    confidence=confidence,
                    weight=ROOM_WEIGHTS.get(room_type, 0.1),
                )
            )
        except Exception as exc:
            logger.warning("CLIP classification skipped for URL %s: %s", url, exc)
            discarded_count += 1

    return {
        "selected": selected,
        "discarded_count": discarded_count,
        "total_input": len(clean_urls),
    }

