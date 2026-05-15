from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.engine.image_condition.services.image_filter import FilteredImage
from app.engine.image_condition.services.vision_image_payload import image_url_to_openai_vision_data_url

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "property_condition_vision_prompt.txt"

_VISION_PROMPT_FALLBACK = """
You are a real estate flip analyst scoring property condition from photos.
Analyze ALL photos and return one object per photo in the SAME ORDER.

Return ONLY a JSON array:
[
  {
    "room_type": "<kitchen|bathroom|living_room|bedroom|basement|exterior|unknown>",
    "condition_score": <0-100>,
    "signals": [<1-2 concise observations>],
    "red_flags": [<serious/structural concerns only>]
  }
]
""".strip()

_MAX_IMAGES_PER_REQUEST = 20


def _load_vision_prompt() -> str:
    if _PROMPT_PATH.exists():
        content = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        if content:
            return content
    return _VISION_PROMPT_FALLBACK


def _chunk_images(images: list[FilteredImage], chunk_size: int = _MAX_IMAGES_PER_REQUEST) -> list[list[FilteredImage]]:
    return [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else []


async def score_from_images(selected_images: list[FilteredImage]) -> dict[str, Any]:
    if not selected_images:
        return {"room_scores": [], "cost_usd": 0.0}
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for condition-score vision analysis.")

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=45.0, max_retries=0)
    room_scores: list[dict[str, Any]] = []
    total_cost = 0.0
    vision_prompt = _load_vision_prompt()

    for chunk in _chunk_images(selected_images):
        content: list[dict[str, Any]] = [{"type": "input_text", "text": vision_prompt}]
        chunk_index_by_image_slot: list[int] = []
        for i, img in enumerate(chunk):
            try:
                payload_url = await image_url_to_openai_vision_data_url(img.image_url)
            except Exception as exc:
                logger.warning(
                    "condition-score: skipping image (cannot prepare for vision): idx=%s url=%s err=%s",
                    i,
                    (img.image_url or "")[:120],
                    exc,
                )
                continue
            if not (payload_url or "").strip():
                logger.warning(
                    "condition-score: skipping image (empty payload): idx=%s",
                    i,
                )
                continue
            content.append({"type": "input_image", "image_url": payload_url})
            chunk_index_by_image_slot.append(i)

        if len(content) <= 1:
            for i, img in enumerate(chunk):
                room_scores.append(
                    {
                        "room_type": img.room_type or "unknown",
                        "score": 50.0,
                        "weight": float(img.weight),
                        "signals": (
                            ["Image could not be loaded or decoded for scoring."]
                            if i not in chunk_index_by_image_slot
                            else []
                        ),
                        "red_flags": [],
                    }
                )
            continue

        response = await client.responses.create(
            model=settings.default_openai_vision_model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=1000,
        )
        raw = getattr(response, "output_text", "") or ""
        parsed_chunk = _parse_json_array(raw)

        usage = getattr(response, "usage", None)
        if usage is not None:
            in_tokens = float(getattr(usage, "input_tokens", 0) or 0)
            out_tokens = float(getattr(usage, "output_tokens", 0) or 0)
            total_cost += ((in_tokens / 1_000_000) * 0.15) + ((out_tokens / 1_000_000) * 0.60)

        filled: list[dict[str, Any] | None] = [None] * len(chunk)
        for slot_idx, result in enumerate(parsed_chunk):
            if slot_idx >= len(chunk_index_by_image_slot):
                break
            chunk_idx = chunk_index_by_image_slot[slot_idx]
            if not isinstance(result, dict):
                result = {}
            try:
                score = float(result.get("condition_score", 50))
            except (TypeError, ValueError):
                score = 50.0
            filled[chunk_idx] = {
                "room_type": str(result.get("room_type", chunk[chunk_idx].room_type or "unknown")),
                "score": max(0.0, min(100.0, score)),
                "weight": float(chunk[chunk_idx].weight),
                "signals": [str(s) for s in result.get("signals", []) if str(s).strip()],
                "red_flags": [str(s) for s in result.get("red_flags", []) if str(s).strip()],
            }

        slot_by_chunk_idx = {cidx: slot for slot, cidx in enumerate(chunk_index_by_image_slot)}
        for i, img in enumerate(chunk):
            if filled[i] is not None:
                room_scores.append(filled[i])
                continue
            if i not in slot_by_chunk_idx:
                signals = ["Image could not be loaded or decoded for scoring."]
            elif slot_by_chunk_idx[i] >= len(parsed_chunk):
                signals = ["Vision response missing this image; neutral score used."]
            else:
                signals = []
            room_scores.append(
                {
                    "room_type": img.room_type or "unknown",
                    "score": 50.0,
                    "weight": float(img.weight),
                    "signals": signals,
                    "red_flags": [],
                }
            )

        if len(parsed_chunk) < len(chunk_index_by_image_slot):
            logger.warning(
                "condition-score vision: model returned %s rows for %s images sent; padding with defaults.",
                len(parsed_chunk),
                len(chunk_index_by_image_slot),
            )

    return {"room_scores": room_scores, "cost_usd": round(total_cost, 6)}

