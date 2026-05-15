from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from openai import APITimeoutError, AsyncOpenAI, RateLimitError

from app.core.config import settings
from app.engine.image_condition.services.image_filter import (
    FilteredImage,
    normalize_house_room_type,
)
from app.engine.image_condition.services.vision_image_payload import (
    bytes_to_openai_vision_data_url,
    image_url_to_openai_vision_data_url,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "property_condition_vision_prompt.txt"

# Matches usage billing in this module (gpt-4o-mini class models, USD per 1M tokens).
_INPUT_USD_PER_1M_TOKENS = 0.15
_OUTPUT_USD_PER_1M_TOKENS = 0.60

_VISION_PROMPT_FALLBACK = """
Score each property photo 0-100 (100 = excellent/move-in, 0 = severe distress).
Good well-maintained rooms should be 90+. Poor or damaged rooms should be below 55.
Return one object per photo in the SAME ORDER. Return ONLY a JSON array with
room_type, condition_score, signals, red_flags.
""".strip()

_RETRYABLE_VISION_ERRORS = (APITimeoutError, RateLimitError)
_DOWNLOAD_CONCURRENCY = 4


class ImageDownloadError(Exception):
    """Raised when every selected listing photo failed to download (e.g. MLS 403)."""

    def __init__(self, *, selected: int, prepared: int, failed: int) -> None:
        self.selected = selected
        self.prepared = prepared
        self.failed = failed
        super().__init__(
            f"Could not download any of {selected} image(s) for vision scoring "
            f"({failed} failed). MLS/CDN URLs often block server access."
        )


def _load_vision_prompt() -> str:
    if _PROMPT_PATH.exists():
        content = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        if content:
            return content
    return _VISION_PROMPT_FALLBACK


def _chunk_images(images: list[FilteredImage]) -> list[list[FilteredImage]]:
    chunk_size = max(1, int(settings.CONDITION_SCORE_VISION_CHUNK_SIZE))
    return [images[i : i + chunk_size] for i in range(0, len(images), chunk_size)]


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("condition-score: vision returned invalid JSON: %s", exc)
        return []
    return parsed if isinstance(parsed, list) else []


def _build_room_score_row(
    *,
    clip_room_type: str,
    score: float,
    signals: list[str],
    red_flags: list[str],
    vision_room_type: str = "",
) -> dict[str, Any] | None:
    canonical = normalize_house_room_type(vision_room_type or clip_room_type, clip_fallback=clip_room_type)
    if canonical is None:
        return None
    return {
        "room_type": canonical,
        "clip_room_type": clip_room_type,
        "score": round(max(0.0, min(100.0, score)), 1),
        "signals": signals,
        "red_flags": red_flags,
    }


def _max_output_tokens_for_chunk(image_count: int) -> int:
    return min(4096, 200 + max(image_count, 1) * 120)


def _estimate_cost_usd(*, input_tokens: float, output_tokens: float) -> float:
    return (input_tokens / 1_000_000) * _INPUT_USD_PER_1M_TOKENS + (
        output_tokens / 1_000_000
    ) * _OUTPUT_USD_PER_1M_TOKENS


async def _prepare_vision_payloads(
    chunk: list[FilteredImage],
) -> tuple[list[int], list[str]]:
    sem = asyncio.Semaphore(_DOWNLOAD_CONCURRENCY)

    async def _one(index: int, image: FilteredImage) -> tuple[int, str | None]:
        async with sem:
            try:
                if image.image_bytes:
                    payload_url = await asyncio.to_thread(
                        bytes_to_openai_vision_data_url, image.image_bytes
                    )
                else:
                    payload_url = await image_url_to_openai_vision_data_url(image.image_url)
                return index, payload_url
            except Exception as exc:
                logger.warning(
                    "condition-score: skipping image (cannot prepare for vision): idx=%s url=%s err=%s",
                    index,
                    (image.image_url or "")[:120],
                    exc,
                )
                return index, None

    results = await asyncio.gather(*[_one(i, img) for i, img in enumerate(chunk)])
    indices: list[int] = []
    payloads: list[str] = []
    for index, payload_url in results:
        if payload_url and payload_url.strip():
            indices.append(index)
            payloads.append(payload_url)
    return indices, payloads


def _is_retryable_vision_error(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_VISION_ERRORS):
        return True
    return exc.__class__.__name__ in {"APIConnectionError", "InternalServerError"}


async def _score_vision_chunk(
    client: AsyncOpenAI,
    *,
    vision_prompt: str,
    chunk: list[FilteredImage],
    chunk_indices: list[int],
    payload_urls: list[str],
) -> tuple[list[dict[str, Any]], float]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": vision_prompt}]
    for payload_url in payload_urls:
        content.append({"type": "input_image", "image_url": payload_url})

    max_retries = max(0, int(settings.OPENAI_CONDITION_SCORE_VISION_MAX_RETRIES))
    last_exc: Exception | None = None
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.responses.create(
                model=settings.default_openai_vision_model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=_max_output_tokens_for_chunk(len(payload_urls)),
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries and _is_retryable_vision_error(exc):
                await asyncio.sleep(1.5 * (2**attempt))
                continue
            raise
    if response is None:
        raise last_exc or RuntimeError("Vision chunk failed without exception.")

    raw = getattr(response, "output_text", "") or ""
    parsed_chunk = _parse_json_array(raw)

    chunk_cost = 0.0
    usage = getattr(response, "usage", None)
    if usage is not None:
        chunk_cost = _estimate_cost_usd(
            input_tokens=float(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=float(getattr(usage, "output_tokens", 0) or 0),
        )

    filled: list[dict[str, Any] | None] = [None] * len(chunk)
    for slot_idx, result in enumerate(parsed_chunk):
        if slot_idx >= len(chunk_indices):
            break
        chunk_idx = chunk_indices[slot_idx]
        if not isinstance(result, dict):
            result = {}
        try:
            score = float(result.get("condition_score", 50))
        except (TypeError, ValueError):
            score = 50.0
        filled[chunk_idx] = _build_room_score_row(
            clip_room_type=chunk[chunk_idx].room_type or "unknown",
            score=score,
            signals=[str(s) for s in result.get("signals", []) if str(s).strip()],
            red_flags=[str(s) for s in result.get("red_flags", []) if str(s).strip()],
            vision_room_type=str(result.get("room_type", "")),
        )

    room_scores: list[dict[str, Any]] = []
    for i, row in enumerate(filled):
        if row is not None:
            room_scores.append(row)

    return room_scores, chunk_cost


async def score_from_images(selected_images: list[FilteredImage]) -> dict[str, Any]:
    if not selected_images:
        return {"room_scores": [], "cost_usd": 0.0}
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for condition-score vision analysis.")

    timeout_s = float(settings.OPENAI_CONDITION_SCORE_VISION_TIMEOUT_SECONDS)
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=timeout_s,
        max_retries=0,
    )
    room_scores: list[dict[str, Any]] = []
    total_cost = 0.0
    images_prepared = 0
    images_failed = 0
    vision_prompt = _load_vision_prompt()
    chunks = _chunk_images(selected_images)
    logger.info(
        "condition-score vision: %s images, %s chunk(s), chunk_size=%s, timeout=%ss, model=%s",
        len(selected_images),
        len(chunks),
        settings.CONDITION_SCORE_VISION_CHUNK_SIZE,
        timeout_s,
        settings.default_openai_vision_model,
    )

    for chunk_num, chunk in enumerate(chunks, start=1):
        chunk_indices, payload_urls = await _prepare_vision_payloads(chunk)
        images_failed += len(chunk) - len(payload_urls)
        if not payload_urls:
            logger.warning(
                "condition-score vision chunk %s/%s: no downloadable images in chunk of %s",
                chunk_num,
                len(chunks),
                len(chunk),
            )
            continue

        images_prepared += len(payload_urls)
        logger.info(
            "condition-score vision chunk %s/%s: calling OpenAI with %s images",
            chunk_num,
            len(chunks),
            len(payload_urls),
        )
        chunk_scores, chunk_cost = await _score_vision_chunk(
            client,
            vision_prompt=vision_prompt,
            chunk=chunk,
            chunk_indices=chunk_indices,
            payload_urls=payload_urls,
        )
        room_scores.extend(chunk_scores)
        total_cost += chunk_cost
        logger.info(
            "condition-score vision chunk %s/%s done cost_usd=%.4f",
            chunk_num,
            len(chunks),
            chunk_cost,
        )

    if not room_scores and selected_images:
        raise ImageDownloadError(
            selected=len(selected_images),
            prepared=images_prepared,
            failed=images_failed or len(selected_images),
        )

    return {
        "room_scores": room_scores,
        "cost_usd": round(total_cost, 6),
        "images_prepared": images_prepared,
        "images_failed": images_failed,
    }
