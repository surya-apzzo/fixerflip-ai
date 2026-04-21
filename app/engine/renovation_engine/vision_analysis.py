"""
Vision call for renovation image condition: OpenAI -> JSON -> ImageConditionEngine.
Used in the renovation estimate endpoint.
Converts vision-detected issues into a 0–100 condition score and normalized output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.engine.renovation_engine.image_condition_engine import (
    CANONICAL_ISSUE_TYPES,
    CANONICAL_POSITIVE_TYPES,
    ImageConditionEngine,
)
from app.schemas import (
    ImageConditionResult,
    IssueDetection,
    PositiveDetection,
    RoomDetection,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "renovation_image_condition_prompt.txt"
ALLOWED_SEVERITY = frozenset({"minor", "moderate", "severe"})
VALID_CONDITION = frozenset({"new", "average", "old"})
MAX_ISSUES_PER_IMAGE = 8
_SEVERITY_RANK = {"severe": 0, "moderate": 1, "minor": 2}
_VISION_FALLBACK_SCORE = 65
_VISION_MAX_RETRIES = 2
_VISION_RETRY_BACKOFF_SECONDS = 0.75

# Reasons for fallback
_VISION_DISABLED_REASON = "vision_unavailable"
_VISION_INVALID_OUTPUT_REASON = "invalid_model_output"
_VISION_REQUEST_FAILED_REASON = "vision_request_failed"
_VISION_EMPTY_IMAGE_URL_REASON = "empty_image_url"


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        'Return JSON: {"room_type":"kitchen|bathroom|exterior|other|unknown",'
        '"issues":["short issue strings"]}'
    )


def _is_retryable_openai_error(exc: Exception) -> bool:
    # Avoid tight coupling to OpenAI exception classes; class names are stable enough for retry gating.
    retryable_names = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
    return exc.__class__.__name__ in retryable_names


async def _sleep_with_backoff(attempt: int) -> None:
    await asyncio.sleep(_VISION_RETRY_BACKOFF_SECONDS * (2**attempt))

# This function is used in the renovation estimate endpoint.

async def analyze_single_image_url(image_url: str) -> tuple[RoomDetection | None, str | None, str | None]:
    """One image URL -> structured room detection via OpenAI."""
    if not (settings.OPENAI_VISION_ENABLED and settings.OPENAI_API_KEY):
        return None, _VISION_DISABLED_REASON, None

    primary_model = settings.default_openai_vision_model
    fallback_model = "gpt-4o-mini"
    try:
        prompt = _load_prompt()
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        last_failed_model: str | None = None
        for model_name in [primary_model, fallback_model]:
            for attempt in range(_VISION_MAX_RETRIES + 1):
                try:
                    response = await client.responses.create(
                        model=model_name,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {"type": "input_image", "image_url": image_url},
                                ],
                            }
                        ],
                        max_output_tokens=512,
                    )
                    raw = getattr(response, "output_text", None) or ""
                    try:
                        parsed = _parse_json_response(raw)
                    except ValueError:
                        logger.warning("Vision returned invalid JSON for URL %s", image_url)
                        return None, _VISION_INVALID_OUTPUT_REASON, model_name
                    if not isinstance(parsed, dict):
                        logger.warning("Vision returned non-object JSON for URL %s", image_url)
                        return None, _VISION_INVALID_OUTPUT_REASON, model_name
                    return _parse_room_detection(parsed), None, model_name
                except Exception as exc:
                    last_failed_model = model_name
                    logger.warning(
                        "Renovation vision analysis failed for URL %s using model %s (attempt %s/%s): %s",
                        image_url,
                        model_name,
                        attempt + 1,
                        _VISION_MAX_RETRIES + 1,
                        exc,
                    )
                    if attempt < _VISION_MAX_RETRIES and _is_retryable_openai_error(exc):
                        await _sleep_with_backoff(attempt)
                        continue
                    break
        return None, _VISION_REQUEST_FAILED_REASON, last_failed_model or primary_model
    except Exception as exc:
        logger.warning("Renovation vision analysis failed for URL %s: %s", image_url, exc)
        return None, _VISION_REQUEST_FAILED_REASON, primary_model


# This function is used in the renovation estimate endpoint.
async def analyze_renovation_image_url(image_url: str) -> ImageConditionResult:
    """Single image URL -> condition result (renovation flow uses one image at a time)."""
    url = (image_url or "").strip()
    if not url:
        return _fallback_image_condition_result(_VISION_EMPTY_IMAGE_URL_REASON)

    engine = ImageConditionEngine()
    one, fallback_reason, model_name = await analyze_single_image_url(url)
    if one is not None:
        return engine.score_from_room_detections([one]).model_copy(
            update={
                "analysis_status": "ai_success",
                "model_used": model_name,
                "fallback_reason": None,
            }
        )

    return _fallback_image_condition_result(fallback_reason or _VISION_REQUEST_FAILED_REASON, model_name)

# Fallback function for the renovation estimate endpoint.
def _fallback_image_condition_result(reason: str, model_name: str | None = None) -> ImageConditionResult:
    return ImageConditionResult(
        condition_score=_VISION_FALLBACK_SCORE,
        issues=[],
        room_type="unknown",
        analysis_status="fallback",
        model_used=model_name,
        fallback_reason=reason,
    )


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_severity(value: str) -> str:
    s = (value or "moderate").strip().lower()
    return s if s in ALLOWED_SEVERITY else "moderate"


def _normalize_condition(value: Any) -> str:
    condition = str(value or "average").strip().lower()
    if condition not in VALID_CONDITION:
        return "average"
    return condition


# parse issues from the vision analysis.

def _parse_issues(engine: ImageConditionEngine, issues_raw: Any) -> list[IssueDetection]:
    if not isinstance(issues_raw, list):
        issues_raw = []

    issues: list[IssueDetection] = []
    for item in issues_raw:
        if isinstance(item, dict):
            issue_type = str(item.get("type") or "").strip()
            if not issue_type:
                continue
            canonical = engine.normalize_issue(issue_type)
            if canonical not in CANONICAL_ISSUE_TYPES:
                logger.warning("Vision issue label not in catalog, ignored: %s", issue_type)
                continue
            severity = _normalize_severity(str(item.get("severity") or "moderate"))
            conf = _safe_float(item.get("confidence"), 0.8)
            issues.append(
                IssueDetection(
                    type=canonical,
                    severity=severity,
                    confidence=max(0.0, min(1.0, conf)),
                )
            )
        elif isinstance(item, str) and item.strip():
            canonical = engine.normalize_issue(item.strip())
            if canonical not in CANONICAL_ISSUE_TYPES:
                logger.warning("Vision issue label not in catalog, ignored: %s", item)
                continue
            issues.append(IssueDetection(type=canonical, severity="moderate", confidence=0.8))

    issues.sort(
        key=lambda i: (_SEVERITY_RANK.get(i.severity, 1), -i.confidence),
    )
    return issues[:MAX_ISSUES_PER_IMAGE]


# parse positives from the vision analysis.

def _parse_positives(engine: ImageConditionEngine, positives_raw: Any) -> list[PositiveDetection]:
    if not isinstance(positives_raw, list):
        positives_raw = []

    positives: list[PositiveDetection] = []
    for item in positives_raw:
        if isinstance(item, dict):
            pos_type = str(item.get("type") or "").strip()
            if not pos_type:
                continue
            canonical_pos = engine.normalize_positive(pos_type)
            if canonical_pos not in CANONICAL_POSITIVE_TYPES:
                logger.warning("Vision positive label not in catalog, ignored: %s", pos_type)
                continue
            conf = _safe_float(item.get("confidence"), 0.8)
            positives.append(
                PositiveDetection(type=canonical_pos, confidence=max(0.0, min(1.0, conf)))
            )
        elif isinstance(item, str) and item.strip():
            canonical_pos = engine.normalize_positive(item.strip())
            if canonical_pos not in CANONICAL_POSITIVE_TYPES:
                logger.warning("Vision positive label not in catalog, ignored: %s", item)
                continue
            positives.append(PositiveDetection(type=canonical_pos, confidence=0.8))
    return positives

# parse room detection from the vision analysis.
def _parse_room_detection(parsed: dict) -> RoomDetection:
    engine = ImageConditionEngine()
    room = engine.normalize_room(str(parsed.get("room") or parsed.get("room_type") or "unknown"))
    condition = _normalize_condition(parsed.get("condition"))
    issues = _parse_issues(engine, parsed.get("issues") or [])
    positives = _parse_positives(engine, parsed.get("positives") or [])
    return RoomDetection(room=room or "unknown", condition=condition, issues=issues, positives=positives)

# parse JSON response from the vision analysis.
def _parse_json_response(text: str) -> Any:
    """Parse model output: strict JSON, or extract JSON from markdown / prose."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("could not parse JSON from model output")


