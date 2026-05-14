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
from app.engine.renovation_engine.image_edit_engine import image_url_as_openai_vision_payload
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

# FIX 1: expanded to cover both old and new prompt condition labels
VALID_CONDITION = frozenset({"new", "good", "fair", "poor", "distressed", "average", "old"})

MAX_ISSUES_PER_IMAGE = 8
_SEVERITY_RANK = {"severe": 0, "moderate": 1, "minor": 2}
_VISION_FALLBACK_SCORE = 65
_VISION_MAX_RETRIES = 2
_VISION_RETRY_BACKOFF_SECONDS = 0.75
_OPENAI_VISION_TIMEOUT_SECONDS = 90.0

_VISION_DISABLED_REASON = "vision_unavailable"
_VISION_INVALID_OUTPUT_REASON = "invalid_model_output"
_VISION_REQUEST_FAILED_REASON = "vision_request_failed"
_VISION_EMPTY_IMAGE_URL_REASON = "empty_image_url"


def _load_condition_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        'Return JSON: {"room_type":"kitchen|bathroom|exterior|other|unknown",'
        '"issues":["short issue strings"]}'
    )


def _is_retryable_openai_exception(exc: Exception) -> bool:
    retryable_names = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
    return exc.__class__.__name__ in retryable_names


async def _wait_for_retry_backoff(attempt: int) -> None:
    await asyncio.sleep(_VISION_RETRY_BACKOFF_SECONDS * (2**attempt))


async def _analyze_single_image_url(image_url: str) -> tuple[RoomDetection | None, str | None, str | None]:
    """One image URL -> structured room detection via OpenAI."""
    if not (settings.OPENAI_VISION_ENABLED and settings.OPENAI_API_KEY):
        return None, _VISION_DISABLED_REASON, None

    primary_model = settings.default_openai_vision_model
    fallback_model = (settings.OPENAI_MODEL or "gpt-4o-mini").strip().lower()
    model_candidates = [primary_model]
    if fallback_model and fallback_model not in model_candidates:
        model_candidates.append(fallback_model)
    try:
        prompt = _load_condition_prompt()
        image_for_model = await image_url_as_openai_vision_payload(image_url)
        client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=_OPENAI_VISION_TIMEOUT_SECONDS,
            max_retries=0,
        )
        last_failed_model: str | None = None
        for model_name in model_candidates:
            for attempt in range(_VISION_MAX_RETRIES + 1):
                try:
                    response = await client.responses.create(
                        model=model_name,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {"type": "input_image", "image_url": image_for_model},
                                ],
                            }
                        ],
                        max_output_tokens=512,
                    )
                    raw = getattr(response, "output_text", None) or ""
                    try:
                        parsed = _parse_response_json(raw)
                    except ValueError:
                        return None, _VISION_INVALID_OUTPUT_REASON, model_name
                    if not isinstance(parsed, dict):
                        return None, _VISION_INVALID_OUTPUT_REASON, model_name
                    return _parse_room_analysis(parsed), None, model_name
                except Exception as exc:
                    last_failed_model = model_name
                    is_retryable = _is_retryable_openai_exception(exc)
                    if attempt < _VISION_MAX_RETRIES and is_retryable:
                        await _wait_for_retry_backoff(attempt)
                        continue
                    logger.warning(
                        "Renovation vision analysis failed for URL %s using model %s (attempt %s/%s): %s",
                        image_url,
                        model_name,
                        attempt + 1,
                        _VISION_MAX_RETRIES + 1,
                        exc,
                    )
                    break
        return None, _VISION_REQUEST_FAILED_REASON, last_failed_model or primary_model
    except Exception as exc:
        logger.warning("Renovation vision analysis failed for URL %s: %s", image_url, exc)
        return None, _VISION_REQUEST_FAILED_REASON, primary_model


async def analyze_renovation_image_url(image_url: str) -> ImageConditionResult:
    """Single image URL -> condition result (renovation flow uses one image at a time)."""
    url = (image_url or "").strip()
    if not url:
        return _build_fallback_condition_result(_VISION_EMPTY_IMAGE_URL_REASON)

    engine = ImageConditionEngine()
    one, fallback_reason, model_name = await _analyze_single_image_url(url)
    if one is not None:
        return engine.score_from_room_detections([one]).model_copy(
            update={
                "analysis_status": "ai_success",
                "model_used": model_name,
                "fallback_reason": None,
            }
        )

    return _build_fallback_condition_result(fallback_reason or _VISION_REQUEST_FAILED_REASON, model_name)


def _build_fallback_condition_result(reason: str, model_name: str | None = None) -> ImageConditionResult:
    return ImageConditionResult(
        condition_score=_VISION_FALLBACK_SCORE,
        issues=[],
        room_type="unknown",
        analysis_status="fallback",
        model_used=model_name,
        fallback_reason=reason,
    )


def _coerce_float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_issue_severity(value: str) -> str:
    s = (value or "moderate").strip().lower()
    return s if s in ALLOWED_SEVERITY else "moderate"


def _normalize_condition_label(value: Any) -> str:
    condition = str(value or "average").strip().lower()
    # FIX 1: now accepts all valid condition labels from both old and new prompt
    if condition not in VALID_CONDITION:
        return "average"
    return condition


def _parse_issue_detections(engine: ImageConditionEngine, issues_raw: Any) -> list[IssueDetection]:
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
                continue
            severity = _normalize_issue_severity(str(item.get("severity") or "moderate"))
            conf = _coerce_float_or_default(item.get("confidence"), 0.8)
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
                continue
            issues.append(IssueDetection(type=canonical, severity="moderate", confidence=0.8))

    issues.sort(
        key=lambda i: (_SEVERITY_RANK.get(i.severity, 1), -i.confidence),
    )
    return issues[:MAX_ISSUES_PER_IMAGE]




def _upsert_issue(
    issues: list[IssueDetection],
    *,
    issue_type: str,
    severity: str,
    confidence: float,
) -> None:
    for idx, existing in enumerate(issues):
        if existing.type != issue_type:
            continue
        merged_severity = severity
        if _SEVERITY_RANK.get(existing.severity, 1) < _SEVERITY_RANK.get(severity, 1):
            merged_severity = existing.severity
        merged_confidence = max(existing.confidence, confidence)
        issues[idx] = IssueDetection(
            type=issue_type,
            severity=merged_severity,
            confidence=merged_confidence,
        )
        return
    issues.append(
        IssueDetection(
            type=issue_type,
            severity=severity,
            confidence=max(0.0, min(1.0, confidence)),
        )
    )


def _enrich_fire_scene_issues(room: str, issues: list[IssueDetection]) -> list[IssueDetection]:
    issue_types = {i.type for i in issues}
    has_fire_or_smoke = "fire damage" in issue_types or "smoke damage" in issue_types
    if not has_fire_or_smoke:
        return issues

    if "damaged ceiling" not in issue_types:
        _upsert_issue(
            issues,
            issue_type="damaged ceiling",
            severity="severe",
            confidence=0.75,
        )
    if "minor wall damage" not in issue_types:
        _upsert_issue(
            issues,
            issue_type="minor wall damage",
            severity="severe",
            confidence=0.7,
        )
    if room == "kitchen" and "outdated cabinets" not in issue_types:
        _upsert_issue(
            issues,
            issue_type="outdated cabinets",
            severity="severe",
            confidence=0.7,
        )
    if "electrical issues" not in issue_types:
        _upsert_issue(
            issues,
            issue_type="electrical issues",
            severity="severe",
            confidence=0.68,
        )
    if room in {"kitchen", "bathroom"} and "plumbing issues" not in issue_types:
        _upsert_issue(
            issues,
            issue_type="plumbing issues",
            severity="moderate",
            confidence=0.55,
        )

    issues.sort(
        key=lambda i: (_SEVERITY_RANK.get(i.severity, 1), -i.confidence),
    )
    return issues[:MAX_ISSUES_PER_IMAGE]

def _parse_positive_detections(engine: ImageConditionEngine, positives_raw: Any) -> list[PositiveDetection]:
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
                continue
            conf = _coerce_float_or_default(item.get("confidence"), 0.8)
            positives.append(
                PositiveDetection(type=canonical_pos, confidence=max(0.0, min(1.0, conf)))
            )
        elif isinstance(item, str) and item.strip():
            canonical_pos = engine.normalize_positive(item.strip())
            if canonical_pos not in CANONICAL_POSITIVE_TYPES:
                continue
            positives.append(PositiveDetection(type=canonical_pos, confidence=0.8))
    return positives


def _parse_room_analysis(parsed: dict) -> RoomDetection:
    engine = ImageConditionEngine()
    room = engine.normalize_room(str(parsed.get("room") or parsed.get("room_type") or "unknown"))
    condition = _normalize_condition_label(parsed.get("condition"))
    issues = _parse_issue_detections(engine, parsed.get("issues") or [])
    issues = _enrich_fire_scene_issues(room, issues)
    positives = _parse_positive_detections(engine, parsed.get("positives") or [])

    # FIX 2: extract overall_score from model output and pass through to RoomDetection
    # This uses the model's own score signal instead of throwing it away.
    # Only applied if RoomDetection schema supports overall_score field.
    overall_score_raw = parsed.get("overall_score") or parsed.get("score")
    try:
        overall_score = int(overall_score_raw) if overall_score_raw is not None else None
        if overall_score is not None:
            overall_score = max(1, min(10, overall_score))
    except (TypeError, ValueError):
        overall_score = None

    # FIX 3: extract renovation_scope hint from model output
    renovation_scope = str(parsed.get("renovation_scope") or "").strip().lower() or None
    valid_scopes = {"cosmetic", "moderate", "heavy", "gut"}
    if renovation_scope not in valid_scopes:
        renovation_scope = None

    # Build kwargs conditionally so we don't break schema if fields are not yet added
    extra_kwargs: dict = {}
    if overall_score is not None:
        extra_kwargs["overall_score"] = overall_score
    if renovation_scope is not None:
        extra_kwargs["renovation_scope"] = renovation_scope

    return RoomDetection(
        room=room or "unknown",
        condition=condition,
        issues=issues,
        positives=positives,
        **extra_kwargs,
    )


def _parse_response_json(text: str) -> Any:
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
