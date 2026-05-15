from __future__ import annotations

from typing import Any

from app.engine.image_condition.services.image_filter import normalize_house_room_type


def _grade_from_score(score: float) -> str:
    if score >= 90:
        return "Excellent - move-in ready"
    if score >= 75:
        return "Good condition - cosmetic work only"
    if score >= 55:
        return "Fair condition - moderate rehab needed"
    return "Poor condition - high flip upside"


def _normalize_room_score_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in raw_rows:
        canonical = normalize_house_room_type(
            str(item.get("room_type", "")),
            clip_fallback=str(item.get("clip_room_type", "")),
        )
        if canonical is None:
            continue
        normalized.append(
            {
                "room_type": canonical,
                "score": float(item.get("score", 50.0)),
                "signals": [str(s) for s in item.get("signals", []) if str(s).strip()],
                "red_flags": [str(s) for s in item.get("red_flags", []) if str(s).strip()],
            }
        )
    return normalized


def aggregate(vision_result: dict[str, Any]) -> dict[str, Any]:
    """
    Final property condition = simple average of per-image vision scores (0-100).

    Each photo is scored independently by the vision model; we average those scores.
    """
    room_scores = _normalize_room_score_rows(list(vision_result.get("room_scores", [])))
    if not room_scores:
        combined = 50.0
        vision_score = None
    else:
        per_image_scores = [float(row["score"]) for row in room_scores]
        vision_score = sum(per_image_scores) / len(per_image_scores)
        combined = vision_score

    positives: list[str] = []
    red_flags: list[str] = []
    for item in room_scores:
        positives.extend(item.get("signals", []))
        red_flags.extend(item.get("red_flags", []))

    dedup_positives = list(dict.fromkeys(positives))[:8]
    dedup_red_flags = list(dict.fromkeys(red_flags))[:5]

    return {
        "condition_score": round(combined, 1),
        "grade": _grade_from_score(combined),
        "text_score": 0.0,
        "vision_score": round(vision_score, 1) if vision_score is not None else None,
        "room_scores": room_scores,
        "positive_signals": dedup_positives,
        "caution_signals": [],
        "red_flags": dedup_red_flags,
        "cost_usd": round(float(vision_result.get("cost_usd", 0.0)), 6),
    }
