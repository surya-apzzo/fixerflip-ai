from __future__ import annotations

from typing import Any


def _grade_from_score(score: float) -> str:
    if score >= 75:
        return "Poor condition - high flip upside"
    if score >= 55:
        return "Fair condition - moderate rehab needed"
    if score >= 35:
        return "Good condition - cosmetic work only"
    return "Move-in ready - low flip upside"


def aggregate(vision_result: dict[str, Any]) -> dict[str, Any]:
    room_scores = vision_result.get("room_scores", [])
    if not room_scores:
        combined = 50.0
        vision_score = None
    else:
        total_weight = sum(float(r.get("weight", 0.0)) for r in room_scores)
        if total_weight > 0:
            vision_score = sum(float(r.get("score", 50.0)) * float(r.get("weight", 0.0)) for r in room_scores) / total_weight
        else:
            vision_score = sum(float(r.get("score", 50.0)) for r in room_scores) / max(len(room_scores), 1)
        combined = vision_score

    red_flags: list[str] = []
    positives: list[str] = []
    for item in room_scores:
        positives.extend([str(s) for s in item.get("signals", []) if str(s).strip()])
        red_flags.extend([str(s) for s in item.get("red_flags", []) if str(s).strip()])

    if red_flags:
        combined = max(0.0, combined - 10.0)

    dedup_positives = list(dict.fromkeys(positives))[:4]
    dedup_red_flags = list(dict.fromkeys(red_flags))[:3]

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

