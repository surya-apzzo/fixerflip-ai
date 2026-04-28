"""
Image Condition AI — renovation engine.
Converts vision-detected issues into a 0–100 condition score and normalized output.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List

from app.core.rules_config import CANONICAL_ISSUE_TYPES, ISSUE_WEIGHTS
from app.schemas import (
    ImageConditionResult,
    IssueDetection,
    RoomDetection,
)

# Normalized Issues
NORMALIZED_ISSUES = {
    "old cabinets": "outdated cabinets",
    "dated kitchen": "outdated cabinets",
    "dated cabinets": "outdated cabinets",
    "kitchen cabinets": "outdated cabinets",
    "dated flooring": "outdated flooring",
    "dirty walls": "stains",
    "wall stains": "stains",
    "old bathroom": "old bathroom",
    "dated bathroom tile": "old tiles",
    "outdated bathroom": "old bathroom",
    "roof wear": "roof damage",
    #a: generic "crack" alone is minor wall damage, not foundation
    "structural crack": "major wall cracks",
    "crack": "minor wall damage",
    "damaged flooring": "floor damage",
    "visible gaps between planks": "floor damage",
    "major crack": "major wall cracks",
    "major wall crack": "major wall cracks",
    # b: "foundation cracks" is not in the prompt allowed list.
    # Closest canonical allowed type is "major wall cracks".
    "foundation crack": "major wall cracks",
    "foundation cracks": "major wall cracks",
    "structural issues": "structural damage",
    "plumbing issue": "water damage",
    "electrical issue": "minor wall damage",
    "hvac issue": "minor wall damage",
    "old hvac": "minor wall damage",
    "ceiling popcorn": "popcorn ceiling",
    "popcorn ceilings": "popcorn ceiling",
    "ceiling damage": "damaged ceiling",
    "ceiling crack": "damaged ceiling",
    "grout missing": "missing grout",
    "missing tile grout": "missing grout",
    "dirty tile grout": "dirty grout",
    "wood rot": "rotted wood",
    "rotting wood": "rotted wood",
    "broken fixture": "broken fixtures",
    "fixture damage": "broken fixtures",
    "appliance damage": "damaged appliances",
    "damaged appliance": "damaged appliances",
    "old appliances": "stains",
    "worn appliance": "stains",
    "dirty surface": "stains",
    "peeling paint": "peeling exterior paint",
    "exterior peeling paint": "peeling exterior paint",
    "broken window": "broken windows",
    # c: smoke and fire map to their own canonical types, NOT structural damage
    "smoke damage": "smoke damage",
    "fire damage": "fire damage",
    "charred walls": "fire damage",
    "burn marks": "smoke damage",
    "soot": "smoke damage",
    # True structural defects only
    "exposed framing": "structural damage",
    "collapsed ceiling": "structural damage",
    "missing wall section": "structural damage",
}


# Severity multiplier
SEVERITY_MULTIPLIER = {
    "minor": 0.7,
    "moderate": 1.0,
    "severe": 1.4,
}

# Prevents dozens of low-weight detections from driving the score to ~0 (model over-listing).
MAX_ISSUE_PENALTY_PER_IMAGE = 80.0

# Positive Weights
POSITIVE_WEIGHTS = {
    "modern kitchen": 10,
    "updated kitchen": 8,
    "renovated bathroom": 12,
    "modern bathroom": 10,
    "updated bathroom": 9,
    "hardwood floor": 8,
    "new flooring": 7,
    "updated flooring": 7,
    "fresh paint": 5,
    "new paint": 5,
    "good lighting": 4,
    "good natural light": 4,
    "recessed lighting": 6,
    "updated fixtures": 5,
    "new appliances": 6,
    "clean condition": 5,
    "new roof": 15,
    "new hvac": 14,
    "new windows": 10,
    "well maintained exterior": 6,
    "move-in ready": 12,
    "turnkey": 13,
}

# Positive Aliases
POSITIVE_ALIASES = {
    "remodeled kitchen": "modern kitchen",
    "updated kitchen": "updated kitchen",
    "gourmet kitchen": "modern kitchen",
    "updated bath": "updated bathroom",
    "modern bath": "modern bathroom",
    "remodeled bathroom": "renovated bathroom",
    "hardwood flooring": "hardwood floor",
    "wood floors": "hardwood floor",
    "new floors": "new flooring",
    "updated floors": "updated flooring",
    "freshly painted": "fresh paint",
    "recent paint": "new paint",
    "natural light": "good natural light",
    "daylight": "good natural light",
    "updated light fixtures": "updated fixtures",
    "updated hardware": "updated fixtures",
    "new appliance": "new appliances",
    "clean interior": "clean condition",
    "clean room": "clean condition",
    "move in ready": "move-in ready",
    "fully updated": "turnkey",
    "fully renovated": "turnkey",
}


CANONICAL_POSITIVE_TYPES: frozenset[str] = frozenset(POSITIVE_WEIGHTS.keys())


CONDITION_SCORE_ADJUSTMENT: dict[str, float] = {
    "new": 8.0,
    "good": 4.0,
    "fair": 0.0,
    "average": 0.0,
    "poor": -12.0,
    "old": -12.0,
    "distressed": -18.0,
}

# Room Weights
ROOM_WEIGHTS = {
    "kitchen": 0.30,
    "bathroom": 0.25,
    "living_room": 0.20,
    "living": 0.20,
    "hall": 0.20,
    "bedroom": 0.10,
    "exterior": 0.15,
    "basement": 0.10,
    "other": 0.10,
    "unknown": 0.10,
}

# Room Aliases
ROOM_ALIASES = {
    "hall": "living_room",
    "hallway": "living_room",
    "living room": "living_room",
    "lounge": "living_room",
    "outside": "exterior",
    "outdoor": "exterior",
    "roof": "exterior",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "bath": "bathroom",
    "living": "living_room",
    "basement": "basement",
}

__all__ = [
    "CANONICAL_ISSUE_TYPES",
    "CANONICAL_POSITIVE_TYPES",
    "ImageConditionEngine",
]


class ImageConditionEngine:
    """Production-style scoring from structured detections."""

    def normalize_issue(self, issue: str) -> str:
        key = re.sub(r"\s+", " ", issue.strip().lower())
        return NORMALIZED_ISSUES.get(key, key)

    def confidence_adjust(self, conf: float) -> float:
        return 0.3 + (_clamp(conf, 0.0, 1.0) * 0.7)

    def normalize_room(self, room: str) -> str:
        key = re.sub(r"\s+", " ", (room or "unknown").strip().lower())
        return ROOM_ALIASES.get(key, key or "unknown")

    def normalize_positive(self, label: str) -> str:
        key = re.sub(r"\s+", " ", (label or "").strip().lower())
        return POSITIVE_ALIASES.get(key, key)

    def calculate_room_score(self, data: RoomDetection) -> float:
        """
        Start from 100, subtract capped weighted issue penalties, apply model `condition`
        adjustment, then add positive bonuses. Unknown issue types use default weight 5.
        """
        total_penalty = 0.0
        for issue in data.issues:
            normalized = self.normalize_issue(issue.type)
            base_penalty = ISSUE_WEIGHTS.get(normalized, 5)
            severity_key = issue.severity.strip().lower()
            severity_factor = SEVERITY_MULTIPLIER.get(severity_key, 1.0)
            conf_factor = self.confidence_adjust(issue.confidence)
            total_penalty += base_penalty * severity_factor * conf_factor

        total_penalty = min(total_penalty, MAX_ISSUE_PENALTY_PER_IMAGE)
        score = 100.0 - total_penalty

        cond = (data.condition or "fair").strip().lower()
        if cond not in CONDITION_SCORE_ADJUSTMENT:
            cond = "fair"
        score += CONDITION_SCORE_ADJUSTMENT[cond]

        for p in data.positives:
            pos = self.normalize_positive(p.type)
            bonus = POSITIVE_WEIGHTS.get(pos, 0)
            score += bonus * _clamp(p.confidence, 0.0, 1.0)

        return _clamp(score, 0.0, 100.0)

    def score_from_room_detections(
        self, detections: List[RoomDetection]
    ) -> ImageConditionResult:
        if not detections:
            return ImageConditionResult(
                condition_score=65, issues=[], room_type="unknown"
            )

        room_scores: dict[str, list[float]] = defaultdict(list)
        all_issues: list[str] = []
        all_issue_details: list[IssueDetection] = []
        seen_issues: set[str] = set()
        issue_counts: dict[str, int] = defaultdict(int)

        for d in detections:
            room = self.normalize_room(d.room)
            room_scores[room].append(self.calculate_room_score(d))
            for issue in d.issues:
                n = self.normalize_issue(issue.type)
                severity = issue.severity.strip().lower() or "moderate"
                confidence = _clamp(issue.confidence, 0.0, 1.0)

                all_issue_details.append(
                    IssueDetection(type=n, severity=severity, confidence=confidence)
                )
                issue_counts[n] = issue_counts.get(n, 0) + 1

                if n not in seen_issues:
                    seen_issues.add(n)
                    all_issues.append(n)

        final = 0.0
        applied_weight = 0.0
        for room, scores in room_scores.items():
            room_avg = sum(scores) / max(1, len(scores))
            weight = ROOM_WEIGHTS.get(room, 0.10)
            final += room_avg * weight
            applied_weight += weight

        if applied_weight > 0:
            final = final / applied_weight

        final = int(round(_clamp(final, 0.0, 100.0)))
        if not room_scores:
            return ImageConditionResult(
                condition_score=final,
                issues=all_issues,
                issue_details=all_issue_details,
                issue_counts=issue_counts,
                room_type="unknown"
            )

        ranked_rooms = sorted(
            room_scores.keys(),
            key=lambda room: (len(room_scores[room]), ROOM_WEIGHTS.get(room, 0.10)),
            reverse=True,
        )
        room_type = (
            ranked_rooms[0] if len(ranked_rooms) == 1 else ",".join(ranked_rooms)
        )
        return ImageConditionResult(
            condition_score=final,
            issues=all_issues,
            issue_details=all_issue_details,
            issue_counts=issue_counts,
            room_type=room_type,
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
