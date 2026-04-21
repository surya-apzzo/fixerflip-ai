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
    "dirty walls": "stains",
    "wall stains": "stains",
    "old bathroom": "old bathroom",
    "dated bathroom tile": "old tiles",
    "outdated bathroom": "old bathroom",
    "roof wear": "roof damage",
    "structural crack": "foundation cracks",
    "crack": "foundation cracks",
    "damaged flooring": "floor damage",
    "visible gaps between planks": "floor damage",
    "major crack": "major wall cracks",
    "major wall crack": "major wall cracks",
    "foundation crack": "foundation cracks",
    "structural issues": "structural damage",
    "plumbing issue": "plumbing issues",
    "electrical issue": "electrical issues",
    "hvac issue": "hvac issues",
    "old hvac": "hvac issues",
    "ceiling popcorn": "popcorn ceiling",
    "popcorn ceilings": "popcorn ceiling",
    "old appliances": "worn appliances",
    "worn appliance": "worn appliances",
    "dirty surface": "dirty surfaces",
    "peeling paint": "peeling exterior paint",
    "exterior peeling paint": "peeling exterior paint",
    "broken window": "broken windows",
    "smoke damage": "structural damage",
    "fire damage": "structural damage",
    "charred walls": "structural damage",
    "burn marks": "major wall cracks",
    "soot": "smoke damage",
}


# Severity multiplier
SEVERITY_MULTIPLIER = {
    "minor": 0.5,
    "moderate": 1.0,
    "severe": 1.7,
}

# Prevents dozens of low-weight detections from driving the score to ~0 (model over-listing).
MAX_ISSUE_PENALTY_PER_IMAGE = 80.0

# Positive Weights
POSITIVE_WEIGHTS = {
    "modern kitchen": 10,
    "updated kitchen": 8,
    "renovated bathroom": 12,
    "updated bathroom": 9,
    "hardwood floor": 8,
    "new flooring": 7,
    "fresh paint": 5,
    "new paint": 5,
    "good lighting": 4,
    "recessed lighting": 6,
    "new roof": 15,
    "new hvac": 14,
    "new windows": 10,
    "well maintained exterior": 6,
    "move-in ready": 12,
    "turnkey": 13,
}

# Positive Aliases
POSITIVE_ALIASES = {
    # Kitchen
    "remodeled kitchen": "modern kitchen",
    "updated kitchen": "updated kitchen",
    "gourmet kitchen": "modern kitchen",

    # Bathroom
    "updated bath": "updated bathroom",
    "remodeled bathroom": "renovated bathroom",

    # Flooring
    "hardwood flooring": "hardwood floor",
    "wood floors": "hardwood floor",
    "new floors": "new flooring",

    # Paint
    "freshly painted": "fresh paint",
    "recent paint": "new paint",

    # Condition
    "move in ready": "move-in ready",
    "fully updated": "turnkey",
    "fully renovated": "turnkey",
}

# Taxonomy used for validation of vision output (single source of truth for weights).
CANONICAL_POSITIVE_TYPES: frozenset[str] = frozenset(POSITIVE_WEIGHTS.keys())

# Overall room condition from the model (new | average | old). Applied after issue penalties.
CONDITION_SCORE_ADJUSTMENT: dict[str, float] = {
    "new": 4.0,
    "average": 0.0,
    "old": -12.0,
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
        # range: 0.5 -> 1.0
        return 0.5 + (_clamp(conf, 0.0, 1.0) * 0.5)

    def normalize_room(self, room: str) -> str:
        key = re.sub(r"\s+", " ", (room or "unknown").strip().lower())
        return ROOM_ALIASES.get(key, key or "unknown")

    def normalize_positive(self, label: str) -> str:
        key = re.sub(r"\s+", " ", (label or "").strip().lower())
        return POSITIVE_ALIASES.get(key, key)

    # Calculate the room score from the room detection.
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

        cond = (data.condition or "average").strip().lower()
        if cond not in CONDITION_SCORE_ADJUSTMENT:
            cond = "average"
        score += CONDITION_SCORE_ADJUSTMENT[cond]

        for p in data.positives:
            pos = self.normalize_positive(p.type)
            bonus = POSITIVE_WEIGHTS.get(pos, 0)
            score += bonus * _clamp(p.confidence, 0.0, 1.0)

        return _clamp(score, 0.0, 100.0)

    # Calculate the room score from the issues.
    def score_from_issues(
        self, issues: List[str], room_type: str = "unknown"
    ) -> ImageConditionResult:
        
        normalized_issues = [
            self.normalize_issue(i) for i in issues if i and str(i).strip()
        ]
        normalized_room = self.normalize_room(room_type)
        room_data = RoomDetection(
            room=normalized_room,
            issues=[
                IssueDetection(type=i, severity="moderate", confidence=1.0)
                for i in normalized_issues
            ],
            positives=[],
        )
        score = int(round(self.calculate_room_score(room_data)))
        return ImageConditionResult(
            condition_score=score,
            issues=normalized_issues,
            room_type=normalized_room,
        )

    # Calculate the room score from the room detections.
    def score_from_room_detections(
        self, detections: List[RoomDetection]
    ) -> ImageConditionResult:
        if not detections:
            return ImageConditionResult(
                condition_score=65, issues=[], room_type="unknown"
            )

        room_scores: dict[str, list[float]] = defaultdict(list)
        all_issues: list[str] = []
        seen_issues: set[str] = set()

        for d in detections:
            room = self.normalize_room(d.room)
            room_scores[room].append(self.calculate_room_score(d))
            for issue in d.issues:
                n = self.normalize_issue(issue.type)
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
                condition_score=final, issues=all_issues, room_type="unknown"
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
            condition_score=final, issues=all_issues, room_type=room_type
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
