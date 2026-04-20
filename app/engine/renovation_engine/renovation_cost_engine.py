from __future__ import annotations

import re
from typing import List, Literal, Tuple

from app.core.rules_config import COST_MAP, ISSUE_COST_WEIGHT
from app.engine.renovation_engine.score_from_issues import (
    compute_gap_score,
    compute_renovation_age_detection,
)
from app.engine.renovation_engine.schemas import (
    ImpactedElementDetail,
    RenovationEstimate,
    RenovationEstimateInput,
    RenovationLineItem,
)


# Main function to estimate renovation cost
def estimate_renovation_cost(data: RenovationEstimateInput) -> RenovationEstimate:
    renovation_class = _renovation_class(data.condition_score, data.issues)
    severity_factor = _compute_severity_score(data.issues)
    quality_factor = _quality_level_factor(data.desired_quality_level)
    issue_factor = _issue_factor(data.issues)
    location_factor = _clamp(((data.labor_index * 0.6) + (data.material_index * 0.4)), 0.7, 2.5)
    combined_factor = severity_factor * quality_factor * issue_factor * location_factor

    line_items = _build_line_items(data, combined_factor)
    subtotal_low = sum(item.cost_low for item in line_items)
    subtotal_high = sum(item.cost_high for item in line_items)

    overhead_low = subtotal_low * 0.12
    overhead_high = subtotal_high * 0.15
    contingency_low = subtotal_low * _contingency_rate(data.issues, data.condition_score, low=True)
    contingency_high = subtotal_high * _contingency_rate(data.issues, data.condition_score, low=False)

    total_low = int(round(subtotal_low + overhead_low + contingency_low))
    total_high = int(round(subtotal_high + overhead_high + contingency_high))

    gap_signal = compute_gap_score(
        listing_price=data.listing_price,
        living_area_sqft=data.sqft,
        avg_area_price_per_sqft=data.avg_area_price_per_sqft,
    )
    age_signal = compute_renovation_age_detection(
        year_built=data.year_built,
        years_since_last_sale=data.years_since_last_sale,
        permit_years_since_last=data.permit_years_since_last,
    )

    timeline_low, timeline_high = _timeline_weeks(
        sqft=data.sqft,
        issue_count=len(data.issues),
        renovation_class=renovation_class,
        days_on_market=data.days_on_market,
        age_score_points=age_signal.score_points,
    )
    confidence_label, confidence_score = _confidence(
        condition_score=data.condition_score,
        issue_count=len(data.issues),
        gap_score_points=gap_signal.score_points,
        age_score_points=age_signal.score_points,
    )
    selected_work_items = _selected_work_items(data.renovation_elements)
    suggested_work_items = selected_work_items or _suggested_work_items(data.issues, data.room_type)
    impacted_elements, impacted_element_details = _impacted_elements(
        data.room_type,
        data.issues,
        suggested_work_items,
        selected_elements=data.renovation_elements,
    )
    explanation_summary = _explanation_summary(
        renovation_class=renovation_class,
        room_type=data.room_type,
        issues=data.issues,
        desired_quality_level=data.desired_quality_level,
        impacted_elements=impacted_elements,
        selected_elements=data.renovation_elements,
    )

    assumptions = [
        "Planning estimate only — not a contractor bid or guaranteed scope.",
        "Figures blend visible issues, home size, quality level, and location factors (with overhead and contingency).",
    ]
    if _is_wood_structure_scope(data):
        assumptions.append(
            "Wood-frame structure signal detected; estimate includes wood-structure repair multiplier for relevant structural scope."
        )
    if data.zip_code:
        assumptions.append(f"Location context: {data.zip_code}.")
    if data.issues:
        assumptions.append("Noted from condition: " + ", ".join(data.issues[:8]) + ".")

    return RenovationEstimate(
        renovation_class=renovation_class,
        minimum_cost=total_low,
        maximum_cost=total_high,
        minimum_timeline_weeks=timeline_low,
        maximum_timeline_weeks=timeline_high,
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        suggested_work_items=suggested_work_items,
        impacted_elements=impacted_elements,
        impacted_element_details=impacted_element_details,
        explanation_summary=explanation_summary,
        line_items=line_items,
        assumptions=assumptions,
    )

# User input cost adjustments
# Phrases appended for cost intent matching when the UI sends `renovation_elements` slugs.
_ELEMENT_SELECTION_SCOPE_PHRASES: dict[str, str] = {
    "flooring": "new flooring",
    "paint": "new paint",
    "kitchen": "kitchen remodel",
    "bathroom": "bathroom remodel",
    "windows": "window replacement",
    "doors": "door replacement",
    "lighting": "electrical upgrade",
}

_SELECTED_ELEMENT_TO_COST_CATEGORY: dict[str, str] = {
    "flooring": "flooring",
    "paint": "paint",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "windows": "window",
    "doors": "doors",
    "lighting": "electrical",
}

_SELECTED_ELEMENT_TO_IMPACTED_LABEL: dict[str, str] = {
    "flooring": "flooring",
    "paint": "paint finish",
    "kitchen": "kitchen surfaces",
    "bathroom": "bathroom surfaces",
    "windows": "windows",
    "doors": "doors",
    "lighting": "lighting",
    "furniture": "furniture",
}

INTENT_MAP = {
    "paint": ["paint", "repaint", "wall color", "wall finish", "new paint"],
    "flooring": ["flooring", "wood floor", "laminate", "vinyl", "carpet", "new flooring"],
    "kitchen": ["kitchen remodel", "kitchen renovation", "new kitchen", "upgrade kitchen", "new kitchen cabinets"],
    "bathroom": ["bathroom remodel", "bathroom renovation", "bath upgrade", "new bathroom"],
    "window": ["window", "windows", "window replacement", "window upgrade", "new window"],
    "roof": ["roof repair", "roof replacement", "roof upgrade", "new roof"],
    "foundation": ["foundation repair", "foundation crack", "structural foundation"],
    "electrical": ["electrical repair", "rewiring", "electrical upgrade"],
    "plumbing": ["plumbing repair", "pipe leak", "pipe replacement"],
    "hvac": ["hvac", "ac repair", "heating system", "cooling system"],
    "exterior": ["exterior remodel", "exterior painting", "facade upgrade", "new exterior"],
    "doors": ["door replacement", "door repair", "new door"],
    "garage": ["garage repair", "garage upgrade", "new garage"],
    "landscaping": ["landscaping", "garden work", "yard improvement", "new landscaping"],
}

SEVERITY_RANGE = {
    "minor": (0.2, 0.4),
    "moderate": (0.4, 0.7),
    "severe": (0.7, 1.0)
}


_NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bnot\b",
    r"\bdon't\b",
    r"\bdo not\b",
    r"\bwithout\b",
    r"\bavoid\b",
    r"\bskip\b",
)

_MAX_USER_INPUT_ADJUSTMENT_PCT = 0.60
_MAX_USER_INPUT_ADJUSTMENT_ABS = 75000
_WOOD_STRUCTURE_MULTIPLIER = 1.12


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", text))


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


# Helper function to match intents
def _match_intents(text: str) -> List[str]:
    """Return matched intents using deterministic, negation-aware matching."""
    matched: list[str] = []
    lowered = f" {text.lower()} "
    negation_regex = "|".join(_NEGATION_PATTERNS)

    for intent, keywords in INTENT_MAP.items():
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword.lower())}\b"
            keyword_match = re.search(pattern, lowered)
            if not keyword_match:
                continue

            # If there's a nearby negation token, ignore this keyword intent.
            window_start = max(0, keyword_match.start() - 24)
            window = lowered[window_start:keyword_match.start()]
            if re.search(negation_regex, window):
                continue

            matched.append(intent)
            break

    return matched


def _detect_severity(issue: str) -> str:
    """Detect severity using word boundaries to avoid false positives."""
    issue = (issue or "").lower()

    # Structural/foundation issues
    severe_patterns = [
        r"\bfoundation\b",
        r"\bstructural\b",
        r"\bmajor\s+crack",
        r"\broof\b",
        r"\bleak",
        r"\bmold\b",
        r"\bwater\s+damage",
    ]
    if any(re.search(p, issue) for p in severe_patterns):
        return "severe"

    # Systems
    moderate_patterns = [
        r"\belectrical\b",
        r"\bplumbing\b",
        r"\bhvac\b",
    ]
    if any(re.search(p, issue) for p in moderate_patterns):
        return "moderate"

    # Wear/age indicators
    wear_patterns = [
        r"\bold\b",
        r"\bworn\b",
        r"\boutdated\b",
        r"\bstain",
    ]
    if any(re.search(p, issue) for p in wear_patterns):
        return "moderate"

    return "minor"


def _compute_severity_score(issues: List[str]) -> float:
    """
    Smart severity score:
    - considers severity level
    - considers cost impact weight
    - returns multiplier (0.85 → 1.6)
    """

    if not issues:
        return 0.9

    total_score = 0.0

    for issue in issues:
        issue_lower = (issue or "").lower()

        severity = _detect_severity(issue_lower)
        low, high = SEVERITY_RANGE.get(severity, (0.4, 0.7))
        severity_value = (low + high) / 2

        # detect cost category weight
        weight = 1.0
        for key, w in ISSUE_COST_WEIGHT.items():
            if key in issue_lower:
                weight = w
                break

        total_score += severity_value * weight

    avg_score = total_score / len(issues)

    # normalize → multiplier
    return _clamp(0.75 + avg_score, 0.85, 1.6)



def apply_user_input_cost_adjustments(
    estimate: RenovationEstimate,
    user_inputs: str,
    sqft: float,
    location_factor: float = 1.0,
    renovation_elements: List[str] | None = None,
) -> RenovationEstimate:
    phrases: list[str] = []
    for el in renovation_elements or []:
        key = str(el).strip().lower()
        phrase = _ELEMENT_SELECTION_SCOPE_PHRASES.get(key)
        if phrase:
            phrases.append(phrase)
    text = " ".join([(user_inputs or "").strip(), *phrases]).strip().lower()
    if not text:
        return estimate

    # Deduplicate intents
    matched_intents = list(set(_match_intents(text)))
    if not matched_intents:
        return estimate

    sqft = max(sqft, 1.0)
    location_factor = max(location_factor, 0.5)

    adjustments: List[Tuple[str, int, int]] = []

    # Severity boost from user text
    severity_boost = 1.0
    if any(word in text for word in ["leak", "damage", "crack", "mold", "structural"]):
        severity_boost = 1.2

    for intent in matched_intents:
        if intent not in COST_MAP:
            continue

        cost_type, low, high = COST_MAP[intent]
        label = f"{intent.replace('_', ' ')} upgrade"

        # Cost calculation
        if cost_type == "sqft":
            adj_low = sqft * low
            adj_high = sqft * high

        elif cost_type == "unit":
            if intent == "window":
                qty = max(4, sqft / 250)
            elif intent == "doors":
                qty = max(3, sqft / 400)
            else:
                qty = 1

            adj_low = qty * low
            adj_high = qty * high

        else:  # fixed
            adj_low = low
            adj_high = high

        # Apply multipliers
        adj_low *= location_factor * severity_boost
        adj_high *= location_factor * severity_boost

        adjustments.append((
            label,
            int(round(adj_low)),
            int(round(adj_high))
        ))

    if not adjustments:
        return estimate

    # Sum adjustments
    low_add = sum(low for _, low, _ in adjustments)
    high_add = sum(high for _, _, high in adjustments)

    # Apply safety cap
    low_add, high_add = _cap_user_input_adjustments(
        estimate.minimum_cost,
        low_add,
        high_add,
    )

    # Notes for transparency
    notes = [
        f"User-input scope adjustment: {label} (+${low:,} to +${high:,})."
        for label, low, high in adjustments
    ]

    notes.append(
        f"User-input total adjustment applied: +${low_add:,} to +${high_add:,} "
        "(location & severity adjusted, capped relative to the base estimate)."
    )

    # Return updated estimate
    return estimate.model_copy(
        update={
            "minimum_cost": estimate.minimum_cost + low_add,
            "maximum_cost": estimate.maximum_cost + high_add,
            "assumptions": [*estimate.assumptions, *notes],
        }
    )


def _cap_user_input_adjustments(base_minimum_cost: int, low_add: int, high_add: int) -> tuple[int, int]:
    pct_cap = int(round(max(base_minimum_cost, 1) * _MAX_USER_INPUT_ADJUSTMENT_PCT))
    hard_cap = _MAX_USER_INPUT_ADJUSTMENT_ABS
    cap = max(0, min(pct_cap, hard_cap))
    return min(low_add, cap), min(high_add, cap)


def _build_line_items(data: RenovationEstimateInput, factor: float) -> List[RenovationLineItem]:
    sqft = max(data.sqft, 1.0)
    kitchen_qty = max(1.0, data.beds / 3)
    bath_qty = max(1.0, data.baths)

    selected_scope = _scope_categories_from_selected_elements(data.renovation_elements)
    scope = selected_scope or sorted(_scope_categories(data.issues, data.room_type, data.condition_score))

    # Map category → quantity
    quantity_map = {
    "paint": sqft,
    "flooring": sqft * 0.8,
    "kitchen": kitchen_qty,
    "bathroom": bath_qty,
    "plumbing": sqft,
    "electrical": sqft,
    "roof": sqft,
    "exterior": sqft * 0.5,
    "window": max(4, sqft / 250),
    "doors": max(3, sqft / 400),
    }

    line_items: List[RenovationLineItem] = []
    wood_structure = _is_wood_structure_scope(data)

    for category in scope:
        if category not in COST_MAP:
            continue

        cost_type, low, high = COST_MAP[category]

        qty = quantity_map.get(category, 1.0)

        if wood_structure and category in {"foundation", "roof", "exterior"}:
            low *= _WOOD_STRUCTURE_MULTIPLIER
            high *= _WOOD_STRUCTURE_MULTIPLIER

        # Adjust based on type
        if cost_type == "sqft":
            unit = "sqft"
            unit_low = low * factor
            unit_high = high * factor

        elif cost_type == "unit":
            unit = "unit"
            unit_low = low * factor
            unit_high = high * factor

        else:  # fixed
            unit = "count"
            qty = 1
            unit_low = low * factor
            unit_high = high * factor

        line_items.append(
            RenovationLineItem(
                category=category.capitalize(),
                quantity=round(qty, 2),
                unit=unit,
                unit_cost_low=round(unit_low, 2),
                unit_cost_high=round(unit_high, 2),
                cost_low=round(qty * unit_low, 2),
                cost_high=round(qty * unit_high, 2),
            )
        )

    # fallback
    if not line_items:
        cost_type, low, high = COST_MAP["paint"]
        line_items.append(
            RenovationLineItem(
                category="Paint",
                quantity=round(sqft, 2),
                unit="sqft",
                unit_cost_low=round(low * factor, 2),
                unit_cost_high=round(high * factor, 2),
                cost_low=round(sqft * low * factor, 2),
                cost_high=round(sqft * high * factor, 2),
            )
        )

    return line_items


def _scope_categories(issues: List[str], room_type: str, condition_score: int) -> set[str]:
    normalized = {(i or "").strip().lower() for i in issues}
    room = (room_type or "").lower()
    scope: set[str] = set()

    category_rules: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        ("kitchen", ("kitchen",), ("kitchen", "cabinet", "cabinets")),
        ("bathroom", ("bathroom", "bath"), ("bathroom", "bath", "tile", "tiles")),
        (
            "exterior",
            ("exterior",),
            ("roof", "exterior", "siding", "facade", "driveway", "gutter", "garage door"),
        ),
        ("flooring", ("living", "bedroom", "hall"), ("floor", "flooring", "carpet")),
        ("paint", (), ("paint", "stain", "wall", "walls")),
        ("plumbing", (), ("water damage", "leak", "mold", "plumbing")),
        ("electrical", (), ("electrical", "rewire", "wiring")),
        ("hvac", (), ("hvac", "heating", "cooling", "ac", "air conditioning")),
        ("roof", (), ("roof damage", "roof", "sagging roof")),
        ("foundation", (), ("foundation", "structural damage")),
    )

    for category, room_terms, issue_terms in category_rules:
        room_match = bool(room_terms) and _contains_any_term(room, room_terms)
        issue_match = any(_contains_any_term(i, issue_terms) for i in normalized)
        if room_match or issue_match:
            scope.add(category)

    if condition_score < 85:
        scope.add("paint")

    return scope


def _scope_categories_from_selected_elements(renovation_elements: List[str]) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()
    for raw in renovation_elements or []:
        key = (raw or "").strip().lower()
        mapped = _SELECTED_ELEMENT_TO_COST_CATEGORY.get(key)
        if not mapped or mapped in seen:
            continue
        seen.add(mapped)
        categories.append(mapped)
    return categories


def _is_wood_structure_scope(data: RenovationEstimateInput) -> bool:
    text_parts = [
        data.user_inputs,
        data.listing_description,
        data.room_type,
        " ".join(data.issues),
    ]
    text = " ".join([p for p in text_parts if p]).lower()
    wood_keywords = (
        "wood frame",
        "wood framing",
        "timber",
        "lumber",
        "stud",
        "studs",
        "framing exposed",
    )
    if any(k in text for k in wood_keywords):
        return True
    major_issue = any(
        ("structural damage" in (issue or "").lower()) or ("major wall cracks" in (issue or "").lower())
        for issue in data.issues
    )
    return (data.room_type or "").strip().lower() == "exterior" and major_issue


def _quality_level_factor(level: str) -> float:
    factors = {
        "cosmetic": 0.90,
        "standard": 1.00,
        "premium": 1.12,
        "luxury": 1.30,
    }
    return factors.get((level or "standard").strip().lower(), 1.00)


def _issue_factor(issues: List[str]) -> float:
    if not issues:
        return 0.95
    if len(issues) <= 3:
        return 1.0
    if len(issues) <= 6:
        return 1.08
    return 1.18


def _has_major_issue(issues: List[str]) -> bool:
    """Severe scope signals for classing and contingency — excludes generic wall cracks (interior finishes)."""
    major = {
        "foundation cracks",
        "structural damage",
        "major wall cracks",
        "roof damage",
        "sagging roof",
        "water damage",
        "mold",
        "electrical issues",
        "plumbing issues",
    }
    return any(any(m in (issue or "").lower() for m in major) for issue in issues)


def _contingency_rate(issues: List[str], condition_score: int, *, low: bool) -> float:
    base = 0.08 if low else 0.14
    if _has_major_issue(issues):
        base += 0.04
    if condition_score < 50:
        base += 0.03
    return base


def _timeline_weeks(
    *,
    sqft: float,
    issue_count: int,
    renovation_class: str,
    days_on_market: int,
    age_score_points: int = 0,
) -> tuple[int, int]:
    base = 6
    if sqft > 1400:
        base += 2
    if sqft > 2200:
        base += 2
    if issue_count >= 5:
        base += 2
    if renovation_class in {"Heavy", "Full Gut"}:
        base += 2
    if days_on_market > 45:
        base += 1
    if age_score_points >= 22:
        base += 2
    elif age_score_points >= 12:
        base += 1
    return base, base + 4


def _confidence(
    *,
    condition_score: int,
    issue_count: int,
    gap_score_points: int = 0,
    age_score_points: int = 0,
) -> tuple[Literal["LOW", "MEDIUM", "HIGH"], int]:
    score = 45
    if issue_count >= 3:
        score += 20
    if issue_count >= 6:
        score += 10
    if condition_score <= 65:
        score += 10
    score += int(round(gap_score_points * 0.30))
    score += int(round(age_score_points * 0.20))
    score = int(_clamp(score, 25, 95))
    if score >= 75:
        return "HIGH", score
    if score >= 50:
        return "MEDIUM", score
    return "LOW", score


def _renovation_class(condition_score: int, issues: List[str]) -> Literal["Cosmetic", "Moderate", "Heavy", "Full Gut"]:
    major = _has_major_issue(issues)
    issue_count = len(issues)
    if major and condition_score < 40:
        return "Full Gut"
    if major or condition_score < 55 or issue_count >= 7:
        return "Heavy"
    if condition_score < 75 or issue_count >= 3:
        return "Moderate"
    return "Cosmetic"


def _suggested_work_items(issues: List[str], room_type: str) -> List[str]:
    normalized = {(i or "").strip().lower() for i in issues}
    items: list[str] = []
    _append_matching_item(
        normalized,
        items,
        ("structural", "foundation", "major wall cracks"),
        "structural stabilization",
    )
    _append_matching_item(
        normalized,
        items,
        ("water damage", "leak", "mold", "plumbing"),
        "water/moisture remediation",
    )
    _append_matching_item(
        normalized,
        items,
        ("fire damage", "smoke damage"),
        "fire/smoke remediation",
    )
    _append_matching_item(
        normalized,
        items,
        ("electrical", "rewire", "wiring"),
        "electrical safety upgrades",
    )
    _append_matching_item(
        normalized,
        items,
        ("roof", "sagging roof"),
        "roof repairs",
    )
    _append_matching_item(normalized, items, ("paint", "stain"), "paint")
    _append_matching_item(normalized, items, ("floor", "carpet"), "flooring")
    _append_matching_item(normalized, items, ("cabinet", "kitchen"), "kitchen update")
    _append_matching_item(normalized, items, ("bath", "tile"), "bathroom update")
    _append_matching_item(normalized, items, ("roof", "siding", "facade", "driveway", "exterior"), "exterior repairs")

    if not items:
        items.extend(_fallback_items_for_room(room_type))
    return items[:8]


def _append_matching_item(
    normalized_issues: set[str],
    items: list[str],
    keywords: tuple[str, ...],
    work_item: str,
) -> None:
    if any(any(_contains_term(issue, keyword) for keyword in keywords) for issue in normalized_issues):
        items.append(work_item)


def _fallback_items_for_room(room_type: str) -> list[str]:
    room = (room_type or "").lower()
    if "kitchen" in room:
        return ["kitchen refresh"]
    if "bath" in room:
        return ["bathroom refresh"]
    if "exterior" in room:
        return ["exterior refresh"]
    return ["paint", "minor repairs"]


def _explanation_summary(
    *,
    renovation_class: str,
    room_type: str,
    issues: List[str],
    desired_quality_level: str,
    impacted_elements: List[str],
    selected_elements: List[str] | None = None,
) -> str:
    selected_names = [str(x).strip() for x in (selected_elements or []) if str(x).strip()]
    if selected_names:
        focus = ", ".join(selected_names[:5])
        return (
            f"Estimate is scoped to user-selected renovation elements ({focus}) "
            f"with {desired_quality_level} finish assumptions."
        )
    room = (room_type or "unknown").replace(",", ", ")
    element_note = ""
    if impacted_elements:
        element_note = " Focus areas: " + ", ".join(impacted_elements[:5]) + "."
    if issues:
        top_issues = ", ".join(issues[:3])
        return (
            f"Property is classified as {renovation_class} rehab based on {room} condition and "
            f"detected issues ({top_issues}). Estimate assumes {desired_quality_level} finish level."
            f"{element_note}"
        )
    return (
        f"Property is classified as {renovation_class} rehab using available room signals ({room}) "
        f"with {desired_quality_level} finish assumptions.{element_note}"
    )


def _impacted_elements(
    room_type: str,
    issues: List[str],
    suggested_work_items: List[str],
    selected_elements: List[str] | None = None,
) -> tuple[List[str], List[ImpactedElementDetail]]:
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    selected_labels: list[str] = []
    selected_details: list[ImpactedElementDetail] = []
    selected_seen: set[str] = set()
    for raw in selected_elements or []:
        key = _norm(raw)
        if not key or key in selected_seen:
            continue
        selected_seen.add(key)
        label = _SELECTED_ELEMENT_TO_IMPACTED_LABEL.get(key, raw)
        selected_labels.append(label)
        selected_details.append(
            ImpactedElementDetail(
                name=label,
                source="user_selected",
                confidence=0.95,
                reason=f"Explicitly selected by user ({raw}).",
            )
        )
    if selected_labels:
        return selected_labels[:8], selected_details[:8]

    room = _norm(room_type)
    issue_text = " ".join(_norm(issue) for issue in issues)
    items_text = " ".join(_norm(item) for item in suggested_work_items)
    combined_text = f"{room} {issue_text} {items_text}".strip()

    room_baseline_elements: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("kitchen",), ("cabinets", "countertops", "backsplash", "fixtures")),
        (("bath", "bathroom"), ("vanity", "tile", "fixtures")),
        (("living", "bedroom", "hall"), ("walls", "flooring", "lighting")),
        (("exterior", "outside"), ("siding", "roofline", "paint")),
    )
    element_detection_rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (("cabinet", "cabinets", "cupboard", "cupboards"), "cabinet fronts"),
        (("counter", "countertop"), "countertops"),
        (("backsplash",), "backsplash"),
        (("paint", "repaint", "wall color", "wall colour", "stain"), "paint finish"),
        (("floor", "flooring", "tile", "carpet", "laminate", "vinyl"), "flooring"),
        (("plumbing", "water damage", "leak", "faucet", "sink"), "plumbing fixtures"),
        (("window", "windows", "glass"), "windows"),
        (("door", "doors"), "doors"),
        (("roof", "roofline", "shingle"), "roofline"),
        (("electrical", "wiring", "rewire"), "electrical fixtures"),
        (("lighting", "light fixture", "lights"), "lighting"),
        (("furniture", "sofa", "table", "chair"), "furniture"),
        (("appliance", "dishwasher", "refrigerator", "fridge", "oven", "range"), "appliances"),
        (("siding", "facade", "façade", "exterior wall"), "siding"),
    )

    elements: list[str] = []
    detail_candidates: list[ImpactedElementDetail] = []

    for room_terms, defaults in room_baseline_elements:
        if any(_contains_term(room, term) for term in room_terms):
            elements.extend(defaults)
            detail_candidates.extend(
                ImpactedElementDetail(
                    name=default,
                    source="room_baseline",
                    confidence=0.65,
                    reason=f"Derived from detected room type '{room_type or 'unknown'}'.",
                )
                for default in defaults
            )
            break

    for keywords, element in element_detection_rules:
        if any(_contains_term(combined_text, term) for term in keywords):
            elements.append(element)
            detail_candidates.append(
                ImpactedElementDetail(
                    name=element,
                    source="issue_or_work_item",
                    confidence=0.85,
                    reason=f"Matched issue/work-item keywords: {', '.join(keywords[:3])}.",
                )
            )

    seen: set[str] = set()
    deduped: list[str] = []
    detail_by_name: dict[str, ImpactedElementDetail] = {}
    for el in elements:
        key = _norm(el)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(el)
    for detail in detail_candidates:
        key = _norm(detail.name)
        if key not in seen:
            continue
        existing = detail_by_name.get(key)
        if existing is None or detail.confidence > existing.confidence:
            detail_by_name[key] = detail
    final_elements = deduped[:8]
    final_details = [detail_by_name[_norm(name)] for name in final_elements if _norm(name) in detail_by_name]
    return final_elements, final_details


def _selected_work_items(renovation_elements: List[str]) -> List[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in renovation_elements or []:
        key = (raw or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(f"{key.replace('_', ' ')} renovation")
    return items[:8]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
