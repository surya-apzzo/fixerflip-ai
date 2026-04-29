"""
Renovation Cost AI — renovation engine.
Estimates the cost of a renovation project based on the condition of the property, the desired quality level, and the location.
"""

from __future__ import annotations

import logging
import re
from typing import List, Literal, Tuple

from app.core.rules_config import COST_MAP, ISSUE_COST_WEIGHT
from app.engine.renovation_engine.score_from_issues import (
    compute_gap_score,
    compute_renovation_age_detection,
)
from app.schemas import (
    ImpactedElementDetail,
    RenovationEstimate,
    RenovationEstimateInput,
    RenovationLineItem,
)

logger = logging.getLogger(__name__)


def estimate_renovation_cost(data: RenovationEstimateInput) -> RenovationEstimate:
    renovation_class = _classify_renovation_class(data.condition_score, data.issues)
    severity_factor = _calculate_severity_multiplier(data.issues)
    quality_factor = _calculate_quality_factor(data.desired_quality_level)
    issue_factor = _calculate_issue_count_factor(data.issues)
    location_factor = _clamp(
        ((data.labor_index * 0.6) + (data.material_index * 0.4)), 0.7, 2.5,)
    combined_factor = severity_factor * quality_factor * issue_factor * location_factor
    combined_factor = _clamp(combined_factor, 0.85, 1.9)



    user_scope_categories = _build_user_scope_categories(
        data.user_inputs,
        data.renovation_elements,
    )
    line_items = _build_cost_line_items(
        data,
        combined_factor,
        user_scope_categories=user_scope_categories,
    )
    subtotal_low = sum(item.cost_low for item in line_items)
    subtotal_high = sum(item.cost_high for item in line_items)


    overhead_low = subtotal_low * 0.12
    overhead_high = subtotal_high * 0.15
    contingency_low = subtotal_low * _calculate_contingency_rate(
        data.issues,
        data.condition_score,
        low=True,
    )
    contingency_high = subtotal_high * _calculate_contingency_rate(
        data.issues,
        data.condition_score,
        low=False,
    )

    total_low = int(round(subtotal_low + overhead_low + contingency_low))
    total_high = int(round(subtotal_high + overhead_high + contingency_high))
    if total_low > total_high:
        total_low, total_high = total_high, total_low


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

    timeline_low, timeline_high = _estimate_timeline_weeks(
        sqft=data.sqft,
        issue_count=len(data.issues),
        renovation_class=renovation_class,
        days_on_market=data.days_on_market,
        age_score_points=age_signal.score_points,
    )
    confidence_label, confidence_score = _calculate_confidence_score(
        condition_score=data.condition_score,
        issue_count=len(data.issues),
        gap_score_points=gap_signal.score_points,
        age_score_points=age_signal.score_points,
    )
    selected_work_items = _build_selected_work_items(data.renovation_elements)
    user_work_items = _build_user_scope_work_items(user_scope_categories)
    issue_work_items = _build_suggested_work_items(data.issues, data.room_type)
    if selected_work_items:
        suggested_work_items = selected_work_items
    elif data.issues and user_work_items:
        # For damaged properties, user-requested upgrades are additive and must not hide remediation scope.
        suggested_work_items = _deduplicate_work_items([*issue_work_items, *user_work_items])
    elif user_work_items:
        suggested_work_items = user_work_items
    else:
        suggested_work_items = issue_work_items
    impacted_elements, impacted_element_details = _build_impacted_element_outputs(
        data.room_type,
        data.issues,
        suggested_work_items,
        selected_elements=data.renovation_elements,
        user_scope_categories=user_scope_categories,
    )
    explanation_summary = _build_explanation_summary(
        renovation_class=renovation_class,
        room_type=data.room_type,
        issues=data.issues,
        desired_quality_level=data.desired_quality_level,
        impacted_elements=impacted_elements,
        suggested_work_items=suggested_work_items,
        selected_elements=data.renovation_elements,
        user_scope_categories=user_scope_categories,
    )

    assumptions = [
        "Planning estimate only — not a contractor bid or guaranteed scope.",
        "Figures blend visible issues, home size, quality level, and location factors (with overhead and contingency).",
    ]
    if _is_wood_structure_context(data):
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
    "flooring": [
        "flooring",
        "floor",
        "wood floor",
        "laminate",
        "vinyl",
        "carpet",
        "tile floor",
        "floor tile",
        "floor tiles",
        "tile flooring",
        "tiles",
        "new flooring",
        "replace floor",
        "replace flooring",
        "replace tile",
        "replace tiles",
    ],
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
_CANONICAL_MAJOR_RISK_ISSUES = frozenset({
    "major wall cracks",
    "structural damage",
    "roof damage",
    "water damage",
    "mold",
    "fire damage",
    "smoke damage",
})
_LEGACY_MAJOR_RISK_ALIASES: dict[str, str] = {
    "foundation cracks": "major wall cracks",
    "sagging roof": "roof damage",
    "electrical issues": "structural damage",
    "plumbing issues": "water damage",
}
_SCOPE_SEVERITY_TERMS = (
    "leak",
    "damage",
    "crack",
    "mold",
    "structural",
    "fire",
    "smoke",
)

_CATEGORY_TO_USER_WORK_ITEM: dict[str, str] = {
    "paint": "paint renovation",
    "flooring": "flooring renovation",
    "kitchen": "kitchen renovation",
    "bathroom": "bathroom renovation",
    "window": "window renovation",
    "roof": "roof renovation",
    "foundation": "foundation renovation",
    "electrical": "electrical renovation",
    "plumbing": "plumbing renovation",
    "hvac": "hvac renovation",
    "exterior": "exterior renovation",
    "doors": "doors renovation",
    "garage": "garage renovation",
    "landscaping": "landscaping renovation",
}

_CATEGORY_TO_IMPACTED_LABEL: dict[str, str] = {
    "paint": "paint finish",
    "flooring": "flooring",
    "kitchen": "kitchen surfaces",
    "bathroom": "bathroom surfaces",
    "window": "windows",
    "roof": "roofline",
    "foundation": "foundation surfaces",
    "electrical": "electrical fixtures",
    "plumbing": "plumbing fixtures",
    "hvac": "hvac",
    "exterior": "siding",
    "doors": "doors",
    "garage": "garage",
    "landscaping": "landscaping",
}


def _text_has_term(text: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", text))


def _text_has_any_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(_text_has_term(text, term) for term in terms)


def _detect_scope_intents(text: str) -> List[str]:
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

            window_start = max(0, keyword_match.start() - 24)
            window = lowered[window_start:keyword_match.start()]
            if re.search(negation_regex, window):
                continue

            matched.append(intent)
            break

    return matched


def _classify_issue_severity(issue: str) -> str:
    """Detect severity using word boundaries to avoid false positives."""
    issue = (issue or "").lower()

    severe_patterns = [
        r"\bfoundation\b",
        r"\bstructural\b",
        r"\bmajor\s+crack",
        r"\broof\b",
        r"\bleak",
        r"\bmold\b",
        r"\bwater\s+damage",
        # FIX 1: fire and smoke are severe — they were missing from severity detection
        r"\bfire\s+damage",
        r"\bsmoke\s+damage",
    ]
    if any(re.search(p, issue) for p in severe_patterns):
        return "severe"

    moderate_patterns = [
        r"\belectrical\b",
        r"\bplumbing\b",
        r"\bhvac\b",
    ]
    if any(re.search(p, issue) for p in moderate_patterns):
        return "moderate"

    wear_patterns = [
        r"\bold\b",
        r"\bworn\b",
        r"\boutdated\b",
        r"\bstain",
    ]
    if any(re.search(p, issue) for p in wear_patterns):
        return "moderate"

    return "minor"


def _calculate_severity_multiplier(issues: List[str]) -> float:
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

        severity = _classify_issue_severity(issue_lower)
        low, high = SEVERITY_RANGE.get(severity, (0.4, 0.7))
        severity_value = (low + high) / 2

        weight = 1.0
        for key, w in ISSUE_COST_WEIGHT.items():
            if key in issue_lower:
                weight = w
                break

        total_score += severity_value * weight

    avg_score = total_score / len(issues)
    return _clamp(0.75 + avg_score, 0.85, 1.6)


def _build_user_scope_text(
    user_inputs: str,
    renovation_elements: List[str] | None,
) -> str:
    phrases = [
        _ELEMENT_SELECTION_SCOPE_PHRASES[key]
        for raw in (renovation_elements or [])
        if (key := str(raw).strip().lower()) in _ELEMENT_SELECTION_SCOPE_PHRASES
    ]
    return " ".join([(user_inputs or "").strip(), *phrases]).strip().lower()


def _build_user_scope_categories(
    user_inputs: str,
    renovation_elements: List[str] | None,
) -> list[str]:
    selected_scope = _map_selected_elements_to_scope(renovation_elements or [])
    intent_scope = list(set(_detect_scope_intents(_build_user_scope_text(user_inputs, renovation_elements))))
    merged = [*selected_scope, *intent_scope]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in merged:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def infer_user_scope_categories(
    user_inputs: str,
    renovation_elements: List[str] | None = None,
) -> list[str]:
    """Public helper for services to consistently detect explicit user-requested scope."""
    return _build_user_scope_categories(user_inputs, renovation_elements)


def _build_user_scope_work_items(categories: List[str]) -> List[str]:
    items: list[str] = []
    seen: set[str] = set()
    for category in categories:
        label = _CATEGORY_TO_USER_WORK_ITEM.get(category)
        if not label or label in seen:
            continue
        seen.add(label)
        items.append(label)
    return items[:8]


def _calculate_scope_severity_boost(text: str) -> float:
    return 1.2 if any(term in text for term in _SCOPE_SEVERITY_TERMS) else 1.0


def _calculate_intent_quantity(intent: str, sqft: float) -> float:
    if intent == "window":
        return max(4, sqft / 250)
    if intent == "doors":
        return max(3, sqft / 400)
    return 1.0


def _calculate_intent_cost_range(
    *,
    intent: str,
    sqft: float,
    location_factor: float,
    severity_boost: float,
) -> tuple[int, int] | None:
    if intent not in COST_MAP:
        return None
    cost_type, low, high = COST_MAP[intent]
    if cost_type == "sqft":
        adj_low, adj_high = sqft * low, sqft * high
    elif cost_type == "unit":
        qty = _calculate_intent_quantity(intent, sqft)
        adj_low, adj_high = qty * low, qty * high
    else:
        adj_low, adj_high = low, high

    multiplier = location_factor * severity_boost
    return int(round(adj_low * multiplier)), int(round(adj_high * multiplier))


def _build_intent_cost_adjustments(
    *,
    intents: List[str],
    sqft: float,
    location_factor: float,
    severity_boost: float,
) -> List[Tuple[str, int, int]]:
    adjustments: List[Tuple[str, int, int]] = []
    for intent in intents:
        computed = _calculate_intent_cost_range(
            intent=intent,
            sqft=sqft,
            location_factor=location_factor,
            severity_boost=severity_boost,
        )
        if computed is None:
            continue
        low, high = computed
        adjustments.append((f"{intent.replace('_', ' ')} upgrade", low, high))
    return adjustments


def apply_user_input_cost_adjustments(
    estimate: RenovationEstimate,
    user_inputs: str,
    sqft: float,
    location_factor: float = 1.0,
    renovation_elements: List[str] | None = None,
) -> RenovationEstimate:
    text = _build_user_scope_text(user_inputs, renovation_elements)
    if not text:
        return estimate

    matched_intents = list(set(_detect_scope_intents(text)))
    if not matched_intents:
        return estimate

    sqft = max(sqft, 1.0)
    location_factor = max(location_factor, 0.5)
    adjustments = _build_intent_cost_adjustments(
        intents=matched_intents,
        sqft=sqft,
        location_factor=location_factor,
        severity_boost=_calculate_scope_severity_boost(text),
    )

    if not adjustments:
        return estimate

    low_add = sum(low for _, low, _ in adjustments)
    high_add = sum(high for _, _, high in adjustments)

    low_add, high_add = _cap_adjustment_range(
        estimate.minimum_cost,
        low_add,
        high_add,
    )

    notes = [
        f"User-input scope adjustment: {label} (+${low:,} to +${high:,})."
        for label, low, high in adjustments
    ]

    notes.append(
        f"User-input total adjustment applied: +${low_add:,} to +${high_add:,} "
        "(location & severity adjusted, capped relative to the base estimate)."
    )

    return estimate.model_copy(
        update={
            "minimum_cost": estimate.minimum_cost + low_add,
            "maximum_cost": estimate.maximum_cost + high_add,
            "assumptions": [*estimate.assumptions, *notes],
        }
    )


def _cap_adjustment_range(base_minimum_cost: int, low_add: int, high_add: int) -> tuple[int, int]:
    pct_cap = int(round(max(base_minimum_cost, 1) * _MAX_USER_INPUT_ADJUSTMENT_PCT))
    hard_cap = _MAX_USER_INPUT_ADJUSTMENT_ABS
    cap = max(0, min(pct_cap, hard_cap))
    return min(low_add, cap), min(high_add, cap)


def _normalize_token_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _build_selected_element_outputs(
    selected_elements: List[str] | None,
) -> tuple[list[str], list[ImpactedElementDetail]]:
    selected_labels: list[str] = []
    selected_details: list[ImpactedElementDetail] = []
    selected_seen: set[str] = set()
    for raw in selected_elements or []:
        key = _normalize_token_text(raw)
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
    return selected_labels[:8], selected_details[:8]


def _build_room_default_elements(room: str, room_type: str) -> tuple[list[str], list[ImpactedElementDetail]]:
    room_baseline_elements: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("kitchen",), ("cabinets", "countertops", "backsplash", "fixtures")),
        (("bath", "bathroom"), ("vanity", "tile", "fixtures")),
        (("living", "bedroom", "hall"), ("walls", "flooring", "lighting")),
        (("exterior", "outside"), ("siding", "roofline", "paint")),
        (("basement",), ("walls", "flooring", "waterproofing")),
    )

    for room_terms, defaults in room_baseline_elements:
        if not any(_text_has_term(room, term) for term in room_terms):
            continue
        details = [
            ImpactedElementDetail(
                name=default,
                source="room_baseline",
                confidence=0.65,
                reason=f"Derived from detected room type '{room_type or 'unknown'}'.",
            )
            for default in defaults
        ]
        return list(defaults), details
    return [], []


def _detect_impacted_elements(combined_text: str) -> tuple[list[str], list[ImpactedElementDetail]]:
    element_detection_rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (("structural damage", "major wall cracks", "foundation cracks", "foundation crack", "collapsed", "collapse"), "structural framing"),
        (("cabinet", "cabinets", "cupboard", "cupboards", "outdated cabinets"), "cabinet fronts"),
        (("counter", "countertop"), "countertops"),
        (("backsplash",), "backsplash"),
        (("paint", "repaint", "wall color", "wall colour", "stain", "paint wear", "ceiling", "damaged ceiling"), "paint finish"),
        (("floor", "flooring", "tile", "carpet", "laminate", "vinyl", "floor damage", "outdated flooring"), "flooring"),
        (("plumbing", "water damage", "leak", "faucet", "sink", "fixture", "fixtures", "broken fixtures"), "plumbing fixtures"),
        (("window", "windows", "glass", "broken windows"), "windows"),
        (("door", "doors"), "doors"),
        (("roof", "roofline", "shingle", "roof damage"), "roofline"),
        (("electrical", "wiring", "rewire"), "electrical fixtures"),
        (("lighting", "light fixture", "lights"), "lighting"),
        (("furniture", "sofa", "table", "chair"), "furniture"),
        (("appliance", "appliances", "damaged appliances", "dishwasher", "refrigerator", "fridge", "oven", "range"), "appliances"),
        (("siding", "facade", "façade", "exterior wall", "rotted wood"), "siding"),
        (("fire damage", "smoke damage", "charred", "soot"), "fire/smoke affected surfaces"),
        (("mold",), "moisture affected surfaces"),
    )

    elements: list[str] = []
    details: list[ImpactedElementDetail] = []
    for keywords, element in element_detection_rules:
        if not any(_text_has_term(combined_text, term) for term in keywords):
            continue
        elements.append(element)
        details.append(
            ImpactedElementDetail(
                name=element,
                source="issue_or_work_item",
                confidence=0.85,
                reason=f"Matched issue/work-item keywords: {', '.join(keywords[:3])}.",
            )
        )
    return elements, details


def _deduplicate_elements_with_details(
    elements: list[str],
    detail_candidates: list[ImpactedElementDetail],
) -> tuple[list[str], list[ImpactedElementDetail]]:
    seen: set[str] = set()
    deduped: list[str] = []
    detail_by_name: dict[str, ImpactedElementDetail] = {}

    for element in elements:
        key = _normalize_token_text(element)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(element)

    for detail in detail_candidates:
        key = _normalize_token_text(detail.name)
        if key not in seen:
            continue
        existing = detail_by_name.get(key)
        if existing is None or detail.confidence > existing.confidence:
            detail_by_name[key] = detail

    final_elements = deduped[:8]
    final_details = [
        detail_by_name[_normalize_token_text(name)]
        for name in final_elements
        if _normalize_token_text(name) in detail_by_name
    ]
    return final_elements, final_details


def _build_cost_line_items(
    data: RenovationEstimateInput,
    factor: float,
    *,
    user_scope_categories: List[str] | None = None,
) -> List[RenovationLineItem]:
    sqft = max(data.sqft, 1.0)
    kitchen_qty = max(1.0, data.beds / 3)
    bath_qty = max(1.0, data.baths)

    selected_scope = _map_selected_elements_to_scope(data.renovation_elements)
    requested_scope = set(user_scope_categories or [])
    critical_scope = _derive_critical_safety_scope(data.issues)
    if selected_scope:
        scope = sorted({*selected_scope, *requested_scope, *critical_scope})
    elif requested_scope and data.issues:
        # For damaged properties, keep remediation scope and add user-requested upgrades.
        scope = sorted(
            {
                *_derive_scope_categories(data.issues, data.room_type, data.condition_score),
                *requested_scope,
                *critical_scope,
            }
        )
    elif requested_scope:
        # For good/no-issue properties, explicit user scope should override room defaults.
        scope = sorted({*requested_scope, *critical_scope})
    else:
        scope = sorted(
            {
                *_derive_scope_categories(data.issues, data.room_type, data.condition_score),
                *requested_scope,
                *critical_scope,
            }
        )

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
    wood_structure = _is_wood_structure_context(data)

    for category in scope:
        if category not in COST_MAP:
            continue

        cost_type, low, high = COST_MAP[category]

        qty = quantity_map.get(category, 1.0)

        if wood_structure and category in {"foundation", "roof", "exterior"}:
            low *= _WOOD_STRUCTURE_MULTIPLIER
            high *= _WOOD_STRUCTURE_MULTIPLIER

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


def _derive_scope_categories(issues: List[str], room_type: str, condition_score: int) -> set[str]:
    normalized = {(i or "").strip().lower() for i in issues}
    room = (room_type or "").lower()
    scope: set[str] = set()

    category_rules: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "kitchen",
            ("kitchen",),
            ("kitchen", "cabinet", "cabinets", "outdated cabinets", "appliance", "appliances", "damaged appliances"),
        ),
        (
            "bathroom",
            ("bathroom", "bath"),
            ("bathroom", "bath", "tile", "tiles", "old bathroom", "old tiles", "missing grout", "dirty grout", "broken fixtures"),
        ),
        (
            "exterior",
            ("exterior",),
            (
                "roof", "exterior", "siding", "facade", "driveway", "gutter", "garage door",
                "peeling exterior paint", "cracked driveway", "poor landscaping", "garage damage", "rotted wood",
                "fire damage", "smoke damage",
            ),
        ),
        (
            "flooring",
            ("living", "bedroom", "hall", "basement"),
            ("floor", "flooring", "carpet", "floor damage", "outdated flooring"),
        ),
        ("paint", (), ("paint", "stain", "wall", "walls", "paint wear", "stains", "ceiling", "damaged ceiling")),
        (
            "plumbing",
            (),
            ("water damage", "leak", "mold", "plumbing"),
        ),
        ("electrical", (), ("electrical", "rewire", "wiring")),
        ("hvac", (), ("hvac", "heating", "cooling", "ac", "air conditioning")),
        ("roof", (), ("roof damage", "roof", "sagging roof")),
        (
            "foundation",
            (),
            (
                "major wall cracks", "structural damage",
                # keep legacy strings for any pre-existing normalized data
                "foundation cracks", "foundation crack",
            ),
        ),
    )

    for category, room_terms, issue_terms in category_rules:
        room_match = bool(room_terms) and _text_has_any_term(room, room_terms)
        issue_match = any(_text_has_any_term(i, issue_terms) for i in normalized)
        if room_match or issue_match:
            scope.add(category)

    if condition_score < 85:
        scope.add("paint")

    return scope


def _map_selected_elements_to_scope(renovation_elements: List[str]) -> list[str]:
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


def _derive_critical_safety_scope(issues: List[str]) -> set[str]:
    normalized = [_normalize_token_text(i) for i in issues]
    scope: set[str] = set()

    has_fire_smoke = any(
        _text_has_any_term(i, ("fire damage", "smoke damage", "charred", "burn"))
        for i in normalized
    )
    has_electrical = any(_text_has_any_term(i, ("electrical", "wiring", "rewire")) for i in normalized)
    has_plumbing = any(_text_has_any_term(i, ("plumbing", "water damage", "leak", "mold")) for i in normalized)
    has_structural = any(
        _text_has_any_term(i, ("structural damage", "major wall cracks", "foundation", "sagging roof", "roof damage"))
        for i in normalized
    )

    if has_fire_smoke:
        scope.update({"electrical", "plumbing"})
    if has_electrical:
        scope.add("electrical")
    if has_plumbing:
        scope.add("plumbing")
    if has_structural:
        if any(_text_has_any_term(i, ("roof", "sagging roof", "roof damage")) for i in normalized):
            scope.add("roof")
        scope.add("foundation")

    return scope


def _is_wood_structure_context(data: RenovationEstimateInput) -> bool:
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
    structural_markers = {"structural damage", "major wall cracks"}
    major_issue = any(
        _LEGACY_MAJOR_RISK_ALIASES.get(_normalize_token_text(issue), _normalize_token_text(issue))
        in structural_markers
        for issue in data.issues
    )
    return (data.room_type or "").strip().lower() == "exterior" and major_issue


def _calculate_quality_factor(level: str) -> float:
    factors = {
        "cosmetic": 0.90,
        "standard": 1.00,
        "premium": 1.12,
        "luxury": 1.30,
    }
    return factors.get((level or "standard").strip().lower(), 1.00)


def _calculate_issue_count_factor(issues: List[str]) -> float:
    if not issues:
        return 0.95
    if len(issues) <= 3:
        return 1.0
    if len(issues) <= 6:
        return 1.08
    return 1.18


def _major_risk_issue(issues: List[str]) -> bool:
    """Severe scope signals for classifying and contingency."""
    for raw_issue in issues:
        normalized = _normalize_token_text(raw_issue)
        canonical = _LEGACY_MAJOR_RISK_ALIASES.get(normalized, normalized)
        if canonical in _CANONICAL_MAJOR_RISK_ISSUES:
            return True
    return False


def _calculate_contingency_rate(issues: List[str], condition_score: int, *, low: bool) -> float:
    base = 0.08 if low else 0.14
    if _major_risk_issue(issues):
        base += 0.04
    if condition_score < 50:
        base += 0.03
    return min(base, 0.22)


def _estimate_timeline_weeks(
    *,
    sqft: float,
    issue_count: int,
    renovation_class: str,
    days_on_market: int,
    age_score_points: int = 0,
) -> tuple[int, int]:
    base = 3
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


def _calculate_confidence_score(
    *,
    condition_score: int,
    issue_count: int,
    gap_score_points: int = 0,
    age_score_points: int = 0,
) -> tuple[Literal["LOW", "MEDIUM", "HIGH"], int]:
    score = 50
    if issue_count == 0 and condition_score >= 80:
        score += 10
    elif issue_count <= 1 and condition_score >= 70:
        score += 6
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


def _classify_renovation_class(condition_score: int, issues: List[str]) -> Literal["Cosmetic", "Moderate", "Heavy", "Full Gut"]:
    major = _major_risk_issue(issues)
    issue_count = len(issues)
    if major and condition_score < 40:
        return "Full Gut"
    if major or condition_score < 55 or issue_count >= 7:
        return "Heavy"
    if condition_score < 75 or issue_count >= 3:
        return "Moderate"
    return "Cosmetic"


def _build_suggested_work_items(issues: List[str], room_type: str) -> List[str]:
    normalized = {(i or "").strip().lower() for i in issues}
    items: list[str] = []
    _add_work_item_when_matched(
        normalized,
        items,
        ("structural damage", "major wall cracks", "foundation", "foundation cracks"),
        "structural stabilization",
    )
    _add_work_item_when_matched(
        normalized,
        items,
        ("water damage", "leak", "mold", "plumbing"),
        "water/moisture remediation",
    )
    _add_work_item_when_matched(
        normalized,
        items,
        ("fire damage", "smoke damage"),
        "fire/smoke remediation",
    )
    _add_work_item_when_matched(
        normalized,
        items,
        ("electrical", "rewire", "wiring"),
        "electrical safety upgrades",
    )
    _add_work_item_when_matched(
        normalized,
        items,
        ("roof damage", "roof", "sagging roof"),
        "roof repairs",
    )
    _add_work_item_when_matched(normalized, items, ("paint wear", "stains", "paint", "damaged ceiling"), "paint")
    _add_work_item_when_matched(normalized, items, ("floor damage", "outdated flooring", "carpet", "floor"), "flooring")
    _add_work_item_when_matched(normalized, items, ("outdated cabinets", "damaged appliances", "cabinet", "kitchen"), "kitchen update")
    _add_work_item_when_matched(
        normalized,
        items,
        ("old bathroom", "old tiles", "missing grout", "dirty grout", "broken fixtures", "bath", "tile"),
        "bathroom update",
    )
    _add_work_item_when_matched(
        normalized,
        items,
        ("roof damage", "peeling exterior paint", "cracked driveway", "poor landscaping", "rotted wood", "exterior", "garage damage"),
        "exterior repairs",
    )

    if not items:
        items.extend(_build_room_fallback_work_items(room_type))
    return items[:8]


def _add_work_item_when_matched(
    normalized_issues: set[str],
    items: list[str],
    keywords: tuple[str, ...],
    work_item: str,
) -> None:
    if any(any(_text_has_term(issue, keyword) for keyword in keywords) for issue in normalized_issues):
        items.append(work_item)


def _build_room_fallback_work_items(room_type: str) -> list[str]:
    room = (room_type or "").lower()
    if "kitchen" in room:
        return ["kitchen refresh"]
    if "bath" in room:
        return ["bathroom refresh"]
    if "exterior" in room:
        return ["exterior refresh"]
    return ["paint", "minor repairs"]


def _build_explanation_summary(
    *,
    renovation_class: str,
    room_type: str,
    issues: List[str],
    desired_quality_level: str,
    impacted_elements: List[str],
    suggested_work_items: List[str],
    selected_elements: List[str] | None = None,
    user_scope_categories: List[str] | None = None,
) -> str:
    selected_names = [str(x).strip() for x in (selected_elements or []) if str(x).strip()]
    if selected_names:
        focus = ", ".join(selected_names[:5])
        return (
            f"Estimate is scoped to user-selected renovation elements ({focus}) "
            f"with {desired_quality_level} finish assumptions."
        )
    if user_scope_categories and not issues:
        user_scope_names = ", ".join(user_scope_categories[:5])
        return (
            f"Estimate is scoped to user-requested updates ({user_scope_names}) "
            f"with {desired_quality_level} finish assumptions."
        )
    room = (room_type or "unknown").replace(",", ", ")
    element_note = ""
    if impacted_elements:
        element_note = " Focus areas: " + ", ".join(impacted_elements[:5]) + "."
    if issues:
        top_issues = ", ".join(issues[:5])
        work_scope = ", ".join(suggested_work_items[:4]) if suggested_work_items else "targeted remediation"
        return (
            f"Property is classified as {renovation_class} rehab based on detected condition issues. "
            f"Detected issues: {top_issues}. "
            f"Recommended work scope: {work_scope}. "
            f"Estimate assumes {desired_quality_level} finish level in {room} areas."
            f"{element_note}"
        )
    return (
        f"Property is classified as {renovation_class} rehab using available room signals ({room}) "
        f"with {desired_quality_level} finish assumptions.{element_note}"
    )


def _build_impacted_element_outputs(
    room_type: str,
    issues: List[str],
    suggested_work_items: List[str],
    selected_elements: List[str] | None = None,
    user_scope_categories: List[str] | None = None,
) -> tuple[List[str], List[ImpactedElementDetail]]:
    selected_labels, selected_details = _build_selected_element_outputs(selected_elements)
    if selected_labels:
        return selected_labels, selected_details
    if user_scope_categories:
        scope_elements: list[str] = []
        scope_details: list[ImpactedElementDetail] = []
        for category in user_scope_categories:
            label = _CATEGORY_TO_IMPACTED_LABEL.get(category)
            if not label:
                continue
            scope_elements.append(label)
            scope_details.append(
                ImpactedElementDetail(
                    name=label,
                    source="user_selected",
                    confidence=0.92,
                    reason=f"Derived from user-requested scope intent ({category}).",
                )
            )
        if scope_elements:
            return _deduplicate_elements_with_details(scope_elements, scope_details)
    if not issues:
        return [], []

    room = _normalize_token_text(room_type)
    issue_text = " ".join(_normalize_token_text(issue) for issue in issues)
    items_text = " ".join(_normalize_token_text(item) for item in suggested_work_items)
    combined_text = f"{room} {issue_text} {items_text}".strip()

    baseline_elements, baseline_details = _build_room_default_elements(room, room_type)
    rule_elements, rule_details = _detect_impacted_elements(combined_text)
    return _deduplicate_elements_with_details(
        [*baseline_elements, *rule_elements],
        [*baseline_details, *rule_details],
    )


def _build_selected_work_items(renovation_elements: List[str]) -> List[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in renovation_elements or []:
        key = (raw or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(f"{key.replace('_', ' ')} renovation")
    return items[:8]


def _deduplicate_work_items(items: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = _normalize_token_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:8]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
