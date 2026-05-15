from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from fastapi import HTTPException

from app.core.config import settings
from app.schemas.requests.renovation import RenovationEstimateRequest

_ALLOWED_QUALITY_LEVELS = {"cosmetic", "standard", "premium", "luxury"}
_VALIDATION_ERROR_CODE = "VALIDATION_ERROR"
_REQUIRED_CONDITION_SOURCE_MESSAGE = "Provide either image_url or condition_score fallback."
_SINGLE_IMAGE_URL_MESSAGE = (
    "Renovation estimate accepts exactly one image_url per request. Send one photo per API call."
)


def _contains_multiple_http_urls(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return False
    return len(re.findall(r"https?://", s, flags=re.IGNORECASE)) > 1
_ALLOWED_RENOVATION_ELEMENTS_INTERIOR: frozenset[str] = frozenset(
    {
        "flooring",
        "paint",
        "lighting",
        "furniture",
        "ceiling",   # renamed from "roof" for clarity
        "cabinet",
        "window",
        "stair",
        "door",
    }
)

_ALLOWED_RENOVATION_ELEMENTS_EXTERIOR: frozenset[str] = frozenset(
    {
        "paint",
        "siding",
        "window",
        "door",
        "landscaping",  # covers yard too, removed redundant "yard"
        "driveway",
        "fence",
        "flooring",
        "stair",
    }
)


def _normalized_renovation_type(value: object) -> str:
    return str(value or "").strip().lower() or "interior"


def _renovation_element_scope(type_of_renovation: object) -> str:
    """Interior vs exterior drives which renovation_elements values are valid."""
    return "exterior" if _normalized_renovation_type(type_of_renovation) == "exterior" else "interior"


def _allowed_renovation_elements(renovation_type: str) -> frozenset[str]:
    if renovation_type == "exterior":
        return _ALLOWED_RENOVATION_ELEMENTS_EXTERIOR
    return _ALLOWED_RENOVATION_ELEMENTS_INTERIOR


def _canonical_renovation_element(raw: str, *, renovation_type: str) -> str:
    """Apply aliases and interior/exterior-specific remaps (e.g. interior roof -> ceiling)."""
    key = (raw or "").strip().lower()
    if not key:
        return ""
    canonical = _RENOVATION_ELEMENT_ALIASES.get(key, key)
    if _renovation_element_scope(renovation_type) == "interior":
        if canonical == "roof":
            return "ceiling"
        return canonical
    # Exterior allowlist has no roof; keep ceiling/paint/siding tokens as-is for validation.
    return canonical


_RENOVATION_ELEMENT_ALIASES = {
    "floor": "flooring",
    "floors": "flooring",
    "tiles": "flooring",
    "tile": "flooring",
    "walls": "paint",
    "wall": "paint",
    "painting": "paint",
    "lights": "lighting",
    "light": "lighting",
    "furnitures": "furniture",
    "roofing": "roof",
    "countertop": "cabinet",
    "cabinets": "cabinet",
    "cabinetry": "cabinet",
    "windows": "window",
    "staircase": "stair",
    "stairs": "stair",
    "backsplash": "cabinet",
    "doors": "door",
    "paints": "paint",
    "exterior paint": "paint",
    "facade paint": "paint",
    "sidings": "siding",
    "cladding": "siding",
    "landscape": "landscaping",
    "garden": "landscaping",
    "lawns": "landscaping",
    "lawn": "landscaping",
    "yards": "landscaping",
    "yard": "landscaping",
    "curb appeal": "landscaping",
    "popcorn ceiling": "ceiling",
    "ceilings": "ceiling",
    "driveways": "driveway",
    "fencing": "fence",
    "fences": "fence",
    "deck": "flooring",
    "decks": "flooring",
    "patio": "flooring",
    "patios": "flooring",
    "porch": "flooring",
    "pavers": "flooring",
    "paver": "flooring",
}

# user_inputs: free-form add/remove/change is OK, but content must stay on
# residential / house / property work (word boundaries; multi-word phrases first).
_HOME_SCOPE_PHRASES: tuple[str, ...] = tuple(
    p.strip()
    for p in (
        "living room, family room, dining room, laundry room, mud room, powder room, "
        "master bath, master bathroom, half bath, full bath, walk in closet, walk-in closet, "
        "crown molding, water heater, air conditioning, dry wall, open concept, hot tub, home office"
    )
    .split(",")
    if p.strip()
)

# One string → split; keeps residential keyword coverage without a huge literal set.
_HOME_SCOPE_WORDS: frozenset[str] = frozenset(
    """
    home house housing household property residence residential dwelling
    condo townhouse duplex triplex apartment flat bungalow cottage mansion homestead
    lot yard curb driveway mailbox interior exterior indoor outdoor facade envelope
    kitchen bathroom bath bedroom bedrooms room rooms loft den basement cellar attic crawlspace crawl
    garage carport mudroom pantry closet closets foyer hallway hall landing stair stairs staircase railing
    balcony terrace sunroom nursery office study
    wall walls ceiling ceilings floor flooring subfloor tile tiles grout carpet hardwood linoleum vinyl laminate
    paint primer trim molding baseboard wainscoting drywall plaster insulation stud studs framing beam joist rafter
    shingle shingles gutter gutters downspout flashing skylight window windows door doors
    cabinet cabinets cabinetry countertop countertops counter counters island backsplash
    sink sinks faucet faucets tub bathtub shower toilet vanity mirror
    lighting fixture fixtures chandelier sconce outlet outlets switch breaker wiring electrical plumbing pipes drain sewer
    heater hvac furnace duct ducts ductwork ventilation boiler radiator fireplace chimney mantel
    roof roofing siding brick stucco veneer porch deck patio fence fencing gate pool spa landscape landscaping furniture
    renovate renovation remodel remodeling rehab repair repairs upgrade upgrades replace refinish restore refresh
    modernize modernise gut demolish finishes layout
    """.split()
)
_OUT_OF_HOME_SCOPE_PATTERN = re.compile(
    r"\b(?:"
    r"car|truck|suv|vehicle|automotive|boat|yacht|aircraft|airplane|helicopter|"
    r"motorcycle|bicycle|scooter|phone|smartphone|laptop|computer|tablet|"
    r"software|website|app"
    r")\b",
    re.IGNORECASE,
)
_HOME_SCOPE_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(w) for w in sorted(_HOME_SCOPE_WORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


class ValidationErrorItem(TypedDict):
    field: str
    message: str


class NumericValidationRuleOverride(TypedDict, total=False):
    minimum: float | None
    maximum: float | None
    min_inclusive: bool
    max_inclusive: bool
    message: str


@dataclass(frozen=True, slots=True)
class NumericValidationRule:
    field: str
    message: str
    minimum: float | None = None
    maximum: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True


def _normalize_renovation_elements(value: object, *, renovation_type: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, list):
        raw = [str(x).strip() for x in value if str(x).strip()]
    else:
        return []

    allowed = _allowed_renovation_elements(renovation_type)
    seen: set[str] = set()
    output: list[str] = []
    for item in raw:
        canonical = _canonical_renovation_element(item, renovation_type=renovation_type)
        if canonical not in allowed:
            continue
        key = canonical
        if key in seen:
            continue
        seen.add(key)
        output.append(key)
        if len(output) >= 5:
            break
    return output


def _extract_raw_renovation_elements(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _find_invalid_renovation_elements(value: object, *, renovation_type: str) -> list[str]:
    raw = _extract_raw_renovation_elements(value)
    allowed = _allowed_renovation_elements(renovation_type)
    invalid: list[str] = []
    seen: set[str] = set()
    for item in raw:
        normalized = item.lower()
        canonical = _canonical_renovation_element(item, renovation_type=renovation_type)
        if canonical in allowed:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        invalid.append(item)
    return invalid


def _normalize_issue_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, list):
        raw = [str(x).strip() for x in value if str(x).strip()]
    else:
        return []

    seen: set[str] = set()
    output: list[str] = []
    for item in raw:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= 8:
            break
    return output


def _normalize_desired_quality_level(value: object) -> str:
    raw = str(value or "standard").strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "standard investor rehab": "standard",
        "investor rehab": "standard",
        "standard rehab": "standard",
        "cosmetic rehab": "cosmetic",
        "light rehab": "cosmetic",
        "premium rehab": "premium",
        "luxury rehab": "luxury",

    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in _ALLOWED_QUALITY_LEVELS else "standard"


# select_elements_to_renovate
# upload my own reference photo
# choose a existing renovation style
# or provide custom instructions related to renovation scope
def _normalize_optional_string(value: object, *, fallback: str = "") -> str:
    normalized = str(value or "").strip()
    return normalized or fallback


def _is_home_or_property_instruction(text: str) -> bool:
    """Allow any wording; require residential/property scope, not vehicles/devices/etc."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    if _OUT_OF_HOME_SCOPE_PATTERN.search(lowered):
        return False
    if any(p in lowered for p in _HOME_SCOPE_PHRASES):
        return True
    return bool(_HOME_SCOPE_PATTERN.search(lowered))


def _append_error(errors: list[ValidationErrorItem], *, field: str, message: str) -> None:
    errors.append({"field": field, "message": message})


def _format_numeric_bound(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _build_numeric_validation_message(
    *,
    field: str,
    minimum: float | None,
    maximum: float | None,
    min_inclusive: bool,
    max_inclusive: bool,
) -> str:
    if minimum is not None and maximum is not None and min_inclusive and max_inclusive:
        return f"{field} must be between {_format_numeric_bound(minimum)} and {_format_numeric_bound(maximum)}"
    if minimum is not None and maximum is not None:
        min_operator = ">=" if min_inclusive else ">"
        max_operator = "<=" if max_inclusive else "<"
        return (
            f"{field} must be {min_operator} {_format_numeric_bound(minimum)} "
            f"and {max_operator} {_format_numeric_bound(maximum)}"
        )
    if minimum is not None:
        if minimum == 0 and not min_inclusive:
            return f"{field} must be greater than 0"
        operator = ">=" if min_inclusive else ">"
        return f"{field} must be {operator} {_format_numeric_bound(minimum)}"
    if maximum is not None:
        operator = "<=" if max_inclusive else "<"
        return f"{field} must be {operator} {_format_numeric_bound(maximum)}"
    return f"{field} is invalid"


def _build_numeric_rule(
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    message: str | None = None,
) -> NumericValidationRule:
    return NumericValidationRule(
        field=field,
        minimum=minimum,
        maximum=maximum,
        min_inclusive=min_inclusive,
        max_inclusive=max_inclusive,
        message=message
        or _build_numeric_validation_message(
            field=field,
            minimum=minimum,
            maximum=maximum,
            min_inclusive=min_inclusive,
            max_inclusive=max_inclusive,
        ),
    )


def _coerce_rule_override_float(*, field: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VALIDATION_RULE_OVERRIDES[{field!r}][{key!r}] must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"VALIDATION_RULE_OVERRIDES[{field!r}][{key!r}] must be finite")
    return numeric


def _coerce_rule_override_bool(*, field: str, key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"VALIDATION_RULE_OVERRIDES[{field!r}][{key!r}] must be boolean")


def _resolve_numeric_rule_override(field: str) -> NumericValidationRuleOverride:
    raw = settings.VALIDATION_RULE_OVERRIDES.get(field, {})
    if not isinstance(raw, dict):
        return {}

    override: NumericValidationRuleOverride = {}
    if "minimum" in raw:
        override["minimum"] = _coerce_rule_override_float(field=field, key="minimum", value=raw["minimum"])
    if "maximum" in raw:
        override["maximum"] = _coerce_rule_override_float(field=field, key="maximum", value=raw["maximum"])
    if "min_inclusive" in raw:
        override["min_inclusive"] = _coerce_rule_override_bool(
            field=field,
            key="min_inclusive",
            value=raw["min_inclusive"],
        )
    if "max_inclusive" in raw:
        override["max_inclusive"] = _coerce_rule_override_bool(
            field=field,
            key="max_inclusive",
            value=raw["max_inclusive"],
        )
    if "message" in raw and raw["message"] is not None:
        override["message"] = str(raw["message"]).strip()
    return override


def _apply_numeric_rule_override(
    base_rule: NumericValidationRule,
    override: NumericValidationRuleOverride,
) -> NumericValidationRule:
    minimum = override.get("minimum", base_rule.minimum)
    maximum = override.get("maximum", base_rule.maximum)
    min_inclusive = override.get("min_inclusive", base_rule.min_inclusive)
    max_inclusive = override.get("max_inclusive", base_rule.max_inclusive)
    message = override.get("message")
    return _build_numeric_rule(
        field=base_rule.field,
        minimum=minimum,
        maximum=maximum,
        min_inclusive=min_inclusive,
        max_inclusive=max_inclusive,
        message=message,
    )


def _base_numeric_validation_rules(_payload: RenovationEstimateRequest) -> tuple[NumericValidationRule, ...]:
    return (
        _build_numeric_rule(
            field="sqft",
            minimum=0.0,
            min_inclusive=False,
        ),
        _build_numeric_rule(field="beds", minimum=0.0),
        _build_numeric_rule(field="baths", minimum=0.0),
        _build_numeric_rule(field="lot_size", minimum=0.0),
        _build_numeric_rule(field="listing_price", minimum=0.0),
        _build_numeric_rule(field="days_on_market", minimum=0.0),
        _build_numeric_rule(field="avg_area_price_per_sqft", minimum=0.0),
        _build_numeric_rule(field="years_since_last_sale", minimum=0.0),
        _build_numeric_rule(field="permit_years_since_last", minimum=0.0),
        _build_numeric_rule(field="condition_score", minimum=0.0, maximum=100.0),
        _build_numeric_rule(field="labor_index", minimum=0.7, maximum=2.5),
        _build_numeric_rule(field="material_index", minimum=0.7, maximum=2.5),
        _build_numeric_rule(field="time_factor", minimum=0.5, maximum=2.5),
        _build_numeric_rule(field="location_factor", minimum=0.5, maximum=2.5),
    )


def _build_numeric_validation_rules(payload: RenovationEstimateRequest) -> tuple[NumericValidationRule, ...]:
    dynamic_rules: list[NumericValidationRule] = []
    for base_rule in _base_numeric_validation_rules(payload):
        override = _resolve_numeric_rule_override(base_rule.field)
        dynamic_rules.append(_apply_numeric_rule_override(base_rule, override))
    return tuple(dynamic_rules)


def _is_within_bounds(value: float, rule: NumericValidationRule) -> bool:
    if rule.minimum is not None:
        if rule.min_inclusive and value < rule.minimum:
            return False
        if not rule.min_inclusive and value <= rule.minimum:
            return False
    if rule.maximum is not None:
        if rule.max_inclusive and value > rule.maximum:
            return False
        if not rule.max_inclusive and value >= rule.maximum:
            return False
    return True


def _validate_numeric_rule(
    *,
    errors: list[ValidationErrorItem],
    payload: RenovationEstimateRequest,
    rule: NumericValidationRule,
) -> None:
    value = getattr(payload, rule.field)
    if value is None:
        return

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        _append_error(
            errors,
            field=rule.field,
            message=f"{rule.field} must be a finite number",
        )
        return

    if not _is_within_bounds(numeric_value, rule):
        _append_error(errors, field=rule.field, message=rule.message)


def _validate_payload_values(
    payload: RenovationEstimateRequest,
    *,
    invalid_renovation_elements: list[str] | None = None,
) -> None:
    errors: list[ValidationErrorItem] = []

    for rule in _build_numeric_validation_rules(payload):
        _validate_numeric_rule(errors=errors, payload=payload, rule=rule)

    if not payload.image_url and payload.condition_score is None:
        _append_error(
            errors,
            field="image_url",
            message=_REQUIRED_CONDITION_SOURCE_MESSAGE,
        )
    if payload.image_url and _contains_multiple_http_urls(payload.image_url):
        _append_error(errors, field="image_url", message=_SINGLE_IMAGE_URL_MESSAGE)
    if invalid_renovation_elements:
        allowed = ", ".join(sorted(_allowed_renovation_elements(_renovation_element_scope(payload.type_of_renovation))))
        invalid = ", ".join(invalid_renovation_elements)
        _append_error(
            errors,
            field="renovation_elements",
            message=f"Unsupported element(s): {invalid}. Allowed values: {allowed}.",
        )
    if payload.user_inputs and not _is_home_or_property_instruction(payload.user_inputs):
        _append_error(
            errors,
            field="user_inputs",
            message=(
                "Instructions must be about home, house, or residential property work only "
                "(rooms, finishes, structure, yard, etc.). Vehicles, electronics, and unrelated topics are not supported."
            ),
        )

    if errors:
        raise HTTPException(
            status_code=422,
            detail={"code": _VALIDATION_ERROR_CODE, "errors": errors},
        )


def validate_and_normalize_renovation_payload(
    payload: RenovationEstimateRequest,
) -> RenovationEstimateRequest:
    type_stored = _normalize_optional_string(payload.type_of_renovation, fallback="interior")
    renovation_type_for_elements = _renovation_element_scope(type_stored)
    invalid_renovation_elements = _find_invalid_renovation_elements(
        payload.renovation_elements,
        renovation_type=renovation_type_for_elements,
    )
    normalized = payload.model_copy(
        update={
            "image_url": _normalize_optional_string(payload.image_url),
            "address": _normalize_optional_string(payload.address),
            "city": _normalize_optional_string(payload.city),
            "zip_code": _normalize_optional_string(payload.zip_code),
            "property_type": _normalize_optional_string(payload.property_type, fallback="SFR"),
            "listing_description": _normalize_optional_string(payload.listing_description),
            "listing_status": _normalize_optional_string(payload.listing_status),
            "type_of_renovation": type_stored,
            "visual_type": _normalize_optional_string(
                payload.visual_type,
                fallback="select_elements_to_renovate",
            ),
            "reference_image_url": _normalize_optional_string(payload.reference_image_url),
            "desired_quality_level": _normalize_desired_quality_level(payload.desired_quality_level),
            "renovation_elements": _normalize_renovation_elements(
                payload.renovation_elements,
                renovation_type=renovation_type_for_elements,
            ),
            "issues": _normalize_issue_list(payload.issues),
            "room_type": _normalize_optional_string(payload.room_type, fallback="unknown"),
            "user_inputs": _normalize_optional_string(payload.user_inputs),
        }
    )
    _validate_payload_values(
        normalized,
        invalid_renovation_elements=invalid_renovation_elements,
    )
    return normalized
