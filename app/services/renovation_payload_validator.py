from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypedDict

from fastapi import HTTPException

from app.core.config import settings
from app.schemas.requests.renovation import RenovationEstimateRequest

_ALLOWED_QUALITY_LEVELS = {"cosmetic", "standard", "premium", "luxury"}
_VALIDATION_ERROR_CODE = "VALIDATION_ERROR"
_REQUIRED_CONDITION_SOURCE_MESSAGE = "Provide either image_url or condition_score fallback."
_ALLOWED_RENOVATION_ELEMENTS = {
    "flooring",
    "paint",
    "lighting",
    "furniture",
    "roof",
    "cabinet",
    "window",
    "stair",
    "door",
}
_MIN_RENOVATION_CONTEXT_TOKENS = 1
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
    "backsplash": "cabinet",
    "doors": "door",

}
_RENOVATION_DOMAIN_KEYWORDS = {
    "home",
    "house",
    "property",
    "room",
    "interior",
    "exterior",
    "renovate",
    "renovation",
    "remodel",
    "repair",
    "upgrade",
    "replace",
    "refinish",
    "paint",
    "wall",
    "ceiling",
    "floor",
    "flooring",
    "tile",
    "lighting",
    "light",
    "fixture",
    "furniture",
    "kitchen",
    "bathroom",
    "cabinet",
    "counter",
    "countertop",
    "backsplash",
    "door",
    "window",
    "stair",
}


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


def _normalize_renovation_elements(value: object) -> list[str]:
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
        canonical = _RENOVATION_ELEMENT_ALIASES.get(key, key)
        if canonical not in _ALLOWED_RENOVATION_ELEMENTS:
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


def _find_invalid_renovation_elements(value: object) -> list[str]:
    raw = _extract_raw_renovation_elements(value)
    invalid: list[str] = []
    seen: set[str] = set()
    for item in raw:
        normalized = item.lower()
        canonical = _RENOVATION_ELEMENT_ALIASES.get(normalized, normalized)
        if canonical in _ALLOWED_RENOVATION_ELEMENTS:
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


def _is_renovation_domain_instruction(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    token_hits = 0
    for keyword in _RENOVATION_DOMAIN_KEYWORDS:
        if keyword in lowered:
            token_hits += 1
            if token_hits >= _MIN_RENOVATION_CONTEXT_TOKENS:
                return True
    return False


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
    if invalid_renovation_elements:
        allowed = ", ".join(sorted(_ALLOWED_RENOVATION_ELEMENTS))
        invalid = ", ".join(invalid_renovation_elements)
        _append_error(
            errors,
            field="renovation_elements",
            message=f"Unsupported element(s): {invalid}. Allowed values: {allowed}.",
        )
    if payload.user_inputs and not _is_renovation_domain_instruction(payload.user_inputs):
        _append_error(
            errors,
            field="user_inputs",
            message=(
                "Instruction must be related to home renovation scope only "
                "(e.g., flooring, paint, lighting, furniture, room updates)."
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
    invalid_renovation_elements = _find_invalid_renovation_elements(payload.renovation_elements)
    normalized = payload.model_copy(
        update={
            "image_url": _normalize_optional_string(payload.image_url),
            "address": _normalize_optional_string(payload.address),
            "city": _normalize_optional_string(payload.city),
            "zip_code": _normalize_optional_string(payload.zip_code),
            "property_type": _normalize_optional_string(payload.property_type, fallback="SFR"),
            "listing_description": _normalize_optional_string(payload.listing_description),
            "listing_status": _normalize_optional_string(payload.listing_status),
            "target_renovation_style": _normalize_optional_string(
                payload.target_renovation_style,
                fallback="investor_standard",
            ),
            "type_of_renovation": _normalize_optional_string(
                payload.type_of_renovation,
                fallback="interior",
            ),
            "visual_type": _normalize_optional_string(
                payload.visual_type,
                fallback="select_elements_to_renovate",
            ),
            "reference_image_url": _normalize_optional_string(payload.reference_image_url),
            "desired_quality_level": _normalize_desired_quality_level(payload.desired_quality_level),
            "renovation_elements": _normalize_renovation_elements(payload.renovation_elements),
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
