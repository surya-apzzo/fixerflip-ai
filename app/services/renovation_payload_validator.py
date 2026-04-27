from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

from fastapi import HTTPException

from app.schemas.requests.renovation import RenovationEstimateRequest

_ALLOWED_QUALITY_LEVELS = {"cosmetic", "standard", "premium", "luxury"}
_VALIDATION_ERROR_CODE = "VALIDATION_ERROR"
_REQUIRED_CONDITION_SOURCE_MESSAGE = "Provide either image_url or condition_score fallback."


class ValidationErrorItem(TypedDict):
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class NumericValidationRule:
    field: str
    message: str
    minimum: float | None = None
    maximum: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True


_NUMERIC_VALIDATION_RULES: tuple[NumericValidationRule, ...] = (
    NumericValidationRule(
        field="sqft",
        minimum=0.0,
        min_inclusive=False,
        message="sqft must be greater than 0",
    ),
    NumericValidationRule(field="beds", minimum=0.0, message="beds must be >= 0"),
    NumericValidationRule(field="baths", minimum=0.0, message="baths must be >= 0"),
    NumericValidationRule(field="lot_size", minimum=0.0, message="lot_size must be >= 0"),
    NumericValidationRule(field="listing_price", minimum=0.0, message="listing_price must be >= 0"),
    NumericValidationRule(field="days_on_market", minimum=0.0, message="days_on_market must be >= 0"),
    NumericValidationRule(
        field="avg_area_price_per_sqft",
        minimum=0.0,
        message="avg_area_price_per_sqft must be >= 0",
    ),
    NumericValidationRule(
        field="years_since_last_sale",
        minimum=0.0,
        message="years_since_last_sale must be >= 0",
    ),
    NumericValidationRule(
        field="permit_years_since_last",
        minimum=0.0,
        message="permit_years_since_last must be >= 0",
    ),
    NumericValidationRule(
        field="condition_score",
        minimum=0.0,
        maximum=100.0,
        message="condition_score must be between 0 and 100",
    ),
    NumericValidationRule(
        field="labor_index",
        minimum=0.7,
        maximum=2.5,
        message="labor_index must be between 0.7 and 2.5",
    ),
    NumericValidationRule(
        field="material_index",
        minimum=0.7,
        maximum=2.5,
        message="material_index must be between 0.7 and 2.5",
    ),
)


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
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= 4:
            break
    return output


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
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in _ALLOWED_QUALITY_LEVELS else "standard"


def _normalize_optional_string(value: object, *, fallback: str = "") -> str:
    normalized = str(value or "").strip()
    return normalized or fallback


def _append_error(errors: list[ValidationErrorItem], *, field: str, message: str) -> None:
    errors.append({"field": field, "message": message})


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


def _validate_payload_values(payload: RenovationEstimateRequest) -> None:
    errors: list[ValidationErrorItem] = []

    for rule in _NUMERIC_VALIDATION_RULES:
        _validate_numeric_rule(errors=errors, payload=payload, rule=rule)

    if not payload.image_url and payload.condition_score is None:
        _append_error(
            errors,
            field="image_url",
            message=_REQUIRED_CONDITION_SOURCE_MESSAGE,
        )

    if errors:
        raise HTTPException(
            status_code=422,
            detail={"code": _VALIDATION_ERROR_CODE, "errors": errors},
        )


def validate_and_normalize_renovation_payload(
    payload: RenovationEstimateRequest,
) -> RenovationEstimateRequest:
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
    _validate_payload_values(normalized)
    return normalized
