from __future__ import annotations

from fastapi import HTTPException

from app.schemas.requests.renovation import RenovationEstimateRequest


_ALLOWED_QUALITY_LEVELS = {"cosmetic", "standard", "premium", "luxury"}


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


def _normalize_desired_quality_level(value: object) -> str:
    raw = str(value or "standard").strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "standard investor rehab": "standard",
        "investor rehab": "standard",
        "standard rehab": "standard",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in _ALLOWED_QUALITY_LEVELS else "standard"


def _validate_payload_values(payload: RenovationEstimateRequest) -> None:
    errors: list[dict[str, str]] = []

    if payload.sqft <= 0:
        errors.append({"field": "sqft", "message": "sqft must be greater than 0"})
    if payload.beds < 0:
        errors.append({"field": "beds", "message": "beds must be >= 0"})
    if payload.baths < 0:
        errors.append({"field": "baths", "message": "baths must be >= 0"})
    if payload.lot_size < 0:
        errors.append({"field": "lot_size", "message": "lot_size must be >= 0"})
    if payload.listing_price < 0:
        errors.append({"field": "listing_price", "message": "listing_price must be >= 0"})
    if payload.days_on_market < 0:
        errors.append({"field": "days_on_market", "message": "days_on_market must be >= 0"})
    if payload.avg_area_price_per_sqft < 0:
        errors.append({"field": "avg_area_price_per_sqft", "message": "avg_area_price_per_sqft must be >= 0"})
    if payload.years_since_last_sale is not None and payload.years_since_last_sale < 0:
        errors.append({"field": "years_since_last_sale", "message": "years_since_last_sale must be >= 0"})
    if payload.permit_years_since_last is not None and payload.permit_years_since_last < 0:
        errors.append({"field": "permit_years_since_last", "message": "permit_years_since_last must be >= 0"})
    if payload.condition_score is not None and not (0 <= payload.condition_score <= 100):
        errors.append({"field": "condition_score", "message": "condition_score must be between 0 and 100"})
    if not (0.7 <= payload.labor_index <= 2.5):
        errors.append({"field": "labor_index", "message": "labor_index must be between 0.7 and 2.5"})
    if not (0.7 <= payload.material_index <= 2.5):
        errors.append({"field": "material_index", "message": "material_index must be between 0.7 and 2.5"})

    if errors:
        raise HTTPException(status_code=422, detail={"code": "VALIDATION_ERROR", "errors": errors})


def validate_and_normalize_renovation_payload(
    payload: RenovationEstimateRequest,
) -> RenovationEstimateRequest:
    normalized = payload.model_copy(
        update={
            "image_url": (payload.image_url or "").strip(),
            "reference_image_url": (payload.reference_image_url or "").strip(),
            "desired_quality_level": _normalize_desired_quality_level(payload.desired_quality_level),
            "renovation_elements": _normalize_renovation_elements(payload.renovation_elements),
        }
    )
    _validate_payload_values(normalized)
    return normalized
