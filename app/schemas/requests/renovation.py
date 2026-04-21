from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.rules_config import DEFAULT_ADMIN_LABOR_INDEX, DEFAULT_ADMIN_MATERIAL_INDEX


class RenovationEstimateRequest(BaseModel):
    image_url: str = ""
    address: str = ""
    city: str = ""
    sqft: float = Field(gt=0)
    beds: int = Field(ge=0)
    baths: float = Field(ge=0)
    lot_size: float = Field(default=0.0, ge=0.0)
    year_built: int | None = None
    property_type: str = "SFR"
    listing_price: float = Field(default=0.0, ge=0.0)
    listing_description: str = ""
    listing_status: str = ""
    days_on_market: int = Field(default=0, ge=0)
    avg_area_price_per_sqft: float = Field(default=0.0, ge=0.0)
    years_since_last_sale: int | None = Field(default=None, ge=0)
    permit_years_since_last: int | None = Field(default=None, ge=0)
    zip_code: str = ""
    target_renovation_style: str = "investor_standard"
    desired_quality_level: str = "standard"
    labor_index: float = Field(default=DEFAULT_ADMIN_LABOR_INDEX, ge=0.7, le=2.5)
    material_index: float = Field(default=DEFAULT_ADMIN_MATERIAL_INDEX, ge=0.7, le=2.5)
    type_of_renovation: str = "interior"
    visual_type: str = "select_elements_to_renovate"
    reference_image_url: str = ""
    renovation_elements: list[str] = Field(default_factory=list, max_length=4)
    condition_score: int | None = Field(default=None, ge=0, le=100)
    issues: list[str] = Field(default_factory=list)
    room_type: str = "unknown"
    user_inputs: str = ""

    @field_validator("image_url")
    @classmethod
    def strip_optional_image_url(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("reference_image_url")
    @classmethod
    def strip_reference_image_url(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("renovation_elements", mode="before")
    @classmethod
    def normalize_renovation_elements(cls, value: object) -> list[str]:
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

    @field_validator("desired_quality_level", mode="before")
    @classmethod
    def normalize_desired_quality_level(cls, value: object) -> str:
        raw = str(value or "standard").strip().lower().replace("-", " ").replace("_", " ")
        aliases = {
            "standard investor rehab": "standard",
            "investor rehab": "standard",
            "standard rehab": "standard",
        }
        normalized = aliases.get(raw, raw)
        allowed = {"cosmetic", "standard", "premium", "luxury"}
        return normalized if normalized in allowed else "standard"
