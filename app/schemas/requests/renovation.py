from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.rules_config import DEFAULT_ADMIN_LABOR_INDEX, DEFAULT_ADMIN_MATERIAL_INDEX


class RenovationEstimateRequest(BaseModel):
    image_url: str = ""
    address: str = ""
    city: str = ""
    sqft: float
    beds: int
    baths: float
    lot_size: float = 0.0
    year_built: int | None = None
    property_type: str = "SFR"
    listing_price: float = 0.0
    listing_description: str = ""
    listing_status: str = ""
    days_on_market: int = 0
    avg_area_price_per_sqft: float = 0.0
    years_since_last_sale: int | None = None
    permit_years_since_last: int | None = None
    zip_code: str = ""
    target_renovation_style: str = "investor_standard"
    desired_quality_level: str = "standard"
    labor_index: float = DEFAULT_ADMIN_LABOR_INDEX
    material_index: float = DEFAULT_ADMIN_MATERIAL_INDEX
    type_of_renovation: str = "interior"
    visual_type: str = "select_elements_to_renovate"
    reference_image_url: str = ""
    renovation_elements: list[str] = Field(default_factory=list)
    condition_score: int | None = None
    issues: list[str] = Field(default_factory=list)
    room_type: str = "unknown"
    user_inputs: str = ""
