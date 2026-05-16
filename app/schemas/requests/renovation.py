from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.rules_config import DEFAULT_ADMIN_LABOR_INDEX, DEFAULT_ADMIN_MATERIAL_INDEX


class RenovationEstimateRequest(BaseModel):
    property_id: str = Field(
        default="",
        description=(
            "Listing/property id for S3 photo cache (renovation/listings/...). "
            "Strongly recommended for Cotality/Trestle URLs on cloud hosts."
        ),
    )
    image_url: str = ""
    source_image_base64: str = Field(
        default="",
        description=(
            "Optional JPEG/PNG as base64 (or data URL). When Cotality blocks server download, "
            "send photo bytes from your app; staged to S3 under renovation/listings/ before vision/edit."
        ),
    )
    sqft: float
    year_built: int | None = None
    listing_price: float = 0.0
    listing_description: str = ""
    days_on_market: int = 0
    avg_area_price_per_sqft: float = 0.0
    years_since_last_sale: int | None = None
    permit_years_since_last: int | None = None
    zip_code: str = ""
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
