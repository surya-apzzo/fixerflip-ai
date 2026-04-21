from typing import List, Literal
from pydantic import BaseModel, Field
from app.schemas.line_item import RenovationLineItem


class ImpactedElementDetail(BaseModel):
    name: str
    source: Literal["room_baseline", "issue_or_work_item", "user_selected"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class RenovationEstimate(BaseModel):
    renovation_class: Literal["Cosmetic", "Moderate", "Heavy", "Full Gut"]
    minimum_cost: int
    maximum_cost: int
    minimum_timeline_weeks: int
    maximum_timeline_weeks: int
    confidence_label: Literal["LOW", "MEDIUM", "HIGH"]
    confidence_score: int = Field(ge=0, le=100)
    suggested_work_items: List[str]
    impacted_elements: List[str] = Field(default_factory=list)
    impacted_element_details: List[ImpactedElementDetail] = Field(default_factory=list)
    explanation_summary: str
    line_items: List[RenovationLineItem]
    assumptions: List[str]


class RenovationEstimateInput(BaseModel):
    sqft: float = Field(gt=0)
    beds: int = Field(ge=0)
    baths: float = Field(ge=0)
    zip_code: str = ""
    condition_score: int = Field(ge=0, le=100)
    issues: List[str] = Field(default_factory=list)
    room_type: str = "unknown"
    labor_index: float = Field(default=1.0, ge=0.7, le=2.5)
    material_index: float = Field(default=1.0, ge=0.7, le=2.5)
    desired_quality_level: Literal["cosmetic", "standard", "premium", "luxury"] = "standard"
    target_renovation_style: str = "investor_standard"
    address: str = ""
    city: str = ""
    property_type: str = "SFR"
    year_built: int | None = None
    lot_size: float = Field(default=0.0, ge=0.0)
    listing_price: float = Field(default=0.0, ge=0.0)
    listing_description: str = ""
    listing_status: str = ""
    days_on_market: int = Field(default=0, ge=0)
    avg_area_price_per_sqft: float = Field(default=0.0, ge=0.0)
    years_since_last_sale: int | None = Field(default=None, ge=0)
    permit_years_since_last: int | None = Field(default=None, ge=0)
    renovation_elements: List[str] = Field(default_factory=list)
    user_inputs: str = ""
