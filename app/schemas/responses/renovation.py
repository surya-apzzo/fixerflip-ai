from typing import List

from pydantic import BaseModel, Field


class RenovationEstimateResponse(BaseModel):
    renovation_class: str
    estimated_renovation_range: str
    estimated_timeline: str
    suggested_work_items: List[str] = Field(default_factory=list)
    confidence_score: str
    explanation_summary: str
    room_type: str = "unknown"
    condition_score: int = Field(
        ge=0,
        le=100,
        description="Condition score used for this estimate (0–100; vision-derived when image_url is sent, else request value).",
    )
    renovated_image_url: str | None = None
