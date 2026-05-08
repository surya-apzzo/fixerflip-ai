from __future__ import annotations

from pydantic import BaseModel, Field


class ConditionScoreRequest(BaseModel):
    property_id: str = Field(..., min_length=1)
    image_urls: list[str] = Field(..., min_length=1)

