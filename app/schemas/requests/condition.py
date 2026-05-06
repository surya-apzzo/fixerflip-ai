from __future__ import annotations

from pydantic import BaseModel, Field


class ImageConditionRequest(BaseModel):
    image_url: str = Field(..., description="Publicly reachable image URL to analyze.")

