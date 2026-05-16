from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ConditionScoreRequest(BaseModel):
    property_id: str = Field(..., min_length=1)
    image_urls: list[str] = Field(
        ...,
        min_length=1,
        description="Listing photo URLs. The server downloads each URL for filtering and vision scoring.",
    )

    @model_validator(mode="after")
    def require_image_urls(self) -> ConditionScoreRequest:
        urls = [u.strip() for u in self.image_urls if u and u.strip()]
        if not urls:
            raise ValueError("Provide at least one non-empty URL in image_urls.")
        return self
