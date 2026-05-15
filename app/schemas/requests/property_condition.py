from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ConditionImageInput(BaseModel):
    """One listing photo: URL only, base64 only, or both (base64 preferred for MLS/CDN)."""

    url: str | None = Field(
        default=None,
        description="Listing photo URL (metadata / heuristics). Not downloaded when base64 is set.",
    )
    base64: str | None = Field(
        default=None,
        description="Raw base64 or data:image/jpeg;base64,... from your app (use when MLS blocks server download).",
    )

    @model_validator(mode="after")
    def require_url_or_base64(self) -> ConditionImageInput:
        has_url = bool((self.url or "").strip())
        has_b64 = bool((self.base64 or "").strip())
        if not has_url and not has_b64:
            raise ValueError("Each image entry needs url and/or base64.")
        return self


class ConditionScoreRequest(BaseModel):
    property_id: str = Field(..., min_length=1)
    image_urls: list[str] = Field(
        default_factory=list,
        description="Legacy: public URLs the server can download. Cotality/CRMLS often return 403.",
    )
    images: list[ConditionImageInput] = Field(
        default_factory=list,
        description="Preferred for blocked MLS feeds: pass base64 from your frontend or re-hosted S3 URLs.",
    )

    @model_validator(mode="after")
    def require_images(self) -> ConditionScoreRequest:
        urls = [u.strip() for u in self.image_urls if u and u.strip()]
        if not urls and not self.images:
            raise ValueError("Provide at least one image via image_urls or images[].")
        return self
