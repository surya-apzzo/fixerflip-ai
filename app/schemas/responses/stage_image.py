from pydantic import BaseModel, Field


class StageListingImageResponse(BaseModel):
    staged_source_image_url: str
    effective_property_id: str
    storage_key: str | None = None
    source: str = Field(
        description="already_staged | s3_cache | client_upload | download",
    )
