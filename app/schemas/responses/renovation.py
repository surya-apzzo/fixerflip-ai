from pydantic import BaseModel

from app.schemas.responses.condition import ImageConditionResult
from app.schemas.responses.estimate import RenovationEstimate


class RenovatedImageResult(BaseModel):
    renovated_image_url: str


class RenovationEstimateResponse(BaseModel):
    image_condition: ImageConditionResult
    estimate: RenovationEstimate
    renovated_image: RenovatedImageResult | None = None
    renovated_image_error: str | None = None
