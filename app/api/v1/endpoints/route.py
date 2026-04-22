from fastapi import APIRouter

from app.engine.renovation_engine.vision_analysis import analyze_renovation_image_url
from app.schemas.requests.renovation import RenovationEstimateRequest
from app.schemas.responses.condition import ImageConditionResult
from app.schemas.responses.renovation import RenovationEstimateResponse
from app.services.renovation_service import build_renovation_estimate

router = APIRouter(prefix="/renovation")


@router.post("/estimate", response_model=RenovationEstimateResponse)
async def renovation_estimate(payload: RenovationEstimateRequest) -> RenovationEstimateResponse:
    return await build_renovation_estimate(payload)



@router.post("/image-condition", response_model=ImageConditionResult)
async def image_condition(image_url: str) -> ImageConditionResult:
    return await analyze_renovation_image_url(image_url)
