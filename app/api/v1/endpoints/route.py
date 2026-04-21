from fastapi import APIRouter

from app.schemas.renovation_api import RenovationEstimateRequest, RenovationEstimateResponse
from app.services.renovation_service import build_renovation_estimate

router = APIRouter(prefix="/renovation")


@router.post("/estimate", response_model=RenovationEstimateResponse)
async def renovation_estimate(payload: RenovationEstimateRequest) -> RenovationEstimateResponse:
    return await build_renovation_estimate(payload)