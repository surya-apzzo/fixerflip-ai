from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/health")


@router.get("")
async def health_check() -> dict[str, str]:
    """
    Lightweight health endpoint for uptime/load-balancer checks.
    """
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
    }
