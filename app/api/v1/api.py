from fastapi import APIRouter

from app.api.v1.endpoints.renovation import router as renovation_router

api_router = APIRouter()
api_router.include_router(renovation_router, tags=["renovation"])
