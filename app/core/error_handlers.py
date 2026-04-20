import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": exc.errors(),
                "code": "VALIDATION_ERROR",
            },
        )

    if not settings.DEBUG:

        @app.exception_handler(Exception)
        async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
            if isinstance(exc, HTTPException):
                return await http_exception_handler(request, exc)
            logger.exception("Unhandled exception")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error", "code": "INTERNAL_ERROR"},
            )
