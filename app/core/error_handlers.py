import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


def _format_request_validation_error(error: dict[str, Any]) -> dict[str, str]:
    loc = error.get("loc", ())
    field_parts = [str(part) for part in loc if str(part) not in {"body", "query", "path"}]
    field = ".".join(field_parts) if field_parts else "payload"
    return {
        "field": field,
        "message": str(error.get("msg") or "Invalid request."),
    }


def _validation_error_content(errors: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, str]] | str]]:
    return {
        "detail": {
            "code": "VALIDATION_ERROR",
            "errors": [_format_request_validation_error(error) for error in errors],
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_validation_error_content(exc.errors()),
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
