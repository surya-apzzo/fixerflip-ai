import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign/propagate X-Request-ID and log request timing."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        request_id = request.headers.get(settings.REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers[settings.REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response
