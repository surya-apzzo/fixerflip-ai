import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.middleware import RequestIdMiddleware, SecurityHeadersMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.PROJECT_NAME, settings.ENVIRONMENT)
    yield


def create_app() -> FastAPI:
    setup_logging()
    openapi_url = f"{settings.API_V1_STR}/openapi.json" if settings.ENABLE_OPENAPI else None
    docs_url = "/docs" if settings.ENABLE_OPENAPI else None
    redoc_url = "/redoc" if settings.ENABLE_OPENAPI else None

    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=openapi_url,
        docs_url=docs_url,
        redoc_url=redoc_url,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIdMiddleware)

    # In production/staging, only listed origins are allowed. In local/dev, allow any origin
    # so tests work from 127.0.0.1 vs localhost, IPv6 ([::1]), arbitrary ports (8080, 5500),
    # and file:// without maintaining a fragile allowlist (see BACKEND_CORS_ORIGINS in .env).
    if settings.is_production:
        cors_origins = [str(origin) for origin in settings.BACKEND_CORS_ORIGINS]
    else:
        cors_origins = ["*"]
        logger.info("CORS allows all origins (non-production environment)")

    if cors_origins:
        allow_all = cors_origins == ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=not allow_all,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if settings.TRUSTED_HOSTS:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.TRUSTED_HOSTS)

    app.include_router(api_router, prefix=settings.API_V1_STR)

    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=False), name="ui")

    return app


app = create_app()
