import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.image_download import image_download_config_summary
from app.core.error_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.middleware import RequestIdMiddleware, SecurityHeadersMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    logger.info("Starting %s (env=%s)", settings.PROJECT_NAME, settings.ENVIRONMENT)
    logger.info(
        "Image download config: renovation=%s condition_score=%s",
        image_download_config_summary("renovation"),
        image_download_config_summary("condition_score"),
    )

    def _warm_condition_score_clip() -> None:
        from app.engine.image_condition.services.image_filter import clip_available

        if clip_available():
            logger.info("Condition-score CLIP ready.")
        else:
            logger.warning(
                "Condition-score CLIP not loaded; dedupe will keep up to 6 photos for OpenAI labeling."
            )

    try:
        await asyncio.to_thread(_warm_condition_score_clip)
    except Exception as exc:
        logger.warning("Condition-score CLIP warm-up failed: %s", exc)

    try:
        yield
    finally:
        from app.services.location_indices_service import close_http_client

        await close_http_client()


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
        allowed_hosts = {host.strip() for host in settings.TRUSTED_HOSTS if host.strip()}
        # Railway health checks can use internal hostnames. Include them so probes
        # are not rejected with 400 by TrustedHostMiddleware.
        if settings.is_production:
            allowed_hosts.update({"*.up.railway.app", "*.railway.internal", "localhost", "127.0.0.1"})
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(allowed_hosts))

    @app.get("/health")
    async def root_health() -> dict[str, str]:
        """
        Root-level health endpoint for platform checks.
        """
        return {
            "status": "ok",
            "service": settings.PROJECT_NAME,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/")
    async def root() -> dict[str, str]:
        """
        Minimal root endpoint so load balancers always receive a fast 200.
        """
        return {"status": "ok"}

    app.include_router(api_router, prefix=settings.API_V1_STR)

    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=False), name="ui")

    return app


app = create_app()
