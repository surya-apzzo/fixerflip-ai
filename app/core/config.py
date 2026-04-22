from typing import List, Union

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    PROJECT_NAME: str = "FastAPI Production App"
    API_V1_STR: str = "/api/v1"

    ENVIRONMENT: str = "local"
    DEBUG: bool = False

    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    LOG_JSON: bool = Field(
        default=False,
        description="Emit JSON logs (recommended in production behind log aggregators)",
    )

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] | List[str] = []

    # Optional database URI for business-rule storage and runtime configuration.
    # Example: sqlite:///./runtime_rules.db
    DATABASE_URL: str = ""

    RULES_RELOAD_TTL_SECONDS: int = Field(
        default=300,
        ge=60,
        le=86400,
        description="Seconds between runtime rule refreshes when using database-backed business rules.",
    )

    DB_POOL_SIZE: int = Field(default=5, ge=1, le=50)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0, le=50)
    DB_POOL_TIMEOUT: float = Field(default=30.0, ge=1.0, le=120.0)

    # If set, only these Host headers are allowed (production behind a reverse proxy)
    TRUSTED_HOSTS: List[str] = []

    REQUEST_ID_HEADER: str = "X-Request-ID"

    # Set to false in production if you do not want /docs and OpenAPI exposed
    ENABLE_OPENAPI: bool = True

    # Optional: OpenAI vision integration for image analysis.
    OPENAI_API_KEY: str = ""
    OPENAI_VISION_MODEL: str = "gpt-4o-mini"
    OPENAI_IMAGE_EDIT_MODEL: str = "gpt-image-1"
    OPENAI_MODEL: str = ""
    OPENAI_VISION_ENABLED: bool = False

    # Optional: Redis cache for image edits.
    REDIS_URL: str = ""
    REDIS_CACHE_TTL_SECONDS: int = Field(default=3600, ge=60, le=86400)

    # Optional: Referer sent when downloading MLS/CDN image URLs (some return 403 without a browser-like Referer).
    # Example: https://www.realty.com/ or the URL your CDN expects.
    IMAGE_DOWNLOAD_REFERER: str = ""
    STORAGE_ENDPOINT_URL: str = ""
    STORAGE_REGION: str = "auto"
    STORAGE_BUCKET_NAME: str = ""
    STORAGE_ACCESS_KEY_ID: str = ""
    STORAGE_SECRET_ACCESS_KEY: str = ""
    STORAGE_PUBLIC_BASE_URL: str = ""
    STORAGE_RENOVATED_IMAGE_PREFIX: str = "renovated"
    STORAGE_PRESIGNED_URL_TTL_SECONDS: int = Field(default=3600, ge=60, le=604800)

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        if isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @field_validator("OPENAI_API_KEY", "OPENAI_VISION_MODEL", "OPENAI_IMAGE_EDIT_MODEL", "OPENAI_MODEL", mode="before")
    @classmethod
    def normalize_openai_strings(cls, v: str | None) -> str:
        return (v or "").strip()

    @field_validator("IMAGE_DOWNLOAD_REFERER", mode="before")
    @classmethod
    def strip_image_download_referer(cls, v: str | None) -> str:
        return (v or "").strip()

    @field_validator(
        "STORAGE_ENDPOINT_URL",
        "STORAGE_REGION",
        "STORAGE_BUCKET_NAME",
        "STORAGE_ACCESS_KEY_ID",
        "STORAGE_SECRET_ACCESS_KEY",
        "STORAGE_PUBLIC_BASE_URL",
        "STORAGE_RENOVATED_IMAGE_PREFIX",
        mode="before",
    )
    @classmethod
    def normalize_storage_strings(cls, v: str | None) -> str:
        return (v or "").strip()

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ("production", "prod", "staging")

    @property
    def default_openai_vision_model(self) -> str:
        return (self.OPENAI_VISION_MODEL or self.OPENAI_MODEL or "gpt-4o-mini").strip().lower()

    @property
    def default_openai_image_edit_model(self) -> str:
        return (self.OPENAI_IMAGE_EDIT_MODEL or "gpt-image-1").strip().lower()


settings = Settings()
