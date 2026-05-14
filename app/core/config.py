import json
import os
from typing import Any, List, Union

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # S3-compatible object storage (Railway buckets, MinIO, R2, AWS, etc.).
    # Env keys are matched case-insensitively. Optional AWS_* / S3_* aliases fill STORAGE_* when empty.
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    PROJECT_NAME: str = "FastAPI Production App"
    API_V1_STR: str = "/api/v1"

    ENVIRONMENT: str = "local"
    DEBUG: bool = False

    LOG_LEVEL: str = Field(default="CRITICAL", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
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

    # Optional JSON object for request-validation overrides.
    # Example:
    # VALIDATION_RULE_OVERRIDES='{"sqft":{"minimum":500,"min_inclusive":true},"labor_index":{"maximum":3.0}}'
    VALIDATION_RULE_OVERRIDES: dict[str, dict[str, Any]] = {}

    # Optional: OpenAI vision integration for image analysis.
    OPENAI_API_KEY: str = ""
    OPENAI_VISION_MODEL: str = "gpt-4o-mini"
    OPENAI_IMAGE_EDIT_MODEL: str = "gpt-image-1"
    OPENAI_IMAGE_EDIT_TIMEOUT_SECONDS: float = Field(
        default=120.0,
        ge=30.0,
        le=600.0,
        description="Per-attempt HTTP timeout for OpenAI images.edit (image generation can exceed 60s).",
    )
    OPENAI_IMAGE_EDIT_MAX_RETRIES: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Extra retries after the first images.edit attempt for rate limits, timeouts, and 5xx.",
    )
    OPENAI_MODEL: str = ""
    OPENAI_VISION_ENABLED: bool = False
    RENOVATION_IMAGE_STRICT_GUARDRAIL: bool = False

    # Optional: Redis cache for image edits.
    REDIS_URL: str = ""
    REDIS_CACHE_TTL_SECONDS: int = Field(default=3600, ge=60, le=86400)

    # Optional: construction cost factors.
    # BLS provides national labor trend/time factor. RSMeans/Gordian provides
    # ZIP/location-specific regional factor.
    BLS_API_KEY: str = ""
    BLS_BASE_WAGE: float = Field(default=34.50, gt=0.0)
    BLS_CONSTRUCTION_WAGE_SERIES_ID: str = "CES2000000008"

    RSMEANS_API_KEY: str = ""
    RSMEANS_BASE_URL: str = ""
    LOCATION_INDEX_CACHE_TTL_SECONDS: int = Field(default=2_592_000, ge=60, le=31_536_000)

    # Optional: Referer sent when downloading MLS/CDN image URLs (some return 403 without a browser-like Referer).
    # Example: https://www.realty.com/ or the URL your CDN expects.
    IMAGE_DOWNLOAD_REFERER: str = ""
    IMAGE_DOWNLOAD_PROXY_TEMPLATE: str = ""
    STORAGE_ENDPOINT_URL: str = ""
    STORAGE_REGION: str = "auto"
    STORAGE_BUCKET_NAME: str = ""
    STORAGE_ACCESS_KEY_ID: str = ""
    STORAGE_SECRET_ACCESS_KEY: str = ""
    STORAGE_PUBLIC_BASE_URL: str = Field(
        default="",
        description=(
            "HTTPS base for permanent renovated-image links (e.g. R2 public URL or CloudFront). "
            "When set, uploads return {base}/{prefix}/{key} instead of expiring presigned URLs; "
            "objects must be readable at that URL (public bucket, public-read ACL, or CDN origin access)."
        ),
    )
    STORAGE_RENOVATED_IMAGE_PREFIX: str = "renovated"
    STORAGE_PRESIGNED_URL_TTL_SECONDS: int = Field(default=3600, ge=60, le=604800)
    # Railway Object Storage (t3.storageapi.dev) requires virtual-hosted-style; MinIO often uses path.
    STORAGE_S3_ADDRESSING_STYLE: str = Field(
        default="virtual",
        description="S3 addressing style: virtual (Railway bucket default) or path",
    )

    @model_validator(mode="before")
    @classmethod
    def merge_aws_into_storage_fields(cls, data: Any) -> Any:
        """Optional: copy common AWS_* / S3_* env names into STORAGE_* when STORAGE_* is empty (hosting templates)."""
        if not isinstance(data, dict):
            return data
        upper_env = {
            k.upper(): (v.strip() if isinstance(v, str) else str(v or "").strip())
            for k, v in os.environ.items()
            if v is not None and str(v).strip() != ""
        }

        def pick(*keys: str) -> str:
            for k in keys:
                if upper_env.get(k):
                    return upper_env[k]
            return ""

        out = dict(data)

        def get_str(key: str) -> str:
            v = out.get(key)
            if v is None:
                return ""
            return str(v).strip()

        if not get_str("STORAGE_ACCESS_KEY_ID"):
            v = pick("STORAGE_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")
            if v:
                out["STORAGE_ACCESS_KEY_ID"] = v
        if not get_str("STORAGE_SECRET_ACCESS_KEY"):
            v = pick("STORAGE_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY")
            if v:
                out["STORAGE_SECRET_ACCESS_KEY"] = v
        if not get_str("STORAGE_ENDPOINT_URL"):
            v = pick("STORAGE_ENDPOINT_URL", "AWS_ENDPOINT_URL", "S3_ENDPOINT_URL", "S3_ENDPOINT")
            if v:
                out["STORAGE_ENDPOINT_URL"] = v
        if not get_str("STORAGE_BUCKET_NAME"):
            v = pick("STORAGE_BUCKET_NAME", "AWS_S3_BUCKET", "S3_BUCKET_NAME", "BUCKET_NAME")
            if v:
                out["STORAGE_BUCKET_NAME"] = v
        region = get_str("STORAGE_REGION")
        if not region or region.lower() == "auto":
            v = pick("STORAGE_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
            if v:
                out["STORAGE_REGION"] = v

        # Railway bucket UI: "Use virtual-hosted-style URLs" for *.storageapi.dev
        ep = get_str("STORAGE_ENDPOINT_URL")
        if ep and "storageapi.dev" in ep.lower():
            if not get_str("STORAGE_S3_ADDRESSING_STYLE"):
                out["STORAGE_S3_ADDRESSING_STYLE"] = "virtual"
        return out

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

    @field_validator("VALIDATION_RULE_OVERRIDES", mode="before")
    @classmethod
    def parse_validation_rule_overrides(cls, v: object) -> dict[str, dict[str, Any]]:
        if v in (None, ""):
            return {}
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("VALIDATION_RULE_OVERRIDES must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("VALIDATION_RULE_OVERRIDES must decode to an object")
            return {
                str(field): value
                for field, value in parsed.items()
                if isinstance(value, dict)
            }
        if isinstance(v, dict):
            return {
                str(field): value
                for field, value in v.items()
                if isinstance(value, dict)
            }
        raise ValueError("VALIDATION_RULE_OVERRIDES must be a dict or JSON object string")

    @field_validator("IMAGE_DOWNLOAD_REFERER", mode="before")
    @classmethod
    def strip_image_download_referer(cls, v: str | None) -> str:
        return (v or "").strip()

    @field_validator(
        "BLS_API_KEY",
        "BLS_CONSTRUCTION_WAGE_SERIES_ID",
        "RSMEANS_API_KEY",
        "RSMEANS_BASE_URL",
        mode="before",
    )
    @classmethod
    def normalize_location_index_strings(cls, v: str | None) -> str:
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

    @field_validator("STORAGE_S3_ADDRESSING_STYLE", mode="before")
    @classmethod
    def normalize_addressing_style(cls, v: str | None) -> str:
        s = (v or "virtual").strip().lower()
        return s if s in ("path", "virtual") else "virtual"

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
