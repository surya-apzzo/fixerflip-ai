from __future__ import annotations

import asyncio
import base64
import logging
import uuid

from botocore.config import Config

from app.core.config import settings

logger = logging.getLogger(__name__)


def _make_s3_client():
    import boto3

    addressing = (settings.STORAGE_S3_ADDRESSING_STYLE or "virtual").strip().lower()
    if addressing not in ("path", "virtual"):
        addressing = "virtual"
    return boto3.client(
        "s3",
        endpoint_url=settings.STORAGE_ENDPOINT_URL,
        region_name=settings.STORAGE_REGION or "auto",
        aws_access_key_id=settings.STORAGE_ACCESS_KEY_ID,
        aws_secret_access_key=settings.STORAGE_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": addressing}),
    )


def _require_storage_config() -> None:
    required = {
        "STORAGE_ENDPOINT_URL": settings.STORAGE_ENDPOINT_URL,
        "STORAGE_BUCKET_NAME": settings.STORAGE_BUCKET_NAME,
        "STORAGE_ACCESS_KEY_ID": settings.STORAGE_ACCESS_KEY_ID,
        "STORAGE_SECRET_ACCESS_KEY": settings.STORAGE_SECRET_ACCESS_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Storage is not configured. Missing: {', '.join(missing)}")


def _build_public_url(key: str) -> str:
    if settings.STORAGE_PUBLIC_BASE_URL:
        return f"{settings.STORAGE_PUBLIC_BASE_URL.rstrip('/')}/{key}"
    return f"{settings.STORAGE_ENDPOINT_URL.rstrip('/')}/{settings.STORAGE_BUCKET_NAME}/{key}"


def _upload_bytes_to_bucket(image_bytes: bytes, *, key: str, content_type: str) -> str:
    logger.debug(
        "Uploading renovated image endpoint=%s bucket=%s key=%s bytes=%s",
        settings.STORAGE_ENDPOINT_URL,
        settings.STORAGE_BUCKET_NAME,
        key,
        len(image_bytes),
    )

    client = _make_s3_client()

    put_kwargs = {
        "Bucket": settings.STORAGE_BUCKET_NAME,
        "Key": key,
        "Body": image_bytes,
        "ContentType": content_type,
    }
    try:
        client.put_object(**put_kwargs)
    except Exception:
        logger.warning(
            "put_object first attempt failed (bucket=%s key=%s); retrying with ACL=public-read",
            settings.STORAGE_BUCKET_NAME,
            key,
            exc_info=True,
        )
        try:
            client.put_object(**put_kwargs, ACL="public-read")
        except Exception:
            logger.exception(
                "put_object failed after ACL retry (bucket=%s key=%s). Check STORAGE_* and network.",
                settings.STORAGE_BUCKET_NAME,
                key,
            )
            raise

    logger.info(
        "Renovated image stored successfully bucket=%s key=%s bytes=%s",
        settings.STORAGE_BUCKET_NAME,
        key,
        len(image_bytes),
    )

    public_base = (settings.STORAGE_PUBLIC_BASE_URL or "").strip()

    try:
        signed = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.STORAGE_BUCKET_NAME, "Key": key},
            ExpiresIn=settings.STORAGE_PRESIGNED_URL_TTL_SECONDS,
        )
        return signed
    except Exception as exc:
        logger.warning(
            "Presigned GET URL failed (bucket=%s key=%s): %s: %s",
            settings.STORAGE_BUCKET_NAME,
            key,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        if public_base and settings.STORAGE_RENOVATED_RESPONSE_USE_PUBLIC_URL:
            logger.warning(
                "Returning unsigned public URL (STORAGE_RENOVATED_RESPONSE_USE_PUBLIC_URL=true); "
                "bucket must allow anonymous GET at STORAGE_PUBLIC_BASE_URL."
            )
            return _build_public_url(key)
        raise ValueError(
            "Uploaded image but failed to generate a presigned download URL. "
            "Fix STORAGE_* / signing and credentials. Optional: set STORAGE_PUBLIC_BASE_URL and "
            "STORAGE_RENOVATED_RESPONSE_USE_PUBLIC_URL=true only if unsigned URLs work in the browser."
        ) from exc


def presigned_get_url_for_renovated_object_key(key: str) -> str:
    """Mint a fresh presigned GET URL for a key under STORAGE_RENOVATED_IMAGE_PREFIX (GET /bucket/file)."""
    _require_storage_config()
    raw = (key or "").strip()
    if not raw or ".." in raw or raw.startswith("/"):
        raise ValueError("Invalid object key.")
    prefix = settings.STORAGE_RENOVATED_IMAGE_PREFIX.strip("/")
    if not (raw == prefix or raw.startswith(prefix + "/")):
        raise ValueError("Key is not allowed for renovated-image downloads.")

    client = _make_s3_client()
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.STORAGE_BUCKET_NAME, "Key": raw},
            ExpiresIn=settings.STORAGE_PRESIGNED_URL_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning(
            "Presigned GET by key failed (bucket=%s key=%s): %s: %s",
            settings.STORAGE_BUCKET_NAME,
            raw,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise ValueError("Could not generate download URL for this key.") from exc


async def upload_base64_image_to_bucket(*, image_base64: str, media_type: str) -> str:
    _require_storage_config()
    if not image_base64:
        raise ValueError("image_base64 is required.")

    image_bytes = base64.b64decode(image_base64)
    extension = media_type.split("/")[-1].lower() if "/" in media_type else "png"
    key = f"{settings.STORAGE_RENOVATED_IMAGE_PREFIX.strip('/')}/{uuid.uuid4().hex}.{extension}"
    return await asyncio.to_thread(
        _upload_bytes_to_bucket,
        image_bytes,
        key=key,
        content_type=media_type or "image/png",
    )
