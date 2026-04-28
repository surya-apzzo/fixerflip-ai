from __future__ import annotations

import asyncio
import base64
import logging
import uuid

from botocore.config import Config

from app.core.config import settings

logger = logging.getLogger(__name__)


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
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=settings.STORAGE_ENDPOINT_URL,
        region_name=settings.STORAGE_REGION or "auto",
        aws_access_key_id=settings.STORAGE_ACCESS_KEY_ID,
        aws_secret_access_key=settings.STORAGE_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    put_kwargs = {
        "Bucket": settings.STORAGE_BUCKET_NAME,
        "Key": key,
        "Body": image_bytes,
        "ContentType": content_type,
    }
    try:
        client.put_object(**put_kwargs)
    except Exception:
        # Keep compatibility for providers that still require ACL.
        client.put_object(**put_kwargs, ACL="public-read")

    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.STORAGE_BUCKET_NAME, "Key": key},
            ExpiresIn=settings.STORAGE_PRESIGNED_URL_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("Failed to generate presigned URL for key '%s': %s", key, exc)
        # Only return direct URL when explicitly configured as public.
        if settings.STORAGE_PUBLIC_BASE_URL:
            return _build_public_url(key)
        raise ValueError(
            "Uploaded image but failed to generate presigned download URL. "
            "Set STORAGE_PUBLIC_BASE_URL for public objects or fix signing config."
        ) from exc


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
