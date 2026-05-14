from fastapi import APIRouter, HTTPException, Query, status

from app.services.storage_service import presigned_get_url_for_renovated_object_key

router = APIRouter(prefix="/bucket")


@router.get("/file")
async def bucket_renovated_file(
    key: str = Query(..., description="S3 object key under STORAGE_RENOVATED_IMAGE_PREFIX (e.g. renovated/<uuid>.png)."),
) -> dict[str, str]:
    """
    Return a fresh presigned GET URL for a renovated preview object.
    Same idea as a separate Node `GET /bucket/file?key=` — URLs expire; call again to refresh.
    """
    try:
        url = presigned_get_url_for_renovated_object_key(key)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage is not available or misconfigured.",
        ) from None
    return {"url": url}
