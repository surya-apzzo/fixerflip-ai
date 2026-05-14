"""Service layer. Prefer importing from concrete modules (e.g. ``renovation_service``) to avoid import cycles."""

__all__ = ["build_renovation_estimate"]


def __getattr__(name: str):
    if name == "build_renovation_estimate":
        from app.services.renovation_service import build_renovation_estimate

        return build_renovation_estimate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
