"""Shared pytest fixtures for API integration tests."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Sync HTTP client against the FastAPI app (runs ASGI in-process)."""
    with TestClient(app) as test_client:
        yield test_client
