from __future__ import annotations


def test_unknown_api_route_returns_url_not_found_message(client):
    response = client.get("/api/v1/this-endpoint-does-not-exist")
    assert response.status_code == 404
    detail = response.json().get("detail", {})
    assert detail.get("code") == "NOT_FOUND"
    assert {"field": "url", "message": "URL is not found."} in detail.get("errors", [])

