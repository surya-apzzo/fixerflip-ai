from __future__ import annotations

import app.api.v1.endpoints.condition_score as condition_endpoint


def test_condition_score_contract(client, monkeypatch):
    monkeypatch.setattr(
        condition_endpoint,
        "classify_and_filter",
        lambda _urls: {
            "selected": [
                {"image_url": "https://example.com/kitchen.jpg", "room_type": "kitchen interior", "weight": 0.35}
            ],
            "discarded_count": 2,
            "total_input": 3,
        },
    )

    async def fake_vision(_selected):
        return {
            "room_scores": [
                {
                    "room_type": "kitchen",
                    "score": 88.0,
                    "weight": 0.35,
                    "signals": ["original cabinets"],
                    "red_flags": [],
                }
            ],
            "cost_usd": 0.00021,
        }

    monkeypatch.setattr(condition_endpoint, "score_from_images", fake_vision)
    monkeypatch.setattr(
        condition_endpoint,
        "aggregate",
        lambda _vision: {
            "condition_score": 82.4,
            "grade": "Poor condition - high flip upside",
            "text_score": 0.0,
            "vision_score": 84.6,
            "room_scores": [
                {
                    "room_type": "kitchen",
                    "score": 88.0,
                    "weight": 0.35,
                    "signals": ["original cabinets"],
                }
            ],
            "positive_signals": ["original cabinets"],
            "caution_signals": [],
            "red_flags": [],
            "cost_usd": 0.00021,
        },
    )

    response = client.post(
        "/api/v1/condition-score",
        json={
            "property_id": "prop_001",
            "image_urls": [
                "https://example.com/1.jpg",
                "https://example.com/2.jpg",
                "https://example.com/3.jpg",
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["property_id"] == "prop_001"
    assert body["condition_score"] == 82.4
    assert body["grade"] == "Poor condition - high flip upside"
    assert body["images_analyzed"] == 1
    assert body["images_discarded"] == 2
    assert body["cost_usd"] == 0.00021

