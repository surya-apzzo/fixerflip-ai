from app.engine.renovation_engine.image_condition_engine import ImageConditionEngine
from app.engine.renovation_engine.schemas import ImageEditResult
from app.engine.renovation_engine.renovation_cost_engine import (
    RenovationEstimateInput,
    estimate_renovation_cost,
)


def test_engine_scores_from_issues():
    engine = ImageConditionEngine()
    r = engine.score_from_issues(
        ["outdated kitchen cabinets", "wall stains", "old carpet"],
        room_type="kitchen",
    )
    assert r.room_type == "kitchen"
    assert len(r.issues) == 3
    assert 0 <= r.condition_score <= 100


def test_estimate_line_items_follow_scope_from_issues_only():
    est = estimate_renovation_cost(
        RenovationEstimateInput(
            sqft=1148,
            beds=3,
            baths=2,
            condition_score=11,
            issues=["major wall cracks", "floor damage", "paint wear"],
            room_type="living_room",
        )
    )
    assert {x.category for x in est.line_items} <= {"Paint", "Flooring"}


def test_estimate_adds_system_scope_for_water_damage():
    est = estimate_renovation_cost(
        RenovationEstimateInput(
            sqft=1800,
            beds=3,
            baths=2,
            condition_score=8,
            issues=["water damage", "floor damage", "paint wear", "minor wall damage"],
            room_type="living_room",
        )
    )
    categories = {x.category for x in est.line_items}
    assert "Plumbing" in categories
    assert "Paint" in categories
    assert "Flooring" in categories
    plumbing_line = next(x for x in est.line_items if x.category == "Plumbing")
    assert plumbing_line.quantity == 1800


def test_renovation_estimate_endpoint_with_manual_fallback(client):
    resp = client.post(
        "/api/v1/renovation/estimate",
        json={
            "sqft": 1600,
            "beds": 3,
            "baths": 2,
            "desired_quality_level": "standard",
            "days_on_market": 12,
            "zip_code": "94103",
            "condition_score": 62,
            "issues": ["old tiles", "paint wear", "roof damage"],
            "room_type": "kitchen,exterior",
            "labor_index": 1.1,
            "material_index": 1.05,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "image_condition" in data
    assert "estimate" in data
    assert data["estimate"]["renovation_class"] in {"Cosmetic", "Moderate", "Heavy", "Full Gut"}
    assert data["estimate"]["minimum_cost"] > 0
    assert data["estimate"]["maximum_cost"] >= data["estimate"]["minimum_cost"]
    assert data["estimate"]["confidence_label"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0 <= data["estimate"]["confidence_score"] <= 100
    assert len(data["estimate"]["suggested_work_items"]) > 0
    assert isinstance(data["estimate"]["explanation_summary"], str)
    assert len(data["estimate"]["line_items"]) > 0
    assert data["renovated_image"] is None
    assert data["image_condition"]["analysis_status"] == "manual_input"
    assert data["image_condition"]["fallback_reason"] == "manual_condition_score"


def test_renovation_estimate_returns_renovated_image_when_user_inputs_present(client, monkeypatch):
    async def _fake_edit_property_image_from_url(*, image_url: str, instruction: str, preserve_elements: str = "") -> ImageEditResult:
        assert image_url == "https://example.com/property/living.jpg"
        assert "Change wall color to warm white" in instruction
        return ImageEditResult(
            revised_prompt="Changed wall color to warm white",
            image_base64="ZmFrZV9lZGl0ZWQ=",
            media_type="image/png",
        )

    monkeypatch.setattr(
        "app.api.v1.endpoints.renovation.edit_property_image_from_url",
        _fake_edit_property_image_from_url,
    )

    resp = client.post(
        "/api/v1/renovation/estimate",
        json={
            "image_url": "https://example.com/property/living.jpg",
            "sqft": 1200,
            "beds": 3,
            "baths": 2,
            "user_inputs": "Change wall color to warm white",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["renovated_image"] is not None
    assert data["renovated_image"]["renovated_image_url"].startswith("data:image/png;base64,")
    assert data["renovated_image"]["renovated_image_url"].endswith("ZmFrZV9lZGl0ZWQ=")
    assert "revised_prompt" not in data["renovated_image"]


def test_renovation_estimate_uses_default_image_instruction_when_blank(client, monkeypatch):
    captured = {}

    async def _fake_edit_property_image_from_url(*, image_url: str, instruction: str, preserve_elements: str = "") -> ImageEditResult:
        captured["instruction"] = instruction
        assert image_url == "https://example.com/property/living.jpg"
        return ImageEditResult(
            revised_prompt="Default renovation applied",
            image_base64="ZmFrZV9lZGl0ZWQ=",
            media_type="image/png",
        )

    monkeypatch.setattr(
        "app.api.v1.endpoints.renovation.edit_property_image_from_url",
        _fake_edit_property_image_from_url,
    )

    resp = client.post(
        "/api/v1/renovation/estimate",
        json={
            "image_url": "https://example.com/property/living.jpg",
            "sqft": 1200,
            "beds": 3,
            "baths": 2,
            "user_inputs": "",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["renovated_image"] is not None
    assert data["renovated_image"]["renovated_image_url"].endswith("ZmFrZV9lZGl0ZWQ=")
    assert captured["instruction"]



def test_renovation_estimate_user_inputs_adjust_cost_range(client):
    base_payload = {
        "sqft": 1200,
        "beds": 3,
        "baths": 2,
        "condition_score": 70,
    }
    base_resp = client.post("/api/v1/renovation/estimate", json=base_payload)
    assert base_resp.status_code == 200
    base_data = base_resp.json()

    adjusted_payload = {
        **base_payload,
        "user_inputs": "Change wall color to warm white and replace windows",
    }
    adjusted_resp = client.post("/api/v1/renovation/estimate", json=adjusted_payload)
    assert adjusted_resp.status_code == 200
    adjusted_data = adjusted_resp.json()

    assert adjusted_data["estimate"]["minimum_cost"] > base_data["estimate"]["minimum_cost"]
    assert adjusted_data["estimate"]["maximum_cost"] > base_data["estimate"]["maximum_cost"]
    assert any(
        "User-input scope adjustment" in x for x in adjusted_data["estimate"]["assumptions"]
    )


def test_renovation_estimate_user_inputs_negation_does_not_adjust(client):
    base_payload = {
        "sqft": 1200,
        "beds": 3,
        "baths": 2,
        "condition_score": 70,
    }
    base_resp = client.post("/api/v1/renovation/estimate", json=base_payload)
    assert base_resp.status_code == 200
    base_data = base_resp.json()

    negated_payload = {
        **base_payload,
        "user_inputs": "Do not replace windows or roof",
    }
    negated_resp = client.post("/api/v1/renovation/estimate", json=negated_payload)
    assert negated_resp.status_code == 200
    negated_data = negated_resp.json()

    assert negated_data["estimate"]["minimum_cost"] == base_data["estimate"]["minimum_cost"]
    assert negated_data["estimate"]["maximum_cost"] == base_data["estimate"]["maximum_cost"]


def test_renovation_estimate_user_input_adjustment_has_cap(client):
    payload = {
        "sqft": 800,
        "beds": 2,
        "baths": 1,
        "condition_score": 80,
        "user_inputs": (
            "kitchen remodel bathroom remodel flooring roof replacement "
            "foundation repair electrical upgrade plumbing repair hvac replacement "
            "window replacement exterior remodel door replacement garage upgrade landscaping"
        ),
    }
    resp = client.post("/api/v1/renovation/estimate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    cap_note = [
        x for x in data["estimate"]["assumptions"]
        if "capped relative to the base estimate" in x
    ]
    assert cap_note
