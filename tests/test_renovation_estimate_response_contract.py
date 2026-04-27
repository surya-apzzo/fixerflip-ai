from __future__ import annotations

from types import SimpleNamespace

import app.services.renovation_service as renovation_service
from app.schemas.responses.condition import ImageConditionResult
from app.schemas.responses.estimate import RenovationEstimate
from app.schemas.responses.line_item import RenovationLineItem


def _build_estimate_fixture() -> RenovationEstimate:
    return RenovationEstimate(
        renovation_class="Moderate",
        minimum_cost=45000,
        maximum_cost=72000,
        minimum_timeline_weeks=6,
        maximum_timeline_weeks=10,
        confidence_label="HIGH",
        confidence_score=82,
        suggested_work_items=["paint", "flooring", "kitchen update", "bathroom update"],
        impacted_elements=[],
        impacted_element_details=[],
        explanation_summary=(
            "Property appears outdated with older kitchen finishes, worn flooring, and original bathrooms. "
            "Recommended as moderate rehab."
        ),
        line_items=[
            RenovationLineItem(
                category="Paint",
                quantity=1000,
                unit="sqft",
                unit_cost_low=2.0,
                unit_cost_high=4.0,
                cost_low=2000,
                cost_high=4000,
            )
        ],
        assumptions=["Noted from condition: fire damage."],
    )


def test_estimate_response_contract_shape(client, monkeypatch):
    async def fake_analyze(_image_url: str) -> ImageConditionResult:
        return ImageConditionResult(
            condition_score=45,
            issues=["fire damage"],
            room_type="kitchen",
            analysis_status="ai_success",
        )

    async def fake_edit_property_image_from_url(*, image_url: str, instruction: str):
        return SimpleNamespace(image_base64="ZmFrZQ==", media_type="image/png")

    async def fake_upload_base64_image_to_bucket(*, image_base64: str, media_type: str) -> str:
        return "https://example.com/renovated.png"

    monkeypatch.setattr(renovation_service, "analyze_renovation_image_url", fake_analyze)
    monkeypatch.setattr(renovation_service, "edit_property_image_from_url", fake_edit_property_image_from_url)
    monkeypatch.setattr(renovation_service, "upload_base64_image_to_bucket", fake_upload_base64_image_to_bucket)
    monkeypatch.setattr(renovation_service, "estimate_renovation_cost", lambda _data: _build_estimate_fixture())
    monkeypatch.setattr(renovation_service, "apply_user_input_cost_adjustments", lambda estimate, *_args, **_kwargs: estimate)

    response = client.post(
        "/api/v1/renovation/estimate",
        json={
            "image_url": "https://example.com/source.png",
            "sqft": 1200,
            "beds": 3,
            "baths": 2,
            "renovation_elements": ["paint", "flooring"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "renovation_class",
        "estimated_renovation_range",
        "estimated_timeline",
        "suggested_work_items",
        "confidence_score",
        "explanation_summary",
    }
    assert body["renovation_class"] == "Moderate"
    assert body["estimated_renovation_range"] == "$45,000 - $72,000"
    assert body["estimated_timeline"] == "6-10 weeks"
    assert body["confidence_score"] == "82%"
    assert "possible systems review" in body["suggested_work_items"]


def test_estimate_validation_error_shape(client):
    response = client.post(
        "/api/v1/renovation/estimate",
        json={
            "sqft": 0,
            "beds": -1,
            "baths": 1,
            "condition_score": 120,
        },
    )

    assert response.status_code == 422
    detail = response.json().get("detail", {})
    assert detail.get("code") == "VALIDATION_ERROR"
    assert isinstance(detail.get("errors"), list)
    fields = {item.get("field") for item in detail["errors"]}
    assert {"sqft", "beds", "condition_score"}.issubset(fields)
