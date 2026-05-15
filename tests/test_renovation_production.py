"""Production-path tests for renovation validation, costing, and API response shape."""

from __future__ import annotations

from app.engine.renovation_engine.renovation_cost_engine import (
    RenovationEstimateInput,
    _build_cost_line_items,
    _deduplicate_work_items,
)
from app.schemas import ImageConditionResult
from app.schemas.requests.renovation import RenovationEstimateRequest
from app.schemas.responses.estimate import RenovationEstimate
from app.services.renovation_payload_validator import (
    _canonical_renovation_element,
    validate_and_normalize_renovation_payload,
)
from app.services.renovation_response_mapper import build_renovation_estimate_response


def _base_request(**overrides) -> RenovationEstimateRequest:
    data = {
        "sqft": 1500.0,
        "condition_score": 70,
        "room_type": "living",
        "issues": [],
    }
    data.update(overrides)
    return RenovationEstimateRequest(**data)


def test_image_condition_result_preserves_analysis_metadata():
    result = ImageConditionResult(
        condition_score=72,
        issues=["stains"],
        room_type="kitchen",
        analysis_status="fallback",
        fallback_reason="vision_unavailable",
        model_used=None,
    )
    dumped = result.model_dump()
    assert dumped["analysis_status"] == "fallback"
    assert dumped["fallback_reason"] == "vision_unavailable"


def test_exterior_ceiling_not_remapped_to_roof():
    assert _canonical_renovation_element("ceiling", renovation_type="exterior") == "ceiling"


def test_interior_roof_maps_to_ceiling():
    assert _canonical_renovation_element("roof", renovation_type="interior") == "ceiling"


def test_validate_normalizes_exterior_paint_element():
    payload = _base_request(
        type_of_renovation="EXTERIOR",
        renovation_elements=["Paint", "Siding"],
    )
    normalized = validate_and_normalize_renovation_payload(payload)
    assert normalized.renovation_elements == ["paint", "siding"]


def test_deduplicate_work_items_prefers_renovation_suffix():
    items = _deduplicate_work_items(
        ["paint", "paint renovation", "kitchen renovation", "kitchen"]
    )
    assert items == ["paint renovation", "kitchen renovation"]


def test_selected_elements_cost_scope_ignores_unrelated_issues():
    data = RenovationEstimateInput(
        sqft=1800,
        condition_score=45,
        issues=["outdated cabinets", "old bathroom", "roof damage"],
        room_type="kitchen",
        renovation_elements=["paint"],
    )
    line_items = _build_cost_line_items(data, factor=1.0)
    categories = {item.category.lower() for item in line_items}
    assert "paint" in categories
    assert "kitchen" not in categories
    assert "bathroom" not in categories
    # Critical safety scope may still add roof/foundation from severe vision issues.
    assert "roof" in categories or "foundation" in categories


def test_response_mapper_builds_estimate_response():
    estimate = RenovationEstimate(
        renovation_class="Cosmetic",
        minimum_cost=10_000,
        maximum_cost=20_000,
        minimum_timeline_weeks=2,
        maximum_timeline_weeks=4,
        confidence_label="MEDIUM",
        confidence_score=75,
        suggested_work_items=["paint renovation"],
        explanation_summary="Test summary.",
        line_items=[],
        assumptions=[],
    )
    response = build_renovation_estimate_response(
        estimate,
        room_type="kitchen",
        condition_score=80,
        renovated_image_url="https://cdn.example.com/edited.png",
    )
    assert response.renovation_class == "Cosmetic"
    assert response.estimated_renovation_range == "$10,000 - $20,000"
    assert response.renovated_image_url.endswith("edited.png")
    assert response.room_type == "kitchen"
    assert response.condition_score == 80
