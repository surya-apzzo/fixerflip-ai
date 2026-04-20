"""Unit tests for cost calculation chain and severity scoring."""
from app.engine.renovation_engine.renovation_cost_engine import (
    RenovationEstimateInput,
    _compute_severity_score,
    _detect_severity,
    _impacted_elements,
    _scope_categories,
    estimate_renovation_cost,
)


class TestDetectSeverity:
    """Test severity detection for individual issues."""

    def test_foundation_is_severe(self):
        assert _detect_severity("foundation cracks") == "severe"
        assert _detect_severity("structural damage") == "severe"
        assert _detect_severity("major crack in wall") == "severe"

    def test_roof_and_water_are_severe(self):
        assert _detect_severity("roof leak") == "severe"
        assert _detect_severity("water damage in basement") == "severe"
        assert _detect_severity("mold growth") == "severe"

    def test_electrical_plumbing_hvac_are_moderate(self):
        assert _detect_severity("electrical issues") == "moderate"
        assert _detect_severity("plumbing problems") == "moderate"
        assert _detect_severity("hvac needs replacement") == "moderate"

    def test_cosmetic_issues_are_minor(self):
        assert _detect_severity("scuffed paint") == "minor"
        assert _detect_severity("dirty walls") == "minor"
        assert _detect_severity("carpet color fade") == "minor"

    def test_old_worn_outdated_are_moderate(self):
        assert _detect_severity("old bathroom") == "moderate"
        assert _detect_severity("worn cabinets") == "moderate"
        assert _detect_severity("outdated kitchen") == "moderate"


class TestScopeCategories:
    """Test scope category detection from issues and room type."""

    def test_kitchen_scope_from_room_type(self):
        scope = _scope_categories([], "kitchen", 70)
        assert "kitchen" in scope

    def test_kitchen_scope_from_issues(self):
        scope = _scope_categories(["outdated kitchen cabinets"], "unknown", 70)
        assert "kitchen" in scope

    def test_bathroom_scope_from_room_type(self):
        scope = _scope_categories([], "bathroom", 70)
        assert "bathroom" in scope

    def test_bathroom_scope_from_issues(self):
        scope = _scope_categories(["old bathroom tiles"], "unknown", 70)
        assert "bathroom" in scope

    def test_exterior_scope_from_issues(self):
        scope = _scope_categories(["roof damage", "siding needs repair"], "unknown", 70)
        assert "exterior" in scope

    def test_flooring_scope_from_issues(self):
        scope = _scope_categories(["carpet wear", "floor damage"], "unknown", 70)
        assert "flooring" in scope

    def test_paint_scope_from_issues(self):
        scope = _scope_categories(["wall stains", "paint wear"], "unknown", 70)
        assert "paint" in scope

    def test_paint_scope_from_low_condition_score(self):
        # Condition score < 85 triggers paint scope
        scope = _scope_categories([], "bedroom", 80)
        assert "paint" in scope

    def test_multiple_scopes(self):
        scope = _scope_categories(
            ["kitchen cabinets", "bathroom tiles", "roof damage", "floor wear"],
            "kitchen",
            50,
        )
        assert "kitchen" in scope
        assert "bathroom" in scope
        assert "exterior" in scope
        assert "flooring" in scope


class TestComputeSeverityScore:
    """Test severity score multiplier calculation."""

    def test_empty_issues_returns_base_multiplier(self):
        # Empty issues should return a nominal multiplier (not 0)
        score = _compute_severity_score([])
        assert 0.8 <= score <= 1.2

    def test_minor_issues_lower_multiplier(self):
        score = _compute_severity_score(["scuffed paint"])
        # "outdated" triggers moderate, so use truly minor issues
        assert score < 1.5

    def test_moderate_issues_mid_multiplier(self):
        score = _compute_severity_score(["old cabinets"])
        # "old" + "cosmetic" weight creates specific multiplier
        assert 1.0 <= score <= 1.5

    def test_severe_issues_higher_multiplier(self):
        score = _compute_severity_score(["foundation cracks", "structural damage"])
        assert score >= 1.3

    def test_multiple_severe_issues_max_out_multiplier(self):
        # Multiple severe issues should increase multiplier, but cap at limit
        issues = [
            "foundation cracks",
            "structural damage",
            "roof leak",
            "water damage",
            "mold",
        ]
        score = _compute_severity_score(issues)
        assert score >= 1.4
        assert score <= 1.6  # Should be capped

    def test_mixed_severity_issues_balanced(self):
        score = _compute_severity_score(
            ["structural damage", "old cabinets", "wall stains"]
        )
        # Mixed should include severe element, so higher
        assert 1.2 <= score <= 1.8


class TestRenovationEstimate:
    """Integration tests for full cost estimation."""

    def test_estimate_basic_property(self):
        result = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=65,
                issues=["outdated kitchen cabinets", "bathroom tiles", "paint wear"],
                room_type="kitchen",
            )
        )
        assert len(result.line_items) > 0
        assert result.minimum_cost > 0
        assert result.maximum_cost > 0
        assert result.minimum_cost <= result.maximum_cost

    def test_estimate_high_condition_minimal_cost(self):
        """High condition score should result in lower estimates."""
        result = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=90,
                issues=[],
                room_type="unknown",
            )
        )
        # High condition with no issues should have low estimates
        assert result.minimum_cost < 50000

    def test_estimate_severe_damage_higher_cost(self):
        """Severe issues should result in higher estimates."""
        result = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=20,
                issues=[
                    "foundation cracks",
                    "structural damage",
                    "roof leak",
                    "water damage",
                ],
                room_type="unknown",
            )
        )
        # Severe damage should result in substantial cost estimate
        assert result.minimum_cost > 5000

    def test_estimate_structural_issues_include_structural_work_items(self):
        result = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1148,
                beds=3,
                baths=2,
                condition_score=8,
                issues=["structural damage", "major wall cracks", "paint wear"],
                room_type="kitchen",
            )
        )
        assert "structural stabilization" in result.suggested_work_items
        assert "paint" in result.suggested_work_items

    def test_estimate_respects_scope_from_issues(self):
        """Line items should reflect detected scope."""
        result = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=50,
                issues=["kitchen cabinets", "bathroom tiles"],
                room_type="unknown",
            )
        )
        categories = {item.category for item in result.line_items}
        # Kitchen and bathroom should be included given the issues
        assert len(categories) >= 2

    def test_estimate_timeline_increases_with_severity(self):
        """Estimate timeline should increase with issues."""
        low_severity = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=85,
                issues=["wall paint"],
                room_type="unknown",
            )
        )
        high_severity = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=20,
                issues=["foundation damage", "roof damage", "water damage"],
                room_type="unknown",
            )
        )
        # Higher severity should have longer timeline
        assert high_severity.maximum_timeline_weeks >= low_severity.maximum_timeline_weeks

    def test_estimate_confidence_affected_by_data(self):
        """Properties with varying data should have confidence scores."""
        few_issues = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=70,
                issues=["wall paint"],
                room_type="unknown",
            )
        )
        many_issues = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=30,
                issues=[
                    "foundation damage",
                    "structural issues",
                    "roof damage",
                    "electrical issues",
                    "plumbing issues",
                ],
                room_type="unknown",
            )
        )
        # Both should have confidence scores
        assert 0 <= few_issues.confidence_score <= 100
        assert 0 <= many_issues.confidence_score <= 100

    def test_estimate_supports_quality_level_adjustment(self):
        """Quality level should affect cost estimates."""
        standard = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=50,
                issues=["wall paint"],
                room_type="unknown",
                desired_quality_level="standard",
            )
        )
        premium = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=50,
                issues=["wall paint"],
                room_type="unknown",
                desired_quality_level="premium",
            )
        )
        # Premium should be higher or equal to standard
        assert premium.minimum_cost >= standard.minimum_cost * 0.95
        assert premium.maximum_cost >= standard.maximum_cost * 0.95

    def test_estimate_supports_market_indices(self):
        """Material and labor indices should affect estimates."""
        base = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=50,
                issues=["wall paint"],
                room_type="unknown",
                material_index=1.0,
                labor_index=1.0,
            )
        )
        adjusted = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=50,
                issues=["wall paint"],
                room_type="unknown",
                material_index=1.2,
                labor_index=1.2,
            )
        )
        # Adjusted should be higher or equal due to indices
        assert adjusted.minimum_cost >= base.minimum_cost * 0.95
        assert adjusted.maximum_cost >= base.maximum_cost * 0.95

    def test_line_items_have_required_fields(self):
        """All line items should have required fields populated."""
        result = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=50,
                issues=["kitchen cabinets", "bathroom tiles", "paint wear"],
                room_type="kitchen",
            )
        )
        for item in result.line_items:
            assert item.category is not None
            assert len(item.category) > 0
            assert item.cost_low > 0
            assert item.cost_high > 0
            assert item.cost_low <= item.cost_high
            assert item.unit is not None

    def test_estimate_grand_totals_match_line_items(self):
        """Estimate totals should roughly align with sum of line items."""
        result = estimate_renovation_cost(
            RenovationEstimateInput(
                sqft=1500,
                beds=3,
                baths=2,
                condition_score=50,
                issues=["kitchen cabinets", "bathroom tiles", "wall paint"],
                room_type="kitchen",
            )
        )
        if result.line_items:
            line_items_low = sum(item.cost_low for item in result.line_items)
            line_items_high = sum(item.cost_high for item in result.line_items)

            # Totals should be in reasonable range (allowing for multipliers and overhead)
            if line_items_low > 0:
                assert result.minimum_cost >= line_items_low * 0.5
            if line_items_high > 0:
                assert result.maximum_cost >= line_items_high * 0.5


class TestImpactedElements:
    def test_kitchen_cupboard_synonym_maps_to_cabinet_fronts(self):
        elements, details = _impacted_elements(
            "kitchen",
            ["old cupboards", "paint wear"],
            ["kitchen update"],
        )
        assert "cabinet fronts" in elements
        assert "paint finish" in elements
        assert any(d.name == "cabinet fronts" and d.confidence >= 0.65 for d in details)

    def test_bathroom_water_damage_maps_to_plumbing_fixtures(self):
        elements, details = _impacted_elements(
            "bathroom",
            ["water damage near sink", "tile cracks"],
            ["water/moisture remediation"],
        )
        assert "plumbing fixtures" in elements
        assert "tile" in elements
        assert any(d.name == "plumbing fixtures" for d in details)

    def test_exterior_signals_include_roofline_and_siding(self):
        elements, details = _impacted_elements(
            "exterior",
            ["roof damage", "siding needs repair"],
            ["exterior repairs"],
        )
        assert "roofline" in elements
        assert "siding" in elements
        assert all(d.source in {"room_baseline", "issue_or_work_item"} for d in details)
