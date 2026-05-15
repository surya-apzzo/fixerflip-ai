from __future__ import annotations

import logging
import time

from app.core.config import settings
from app.schemas import ImageConditionResult
from app.engine.renovation_engine.image_edit_engine import (
    build_instruction_for_edit,
    edit_property_image_from_url,
)
from app.engine.renovation_engine.renovation_cost_engine import (
    RenovationEstimateInput,
    apply_user_input_cost_adjustments,
    build_renovation_sqft_context,
    estimate_renovation_cost,
    infer_user_scope_categories,
)
from app.engine.renovation_engine.vision_analysis import analyze_renovation_image_url
from app.schemas.requests.renovation import RenovationEstimateRequest
from app.schemas.responses.renovation import RenovationEstimateResponse
from app.services.renovation_payload_validator import validate_and_normalize_renovation_payload
from app.services.renovation_response_mapper import build_renovation_estimate_response
from app.services.storage_service import upload_base64_image_to_bucket

logger = logging.getLogger(__name__)

_PUBLIC_IMAGE_EDIT_ERROR = "Image edit failed. Retry later or inspect server logs."


def _cost_adjustment_factor(*, labor_index: float, material_index: float) -> float:
    return max((labor_index * 0.6) + (material_index * 0.4), 0.5)


def _has_style_upgrade_request(payload: RenovationEstimateRequest) -> bool:
    if payload.renovation_elements:
        return True
    text = (payload.user_inputs or "").strip().lower()
    if not text:
        return False
    style_markers = (
        "color",
        "colour",
        "tile",
        "tiles",
        "paint",
        "mosaic",
        "backsplash",
        "counter",
        "countertop",
        "cabinet",
        "kitchen",
        "bathroom",
        "modern",
        "style",
        "theme",
        "luxury",
        "premium",
        "renovate",
        "remodel",
        "rehab",
        "modernize",
        "modernise",
        "upgrade",
        "refresh",
        "gut",
        "redo",
        "overhaul",
    )
    return any(marker in text for marker in style_markers)


def _resolve_visual_scope_hint(payload: RenovationEstimateRequest) -> str:
    """Derived edit-scope tag (reference vs elements). Quality/finish is only desired_quality_level."""
    if (payload.reference_image_url or "").strip():
        return "reference_style"
    if payload.visual_type == "choose_an_existing_style":
        return "existing_style"
    if payload.visual_type == "select_elements_to_renovate" and payload.renovation_elements:
        if len(payload.renovation_elements) == 1:
            return payload.renovation_elements[0].replace(" ", "_").lower()
        return "targeted_elements"
    return "investor_standard"


def _build_fallback_image_condition() -> ImageConditionResult:
    return ImageConditionResult(
        condition_score=65,
        issues=[],
        room_type="unknown",
        analysis_status="fallback",
        fallback_reason="vision_request_failed",
    )


async def _resolve_image_condition(
    payload: RenovationEstimateRequest,
    pipeline_warnings: list[str],
) -> ImageConditionResult:
    try:
        return await analyze_renovation_image_url(payload.image_url)
    except Exception as vision_exc:
        logger.warning("Renovation vision analysis failed: %s", vision_exc)
        pipeline_warnings.append("Vision analysis failed; estimate used fallback condition inputs.")
        return _build_fallback_image_condition()


def _resolve_edit_scope(
    *,
    payload: RenovationEstimateRequest,
    user_scope_categories: list[str],
    has_detected_issues: bool,
) -> tuple[str, list[str]]:
    effective_elements = payload.renovation_elements or user_scope_categories
    effective_visual_type = payload.visual_type
    if effective_elements and not payload.renovation_elements and not has_detected_issues:
        effective_visual_type = "select_elements_to_renovate"
    return effective_visual_type, effective_elements


async def _upload_renovated_image(
    *,
    image_base64: str,
    media_type: str,
    pipeline_warnings: list[str],
) -> str | None:
    try:
        return await upload_base64_image_to_bucket(
            image_base64=image_base64,
            media_type=media_type,
        )
    except Exception as upload_exc:
        logger.warning("Renovated image upload failed: %s", upload_exc)
        pipeline_warnings.append("Renovated image upload failed; estimate data remains available.")
        return None


async def _enforce_repair_only_guardrail(
    *,
    payload: RenovationEstimateRequest,
    image_condition: ImageConditionResult,
    visual_scope_hint: str,
    candidate_url: str | None,
    has_detected_issues: bool,
    pipeline_warnings: list[str],
) -> str | None:
    if not settings.RENOVATION_IMAGE_STRICT_GUARDRAIL:
        return candidate_url

    if not candidate_url or not has_detected_issues or _has_style_upgrade_request(payload):
        return candidate_url

    try:
        edited_condition = await analyze_renovation_image_url(candidate_url)
    except Exception as edited_analysis_exc:
        logger.warning("Edited image validation failed: %s", edited_analysis_exc)
        return candidate_url

    if edited_condition.condition_score < 90 or edited_condition.issues:
        return candidate_url

    strict_retry_instruction = build_instruction_for_edit(
        user_inputs=(
            "Repair only visible damage and issues in place. "
            "Do not add new furniture, kitchen sets, decor, fixtures, or redesign."
        ),
        type_of_renovation=payload.type_of_renovation,
        visual_type=payload.visual_type,
        desired_quality_level=payload.desired_quality_level,
        visual_scope_hint=visual_scope_hint,
        reference_image_url=payload.reference_image_url,
        renovation_elements=[],
        detected_issues=image_condition.issues,
    )
    try:
        retry_edit_result = await edit_property_image_from_url(
            image_url=payload.image_url,
            instruction=strict_retry_instruction,
            reference_image_url="",
        )
        retry_url = await upload_base64_image_to_bucket(
            image_base64=retry_edit_result.image_base64,
            media_type=retry_edit_result.media_type,
        )
        retry_condition = await analyze_renovation_image_url(retry_url)
    except Exception as retry_exc:
        logger.warning("Strict repair-only retry failed: %s", retry_exc)
        pipeline_warnings.append(
            "Strict repair-only retry failed; first preview image kept for display."
        )
        return candidate_url

    if retry_condition.condition_score >= 90 and not retry_condition.issues:
        pipeline_warnings.append(
            "Strict repair-only retry still looked fully restored; first preview image kept for display."
        )
        return candidate_url

    pipeline_warnings.append("Applied strict repair-only retry to avoid over-restoration.")
    return retry_url


async def _generate_renovated_image_url(
    *,
    payload: RenovationEstimateRequest,
    image_condition: ImageConditionResult,
    visual_scope_hint: str,
    user_scope_categories: list[str],
    pipeline_warnings: list[str],
) -> str | None:
    explicit_user_request = bool((payload.user_inputs or "").strip()) or bool(payload.renovation_elements)
    has_detected_issues = bool(image_condition.issues)
    if not (explicit_user_request or has_detected_issues):
        return payload.image_url

    effective_visual_type, effective_elements = _resolve_edit_scope(
        payload=payload,
        user_scope_categories=user_scope_categories,
        has_detected_issues=has_detected_issues,
    )
    if (payload.reference_image_url or "").strip():
        source = (payload.image_url or "").lower()
        if "renovated-images" in source or "/renovated/" in source:
            pipeline_warnings.append(
                "image_url appears to be a prior renovation preview; use the original listing photo "
                "as image_url for best reference-style results."
            )
    instruction_for_edit = build_instruction_for_edit(
        user_inputs=payload.user_inputs,
        type_of_renovation=payload.type_of_renovation,
        visual_type=effective_visual_type,
        desired_quality_level=payload.desired_quality_level,
        visual_scope_hint=visual_scope_hint,
        reference_image_url=payload.reference_image_url,
        renovation_elements=effective_elements,
        detected_issues=image_condition.issues,
    )
    try:
        edit_result = await edit_property_image_from_url(
            image_url=payload.image_url,
            instruction=instruction_for_edit,
            reference_image_url=payload.reference_image_url,
        )
    except Exception as edit_exc:
        logger.warning("Renovation image edit failed: %s", edit_exc)
        error_message = str(edit_exc) if isinstance(edit_exc, ValueError) else _PUBLIC_IMAGE_EDIT_ERROR
        pipeline_warnings.append(error_message)
        # Return source image URL when edit fails so clients always receive a usable image link.
        return payload.image_url

    uploaded_url = await _upload_renovated_image(
        image_base64=edit_result.image_base64,
        media_type=edit_result.media_type,
        pipeline_warnings=pipeline_warnings,
    )
    if uploaded_url is None:
        logger.error(
            "Renovation image edit succeeded but storage upload returned no URL. "
            "Configure STORAGE_* (or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION) on this environment."
        )
    return await _enforce_repair_only_guardrail(
        payload=payload,
        image_condition=image_condition,
        visual_scope_hint=visual_scope_hint,
        candidate_url=uploaded_url,
        has_detected_issues=has_detected_issues,
        pipeline_warnings=pipeline_warnings,
    )


def _build_renovation_estimate_input(
    payload: RenovationEstimateRequest,
    *,
    image_condition: ImageConditionResult,
) -> RenovationEstimateInput:
    return RenovationEstimateInput(
        sqft=payload.sqft,
        zip_code=payload.zip_code,
        year_built=payload.year_built,
        listing_price=payload.listing_price,
        listing_description=payload.listing_description,
        days_on_market=payload.days_on_market,
        avg_area_price_per_sqft=payload.avg_area_price_per_sqft,
        years_since_last_sale=payload.years_since_last_sale,
        permit_years_since_last=payload.permit_years_since_last,
        condition_score=image_condition.condition_score,
        issues=image_condition.issues,
        room_type=image_condition.room_type,
        desired_quality_level=payload.desired_quality_level,
        labor_index=payload.labor_index,
        material_index=payload.material_index,
        renovation_elements=payload.renovation_elements,
        user_inputs=payload.user_inputs,
    )


async def build_renovation_estimate(payload: RenovationEstimateRequest) -> RenovationEstimateResponse:
    started = time.perf_counter()
    logger.info(
        "Renovation estimate request start has_image=%s zip=%s sqft=%s elements=%s",
        bool((payload.image_url or "").strip()),
        (payload.zip_code or "").strip() or None,
        payload.sqft,
        len(payload.renovation_elements or []),
    )

    t0 = time.perf_counter()
    payload = validate_and_normalize_renovation_payload(payload)
    logger.debug("Renovation payload validated in %.1fms", (time.perf_counter() - t0) * 1000)

    pipeline_warnings: list[str] = []
    renovated_image_url: str | None = None
    user_scope_categories = infer_user_scope_categories(
        payload.user_inputs,
        payload.renovation_elements,
    )

    if payload.image_url:
        visual_scope_hint = _resolve_visual_scope_hint(payload)
        t2 = time.perf_counter()
        image_condition = await _resolve_image_condition(payload, pipeline_warnings)
        logger.debug(
            "Renovation vision resolved in %.1fms status=%s issues=%s room=%s",
            (time.perf_counter() - t2) * 1000,
            image_condition.analysis_status,
            len(image_condition.issues or []),
            image_condition.room_type,
        )

        t3 = time.perf_counter()
        renovated_image_url = await _generate_renovated_image_url(
            payload=payload,
            image_condition=image_condition,
            visual_scope_hint=visual_scope_hint,
            user_scope_categories=user_scope_categories,
            pipeline_warnings=pipeline_warnings,
        )
        logger.debug(
            "Renovation image edit/upload pipeline completed in %.1fms returned_url=%s",
            (time.perf_counter() - t3) * 1000,
            bool(renovated_image_url),
        )
    else:
        manual_condition_score = payload.condition_score
        if manual_condition_score is None:
            raise RuntimeError("Validated renovation payload is missing condition_score.")
        image_condition = ImageConditionResult(
            condition_score=manual_condition_score,
            issues=payload.issues,
            room_type=payload.room_type,
            analysis_status="manual_input",
            fallback_reason="manual_condition_score",
        )

    t4 = time.perf_counter()
    estimate_input = _build_renovation_estimate_input(
        payload,
        image_condition=image_condition,
    )
    sqft_ctx = build_renovation_sqft_context(estimate_input.sqft, estimate_input.room_type)
    estimate = estimate_renovation_cost(estimate_input, sqft_context=sqft_ctx)
    cost_multiplier = _cost_adjustment_factor(
        labor_index=estimate_input.labor_index,
        material_index=estimate_input.material_index,
    )
    logger.debug("Renovation base estimate computed in %.1fms", (time.perf_counter() - t4) * 1000)

    if payload.renovation_elements:
        estimate = estimate.model_copy(
            update={
                "assumptions": [
                    *estimate.assumptions,
                    "Estimate baseline uses user-selected renovation elements; explicit user instructions are priced as additive scope when applicable.",
                ]
            }
        )
    elif user_scope_categories:
        estimate = estimate.model_copy(
            update={
                "assumptions": [
                    *estimate.assumptions,
                    "Estimate scope follows explicit user-requested renovation intent.",
                ]
            }
        )

    estimate = apply_user_input_cost_adjustments(
        estimate,
        payload.user_inputs,
        sqft_ctx.effective_sqft,
        location_factor=cost_multiplier,
        renovation_elements=payload.renovation_elements,
    )

    analysis_status = image_condition.analysis_status
    fallback_reason = image_condition.fallback_reason
    if analysis_status != "ai_success":
        fallback_note = f"Condition source: {analysis_status}"
        if fallback_reason:
            fallback_note += f" ({fallback_reason})."
        else:
            fallback_note += "."
        estimate = estimate.model_copy(update={"assumptions": [*estimate.assumptions, fallback_note]})

    if pipeline_warnings:
        estimate = estimate.model_copy(update={"assumptions": [*estimate.assumptions, *pipeline_warnings]})

    response = build_renovation_estimate_response(
        estimate,
        room_type=estimate_input.room_type,
        condition_score=int(image_condition.condition_score),
        renovated_image_url=renovated_image_url,
    )
    logger.info(
        "Renovation estimate request end class=%s range=%s warnings=%s duration_ms=%.1f",
        getattr(response, "renovation_class", None),
        getattr(response, "estimated_renovation_range", None),
        len(pipeline_warnings),
        (time.perf_counter() - started) * 1000,
    )
    return response
