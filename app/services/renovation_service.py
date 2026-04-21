from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException

from app.engine.renovation_engine.image_condition_engine import ImageConditionResult
from app.engine.renovation_engine.image_edit_engine import build_instruction_for_edit, edit_property_image_from_url
from app.engine.renovation_engine.renovation_cost_engine import (
    RenovationEstimateInput,
    apply_user_input_cost_adjustments,
    estimate_renovation_cost,
)
from app.engine.renovation_engine.vision_analysis import analyze_renovation_image_url
from app.schemas.renovation_api import RenovatedImageResult, RenovationEstimateRequest, RenovationEstimateResponse

logger = logging.getLogger(__name__)

_PUBLIC_IMAGE_EDIT_ERROR = "Image edit failed. Retry later or inspect server logs."


def _resolve_target_renovation_style(payload: RenovationEstimateRequest) -> str:
    if payload.target_renovation_style and payload.target_renovation_style != "investor_standard":
        return payload.target_renovation_style
    if payload.visual_type == "upload_my_own_reference_photo" and payload.reference_image_url:
        return "reference_style"
    if payload.visual_type == "choose_an_existing_style":
        return "existing_style"
    if payload.visual_type == "select_elements_to_renovate" and payload.renovation_elements:
        if len(payload.renovation_elements) == 1:
            return payload.renovation_elements[0].replace(" ", "_").lower()
        return "targeted_elements"
    return payload.target_renovation_style


def _build_estimate_input(
    payload: RenovationEstimateRequest,
    *,
    image_condition: ImageConditionResult,
    resolved_target_style: str,
) -> RenovationEstimateInput:
    return RenovationEstimateInput(
        sqft=payload.sqft,
        beds=payload.beds,
        baths=payload.baths,
        address=payload.address,
        city=payload.city,
        zip_code=payload.zip_code,
        property_type=payload.property_type,
        year_built=payload.year_built,
        lot_size=payload.lot_size,
        listing_price=payload.listing_price,
        listing_description=payload.listing_description,
        listing_status=payload.listing_status,
        days_on_market=payload.days_on_market,
        avg_area_price_per_sqft=payload.avg_area_price_per_sqft,
        years_since_last_sale=payload.years_since_last_sale,
        permit_years_since_last=payload.permit_years_since_last,
        condition_score=image_condition.condition_score,
        issues=image_condition.issues,
        room_type=image_condition.room_type,
        target_renovation_style=resolved_target_style,
        desired_quality_level=payload.desired_quality_level,
        labor_index=payload.labor_index,
        material_index=payload.material_index,
        renovation_elements=payload.renovation_elements,
        user_inputs=payload.user_inputs,
    )


async def build_renovation_estimate(payload: RenovationEstimateRequest) -> RenovationEstimateResponse:
    renovated_image: RenovatedImageResult | None = None
    renovated_image_error: str | None = None

    if payload.image_url:
        resolved_target_style = _resolve_target_renovation_style(payload)
        instruction_for_edit = build_instruction_for_edit(
            user_inputs=payload.user_inputs,
            type_of_renovation=payload.type_of_renovation,
            visual_type=payload.visual_type,
            desired_quality_level=payload.desired_quality_level,
            resolved_target_style=resolved_target_style,
            reference_image_url=payload.reference_image_url,
            renovation_elements=payload.renovation_elements,
        )
        vision_result, edit_result = await asyncio.gather(
            analyze_renovation_image_url(payload.image_url),
            edit_property_image_from_url(
                image_url=payload.image_url,
                instruction=instruction_for_edit,
            ),
            return_exceptions=True,
        )

        if isinstance(vision_result, Exception):
            logger.warning("Renovation vision analysis failed: %s", vision_result)
            image_condition = ImageConditionResult(
                condition_score=65,
                issues=[],
                room_type="unknown",
                analysis_status="fallback",
                fallback_reason="vision_request_failed",
            )
        else:
            image_condition = vision_result

        if isinstance(edit_result, Exception):
            logger.warning("Renovation image edit failed: %s", edit_result)
            renovated_image_error = str(edit_result) if isinstance(edit_result, ValueError) else _PUBLIC_IMAGE_EDIT_ERROR
        else:
            renovated_image = RenovatedImageResult(
                renovated_image_url=f"data:{edit_result.media_type};base64,{edit_result.image_base64}",
            )
    else:
        if payload.condition_score is None:
            raise HTTPException(
                status_code=400,
                detail="Provide either image_url or condition_score fallback.",
            )
        image_condition = ImageConditionResult(
            condition_score=payload.condition_score,
            issues=payload.issues,
            room_type=payload.room_type,
            analysis_status="manual_input",
            fallback_reason="manual_condition_score",
        )
        resolved_target_style = _resolve_target_renovation_style(payload)

    estimate_input = _build_estimate_input(
        payload,
        image_condition=image_condition,
        resolved_target_style=resolved_target_style,
    )
    estimate = estimate_renovation_cost(estimate_input)
    location_factor = max((payload.labor_index * 0.6) + (payload.material_index * 0.4), 0.5)

    if payload.renovation_elements:
        estimate = estimate.model_copy(
            update={
                "assumptions": [
                    *estimate.assumptions,
                    "Estimate scope is constrained to user-selected renovation elements only.",
                ]
            }
        )
    else:
        estimate = apply_user_input_cost_adjustments(
            estimate,
            payload.user_inputs,
            payload.sqft,
            location_factor=location_factor,
            renovation_elements=payload.renovation_elements,
        )

    if image_condition.analysis_status != "ai_success":
        fallback_note = f"Condition source: {image_condition.analysis_status}"
        if image_condition.fallback_reason:
            fallback_note += f" ({image_condition.fallback_reason})."
        else:
            fallback_note += "."
        estimate = estimate.model_copy(update={"assumptions": [*estimate.assumptions, fallback_note]})

    return RenovationEstimateResponse(
        image_condition=image_condition,
        estimate=estimate,
        renovated_image=renovated_image,
        renovated_image_error=renovated_image_error,
    )
