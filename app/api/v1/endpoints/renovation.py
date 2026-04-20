from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.rules_config import DEFAULT_ADMIN_LABOR_INDEX, DEFAULT_ADMIN_MATERIAL_INDEX
from app.engine.renovation_engine.image_condition_engine import ImageConditionResult
from app.engine.renovation_engine.image_edit_engine import (
    build_instruction_for_edit,
    edit_property_image_from_url,
)
from app.engine.renovation_engine.renovation_cost_engine import (
    RenovationEstimate,
    RenovationEstimateInput,
    apply_user_input_cost_adjustments,
    estimate_renovation_cost,
)
from app.engine.renovation_engine.vision_analysis import analyze_renovation_image_url

router = APIRouter(prefix="/renovation")
logger = logging.getLogger(__name__)

_PUBLIC_IMAGE_EDIT_ERROR = "Image edit failed. Retry later or inspect server logs."


class RenovationEstimateRequest(BaseModel):
    """
    Backend-facing payload contract for /renovation/estimate.

    Notes:
    - Backend sends these fields to this endpoint.
    - `condition_score` / `issues` / `room_type` are used as manual fallback
      when `image_url` is omitted.
    - This payload is later mapped into internal `RenovationEstimateInput`
      before cost calculation.
    """

    image_url: str = ""
    address: str = ""
    city: str = ""
    sqft: float = Field(gt=0)
    beds: int = Field(ge=0)
    baths: float = Field(ge=0)
    lot_size: float = Field(default=0.0, ge=0.0)
    year_built: int | None = None
    property_type: str = "SFR"
    listing_price: float = Field(default=0.0, ge=0.0)
    listing_description: str = ""
    listing_status: str = ""
    days_on_market: int = Field(default=0, ge=0)
    avg_area_price_per_sqft: float = Field(default=0.0, ge=0.0)
    years_since_last_sale: int | None = Field(default=None, ge=0)
    permit_years_since_last: int | None = Field(default=None, ge=0)
    zip_code: str = ""
    target_renovation_style: str = "investor_standard"
    desired_quality_level: str = "standard"
    labor_index: float = Field(default=DEFAULT_ADMIN_LABOR_INDEX, ge=0.7, le=2.5)
    material_index: float = Field(default=DEFAULT_ADMIN_MATERIAL_INDEX, ge=0.7, le=2.5)
    type_of_renovation: str = "interior"
    visual_type: str = "select_elements_to_renovate"
    reference_image_url: str = ""
    renovation_elements: list[str] = Field(default_factory=list, max_length=4)
    # Optional manual fallback values if no images are provided.
    condition_score: int | None = Field(default=None, ge=0, le=100)
    issues: list[str] = Field(default_factory=list)
    room_type: str = "unknown"
    user_inputs: str = ""

    @field_validator("image_url")
    @classmethod
    def strip_optional_image_url(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("reference_image_url")
    @classmethod
    def strip_reference_image_url(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("renovation_elements", mode="before")
    @classmethod
    def normalize_renovation_elements(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            raw = [x.strip() for x in v.split(",") if x.strip()]
        elif isinstance(v, list):
            raw = [str(x).strip() for x in v if str(x).strip()]
        else:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for x in raw:
            key = x.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
            if len(out) >= 4:
                break
        return out

    @field_validator("desired_quality_level", mode="before")
    @classmethod
    def normalize_desired_quality_level(cls, v: object) -> str:
        raw = str(v or "standard").strip().lower().replace("-", " ").replace("_", " ")
        aliases = {
            "standard investor rehab": "standard",
            "investor rehab": "standard",
            "standard rehab": "standard",
        }
        normalized = aliases.get(raw, raw)
        allowed = {"cosmetic", "standard", "premium", "luxury"}
        return normalized if normalized in allowed else "standard"


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
    """
    Map backend request payload + resolved image condition into internal estimator input.

    This is the boundary between:
    1) what backend clients send (`RenovationEstimateRequest`)
    2) what cost engine consumes (`RenovationEstimateInput`)
    """
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


class RenovationEstimateResponse(BaseModel):
    class RenovatedImageResult(BaseModel):
        """Client-facing preview only — internal edit prompt is not exposed."""

        renovated_image_url: str

    image_condition: ImageConditionResult
    estimate: RenovationEstimate
    renovated_image: RenovatedImageResult | None = None
    renovated_image_error: str | None = None

@router.post("/estimate", response_model=RenovationEstimateResponse)
async def renovation_estimate(payload: RenovationEstimateRequest) -> RenovationEstimateResponse:
    """
    Renovation estimate from image condition + property facts.
    """
    renovated_image: RenovationEstimateResponse.RenovatedImageResult | None = None
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
        vision_task = analyze_renovation_image_url(payload.image_url)
        edit_task = edit_property_image_from_url(
            image_url=payload.image_url,
            instruction=instruction_for_edit,
        )
        vision_result, edit_result = await asyncio.gather(
            vision_task,
            edit_task,
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
            if isinstance(edit_result, ValueError):
                renovated_image_error = str(edit_result)
            else:
                renovated_image_error = _PUBLIC_IMAGE_EDIT_ERROR
        else:
            renovated_image = RenovationEstimateResponse.RenovatedImageResult(
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
    # Make non-AI/manual fallback visible to API consumers so estimate quality can be monitored.
    if image_condition.analysis_status != "ai_success":
        fallback_note = (
            f"Condition source: {image_condition.analysis_status}"
            + (
                f" ({image_condition.fallback_reason})."
                if image_condition.fallback_reason
                else "."
            )
        )
        estimate = estimate.model_copy(update={"assumptions": [*estimate.assumptions, fallback_note]})

    return RenovationEstimateResponse(
        image_condition=image_condition,
        estimate=estimate,
        renovated_image=renovated_image,
        renovated_image_error=renovated_image_error,
    )
