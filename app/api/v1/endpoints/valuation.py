from __future__ import annotations

from fastapi import APIRouter

from app.engine.valuation_engine.valuation_service import (
    ValuationInput,
    ValuationResult,
    run_valuation_engine,
)

router = APIRouter(prefix="/valuation")


@router.post("/analyze", response_model=ValuationResult)
async def analyze_valuation(payload: ValuationInput) -> ValuationResult:
    """
    ARV + ROI valuation engine:
    comp-based ARV, profitability, ROI%, margin, and sensitivity.
    """
    return run_valuation_engine(payload)
