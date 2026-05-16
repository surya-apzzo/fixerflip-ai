from __future__ import annotations

from pydantic import BaseModel, Field


class RoomScore(BaseModel):
    room_type: str
    score: float = Field(ge=0, le=100, description="Vision condition score for this photo (0-100).")
    signals: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)


class ConditionScoreResponse(BaseModel):
    property_id: str
    condition_score: float = Field(ge=0, le=100)
    grade: str
    text_score: float = Field(ge=0, le=100)
    vision_score: float | None = Field(default=None, ge=0, le=100)
    room_scores: list[RoomScore] = Field(default_factory=list)
    positive_signals: list[str] = Field(default_factory=list)
    caution_signals: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    images_analyzed: int = Field(
        ge=0,
        description="Unique room-type photos sent to vision (after dedupe).",
    )
    images_discarded: int = Field(ge=0, description="Removed by CLIP/heuristics or failed download.")
    images_after_filter: int = Field(
        default=0,
        ge=0,
        description="House photos kept after filter, before dedupe by room_type.",
    )
    images_deduplicated: int = Field(
        default=0,
        ge=0,
        description="Extra photos skipped because the same room_type was already represented.",
    )
    urls_received: int = Field(
        default=0,
        ge=0,
        description="Non-empty image_urls in the request body.",
    )
    urls_processed: int = Field(
        default=0,
        ge=0,
        description="URLs actually downloaded/CLIP'd (after CONDITION_SCORE_MAX_INPUT_URLS sampling).",
    )
    urls_truncated: int = Field(
        default=0,
        ge=0,
        description="URLs skipped because the feed exceeded CONDITION_SCORE_MAX_INPUT_URLS.",
    )
    cost_usd: float = Field(ge=0)

