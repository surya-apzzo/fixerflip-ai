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
    images_analyzed: int = Field(ge=0)
    images_discarded: int = Field(ge=0)
    cost_usd: float = Field(ge=0)

