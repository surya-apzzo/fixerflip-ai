from typing import List

from pydantic import BaseModel, Field


class ImageConditionResult(BaseModel):
    """API response shape for image condition analysis."""

    condition_score: int = Field(ge=0, le=100, description="100 = excellent, 0 = severe distress")
    issues: List[str] = Field(default_factory=list)
    room_type: str = Field(
        default="unknown",
        description="Primary room/area: kitchen, bathroom, exterior, living, bedroom, other",
    )
    issue_details: List["IssueDetection"] = Field(
        default_factory=list,
        description="Structured issue detections including type, severity, and confidence.",
    )


class IssueDetection(BaseModel):
    type: str
    severity: str = "moderate"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class PositiveDetection(BaseModel):
    type: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class RoomDetection(BaseModel):
    room: str = "unknown"
    condition: str = "average"
    issues: List[IssueDetection] = Field(default_factory=list)
    positives: List[PositiveDetection] = Field(default_factory=list)
