from pydantic import BaseModel, Field


class RenovationLineItem(BaseModel):
    category: str
    quantity: float = Field(ge=0)
    unit: str
    unit_cost_low: float = Field(ge=0)
    unit_cost_high: float = Field(ge=0)
    cost_low: float = Field(ge=0)
    cost_high: float = Field(ge=0)
