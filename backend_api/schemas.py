from pydantic import BaseModel
from datetime import datetime


class PredictionRequest(BaseModel):
    review: str


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    processed_text: str


class HistoryResponse(BaseModel):
    id: int
    review: str
    sentiment: str
    confidence: float
    processed_text: str
    created_at: datetime

    class Config:
        from_attributes = True