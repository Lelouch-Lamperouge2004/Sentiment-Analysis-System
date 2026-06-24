from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

from backend_api.database import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    review = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    processed_text = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)