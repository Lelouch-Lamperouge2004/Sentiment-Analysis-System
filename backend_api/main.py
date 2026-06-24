from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

import backend

from backend_api.database import engine, get_db
from backend_api.models import Base, PredictionHistory
from backend_api.schemas import PredictionRequest, PredictionResponse, HistoryResponse

from backend_api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HistoryResponse,
    BatchPredictionRequest,
    BatchPredictionItem
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "running",
        "model": "Linear SVC",
        "message": "Sentiment Analysis API is active"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest, db: Session = Depends(get_db)):
    if request.review.strip() == "":
        raise HTTPException(status_code=400, detail="Review text cannot be empty")

    result = backend.predict_sentiment(request.review)

    history = PredictionHistory(
        review=request.review,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        processed_text=result["processed_text"]
    )

    db.add(history)
    db.commit()
    db.refresh(history)

    return result


@app.get("/history", response_model=list[HistoryResponse])
def get_prediction_history(db: Session = Depends(get_db)):
    return (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .limit(100)
        .all()
    )


@app.get("/analytics")
def get_prediction_analytics(db: Session = Depends(get_db)):
    total = db.query(PredictionHistory).count()

    positive = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.sentiment == "Positive")
        .count()
    )

    negative = (
        db.query(PredictionHistory)
        .filter(PredictionHistory.sentiment == "Negative")
        .count()
    )

    avg_confidence = db.query(func.avg(PredictionHistory.confidence)).scalar()

    return {
        "total_predictions": total,
        "positive_predictions": positive,
        "negative_predictions": negative,
        "average_confidence": round(avg_confidence or 0, 2)
    }

@app.post("/predict-batch", response_model=list[BatchPredictionItem])
def predict_batch(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db)
):
    results = []

    for review in request.reviews:
        if not str(review).strip():
            continue

        result = backend.predict_sentiment(str(review))

        history = PredictionHistory(
            review=str(review),
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            processed_text=result["processed_text"]
        )

        db.add(history)

        results.append({
            "review": str(review),
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "processed_text": result["processed_text"]
        })

    db.commit()

    return results