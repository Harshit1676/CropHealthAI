"""
CropHealth AI — FastAPI backend.

Run with:
    uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from backend import advisory, database, risk_engine, weather
from backend.predictor import InvalidImageError, get_predictor
from backend.schemas import (
    AdvisoryResponse, HealthResponse, PredictionCreate, PredictionRecord,
    PredictionResponse, RiskRequest, RiskResponse, StatisticsResponse,
    WeatherRequest, WeatherResponse,
)

logger = config.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        database.init_db()
    except Exception as exc:  # noqa: BLE001
        logger.error("Database initialization failed: %s", exc)
    # Warm-load the model (non-fatal if missing).
    get_predictor()
    yield


app = FastAPI(
    title="CropHealth AI API",
    description="Crop Disease & Pest Detection and Risk Advisory System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "project": "CropHealth AI",
        "description": "Crop Disease & Pest Detection and Risk Advisory System",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    predictor = get_predictor()
    db_ok = True
    try:
        database.count_predictions()
    except Exception:  # noqa: BLE001
        db_ok = False

    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        demo_mode=config.DEMO_MODE,
        num_classes=len(predictor.class_names),
        database_ok=db_ok,
        weather_configured=weather.is_weather_api_configured(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["ml"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPG or PNG image.",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        predictor = get_predictor()
        result = predictor.predict(image_bytes)
        return PredictionResponse(**result)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.") from exc


@app.post("/risk", response_model=RiskResponse, tags=["risk"])
def risk(payload: RiskRequest) -> RiskResponse:
    try:
        result = risk_engine.calculate_risk(
            crop=payload.crop,
            condition=payload.condition,
            category=payload.category,
            confidence=payload.confidence,
            temperature=payload.temperature,
            humidity=payload.humidity,
            rainfall=payload.rainfall,
            crop_stage=payload.crop_stage,
            historical_risk=payload.historical_risk,
        )
        return RiskResponse(**result)
    except Exception as exc:  # noqa: BLE001
        logger.error("Risk calculation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Risk calculation failed due to an internal error.") from exc


@app.get("/advisory/{crop}/{condition}", response_model=AdvisoryResponse, tags=["advisory"])
def get_advisory_endpoint(crop: str, condition: str) -> AdvisoryResponse:
    record = advisory.get_advisory(crop, condition)
    if record is None:
        return AdvisoryResponse(
            found=False,
            crop=crop,
            condition=condition,
            message=f"No advisory entry found for '{crop} — {condition}'.",
        )
    return AdvisoryResponse(found=True, crop=crop.capitalize(), condition=condition, **record)


@app.get("/advisory-list", tags=["advisory"])
def list_advisory_conditions() -> list[dict]:
    return advisory.list_all_conditions()


@app.get("/predictions", response_model=list[PredictionRecord], tags=["history"])
def get_predictions(limit: int = 100, offset: int = 0) -> list[PredictionRecord]:
    try:
        rows = database.get_predictions(limit=limit, offset=offset)
        return [PredictionRecord(**row) for row in rows]
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch predictions: %s", exc)
        raise HTTPException(status_code=500, detail="Could not read prediction history.") from exc


@app.post("/predictions", response_model=PredictionRecord, tags=["history"])
def create_prediction(payload: PredictionCreate) -> PredictionRecord:
    try:
        new_id = database.insert_prediction(**payload.model_dump())
        rows = database.get_predictions(limit=1, offset=0)
        return PredictionRecord(**rows[0]) if rows else PredictionRecord(
            id=new_id, timestamp=datetime.utcnow(), **payload.model_dump()
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save prediction: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save prediction to the database.") from exc


@app.get("/statistics", response_model=StatisticsResponse, tags=["history"])
def get_statistics() -> StatisticsResponse:
    try:
        rows = database.get_all_predictions()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to compute statistics: %s", exc)
        raise HTTPException(status_code=500, detail="Could not read prediction history.") from exc

    disease_dist: Counter = Counter()
    pest_dist: Counter = Counter()
    healthy_dist: Counter = Counter()
    risk_dist: Counter = Counter()
    crop_dist: Counter = Counter()
    by_date: defaultdict = defaultdict(int)

    for row in rows:
        crop_dist[row["crop"]] += 1
        pred_label = row["prediction"]
        lower = pred_label.lower()
        if "healthy" in lower:
            healthy_dist[pred_label] += 1
        else:
            from ml.model_utils import parse_class_name
            meta = parse_class_name(f"{row['crop']}_{pred_label}")
            if meta["category"] == "pest":
                pest_dist[pred_label] += 1
            else:
                disease_dist[pred_label] += 1

        if row["risk_level"]:
            risk_dist[row["risk_level"]] += 1
        if row["timestamp"]:
            date_part = row["timestamp"][:10]
            by_date[date_part] += 1

    predictor = get_predictor()
    return StatisticsResponse(
        total_predictions=len(rows),
        disease_distribution=dict(disease_dist),
        pest_distribution=dict(pest_dist),
        healthy_distribution=dict(healthy_dist),
        risk_distribution=dict(risk_dist),
        crop_distribution=dict(crop_dist),
        predictions_over_time=[{"date": d, "count": c} for d, c in sorted(by_date.items())],
        model_trained=predictor.is_loaded,
        num_classes=len(predictor.class_names),
    )


@app.post("/weather", response_model=WeatherResponse, tags=["weather"])
def get_weather(payload: WeatherRequest) -> WeatherResponse:
    result = weather.fetch_weather(payload.location)
    return WeatherResponse(**result)
