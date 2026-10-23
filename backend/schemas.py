"""Pydantic schemas shared across the FastAPI backend."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TopPrediction(BaseModel):
    label: str
    crop: str
    condition: str
    category: str
    confidence: float


class PredictionResponse(BaseModel):
    success: bool
    crop: Optional[str] = None
    condition: Optional[str] = None
    category: Optional[str] = None  # "disease" | "pest" | "healthy"
    label: Optional[str] = None
    confidence: Optional[float] = None

    # Prediction reliability information
    status: Optional[str] = None
    top1_confidence: Optional[float] = None
    top2_confidence: Optional[float] = None
    margin: Optional[float] = None
    uncertain: bool = False
    confusion_warning: bool = False

    top_predictions: list[TopPrediction] = Field(default_factory=list)
    demo_mode: bool = False
    message: Optional[str] = None


class RiskRequest(BaseModel):
    crop: str
    condition: str = "Healthy"
    category: str = "healthy"
    confidence: float = Field(default=80.0, ge=0, le=100)
    temperature: Optional[float] = Field(default=None, description="Degrees Celsius")
    humidity: Optional[float] = Field(default=None, ge=0, le=100, description="Percent")
    rainfall: Optional[float] = Field(default=None, ge=0, description="mm, last 24h")
    crop_stage: str = Field(default="Vegetative")
    historical_risk: str = Field(default="Medium", description="Low | Medium | High")


class RiskFactor(BaseModel):
    name: str
    contribution: float
    explanation: str


class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    factors: list[RiskFactor]
    recommended_actions: list[str]
    disclaimer: str = (
        "This is an Explainable Crop Health Risk Score, a rule-based "
        "decision-support estimate. It is not a scientifically validated "
        "agricultural forecasting model."
    )


class AdvisoryResponse(BaseModel):
    found: bool
    crop: Optional[str] = None
    condition: Optional[str] = None
    category: Optional[str] = None
    symptoms: list[str] = Field(default_factory=list)
    likely_cause: Optional[str] = None
    prevention: list[str] = Field(default_factory=list)
    ipm_actions: list[str] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list)
    general_treatment: Optional[str] = None
    monitoring: list[str] = Field(default_factory=list)
    message: Optional[str] = None


class PredictionRecord(BaseModel):
    id: int
    timestamp: datetime
    crop: str
    prediction: str
    confidence: float
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    location: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionCreate(BaseModel):
    crop: str
    prediction: str
    confidence: float
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    location: Optional[str] = None


class StatisticsResponse(BaseModel):
    total_predictions: int
    disease_distribution: dict[str, int]
    pest_distribution: dict[str, int]
    healthy_distribution: dict[str, int]
    risk_distribution: dict[str, int]
    crop_distribution: dict[str, int]
    predictions_over_time: list[dict]
    model_trained: bool
    num_classes: int


class WeatherRequest(BaseModel):
    location: str


class WeatherResponse(BaseModel):
    success: bool
    source: str  # "api" | "manual" | "unavailable"
    location: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    description: Optional[str] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    demo_mode: bool
    num_classes: int
    database_ok: bool
    weather_configured: bool
