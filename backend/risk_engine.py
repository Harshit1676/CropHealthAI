"""
Explainable Crop Health Risk Score engine.

This is a transparent, rule-based scoring system — NOT a scientifically
validated agricultural forecasting model. It combines disease/pest
severity, model confidence, weather conditions, crop growth stage and
historical occurrence into a single 0-100 risk score with a fully
itemized breakdown of contributing factors.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = config.get_logger(__name__)

# Base severity per condition category (0-100 scale contribution before
# weighting). Healthy plants contribute very little baseline risk.
CATEGORY_BASE_SEVERITY = {
    "healthy": 5,
    "pest": 55,
    "disease": 65,
}

CROP_STAGE_MULTIPLIER = {
    "Seedling": 1.15,
    "Vegetative": 1.0,
    "Flowering": 1.1,
    "Fruiting": 1.2,
    "Maturity": 0.85,
}

HISTORICAL_RISK_POINTS = {
    "Low": 5,
    "Medium": 12,
    "High": 20,
}


def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def _risk_level(score: float) -> str:
    for low, high, label in config.RISK_LEVELS:
        if low <= score <= high:
            return label
    return "Severe" if score > 100 else "Low"


def _environmental_risk(temperature: float | None, humidity: float | None,
                         rainfall: float | None) -> tuple[float, str]:
    """
    Most fungal/bacterial crop diseases and many pests thrive in warm,
    humid, wet conditions. This is a simplified heuristic, not a
    pathogen-specific epidemiological model.
    """
    if temperature is None and humidity is None and rainfall is None:
        return 0.0, "No weather data provided — environmental factor not applied."

    points = 0.0
    reasons = []

    if humidity is not None:
        if humidity >= 85:
            points += 25
            reasons.append(f"very high humidity ({humidity:.0f}%)")
        elif humidity >= 70:
            points += 15
            reasons.append(f"high humidity ({humidity:.0f}%)")
        elif humidity >= 50:
            points += 6
            reasons.append(f"moderate humidity ({humidity:.0f}%)")

    if temperature is not None:
        if 20 <= temperature <= 32:
            points += 15
            reasons.append(f"temperature in favorable disease/pest range ({temperature:.0f}°C)")
        elif 15 <= temperature < 20 or 32 < temperature <= 36:
            points += 8
            reasons.append(f"moderately favorable temperature ({temperature:.0f}°C)")

    if rainfall is not None:
        if rainfall >= 20:
            points += 20
            reasons.append(f"heavy rainfall ({rainfall:.0f} mm)")
        elif rainfall >= 5:
            points += 10
            reasons.append(f"moderate rainfall ({rainfall:.0f} mm)")

    points = _clamp(points, 0, 60)
    explanation = ("Favorable conditions detected: " + ", ".join(reasons)) if reasons \
        else "Weather conditions are not particularly favorable for disease/pest spread."
    return points, explanation


def calculate_risk(
    crop: str,
    condition: str,
    category: str,
    confidence: float,
    temperature: float | None = None,
    humidity: float | None = None,
    rainfall: float | None = None,
    crop_stage: str = "Vegetative",
    historical_risk: str = "Medium",
) -> dict:
    """
    Compute the Explainable Crop Health Risk Score.

    Weighted components (sum to 100 max before crop-stage multiplier):
      - Disease/pest severity base   : up to 40 points
      - Model confidence factor      : up to 15 points
      - Environmental risk           : up to 25 points
      - Historical occurrence risk   : up to 20 points
    Then adjusted by a crop-growth-stage multiplier.
    """
    category = category.lower()
    base_severity_raw = CATEGORY_BASE_SEVERITY.get(category, 50)
    severity_points = _clamp(base_severity_raw * 0.4, 0, 40)

    confidence_points = _clamp((confidence / 100.0) * 15, 0, 15) if category != "healthy" else \
        _clamp(3, 0, 15)

    env_points, env_explanation = _environmental_risk(temperature, humidity, rainfall)
    env_points_scaled = _clamp(env_points / 60 * 25, 0, 25)

    hist_points = HISTORICAL_RISK_POINTS.get(historical_risk, 12)

    raw_score = severity_points + confidence_points + env_points_scaled + hist_points

    multiplier = CROP_STAGE_MULTIPLIER.get(crop_stage, 1.0)
    final_score = _clamp(raw_score * multiplier)

    if category == "healthy":
        # Healthy plants should never be scored as high-risk regardless of
        # weather; cap firmly.
        final_score = _clamp(min(final_score, 20))

    level = _risk_level(final_score)

    factors = [
        {
            "name": "Disease/Pest Severity",
            "contribution": round(severity_points, 1),
            "explanation": f"Base severity for category '{category}' ({condition}).",
        },
        {
            "name": "Model Confidence",
            "contribution": round(confidence_points, 1),
            "explanation": f"Detection confidence was {confidence:.1f}%.",
        },
        {
            "name": "Environmental Risk",
            "contribution": round(env_points_scaled, 1),
            "explanation": env_explanation,
        },
        {
            "name": "Historical Occurrence Risk",
            "contribution": round(hist_points, 1),
            "explanation": f"Historical risk level for this crop/region set to '{historical_risk}'.",
        },
        {
            "name": "Crop Growth Stage Adjustment",
            "contribution": round(final_score - raw_score, 1) if category != "healthy" else 0.0,
            "explanation": f"Crop stage '{crop_stage}' applies a "
                            f"{multiplier:.2f}x multiplier to overall risk.",
        },
    ]

    actions = _recommended_actions(level, category)

    return {
        "risk_score": round(final_score, 1),
        "risk_level": level,
        "factors": factors,
        "recommended_actions": actions,
    }


def _recommended_actions(level: str, category: str) -> list[str]:
    if category == "healthy":
        return [
            "Continue routine field monitoring (weekly visual inspection).",
            "Maintain balanced irrigation and fertilization schedules.",
            "Re-scan periodically, especially after weather changes.",
        ]

    actions = {
        "Low": [
            "Continue regular monitoring; no immediate intervention required.",
            "Record the observation for trend tracking.",
            "Re-check the affected plants within 7-10 days.",
        ],
        "Moderate": [
            "Increase field scouting frequency to twice weekly.",
            "Isolate/mark affected plants and monitor spread.",
            "Consult the Treatment Advisor page for condition-specific IPM guidance.",
            "Follow the pesticide label and local agricultural extension recommendations if treatment is needed.",
        ],
        "High": [
            "Inspect surrounding plants immediately for spread.",
            "Apply recommended IPM measures from the Treatment Advisor without delay.",
            "Consider consulting a local agricultural extension officer.",
            "Follow the pesticide label and local agricultural extension recommendations for any chemical control.",
        ],
        "Severe": [
            "Take immediate action — risk of significant yield loss.",
            "Remove/quarantine severely affected plant material where appropriate.",
            "Contact a local agricultural extension officer or plant pathologist promptly.",
            "Follow the pesticide label and local agricultural extension recommendations strictly for any chemical intervention.",
            "Reassess the field within 48-72 hours.",
        ],
    }
    return actions.get(level, actions["Moderate"])
