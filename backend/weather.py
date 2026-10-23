"""
Optional weather integration using the OpenWeather API.

The application must never crash if the API key is missing or the API
call fails — callers always get a well-formed WeatherResponse-compatible
dict, with `source` indicating whether real data, manual fallback, or
"unavailable" was used.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = config.get_logger(__name__)


def is_weather_api_configured() -> bool:
    return bool(config.OPENWEATHER_API_KEY)


def fetch_weather(location: str) -> dict:
    """
    Fetch current weather for `location` from OpenWeather.

    Returns a dict compatible with schemas.WeatherResponse. Never raises —
    any failure is captured and returned as source="unavailable".
    """
    if not location or not location.strip():
        return {
            "success": False,
            "source": "unavailable",
            "message": "No location provided.",
        }

    if not is_weather_api_configured():
        return {
            "success": False,
            "source": "unavailable",
            "message": "OPENWEATHER_API_KEY is not configured. Enter weather values manually.",
        }

    try:
        import requests
        response = requests.get(
            config.OPENWEATHER_BASE_URL,
            params={
                "q": location,
                "appid": config.OPENWEATHER_API_KEY,
                "units": "metric",
            },
            timeout=8,
        )
        if response.status_code != 200:
            logger.warning("OpenWeather API returned status %s for %s", response.status_code, location)
            return {
                "success": False,
                "source": "unavailable",
                "message": f"Weather API returned an error (status {response.status_code}). "
                           f"Enter weather values manually.",
            }
        data = response.json()
        rainfall = 0.0
        if "rain" in data:
            rainfall = data["rain"].get("1h", data["rain"].get("3h", 0.0))

        return {
            "success": True,
            "source": "api",
            "location": data.get("name", location),
            "temperature": data.get("main", {}).get("temp"),
            "humidity": data.get("main", {}).get("humidity"),
            "rainfall": rainfall,
            "description": (data.get("weather", [{}])[0].get("description", "") or "").capitalize(),
        }
    except Exception as exc:  # noqa: BLE001 - never let weather crash the app
        logger.warning("Weather fetch failed for '%s': %s", location, exc)
        return {
            "success": False,
            "source": "unavailable",
            "message": "Could not reach the weather service. Enter weather values manually.",
        }
