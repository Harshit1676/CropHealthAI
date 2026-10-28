"""
Thin HTTP client wrapper used by all Streamlit pages to talk to the
FastAPI backend. Every function fails gracefully (returns None / a status
dict) rather than raising, so the UI can show a friendly message instead
of a stack trace if the backend is unreachable.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests
import streamlit as st

def _get_backend_url() -> str:
    """
    Resolve the backend URL in priority order:
    1. Streamlit secrets (Streamlit Community Cloud)
    2. BACKEND_URL environment variable
    3. localhost fallback for local development
    """
    try:
        # st.secrets is available on Streamlit Community Cloud
        url = st.secrets.get("BACKEND_URL", "")
        if url:
            return url.rstrip("/")
    except Exception:  # noqa: BLE001
        pass
    return os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


BACKEND_URL = _get_backend_url()
TIMEOUT = 30


def _handle_request(method: str, path: str, **kwargs) -> tuple[bool, Any]:
    url = f"{BACKEND_URL}{path}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:  # noqa: BLE001
                detail = response.text
            return False, f"Backend error ({response.status_code}): {detail}"
        return True, response.json()
    except requests.exceptions.ConnectionError:
        return False, (
            f"Cannot reach the backend API at {BACKEND_URL}. "
            f"Make sure it is running: uvicorn backend.app:app --reload"
        )
    except requests.exceptions.Timeout:
        return False, "The backend request timed out. Please try again."
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error contacting backend: {exc}"


def check_health() -> Optional[dict]:
    ok, data = _handle_request("GET", "/health")
    return data if ok else None


def predict_image(file_bytes: bytes, filename: str, content_type: str) -> tuple[bool, Any]:
    files = {"file": (filename, file_bytes, content_type)}
    return _handle_request("POST", "/predict", files=files)


def calculate_risk(payload: dict) -> tuple[bool, Any]:
    return _handle_request("POST", "/risk", json=payload)


def get_advisory(crop: str, condition: str) -> tuple[bool, Any]:
    return _handle_request("GET", f"/advisory/{crop}/{condition}")


def list_advisory_conditions() -> tuple[bool, Any]:
    return _handle_request("GET", "/advisory-list")


def get_predictions(limit: int = 200) -> tuple[bool, Any]:
    return _handle_request("GET", "/predictions", params={"limit": limit})


def save_prediction(payload: dict) -> tuple[bool, Any]:
    return _handle_request("POST", "/predictions", json=payload)


def get_statistics() -> tuple[bool, Any]:
    return _handle_request("GET", "/statistics")


def fetch_weather(location: str) -> tuple[bool, Any]:
    return _handle_request("POST", "/weather", json={"location": location})


def backend_status_banner() -> bool:
    """Show a small backend-connectivity banner. Returns True if healthy."""
    health = check_health()
    if health is None:
        st.error(
            f"⚠️ Backend not reachable at `{BACKEND_URL}`. "
            f"Start it with: `run_backend.bat` or "
            f"`uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000`"
        )
        return False
    return True
