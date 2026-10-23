"""CropHealth AI — Risk Assessment page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

import config
from frontend.components import api_client, styling
from ml.model_utils import parse_class_name

st.set_page_config(page_title="Risk Assessment | CropHealth AI", page_icon="⚠️", layout="wide")
styling.inject_css()
styling.page_header("⚠️ Risk Assessment", "Compute an Explainable Crop Health Risk Score from condition + weather + crop stage.")

backend_ok = api_client.backend_status_banner()

conditions_ok, conditions = api_client.list_advisory_conditions() if backend_ok else (False, [])
crops = sorted({c["crop"] for c in conditions}) if conditions_ok else config.KNOWN_CROPS

col_input, col_weather = st.columns(2)

with col_input:
    styling.card_start()
    st.markdown("#### Crop & Condition")
    crop = st.selectbox("Crop", crops)
    if conditions_ok:
        options = [c["condition"] for c in conditions if c["crop"] == crop]
    else:
        options = ["Healthy"]
    condition = st.selectbox("Detected condition", options)

    category = parse_class_name(f"{crop}_{condition}")["category"]

    confidence = st.slider("Detection confidence (%)", 0, 100, 85)
    crop_stage = st.selectbox(
        "Crop growth stage",
        ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"],
        index=1,
    )
    historical_risk = st.selectbox("Historical occurrence risk (this field/region)", ["Low", "Medium", "High"], index=1)
    styling.card_end()

with col_weather:
    styling.card_start()
    st.markdown("#### Weather Conditions")
    st.caption("Enter a location to auto-fetch weather (if OpenWeather API key is configured), or enter values manually.")
    location = st.text_input("Location (city name)", value="")
    fetch_col, _ = st.columns([1, 2])
    weather_data = None
    with fetch_col:
        if st.button("🌤️ Fetch Weather", disabled=not backend_ok):
            ok, weather_data = api_client.fetch_weather(location)
            if ok and weather_data.get("success"):
                st.success(f"Fetched weather for {weather_data.get('location', location)}")
                st.session_state["risk_temp"] = weather_data.get("temperature", 25.0)
                st.session_state["risk_humidity"] = weather_data.get("humidity", 60.0)
                st.session_state["risk_rainfall"] = weather_data.get("rainfall", 0.0)
            else:
                msg = weather_data.get("message") if isinstance(weather_data, dict) else weather_data
                st.warning(msg or "Weather unavailable — enter values manually below.")

    temperature = st.number_input("Temperature (°C)", value=float(st.session_state.get("risk_temp", 26.0)),
                                   min_value=-10.0, max_value=55.0, step=0.5)
    humidity = st.number_input("Humidity (%)", value=float(st.session_state.get("risk_humidity", 65.0)),
                                min_value=0.0, max_value=100.0, step=1.0)
    rainfall = st.number_input("Rainfall, last 24h (mm)", value=float(st.session_state.get("risk_rainfall", 2.0)),
                                min_value=0.0, max_value=500.0, step=1.0)
    styling.card_end()

if st.button("📊 Calculate Risk Score", type="primary", use_container_width=True, disabled=not backend_ok):
    payload = {
        "crop": crop, "condition": condition, "category": category, "confidence": float(confidence),
        "temperature": temperature, "humidity": humidity, "rainfall": rainfall,
        "crop_stage": crop_stage, "historical_risk": historical_risk,
    }
    ok, result = api_client.calculate_risk(payload)
    if not ok:
        st.error(result)
    else:
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            styling.card_start()
            st.markdown("### Result")
            st.metric("Risk Score", f"{result['risk_score']} / 100")
            st.markdown(styling.risk_badge_html(result["risk_level"]), unsafe_allow_html=True)
            st.progress(min(result["risk_score"] / 100, 1.0))
            st.caption(result["disclaimer"])
            styling.card_end()
        with res_col2:
            styling.card_start()
            st.markdown("### Contributing Factors")
            for factor in result["factors"]:
                st.write(f"**{factor['name']}**: +{factor['contribution']} pts")
                st.caption(factor["explanation"])
            styling.card_end()

        styling.card_start()
        st.markdown("### Recommended Actions")
        for action in result["recommended_actions"]:
            st.write(f"- {action}")
        styling.card_end()
