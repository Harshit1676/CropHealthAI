"""CropHealth AI — Home page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from frontend.components import api_client, styling

st.set_page_config(
    page_title="CropHealth AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)
styling.inject_css()

with st.sidebar:
    st.markdown("## 🌾 CropHealth AI")
    st.caption("Crop Disease & Pest Detection and Risk Advisory System")
    st.markdown("---")
    st.markdown("**Navigate using the pages above:**")
    st.markdown(
        "- 🔬 Disease Detection\n"
        "- 🐛 Pest Detection\n"
        "- ⚠️ Risk Assessment\n"
        "- 💊 Treatment Advisor\n"
        "- 📊 History & Analytics"
    )
    st.markdown("---")
    st.caption(f"Backend: `{api_client.BACKEND_URL}`")

styling.page_header(
    "🌾 CropHealth AI",
    "Crop Disease & Pest Detection and Risk Advisory System — Cashew · Cassava · Maize · Tomato",
)

backend_ok = api_client.backend_status_banner()

health = api_client.check_health() if backend_ok else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    styling.card_start()
    st.metric("Supported Crops", "4")
    st.caption("Cashew, Cassava, Maize, Tomato")
    styling.card_end()
with col2:
    styling.card_start()
    num_classes = health["num_classes"] if health else 0
    st.metric("Disease/Pest Classes", num_classes if num_classes else "—")
    st.caption("Trained model classes" if num_classes else "Model not trained yet")
    styling.card_end()
with col3:
    styling.card_start()
    total = 0
    if backend_ok:
        ok, data = api_client.get_statistics()
        if ok:
            total = data["total_predictions"]
    st.metric("Total Predictions", total)
    st.caption("Stored in local history")
    styling.card_end()
with col4:
    styling.card_start()
    model_status = "✅ Trained" if (health and health["model_loaded"]) else "⏳ Not trained"
    st.metric("Model Status", model_status)
    st.caption("DEMO MODE" if (health and health.get("demo_mode")) else "Real inference" if health else "—")
    styling.card_end()

if health and not health["model_loaded"] and not health.get("demo_mode"):
    st.warning(
        "The ML model has not been trained yet. Run:\n\n"
        "`python -m ml.train --epochs 10 --batch-size 32`\n\n"
        "Place the extracted Kaggle dataset into the `dataset/` folder first. "
        "See `dataset/README_PLACE_DATASET_HERE.txt` for details."
    )

st.markdown("### About this system")
styling.card_start()
st.markdown(
    """
CropHealth AI helps identify crop diseases and pests from a leaf photo, estimates
an **Explainable Crop Health Risk Score** that combines detection confidence with
weather conditions and crop stage, and provides structured, safety-conscious
treatment/management guidance.

**Supported crops & example conditions:**
- 🌰 **Cashew** — Anthracnose, Gummosis, Leaf Miner, Red Rust, Healthy
- 🌿 **Cassava** — Bacterial Blight, Brown Spot, Green Mite, Mosaic, Healthy
- 🌽 **Maize** — Fall Armyworm, Grasshopper, Leaf Beetle, Leaf Blight, Leaf Spot, Streak Virus, Healthy
- 🍅 **Tomato** — Leaf Blight, Leaf Curl, Septoria Leaf Spot, Verticillium Wilt, Healthy
"""
)
styling.card_end()

st.markdown("### Recent Predictions")
if backend_ok:
    ok, data = api_client.get_predictions(limit=5)
    if ok and data:
        df = pd.DataFrame(data)[["timestamp", "crop", "prediction", "confidence", "risk_level"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif ok:
        st.info("No predictions recorded yet. Try the Disease Detection page.")
    else:
        st.error(data)
else:
    st.info("Connect the backend to see recent prediction history.")
