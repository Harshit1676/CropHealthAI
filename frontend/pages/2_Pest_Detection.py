"""CropHealth AI — Pest Detection page.

Uses the same trained classifier as Disease Detection (it already contains
pest classes such as Fall Armyworm, Grasshopper, Leaf Beetle, Leaf Miner,
Green Mite) but presents a dedicated pest-focused workflow and messaging.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from PIL import Image

from frontend.components import api_client, styling

st.set_page_config(page_title="Pest Detection | CropHealth AI", page_icon="🐛", layout="wide")
styling.inject_css()
styling.page_header("🐛 Pest Detection", "Identify pest infestations (e.g. Fall Armyworm, Leaf Miner, Green Mite) from a leaf photo.")

backend_ok = api_client.backend_status_banner()

st.info(
    "This page uses the same trained CropHealth AI model as Disease Detection — "
    "it recognizes both disease and pest classes. Results are automatically "
    "labeled by category (pest / disease / healthy)."
)

col_upload, col_result = st.columns([1, 1.3])

with col_upload:
    styling.card_start()
    st.markdown("#### 1. Upload a leaf image showing pest damage")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], key="pest_upload"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
    styling.card_end()

    run_prediction = st.button("🔎 Analyze for Pests", type="primary", use_container_width=True,
                                disabled=(uploaded_file is None or not backend_ok))

with col_result:
    styling.card_start()
    st.markdown("#### 2. Result")

    if "pest_result" not in st.session_state:
        st.session_state.pest_result = None

    if run_prediction and uploaded_file is not None:
        with st.spinner("Running inference..."):
            uploaded_file.seek(0)
            ok, result = api_client.predict_image(
                uploaded_file.read(), uploaded_file.name, uploaded_file.type or "image/jpeg"
            )
        if not ok:
            st.error(result)
            st.session_state.pest_result = None
        else:
            st.session_state.pest_result = result

    result = st.session_state.pest_result

    if result is None:
        st.info("Upload an image and click **Analyze for Pests** to see results here.")
    elif not result.get("success"):
        st.warning(result.get("message") or "Model not trained yet.")
    else:
        if result.get("demo_mode"):
            st.warning(f"🧪 {result.get('message')}")

        category = result["category"]
        if category == "pest":
            st.success(f"🐛 Pest detected: **{result['label']}**")
        elif category == "healthy":
            st.success(f"✅ No pest detected — plant appears **{result['label']}**")
        else:
            st.info(f"ℹ️ This looks like a **disease** rather than a pest: **{result['label']}**")

        st.markdown(styling.category_badge_html(category), unsafe_allow_html=True)
        st.metric("Confidence", f"{result['confidence']:.2f}%")
        st.progress(min(result["confidence"] / 100, 1.0))

        st.markdown("##### Top-3 Predictions")
        for i, pred in enumerate(result["top_predictions"], start=1):
            tag = "🐛" if pred["category"] == "pest" else ("✅" if pred["category"] == "healthy" else "🦠")
            st.write(f"{i}. {tag} **{pred['label']}** — {pred['confidence']:.2f}%")

    styling.card_end()

result = st.session_state.get("pest_result")
if result and result.get("success") and result["category"] == "pest":
    st.markdown("---")
    st.markdown("### 3. Recommended IPM Actions")
    ok, adv = api_client.get_advisory(result["crop"], result["condition"])
    if ok and adv.get("found"):
        cols = st.columns(2)
        with cols[0]:
            styling.card_start()
            st.markdown("**Integrated Pest Management (IPM):**")
            for a in adv["ipm_actions"]:
                st.write(f"- {a}")
            styling.card_end()
        with cols[1]:
            styling.card_start()
            st.markdown("**Immediate actions:**")
            for a in adv["immediate_actions"]:
                st.write(f"- {a}")
            st.caption("See Treatment Advisor page for full details.")
            styling.card_end()
    else:
        st.info("No IPM advisory found for this pest class.")
