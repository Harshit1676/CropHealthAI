"""CropHealth AI — Disease Detection page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from PIL import Image

from frontend.components import api_client, styling

st.set_page_config(page_title="Disease Detection | CropHealth AI", page_icon="🔬", layout="wide")
styling.inject_css()
styling.page_header("🔬 Disease Detection", "Upload a leaf photo to detect crop disease and estimate confidence.")

backend_ok = api_client.backend_status_banner()

col_upload, col_result = st.columns([1, 1.3])

with col_upload:
    styling.card_start()
    st.markdown("#### 1. Upload a leaf image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], key="disease_upload"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
    styling.card_end()

    run_prediction = st.button("🔎 Analyze Image", type="primary", use_container_width=True,
                                disabled=(uploaded_file is None or not backend_ok))

with col_result:
    styling.card_start()
    st.markdown("#### 2. Prediction Result")

    if "disease_result" not in st.session_state:
        st.session_state.disease_result = None

    if run_prediction and uploaded_file is not None:
        with st.spinner("Running inference..."):
            uploaded_file.seek(0)
            ok, result = api_client.predict_image(
                uploaded_file.read(), uploaded_file.name, uploaded_file.type or "image/jpeg"
            )
        if not ok:
            st.error(result)
            st.session_state.disease_result = None
        else:
            st.session_state.disease_result = result

    result = st.session_state.disease_result

    if result is None:
        st.info("Upload an image and click **Analyze Image** to see results here.")
    elif not result.get("success"):
        st.warning(result.get("message") or "Model not trained yet.")
    else:
        if result.get("demo_mode"):
            st.warning(f"🧪 {result.get('message')}")

        st.markdown(f"### {result['label']}")
        st.markdown(styling.category_badge_html(result["category"]), unsafe_allow_html=True)
        st.markdown("")

        # ---------------------------------------------------------
        # Prediction reliability
        # ---------------------------------------------------------
        status = result.get("status", "confident")
        confidence = float(result.get("confidence", 0))
        margin = float(result.get("margin", 0))

        if status == "uncertain":
            st.error("🔴 Uncertain Prediction")
            st.warning(
                result.get("message")
                or "The model is not sufficiently confident in this prediction."
            )

        elif status == "caution":
            st.warning("🟡 Caution — Visually Similar Condition")
            st.info(
                result.get("message")
                or "This condition can be visually similar to other classes."
            )

        else:
            st.success("🟢 Confident Prediction")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric("Confidence", f"{confidence:.2f}%")

        with metric_col2:
            st.metric("Top-1 vs Top-2 Margin", f"{margin:.2f}%")

        st.progress(min(confidence / 100, 1.0))

        st.markdown("##### Top-3 Predictions")
        for i, pred in enumerate(result["top_predictions"], start=1):
            st.write(f"{i}. **{pred['label']}** — {pred['confidence']:.2f}%")
            st.progress(min(pred["confidence"] / 100, 1.0))

    styling.card_end()

if st.session_state.get("disease_result") and st.session_state.disease_result.get("success"):
    result = st.session_state.disease_result
    st.markdown("---")
    st.markdown("### 3. Risk & Advisory")

    risk_col, advisory_col = st.columns(2)

    risk_ok = False
    risk_data = None

    with risk_col:
        styling.card_start()
        st.markdown("#### ⚠️ Quick Risk Estimate")
        st.caption("Uses default environmental assumptions. Visit **Risk Assessment** for full control.")
        payload = {
            "crop": result["crop"], "condition": result["condition"],
            "category": result["category"], "confidence": result["confidence"],
            "crop_stage": "Vegetative", "historical_risk": "Medium",
        }
        risk_ok, risk_data = api_client.calculate_risk(payload)
        if risk_ok:
            st.markdown(f"**Risk Score:** {risk_data['risk_score']} / 100")
            st.markdown(styling.risk_badge_html(risk_data["risk_level"]), unsafe_allow_html=True)
            with st.expander("Why this score?"):
                for factor in risk_data["factors"]:
                    st.write(f"- **{factor['name']}**: +{factor['contribution']} — {factor['explanation']}")
        else:
            st.error(risk_data)
        styling.card_end()

    with advisory_col:
        styling.card_start()
        st.markdown("#### 💊 Advisory Summary")
        adv_ok, adv = api_client.get_advisory(result["crop"], result["condition"])
        if adv_ok and adv.get("found"):
            st.markdown("**Symptoms:**")
            for s in adv["symptoms"]:
                st.write(f"- {s}")
            st.markdown("**Immediate actions:**")
            for a in adv["immediate_actions"]:
                st.write(f"- {a}")
            st.caption("See Treatment Advisor page for full guidance.")
        else:
            st.info("No advisory entry available for this condition.")
        styling.card_end()

    save_key = f"saved_{uploaded_file.name}_{result['confidence']}" if uploaded_file else None
    if backend_ok and save_key and not st.session_state.get(save_key):
        save_payload = {
            "crop": result["crop"],
            "prediction": result["condition"],
            "confidence": result["confidence"],
            "risk_score": risk_data.get("risk_score") if risk_ok else None,
            "risk_level": risk_data.get("risk_level") if risk_ok else None,
            "temperature": None, "humidity": None, "rainfall": None, "location": None,
        }
        saved_ok, _ = api_client.save_prediction(save_payload)
        if saved_ok:
            st.session_state[save_key] = True
