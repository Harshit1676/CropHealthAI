"""CropHealth AI — Treatment Advisor page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

import config
from frontend.components import api_client, styling

st.set_page_config(page_title="Treatment Advisor | CropHealth AI", page_icon="💊", layout="wide")
styling.inject_css()
styling.page_header("💊 Treatment Advisor", "Search or select a disease/pest to view symptoms, prevention, IPM and monitoring guidance.")

backend_ok = api_client.backend_status_banner()

conditions_ok, conditions = api_client.list_advisory_conditions() if backend_ok else (False, [])

if not conditions_ok:
    st.error("Could not load the advisory list from the backend.")
    st.stop()

crops = sorted({c["crop"] for c in conditions})

col1, col2 = st.columns(2)
with col1:
    crop = st.selectbox("Crop", crops)
with col2:
    options = [c["condition"] for c in conditions if c["crop"] == crop]
    condition = st.selectbox("Disease / Pest / Status", options)

ok, adv = api_client.get_advisory(crop, condition)

if not ok:
    st.error(adv)
elif not adv.get("found"):
    st.info(adv.get("message") or "No advisory entry found for this selection.")
else:
    category = adv.get("category", "disease")
    st.markdown(f"## {crop} — {condition}")
    st.markdown(styling.category_badge_html(category), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        styling.card_start()
        st.markdown("#### 🩺 Symptoms")
        for s in adv["symptoms"]:
            st.write(f"- {s}")
        st.markdown("#### 🔍 Likely Cause")
        st.write(adv["likely_cause"])
        styling.card_end()

        styling.card_start()
        st.markdown("#### 🛡️ Prevention")
        for p in adv["prevention"]:
            st.write(f"- {p}")
        styling.card_end()

    with col_b:
        styling.card_start()
        st.markdown("#### 🌱 Integrated Pest Management (IPM)")
        for i in adv["ipm_actions"]:
            st.write(f"- {i}")
        styling.card_end()

        styling.card_start()
        st.markdown("#### ⏱️ Immediate Actions")
        for a in adv["immediate_actions"]:
            st.write(f"- {a}")
        styling.card_end()

    styling.card_start()
    st.markdown("#### 💊 General Treatment Guidance")
    st.write(adv["general_treatment"])
    st.caption("⚠️ Always follow the pesticide label and local agricultural extension recommendations. "
               "This system does not provide specific dosage instructions.")
    styling.card_end()

    styling.card_start()
    st.markdown("#### 👁️ Monitoring Advice")
    for m in adv["monitoring"]:
        st.write(f"- {m}")
    styling.card_end()

st.markdown("---")
with st.expander("📋 View all supported conditions"):
    import pandas as pd
    st.dataframe(pd.DataFrame(conditions), use_container_width=True, hide_index=True)
