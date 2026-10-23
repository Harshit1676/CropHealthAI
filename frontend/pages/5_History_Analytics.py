"""CropHealth AI — History & Analytics page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components import api_client, styling

st.set_page_config(page_title="History & Analytics | CropHealth AI", page_icon="📊", layout="wide")
styling.inject_css()
styling.page_header("📊 History & Analytics", "Explore prediction history and aggregate trends across all scans.")

backend_ok = api_client.backend_status_banner()
if not backend_ok:
    st.stop()

ok, stats = api_client.get_statistics()
if not ok:
    st.error(stats)
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    styling.card_start(); st.metric("Total Predictions", stats["total_predictions"]); styling.card_end()
with col2:
    styling.card_start(); st.metric("Diseases Detected", sum(stats["disease_distribution"].values())); styling.card_end()
with col3:
    styling.card_start(); st.metric("Pests Detected", sum(stats["pest_distribution"].values())); styling.card_end()
with col4:
    styling.card_start(); st.metric("Healthy Scans", sum(stats["healthy_distribution"].values())); styling.card_end()

if stats["total_predictions"] == 0:
    st.info("No predictions recorded yet. Use the Disease Detection or Pest Detection pages to get started.")
    st.stop()

st.markdown("### Distribution Overview")
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    styling.card_start()
    st.markdown("#### Disease Distribution")
    if stats["disease_distribution"]:
        df = pd.DataFrame(list(stats["disease_distribution"].items()), columns=["Disease", "Count"])
        fig = px.bar(df, x="Disease", y="Count", color="Count", color_continuous_scale="Reds")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No disease detections recorded yet.")
    styling.card_end()

with row1_col2:
    styling.card_start()
    st.markdown("#### Pest Distribution")
    if stats["pest_distribution"]:
        df = pd.DataFrame(list(stats["pest_distribution"].items()), columns=["Pest", "Count"])
        fig = px.bar(df, x="Pest", y="Count", color="Count", color_continuous_scale="Oranges")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No pest detections recorded yet.")
    styling.card_end()

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    styling.card_start()
    st.markdown("#### Risk Level Distribution")
    if stats["risk_distribution"]:
        df = pd.DataFrame(list(stats["risk_distribution"].items()), columns=["Risk Level", "Count"])
        color_map = {"Low": "#22C55E", "Moderate": "#F59E0B", "High": "#F97316", "Severe": "#DC2626"}
        fig = px.pie(df, names="Risk Level", values="Count",
                     color="Risk Level", color_discrete_map=color_map, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk scores recorded yet.")
    styling.card_end()

with row2_col2:
    styling.card_start()
    st.markdown("#### Crop Distribution")
    if stats["crop_distribution"]:
        df = pd.DataFrame(list(stats["crop_distribution"].items()), columns=["Crop", "Count"])
        fig = px.pie(df, names="Crop", values="Count", hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No crop data recorded yet.")
    styling.card_end()

styling.card_start()
st.markdown("#### Predictions Over Time")
if stats["predictions_over_time"]:
    df = pd.DataFrame(stats["predictions_over_time"])
    fig = px.line(df, x="date", y="count", markers=True)
    fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Predictions")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data yet to show a trend.")
styling.card_end()

st.markdown("### Recent Prediction Records")
pred_ok, predictions = api_client.get_predictions(limit=200)
if pred_ok and predictions:
    df = pd.DataFrame(predictions)
    display_cols = ["timestamp", "crop", "prediction", "confidence", "risk_score",
                     "risk_level", "temperature", "humidity", "rainfall", "location"]
    df = df[[c for c in display_cols if c in df.columns]]
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download history as CSV", csv, "crophealth_predictions.csv", "text/csv")
elif pred_ok:
    st.info("No prediction records found.")
else:
    st.error(predictions)
