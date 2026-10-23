"""Shared CSS + small UI helpers used across all Streamlit pages."""
from __future__ import annotations

import streamlit as st

PRIMARY_GREEN = "#1B7340"
ACCENT_GREEN = "#2FA866"
BG_SOFT = "#F4F8F4"
WARN = "#D97706"
DANGER = "#DC2626"

CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: {BG_SOFT};
    }}
    .ch-card {{
        background: white;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border: 1px solid #E5EAE5;
        margin-bottom: 1rem;
    }}
    .ch-header {{
        background: linear-gradient(135deg, {PRIMARY_GREEN} 0%, {ACCENT_GREEN} 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }}
    .ch-header h1 {{
        color: white !important;
        margin-bottom: 0.2rem;
    }}
    .ch-header p {{
        color: #EAF5EC;
        margin: 0;
        font-size: 1.05rem;
    }}
    .ch-badge {{
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .ch-badge-low {{ background:#DCFCE7; color:#166534; }}
    .ch-badge-moderate {{ background:#FEF3C7; color:#92400E; }}
    .ch-badge-high {{ background:#FFEDD5; color:#C2410C; }}
    .ch-badge-severe {{ background:#FEE2E2; color:#991B1B; }}
    .ch-badge-healthy {{ background:#DCFCE7; color:#166534; }}
    .ch-badge-disease {{ background:#FFEDD5; color:#C2410C; }}
    .ch-badge-pest {{ background:#FEF3C7; color:#92400E; }}
    section[data-testid="stSidebar"] {{
        background-color: #103A22;
    }}
    section[data-testid="stSidebar"] * {{
        color: #F4F8F4 !important;
    }}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""<div class="ch-header"><h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def risk_badge_html(level: str) -> str:
    css_class = {
        "Low": "ch-badge-low", "Moderate": "ch-badge-moderate",
        "High": "ch-badge-high", "Severe": "ch-badge-severe",
    }.get(level, "ch-badge-moderate")
    return f'<span class="ch-badge {css_class}">{level} Risk</span>'


def category_badge_html(category: str) -> str:
    """Generate a category badge safely, even when category is missing."""
    category = str(category or "unknown").strip()
    css_class = category.lower().replace(" ", "-")
    return f'<span class="ch-badge {css_class}">{category.capitalize()}</span>'

def card_start() -> None:
    st.markdown('<div class="ch-card">', unsafe_allow_html=True)


def card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
