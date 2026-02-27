"""Centralized Streamlit theme helpers for FLUX.2 UI."""

from __future__ import annotations

import streamlit as st


BLACK_UI_PALETTE: dict[str, str] = {
    "background": "#000000",
    "surface": "#111111",
    "surface_alt": "#1A1A1A",
    "code_surface": "#0A0A0A",
    "border": "#333333",
    "text_primary": "#FFFFFF",
}


def build_theme_css() -> str:
    """Return the global dark theme CSS for Streamlit."""
    p = BLACK_UI_PALETTE
    return f"""
    <style>
        :root {{ color-scheme: dark; }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        [data-testid="stSidebar"], [data-testid="stSidebarContent"],
        [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            background: {p['background']} !important;
        }}
        body, .stApp, p, li, label, span, div, small, .stMarkdown, .stCaption,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        [data-testid="stExpander"] {{
            color: {p['text_primary']} !important;
        }}
        input, textarea, select,
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
            background: {p['background']} !important;
            color: {p['text_primary']} !important;
            border-color: {p['border']} !important;
        }}
        [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background: {p['surface']} !important;
            color: {p['text_primary']} !important;
        }}
        [data-testid="stButton"] button {{
            background: {p['surface']} !important;
            color: {p['text_primary']} !important;
            border: 1px solid {p['border']} !important;
        }}
        [data-testid="stButton"] button:hover {{
            background: {p['surface_alt']} !important;
        }}
        [data-testid="stButton"] button:focus-visible,
        .stTextInput input:focus-visible,
        .stTextArea textarea:focus-visible,
        .stSelectbox div[data-baseweb="select"]:focus-within,
        [role="tab"]:focus-visible,
        [data-testid="stCheckbox"] input:focus-visible,
        [data-testid="stRadio"] input:focus-visible,
        details summary:focus-visible {{
            outline: 2px solid {p['text_primary']} !important;
            outline-offset: 2px;
        }}
        .stCodeBlock, pre, code {{
            background: {p['code_surface']} !important;
            color: {p['text_primary']} !important;
        }}
    </style>
    """


def apply_theme() -> None:
    """Apply the centralized global Streamlit theme."""
    st.markdown(build_theme_css(), unsafe_allow_html=True)
