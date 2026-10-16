"""
style.py
Clinical design stylesheet and layout styling for the Type 2 Diabetes Digital Twin.

Provides a calm, restrained light clinical theme:
- Warm off-white page (#FBFBFA), white panels, dark slate text (#2D3748)
- Muted primary teal accent (#1F5F5B)
- Clinical risk tier colors: Green (#3E7B5A), Amber (#B7791F), Red (#A23B3B)
- Serif headings (Georgia), clean sans-serif body
- Flat presentation with thin 1px dividers, zero shadows, no neon, no emojis
- Hidden Streamlit chrome (deploy button, hamburger menu, header, footer)
"""

from typing import Tuple
import streamlit as st

# Muted clinical color palette
COLOR_TEAL = "#1F5F5B"
COLOR_PAGE_BG = "#FBFBFA"
COLOR_PANEL_BG = "#FFFFFF"
COLOR_TEXT = "#2D3748"
COLOR_MUTED_TEXT = "#64748B"
COLOR_BORDER = "#E2E8F0"

COLOR_RISK_LOW = "#3E7B5A"       # < 0.20
COLOR_RISK_MODERATE = "#B7791F"  # 0.20 - 0.40
COLOR_RISK_HIGH = "#A23B3B"      # >= 0.40


def get_clinical_risk_tier(risk: float) -> Tuple[str, str]:
    """
    Returns (label, hex_color) in clean clinical wording without emojis.
    - Green: risk < 0.20
    - Amber: 0.20 <= risk < 0.40
    - Red: risk >= 0.40
    """
    if risk < 0.20:
        return "Low risk", COLOR_RISK_LOW
    elif risk < 0.40:
        return "Moderate risk", COLOR_RISK_MODERATE
    else:
        return "High risk", COLOR_RISK_HIGH


def inject_clinical_css():
    """Injects the clinical CSS overrides into the Streamlit app."""
    custom_css = f"""
    <style>
    /* 1. Global typography and page background */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_PAGE_BG} !important;
        color: {COLOR_TEXT} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}

    /* 2. Serif headings in sentence case */
    h1, h2, h3, h4, h5, h6 {{
        font-family: Georgia, "Times New Roman", Cambria, serif !important;
        font-weight: 500 !important;
        color: {COLOR_TEXT} !important;
        letter-spacing: -0.01em;
        margin-top: 0.4em;
        margin-bottom: 0.3em;
    }}

    h1 {{ font-size: 1.85rem !important; }}
    h2 {{ font-size: 1.45rem !important; }}
    h3 {{ font-size: 1.20rem !important; }}
    h4 {{ font-size: 1.05rem !important; }}

    /* 3. Hide Streamlit chrome: Deploy button, header decorations, hamburger menu, footer */
    #MainMenu {{ visibility: hidden !important; display: none !important; }}
    header {{ visibility: hidden !important; display: none !important; }}
    footer {{ visibility: hidden !important; display: none !important; }}
    [data-testid="stToolbar"] {{ visibility: hidden !important; display: none !important; }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    button[title="View app in GitHub"] {{ display: none !important; }}

    /* 4. Remove all card drop shadows, neon borders, and glowing states */
    * {{
        box-shadow: none !important;
        text-shadow: none !important;
    }}

    /* 5. Clean, quiet button styling */
    .stButton > button {{
        background-color: #FFFFFF !important;
        color: {COLOR_TEAL} !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 3px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.75rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: #F8FAFC !important;
        border-color: {COLOR_TEAL} !important;
        color: {COLOR_TEAL} !important;
    }}
    .stButton > button:active {{
        background-color: #EEF2F6 !important;
    }}

    /* 6. Clean sliders */
    [data-testid="stSlider"] > div {{
        padding-top: 0.2rem;
        padding-bottom: 0.5rem;
    }}

    /* 7. Thin 1px dividers */
    hr {{
        border: 0 !important;
        border-top: 1px solid {COLOR_BORDER} !important;
        margin: 1.2rem 0 !important;
    }}

    /* 8. Hospital worklist table layout */
    .clinical-worklist-header {{
        display: grid;
        grid-template-columns: 2.2fr 1.0fr 1.3fr 1.4fr 1.6fr 1.8fr 1.1fr;
        padding: 0.55rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {COLOR_MUTED_TEXT};
        border-bottom: 1px solid #CBD5E1;
        background-color: #F4F4F1;
        border-radius: 3px 3px 0 0;
    }}

    .clinical-worklist-row {{
        display: grid;
        grid-template-columns: 2.2fr 1.0fr 1.3fr 1.4fr 1.6fr 1.8fr 1.1fr;
        padding: 0.65rem 0.75rem;
        align-items: center;
        border-bottom: 1px solid {COLOR_BORDER};
        background-color: #FFFFFF;
        font-size: 0.88rem;
    }}

    .clinical-worklist-row:hover {{
        background-color: #F8F9FA;
    }}

    /* 9. Progress bar for model risk in worklist */
    .risk-bar-container {{
        width: 100%;
        background-color: #EDF2F7;
        height: 6px;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 3px;
    }}
    .risk-bar-fill {{
        height: 100%;
        border-radius: 2px;
    }}

    /* 10. Quiet clinical meta readout line */
    .clinical-metalist {{
        font-size: 0.88rem;
        color: {COLOR_MUTED_TEXT};
        padding: 0.35rem 0;
        margin-bottom: 1.0rem;
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    .clinical-metalist span {{
        color: {COLOR_TEXT};
        font-weight: 500;
        margin-right: 0.85rem;
    }}

    /* 11. Model score presentation box */
    .clinical-score-panel {{
        background-color: #FFFFFF;
        border: 1px solid {COLOR_BORDER};
        border-left: 4px solid {COLOR_TEAL};
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.2rem;
        border-radius: 2px;
    }}

    /* 12. Quiet disclaimer text */
    .clinical-disclaimer {{
        font-size: 0.80rem;
        color: {COLOR_MUTED_TEXT};
        padding: 1.0rem 0 0.5rem 0;
        border-top: 1px solid {COLOR_BORDER};
        line-height: 1.4;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
