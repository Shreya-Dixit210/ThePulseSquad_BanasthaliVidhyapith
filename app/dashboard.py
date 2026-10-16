"""
dashboard.py
Clinical decision-support dashboard for the Type 2 Diabetes Digital Twin.
Hackathon: Happiest Health Digital Twin Challenge 2026.

Run with:
    streamlit run app/dashboard.py
"""

from typing import Dict, Any, List
import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Ensure app directory is on path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from demo_data import get_all_patients
from model_adapter import (
    compute_features,
    predict_risk,
    get_feature_contributions,
)


from model_adapter import validation_rate

from style import (
    inject_clinical_css,
    get_clinical_risk_tier,
    COLOR_TEAL,
    COLOR_RISK_LOW,
    COLOR_RISK_MODERATE,
    COLOR_RISK_HIGH,
    COLOR_BORDER,
    COLOR_MUTED_TEXT,
)

# Page configuration
st.set_page_config(
    page_title="Type 2 Diabetes Digital Twin",
    layout="wide",
)

# Inject clinical CSS
inject_clinical_css()


def render_footer():
    """Renders the mandatory regulatory & decision-support disclaimer on every page."""
    st.markdown(
        '<div class="clinical-disclaimer">'
        "Decision-support prototype built on synthetic and open data. Not medical advice. "
        "Final decisions rest with the doctor."
        "</div>",
        unsafe_allow_html=True,
    )


def build_clinical_cgm_chart(history_df: pd.DataFrame, patient_name: str) -> go.Figure:
    """
    Creates a 24-hour clinical CGM line chart:
    - White background, thin teal line
    - Hourly horizontal tick labels
    - Fully visible Y-axis title
    - Light band above 180 mg/dL
    - Vertical 'now' line via add_shape and add_annotation (avoids categorical vline error)
    - Meal markers labelled like 'Lunch 70 g' that avoid hover tooltip clipping
    """
    fig = go.Figure()

    # 1. Light shaded band above 180 mg/dL (hyperglycemia region)
    fig.add_hrect(
        y0=180,
        y1=260,
        fillcolor="rgba(162, 59, 59, 0.05)",
        line_width=0,
    )

    # 2. 180 mg/dL threshold line
    fig.add_hline(
        y=180,
        line_dash="dash",
        line_color=COLOR_RISK_HIGH,
        line_width=1.0,
        annotation_text="180 mg/dL threshold",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_RISK_HIGH, size=10, family="sans-serif"),
    )

    # 3. Glucose time-series trace (thin teal line)
    fig.add_trace(
        go.Scatter(
            x=history_df["time_str"],
            y=history_df["glucose"],
            mode="lines",
            name="Glucose",
            line=dict(color=COLOR_TEAL, width=1.75),
            hovertemplate="Time: %{x}<br>Glucose: %{y:.1f} mg/dL<extra></extra>",
        )
    )

    # 4. Vertical "Now" line at the latest observation (uses shape + annotation to support categorical x-axis)
    now_time = history_df["time_str"].iloc[-1]
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=now_time,
        x1=now_time,
        y0=0,
        y1=1,
        line=dict(color=COLOR_TEAL, width=1.2, dash="dot"),
    )
    fig.add_annotation(
        xref="x",
        yref="paper",
        x=now_time,
        y=1,
        text="Now",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(color=COLOR_TEAL, size=10, family="sans-serif"),
    )

    # 5. Logged meal markers with clean clinical labels (e.g., 'Lunch 70 g')
    meal_rows = history_df[history_df["is_meal"]]
    if not meal_rows.empty:
        cleaned_labels = []
        for name in meal_rows["meal_name"]:
            c_name = name.replace(" (", " ").replace("g)", " g").replace(")", "")
            cleaned_labels.append(c_name)

        fig.add_trace(
            go.Scatter(
                x=meal_rows["time_str"],
                y=meal_rows["glucose"],
                mode="markers+text",
                name="Meals",
                marker=dict(symbol="circle", size=7, color=COLOR_RISK_MODERATE),
                text=cleaned_labels,
                textposition="top center",
                textfont=dict(size=10, color="#4A5568", family="sans-serif"),
                cliponaxis=False,
                hovertemplate="%{text}<br>Time: %{x}<br>Glucose: %{y:.1f} mg/dL<extra></extra>",
            )
        )

    # Extract hourly horizontal tick marks (times ending in :00)
    hourly_ticks = [t for t in history_df["time_str"] if t.endswith(":00")]

    fig.update_layout(
        title=dict(
            text=f"24-hour continuous glucose profile — {patient_name}",
            font=dict(family="Georgia, serif", size=15, color="#2D3748"),
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(text="Time of day", font=dict(size=11, color="#64748B")),
            tickmode="array",
            tickvals=hourly_ticks,
            ticktext=hourly_ticks,
            tickangle=0,
            showgrid=True,
            gridcolor="#F0F1EE",
            linecolor=COLOR_BORDER,
            tickfont=dict(size=10, color="#64748B"),
        ),
        yaxis=dict(
            title=dict(text="Glucose (mg/dL)", font=dict(size=11, color="#2D3748")),
            range=[70, 240],
            showgrid=True,
            gridcolor="#F0F1EE",
            linecolor=COLOR_BORDER,
            tickfont=dict(size=10, color="#64748B"),
        ),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=55, r=35, t=45, b=35),
        height=320,
    )
    return fig

def build_clinical_why_chart(contributions_df: pd.DataFrame) -> go.Figure:
    """Display signed SHAP contributions on the model's raw output scale."""

    label_map = {
        "Current Glucose": "Now | Current glucose",
        "Current Glucose (glu_now)": "Now | Current glucose",
        "Recent Meal Carbs (meal_carbs)": "Now | Meal carbohydrates",
        "Meal Carbohydrates": "Now | Meal carbohydrates",
        "30-min Trend (glu_change_30)": "Now | 30-min glucose trend",
        "30-min Glucose Trend": "Now | 30-min glucose trend",
        "Time Since Meal (mins_since_meal)": "Now | Time since meal",
        "Time Since Meal": "Now | Time since meal",
        "Baseline A1c": "Baseline | HbA1c",
        "Body Mass Index (BMI)": "Baseline | BMI",
        "Fasting Glucose": "Baseline | Fasting glucose",
        "Age": "Baseline | Age",
        "Gender": "Baseline | Gender",
        "HDL": "Baseline | HDL",
        "Insulin": "Baseline | Insulin",
        "Triglycerides": "Baseline | Triglycerides",
    }

    if contributions_df.empty or not {"Feature", "Impact"}.issubset(
        contributions_df.columns
    ):
        fig = go.Figure()
        fig.add_annotation(
            text="Feature explanations are currently unavailable.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13, color="#64748B"),
        )
        fig.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=260,
            margin=dict(l=15, r=15, t=35, b=25),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    chart_df = contributions_df.copy()
    chart_df["Impact"] = pd.to_numeric(chart_df["Impact"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Impact"])
    chart_df["Display"] = chart_df["Feature"].map(
        lambda name: label_map.get(name, name)
    )
    chart_df["Magnitude"] = chart_df["Impact"].abs()
    chart_df = chart_df.sort_values("Magnitude", ascending=True).tail(10)

    bar_colors = [
        COLOR_RISK_HIGH if value > 0 else COLOR_RISK_LOW
        for value in chart_df["Impact"]
    ]

    fig = go.Figure(
        go.Bar(
            x=chart_df["Impact"],
            y=chart_df["Display"],
            orientation="h",
            marker=dict(color=bar_colors),
            customdata=chart_df["Magnitude"],
            hovertemplate=(
                "%{y}<br>Signed SHAP contribution: %{x:+.3f}"
                "<br>Magnitude: %{customdata:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Top contributing factors",
            font=dict(family="Georgia, serif", size=14, color="#2D3748"),
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(
                text="SHAP contribution (raw model-output scale)",
                font=dict(size=10, color="#64748B"),
            ),
            showgrid=True,
            gridcolor="#F0F1EE",
            linecolor=COLOR_BORDER,
            zeroline=True,
            zerolinecolor="#64748B",
            tickfont=dict(size=10, color="#64748B"),
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="#2D3748"),
            linecolor=COLOR_BORDER,
        ),
        margin=dict(l=10, r=20, t=35, b=40),
        height=300,
    )
    return fig



def show_patient_list(patients: List[Dict[str, Any]]):
    """
    Page 1: Compact hospital-worklist style table.
    Columns: Patient, Age/Sex, Glucose now, 30-min change with arrow, Last meal,
             Model score with a small bar, Open link.
    Sorted by score. One summary line instead of four big tiles.
    """
    st.markdown("<h1>Simulated patients</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size: 0.92rem; color: #2D3748; margin-bottom: 0.5rem;">'
        "Simulated patients. Model risk scores for reaching 180 mg/dL within the next 2 hours."
        "</div>",
        unsafe_allow_html=True,
    )

    # Compute risk for all patients
    patient_rows = []
    for p in patients:
        feat_df = compute_features(
            history=p["history"],
            static=p["static"],
            mins_since_meal=p["mins_since_meal"],
            meal_carbs=p["meal_carbs"],
        )
        risk = predict_risk(feat_df)
        tier_label, tier_color = get_clinical_risk_tier(risk)

        patient_rows.append({
            "id": p["id"],
            "name": p["name"],
            "risk": risk,
            "tier_label": tier_label,
            "tier_color": tier_color,
            "age": int(p["static"]["Age"]),
            "sex": "M" if p["static"]["Gender"] == 1 else "F",
            "glu_now": p["glu_now"],
            "glu_change_30": p["glu_change_30"],
            "mins_since_meal": int(p["mins_since_meal"]),
            "meal_carbs": int(p["meal_carbs"]),
        })

    # Sort cohort by risk descending
    patient_rows.sort(key=lambda x: x["risk"], reverse=True)

    # Triage summary counts
    high_count = sum(1 for p in patient_rows if p["risk"] >= 0.40)
    mod_count = sum(1 for p in patient_rows if 0.20 <= p["risk"] < 0.40)
    low_count = sum(1 for p in patient_rows if p["risk"] < 0.20)

    # One summary line instead of four big tiles
    st.markdown(
        f'<div style="font-size: 0.85rem; color: {COLOR_MUTED_TEXT}; margin-bottom: 1.1rem; padding-bottom: 0.6rem; border-bottom: 1px solid {COLOR_BORDER};">'
        f"Simulated patients: <strong>{len(patient_rows)} total</strong> &nbsp;·&nbsp; "
        f'<span style="color: {COLOR_RISK_HIGH}; font-weight: 500;">{high_count} high risk</span> &nbsp;·&nbsp; '
        f'<span style="color: {COLOR_RISK_MODERATE}; font-weight: 500;">{mod_count} moderate</span> &nbsp;·&nbsp; '
        f'<span style="color: {COLOR_RISK_LOW}; font-weight: 500;">{low_count} low</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Compact hospital-worklist table header
    header_cols = st.columns([2.2, 1.0, 1.3, 1.4, 1.6, 2.0, 0.9])
    header_cols[0].caption("PATIENT")
    header_cols[1].caption("AGE / SEX")
    header_cols[2].caption("GLUCOSE NOW")
    header_cols[3].caption("30-MIN CHANGE")
    header_cols[4].caption("LAST MEAL")
    header_cols[5].caption("MODEL RISK SCORE")
    header_cols[6].caption("")

    st.markdown(
        '<div style="border-bottom: 1px solid #CBD5E1; margin: -6px 0 6px 0;"></div>',
        unsafe_allow_html=True,
    )

    # Patient worklist rows
    for p in patient_rows:
        cols = st.columns([2.2, 1.0, 1.3, 1.4, 1.6, 2.0, 0.9])
        with cols[0]:
            st.markdown(
                f"<div style='font-weight: 500; font-size: 0.90rem;'>{p['name']}</div>"
                f"<div style='font-size: 0.78rem; color: {COLOR_MUTED_TEXT};'>{p['id']}</div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.write(f"{p['age']} / {p['sex']}")
        with cols[2]:
            st.write(f"{p['glu_now']:.1f} mg/dL")
        with cols[3]:
            # Arrow indicator
            arrow = "↑" if p["glu_change_30"] > 0 else ("↓" if p["glu_change_30"] < 0 else "→")
            st.write(f"{arrow} {p['glu_change_30']:+.1f} mg/dL")
        with cols[4]:
            st.write(f"{p['meal_carbs']} g, {p['mins_since_meal']} min ago")
        with cols[5]:
            bar_pct = min(max(p["risk"] * 100, 3), 100)
            st.markdown(
                f"<div style='font-weight: 600; font-size: 0.88rem; color: {p['tier_color']};'>"
                f"Model score: {p['risk']:.2f} <span style='font-weight: 400; font-size: 0.78rem;'>({p['tier_label']})</span>"
                f"</div>"
                f"<div class='risk-bar-container'>"
                f"<div class='risk-bar-fill' style='width: {bar_pct:.1f}%; background-color: {p['tier_color']};'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with cols[6]:
            if st.button("Open", key=f"open_{p['id']}", width="stretch"):
                st.session_state["selected_patient_id"] = p["id"]
                st.rerun()

        st.markdown(
            f'<div style="border-bottom: 1px solid {COLOR_BORDER}; margin: 2px 0 6px 0;"></div>',
            unsafe_allow_html=True,
        )

    render_footer()


def show_patient_detail(patient: Dict[str, Any]):
    """
    Page 2: Clinical patient detail view.
    - Header on one line with quiet line of labs, no boxed profile card
    - Model risk score with validation sentence
    - 24-hour CGM chart
    - Why and What-if side by side, separated by a thin 1px divider
    """
    # Back navigation link
    if st.button("← Patients"):
        st.session_state["selected_patient_id"] = None
        st.rerun()

    # Feature preparation and baseline prediction
    feat_df = compute_features(
        history=patient["history"],
        static=patient["static"],
        mins_since_meal=patient["mins_since_meal"],
        meal_carbs=patient["meal_carbs"],
    )
    baseline_risk = predict_risk(feat_df)
    threshold_reached = patient["glu_now"] >= 180
    tier_label, tier_color = get_clinical_risk_tier(baseline_risk)

    # 1. Header on one line
    st.markdown(
        f"<h2 style='margin-bottom: 0.2rem;'>"
        f"{patient['name']} <span style='color: {COLOR_MUTED_TEXT}; font-size: 1.10rem;'>({patient['id']})</span>"
        f"</h2>",
        unsafe_allow_html=True,
    )

    # 2. Quiet line of labs (no boxed profile card)
    gender_str = "M" if patient["static"]["Gender"] == 1 else "F"
    st.markdown(
        f'<div class="clinical-metalist">'
        f"Age: <span>{int(patient['static']['Age'])}</span> · "
        f"Sex: <span>{gender_str}</span> · "
        f"BMI: <span>{patient['static']['BMI']:.1f}</span> · "
        f"HbA1c: <span>{patient['static']['A1c']:.1f}%</span> · "
        f"Fasting glucose: <span>{int(patient['static']['fasting_glu'])} mg/dL</span> · "
        f"Fasting insulin: <span>{int(patient['static']['Insulin'])} µIU/mL</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 3. Model risk score section with validation sentence

    if threshold_reached:
        score_heading = "Current glucose status"
        score_display = "Threshold already reached"
        val_sentence = (
            "Current glucose is at or above 180 mg/dL. "
            "This is a threshold status, not a prediction of a future event."
        )
    else:
        score_heading = "Model risk score"
        score_display = f"{baseline_risk:.2f} — {tier_label}"
        val_result = validation_rate(baseline_risk)

        if val_result is None:
            val_sentence = (
                "Calibration data is unavailable; this score should not be "
                "interpreted as a calibrated probability."
            )
        else:
            if isinstance(val_result, dict):
                rate = val_result["rate"]
                rows = val_result["rows"]
            else:
                rate, rows = val_result

            val_sentence = (
                f"In the saved calibration data, about {rate:.0%} of readings "
                f"in this score range had the target event ({rows:,} readings; "
                "neighbouring five-minute readings may be correlated)."
            )


    st.html(
        f"""
        <div class="clinical-score-panel" style="border-left-color: {tier_color};">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: {COLOR_MUTED_TEXT}; font-weight: 600;">
                {score_heading}
            </div>
            <div style="font-size: 1.55rem; font-weight: 600; color: {tier_color}; margin: 2px 0;">
                {score_display}
            </div>
            <div style="font-size: 0.90rem; color: #2D3748; margin-top: 4px;">
                A relative score from the model, not a calibrated probability.
            </div>
            <div style="font-size: 0.82rem; color: {COLOR_MUTED_TEXT}; margin-top: 4px; line-height: 1.4;">
                {val_sentence}
            </div>
        </div>
        """
    )





    # 4. 24-Hour Clinical Glucose Chart
    cgm_fig = build_clinical_cgm_chart(patient["history"], patient["name"])
    st.plotly_chart(cgm_fig, width="stretch")

    # Thin divider before Why and What-if panels
    st.markdown('<hr>', unsafe_allow_html=True)

    # 5. Why and What-if side by side, separated by a thin vertical divider
    col_why, col_div, col_whatif = st.columns([1.0, 0.05, 1.0])

    with col_why:
        st.markdown("<h3>Why: top contributing factors</h3>", unsafe_allow_html=True)
        contributions_df = get_feature_contributions(feat_df)
        why_fig = build_clinical_why_chart(contributions_df)
        st.plotly_chart(why_fig, width="stretch")
        st.caption(
    "Positive and negative bars show the direction of each feature's contribution "
    "to the model's raw output. Magnitude is not a percentage or a causal effect."
)

    with col_div:
        # Thin 1px vertical line
        st.markdown(
            f'<div style="border-left: 1px solid {COLOR_BORDER}; min-height: 400px; height: 100%; margin: 0 auto; width: 1px;"></div>',
            unsafe_allow_html=True,
        )

    with col_whatif:
        st.markdown("<h3>What-if simulation</h3>", unsafe_allow_html=True)
        st.caption("Simulate carbohydrate intake and postprandial timing adjustments.")

        # Sliders (strictly meal carbs & minutes since meal - no activity or heart rate)
        sim_carbs = st.slider(
            "Carbohydrates in last meal (g)",
            min_value=0,
            max_value=120,
            value=int(patient["meal_carbs"]),
            step=5,
            help="Simulate an adjusted carbohydrate intake for the last meal.",
        )

        sim_mins = st.slider(
            "Minutes elapsed since last meal",
            min_value=15,
            max_value=240,
            value=int(patient["mins_since_meal"]),
            step=5,
            help="Simulate the passage of time or earlier meal scheduling.",
        )

        # Re-run predict_risk under simulation
        sim_feat_df = compute_features(
            history=patient["history"],
            static=patient["static"],
            mins_since_meal=float(sim_mins),
            meal_carbs=float(sim_carbs),
        )
        sim_risk = predict_risk(sim_feat_df)
        sim_tier_label, sim_tier_color = get_clinical_risk_tier(sim_risk)

        # Comparison calculation in 2-decimal scores (never percentage points)
        delta_risk = sim_risk - baseline_risk
        abs_delta = abs(delta_risk)

        st.markdown(
            f'<div style="margin-top: 12px; margin-bottom: 8px; font-size: 0.90rem; font-weight: 500;">'
            f"Baseline score: <code>{baseline_risk:.3f}</code> &nbsp;→&nbsp; "
            f"Simulated score: <code>{sim_risk:.3f}</code> &nbsp;(change: <code>{delta_risk:+.3f}</code>)"

            f"</div>",
            unsafe_allow_html=True,
        )

        # Exact sentences as specified
        if delta_risk < -0.005:
            st.info(f"According to the model, these meal inputs lower the risk score by {abs_delta:.3f}.")
        elif delta_risk > 0.005:
            st.warning(f"According to the model, these meal inputs raise the risk score by {abs_delta:.3f}.")
        else:
            st.info("According to the model, the risk score is essentially unchanged.")

        st.caption(
            "This shows how sensitive the model is to the meal inputs. The patient's glucose history "
            "is kept fixed, so it is not a forecast of what would really happen."
        )

        st.caption(
            f'Simulated classification: <span style="color: {sim_tier_color}; font-weight: 600;">{sim_tier_label}</span> '
            f'(score: {sim_risk:.2f})',
            unsafe_allow_html=True,
        )

    render_footer()


def main():
    # Load synthetic cohort
    patients = get_all_patients()
    patient_map = {p["id"]: p for p in patients}

    # Initialize session navigation state
    if "selected_patient_id" not in st.session_state:
        st.session_state["selected_patient_id"] = None

    selected_id = st.session_state["selected_patient_id"]

    if selected_id and selected_id in patient_map:
        show_patient_detail(patient_map[selected_id])
    else:
        show_patient_list(patients)


if __name__ == "__main__":
    main()
