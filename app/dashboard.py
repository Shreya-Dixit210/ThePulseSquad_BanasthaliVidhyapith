"""
dashboard.py
Streamlit Doctor Dashboard for the Type 2 Diabetes Digital Twin Prototype.
Hackathon: Happiest Health Digital Twin Challenge 2026.

Run with:
    streamlit run app/dashboard.py
"""

from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import os
import sys

# Ensure app/ directory is on sys.path regardless of execution working directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from demo_data import get_all_patients
from model_adapter import (
    compute_features,
    predict_risk,
    get_risk_badge,
    get_feature_contributions,
    validation_rate,
)

# Page configuration
st.set_page_config(
    page_title="T2D Digital Twin - Doctor Dashboard",
    page_icon="🩺",
    layout="wide",
)


def render_footer():
    """Renders the mandatory regulatory & decision-support disclaimer on all views."""
    st.markdown("---")
    st.caption(
        "⚖️ **Disclaimer:** Decision-support prototype built on synthetic and open data. "
        "Not medical advice. Final decisions rest with the doctor."
    )


def build_cgm_chart(history_df: pd.DataFrame, patient_name: str) -> go.Figure:
    """Creates a 24-hour interactive CGM line chart with a 180 mg/dL threshold and meal pins."""
    fig = go.Figure()

    # 1. Glucose trace
    fig.add_trace(
        go.Scatter(
            x=history_df["time_str"],
            y=history_df["glucose"],
            mode="lines",
            name="Glucose (mg/dL)",
            line=dict(color="#1f77b4", width=2.5),
            hovertemplate="Time: %{x}<br>Glucose: %{y:.1f} mg/dL<extra></extra>",
        )
    )

    # 2. Hyperglycemia risk threshold line (180 mg/dL)
    fig.add_hline(
        y=180,
        line_dash="dash",
        line_color="#dc3545",
        line_width=2,
        annotation_text="High Glucose Threshold (180 mg/dL)",
        annotation_position="top right",
        annotation_font=dict(color="#dc3545", size=11),
    )

    # 3. Meal markers
    meal_rows = history_df[history_df["is_meal"]]
    if not meal_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=meal_rows["time_str"],
                y=meal_rows["glucose"],
                mode="markers+text",
                name="Logged Meals",
                marker=dict(symbol="triangle-up", size=14, color="#fd7e14"),
                text=meal_rows["meal_name"],
                textposition="top center",
                hovertemplate="Meal: %{text}<br>Time: %{x}<br>Glucose at meal: %{y:.1f} mg/dL<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"24-Hour Continuous Glucose Profile — {patient_name}",
        xaxis_title="Time of Day (5-min intervals)",
        yaxis_title="Glucose (mg/dL)",
        yaxis=dict(range=[60, 260]),
        hovermode="x unified",
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    return fig


def build_why_chart(contributions_df: pd.DataFrame) -> go.Figure:
    """Builds a horizontal bar chart displaying top factors driving the current prediction."""
    colors = ["#dc3545" if val > 0 else "#28a745" for val in contributions_df["Impact"]]

    fig = go.Figure(
        go.Bar(
            x=contributions_df["Impact"],
            y=contributions_df["Feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="Feature: %{y}<br>Impact: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="What drives this score (from the model)",
        xaxis_title="Push on the risk score (right = higher risk, left = lower risk)",
        yaxis=dict(automargin=True),
        margin=dict(l=10, r=20, t=40, b=20),
        height=320,
    )
    
    return fig


def show_patient_list(patients: List[Dict[str, Any]]):
    """Page 1: Patient Cohort List with risk sorting and triage badges."""
    st.title("🩺 Type 2 Diabetes: Digital Twin Risk Monitor")
    st.markdown(
        "**Physician Decision-Support Overview** — Simulated patients. Model risk scores "
        "for reaching 180 mg/dL within the next 2 hours."
        
    )

    # Calculate current risk for all patients
    patient_rows = []
    for p in patients:
        feat_df = compute_features(
            history=p["history"],
            static=p["static"],
            mins_since_meal=p["mins_since_meal"],
            meal_carbs=p["meal_carbs"],
        )
        risk = predict_risk(feat_df)
        label, hex_color, badge = get_risk_badge(risk)

        patient_rows.append({
            "id": p["id"],
            "name": p["name"],
            "risk": risk,
            "badge": badge,
            "hex_color": hex_color,
            "label": label,
            "age": int(p["static"]["Age"]),
            "gender": "Male" if p["static"]["Gender"] == 1 else "Female",
            "bmi": p["static"]["BMI"],
            "a1c": p["static"]["A1c"],
            "fasting_glu": p["static"]["fasting_glu"],
            "glu_now": p["glu_now"],
            "glu_change_30": p["glu_change_30"],
            "mins_since_meal": int(p["mins_since_meal"]),
            "meal_carbs": p["meal_carbs"],
        })

    # Sort cohort by risk descending (highest risk first)
    patient_rows.sort(key=lambda x: x["risk"], reverse=True)

    # Triage summary counters
    high_count = sum(1 for p in patient_rows if p["risk"] >= 0.4)
    mod_count = sum(1 for p in patient_rows if 0.2 <= p["risk"] < 0.4)
    low_count = sum(1 for p in patient_rows if p["risk"] < 0.2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monitored Patients", len(patient_rows))
    col2.metric("High Risk (🔴 ≥ 0.40)", high_count)
    col3.metric("Moderate Risk (🟡 0.20–0.40)", mod_count)
    col4.metric("Low Risk (🟢 < 0.20)", low_count)

    st.markdown("### Patient Triage List")
    st.caption("Click **Open Virtual Patient** to inspect full CGM history, factor explanations, and simulate what-if scenarios.")

    for p in patient_rows:
        with st.container(border=True):
            cols = st.columns([1.5, 2.2, 1.8, 1.5, 1.8, 2.0])
            with cols[0]:
                st.write(f"**{p['id']}**")
                st.caption(f"{p['gender']}, {p['age']} yrs")
            with cols[1]:
                st.write(f"**{p['name']}**")
                st.caption(f"BMI: {p['bmi']:.1f} | HbA1c: {p['a1c']:.1f}%")
            with cols[2]:
                st.write(f"**Current:** {p['glu_now']} mg/dL")
                delta_str = f"{p['glu_change_30']:+0.1f} mg/dL (30m)"
                st.caption(delta_str)
            with cols[3]:
                st.write(f"Last Meal: {int(p['meal_carbs'])}g")
                st.caption(f"{p['mins_since_meal']} mins ago")
            with cols[4]:
                st.markdown(
                    f"<div style='font-size: 1.1rem; font-weight: bold; color: {p['hex_color']};'>"
                    f"{p['badge']}</div>"
                    f"<div style='font-size: 0.82rem; color: #666;'>"
                    f"Model score: {p['risk']:.2f}</div>",
                    unsafe_allow_html=True,
                )
            with cols[5]:
                if st.button("Open Virtual Patient", key=f"btn_{p['id']}", use_container_width=True):
                    st.session_state["selected_patient_id"] = p["id"]
                    st.rerun()

    render_footer()


def show_patient_detail(patient: Dict[str, Any]):
    """Page 2: Patient Detail view (Virtual Patient) with CGM, Risk, Why, and What-if panels."""
    if st.button("⬅ Back to Patient Cohort"):
        st.session_state["selected_patient_id"] = None
        st.rerun()

    # Compute baseline risk
    feat_df = compute_features(
        history=patient["history"],
        static=patient["static"],
        mins_since_meal=patient["mins_since_meal"],
        meal_carbs=patient["meal_carbs"],
    )
    baseline_risk = predict_risk(feat_df)
    label, hex_color, badge = get_risk_badge(baseline_risk)

    # Header
    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.title(f"Virtual Patient: {patient['name']} ({patient['id']})")
    with head_right:
        st.markdown(
            f"<div style='text-align: right; padding-top: 15px; font-size: 1.25rem; font-weight: bold; color: {hex_color};'>"
            f"{badge}</div>",
            unsafe_allow_html=True,
        )

    # 1. Profile Card
    with st.container(border=True):
        st.subheader("Patient Clinical Profile")
        p_cols = st.columns(6)
        p_cols[0].metric("Age", f"{int(patient['static']['Age'])} yrs")
        p_cols[1].metric("Gender", "Male" if patient['static']['Gender'] == 1 else "Female")
        p_cols[2].metric("BMI", f"{patient['static']['BMI']:.1f}")
        p_cols[3].metric("HbA1c", f"{patient['static']['A1c']:.1f}%")
        p_cols[4].metric("Fasting Glucose", f"{patient['static']['fasting_glu']:.0f} mg/dL")
        p_cols[5].metric("Fasting Insulin", f"{patient['static']['Insulin']:.0f} µIU/mL")

    # 2. Current Risk Banner
        # 2. Current Risk Banner
    # Validation line: how often cases with a similar score really reached 180 mg/dL
    val = validation_rate(baseline_risk)
    if val is not None:
        val_rate, val_rows = val
       
        validation_text = (
            f"In validation on unseen participants, about <strong>{val_rate:.0%}</strong> of readings "
            f"with a score in this range were followed by a rise to 180 mg/dL within 2 hours "
            f"({val_rows:,} five-minute readings; neighbouring readings are highly correlated)."
        )
    else:
        validation_text = ""

    st.markdown(
        f"""
        <div style="background-color: rgba(220, 53, 69, 0.08); border-left: 6px solid {hex_color};
                    padding: 14px 20px; border-radius: 6px; margin: 12px 0 20px 0;">
            <div style="font-size: 0.95rem; color: #555; text-transform: uppercase; font-weight: 600;">
                Model risk score (reaching 180 mg/dL within 2 hours)
            </div>
            <div style="font-size: 1.55rem; font-weight: 700; color: {hex_color}; margin: 4px 0;">
                {baseline_risk:.2f} — {label}
            </div>
            <div style="font-size: 1.0rem; color: #333;">
                A relative score from the model, not a calibrated probability. {validation_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    

    # 3. 24-Hour Glucose Line Chart
    cgm_fig = build_cgm_chart(patient["history"], patient["name"])
    st.plotly_chart(cgm_fig, use_container_width=True)

    # 4. Explanations & What-if Simulation
    col_why, col_whatif = st.columns(2)

    with col_why:
        with st.container(border=True):
            st.subheader("💡 'Why' Panel: Factor Contributions")
            st.caption(
                "Key drivers pushing risk towards or away from the 180 mg/dL threshold."
            )
            contributions_df = get_feature_contributions(feat_df)
            why_fig = build_why_chart(contributions_df)
            st.plotly_chart(why_fig, use_container_width=True)
            
            st.caption(
                "Bars show how much each factor moved this patient's score, according to the model. "
                "'Baseline' items are fixed for the patient; 'Now' items reflect the current situation. "
                "Bar size is a relative measure, not a percentage, and does not prove a cause."
            )

    with col_whatif:
        with st.container(border=True):
            st.subheader("🧪 'What-If' Simulation (Digital Twin)")
            st.caption(
                "Simulate carbohydrate and meal timing adjustments to explore risk sensitivity."
            )

            # Sliders (Strictly meal carbs & minutes since meal - NO activity or heart rate)
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

            # Re-run prediction under simulation
            sim_feat_df = compute_features(
                history=patient["history"],
                static=patient["static"],
                mins_since_meal=float(sim_mins),
                meal_carbs=float(sim_carbs),
            )
            sim_risk = predict_risk(sim_feat_df)
            sim_label, sim_hex, sim_badge = get_risk_badge(sim_risk)

            # Before and after comparison
                    
            delta_risk = sim_risk - baseline_risk

            st.markdown("#### Simulation Outcome")
            st.write(
                f"**Baseline score:** `{baseline_risk:.2f}` "
                f"&nbsp; ➔ &nbsp; **Simulated score:** `{sim_risk:.2f}` "
                f"&nbsp; (change {delta_risk:+.2f})"
            )

            if delta_risk < -0.005:
                st.success(
                    f"According to the model, these meal inputs lower the risk score by {abs(delta_risk):.2f}."
                )
            elif delta_risk > 0.005:
                st.warning(
                    f"According to the model, these meal inputs raise the risk score by {delta_risk:.2f}."
                )
            else:
                st.info("According to the model, the risk score is essentially unchanged.")

            st.caption(
                "This shows how sensitive the model is to the meal inputs. The patient's glucose "
                "history is kept fixed, so it is not a forecast of what would really happen."
            )
            
            st.caption(
                f"Simulated classification: <span style='color: {sim_hex}; font-weight: bold;'>{sim_badge}</span>",
                unsafe_allow_html=True,
            )

    render_footer()


def main():
    # Load synthetic cohort
    patients = get_all_patients()
    patient_map = {p["id"]: p for p in patients}

    # Initialize navigation state
    if "selected_patient_id" not in st.session_state:
        st.session_state["selected_patient_id"] = None

    selected_id = st.session_state["selected_patient_id"]

    if selected_id and selected_id in patient_map:
        show_patient_detail(patient_map[selected_id])
    else:
        show_patient_list(patients)


if __name__ == "__main__":
    main()
