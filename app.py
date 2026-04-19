"""Modern SaaS-style Streamlit dashboard for Vehicle Maintenance AI."""

from __future__ import annotations

import os
import uuid
from html import escape

import altair as alt
import joblib
import pandas as pd
import streamlit as st

try:
    from langgraph.types import Command
    from agent import fleet_graph, health_report_graph, run_chat_agent, schedule_graph

    AGENT_AVAILABLE = True
    AGENT_IMPORT_ERROR = ""
except Exception as exc:
    AGENT_AVAILABLE = False
    AGENT_IMPORT_ERROR = str(exc)


st.set_page_config(page_title="Vehicle Maintenance AI", layout="wide", page_icon="🚗")

st.markdown(
    """
<style>
:root {
  --bg: #0a1020;
  --panel: #111a2e;
  --panel-soft: #131f38;
  --panel-border: #233354;
  --text: #e6edf7;
  --muted: #9fb2d1;
  --accent: #4f8cff;
  --accent-2: #19c2b8;
  --good: #1fc16b;
  --bad: #ff5d6c;
  --warning: #f5b14c;
}

.stApp {
  background: radial-gradient(1200px 600px at 90% -20%, #203a74 0%, rgba(32,58,116,0) 55%),
              radial-gradient(900px 500px at -10% -20%, #17305e 0%, rgba(23,48,94,0) 45%),
              var(--bg);
  color: var(--text);
    font-family: "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif;
}

.block-container {
  max-width: 100% !important;
  padding-top: 1.2rem;
  padding-left: 2rem;
  padding-right: 2rem;
  padding-bottom: 2rem;
  animation: fadeIn 280ms ease-out;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

h1, h2, h3 {
  letter-spacing: -0.02em;
}

[data-testid="stHeader"] {
  background: transparent;
}

div[data-testid="stTabs"] button {
  color: var(--muted);
    border-radius: 999px;
    border: 1px solid rgba(98, 128, 181, 0.22);
    background: rgba(18, 29, 53, 0.7);
    margin-right: 0.35rem;
    padding: 0.45rem 0.9rem;
  transition: all 160ms ease;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--text);
    border-color: rgba(106, 149, 231, 0.55);
    background: linear-gradient(135deg, rgba(79,140,255,0.32), rgba(25,194,184,0.25));
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: rgba(14, 24, 44, 0.92) !important;
    border-color: rgba(87, 120, 181, 0.45) !important;
    border-radius: 10px !important;
}

input, textarea {
    color: #e6edf7 !important;
}

div[data-baseweb="select"] * {
    color: #dbe8ff !important;
}

/* AI chat/search field visibility */
[data-testid="stChatInput"] {
    background: rgba(13, 22, 41, 0.95) !important;
    border: 1px solid rgba(95, 130, 189, 0.45) !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    color: #eef4ff !important;
    caret-color: #eef4ff !important;
}

[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
    color: #95abd0 !important;
    opacity: 1 !important;
}

div[data-testid="stForm"] {
  background: rgba(17, 26, 46, 0.78);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  padding: 1.1rem 1rem 0.7rem 1rem;
  box-shadow: 0 12px 34px rgba(5, 10, 20, 0.35);
}

/* Card feel for each input group column */
[data-testid="stForm"] div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
  background: rgba(19, 31, 56, 0.9);
  border: 1px solid rgba(85, 117, 176, 0.28);
  border-radius: 14px;
  padding: 0.95rem 0.8rem 0.2rem 0.8rem;
  min-height: 100%;
}

label, [data-testid="stMarkdownContainer"] p {
  color: var(--muted);
}

.stButton > button,
.stFormSubmitButton > button {
  width: 100%;
  border: 1px solid rgba(114, 148, 212, 0.45);
  border-radius: 12px;
  min-height: 2.7rem;
  background: linear-gradient(135deg, rgba(79,140,255,0.96), rgba(25,194,184,0.86));
  color: #f8fbff;
  font-weight: 600;
  transition: transform 140ms ease, box-shadow 140ms ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
  transform: translateY(-1px) scale(1.01);
  box-shadow: 0 10px 26px rgba(18, 63, 140, 0.35);
}

.section-header {
    background: linear-gradient(135deg, rgba(32, 52, 91, 0.95), rgba(19, 33, 62, 0.95));
    border: 1px solid rgba(97, 129, 186, 0.35);
  border-radius: 14px;
    padding: 1rem 1.1rem;
  margin: 0.15rem 0 1rem 0;
    box-shadow: 0 10px 24px rgba(6, 11, 24, 0.25);
}

.section-title {
  color: var(--text);
  font-size: 1.06rem;
  font-weight: 700;
  margin: 0;
}

.section-subtitle {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 0.3rem 0 0 0;
}

.kpi-card {
  background: linear-gradient(145deg, rgba(20, 33, 60, 0.95), rgba(15, 27, 50, 0.95));
  border: 1px solid rgba(87, 120, 181, 0.28);
  border-radius: 14px;
  padding: 0.9rem 1rem;
  box-shadow: 0 8px 20px rgba(8, 14, 30, 0.22);
}

.panel-card {
    background: rgba(17, 26, 46, 0.82);
    border: 1px solid rgba(93, 122, 176, 0.28);
    border-radius: 14px;
    padding: 0.95rem;
    box-shadow: 0 10px 24px rgba(8, 14, 28, 0.25);
}

.chart-title {
    color: #dfeaff;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
}

.chart-sub {
    color: #96add4;
    font-size: 0.85rem;
    margin: 0 0 0.8rem 0;
}

.kpi-label {
  color: var(--muted);
  font-size: 0.82rem;
  margin-bottom: 0.35rem;
}

.kpi-value {
  color: var(--text);
  font-size: 1.32rem;
  font-weight: 700;
}

.status-card {
  border-radius: 14px;
  padding: 0.95rem 1rem;
  border: 1px solid transparent;
  margin-bottom: 0.9rem;
}

.status-good {
  background: linear-gradient(145deg, rgba(19, 74, 50, 0.45), rgba(18, 54, 40, 0.4));
  border-color: rgba(31, 193, 107, 0.38);
}

.status-bad {
  background: linear-gradient(145deg, rgba(93, 20, 34, 0.45), rgba(70, 15, 26, 0.4));
  border-color: rgba(255, 93, 108, 0.36);
}

.status-text {
  color: var(--text);
  font-weight: 650;
  margin: 0;
}

.badge {
  display: inline-block;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  padding: 0.24rem 0.54rem;
  margin-bottom: 0.5rem;
}

.badge-high {
  color: #ffd8db;
  background: rgba(255, 93, 108, 0.2);
  border: 1px solid rgba(255, 93, 108, 0.42);
}

.badge-routine {
  color: #d8ffe9;
  background: rgba(31, 193, 107, 0.2);
  border: 1px solid rgba(31, 193, 107, 0.42);
}

.ai-card {
  background: rgba(17, 26, 46, 0.78);
  border: 1px solid var(--panel-border);
  border-radius: 14px;
    padding: 0.95rem;
  margin-bottom: 0.7rem;
    box-shadow: 0 10px 24px rgba(8, 14, 28, 0.22);
}

.chat-shell {
  background: rgba(15, 24, 43, 0.86);
  border: 1px solid rgba(85, 117, 176, 0.26);
  border-radius: 14px;
    padding: 0.85rem;
  max-height: 420px;
  overflow-y: auto;
}

.chat-empty {
    color: #95abd0;
    text-align: center;
    padding: 1.1rem 0.7rem;
    border: 1px dashed rgba(113, 142, 196, 0.35);
    border-radius: 10px;
    background: rgba(17, 27, 49, 0.6);
}

.chat-row {
  display: flex;
  margin: 0.4rem 0;
}

.chat-row.user {
  justify-content: flex-end;
}

.chat-row.assistant {
  justify-content: flex-start;
}

.bubble {
  max-width: 80%;
  padding: 0.62rem 0.78rem;
  border-radius: 12px;
  font-size: 0.92rem;
  line-height: 1.36;
  white-space: pre-wrap;
}

.bubble.user {
  color: #ecf4ff;
  background: linear-gradient(135deg, rgba(79, 140, 255, 0.88), rgba(44, 92, 200, 0.84));
  border-bottom-right-radius: 6px;
}

.bubble.assistant {
  color: #dce7f8;
  background: rgba(39, 56, 88, 0.85);
  border: 1px solid rgba(94, 121, 174, 0.34);
  border-bottom-left-radius: 6px;
}

hr {
  border: none;
  border-top: 1px solid rgba(97, 124, 171, 0.28);
}

.tool-pill {
    display: inline-block;
    margin: 0.2rem 0.35rem 0.2rem 0;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    color: #d8e6ff;
    background: rgba(62, 96, 158, 0.45);
    border: 1px solid rgba(112, 148, 213, 0.35);
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def section_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="section-header">
  <p class="section-title">{icon} {escape(title)}</p>
  <p class="section-subtitle">{escape(subtitle)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{escape(label)}</div>
  <div class="kpi-value">{escape(value)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts() -> tuple[object | None, object | None]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "maintenance_model.pkl")
    preprocessor_path = os.path.join(base_dir, "models", "preprocessor.pkl")

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception:
        return None, None


def ensure_session_state() -> None:
    defaults = {
        "risk_score": None,
        "vehicle_data": None,
        "chat_history": [],
        "previous_report": "",
        "triage_result": None,
        "pending_schedule": None,
        "final_schedule": None,
        "schedule_thread_id": None,
        "schedule_urgency": None,
        "chat_thread_id": str(uuid.uuid4()),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def vehicle_form() -> tuple[bool, pd.DataFrame, dict]:
    with st.form("vehicle_form"):
        section_header("🚘", "Vehicle Input", "Structured details for prediction and AI analysis")

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown("**Vehicle Profile**")
            vehicle_model = st.selectbox("Model", ["Car", "SUV", "Van", "Truck", "Bus", "Motorcycle"])
            vehicle_age = st.number_input("Age (years)", min_value=1, max_value=10, value=5)
            engine_size = st.selectbox("Engine (cc)", [800, 1000, 1500, 2000, 2500], index=2)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
            transmission_type = st.selectbox("Transmission", ["Automatic", "Manual"])
            owner_type = st.selectbox("Owner Type", ["First", "Second", "Third"])

        with col2:
            st.markdown("**Usage & Service**")
            mileage = st.number_input("Mileage (km)", min_value=30000, max_value=80000, value=55000, step=5000)
            odometer_reading = st.number_input("Odometer (km)", min_value=1000, max_value=150000, value=75000, step=5000)
            service_history = st.number_input("Past Services", min_value=1, max_value=10, value=5)
            last_service_days_ago = st.number_input("Days Since Last Service", min_value=700, max_value=1100, value=896, step=10)
            warranty_days_remaining = st.number_input("Warranty Days Left", min_value=-700, max_value=50, value=-320, step=10)
            reported_issues = st.selectbox("Reported Issues", [0, 1, 2, 3, 4, 5], index=2)
            accident_history = st.selectbox("Accident Count", [0, 1, 2, 3], index=1)

        with col3:
            st.markdown("**Components & Cost**")
            tire_condition = st.selectbox("Tire Condition", ["New", "Good", "Worn Out"])
            brake_condition = st.selectbox("Brake Condition", ["New", "Good", "Worn Out"])
            battery_status = st.selectbox("Battery", ["New", "Good", "Weak"])
            maintenance_history = st.selectbox("Maintenance History", ["Good", "Average", "Poor"], index=1)
            fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=10.0, max_value=20.0, value=15.0, step=0.5)
            insurance_premium = st.number_input("Insurance Premium (INR)", min_value=5000, max_value=30000, value=17500, step=1000)

        submitted = st.form_submit_button("Predict Maintenance Need", use_container_width=True)

    input_df = pd.DataFrame(
        [
            {
                "Mileage": mileage,
                "Reported_Issues": reported_issues,
                "Vehicle_Age": vehicle_age,
                "Engine_Size": engine_size,
                "Odometer_Reading": odometer_reading,
                "Insurance_Premium": float(insurance_premium),
                "Service_History": service_history,
                "Accident_History": accident_history,
                "Fuel_Efficiency": fuel_efficiency,
                "Last_Service_Days_Ago": last_service_days_ago,
                "Warranty_Days_Remaining": warranty_days_remaining,
                "Vehicle_Model": vehicle_model,
                "Maintenance_History": maintenance_history,
                "Fuel_Type": fuel_type,
                "Transmission_Type": transmission_type,
                "Owner_Type": owner_type,
                "Tire_Condition": tire_condition,
                "Brake_Condition": brake_condition,
                "Battery_Status": battery_status,
            }
        ]
    )

    vehicle_context = {
        "Vehicle_Model": str(vehicle_model),
        "Vehicle_Age": int(vehicle_age),
        "Mileage": int(mileage),
        "Fuel_Type": str(fuel_type),
        "Engine_Size": int(engine_size),
        "Last_Service_Days_Ago": int(last_service_days_ago),
        "Warranty_Days_Remaining": int(warranty_days_remaining),
        "Tire_Condition": str(tire_condition),
        "Brake_Condition": str(brake_condition),
        "Battery_Status": str(battery_status),
        "Maintenance_History": str(maintenance_history),
        "Accident_History": int(accident_history),
        "Reported_Issues": int(reported_issues),
        "Odometer_Reading": int(odometer_reading),
    }

    return submitted, input_df, vehicle_context


def render_result_cards(risk_score: float, vehicle_data: dict) -> None:
    status_class = "status-bad" if risk_score >= 0.5 else "status-good"
    status_icon = "⚠️" if risk_score >= 0.5 else "✅"
    status_text = "Maintenance Required" if risk_score >= 0.5 else "No Immediate Maintenance Needed"

    st.markdown(
        f"""
<div class="status-card {status_class}">
  <p class="status-text">{status_icon} {status_text} - Risk Score: {risk_score:.2f}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3, gap="large")
    with k1:
        render_kpi("Risk Score", f"{risk_score:.2f}")
    with k2:
        render_kpi("Vehicle Age", f"{vehicle_data['Vehicle_Age']} years")
    with k3:
        render_kpi("Reported Issues", str(vehicle_data["Reported_Issues"]))

    st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)

    prob_df = pd.DataFrame(
        {
            "Status": ["No Maintenance", "Needs Maintenance"],
            "Probability": [1 - risk_score, risk_score],
        }
    )

    chart = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Status:N", axis=alt.Axis(labelColor="#c7d6f3", title=None)),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labelColor="#c7d6f3", title=None)),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(range=["#19c2b8", "#ff5d6c"]),
                legend=None,
            ),
            tooltip=["Status", alt.Tooltip("Probability:Q", format=".2f")],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
    )

    st.markdown(
        """
<div class="panel-card">
    <p class="chart-title">📊 Prediction Confidence</p>
    <p class="chart-sub">Model confidence split across prediction classes</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.altair_chart(chart, use_container_width=True)


def render_chat_history() -> None:
    st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
    if not st.session_state["chat_history"]:
        st.markdown(
            """
<div class="chat-empty">No conversation yet. Ask about urgency, expected repair cost, or maintenance steps.</div>
""",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state["chat_history"]:
            role = "user" if msg["role"] == "user" else "assistant"
            content = escape(msg["content"]).replace("\n", "<br>")
            st.markdown(
                f"""
<div class="chat-row {role}">
  <div class="bubble {role}">{content}</div>
</div>
""",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_section() -> None:
    section_header("🤖", "AI Assistant", "Agentic health report, schedule planning, and tool-driven chat")

    ai_health, ai_schedule, ai_chat = st.tabs(["🩺 Health Report", "📅 Schedule Planner", "💬 AI Chat"])
    risk_score = st.session_state["risk_score"]
    vehicle_data = st.session_state["vehicle_data"]

    with ai_health:
        st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
        if st.button("Generate Health Report", key="health_report_btn", use_container_width=True):
            with st.spinner("Generating health report..."):
                try:
                    result = health_report_graph.invoke(
                        {
                            "vehicle_data": vehicle_data,
                            "risk_score": risk_score,
                            "retrieved_context": None,
                            "triage_result": None,
                            "report": None,
                        }
                    )
                    st.session_state["previous_report"] = result.get("report", "")
                    st.session_state["triage_result"] = result.get("triage_result", "unknown")
                except Exception as exc:
                    st.error(f"Health report failed: {exc}")

        if st.session_state.get("triage_result"):
            triage = st.session_state["triage_result"]
            if triage == "high_risk":
                st.markdown('<span class="badge badge-high">HIGH RISK ROUTE</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-routine">ROUTINE ROUTE</span>', unsafe_allow_html=True)

        if st.session_state.get("previous_report"):
            st.text_area("AI Health Report", st.session_state["previous_report"], height=340, key="report_box")
        st.markdown("</div>", unsafe_allow_html=True)

    with ai_schedule:
        st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
        if st.button("Generate 90-Day Schedule", key="generate_schedule_btn", use_container_width=True):
            st.session_state["schedule_thread_id"] = str(uuid.uuid4())
            st.session_state["pending_schedule"] = None
            st.session_state["final_schedule"] = None

            with st.spinner("Planning schedule..."):
                try:
                    out = schedule_graph.invoke(
                        {
                            "vehicle_data": vehicle_data,
                            "risk_score": risk_score,
                            "schedule": None,
                            "urgency_level": None,
                            "approval_status": None,
                            "manager_notes": None,
                            "final_schedule": None,
                        },
                        config={"configurable": {"thread_id": st.session_state["schedule_thread_id"]}},
                    )
                    if out.get("final_schedule"):
                        st.session_state["final_schedule"] = out.get("final_schedule")
                    else:
                        st.session_state["pending_schedule"] = out.get("schedule", "")
                        st.session_state["schedule_urgency"] = out.get("urgency_level", "HIGH")
                except Exception as exc:
                    st.error(f"Schedule generation failed: {exc}")

        if st.session_state.get("pending_schedule") and not st.session_state.get("final_schedule"):
            st.warning("Human approval required before finalizing this schedule.")
            st.text_area("Pending Schedule", st.session_state["pending_schedule"], height=240, key="pending_schedule_box")
            notes = st.text_input("Manager Notes", key="manager_notes")
            a_col, m_col = st.columns(2)

            with a_col:
                if st.button("Approve", key="approve_schedule_btn", use_container_width=True):
                    try:
                        resumed = schedule_graph.invoke(
                            Command(resume={"approval": "approved", "notes": notes}),
                            config={"configurable": {"thread_id": st.session_state["schedule_thread_id"]}},
                        )
                        st.session_state["final_schedule"] = resumed.get("final_schedule", "")
                        st.session_state["pending_schedule"] = None
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Approval failed: {exc}")

            with m_col:
                if st.button("Modify & Approve", key="modify_schedule_btn", use_container_width=True):
                    try:
                        resumed = schedule_graph.invoke(
                            Command(resume={"approval": "modified", "notes": notes or "Modified"}),
                            config={"configurable": {"thread_id": st.session_state["schedule_thread_id"]}},
                        )
                        st.session_state["final_schedule"] = resumed.get("final_schedule", "")
                        st.session_state["pending_schedule"] = None
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Modify failed: {exc}")

        if st.session_state.get("final_schedule"):
            st.success("Schedule finalized.")
            st.text_area("Final Schedule", st.session_state["final_schedule"], height=300, key="final_schedule_box")

        st.markdown("</div>", unsafe_allow_html=True)

    with ai_chat:
        st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
        render_chat_history()

        user_prompt = st.chat_input("Ask the assistant about urgency, cost, or procedures")
        if user_prompt:
            st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
            try:
                answer, tool_trace = run_chat_agent(
                    vehicle_data=vehicle_data,
                    risk_score=risk_score,
                    previous_report=st.session_state.get("previous_report", ""),
                    user_query=user_prompt,
                    history=st.session_state["chat_history"][:-1],
                    thread_id=st.session_state["chat_thread_id"],
                )
                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "tool_trace": tool_trace,
                    }
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Chat failed: {exc}")

        if st.session_state["chat_history"]:
            last_assistant = next(
                (m for m in reversed(st.session_state["chat_history"]) if m["role"] == "assistant"),
                None,
            )
            if last_assistant and last_assistant.get("tool_trace"):
                with st.expander("Tool Usage", expanded=False):
                    for idx, tool_name in enumerate(last_assistant["tool_trace"], 1):
                        st.markdown(
                            f"<span class='tool-pill'>Step {idx}: {escape(tool_name)}</span>",
                            unsafe_allow_html=True,
                        )

        st.markdown("</div>", unsafe_allow_html=True)


def render_fleet_dashboard() -> None:
    section_header("🌐", "Fleet Dashboard", "Snapshot of fleet-level health and strategic action")

    if not AGENT_AVAILABLE:
        st.warning("AI features not available. Install requirements to enable fleet analysis.")
        if AGENT_IMPORT_ERROR:
            st.caption(f"Import error: {AGENT_IMPORT_ERROR}")
        return

    fleet_stats = {
        "total_vehicles": 250,
        "high_risk_count": 32,
        "medium_risk_count": 89,
        "low_risk_count": 129,
        "top_vehicles": (
            "1. Truck (ID: T-402), 8 years old, Risk: 0.92 - Brake failure imminent\n"
            "2. SUV (ID: S-119), 6 years old, Risk: 0.88 - Worn tires, high mileage\n"
            "3. Van (ID: V-044), 9 years old, Risk: 0.85 - Engine knocking, no warranty"
        ),
        "vehicle_type_summary": (
            "Trucks: 80 vehicles, avg age 6 yrs | "
            "SUVs: 100 vehicles, avg age 4 yrs | "
            "Vans: 70 vehicles, avg age 5 yrs"
        ),
    }

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        render_kpi("High Risk", str(fleet_stats["high_risk_count"]))
    with c2:
        render_kpi("Medium Risk", str(fleet_stats["medium_risk_count"]))
    with c3:
        render_kpi("Low Risk", str(fleet_stats["low_risk_count"]))

    st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
    if st.button("Generate Fleet Strategy Report", use_container_width=True):
        with st.spinner("Generating fleet strategy report..."):
            try:
                result = fleet_graph.invoke({"fleet_stats": fleet_stats, "dashboard_report": None})
                st.text_area("Fleet Strategy Report", result.get("dashboard_report", ""), height=430)
            except Exception as exc:
                st.error(f"Fleet report failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


model, preprocessor = load_artifacts()
if model is None:
    st.error("Model artifacts missing. Please run `python train.py` first.")
    st.stop()

ensure_session_state()

st.title("Vehicle Maintenance AI")
st.caption("Modern predictive maintenance dashboard with agentic decision support")

single_tab, ai_copilot_tab, fleet_tab = st.tabs(
    ["🚘 Single Vehicle", "🤖 AI Copilot", "📡 Fleet Overview"]
)

with single_tab:
    submitted, input_df, vehicle_context = vehicle_form()

    if submitted:
        try:
            risk_score = float(model.predict_proba(preprocessor.transform(input_df))[0][1])
            st.session_state["risk_score"] = risk_score
            st.session_state["vehicle_data"] = vehicle_context
            st.session_state["chat_history"] = []
            st.session_state["previous_report"] = ""
            st.session_state["triage_result"] = None
            st.session_state["pending_schedule"] = None
            st.session_state["final_schedule"] = None
            st.session_state["schedule_thread_id"] = None
            st.session_state["chat_thread_id"] = str(uuid.uuid4())
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if st.session_state.get("risk_score") is not None and st.session_state.get("vehicle_data"):
        render_result_cards(st.session_state["risk_score"], st.session_state["vehicle_data"])


with ai_copilot_tab:
    section_header("🤖", "AI Copilot Workspace", "Dedicated assistant area for health analysis, scheduling, and chat")

    if not AGENT_AVAILABLE:
        st.warning("AI agent modules are not available. Install dependencies and check imports.")
        if AGENT_IMPORT_ERROR:
            st.caption(f"Import error: {AGENT_IMPORT_ERROR}")
    elif st.session_state.get("risk_score") is None or st.session_state.get("vehicle_data") is None:
        st.info("Run a prediction in the Single Vehicle tab first to activate AI Copilot with vehicle context.")
    else:
        risk_score = st.session_state["risk_score"]
        vehicle_data = st.session_state["vehicle_data"]
        st.markdown(
            f"""
<div class="panel-card" style="margin-bottom: 0.8rem;">
  <p class="chart-title" style="margin-bottom: 0.2rem;">Active Vehicle Context</p>
  <p class="chart-sub">Model: {escape(str(vehicle_data.get('Vehicle_Model', 'N/A')))} | Risk Score: {risk_score:.2f}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        render_ai_section()

with fleet_tab:
    render_fleet_dashboard()
