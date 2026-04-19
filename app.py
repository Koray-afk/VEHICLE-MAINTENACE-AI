"""
app.py — Vehicle Maintenance AI (Streamlit Dashboard)
======================================================
Two main tabs:
  1. Single Vehicle Analysis  — predict + AI health report + schedule + chat
  2. Fleet Dashboard          — fleet-level AI report

Agentic patterns surfaced in UI:
  • Health Report : Conditional Routing triage badge
  • Schedule      : Human-in-the-Loop approval flow
  • Chat          : ReAct tool-call trace expander
"""

"""
app.py — Vehicle Maintenance AI (Streamlit Dashboard)
======================================================
Two main tabs:
  1. Single Vehicle Analysis  — predict + AI health report + schedule + chat
  2. Fleet Dashboard          — fleet-level AI report

Agentic patterns surfaced in UI:
  • Health Report : Conditional Routing triage badge
  • Schedule      : Human-in-the-Loop approval flow
  • Chat          : ReAct tool-call trace expander
"""

import streamlit as st
import pandas as pd
import joblib
import os
import uuid

from langgraph.types import Command

# ── AI agent imports ──────────────────────────────────────────────────────────
try:
    from agent import (
        health_report_graph,
        schedule_graph,
        fleet_graph,
        run_chat_agent,
    )
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Vehicle Maintenance AI", layout="wide", page_icon="🚗")

# ── Global CSS & Animations ───────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Fade-in for the whole app ---- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
section.main > div { animation: fadeIn 0.6s ease both; }

/* ---- Slide-up for metric cards ---- */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stMetric"] { animation: slideUp 0.5s ease both; }

/* ---- Pulse on primary buttons ---- */
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,75,75,0.4); }
    70%  { box-shadow: 0 0 0 10px rgba(255,75,75,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,75,75,0); }
}
[data-testid="stFormSubmitButton"] button,
button[kind="primary"] { animation: pulse 2s infinite; }

/* ---- Tab underline transition ---- */
[data-testid="stTab"] { transition: color 0.3s ease; }
</style>
""", unsafe_allow_html=True)

# ── Splash screen also (shown only once per session) ───────────────────────────────
if "splash_done" not in st.session_state:
    st.session_state["splash_done"] = False

if not st.session_state["splash_done"]:
    st.markdown("""
    <style>
    #splash {
        position: fixed; inset: 0; z-index: 9999;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        animation: splashFade 0.6s ease 3.2s forwards;
    }
    @keyframes splashFade { to { opacity: 0; pointer-events: none; } }

    /* ---- Car body ---- */
    .car-wrap { position: relative; width: 260px; height: 120px; }

    @keyframes carBounce {
        0%,100% { transform: translateY(0); }
        50%      { transform: translateY(-6px); }
    }
    .car { animation: carBounce 0.8s ease-in-out infinite; }

    /* ---- Wheels spinning ---- */
    @keyframes spin { to { transform: rotate(360deg); } }
    .wheel { animation: spin 0.6s linear infinite; transform-origin: center; }

    /* ---- Wrench swinging ---- */
    @keyframes swing {
        0%,100% { transform: rotate(-30deg); }
        50%      { transform: rotate(30deg); }
    }
    .wrench { animation: swing 0.7s ease-in-out infinite; transform-origin: bottom center; }

    /* ---- Sparks flying ---- */
    @keyframes spark1 { 0%{opacity:1;transform:translate(0,0) scale(1)} 100%{opacity:0;transform:translate(20px,-30px) scale(0)} }
    @keyframes spark2 { 0%{opacity:1;transform:translate(0,0) scale(1)} 100%{opacity:0;transform:translate(-15px,-25px) scale(0)} }
    @keyframes spark3 { 0%{opacity:1;transform:translate(0,0) scale(1)} 100%{opacity:0;transform:translate(10px,-35px) scale(0)} }
    .s1 { animation: spark1 0.6s ease-out infinite; }
    .s2 { animation: spark2 0.6s ease-out 0.2s infinite; }
    .s3 { animation: spark3 0.6s ease-out 0.1s infinite; }

    /* ---- Progress bar ---- */
    @keyframes loadBar { from{width:0} to{width:100%} }
    .progress-track { width: 260px; height: 6px; background: rgba(255,255,255,0.15); border-radius: 4px; margin-top: 24px; overflow: hidden; }
    .progress-fill  { height: 100%; background: linear-gradient(90deg, #f7971e, #ffd200); border-radius: 4px; animation: loadBar 3s ease forwards; }

    .splash-title { color: #fff; font-family: sans-serif; font-size: 28px; font-weight: 700; margin-top: 20px; letter-spacing: 1px; }
    .splash-sub   { color: rgba(255,255,255,0.55); font-family: sans-serif; font-size: 14px; margin-top: 6px; }
    </style>

    <div id="splash">
      <!-- Animated car SVG -->
      <div class="car-wrap">
        <svg class="car" width="260" height="100" viewBox="0 0 260 100" fill="none" xmlns="http://www.w3.org/2000/svg">
          <!-- Body -->
          <rect x="20" y="50" width="220" height="40" rx="8" fill="#f7971e"/>
          <!-- Roof -->
          <path d="M70 50 L95 20 H175 L200 50 Z" fill="#ffd200"/>
          <!-- Windshield -->
          <path d="M100 50 L118 26 H158 L178 50 Z" fill="#a8d8ea" opacity="0.8"/>
          <!-- Wheels -->
          <circle cx="65"  cy="90" r="18" fill="#222"/>
          <circle cx="65"  cy="90" r="9"  fill="#aaa"/>
          <circle class="wheel" cx="65" cy="90" r="5" fill="#555"/>
          <circle cx="195" cy="90" r="18" fill="#222"/>
          <circle cx="195" cy="90" r="9"  fill="#aaa"/>
          <circle class="wheel" cx="195" cy="90" r="5" fill="#555"/>
          <!-- Headlight -->
          <ellipse cx="238" cy="68" rx="8" ry="6" fill="#fffde7" opacity="0.9"/>
        </svg>

        <!-- Wrench above the car -->
        <svg class="wrench" style="position:absolute;top:-10px;right:30px;" width="36" height="60" viewBox="0 0 36 60" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="14" y="10" width="8" height="42" rx="4" fill="#aaa"/>
          <rect x="8"  y="6"  width="20" height="12" rx="6" fill="#ccc"/>
          <rect x="10" y="4"  width="6"  height="8"  rx="3" fill="#888"/>
          <rect x="20" y="4"  width="6"  height="8"  rx="3" fill="#888"/>
        </svg>

        <!-- Sparks -->
        <svg style="position:absolute;top:0;right:25px;" width="40" height="40" viewBox="0 0 40 40">
          <circle class="s1" cx="20" cy="30" r="3" fill="#ffd200"/>
          <circle class="s2" cx="20" cy="30" r="2" fill="#f7971e"/>
          <circle class="s3" cx="20" cy="30" r="2" fill="#fff"/>
        </svg>
      </div>

      <div class="splash-title">🚗 Vehicle Maintenance AI</div>
      <div class="splash-sub">Initialising systems...</div>
      <div class="progress-track"><div class="progress-fill"></div></div>
    </div>

    <script>
      // Remove splash from DOM after animation completes
      setTimeout(() => {
        const el = document.getElementById('splash');
        if (el) el.remove();
      }, 3800);
    </script>
        """, unsafe_allow_html=True)

    import time
    time.sleep(3.2)
    st.session_state["splash_done"] = True
    st.rerun()



# ── Load ML model ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
def load_artifacts():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "models", "maintenance_model.pkl"))
        preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.pkl"))
        model = joblib.load(os.path.join(BASE_DIR, "models", "maintenance_model.pkl"))
        preprocessor = joblib.load(os.path.join(BASE_DIR, "models", "preprocessor.pkl"))
        return model, preprocessor
    except Exception:
    except Exception:
        return None, None

model, preprocessor = load_artifacts()
model, preprocessor = load_artifacts()

if model is None:
    st.error("Model artifacts missing. Please run `python train.py` first.")
    st.error("Model artifacts missing. Please run `python train.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Maintenance AI")
st.caption("Predict maintenance needs and get AI-powered service advice for your fleet.")

tab_vehicle, tab_fleet = st.tabs(["🔍 Single Vehicle Analysis", "🌐 Fleet Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Vehicle Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_vehicle:

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("vehicle_form"):
        st.subheader("Vehicle Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Specifications**")
            Vehicle_Model       = st.selectbox("Model", ["Car", "SUV", "Van", "Truck", "Bus", "Motorcycle"])
            Vehicle_Age         = st.number_input("Age (years)", 1, 10, 5)
            Engine_Size         = st.selectbox("Engine (cc)", [800, 1000, 1500, 2000, 2500], index=2)
            Fuel_Type           = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
            Transmission_Type   = st.selectbox("Transmission", ["Automatic", "Manual"])
            Owner_Type          = st.selectbox("Owner", ["First", "Second", "Third"])
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Maintenance AI")
st.caption("Predict maintenance needs and get AI-powered service advice for your fleet.")

tab_vehicle, tab_fleet = st.tabs(["🔍 Single Vehicle Analysis", "🌐 Fleet Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Vehicle Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_vehicle:

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("vehicle_form"):
        st.subheader("Vehicle Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Specifications**")
            Vehicle_Model       = st.selectbox("Model", ["Car", "SUV", "Van", "Truck", "Bus", "Motorcycle"])
            Vehicle_Age         = st.number_input("Age (years)", 1, 10, 5)
            Engine_Size         = st.selectbox("Engine (cc)", [800, 1000, 1500, 2000, 2500], index=2)
            Fuel_Type           = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
            Transmission_Type   = st.selectbox("Transmission", ["Automatic", "Manual"])
            Owner_Type          = st.selectbox("Owner", ["First", "Second", "Third"])

        with c2:
            st.markdown("**Usage & History**")
            Mileage                 = st.number_input("Mileage (km)", 30000, 80000, 55000, 5000)
            Odometer_Reading        = st.number_input("Odometer (km)", 1000, 150000, 75000, 5000)
            Service_History         = st.number_input("Past Services", 1, 10, 5)
            Last_Service_Days_Ago   = st.number_input("Days Since Last Service", 700, 1100, 896, 10)
            Warranty_Days_Remaining = st.number_input("Warranty Days Left", -700, 50, -320, 10)
            Reported_Issues         = st.selectbox("Reported Issues", [0, 1, 2, 3, 4, 5], index=2)
            Accident_History        = st.selectbox("Accidents", [0, 1, 2, 3], index=1)

        with c3:
            st.markdown("**Component Condition**")
            Tire_Condition      = st.selectbox("Tires", ["New", "Good", "Worn Out"])
            Brake_Condition     = st.selectbox("Brakes", ["New", "Good", "Worn Out"])
            Battery_Status      = st.selectbox("Battery", ["New", "Good", "Weak"])
            Maintenance_History = st.selectbox("Maintenance History", ["Good", "Average", "Poor"], index=1)
            Fuel_Efficiency     = st.number_input("Fuel Efficiency (km/l)", 10.0, 20.0, 15.0, 0.5)
            Insurance_Premium   = st.number_input("Insurance Premium (₹)", 5000, 30000, 17500, 1000)

        submitted = st.form_submit_button("🔍 Predict Maintenance Need", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        input_df = pd.DataFrame([{
            "Mileage": Mileage,
            "Reported_Issues": Reported_Issues,
            "Vehicle_Age": Vehicle_Age,
            "Engine_Size": Engine_Size,
            "Odometer_Reading": Odometer_Reading,
            "Insurance_Premium": float(Insurance_Premium),
            "Service_History": Service_History,
            "Accident_History": Accident_History,
            "Fuel_Efficiency": Fuel_Efficiency,
            "Last_Service_Days_Ago": Last_Service_Days_Ago,
            "Warranty_Days_Remaining": Warranty_Days_Remaining,
            "Vehicle_Model": Vehicle_Model,
            "Maintenance_History": Maintenance_History,
            "Fuel_Type": Fuel_Type,
            "Transmission_Type": Transmission_Type,
            "Owner_Type": Owner_Type,
            "Tire_Condition": Tire_Condition,
            "Brake_Condition": Brake_Condition,
            "Battery_Status": Battery_Status,
        }])
        with c2:
            st.markdown("**Usage & History**")
            Mileage                 = st.number_input("Mileage (km)", 30000, 80000, 55000, 5000)
            Odometer_Reading        = st.number_input("Odometer (km)", 1000, 150000, 75000, 5000)
            Service_History         = st.number_input("Past Services", 1, 10, 5)
            Last_Service_Days_Ago   = st.number_input("Days Since Last Service", 700, 1100, 896, 10)
            Warranty_Days_Remaining = st.number_input("Warranty Days Left", -700, 50, -320, 10)
            Reported_Issues         = st.selectbox("Reported Issues", [0, 1, 2, 3, 4, 5], index=2)
            Accident_History        = st.selectbox("Accidents", [0, 1, 2, 3], index=1)

        with c3:
            st.markdown("**Component Condition**")
            Tire_Condition      = st.selectbox("Tires", ["New", "Good", "Worn Out"])
            Brake_Condition     = st.selectbox("Brakes", ["New", "Good", "Worn Out"])
            Battery_Status      = st.selectbox("Battery", ["New", "Good", "Weak"])
            Maintenance_History = st.selectbox("Maintenance History", ["Good", "Average", "Poor"], index=1)
            Fuel_Efficiency     = st.number_input("Fuel Efficiency (km/l)", 10.0, 20.0, 15.0, 0.5)
            Insurance_Premium   = st.number_input("Insurance Premium (₹)", 5000, 30000, 17500, 1000)

        submitted = st.form_submit_button("🔍 Predict Maintenance Need", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        input_df = pd.DataFrame([{
            "Mileage": Mileage,
            "Reported_Issues": Reported_Issues,
            "Vehicle_Age": Vehicle_Age,
            "Engine_Size": Engine_Size,
            "Odometer_Reading": Odometer_Reading,
            "Insurance_Premium": float(Insurance_Premium),
            "Service_History": Service_History,
            "Accident_History": Accident_History,
            "Fuel_Efficiency": Fuel_Efficiency,
            "Last_Service_Days_Ago": Last_Service_Days_Ago,
            "Warranty_Days_Remaining": Warranty_Days_Remaining,
            "Vehicle_Model": Vehicle_Model,
            "Maintenance_History": Maintenance_History,
            "Fuel_Type": Fuel_Type,
            "Transmission_Type": Transmission_Type,
            "Owner_Type": Owner_Type,
            "Tire_Condition": Tire_Condition,
            "Brake_Condition": Brake_Condition,
            "Battery_Status": Battery_Status,
        }])

        try:
            prob = float(model.predict_proba(preprocessor.transform(input_df))[0][1])

            # Store in session state for AI tools
            st.session_state["risk_score"] = prob
            st.session_state["vehicle_data"] = {
                "Vehicle_Model": str(Vehicle_Model),
                "Vehicle_Age": int(Vehicle_Age),
                "Mileage": int(Mileage),
                "Fuel_Type": str(Fuel_Type),
                "Engine_Size": int(Engine_Size),
                "Last_Service_Days_Ago": int(Last_Service_Days_Ago),
                "Warranty_Days_Remaining": int(Warranty_Days_Remaining),
                "Tire_Condition": str(Tire_Condition),
                "Brake_Condition": str(Brake_Condition),
                "Battery_Status": str(Battery_Status),
                "Maintenance_History": str(Maintenance_History),
                "Accident_History": int(Accident_History),
                "Reported_Issues": int(Reported_Issues),
                "Odometer_Reading": int(Odometer_Reading),
            }
            st.session_state["chat_history"] = []
            st.session_state["previous_report"] = ""
            st.session_state.pop("triage_result", None)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Show result badge
    if "risk_score" in st.session_state:
        prob = st.session_state["risk_score"]
        try:
            prob = float(model.predict_proba(preprocessor.transform(input_df))[0][1])

            # Store in session state for AI tools
            st.session_state["risk_score"] = prob
            st.session_state["vehicle_data"] = {
                "Vehicle_Model": str(Vehicle_Model),
                "Vehicle_Age": int(Vehicle_Age),
                "Mileage": int(Mileage),
                "Fuel_Type": str(Fuel_Type),
                "Engine_Size": int(Engine_Size),
                "Last_Service_Days_Ago": int(Last_Service_Days_Ago),
                "Warranty_Days_Remaining": int(Warranty_Days_Remaining),
                "Tire_Condition": str(Tire_Condition),
                "Brake_Condition": str(Brake_Condition),
                "Battery_Status": str(Battery_Status),
                "Maintenance_History": str(Maintenance_History),
                "Accident_History": int(Accident_History),
                "Reported_Issues": int(Reported_Issues),
                "Odometer_Reading": int(Odometer_Reading),
            }
            st.session_state["chat_history"] = []
            st.session_state["previous_report"] = ""
            st.session_state.pop("triage_result", None)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Show result badge
    if "risk_score" in st.session_state:
        prob = st.session_state["risk_score"]
        st.divider()
        if prob >= 0.5:
            st.error(f"⚠️  Maintenance Required  —  Risk Score: {prob:.2f}")
            st.error(f"⚠️  Maintenance Required  —  Risk Score: {prob:.2f}")
        else:
            st.success(f"✅  No Maintenance Needed  —  Risk Score: {prob:.2f}")

    # ── AI Tools (visible only after prediction) ──────────────────────────────
    if "risk_score" in st.session_state and AGENT_AVAILABLE:
        st.divider()
        st.subheader("🤖 AI Assistant")

        ai_tab1, ai_tab2, ai_tab3 = st.tabs(["🩺 Health Report", "📅 Schedule Planner", "💬 Chat"])

        prob         = st.session_state["risk_score"]
        vehicle_data = st.session_state["vehicle_data"]

        # ── AI Tab 1: Health Report (Conditional Routing) ────────────────
        with ai_tab1:
            st.write("Generate a detailed health report. The AI **triages** the vehicle's risk and routes itself to either a Critical or Routine analysis path.")
            if st.button("Generate Health Report", key="btn_health", use_container_width=True):
                with st.spinner("🔍 Triaging risk and generating report..."):
                    try:
                        result = health_report_graph.invoke({
                            "vehicle_data": vehicle_data,
                            "risk_score": prob,
                            "retrieved_context": None,
                            "triage_result": None,
                            "report": None,
                        })
                        report_text   = result.get("report", "No report generated.")
                        triage_result = result.get("triage_result", "unknown")
                        st.session_state["previous_report"] = report_text
                        st.session_state["triage_result"]   = triage_result
                    except Exception as e:
                        st.error(f"Failed: {e}")

            # ── Show triage badge ────────────────────────────────────────
            if st.session_state.get("triage_result"):
                triage = st.session_state["triage_result"]
                if triage == "high_risk":
                    st.error(
                        "🔴 **AGENTIC ROUTING** → **CRITICAL PATH** taken  "
                        "\n_The graph detected HIGH risk / worn components and routed to Deep RAG Analysis._"
                    )
                else:
                    st.success(
                        "🟢 **AGENTIC ROUTING** → **ROUTINE PATH** taken  "
                        "\n_The graph detected LOW risk and routed to Routine Summary (no heavy RAG)._"
                    )

            if st.session_state.get("previous_report"):
                st.text_area("Health Report", st.session_state["previous_report"], height=350, key="health_report_area")

        # ── AI Tab 2: Schedule Planner (Human-in-the-Loop) ────────────────
        with ai_tab2:
            st.write("Get a 90-day service schedule. For **CRITICAL / HIGH** urgency vehicles, the AI pauses and asks for Fleet Manager approval before finalising.")

            # ── Step 1: Generate schedule ────────────────────────────────
            if st.button("Generate Schedule", key="btn_schedule", use_container_width=True):
                thread_id = str(uuid.uuid4())
                st.session_state["schedule_thread_id"] = thread_id
                st.session_state.pop("pending_schedule", None)
                st.session_state.pop("final_schedule", None)

                config = {"configurable": {"thread_id": thread_id}}
                with st.spinner("📅 Planning schedule and checking urgency..."):
                    try:
                        sched_result = schedule_graph.invoke(
                            {
                                "vehicle_data": vehicle_data,
                                "risk_score": prob,
                                "schedule": None,
                                "urgency_level": None,
                                "approval_status": None,
                                "manager_notes": None,
                                "final_schedule": None,
                            },
                            config=config,
                        )

                        if sched_result.get("final_schedule"):
                            # Auto-approved (ROUTINE/MEDIUM)
                            st.session_state["final_schedule"] = sched_result["final_schedule"]
                            st.session_state["schedule_urgency"] = sched_result.get("urgency_level", "ROUTINE")
                        else:
                            # HITL interrupt triggered — graph paused
                            st.session_state["pending_schedule"]  = sched_result.get("schedule", "")
                            st.session_state["schedule_urgency"]  = sched_result.get("urgency_level", "HIGH")
                    except Exception as e:
                        st.error(f"Schedule generation failed: {e}")

            # ── Step 2: HITL approval UI (shown when interrupt triggered) ─
            if st.session_state.get("pending_schedule") and not st.session_state.get("final_schedule"):
                urgency = st.session_state.get("schedule_urgency", "HIGH")
                st.warning(
                    f"⚠️ **HUMAN-IN-THE-LOOP** — This schedule is **{urgency}** urgency.  "
                    "\n_The AI agent has paused and is waiting for Fleet Manager approval before finalising._"
                )
                st.text_area("Proposed Schedule (pending approval)", st.session_state["pending_schedule"], height=300, key="pending_sched_area")
                manager_notes = st.text_input("Fleet Manager Notes (optional)", key="mgr_notes")

                col_approve, col_modify = st.columns(2)
                
                # Retrieve thread_id reliably
                thread_id = st.session_state.get("schedule_thread_id")
                
                with col_approve:
                    if st.button("✅ Approve Schedule", key="btn_approve", use_container_width=True):
                        if not thread_id:
                            st.error("Lost thread ID. Please generate schedule again.")
                        else:
                            config = {"configurable": {"thread_id": thread_id}}
                            with st.spinner("Resuming agent and finalising schedule..."):
                                try:
                                    resumed = schedule_graph.invoke(
                                        Command(resume={"approval": "approved", "notes": manager_notes}),
                                        config=config,
                                    )
                                    st.session_state["final_schedule"] = resumed.get("final_schedule", "Approval succeeded, but no schedule returned.")
                                    st.session_state.pop("pending_schedule", None)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Resume failed (possible session reset or rate limit): {e}")

                with col_modify:
                    if st.button("✏️ Modify & Approve", key="btn_modify", use_container_width=True):
                        if not thread_id:
                            st.error("Lost thread ID. Please generate schedule again.")
                        else:
                            config = {"configurable": {"thread_id": thread_id}}
                            with st.spinner("Resuming agent with modifications..."):
                                try:
                                    resumed = schedule_graph.invoke(
                                        Command(resume={"approval": "modified", "notes": manager_notes or "Modifications requested."}),
                                        config=config,
                                    )
                                    st.session_state["final_schedule"] = resumed.get("final_schedule", "Approval succeeded, but no schedule returned.")
                                    st.session_state.pop("pending_schedule", None)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Resume failed (possible session reset or rate limit): {e}")

            # ── Step 3: Show finalised schedule ─────────────────────────
            if st.session_state.get("final_schedule"):
                urgency = st.session_state.get("schedule_urgency", "ROUTINE")
                if urgency in ("CRITICAL", "HIGH"):
                    st.success("✅ **Schedule approved and finalised by Fleet Manager.**")
                else:
                    st.info("ℹ️ Schedule auto-approved (ROUTINE urgency — no human approval needed).")
                st.text_area("Final Service Schedule", st.session_state["final_schedule"], height=350, key="final_sched_area")

        # ── AI Tab 3: Chat (ReAct Tool-Calling Agent) ─────────────────────
        with ai_tab3:
            st.write("Ask any follow-up question. The AI **autonomously decides which tools to call** (RAG search, cost estimator, urgency checker) before answering.")

            if not st.session_state.get("previous_report"):
                st.info("Generate a Health Report first to give the assistant full context.")

            # Display chat history
            for msg in st.session_state.get("chat_history", []):
                with st.chat_message(msg["role"]):
                    st.text(msg["content"])
                    # Show tool trace for assistant messages
                    if msg["role"] == "assistant" and msg.get("tool_trace"):
                        with st.expander("🔧 Agent tool calls", expanded=False):
                            for i, tool_name in enumerate(msg["tool_trace"], 1):
                                st.markdown(f"`Step {i}` → **{tool_name}**")

            # Chat input
            user_input = st.chat_input("E.g., How much will tire replacement cost in India?")
            if user_input:
                # Derive or create a stable chat thread ID
                if "chat_thread_id" not in st.session_state:
                    st.session_state["chat_thread_id"] = str(uuid.uuid4())

                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.text(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Agent thinking and calling tools..."):
                        try:
                            reply, tool_trace = run_chat_agent(
                                vehicle_data=vehicle_data,
                                risk_score=prob,
                                previous_report=st.session_state.get("previous_report", ""),
                                user_query=user_input,
                                history=st.session_state.chat_history[:-1],
                                thread_id=st.session_state["chat_thread_id"],
                            )
                            st.text(reply)
                            if tool_trace:
                                with st.expander("🔧 Agent tool calls", expanded=True):
                                    for i, tool_name in enumerate(tool_trace, 1):
                                        st.markdown(f"`Step {i}` → **{tool_name}**")
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": reply,
                                "tool_trace": tool_trace,
                            })
                        except Exception as e:
                            st.error(f"Chat failed: {e}")

    elif "risk_score" in st.session_state and not AGENT_AVAILABLE:
        st.warning("AI features not available. Run `pip install -r requirements.txt` to enable them.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Fleet Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_fleet:
    st.subheader("Fleet Health Dashboard")
    st.write("Generate a strategic AI report across all vehicles in the fleet.")

    if not AGENT_AVAILABLE:
        st.warning("AI features not available. Run `pip install -r requirements.txt` to enable them.")
    else:
        # Simulated fleet stats (realistic synthetic snapshot for demo)
        fleet_stats = {
            "total_vehicles": 250,
            "high_risk_count": 32,
            "medium_risk_count": 89,
            "low_risk_count": 129,
            "top_vehicles": (
                "1. Truck (ID: T-402), 8 years old, Risk: 0.92 — Brake failure imminent\n"
                "2. SUV (ID: S-119), 6 years old, Risk: 0.88 — Worn tires, high mileage\n"
                "3. Van (ID: V-044), 9 years old, Risk: 0.85 — Engine knocking, no warranty"
            ),
            "vehicle_type_summary": (
                "Trucks: 80 vehicles, avg age 6 yrs | "
                "SUVs: 100 vehicles, avg age 4 yrs | "
                "Vans: 70 vehicles, avg age 5 yrs"
            ),
        }

        # Show fleet metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 High Risk", fleet_stats["high_risk_count"])
        m2.metric("🟡 Medium Risk", fleet_stats["medium_risk_count"])
        m3.metric("🟢 Low Risk", fleet_stats["low_risk_count"])

        st.divider()

        if st.button("Generate Fleet Dashboard Report", use_container_width=True):
            with st.spinner("Generating fleet-wide analysis..."):
                try:
                    result = fleet_graph.invoke({
                        "fleet_stats": fleet_stats,
                        "dashboard_report": None,
                    })
                    st.text_area(
                        "Fleet Strategy Report",
                        result.get("dashboard_report", ""),
                        height=500,
                    )
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.success(f"✅  No Maintenance Needed  —  Risk Score: {prob:.2f}")

    # ── AI Tools (visible only after prediction) ──────────────────────────────
    if "risk_score" in st.session_state and AGENT_AVAILABLE:
        st.divider()
        st.subheader("🤖 AI Assistant")

        ai_tab1, ai_tab2, ai_tab3 = st.tabs(["🩺 Health Report", "📅 Schedule Planner", "💬 Chat"])

        prob         = st.session_state["risk_score"]
        vehicle_data = st.session_state["vehicle_data"]

        # ── AI Tab 1: Health Report (Conditional Routing) ────────────────
        with ai_tab1:
            st.write("Generate a detailed health report. The AI **triages** the vehicle's risk and routes itself to either a Critical or Routine analysis path.")
            if st.button("Generate Health Report", key="btn_health", use_container_width=True):
                with st.spinner("🔍 Triaging risk and generating report..."):
                    try:
                        result = health_report_graph.invoke({
                            "vehicle_data": vehicle_data,
                            "risk_score": prob,
                            "retrieved_context": None,
                            "triage_result": None,
                            "report": None,
                        })
                        report_text   = result.get("report", "No report generated.")
                        triage_result = result.get("triage_result", "unknown")
                        st.session_state["previous_report"] = report_text
                        st.session_state["triage_result"]   = triage_result
                    except Exception as e:
                        st.error(f"Failed: {e}")

            # ── Show triage badge ────────────────────────────────────────
            if st.session_state.get("triage_result"):
                triage = st.session_state["triage_result"]
                if triage == "high_risk":
                    st.error(
                        "🔴 **AGENTIC ROUTING** → **CRITICAL PATH** taken  "
                        "\n_The graph detected HIGH risk / worn components and routed to Deep RAG Analysis._"
                    )
                else:
                    st.success(
                        "🟢 **AGENTIC ROUTING** → **ROUTINE PATH** taken  "
                        "\n_The graph detected LOW risk and routed to Routine Summary (no heavy RAG)._"
                    )

            if st.session_state.get("previous_report"):
                st.text_area("Health Report", st.session_state["previous_report"], height=350, key="health_report_area")

        # ── AI Tab 2: Schedule Planner (Human-in-the-Loop) ────────────────
        with ai_tab2:
            st.write("Get a 90-day service schedule. For **CRITICAL / HIGH** urgency vehicles, the AI pauses and asks for Fleet Manager approval before finalising.")

            # ── Step 1: Generate schedule ────────────────────────────────
            if st.button("Generate Schedule", key="btn_schedule", use_container_width=True):
                thread_id = str(uuid.uuid4())
                st.session_state["schedule_thread_id"] = thread_id
                st.session_state.pop("pending_schedule", None)
                st.session_state.pop("final_schedule", None)

                config = {"configurable": {"thread_id": thread_id}}
                with st.spinner("📅 Planning schedule and checking urgency..."):
                    try:
                        sched_result = schedule_graph.invoke(
                            {
                                "vehicle_data": vehicle_data,
                                "risk_score": prob,
                                "schedule": None,
                                "urgency_level": None,
                                "approval_status": None,
                                "manager_notes": None,
                                "final_schedule": None,
                            },
                            config=config,
                        )

                        if sched_result.get("final_schedule"):
                            # Auto-approved (ROUTINE/MEDIUM)
                            st.session_state["final_schedule"] = sched_result["final_schedule"]
                            st.session_state["schedule_urgency"] = sched_result.get("urgency_level", "ROUTINE")
                        else:
                            # HITL interrupt triggered — graph paused
                            st.session_state["pending_schedule"]  = sched_result.get("schedule", "")
                            st.session_state["schedule_urgency"]  = sched_result.get("urgency_level", "HIGH")
                    except Exception as e:
                        st.error(f"Schedule generation failed: {e}")

            # ── Step 2: HITL approval UI (shown when interrupt triggered) ─
            if st.session_state.get("pending_schedule") and not st.session_state.get("final_schedule"):
                urgency = st.session_state.get("schedule_urgency", "HIGH")
                st.warning(
                    f"⚠️ **HUMAN-IN-THE-LOOP** — This schedule is **{urgency}** urgency.  "
                    "\n_The AI agent has paused and is waiting for Fleet Manager approval before finalising._"
                )
                st.text_area("Proposed Schedule (pending approval)", st.session_state["pending_schedule"], height=300, key="pending_sched_area")
                manager_notes = st.text_input("Fleet Manager Notes (optional)", key="mgr_notes")

                col_approve, col_modify = st.columns(2)
                
                # Retrieve thread_id reliably
                thread_id = st.session_state.get("schedule_thread_id")
                
                with col_approve:
                    if st.button("✅ Approve Schedule", key="btn_approve", use_container_width=True):
                        if not thread_id:
                            st.error("Lost thread ID. Please generate schedule again.")
                        else:
                            config = {"configurable": {"thread_id": thread_id}}
                            with st.spinner("Resuming agent and finalising schedule..."):
                                try:
                                    resumed = schedule_graph.invoke(
                                        Command(resume={"approval": "approved", "notes": manager_notes}),
                                        config=config,
                                    )
                                    st.session_state["final_schedule"] = resumed.get("final_schedule", "Approval succeeded, but no schedule returned.")
                                    st.session_state.pop("pending_schedule", None)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Resume failed (possible session reset or rate limit): {e}")

                with col_modify:
                    if st.button("✏️ Modify & Approve", key="btn_modify", use_container_width=True):
                        if not thread_id:
                            st.error("Lost thread ID. Please generate schedule again.")
                        else:
                            config = {"configurable": {"thread_id": thread_id}}
                            with st.spinner("Resuming agent with modifications..."):
                                try:
                                    resumed = schedule_graph.invoke(
                                        Command(resume={"approval": "modified", "notes": manager_notes or "Modifications requested."}),
                                        config=config,
                                    )
                                    st.session_state["final_schedule"] = resumed.get("final_schedule", "Approval succeeded, but no schedule returned.")
                                    st.session_state.pop("pending_schedule", None)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Resume failed (possible session reset or rate limit): {e}")

            # ── Step 3: Show finalised schedule ─────────────────────────
            if st.session_state.get("final_schedule"):
                urgency = st.session_state.get("schedule_urgency", "ROUTINE")
                if urgency in ("CRITICAL", "HIGH"):
                    st.success("✅ **Schedule approved and finalised by Fleet Manager.**")
                else:
                    st.info("ℹ️ Schedule auto-approved (ROUTINE urgency — no human approval needed).")
                st.text_area("Final Service Schedule", st.session_state["final_schedule"], height=350, key="final_sched_area")

        # ── AI Tab 3: Chat (ReAct Tool-Calling Agent) ─────────────────────
        with ai_tab3:
            st.write("Ask any follow-up question. The AI **autonomously decides which tools to call** (RAG search, cost estimator, urgency checker) before answering.")

            if not st.session_state.get("previous_report"):
                st.info("Generate a Health Report first to give the assistant full context.")

            # Display chat history
            for msg in st.session_state.get("chat_history", []):
                with st.chat_message(msg["role"]):
                    st.text(msg["content"])
                    # Show tool trace for assistant messages
                    if msg["role"] == "assistant" and msg.get("tool_trace"):
                        with st.expander("🔧 Agent tool calls", expanded=False):
                            for i, tool_name in enumerate(msg["tool_trace"], 1):
                                st.markdown(f"`Step {i}` → **{tool_name}**")

            # Chat input
            user_input = st.chat_input("E.g., How much will tire replacement cost in India?")
            if user_input:
                # Derive or create a stable chat thread ID
                if "chat_thread_id" not in st.session_state:
                    st.session_state["chat_thread_id"] = str(uuid.uuid4())

                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.text(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Agent thinking and calling tools..."):
                        try:
                            reply, tool_trace = run_chat_agent(
                                vehicle_data=vehicle_data,
                                risk_score=prob,
                                previous_report=st.session_state.get("previous_report", ""),
                                user_query=user_input,
                                history=st.session_state.chat_history[:-1],
                                thread_id=st.session_state["chat_thread_id"],
                            )
                            st.text(reply)
                            if tool_trace:
                                with st.expander("🔧 Agent tool calls", expanded=True):
                                    for i, tool_name in enumerate(tool_trace, 1):
                                        st.markdown(f"`Step {i}` → **{tool_name}**")
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": reply,
                                "tool_trace": tool_trace,
                            })
                        except Exception as e:
                            st.error(f"Chat failed: {e}")

    elif "risk_score" in st.session_state and not AGENT_AVAILABLE:
        st.warning("AI features not available. Run `pip install -r requirements.txt` to enable them.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Fleet Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_fleet:
    st.subheader("Fleet Health Dashboard")
    st.write("Generate a strategic AI report across all vehicles in the fleet.")

    if not AGENT_AVAILABLE:
        st.warning("AI features not available. Run `pip install -r requirements.txt` to enable them.")
    else:
        # Simulated fleet stats (realistic synthetic snapshot for demo)
        fleet_stats = {
            "total_vehicles": 250,
            "high_risk_count": 32,
            "medium_risk_count": 89,
            "low_risk_count": 129,
            "top_vehicles": (
                "1. Truck (ID: T-402), 8 years old, Risk: 0.92 — Brake failure imminent\n"
                "2. SUV (ID: S-119), 6 years old, Risk: 0.88 — Worn tires, high mileage\n"
                "3. Van (ID: V-044), 9 years old, Risk: 0.85 — Engine knocking, no warranty"
            ),
            "vehicle_type_summary": (
                "Trucks: 80 vehicles, avg age 6 yrs | "
                "SUVs: 100 vehicles, avg age 4 yrs | "
                "Vans: 70 vehicles, avg age 5 yrs"
            ),
        }

        # Show fleet metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 High Risk", fleet_stats["high_risk_count"])
        m2.metric("🟡 Medium Risk", fleet_stats["medium_risk_count"])
        m3.metric("🟢 Low Risk", fleet_stats["low_risk_count"])

        st.divider()

        if st.button("Generate Fleet Dashboard Report", use_container_width=True):
            with st.spinner("Generating fleet-wide analysis..."):
                try:
                    result = fleet_graph.invoke({
                        "fleet_stats": fleet_stats,
                        "dashboard_report": None,
                    })
                    st.text_area(
                        "Fleet Strategy Report",
                        result.get("dashboard_report", ""),
                        height=500,
                    )
                except Exception as e:
                    st.error(f"Failed: {e}")