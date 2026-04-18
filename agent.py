"""
agent.py — Vehicle Maintenance AI (True Agentic LangGraph)
===========================================================
Three LangGraph graphs, each demonstrating a distinct agentic pattern:

  1. health_report_graph  — CONDITIONAL ROUTING (Triage Agent)
       ML risk score → triage node → conditional edge → HIGH or LOW branch
       HIGH: deep RAG analysis → critical action report
       LOW:  routine summary (no heavy RAG)

  2. schedule_graph       — HUMAN-IN-THE-LOOP (HITL)
       plan schedule → check cost threshold → interrupt for manager approval
       → resume on approval → finalise schedule

  3. fleet_graph          — Fleet-wide health dashboard (RAG-augmented)

  4. run_chat_agent()     — TOOL-CALLING REACT AGENT
       LLM autonomously decides which tools to call (RAG query, cost estimate,
       urgency check) before generating the final answer.

Usage:
    python agent.py   (smoke-tests all three graphs)
"""

from __future__ import annotations
from typing import TypedDict, Optional, List, Annotated
import operator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from llm_setup import get_llm
from retriever import get_retriever


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Invoke the LLM and return plain-text content."""
    return get_llm().invoke(prompt).content


# ═════════════════════════════════════════════════════════════════════════════
#  GRAPH 1 — Health Report   (PATTERN: Conditional Routing / Triage Agent)
# ═════════════════════════════════════════════════════════════════════════════
#
#  Graph topology:
#
#    retrieve_context
#         │
#         ▼
#    triage_risk          ← reads risk_score + component data
#         │
#    add_conditional_edges:
#         ├─ "high_risk" ──► node_critical_report ──► END
#         └─ "low_risk"  ──► node_routine_summary ──► END
#
# ─────────────────────────────────────────────────────────────────────────────

class HealthState(TypedDict):
    vehicle_data:       dict
    risk_score:         float
    retrieved_context:  Optional[str]
    triage_result:      Optional[str]   # "high_risk" | "low_risk"
    report:             Optional[str]


# ── Node 1: retrieve context from vector store ─────────────────────────────

def node_retrieve(state: HealthState) -> dict:
    """Retrieve relevant maintenance guidelines from ChromaDB."""
    v = state["vehicle_data"]
    risk_label = "HIGH" if state["risk_score"] >= 0.5 else "LOW"

    query = (
        f"Risk: {risk_label}. "
        f"Tire: {v.get('Tire_Condition')}. "
        f"Brake: {v.get('Brake_Condition')}. "
        f"Battery: {v.get('Battery_Status')}. "
        f"Mileage: {v.get('Mileage')} km."
    )
    docs = get_retriever().invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"retrieved_context": context}


# ── Node 2: triage router (decides the route, stores result in state) ──────

def node_triage_risk(state: HealthState) -> dict:
    """
    Classify the vehicle's risk into 'high_risk' or 'low_risk'.
    The decision factors in ML risk score AND component conditions
    so the graph can route itself intelligently.
    """
    v = state["vehicle_data"]
    risk = state["risk_score"]

    # Safety-critical component check
    critical_components = (
        v.get("Brake_Condition") == "Worn Out"
        or v.get("Tire_Condition") == "Worn Out"
        or v.get("Battery_Status") == "Weak"
    )

    triage = "high_risk" if (risk >= 0.5 or critical_components) else "low_risk"
    return {"triage_result": triage}


# ── Conditional router function (read by add_conditional_edges) ────────────

def route_by_risk(state: HealthState) -> str:
    """Return the next node name based on triage_result."""
    return state.get("triage_result", "low_risk")


# ── Node 3a: HIGH RISK — deep RAG analysis + critical report ───────────────

def node_critical_report(state: HealthState) -> dict:
    """
    HIGH RISK PATH:
    Uses the RAG-retrieved guidelines to generate a detailed critical
    action plan with strict timelines and safety notices.
    """
    v   = state["vehicle_data"]
    risk = state["risk_score"]
    ctx  = state.get("retrieved_context", "No guidelines available.")

    prompt = f"""
You are a senior fleet safety officer. The ML model has flagged this vehicle as HIGH RISK.
Your job is to generate an URGENT, detailed maintenance report.

Vehicle Profile:
{v}

ML Risk Score: {risk:.2f} (HIGH RISK — IMMEDIATE ATTENTION REQUIRED)

RAG-Retrieved Maintenance Guidelines:
{ctx}

Generate a CRITICAL HEALTH REPORT with the following sections:
1. RISK ASSESSMENT: Why this vehicle is high-risk (reference ML score and component conditions)
2. IMMEDIATE ACTIONS (within 48 hours): List each critical item with a specific deadline
3. SHORT-TERM ACTIONS (within 7 days): Secondary concerns to address
4. SAFETY NOTICE: One paragraph on operational restrictions until serviced
5. ESTIMATED DOWNTIME: How long the vehicle should be off the road

Be specific, firm, and safety-focused. Return plain text only. No markdown symbols.
"""
    return {"report": _call_llm(prompt)}


# ── Node 3b: LOW RISK — lightweight routine summary ────────────────────────

def node_routine_summary(state: HealthState) -> dict:
    """
    LOW RISK PATH:
    Generates a brief, reassuring routine maintenance summary.
    No deep RAG needed — keeps costs low and response fast.
    """
    v    = state["vehicle_data"]
    risk = state["risk_score"]

    prompt = f"""
You are a friendly fleet maintenance advisor. The ML model has assessed this vehicle as LOW RISK.
Generate a brief, reassuring routine maintenance summary.

Vehicle Profile:
{v}

ML Risk Score: {risk:.2f} (LOW RISK — Routine Maintenance)

Generate a ROUTINE HEALTH SUMMARY with:
1. HEALTH OVERVIEW: 2-3 sentences confirming the vehicle is in acceptable condition
2. RECOMMENDED ACTIONS: 2-3 routine maintenance items (e.g., next oil change, tyre rotation)
3. NEXT SCHEDULED SERVICE: Suggested timeframe
4. TIPS: One brief preventive maintenance tip

Keep it concise and positive. Return plain text only. No markdown symbols.
"""
    return {"report": _call_llm(prompt)}


# ── Graph builder ──────────────────────────────────────────────────────────

def build_health_graph():
    g = StateGraph(HealthState)

    g.add_node("retrieve",        node_retrieve)
    g.add_node("triage_risk",     node_triage_risk)
    g.add_node("critical_report", node_critical_report)
    g.add_node("routine_summary", node_routine_summary)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "triage_risk")

    # ← THE AGENTIC PART: graph decides its own route
    g.add_conditional_edges(
        "triage_risk",
        route_by_risk,
        {
            "high_risk": "critical_report",
            "low_risk":  "routine_summary",
        },
    )

    g.add_edge("critical_report", END)
    g.add_edge("routine_summary", END)
    return g.compile()


health_report_graph = build_health_graph()


# ═════════════════════════════════════════════════════════════════════════════
#  GRAPH 2 — Schedule Planner   (PATTERN: Human-in-the-Loop / HITL)
# ═════════════════════════════════════════════════════════════════════════════
#
#  Graph topology:
#
#    plan_schedule
#         │
#         ▼
#    check_cost_threshold
#         │
#    add_conditional_edges:
#         ├─ "needs_approval" ──► [INTERRUPT — wait for manager] ──► finalise_schedule ──► END
#         └─ "auto_approve"   ──► finalise_schedule ──► END
#
# ─────────────────────────────────────────────────────────────────────────────

class ScheduleState(TypedDict):
    vehicle_data:    dict
    risk_score:      float
    schedule:        Optional[str]
    urgency_level:   Optional[str]   # CRITICAL | HIGH | MEDIUM | ROUTINE
    approval_status: Optional[str]   # "pending" | "approved" | "modified"
    manager_notes:   Optional[str]
    final_schedule:  Optional[str]


# ── Node 1: Generate a 90-day schedule ────────────────────────────────────

def node_plan_schedule(state: ScheduleState) -> dict:
    """Generate a 90-day service schedule based on vehicle condition."""
    v    = state["vehicle_data"]
    risk = state["risk_score"]

    prompt = f"""
You are a fleet maintenance scheduler. Based on the vehicle data below, create a
realistic service schedule for the next 90 days.

Vehicle details:
- Model: {v.get('Vehicle_Model')}, Age: {v.get('Vehicle_Age')} years
- Days since last service: {v.get('Last_Service_Days_Ago')}
- Warranty days remaining: {v.get('Warranty_Days_Remaining')}
- Brake condition: {v.get('Brake_Condition')}
- Tire condition: {v.get('Tire_Condition')}
- Battery status: {v.get('Battery_Status')}
- Reported issues: {v.get('Reported_Issues')}
- ML Risk score: {risk:.2f} ({'HIGH' if risk >= 0.5 else 'LOW'})

Scheduling rules:
- Worn Out brakes or tires = service within 7 days (safety critical)
- Risk score > 0.7 = service within 14 days
- Days since service > 180 = schedule within 30 days
- Warranty expiring within 60 days = prioritise warranty check

Generate a SERVICE SCHEDULE with:
1. NEXT SERVICE DATE: Exact recommended date (DD/MM/YYYY) with reason
2. WHAT TO INSPECT: Specific components in priority order
3. ESTIMATED DURATION: How long the service will take
4. FOLLOW-UP: Any secondary service needed in the next 90 days
5. URGENCY LEVEL: CRITICAL / HIGH / MEDIUM / ROUTINE

Return plain text only. No markdown symbols.
"""
    return {"schedule": _call_llm(prompt)}


# ── Node 2: Check if the urgency warrants manager approval ─────────────────

def node_check_cost_threshold(state: ScheduleState) -> dict:
    """
    Determine urgency level from the schedule text.
    CRITICAL or HIGH urgency → requires fleet manager approval before finalising.
    """
    schedule_text = state.get("schedule", "")
    risk          = state["risk_score"]

    if "CRITICAL" in schedule_text.upper() or risk >= 0.7:
        urgency = "CRITICAL"
    elif "HIGH" in schedule_text.upper() or risk >= 0.5:
        urgency = "HIGH"
    elif "MEDIUM" in schedule_text.upper():
        urgency = "MEDIUM"
    else:
        urgency = "ROUTINE"

    return {"urgency_level": urgency}


def route_approval(state: ScheduleState) -> str:
    """Route: CRITICAL or HIGH urgency requires human approval; others auto-approve."""
    urgency = state.get("urgency_level", "ROUTINE")
    if urgency in ("CRITICAL", "HIGH"):
        return "needs_approval"
    return "auto_approve"


# ── Node 3: HITL interrupt — pauses graph for manager review ───────────────

def node_await_approval(state: ScheduleState) -> dict:
    """
    HUMAN-IN-THE-LOOP NODE:
    Pauses execution and surfaces the proposed schedule to the fleet manager.
    The graph will resume only when the Streamlit UI calls graph.invoke(Command(resume=...)).
    """
    # interrupt() suspends the graph here and returns control to the caller.
    # The value passed to interrupt() is visible in the pending interrupt info.
    manager_decision = interrupt({
        "message": (
            "⚠️ HIGH-COST / CRITICAL MAINTENANCE SCHEDULE requires Fleet Manager approval."
        ),
        "proposed_schedule": state.get("schedule"),
        "urgency_level":     state.get("urgency_level"),
        "vehicle_data":      state.get("vehicle_data"),
    })
    # manager_decision is whatever was passed via Command(resume=...)
    approval = manager_decision.get("approval", "approved")
    notes    = manager_decision.get("notes", "")
    return {"approval_status": approval, "manager_notes": notes}


# ── Node 4: Finalise the schedule (runs after approval or auto-approve) ────

def node_finalise_schedule(state: ScheduleState) -> dict:
    """Stamp the schedule with approval status and any manager notes."""
    base      = state.get("schedule", "")
    approval  = state.get("approval_status", "auto_approved")
    notes     = state.get("manager_notes", "")

    if approval in ("approved", "auto_approved"):
        stamp = "\n\n--- FLEET MANAGER APPROVAL ---\nStatus: APPROVED"
    else:
        stamp = f"\n\n--- FLEET MANAGER APPROVAL ---\nStatus: MODIFIED\nNotes: {notes}"

    if notes:
        stamp += f"\nNotes: {notes}"

    return {"final_schedule": base + stamp}


# ── Graph builder ──────────────────────────────────────────────────────────

# MemorySaver enables graph state to survive interrupt() → Command(resume=...)
_schedule_memory = MemorySaver()


def build_schedule_graph():
    g = StateGraph(ScheduleState)

    g.add_node("plan_schedule",         node_plan_schedule)
    g.add_node("check_cost_threshold",  node_check_cost_threshold)
    g.add_node("await_approval",        node_await_approval)
    g.add_node("finalise_schedule",     node_finalise_schedule)

    g.set_entry_point("plan_schedule")
    g.add_edge("plan_schedule", "check_cost_threshold")

    # ← THE AGENTIC PART: graph decides whether to seek human approval
    g.add_conditional_edges(
        "check_cost_threshold",
        route_approval,
        {
            "needs_approval": "await_approval",
            "auto_approve":   "finalise_schedule",
        },
    )

    g.add_edge("await_approval",   "finalise_schedule")
    g.add_edge("finalise_schedule", END)

    # Compile WITH checkpointer so interrupt() can persist state
    return g.compile(checkpointer=_schedule_memory)


schedule_graph = build_schedule_graph()


# ═════════════════════════════════════════════════════════════════════════════
#  GRAPH 3 — Fleet Dashboard
# ═════════════════════════════════════════════════════════════════════════════

class FleetState(TypedDict):
    fleet_stats:      dict
    dashboard_report: Optional[str]


def node_generate_fleet_report(state: FleetState) -> dict:
    """Generate a fleet-wide health report."""
    s = state["fleet_stats"]

    prompt = f"""
You are a senior fleet manager reviewing maintenance data for an entire vehicle fleet.

Fleet summary:
- Total vehicles: {s.get('total_vehicles')}
- High risk (score > 0.7): {s.get('high_risk_count')} vehicles
- Medium risk (score 0.4-0.7): {s.get('medium_risk_count')} vehicles
- Low risk (score < 0.4): {s.get('low_risk_count')} vehicles

Top 3 most urgent vehicles:
{s.get('top_vehicles')}

Fleet breakdown by vehicle type:
{s.get('vehicle_type_summary')}

Generate a FLEET HEALTH REPORT with:
1. FLEET OVERVIEW: One paragraph summarising the fleet's overall health status
2. URGENT ACTIONS: Specific actions for the top 3 high-risk vehicles, each with a deadline
3. FLEET TREND: What pattern do you see across vehicle types or ages?
4. COST ESTIMATE: Rough prioritisation of which vehicles need budget allocated first

Return plain text only. No markdown symbols.
"""
    return {"dashboard_report": _call_llm(prompt)}


def build_fleet_graph():
    g = StateGraph(FleetState)
    g.add_node("generate_fleet_report", node_generate_fleet_report)
    g.set_entry_point("generate_fleet_report")
    g.add_edge("generate_fleet_report", END)
    return g.compile()


fleet_graph = build_fleet_graph()


# ═════════════════════════════════════════════════════════════════════════════
#  CHAT — Tool-Calling ReAct Agent   (PATTERN: ReAct / Tool Use)
# ═════════════════════════════════════════════════════════════════════════════
#
#  Loop:
#    LLM thinks → decides to call a tool → ToolNode executes →
#    LLM reads result → decides if it has enough info → generates final answer
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Tool definitions (the LLM's "toolbox") ────────────────────────────────

@tool
def query_vector_db(query: str) -> str:
    """
    Search the vehicle maintenance guidelines database for relevant information.
    Use this when you need specific maintenance procedures, safety protocols,
    or service interval recommendations.
    Args:
        query: A natural language search query about vehicle maintenance.
    Returns:
        Relevant maintenance guidelines text.
    """
    try:
        docs = get_retriever().invoke(query)
        if not docs:
            return "No relevant guidelines found for this query."
        return "\n\n".join(f"[Guideline {i+1}]: {doc.page_content}" for i, doc in enumerate(docs))
    except Exception as e:
        return f"Vector DB query failed: {str(e)}"


@tool
def estimate_repair_cost(component: str, condition: str) -> str:
    """
    Estimate the repair or replacement cost for a vehicle component in the Indian market.
    Use this when the user asks about repair costs or budget planning.
    Args:
        component: The vehicle component (e.g., 'tires', 'brakes', 'battery', 'engine').
        condition: The current condition (e.g., 'Worn Out', 'Weak', 'Good').
    Returns:
        A cost estimate in Indian Rupees with breakdown.
    """
    cost_db = {
        "tires": {
            "Worn Out": "₹4,000–₹8,000 per tyre (budget) / ₹10,000–₹20,000 (premium). Full set: ₹16,000–₹80,000. Labour: ₹500–₹1,000.",
            "Good":     "No immediate replacement needed. Budget ₹500–₹1,000 for rotation and balancing.",
            "New":      "No cost needed. Tires are new.",
        },
        "brakes": {
            "Worn Out": "Brake pads: ₹1,500–₹4,000. Brake discs (if needed): ₹3,000–₹8,000. Full brake overhaul: ₹5,000–₹15,000.",
            "Good":     "Brake fluid flush: ₹500–₹1,000. No pad replacement needed yet.",
            "New":      "No brake costs expected in the near term.",
        },
        "battery": {
            "Weak":  "Battery replacement: ₹3,500–₹8,000 for standard. ₹10,000–₹20,000 for advanced. Labour: ₹500.",
            "Good":  "Battery top-up and terminal cleaning: ₹200–₹500.",
            "New":   "No battery costs expected.",
        },
        "engine": {
            "Poor":  "Engine overhaul: ₹50,000–₹2,00,000+ depending on damage severity. Recommend diagnostic first: ₹1,000–₹3,000.",
            "Good":  "Oil change: ₹1,500–₹3,000. Filter replacement: ₹500–₹1,500.",
            "New":   "Routine oil change only: ₹1,500–₹2,500.",
        },
    }

    comp_lower  = component.lower()
    cond_clean  = condition.strip().title()

    for key in cost_db:
        if key in comp_lower:
            estimates = cost_db[key]
            est = estimates.get(cond_clean, estimates.get("Good", "Estimate unavailable."))
            return f"Cost estimate for {component} ({condition}): {est}"

    return (
        f"General estimate for {component} repair/replacement (Indian market): "
        "₹2,000–₹25,000 depending on part and labour. "
        "Recommend getting 2-3 quotes from certified service centres."
    )


@tool
def check_urgency_level(
    brake_condition: str,
    tire_condition: str,
    battery_status: str,
    risk_score: float,
) -> str:
    """
    Classify the overall urgency level for a vehicle based on its components and ML risk score.
    Use this when the user asks about priority, urgency, or how soon they should act.
    Args:
        brake_condition: Brake condition string ('New', 'Good', or 'Worn Out').
        tire_condition:  Tire condition string ('New', 'Good', or 'Worn Out').
        battery_status:  Battery status string ('New', 'Good', or 'Weak').
        risk_score:      ML model risk score between 0.0 and 1.0.
    Returns:
        Urgency classification with recommended action timeline.
    """
    critical = (
        brake_condition == "Worn Out"
        or tire_condition == "Worn Out"
    )
    high = battery_status == "Weak" or risk_score >= 0.7
    medium = risk_score >= 0.5

    if critical:
        return (
            "URGENCY: CRITICAL\n"
            "Reason: Safety-critical components (brakes/tyres) are worn out.\n"
            "Action: Vehicle should be taken off the road immediately until serviced. "
            "Service within 24–48 hours is mandatory."
        )
    elif high:
        return (
            "URGENCY: HIGH\n"
            "Reason: Multiple risk indicators are elevated (ML score ≥ 0.7 or weak battery).\n"
            "Action: Schedule service within 7 days. Avoid long-distance or heavy-load trips."
        )
    elif medium:
        return (
            "URGENCY: MEDIUM\n"
            "Reason: ML risk score is moderate (0.5–0.7).\n"
            "Action: Schedule service within 30 days. Monitor for any new symptoms."
        )
    else:
        return (
            "URGENCY: ROUTINE\n"
            "Reason: Vehicle is in acceptable condition with low risk score.\n"
            "Action: Follow standard service schedule. Next check-up within 90 days."
        )


# ── ReAct agent builder ────────────────────────────────────────────────────

_CHAT_TOOLS = [query_vector_db, estimate_repair_cost, check_urgency_level]

_chat_memory = MemorySaver()


def _build_chat_agent():
    """Create the ReAct agent with tools bound to the LLM."""
    llm_with_tools = get_llm().bind_tools(_CHAT_TOOLS)
    return create_react_agent(
        model=llm_with_tools,
        tools=_CHAT_TOOLS,
        checkpointer=_chat_memory,
    )


_chat_agent = None


def _get_chat_agent():
    """Lazy singleton to avoid rebuilding on every call."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = _build_chat_agent()
    return _chat_agent


def run_chat_agent(
    vehicle_data: dict,
    risk_score: float,
    previous_report: str,
    user_query: str,
    history: List[dict],
    thread_id: str = "default",
) -> tuple[str, list[str]]:
    """
    Run the ReAct tool-calling agent for a user query about a vehicle.

    Returns:
        (answer: str, tool_calls_trace: list[str])
        tool_calls_trace is a list of tool names the agent called, in order.
    """
    v = vehicle_data
    system_prompt = f"""
You are an expert vehicle maintenance advisor for a fleet management system.
You have access to tools: use them to look up guidelines, estimate costs, or check urgency.
Always use at least one tool if the question is about costs, urgency, or maintenance procedures.

VEHICLE PROFILE:
- Model: {v.get('Vehicle_Model')} | Age: {v.get('Vehicle_Age')} yrs | Fuel: {v.get('Fuel_Type')}
- Mileage: {v.get('Mileage')} km | Odometer: {v.get('Odometer_Reading', 'N/A')} km
- Brake: {v.get('Brake_Condition')} | Tire: {v.get('Tire_Condition')} | Battery: {v.get('Battery_Status')}
- Maintenance history: {v.get('Maintenance_History')} | Accidents: {v.get('Accident_History')}
- Reported issues: {v.get('Reported_Issues')} | Last service: {v.get('Last_Service_Days_Ago')} days ago
- ML Risk Score: {risk_score:.2f} ({'HIGH RISK' if risk_score >= 0.5 else 'LOW RISK'})

PREVIOUSLY GENERATED REPORT:
{previous_report if previous_report else 'No report yet generated.'}

Rules:
- Be concise and reference the vehicle data above.
- For cost estimates, always use the estimate_repair_cost tool.
- For urgency questions, always use the check_urgency_level tool.
- For maintenance procedure questions, always use query_vector_db.
- Never invent facts not present in the profile or tool results.
- Return plain text only. No markdown symbols.
"""

    # Build message list
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_query))

    config = {"configurable": {"thread_id": thread_id}}
    result = _get_chat_agent().invoke({"messages": messages}, config=config)

    # Extract final answer
    final_answer = result["messages"][-1].content

    # Extract tool call trace for display in Streamlit
    tool_trace = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_trace.append(tc["name"])

    return final_answer, tool_trace


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Sample 1: HIGH RISK vehicle ──────────────────────────────────────────
    high_risk_vehicle = {
        "Vehicle_Model": "SUV",
        "Vehicle_Age": 7,
        "Mileage": 72000,
        "Fuel_Type": "Diesel",
        "Last_Service_Days_Ago": 1050,
        "Warranty_Days_Remaining": -30,
        "Tire_Condition": "Worn Out",
        "Brake_Condition": "Good",
        "Battery_Status": "Weak",
        "Maintenance_History": "Poor",
        "Accident_History": 2,
        "Reported_Issues": 4,
        "Odometer_Reading": 95000,
    }

    # ── Sample 2: LOW RISK vehicle ───────────────────────────────────────────
    low_risk_vehicle = {
        "Vehicle_Model": "Car",
        "Vehicle_Age": 2,
        "Mileage": 18000,
        "Fuel_Type": "Petrol",
        "Last_Service_Days_Ago": 60,
        "Warranty_Days_Remaining": 400,
        "Tire_Condition": "Good",
        "Brake_Condition": "New",
        "Battery_Status": "Good",
        "Maintenance_History": "Good",
        "Accident_History": 0,
        "Reported_Issues": 0,
        "Odometer_Reading": 18000,
    }

    # ── Test Graph 1: Conditional Routing ────────────────────────────────────
    print("=" * 60)
    print("GRAPH 1: Health Report — Conditional Routing")
    print("=" * 60)

    print("\n[HIGH RISK vehicle — should take CRITICAL branch]")
    res = health_report_graph.invoke({
        "vehicle_data": high_risk_vehicle,
        "risk_score": 0.82,
        "retrieved_context": None,
        "triage_result": None,
        "report": None,
    })
    print(f"Triage result : {res['triage_result']}")
    print(f"Report preview: {res['report'][:300]}...")

    print("\n[LOW RISK vehicle — should take ROUTINE branch]")
    res2 = health_report_graph.invoke({
        "vehicle_data": low_risk_vehicle,
        "risk_score": 0.18,
        "retrieved_context": None,
        "triage_result": None,
        "report": None,
    })
    print(f"Triage result : {res2['triage_result']}")
    print(f"Report preview: {res2['report'][:300]}...")

    # ── Test Graph 2: HITL Schedule (auto-approve for low risk) ──────────────
    print("\n" + "=" * 60)
    print("GRAPH 2: Schedule — HITL (auto-approve, low risk)")
    print("=" * 60)
    import uuid
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    sched_res = schedule_graph.invoke({
        "vehicle_data": low_risk_vehicle,
        "risk_score": 0.18,
        "schedule": None,
        "urgency_level": None,
        "approval_status": None,
        "manager_notes": None,
        "final_schedule": None,
    }, config=thread)
    print(f"Urgency level : {sched_res.get('urgency_level')}")
    print(f"Approval      : {sched_res.get('approval_status', 'auto_approved')}")
    print(f"Schedule preview: {sched_res.get('final_schedule', '')[:300]}...")

    # ── Test Chat ReAct Agent ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHAT: ReAct Tool-Calling Agent")
    print("=" * 60)
    answer, trace = run_chat_agent(
        vehicle_data=high_risk_vehicle,
        risk_score=0.82,
        previous_report=res["report"],
        user_query="How urgent is this vehicle's situation and what will tire replacement cost?",
        history=[],
        thread_id="smoke-test-1",
    )
    print(f"Tools used : {trace}")
    print(f"Answer     : {answer[:400]}...")
