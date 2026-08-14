"""
ECOsphere — Mission Console v2
================================
Judge-facing, human-first mission console over the validated ECOsphere runtime.

Design goal:
    Make the rover's behavior understandable before exposing technical detail.

The validated runtime, model files, benchmark methodology, and physical-AI
logic are not modified by this UI layer.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import sys
import time as _time
from pathlib import Path

import streamlit as st
from rover_visualization import render_rover_panel, reset_rover

ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "08_master_pipeline.py"
RUNTIME_PATH = ROOT / "18_ecosphere_runtime.py"
BASELINE_MODEL = ROOT / "hazard_model.joblib"
OPTIMIZED_MODEL = ROOT / "hazard_model_optimized.joblib"

EVIDENCE = {
    "baseline_kb": 169.44,
    "optimized_kb": 9.84,
    "size_ratio": 17.21,
    "baseline_ms_median": 17.307,
    "optimized_ms_median": 1.504,
    "baseline_ms_stdev": 0.495,
    "optimized_ms_stdev": 0.039,
    "latency_ratio_median": 11.51,
    "latency_reps": 10,
    "baseline_accuracy": 99.49,
    "optimized_accuracy": 100.00,
    "regression_agreement": "1000/1000",
    "cache_hits": 187,
    "cache_hit_mismatches": 0,
    "telemetry_reduction": 98.74,
    "telemetry_readings": 1900,
    "telemetry_events": 24,
    "telemetry_missed_transitions": 0,
}

SCENARIOS = {
    "Normal": (75.0, 2.0, 0.04),
    "Gas hazard": (480.0, 3.0, 0.05),
    "Structural hazard": (90.0, 20.0, 0.45),
    "Combined critical": (700.0, 27.0, 0.70),
}

MISSION = {
    "objective": "Survey the environment while responding locally to changing hazards.",
    "priority": "Safety-first coverage",
    "strategy": "Sense → understand → predict → act → verify → recover",
    "success_text": "The rover must respond to the hazard and finish in a safe state.",
    "steps": [
        ("Normal", SCENARIOS["Normal"]),
        ("Gas hazard", SCENARIOS["Gas hazard"]),
        ("Structural hazard", SCENARIOS["Structural hazard"]),
        ("Combined critical", SCENARIOS["Combined critical"]),
        ("Recovery", SCENARIOS["Normal"]),
    ],
}

st.set_page_config(
    page_title="ECOsphere — Mission Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Visual system: restrained field-mission HMI, not a generic AI dashboard.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --bg:#081019; --panel:#0d1722; --panel2:#101d2a;
  --line:rgba(148,163,184,.18); --muted:#8da0b5; --text:#edf4fa;
  --safe:#35d07f; --watch:#f4b942; --critical:#ff5d5d; --route:#58a6ff;
}
.block-container { max-width:1500px; padding:.7rem 1.15rem 1.4rem; }
* { font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
code,.mono { font-family:"SF Mono",Consolas,monospace; }
.header { display:flex; justify-content:space-between; align-items:flex-end; padding:.2rem 0 .7rem; border-bottom:1px solid var(--line); }
.brand { font-size:1.2rem; font-weight:800; letter-spacing:.12em; }
.subbrand { color:var(--muted); font-size:.68rem; letter-spacing:.08em; margin-top:.18rem; }
.live { font-size:.67rem; letter-spacing:.08em; color:#b9c7d5; text-align:right; }
.dot { color:var(--safe); }
.hero { border:1px solid rgba(148,163,184,.25); background:linear-gradient(180deg,rgba(16,29,42,.9),rgba(9,17,25,.95)); border-radius:8px; padding:1rem; margin:.8rem 0 .65rem; }
.kicker { text-transform:uppercase; letter-spacing:.14em; color:#71869b; font-size:.62rem; font-weight:700; }
.headline { font-size:2.45rem; font-weight:850; line-height:1.0; margin:.25rem 0; }
.intent { font-size:.98rem; color:#d9e5ee; max-width:850px; line-height:1.45; }
.state-safe { color:var(--safe); } .state-watch { color:var(--watch); } .state-critical { color:var(--critical); } .state-route { color:var(--route); }
.panel { border:1px solid var(--line); background:rgba(13,23,34,.72); border-radius:7px; padding:.75rem .85rem; height:100%; }
.panel-title { font-size:.62rem; text-transform:uppercase; letter-spacing:.13em; color:#71869b; font-weight:700; margin-bottom:.45rem; }
.big-number { font-size:1.65rem; font-weight:800; }
.sensor-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem; }
.sensor { border:1px solid var(--line); border-radius:6px; padding:.55rem .6rem; background:rgba(255,255,255,.018); }
.sensor-name { font-size:.6rem; color:#71869b; letter-spacing:.12em; text-transform:uppercase; }
.sensor-value { font-size:1.1rem; font-weight:750; margin-top:.12rem; }
.explain { border-left:3px solid var(--route); padding:.6rem .75rem; background:rgba(88,166,255,.055); border-radius:0 6px 6px 0; }
.explain b { color:#e7f2ff; }
.next { border-left:3px solid var(--safe); padding:.6rem .75rem; background:rgba(53,208,127,.045); border-radius:0 6px 6px 0; }
.trace { display:grid; grid-template-columns:repeat(6,1fr); gap:0; margin:.55rem 0; }
.trace-step { border-top:2px solid var(--line); padding:.48rem .45rem .2rem 0; min-height:3rem; }
.trace-step b { display:block; font-size:.59rem; letter-spacing:.08em; color:#c7d3df; }
.trace-step span { display:block; font-size:.62rem; color:#72879b; margin-top:.15rem; }
.timeline { display:flex; gap:.6rem; align-items:flex-start; padding:.45rem 0; border-bottom:1px solid rgba(148,163,184,.10); }
.timeline-dot { width:9px; height:9px; border-radius:50%; margin-top:.28rem; flex:0 0 auto; background:#64748b; }
.timeline-main { font-size:.72rem; } .timeline-sub { color:#71869b; font-size:.62rem; margin-top:.1rem; }
.proof { border:1px solid var(--line); border-radius:7px; padding:.7rem; background:rgba(255,255,255,.018); }
.proof-value { font-size:1.3rem; font-weight:800; }
.footer { text-align:center; color:#526476; font-size:.6rem; padding-top:.8rem; }
div[data-testid="stTabs"] button { font-size:.68rem; letter-spacing:.1em; }
div[data-testid="stButton"] button { border-radius:6px; min-height:2.35rem; }
div[data-testid="stMetricValue"] { font-size:1rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@st.cache_resource
def runtime_factory():
    module = load_module(str(RUNTIME_PATH), "ecosphere_runtime_field_console_v2")
    return module.ECOSphereRuntime(enable_cache=True, enable_events=True, enable_cloud_feedback=True)


@st.cache_resource
def master_module():
    return load_module(str(MASTER_PATH), "ecosphere_master_field_console_v2")


@st.cache_resource
def baseline_pipeline():
    return master_module().build_pipeline(model_path=str(BASELINE_MODEL), use_numpy=False)


@st.cache_resource
def optimized_pipeline():
    return master_module().build_pipeline(model_path=str(OPTIMIZED_MODEL), use_numpy=True)


def clear_runtime_caches():
    runtime_factory.clear()
    baseline_pipeline.clear()
    optimized_pipeline.clear()


DEFAULTS = {
    "gas_value": 75.0, "tilt_value": 2.0, "vibration_value": 0.04,
    "last_input": None, "last_result": None, "mission_history": [],
    "cloud_online": True, "pace": "Normal", "last_cloud_update": None,
    "pending_sensor_values": None, "autonomous_trace": [],
    "autonomous_pace": "Normal", "autonomous_complete": False,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

PACE_DELAY = {"Slow": 1.25, "Normal": 0.75, "Fast": 0.35}


def active_profile_label(gas, tilt, vibration):
    current = (round(gas, 2), round(tilt, 2), round(vibration, 2))
    for name, values in SCENARIOS.items():
        if tuple(round(v, 2) for v in values) == current:
            return name
    return "Custom"


def apply_scenario(name):
    g, t, v = SCENARIOS[name]
    st.session_state.gas_value = g
    st.session_state.tilt_value = t
    st.session_state.vibration_value = v


def risk_class(level):
    level = str(level).upper()
    if level == "CRITICAL": return "state-critical"
    if level in {"HIGH", "MODERATE"}: return "state-watch"
    return "state-safe"


def hazard_class(state):
    state = str(state).upper()
    if state == "CRITICAL": return "state-critical"
    if state == "ELEVATED": return "state-watch"
    return "state-safe"


def confidence_of(result):
    return max((float(v) for v in result.get("confidence", {}).values()), default=0.0)


def action_phrase(decision):
    action = decision.get("action", "CONTINUE")
    return {
        "CONTINUE": "Continue surveying the environment.",
        "MONITOR": "Continue carefully while watching the environment.",
        "SLOW_AND_MONITOR": "Reduce movement speed while the hazard is escalating.",
        "PAUSE_AND_VERIFY": "Pause movement and verify the physical evidence.",
        "STOP_AND_REROUTE": "Stop now and move around the hazardous area.",
    }.get(action, str(action))


def next_phrase(result):
    decision = result.get("decision", {})
    verification = result.get("verification")
    recovery = result.get("recovery")
    if verification and verification.get("next_action") == "ESCALATE":
        return "Escalate the response because the previous intervention did not reduce the hazard."
    if recovery and recovery.get("recovery_confirmed"):
        return "Recovery is confirmed; safe movement can resume."
    if decision.get("action") == "STOP_AND_REROUTE":
        return "Wait for the hazard to clear, then verify recovery before resuming."
    if decision.get("verification_required"):
        return "Observe the next reading and verify whether the intervention worked."
    return "Continue sensing the environment."


def why_phrase(result):
    ra = result.get("risk_assessment", {})
    rationale = ra.get("rationale", [])
    if rationale:
        return rationale[0] if len(rationale) == 1 else " ".join(rationale[:3])
    env = result.get("environment", {})
    return env.get("context", "Environmental conditions are being evaluated locally.")


def execute_current_input():
    runtime = runtime_factory()
    values = (float(st.session_state.gas_value), float(st.session_state.tilt_value), float(st.session_state.vibration_value))
    key = tuple(round(x, 4) for x in values)
    is_new_reading = st.session_state.last_input != key
    if is_new_reading:
        events_before = len(runtime.event_log)
        result = runtime.step(*values)
        st.session_state.last_input = key
        st.session_state.last_result = result
        st.session_state.mission_history.append({
            "label": active_profile_label(*values), "gas": values[0], "tilt": values[1],
            "vibration": values[2], "state": result["hazard_state"],
            "risk": result["environment"]["risk_features"]["joint_risk"],
            "action": result["decision"]["action"], "cache": result["cache_hit"],
        })
        new_events = len(runtime.event_log) > events_before
        if st.session_state.cloud_online and new_events:
            applied, config = runtime.get_cloud_feedback()
            if applied:
                st.session_state.last_cloud_update = config
    return st.session_state.last_result, is_new_reading


def reset_mission():
    clear_runtime_caches()
    reset_rover()
    for key, value in DEFAULTS.items():
        st.session_state[key] = deepcopy(value)
    st.session_state.gas_value, st.session_state.tilt_value, st.session_state.vibration_value = SCENARIOS["Normal"]


def mission_success(trace):
    return bool(trace) and trace[-1]["state"] == "SAFE"


runtime = runtime_factory()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="header">
  <div><div class="brand">◆ ECOSPHERE</div>
  <div class="subbrand">AUTONOMOUS ENVIRONMENTAL FIELD UNIT · MISSION CONSOLE</div></div>
  <div class="live"><span class="dot">●</span> EDGE ACTIVE<br>
  {'CLOUD LINK ACTIVE' if st.session_state.cloud_online else 'CLOUD OFFLINE · LOCAL MODE'}</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_live, tab_mission, tab_prove = st.tabs(["LIVE MISSION", "MISSION CONTROL", "PROVE"])

# ---------------------------------------------------------------------------
# LIVE MISSION — the judge-facing story
# ---------------------------------------------------------------------------
with tab_live:
    pending = st.session_state.get("pending_sensor_values")
    if pending is not None:
        st.session_state.gas_value, st.session_state.tilt_value, st.session_state.vibration_value = pending
        st.session_state.pending_sensor_values = None

    result, is_new_reading = execute_current_input()
    state = result["hazard_state"]
    decision = result["decision"]
    risk = result["environment"]["risk_features"]["joint_risk"]
    risk_level = result["risk_assessment"]["risk_level"]
    temporal = result["temporal"]
    projection = result["projection"]
    rover_state = result.get("rover_state", {})
    verification = result.get("verification")
    recovery = result.get("recovery")

    # Hero: answer the five human questions immediately.
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">WHAT IS HAPPENING?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="headline {hazard_class(state)}">{state}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="intent"><b>ROVER:</b> {action_phrase(decision)} &nbsp;·&nbsp; <b>NEXT:</b> {next_phrase(result)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    rover_col, info_col = st.columns([1.45, 1])
    with rover_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">ROVER · CURRENT BEHAVIOR</div>', unsafe_allow_html=True)
        render_rover_panel(state, is_new_reading=is_new_reading)
        if rover_state:
            st.caption(f"Physical action state · {rover_state.get('motion_state','—')} · {rover_state.get('speed_mode','—')} · {rover_state.get('route_state','—')}")
        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">WHY DID IT DO THAT?</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="explain"><b>{decision.get("trigger","SYSTEM_STATE")}</b><br>{why_phrase(result)}</div>', unsafe_allow_html=True)
        st.markdown('<br><div class="panel-title">WHAT HAPPENS NEXT?</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="next">{next_phrase(result)}</div>', unsafe_allow_html=True)
        st.markdown('<br><div class="panel-title">DECISION</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-number {risk_class(risk_level)}">{decision.get("action","—")}</div>', unsafe_allow_html=True)
        st.caption(f"Priority · {decision.get('priority','—')} · Verification required · {'YES' if decision.get('verification_required') else 'NO'}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Environment: semantic first, numbers second.
    st.markdown('<br><div class="panel-title">ENVIRONMENT</div>', unsafe_allow_html=True)
    env = result["environment"]
    ev = env["evidence"]
    vals = result["corrected_input"]
    st.markdown(
        f"""
<div class="sensor-grid">
  <div class="sensor"><div class="sensor-name">Gas</div><div class="sensor-value">{vals['gas_ppm']:.0f} ppm · {ev['gas']}</div></div>
  <div class="sensor"><div class="sensor-name">Tilt</div><div class="sensor-value">{vals['tilt_deg']:.1f}° · {ev['tilt']}</div></div>
  <div class="sensor"><div class="sensor-name">Vibration</div><div class="sensor-value">{vals['vibration_g']:.2f} g · {ev['vibration']}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Physical risk", f"{risk:.3f}")
    c2.metric("Trend", temporal.get("trend", "—"))
    c3.metric("Near-term", projection.get("projected_level", "—"))
    c4.metric("Sensor agreement", env.get("sensor_agreement", "—"))

    st.markdown('<div class="trace">', unsafe_allow_html=True)
    stages = [
        ("01 · SENSE", "raw environment"),
        ("02 · UNDERSTAND", env.get("context", "—")),
        ("03 · PREDICT", projection.get("trajectory", "—")),
        ("04 · DECIDE", decision.get("action", "—")),
        ("05 · ACT", rover_state.get("motion_state", "—") if rover_state else "—"),
        ("06 · VERIFY", (verification or {}).get("verification", "awaiting next reading")),
    ]
    for title, text in stages:
        st.markdown(f'<div class="trace-step"><b>{title}</b><span>{text}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if recovery:
        st.markdown(f"**RECOVERY:** `{recovery.get('state','—')}` · {recovery.get('falling_count',0)}/{recovery.get('required_observations',2)} falling observations · {'CONFIRMED' if recovery.get('recovery_confirmed') else 'VERIFYING'}")

    with st.expander("Technical detail", expanded=False):
        st.json({
            "hazard_state": state,
            "risk_assessment": result["risk_assessment"],
            "temporal": temporal,
            "projection": projection,
            "decision": decision,
            "rover_state": rover_state,
            "verification": verification,
            "recovery": recovery,
            "cache_hit": result["cache_hit"],
            "rf_executed": result["rf_executed"],
        })

    with st.expander("Test an environment", expanded=False):
        st.caption("For controlled demonstration/testing. The values below still pass through the real ECOsphere runtime.")
        q = st.columns(4)
        for idx, name in enumerate(SCENARIOS):
            with q[idx]:
                st.button(name, key=f"live_scen_{idx}", use_container_width=True, on_click=apply_scenario, args=(name,))
        s1, s2, s3 = st.columns(3)
        with s1: st.slider("Gas", 0.0, 800.0, key="gas_value", step=1.0, format="%.0f ppm")
        with s2: st.slider("Tilt", 0.0, 35.0, key="tilt_value", step=0.5, format="%.1f°")
        with s3: st.slider("Vibration", 0.0, 1.0, key="vibration_value", step=0.01, format="%.2f g")

# ---------------------------------------------------------------------------
# MISSION CONTROL — one-click story, replay, connectivity, reset
# ---------------------------------------------------------------------------
with tab_mission:
    st.markdown('<div class="kicker">MISSION OBJECTIVE</div>', unsafe_allow_html=True)
    st.markdown(f"### Environmental survey")
    st.markdown(f"{MISSION['objective']}  \n**Priority:** {MISSION['priority']}  ·  **Strategy:** {MISSION['strategy']}")

    a, b = st.columns([1.2, 1])
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">AUTONOMOUS RUN</div>', unsafe_allow_html=True)
        st.markdown("**Press once. Watch the whole loop.**")
        pace = st.select_slider("Demo pace", options=["Slow", "Normal", "Fast"], key="pace", label_visibility="collapsed")
        if st.button("▶  START AUTONOMOUS MISSION", type="primary", use_container_width=True):
            clear_runtime_caches()
            reset_rover()
            runtime = runtime_factory()
            st.session_state.mission_history = []
            st.session_state.last_result = None
            st.session_state.last_input = None
            st.session_state.last_cloud_update = None
            st.session_state.autonomous_trace = []
            st.session_state.autonomous_pace = pace
            st.session_state.autonomous_complete = False

            progress = st.progress(0, text="Mission initializing…")
            hero_box = st.empty()
            detail_box = st.empty()
            for idx, (label, values) in enumerate(MISSION["steps"]):
                g, t, v = values
                events_before = len(runtime.event_log)
                live = runtime.step(g, t, v)
                if st.session_state.cloud_online and len(runtime.event_log) > events_before:
                    applied, config = runtime.get_cloud_feedback()
                    if applied: st.session_state.last_cloud_update = config

                live_state = live["hazard_state"]
                live_decision = live["decision"]
                item = {
                    "step": idx + 1, "label": label, "gas": g, "tilt": t, "vibration": v,
                    "state": live_state, "action": live_decision["action"],
                    "risk": live["environment"]["risk_features"]["joint_risk"],
                    "trend": live["temporal"]["trend"],
                    "verification": live.get("verification"),
                    "recovery": live.get("recovery"),
                    "cache": live["cache_hit"], "rf": live["rf_executed"],
                }
                st.session_state.autonomous_trace.append(item)
                st.session_state.mission_history.append(item)

                with hero_box.container():
                    st.markdown(f'<div class="hero"><div class="kicker">MISSION STEP {idx+1}/{len(MISSION["steps"])}</div><div class="headline {hazard_class(live_state)}">{live_state}</div><div class="intent"><b>{action_phrase(live_decision)}</b></div></div>', unsafe_allow_html=True)
                    render_rover_panel(live_state, is_new_reading=True)
                detail_box.markdown(f"**{label}** · risk `{item['risk']:.3f}` · trend `{item['trend']}` · decision `{item['action']}`")
                progress.progress((idx + 1) / len(MISSION["steps"]), text=f"{label} · {live_state}")
                _time.sleep(PACE_DELAY[pace])

            st.session_state.pending_sensor_values = MISSION["steps"][-1][1]
            st.session_state.last_input = None
            st.session_state.autonomous_complete = True
            success = mission_success(st.session_state.autonomous_trace)
            st.success("MISSION COMPLETE — recovery state reached SAFE." if success else "MISSION FINISHED — inspect the trace.")

    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">MISSION MEMORY</div>', unsafe_allow_html=True)
        trace = st.session_state.autonomous_trace
        if trace:
            for item in trace:
                dot = "#35d07f" if item["state"] == "SAFE" else "#f4b942" if item["state"] == "ELEVATED" else "#ff5d5d"
                st.markdown(f'<div class="timeline"><span class="timeline-dot" style="background:{dot}"></span><div><div class="timeline-main"><b>{item["label"]}</b> · {item["state"]} · {item["action"]}</div><div class="timeline-sub">risk {item["risk"]:.3f} · trend {item["trend"]}</div></div></div>', unsafe_allow_html=True)
        else:
            st.caption("No mission recorded yet. Start the autonomous mission to build the timeline.")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.autonomous_complete and st.session_state.autonomous_trace:
        r1, r2 = st.columns(2)
        with r1:
            if st.button("↻  REPLAY LAST MISSION", use_container_width=True):
                trace = st.session_state.autonomous_trace
                saved = {k: st.session_state.get(k) for k in ["rover_position", "rover_reroute_points", "rover_stopped"]}
                reset_rover()
                box = st.empty()
                for item in trace:
                    with box.container():
                        st.markdown(f'<div class="hero"><div class="kicker">REPLAY · STEP {item["step"]}</div><div class="headline {hazard_class(item["state"])}">{item["state"]}</div><div class="intent"><b>{item["action"]}</b> · risk {item["risk"]:.3f}</div></div>', unsafe_allow_html=True)
                        render_rover_panel(item["state"], is_new_reading=True)
                    _time.sleep(PACE_DELAY[st.session_state.autonomous_pace])
                for k, v in saved.items():
                    if v is not None: st.session_state[k] = v
        with r2:
            if st.button("⟲  RESET MISSION", use_container_width=True, on_click=reset_mission):
                pass

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel-title">CONNECTIVITY</div>', unsafe_allow_html=True)
        st.toggle("Cloud feedback link", key="cloud_online")
        if st.session_state.cloud_online:
            st.info("Cloud feedback is available; local inference remains independent.")
        else:
            st.warning("Cloud offline. Local inference continues and the last edge configuration is retained.")
    with c2:
        st.markdown('<div class="panel-title">MISSION SUCCESS CONDITION</div>', unsafe_allow_html=True)
        st.markdown("**Response:** hazard action must occur.  \n**Recovery:** environment must return to SAFE.  \n**Proof:** recovery is tracked across observations, not assumed from one reading.")

# ---------------------------------------------------------------------------
# PROVE — measured engineering evidence, progressively disclosed
# ---------------------------------------------------------------------------
with tab_prove:
    st.caption("Measured claims only. x86 measurements are from the documented validation runs, not ARM/QRB2210 field measurements.")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown('<div class="proof"><div class="panel-title">EDGE MODEL</div><div class="proof-value">17.21× smaller</div><div class="mono">169.44 KB → 9.84 KB</div><br>Held-out accuracy: 99.49% → 100.00% on the stated synthetic dataset.</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="proof"><div class="panel-title">PIPELINE LATENCY · x86</div><div class="proof-value">11.51× median improvement</div><div class="mono">17.307 ms → 1.504 ms</div><br>10 independent repetitions; randomized order per repetition.</div>', unsafe_allow_html=True)
    with p3:
        st.markdown('<div class="proof"><div class="panel-title">CACHE REGRESSION</div><div class="proof-value">1000 / 1000</div><div class="mono">reference agreement</div><br>187 cache hits · 0 label mismatches.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    p4, p5 = st.columns(2)
    with p4:
        st.markdown('<div class="proof"><div class="panel-title">SELECTIVE TELEMETRY</div><div class="proof-value">98.74% reduction</div>24 events from 1900 readings · 0 missed meaningful transitions in the controlled stream.</div>', unsafe_allow_html=True)
    with p5:
        st.markdown('<div class="proof"><div class="panel-title">CLOSED LOOP</div><div class="proof-value">SENSE → ACT → VERIFY → RECOVER</div>Action verification and recovery tracking are exercised by the integrated runtime.</div>', unsafe_allow_html=True)

    with st.expander("Show validation details", expanded=False):
        st.json(EVIDENCE)
    with st.expander("What is simulated vs. physical?", expanded=False):
        st.markdown("The current Field Console demonstrates the validated software decision loop and a simulated rover state/route visualization. Hardware actuation should only be claimed once the hardware adapter is connected and tested on the real rover.")

st.markdown('<div class="footer">ECOsphere · Mission Console · human-first interface · measured claims only</div>', unsafe_allow_html=True)
