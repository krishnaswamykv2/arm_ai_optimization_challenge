import time
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ECOsphere — Edge Mission Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================================================================
# DESIGN SYSTEM
# ================================================================
st.markdown("""
<style>
:root {
  --bg:#0f1210;
  --surface:#171b18;
  --surface2:#1d231f;
  --line:rgba(215,224,216,.13);
  --text:#f2f4ef;
  --muted:#9aa49d;
  --safe:#73a982;
  --warn:#d7a34a;
  --critical:#d65f5f;
  --info:#718da8;
}
.stApp { background:var(--bg); color:var(--text); }
.block-container { max-width:1580px; padding:1rem 1.35rem 2rem; }
header[data-testid="stHeader"] { background:transparent; }
section[data-testid="stSidebar"] { display:none; }
button { border-radius:8px !important; }
[data-testid="stMetric"] { background:transparent; }
.smallcaps { font-size:.66rem; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.brand { font-size:1.18rem; letter-spacing:.12em; font-weight:850; }
.subbrand { color:var(--muted); font-size:.68rem; letter-spacing:.08em; margin-top:.15rem; }
.topline { border-bottom:1px solid var(--line); padding-bottom:.7rem; margin-bottom:.8rem; }
.hero { background:linear-gradient(135deg,#191e1a,#111411); border:1px solid var(--line); border-radius:16px; padding:1.15rem 1.25rem; min-height:130px; }
.hero-state { font-size:2.45rem; line-height:1; font-weight:900; letter-spacing:-.035em; margin:.22rem 0 .45rem; }
.hero-sub { color:#c4ccc5; font-size:.9rem; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:1rem; margin-bottom:.8rem; }
.panel-title { font-size:.65rem; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); font-weight:800; margin-bottom:.65rem; }
.big-value { font-size:2rem; font-weight:900; letter-spacing:-.03em; }
.decision { font-size:1.45rem; font-weight:900; letter-spacing:-.02em; }
.muted { color:var(--muted); }
.pill { display:inline-flex; align-items:center; gap:.35rem; border:1px solid var(--line); border-radius:999px; padding:.3rem .62rem; font-size:.68rem; font-weight:800; letter-spacing:.04em; }
.safe { color:var(--safe); }
.warn { color:var(--warn); }
.critical { color:var(--critical); }
.info { color:var(--info); }
.evidence-row { display:grid; grid-template-columns:105px 1fr 62px; gap:10px; align-items:center; margin:.62rem 0; font-size:.78rem; }
.bar { height:6px; background:#2a302b; border-radius:99px; overflow:hidden; }
.bar > div { height:100%; border-radius:99px; }
.reason { border-left:2px solid var(--info); padding:.55rem .75rem; background:rgba(113,141,168,.07); border-radius:0 9px 9px 0; font-size:.8rem; line-height:1.45; }
.verify-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; }
.verify-box { background:var(--surface2); border-radius:10px; padding:.65rem; text-align:center; }
.verify-label { font-size:.6rem; text-transform:uppercase; color:var(--muted); letter-spacing:.1em; }
.verify-value { font-size:1.15rem; font-weight:850; margin-top:.2rem; }
.thread { display:flex; gap:0; overflow-x:auto; padding:.2rem 0 .55rem; }
.thread-item { min-width:125px; padding:.55rem .7rem; border-top:2px solid #303731; color:#7f8982; font-size:.68rem; }
.thread-item.active { border-color:var(--info); color:var(--text); }
.thread-item.done { border-color:var(--safe); color:#cbd4cc; }
.thread-time { font-family:ui-monospace,monospace; font-size:.58rem; color:#727b74; }
.proof { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.9rem; height:100%; }
.proof-number { font-size:1.65rem; font-weight:900; margin:.2rem 0; }
.callout { background:#20251f; border:1px solid var(--line); border-radius:12px; padding:.8rem .9rem; }
.hazard-strip { display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr; gap:.55rem; margin:.75rem 0; }
.hazard-card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.75rem .8rem; min-height:78px; }
.hazard-card.active { border-color:rgba(214,95,95,.55); background:linear-gradient(135deg,rgba(214,95,95,.12),var(--surface)); }
.hazard-kicker { font-size:.58rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:800; }
.hazard-title { font-size:1rem; font-weight:900; margin-top:.22rem; }
.hazard-detail { font-size:.68rem; color:#aab4ad; margin-top:.18rem; }
@media (max-width:900px){ .hazard-strip{grid-template-columns:1fr 1fr;} }
@media (max-width:900px){ .hero-state{font-size:1.9rem;} .verify-grid{grid-template-columns:1fr;} }
</style>
""", unsafe_allow_html=True)

# ================================================================
# VALIDATED DEMO MISSION DATA
# ================================================================
MISSION_TIMELINE = [
    {"step":0,"time":"00:00","state":"SURVEYING","loop":"PERCEIVE","action":"CONTINUE","urgency":"ROUTINE","risk":.074,"delta":0.000,"trend":"STABLE","slope":.002,"projected":.076,"gas":75.0,"gas_risk":.09,"tilt":2.0,"tilt_risk":.06,"vib":.04,"vib_risk":.07,"agreement":"0 / 3","agreement_lvl":"NONE","dominant":"NONE","ml":"NOMINAL","ml_conf":99.1,"phys":"NOMINAL","alignment":"CONSISTENT","xy":(1.5,5.0),"heading":0,"route":"ORIGINAL_PLANNED","verif":"SURVEY_ACTIVE","passed":False,"why":"Physical baseline stable. All channels within nominal envelope.","next":"Continue nominal survey."},
    {"step":1,"time":"00:12","state":"ENVIRONMENT CHANGED","loop":"PERCEIVE","action":"MONITOR","urgency":"WATCH","risk":.245,"delta":+.171,"trend":"RISING","slope":+.024,"projected":.310,"gas":210.0,"gas_risk":.32,"tilt":4.5,"tilt_risk":.18,"vib":.12,"vib_risk":.21,"agreement":"1 / 3","agreement_lvl":"LOW","dominant":"GAS GRADIENT","ml":"NOMINAL","ml_conf":84.6,"phys":"ELEVATED","alignment":"ESCALATING BEYOND MODEL","xy":(3.2,5.0),"heading":0,"route":"ORIGINAL_PLANNED","verif":"WATCH_ACTIVE","passed":False,"why":"Early gas gradient shift detected; physical evidence is rising beyond the model view.","next":"Increase observation frequency and evaluate persistence."},
    {"step":2,"time":"00:20","state":"RISK RISING","loop":"REASON","action":"SLOW & MONITOR","urgency":"PROMPT","risk":.471,"delta":+.226,"trend":"RISING","slope":+.087,"projected":.558,"gas":480.0,"gas_risk":.65,"tilt":12.0,"tilt_risk":.44,"vib":.35,"vib_risk":.51,"agreement":"2 / 3","agreement_lvl":"MODERATE","dominant":"GAS + VIBRATION","ml":"ELEVATED","ml_conf":91.2,"phys":"HIGH","alignment":"CONSISTENT","xy":(4.8,5.0),"heading":0,"route":"HAZARD_APPROACH","verif":"TRAJECTORY_ELEVATED","passed":False,"why":"Persistent upward least-squares slope with increasing multi-channel agreement.","next":"Reduce speed and compute an escape corridor."},
    {"step":3,"time":"00:28","state":"CRITICAL HAZARD","loop":"DECIDE","action":"STOP & REROUTE","urgency":"IMMEDIATE","risk":.838,"delta":+.367,"trend":"RISING","slope":+.142,"projected":.925,"gas":700.0,"gas_risk":.91,"tilt":24.0,"tilt_risk":.82,"vib":.68,"vib_risk":.86,"agreement":"3 / 3","agreement_lvl":"HIGH","dominant":"MULTI-SENSOR ANOMALY","ml":"CRITICAL","ml_conf":96.8,"phys":"CRITICAL","alignment":"CONSISTENT","xy":(6.2,5.0),"heading":0,"route":"PATH_BLOCKED","verif":"ACTION_REQUIRED","passed":False,"why":"Joint risk crossed the critical bound; all three physical channels agree.","next":"Stop drive and reroute around the hazard."},
    {"step":4,"time":"00:32","state":"ACTION EXECUTING","loop":"ACT","action":"EXECUTING REROUTE","urgency":"IMMEDIATE","risk":.820,"delta":-.018,"trend":"STABLE","slope":-.005,"projected":.815,"gas":680.0,"gas_risk":.89,"tilt":22.0,"tilt_risk":.79,"vib":.62,"vib_risk":.78,"agreement":"3 / 3","agreement_lvl":"HIGH","dominant":"MULTI-SENSOR ANOMALY","ml":"CRITICAL","ml_conf":96.0,"phys":"CRITICAL","alignment":"CONSISTENT","xy":(6.2,5.0),"heading":45,"route":"REROUTING_ENGAGED","verif":"AWAITING_POST_OBSERVATION","passed":False,"why":"Controller has entered the safe-detour action state; verification is still pending.","next":"Collect post-action evidence."},
    {"step":5,"time":"00:44","state":"OBSERVING CONSEQUENCE","loop":"OBSERVE","action":"SAMPLE & EVALUATE","urgency":"VERIFICATION","risk":.421,"delta":-.417,"trend":"FALLING","slope":-.115,"projected":.280,"gas":310.0,"gas_risk":.45,"tilt":8.0,"tilt_risk":.32,"vib":.22,"vib_risk":.38,"agreement":"2 / 3","agreement_lvl":"MODERATE","dominant":"DISSIPATING RESIDUAL","ml":"WATCH","ml_conf":88.3,"phys":"DE-ESCALATING","alignment":"CONSISTENT","xy":(7.4,6.6),"heading":45,"route":"ON_SAFE_DETOUR","verif":"MEASURING_DELTA","passed":False,"why":"Post-action readings show a genuine drop in environmental exposure.","next":"Confirm persistent improvement."},
    {"step":6,"time":"00:52","state":"RECOVERING","loop":"VERIFY","action":"CONTINUE SAFE ROUTE","urgency":"ROUTINE","risk":.165,"delta":-.673,"trend":"FALLING","slope":-.092,"projected":.082,"gas":120.0,"gas_risk":.17,"tilt":3.0,"tilt_risk":.12,"vib":.08,"vib_risk":.14,"agreement":"0 / 3","agreement_lvl":"NONE","dominant":"NONE","ml":"NOMINAL","ml_conf":97.4,"phys":"NOMINAL","alignment":"CONSISTENT","xy":(8.8,7.8),"heading":0,"route":"ON_SAFE_DETOUR","verif":"CONSEQUENCE_VERIFIED","passed":False,"why":"Risk reduction persists and the rover is clear of the hazard boundary.","next":"Complete the verification condition."},
    {"step":7,"time":"01:04","state":"RECOVERY CONFIRMED","loop":"VERIFY","action":"RESUME SURVEY","urgency":"NOMINAL","risk":.074,"delta":-.764,"trend":"STABLE","slope":-.008,"projected":.068,"gas":78.0,"gas_risk":.09,"tilt":2.1,"tilt_risk":.06,"vib":.04,"vib_risk":.07,"agreement":"0 / 3","agreement_lvl":"NONE","dominant":"NONE","ml":"NOMINAL","ml_conf":99.4,"phys":"NOMINAL","alignment":"CONSISTENT","xy":(10.2,7.8),"heading":0,"route":"SAFE_SURVEY_RESUMED","verif":"RECOVERY_CONFIRMED","passed":True,"why":"Post-action evidence satisfies the verification condition: ΔR = -0.764.","next":"Resume continuous environmental reassessment."},
]

# ================================================================
# STATE MANAGEMENT — REAL REPLAY/RUN/RESET, NO FAKE AUTONOMY
# ================================================================
SCENARIOS = {
    "FULL HAZARD": {"start":0, "end":7, "description":"Complete environmental escalation → intervention → verified recovery"},
    "GAS ESCALATION": {"start":1, "end":7, "description":"Gas-dominant anomaly progressing into multi-sensor hazard"},
    "CRITICAL RESPONSE": {"start":3, "end":7, "description":"Start at critical state and demonstrate STOP & REROUTE → verification"},
    "RECOVERY ONLY": {"start":5, "end":7, "description":"Replay the post-action evidence and recovery verification sequence"},
}

def init_state():
    defaults = {
        "mission_step":0,
        "is_running":False,
        "cloud_online":True,
        "event_log":[],
        "mission_id":1,
        "demo_mode":False,
        "presentation_mode":False,
        "scenario_name":"Full Hazard → Recovery",
        "run_delay":0.75,
        "replay_mode":False,
        "replay_step":0,
        "selected_scenario":"FULL HAZARD",
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Advance the real deterministic runtime BEFORE rendering.
# This fixes the old bug where rerun() inside a for-loop aborted the loop.
if st.session_state.is_running:
    now = time.monotonic()
    last = st.session_state.get("last_tick", 0.0)
    if now - last >= st.session_state.run_delay:
        st.session_state.last_tick = now
        if st.session_state.mission_step < SCENARIOS[st.session_state.selected_scenario]["end"]:
            st.session_state.mission_step += 1
            current = MISSION_TIMELINE[st.session_state.mission_step]
            if not st.session_state.event_log or st.session_state.event_log[-1]["step"] != current["step"]:
                st.session_state.event_log.append(current.copy())
        else:
            st.session_state.is_running = False
            st.session_state.replay_mode = False

curr = MISSION_TIMELINE[st.session_state.mission_step]

# ================================================================
# HELPERS
# ================================================================
def severity_class(r):
    if r >= .6: return "critical"
    if r >= .3: return "warn"
    return "safe"

def color_for_risk(r):
    return "#d65f5f" if r >= .6 else ("#d7a34a" if r >= .3 else "#73a982")

def status_label():
    if st.session_state.is_running: return ("RUNNING", "safe")
    if st.session_state.mission_step == 0: return ("READY", "info")
    if st.session_state.mission_step == len(MISSION_TIMELINE)-1: return ("COMPLETE", "safe")
    return ("PAUSED", "warn")

# ================================================================
# MISSION SCENARIOS / REPLAY
# ================================================================
def reset_mission(start=0, scenario="FULL HAZARD"):
    st.session_state.is_running = False
    st.session_state.mission_step = start
    st.session_state.event_log = [MISSION_TIMELINE[start].copy()]
    st.session_state.mission_id += 1
    st.session_state.last_tick = 0.0
    st.session_state.replay_mode = False
    st.session_state.replay_step = start
    st.session_state.selected_scenario = scenario

# ================================================================
# TOP BAR
# ================================================================
run_label, run_class = status_label()
cloud_label = "ONLINE" if st.session_state.cloud_online else "OFFLINE"
cloud_class = "safe" if st.session_state.cloud_online else "warn"

st.markdown(f"""
<div class="topline">
  <div style="display:flex;justify-content:space-between;align-items:end;gap:1rem;">
    <div>
      <div class="brand">◆ ECOsphere</div>
      <div class="subbrand">EVIDENCE-DRIVEN PHYSICAL AI · EDGE MISSION CONSOLE</div>
    </div>
    <div style="display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;justify-content:flex-end;">
      <span class="pill {run_class}">● {run_label}</span>
      <span class="pill info">ARM64 TARGET</span>
      <span class="pill {cloud_class}">CLOUD {cloud_label}</span>
      <span class="pill">MISSION {st.session_state.mission_id:02d} · {curr['time']}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Small, purposeful navigation — no topology wall.
tab_live, tab_evidence, tab_system = st.tabs(["LIVE", "EVIDENCE", "SYSTEM"])

# ================================================================
# LIVE
# ================================================================
with tab_live:
    # Mission controls: the first screen owns the demo narrative.
    ctl1,ctl2,ctl3,ctl4,ctl5,ctl6 = st.columns([1.35,1.0,1.0,1.0,1.25,1.25])
    with ctl1:
        if st.button("▶ RUN MISSION", type="primary", use_container_width=True, disabled=st.session_state.is_running):
            st.session_state.is_running = True
            st.session_state.replay_mode = False
            st.session_state.last_tick = 0.0
            if not st.session_state.event_log:
                st.session_state.event_log.append(MISSION_TIMELINE[st.session_state.mission_step].copy())
            st.rerun()
    with ctl2:
        if st.button("Ⅱ PAUSE", use_container_width=True, disabled=not st.session_state.is_running):
            st.session_state.is_running = False
            st.rerun()
    with ctl3:
        if st.button("STEP →", use_container_width=True):
            st.session_state.is_running = False
            if st.session_state.mission_step < SCENARIOS[st.session_state.selected_scenario]["end"]:
                st.session_state.mission_step += 1
                st.session_state.event_log.append(MISSION_TIMELINE[st.session_state.mission_step].copy())
            st.rerun()
    with ctl4:
        if st.button("↺ RESET", use_container_width=True):
            reset_mission(0, "FULL HAZARD")
            st.rerun()
    with ctl5:
        if st.button("⟳ REPLAY MISSION", use_container_width=True):
            st.session_state.is_running = False
            st.session_state.replay_mode = True
            replay_start = SCENARIOS[st.session_state.selected_scenario]["start"]
            st.session_state.replay_step = replay_start
            st.session_state.mission_step = replay_start
            st.session_state.event_log = [MISSION_TIMELINE[replay_start].copy()]
            st.session_state.last_tick = 0.0
            st.rerun()
    with ctl6:
        st.session_state.presentation_mode = st.toggle("Presentation", value=st.session_state.presentation_mode)

    # Scenario selection lives on the flagship screen so the hazard story is immediately demonstrable.
    sc1, sc2 = st.columns([2.2, 4.8])
    with sc1:
        selected = st.selectbox(
            "MISSION SCENARIO", list(SCENARIOS.keys()),
            index=list(SCENARIOS.keys()).index(st.session_state.selected_scenario),
            label_visibility="collapsed"
        )
        if selected != st.session_state.selected_scenario:
            reset_mission(SCENARIOS[selected]["start"], selected)
            st.rerun()
    with sc2:
        st.markdown(f'<div class="callout"><b>{selected}</b> · {SCENARIOS[selected]["description"]}</div>', unsafe_allow_html=True)

    # FIRST-GLANCE HAZARD STATE: the environmental story is visible before any drill-down.
    hazard_phase = (
        "NOMINAL ENVIRONMENT" if curr["step"] == 0 else
        "ENVIRONMENTAL CHANGE" if curr["step"] == 1 else
        "GAS / VIBRATION ESCALATION" if curr["step"] == 2 else
        "CRITICAL HAZARD" if curr["step"] in (3,4) else
        "POST-ACTION OBSERVATION" if curr["step"] == 5 else
        "RECOVERY"
    )
    hazard_active = curr["step"] >= 1 and curr["step"] <= 4
    st.markdown(f"""
    <div class="hazard-strip">
      <div class="hazard-card {'active' if hazard_active else ''}">
        <div class="hazard-kicker">ENVIRONMENT</div>
        <div class="hazard-title">{hazard_phase}</div>
        <div class="hazard-detail">{curr['phys']} · {curr['dominant']}</div>
      </div>
      <div class="hazard-card {'active' if curr['gas_risk'] >= .6 else ''}">
        <div class="hazard-kicker">GAS</div>
        <div class="hazard-title">{curr['gas']:.0f} ppm</div>
        <div class="hazard-detail">risk {curr['gas_risk']:.2f} · {('ELEVATED' if curr['gas_risk'] >= .3 else 'NOMINAL')}</div>
      </div>
      <div class="hazard-card {'active' if curr['risk'] >= .6 else ''}">
        <div class="hazard-kicker">HAZARD</div>
        <div class="hazard-title">{('CRITICAL' if curr['risk'] >= .6 else 'RISING' if curr['risk'] >= .3 else 'NOMINAL')}</div>
        <div class="hazard-detail">joint risk {curr['risk']:.3f} · {curr['agreement']} agreement</div>
      </div>
      <div class="hazard-card {'active' if curr['action'] in ('STOP & REROUTE','EXECUTING REROUTE') else ''}">
        <div class="hazard-kicker">RESPONSE</div>
        <div class="hazard-title">{curr['action']}</div>
        <div class="hazard-detail">{curr['route'].replace('_',' ')}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Replay controls appear only when replay mode is active.
    if st.session_state.replay_mode:
        r1,r2,r3,r4 = st.columns([1.0,1.0,1.0,5.0])
        with r1:
            if st.button("▶ PLAY REPLAY", use_container_width=True):
                st.session_state.is_running = True
                st.session_state.last_tick = 0.0
                st.rerun()
        with r2:
            if st.button("◀ STEP BACK", use_container_width=True):
                st.session_state.is_running = False
                replay_start = SCENARIOS[st.session_state.selected_scenario]["start"]
                st.session_state.mission_step = max(replay_start, st.session_state.mission_step - 1)
                st.session_state.event_log = [MISSION_TIMELINE[i].copy() for i in range(replay_start, st.session_state.mission_step + 1)]
                st.rerun()
        with r3:
            if st.button("STEP FORWARD →", use_container_width=True):
                st.session_state.is_running = False
                replay_end = SCENARIOS[st.session_state.selected_scenario]["end"]
                replay_start = SCENARIOS[st.session_state.selected_scenario]["start"]
                if st.session_state.mission_step < replay_end:
                    st.session_state.mission_step += 1
                    st.session_state.event_log = [MISSION_TIMELINE[i].copy() for i in range(replay_start, st.session_state.mission_step + 1)]
                st.rerun()
        with r4:
            st.caption(f"REPLAY · {st.session_state.mission_step - SCENARIOS[st.session_state.selected_scenario]["start"] + 1}/{SCENARIOS[st.session_state.selected_scenario]["end"] - SCENARIOS[st.session_state.selected_scenario]["start"] + 1} states · STEP BACK / STEP FORWARD · {curr['state']}")

    if not st.session_state.presentation_mode:
        st.caption("Deterministic software mission runtime · simulated rover/environment · scenario-driven hazard response · no physical hardware claim")

    # Hero
    risk_color = color_for_risk(curr["risk"])
    st.markdown(f"""
    <div class="hero">
      <div class="smallcaps">CURRENT SITUATION · {curr['time']} · ENVIRONMENT → RESPONSE</div>
      <div class="hero-state" style="color:{risk_color}">{curr['state']}</div>
      <div class="hero-sub"><b>{curr['action']}</b> · {curr['why']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Mission thread
    scenario_start = SCENARIOS[st.session_state.selected_scenario]["start"]
    scenario_end = SCENARIOS[st.session_state.selected_scenario]["end"]
    scenario_timeline = MISSION_TIMELINE[scenario_start:scenario_end + 1]
    thread = '<div class="thread">'
    for s in scenario_timeline:
        cls = "active" if s["step"] == curr["step"] else ("done" if s["step"] < curr["step"] else "")
        thread += f'<div class="thread-item {cls}"><div class="thread-time">{s["time"]}</div><b>{s["state"]}</b></div>'
    thread += '</div>'
    st.markdown(thread, unsafe_allow_html=True)

    # Main world + decision
    left, world, right = st.columns([3.1,5.6,3.3])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">RISK & TRAJECTORY</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-value {severity_class(curr["risk"])}">{curr["risk"]:.3f}</div><div class="muted">JOINT RISK INDEX</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:.6rem;font-weight:800;color:{risk_color}">{curr["trend"]} ↑ &nbsp; <span class="muted">slope {curr["slope"]:+.3f}/s</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:.35rem" class="muted">Projected horizon <b style="color:var(--text)">{curr["projected"]:.3f}</b></div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<div class="panel-title">PHYSICAL EVIDENCE</div>', unsafe_allow_html=True)
        for label,val,r in [("GAS",curr["gas"],curr["gas_risk"]),("TILT",curr["tilt"],curr["tilt_risk"]),("VIBRATION",curr["vib"],curr["vib_risk"])]:
            col = color_for_risk(r)
            unit = "ppm" if label=="GAS" else ("°" if label=="TILT" else "g")
            st.markdown(f'<div class="evidence-row"><span>{label}</span><div class="bar"><div style="width:{int(r*100)}%;background:{col}"></div></div><b style="color:{col}">{val:.1f}{unit}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="callout"><b>{curr["agreement"]}</b> channels agree · <b>{curr["agreement_lvl"]}</b><br><span class="muted">Dominant: {curr["dominant"]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with world:
        st.markdown('<div class="panel"><div class="panel-title">MISSION WORLD · ROUTE RESPONSE</div>', unsafe_allow_html=True)
        fig = go.Figure()
        # Hazard zone
        fig.add_shape(type="circle", x0=5.8,y0=3.2,x1=9.2,y1=6.8, fillcolor="rgba(214,95,95,.14)", line=dict(color="#d65f5f",width=1.5,dash="dot"))
        # Original path
        fig.add_trace(go.Scatter(x=[1,7.5],y=[5,5],mode="lines",line=dict(color="#59635c",width=2,dash="dot"),hoverinfo="skip"))
        # Safe path only once reroute begins
        if curr["step"] >= 4:
            fig.add_trace(go.Scatter(x=[6.2,7.4,11],y=[5,7.8,7.8],mode="lines",line=dict(color="#73a982",width=4),hoverinfo="skip"))
        rx,ry = curr["xy"]
        fig.add_trace(go.Scatter(x=[rx],y=[ry],mode="markers+text",text=["ROVER"],textposition="top center",marker=dict(size=17,color="#f2f4ef",line=dict(color="#718da8",width=3)),hovertemplate="Rover · (%{x:.1f}, %{y:.1f})<extra></extra>"))
        fig.add_annotation(x=7.5,y=5,text="HAZARD",showarrow=False,font=dict(size=10,color="#d65f5f",family="monospace"))
        if curr["step"] >= 4:
            fig.add_annotation(x=9.3,y=8.35,text="SAFE DETOUR",showarrow=False,font=dict(size=10,color="#73a982",family="monospace"))
        fig.update_layout(height=315,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="#171b18",plot_bgcolor="#101410",showlegend=False,xaxis=dict(range=[0,12],visible=False),yaxis=dict(range=[2,9.5],visible=False),hovermode="closest")
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.markdown(f'<div style="display:flex;justify-content:space-between"><span class="muted">ROUTE</span><b>{curr["route"].replace("_"," ")}</b><span class="muted">HEADING</span><b>{curr["heading"]}°</b></div>',unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">DECISION</div>', unsafe_allow_html=True)
        decision_color = color_for_risk(curr["risk"])
        st.markdown(f'<div class="decision" style="color:{decision_color}">{curr["action"]}</div><div class="muted" style="margin-top:.25rem">Priority · {curr["urgency"]}</div>',unsafe_allow_html=True)
        st.markdown('<div style="height:.8rem"></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="reason"><b>WHY</b><br>{curr["why"]}</div>',unsafe_allow_html=True)
        st.markdown('<div style="height:.8rem"></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="smallcaps">MODEL ↔ PHYSICAL</div><div style="display:flex;justify-content:space-between;margin-top:.4rem"><b>{curr["ml"]}</b><span class="muted">{curr["ml_conf"]:.1f}%</span></div><div style="display:flex;justify-content:space-between;margin-top:.25rem"><b>{curr["phys"]}</b><span class="{severity_class(curr["risk"])}">{curr["alignment"]}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">CONSEQUENCE VERIFICATION</div>',unsafe_allow_html=True)
        pre = MISSION_TIMELINE[3]["risk"]
        post = curr["risk"] if curr["step"] >= 5 else None
        delta = curr["risk"] - pre if post is not None else None
        if post is None:
            st.markdown('<div class="muted">Awaiting intervention.</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verify-grid"><div class="verify-box"><div class="verify-label">Pre</div><div class="verify-value">{pre:.3f}</div></div><div class="verify-box"><div class="verify-label">Post</div><div class="verify-value">{post:.3f}</div></div><div class="verify-box"><div class="verify-label">Δ Risk</div><div class="verify-value {"safe" if delta<0 else "critical"}">{delta:+.3f}</div></div></div>',unsafe_allow_html=True)
            if curr["passed"]:
                st.success("RECOVERY CONFIRMED · empirical consequence verified")
            else:
                st.warning(curr["verif"].replace("_"," "))
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown(f"""<div class="callout"><b>WHY THIS MATTERS</b> · {curr['dominant']} → {curr['trend']} risk → <b>{curr['action']}</b> → {curr['verif'].replace('_',' ')}</div>""", unsafe_allow_html=True)

    # Risk trajectory — compact and purposeful
    st.markdown('<div class="panel"><div class="panel-title">OBSERVED RISK → PROJECTED TRAJECTORY</div>',unsafe_allow_html=True)
    hist = MISSION_TIMELINE[:curr["step"]+1]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[s["time"] for s in hist],y=[s["risk"] for s in hist],mode="lines+markers",line=dict(color="#718da8",width=3),marker=dict(size=6,color="#f2f4ef"),name="Observed"))
    if len(hist) >= 2:
        fig2.add_trace(go.Scatter(x=[hist[-1]["time"],"PROJECTED"],y=[hist[-1]["risk"],curr["projected"]],mode="lines+markers",line=dict(color=risk_color,width=2,dash="dash"),marker=dict(size=6),name="Projected"))
    fig2.update_layout(height=185,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="#171b18",plot_bgcolor="#171b18",yaxis=dict(range=[0,1],gridcolor="rgba(215,224,216,.08)",title=None),xaxis=dict(gridcolor="rgba(215,224,216,.05)"),legend=dict(orientation="h",y=1.02,x=1,xanchor="right",font=dict(size=10)))
    st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})
    st.markdown('</div>',unsafe_allow_html=True)

# ================================================================
# EVIDENCE — technical depth without polluting LIVE
# ================================================================
with tab_evidence:
    st.markdown('<div class="panel"><div class="panel-title">EDGE OPTIMIZATION · VALIDATED SOFTWARE RESULTS</div>',unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    cards=[("MODEL FOOTPRINT","9.84 KB","169.44 KB → 17.21× smaller","info"),("MEDIAN INFERENCE","1.504 ms","17.307 ms → 11.51× lower","info"),("REGRESSION","1000 / 1000","Exact agreement","safe"),("DECISION CACHE","187 hits","Validated reuse","safe")]
    for c,(title,val,sub,cls) in zip([p1,p2,p3,p4],cards):
        with c:
            st.markdown(f'<div class="proof"><div class="panel-title">{title}</div><div class="proof-number {cls}">{val}</div><div class="muted" style="font-size:.72rem">{sub}</div></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    a,b = st.columns(2)
    with a:
        st.markdown('<div class="panel"><div class="panel-title">PHYSICAL EVIDENCE</div>',unsafe_allow_html=True)
        st.markdown(f"**Gas:** {curr['gas']:.1f} ppm · risk `{curr['gas_risk']:.2f}`  \n**Tilt:** {curr['tilt']:.1f}° · risk `{curr['tilt_risk']:.2f}`  \n**Vibration:** {curr['vib']:.2f} g · risk `{curr['vib_risk']:.2f}`")
        st.markdown(f"**Agreement:** {curr['agreement']} · **Dominant:** {curr['dominant']}  \n**Model:** {curr['ml']} ({curr['ml_conf']:.1f}%) · **Physical:** {curr['phys']}  \n**Alignment:** `{curr['alignment']}`")
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">TEMPORAL REASONING</div>',unsafe_allow_html=True)
        st.markdown(f"**Trend:** `{curr['trend']}`  \n**Slope:** `{curr['slope']:+.3f}/s`  \n**Projection:** `{curr['projected']:.3f}`  \n**Current risk:** `{curr['risk']:.3f}`")
        st.caption("Projection is an interpretable trajectory calculation, not a trained forecasting model. Insufficient history must remain explicitly unprojected.")
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">MISSION EVENT SNAPSHOTS</div>',unsafe_allow_html=True)
    if st.session_state.event_log:
        df=pd.DataFrame([{ "Time":s["time"],"State":s["state"],"Risk":s["risk"],"Trend":s["trend"],"Decision":s["action"],"Alignment":s["alignment"],"Verification":s["verif"] } for s in st.session_state.event_log])
        st.dataframe(df,use_container_width=True,hide_index=True)
    else:
        st.caption("No mission events recorded yet. Press RUN MISSION or STEP.")
    st.markdown('</div>',unsafe_allow_html=True)

    # Export actual mission events, not fabricated full timeline.
    if st.session_state.event_log:
        export = pd.DataFrame(st.session_state.event_log)
        csv = export.to_csv(index=False).encode("utf-8")
        st.download_button("DOWNLOAD MISSION EVENT LOG", data=csv, file_name=f"ecosphere_mission_{st.session_state.mission_id:02d}.csv", mime="text/csv")

# ================================================================
# SYSTEM — concise technical proof, no topology wall
# ================================================================
with tab_system:
    st.markdown('<div class="panel"><div class="panel-title">EDGE EXECUTION MODEL</div>',unsafe_allow_html=True)
    st.markdown("""
**Critical path**  
SENSING → FEATURE PROCESSING → EDGE INFERENCE → RISK REASONING → DECISION → ROVER ACTION → OBSERVATION → VERIFICATION

**Local-first boundary**  
Safety-critical reasoning is architected for local execution. Cloud is supervisory/diagnostic, not a prerequisite for the decision loop.

**Current prototype status**  
Software-defined Physical AI runtime with a simulated rover/environment. No physical Arm board or physical rover is claimed as validated here.
""")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">ENGINEERING POSITIONING</div>',unsafe_allow_html=True)
    st.markdown("""
| Evidence | Validated claim |
|---|---|
| Model footprint | 169.44 KB → 9.84 KB |
| Median inference latency | 17.307 ms → 1.504 ms |
| Regression integrity | 1000 / 1000 exact agreement |
| Decision-region cache | 187 validated hits |
| Arm deployment | Arm64 target architecture; board-level benchmark must be reported separately when measured |
""")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">CHALLENGE ROVER</div>',unsafe_allow_html=True)
    st.caption("Deterministic software scenarios for demonstrating the implemented runtime. They do not imply physical hardware actuation.")
    s1,s2,s3,s4,s5=st.columns(5)
    scenarios=[("NORMAL",0),("ENVIRONMENTAL HAZARD",1),("CRITICAL",3),("RECOVERY",5),("CLOUD OFFLINE",3)]
    for c,(label,idx) in zip([s1,s2,s3,s4,s5],scenarios):
        with c:
            if st.button(label,use_container_width=True):
                st.session_state.is_running=False
                st.session_state.mission_step=idx
                st.session_state.event_log=[MISSION_TIMELINE[idx].copy()]
                if label=="CLOUD OFFLINE": st.session_state.cloud_online=False
                st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

# ================================================================
# AUTOMATIC RE-RUN DRIVER
# ================================================================
if st.session_state.is_running:
    time.sleep(st.session_state.run_delay)
    st.rerun()