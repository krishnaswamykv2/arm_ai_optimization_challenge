"""
ECOsphere — Mission Control
===========================

Final judge-facing product/demo interface.

Design:
    - Landscape-first, 16:9 desktop composition.
    - No reset/inference button for normal use.
    - Scenario selection and custom sensor controls update the live decision.
    - One-click autonomous mission replay demonstrates the complete loop.
    - Operator customization is available without exposing developer/debug UI.
    - Technical evidence is progressive disclosure, not the default experience.
    - Uses the validated ECOsphere runtime and existing baseline/optimized
      pipelines; it does not change benchmark methodology.
"""

from __future__ import annotations
from rover_visualization import render_rover_panel

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "08_master_pipeline.py"
RUNTIME_PATH = ROOT / "18_ecosphere_runtime.py"
BASELINE_MODEL = ROOT / "hazard_model.joblib"
OPTIMIZED_MODEL = ROOT / "hazard_model_optimized.joblib"

# Validated evidence — display only; not used to fabricate live measurements.
EVIDENCE = {
    "baseline_kb": 169.44,
    "optimized_kb": 9.84,
    "size_ratio": 17.21,
    "baseline_ms": 8.280,
    "optimized_ms": 0.754,
    "latency_ratio": 10.98,
    "baseline_accuracy": 99.49,
    "optimized_accuracy": 100.00,
    "telemetry_reduction": 98.74,
}

SCENARIOS = {
    "Normal": (75.0, 2.0, 0.04),
    "Gas hazard": (480.0, 3.0, 0.05),
    "Structural hazard": (90.0, 20.0, 0.45),
    "Combined critical": (700.0, 27.0, 0.70),
}

MISSION_PROFILES = {
    "Standard mission": [
        ("Normal", SCENARIOS["Normal"]),
        ("Gas hazard", SCENARIOS["Gas hazard"]),
        ("Structural hazard", SCENARIOS["Structural hazard"]),
        ("Combined critical", SCENARIOS["Combined critical"]),
        ("Recovery", SCENARIOS["Normal"]),
    ],
    "Hazard escalation": [
        ("Normal", SCENARIOS["Normal"]),
        ("Gas rise", (260.0, 3.0, 0.05)),
        ("Elevated gas", (480.0, 3.0, 0.05)),
        ("Structural instability", (480.0, 20.0, 0.45)),
        ("Critical combination", SCENARIOS["Combined critical"]),
    ],
    "Structural-first": [
        ("Normal", SCENARIOS["Normal"]),
        ("Tilt detected", (80.0, 15.0, 0.25)),
        ("Structural hazard", SCENARIOS["Structural hazard"]),
        ("Combined critical", SCENARIOS["Combined critical"]),
        ("Recovery", SCENARIOS["Normal"]),
    ],
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ECOsphere — Mission Control",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 1540px;
    padding: 1rem 2rem 2.2rem;
}
.hero {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 20px;
    padding: 1.05rem 1.35rem;
    background: linear-gradient(110deg, rgba(34,197,94,.09), rgba(56,189,248,.05));
}
.hero-title {font-size:2.25rem;font-weight:800;letter-spacing:-.05em;}
.hero-sub {opacity:.68;font-size:.95rem;}
.pill {
    display:inline-block;padding:.22rem .62rem;border-radius:999px;
    border:1px solid rgba(128,128,128,.25);font-size:.72rem;
    margin:.45rem .25rem 0 0;
}
.panel {
    border:1px solid rgba(128,128,128,.19);
    border-radius:17px;padding:1rem 1.1rem;height:100%;
}
.kicker {
    font-size:.70rem;text-transform:uppercase;letter-spacing:.11em;opacity:.55;
}
.big-state {font-size:2.35rem;font-weight:850;letter-spacing:-.05em;}
.safe {color:#22c55e}.elevated {color:#f59e0b}.critical {color:#ef4444}
.action {
    font-size:1.02rem;font-weight:700;margin-top:.2rem;
}
.flow {
    display:flex;align-items:center;gap:.32rem;flex-wrap:wrap;
    margin:.6rem 0;
}
.node {
    border:1px solid rgba(128,128,128,.22);border-radius:9px;
    padding:.38rem .52rem;font-size:.72rem;
}
.arrow {opacity:.45}
.signal {
    border-radius:12px;padding:.65rem .8rem;
    border:1px solid rgba(34,197,94,.30);
    background:rgba(34,197,94,.06);
}
.signal-warn {
    border-color:rgba(245,158,11,.35);
    background:rgba(245,158,11,.06);
}
.signal-danger {
    border-color:rgba(239,68,68,.35);
    background:rgba(239,68,68,.06);
}
.compare {
    border:1px solid rgba(128,128,128,.20);
    border-radius:14px;padding:.85rem 1rem;
}
.compare-winner {
    border-color:rgba(34,197,94,.40);
    box-shadow:inset 0 0 0 1px rgba(34,197,94,.07);
}
.big-number {font-size:1.55rem;font-weight:800;}
.muted {opacity:.58;font-size:.76rem;}
.timeline {
    border-left:2px solid rgba(128,128,128,.22);
    padding-left:1rem;margin:.4rem 0;
}
.timeline-row {padding:.3rem 0;}
.footer {text-align:center;opacity:.45;font-size:.72rem;padding-top:1rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

@st.cache_resource
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path.name}")

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
    module = load_module(RUNTIME_PATH, "ecosphere_runtime_mission_control")
    return module.ECOSphereRuntime(
        enable_cache=True,
        enable_events=True,
        enable_cloud_feedback=True,
    )


@st.cache_resource
def master_module():
    return load_module(MASTER_PATH, "ecosphere_master_mission_control")


@st.cache_resource
def baseline_pipeline():
    return master_module().build_pipeline(
        model_path=str(BASELINE_MODEL),
        use_numpy=False,
    )


@st.cache_resource
def optimized_pipeline():
    return master_module().build_pipeline(
        model_path=str(OPTIMIZED_MODEL),
        use_numpy=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = {
    "gas_value": 75.0,
    "tilt_value": 2.0,
    "vibration_value": 0.04,
    "scenario": "Normal",
    "last_input": None,
    "last_result": None,
    "mission_history": [],
    "mission_running": False,
    "cloud_online": True,
    "profile": "Standard mission",
    "speed": 0.75,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def set_scenario(name):
    st.session_state.scenario = name
    if name in SCENARIOS:
        g, t, v = SCENARIOS[name]
        st.session_state.gas_value = g
        st.session_state.tilt_value = t
        st.session_state.vibration_value = v


def state_meta(state):
    state = str(state).upper()
    if state == "CRITICAL":
        return "CRITICAL", "STOP / REROUTE", "critical", "⛔"
    if state == "ELEVATED":
        return "ELEVATED", "SLOW DOWN / MONITOR", "elevated", "⚠"
    return "SAFE", "CONTINUE MISSION", "safe", "✓"


def confidence_of(result):
    return max(
        (float(v) for v in result.get("confidence", {}).values()),
        default=0.0,
    )


def execute_current_input():
    runtime = runtime_factory()
    values = (
        float(st.session_state.gas_value),
        float(st.session_state.tilt_value),
        float(st.session_state.vibration_value),
    )
    key = tuple(round(x, 4) for x in values)

    is_new_reading = st.session_state.last_input != key

    if is_new_reading:
        result = runtime.step(*values)
        st.session_state.last_input = key
        st.session_state.last_result = result
        st.session_state.mission_history.append(
            {
                "label": st.session_state.scenario,
                "gas": values[0],
                "tilt": values[1],
                "vibration": values[2],
                "state": result["hazard_state"],
                "cache": result["cache_hit"],
                "rf": result["rf_executed"],
            }
        )

    return st.session_state.last_result, is_new_reading


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
  <div class="hero-title">🌍 ECOsphere</div>
  <div class="hero-sub">Mission Control · Autonomous Environmental Intelligence</div>
  <span class="pill">● EDGE-FIRST</span>
  <span class="pill">PHYSICAL AI</span>
  <span class="pill">LOCAL DECISION MAKING</span>
</div>
""",
    unsafe_allow_html=True,
)

top_a, top_b, top_c = st.columns([1.5, 1, 1])

with top_a:
    st.markdown("#### Live mission")
    st.caption("The interface reacts to sensor state automatically.")

with top_b:
    cloud_label = "● CLOUD CONNECTED" if st.session_state.cloud_online else "● CLOUD OFFLINE"
    st.markdown(
        f"<div style='text-align:right'><b>{cloud_label}</b></div>",
        unsafe_allow_html=True,
    )

with top_c:
    st.markdown(
        "<div style='text-align:right'><b>● EDGE OPERATIONAL</b></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Mission controls — operator-level, not developer-level
# ---------------------------------------------------------------------------

control_left, control_mid, control_right = st.columns([1.25, 1, 1])

with control_left:
    st.selectbox(
        "Mission profile",
        list(MISSION_PROFILES.keys()),
        key="profile",
        label_visibility="visible",
    )

with control_mid:
    st.select_slider(
        "Mission pace",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
        key="pace",
    )

with control_right:
    cloud_choice = st.toggle(
        "Cloud link available",
        value=st.session_state.cloud_online,
        key="cloud_online",
    )

pace_delay = {"Slow": 1.25, "Normal": 0.75, "Fast": 0.35}[st.session_state.pace]


# ---------------------------------------------------------------------------
# Scenario controls
# ---------------------------------------------------------------------------

st.markdown("### Environment")

scenario_cols = st.columns(5)
scenario_names = ["Normal", "Gas hazard", "Structural hazard", "Combined critical", "Custom"]

for idx, name in enumerate(scenario_names):
    with scenario_cols[idx]:
        st.button(
            name,
            key=f"choose_{idx}",
            use_container_width=True,
            type="primary" if st.session_state.scenario == name else "secondary",
            on_click=set_scenario if name != "Custom" else None,
            args=(name,) if name != "Custom" else (),
        )

sensor_cols = st.columns(3)

with sensor_cols[0]:
    gas = st.slider(
        "Gas concentration",
        0.0, 800.0,
        key="gas_value",
        step=1.0,
        format="%.0f ppm",
    )

with sensor_cols[1]:
    tilt = st.slider(
        "Tilt",
        0.0, 35.0,
        key="tilt_value",
        step=0.5,
        format="%.1f°",
    )

with sensor_cols[2]:
    vibration = st.slider(
        "Vibration",
        0.0, 1.0,
        key="vibration_value",
        step=0.01,
        format="%.2f g",
    )

# Current state automatically computes if input changed.
result, is_new_reading = execute_current_input()
runtime = runtime_factory()
render_rover_panel(
    result["hazard_state"],
    is_new_reading=is_new_reading,
)

state, action, state_class, icon = state_meta(result["hazard_state"])
confidence = confidence_of(result)
summary = runtime.summary()


# ---------------------------------------------------------------------------
# Hero decision
# ---------------------------------------------------------------------------

d1, d2, d3 = st.columns([1.05, 1.35, 1])

with d1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Local decision</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="big-state {state_class}">{icon} {state}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="action">{action}</div>', unsafe_allow_html=True)
    st.progress(
        min(max(confidence, 0.0), 1.0),
        text=f"Confidence · {confidence:.1%}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with d2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">How the edge decided</div>', unsafe_allow_html=True)
    rf = "RF BYPASSED" if result["cache_hit"] else "OPTIMIZED RF EXECUTED"
    rf_style = "signal" if result["cache_hit"] else "signal-warn"

    st.markdown(
        f"""
<div class="{rf_style}">
<b>⚡ {rf}</b><br>
<span class="muted">
{"The current state remained inside the same decision region." if result["cache_hit"]
 else "The current state crossed a decision boundary and required fresh inference."}
</span>
</div>
<div class="flow">
<div class="node">SENSE</div><span class="arrow">→</span>
<div class="node">FUSION</div><span class="arrow">→</span>
<div class="node">CACHE</div><span class="arrow">→</span>
<div class="node">DECIDE</div><span class="arrow">→</span>
<div class="node">ACT</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with d3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Edge state</div>', unsafe_allow_html=True)
    st.metric("Configuration", f"v{result.get('config_version', 1)}")
    st.metric("Cache hits", summary["cache_hits"])
    st.metric("Meaningful events", summary["events_generated"])
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Compute transformation
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### The same environment — radically less edge compute")

baseline = baseline_pipeline()(gas, tilt, vibration)
optimized = optimized_pipeline()(gas, tilt, vibration)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
<div class="compare">
<div class="kicker">Baseline</div>
<div class="big-number">{EVIDENCE["baseline_kb"]:.2f} KB</div>
100 trees · depth 6<br>
<span class="muted">x86 median: {EVIDENCE["baseline_ms"]:.3f} ms</span>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
<div class="compare compare-winner">
<div class="kicker">ECOsphere edge model</div>
<div class="big-number">{EVIDENCE["optimized_kb"]:.2f} KB</div>
5 trees · depth 4<br>
<span class="muted">x86 median: {EVIDENCE["optimized_ms"]:.3f} ms</span>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
<div class="compare">
<div class="kicker">Transformation</div>
<div class="big-number">{EVIDENCE["size_ratio"]:.2f}× smaller</div>
<div class="big-number">{EVIDENCE["latency_ratio"]:.2f}× faster</div>
<span class="muted">Same sensor input · same classification task</span>
</div>
""",
        unsafe_allow_html=True,
    )

match = baseline["hazard_state"] == optimized["hazard_state"]
st.success(
    "✓ Baseline and optimized models agree on the current decision."
    if match
    else "⚠ Baseline and optimized models disagree — investigate before claiming parity."
)


# ---------------------------------------------------------------------------
# Mission replay — one-click, autonomous
# ---------------------------------------------------------------------------

st.divider()
mleft, mright = st.columns([1.2, 1])

with mleft:
    st.markdown("### Autonomous mission replay")
    st.caption(
        "One click. ECOsphere moves through the selected environment profile "
        "and updates the decision, cache path, action, and event stream."
    )

    if st.button("▶ Start autonomous mission", type="primary", use_container_width=True):
        profile = MISSION_PROFILES[st.session_state.profile]
        runtime = runtime_factory()
        progress = st.progress(0, text="Mission initializing…")
        mission_box = st.empty()

        st.session_state.mission_history = []

        for idx, (label, values) in enumerate(profile):
            g, t, v = values
            live_result = runtime.step(g, t, v)
            live_state, live_action, live_class, live_icon = state_meta(
                live_result["hazard_state"]
            )

            st.session_state.mission_history.append(
                {
                    "label": label,
                    "gas": g,
                    "tilt": t,
                    "vibration": v,
                    "state": live_state,
                    "cache": live_result["cache_hit"],
                    "rf": live_result["rf_executed"],
                }
            )

            cache_text = (
                "⚡ CACHE HIT · RF BYPASSED"
                if live_result["cache_hit"]
                else "⚙ CACHE MISS · RF EXECUTED"
            )

            mission_box.markdown(
                f"""
<div class="panel">
<div class="kicker">Mission step {idx+1}/{len(profile)}</div>
<h2>{live_icon} {live_state}</h2>
<b>{label}</b><br>
Gas {g:.0f} ppm · Tilt {t:.1f}° · Vibration {v:.2f} g
<hr>
<b>{live_action}</b><br>
<span class="muted">{cache_text}</span>
</div>
""",
                unsafe_allow_html=True,
            )
            progress.progress(
                (idx + 1) / len(profile),
                text=f"{label} · {live_state}",
            )
            time.sleep(pace_delay)

        st.success("Mission complete.")

with mright:
    st.markdown("### Mission timeline")

    history = st.session_state.mission_history
    if history:
        for item in history[-8:]:
            cache = "HIT" if item["cache"] else "MISS"
            st.markdown(
                f"""
<div class="timeline">
<div class="timeline-row">
<b>{item["label"]}</b> · {item["state"]}<br>
<span class="muted">
{item["gas"]:.0f} ppm · {item["tilt"]:.1f}° · {item["vibration"]:.2f} g
· Cache {cache}
</span>
</div>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.info("Start a mission to build the live mission timeline.")


# ---------------------------------------------------------------------------
# Resilience control
# ---------------------------------------------------------------------------

st.divider()
r1, r2 = st.columns([1.25, 1])

with r1:
    st.markdown("### Edge autonomy")

    if st.session_state.cloud_online:
        st.markdown(
            """
<div class="signal">
<b>☁ CLOUD CONNECTED</b><br>
<span class="muted">
Event analysis and configuration feedback are available. Local decisions
remain independent of the cloud.
</span>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="signal-danger">
<b>☁ CLOUD OFFLINE — EDGE STILL OPERATIONAL</b><br>
<span class="muted">
Local inference, decision making, cache operation, and physical response
continue using the last valid configuration.
</span>
</div>
""",
            unsafe_allow_html=True,
        )

with r2:
    st.markdown("### Operator customization")

    with st.expander("Customize mission behaviour", expanded=False):
        st.caption(
            "These are operator controls. They do not change the trained model."
        )

        show_compute = st.toggle("Show compute details", value=True)
        show_evidence = st.toggle("Show evidence panel", value=False)
        st.selectbox(
            "Mission profile",
            list(MISSION_PROFILES.keys()),
            key="profile_duplicate",
        )

        if show_compute:
            st.info(
                "Compute path is visible: cache hit/miss and RF execution are "
                "reported from the integrated runtime."
            )

        if show_evidence:
            st.markdown(
                f"""
**Measured evidence**

- Model: {EVIDENCE["size_ratio"]:.2f}× smaller
- x86 pipeline: {EVIDENCE["latency_ratio"]:.2f}× faster
- Held-out optimized accuracy: {EVIDENCE["optimized_accuracy"]:.2f}%
- Controlled telemetry reduction: {EVIDENCE["telemetry_reduction"]:.2f}%
"""
            )


# ---------------------------------------------------------------------------
# Progressive technical evidence
# ---------------------------------------------------------------------------

with st.expander("Technical evidence — open when a judge asks"):
    tab1, tab2, tab3 = st.tabs(
        ["Optimization", "Cache + integration", "Edge-cloud"]
    )

    with tab1:
        st.markdown(
            f"""
**Model compression**

`{EVIDENCE["baseline_kb"]:.2f} KB → {EVIDENCE["optimized_kb"]:.2f} KB`

**{EVIDENCE["size_ratio"]:.2f}× smaller**

100 trees / depth 6 → 5 trees / depth 4

**Held-out validation**

Baseline: {EVIDENCE["baseline_accuracy"]:.2f}%  
Optimized: {EVIDENCE["optimized_accuracy"]:.2f}%

The held-out dataset is synthetic; this is not presented as proof of
real-world accuracy.
"""
        )

    with tab2:
        st.markdown(
            """
**Integrated regression**

- 1,000/1,000 reference-vs-integrated agreement
- 187 cache hits
- 0 observed cache-hit label mismatches
- cloud-disabled inference remained operational

A cache hit is accepted only when the feature vector remains in the same leaf
region of every tree.
"""
        )

    with tab3:
        st.markdown(
            f"""
**Selective telemetry**

Controlled experiment: **{EVIDENCE["telemetry_reduction"]:.2f}% telemetry
reduction**.

The experiment processed 1,900 readings, generated 24 events, captured all
5 meaningful state transitions, and continued edge processing during the
simulated cloud outage.

This is controlled validation, not a real-world network bandwidth claim.
"""
        )


st.markdown(
    """
<div class="footer">
ECOsphere · Mission Control · Edge-first Physical AI · measured claims only
</div>
""",
    unsafe_allow_html=True,
)