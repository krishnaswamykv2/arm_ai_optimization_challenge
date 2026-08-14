"""
ECOsphere — Field Console
=========================

Judge-facing product/demo interface for an edge-first Physical AI system.

Rewrite notes (from the prior version):
    - Cloud Online/Offline now has a real effect: cloud-feedback analysis
      (runtime.get_cloud_feedback()) is only invoked while the toggle is
      on. Flip it off and the edge configuration genuinely freezes at
      its last value, even while new hazard events keep accumulating --
      matching what 17_cloud_feedback_engine.py / 19_integration_regression.py
      actually validate. Previously the toggle only changed display text.
    - Added a real Reset Mission control. It clears the cached runtime
      and pipeline singletons (st.cache_resource) as well as rover state
      and session history, so "reset" means a genuinely fresh runtime,
      not just a cleared display.
    - Removed two dead controls: the "Custom" scenario button (did
      nothing on click) and the duplicate "Mission profile" selectbox in
      Operator Customization (its value was read nowhere). The active
      profile label is now computed live from the current sensor values
      instead, so manually dragging a slider away from a preset
      correctly shows "Custom" without needing a separate button.
    - Fixed a Streamlit widget bug: the cloud toggle was constructed with
      both `value=` and a session_state-backed `key=`, which Streamlit
      flags as a contradiction on every rerun.
    - PROVE tab numbers are sourced directly from the latest
      11_verify_benchmarks.py / 19_integration_regression.py /
      16_edge_event_intelligence.py runs (see EVIDENCE below), not
      estimated -- and now show the repeated-trial latency distribution
      (median + spread) rather than a single number, since single-block
      x86 timings on a shared machine vary run to run.
    - Restructured into three views per the product spec: OPERATE
      (default -- perceive/decide/act loop), MISSION (scenario/profile
      control, cloud link, reset), PROVE (collapsible engineering
      evidence). No ML/runtime/benchmark methodology was touched.

Design rules preserved from the original:
    - Uses the validated ECOsphere runtime and existing baseline/optimized
      pipelines; does not change benchmark methodology.
    - 08_master_pipeline.py, hazard_model.joblib, hazard_model_optimized.joblib,
      18_ecosphere_runtime.py are not modified.
"""

from __future__ import annotations

from copy import deepcopy
from rover_visualization import render_rover_panel, reset_rover

import importlib.util
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "08_master_pipeline.py"
RUNTIME_PATH = ROOT / "18_ecosphere_runtime.py"
BASELINE_MODEL = ROOT / "hazard_model.joblib"
OPTIMIZED_MODEL = ROOT / "hazard_model_optimized.joblib"

# ---------------------------------------------------------------------------
# Validated evidence -- display only, sourced from actual script runs:
#   size/accuracy   : 11_verify_benchmarks.py
#   latency         : 11_verify_benchmarks.py, 10 independent repetitions,
#                      400 timed calls/rep, x86, config order randomized
#                      per repetition (see that script for methodology)
#   cache/regression: 19_integration_regression.py (1000-reading stream)
#   telemetry       : 16_edge_event_intelligence.py (1900-reading, 5-regime
#                      persistent stream)
# Never presented as ARM/QRB2210 numbers or as real-world field data.
# ---------------------------------------------------------------------------
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
    page_title="ECOsphere — Field Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Flat, utilitarian console styling -- no gradients, no glow, no cards-on-cards.
st.markdown(
    """
<style>
.block-container { max-width:1480px; padding:.55rem 1.15rem 1rem; }
* { font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
code,.mono-small,.status-line { font-family:"SF Mono",Consolas,monospace; }
.console-header { display:flex; align-items:center; justify-content:space-between; padding:.15rem 0 .55rem; margin-bottom:.45rem; border-bottom:1px solid rgba(148,163,184,.24); }
.console-title { font-size:1.12rem; font-weight:760; letter-spacing:.08em; }
.console-sub { font-size:.72rem; opacity:.52; }
.console-mark { font-size:.66rem; letter-spacing:.12em; opacity:.48; }
.status-line { font-size:.68rem; text-align:right; opacity:.72; }
.mission-strip { display:flex; align-items:center; justify-content:space-between; border:1px solid rgba(148,163,184,.22); border-radius:4px; padding:.45rem .65rem; margin:.2rem 0 .45rem; background:rgba(148,163,184,.035); }
.mission-strip b { letter-spacing:.06em; font-size:.72rem; }
.mission-strip span { font-size:.66rem; opacity:.55; }
.panel { border:1px solid rgba(148,163,184,.22); border-radius:4px; padding:.65rem .75rem; height:100%; background:rgba(148,163,184,.025); }
.hero-panel { border:1px solid rgba(148,163,184,.28); border-radius:5px; padding:.45rem .55rem .2rem; background:rgba(15,23,42,.12); }
.kicker { font-size:.6rem; text-transform:uppercase; letter-spacing:.13em; opacity:.48; margin-bottom:.25rem; }
.big-state { font-size:2.05rem; font-weight:800; line-height:1; letter-spacing:.015em; }
.safe { color:#22c55e } .elevated { color:#f59e0b } .critical { color:#ef4444 }
.action { font-size:.78rem; font-weight:700; margin-top:.22rem; opacity:.82; letter-spacing:.04em; }
.signal { border-radius:4px; padding:.42rem .55rem; border:1px solid rgba(34,197,94,.32); background:rgba(34,197,94,.045); }
.signal-warn { border-color:rgba(245,158,11,.34); background:rgba(245,158,11,.045); }
.signal-danger { border-color:rgba(239,68,68,.34); background:rgba(239,68,68,.045); }
.trace { display:grid; grid-template-columns:repeat(5,1fr); gap:0; margin-top:.45rem; }
.trace-step { border-top:1px solid rgba(148,163,184,.25); padding:.38rem .38rem .15rem 0; font-size:.64rem; min-height:2.45rem; }
.trace-step + .trace-step { padding-left:.38rem; border-left:1px solid rgba(148,163,184,.12); }
.trace-step b { display:block; font-size:.61rem; letter-spacing:.08em; opacity:.72; }
.trace-step span { opacity:.52; font-size:.62rem; }
.timeline { border-left:1px solid rgba(148,163,184,.28); padding-left:.65rem; margin:.28rem 0; }
.timeline-row { padding:.18rem 0; font-size:.72rem; }
.mission-objective { border-left:2px solid #94a3b8; padding:.4rem .6rem; margin:.25rem 0 .55rem; background:rgba(148,163,184,.035); }
.outcome { border:1px solid rgba(148,163,184,.22); border-radius:4px; padding:.55rem .65rem; margin-top:.45rem; }
div[data-testid="stMetricValue"] { font-size:1rem; }
div[data-testid="stTabs"] button { font-size:.68rem; letter-spacing:.08em; }
div[data-testid="stButton"] button { border-radius:4px; }
.footer { text-align:center; opacity:.32; font-size:.62rem; padding-top:.55rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Dynamic imports / cached resources
#
# NOTE: st.cache_resource caches are process-global, not per-session. That's
# fine for this console's intended use (one operator, one laptop, one demo
# at a time) but means Reset Mission resets the runtime for ANY concurrent
# session on the same server process -- acceptable here, flagged for anyone
# adapting this into a genuinely multi-user deployment.
# ---------------------------------------------------------------------------

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
    module = load_module(str(RUNTIME_PATH), "ecosphere_runtime_field_console")
    return module.ECOSphereRuntime(
        enable_cache=True,
        enable_events=True,
        enable_cloud_feedback=True,
    )


@st.cache_resource
def master_module():
    return load_module(str(MASTER_PATH), "ecosphere_master_field_console")


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


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

DEFAULTS = {
    "gas_value": 75.0,
    "tilt_value": 2.0,
    "vibration_value": 0.04,
    "last_input": None,
    "last_result": None,
    "mission_history": [],
    "cloud_online": True,
    "profile": "Standard mission",
    "pace": "Normal",
    "last_cloud_update": None,
    "pending_sensor_values": None,
}
for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

PACE_DELAY = {"Slow": 1.25, "Normal": 0.75, "Fast": 0.35}


def active_profile_label(gas, tilt, vibration):
    """Live-computed label -- 'Custom' the moment any slider is dragged off
    a preset, so there's no separate dead 'Custom' button to maintain."""
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


def state_meta(state):
    state = str(state).upper()
    if state == "CRITICAL":
        return "CRITICAL", "STOP / REROUTE", "critical"
    if state == "ELEVATED":
        return "ELEVATED", "SLOW DOWN / MONITOR", "elevated"
    return "SAFE", "CONTINUE MISSION", "safe"


def confidence_of(result):
    return max((float(v) for v in result.get("confidence", {}).values()), default=0.0)


def execute_current_input():
    """Runs a new reading through the runtime only when the sensor values
    actually changed (is_new_reading dedup) -- unrelated Streamlit reruns
    (any widget interaction anywhere on the page) must not re-advance the
    rover or re-trigger cloud analysis."""
    runtime = runtime_factory()
    values = (
        float(st.session_state.gas_value),
        float(st.session_state.tilt_value),
        float(st.session_state.vibration_value),
    )
    key = tuple(round(x, 4) for x in values)
    is_new_reading = st.session_state.last_input != key

    if is_new_reading:
        events_before = len(runtime.event_log)
        result = runtime.step(*values)
        st.session_state.last_input = key
        st.session_state.last_result = result
        st.session_state.mission_history.append({
            "label": active_profile_label(*values),
            "gas": values[0], "tilt": values[1], "vibration": values[2],
            "state": result["hazard_state"],
            "cache": result["cache_hit"],
            "rf": result["rf_executed"],
        })

        # Cloud toggle's real effect: analysis only runs while online, and
        # only when this reading actually generated a new edge event (not
        # on every heartbeat reading) -- get_cloud_feedback() re-scans the
        # runtime's FULL cumulative event history each call, so calling it
        # unconditionally on every reading would re-evaluate needlessly.
        # Offline, the edge config is left exactly as it was -- config
        # freezes, it doesn't reset -- matching the offline-retention
        # behavior validated in 19_integration_regression.py.
        new_events = len(runtime.event_log) > events_before
        if st.session_state.cloud_online and new_events:
            applied, config = runtime.get_cloud_feedback()
            if applied:
                st.session_state.last_cloud_update = config

    return st.session_state.last_result, is_new_reading


def reset_mission():
    clear_runtime_caches()
    reset_rover()
    for _key, _value in DEFAULTS.items():
        # Reset runs as a button callback, before widgets are instantiated
        # on the next rerun. deepcopy also prevents mutable defaults such as
        # mission_history from sharing the same list object with DEFAULTS.
        st.session_state[_key] = deepcopy(_value)
    st.session_state.gas_value, st.session_state.tilt_value, st.session_state.vibration_value = SCENARIOS["Normal"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

runtime = runtime_factory()

hcol1, hcol2 = st.columns([2.2, 1])
with hcol1:
    st.markdown(
        """
<div class="console-header">
  <div>
    <div class="console-title">◆ ECOSPHERE</div>
    <div class="console-sub">AUTONOMOUS ENVIRONMENTAL FIELD UNIT · EDGE INTELLIGENCE</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with hcol2:
    cloud_txt = "CLOUD: ONLINE" if st.session_state.cloud_online else "CLOUD: OFFLINE (config frozen)"
    st.markdown(
        f'<div class="status-line">EDGE: OPERATIONAL &nbsp;·&nbsp; {cloud_txt}</div>',
        unsafe_allow_html=True,
    )

tab_operate, tab_mission, tab_prove = st.tabs(["OPERATE", "MISSION", "PROVE"])


# ---------------------------------------------------------------------------
# OPERATE -- perceive / decide / act, default view
# ---------------------------------------------------------------------------

with tab_operate:
    # Apply mission replay's final sensor state before creating the widgets.
    # This keeps widget ownership unambiguous and avoids Streamlit's
    # "cannot be modified after widget is instantiated" exception.
    pending = st.session_state.get("pending_sensor_values")
    if pending is not None:
        st.session_state.gas_value, st.session_state.tilt_value, st.session_state.vibration_value = pending
        st.session_state.pending_sensor_values = None

    st.markdown(
        '<div class="mission-strip"><b>LIVE FIELD MISSION</b>'
        '<span>PERCEIVE → DECIDE → ACT → ADAPT</span></div>',
        unsafe_allow_html=True,
    )

    sensor_cols = st.columns(3)
    with sensor_cols[0]:
        gas = st.slider("Gas concentration", 0.0, 800.0, key="gas_value", step=1.0, format="%.0f ppm")
    with sensor_cols[1]:
        tilt = st.slider("Tilt", 0.0, 35.0, key="tilt_value", step=0.5, format="%.1f°")
    with sensor_cols[2]:
        vibration = st.slider("Vibration", 0.0, 1.0, key="vibration_value", step=0.01, format="%.2f g")

    label = active_profile_label(gas, tilt, vibration)
    st.caption(f"Current environment: **{label}**" + ("" if label != "Custom" else " (manually set)"))

    result, is_new_reading = execute_current_input()
    state, action, state_class = state_meta(result["hazard_state"])
    confidence = confidence_of(result)
    summary = runtime.summary()

    st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">MISSION SPACE · LIVE RESPONSE</div>', unsafe_allow_html=True)
    render_rover_panel(result["hazard_state"], is_new_reading=is_new_reading)
    st.markdown('</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns([1.0, 1.25, 0.9])

    with d1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="kicker">Local decision</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-state {state_class}">{state}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="action">{action}</div>', unsafe_allow_html=True)
        st.progress(min(max(confidence, 0.0), 1.0), text=f"Confidence · {confidence:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="kicker">How the edge decided</div>', unsafe_allow_html=True)
        rf = "CACHE HIT" if result["cache_hit"] else "FRESH RF"
        rf_style = "signal" if result["cache_hit"] else "signal-warn"
        detail = (
            "Same decision region → inference bypassed."
            if result["cache_hit"]
            else "Decision boundary changed → optimized model executed."
        )
        action_short = {"SAFE": "CONTINUE", "ELEVATED": "SLOW", "CRITICAL": "STOP / REROUTE"}[state]
        adapt = "CONFIG UPDATED" if st.session_state.last_cloud_update else "NO ADAPTATION"
        st.markdown(
            f"""<div class="{rf_style}"><b>{rf}</b> · {detail}</div>
<div class="trace">
<div class="trace-step"><b>01 · PERCEIVE</b><span>{gas:.0f} ppm · {tilt:.1f}° · {vibration:.2f}g</span></div>
<div class="trace-step"><b>02 · INFER</b><span>{confidence:.1%} confidence</span></div>
<div class="trace-step"><b>03 · DECIDE</b><span>{state}</span></div>
<div class="trace-step"><b>04 · ACT</b><span>{action_short}</span></div>
<div class="trace-step"><b>05 · ADAPT</b><span>{adapt}</span></div>
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="mono-small" style="margin-top:.35rem;opacity:.5;">'
        f'EDGE CONFIG v{summary["config_version"]} · '
        f'CACHE BYPASS {summary["cache_bypass_percent"]:.0f}% · '
        f'EVENTS {summary["events_generated"]}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# MISSION -- scenario/profile control, cloud link, reset
# ---------------------------------------------------------------------------

with tab_mission:
    mcol1, mcol2 = st.columns([1.3, 1])

    with mcol1:
        st.markdown("##### ENVIRONMENT STATES")
        scen_cols = st.columns(4)
        for idx, name in enumerate(SCENARIOS.keys()):
            with scen_cols[idx]:
                st.button(
                    name, key=f"scen_{idx}", use_container_width=True,
                    type="primary" if label == name else "secondary",
                    on_click=apply_scenario, args=(name,),
                )

        st.markdown("##### MISSION REPLAY")
        st.selectbox("Mission profile", list(MISSION_PROFILES.keys()), key="profile")
        st.select_slider("Mission pace", options=["Slow", "Normal", "Fast"], key="pace")

        if st.button("▶ Start autonomous mission", type="primary", use_container_width=True):
            import time as _time

            # A mission replay is a fresh mission run, but it preserves the
            # operator's selected profile and pace. Do not mutate widget-backed
            # sensor keys here; only runtime/rover/history state is reset.
            clear_runtime_caches()
            reset_rover()
            runtime = runtime_factory()
            st.session_state.mission_history = []
            st.session_state.last_result = None
            st.session_state.last_input = None
            st.session_state.last_cloud_update = None

            profile = MISSION_PROFILES[st.session_state.profile]
            st.markdown(
                '<div class="mission-objective"><b>MISSION OBJECTIVE</b><br>'
                'Reach the survey endpoint while responding locally to environmental hazards. '
                'Cloud feedback may adapt future edge configuration, but immediate action remains local.</div>',
                unsafe_allow_html=True,
            )
            progress = st.progress(0, text="Mission initializing…")
            mission_box = st.empty()
            mission_rover_box = st.empty()

            for idx, (step_label, values) in enumerate(profile):
                g, t, v = values
                events_before = len(runtime.event_log)
                live_result = runtime.step(g, t, v)
                new_events = len(runtime.event_log) > events_before
                if st.session_state.cloud_online and new_events:
                    applied, config = runtime.get_cloud_feedback()
                    if applied:
                        st.session_state.last_cloud_update = config
                live_state, live_action, _ = state_meta(live_result["hazard_state"])

                # The replay is not just a log: each validated edge decision
                # drives the same rover state machine used by OPERATE.
                with mission_rover_box.container():
                    render_rover_panel(
                        live_result["hazard_state"],
                        is_new_reading=True,
                    )

                st.session_state.mission_history.append({
                    "label": step_label, "gas": g, "tilt": t, "vibration": v,
                    "state": live_state,
                    "cache": live_result["cache_hit"], "rf": live_result["rf_executed"],
                })
                cache_text = "CACHE HIT" if live_result["cache_hit"] else "RF EXECUTED"
                mission_box.markdown(
                    f'<div class="panel"><div class="kicker">Step {idx+1}/{len(profile)}</div>'
                    f'<div class="big-state">{live_state}</div><b>{step_label}</b><br>'
                    f'Gas {g:.0f} ppm · Tilt {t:.1f}° · Vibration {v:.2f} g<hr>'
                    f'<b>{live_action}</b><br><span class="mono-small">{cache_text}</span></div>',
                    unsafe_allow_html=True,
                )
                progress.progress((idx + 1) / len(profile), text=f"{step_label} · {live_state}")
                _time.sleep(PACE_DELAY[st.session_state.pace])
            # Queue the final mission reading for the next Streamlit rerun.
            # Do not mutate widget-backed keys after the sliders are instantiated.
            st.session_state.pending_sensor_values = profile[-1][1]
            st.session_state.last_input = None  # force OPERATE tab to re-sync on next view

            final_state = st.session_state.mission_history[-1]["state"]
            critical_steps = sum(1 for item in st.session_state.mission_history if item["state"] == "CRITICAL")
            elevated_steps = sum(1 for item in st.session_state.mission_history if item["state"] == "ELEVATED")
            cache_hits = sum(1 for item in st.session_state.mission_history if item["cache"])

            st.markdown(
                f"""<div class="outcome">
<b>MISSION OUTCOME</b> · COMPLETE<br>
<span class="mono-small">
{len(profile)} environment states observed · {critical_steps} critical ·
{elevated_steps} elevated · {cache_hits} cache-assisted decisions ·
final state {final_state}
</span>
</div>""",
                unsafe_allow_html=True,
            )

    with mcol2:
        st.markdown("##### CONNECTIVITY")
        st.toggle("Cloud link available", key="cloud_online")
        if st.session_state.cloud_online:
            st.markdown(
                '<div class="signal"><b>CLOUD CONNECTED</b><br>'
                '<span class="mono-small">Edge config accepts new recommendations from accumulated events.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="signal-danger"><b>CLOUD OFFLINE</b><br>'
                '<span class="mono-small">Local inference continues. Edge config is frozen at its last value '
                '(no analysis runs while offline).</span></div>',
                unsafe_allow_html=True,
            )
        if st.session_state.last_cloud_update:
            st.caption(f"Last applied config: {st.session_state.last_cloud_update}")

        st.markdown("##### MISSION CONTROL")
        st.caption("Clears runtime, rover, cache, config, history, and inputs back to a known baseline.")
        st.button(
            "⟲ Reset Mission",
            use_container_width=True,
            on_click=reset_mission,
        )

        st.markdown("##### MISSION RECORD")
        history = st.session_state.mission_history
        if history:
            for item in history[-8:]:
                cache = "HIT" if item["cache"] else "MISS"
                st.markdown(
                    f'<div class="timeline"><div class="timeline-row"><b>{item["label"]}</b> · {item["state"]}<br>'
                    f'<span class="mono-small">{item["gas"]:.0f} ppm · {item["tilt"]:.1f}° · '
                    f'{item["vibration"]:.2f} g · Cache {cache}</span></div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No readings yet.")


# ---------------------------------------------------------------------------
# PROVE -- collapsible engineering evidence
# ---------------------------------------------------------------------------

with tab_prove:
    st.caption("Measured claims only. x86 = this machine, not the QRB2210 target. See source scripts for full methodology.")

    ev1, ev2, ev3 = st.columns(3)
    with ev1:
        st.markdown("**Model compression**")
        st.markdown(f"`{EVIDENCE['baseline_kb']:.2f} KB → {EVIDENCE['optimized_kb']:.2f} KB`")
        st.markdown(f"**{EVIDENCE['size_ratio']:.2f}× smaller** · 100 trees/depth 6 → 5 trees/depth 4")
        st.markdown(
            f"Held-out accuracy: baseline {EVIDENCE['baseline_accuracy']:.2f}% · "
            f"optimized {EVIDENCE['optimized_accuracy']:.2f}% "
            f"(no measurable degradation; synthetic dataset, not a real-world accuracy claim)"
        )

    with ev2:
        st.markdown("**Measured latency (x86, complete pipeline)**")
        st.markdown(
            f"`{EVIDENCE['baseline_ms_median']:.2f} ms → {EVIDENCE['optimized_ms_median']:.2f} ms` (median)"
        )
        st.markdown(f"**{EVIDENCE['latency_ratio_median']:.2f}× faster**")
        st.caption(
            f"Median of {EVIDENCE['latency_reps']} independent repetitions, order randomized "
            f"per rep. stdev: {EVIDENCE['baseline_ms_stdev']:.2f} / {EVIDENCE['optimized_ms_stdev']:.2f} ms. "
            "Not an ARM/QRB2210 measurement."
        )

    with ev3:
        st.markdown("**Decision-region cache**")
        st.markdown(f"Regression stream: **{EVIDENCE['regression_agreement']}** reference agreement")
        st.markdown(f"{EVIDENCE['cache_hits']} cache hits, {EVIDENCE['cache_hit_mismatches']} label mismatches")
        st.caption("From 19_integration_regression.py — a controlled 1,000-reading synthetic stream.")

    st.divider()
    st.markdown("**Selective telemetry**")
    st.markdown(f"**{EVIDENCE['telemetry_reduction']:.2f}% reduction** in events sent vs. readings taken")
    st.caption(
        f"{EVIDENCE['telemetry_events']} events generated from {EVIDENCE['telemetry_readings']} readings across "
        f"a 5-regime persistent stream; {EVIDENCE['telemetry_missed_transitions']} meaningful hazard transitions "
        "missed. Cloud connectivity was simulated with an in-memory sink — feasibility evidence, "
        "not real-world network validation. From 16_edge_event_intelligence.py."
    )


st.markdown('<div class="footer">ECOsphere · Field Console · measured claims only</div>', unsafe_allow_html=True)