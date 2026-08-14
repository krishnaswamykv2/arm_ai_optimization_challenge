"""
ECOsphere — Flagship Physical AI Dashboard
==========================================

Laptop-only competition demo.

Design goals:
- Use the REAL ECOsphere pipeline; no duplicated classification logic.
- Make the Physical AI loop understandable in <30 seconds.
- Make the optimization result the visual centerpiece.
- Keep hardware explicitly optional for this validation/demo path.
"""

from pathlib import Path
import importlib.util
import sys
import time

import streamlit as st


# ---------------------------------------------------------------------
# REAL PROJECT FILES
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MASTER_PATH = ROOT / "08_master_pipeline.py"
BASELINE_MODEL = ROOT / "hazard_model.joblib"
OPTIMIZED_MODEL = ROOT / "hazard_model_optimized.joblib"

if not MASTER_PATH.exists():
    st.error(f"Missing {MASTER_PATH.name}. Put this dashboard beside your project files.")
    st.stop()

if not BASELINE_MODEL.exists() or not OPTIMIZED_MODEL.exists():
    st.error(
        "Both hazard_model.joblib and hazard_model_optimized.joblib must be "
        "in the same folder as this dashboard."
    )
    st.stop()

spec = importlib.util.spec_from_file_location("ecosystem_master", MASTER_PATH)
master = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master)


# ---------------------------------------------------------------------
# VERIFIED BENCHMARK RESULTS
# ---------------------------------------------------------------------
BASELINE_SIZE_KB = 169.44
OPTIMIZED_SIZE_KB = 9.84
SIZE_MULTIPLIER = 17.21

BASELINE_TREES = 100
OPTIMIZED_TREES = 5
BASELINE_DEPTH = 6
OPTIMIZED_DEPTH = 4

BASELINE_MEDIAN_MS = 8.280
OPTIMIZED_MEDIAN_MS = 0.754
LATENCY_MULTIPLIER = 10.98

BASELINE_ACCURACY = 99.49
OPTIMIZED_ACCURACY = 100.00


# ---------------------------------------------------------------------
# SCENARIOS
# ---------------------------------------------------------------------
SCENARIOS = {
    "Normal": {
        "gas": 75.0,
        "tilt": 2.0,
        "vibration": 0.04,
        "description": "Nominal environment",
    },
    "Gas hazard": {
        "gas": 480.0,
        "tilt": 3.0,
        "vibration": 0.05,
        "description": "Gas concentration rising",
    },
    "Structural hazard": {
        "gas": 90.0,
        "tilt": 20.0,
        "vibration": 0.45,
        "description": "Unstable structure / terrain",
    },
    "Combined critical": {
        "gas": 700.0,
        "tilt": 27.0,
        "vibration": 0.70,
        "description": "Multiple hazards detected",
    },
}


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
defaults = {
    "scenario": "Normal",
    "gas": 75.0,
    "tilt": 2.0,
    "vibration": 0.04,
    "calibrated": False,
    "baseline_pipeline": None,
    "optimized_pipeline": None,
    "last_baseline": None,
    "last_optimized": None,
    "event_count": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_pipelines():
    st.session_state.baseline_pipeline = None
    st.session_state.optimized_pipeline = None
    st.session_state.calibrated = False
    st.session_state.last_baseline = None
    st.session_state.last_optimized = None
    st.session_state.event_count = 0


def set_scenario(name):
    preset = SCENARIOS[name]
    st.session_state.scenario = name
    st.session_state.gas = preset["gas"]
    st.session_state.tilt = preset["tilt"]
    st.session_state.vibration = preset["vibration"]


def get_pipeline(kind):
    if kind == "baseline":
        if st.session_state.baseline_pipeline is None:
            st.session_state.baseline_pipeline = master.build_pipeline(
                model_path=str(BASELINE_MODEL),
                use_numpy=False,
            )
        return st.session_state.baseline_pipeline

    if st.session_state.optimized_pipeline is None:
        st.session_state.optimized_pipeline = master.build_pipeline(
            model_path=str(OPTIMIZED_MODEL),
            use_numpy=True,
        )
    return st.session_state.optimized_pipeline


def calibrate():
    """
    Establish the same safe reference state for both independent pipelines.

    The master pipeline's tracker uses 15 calibration readings. We feed a
    stable nominal environment to each pipeline so the dashboard starts
    from a known state instead of pretending calibration happened.
    """
    safe_reading = (75.0, 2.0, 0.04)

    baseline = get_pipeline("baseline")
    optimized = get_pipeline("optimized")

    for _ in range(15):
        baseline(*safe_reading)
        optimized(*safe_reading)

    st.session_state.calibrated = True
    st.session_state.last_baseline = None
    st.session_state.last_optimized = None


def run_both(gas, tilt, vibration):
    baseline = get_pipeline("baseline")
    optimized = get_pipeline("optimized")

    t0 = time.perf_counter()
    baseline_result = baseline(gas, tilt, vibration)
    baseline_ui_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    optimized_result = optimized(gas, tilt, vibration)
    optimized_ui_time = (time.perf_counter() - t0) * 1000

    st.session_state.last_baseline = baseline_result
    st.session_state.last_optimized = optimized_result
    st.session_state.event_count += 1

    return baseline_result, optimized_result, baseline_ui_time, optimized_ui_time


def hazard_style(hazard):
    hazard = hazard.upper()
    if hazard == "CRITICAL":
        return "CRITICAL", "STOP / AVOID / REROUTE"
    if hazard == "ELEVATED":
        return "ELEVATED", "PROCEED WITH CAUTION"
    return "SAFE", "CONTINUE MISSION"


def confidence_pct(confidence, label):
    return float(confidence.get(label, 0.0)) * 100.0


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ECOsphere | Physical AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# VISUAL SYSTEM
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(128,128,128,.24);
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(40,120,100,.13),
            rgba(60,90,160,.08)
        );
        margin-bottom: 1rem;
    }

    .eyebrow {
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .16em;
        opacity: .68;
        margin-bottom: .35rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 850;
        line-height: 1.05;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: .76;
        margin-top: .45rem;
    }

    .pill {
        display: inline-block;
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 999px;
        padding: .25rem .65rem;
        margin-right: .35rem;
        font-size: .72rem;
        font-weight: 700;
    }

    .section-kicker {
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .12em;
        opacity: .58;
        text-transform: uppercase;
        margin-bottom: .2rem;
    }

    .pipeline {
        display: flex;
        align-items: center;
        gap: .35rem;
        overflow-x: auto;
        padding: .7rem 0 1rem;
    }

    .pipeline-box {
        min-width: 145px;
        padding: .75rem .65rem;
        border: 1px solid rgba(128,128,128,.24);
        border-radius: 12px;
        text-align: center;
        font-size: .82rem;
        font-weight: 700;
    }

    .arrow {
        opacity: .45;
        font-size: 1.2rem;
    }

    .card {
        border: 1px solid rgba(128,128,128,.23);
        border-radius: 15px;
        padding: 1rem;
        min-height: 100%;
    }

    .card-title {
        font-size: .78rem;
        letter-spacing: .1em;
        text-transform: uppercase;
        font-weight: 800;
        opacity: .62;
        margin-bottom: .55rem;
    }

    .big-number {
        font-size: 2.15rem;
        font-weight: 850;
        line-height: 1;
    }

    .delta {
        font-size: .82rem;
        opacity: .72;
        margin-top: .25rem;
    }

    .comparison {
        border: 1px solid rgba(128,128,128,.24);
        border-radius: 16px;
        padding: 1rem;
    }

    .comparison-row {
        display: grid;
        grid-template-columns: 1.4fr 1fr 1fr;
        gap: .5rem;
        padding: .55rem .35rem;
        border-bottom: 1px solid rgba(128,128,128,.12);
        align-items: center;
    }

    .comparison-row:last-child { border-bottom: 0; }

    .comparison-head {
        font-weight: 800;
        font-size: .75rem;
        opacity: .62;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .win-banner {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-top: .8rem;
    }

    .win-number {
        font-size: 2.5rem;
        font-weight: 900;
        line-height: 1;
    }

    .tiny {
        font-size: .72rem;
        opacity: .58;
    }

    .response-box {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 15px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">PHYSICAL AI · EDGE OPTIMIZATION · LIVE VALIDATION</div>
        <div class="hero-title">🌍 ECOsphere</div>
        <div class="hero-subtitle">
            Drift-aware environmental perception for an autonomous physical system —
            optimized to make edge inference smaller and faster.
        </div>
        <div style="margin-top:.8rem">
            <span class="pill">MULTI-SENSOR PERCEPTION</span>
            <span class="pill">EDGE AI</span>
            <span class="pill">PHYSICAL RESPONSE</span>
            <span class="pill">MEASURED OPTIMIZATION</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# PIPELINE OVERVIEW
# ---------------------------------------------------------------------
st.markdown('<div class="section-kicker">System architecture</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="pipeline">
        <div class="pipeline-box">🌐<br>Environment</div>
        <div class="arrow">→</div>
        <div class="pipeline-box">📡<br>Sensor Input</div>
        <div class="arrow">→</div>
        <div class="pipeline-box">↻<br>Drift Correction</div>
        <div class="arrow">→</div>
        <div class="pipeline-box">⚙<br>Feature Fusion</div>
        <div class="arrow">→</div>
        <div class="pipeline-box">🧠<br>Optimized Edge AI</div>
        <div class="arrow">→</div>
        <div class="pipeline-box">🚨<br>Hazard Decision</div>
        <div class="arrow">→</div>
        <div class="pipeline-box">🤖<br>Physical Response</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛 Demo controls")
    st.caption("This is the laptop validation path. Sensor values are simulated.")

    st.markdown("**Scenario**")
    scenario_cols = st.columns(2)
    names = list(SCENARIOS.keys())
    for i, name in enumerate(names):
        scenario_cols[i % 2].button(
            name,
            key=f"scenario_{i}",
            use_container_width=True,
            on_click=set_scenario,
            args=(name,),
        )

    st.divider()

    st.markdown("**Live sensor input**")
    gas = st.slider(
        "Gas concentration",
        0.0,
        800.0,
        key="gas",
        step=1.0,
        help="Simulated gas sensor value in ppm.",
    )
    tilt = st.slider(
        "Tilt",
        0.0,
        35.0,
        key="tilt",
        step=0.5,
        help="Simulated tilt sensor value in degrees.",
    )
    vibration = st.slider(
        "Vibration",
        0.0,
        1.0,
        key="vibration",
        step=0.01,
        help="Simulated vibration sensor value in g.",
    )

    st.divider()

    if not st.session_state.calibrated:
        st.warning("Calibration required before drift-aware live testing.")
    else:
        st.success("Calibration complete")

    if st.button(
        "Calibrate safe baseline",
        use_container_width=True,
        type="primary",
    ):
        calibrate()
        st.rerun()

    if st.button("Reset all state", use_container_width=True):
        reset_pipelines()
        st.rerun()

    st.divider()
    st.caption(
        "Hardware deployment is intentionally isolated. This demo does not "
        "claim to control a rover or actuator."
    )


# ---------------------------------------------------------------------
# CALIBRATION GATE
# ---------------------------------------------------------------------
if not st.session_state.calibrated:
    st.info(
        "### Start here\n"
        "Click **Calibrate safe baseline** in the sidebar. The dashboard will "
        "feed the real pipeline's 15-reading calibration sequence to both "
        "baseline and optimized paths before live scenarios begin."
    )

    st.markdown(
        """
        <div class="comparison">
            <div class="card-title">Why calibration exists</div>
            The real pipeline assumes the system begins in a known-safe state.
            It establishes a reference baseline before adaptive drift correction
            starts. We expose that step instead of hiding it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ---------------------------------------------------------------------
# RUN BOTH REAL PIPELINES
# ---------------------------------------------------------------------
baseline_result, optimized_result, _, _ = run_both(gas, tilt, vibration)

baseline_hazard, baseline_response = hazard_style(baseline_result["hazard_state"])
optimized_hazard, optimized_response = hazard_style(optimized_result["hazard_state"])

same_decision = baseline_hazard == optimized_hazard


# ---------------------------------------------------------------------
# LIVE PERCEPTION
# ---------------------------------------------------------------------
st.markdown('<div class="section-kicker">Live perception</div>', unsafe_allow_html=True)
st.subheader(f"{SCENARIOS[st.session_state.scenario]['description']}")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Gas", f"{gas:.0f} ppm")
with s2:
    st.metric("Tilt", f"{tilt:.1f}°")
with s3:
    st.metric("Vibration", f"{vibration:.2f} g")
with s4:
    st.metric("Reading", f"#{st.session_state.event_count}")


# ---------------------------------------------------------------------
# RAW → CORRECTED → DECISION
# ---------------------------------------------------------------------
left, center, right = st.columns([1, 1, 1.15])

with left:
    st.markdown("### Raw → corrected")
    raw = optimized_result["raw_input"]
    corrected = optimized_result["corrected_input"]

    st.markdown(
        f"""
        <div class="card">
        <div class="card-title">Gas</div>
        <div class="big-number">{corrected['gas_ppm']:.1f}</div>
        <div class="delta">ppm corrected · raw {raw['gas_ppm']:.1f}</div>
        <br>
        <div class="card-title">Tilt</div>
        <div class="big-number">{corrected['tilt_deg']:.1f}°</div>
        <div class="delta">corrected · raw {raw['tilt_deg']:.1f}°</div>
        <br>
        <div class="card-title">Vibration</div>
        <div class="big-number">{corrected['vibration_g']:.3f}</div>
        <div class="delta">g corrected · raw {raw['vibration_g']:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with center:
    st.markdown("### Perception path")
    for step in [
        ("01", "Raw sensor reading"),
        ("02", "Adaptive drift correction"),
        ("03", "Feature fusion"),
        ("04", "Hazard classification"),
        ("05", "Physical response"),
    ]:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <b>{step[0]}</b>&nbsp;&nbsp; {step[1]}
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### Decision")

    if optimized_hazard == "CRITICAL":
        st.error(f"## ⛔ {optimized_hazard}")
    elif optimized_hazard == "ELEVATED":
        st.warning(f"## ⚠ {optimized_hazard}")
    else:
        st.success(f"## ✓ {optimized_hazard}")

    st.write("Optimized model confidence")
    for label, probability in optimized_result["confidence"].items():
        st.progress(
            float(probability),
            text=f"{label.upper()} · {float(probability):.1%}",
        )

    st.caption("Current inference path: **optimized**")


# ---------------------------------------------------------------------
# MODEL-TO-MODEL LIVE COMPARISON
# ---------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-kicker">A/B inference comparison</div>', unsafe_allow_html=True)
st.subheader("Same environment. Same task. Different compute budget.")

st.caption(
    "Both models receive the same current sensor reading through independent "
    "pipeline instances. The comparison is about the AI configuration, not a "
    "different hazard rule."
)

st.markdown(
    f"""
    <div class="comparison">
        <div class="comparison-row comparison-head">
            <div>Metric</div><div>Baseline</div><div>Optimized</div>
        </div>
        <div class="comparison-row">
            <div>Model</div><div>100 trees / depth 6</div><div>5 trees / depth 4</div>
        </div>
        <div class="comparison-row">
            <div>Model size</div><div>{BASELINE_SIZE_KB:.2f} KB</div><div><b>{OPTIMIZED_SIZE_KB:.2f} KB</b></div>
        </div>
        <div class="comparison-row">
            <div>Median pipeline latency</div><div>{BASELINE_MEDIAN_MS:.3f} ms</div><div><b>{OPTIMIZED_MEDIAN_MS:.3f} ms</b></div>
        </div>
        <div class="comparison-row">
            <div>Held-out accuracy</div><div>{BASELINE_ACCURACY:.2f}%</div><div>{OPTIMIZED_ACCURACY:.2f}%</div>
        </div>
        <div class="comparison-row">
            <div>Current decision</div><div><b>{baseline_hazard}</b></div><div><b>{optimized_hazard}</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if same_decision:
    st.success(
        f"✓ Decision preserved: both configurations classify the current "
        f"environment as **{optimized_hazard}**."
    )
else:
    st.warning(
        "The two configurations currently disagree on this reading. "
        "Investigate before presenting this scenario as a parity example."
    )

w1, w2 = st.columns(2)
with w1:
    st.markdown(
        f"""
        <div class="win-banner">
            <div class="tiny">MODEL FOOTPRINT</div>
            <div class="win-number">{SIZE_MULTIPLIER:.1f}×</div>
            <div><b>smaller</b></div>
            <div class="tiny">94.2% reduction in model size</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with w2:
    st.markdown(
        f"""
        <div class="win-banner">
            <div class="tiny">END-TO-END PIPELINE</div>
            <div class="win-number">{LATENCY_MULTIPLIER:.1f}×</div>
            <div><b>faster</b></div>
            <div class="tiny">90.9% lower median x86 latency</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# PHYSICAL AI RESPONSE
# ---------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-kicker">Physical AI action layer</div>', unsafe_allow_html=True)
st.subheader("Decision → action")

r1, r2 = st.columns([1, 2])

with r1:
    if optimized_hazard == "CRITICAL":
        st.error("⛔ CRITICAL")
    elif optimized_hazard == "ELEVATED":
        st.warning("⚠ ELEVATED")
    else:
        st.success("✓ SAFE")

with r2:
    if optimized_hazard == "CRITICAL":
        st.markdown("### Recommended response: STOP / AVOID / REROUTE")
        st.write(
            "A physical agent should avoid continuing into the detected "
            "hazard region."
        )
    elif optimized_hazard == "ELEVATED":
        st.markdown("### Recommended response: PROCEED WITH CAUTION")
        st.write(
            "Increase monitoring and prepare for a route or behavior change."
        )
    else:
        st.markdown("### Recommended response: CONTINUE MISSION")
        st.write("No elevated hazard is currently indicated.")

st.caption(
    "This action layer is a simulated Physical AI response for the laptop demo. "
    "It does not claim that a rover actuator was triggered."
)


# ---------------------------------------------------------------------
# ENGINEERING NOTES — GOOD FOR JUDGES
# ---------------------------------------------------------------------
with st.expander("🔬 Engineering notes — what is actually measured?"):
    st.markdown(
        """
        **Optimization decision**

        The baseline Random Forest uses 100 trees with maximum depth 6.
        The optimized candidate uses 5 trees with maximum depth 4.

        **Measured result**

        - Model size: 169.44 KB → 9.84 KB (**17.21× smaller**)
        - Median complete-pipeline latency: 8.280 ms → 0.754 ms
          (**10.98× faster**)
        - Held-out accuracy: 99.49% → 100.00%
        - Interpretation: **no measurable accuracy degradation** on the
          current evaluation.

        **Why we are careful with the claims**

        Accuracy was measured on a synthetic dataset's held-out split.
        Latency was measured on x86, not on the ARM/QRB2210 target.
        Therefore the dashboard does not present the x86 latency as ARM
        performance or claim real-world accuracy parity.

        **Physical deployment**

        The hardware path is intentionally separate from this laptop
        validation demo. The architecture can later consume real sensor
        streams without making the competition demo depend on hardware access.
        """
    )


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "ECOsphere · Physical AI · Edge optimization · Laptop validation path · "
    "Measured claims only"
)
