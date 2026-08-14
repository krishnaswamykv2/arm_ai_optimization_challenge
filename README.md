# ECOsphere

## Evidence-Driven Physical AI for Autonomous Environmental Response

**Arm AI Optimization Challenge**

ECOsphere is a software-defined Physical AI runtime that turns heterogeneous environmental observations into **interpretable risk, temporal understanding, autonomous rover action, and post-action verification**—while keeping the critical autonomy path local and optimized for constrained Arm-class edge execution.

> **Perceive → Interpret → Reason → Decide → Act → Observe → Verify → Reassess**

---

## Why ECOsphere?

Most environmental AI pipelines end at:

```text
SENSE → CLASSIFY → ALERT
```

That is insufficient for autonomous physical systems.

A classifier can identify a hazard, but autonomy requires answering a larger chain of questions:

* What physical evidence supports the prediction?
* Which sensor is driving the anomaly?
* Is the risk rising, falling, or stable?
* Does the model agree with the physical evidence?
* How urgent is the situation?
* What should the rover do?
* Can that decision happen without cloud connectivity?
* **Did the intervention actually improve the physical state?**

ECOsphere closes that loop.

```text
PHYSICAL ENVIRONMENT
        ↓
     SENSING
        ↓
 FEATURE ENGINEERING
        ↓
 EDGE INFERENCE
        ↓
 PHYSICAL EVIDENCE
        ↓
 TEMPORAL REASONING
        ↓
 TRAJECTORY PROJECTION
        ↓
 MODEL ↔ PHYSICAL RECONCILIATION
        ↓
 DETERMINISTIC DECISION
        ↓
 ROVER ACTION
        ↓
 POST-ACTION OBSERVATION
        ↓
 CONSEQUENCE VERIFICATION
        ↓
 RE-EVALUATION
```

The important shift is:

> **An action is not considered successful merely because it was executed. ECOsphere requires physical evidence of its consequence.**

---

# What We Built

ECOsphere is composed of several cooperating runtime layers.

### 1. Physical-world representation

Environmental observations are represented through:

* Gas
* Tilt
* Vibration

These are transformed into normalized reasoning features including:

* Gas Risk
* Tilt Risk
* Vibration Risk
* Structural Risk
* Joint Risk

This creates a common representation for heterogeneous physical signals.

---

### 2. Edge inference

A validated Random Forest inference path was aggressively optimized for constrained execution.

| Metric                   |  Baseline |         ECOsphere |               Improvement |
| ------------------------ | --------: | ----------------: | ------------------------: |
| Model footprint          | 169.44 KB |       **9.84 KB** |        **17.21× smaller** |
| Median inference latency | 17.307 ms |      **1.504 ms** |          **11.51× lower** |
| Regression vectors       |      1000 | **1000 matching** |        **100% agreement** |
| Decision-cache hits      |         — | **187 validated** | Reuse of repeated regions |

The optimization objective was not simply "make it smaller."

It was:

> **Reduce edge cost while preserving validated decision behaviour.**

---

### 3. Physical Evidence Intelligence

The classifier is not treated as the sole authority.

ECOsphere independently interprets the physical evidence:

```text
GAS
TILT
VIBRATION
   ↓
Severity
Abnormal channels
Dominant signal
Agreement
Structural risk
Joint risk
   ↓
Environmental context
```

This enables interpretable states such as:

```text
GAS-DOMINANT ANOMALY
TILT ANOMALY
VIBRATION ANOMALY
MULTI-SENSOR ENVIRONMENTAL ANOMALY
```

---

### 4. Multi-sensor agreement

ECOsphere explicitly measures agreement between independent physical channels.

```text
0 / 3 → NONE
1 / 3 → LOW
2 / 3 → MODERATE
3 / 3 → HIGH
```

This gives the decision engine an evidence relationship rather than relying exclusively on a model confidence score.

---

### 5. Temporal intelligence

Risk is not interpreted as an isolated number.

ECOsphere maintains recent joint-risk history and distinguishes:

```text
RISING
FALLING
STABLE
INSUFFICIENT DATA
```

Therefore:

```text
HIGH RISK
```

and

```text
HIGH + RAPIDLY RISING RISK
```

are not treated as equivalent states.

When the history is insufficient, ECOsphere explicitly reports:

```text
INSUFFICIENT DATA
```

rather than inventing a trend.

---

### 6. Near-term trajectory

Current severity and trajectory are treated separately.

ECOsphere derives an interpretable near-term projection from the current joint risk and observed temporal slope.

The output includes:

* Current risk
* Trajectory
* Projected risk
* Projected severity
* Confidence/data sufficiency state

This is deliberately an **interpretable trajectory projection**, not a claimed trained forecasting model.

---

### 7. Model ↔ Physical Reconciliation

ECOsphere maintains two independent views:

```text
MODEL VIEW
What does the classifier indicate?

        ↕

PHYSICAL VIEW
What do the environmental signals indicate?
```

The runtime can identify:

```text
CONSISTENT
ESCALATING BEYOND MODEL
MODEL / PHYSICAL DISAGREEMENT
```

Disagreement is therefore promoted from a hidden failure mode into an explicit runtime signal.

---

### 8. Deterministic autonomy policy

Environmental reasoning becomes an actionable rover policy.

```text
LOW / NORMAL
    → CONTINUE

MODERATE / WATCH
    → MONITOR

HIGH / ESCALATING
    → SLOW & MONITOR

CRITICAL
    → STOP & REROUTE

MODEL / PHYSICAL DISAGREEMENT
    → PAUSE & VERIFY
```

The decision path is explicit, auditable, and state-driven.

---

# The Defining Feature: Consequence Verification

ECOsphere does not stop at:

```text
HAZARD DETECTED
        ↓
ACTION ISSUED
```

Instead:

```text
PRE-ACTION RISK
      ↓
INTERVENTION
      ↓
SUBSEQUENT OBSERVATION
      ↓
POST-ACTION RISK
      ↓
ΔRISK
      ↓
RECOVERY ASSESSMENT
```

The runtime can distinguish:

```text
NO IMPROVEMENT
PARTIAL IMPROVEMENT
RECOVERY
RECOVERY CONFIRMED
```

The system does **not** infer recovery from the fact that an actuator command was issued.

> **Action is not success. Evidence of consequence is success.**

This is the architectural principle that turns ECOsphere from a hazard classifier into a closed-loop Physical AI system.

---

# Edge Efficiency

ECOsphere combines two complementary optimization strategies.

### Model-path optimization

```text
169.44 KB
     ↓
  9.84 KB
```

### Decision-region reuse

```text
SENSOR STATE
    ↓
REGION SIGNATURE
    ↓
CACHE HIT ─────→ REUSE
    │
CACHE MISS
    ↓
INFERENCE
    ↓
CACHE RESULT
```

**187 validated cache hits** demonstrate reuse of repeated decision regions in the runtime evaluation.

Together, these reduce unnecessary computation while preserving the validated inference path.

---

# Arm Deployment Architecture

The critical autonomy path is designed around local execution:

```text
Sensors
   ↓
Feature Processing
   ↓
Inference / Cache
   ↓
Physical Reasoning
   ↓
Decision Engine
   ↓
Action Controller
   ↓
Rover Behaviour
```

Remote services are separated from safety-critical execution:

```text
LOCAL EDGE
──────────
Sensing
Inference
Reasoning
Decision
Action
Verification

        │
        │ asynchronous / non-critical
        ▼

SUPERVISORY LAYER
─────────────────
Telemetry
Configuration
Remote monitoring
Offline auditing
```

The architecture is therefore **cloud-independent for the critical autonomy loop**.

The optimization results demonstrate the software characteristics required for constrained Arm execution. **Physical Arm-board measurements should be reported separately from these host/runtime validation results rather than being implied.**

---

# Repository Structure

```text
arm_ai_optimization_challenge/
│
├── 00_load_real_gas_data.py
├── 01_generate_data.py
├── 02_feature_engineering.py
├── 03_train_model.py
├── 04_predict_live.py
├── 05_drift_demonstration.py
├── 06_drift_curve.py
├── 07_drift_correction_comparison.py
│
├── 08_master_pipeline.py
│
├── 09_train_keras_model.py
├── 10_convert_to_tflite.py
├── 11_tflite_inference.py
├── 11_verify_benchmarks.py
├── 12_laptop_live_demo.py
├── 12_live_rover_pipeline.py
├── 13_hardware_deployment_pipeline.py
│
├── 15_adaptive_cache_experiment.py
├── 16_edge_event_intelligence.py
├── 17_cloud_feedback_engine.py
├── 18_ecosphere_runtime.py
├── 19_integration_regression.py
│
├── ECOsphere_final_dashboard.py
│
├── environmental_context.py
├── risk_reasoner.py
├── temporal_intelligence.py
├── near_term_projection.py
├── decision_engine.py
├── rover_action_controller.py
├── rover_visualization.py
├── action_verifier.py
├── recovery_tracker.py
├── closed_loop_test.py
├── ecosphere_ui_adapter.py
│
├── arm64_validation.py
├── feature_engineering_helper.py
├── HARDWARE_REQUIREMENTS.md
│
├── hazard_model_optimized.joblib
├── hazard_model.tflite
├── label_classes.npy
├── norm_mean.npy
├── norm_std.npy
│
└── uno_q_sketch/
```

Archived dashboards, obsolete experiments, and handoff artifacts are intentionally excluded from the final submission surface.

---

# Quick Start

## 1. Clone

```bash
git clone https://github.com/krishnaswamykv2/arm_ai_optimization_challenge.git
cd arm_ai_optimization_challenge
```

## 2. Create an environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / Arm64

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the repository does not yet contain a `requirements.txt`, install the runtime dependencies used by the included Python modules before running the dashboard or validation scripts.

---

# Run ECOsphere

Launch the flagship runtime:

```bash
python ECOsphere_final_dashboard.py
```

The dashboard exposes the ECOsphere runtime concepts including:

* Environmental state
* Gas / tilt / vibration evidence
* Hazard interpretation
* Joint risk
* Temporal state
* Trajectory
* Model/physical alignment
* Autonomous action
* Mission state
* Recovery
* Post-action verification
* Runtime telemetry

---

# Validate the Runtime

Run the integration validation:

```bash
python 19_integration_regression.py
```

Run closed-loop verification:

```bash
python closed_loop_test.py
```

Run benchmark validation:

```bash
python 11_verify_benchmarks.py
```

For Arm64-oriented validation:

```bash
python arm64_validation.py
```

The exact validation command should be run in the target environment with the corresponding dependencies installed.

---

# Validation Results

The current validated results include:

### Classification integrity

**1000 / 1000 regression agreement**

The optimized path preserved the validated classification behaviour across the regression vectors used.

### Model footprint

**169.44 KB → 9.84 KB**

**17.21× reduction**

### Median inference latency

**17.307 ms → 1.504 ms**

**11.51× reduction**

### Decision-region reuse

**187 validated cache hits**

### Runtime validation

The closed-loop runtime has been exercised across:

* Scenario transitions
* Repeated readings
* Mission replay
* Reset behaviour
* Recovery sequences
* Post-action verification
* Conditional reasoning output

---

# Physical AI, Not Just AI

The distinction is architectural.

A conventional environmental AI system:

```text
OBSERVE → CLASSIFY → REPORT
```

ECOsphere:

```text
OBSERVE
   ↓
INTERPRET
   ↓
REASON
   ↓
DECIDE
   ↓
ACT
   ↓
OBSERVE AGAIN
   ↓
VERIFY
   ↓
REASSESS
```

The model is one component.

The intelligence emerges from the **closed-loop relationship between physical evidence, reasoning, action, and consequence**.

---

# What Is Novel

ECOsphere does not claim novelty from any individual primitive.

Not:

* Random Forest alone
* Sensor fusion alone
* Caching alone
* Trend analysis alone
* Rover simulation alone

The architectural novelty is their integration into an evidence-driven autonomy loop:

> **The system interprets physical evidence, reasons about its evolution, acts on that reasoning, and evaluates the physical consequence of its own intervention.**

That creates six tightly connected capabilities:

1. **Physical Evidence Intelligence**
2. **Temporal Risk Intelligence**
3. **Model ↔ Physical Reconciliation**
4. **Evidence-Based Autonomous Action**
5. **Closed-Loop Action Verification**
6. **Edge-Efficient Inference**

---

# Design Philosophy

### Local first

Critical autonomy does not require continuous cloud connectivity.

### Evidence over confidence

Model confidence is not treated as unquestionable truth.

### Trends over snapshots

Risk evolution matters alongside current severity.

### Action requires verification

An issued command is not evidence that the world changed as intended.

### Optimization without behavioural drift

Edge efficiency must preserve validated decision behaviour.

### Honest uncertainty

When the system does not have enough evidence, it says so.

```text
INSUFFICIENT DATA
```

is preferable to fabricated certainty.

---

# Current Prototype Scope

ECOsphere is a:

> **Software-defined Physical AI prototype.**

The current implementation validates the software/runtime architecture for:

* Environmental sensing representation
* Feature engineering
* Edge inference
* Decision-region caching
* Physical evidence reasoning
* Multi-sensor agreement
* Temporal intelligence
* Near-term trajectory projection
* Model/physical reconciliation
* Deterministic action policy
* Rover action control
* Mission-state transitions
* Consequence verification
* Recovery assessment
* Cloud-independent autonomy logic
* Edge optimization and regression integrity

The current prototype does **not** claim a completed physical rover deployment where one has not been measured.

---

# The Core Idea

ECOsphere treats autonomy as an **evidence loop**, not a prediction.

```text
UNDERSTAND THE ENVIRONMENT
          ↓
UNDERSTAND HOW IT IS CHANGING
          ↓
DECIDE
          ↓
ACT
          ↓
OBSERVE THE CONSEQUENCE
          ↓
VERIFY
          ↓
REASSESS
```

## ECOsphere

### **From environmental prediction to verifiable physical autonomy.**
