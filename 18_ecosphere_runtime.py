"""
ECOsphere -- Integrated Edge Runtime
====================================

Production-oriented integration layer around the already validated components.

Flow:
    raw sensors
      -> drift correction
      -> feature fusion
      -> decision-region cache
      -> optimized RF on cache miss
      -> local hazard result
      -> environmental analysis
      -> temporal intelligence
      -> near-term projection
      -> interpretable risk reasoning
      -> edge event intelligence
      -> optional cloud feedback

Design rules:
    - 08_master_pipeline.py is NOT modified.
    - hazard_model.joblib is NOT modified.
    - hazard_model_optimized.joblib is used for the integrated edge path.
    - Cloud feedback is optional and never required for local inference.
    - Cache hits reuse the previous exact RF result only while the feature
      vector reaches the same leaf in every tree.
    - Temporal intelligence reasons over existing risk features.
    - Near-term projection is interpretable and not a trained ML model.
    - This file is the orchestration layer; experiments remain reproducible
      and independent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from environmental_context import analyze_environment
from near_term_projection import project_risk
from risk_reasoner import assess_risk
from temporal_intelligence import TemporalIntelligence
from decision_engine import decide
from rover_action_controller import RoverActionController
from action_verifier import verify_action
from recovery_tracker import RecoveryTracker


ROOT = Path(__file__).resolve().parent

MASTER_PATH = ROOT / "08_master_pipeline.py"
MODEL_PATH = ROOT / "hazard_model_optimized.joblib"
EVENT_MODULE_PATH = ROOT / "16_edge_event_intelligence.py"
FEEDBACK_MODULE_PATH = ROOT / "17_cloud_feedback_engine.py"

FEATURE_COLUMNS = [
    "gas_ppm",
    "tilt_deg",
    "vibration_g",
    "gas_risk",
    "tilt_risk",
    "vibration_risk",
    "structural_risk",
    "joint_risk",
]


def load_module(path: Path, name: str):
    """
    Dynamically load a Python module from a file path.

    The module is registered in sys.modules before execution so that
    dataclasses and other runtime introspection mechanisms can resolve
    the module correctly.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}")

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Register before executing the module.
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return module


@dataclass
class RuntimeStats:
    readings: int = 0
    cache_hits: int = 0
    rf_executions: int = 0
    events_generated: int = 0


class DecisionRegionCache:
    """
    Exact decision-region cache for the optimized Random Forest.

    A cache hit is considered safe only when the current feature vector
    reaches exactly the same leaf in every Random Forest tree as the
    previous feature vector.

    This avoids relying on manually reconstructed floating-point bounds.
    """

    def __init__(self, model):
        self.model = model

        self.previous_leaf_ids = None
        self.previous_result = None

    def _current_leaf_ids(self, x):
        """
        Return the tuple of leaf IDs reached by the feature vector
        across every tree in the Random Forest.
        """

        row = np.asarray(
            x,
            dtype=float,
        ).reshape(1, -1)

        return tuple(
            int(tree.apply(row)[0])
            for tree in self.model.estimators_
        )

    def lookup(self, features):
        """
        Look up a feature vector in the decision-region cache.

        Returns
        -------
        tuple
            (cache_hit, classification_result)
        """

        x = np.asarray(
            [
                features[column]
                for column in FEATURE_COLUMNS
            ],
            dtype=float,
        )

        current_leaf_ids = self._current_leaf_ids(x)

        # Exact cache validation:
        # every tree must reach the same leaf as before.
        if (
            self.previous_leaf_ids is not None
            and current_leaf_ids == self.previous_leaf_ids
            and self.previous_result is not None
        ):
            return True, dict(self.previous_result)

        # Cache miss: execute the actual Random Forest.
        frame = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS,
        )

        prediction = self.model.predict(frame)[0]
        probabilities = self.model.predict_proba(frame)[0]

        confidence = dict(
            zip(
                self.model.classes_,
                probabilities.round(3),
            )
        )

        result = {
            "hazard_state": str(prediction).upper(),
            "confidence": confidence,
        }

        # Store the exact leaf signature and RF result.
        self.previous_leaf_ids = current_leaf_ids
        self.previous_result = dict(result)

        return False, result


class EdgeEventAdapter:
    """
    Thin adapter around the validated edge-event selector.
    """

    def __init__(self):
        module = load_module(
            EVENT_MODULE_PATH,
            "ecosphere_edge_event_module",
        )

        self.selector = module.EdgeEventIntelligence()

    def observe(self, result):
        return self.selector.observe(result)


class OptionalCloudFeedback:
    """
    Cloud-feedback adapter.

    This deliberately does not put a network call in the inference path.

    Call analyze_and_apply() only when the application has accumulated
    events and explicitly wants a feedback update.
    """

    def __init__(self):
        self.module = load_module(
            FEEDBACK_MODULE_PATH,
            "ecosphere_cloud_feedback_module",
        )

        self.engine = self.module.CloudFeedbackEngine()
        self.edge_config = self.module.EdgeConfig()

    def analyze_and_apply(self, events):
        """
        Analyze accumulated events and apply a newer configuration
        if recommended by the cloud-feedback engine.
        """

        recommendation = self.engine.analyze(
            events,
            self.edge_config,
        )

        if (
            recommendation.config_version
            > self.edge_config.config_version
        ):
            self.edge_config = recommendation

            return True, asdict(self.edge_config)

        return False, asdict(self.edge_config)


class ECOSphereRuntime:
    """
    One stateful local runtime instance.

    The runtime owns its own:

        - drift trackers
        - decision-region cache
        - temporal intelligence state
        - event selector
        - cloud-feedback configuration
        - runtime statistics
    """

    def __init__(
        self,
        enable_cache=True,
        enable_events=True,
        enable_cloud_feedback=True,
    ):
        # Load the validated master pipeline without modifying it.
        self.master = load_module(
            MASTER_PATH,
            "ecosphere_master_runtime",
        )

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {MODEL_PATH.name}"
            )

        # Load the optimized model used by the integrated edge path.
        self.model = joblib.load(MODEL_PATH)

        # Validate feature ordering when available.
        if hasattr(self.model, "feature_names_in_"):
            actual = list(
                self.model.feature_names_in_
            )

            if actual != FEATURE_COLUMNS:
                raise ValueError(
                    "Optimized model feature order mismatch.\n"
                    f"Expected: {FEATURE_COLUMNS}\n"
                    f"Actual:   {actual}"
                )

        self.enable_cache = enable_cache
        self.enable_events = enable_events
        self.enable_cloud_feedback = enable_cloud_feedback

        # Independent state:
        # production module-level trackers remain untouched.
        self.gas_tracker = (
            self.master.BaselineDriftTracker()
        )

        self.tilt_tracker = (
            self.master.BaselineDriftTracker()
        )

        self.vibration_tracker = (
            self.master.BaselineDriftTracker()
        )

        self.cache = (
            DecisionRegionCache(self.model)
            if enable_cache
            else None
        )

        self.events = (
            EdgeEventAdapter()
            if enable_events
            else None
        )

        self.feedback = (
            OptionalCloudFeedback()
            if enable_cloud_feedback
            else None
        )

        self.stats = RuntimeStats()

        self.event_log = []

        # Temporal state belongs to this runtime instance.
        self.temporal = TemporalIntelligence(
            window_size=5
        )

        # Physical-AI action state.
        self.rover_controller = RoverActionController()

        self.previous_action = None
        self.previous_risk = None
        self.previous_trend = None
        
        self.recovery_tracker = RecoveryTracker(
             required_falling_observations=2
        )

    def _correct_and_fuse(
        self,
        gas_ppm,
        tilt_deg,
        vibration_g,
    ):
        """
        Apply independent drift correction and then feature fusion.
        """

        corrected_gas = (
            self.gas_tracker.update_and_correct(
                gas_ppm
            )
        )

        corrected_tilt = (
            self.tilt_tracker.update_and_correct(
                tilt_deg
            )
        )

        corrected_vib = (
            self.vibration_tracker.update_and_correct(
                vibration_g
            )
        )

        features = self.master.apply_fusion(
            corrected_gas,
            corrected_tilt,
            corrected_vib,
        )

        corrected = {
            "gas_ppm": corrected_gas,
            "tilt_deg": corrected_tilt,
            "vibration_g": corrected_vib,
        }

        return corrected, features

    def _classify_without_cache(self, features):
        """
        Execute the optimized Random Forest directly.

        Used when the decision-region cache is disabled.
        """

        frame = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS,
        )

        prediction = self.model.predict(frame)[0]

        probabilities = self.model.predict_proba(frame)[0]

        confidence = dict(
            zip(
                self.model.classes_,
                probabilities.round(3),
            )
        )

        return {
            "hazard_state": str(prediction).upper(),
            "confidence": confidence,
        }

    def step(
        self,
        gas_ppm,
        tilt_deg,
        vibration_g,
    ):
        """
        Process one sensor observation through the complete
        integrated edge pipeline.
        """

        raw = {
            "gas_ppm": gas_ppm,
            "tilt_deg": tilt_deg,
            "vibration_g": vibration_g,
        }

        # ---------------------------------------------------------
        # 1. Drift correction + feature fusion
        # ---------------------------------------------------------

        corrected, features = self._correct_and_fuse(
            gas_ppm,
            tilt_deg,
            vibration_g,
        )

        # ---------------------------------------------------------
        # 2. Optimized RF + exact decision-region cache
        # ---------------------------------------------------------

        if self.enable_cache:
            cache_hit, classification = (
                self.cache.lookup(features)
            )
        else:
            classification = (
                self._classify_without_cache(
                    features
                )
            )

            cache_hit = False

        # ---------------------------------------------------------
        # 3. Environmental analysis
        # ---------------------------------------------------------

        environment = analyze_environment(
            features,
            classification["hazard_state"],
            classification["confidence"],
        )

        # ---------------------------------------------------------
        # 4. Temporal intelligence
        # ---------------------------------------------------------

        temporal = self.temporal.update(
            features["joint_risk"]
        )

        # ---------------------------------------------------------
        # 5. Near-term risk projection
        # ---------------------------------------------------------

        projection = project_risk(
            features["joint_risk"],
            temporal["slope"],
            temporal["trend"],
            horizon_steps=1,
        )

        # Temporal projection is not reliable until at least
        # three observations are available.
        if temporal["trend"] == "INSUFFICIENT_DATA":
            projection["trajectory"] = (
                "INSUFFICIENT_DATA"
            )

            projection["projected_level"] = (
                "UNKNOWN"
            )

        # ---------------------------------------------------------
        # 6. Interpretable risk reasoning
        # ---------------------------------------------------------

        model_confidence = max(
            classification["confidence"].values()
        )

        risk_assessment = assess_risk(
            classification["hazard_state"],
            model_confidence,
            environment,
            temporal,
            projection,
        )

        # ---------------------------------------------------------
        # 7. Verify the previous physical intervention
        # ---------------------------------------------------------

        current_risk = features["joint_risk"]
        current_trend = temporal["trend"]

        verification = None

        # Only intervention-oriented actions require consequence
        # verification. CONTINUE/MONITOR do not create an explicit
        # recovery expectation.
        intervention_actions = {
            "SLOW_AND_MONITOR",
            "PAUSE_AND_VERIFY",
            "STOP_AND_REROUTE",
        }

        if (
            self.previous_action in intervention_actions
            and self.previous_risk is not None
            and self.previous_trend is not None
        ):
            verification = verify_action(
                self.previous_action,
                self.previous_risk,
                current_risk,
                self.previous_trend,
                current_trend,
            )
            
        # ---------------------------------------------------------
        # 8. Recovery tracking
        # ---------------------------------------------------------

        recovery = self.recovery_tracker.update(
            current_trend,
            current_risk,
        )    

        # ---------------------------------------------------------
        # 9. Make the current decision
        # ---------------------------------------------------------

        decision = decide(risk_assessment)

        # ---------------------------------------------------------
        # 10. Execute the current physical action
        # ---------------------------------------------------------

        rover_state = self.rover_controller.execute(
            decision
        )

        # Store this observation/action pair so the NEXT sensor
        # observation can verify its environmental consequence.
        self.previous_action = decision["action"]
        self.previous_risk = current_risk
        self.previous_trend = current_trend

        # ---------------------------------------------------------
        # 11. Runtime statistics
        # ---------------------------------------------------------

        self.stats.readings += 1

        if cache_hit:
            self.stats.cache_hits += 1
        else:
            self.stats.rf_executions += 1

        # ---------------------------------------------------------
        # 12. Integrated result
        # ---------------------------------------------------------

        result = {
            "raw_input": raw,
            "corrected_input": corrected,

            "hazard_state": (
                classification["hazard_state"]
            ),

            "confidence": (
                classification["confidence"]
            ),

            "environment": environment,

            "temporal": temporal,

            "projection": projection,

            "risk_assessment": risk_assessment,

            "decision": decision,

            "verification": verification,

            "rover_state": rover_state,
            "recovery": recovery,
            "cache_hit": cache_hit,

            "rf_executed": not cache_hit,

            "config_version": (
                self.feedback.edge_config.config_version
                if self.feedback is not None
                else 1
            ),
        }

        # ---------------------------------------------------------
        # 13. Edge event intelligence
        # ---------------------------------------------------------

        if self.enable_events:
            generated = self.events.observe(
                result
            )

            self.event_log.extend(
                generated
            )

            self.stats.events_generated += (
                len(generated)
            )

        return result

    def get_cloud_feedback(self):
        """
        Analyze currently accumulated edge events.

        This is intentionally separate from step():
        cloud feedback cannot block real-time local inference.
        """

        if not self.enable_cloud_feedback:
            return False, None

        applied, config = (
            self.feedback.analyze_and_apply(
                self.event_log
            )
        )

        return applied, config

    def summary(self):
        """
        Return runtime statistics and current configuration state.
        """

        readings = self.stats.readings

        hit_rate = (
            100.0 * self.stats.cache_hits / readings
            if readings
            else 0.0
        )

        return {
            "readings": readings,

            "cache_hits": self.stats.cache_hits,

            "cache_bypass_percent": round(
                hit_rate,
                2,
            ),

            "rf_executions": (
                self.stats.rf_executions
            ),

            "events_generated": (
                self.stats.events_generated
            ),

            "config_version": (
                self.feedback.edge_config.config_version
                if self.feedback is not None
                else 1
            ),
        }


def main():
    """
    Small integration smoke test.

    This is intentionally not a performance benchmark.

    Use 11_verify_benchmarks.py for benchmark claims.
    """

    runtime = ECOSphereRuntime(
        enable_cache=True,
        enable_events=True,
        enable_cloud_feedback=True,
    )

    test_stream = [
        (75, 2, 0.04),
        (75, 2, 0.04),
        (480, 3, 0.05),
        (700, 27, 0.70),
        (700, 27, 0.70),
        (75, 2, 0.04),
    ]

    print("=" * 72)

    print(
        "ECOsphere -- Integrated Runtime Smoke Test"
    )

    print("=" * 72)

    for i, reading in enumerate(
        test_stream,
        start=1,
    ):
        result = runtime.step(*reading)

        print(
            f"{i:02d}: "
            f"hazard={result['hazard_state']:<9} "
            f"cache_hit={str(result['cache_hit']):<5} "
            f"RF={'yes' if result['rf_executed'] else 'no'}"
        )

        print(
            f"    temporal="
            f"{result['temporal']['trend']:<17} "
            f"risk="
            f"{result['risk_assessment']['risk_level']:<9} "
            f"urgency="
            f"{result['risk_assessment']['urgency']}"
        )

        if result["verification"] is not None:
            print(
                f"    verification="
                f"{result['verification']}"
            )

        print(
            f"    action="
            f"{result['decision']['action']}"
        )

    # -------------------------------------------------------------
    # Optional cloud feedback
    # -------------------------------------------------------------

    applied, config = (
        runtime.get_cloud_feedback()
    )

    print("\n--- Runtime summary ---")

    print(
        json.dumps(
            runtime.summary(),
            indent=2,
        )
    )

    print("\n--- Cloud feedback ---")

    print(
        "configuration update applied : "
        f"{'YES' if applied else 'NO'}"
    )

    print(
        json.dumps(
            config,
            indent=2,
        )
    )

    print("\n--- Integration guarantees ---")

    print(
        "local inference path         : AVAILABLE"
    )

    print(
        "decision-region cache        : ENABLED"
    )

    print(
        "temporal intelligence        : ENABLED"
    )

    print(
        "near-term projection         : ENABLED"
    )

    print(
        "risk reasoning               : ENABLED"
    )

    print(
        "event intelligence           : ENABLED"
    )

    print(
        "cloud feedback               : "
        "OPTIONAL / NON-BLOCKING"
    )

    print(
        "benchmark methodology        : UNCHANGED"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()