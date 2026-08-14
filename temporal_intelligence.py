"""
ECOsphere Temporal Intelligence Layer

Tracks recent environmental risk and determines whether the
environment is stable, improving, or deteriorating.

This layer does not replace the classifier.
It reasons over the temporal evolution of existing risk features.
"""

from collections import deque


class TemporalIntelligence:
    def __init__(self, window_size=5):
        if window_size < 3:
            raise ValueError("window_size must be at least 3")

        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, joint_risk):
        """
        Add a new joint-risk observation and return temporal analysis.
        """

        value = float(joint_risk)
        self.history.append(value)

        values = list(self.history)

        # Not enough observations to establish a reliable trend.
        if len(values) < 3:
            return {
                "trend": "INSUFFICIENT_DATA",
                "slope": 0.0,
                "trend_strength": 0.0,
                "persistence": len(values),
                "window_size": self.window_size,
            }

        # Simple least-squares slope over equally spaced observations.
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum(
            (i - x_mean) * (y - y_mean)
            for i, y in enumerate(values)
        )

        denominator = sum(
            (i - x_mean) ** 2
            for i in range(n)
        )

        slope = numerator / denominator if denominator else 0.0

        # Convert slope magnitude into a bounded 0-1 strength.
        trend_strength = min(abs(slope) * 5.0, 1.0)

        if slope > 0.02:
            trend = "RISING"
        elif slope < -0.02:
            trend = "FALLING"
        else:
            trend = "STABLE"

        return {
            "trend": trend,
            "slope": round(slope, 4),
            "trend_strength": round(trend_strength, 3),
            "persistence": len(values),
            "window_size": self.window_size,
        }


if __name__ == "__main__":
    scenarios = {
        "Stable": [0.40, 0.41, 0.39, 0.40, 0.42],
        "Rising": [0.30, 0.42, 0.55, 0.68, 0.80],
        "Falling": [0.82, 0.70, 0.55, 0.40, 0.25],
        "Rapid escalation": [0.30, 0.50, 0.72, 0.88, 0.95],
    }

    for name, readings in scenarios.items():
        tracker = TemporalIntelligence(window_size=5)

        result = None

        for reading in readings:
            result = tracker.update(reading)

        print(f"\n--- {name} ---")
        print(result)