"""
ECOsphere Recovery Tracker

Confirms environmental recovery using temporal evidence rather than
a single improved sensor reading.

Recovery requires two consecutive FALLING observations.
"""

class RecoveryTracker:
    def __init__(self, required_falling_observations=2):
        self.required_falling_observations = (
            required_falling_observations
        )

        self.falling_count = 0
        self.state = "NO_RECOVERY"

    def update(self, trend, risk):
        trend = str(trend).upper()

        if trend == "FALLING":
            self.falling_count += 1

            if (
                self.falling_count
                >= self.required_falling_observations
            ):
                self.state = "RECOVERY_CONFIRMED"

            else:
                self.state = "RECOVERY_PROGRESS"

        elif trend == "RISING":
            self.falling_count = 0
            self.state = "RECOVERY_ABORTED"

        else:
            # Stable / insufficient data does not confirm recovery.
            self.state = "RECOVERY_PENDING"

        return {
            "state": self.state,
            "falling_count": self.falling_count,
            "required_observations": (
                self.required_falling_observations
            ),
            "risk": round(float(risk), 3),
            "recovery_confirmed": (
                self.state == "RECOVERY_CONFIRMED"
            ),
        }


if __name__ == "__main__":

    tracker = RecoveryTracker(
        required_falling_observations=2
    )

    test_sequence = [
        ("RISING", 0.838),
        ("RISING", 0.367),
        ("FALLING", 0.178),
        ("FALLING", 0.131),
        ("FALLING", 0.089),
        ("RISING", 0.250),
    ]

    for i, (trend, risk) in enumerate(
        test_sequence,
        start=1,
    ):
        result = tracker.update(
            trend,
            risk,
        )

        print(
            f"{i}: trend={trend:<10} "
            f"risk={risk:.3f} "
            f"state={result['state']:<20} "
            f"falling_count={result['falling_count']} "
            f"confirmed={result['recovery_confirmed']}"
        )