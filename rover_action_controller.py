"""
ECOsphere Rover Action Controller

Converts an autonomous decision into a rover state transition.

This module does NOT directly control physical motors.
It provides a hardware-agnostic action interface that can later
be connected to a simulation adapter or real rover hardware.
"""


class RoverActionController:

    VALID_ACTIONS = {
        "CONTINUE",
        "MONITOR",
        "SLOW_AND_MONITOR",
        "PAUSE_AND_VERIFY",
        "STOP_AND_REROUTE",
    }

    def __init__(self):
        self.state = {
            "motion_state": "MOVING",
            "speed_mode": "NORMAL",
            "route_state": "PRIMARY",
            "last_action": "CONTINUE",
        }

        self.action_history = []

    def execute(self, decision):
        """
        Apply a decision to the rover's logical state.

        Returns the resulting rover state.
        """

        action = str(
            decision.get("action", "CONTINUE")
        ).upper()

        if action not in self.VALID_ACTIONS:
            raise ValueError(
                f"Unsupported rover action: {action}"
            )

        if action == "CONTINUE":
            self.state.update({
                "motion_state": "MOVING",
                "speed_mode": "NORMAL",
                "route_state": "PRIMARY",
            })

        elif action == "MONITOR":
            self.state.update({
                "motion_state": "MOVING",
                "speed_mode": "NORMAL",
                "route_state": "PRIMARY",
            })

        elif action == "SLOW_AND_MONITOR":
            self.state.update({
                "motion_state": "MOVING",
                "speed_mode": "REDUCED",
                "route_state": "PRIMARY",
            })

        elif action == "PAUSE_AND_VERIFY":
            self.state.update({
                "motion_state": "PAUSED",
                "speed_mode": "STOPPED",
                "route_state": "PRIMARY",
            })

        elif action == "STOP_AND_REROUTE":
            self.state.update({
                "motion_state": "STOPPED",
                "speed_mode": "STOPPED",
                "route_state": "REROUTING",
            })

        self.state["last_action"] = action

        self.action_history.append({
            "action": action,
            "state": self.state.copy(),
        })

        return self.state.copy()


if __name__ == "__main__":

    controller = RoverActionController()

    decisions = [
        {
            "action": "CONTINUE"
        },
        {
            "action": "MONITOR"
        },
        {
            "action": "SLOW_AND_MONITOR"
        },
        {
            "action": "PAUSE_AND_VERIFY"
        },
        {
            "action": "STOP_AND_REROUTE"
        },
    ]

    for decision in decisions:

        state = controller.execute(decision)

        print(
            f"{decision['action']:<20} → "
            f"motion={state['motion_state']:<8} "
            f"speed={state['speed_mode']:<8} "
            f"route={state['route_state']}"
        )