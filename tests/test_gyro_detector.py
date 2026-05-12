import sys
import types
from pathlib import Path

import numpy as np


try:
    import pylsl  # noqa: F401
except Exception:
    pylsl_stub = types.ModuleType("pylsl")
    pylsl_stub.StreamInlet = object
    pylsl_stub.resolve_byprop = lambda *args, **kwargs: []
    sys.modules["pylsl"] = pylsl_stub


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gyro_detector import VelocityDetector


def make_detector(**overrides):
    defaults = {
        "vel_forward": 30.0,
        "vel_backward": 30.0,
        "vel_left": 80.0,
        "vel_right": 80.0,
        "scale_factor": 1.0,
        "smoothing_window": 1,
        "deadzone_x": 5.0,
        "deadzone_y": 5.0,
        "deadzone_z": 5.0,
    }
    defaults.update(overrides)
    return VelocityDetector(**defaults)


def test_forward_velocity_above_legacy_auto_reset_stays_forward():
    detector = make_detector()

    direction, velocity = detector.process_sample(np.array([0.0, 70.0, 0.0]))

    assert direction == "forward"
    assert velocity[1] == 70.0
    assert detector.baseline[1] == 0.0


def test_auto_reset_can_be_enabled_explicitly():
    detector = make_detector(auto_reset_multiplier=2.0)

    direction, _velocity = detector.process_sample(np.array([0.0, 70.0, 0.0]))

    assert direction is None
    assert detector.baseline[1] == 70.0


def test_gamepad_continued_forward_does_not_center_without_opposite():
    detector = make_detector(
        gamepad_mode=True,
        gamepad_activation_samples=1,
    )

    first_direction, _first_velocity = detector.process_sample(np.array([0.0, 70.0, 0.0]))
    next_direction, _next_velocity = detector.process_sample(np.array([0.0, 75.0, 0.0]))

    assert first_direction == "forward"
    assert next_direction is None
    assert detector.current_direction == "forward"


def test_gamepad_opposite_motion_returns_to_center_first():
    detector = make_detector(
        gamepad_mode=True,
        gamepad_activation_samples=1,
    )

    first_direction, _first_velocity = detector.process_sample(np.array([0.0, -40.0, 0.0]))
    next_direction, _next_velocity = detector.process_sample(np.array([0.0, 45.0, 0.0]))

    assert first_direction == "backward"
    assert next_direction == "center"
    assert detector.current_direction == "center"
