import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
from std_msgs.msg import Bool


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "realsense_yolo_detector.py"
)
SPEC = importlib.util.spec_from_file_location(
    "realsense_yolo_detector",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_loads_dedicated_realsense_yolo_settings(tmp_path):
    ini = tmp_path / "settings.ini"
    ini.write_text(
        """
[realsense_yolo]
model = /tmp/realsense.engine
goal_class = goal
backboard_class = backboard
ball_class = ball
max_fps = 30
publish_debug_image = true
""".strip(),
        encoding="utf-8",
    )

    config = MODULE.load_config(str(ini))

    assert config["model"] == "/tmp/realsense.engine"
    assert config["goal_class"] == "goal"
    assert config["backboard_class"] == "backboard"
    assert config["ball_class"] == "ball"
    assert config["max_fps"] == 30.0
    assert config["publish_debug_image"] is True


def test_ball_and_hoop_views_have_yellow_boxes_and_information_panels():
    detector = MODULE.RealSenseYoloDetector.__new__(
        MODULE.RealSenseYoloDetector
    )
    detector.cfg = {
        "ball_class": "ball",
        "backboard_class": "backboard",
        "goal_class": "goal",
    }
    detector.centerline_x_px = 120.0
    detector.ball_active = True
    detector.hoop_active = False

    ball = MODULE.Detection("ball", 2, 0.91, 40, 175, 90, 225)
    backboard = MODULE.Detection(
        "backboard",
        1,
        0.88,
        145,
        155,
        260,
        225,
    )
    goal = MODULE.Detection("goal", 0, 0.72, 165, 170, 235, 215)
    ball_state = {
        "realsense_ball_detected": True,
        "realsense_ball_distance_cm": 81.2,
        "realsense_ball_angle_error": -4.3,
        "raw_x_m": -0.06,
        "raw_y_m": 0.14,
        "raw_z_m": 0.79,
    }
    hoop_state = {
        "detected": True,
        "center_x": 202.5,
        "center_y": 87.5,
        "realsense_goal_distance_cm": 170.0,
        "realsense_goal_angle": 8.0,
        "center_depth_cm": 168.0,
    }
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    ball_view = detector._draw_debug_view(
        frame,
        "ball",
        ball,
        backboard,
        goal,
        ball_state,
        hoop_state,
        12.4,
    )
    hoop_view = detector._draw_debug_view(
        frame,
        "hoop",
        ball,
        backboard,
        goal,
        ball_state,
        hoop_state,
        12.4,
    )
    combined_view = detector._draw_debug_view(
        frame,
        "combined",
        ball,
        backboard,
        goal,
        ball_state,
        hoop_state,
        12.4,
    )

    yellow = np.array(MODULE.YELLOW, dtype=np.uint8)
    assert np.array_equal(ball_view[175, 40], yellow)
    assert np.array_equal(hoop_view[155, 145], yellow)
    assert np.array_equal(combined_view[175, 40], yellow)
    assert np.array_equal(combined_view[155, 145], yellow)
    assert np.count_nonzero(cv2.inRange(ball_view, yellow, yellow)) > 100
    assert np.count_nonzero(cv2.inRange(hoop_view, yellow, yellow)) > 100
    assert not np.array_equal(ball_view, hoop_view)


def test_display_mode_survives_off_then_switches_on_next_active_signal():
    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Logger:
        def info(self, _message):
            pass

    detector = MODULE.RealSenseYoloDetector.__new__(
        MODULE.RealSenseYoloDetector
    )
    detector.ball_active = True
    detector.hoop_active = False
    detector.display_mode = "ball"
    detector.ball_state_pub = Publisher()
    detector.ball_detected_pub = Publisher()
    detector.hoop_state_pub = Publisher()
    detector.hoop_detected_pub = Publisher()
    detector.get_logger = lambda: Logger()

    detector.cb_ball_active(Bool(data=False))
    assert detector.ball_active is False
    assert detector.display_mode == "ball"

    detector.cb_hoop_active(Bool(data=True))
    assert detector.hoop_active is True
    assert detector.display_mode == "hoop"
