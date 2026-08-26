from collections import deque
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
from std_msgs.msg import Bool


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ball_vision_fusion import BallVisionFusionNode  # noqa: E402
from ball_detector_core import BallMatchConfig  # noqa: E402


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def test_inactive_ball_callbacks_skip_image_and_webcam_processing():
    harness = SimpleNamespace(ball_detection_active=False)

    BallVisionFusionNode.cb_realsense_images(harness, object(), object())
    BallVisionFusionNode.cb_webcam_state(harness, object())


def test_ball_activity_switch_clears_state_and_controls_subscriptions():
    logger = _Logger()
    enabled_values = []
    harness = SimpleNamespace(
        ball_detection_active=True,
        latest_realsense={"realsense_ball_detected": True},
        latest_realsense_time=1.0,
        last_realsense_detection={"realsense_ball_detected": True},
        realsense_lost_frames=0,
        realsense_hold_frames=3,
        realsense_track_confirmed=True,
        realsense_track_velocity_px=(3.0, 1.0),
        raw_detection_history=deque([{"raw": True}], maxlen=3),
        latest_webcam={"webcam_ball_detected": True},
        latest_webcam_time=1.0,
        get_logger=lambda: logger,
    )
    harness._clear_ball_detection_state = MethodType(
        BallVisionFusionNode._clear_ball_detection_state,
        harness,
    )
    harness.ball_status_publisher = SimpleNamespace(
        set_detection_enabled=lambda enabled: enabled_values.append(enabled)
    )

    BallVisionFusionNode.cb_ball_active(harness, Bool(data=False))
    assert harness.ball_detection_active is False
    assert harness.latest_realsense is None
    assert harness.latest_webcam is None
    assert harness.realsense_track_confirmed is False
    assert list(harness.raw_detection_history) == []
    assert enabled_values == [False]

    BallVisionFusionNode.cb_ball_active(harness, Bool(data=True))
    assert harness.ball_detection_active is True
    assert enabled_values == [False, True]


def _raw_candidate(x=200.0, y=220.0, z=1.0, radius=16.0):
    return {
        "realsense_ball_detected": True,
        "realsense_ball_distance_cm": z * 100.0,
        "realsense_ball_angle_error": 0.0,
        "raw_ball_x": x,
        "raw_ball_y": y,
        "raw_z_m": z,
        "raw_radius": radius,
        "held_previous_detection": False,
    }


def _tracking_harness():
    harness = SimpleNamespace(
        confirmation_window_frames=3,
        confirmation_required_hits=2,
        ball_match_config=BallMatchConfig(),
        raw_detection_history=deque(maxlen=3),
        realsense_track_confirmed=False,
        realsense_track_velocity_px=(0.0, 0.0),
        realsense_lost_frames=3,
        realsense_hold_frames=3,
        last_realsense_detection=None,
        latest_realsense=None,
        latest_realsense_time=0.0,
    )
    for method_name in (
        "_empty_realsense_state",
        "_confirmation_hits",
        "_accept_confirmed_detection",
        "_update_realsense_tracking",
    ):
        setattr(
            harness,
            method_name,
            MethodType(getattr(BallVisionFusionNode, method_name), harness),
        )
    return harness


def test_two_of_three_raw_detections_confirm_but_held_does_not_vote():
    harness = _tracking_harness()
    first = _raw_candidate()
    second = _raw_candidate(x=245.0, y=235.0, z=1.1, radius=17.0)

    held = harness._update_realsense_tracking(
        detection=first,
        rejection_diagnostic={"category": "shape", "detail": "test"},
        now=1.0,
    )
    assert held is False
    assert harness.realsense_track_confirmed is False
    assert harness.latest_realsense["realsense_ball_detected"] is False

    # 중간 miss는 raw 창에 None으로 들어가며 HELD 복사본은 들어가지 않는다.
    harness._update_realsense_tracking(
        detection=None,
        rejection_diagnostic={"category": "shape", "detail": "test"},
        now=2.0,
    )
    harness._update_realsense_tracking(
        detection=second,
        rejection_diagnostic={"category": "shape", "detail": "test"},
        now=3.0,
    )

    assert harness.realsense_track_confirmed is True
    assert harness.latest_realsense["realsense_ball_detected"] is True


def test_held_detection_is_display_only_and_never_action_true():
    harness = _tracking_harness()
    first = _raw_candidate()
    second = _raw_candidate(x=220.0)
    harness._update_realsense_tracking(
        detection=first,
        rejection_diagnostic={"category": "shape", "detail": "test"},
        now=1.0,
    )
    harness._update_realsense_tracking(
        detection=second,
        rejection_diagnostic={"category": "shape", "detail": "test"},
        now=2.0,
    )

    held = harness._update_realsense_tracking(
        detection=None,
        rejection_diagnostic={"category": "shape", "detail": "blur"},
        now=3.0,
    )

    assert held is True
    assert harness.latest_realsense["held_previous_detection"] is True
    assert harness.latest_realsense["realsense_ball_detected"] is False


def test_runtime_detector_uses_inner_depth_and_soft_support_score():
    harness = SimpleNamespace(
        edge_ball_margin_px=3,
        min_contour_area=120.0,
        edge_min_contour_area_ratio=0.65,
        min_aspect_ratio=0.55,
        max_aspect_ratio=1.80,
        edge_min_aspect_ratio=0.45,
        edge_max_aspect_ratio=2.20,
        min_circularity=0.38,
        edge_min_circularity=0.30,
        max_circle_ratio_error=0.65,
        edge_max_circle_ratio_error=0.75,
        fx=600.0,
        fy=600.0,
        cx_intr=100.0,
        cy_intr=100.0,
        ball_diameter_min_m=0.050,
        ball_diameter_max_m=0.070,
        radius_size_min_ratio=0.45,
        radius_size_max_ratio=1.70,
        edge_radius_size_min_ratio=0.60,
        edge_radius_size_max_ratio=1.50,
        depth_scale=0.001,
        depth_threshold_m=1.5,
        depth_inner_radius_ratio=0.50,
        depth_min_valid_pixels=8,
        depth_min_valid_ratio=0.15,
        support_score_weight=1.0,
        support_diameter_m=0.15,
        support_v_max=75,
        support_black_ratio_min=0.30,
        edge_support_black_ratio_min=0.25,
        support_ball_color_ratio_max=0.15,
        support_floor_ratio_max=0.35,
        support_sector_black_ratio_min=0.18,
        support_min_sectors=3,
        edge_support_min_sectors=2,
        support_min_visible_fraction=0.45,
        edge_support_min_visible_fraction=0.12,
        realsense_use_euclidean_distance=False,
    )
    for method_name in ("_support_config", "_read_depth_m", "_find_best_ball"):
        setattr(
            harness,
            method_name,
            MethodType(getattr(BallVisionFusionNode, method_name), harness),
        )

    hsv = np.empty((200, 200, 3), dtype=np.uint8)
    hsv[:, :] = (5, 200, 160)  # red floor; intentionally no black support
    cv2.circle(hsv, (100, 100), 12, (14, 230, 220), thickness=-1)
    ball_mask = cv2.inRange(hsv, (9, 120, 80), (20, 255, 255))
    floor_mask = cv2.inRange(hsv, (0, 80, 40), (8, 255, 255))
    depth_mm = np.full((200, 200), 1000, dtype=np.uint16)

    detection = harness._find_best_ball(
        mask=ball_mask,
        hsv=hsv,
        ball_color_mask=ball_mask,
        floor_mask=floor_mask,
        depth=depth_mm,
        roi_x_start=0,
        roi_y_start=0,
        frame_w=200,
        frame_h=200,
        rejection_counts={},
    )

    # 받침대 hard gate라면 None이어야 하지만 이제 후보는 유지되고 낮은
    # support confidence가 후보 순위에만 반영된다.
    assert detection is not None
    assert detection["raw_support_passed"] is False
    assert detection["raw_support_confidence"] < 0.5
    assert detection["raw_depth_sample_pixels"] > 9
    assert detection["raw_depth_valid_ratio"] == 1.0
