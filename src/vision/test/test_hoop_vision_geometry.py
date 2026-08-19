#!/usr/bin/env python3
"""백보드 중심 거리와 로봇 중심선 기준 각도 계산 테스트."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest

import numpy as np
from std_msgs.msg import String


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ball_vision_fusion import BallVisionFusionNode  # noqa: E402
from hoop_vision import HoopVisionNode  # noqa: E402
from realsense_debug_selector import RealSenseDebugSelector  # noqa: E402


class HoopGeometryTest(unittest.TestCase):
    def test_robot_reference_is_exact_screen_bottom_center(self) -> None:
        point = HoopVisionNode._robot_reference_point(
            frame_width=640,
            frame_height=480,
        )

        self.assertEqual(point, (320.0, 479.0))

    def test_rectangle_center_is_diagonal_intersection(self) -> None:
        box = np.array(
            [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]],
            dtype=np.float32,
        )

        center = HoopVisionNode._rectangle_center(box)

        np.testing.assert_allclose(center, [30.0, 40.0])

    def test_centerline_angle_is_signed_left_and_right(self) -> None:
        right = HoopVisionNode._centerline_error_angle_deg(
            center_x=340.0,
            center_y=300.0,
            robot_x=320.0,
            robot_y=480.0,
        )
        left = HoopVisionNode._centerline_error_angle_deg(
            center_x=300.0,
            center_y=300.0,
            robot_x=320.0,
            robot_y=480.0,
        )
        centered = HoopVisionNode._centerline_error_angle_deg(
            center_x=320.0,
            center_y=300.0,
            robot_x=320.0,
            robot_y=480.0,
        )

        self.assertIsNotNone(right)
        self.assertIsNotNone(left)
        self.assertGreater(right, 0.0)
        self.assertLess(left, 0.0)
        self.assertEqual(centered, 0.0)
        self.assertAlmostEqual(right, -left)
        self.assertAlmostEqual(
            right,
            math.degrees(math.atan2(20.0, 180.0)),
        )

    def test_center_pixel_offsets_use_robot_bottom_center(self) -> None:
        dx_px, dy_px = HoopVisionNode._center_pixel_offsets(
            center_x=350.0,
            center_y=180.0,
            robot_x=320.0,
            robot_y=479.0,
        )

        self.assertEqual(dx_px, 30.0)
        self.assertEqual(dy_px, 299.0)

    def test_hold_is_active_for_half_second(self) -> None:
        self.assertTrue(
            HoopVisionNode._hold_is_active(
                last_detection_time=10.0,
                current_time=10.5,
                hold_seconds=0.5,
            )
        )
        self.assertFalse(
            HoopVisionNode._hold_is_active(
                last_detection_time=10.0,
                current_time=10.5001,
                hold_seconds=0.5,
            )
        )

    def test_center_depth_uses_valid_patch_median(self) -> None:
        harness = SimpleNamespace(
            center_depth_patch_radius=1,
            min_valid_center_depth_pixels=3,
            depth_min_m=0.08,
            depth_max_m=2.0,
        )
        depth = np.zeros((5, 5), dtype=np.float32)
        depth[1:4, 1:4] = np.array(
            [
                [0.0, 1.0, 1.1],
                [1.2, 1.3, 1.4],
                [1.5, 1.6, 3.0],
            ],
            dtype=np.float32,
        )

        result = HoopVisionNode._center_depth_m(harness, depth, 2.0, 2.0)

        self.assertAlmostEqual(result, 1.3, places=6)

    def test_center_distance_is_euclidean_camera_to_center_distance(self) -> None:
        harness = SimpleNamespace(
            fx=100.0,
            fy=100.0,
            cx_intr=50.0,
            cy_intr=50.0,
        )

        result = HoopVisionNode._center_distance_m(
            harness,
            center_x=60.0,
            center_y=50.0,
            depth_m=2.0,
        )

        self.assertAlmostEqual(result, math.sqrt(4.04), places=6)


class HoopIntegrationTest(unittest.TestCase):
    def test_valid_hoop_state_is_kept_for_ball_result(self) -> None:
        harness = SimpleNamespace(
            latest_hoop=None,
            latest_hoop_time=0.0,
            get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
            _empty_hoop_state=BallVisionFusionNode._empty_hoop_state,
        )
        message = String()
        message.data = json.dumps(
            {
                "detected": True,
                "realsense_goal_distance_cm": 123.4,
                "realsense_goal_angle": -8.5,
            }
        )

        BallVisionFusionNode.cb_hoop_state(harness, message)

        self.assertTrue(harness.latest_hoop["hoop_detected"])
        self.assertEqual(
            harness.latest_hoop["realsense_goal_distance_cm"],
            123.4,
        )
        self.assertEqual(harness.latest_hoop["realsense_goal_angle"], -8.5)
        self.assertGreater(harness.latest_hoop_time, 0.0)

    def test_hoop_debug_image_has_priority_while_hoop_is_detected(self) -> None:
        now = time.monotonic()
        harness = SimpleNamespace(
            ball_detected=True,
            hurdle_detected=True,
            hoop_detected=True,
            ball_state_time=now,
            hurdle_state_time=now,
            hoop_state_time=now,
            state_timeout_sec=0.5,
        )

        source = RealSenseDebugSelector._active_source(harness)

        self.assertEqual(source, "hoop")


if __name__ == "__main__":
    unittest.main()
