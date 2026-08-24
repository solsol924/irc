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
    @staticmethod
    def _detector_harness() -> HoopVisionNode:
        node = HoopVisionNode.__new__(HoopVisionNode)
        node.min_contour_area = 200.0
        node.min_backboard_aspect_ratio = 1.05
        node.max_backboard_aspect_ratio = 6.0
        node.top_band_ratio = 0.15
        node.side_band_ratio = 0.10
        node.side_vertical_end_ratio = 0.75
        node.red_ratio_min = 0.55
        node.white_inner_ratio_min = 0.50
        node.occlusion_merge_gap_px = 41
        node.min_visible_red_bands = 2
        node.red_band_average_min = 0.40
        node.depth_min_m = 0.08
        node.depth_max_m = 2.0
        node.min_valid_depth_pixels = 20
        node.center_depth_patch_radius = 5
        node.min_valid_center_depth_pixels = 5
        node.fx = 607.0
        node.fy = 606.0
        node.cx_intr = 160.0
        node.cy_intr = 120.0
        return node

    @staticmethod
    def _synthetic_backboard_masks():
        red = np.zeros((240, 320), dtype=np.uint8)
        white = np.zeros_like(red)
        depth = np.ones(red.shape, dtype=np.float32)

        red[40:59, 60:261] = 255
        red[40:156, 60:79] = 255
        red[40:156, 242:261] = 255
        white[59:128, 79:242] = 255
        return red, white, depth

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

    def test_red_band_evidence_allows_one_partially_occluded_band(self) -> None:
        passed, visible_count, average = (
            HoopVisionNode._red_band_evidence_passes(
                (0.82, 0.18, 0.76),
                red_ratio_min=0.55,
                min_visible_red_bands=2,
                red_band_average_min=0.40,
            )
        )

        self.assertTrue(passed)
        self.assertEqual(visible_count, 2)
        self.assertAlmostEqual(average, (0.82 + 0.18 + 0.76) / 3.0)

    def test_red_band_evidence_rejects_only_one_visible_band(self) -> None:
        passed, visible_count, _ = HoopVisionNode._red_band_evidence_passes(
            (0.80, 0.10, 0.12),
            red_ratio_min=0.55,
            min_visible_red_bands=2,
            red_band_average_min=0.40,
        )

        self.assertFalse(passed)
        self.assertEqual(visible_count, 1)

    def test_detects_backboard_when_top_border_is_split_by_occlusion(self) -> None:
        node = self._detector_harness()
        red, white, depth = self._synthetic_backboard_masks()
        red[36:65, 143:178] = 0

        node.occlusion_merge_gap_px = 0
        without_fragment_merge = node._find_best_hoop(
            red_mask=red,
            white_mask=white,
            roi_depth_m=depth,
            roi_x_start=0,
            roi_y_start=0,
            frame_width=320,
            frame_height=240,
        )
        self.assertIsNone(without_fragment_merge)

        node.occlusion_merge_gap_px = 41
        detection = node._find_best_hoop(
            red_mask=red,
            white_mask=white,
            roi_depth_m=depth,
            roi_x_start=0,
            roi_y_start=0,
            frame_width=320,
            frame_height=240,
        )

        self.assertIsNotNone(detection)
        self.assertTrue(detection["detected"])
        self.assertGreaterEqual(detection["visible_red_bands"], 2)

    def test_detects_backboard_when_one_side_is_mostly_occluded(self) -> None:
        node = self._detector_harness()
        red, white, depth = self._synthetic_backboard_masks()
        red[64:126, 56:83] = 0

        detection = node._find_best_hoop(
            red_mask=red,
            white_mask=white,
            roi_depth_m=depth,
            roi_x_start=0,
            roi_y_start=0,
            frame_width=320,
            frame_height=240,
        )

        self.assertIsNotNone(detection)
        self.assertEqual(detection["visible_red_bands"], 2)
        self.assertTrue(detection["occlusion_tolerant"])


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

    def test_hoop_debug_image_has_priority_while_hoop_mode_is_active(self) -> None:
        now = time.monotonic()
        harness = SimpleNamespace(
            ball_detected=True,
            hurdle_detected=True,
            hoop_detected=True,
            ball_state_time=now,
            hurdle_state_time=now,
            hoop_state_time=now,
            state_timeout_sec=0.5,
            selected_source="hoop",
            ball_enabled=False,
            hoop_enabled=True,
        )

        source = RealSenseDebugSelector._active_source(harness)

        self.assertEqual(source, "hoop")


if __name__ == "__main__":
    unittest.main()
