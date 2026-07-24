#!/usr/bin/env python3
"""Synthetic tests for ball/support context validation."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ball_detector_core import (  # noqa: E402
    BallSupportConfig,
    black_support_mask,
    evaluate_ball_support,
    hsv_range_mask,
)


class BallSupportValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 480
        self.width = 640
        self.fx = 600.0
        self.depth_m = 1.0
        self.ball_radius = 17
        self.support_radius = 45
        self.config = BallSupportConfig()

    @staticmethod
    def _masks(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ball = hsv_range_mask(hsv, (9, 120, 80), (20, 255, 255))
        floor = hsv_range_mask(hsv, (0, 80, 40), (8, 255, 255))
        return ball, floor

    def _scene(
        self,
        center: tuple[int, int],
        with_support: bool,
    ) -> np.ndarray:
        hsv = np.empty((self.height, self.width, 3), dtype=np.uint8)
        hsv[:, :] = (5, 200, 160)  # red competition floor
        if with_support:
            cv2.circle(
                hsv,
                center,
                self.support_radius,
                # Dark pixels may carry arbitrary H/S noise and overlap the
                # floor profile.  V must take precedence for black support.
                (5, 200, 60),
                thickness=-1,
            )
        cv2.circle(
            hsv,
            center,
            self.ball_radius,
            (14, 230, 220),
            thickness=-1,
        )
        return hsv

    def _evaluate(
        self,
        hsv: np.ndarray,
        center: tuple[int, int],
        touches_edge: bool,
    ) -> dict:
        ball_mask, floor_mask = self._masks(hsv)
        support_mask = black_support_mask(
            hsv,
            self.config.support_v_max,
        )
        return evaluate_ball_support(
            hsv=hsv,
            ball_color_mask=ball_mask,
            floor_mask=floor_mask,
            center=center,
            detected_radius_px=float(self.ball_radius),
            expected_ball_radius_px=16.5,
            depth_m=self.depth_m,
            focal_x_px=self.fx,
            touches_edge=touches_edge,
            config=self.config,
            black_mask=support_mask,
        )

    def test_center_ball_on_black_support_is_accepted(self) -> None:
        center = (320, 240)
        result = self._evaluate(
            self._scene(center, with_support=True),
            center,
            touches_edge=False,
        )
        self.assertTrue(result["accepted"], result)
        self.assertGreater(result["black_ratio"], 0.9)

    def test_orange_patch_on_red_floor_is_rejected(self) -> None:
        center = (320, 240)
        result = self._evaluate(
            self._scene(center, with_support=False),
            center,
            touches_edge=False,
        )
        self.assertFalse(result["accepted"], result)
        self.assertIn("not_enough_black", result["reason"])
        self.assertGreater(result["floor_ratio"], 0.9)

    def test_partially_visible_edge_support_is_accepted(self) -> None:
        center = (320, 474)
        result = self._evaluate(
            self._scene(center, with_support=True),
            center,
            touches_edge=True,
        )
        self.assertTrue(result["accepted"], result)
        self.assertGreaterEqual(
            result["qualified_sectors"],
            self.config.edge_min_sectors,
        )


if __name__ == "__main__":
    unittest.main()
