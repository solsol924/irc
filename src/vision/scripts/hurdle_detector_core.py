#!/usr/bin/env python3
"""
Pure OpenCV hurdle detector for aligned RealSense color/depth frames.

The implementation combines two reference approaches:
- senior hurdle code: Lab color filtering, horizontal morphology, elongated
  contour filtering, cv2.fitLine angle, and short lost-frame holding;
- expo code: yellow HSV+BGR filtering, empty-floor depth calibration,
  height-above-floor verification, height smoothing, and CROSSING by image y.

There is deliberately no ROS dependency in this file.  The ROS2 wrapper is
hurdle_vision_fusion.py.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class HurdleVisionState:
    CALIBRATING = "CALIBRATING"
    WALKING = "WALKING"
    DETECTING = "DETECTING"
    CROSSING = "CROSSING"


@dataclass
class HurdleDetectorConfig:
    # ROI ratios in the full RealSense color/depth image.
    roi_left_ratio: float = 0.0
    roi_right_ratio: float = 1.0
    roi_top_ratio: float = 0.0
    roi_bottom_ratio: float = 1.0

    # Color mask selection: hsv, lab, and, or.
    # "or" gives good recall; depth and shape filters reject most false hits.
    color_combine_mode: str = "or"
    use_hsv_mask: bool = True
    use_lab_mask: bool = True

    # expo.py yellow HSV initial values.
    h_low: int = 20
    h_high: int = 38
    s_low: int = 120
    s_high: int = 255
    v_low: int = 80
    v_high: int = 255

    # expo.py BGR guard against brown floor and white robot parts.
    use_bgr_guard: bool = True
    bgr_min_r: int = 120
    bgr_min_g: int = 120
    bgr_min_r_minus_b: int = 80
    bgr_min_g_minus_b: int = 70
    bgr_max_abs_r_minus_g: int = 60

    # Senior code Lab initial values.
    l_low: int = 40
    l_high: int = 255
    a_low: int = 90
    a_high: int = 160
    b_low: int = 140
    b_high: int = 255

    # Depth and empty-floor calibration.
    depth_min_m: float = 0.05
    depth_max_m: float = 1.50
    floor_calibration_frames: int = 45
    calibration_max_strict_yellow_pixels: int = 300
    calibration_min_valid_ratio: float = 0.55
    compensate_floor_offset: bool = True
    max_floor_offset_m: float = 0.08

    # Height estimate: floor_delta, absolute, hybrid.
    # floor_delta is recommended when the robot body can pitch slightly.
    height_mode: str = "floor_delta"
    camera_height_m: float = 0.430
    hybrid_floor_weight: float = 0.75
    height_min_m: float = 0.010
    height_max_m: float = 0.150
    # expo.py used +1.8 cm.  Measure the real error and retune this value.
    height_offset_m: float = 0.018
    height_top_fraction: float = 0.20

    # Morphology and horizontal-bar shape filtering.
    morph_kernel_size: int = 5
    horizontal_close_width: int = 21
    horizontal_close_height: int = 3
    min_contour_area: float = 450.0
    min_span_ratio: float = 0.18
    min_aspect_ratio: float = 3.0
    min_fill_ratio: float = 0.10
    min_depth_support_ratio: float = 0.15
    max_horizontal_angle_deg: float = 40.0

    # Temporal smoothing.  Senior code held the previous result for 3 frames;
    # expo.py used a 12-frame height history.
    detection_confirm_frames: int = 3
    lost_hold_frames: int = 3
    history_size: int = 12

    # CROSSING threshold inside the ROI.
    cross_y_ratio: float = 0.55
    cross_hysteresis: float = 0.03

    # Disabled for competition use.  When true, a valid color/shape contour can
    # be accepted even when the depth verification has holes.
    allow_color_only_fallback: bool = False

    def validated(self) -> "HurdleDetectorConfig":
        cfg = replace(self)

        if not (0.0 <= cfg.roi_left_ratio < cfg.roi_right_ratio <= 1.0):
            raise ValueError("ROI must satisfy 0 <= left < right <= 1")
        if not (0.0 <= cfg.roi_top_ratio < cfg.roi_bottom_ratio <= 1.0):
            raise ValueError("ROI must satisfy 0 <= top < bottom <= 1")

        cfg.color_combine_mode = str(cfg.color_combine_mode).strip().lower()
        if cfg.color_combine_mode not in {"hsv", "lab", "and", "or"}:
            raise ValueError("color_combine_mode must be hsv, lab, and, or")
        if not cfg.use_hsv_mask and not cfg.use_lab_mask:
            raise ValueError("At least one of use_hsv_mask/use_lab_mask must be true")

        if not (0 <= cfg.h_low <= cfg.h_high <= 179):
            raise ValueError("Invalid HSV H range")
        if not (0 <= cfg.s_low <= cfg.s_high <= 255):
            raise ValueError("Invalid HSV S range")
        if not (0 <= cfg.v_low <= cfg.v_high <= 255):
            raise ValueError("Invalid HSV V range")
        if not (0 <= cfg.l_low <= cfg.l_high <= 255):
            raise ValueError("Invalid Lab L range")
        if not (0 <= cfg.a_low <= cfg.a_high <= 255):
            raise ValueError("Invalid Lab a range")
        if not (0 <= cfg.b_low <= cfg.b_high <= 255):
            raise ValueError("Invalid Lab b range")

        if not (0.0 < cfg.depth_min_m < cfg.depth_max_m):
            raise ValueError("Invalid depth range")
        if not (0.0 <= cfg.height_min_m < cfg.height_max_m):
            raise ValueError("Invalid height range")
        if cfg.floor_calibration_frames < 1:
            raise ValueError("floor_calibration_frames must be >= 1")
        if not (0.0 < cfg.calibration_min_valid_ratio <= 1.0):
            raise ValueError("calibration_min_valid_ratio must be in (0, 1]")
        if cfg.height_mode not in {"floor_delta", "absolute", "hybrid"}:
            raise ValueError("height_mode must be floor_delta, absolute, or hybrid")
        if not (0.0 <= cfg.hybrid_floor_weight <= 1.0):
            raise ValueError("hybrid_floor_weight must be in [0, 1]")
        if not (0.0 < cfg.height_top_fraction <= 1.0):
            raise ValueError("height_top_fraction must be in (0, 1]")

        if cfg.min_contour_area < 0.0:
            raise ValueError("min_contour_area must be >= 0")
        if not (0.0 <= cfg.min_span_ratio <= 1.0):
            raise ValueError("min_span_ratio must be in [0, 1]")
        if cfg.min_aspect_ratio < 1.0:
            raise ValueError("min_aspect_ratio must be >= 1")
        if not (0.0 <= cfg.min_fill_ratio <= 1.0):
            raise ValueError("min_fill_ratio must be in [0, 1]")
        if not (0.0 <= cfg.min_depth_support_ratio <= 1.0):
            raise ValueError("min_depth_support_ratio must be in [0, 1]")
        if not (0.0 <= cfg.max_horizontal_angle_deg <= 90.0):
            raise ValueError("max_horizontal_angle_deg must be in [0, 90]")

        if cfg.detection_confirm_frames < 1:
            raise ValueError("detection_confirm_frames must be >= 1")
        if cfg.lost_hold_frames < 0:
            raise ValueError("lost_hold_frames must be >= 0")
        if cfg.history_size < 1:
            raise ValueError("history_size must be >= 1")
        if not (0.0 <= cfg.cross_y_ratio <= 1.0):
            raise ValueError("cross_y_ratio must be in [0, 1]")
        if not (0.0 <= cfg.cross_hysteresis < 0.5):
            raise ValueError("cross_hysteresis must be in [0, 0.5)")

        cfg.morph_kernel_size = max(1, int(cfg.morph_kernel_size))
        if cfg.morph_kernel_size % 2 == 0:
            cfg.morph_kernel_size += 1
        cfg.horizontal_close_width = max(1, int(cfg.horizontal_close_width))
        if cfg.horizontal_close_width % 2 == 0:
            cfg.horizontal_close_width += 1
        cfg.horizontal_close_height = max(1, int(cfg.horizontal_close_height))
        if cfg.horizontal_close_height % 2 == 0:
            cfg.horizontal_close_height += 1

        return cfg


class HurdleOpenCVDetector:
    def __init__(self, config: Optional[HurdleDetectorConfig] = None) -> None:
        self.cfg = (config or HurdleDetectorConfig()).validated()
        self._rebuild_kernels()

        self.floor_depth_map: Optional[np.ndarray] = None
        self.calibration_buffer: list[np.ndarray] = []
        self.calibration_blocked_reason: Optional[str] = None
        self.roi_shape: Optional[Tuple[int, int]] = None

        self.hit_count = 0
        self.lost_count = 0
        self.confirmed = False
        self.state = HurdleVisionState.CALIBRATING
        self.last_stable_detection: Optional[Dict[str, Any]] = None
        self.detection_history: deque[Dict[str, Any]] = deque(
            maxlen=self.cfg.history_size
        )

    def update_config(self, config: HurdleDetectorConfig, reset_floor: bool = False) -> None:
        old_history = list(self.detection_history)
        self.cfg = config.validated()
        self._rebuild_kernels()
        self.detection_history = deque(
            old_history[-self.cfg.history_size :],
            maxlen=self.cfg.history_size,
        )
        if reset_floor:
            self.reset_calibration()

    def reset_calibration(self) -> None:
        self.floor_depth_map = None
        self.calibration_buffer.clear()
        self.calibration_blocked_reason = None
        self.roi_shape = None
        self.hit_count = 0
        self.lost_count = 0
        self.confirmed = False
        self.state = HurdleVisionState.CALIBRATING
        self.last_stable_detection = None
        self.detection_history.clear()

    @property
    def calibration_done(self) -> bool:
        return self.floor_depth_map is not None

    @property
    def calibration_progress(self) -> float:
        if self.calibration_done:
            return 1.0
        return min(
            1.0,
            len(self.calibration_buffer) / float(max(1, self.cfg.floor_calibration_frames)),
        )

    def _rebuild_kernels(self) -> None:
        k = self.cfg.morph_kernel_size
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.horizontal_close_width, self.cfg.horizontal_close_height),
        )

    def _roi_bounds(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        x1 = int(round(frame_w * self.cfg.roi_left_ratio))
        x2 = int(round(frame_w * self.cfg.roi_right_ratio))
        y1 = int(round(frame_h * self.cfg.roi_top_ratio))
        y2 = int(round(frame_h * self.cfg.roi_bottom_ratio))
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(y1 + 1, min(y2, frame_h))
        return x1, y1, x2, y2

    def _color_masks(self, roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        empty = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)

        hsv_mask = empty.copy()
        if self.cfg.use_hsv_mask:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(
                hsv,
                np.array([self.cfg.h_low, self.cfg.s_low, self.cfg.v_low], dtype=np.uint8),
                np.array([self.cfg.h_high, self.cfg.s_high, self.cfg.v_high], dtype=np.uint8),
            )
            if self.cfg.use_bgr_guard:
                bgr16 = roi_bgr.astype(np.int16)
                b, g, r = cv2.split(bgr16)
                guard = (
                    (r >= self.cfg.bgr_min_r)
                    & (g >= self.cfg.bgr_min_g)
                    & ((r - b) >= self.cfg.bgr_min_r_minus_b)
                    & ((g - b) >= self.cfg.bgr_min_g_minus_b)
                    & (np.abs(r - g) <= self.cfg.bgr_max_abs_r_minus_g)
                )
                hsv_mask = cv2.bitwise_and(hsv_mask, guard.astype(np.uint8) * 255)

        lab_mask = empty.copy()
        if self.cfg.use_lab_mask:
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            lab_mask = cv2.inRange(
                lab,
                np.array([self.cfg.l_low, self.cfg.a_low, self.cfg.b_low], dtype=np.uint8),
                np.array([self.cfg.l_high, self.cfg.a_high, self.cfg.b_high], dtype=np.uint8),
            )

        mode = self.cfg.color_combine_mode
        if mode == "hsv" or (self.cfg.use_hsv_mask and not self.cfg.use_lab_mask):
            combined = hsv_mask
        elif mode == "lab" or (self.cfg.use_lab_mask and not self.cfg.use_hsv_mask):
            combined = lab_mask
        elif mode == "and":
            combined = cv2.bitwise_and(hsv_mask, lab_mask)
        else:
            combined = cv2.bitwise_or(hsv_mask, lab_mask)

        # Strict mask is used only to prevent calibrating while a clear yellow
        # hurdle occupies the ROI.  It is intentionally stricter than OR mode.
        if self.cfg.use_hsv_mask and self.cfg.use_lab_mask:
            strict = cv2.bitwise_and(hsv_mask, lab_mask)
        elif self.cfg.use_hsv_mask:
            strict = hsv_mask.copy()
        else:
            strict = lab_mask.copy()

        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.small_kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.horizontal_kernel)
        strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, self.small_kernel)
        return combined, hsv_mask, lab_mask, strict

    def _valid_depth_mask(self, depth_m: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(depth_m)
            & (depth_m >= self.cfg.depth_min_m)
            & (depth_m <= self.cfg.depth_max_m)
        )

    def _calibrate(self, roi_depth_m: np.ndarray, strict_color_mask: np.ndarray) -> Dict[str, Any]:
        valid = self._valid_depth_mask(roi_depth_m)
        valid_ratio = float(np.count_nonzero(valid)) / float(max(1, valid.size))
        strict_pixels = int(np.count_nonzero(strict_color_mask))

        self.calibration_blocked_reason = None
        if strict_pixels > self.cfg.calibration_max_strict_yellow_pixels:
            self.calibration_blocked_reason = "yellow_hurdle_in_roi"
        elif valid_ratio < self.cfg.calibration_min_valid_ratio:
            self.calibration_blocked_reason = "insufficient_valid_depth"

        if self.calibration_blocked_reason is None:
            self.calibration_buffer.append(
                np.where(valid, roi_depth_m, np.nan).astype(np.float32)
            )

        if len(self.calibration_buffer) >= self.cfg.floor_calibration_frames:
            stack = np.stack(self.calibration_buffer, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                floor = np.nanmedian(stack, axis=0).astype(np.float32)

            floor_valid_ratio = float(np.count_nonzero(np.isfinite(floor))) / float(max(1, floor.size))
            if floor_valid_ratio >= self.cfg.calibration_min_valid_ratio:
                self.floor_depth_map = floor
                self.calibration_buffer.clear()
                self.state = HurdleVisionState.WALKING
            else:
                self.calibration_buffer.clear()
                self.calibration_blocked_reason = "invalid_floor_map"

        return {
            "calibration_progress": self.calibration_progress,
            "calibration_valid_depth_ratio": valid_ratio,
            "calibration_blocked_reason": self.calibration_blocked_reason,
            "strict_yellow_pixels": strict_pixels,
        }

    def _height_map(
        self,
        roi_depth_m: np.ndarray,
        color_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, int]:
        if self.floor_depth_map is None:
            raise RuntimeError("Floor calibration is not complete")

        valid = self._valid_depth_mask(roi_depth_m)
        floor_valid = np.isfinite(self.floor_depth_map)
        color = color_mask > 0

        floor_offset_m = 0.0
        sample_count = 0
        if self.cfg.compensate_floor_offset:
            background = valid & floor_valid & (~color)
            offsets = roi_depth_m[background] - self.floor_depth_map[background]
            offsets = offsets[np.isfinite(offsets)]
            sample_count = int(offsets.size)
            if offsets.size:
                floor_offset_m = float(np.median(offsets))
                floor_offset_m = float(
                    np.clip(
                        floor_offset_m,
                        -self.cfg.max_floor_offset_m,
                        self.cfg.max_floor_offset_m,
                    )
                )

        adjusted_floor = self.floor_depth_map + floor_offset_m
        floor_delta_h = adjusted_floor - roi_depth_m
        absolute_h = self.cfg.camera_height_m - roi_depth_m

        if self.cfg.height_mode == "absolute":
            height = absolute_h
        elif self.cfg.height_mode == "hybrid":
            w = self.cfg.hybrid_floor_weight
            height = w * floor_delta_h + (1.0 - w) * absolute_h
        else:
            height = floor_delta_h

        height = height.astype(np.float32)
        height[~(valid & floor_valid)] = np.nan
        return height, floor_offset_m, sample_count

    @staticmethod
    def _fit_line(contour: np.ndarray) -> Tuple[float, float, float, float, float]:
        vx_a, vy_a, x0_a, y0_a = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(np.asarray(vx_a).reshape(-1)[0])
        vy = float(np.asarray(vy_a).reshape(-1)[0])
        x0 = float(np.asarray(x0_a).reshape(-1)[0])
        y0 = float(np.asarray(y0_a).reshape(-1)[0])
        if vx < 0.0 or (abs(vx) < 1e-9 and vy < 0.0):
            vx, vy = -vx, -vy
        angle_deg = math.degrees(math.atan2(vy, vx))
        while angle_deg >= 90.0:
            angle_deg -= 180.0
        while angle_deg < -90.0:
            angle_deg += 180.0
        return vx, vy, x0, y0, float(angle_deg)

    @staticmethod
    def _top_fraction_mean(values: np.ndarray, fraction: float) -> Optional[float]:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        values = np.sort(values)
        start = int(math.floor(values.size * (1.0 - fraction)))
        start = max(0, min(start, values.size - 1))
        return float(np.mean(values[start:]))

    def _best_candidate(
        self,
        candidate_mask: np.ndarray,
        color_mask: np.ndarray,
        height_map: np.ndarray,
        roi_depth_m: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
        depth_verified: bool,
    ) -> Optional[Dict[str, Any]]:
        contours, _ = cv2.findContours(
            candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        roi_x, roi_y, roi_x2, roi_y2 = roi_rect
        roi_w = roi_x2 - roi_x
        roi_h = roi_y2 - roi_y
        best: Optional[Dict[str, Any]] = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.cfg.min_contour_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue

            (_rcx, _rcy), (rw, rh), _rang = cv2.minAreaRect(contour)
            long_side = max(float(rw), float(rh), float(bw))
            short_side = max(1.0, min(float(rw), float(rh), float(bh)))
            aspect_ratio = long_side / short_side
            span_ratio = long_side / float(max(1, roi_w))
            fill_ratio = area / float(max(1, bw * bh))

            vx, vy, line_x, line_y, angle_deg = self._fit_line(contour)
            if span_ratio < self.cfg.min_span_ratio:
                continue
            if aspect_ratio < self.cfg.min_aspect_ratio:
                continue
            if fill_ratio < self.cfg.min_fill_ratio:
                continue
            if abs(angle_deg) > self.cfg.max_horizontal_angle_deg:
                continue

            contour_mask = np.zeros_like(candidate_mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            contour_pixels = contour_mask > 0
            color_pixels = contour_pixels & (color_mask > 0)
            verified_pixels = contour_pixels & np.isfinite(height_map)
            verified_pixels &= (
                (height_map >= self.cfg.height_min_m)
                & (height_map <= self.cfg.height_max_m)
            )
            depth_support_ratio = float(np.count_nonzero(verified_pixels)) / float(
                max(1, np.count_nonzero(color_pixels))
            )

            if depth_verified and depth_support_ratio < self.cfg.min_depth_support_ratio:
                continue

            height_value = self._top_fraction_mean(
                height_map[verified_pixels],
                self.cfg.height_top_fraction,
            )
            if height_value is not None:
                height_value += self.cfg.height_offset_m

            depth_values = roi_depth_m[contour_pixels]
            depth_values = depth_values[
                np.isfinite(depth_values)
                & (depth_values >= self.cfg.depth_min_m)
                & (depth_values <= self.cfg.depth_max_m)
            ]
            distance_m = float(np.median(depth_values)) if depth_values.size else None

            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-6:
                cx_roi = float(moments["m10"] / moments["m00"])
                cy_roi = float(moments["m01"] / moments["m00"])
            else:
                cx_roi = float(bx + bw / 2.0)
                cy_roi = float(by + bh / 2.0)

            # Line endpoints across the ROI for debug visualization.
            eps = 1e-9
            left_y = line_y + (0.0 - line_x) * vy / (vx + eps)
            right_y = line_y + ((roi_w - 1.0) - line_x) * vy / (vx + eps)

            score = (
                2.5 * span_ratio
                + 1.0 * fill_ratio
                + 0.5 * min(1.0, area / float(max(1, roi_w * roi_h)))
                + 0.5 * depth_support_ratio
                - 0.01 * abs(angle_deg)
            )

            candidate: Dict[str, Any] = {
                "hurdle_x": float(cx_roi + roi_x),
                "hurdle_y": float(cy_roi + roi_y),
                "hurdle_x_offset_px": float(cx_roi + roi_x - (roi_x + roi_w / 2.0)),
                "hurdle_y_ratio": float(cy_roi / float(max(1, roi_h))),
                "hurdle_bbox": [
                    float(bx + roi_x),
                    float(by + roi_y),
                    float(bx + bw + roi_x),
                    float(by + bh + roi_y),
                ],
                "hurdle_height_m": height_value,
                "hurdle_distance_m": distance_m,
                "horizontal_angle_deg": float(angle_deg),
                "contour_area": area,
                "span_ratio": float(span_ratio),
                "aspect_ratio": float(aspect_ratio),
                "fill_ratio": float(fill_ratio),
                "depth_support_ratio": float(depth_support_ratio),
                "hurdle_pixels": int(np.count_nonzero(verified_pixels)),
                "depth_verified": bool(depth_verified),
                "line_left_y": float(left_y + roi_y),
                "line_right_y": float(right_y + roi_y),
                "score": float(score),
                "contour": contour,
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

        return best

    def _detect(
        self,
        roi_depth_m: np.ndarray,
        color_mask: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
    ) -> Tuple[Optional[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        height_map, floor_offset_m, floor_sample_count = self._height_map(
            roi_depth_m,
            color_mask,
        )
        valid_height = (
            np.isfinite(height_map)
            & (height_map >= self.cfg.height_min_m)
            & (height_map <= self.cfg.height_max_m)
        )
        depth_mask = ((color_mask > 0) & valid_height).astype(np.uint8) * 255
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, self.horizontal_kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, self.small_kernel)

        candidate = self._best_candidate(
            depth_mask,
            color_mask,
            height_map,
            roi_depth_m,
            roi_rect,
            depth_verified=True,
        )

        used_color_fallback = False
        if candidate is None and self.cfg.allow_color_only_fallback:
            candidate = self._best_candidate(
                color_mask,
                color_mask,
                height_map,
                roi_depth_m,
                roi_rect,
                depth_verified=False,
            )
            used_color_fallback = candidate is not None

        metrics = {
            "floor_offset_m": float(floor_offset_m),
            "floor_offset_sample_count": int(floor_sample_count),
            "used_color_only_fallback": bool(used_color_fallback),
            "depth_candidate_pixels": int(np.count_nonzero(depth_mask)),
        }
        return candidate, depth_mask, metrics

    @staticmethod
    def _median(history: list[Dict[str, Any]], key: str) -> Optional[float]:
        values = []
        for item in history:
            value = item.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append(number)
        return float(np.median(values)) if values else None

    def _filtered_detection(self, latest: Dict[str, Any]) -> Dict[str, Any]:
        history = list(self.detection_history)
        result = dict(latest)
        for key in [
            "hurdle_x",
            "hurdle_y",
            "hurdle_x_offset_px",
            "hurdle_y_ratio",
            "hurdle_height_m",
            "hurdle_distance_m",
            "horizontal_angle_deg",
            "line_left_y",
            "line_right_y",
        ]:
            value = self._median(history, key)
            if value is not None:
                result[key] = value
        return result

    def _stabilize(
        self,
        raw_detection: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[Dict[str, Any]], bool]:
        held_previous = False

        if raw_detection is not None:
            self.hit_count += 1
            self.lost_count = 0
            serializable = {k: v for k, v in raw_detection.items() if k != "contour"}
            self.detection_history.append(serializable)
            filtered = self._filtered_detection(serializable)

            if self.hit_count >= self.cfg.detection_confirm_frames:
                self.confirmed = True
                self.last_stable_detection = filtered
            elif self.confirmed:
                self.last_stable_detection = filtered
        else:
            self.hit_count = 0
            if self.confirmed:
                self.lost_count += 1
                if self.lost_count <= self.cfg.lost_hold_frames:
                    held_previous = self.last_stable_detection is not None
                else:
                    self.confirmed = False
                    self.last_stable_detection = None
                    self.detection_history.clear()
            else:
                self.lost_count = min(self.cfg.lost_hold_frames + 1, self.lost_count + 1)
                self.detection_history.clear()

        stable = bool(self.confirmed and self.last_stable_detection is not None)
        stable_detection = dict(self.last_stable_detection) if stable else None
        return stable, stable_detection, held_previous

    def _update_state(self, detected: bool, detection: Optional[Dict[str, Any]]) -> None:
        if not self.calibration_done:
            self.state = HurdleVisionState.CALIBRATING
            return
        if not detected or detection is None:
            self.state = HurdleVisionState.WALKING
            return

        y_ratio = float(detection.get("hurdle_y_ratio", 0.0))
        if self.state == HurdleVisionState.CROSSING:
            if y_ratio < self.cfg.cross_y_ratio - self.cfg.cross_hysteresis:
                self.state = HurdleVisionState.DETECTING
        elif y_ratio >= self.cfg.cross_y_ratio:
            self.state = HurdleVisionState.CROSSING
        else:
            self.state = HurdleVisionState.DETECTING

    def process(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        if color_bgr is None or depth_m is None:
            raise ValueError("color_bgr and depth_m are required")
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError("color_bgr must be HxWx3")
        if depth_m.ndim != 2:
            raise ValueError("depth_m must be HxW")
        if color_bgr.shape[:2] != depth_m.shape[:2]:
            raise ValueError("Color and aligned depth dimensions must match")

        frame_h, frame_w = color_bgr.shape[:2]
        roi_rect = self._roi_bounds(frame_w, frame_h)
        x1, y1, x2, y2 = roi_rect
        roi_bgr = color_bgr[y1:y2, x1:x2]
        roi_depth_m = depth_m[y1:y2, x1:x2].astype(np.float32, copy=False)

        color_mask, hsv_mask, lab_mask, strict_mask = self._color_masks(roi_bgr)

        if self.roi_shape is None:
            self.roi_shape = roi_depth_m.shape
        elif self.roi_shape != roi_depth_m.shape:
            self.reset_calibration()
            self.roi_shape = roi_depth_m.shape

        payload: Dict[str, Any] = {
            "hurdle_detected": False,
            "raw_detected": False,
            "depth_verified": False,
            "held_previous_detection": False,
            "vision_state": self.state,
            "crossing_ready": False,
            "calibration_done": self.calibration_done,
            "calibration_progress": self.calibration_progress,
            "calibration_blocked_reason": self.calibration_blocked_reason,
            "hurdle_x": None,
            "hurdle_y": None,
            "hurdle_x_offset_px": None,
            "hurdle_y_ratio": None,
            "hurdle_bbox": [],
            "hurdle_height_m": None,
            "hurdle_distance_m": None,
            "horizontal_angle_deg": None,
            "line_left_y": None,
            "line_right_y": None,
            "contour_area": 0.0,
            "span_ratio": 0.0,
            "aspect_ratio": 0.0,
            "fill_ratio": 0.0,
            "depth_support_ratio": 0.0,
            "hurdle_pixels": 0,
            "hsv_pixels": int(np.count_nonzero(hsv_mask)),
            "lab_pixels": int(np.count_nonzero(lab_mask)),
            "color_pixels": int(np.count_nonzero(color_mask)),
            "strict_yellow_pixels": int(np.count_nonzero(strict_mask)),
            "depth_candidate_pixels": 0,
            "floor_offset_m": None,
            "floor_offset_sample_count": 0,
            "used_color_only_fallback": False,
            "hit_count": int(self.hit_count),
            "lost_count": int(self.lost_count),
            "roi": [int(x1), int(y1), int(x2), int(y2)],
        }

        raw_detection: Optional[Dict[str, Any]] = None
        depth_mask = np.zeros_like(color_mask)

        if not self.calibration_done:
            payload.update(self._calibrate(roi_depth_m, strict_mask))
            payload["calibration_done"] = self.calibration_done
            payload["calibration_progress"] = self.calibration_progress
            self._update_state(False, None)
            payload["vision_state"] = self.state
            debug = self.draw_debug(
                color_bgr,
                color_mask,
                hsv_mask,
                lab_mask,
                depth_mask,
                payload,
                roi_rect,
                None,
            )
            return payload, debug

        raw_detection, depth_mask, metrics = self._detect(
            roi_depth_m,
            color_mask,
            roi_rect,
        )
        stable_detected, stable_detection, held_previous = self._stabilize(raw_detection)
        self._update_state(stable_detected, stable_detection)

        payload.update(metrics)
        payload["calibration_done"] = True
        payload["calibration_progress"] = 1.0
        payload["raw_detected"] = raw_detection is not None
        payload["hurdle_detected"] = stable_detected
        payload["held_previous_detection"] = held_previous
        payload["vision_state"] = self.state
        payload["crossing_ready"] = self.state == HurdleVisionState.CROSSING
        payload["hit_count"] = int(self.hit_count)
        payload["lost_count"] = int(self.lost_count)

        display_detection = stable_detection if stable_detection is not None else raw_detection
        if display_detection is not None:
            for key, value in display_detection.items():
                if key != "contour":
                    payload[key] = value

        debug = self.draw_debug(
            color_bgr,
            color_mask,
            hsv_mask,
            lab_mask,
            depth_mask,
            payload,
            roi_rect,
            raw_detection,
        )
        return payload, debug

    def draw_debug(
        self,
        frame: np.ndarray,
        color_mask: np.ndarray,
        hsv_mask: np.ndarray,
        lab_mask: np.ndarray,
        depth_mask: np.ndarray,
        payload: Dict[str, Any],
        roi_rect: Tuple[int, int, int, int],
        raw_detection: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        debug = frame.copy()
        frame_h, frame_w = debug.shape[:2]
        x1, y1, x2, y2 = roi_rect
        cross_y = int(round(y1 + (y2 - y1) * self.cfg.cross_y_ratio))

        state = str(payload.get("vision_state", HurdleVisionState.WALKING))
        state_color = {
            HurdleVisionState.CALIBRATING: (0, 220, 220),
            HurdleVisionState.WALKING: (180, 180, 0),
            HurdleVisionState.DETECTING: (0, 140, 255),
            HurdleVisionState.CROSSING: (200, 0, 200),
        }.get(state, (220, 220, 220))

        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (200, 50, 0), 2)
        cv2.line(debug, (x1, cross_y), (x2 - 1, cross_y), (110, 110, 110), 2)
        cv2.putText(
            debug,
            "CROSSING LINE",
            (x1 + 4, max(14, cross_y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

        bbox = payload.get("hurdle_bbox") or []
        if len(bbox) == 4:
            bx1, by1, bx2, by2 = [int(round(float(v))) for v in bbox]
            box_color = (0, 255, 255) if payload.get("held_previous_detection") else state_color
            cv2.rectangle(debug, (bx1, by1), (bx2, by2), box_color, 3)
            hx = payload.get("hurdle_x")
            hy = payload.get("hurdle_y")
            if hx is not None and hy is not None:
                cv2.circle(debug, (int(round(float(hx))), int(round(float(hy)))), 5, (0, 0, 255), -1)

        left_y = payload.get("line_left_y")
        right_y = payload.get("line_right_y")
        if left_y is not None and right_y is not None:
            cv2.line(
                debug,
                (x1, int(round(float(left_y)))),
                (x2 - 1, int(round(float(right_y)))),
                state_color,
                2,
                cv2.LINE_AA,
            )

        if raw_detection is not None and raw_detection.get("contour") is not None:
            contour = raw_detection["contour"].copy()
            contour[:, 0, 0] += x1
            contour[:, 0, 1] += y1
            cv2.drawContours(debug, [contour], -1, (255, 255, 0), 2)

        if state == HurdleVisionState.CROSSING:
            cv2.rectangle(debug, (4, 4), (frame_w - 5, frame_h - 5), state_color, 4)

        def fmt(value: Any, scale: float = 1.0, suffix: str = "", digits: int = 1) -> str:
            if value is None:
                return "--"
            try:
                number = float(value) * scale
            except (TypeError, ValueError):
                return "--"
            if not math.isfinite(number):
                return "--"
            return f"{number:.{digits}f}{suffix}"

        text_lines = [
            f"state:{state} stable:{int(bool(payload.get('hurdle_detected')))} raw:{int(bool(payload.get('raw_detected')))}",
            f"dist:{fmt(payload.get('hurdle_distance_m'), 100.0, 'cm')} height:{fmt(payload.get('hurdle_height_m'), 100.0, 'cm')}",
            f"angle:{fmt(payload.get('horizontal_angle_deg'), 1.0, 'deg')} y:{fmt(payload.get('hurdle_y_ratio'), 100.0, '%')}",
            f"depth_hit:{payload.get('depth_candidate_pixels', 0)} hit:{payload.get('hit_count', 0)}/{self.cfg.detection_confirm_frames} lost:{payload.get('lost_count', 0)}/{self.cfg.lost_hold_frames}",
        ]
        if not payload.get("calibration_done", False):
            reason = payload.get("calibration_blocked_reason") or "collecting_floor"
            text_lines.append(
                f"calib:{float(payload.get('calibration_progress', 0.0))*100.0:.0f}% {reason}"
            )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.40
        font_thickness = 1
        padding_x = 6
        padding_y = 5
        line_gap = 4
        text_sizes = [
            cv2.getTextSize(
                text,
                font,
                font_scale,
                font_thickness,
            )[0]
            for text in text_lines
        ]
        text_height = max(size[1] for size in text_sizes)
        line_height = text_height + line_gap
        panel_w = min(
            frame_w - 10,
            max(size[0] for size in text_sizes) + 2 * padding_x,
        )
        panel_h = min(
            frame_h - 10,
            2 * padding_y
            + len(text_lines) * text_height
            + (len(text_lines) - 1) * line_gap,
        )
        cv2.rectangle(
            debug,
            (5, 5),
            (5 + panel_w, 5 + panel_h),
            (20, 20, 20),
            -1,
        )
        first_text_y = 5 + padding_y + text_height
        for index, text in enumerate(text_lines):
            cv2.putText(
                debug,
                text,
                (5 + padding_x, first_text_y + index * line_height),
                font,
                font_scale,
                (235, 235, 235),
                font_thickness,
                cv2.LINE_AA,
            )

        # Preview legend: green=HSV, blue=Lab, cyan=combined, magenta=depth verified.
        preview = np.zeros((*color_mask.shape, 3), dtype=np.uint8)
        preview[hsv_mask > 0] = (0, 160, 0)
        preview[lab_mask > 0] = (160, 0, 0)
        preview[color_mask > 0] = (180, 180, 0)
        preview[depth_mask > 0] = (200, 0, 200)
        preview_w = min(140, max(1, frame_w // 5))
        if preview.shape[1] > 0:
            scale = preview_w / float(preview.shape[1])
            preview_h = max(1, int(round(preview.shape[0] * scale)))
            resized = cv2.resize(preview, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
            px = max(0, frame_w - preview_w - 5)
            py = 5
            ph = min(preview_h, frame_h - py)
            pw = min(preview_w, frame_w - px)
            if ph > 0 and pw > 0:
                debug[py : py + ph, px : px + pw] = resized[:ph, :pw]

        return debug
