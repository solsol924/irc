#!/usr/bin/env python3
"""
ROS 2 + RealSense HSV/Depth calibration tool.

Purpose
-------
* Tune HSV and depth ranges at whatever venue/lighting is present now.
* Keep one current profile for ball and one current profile for hoop.
* Sample a dragged ROI over multiple frames and suggest robust HSV bounds.
* Preview raw HSV mask, cleaned mask, depth mask, combined mask, and contours.
* Save the current profile, screenshots, depth arrays, and CSV measurements.
* Export the latest ball HSV as a ROS parameter file used by robot_bringup.

Mouse
-----
* Left-drag on the "Calibration" window: select a tight object ROI.
* Right-click: clear ROI.

Keys
----
* b / k / f / g: select ball / black support / red floor / hoop
* SPACE       : add current ROI samples to the selected target's sample bank
* a           : auto-fit HSV bounds from sample bank (or current ROI)
* d           : toggle detection preview using the fitted/current values
* r           : restore the values that were active before the first auto-fit
* n           : start a new tuning session for the current target
                (clears ROI and accumulated samples; keeps slider values)
* x           : clear only the selected target's sample bank
* s           : save the current ball/hoop profiles to YAML
* l           : reload profiles from YAML
* i           : save image/depth snapshot and append a CSV row
* q / ESC     : quit

Notes
-----
* There are no fixed bright/normal/dark modes. Run this tool at the current
  venue, collect ROI samples, press A to fit, inspect the masks, then press S.
* HSV H is OpenCV's 0..179 range. If H low > H high, the range wraps through 0
  (useful for red hues near both 179 and 0).
* depth_scale_to_mm multiplies raw depth values before comparison. RealSense ROS
  commonly publishes 16UC1 depth in millimetres, so the default is 1.0. Verify
  your topic encoding and change this parameter when necessary.
* environment_label is optional and is used only in saved records. It does not
  select or alter HSV values. Example: -p environment_label:=gym_A

"""

from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from ball_detector_core import (
    BallSupportConfig,
    black_support_mask,
    evaluate_ball_support,
    hsv_range_mask,
)


# -----------------------------------------------------------------------------
# Easy-to-edit calibration heuristics
# -----------------------------------------------------------------------------
HUE_COVERAGE = 0.97       # shortest hue arc covering 97% of sampled hue pixels
HUE_MARGIN = 4            # expand the fitted hue arc by this many OpenCV H units
SV_LOW_PERCENTILE = 2.0   # ignore extreme dark/desaturated outlier pixels
SV_HIGH_PERCENTILE = 99.0
S_LOW_MARGIN = 15
S_HIGH_MARGIN = 5
V_LOW_MARGIN = 15
V_HIGH_MARGIN = 5
SUPPORT_V_PERCENTILE = 95.0
SUPPORT_V_MARGIN = 10
MAX_BANK_PIXELS = 200_000
MAX_ROI_PIXELS_PER_ADD = 20_000
UNDEREXPOSED_V = 25
OVEREXPOSED_V = 245

# Keep these ball-preview checks aligned with irc/src/vision/scripts/
# ball_vision_fusion.py.  They intentionally do not use the looser calibrator
# contour sliders, so D mode answers whether the fitted HSV is likely to pass
# the current competition ball detector rather than only its color mask.
BALL_PREVIEW_DEPTH_MAX_MM = 1_500.0
BALL_PREVIEW_MIN_AREA = 300.0
BALL_PREVIEW_MAX_CIRCLE_RATIO_ERROR = 0.45
BALL_PREVIEW_EDGE_MARGIN_PX = 3
BALL_PREVIEW_EDGE_MIN_AREA_RATIO = 0.65
BALL_PREVIEW_EDGE_MAX_CIRCLE_RATIO_ERROR = 0.65
BALL_PREVIEW_DIAMETER_MIN_M = 0.050
BALL_PREVIEW_DIAMETER_MAX_M = 0.070
BALL_PREVIEW_RADIUS_MIN_RATIO = 0.45
BALL_PREVIEW_RADIUS_MAX_RATIO = 1.70
BALL_PREVIEW_EDGE_RADIUS_MIN_RATIO = 0.60
BALL_PREVIEW_EDGE_RADIUS_MAX_RATIO = 1.50
BALL_PREVIEW_MIN_CIRCULARITY = 0.55
BALL_PREVIEW_EDGE_MIN_CIRCULARITY = 0.48
BALL_PREVIEW_MIN_ASPECT = 0.55
BALL_PREVIEW_MAX_ASPECT = 1.80
BALL_PREVIEW_EDGE_MIN_ASPECT = 0.45
BALL_PREVIEW_EDGE_MAX_ASPECT = 2.20
BALL_PREVIEW_MORPH_SIZE = 5
BALL_PREVIEW_HOLD_FRAMES = 3

TARGETS = ("ball", "support", "floor", "hoop")

MAIN_WINDOW = "Calibration"
MASK_WINDOW = "Masks"
CONTROL_WINDOW = "Controls"

TRACKBARS = {
    "H low": 179,
    "H high": 179,
    "S low": 255,
    "S high": 255,
    "V low": 255,
    "V high": 255,
    "Depth min mm": 10_000,
    "Depth max mm": 10_000,
    "Blur": 21,
    "Morph": 21,
    "Min area": 30_000,
    "Ball circ %": 100,
}


def noop(_: int) -> None:
    """Provide a trackbar callback placeholder."""


def clamp_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def odd_kernel(value: int, allow_zero: bool = False) -> int:
    """Return a valid odd OpenCV kernel size."""
    value = int(value)
    if allow_zero and value <= 0:
        return 0
    value = max(1, value)
    return value if value % 2 == 1 else value + 1


def shortest_hue_interval(
    hues: np.ndarray,
    coverage: float = HUE_COVERAGE,
    margin: int = HUE_MARGIN,
) -> Tuple[int, int]:
    """
    Find the shortest circular interval on OpenCV hue space [0, 179].

    A returned low > high means the interval wraps through hue 0.
    """
    values = np.asarray(hues, dtype=np.int16).reshape(-1)
    values = values[(values >= 0) & (values <= 179)]
    if values.size == 0:
        return 0, 179

    values.sort()
    n = values.size
    keep = max(1, min(n, int(math.ceil(n * coverage))))
    extended = np.concatenate([values, values + 180])

    starts = np.arange(n)
    ends = starts + keep - 1
    widths = extended[ends] - extended[starts]
    best = int(np.argmin(widths))

    start = int(extended[best]) - int(margin)
    end = int(extended[best + keep - 1]) + int(margin)
    if end - start >= 179:
        return 0, 179

    return start % 180, end % 180


def hue_mask(hsv: np.ndarray, profile: Dict[str, Any]) -> np.ndarray:
    """Create an HSV mask with support for hue wrap-around."""
    h_low = clamp_int(profile["h_low"], 0, 179)
    h_high = clamp_int(profile["h_high"], 0, 179)
    s_low = clamp_int(profile["s_low"], 0, 255)
    s_high = clamp_int(profile["s_high"], 0, 255)
    v_low = clamp_int(profile["v_low"], 0, 255)
    v_high = clamp_int(profile["v_high"], 0, 255)

    if s_low > s_high:
        s_low, s_high = s_high, s_low
    if v_low > v_high:
        v_low, v_high = v_high, v_low

    if h_low <= h_high:
        return cv2.inRange(
            hsv,
            np.array([h_low, s_low, v_low], dtype=np.uint8),
            np.array([h_high, s_high, v_high], dtype=np.uint8),
        )

    # Wrapped interval, e.g. H=170..10 for red.
    lower_part = cv2.inRange(
        hsv,
        np.array([0, s_low, v_low], dtype=np.uint8),
        np.array([h_high, s_high, v_high], dtype=np.uint8),
    )
    upper_part = cv2.inRange(
        hsv,
        np.array([h_low, s_low, v_low], dtype=np.uint8),
        np.array([179, s_high, v_high], dtype=np.uint8),
    )
    return cv2.bitwise_or(lower_part, upper_part)


def default_profile(target: str) -> Dict[str, Any]:
    """Return a placeholder profile to be replaced by calibration data."""
    common: Dict[str, Any] = {
        "h_low": 8,
        "h_high": 60,
        "s_low": 60,
        "s_high": 255,
        "v_low": 20,
        "v_high": 255,
        "depth_min_mm": 100,
        "depth_max_mm": 6_000,
        "blur_size": 3,
        "morph_size": 5,
        "min_area": 120,
        "ball_circularity_min": 0.45,
    }
    if target == "support":
        # Black hue is unstable; production uses the fitted V high only.
        common.update(
            {
                "h_low": 0,
                "h_high": 179,
                "s_low": 0,
                "s_high": 255,
                "v_low": 0,
                "v_high": 75,
            }
        )
    elif target == "floor":
        common.update(
            {
                "h_low": 0,
                "h_high": 12,
                "s_low": 50,
                "s_high": 255,
                "v_low": 25,
                "v_high": 255,
            }
        )
    elif target == "hoop":
        common["min_area"] = 250
        # The calibrator does not assume whether you detect rim or backboard.
        # Shape validation should be specialized in the production detector.
    return common


def default_detector_settings() -> Dict[str, Any]:
    """Support geometry and context thresholds shared with production."""
    return {
        "support_diameter_m": 0.150,
        "support_black_ratio_min": 0.30,
        "edge_support_black_ratio_min": 0.25,
        "support_ball_color_ratio_max": 0.15,
        "support_floor_ratio_max": 0.35,
        "support_sector_black_ratio_min": 0.18,
        "support_min_sectors": 3,
        "edge_support_min_sectors": 2,
        "support_min_visible_fraction": 0.45,
        "edge_support_min_visible_fraction": 0.12,
    }


def default_store() -> Dict[str, Any]:
    return {
        "version": 3,
        "profiles": {
            target: default_profile(target)
            for target in TARGETS
        },
        "detector": default_detector_settings(),
        "metadata": {
            "note": (
                "Separate ball, black-support, floor and hoop profiles. "
                "Tune at the venue that is present now."
            ),
        },
    }


class ProfileStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = default_store()
        self.load()

    def _ensure_structure(self) -> None:
        """Normalize current and legacy YAML structures to one profile per target."""
        profiles = self.data.setdefault("profiles", {})
        for target in TARGETS:
            loaded = profiles.get(target, {})

            # Migration from the previous bright/normal/dark structure.
            if isinstance(loaded, dict) and any(
                key in loaded for key in ("bright", "normal", "dark")
            ):
                legacy = loaded.get("normal")
                if not isinstance(legacy, dict):
                    legacy = next(
                        (
                            value
                            for key, value in loaded.items()
                            if key in ("bright", "normal", "dark")
                            and isinstance(value, dict)
                        ),
                        {},
                    )
                loaded = legacy

            base = default_profile(target)
            if isinstance(loaded, dict):
                base.update(loaded)
            profiles[target] = base

        detector = default_detector_settings()
        loaded_detector = self.data.get("detector", {})
        if isinstance(loaded_detector, dict):
            detector.update(loaded_detector)
        self.data["detector"] = detector
        self.data["version"] = 3
        self.data.setdefault("metadata", {})

    def load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as file:
                    loaded = yaml.safe_load(file)
                if isinstance(loaded, dict):
                    self.data = loaded
            except (OSError, yaml.YAMLError) as exc:
                print(f"[ProfileStore] Could not load {self.path}: {exc}")
        self._ensure_structure()

    def save(self) -> None:
        self._ensure_structure()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["metadata"]["last_saved_unix"] = time.time()
        with self.path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                self.data,
                file,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    def get(self, target: str) -> Dict[str, Any]:
        return dict(self.data["profiles"][target])

    def set(self, target: str, profile: Dict[str, Any]) -> None:
        self.data["profiles"][target] = dict(profile)

    def get_detector(self) -> Dict[str, Any]:
        return dict(self.data["detector"])


class HSVCalibratorNode(Node):
    def __init__(self) -> None:
        super().__init__("realsense_hsv_calibrator")

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/camera/color/camera_info",
        )
        self.declare_parameter(
            "profile_file",
            str(
                Path.home()
                / "irc"
                / "src"
                / "vision"
                / "config"
                / "hsv_profiles.yaml"
            ),
        )
        self.declare_parameter(
            "ball_params_file",
            str(
                Path.home()
                / "irc"
                / "src"
                / "vision"
                / "config"
                / "ball_hsv.yaml"
            ),
        )
        self.declare_parameter(
            "output_dir",
            str(Path.home() / ".ros" / "vision" / "calibration"),
        )
        self.declare_parameter("depth_scale_to_mm", 1.0)
        self.declare_parameter("sync_queue", 3)
        self.declare_parameter("sync_slop", 0.08)
        self.declare_parameter("target", "ball")
        self.declare_parameter("environment_label", "current")

        color_topic = str(self.get_parameter("color_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.profile_path = Path(
            str(self.get_parameter("profile_file").value)
        ).expanduser()
        self.ball_params_path = Path(
            str(self.get_parameter("ball_params_file").value)
        ).expanduser()
        self.output_dir = Path(
            str(self.get_parameter("output_dir").value)
        ).expanduser()
        self.depth_scale_to_mm = float(
            self.get_parameter("depth_scale_to_mm").value
        )

        requested_target = str(self.get_parameter("target").value).lower()
        self.target = requested_target if requested_target in TARGETS else "ball"
        self.environment_label = str(
            self.get_parameter("environment_label").value
        ).strip() or "current"

        self.bridge = CvBridge()
        self.store = ProfileStore(self.profile_path)

        self.color_sub = Subscriber(
            self,
            Image,
            color_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=int(self.get_parameter("sync_queue").value),
            slop=float(self.get_parameter("sync_slop").value),
        )
        self.sync.registerCallback(self.image_callback)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_hsv: Optional[np.ndarray] = None
        self.latest_depth_mm: Optional[np.ndarray] = None
        self.latest_overlay: Optional[np.ndarray] = None
        self.last_outputs: Dict[str, np.ndarray] = {}
        self.last_metrics: Dict[str, Any] = {}
        self.last_roi_stats: Dict[str, Any] = {}

        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_current: Optional[Tuple[int, int]] = None
        self.current_roi_pixels: Optional[np.ndarray] = None
        self.sample_banks: Dict[str, np.ndarray] = {}
        self.sample_bank_stats: Dict[str, Dict[str, Any]] = {}
        self.sample_add_counts: Dict[str, int] = {}

        # UI and reversible auto-fit state.  Nothing here changes IRC files or
        # a running detector node; D mode is a local, read-only preview.
        self.view_mode = "calibration"
        self.context_panel_modes: Dict[str, str] = {
            target: "none" for target in TARGETS
        }
        self.last_fitted_profiles: Dict[str, Dict[str, Any]] = {}
        self.pre_fit_profiles: Dict[str, Dict[str, Any]] = {}
        self.session_profile_backup: Optional[Path] = None

        # Camera intrinsics and short hold state mirror the production ball
        # detector.  CameraInfo replaces these fallbacks as soon as it arrives.
        self.fx = 607.0
        self.fy = 606.0
        self.cx_intr = 325.5
        self.cy_intr = 239.4
        self.camera_info_received = False
        self.preview_last_ball: Optional[Dict[str, Any]] = None
        self.preview_last_rejection: Optional[Dict[str, Any]] = None
        self.preview_ball_lost_frames = BALL_PREVIEW_HOLD_FRAMES

        self.last_frame_time = time.monotonic()
        self.fps_ema = 0.0
        self.warned_shape = False

        self._setup_ui()
        self._apply_profile_to_controls(self.store.get(self.target))
        self._show_waiting_ui()

        # OpenCV window events must keep running even when the RealSense stops
        # publishing frames.  Keeping waitKey() inside image_callback() made
        # Ubuntu report "Unknown is not responding" during a UVC timeout.
        self.gui_timer = self.create_timer(0.02, self._poll_gui)

        self.get_logger().info(
            f"Calibration node ready | color={color_topic} | depth={depth_topic}"
        )
        self.get_logger().info(
            "Keys: b/k/f/g target, SPACE add ROI, a auto-fit, d preview, "
            "r restore pre-fit, n new, x clear bank, s save, l load, "
            "i snapshot, q quit"
        )

    # ------------------------------------------------------------------ UI --
    def _setup_ui(self) -> None:
        cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

        # Some Ubuntu/Qt themes render OpenCV trackbar labels as white text on
        # a white background.  Use the normal GUI mode (without the image
        # toolbar when supported) and draw our own readable guide above the
        # sliders so the control order and current values are always visible.
        control_flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            control_flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(CONTROL_WINDOW, control_flags)
        cv2.resizeWindow(CONTROL_WINDOW, 780, 760)
        cv2.setMouseCallback(MAIN_WINDOW, self._mouse_callback)

        initial = default_profile("ball")
        values = {
            "H low": initial["h_low"],
            "H high": initial["h_high"],
            "S low": initial["s_low"],
            "S high": initial["s_high"],
            "V low": initial["v_low"],
            "V high": initial["v_high"],
            "Depth min mm": initial["depth_min_mm"],
            "Depth max mm": initial["depth_max_mm"],
            "Blur": initial["blur_size"],
            "Morph": initial["morph_size"],
            "Min area": initial["min_area"],
            "Ball circ %": int(initial["ball_circularity_min"] * 100),
        }
        for name, maximum in TRACKBARS.items():
            cv2.createTrackbar(
                name,
                CONTROL_WINDOW,
                clamp_int(values[name], 0, maximum),
                maximum,
                noop,
            )

    def _show_waiting_ui(self) -> None:
        """Draw useful windows before the first synchronized camera frame."""
        waiting = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            waiting,
            "Waiting for synchronized RealSense frames...",
            (28, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            waiting,
            "The window stays responsive; Q or ESC exits.",
            (28, 252),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(MAIN_WINDOW, waiting)
        cv2.imshow(MASK_WINDOW, waiting)
        cv2.imshow(
            CONTROL_WINDOW,
            self._make_control_guide(self.store.get(self.target)),
        )

    def _poll_gui(self) -> None:
        """Process OpenCV input independently of the camera callbacks."""
        key = cv2.waitKey(1) & 0xFF
        self._handle_key(key)

    def _read_controls(self) -> Dict[str, Any]:
        def pos(name: str) -> int:
            return cv2.getTrackbarPos(name, CONTROL_WINDOW)

        depth_min = pos("Depth min mm")
        depth_max = pos("Depth max mm")
        if depth_min > depth_max:
            depth_min, depth_max = depth_max, depth_min

        return {
            "h_low": pos("H low"),
            "h_high": pos("H high"),
            "s_low": pos("S low"),
            "s_high": pos("S high"),
            "v_low": pos("V low"),
            "v_high": pos("V high"),
            "depth_min_mm": depth_min,
            "depth_max_mm": depth_max,
            "blur_size": odd_kernel(pos("Blur"), allow_zero=True),
            "morph_size": odd_kernel(pos("Morph")),
            "min_area": pos("Min area"),
            "ball_circularity_min": pos("Ball circ %") / 100.0,
        }

    def _apply_profile_to_controls(self, profile: Dict[str, Any]) -> None:
        values = {
            "H low": profile.get("h_low", 0),
            "H high": profile.get("h_high", 179),
            "S low": profile.get("s_low", 0),
            "S high": profile.get("s_high", 255),
            "V low": profile.get("v_low", 0),
            "V high": profile.get("v_high", 255),
            "Depth min mm": profile.get("depth_min_mm", 0),
            "Depth max mm": profile.get("depth_max_mm", 6_000),
            "Blur": profile.get("blur_size", 0),
            "Morph": profile.get("morph_size", 5),
            "Min area": profile.get("min_area", 100),
            "Ball circ %": int(
                float(profile.get("ball_circularity_min", 0.45)) * 100
            ),
        }
        for name, maximum in TRACKBARS.items():
            cv2.setTrackbarPos(
                name,
                CONTROL_WINDOW,
                clamp_int(values[name], 0, maximum),
            )

    def _commit_current_profile(self) -> Dict[str, Any]:
        profile = self._read_controls()
        self.store.set(self.target, profile)
        return profile

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) < 9:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return
        self.fx = fx
        self.fy = fy
        self.cx_intr = cx
        self.cy_intr = cy
        if not self.camera_info_received:
            self.camera_info_received = True
            self.get_logger().info(
                "CameraInfo ready for production ball preview: "
                f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
            )

    def _reset_preview_state(self) -> None:
        self.preview_last_ball = None
        self.preview_last_rejection = None
        self.preview_ball_lost_frames = BALL_PREVIEW_HOLD_FRAMES

    def _toggle_detection_preview(self) -> None:
        if self.view_mode == "detection":
            self.view_mode = "calibration"
            self._reset_preview_state()
            self.get_logger().info("View -> CALIBRATION")
            return

        if self.target not in self.last_fitted_profiles:
            self.get_logger().warning(
                "Press A successfully before opening detection preview."
            )
            return

        self.view_mode = "detection"
        self._reset_preview_state()
        self.get_logger().info(
            f"View -> DETECTION PREVIEW ({self.target}); press D to return"
        )

    def _restore_pre_fit(self) -> None:
        original = self.pre_fit_profiles.get(self.target)
        if original is None:
            self.get_logger().warning(
                "No pre-auto-fit values are available for this target."
            )
            return

        self._apply_profile_to_controls(original)
        self.store.set(self.target, original)
        self.pre_fit_profiles.pop(self.target, None)
        self.last_fitted_profiles.pop(self.target, None)
        self.context_panel_modes[self.target] = (
            "bank" if self.target in self.sample_bank_stats else "none"
        )
        self.view_mode = "calibration"
        self._reset_preview_state()
        self.get_logger().info(
            f"Restored pre-auto-fit values -> {self.target}; "
            "press S only if you also want to restore the YAML file"
        )

    def _switch_target(self, target: str) -> None:
        if target not in TARGETS or target == self.target:
            return
        self._commit_current_profile()
        self.target = target
        self._apply_profile_to_controls(self.store.get(self.target))
        self._clear_roi()
        self.view_mode = "calibration"
        self._reset_preview_state()
        self.get_logger().info(f"Target -> {self.target}")

    # --------------------------------------------------------------- Mouse --
    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _userdata: Any,
    ) -> None:
        if self.latest_frame is None or self.view_mode != "calibration":
            return

        height, width = self.latest_frame.shape[:2]
        x = clamp_int(x, 0, width - 1)
        y = clamp_int(y, 0, height - 1)

        if event == cv2.EVENT_RBUTTONDOWN:
            self._clear_roi()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_current = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_current = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_current = (x, y)
            if self.drag_start is None:
                return

            x1, y1 = self.drag_start
            x2, y2 = x, y
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            if right - left < 4 or bottom - top < 4:
                self._clear_roi()
                return

            self.roi = (left, top, right, bottom)
            self._refresh_roi_pixels()

    def _clear_roi(self) -> None:
        self.roi = None
        self.dragging = False
        self.drag_start = None
        self.drag_current = None
        self.current_roi_pixels = None
        self.last_roi_stats = {}

    def _refresh_roi_pixels(self) -> None:
        if self.roi is None or self.latest_hsv is None:
            return

        x1, y1, x2, y2 = self.roi
        roi_hsv = self.latest_hsv[y1:y2, x1:x2]
        if roi_hsv.size == 0:
            return

        height, width = roi_hsv.shape[:2]
        sample_mask = np.ones((height, width), dtype=bool)

        # A rectangular ROI around a ball contains background at its corners.
        # Use an inner ellipse for ball samples to reduce contamination.
        if self.target == "ball" and width >= 6 and height >= 6:
            yy, xx = np.ogrid[:height, :width]
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            rx = max(1.0, width * 0.46)
            ry = max(1.0, height * 0.46)
            sample_mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

        pixels = roi_hsv[sample_mask].reshape(-1, 3)
        if pixels.size == 0:
            return

        if pixels.shape[0] > MAX_ROI_PIXELS_PER_ADD:
            indices = np.linspace(
                0,
                pixels.shape[0] - 1,
                MAX_ROI_PIXELS_PER_ADD,
                dtype=np.int64,
            )
            pixels = pixels[indices]

        self.current_roi_pixels = pixels.copy()
        self.last_roi_stats = self._describe_samples(pixels)

    # ----------------------------------------------------------- Statistics --
    def _describe_samples(self, pixels: np.ndarray) -> Dict[str, Any]:
        pixels = np.asarray(pixels).reshape(-1, 3)
        h = pixels[:, 0].astype(np.int16)
        s = pixels[:, 1].astype(np.float32)
        v = pixels[:, 2].astype(np.float32)

        hue_valid = h[(s > 10) & (v > 10)]
        if hue_valid.size < 20:
            hue_valid = h
        h_low, h_high = shortest_hue_interval(hue_valid)

        return {
            "count": int(pixels.shape[0]),
            "h_fit_low": h_low,
            "h_fit_high": h_high,
            "s_p02": float(np.percentile(s, 2)),
            "s_p50": float(np.percentile(s, 50)),
            "s_p99": float(np.percentile(s, 99)),
            "v_p02": float(np.percentile(v, 2)),
            "v_p50": float(np.percentile(v, 50)),
            "v_p99": float(np.percentile(v, 99)),
        }

    def _frame_metrics(self, hsv: np.ndarray) -> Dict[str, Any]:
        value = hsv[:, :, 2]
        p10, p50, p90 = np.percentile(value, (10, 50, 90))
        return {
            "frame_v_p10": float(p10),
            "frame_v_p50": float(p50),
            "frame_v_p90": float(p90),
            "underexposed_pct": float(np.mean(value <= UNDEREXPOSED_V) * 100.0),
            "overexposed_pct": float(np.mean(value >= OVEREXPOSED_V) * 100.0),
        }

    def _add_roi_to_bank(self) -> None:
        self._refresh_roi_pixels()
        if self.current_roi_pixels is None:
            self.get_logger().warning("Select a tight ROI first.")
            return

        key = self.target
        existing = self.sample_banks.get(key)
        if existing is None:
            bank = self.current_roi_pixels.copy()
        else:
            bank = np.concatenate([existing, self.current_roi_pixels], axis=0)

        if bank.shape[0] > MAX_BANK_PIXELS:
            indices = np.linspace(
                0,
                bank.shape[0] - 1,
                MAX_BANK_PIXELS,
                dtype=np.int64,
            )
            bank = bank[indices]

        self.sample_banks[key] = bank
        stats = self._describe_samples(bank)
        self.sample_bank_stats[key] = stats
        self.sample_add_counts[key] = self.sample_add_counts.get(key, 0) + 1
        # New samples make an older fit stale.  Return the context panel to
        # ROI-bank mode until A is pressed again.
        self.last_fitted_profiles.pop(key, None)
        self.context_panel_modes[key] = "bank"
        self.view_mode = "calibration"
        self._reset_preview_state()
        self.get_logger().info(
            f"Added ROI -> {key}, bank={stats['count']} px, "
            f"H={stats['h_fit_low']}..{stats['h_fit_high']}, "
            f"S p02/p99={stats['s_p02']:.0f}/{stats['s_p99']:.0f}, "
            f"V p02/p99={stats['v_p02']:.0f}/{stats['v_p99']:.0f}"
        )

    def _clear_bank(self) -> None:
        key = self.target
        self.sample_banks.pop(key, None)
        self.sample_bank_stats.pop(key, None)
        self.sample_add_counts.pop(key, None)
        self.context_panel_modes[key] = (
            "fit" if key in self.last_fitted_profiles else "none"
        )
        self.get_logger().info(f"Cleared sample bank -> {key}")

    def _new_tuning_session(self) -> None:
        """Start sampling the current target again without preset light classes."""
        self._clear_bank()
        self._clear_roi()
        self.last_fitted_profiles.pop(self.target, None)
        self.context_panel_modes[self.target] = "none"
        self.view_mode = "calibration"
        self._reset_preview_state()
        self.get_logger().info(
            f"New tuning session -> target={self.target}, "
            f"environment={self.environment_label}"
        )

    def _auto_fit(self) -> None:
        key = self.target
        pixels = self.sample_banks.get(key)
        if pixels is None:
            self._refresh_roi_pixels()
            pixels = self.current_roi_pixels
        if pixels is None or pixels.size == 0:
            self.get_logger().warning("Select/add ROI samples before auto-fit.")
            return

        # Preserve the values from before the first A in this tuning cycle.
        # Repeated auto-fit operations must not overwrite this restore point.
        self.pre_fit_profiles.setdefault(key, self._read_controls().copy())

        h = pixels[:, 0].astype(np.int16)
        s = pixels[:, 1].astype(np.float32)
        v = pixels[:, 2].astype(np.float32)

        if key == "support":
            # H is undefined for black and S varies with reflections.  Keep
            # both fully open and fit only a robust upper brightness bound.
            h_low, h_high = 0, 179
            s_low, s_high = 0, 255
            v_low = 0
            v_high = clamp_int(
                np.percentile(v, SUPPORT_V_PERCENTILE) + SUPPORT_V_MARGIN,
                0,
                255,
            )
        else:
            hue_valid = h[(s > 10) & (v > 10)]
            if hue_valid.size < 20:
                hue_valid = h
            h_low, h_high = shortest_hue_interval(hue_valid)

            s_low = clamp_int(
                np.percentile(s, SV_LOW_PERCENTILE) - S_LOW_MARGIN,
                0,
                255,
            )
            s_high = clamp_int(
                np.percentile(s, SV_HIGH_PERCENTILE) + S_HIGH_MARGIN,
                0,
                255,
            )
            v_low = clamp_int(
                np.percentile(v, SV_LOW_PERCENTILE) - V_LOW_MARGIN,
                0,
                255,
            )
            # A brighter view of the same orange ball should not be rejected
            # just because it exceeds the sampled V maximum.
            v_high = (
                255
                if key == "ball"
                else clamp_int(
                    np.percentile(v, SV_HIGH_PERCENTILE) + V_HIGH_MARGIN,
                    0,
                    255,
                )
            )

        cv2.setTrackbarPos("H low", CONTROL_WINDOW, h_low)
        cv2.setTrackbarPos("H high", CONTROL_WINDOW, h_high)
        cv2.setTrackbarPos("S low", CONTROL_WINDOW, s_low)
        cv2.setTrackbarPos("S high", CONTROL_WINDOW, s_high)
        cv2.setTrackbarPos("V low", CONTROL_WINDOW, v_low)
        cv2.setTrackbarPos("V high", CONTROL_WINDOW, v_high)
        fitted_profile = self._commit_current_profile()
        self.last_fitted_profiles[key] = dict(fitted_profile)
        self.context_panel_modes[key] = "fit"
        self.view_mode = "calibration"
        self._reset_preview_state()

        if key == "support":
            self.get_logger().info(
                f"Auto-fit support: black V <= {v_high} "
                f"(p{SUPPORT_V_PERCENTILE:.0f} + {SUPPORT_V_MARGIN})"
            )
        else:
            wrap_text = " (wraps through 0)" if h_low > h_high else ""
            self.get_logger().info(
                f"Auto-fit {key}: H={h_low}..{h_high}{wrap_text}, "
                f"S={s_low}..{s_high}, V={v_low}..{v_high}"
            )

    # --------------------------------------------------------------- Masks --
    def _process_masks(
        self,
        frame: np.ndarray,
        depth_mm: np.ndarray,
        profile: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        blur_size = int(profile["blur_size"])
        if blur_size >= 3:
            processed = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        else:
            processed = frame

        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        raw_color_mask = hue_mask(hsv, profile)

        valid_depth = np.isfinite(depth_mm) & (depth_mm > 0)
        depth_mask = np.zeros(depth_mm.shape, dtype=np.uint8)
        depth_ok = (
            valid_depth
            & (depth_mm >= float(profile["depth_min_mm"]))
            & (depth_mm <= float(profile["depth_max_mm"]))
        )
        depth_mask[depth_ok] = 255

        morph_size = int(profile["morph_size"])
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_size, morph_size),
        )
        color_mask = cv2.morphologyEx(
            raw_color_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_OPEN,
            kernel,
        )
        clean_depth_mask = cv2.morphologyEx(
            depth_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )
        clean_depth_mask = cv2.morphologyEx(
            clean_depth_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        combined_mask = cv2.bitwise_and(color_mask, clean_depth_mask)
        masked_color = cv2.bitwise_and(frame, frame, mask=combined_mask)

        return {
            "hsv": hsv,
            "raw_color_mask": raw_color_mask,
            "color_mask": color_mask,
            "depth_mask": clean_depth_mask,
            "combined_mask": combined_mask,
            "masked_color": masked_color,
        }

    # -------------------------------------------- Production ball preview --
    @staticmethod
    def _preview_depth_m(
        depth_mm: np.ndarray,
        cx: int,
        cy: int,
    ) -> Optional[float]:
        frame_h, frame_w = depth_mm.shape[:2]
        x1 = max(0, cx - 1)
        x2 = min(frame_w, cx + 2)
        y1 = max(0, cy - 1)
        y2 = min(frame_h, cy + 2)
        patch = depth_mm[y1:y2, x1:x2]
        valid = patch[
            np.isfinite(patch)
            & (patch > 0.0)
            & (patch <= BALL_PREVIEW_DEPTH_MAX_MM)
        ]
        if valid.size == 0:
            return None
        return float(np.median(valid) * 0.001)

    def _preview_support_config(self) -> BallSupportConfig:
        settings = self.store.get_detector()
        support_profile = self.store.get("support")
        return BallSupportConfig(
            support_diameter_m=float(settings["support_diameter_m"]),
            support_v_max=int(support_profile["v_high"]),
            black_ratio_min=float(
                settings["support_black_ratio_min"]
            ),
            edge_black_ratio_min=float(
                settings["edge_support_black_ratio_min"]
            ),
            surrounding_ball_ratio_max=float(
                settings["support_ball_color_ratio_max"]
            ),
            floor_ratio_max=float(
                settings["support_floor_ratio_max"]
            ),
            sector_black_ratio_min=float(
                settings["support_sector_black_ratio_min"]
            ),
            min_sectors=int(settings["support_min_sectors"]),
            edge_min_sectors=int(
                settings["edge_support_min_sectors"]
            ),
            min_visible_fraction=float(
                settings["support_min_visible_fraction"]
            ),
            edge_min_visible_fraction=float(
                settings["edge_support_min_visible_fraction"]
            ),
        )

    def _find_production_ball(
        self,
        mask: np.ndarray,
        hsv: np.ndarray,
        ball_color_mask: np.ndarray,
        floor_mask: np.ndarray,
        depth_mm: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Mirror the current IRC RealSense ball acceptance checks."""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        frame_h, frame_w = mask.shape[:2]
        best: Optional[Dict[str, Any]] = None
        support_config = self._preview_support_config()
        support_black_mask = black_support_mask(
            hsv,
            support_config.support_v_max,
        )
        self.preview_last_rejection = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            bx, by, bw, bh = cv2.boundingRect(contour)
            margin = BALL_PREVIEW_EDGE_MARGIN_PX
            touches_edge = bool(
                bx <= margin
                or by <= margin
                or bx + bw >= frame_w - margin
                or by + bh >= frame_h - margin
            )

            min_area = BALL_PREVIEW_MIN_AREA
            if touches_edge:
                min_area *= BALL_PREVIEW_EDGE_MIN_AREA_RATIO
            if area < min_area:
                continue

            aspect = float(bw) / max(float(bh), 1.0)
            min_aspect = (
                BALL_PREVIEW_EDGE_MIN_ASPECT
                if touches_edge
                else BALL_PREVIEW_MIN_ASPECT
            )
            max_aspect = (
                BALL_PREVIEW_EDGE_MAX_ASPECT
                if touches_edge
                else BALL_PREVIEW_MAX_ASPECT
            )
            if not (min_aspect <= aspect <= max_aspect):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            required_circularity = (
                BALL_PREVIEW_EDGE_MIN_CIRCULARITY
                if touches_edge
                else BALL_PREVIEW_MIN_CIRCULARITY
            )
            if circularity < required_circularity:
                continue

            (cx_float, cy_float), radius = cv2.minEnclosingCircle(contour)
            if radius <= 1e-6:
                continue
            circle_area = math.pi * radius * radius
            ratio_error = abs((area / circle_area) - 1.0)
            max_ratio_error = (
                BALL_PREVIEW_EDGE_MAX_CIRCLE_RATIO_ERROR
                if touches_edge
                else BALL_PREVIEW_MAX_CIRCLE_RATIO_ERROR
            )
            if ratio_error > max_ratio_error:
                continue

            cx = int(round(cx_float))
            cy = int(round(cy_float))
            z_m = self._preview_depth_m(depth_mm, cx, cy)
            if z_m is None or self.fx <= 0.0 or self.fy <= 0.0:
                continue

            expected_min = self.fx * (BALL_PREVIEW_DIAMETER_MIN_M * 0.5) / z_m
            expected_max = self.fx * (BALL_PREVIEW_DIAMETER_MAX_M * 0.5) / z_m
            size_min_ratio = (
                BALL_PREVIEW_EDGE_RADIUS_MIN_RATIO
                if touches_edge
                else BALL_PREVIEW_RADIUS_MIN_RATIO
            )
            size_max_ratio = (
                BALL_PREVIEW_EDGE_RADIUS_MAX_RATIO
                if touches_edge
                else BALL_PREVIEW_RADIUS_MAX_RATIO
            )
            if not (
                expected_min * size_min_ratio
                <= radius
                <= expected_max * size_max_ratio
            ):
                continue

            expected_mid = 0.5 * (expected_min + expected_max)
            radius_ratio = radius / max(expected_mid, 1e-6)
            support = evaluate_ball_support(
                hsv=hsv,
                ball_color_mask=ball_color_mask,
                floor_mask=floor_mask,
                center=(float(cx_float), float(cy_float)),
                detected_radius_px=float(radius),
                expected_ball_radius_px=float(expected_mid),
                depth_m=float(z_m),
                focal_x_px=float(self.fx),
                touches_edge=touches_edge,
                config=support_config,
                black_mask=support_black_mask,
            )
            if not bool(support["accepted"]):
                rejection = {
                    **support,
                    "circularity": float(circularity),
                    "radius_ratio": float(radius / max(expected_mid, 1e-6)),
                }
                if (
                    self.preview_last_rejection is None
                    or float(rejection["black_ratio"])
                    > float(self.preview_last_rejection["black_ratio"])
                ):
                    self.preview_last_rejection = rejection
                continue

            score = (
                ratio_error
                + (1.0 - min(circularity, 1.0))
                + abs(radius_ratio - 1.0)
                + 0.5 * (1.0 - float(support["black_ratio"]))
                + float(support["surrounding_ball_ratio"])
                + float(support["floor_ratio"])
            )
            if best is not None and score >= float(best["score"]):
                continue

            x_m = (cx - self.cx_intr) * z_m / self.fx
            angle_deg = math.degrees(math.atan2(x_m, z_m))
            best = {
                "score": float(score),
                "cx": cx,
                "cy": cy,
                "radius": float(radius),
                "distance_cm": float(z_m * 100.0),
                "angle_deg": float(angle_deg),
                "area": area,
                "circularity": float(circularity),
                "aspect": float(aspect),
                "radius_ratio": float(radius_ratio),
                "touches_edge": touches_edge,
                "support_black_ratio": float(support["black_ratio"]),
                "support_ball_ratio": float(
                    support["surrounding_ball_ratio"]
                ),
                "support_floor_ratio": float(support["floor_ratio"]),
                "support_sectors": int(support["qualified_sectors"]),
                "support_visible_fraction": float(
                    support["visible_fraction"]
                ),
                "support_outer_radius": float(support["outer_radius_px"]),
            }

        return best

    def _draw_ball_detection_preview(
        self,
        frame: np.ndarray,
        depth_mm: np.ndarray,
        profile: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """Draw a local preview equivalent to IRC's RealSense ball path."""
        overlay = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        ball_color_mask = empty_mask.copy()
        depth_mask = empty_mask.copy()
        mask = empty_mask.copy()
        h_low = int(profile["h_low"])
        h_high = int(profile["h_high"])
        compatible = h_low <= h_high

        detection: Optional[Dict[str, Any]] = None
        if compatible:
            ball_color_mask = cv2.inRange(
                hsv,
                np.array(
                    [h_low, int(profile["s_low"]), int(profile["v_low"])],
                    dtype=np.uint8,
                ),
                np.array(
                    [h_high, int(profile["s_high"]), int(profile["v_high"])],
                    dtype=np.uint8,
                ),
            )
            floor_profile = self.store.get("floor")
            floor_mask = hsv_range_mask(
                hsv,
                (
                    int(floor_profile["h_low"]),
                    int(floor_profile["s_low"]),
                    int(floor_profile["v_low"]),
                ),
                (
                    int(floor_profile["h_high"]),
                    int(floor_profile["s_high"]),
                    int(floor_profile["v_high"]),
                ),
            )
            valid_depth = (
                np.isfinite(depth_mm)
                & (depth_mm > 0.0)
                & (depth_mm <= BALL_PREVIEW_DEPTH_MAX_MM)
            )
            depth_mask[valid_depth] = 255
            mask = cv2.bitwise_and(ball_color_mask, depth_mask)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (BALL_PREVIEW_MORPH_SIZE, BALL_PREVIEW_MORPH_SIZE),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            detection = self._find_production_ball(
                mask,
                hsv,
                ball_color_mask,
                floor_mask,
                depth_mm,
            )

        masked_color = cv2.bitwise_and(frame, frame, mask=mask)
        outputs = {
            "hsv": hsv,
            "raw_color_mask": ball_color_mask,
            "color_mask": mask,
            "depth_mask": depth_mask,
            "combined_mask": mask,
            "masked_color": masked_color,
        }

        held = False
        if detection is not None:
            self.preview_last_ball = dict(detection)
            self.preview_ball_lost_frames = 0
            display = detection
        elif (
            compatible
            and self.preview_last_ball is not None
            and self.preview_ball_lost_frames < BALL_PREVIEW_HOLD_FRAMES
        ):
            self.preview_ball_lost_frames += 1
            held = True
            display = self.preview_last_ball
        else:
            self.preview_ball_lost_frames = BALL_PREVIEW_HOLD_FRAMES
            display = None

        if not compatible:
            state = "UNSUPPORTED HUE WRAP"
            color = (0, 0, 255)
        elif detection is not None:
            state = "DETECTED"
            color = (0, 255, 0)
        elif held:
            state = "HOLD"
            color = (0, 255, 255)
        else:
            state = "MISS"
            color = (0, 0, 255)

        if display is not None:
            center = (int(display["cx"]), int(display["cy"]))
            radius = max(1, int(round(float(display["radius"]))))
            support_radius = max(
                1,
                int(round(float(display["support_outer_radius"]))),
            )
            cv2.circle(overlay, center, radius, color, 3)
            cv2.circle(
                overlay,
                center,
                support_radius,
                (255, 180, 0),
                1,
            )
            cv2.circle(overlay, center, 4, (0, 0, 255), -1)
            detail = (
                f"{float(display['distance_cm']):.1f}cm  "
                f"{float(display['angle_deg']):+.1f}deg  "
                f"circ {float(display['circularity']):.2f}"
            )
            support_detail = (
                f"black {float(display['support_black_ratio']):.2f}  "
                f"floor {float(display['support_floor_ratio']):.2f}  "
                f"orange {float(display['support_ball_ratio']):.2f}  "
                f"sectors {int(display['support_sectors'])}"
            )
        else:
            detail = "No ball passes IRC depth/shape/physical-size checks"
            if self.preview_last_rejection is not None:
                rejection = self.preview_last_rejection
                detail = f"REJECT: {rejection['reason']}"
                support_detail = (
                    f"black {float(rejection['black_ratio']):.2f}  "
                    f"floor {float(rejection['floor_ratio']):.2f}  "
                    f"orange "
                    f"{float(rejection['surrounding_ball_ratio']):.2f}  "
                    f"sectors {int(rejection['qualified_sectors'])}"
                )
            else:
                support_detail = (
                    "Every ball must pass black-support/floor checks"
                )

        cv2.rectangle(overlay, (5, 5), (480, 92), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"IRC BALL PREVIEW: {state}",
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            detail,
            (12, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            support_detail,
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            (
                f"HSV H {profile['h_low']}..{profile['h_high']}  "
                f"S {profile['s_low']}..{profile['s_high']}  "
                f"V {profile['v_low']}..{profile['v_high']} | D return"
            ),
            (12, 87),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        preview_w = min(150, max(1, overlay.shape[1] // 4))
        scale = preview_w / float(max(1, mask.shape[1]))
        preview_h = max(1, int(round(mask.shape[0] * scale)))
        mask_preview = cv2.cvtColor(
            cv2.resize(
                mask,
                (preview_w, preview_h),
                interpolation=cv2.INTER_NEAREST,
            ),
            cv2.COLOR_GRAY2BGR,
        )
        px = max(0, overlay.shape[1] - preview_w - 5)
        py = 5
        ph = min(preview_h, overlay.shape[0] - py)
        overlay[py:py + ph, px:px + preview_w] = mask_preview[:ph]

        preview_metrics = {
            "preview_irc_compatible": compatible,
            "preview_detected": detection is not None,
            "preview_held": held,
            "preview_state": state,
            "candidate_count": int(detection is not None),
            "accepted_count": int(detection is not None),
            "largest_area": (
                float(detection["area"])
                if detection is not None
                else 0.0
            ),
            "best_circularity": (
                float(detection["circularity"])
                if detection is not None
                else 0.0
            ),
        }
        return overlay, outputs, preview_metrics

    def _draw_candidates(
        self,
        frame: np.ndarray,
        combined_mask: np.ndarray,
        depth_mm: np.ndarray,
        profile: Dict[str, Any],
        draw_preview: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        overlay = frame.copy()
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        min_area = float(profile["min_area"])
        circularity_min = float(profile["ball_circularity_min"])

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                4.0 * math.pi * area / (perimeter * perimeter)
                if perimeter > 1e-6
                else 0.0
            )
            x, y, width, height = cv2.boundingRect(contour)
            aspect = width / float(max(1, height))
            rectangularity = area / float(max(1, width * height))

            local_mask = np.zeros((height, width), dtype=np.uint8)
            shifted = contour.copy()
            shifted[:, 0, 0] -= x
            shifted[:, 0, 1] -= y
            cv2.drawContours(local_mask, [shifted], -1, 255, thickness=-1)
            local_depth = depth_mm[y:y + height, x:x + width]
            valid = (
                (local_mask > 0)
                & np.isfinite(local_depth)
                & (local_depth > 0)
            )
            median_depth = float(np.median(local_depth[valid])) if np.any(valid) else 0.0

            accepted = True
            if self.target == "ball":
                accepted = circularity >= circularity_min

            score = area * (max(circularity, 0.05) if self.target == "ball" else 1.0)
            candidates.append(
                {
                    "contour": contour,
                    "area": area,
                    "circularity": circularity,
                    "aspect": aspect,
                    "rectangularity": rectangularity,
                    "depth_mm": median_depth,
                    "bbox": (x, y, width, height),
                    "accepted": accepted,
                    "score": score,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        accepted_count = sum(bool(item["accepted"]) for item in candidates)
        accepted_candidates = [item for item in candidates if item["accepted"]]

        if draw_preview:
            best = accepted_candidates[0] if accepted_candidates else None
            state = "DETECTED" if best is not None else "MISS"
            color = (0, 255, 0) if best is not None else (0, 0, 255)
            if best is not None:
                x, y, width, height = best["bbox"]
                cv2.rectangle(
                    overlay,
                    (x, y),
                    (x + width, y + height),
                    color,
                    3,
                )
                cv2.drawContours(
                    overlay,
                    [best["contour"]],
                    -1,
                    (255, 255, 0),
                    2,
                )
                detail = (
                    f"area {best['area']:.0f}  depth {best['depth_mm']:.0f}mm"
                )
            else:
                detail = "No contour passes the current calibration checks"
            cv2.rectangle(overlay, (5, 5), (430, 58), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                f"CALIBRATOR {self.target.upper()} PREVIEW: {state}",
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"{detail} | D return",
                (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

        # Candidate measurements are useful for metrics and snapshots, but
        # drawing every automatically detected contour looks like an ROI and
        # makes manual sampling confusing.  Keep the calculations above while
        # leaving the main view clean; only the user-drawn ROI is rendered in
        # _draw_status().

        metrics: Dict[str, Any] = {
            "candidate_count": len(candidates),
            "accepted_count": accepted_count,
            "largest_area": candidates[0]["area"] if candidates else 0.0,
            "best_circularity": candidates[0]["circularity"] if candidates else 0.0,
            "best_depth_mm": candidates[0]["depth_mm"] if candidates else 0.0,
        }
        return overlay, metrics

    # -------------------------------------------------------------- Display --
    @staticmethod
    def _panel(image: np.ndarray, label: str, size: Tuple[int, int] = (320, 240)) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(panel, (0, 0), (size[0], 25), (0, 0, 0), thickness=-1)
        cv2.putText(
            panel,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _make_control_guide(self, profile: Dict[str, Any]) -> np.ndarray:
        """Draw a readable guide because native Qt trackbar labels may vanish."""
        width, height = 760, 390
        panel = np.zeros((height, width, 3), dtype=np.uint8)

        target_name = self.target.upper()
        title = f"HSV / DEPTH CONTROLS   TARGET: {target_name}"
        cv2.putText(
            panel,
            title,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Move the 12 sliders below.  Their exact top-to-bottom order is:",
            (18, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        rows = [
            ("01  H low", profile["h_low"], "minimum hue (color), 0-179"),
            ("02  H high", profile["h_high"], "maximum hue (color), 0-179"),
            ("03  S low", profile["s_low"], "minimum saturation, 0-255"),
            ("04  S high", profile["s_high"], "maximum saturation, 0-255"),
            ("05  V low", profile["v_low"], "minimum brightness, 0-255"),
            ("06  V high", profile["v_high"], "maximum brightness, 0-255"),
            ("07  Depth min", profile["depth_min_mm"], "nearest accepted distance (mm)"),
            ("08  Depth max", profile["depth_max_mm"], "farthest accepted distance (mm)"),
            ("09  Blur", profile["blur_size"], "image smoothing; 0 disables it"),
            ("10  Morph", profile["morph_size"], "remove dots / fill small mask holes"),
            ("11  Min area", profile["min_area"], "reject smaller connected regions"),
            (
                "12  Ball circ %",
                int(round(profile["ball_circularity_min"] * 100)),
                "minimum roundness; ignored for hoop",
            ),
        ]

        start_y = 82
        line_h = 24
        for index, (name, value, description) in enumerate(rows):
            y = start_y + index * line_h
            cv2.putText(
                panel,
                f"{name:<18} = {value:>5}",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                description,
                (300, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

        note_y = start_y + len(rows) * line_h + 10
        cv2.putText(
            panel,
            "Targets: B ball | K black support | F red floor | G hoop",
            (18, note_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Workflow: ROI -> SPACE xN -> A fit -> D preview -> R restore or S save",
            (18, note_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _make_grid(self, frame: np.ndarray, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        panels = [
            self._panel(frame, "Original"),
            self._panel(outputs["raw_color_mask"], "Raw HSV mask"),
            self._panel(outputs["color_mask"], "Clean HSV mask"),
            self._panel(outputs["depth_mask"], "Depth mask"),
            self._panel(outputs["combined_mask"], "HSV AND depth"),
            self._panel(outputs["masked_color"], "Combined result"),
        ]
        top = np.hstack(panels[:3])
        bottom = np.hstack(panels[3:])
        return np.vstack([top, bottom])

    @staticmethod
    def _draw_text_panel(
        image: np.ndarray,
        lines: list[str],
        anchor: str,
        color: Tuple[int, int, int],
        scale: float,
    ) -> Tuple[int, int, int, int]:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        padding_x = 6
        padding_y = 5
        line_gap = 4
        sizes = [
            cv2.getTextSize(text, font, scale, thickness)[0]
            for text in lines
        ]
        text_h = max(size[1] for size in sizes)
        line_h = text_h + line_gap
        panel_w = min(
            image.shape[1] - 10,
            max(size[0] for size in sizes) + 2 * padding_x,
        )
        panel_h = (
            2 * padding_y
            + len(lines) * text_h
            + (len(lines) - 1) * line_gap
        )
        panel_x = 5 if anchor == "left" else max(5, image.shape[1] - panel_w - 5)
        panel_y = 5
        cv2.rectangle(
            image,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (0, 0, 0),
            -1,
        )
        first_y = panel_y + padding_y + text_h
        for index, text in enumerate(lines):
            cv2.putText(
                image,
                text,
                (panel_x + padding_x, first_y + index * line_h),
                font,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        return panel_x, panel_y, panel_w, panel_h

    def _draw_status(
        self,
        overlay: np.ndarray,
        profile: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        # Requested layout: general live information at top-left and the
        # sampling/fit result at top-right.
        status_lines = [
            f"{self.target.upper()} | ENV {self.environment_label} | FPS {self.fps_ema:.1f}",
            (
                f"Current H {profile['h_low']}..{profile['h_high']}  "
                f"S {profile['s_low']}..{profile['s_high']}  "
                f"V {profile['v_low']}..{profile['v_high']}"
            ),
            (
                f"Depth {profile['depth_min_mm']}..{profile['depth_max_mm']}mm  "
                f"Morph {profile['morph_size']}  Area {profile['min_area']}"
            ),
            (
                f"Scene V {metrics['frame_v_p10']:.0f}/"
                f"{metrics['frame_v_p50']:.0f}/{metrics['frame_v_p90']:.0f}  "
                f"dark {metrics['underexposed_pct']:.1f}%  "
                f"clip {metrics['overexposed_pct']:.1f}%"
            ),
            "B ball | K support | F floor | G hoop",
            "SPACE sample | A fit | D preview | R restore | S save",
        ]
        self._draw_text_panel(
            overlay,
            status_lines,
            anchor="left",
            color=(255, 255, 255),
            scale=0.32,
        )

        key = self.target
        context_mode = self.context_panel_modes.get(key, "none")
        context_lines: list[str] = []
        context_color = (80, 255, 80)

        if context_mode == "bank":
            bank_stats = self.sample_bank_stats.get(key)
            if bank_stats:
                add_count = self.sample_add_counts.get(key, 0)
                if key == "support":
                    context_lines = [
                        f"BLACK SUPPORT BANK x{add_count} | "
                        f"{bank_stats['count']} px",
                        (
                            f"V p02/p50/p99 "
                            f"{bank_stats['v_p02']:.0f}/"
                            f"{bank_stats['v_p50']:.0f}/"
                            f"{bank_stats['v_p99']:.0f}"
                        ),
                        "Press A to fit the black V ceiling",
                    ]
                else:
                    hue_wrap = (
                        " wrap"
                        if bank_stats["h_fit_low"] > bank_stats["h_fit_high"]
                        else ""
                    )
                    context_lines = [
                        f"ROI BANK x{add_count} | {bank_stats['count']} px",
                        (
                            f"sample H {bank_stats['h_fit_low']}.."
                            f"{bank_stats['h_fit_high']}{hue_wrap}  "
                            f"S {bank_stats['s_p02']:.0f}.."
                            f"{bank_stats['s_p99']:.0f}  "
                            f"V {bank_stats['v_p02']:.0f}.."
                            f"{bank_stats['v_p99']:.0f}"
                        ),
                        "Press A to calculate final HSV",
                    ]
        elif context_mode == "fit":
            fitted = self.last_fitted_profiles.get(key)
            if fitted:
                if key == "support":
                    context_lines = [
                        f"AUTO-FIT RESULT | SUPPORT V <= {fitted['v_high']}",
                        "H/S ignored for black; V ceiling is exported",
                        "Switch B, then D to validate ball + support",
                    ]
                    context_color = (0, 220, 255)
                    fitted = None
            if fitted:
                hsv_keys = (
                    "h_low", "h_high", "s_low", "s_high", "v_low", "v_high"
                )
                manually_modified = any(
                    int(profile[name]) != int(fitted[name]) for name in hsv_keys
                )
                suffix = " | sliders modified" if manually_modified else ""
                hue_wrap = (
                    " wrap" if int(fitted["h_low"]) > int(fitted["h_high"]) else ""
                )
                context_lines = [
                    f"AUTO-FIT RESULT | {key.upper()}{suffix}",
                    (
                        f"H {fitted['h_low']}..{fitted['h_high']}{hue_wrap}  "
                        f"S {fitted['s_low']}..{fitted['s_high']}  "
                        f"V {fitted['v_low']}..{fitted['v_high']}"
                    ),
                    "D detection preview | R restore pre-A",
                ]
                context_color = (0, 220, 255)

        if context_lines:
            self._draw_text_panel(
                overlay,
                context_lines,
                anchor="right",
                color=context_color,
                scale=0.34,
            )

        roi_to_draw = self.roi
        if self.dragging and self.drag_start and self.drag_current:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            roi_to_draw = (left, top, right, bottom)
        if roi_to_draw:
            x1, y1, x2, y2 = roi_to_draw
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # --------------------------------------------------------------- Saving --
    def _backup_profile_file_once(self) -> None:
        if self.session_profile_backup is not None or not self.profile_path.exists():
            return
        backup_dir = self.profile_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"hsv_profiles_before_save_{stamp}.yaml"
        suffix = 1
        while backup_path.exists():
            backup_path = backup_dir / (
                f"hsv_profiles_before_save_{stamp}_{suffix}.yaml"
            )
            suffix += 1
        shutil.copy2(self.profile_path, backup_path)
        self.session_profile_backup = backup_path
        self.get_logger().info(f"Backed up previous profiles -> {backup_path}")

    def _save_profiles(self) -> None:
        self._commit_current_profile()
        try:
            self._backup_profile_file_once()
            self.store.save()
            self.get_logger().info(f"Saved profiles -> {self.profile_path}")
            self._save_ball_ros_params()
        except OSError as exc:
            self.get_logger().error(f"Could not save profiles: {exc}")

    def _save_ball_ros_params(self) -> None:
        """Export ball, black-support and floor calibration for production."""
        profile = self.store.get("ball")
        support_profile = self.store.get("support")
        floor_profile = self.store.get("floor")
        detector = self.store.get_detector()
        hsv_keys = (
            "h_low",
            "h_high",
            "s_low",
            "s_high",
            "v_low",
            "v_high",
        )
        values: Dict[str, Any] = {
            name: int(profile[name]) for name in hsv_keys
        }
        values["support_v_max"] = int(support_profile["v_high"])
        for name in hsv_keys:
            values[f"floor_{name}"] = int(floor_profile[name])
        values.update(
            {
                "support_diameter_m": float(
                    detector["support_diameter_m"]
                ),
                "support_black_ratio_min": float(
                    detector["support_black_ratio_min"]
                ),
                "edge_support_black_ratio_min": float(
                    detector["edge_support_black_ratio_min"]
                ),
                "support_ball_color_ratio_max": float(
                    detector["support_ball_color_ratio_max"]
                ),
                "support_floor_ratio_max": float(
                    detector["support_floor_ratio_max"]
                ),
                "support_sector_black_ratio_min": float(
                    detector["support_sector_black_ratio_min"]
                ),
                "support_min_sectors": int(
                    detector["support_min_sectors"]
                ),
                "edge_support_min_sectors": int(
                    detector["edge_support_min_sectors"]
                ),
                "support_min_visible_fraction": float(
                    detector["support_min_visible_fraction"]
                ),
                "edge_support_min_visible_fraction": float(
                    detector["edge_support_min_visible_fraction"]
                ),
            }
        )

        # The competition detector currently uses one cv2.inRange call and
        # therefore cannot consume a circular hue interval such as 170..10.
        # Keep the last valid production file instead of exporting a value the
        # detector would reject at startup.
        if values["h_low"] > values["h_high"]:
            self.get_logger().error(
                "Ball HSV was saved to the calibration profile, but not to "
                "ball_hsv.yaml: IRC ball detection does not support hue wrap."
            )
            return

        payload = {
            "ball_vision_fusion": {
                "ros__parameters": values,
            }
        }
        self.ball_params_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.ball_params_path.with_suffix(
            self.ball_params_path.suffix + ".tmp"
        )
        with temp_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                payload,
                file,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
        temp_path.replace(self.ball_params_path)
        self.get_logger().info(
            "Exported ball/support/floor detector calibration -> "
            f"{self.ball_params_path}"
        )

    def _reload_profiles(self) -> None:
        self.store.load()
        self._apply_profile_to_controls(self.store.get(self.target))
        self.pre_fit_profiles.clear()
        self.last_fitted_profiles.clear()
        for target in TARGETS:
            self.context_panel_modes[target] = (
                "bank" if target in self.sample_bank_stats else "none"
            )
        self.view_mode = "calibration"
        self._reset_preview_state()
        self.get_logger().info(f"Reloaded profiles <- {self.profile_path}")

    def _save_snapshot(self) -> None:
        if self.latest_frame is None or not self.last_outputs:
            self.get_logger().warning("No frame available yet.")
            return

        profile = self._commit_current_profile()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = self.output_dir / self.target / stamp
        folder.mkdir(parents=True, exist_ok=True)

        images = {
            "color.png": self.latest_frame,
            "overlay.png": self.latest_overlay,
            "raw_hsv_mask.png": self.last_outputs["raw_color_mask"],
            "clean_hsv_mask.png": self.last_outputs["color_mask"],
            "depth_mask.png": self.last_outputs["depth_mask"],
            "combined_mask.png": self.last_outputs["combined_mask"],
            "combined_color.png": self.last_outputs["masked_color"],
        }
        for filename, image in images.items():
            if image is not None:
                cv2.imwrite(str(folder / filename), image)
        if self.latest_depth_mm is not None:
            np.save(folder / "depth_mm.npy", self.latest_depth_mm)

        metadata = {
            "target": self.target,
            "environment_label": self.environment_label,
            "profile": profile,
            "metrics": self.last_metrics,
            "roi_stats": self.last_roi_stats,
            "depth_scale_to_mm": self.depth_scale_to_mm,
        }
        with (folder / "metadata.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file, sort_keys=False, allow_unicode=True)

        csv_path = self.output_dir / "measurements.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        profile_fields = [
            "h_low", "h_high", "s_low", "s_high", "v_low", "v_high",
            "depth_min_mm", "depth_max_mm", "blur_size", "morph_size",
            "min_area", "ball_circularity_min",
        ]
        metric_fields = [
            "frame_v_p10", "frame_v_p50", "frame_v_p90",
            "underexposed_pct", "overexposed_pct", "candidate_count",
            "accepted_count", "largest_area", "best_circularity",
            "best_depth_mm",
        ]
        roi_fields = [
            "count", "h_fit_low", "h_fit_high", "s_p02", "s_p50",
            "s_p99", "v_p02", "v_p50", "v_p99",
        ]
        fieldnames = (
            ["timestamp", "target", "environment_label"]
            + profile_fields
            + metric_fields
            + [f"roi_{name}" for name in roi_fields]
            + ["snapshot_dir"]
        )
        row = {
            "timestamp": stamp,
            "target": self.target,
            "environment_label": self.environment_label,
            **{name: profile.get(name, "") for name in profile_fields},
            **{name: self.last_metrics.get(name, "") for name in metric_fields},
            **{
                f"roi_{name}": self.last_roi_stats.get(name, "")
                for name in roi_fields
            },
            "snapshot_dir": str(folder),
        }
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.get_logger().info(f"Saved snapshot -> {folder}")

    # ------------------------------------------------------------- Callback --
    def image_callback(self, color_msg: Image, depth_msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                color_msg,
                desired_encoding="bgr8",
            )
            depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except CvBridgeError as exc:
            self.get_logger().error(f"CvBridge conversion failed: {exc}")
            return

        depth_mm = np.asarray(depth_raw, dtype=np.float32) * self.depth_scale_to_mm
        depth_mm[~np.isfinite(depth_mm)] = 0.0

        if depth_mm.shape[:2] != frame.shape[:2]:
            if not self.warned_shape:
                self.get_logger().error(
                    "Color/depth image sizes differ. Use aligned_depth_to_color and "
                    "matching profiles; frame is skipped to avoid wrong pixel fusion."
                )
                self.warned_shape = True
            return

        now = time.monotonic()
        dt = max(now - self.last_frame_time, 1e-6)
        instant_fps = 1.0 / dt
        self.fps_ema = instant_fps if self.fps_ema <= 0 else 0.9 * self.fps_ema + 0.1 * instant_fps
        self.last_frame_time = now

        profile = self._read_controls()
        if self.view_mode == "detection" and self.target == "ball":
            overlay, outputs, preview_metrics = (
                self._draw_ball_detection_preview(frame, depth_mm, profile)
            )
            metrics = {
                **self._frame_metrics(outputs["hsv"]),
                **preview_metrics,
            }
        else:
            outputs = self._process_masks(frame, depth_mm, profile)
            draw_preview = self.view_mode == "detection"
            overlay, candidate_metrics = self._draw_candidates(
                frame,
                outputs["combined_mask"],
                depth_mm,
                profile,
                draw_preview=draw_preview,
            )
            metrics = {
                **self._frame_metrics(outputs["hsv"]),
                **candidate_metrics,
            }
            if draw_preview:
                metrics["preview_state"] = (
                    "DETECTED"
                    if candidate_metrics["accepted_count"] > 0
                    else "MISS"
                )

        self.latest_frame = frame.copy()
        self.latest_hsv = outputs["hsv"].copy()
        self.latest_depth_mm = depth_mm.copy()
        self.last_outputs = outputs
        self.last_metrics = metrics

        # Refresh ROI statistics every frame while keeping the same ROI.
        if self.view_mode == "calibration" and self.roi is not None:
            self._refresh_roi_pixels()

        if self.view_mode == "calibration":
            self._draw_status(overlay, profile, metrics)
        self.latest_overlay = overlay.copy()

        cv2.imshow(MAIN_WINDOW, overlay)
        cv2.imshow(MASK_WINDOW, self._make_grid(frame, outputs))
        cv2.imshow(CONTROL_WINDOW, self._make_control_guide(profile))

    def _handle_key(self, key: int) -> None:
        if key in (255,):
            return
        if key in (ord("q"), 27):
            if rclpy.ok():
                rclpy.shutdown()
        elif key == ord("b"):
            self._switch_target("ball")
        elif key == ord("k"):
            self._switch_target("support")
        elif key == ord("f"):
            self._switch_target("floor")
        elif key == ord("g"):
            self._switch_target("hoop")
        elif key == ord("h"):
            self.get_logger().warning(
                "IRC hurdle detection currently uses webcam YOLO, not "
                "RealSense HSV; this calibrator cannot validate it."
            )
        elif key == ord("d"):
            self._toggle_detection_preview()
        elif key == ord("r"):
            self._restore_pre_fit()
        elif key == ord("s"):
            self._save_profiles()
        elif key == ord("l"):
            self._reload_profiles()
        elif key == ord("i"):
            self._save_snapshot()
        elif self.view_mode == "detection":
            self.get_logger().warning(
                "Detection preview is read-only; press D to return first."
            )
        elif key == ord("n"):
            self._new_tuning_session()
        elif key == 32:  # SPACE
            self._add_roi_to_bank()
        elif key == ord("a"):
            self._auto_fit()
        elif key == ord("x"):
            self._clear_bank()

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = HSVCalibratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
