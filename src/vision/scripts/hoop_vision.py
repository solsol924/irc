#!/usr/bin/env python3
"""
RealSense OpenCV Hoop 비전 노드

역할
1. RealSense color + aligned depth 영상을 시간 동기화해 받는다.
2. HSV 색공간에서 빨간 테두리와 흰색 내부를 분리한다.
3. 빨간 컨투어의 회전 사각형을 기준으로 위/왼쪽/오른쪽 테두리 비율을 검사한다.
4. 내부 흰색 비율과 depth 유효성을 함께 검사해 골대 후보를 확정한다.
5. 백보드 중심까지의 3차원 거리와 로봇 중심선 기준 좌우 오차각을 계산한다.
6. JSON 상태와 디버그 이미지를 ROS 2 토픽으로 발행한다.

입력
- /camera/color/image_raw
- /camera/aligned_depth_to_color/image_raw
- /camera/color/camera_info
- /vision/hoop_active (선택, Bool)

출력
- /hoop/vision_state        (std_msgs/String, JSON)
- /hoop/detected            (std_msgs/Bool)
- /hoop/debug_image         (sensor_msgs/Image)

주의
- 반드시 aligned_depth_to_color 토픽을 사용해야 컬러 픽셀과 depth 픽셀이 일치한다.
- HSV 및 면적/비율 기준은 실제 경기장 조명과 골대 크기에 맞게 조정해야 한다.
- 이 파일은 검출값만 발행한다. 접근/좌우이동/슛 등의 모션 판단은 Decision 노드에서 한다.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String


class HoopVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("hoop_vision")

        source_config = (
            Path(__file__).resolve().parent.parent / "config" / "hoop_hsv.yaml"
        )
        default_hsv_config = (
            source_config
            if source_config.exists()
            else Path.home()
            / "irc"
            / "src"
            / "vision"
            / "config"
            / "hoop_hsv.yaml"
        )
        self.declare_parameter("hsv_config_file", str(default_hsv_config))
        self.hsv_config_path = Path(
            str(self.get_parameter("hsv_config_file").value)
        ).expanduser()
        hsv_defaults = self._load_hsv_defaults(self.hsv_config_path)

        # =========================================================
        # ROS 토픽
        # =========================================================
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("active_topic", "/vision/hoop_active")
        self.declare_parameter("state_topic", "/hoop/vision_state")
        self.declare_parameter("detected_topic", "/hoop/detected")
        self.declare_parameter("debug_image_topic", "/hoop/debug_image")

        # 통합 실행에서는 ball_vision_fusion이 현재 모드를 transient-local로
        # 전달한다. 단독 hoop 테스트만 파라미터로 True를 지정한다.
        self.declare_parameter("active_on_start", False)

        # =========================================================
        # ROI: 기본값은 카메라 화면 전체를 사용한다.
        # =========================================================
        self.declare_parameter("roi_left_ratio", 0.0)
        self.declare_parameter("roi_right_ratio", 1.0)
        self.declare_parameter("roi_top_ratio", 0.0)
        self.declare_parameter("roi_bottom_ratio", 1.0)

        # =========================================================
        # HSV 기준
        # OpenCV H 범위는 0~179이며 빨강이 0과 179 양 끝에 걸쳐 있다.
        # =========================================================
        self.declare_parameter("red_h1_low", hsv_defaults["red_h1_low"])
        self.declare_parameter("red_h1_high", hsv_defaults["red_h1_high"])
        self.declare_parameter("red_h2_low", hsv_defaults["red_h2_low"])
        self.declare_parameter("red_h2_high", hsv_defaults["red_h2_high"])
        self.declare_parameter("red_s_low", hsv_defaults["red_s_low"])
        self.declare_parameter("red_v_low", hsv_defaults["red_v_low"])

        self.declare_parameter("white_s_high", hsv_defaults["white_s_high"])
        self.declare_parameter("white_v_low", hsv_defaults["white_v_low"])

        # =========================================================
        # 후보 형상 및 색 비율 조건
        # =========================================================
        self.declare_parameter("min_contour_area", 200.0)
        self.declare_parameter("min_backboard_aspect_ratio", 1.05)
        self.declare_parameter("max_backboard_aspect_ratio", 6.0)

        self.declare_parameter("top_band_ratio", 0.15)
        self.declare_parameter("side_band_ratio", 0.10)
        self.declare_parameter("side_vertical_end_ratio", 0.75)

        self.declare_parameter("red_ratio_min", 0.55)
        self.declare_parameter("white_inner_ratio_min", 0.50)

        # 일부가 가려져 빨간 테두리가 끊겨도 작은 간격은 후보 생성 단계에서
        # 다시 연결한다. 최종 색 비율은 연결 전 원본 마스크로 검사하므로,
        # 이 값이 곧바로 빨간 픽셀 증거를 부풀리지는 않는다.
        self.declare_parameter("occlusion_merge_gap_px", 41)
        # 위/왼쪽/오른쪽 테두리 중 하나가 가려져도 나머지 두 구간과 전체
        # 빨간 비율이 충분하면 백보드로 인정한다.
        self.declare_parameter("min_visible_red_bands", 2)
        self.declare_parameter("red_band_average_min", 0.40)

        # =========================================================
        # Depth 조건
        # 16UC1 depth가 mm인 환경을 기준으로 한다.
        # =========================================================
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("depth_min_m", 0.08)
        self.declare_parameter("depth_max_m", 2.0)
        self.declare_parameter("min_valid_depth_pixels", 20)
        # 백보드 중심 픽셀에 depth hole이 생겨도 주변의 작은 영역에서
        # 안정적으로 중앙값을 구할 수 있도록 한다.
        self.declare_parameter("center_depth_patch_radius", 5)
        self.declare_parameter("min_valid_center_depth_pixels", 5)

        # RealSense 화면의 정확한 하단 중앙을 로봇 기준점으로 사용한다.

        # =========================================================
        # 안정화 및 출력
        # =========================================================
        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("hold_seconds", 0.5)
        self.declare_parameter("smoothing_window", 5)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("show_window", False)
        self.declare_parameter("print_every_n_frames", 10)

        # CameraInfo를 받기 전 임시값
        self.declare_parameter("fallback_fx", 607.0)
        self.declare_parameter("fallback_fy", 606.0)
        self.declare_parameter("fallback_cx", 325.5)
        self.declare_parameter("fallback_cy", 239.4)

        # =========================================================
        # 파라미터 로드
        # =========================================================
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.active_topic = str(self.get_parameter("active_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.detected_topic = str(self.get_parameter("detected_topic").value)
        self.debug_image_topic = str(
            self.get_parameter("debug_image_topic").value
        )

        self.active = bool(self.get_parameter("active_on_start").value)

        self.roi_left_ratio = float(self.get_parameter("roi_left_ratio").value)
        self.roi_right_ratio = float(
            self.get_parameter("roi_right_ratio").value
        )
        self.roi_top_ratio = float(self.get_parameter("roi_top_ratio").value)
        self.roi_bottom_ratio = float(
            self.get_parameter("roi_bottom_ratio").value
        )

        self.red_h1_low = int(self.get_parameter("red_h1_low").value)
        self.red_h1_high = int(self.get_parameter("red_h1_high").value)
        self.red_h2_low = int(self.get_parameter("red_h2_low").value)
        self.red_h2_high = int(self.get_parameter("red_h2_high").value)
        self.red_s_low = int(self.get_parameter("red_s_low").value)
        self.red_v_low = int(self.get_parameter("red_v_low").value)
        self.white_s_high = int(self.get_parameter("white_s_high").value)
        self.white_v_low = int(self.get_parameter("white_v_low").value)

        self.min_contour_area = float(
            self.get_parameter("min_contour_area").value
        )
        self.min_backboard_aspect_ratio = float(
            self.get_parameter("min_backboard_aspect_ratio").value
        )
        self.max_backboard_aspect_ratio = float(
            self.get_parameter("max_backboard_aspect_ratio").value
        )
        self.top_band_ratio = float(
            self.get_parameter("top_band_ratio").value
        )
        self.side_band_ratio = float(
            self.get_parameter("side_band_ratio").value
        )
        self.side_vertical_end_ratio = float(
            self.get_parameter("side_vertical_end_ratio").value
        )
        self.red_ratio_min = float(self.get_parameter("red_ratio_min").value)
        self.white_inner_ratio_min = float(
            self.get_parameter("white_inner_ratio_min").value
        )
        self.occlusion_merge_gap_px = max(
            0,
            int(self.get_parameter("occlusion_merge_gap_px").value),
        )
        self.min_visible_red_bands = max(
            1,
            min(3, int(self.get_parameter("min_visible_red_bands").value)),
        )
        self.red_band_average_min = float(
            self.get_parameter("red_band_average_min").value
        )

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.min_valid_depth_pixels = int(
            self.get_parameter("min_valid_depth_pixels").value
        )
        self.center_depth_patch_radius = max(
            0,
            int(self.get_parameter("center_depth_patch_radius").value),
        )
        self.min_valid_center_depth_pixels = max(
            1,
            int(self.get_parameter("min_valid_center_depth_pixels").value),
        )

        self.hold_seconds = max(
            0.0,
            float(self.get_parameter("hold_seconds").value),
        )
        self.smoothing_window = max(
            1, int(self.get_parameter("smoothing_window").value)
        )
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        self.show_window = bool(self.get_parameter("show_window").value)
        self.print_every_n_frames = max(
            1, int(self.get_parameter("print_every_n_frames").value)
        )

        self.fx = float(self.get_parameter("fallback_fx").value)
        self.fy = float(self.get_parameter("fallback_fy").value)
        self.cx_intr = float(self.get_parameter("fallback_cx").value)
        self.cy_intr = float(self.get_parameter("fallback_cy").value)
        self.camera_info_received = False

        self._rebuild_kernel()
        self.add_on_set_parameters_callback(self.parameter_callback)

        # =========================================================
        # 상태
        # =========================================================
        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_detection: Optional[Dict[str, Any]] = None
        self.last_detection_time: Optional[float] = None
        self.history: Deque[Dict[str, Any]] = deque(
            maxlen=self.smoothing_window
        )
        self.image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # =========================================================
        # ROS I/O
        # =========================================================
        # RealSense 구독과 synchronizer는 프로세스 수명 동안 유지한다.
        # 모드 전환 때 구독을 재생성하지 않아 다음 카메라 프레임부터
        # 즉시 hoop 처리로 넘어갈 수 있게 한다.
        self.color_sub = None
        self.depth_sub = None
        self.sync = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )
        active_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.active_sub = self.create_subscription(
            Bool,
            self.active_topic,
            self.active_callback,
            active_qos,
        )

        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.detected_pub = self.create_publisher(
            Bool, self.detected_topic, 10
        )
        self.debug_pub = self.create_publisher(
            Image, self.debug_image_topic, self.image_qos
        )

        self._start_image_subscriptions()

        self.get_logger().info("HoopVisionNode started.")
        self.get_logger().info(f"Color topic: {self.color_topic}")
        self.get_logger().info(f"Aligned depth topic: {self.depth_topic}")
        self.get_logger().info(f"State output: {self.state_topic}")

    def _load_hsv_defaults(self, path: Path) -> Dict[str, int]:
        fallback = {
            "red_h1_low": 0,
            "red_h1_high": 10,
            "red_h2_low": 160,
            "red_h2_high": 179,
            "red_s_low": 80,
            "red_v_low": 60,
            "white_s_high": 80,
            "white_v_low": 80,
        }
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}
            loaded = payload.get("hoop_vision", {}).get("ros__parameters", {})
            values = {
                name: int(loaded.get(name, default))
                for name, default in fallback.items()
            }
            if not (
                0 <= values["red_h1_low"] <= values["red_h1_high"] <= 179
                and 0 <= values["red_h2_low"] <= values["red_h2_high"] <= 179
                and 0 <= values["red_s_low"] <= 255
                and 0 <= values["red_v_low"] <= 255
                and 0 <= values["white_s_high"] <= 255
                and 0 <= values["white_v_low"] <= 255
            ):
                raise ValueError("HSV value outside the OpenCV range")
            self.get_logger().info(
                "Loaded hoop red/white calibration from "
                f"{path}: red H={values['red_h1_low']}.."
                f"{values['red_h1_high']} + {values['red_h2_low']}.."
                f"{values['red_h2_high']}, S>={values['red_s_low']}, "
                f"V>={values['red_v_low']}; white "
                f"S<={values['white_s_high']}, V>={values['white_v_low']}"
            )
            return values
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            self.get_logger().warning(
                f"Could not load hoop HSV calibration {path}: {exc}; "
                "using built-in defaults"
            )
            return fallback

    # =============================================================
    # 파라미터
    # =============================================================
    def _rebuild_kernel(self) -> None:
        size = max(1, int(self.get_parameter("morph_kernel_size").value))
        if size % 2 == 0:
            size += 1
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size),
        )

    def parameter_callback(self, params) -> SetParametersResult:
        int_names = {
            "red_h1_low",
            "red_h1_high",
            "red_h2_low",
            "red_h2_high",
            "red_s_low",
            "red_v_low",
            "white_s_high",
            "white_v_low",
            "min_valid_depth_pixels",
            "center_depth_patch_radius",
            "min_valid_center_depth_pixels",
            "occlusion_merge_gap_px",
            "min_visible_red_bands",
            "smoothing_window",
            "print_every_n_frames",
        }
        float_names = {
            "roi_left_ratio",
            "roi_right_ratio",
            "roi_top_ratio",
            "roi_bottom_ratio",
            "min_contour_area",
            "min_backboard_aspect_ratio",
            "max_backboard_aspect_ratio",
            "top_band_ratio",
            "side_band_ratio",
            "side_vertical_end_ratio",
            "red_ratio_min",
            "red_band_average_min",
            "white_inner_ratio_min",
            "depth_scale",
            "depth_min_m",
            "depth_max_m",
            "hold_seconds",
        }

        try:
            for param in params:
                if param.name in int_names:
                    setattr(self, param.name, int(param.value))
                elif param.name in float_names:
                    setattr(self, param.name, float(param.value))
                elif param.name == "publish_debug_image":
                    self.publish_debug_image = bool(param.value)
                elif param.name == "show_window":
                    self.show_window = bool(param.value)
                elif param.name == "active_on_start":
                    self.active = bool(param.value)
                elif param.name == "morph_kernel_size":
                    size = max(1, int(param.value))
                    if size % 2 == 0:
                        size += 1
                    self.kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (size, size),
                    )
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        if not (
            0 <= self.red_h1_low <= self.red_h1_high <= 179
            and 0 <= self.red_h2_low <= self.red_h2_high <= 179
            and 0 <= self.red_s_low <= 255
            and 0 <= self.red_v_low <= 255
            and 0 <= self.white_s_high <= 255
            and 0 <= self.white_v_low <= 255
        ):
            return SetParametersResult(
                successful=False,
                reason="Invalid HSV parameter range",
            )

        if self.depth_min_m >= self.depth_max_m:
            return SetParametersResult(
                successful=False,
                reason="depth_min_m must be smaller than depth_max_m",
            )

        self.center_depth_patch_radius = max(
            0,
            int(self.center_depth_patch_radius),
        )
        self.min_valid_center_depth_pixels = max(
            1,
            int(self.min_valid_center_depth_pixels),
        )
        self.occlusion_merge_gap_px = max(
            0,
            int(self.occlusion_merge_gap_px),
        )
        self.min_visible_red_bands = max(
            1,
            min(3, int(self.min_visible_red_bands)),
        )

        if not (0.0 <= self.red_band_average_min <= 1.0):
            return SetParametersResult(
                successful=False,
                reason="red_band_average_min must be between 0 and 1",
            )

        # smoothing_window 변경 시 deque 크기도 갱신한다.
        new_window = max(1, int(self.smoothing_window))
        if self.history.maxlen != new_window:
            self.history = deque(list(self.history)[-new_window:], maxlen=new_window)

        self.hold_seconds = max(0.0, float(self.hold_seconds))
        self.print_every_n_frames = max(1, int(self.print_every_n_frames))
        return SetParametersResult(successful=True)

    # =============================================================
    # ROS 콜백
    # =============================================================
    def _start_image_subscriptions(self) -> None:
        if self.color_sub is not None or self.depth_sub is not None:
            return

        self.color_sub = Subscriber(
            self,
            Image,
            self.color_topic,
            qos_profile=self.image_qos,
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=self.image_qos,
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=2,
            slop=0.1,
        )
        self.sync.registerCallback(self.image_callback)

    def active_callback(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested == self.active:
            return

        # 구독을 끊지 않고 처리 플래그만 바꾼다. 이 방식은 DDS discovery와
        # synchronizer 재충전으로 생기던 전환 공백을 피한다.
        self.active = requested
        self.history.clear()
        self.last_detection = None
        self.last_detection_time = None
        if not requested:
            self._publish_state(
                detection=None,
                process_ms=0.0,
                stamp_sec=time.monotonic(),
            )

        self.get_logger().info(
            f"Hoop image processing switched {'ON' if requested else 'OFF'}"
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
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
                "CameraInfo received: "
                f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
            )

    def image_callback(self, color_msg: Image, depth_msg: Image) -> None:
        if not self.active:
            return

        start_time = time.perf_counter()

        try:
            frame = self.bridge.imgmsg_to_cv2(
                color_msg,
                desired_encoding="bgr8",
            )
            depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}")
            return

        depth = np.asarray(depth_raw, dtype=np.float32)
        if frame is None or depth.ndim != 2:
            return

        frame_h, frame_w = frame.shape[:2]
        if depth.shape != (frame_h, frame_w):
            self.get_logger().warn(
                "Color/depth size mismatch. Use aligned_depth_to_color."
            )
            return

        x1, y1, x2, y2 = self._get_roi(frame_w, frame_h)
        roi_color = frame[y1:y2, x1:x2]
        roi_depth = depth[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(
            hsv,
            (self.red_h1_low, self.red_s_low, self.red_v_low),
            (self.red_h1_high, 255, 255),
        )
        red_mask2 = cv2.inRange(
            hsv,
            (self.red_h2_low, self.red_s_low, self.red_v_low),
            (self.red_h2_high, 255, 255),
        )
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        white_mask = cv2.inRange(
            hsv,
            (0, 0, self.white_v_low),
            (179, self.white_s_high, 255),
        )

        depth_m = roi_depth * self.depth_scale
        invalid_depth = (
            ~np.isfinite(depth_m)
            | (depth_m < self.depth_min_m)
            | (depth_m > self.depth_max_m)
        )
        # 빨간 후보는 거리 범위로 제한하지만, 흰 내부 마스크는 depth hole 때문에
        # 색상 픽셀을 지우지 않는다. 내부 depth 유효성은 후보 확정 단계에서 별도로 검사한다.
        red_mask[invalid_depth] = 0

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            self.kernel,
        )
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            self.kernel,
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
        )

        raw_detection = self._find_best_hoop(
            red_mask=red_mask,
            white_mask=white_mask,
            roi_depth_m=depth_m,
            roi_x_start=x1,
            roi_y_start=y1,
            frame_width=frame_w,
            frame_height=frame_h,
        )

        held_previous = False
        if raw_detection is not None:
            self.history.append(raw_detection)
            smoothed = self._smooth_detection(raw_detection)
            self.last_detection = dict(smoothed)
            self.last_detection_time = start_time
            published_detection = smoothed
        elif (
            self.last_detection is not None
            and self._hold_is_active(
                last_detection_time=self.last_detection_time,
                current_time=start_time,
                hold_seconds=self.hold_seconds,
            )
        ):
            held_previous = True
            published_detection = dict(self.last_detection)
            published_detection["held_previous_detection"] = True
            published_detection["raw_detected"] = False
        else:
            self.history.clear()
            self.last_detection = None
            self.last_detection_time = None
            published_detection = None

        process_ms = (time.perf_counter() - start_time) * 1000.0
        self._publish_state(
            detection=published_detection,
            process_ms=process_ms,
            stamp_sec=(color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9),
        )

        if self.publish_debug_image or self.show_window:
            debug = self._draw_debug(
                frame=frame,
                red_mask=red_mask,
                white_mask=white_mask,
                detection=published_detection,
                raw_detected=raw_detection is not None,
                held_previous=held_previous,
                roi=(x1, y1, x2, y2),
                process_ms=process_ms,
            )

            if self.publish_debug_image:
                debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
                debug_msg.header = color_msg.header
                self.debug_pub.publish(debug_msg)

            if self.show_window:
                cv2.imshow("Hoop Detection", debug)
                cv2.waitKey(1)

        self.frame_count += 1
        if self.frame_count % self.print_every_n_frames == 0:
            if published_detection is None:
                self.get_logger().info(
                    f"hoop miss | process={process_ms:.1f} ms"
                )
            else:
                self.get_logger().info(
                    "hoop detected "
                    f"dist={published_detection['realsense_goal_distance_cm']:.1f} cm "
                    f"center_ang={published_detection['realsense_goal_angle']:+.1f} deg "
                    f"held={published_detection.get('held_previous_detection', False)} "
                    f"process={process_ms:.1f} ms"
                )

    # =============================================================
    # 검출
    # =============================================================
    def _get_roi(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        x1 = int(frame_w * self.roi_left_ratio)
        x2 = int(frame_w * self.roi_right_ratio)
        y1 = int(frame_h * self.roi_top_ratio)
        y2 = int(frame_h * self.roi_bottom_ratio)

        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(y1 + 1, min(y2, frame_h))
        return x1, y1, x2, y2

    @staticmethod
    def _order_quad(points: np.ndarray) -> np.ndarray:
        """네 꼭지점을 [좌상, 우상, 우하, 좌하] 순서로 정렬한다."""
        pts = np.asarray(points, dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)

        tl = pts[np.argmin(sums)]
        br = pts[np.argmax(sums)]
        tr = pts[np.argmin(diffs)]
        bl = pts[np.argmax(diffs)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _fill_polygon_mask(
        shape: Tuple[int, int],
        polygon: np.ndarray,
    ) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        poly = np.round(polygon).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_8)
        return mask

    @staticmethod
    def _masked_ratio(source_mask: np.ndarray, region_mask: np.ndarray) -> float:
        area = int(cv2.countNonZero(region_mask))
        if area <= 0:
            return 0.0
        hits = int(cv2.countNonZero(cv2.bitwise_and(source_mask, region_mask)))
        return float(hits) / float(area)

    @staticmethod
    def _rectangle_center(box: np.ndarray) -> np.ndarray:
        """직사각형의 두 대각선이 만나는 중심점을 반환한다."""
        points = np.asarray(box, dtype=np.float32)
        return 0.5 * (points[0] + points[2])

    @staticmethod
    def _robot_reference_point(
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        """화면 최하단에서 로봇 중심선의 기준점을 반환한다."""
        return (
            float(max(1, frame_width)) / 2.0,
            float(max(1, frame_height) - 1),
        )

    @staticmethod
    def _centerline_error_angle_deg(
        center_x: float,
        center_y: float,
        robot_x: float,
        robot_y: float,
    ) -> Optional[float]:
        """공 각도와 같은 화면 기하식으로 백보드 중심의 좌우 각도를 계산한다."""
        x_distance = center_x - robot_x
        y_distance = abs(robot_y - center_y)
        if y_distance <= 0.0:
            return None
        return math.degrees(math.atan2(x_distance, y_distance))

    @staticmethod
    def _center_pixel_offsets(
        center_x: float,
        center_y: float,
        robot_x: float,
        robot_y: float,
    ) -> Tuple[float, float]:
        """로봇 하단 중심 기준 백보드 중심의 수평/수직 픽셀 차이를 반환한다."""
        return center_x - robot_x, robot_y - center_y

    @staticmethod
    def _hold_is_active(
        last_detection_time: Optional[float],
        current_time: float,
        hold_seconds: float,
    ) -> bool:
        """마지막 검출 후 설정 시간 안이면 이전 검출을 유지한다."""
        if last_detection_time is None:
            return False
        return (current_time - last_detection_time) <= max(0.0, hold_seconds)

    def _center_depth_m(
        self,
        roi_depth_m: np.ndarray,
        center_x: float,
        center_y: float,
    ) -> Optional[float]:
        """백보드 중심 주변의 유효 depth 중앙값을 반환한다."""
        height, width = roi_depth_m.shape[:2]
        cx = int(round(center_x))
        cy = int(round(center_y))
        radius = self.center_depth_patch_radius
        x1 = max(0, cx - radius)
        x2 = min(width, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(height, cy + radius + 1)
        patch = roi_depth_m[y1:y2, x1:x2]
        valid = patch[
            np.isfinite(patch)
            & (patch >= self.depth_min_m)
            & (patch <= self.depth_max_m)
        ]
        if valid.size < self.min_valid_center_depth_pixels:
            return None
        return float(np.median(valid))

    def _center_distance_m(
        self,
        center_x: float,
        center_y: float,
        depth_m: float,
    ) -> float:
        """카메라 원점에서 백보드 중심까지의 3차원 직선거리를 계산한다."""
        x_m = (center_x - self.cx_intr) * depth_m / self.fx
        y_m = (center_y - self.cy_intr) * depth_m / self.fy
        return math.sqrt(x_m * x_m + y_m * y_m + depth_m * depth_m)

    @staticmethod
    def _build_occlusion_tolerant_candidate_mask(
        red_mask: np.ndarray,
        merge_gap_px: int,
    ) -> np.ndarray:
        """가림으로 끊긴 수평/수직 테두리 조각을 후보 생성용으로 연결한다."""
        gap = max(0, int(merge_gap_px))
        if gap <= 1:
            return red_mask.copy()
        if gap % 2 == 0:
            gap += 1

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (gap, 1),
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, gap),
        )
        horizontal = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            horizontal_kernel,
        )
        vertical = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            vertical_kernel,
        )
        return cv2.bitwise_or(red_mask, cv2.bitwise_or(horizontal, vertical))

    @staticmethod
    def _red_band_evidence_passes(
        band_ratios: Tuple[float, float, float],
        red_ratio_min: float,
        min_visible_red_bands: int,
        red_band_average_min: float,
    ) -> Tuple[bool, int, float]:
        """부분 가림을 허용하면서 충분한 빨간 테두리 증거가 있는지 검사한다."""
        visible_count = sum(
            ratio >= red_ratio_min for ratio in band_ratios
        )
        average_ratio = float(sum(band_ratios)) / float(len(band_ratios))
        required_count = max(1, min(len(band_ratios), min_visible_red_bands))
        passed = (
            visible_count >= required_count
            and average_ratio >= red_band_average_min
        )
        return passed, visible_count, average_ratio

    def _find_best_hoop(
        self,
        red_mask: np.ndarray,
        white_mask: np.ndarray,
        roi_depth_m: np.ndarray,
        roi_x_start: int,
        roi_y_start: int,
        frame_width: int,
        frame_height: int,
    ) -> Optional[Dict[str, Any]]:
        candidate_mask = self._build_occlusion_tolerant_candidate_mask(
            red_mask,
            self.occlusion_merge_gap_px,
        )
        contours, _ = cv2.findContours(
            candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        roi_h, roi_w = red_mask.shape[:2]

        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < self.min_contour_area:
                continue

            rect = cv2.minAreaRect(contour)
            raw_box = cv2.boxPoints(rect)
            box = self._order_quad(raw_box)
            tl, tr, br, bl = box

            width = 0.5 * (
                float(np.linalg.norm(tr - tl))
                + float(np.linalg.norm(br - bl))
            )
            height = 0.5 * (
                float(np.linalg.norm(bl - tl))
                + float(np.linalg.norm(br - tr))
            )
            if width < 2.0 or height < 2.0:
                continue

            aspect_ratio = width / height
            if not (
                self.min_backboard_aspect_ratio
                <= aspect_ratio
                <= self.max_backboard_aspect_ratio
            ):
                continue

            canonical_w = max(2.0, width)
            canonical_h = max(2.0, height)
            canonical = np.array(
                [
                    [0.0, 0.0],
                    [canonical_w, 0.0],
                    [canonical_w, canonical_h],
                    [0.0, canonical_h],
                ],
                dtype=np.float32,
            )
            inverse_h = cv2.getPerspectiveTransform(canonical, box)

            top_h = max(1.0, canonical_h * self.top_band_ratio)
            side_w = max(1.0, canonical_w * self.side_band_ratio)
            side_y_end = max(
                top_h + 1.0,
                canonical_h * self.side_vertical_end_ratio,
            )
            side_y_end = min(side_y_end, canonical_h)

            top_poly_c = np.array(
                [[0, 0], [canonical_w, 0], [canonical_w, top_h], [0, top_h]],
                dtype=np.float32,
            )
            left_poly_c = np.array(
                [[0, top_h], [side_w, top_h], [side_w, side_y_end], [0, side_y_end]],
                dtype=np.float32,
            )
            right_poly_c = np.array(
                [
                    [canonical_w - side_w, top_h],
                    [canonical_w, top_h],
                    [canonical_w, side_y_end],
                    [canonical_w - side_w, side_y_end],
                ],
                dtype=np.float32,
            )
            inner_poly_c = np.array(
                [
                    [side_w, top_h],
                    [canonical_w - side_w, top_h],
                    [canonical_w - side_w, side_y_end],
                    [side_w, side_y_end],
                ],
                dtype=np.float32,
            )

            def transform(poly: np.ndarray) -> np.ndarray:
                return cv2.perspectiveTransform(
                    poly.reshape(-1, 1, 2),
                    inverse_h,
                ).reshape(-1, 2)

            top_poly = transform(top_poly_c)
            left_poly = transform(left_poly_c)
            right_poly = transform(right_poly_c)
            inner_poly = transform(inner_poly_c)

            top_mask = self._fill_polygon_mask((roi_h, roi_w), top_poly)
            left_mask = self._fill_polygon_mask((roi_h, roi_w), left_poly)
            right_mask = self._fill_polygon_mask((roi_h, roi_w), right_poly)
            inner_mask = self._fill_polygon_mask((roi_h, roi_w), inner_poly)

            top_red_ratio = self._masked_ratio(red_mask, top_mask)
            left_red_ratio = self._masked_ratio(red_mask, left_mask)
            right_red_ratio = self._masked_ratio(red_mask, right_mask)
            white_inner_ratio = self._masked_ratio(white_mask, inner_mask)

            red_bands_pass, visible_red_bands, red_band_ratio = (
                self._red_band_evidence_passes(
                    (
                        top_red_ratio,
                        left_red_ratio,
                        right_red_ratio,
                    ),
                    self.red_ratio_min,
                    self.min_visible_red_bands,
                    self.red_band_average_min,
                )
            )
            if not red_bands_pass or white_inner_ratio < self.white_inner_ratio_min:
                continue

            inner_depth = roi_depth_m[inner_mask.astype(bool)]
            valid_inner_depth = inner_depth[
                np.isfinite(inner_depth)
                & (inner_depth >= self.depth_min_m)
                & (inner_depth <= self.depth_max_m)
            ]
            if valid_inner_depth.size < self.min_valid_depth_pixels:
                continue

            # 회전 직사각형의 대각선 교점이 백보드 중심이다.
            center_roi = self._rectangle_center(box)
            center_x = float(center_roi[0] + roi_x_start)
            center_y = float(center_roi[1] + roi_y_start)

            center_depth_m = self._center_depth_m(
                roi_depth_m,
                float(center_roi[0]),
                float(center_roi[1]),
            )
            if center_depth_m is None:
                continue
            distance_m = self._center_distance_m(
                center_x,
                center_y,
                center_depth_m,
            )

            robot_x, robot_y = self._robot_reference_point(
                frame_width,
                frame_height,
            )
            realsense_goal_angle = self._centerline_error_angle_deg(
                center_x,
                center_y,
                robot_x,
                robot_y,
            )
            if realsense_goal_angle is None:
                continue
            goal_center_dx_px, goal_center_dy_px = self._center_pixel_offsets(
                center_x,
                center_y,
                robot_x,
                robot_y,
            )

            score = red_band_ratio + 0.5 * white_inner_ratio

            if score <= best_score:
                continue

            full_box = box + np.array(
                [roi_x_start, roi_y_start],
                dtype=np.float32,
            )
            best_score = score
            best = {
                "detected": True,
                "raw_detected": True,
                "held_previous_detection": False,
                "center_x": center_x,
                "center_y": center_y,
                "realsense_goal_distance_cm": distance_m * 100.0,
                "realsense_goal_angle": realsense_goal_angle,
                "goal_center_dx_px": goal_center_dx_px,
                "goal_center_dy_px": goal_center_dy_px,
                "center_depth_cm": center_depth_m * 100.0,
                "robot_center_x": float(robot_x),
                "robot_bottom_y": float(robot_y),
                "top_red_ratio": top_red_ratio,
                "left_red_ratio": left_red_ratio,
                "right_red_ratio": right_red_ratio,
                "white_inner_ratio": white_inner_ratio,
                "red_band_ratio": red_band_ratio,
                "visible_red_bands": visible_red_bands,
                "occlusion_tolerant": visible_red_bands < 3,
                "contour_area": contour_area,
                "aspect_ratio": aspect_ratio,
                "score": score,
                "box": full_box.astype(np.int32).tolist(),
            }

        return best

    def _smooth_detection(
        self,
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        """최근 검출값의 중앙값을 사용해 거리와 색상 비율 흔들림을 줄인다."""
        output = dict(current)
        numeric_keys = (
            "realsense_goal_distance_cm",
            "center_depth_cm",
            "top_red_ratio",
            "left_red_ratio",
            "right_red_ratio",
            "white_inner_ratio",
            "red_band_ratio",
            "score",
        )

        for key in numeric_keys:
            values = [
                float(item[key])
                for item in self.history
                if item.get(key) is not None
            ]
            if values:
                output[key] = float(np.median(values))

        # 디버그 박스는 현재 프레임의 실제 검출 박스를 사용한다.
        output["box"] = current["box"]
        return output

    # =============================================================
    # 출력
    # =============================================================
    def _publish_state(
        self,
        detection: Optional[Dict[str, Any]],
        process_ms: float,
        stamp_sec: float,
    ) -> None:
        if detection is None:
            output: Dict[str, Any] = {
                "detected": False,
                "raw_detected": False,
                "held_previous_detection": False,
                "center_x": None,
                "center_y": None,
                "realsense_goal_distance_cm": None,
                "realsense_goal_angle": None,
                "goal_center_dx_px": None,
                "goal_center_dy_px": None,
                "center_depth_cm": None,
                "robot_center_x": None,
                "robot_bottom_y": None,
                "top_red_ratio": None,
                "left_red_ratio": None,
                "right_red_ratio": None,
                "white_inner_ratio": None,
                "red_band_ratio": None,
                "visible_red_bands": 0,
                "occlusion_tolerant": False,
                "score": None,
            }
        else:
            output = {
                key: value
                for key, value in detection.items()
                if key != "box"
            }

        output.update(
            {
                "active": self.active,
                "camera_info_received": self.camera_info_received,
                "process_ms": float(process_ms),
                "stamp_sec": float(stamp_sec),
            }
        )

        self.state_pub.publish(
            String(data=json.dumps(output, ensure_ascii=False))
        )
        self.detected_pub.publish(Bool(data=bool(output["detected"])))

    def _draw_debug(
        self,
        frame: np.ndarray,
        red_mask: np.ndarray,
        white_mask: np.ndarray,
        detection: Optional[Dict[str, Any]],
        raw_detected: bool,
        held_previous: bool,
        roi: Tuple[int, int, int, int],
        process_ms: float,
    ) -> np.ndarray:
        debug = frame.copy()
        x1, y1, x2, y2 = roi

        pastel_yellow = (170, 235, 255)
        pastel_pink = (210, 190, 255)
        pastel_cyan = (245, 235, 180)
        white = (255, 255, 255)
        color = pastel_yellow if detection is not None else (170, 170, 255)

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        if detection is not None:
            box = np.asarray(detection.get("box", []), dtype=np.int32)
            if box.shape == (4, 2):
                cv2.polylines(debug, [box], True, color, 2)

            cx = int(round(float(detection["center_x"])))
            cy = int(round(float(detection["center_y"])))
            cv2.circle(debug, (cx, cy), 5, pastel_pink, -1)

            robot_x = int(round(float(detection["robot_center_x"])))
            robot_y = int(round(float(detection["robot_bottom_y"])))
            # 흰 선은 로봇의 정면 중심선, 하늘색 선은 로봇 기준점에서
            # 백보드 중심으로 향하는 선이다. 두 선이 만나는 각도가
            # realsense_goal_angle이며 오른쪽은 양수, 왼쪽은 음수이다.
            cv2.line(
                debug,
                (robot_x, 0),
                (robot_x, debug.shape[0] - 1),
                white,
                1,
            )
            cv2.line(
                debug,
                (robot_x, robot_y),
                (cx, cy),
                pastel_cyan,
                2,
            )
            cv2.circle(debug, (robot_x, robot_y), 5, white, -1)

            target_screen_angle = (
                math.degrees(math.atan2(cy - robot_y, cx - robot_x))
                + 360.0
            ) % 360.0
            vertical_screen_angle = 270.0
            arc_start = min(target_screen_angle, vertical_screen_angle)
            arc_end = max(target_screen_angle, vertical_screen_angle)
            arc_radius = max(20, min(debug.shape[:2]) // 10)
            cv2.ellipse(
                debug,
                (robot_x, robot_y),
                (arc_radius, arc_radius),
                0.0,
                arc_start,
                arc_end,
                pastel_cyan,
                2,
            )

            angle_value = float(detection["realsense_goal_angle"])
            angle_label_x = robot_x + 8 if angle_value >= 0.0 else robot_x - 76
            angle_label_y = max(16, robot_y - arc_radius - 6)
            cv2.putText(
                debug,
                f"{angle_value:+.1f}deg",
                (angle_label_x, angle_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                pastel_cyan,
                1,
                cv2.LINE_AA,
            )

            state_name = "DETECTED" if raw_detected else "HOLD"
            panel_lines = [
                f"HOOP:{state_name}",
                (
                    "realsense_goal_distance_cm:"
                    f"{detection['realsense_goal_distance_cm']:.1f}"
                ),
                f"realsense_goal_angle:{detection['realsense_goal_angle']:+.1f}deg",
                f"goal_center_dx_px:{detection['goal_center_dx_px']:+.1f}",
                f"goal_center_dy_px:{detection['goal_center_dy_px']:.1f}",
                (
                    "red_bands_visible:"
                    f"{int(detection.get('visible_red_bands', 3))}/3"
                ),
            ]
        else:
            panel_lines = [
                "HOOP:MISS",
                "realsense_goal_distance_cm:N/A",
                "realsense_goal_angle:N/A",
                "goal_center_dx_px:N/A",
                "goal_center_dy_px:N/A",
                "red_bands_visible:0/3",
            ]

        # 라인/공 화면과 같은 형태로 정보를 왼쪽 위의 작은 패널에 모은다.
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.36
        font_thickness = 1
        padding_x, padding_y = 6, 5
        line_gap = 4
        text_sizes = [
            cv2.getTextSize(
                text,
                font,
                font_scale,
                font_thickness,
            )[0]
            for text in panel_lines
        ]
        text_height = max(size[1] for size in text_sizes)
        line_height = text_height + line_gap
        panel_width = max(size[0] for size in text_sizes) + 2 * padding_x
        panel_height = (
            2 * padding_y
            + len(panel_lines) * text_height
            + (len(panel_lines) - 1) * line_gap
        )
        panel_x, panel_y = 4, 4
        panel_right = min(debug.shape[1] - 1, panel_x + panel_width)
        panel_bottom = min(debug.shape[0] - 1, panel_y + panel_height)

        cv2.rectangle(
            debug,
            (panel_x, panel_y),
            (panel_right, panel_bottom),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            debug,
            (panel_x, panel_y),
            (panel_right, panel_bottom),
            pastel_yellow,
            1,
        )

        text_x = panel_x + padding_x
        first_baseline_y = panel_y + padding_y + text_height
        for index, text in enumerate(panel_lines):
            cv2.putText(
                debug,
                text,
                (text_x, first_baseline_y + index * line_height),
                font,
                font_scale,
                pastel_yellow,
                font_thickness,
                cv2.LINE_AA,
            )

        return debug

    def destroy_node(self):
        if self.show_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HoopVisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
