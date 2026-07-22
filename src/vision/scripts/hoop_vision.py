#!/usr/bin/env python3
"""
RealSense OpenCV Hoop Vision Node

역할
1. RealSense color + aligned depth 영상을 시간 동기화해 받는다.
2. HSV 색공간에서 빨간 테두리와 흰색 내부를 분리한다.
3. 빨간 컨투어의 회전 사각형을 기준으로 위/왼쪽/오른쪽 테두리 비율을 검사한다.
4. 내부 흰색 비율과 depth 유효성을 함께 검사해 골대 후보를 확정한다.
5. 골대 중심, 거리, 화면 중심 오차각, 백보드 yaw를 계산한다.
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
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String


class HoopVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("hoop_vision")

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

        # active 토픽이 아직 오지 않아도 단독 테스트할 수 있도록 기본 True.
        self.declare_parameter("active_on_start", True)

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
        self.declare_parameter("red_h1_low", 0)
        self.declare_parameter("red_h1_high", 10)
        self.declare_parameter("red_h2_low", 160)
        self.declare_parameter("red_h2_high", 179)
        self.declare_parameter("red_s_low", 80)
        self.declare_parameter("red_v_low", 60)

        self.declare_parameter("white_s_high", 80)
        self.declare_parameter("white_v_low", 80)

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

        # =========================================================
        # Depth 조건
        # 16UC1 depth가 mm인 환경을 기준으로 한다.
        # =========================================================
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("depth_min_m", 0.08)
        self.declare_parameter("depth_max_m", 2.0)
        self.declare_parameter("min_valid_depth_pixels", 20)

        # =========================================================
        # 안정화 및 출력
        # =========================================================
        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("hold_frames", 3)
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

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.min_valid_depth_pixels = int(
            self.get_parameter("min_valid_depth_pixels").value
        )

        self.hold_frames = max(0, int(self.get_parameter("hold_frames").value))
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
        self.lost_frames = self.hold_frames
        self.history: Deque[Dict[str, Any]] = deque(
            maxlen=self.smoothing_window
        )

        # =========================================================
        # ROS I/O
        # =========================================================
        self.color_sub = Subscriber(self, Image, self.color_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=5,
            slop=0.1,
        )
        self.sync.registerCallback(self.image_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )
        self.active_sub = self.create_subscription(
            Bool,
            self.active_topic,
            self.active_callback,
            10,
        )

        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.detected_pub = self.create_publisher(
            Bool, self.detected_topic, 10
        )
        self.debug_pub = self.create_publisher(
            Image, self.debug_image_topic, 10
        )

        self.get_logger().info("HoopVisionNode started.")
        self.get_logger().info(f"Color topic: {self.color_topic}")
        self.get_logger().info(f"Aligned depth topic: {self.depth_topic}")
        self.get_logger().info(f"State output: {self.state_topic}")

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
            "hold_frames",
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
            "white_inner_ratio_min",
            "depth_scale",
            "depth_min_m",
            "depth_max_m",
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

        # smoothing_window 변경 시 deque 크기도 갱신한다.
        new_window = max(1, int(self.smoothing_window))
        if self.history.maxlen != new_window:
            self.history = deque(list(self.history)[-new_window:], maxlen=new_window)

        self.hold_frames = max(0, int(self.hold_frames))
        self.print_every_n_frames = max(1, int(self.print_every_n_frames))
        return SetParametersResult(successful=True)

    # =============================================================
    # ROS 콜백
    # =============================================================
    def active_callback(self, msg: Bool) -> None:
        self.active = bool(msg.data)

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
        )

        held_previous = False
        if raw_detection is not None:
            self.lost_frames = 0
            self.history.append(raw_detection)
            smoothed = self._smooth_detection(raw_detection)
            self.last_detection = dict(smoothed)
            published_detection = smoothed
        elif self.last_detection is not None and self.lost_frames < self.hold_frames:
            self.lost_frames += 1
            held_previous = True
            published_detection = dict(self.last_detection)
            published_detection["held_previous_detection"] = True
            published_detection["raw_detected"] = False
        else:
            self.lost_frames = self.hold_frames
            self.history.clear()
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
                    f"dist={published_detection['distance_cm']:.1f} cm "
                    f"center_ang={published_detection['center_angle_deg']:+.1f} deg "
                    f"yaw={published_detection['yaw_deg']:+.1f} deg "
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

    def _find_best_hoop(
        self,
        red_mask: np.ndarray,
        white_mask: np.ndarray,
        roi_depth_m: np.ndarray,
        roi_x_start: int,
        roi_y_start: int,
    ) -> Optional[Dict[str, Any]]:
        contours, _ = cv2.findContours(
            red_mask.copy(),
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

            if (
                top_red_ratio < self.red_ratio_min
                or left_red_ratio < self.red_ratio_min
                or right_red_ratio < self.red_ratio_min
                or white_inner_ratio < self.white_inner_ratio_min
            ):
                continue

            inner_depth = roi_depth_m[inner_mask.astype(bool)]
            valid_inner_depth = inner_depth[
                np.isfinite(inner_depth)
                & (inner_depth >= self.depth_min_m)
                & (inner_depth <= self.depth_max_m)
            ]
            if valid_inner_depth.size < self.min_valid_depth_pixels:
                continue

            distance_m = float(np.median(valid_inner_depth))

            left_depth_values = roi_depth_m[left_mask.astype(bool)]
            right_depth_values = roi_depth_m[right_mask.astype(bool)]
            valid_left = left_depth_values[
                np.isfinite(left_depth_values)
                & (left_depth_values >= self.depth_min_m)
                & (left_depth_values <= self.depth_max_m)
            ]
            valid_right = right_depth_values[
                np.isfinite(right_depth_values)
                & (right_depth_values >= self.depth_min_m)
                & (right_depth_values <= self.depth_max_m)
            ]

            if (
                valid_left.size < self.min_valid_depth_pixels
                or valid_right.size < self.min_valid_depth_pixels
            ):
                continue

            depth_left_m = float(np.median(valid_left))
            depth_right_m = float(np.median(valid_right))

            center_roi = box.mean(axis=0)
            center_x = float(center_roi[0] + roi_x_start)
            center_y = float(center_roi[1] + roi_y_start)
            center_angle_deg = math.degrees(
                math.atan2(center_x - self.cx_intr, self.fx)
            )

            left_center_roi = left_poly.mean(axis=0)
            right_center_roi = right_poly.mean(axis=0)
            left_x_full = float(left_center_roi[0] + roi_x_start)
            right_x_full = float(right_center_roi[0] + roi_x_start)

            x_left_m = (left_x_full - self.cx_intr) * depth_left_m / self.fx
            x_right_m = (right_x_full - self.cx_intr) * depth_right_m / self.fx
            dx_m = x_right_m - x_left_m
            dz_m = depth_left_m - depth_right_m

            if abs(dx_m) < 1e-4:
                continue

            # 양수: 오른쪽 테두리가 카메라에 더 가까운 방향.
            yaw_deg = math.degrees(math.atan2(dz_m, dx_m))

            red_band_ratio = (
                top_red_ratio + left_red_ratio + right_red_ratio
            ) / 3.0
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
                "distance_cm": distance_m * 100.0,
                "center_angle_deg": center_angle_deg,
                "yaw_deg": yaw_deg,
                "depth_left_cm": depth_left_m * 100.0,
                "depth_right_cm": depth_right_m * 100.0,
                "top_red_ratio": top_red_ratio,
                "left_red_ratio": left_red_ratio,
                "right_red_ratio": right_red_ratio,
                "white_inner_ratio": white_inner_ratio,
                "red_band_ratio": red_band_ratio,
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
        """최근 검출값의 중앙값을 사용해 거리와 각도 흔들림을 줄인다."""
        output = dict(current)
        numeric_keys = (
            "center_x",
            "center_y",
            "distance_cm",
            "center_angle_deg",
            "yaw_deg",
            "depth_left_cm",
            "depth_right_cm",
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
                "distance_cm": None,
                "center_angle_deg": None,
                "yaw_deg": None,
                "depth_left_cm": None,
                "depth_right_cm": None,
                "top_red_ratio": None,
                "left_red_ratio": None,
                "right_red_ratio": None,
                "white_inner_ratio": None,
                "red_band_ratio": None,
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

        color = (0, 0, 255)
        if detection is not None:
            color = (0, 255, 0)
        if held_previous:
            color = (0, 255, 255)

        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

        if detection is not None:
            box = np.asarray(detection.get("box", []), dtype=np.int32)
            if box.shape == (4, 2):
                cv2.polylines(debug, [box], True, color, 2)

            cx = int(round(float(detection["center_x"])))
            cy = int(round(float(detection["center_y"])))
            cv2.circle(debug, (cx, cy), 5, (255, 0, 255), -1)

            state_name = "DETECTED" if raw_detected else "HOLD"
            panel_lines = [
                f"HOOP:{state_name}",
                (
                    f"dist:{detection['distance_cm']:.1f}cm "
                    f"ang:{detection['center_angle_deg']:+.1f}deg"
                ),
                (
                    f"yaw:{detection['yaw_deg']:+.1f}deg "
                    f"score:{detection['score']:.2f}"
                ),
                (
                    f"R t:{detection['top_red_ratio']:.2f} "
                    f"l:{detection['left_red_ratio']:.2f} "
                    f"r:{detection['right_red_ratio']:.2f}"
                ),
                (
                    f"white:{detection['white_inner_ratio']:.2f} "
                    f"proc:{process_ms:.1f}ms"
                ),
            ]
        else:
            panel_lines = [
                "HOOP:MISS",
                "dist:N/A ang:N/A",
                "yaw:N/A score:N/A",
                f"proc:{process_ms:.1f}ms",
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
            (0, 255, 0),
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
                (255, 255, 255),
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
