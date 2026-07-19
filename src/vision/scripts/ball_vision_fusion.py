#!/usr/bin/env python3

"""
Ball Vision Fusion Node

역할
1. RealSense color + aligned depth 영상에서 OpenCV로 주황색 공을 직접 검출한다.
2. 검출된 공 중심 픽셀과 depth를 이용해 공의 3차원 위치, 거리, 좌우 각도를 계산한다.
3. 웹캠 YOLO가 /line_tracker/state로 보내는 공 중심 좌표를 구독한다.
4. RealSense + 웹캠 값을 BallStatusPublisher에 전달한다.
5. 디버깅용으로 /ball/vision_state와 /ball/realsense_debug_image를 발행한다.

입력
- /camera/color/image_raw
- /camera/aligned_depth_to_color/image_raw
- /camera/color/camera_info
- /line_tracker/state
- /ball/in_hand

출력
- ball_result
- /ball/vision_state
- /ball/realsense_debug_image

주의
- OpenCV HSV 값은 경기장 조명과 공 색상에 맞게 반드시 조정해야 한다.
- aligned depth 토픽을 사용하므로 color 픽셀과 depth 픽셀을 같은 좌표로 사용한다.
"""

import json
import math
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from ball_status_publisher import BallStatusPublisher


class BallVisionFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("ball_vision_fusion")

        # =========================================================
        # ROS 토픽
        # =========================================================
        self.declare_parameter("realsense_color_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "realsense_depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "realsense_camera_info_topic",
            "/camera/color/camera_info",
        )
        self.declare_parameter("webcam_state_topic", "/line_tracker/state")
        self.declare_parameter("ball_in_hand_topic", "/ball/in_hand")
        self.declare_parameter("vision_state_topic", "/ball/vision_state")
        self.declare_parameter(
            "realsense_debug_image_topic",
            "/ball/realsense_debug_image",
        )
        self.declare_parameter("ball_result_topic", "ball_result")

        # =========================================================
        # RealSense OpenCV 공 검출 파라미터
        # =========================================================
        # RealSense 전체 카메라 프레임을 검출 ROI로 사용한다.
        self.declare_parameter("rs_roi_left_ratio", 0.0)
        self.declare_parameter("rs_roi_right_ratio", 1.0)
        self.declare_parameter("rs_roi_top_ratio", 0.0)
        self.declare_parameter("rs_roi_bottom_ratio", 1.0)

        # 주황색 공 HSV 범위. 실제 환경에서 ros2 param set으로 조정한다.
        self.declare_parameter("h_low", 8)
        self.declare_parameter("h_high", 60)
        self.declare_parameter("s_low", 60)
        self.declare_parameter("s_high", 255)
        self.declare_parameter("v_low", 60)
        self.declare_parameter("v_high", 255)

        self.declare_parameter("depth_threshold_m", 1.5)
        self.declare_parameter("depth_scale", 0.001)  # 16UC1 mm -> m
        # 먼 거리에서 작아진 공 마스크도 후보로 유지한다.
        self.declare_parameter("min_contour_area", 300.0)
        # 검은 무늬·반사광·depth hole로 원 내부가 비어도 허용한다.
        self.declare_parameter("max_circle_ratio_error", 0.45)
        # 화면 경계에 걸려 일부가 잘린 공은 면적과 원형도 조건을 완화한다.
        self.declare_parameter("edge_ball_margin_px", 3)
        self.declare_parameter("edge_min_contour_area_ratio", 0.45)
        self.declare_parameter("edge_max_circle_ratio_error", 0.80)
        # 실제 공 지름은 약 5~6 cm이다. Depth와 CameraInfo로 영상에서
        # 예상되는 픽셀 반지름을 계산해 지나치게 크거나 작은 색상 물체를 제거한다.
        # 마스크가 공 전체를 채우지 않을 수 있어 초기 허용 범위는 넉넉하게 둔다.
        self.declare_parameter("ball_diameter_min_m", 0.050)
        self.declare_parameter("ball_diameter_max_m", 0.060)
        self.declare_parameter("radius_size_min_ratio", 0.45)
        self.declare_parameter("radius_size_max_ratio", 1.70)
        self.declare_parameter("edge_radius_size_min_ratio", 0.25)
        self.declare_parameter("edge_radius_size_max_ratio", 2.00)
        # 최소 외접원 면적뿐 아니라 contour 둘레 기반 원형도와 bbox 비율도 검사한다.
        self.declare_parameter("min_circularity", 0.55)
        self.declare_parameter("edge_min_circularity", 0.30)
        self.declare_parameter("min_aspect_ratio", 0.55)
        self.declare_parameter("max_aspect_ratio", 1.80)
        self.declare_parameter("edge_min_aspect_ratio", 0.30)
        self.declare_parameter("edge_max_aspect_ratio", 3.30)
        self.declare_parameter("morph_kernel_size", 5)
        self.declare_parameter("depth_patch_radius", 1)
        self.declare_parameter("realsense_hold_frames", 10)

        # CameraInfo를 아직 받지 못했을 때 사용할 선배 코드의 기본 내부 파라미터
        self.declare_parameter("fallback_fx", 607.0)
        self.declare_parameter("fallback_fy", 606.0)
        self.declare_parameter("fallback_cx", 325.5)
        self.declare_parameter("fallback_cy", 239.4)

        # 디버그 영상 발행 여부. imshow는 headless 환경 문제 때문에 기본 False.
        self.declare_parameter("publish_realsense_debug_image", True)
        self.declare_parameter("show_realsense_window", False)

        # =========================================================
        # 웹캠 기하 파라미터
        # =========================================================
        self.declare_parameter("webcam_frame_width", 640.0)
        self.declare_parameter("webcam_robot_center_x", 320.0)
        self.declare_parameter("webcam_robot_center_y", 420.0)
        self.declare_parameter("webcam_fov_x_deg", 60.0)

        # =========================================================
        # 유효시간 및 출력 주기
        # =========================================================
        self.declare_parameter("realsense_timeout_sec", 0.5)
        self.declare_parameter("webcam_timeout_sec", 0.5)
        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("print_every_n_frames", 10)
        self.declare_parameter("realsense_use_euclidean_distance", False)

        # =========================================================
        # 파라미터 로드
        # =========================================================
        self.realsense_color_topic = str(
            self.get_parameter("realsense_color_topic").value
        )
        self.realsense_depth_topic = str(
            self.get_parameter("realsense_depth_topic").value
        )
        self.realsense_camera_info_topic = str(
            self.get_parameter("realsense_camera_info_topic").value
        )
        self.webcam_state_topic = str(
            self.get_parameter("webcam_state_topic").value
        )
        self.ball_in_hand_topic = str(
            self.get_parameter("ball_in_hand_topic").value
        )
        self.vision_state_topic = str(
            self.get_parameter("vision_state_topic").value
        )
        self.realsense_debug_image_topic = str(
            self.get_parameter("realsense_debug_image_topic").value
        )
        self.ball_result_topic = str(
            self.get_parameter("ball_result_topic").value
        )

        self.rs_roi_left_ratio = float(
            self.get_parameter("rs_roi_left_ratio").value
        )
        self.rs_roi_right_ratio = float(
            self.get_parameter("rs_roi_right_ratio").value
        )
        self.rs_roi_top_ratio = float(
            self.get_parameter("rs_roi_top_ratio").value
        )
        self.rs_roi_bottom_ratio = float(
            self.get_parameter("rs_roi_bottom_ratio").value
        )

        self.h_low = int(self.get_parameter("h_low").value)
        self.h_high = int(self.get_parameter("h_high").value)
        self.s_low = int(self.get_parameter("s_low").value)
        self.s_high = int(self.get_parameter("s_high").value)
        self.v_low = int(self.get_parameter("v_low").value)
        self.v_high = int(self.get_parameter("v_high").value)

        self.depth_threshold_m = float(
            self.get_parameter("depth_threshold_m").value
        )
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_contour_area = float(
            self.get_parameter("min_contour_area").value
        )
        self.max_circle_ratio_error = float(
            self.get_parameter("max_circle_ratio_error").value
        )
        self.edge_ball_margin_px = max(
            0, int(self.get_parameter("edge_ball_margin_px").value)
        )
        self.edge_min_contour_area_ratio = float(
            self.get_parameter("edge_min_contour_area_ratio").value
        )
        self.edge_max_circle_ratio_error = float(
            self.get_parameter("edge_max_circle_ratio_error").value
        )
        self.ball_diameter_min_m = float(
            self.get_parameter("ball_diameter_min_m").value
        )
        self.ball_diameter_max_m = float(
            self.get_parameter("ball_diameter_max_m").value
        )
        self.radius_size_min_ratio = float(
            self.get_parameter("radius_size_min_ratio").value
        )
        self.radius_size_max_ratio = float(
            self.get_parameter("radius_size_max_ratio").value
        )
        self.edge_radius_size_min_ratio = float(
            self.get_parameter("edge_radius_size_min_ratio").value
        )
        self.edge_radius_size_max_ratio = float(
            self.get_parameter("edge_radius_size_max_ratio").value
        )
        self.min_circularity = float(
            self.get_parameter("min_circularity").value
        )
        self.edge_min_circularity = float(
            self.get_parameter("edge_min_circularity").value
        )
        self.min_aspect_ratio = float(
            self.get_parameter("min_aspect_ratio").value
        )
        self.max_aspect_ratio = float(
            self.get_parameter("max_aspect_ratio").value
        )
        self.edge_min_aspect_ratio = float(
            self.get_parameter("edge_min_aspect_ratio").value
        )
        self.edge_max_aspect_ratio = float(
            self.get_parameter("edge_max_aspect_ratio").value
        )
        self.depth_patch_radius = max(
            0, int(self.get_parameter("depth_patch_radius").value)
        )
        self.realsense_hold_frames = max(
            0, int(self.get_parameter("realsense_hold_frames").value)
        )

        kernel_size = max(
            1, int(self.get_parameter("morph_kernel_size").value)
        )
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.morph_kernel_size = kernel_size
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        self.fx = float(self.get_parameter("fallback_fx").value)
        self.fy = float(self.get_parameter("fallback_fy").value)
        self.cx_intr = float(self.get_parameter("fallback_cx").value)
        self.cy_intr = float(self.get_parameter("fallback_cy").value)
        self.camera_info_received = False

        self.publish_realsense_debug_image = bool(
            self.get_parameter("publish_realsense_debug_image").value
        )
        self.show_realsense_window = bool(
            self.get_parameter("show_realsense_window").value
        )

        self.webcam_frame_width = float(
            self.get_parameter("webcam_frame_width").value
        )
        self.webcam_robot_center_x = float(
            self.get_parameter("webcam_robot_center_x").value
        )
        self.webcam_robot_center_y = float(
            self.get_parameter("webcam_robot_center_y").value
        )
        self.webcam_fov_x_deg = float(
            self.get_parameter("webcam_fov_x_deg").value
        )

        self.realsense_timeout_sec = float(
            self.get_parameter("realsense_timeout_sec").value
        )
        self.webcam_timeout_sec = float(
            self.get_parameter("webcam_timeout_sec").value
        )
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.print_every_n_frames = max(
            1,
            int(self.get_parameter("print_every_n_frames").value),
        )
        self.realsense_use_euclidean_distance = bool(
            self.get_parameter("realsense_use_euclidean_distance").value
        )

        self._update_hsv_arrays()
        self.add_on_set_parameters_callback(self.parameter_callback)

        # =========================================================
        # 상태
        # =========================================================
        self.bridge = CvBridge()

        self.latest_realsense: Optional[Dict[str, Any]] = None
        self.latest_realsense_time = 0.0
        self.last_realsense_detection: Optional[Dict[str, Any]] = None
        self.realsense_lost_frames = self.realsense_hold_frames

        self.latest_webcam: Optional[Dict[str, Any]] = None
        self.latest_webcam_time = 0.0

        self.ball_in_hand = False
        self.frame_count = 0

        # =========================================================
        # ROS I/O
        # =========================================================
        # RealSense color/depth를 시간 동기화해 직접 OpenCV 처리한다.
        self.rs_color_sub = Subscriber(
            self,
            Image,
            self.realsense_color_topic,
        )
        self.rs_depth_sub = Subscriber(
            self,
            Image,
            self.realsense_depth_topic,
        )
        self.rs_sync = ApproximateTimeSynchronizer(
            [self.rs_color_sub, self.rs_depth_sub],
            queue_size=5,
            slop=0.1,
        )
        self.rs_sync.registerCallback(self.cb_realsense_images)

        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            self.realsense_camera_info_topic,
            self.cb_camera_info,
            10,
        )
        self.sub_webcam = self.create_subscription(
            String,
            self.webcam_state_topic,
            self.cb_webcam_state,
            10,
        )
        self.sub_ball_in_hand = self.create_subscription(
            Bool,
            self.ball_in_hand_topic,
            self.cb_ball_in_hand,
            10,
        )

        self.ball_status_publisher = BallStatusPublisher(
            self,
            topic_name=self.ball_result_topic,
        )

        self.pub_vision_state = self.create_publisher(
            String,
            self.vision_state_topic,
            10,
        )
        self.pub_realsense_debug = self.create_publisher(
            Image,
            self.realsense_debug_image_topic,
            10,
        )

        timer_period = 1.0 / max(self.publish_hz, 1.0)
        self.timer = self.create_timer(
            timer_period,
            self.publish_ball_features,
        )

        self.get_logger().info("BallVisionFusionNode started.")
        self.get_logger().info(
            f"RealSense color: {self.realsense_color_topic}"
        )
        self.get_logger().info(
            f"RealSense aligned depth: {self.realsense_depth_topic}"
        )
        self.get_logger().info(
            f"Webcam YOLO input: {self.webcam_state_topic}"
        )
        self.get_logger().info(
            f"BallResult output: {self.ball_result_topic}"
        )

    # =============================================================
    # 파라미터
    # =============================================================
    def _update_hsv_arrays(self) -> None:
        self.lower_hsv = np.array(
            [self.h_low, self.s_low, self.v_low],
            dtype=np.uint8,
        )
        self.upper_hsv = np.array(
            [self.h_high, self.s_high, self.v_high],
            dtype=np.uint8,
        )

    def parameter_callback(self, params) -> SetParametersResult:
        values = {
            "h_low": self.h_low,
            "h_high": self.h_high,
            "s_low": self.s_low,
            "s_high": self.s_high,
            "v_low": self.v_low,
            "v_high": self.v_high,
        }

        for param in params:
            if param.name in values:
                try:
                    values[param.name] = int(param.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f"{param.name} must be an integer",
                    )

        if not (
            0 <= values["h_low"] <= values["h_high"] <= 179
            and 0 <= values["s_low"] <= values["s_high"] <= 255
            and 0 <= values["v_low"] <= values["v_high"] <= 255
        ):
            return SetParametersResult(
                successful=False,
                reason="Invalid HSV range",
            )

        self.h_low = values["h_low"]
        self.h_high = values["h_high"]
        self.s_low = values["s_low"]
        self.s_high = values["s_high"]
        self.v_low = values["v_low"]
        self.v_high = values["v_high"]
        self._update_hsv_arrays()

        for param in params:
            if param.name == "depth_threshold_m":
                self.depth_threshold_m = float(param.value)
            elif param.name == "min_contour_area":
                self.min_contour_area = float(param.value)
            elif param.name == "max_circle_ratio_error":
                self.max_circle_ratio_error = float(param.value)
            elif param.name in {
                "ball_diameter_min_m", "ball_diameter_max_m",
                "radius_size_min_ratio", "radius_size_max_ratio",
                "edge_radius_size_min_ratio", "edge_radius_size_max_ratio",
                "min_circularity", "edge_min_circularity",
                "min_aspect_ratio", "max_aspect_ratio",
                "edge_min_aspect_ratio", "edge_max_aspect_ratio",
            }:
                setattr(self, param.name, float(param.value))

        return SetParametersResult(successful=True)

    # =============================================================
    # CameraInfo
    # =============================================================
    def cb_camera_info(self, msg: CameraInfo) -> None:
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
                "RealSense CameraInfo received: "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx_intr:.2f}, cy={self.cy_intr:.2f}"
            )

    # =============================================================
    # RealSense OpenCV 검출
    # =============================================================
    def cb_realsense_images(
        self,
        color_msg: Image,
        depth_msg: Image,
    ) -> None:
        now = time.monotonic()

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
            self.get_logger().warn(
                f"RealSense image conversion failed: {exc}"
            )
            return

        if frame is None or depth_raw is None:
            return

        depth = np.asarray(depth_raw, dtype=np.float32)
        if depth.ndim != 2:
            self.get_logger().warn("Depth image must be single-channel")
            return

        frame_h, frame_w = frame.shape[:2]
        if depth.shape[0] != frame_h or depth.shape[1] != frame_w:
            self.get_logger().warn(
                "Color/depth size mismatch. "
                "Use aligned_depth_to_color topic."
            )
            return

        x_start = int(frame_w * self.rs_roi_left_ratio)
        x_end = int(frame_w * self.rs_roi_right_ratio)
        y_start = int(frame_h * self.rs_roi_top_ratio)
        y_end = int(frame_h * self.rs_roi_bottom_ratio)

        x_start = max(0, min(x_start, frame_w - 1))
        x_end = max(x_start + 1, min(x_end, frame_w))
        y_start = max(0, min(y_start, frame_h - 1))
        y_end = max(y_start + 1, min(y_end, frame_h))

        roi_color = frame[y_start:y_end, x_start:x_end]
        roi_depth = depth[y_start:y_end, x_start:x_end]

        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(
            hsv,
            self.lower_hsv,
            self.upper_hsv,
        )

        # depth가 0이거나 비정상이거나 설정 거리보다 먼 픽셀은 제거한다.
        roi_depth_m = roi_depth * self.depth_scale
        invalid_depth = (
            ~np.isfinite(roi_depth_m)
            | (roi_depth_m <= 0.0)
            | (roi_depth_m > self.depth_threshold_m)
        )
        raw_mask[invalid_depth] = 0

        mask = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
        )

        detection = self._find_best_ball(
            mask=mask,
            depth=depth,
            roi_x_start=x_start,
            roi_y_start=y_start,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        held_previous = False

        if detection is not None:
            self.realsense_lost_frames = 0
            self.last_realsense_detection = dict(detection)
            self.latest_realsense = dict(detection)
            self.latest_realsense_time = now

        elif (
            self.last_realsense_detection is not None
            and self.realsense_lost_frames < self.realsense_hold_frames
        ):
            # 선배 코드처럼 짧은 미검출은 직전 위치를 유지한다.
            self.realsense_lost_frames += 1
            held_previous = True
            self.latest_realsense = dict(
                self.last_realsense_detection
            )
            self.latest_realsense["held_previous_detection"] = True
            self.latest_realsense_time = now

        else:
            self.realsense_lost_frames = self.realsense_hold_frames
            self.latest_realsense = self._empty_realsense_state()
            self.latest_realsense_time = now

        if self.publish_realsense_debug_image or self.show_realsense_window:
            debug = self._draw_realsense_debug(
                frame=frame,
                raw_mask=raw_mask,
                mask=mask,
                detection=detection,
                held_previous=held_previous,
                roi=(x_start, y_start, x_end, y_end),
            )

            if self.publish_realsense_debug_image:
                debug_msg = self.bridge.cv2_to_imgmsg(
                    debug,
                    encoding="bgr8",
                )
                debug_msg.header = color_msg.header
                self.pub_realsense_debug.publish(debug_msg)

            if self.show_realsense_window:
                cv2.imshow("Ball RealSense OpenCV", debug)
                cv2.waitKey(1)

    def _find_best_ball(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        roi_x_start: int,
        roi_y_start: int,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Dict[str, Any]]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best: Optional[
            Tuple[float, np.ndarray, float, float, float, float, float,
                  float, float, float]
        ] = None

        for contour in contours:
            area = float(cv2.contourArea(contour))
            bx, by, bw, bh = cv2.boundingRect(contour)
            mask_h, mask_w = mask.shape[:2]
            margin = self.edge_ball_margin_px
            touches_edge = bool(
                bx <= margin
                or by <= margin
                or bx + bw >= mask_w - margin
                or by + bh >= mask_h - margin
            )

            min_area = self.min_contour_area
            if touches_edge:
                min_area *= self.edge_min_contour_area_ratio
            if area < min_area:
                continue

            aspect_ratio = float(bw) / max(float(bh), 1.0)
            min_aspect = (
                self.edge_min_aspect_ratio if touches_edge
                else self.min_aspect_ratio
            )
            max_aspect = (
                self.edge_max_aspect_ratio if touches_edge
                else self.max_aspect_ratio
            )
            if not (min_aspect <= aspect_ratio <= max_aspect):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            required_circularity = (
                self.edge_min_circularity if touches_edge
                else self.min_circularity
            )
            if circularity < required_circularity:
                continue

            (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(contour)
            if radius <= 1e-6:
                continue

            circle_area = math.pi * radius * radius
            ratio_error = abs((area / circle_area) - 1.0)

            max_ratio_error = (
                self.edge_max_circle_ratio_error
                if touches_edge
                else self.max_circle_ratio_error
            )
            if ratio_error > max_ratio_error:
                continue

            cx_img = int(round(cx_roi + roi_x_start))
            cy_img = int(round(cy_roi + roi_y_start))
            z_m = self._read_depth_m(
                depth=depth,
                cx=cx_img,
                cy=cy_img,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            if z_m is None or self.fx <= 0.0:
                continue

            expected_radius_min = (
                self.fx * (self.ball_diameter_min_m * 0.5) / z_m
            )
            expected_radius_max = (
                self.fx * (self.ball_diameter_max_m * 0.5) / z_m
            )
            size_min_ratio = (
                self.edge_radius_size_min_ratio if touches_edge
                else self.radius_size_min_ratio
            )
            size_max_ratio = (
                self.edge_radius_size_max_ratio if touches_edge
                else self.radius_size_max_ratio
            )
            allowed_radius_min = expected_radius_min * size_min_ratio
            allowed_radius_max = expected_radius_max * size_max_ratio
            if not (allowed_radius_min <= radius <= allowed_radius_max):
                continue

            expected_radius_mid = 0.5 * (
                expected_radius_min + expected_radius_max
            )
            radius_ratio = radius / max(expected_radius_mid, 1e-6)
            size_error = abs(radius_ratio - 1.0)
            score = ratio_error + (1.0 - min(circularity, 1.0)) + size_error

            if best is None or score < best[0]:
                best = (
                    score,
                    contour,
                    float(cx_roi),
                    float(cy_roi),
                    float(radius),
                    area,
                    float(z_m),
                    float(circularity),
                    float(aspect_ratio),
                    float(radius_ratio),
                )

        if best is None:
            return None

        (
            _score, _contour, cx_roi, cy_roi, radius, area, z_m,
            circularity, aspect_ratio, radius_ratio,
        ) = best
        circle_area = math.pi * radius * radius
        ratio_error = abs((area / circle_area) - 1.0)
        cx_img = int(round(cx_roi + roi_x_start))
        cy_img = int(round(cy_roi + roi_y_start))

        # pinhole camera model을 이용한 3차원 좌표
        x_m = (cx_img - self.cx_intr) * z_m / self.fx
        y_m = (cy_img - self.cy_intr) * z_m / self.fy

        if self.realsense_use_euclidean_distance:
            distance_m = math.sqrt(
                x_m * x_m + y_m * y_m + z_m * z_m
            )
        else:
            distance_m = z_m

        angle_error_deg = math.degrees(math.atan2(x_m, z_m))

        return {
            "realsense_ball_detected": True,
            "realsense_ball_distance_cm": float(distance_m * 100.0),
            "realsense_ball_angle_error": float(angle_error_deg),
            "raw_x_m": float(x_m),
            "raw_y_m": float(y_m),
            "raw_z_m": float(z_m),
            "raw_ball_x": float(cx_img),
            "raw_ball_y": float(cy_img),
            "raw_radius": float(radius),
            "raw_contour_area": float(area),
            "raw_circle_ratio_error": float(ratio_error),
            "raw_circularity": float(circularity),
            "raw_aspect_ratio": float(aspect_ratio),
            "raw_radius_size_ratio": float(radius_ratio),
            "held_previous_detection": False,
        }

    def _read_depth_m(
        self,
        depth: np.ndarray,
        cx: int,
        cy: int,
        frame_w: int,
        frame_h: int,
    ) -> Optional[float]:
        radius = self.depth_patch_radius

        x1 = max(0, cx - radius)
        x2 = min(frame_w, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(frame_h, cy + radius + 1)

        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        patch_m = patch.astype(np.float32) * self.depth_scale
        valid = patch_m[
            np.isfinite(patch_m)
            & (patch_m > 0.0)
            & (patch_m <= self.depth_threshold_m)
        ]

        if valid.size == 0:
            return None

        # 평균보다 outlier에 강한 median 사용
        return float(np.median(valid))

    def _draw_realsense_debug(
        self,
        frame: np.ndarray,
        raw_mask: np.ndarray,
        mask: np.ndarray,
        detection: Optional[Dict[str, Any]],
        held_previous: bool,
        roi: Tuple[int, int, int, int],
    ) -> np.ndarray:
        debug = frame.copy()
        x_start, y_start, x_end, y_end = roi

        color = (0, 255, 0) if detection is not None else (0, 0, 255)
        if held_previous:
            color = (0, 255, 255)

        cv2.rectangle(
            debug,
            (x_start, y_start),
            (x_end, y_end),
            color,
            2,
        )

        draw_state = detection
        if draw_state is None and held_previous:
            draw_state = self.last_realsense_detection

        if draw_state is not None:
            cx = int(round(draw_state["raw_ball_x"]))
            cy = int(round(draw_state["raw_ball_y"]))
            radius = int(round(draw_state["raw_radius"]))

            cv2.circle(debug, (cx, cy), radius, color, 2)
            cv2.circle(debug, (cx, cy), 4, (0, 0, 255), -1)

            state_text = "HOLD" if held_previous else "DETECTED"
            text = (
                f"{state_text} "
                f"{draw_state['realsense_ball_distance_cm']:.1f}cm "
                f"{draw_state['realsense_ball_angle_error']:+.1f}deg"
            )
        else:
            text = "BALL MISS"

        cv2.putText(
            debug,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        hsv_text = (
            f"HSV [{self.h_low},{self.s_low},{self.v_low}]"
            f"-[{self.h_high},{self.s_high},{self.v_high}]"
        )
        cv2.putText(
            debug,
            hsv_text,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # 마스크를 작은 미리보기로 합성
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview_w = min(240, debug.shape[1] // 3)
        if preview_w > 0 and mask_bgr.shape[1] > 0:
            scale = preview_w / mask_bgr.shape[1]
            preview_h = max(1, int(mask_bgr.shape[0] * scale))
            preview = cv2.resize(mask_bgr, (preview_w, preview_h))
            y0 = 5
            x0 = max(0, debug.shape[1] - preview_w - 5)
            y1 = min(debug.shape[0], y0 + preview_h)
            x1 = min(debug.shape[1], x0 + preview_w)
            debug[y0:y1, x0:x1] = preview[: y1 - y0, : x1 - x0]

        return debug

    def _empty_realsense_state(self) -> Dict[str, Any]:
        return {
            "realsense_ball_detected": False,
            "realsense_ball_distance_cm": None,
            "realsense_ball_angle_error": None,
            "raw_x_m": None,
            "raw_y_m": None,
            "raw_z_m": None,
            "raw_ball_x": None,
            "raw_ball_y": None,
            "raw_radius": None,
            "raw_contour_area": None,
            "raw_circle_ratio_error": None,
            "raw_circularity": None,
            "raw_aspect_ratio": None,
            "raw_radius_size_ratio": None,
            "held_previous_detection": False,
        }

    # =============================================================
    # Webcam YOLO
    # =============================================================
    def cb_webcam_state(self, msg: String) -> None:
        now = time.monotonic()

        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(
                "Failed to parse /line_tracker/state JSON"
            )
            return

        ball_detected = bool(payload.get("ball_detected", False))

        if not ball_detected:
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        try:
            ball_x = float(payload.get("ball_x", -1.0))
            ball_y = float(payload.get("ball_y", -1.0))
            ball_conf = float(payload.get("ball_conf", 0.0))
        except (TypeError, ValueError):
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        ball_bbox = payload.get("ball_bbox", [])

        if (
            not math.isfinite(ball_x)
            or not math.isfinite(ball_y)
            or ball_x < 0.0
            or ball_y < 0.0
        ):
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        dx = ball_x - self.webcam_robot_center_x
        dy = ball_y - self.webcam_robot_center_y
        distance_px = math.hypot(dx, dy)

        angle_error_deg: Optional[float]
        if (
            0.0 < self.webcam_fov_x_deg < 180.0
            and self.webcam_frame_width > 0.0
        ):
            focal_px = self.webcam_frame_width / (
                2.0
                * math.tan(
                    math.radians(self.webcam_fov_x_deg) / 2.0
                )
            )
            angle_error_deg = float(
                math.degrees(math.atan2(dx, focal_px))
            )
        else:
            angle_error_deg = None

        self.latest_webcam = {
            "webcam_ball_detected": True,
            "webcam_ball_x_distance": float(dx),
            "webcam_ball_angle_error": angle_error_deg,
            "webcam_ball_distance_px": float(distance_px),
            "raw_ball_x": ball_x,
            "raw_ball_y": ball_y,
            "raw_ball_conf": ball_conf,
            "raw_ball_bbox": ball_bbox,
        }
        self.latest_webcam_time = now

    def _empty_webcam_state(self) -> Dict[str, Any]:
        return {
            "webcam_ball_detected": False,
            "webcam_ball_x_distance": None,
            "webcam_ball_angle_error": None,
            "webcam_ball_distance_px": None,
            "raw_ball_x": None,
            "raw_ball_y": None,
            "raw_ball_conf": 0.0,
            "raw_ball_bbox": [],
        }

    # =============================================================
    # ball_in_hand
    # =============================================================
    def cb_ball_in_hand(self, msg: Bool) -> None:
        self.ball_in_hand = bool(msg.data)

    # =============================================================
    # BallFeatures 생성 및 알고리즘 전달
    # =============================================================
    def publish_ball_features(self) -> None:
        now = time.monotonic()

        realsense_age = (
            now - self.latest_realsense_time
            if self.latest_realsense is not None
            else None
        )
        webcam_age = (
            now - self.latest_webcam_time
            if self.latest_webcam is not None
            else None
        )

        realsense_valid = bool(
            self.latest_realsense is not None
            and realsense_age is not None
            and realsense_age <= self.realsense_timeout_sec
            and self.latest_realsense.get(
                "realsense_ball_detected",
                False,
            )
        )

        webcam_valid = bool(
            self.latest_webcam is not None
            and webcam_age is not None
            and webcam_age <= self.webcam_timeout_sec
            and self.latest_webcam.get(
                "webcam_ball_detected",
                False,
            )
        )

        features: Dict[str, Any] = {
            "realsense_ball_detected": False,
            "realsense_ball_distance_cm": None,
            "realsense_ball_angle_error": None,
            "webcam_ball_detected": False,
            "webcam_ball_x_distance": None,
            "webcam_ball_angle_error": None,
            "webcam_ball_distance_px": None,
            "ball_in_hand": bool(self.ball_in_hand),
        }

        if realsense_valid and self.latest_realsense is not None:
            features.update(
                {
                    "realsense_ball_detected": True,
                    "realsense_ball_distance_cm":
                        self.latest_realsense[
                            "realsense_ball_distance_cm"
                        ],
                    "realsense_ball_angle_error":
                        self.latest_realsense[
                            "realsense_ball_angle_error"
                        ],
                }
            )

        if webcam_valid and self.latest_webcam is not None:
            features.update(
                {
                    "webcam_ball_detected": True,
                    "webcam_ball_x_distance":
                        self.latest_webcam[
                            "webcam_ball_x_distance"
                        ],
                    "webcam_ball_angle_error":
                        self.latest_webcam[
                            "webcam_ball_angle_error"
                        ],
                    "webcam_ball_distance_px":
                        self.latest_webcam[
                            "webcam_ball_distance_px"
                        ],
                }
            )

        status, angle = (
            self.ball_status_publisher.publish_ball_status(
                **features
            )
        )

        if webcam_valid:
            source_priority = "webcam"
        elif realsense_valid:
            source_priority = "realsense"
        else:
            source_priority = "none"

        output: Dict[str, Any] = dict(features)
        output.update(
            {
                "source_priority": source_priority,
                "realsense_detection_method": "opencv_hsv_depth",
                "realsense_age_sec": realsense_age,
                "webcam_age_sec": webcam_age,
                "ball_status": int(status),
                "ball_status_angle": float(angle),
                "camera_info_received": self.camera_info_received,
            }
        )

        if realsense_valid and self.latest_realsense is not None:
            output["realsense_raw"] = {
                "x_m": self.latest_realsense["raw_x_m"],
                "y_m": self.latest_realsense["raw_y_m"],
                "z_m": self.latest_realsense["raw_z_m"],
                "ball_x": self.latest_realsense["raw_ball_x"],
                "ball_y": self.latest_realsense["raw_ball_y"],
                "radius": self.latest_realsense["raw_radius"],
                "contour_area":
                    self.latest_realsense["raw_contour_area"],
                "circle_ratio_error":
                    self.latest_realsense[
                        "raw_circle_ratio_error"
                    ],
                "circularity":
                    self.latest_realsense["raw_circularity"],
                "aspect_ratio":
                    self.latest_realsense["raw_aspect_ratio"],
                "radius_size_ratio":
                    self.latest_realsense["raw_radius_size_ratio"],
                "held_previous_detection":
                    self.latest_realsense[
                        "held_previous_detection"
                    ],
            }

        if webcam_valid and self.latest_webcam is not None:
            output["webcam_raw"] = {
                "ball_x": self.latest_webcam["raw_ball_x"],
                "ball_y": self.latest_webcam["raw_ball_y"],
                "ball_conf":
                    self.latest_webcam["raw_ball_conf"],
                "ball_bbox":
                    self.latest_webcam["raw_ball_bbox"],
            }

        self.pub_vision_state.publish(
            String(data=json.dumps(output, ensure_ascii=False))
        )

        self.frame_count += 1
        if self.frame_count % self.print_every_n_frames == 0:
            self.get_logger().info(
                "ball_vision "
                f"src={source_priority} "
                f"rs={features['realsense_ball_detected']} "
                f"rs_dist={features['realsense_ball_distance_cm']} "
                f"rs_ang={features['realsense_ball_angle_error']} "
                f"webcam={features['webcam_ball_detected']} "
                f"webcam_x={features['webcam_ball_x_distance']} "
                f"webcam_dist={features['webcam_ball_distance_px']} "
                f"hand={features['ball_in_hand']} "
                f"status={status} angle={angle:.2f}"
            )

    def destroy_node(self):
        if self.show_realsense_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallVisionFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
