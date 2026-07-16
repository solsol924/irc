#!/usr/bin/env python3
"""
RealSense OpenCV + webcam YOLO hurdle fusion node.

RealSense does not run YOLO here.  It uses aligned color/depth images and the
OpenCV detector in hurdle_detector_core.py.  The existing yolo_detector.py keeps
placing hurdle_detected/hurdle_x/hurdle_y/hurdle_conf/hurdle_bbox in
/line_tracker/state.  This node fuses both sources and is intended to be the
only publisher of hurdle_result.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import fields, replace
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from hurdle_detector_core import (
    HurdleDetectorConfig,
    HurdleOpenCVDetector,
    HurdleVisionState,
)
from hurdle_status_publisher import HurdleStatusPublisher


DETECTOR_FIELDS = {item.name for item in fields(HurdleDetectorConfig)}
CALIBRATION_RESET_FIELDS = {
    "roi_left_ratio",
    "roi_right_ratio",
    "roi_top_ratio",
    "roi_bottom_ratio",
    "depth_min_m",
    "depth_max_m",
    "floor_calibration_frames",
    "calibration_min_valid_ratio",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


class HurdleVisionFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("hurdle_vision_fusion")

        # ---------------------------------------------------------
        # ROS topics
        # ---------------------------------------------------------
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
        self.declare_parameter("vision_state_topic", "/hurdle/vision_state")
        self.declare_parameter(
            "realsense_debug_image_topic",
            "/hurdle/realsense_debug_image",
        )
        self.declare_parameter("hurdle_result_topic", "hurdle_result")
        self.declare_parameter("recalibrate_service", "/hurdle/recalibrate")

        # ---------------------------------------------------------
        # Fusion and timing
        # ---------------------------------------------------------
        # or, and, realsense_only, webcam_only
        self.declare_parameter("fusion_mode", "or")
        self.declare_parameter("realsense_timeout_sec", 0.6)
        self.declare_parameter("webcam_timeout_sec", 0.6)
        self.declare_parameter("webcam_min_conf", 0.35)
        self.declare_parameter("webcam_frame_width", 640.0)
        self.declare_parameter("webcam_frame_height", 480.0)
        self.declare_parameter("webcam_fov_x_deg", 60.0)
        self.declare_parameter("webcam_cross_y_ratio", 0.72)
        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("print_every_n_frames", 10)

        # HurdleResult.angle source: orientation, bearing, zero.
        self.declare_parameter("result_angle_source", "orientation")
        self.declare_parameter("result_angle_absolute", False)

        # ---------------------------------------------------------
        # Image conversion / debug
        # ---------------------------------------------------------
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_sec", 0.08)
        self.declare_parameter("publish_realsense_debug_image", True)
        self.declare_parameter("show_realsense_window", False)

        # CameraInfo fallback values from the ball fusion reference.
        self.declare_parameter("fallback_fx", 607.0)
        self.declare_parameter("fallback_fy", 606.0)
        self.declare_parameter("fallback_cx", 325.5)
        self.declare_parameter("fallback_cy", 239.4)

        # Declare all OpenCV detector parameters with dataclass defaults.
        detector_defaults = HurdleDetectorConfig()
        for item in fields(HurdleDetectorConfig):
            self.declare_parameter(item.name, getattr(detector_defaults, item.name))

        # ---------------------------------------------------------
        # Load parameters
        # ---------------------------------------------------------
        self.realsense_color_topic = str(self.get_parameter("realsense_color_topic").value)
        self.realsense_depth_topic = str(self.get_parameter("realsense_depth_topic").value)
        self.realsense_camera_info_topic = str(
            self.get_parameter("realsense_camera_info_topic").value
        )
        self.webcam_state_topic = str(self.get_parameter("webcam_state_topic").value)
        self.vision_state_topic = str(self.get_parameter("vision_state_topic").value)
        self.realsense_debug_image_topic = str(
            self.get_parameter("realsense_debug_image_topic").value
        )
        self.hurdle_result_topic = str(self.get_parameter("hurdle_result_topic").value)
        self.recalibrate_service_name = str(
            self.get_parameter("recalibrate_service").value
        )

        self.fusion_mode = self._validated_fusion_mode(
            str(self.get_parameter("fusion_mode").value)
        )
        self.realsense_timeout_sec = float(
            self.get_parameter("realsense_timeout_sec").value
        )
        self.webcam_timeout_sec = float(self.get_parameter("webcam_timeout_sec").value)
        self.webcam_min_conf = float(self.get_parameter("webcam_min_conf").value)
        self.webcam_frame_width = float(self.get_parameter("webcam_frame_width").value)
        self.webcam_frame_height = float(self.get_parameter("webcam_frame_height").value)
        self.webcam_fov_x_deg = float(self.get_parameter("webcam_fov_x_deg").value)
        self.webcam_cross_y_ratio = float(
            self.get_parameter("webcam_cross_y_ratio").value
        )
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.print_every_n_frames = max(
            1,
            int(self.get_parameter("print_every_n_frames").value),
        )
        self.result_angle_source = self._validated_angle_source(
            str(self.get_parameter("result_angle_source").value)
        )
        self.result_angle_absolute = bool(
            self.get_parameter("result_angle_absolute").value
        )

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.sync_queue_size = max(2, int(self.get_parameter("sync_queue_size").value))
        self.sync_slop_sec = max(0.0, float(self.get_parameter("sync_slop_sec").value))
        self.publish_realsense_debug_image = bool(
            self.get_parameter("publish_realsense_debug_image").value
        )
        self.show_realsense_window = bool(
            self.get_parameter("show_realsense_window").value
        )

        self.fx = float(self.get_parameter("fallback_fx").value)
        self.fy = float(self.get_parameter("fallback_fy").value)
        self.cx_intr = float(self.get_parameter("fallback_cx").value)
        self.cy_intr = float(self.get_parameter("fallback_cy").value)
        self.camera_info_received = False

        detector_values = {
            name: self.get_parameter(name).value
            for name in DETECTOR_FIELDS
        }
        self.detector = HurdleOpenCVDetector(HurdleDetectorConfig(**detector_values))
        self.add_on_set_parameters_callback(self.parameter_callback)

        # ---------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------
        self.bridge = CvBridge()
        self.latest_realsense: Optional[Dict[str, Any]] = None
        self.latest_realsense_time = 0.0
        self.latest_webcam: Optional[Dict[str, Any]] = None
        self.latest_webcam_time = 0.0
        self.publish_count = 0

        # ---------------------------------------------------------
        # ROS I/O
        # ---------------------------------------------------------
        self.rs_color_sub = Subscriber(
            self,
            Image,
            self.realsense_color_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.rs_depth_sub = Subscriber(
            self,
            Image,
            self.realsense_depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.rs_sync = ApproximateTimeSynchronizer(
            [self.rs_color_sub, self.rs_depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.rs_sync.registerCallback(self.cb_realsense_images)

        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            self.realsense_camera_info_topic,
            self.cb_camera_info,
            qos_profile_sensor_data,
        )
        self.sub_webcam = self.create_subscription(
            String,
            self.webcam_state_topic,
            self.cb_webcam_state,
            10,
        )

        self.hurdle_status_publisher = HurdleStatusPublisher(
            self,
            topic_name=self.hurdle_result_topic,
        )
        self.pub_vision_state = self.create_publisher(String, self.vision_state_topic, 10)
        self.pub_realsense_debug = self.create_publisher(
            Image,
            self.realsense_debug_image_topic,
            qos_profile_sensor_data,
        )
        self.recalibrate_srv = self.create_service(
            Trigger,
            self.recalibrate_service_name,
            self.cb_recalibrate,
        )

        self.timer = self.create_timer(
            1.0 / max(1.0, self.publish_hz),
            self.publish_hurdle_features,
        )

        self.get_logger().info("HurdleVisionFusionNode started")
        self.get_logger().info(f"RealSense color: {self.realsense_color_topic}")
        self.get_logger().info(f"RealSense aligned depth: {self.realsense_depth_topic}")
        self.get_logger().info(f"Webcam YOLO state: {self.webcam_state_topic}")
        self.get_logger().info(
            f"fusion_mode={self.fusion_mode}, hurdle_result={self.hurdle_result_topic}"
        )
        self.get_logger().info(
            "Floor calibration starts automatically; keep the RealSense ROI clear."
        )

    # -------------------------------------------------------------
    # Parameter handling
    # -------------------------------------------------------------
    @staticmethod
    def _validated_fusion_mode(value: str) -> str:
        mode = value.strip().lower()
        if mode not in {"or", "and", "realsense_only", "webcam_only"}:
            raise ValueError("fusion_mode must be or, and, realsense_only, webcam_only")
        return mode

    @staticmethod
    def _validated_angle_source(value: str) -> str:
        source = value.strip().lower()
        if source not in {"orientation", "bearing", "zero"}:
            raise ValueError("result_angle_source must be orientation, bearing, or zero")
        return source

    @staticmethod
    def _cast_like(value: Any, current: Any) -> Any:
        if isinstance(current, bool):
            return bool(value)
        if isinstance(current, int) and not isinstance(current, bool):
            return int(value)
        if isinstance(current, float):
            return float(value)
        if isinstance(current, str):
            return str(value)
        return value

    def parameter_callback(self, params) -> SetParametersResult:
        try:
            new_cfg = replace(self.detector.cfg)
            reset_floor = False
            new_fusion_mode = self.fusion_mode
            new_angle_source = self.result_angle_source
            new_values = {
                "realsense_timeout_sec": self.realsense_timeout_sec,
                "webcam_timeout_sec": self.webcam_timeout_sec,
                "webcam_min_conf": self.webcam_min_conf,
                "webcam_cross_y_ratio": self.webcam_cross_y_ratio,
                "result_angle_absolute": self.result_angle_absolute,
            }

            for param in params:
                if param.name in DETECTOR_FIELDS:
                    current = getattr(new_cfg, param.name)
                    setattr(new_cfg, param.name, self._cast_like(param.value, current))
                    if param.name in CALIBRATION_RESET_FIELDS:
                        reset_floor = True
                elif param.name == "fusion_mode":
                    new_fusion_mode = self._validated_fusion_mode(str(param.value))
                elif param.name == "result_angle_source":
                    new_angle_source = self._validated_angle_source(str(param.value))
                elif param.name in new_values:
                    current = new_values[param.name]
                    new_values[param.name] = self._cast_like(param.value, current)

            new_cfg = new_cfg.validated()
            if not (0.0 <= float(new_values["webcam_cross_y_ratio"]) <= 1.0):
                raise ValueError("webcam_cross_y_ratio must be in [0, 1]")
            if float(new_values["realsense_timeout_sec"]) <= 0.0:
                raise ValueError("realsense_timeout_sec must be > 0")
            if float(new_values["webcam_timeout_sec"]) <= 0.0:
                raise ValueError("webcam_timeout_sec must be > 0")

            self.detector.update_config(new_cfg, reset_floor=reset_floor)
            self.fusion_mode = new_fusion_mode
            self.result_angle_source = new_angle_source
            self.realsense_timeout_sec = float(new_values["realsense_timeout_sec"])
            self.webcam_timeout_sec = float(new_values["webcam_timeout_sec"])
            self.webcam_min_conf = float(new_values["webcam_min_conf"])
            self.webcam_cross_y_ratio = float(new_values["webcam_cross_y_ratio"])
            self.result_angle_absolute = bool(new_values["result_angle_absolute"])
            return SetParametersResult(successful=True)
        except Exception as exc:
            return SetParametersResult(successful=False, reason=str(exc))

    # -------------------------------------------------------------
    # RealSense callbacks
    # -------------------------------------------------------------
    def cb_camera_info(self, msg: CameraInfo) -> None:
        if len(msg.k) < 9:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return
        self.fx, self.fy = fx, fy
        self.cx_intr, self.cy_intr = cx, cy
        if not self.camera_info_received:
            self.camera_info_received = True
            self.get_logger().info(
                f"CameraInfo received fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
            )

    def _depth_to_meters(self, depth_raw: np.ndarray, encoding: str) -> np.ndarray:
        depth = np.asarray(depth_raw)
        enc = str(encoding or "").upper()
        if np.issubdtype(depth.dtype, np.floating) or "32FC1" in enc:
            return depth.astype(np.float32, copy=False)
        return depth.astype(np.float32) * self.depth_scale

    def cb_realsense_images(self, color_msg: Image, depth_msg: Image) -> None:
        now = time.monotonic()
        try:
            color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"RealSense image conversion failed: {exc}")
            return

        if color_bgr is None or depth_raw is None:
            return
        depth_m = self._depth_to_meters(depth_raw, depth_msg.encoding)
        if depth_m.ndim != 2 or color_bgr.shape[:2] != depth_m.shape[:2]:
            self.get_logger().warn(
                "Color/depth size mismatch. Use /camera/aligned_depth_to_color/image_raw."
            )
            return

        try:
            state, debug = self.detector.process(color_bgr, depth_m)
        except Exception as exc:
            self.get_logger().error(f"Hurdle OpenCV processing failed: {exc}")
            return

        hurdle_x = state.get("hurdle_x")
        if hurdle_x is not None and self.fx > 0.0:
            x_normalized = (float(hurdle_x) - self.cx_intr) / self.fx
            state["bearing_angle_deg"] = float(math.degrees(math.atan2(x_normalized, 1.0)))
        else:
            state["bearing_angle_deg"] = None

        state["camera_info_received"] = bool(self.camera_info_received)
        self.latest_realsense = _json_safe(state)
        self.latest_realsense_time = now

        if self.publish_realsense_debug_image:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = color_msg.header
            self.pub_realsense_debug.publish(debug_msg)

        if self.show_realsense_window:
            cv2.imshow("Hurdle RealSense OpenCV", debug)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rclpy.shutdown()

    # -------------------------------------------------------------
    # Webcam YOLO callback
    # -------------------------------------------------------------
    @staticmethod
    def _empty_webcam_state() -> Dict[str, Any]:
        return {
            "webcam_hurdle_detected": False,
            "webcam_hurdle_x": None,
            "webcam_hurdle_y": None,
            "webcam_hurdle_conf": 0.0,
            "webcam_hurdle_bbox": [],
            "webcam_hurdle_x_offset_px": None,
            "webcam_hurdle_y_ratio": None,
            "webcam_hurdle_bearing_deg": None,
            "webcam_crossing_ready": False,
        }

    def cb_webcam_state(self, msg: String) -> None:
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn("Failed to parse /line_tracker/state JSON")
            return

        detected = bool(payload.get("hurdle_detected", False))
        try:
            conf = float(payload.get("hurdle_conf", 0.0))
            x = float(payload.get("hurdle_x", -1.0))
            y = float(payload.get("hurdle_y", -1.0))
        except (TypeError, ValueError):
            detected = False
            conf, x, y = 0.0, -1.0, -1.0

        if (
            not detected
            or conf < self.webcam_min_conf
            or not math.isfinite(x)
            or not math.isfinite(y)
            or x < 0.0
            or y < 0.0
        ):
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        x_offset = x - self.webcam_frame_width / 2.0
        y_ratio = y / max(1.0, self.webcam_frame_height)
        bearing = None
        if 0.0 < self.webcam_fov_x_deg < 180.0 and self.webcam_frame_width > 0.0:
            focal_px = self.webcam_frame_width / (
                2.0 * math.tan(math.radians(self.webcam_fov_x_deg) / 2.0)
            )
            bearing = float(math.degrees(math.atan2(x_offset, focal_px)))

        self.latest_webcam = {
            "webcam_hurdle_detected": True,
            "webcam_hurdle_x": x,
            "webcam_hurdle_y": y,
            "webcam_hurdle_conf": conf,
            "webcam_hurdle_bbox": payload.get("hurdle_bbox", []),
            "webcam_hurdle_x_offset_px": float(x_offset),
            "webcam_hurdle_y_ratio": float(y_ratio),
            "webcam_hurdle_bearing_deg": bearing,
            "webcam_crossing_ready": bool(y_ratio >= self.webcam_cross_y_ratio),
        }
        self.latest_webcam_time = now

    # -------------------------------------------------------------
    # Recalibration service
    # -------------------------------------------------------------
    def cb_recalibrate(self, _request: Trigger.Request, response: Trigger.Response):
        self.detector.reset_calibration()
        self.latest_realsense = None
        self.latest_realsense_time = 0.0
        response.success = True
        response.message = "Hurdle floor calibration restarted; keep the ROI clear."
        self.get_logger().info(response.message)
        return response

    # -------------------------------------------------------------
    # Fusion and publishing
    # -------------------------------------------------------------
    def _select_result_angle(self, realsense_valid: bool) -> float:
        if not realsense_valid or self.latest_realsense is None:
            return 0.0
        if self.result_angle_source == "orientation":
            value = self.latest_realsense.get("horizontal_angle_deg")
        elif self.result_angle_source == "bearing":
            value = self.latest_realsense.get("bearing_angle_deg")
        else:
            value = 0.0
        try:
            angle = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            angle = 0.0
        if not math.isfinite(angle):
            angle = 0.0
        return abs(angle) if self.result_angle_absolute else angle

    def publish_hurdle_features(self) -> None:
        now = time.monotonic()
        rs_age = (
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
            and rs_age is not None
            and rs_age <= self.realsense_timeout_sec
            and self.latest_realsense.get("hurdle_detected", False)
        )
        webcam_valid = bool(
            self.latest_webcam is not None
            and webcam_age is not None
            and webcam_age <= self.webcam_timeout_sec
            and self.latest_webcam.get("webcam_hurdle_detected", False)
        )

        if self.fusion_mode == "and":
            fused_detected = realsense_valid and webcam_valid
        elif self.fusion_mode == "realsense_only":
            fused_detected = realsense_valid
        elif self.fusion_mode == "webcam_only":
            fused_detected = webcam_valid
        else:
            fused_detected = realsense_valid or webcam_valid

        if realsense_valid and webcam_valid:
            source = "both"
        elif realsense_valid:
            source = "realsense"
        elif webcam_valid:
            source = "webcam"
        else:
            source = "none"

        rs_crossing = bool(
            realsense_valid
            and self.latest_realsense is not None
            and self.latest_realsense.get("crossing_ready", False)
        )
        webcam_crossing = bool(
            webcam_valid
            and self.latest_webcam is not None
            and self.latest_webcam.get("webcam_crossing_ready", False)
        )
        crossing_ready = bool(fused_detected and (rs_crossing or webcam_crossing))

        result_angle = self._select_result_angle(realsense_valid)
        # 프로젝트에서 사용 중인 hurdle_status_publisher.py는
        # hurdle_detected 인자만 받는다. 측정 각도는 디버그 JSON에 남기고,
        # 최종 HurdleResult는 기존 판단 코드(status 18/99, angle 0.0)를 그대로 사용한다.
        status, published_angle = self.hurdle_status_publisher.publish_hurdle_status(
            hurdle_detected=bool(fused_detected),
        )

        output: Dict[str, Any] = {
            "fused_hurdle_detected": bool(fused_detected),
            "source": source,
            "fusion_mode": self.fusion_mode,
            "crossing_ready": crossing_ready,
            "hurdle_status": int(status),
            "hurdle_status_angle": float(published_angle),
            "result_angle_source": self.result_angle_source,
            "measured_hurdle_angle_deg": float(result_angle),
            "realsense_valid": bool(realsense_valid),
            "webcam_valid": bool(webcam_valid),
            "realsense_age_sec": rs_age,
            "webcam_age_sec": webcam_age,
            "realsense_detection_method": "opencv_hsv_lab_bgr_depth_shape",
            "camera_info_received": bool(self.camera_info_received),
        }

        if self.latest_realsense is not None:
            output["realsense"] = dict(self.latest_realsense)
        else:
            output["realsense"] = {
                "hurdle_detected": False,
                "vision_state": HurdleVisionState.CALIBRATING,
                "calibration_done": False,
            }

        if self.latest_webcam is not None:
            output["webcam"] = dict(self.latest_webcam)
        else:
            output["webcam"] = self._empty_webcam_state()

        self.pub_vision_state.publish(
            String(data=json.dumps(_json_safe(output), ensure_ascii=False))
        )

        self.publish_count += 1
        if self.publish_count % self.print_every_n_frames == 0:
            rs = output["realsense"]
            self.get_logger().info(
                "hurdle_vision "
                f"src={source} fused={fused_detected} cross={crossing_ready} "
                f"rs={realsense_valid} dist={rs.get('hurdle_distance_m')} "
                f"height={rs.get('hurdle_height_m')} "
                f"orient={rs.get('horizontal_angle_deg')} "
                f"webcam={webcam_valid} status={status} angle={published_angle:.1f}"
            )

    def destroy_node(self):
        if self.show_realsense_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HurdleVisionFusionNode()
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
