#!/usr/bin/env python3
"""RealSense YOLO와 webcam YOLO 상태를 로봇의 공 상태로 융합한다.

영상을 다시 검출하거나 그리지 않고 최신 JSON 상태만 처리한다. 따라서 화면
발행은 ``realsense_yolo_detector.py``가 계속 담당하며 모드 전환 때 카메라나
검출 노드를 재시작하지 않는다.
"""

import json
import math
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from ball_status_publisher import BallStatusPublisher


class BallVisionFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("ball_vision_fusion")

        self.declare_parameter(
            "realsense_yolo_state_topic", "/realsense_yolo/ball_state"
        )
        self.declare_parameter("webcam_state_topic", "/line_tracker/state")
        self.declare_parameter("hoop_state_topic", "/hoop/vision_state")
        self.declare_parameter("active_topic", "/vision/ball_active")
        self.declare_parameter("hoop_active_topic", "/vision/hoop_active")
        self.declare_parameter("active_on_start", True)
        self.declare_parameter("manage_activity_from_ball_in_hand", True)
        self.declare_parameter("raw_ball_in_hand_topic", "/raw_ball_in_hand")
        self.declare_parameter("vision_state_topic", "/ball/vision_state")
        self.declare_parameter("ball_result_topic", "ball_result")
        self.declare_parameter("webcam_frame_width", 640.0)
        self.declare_parameter("webcam_robot_center_x", 320.0)
        self.declare_parameter("webcam_robot_center_y", 420.0)
        self.declare_parameter("webcam_fov_x_deg", 60.0)
        self.declare_parameter("realsense_timeout_sec", 0.5)
        self.declare_parameter("webcam_timeout_sec", 0.5)
        self.declare_parameter("hoop_timeout_sec", 0.5)
        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("print_every_n_frames", 10)

        def text_param(name: str) -> str:
            return str(self.get_parameter(name).value)

        def float_param(name: str) -> float:
            return float(self.get_parameter(name).value)

        self.realsense_yolo_state_topic = text_param(
            "realsense_yolo_state_topic"
        )
        self.webcam_state_topic = text_param("webcam_state_topic")
        self.hoop_state_topic = text_param("hoop_state_topic")
        self.active_topic = text_param("active_topic")
        self.hoop_active_topic = text_param("hoop_active_topic")
        self.raw_ball_in_hand_topic = text_param("raw_ball_in_hand_topic")
        self.vision_state_topic = text_param("vision_state_topic")
        self.ball_result_topic = text_param("ball_result_topic")
        self.manage_activity_from_ball_in_hand = bool(
            self.get_parameter("manage_activity_from_ball_in_hand").value
        )
        self.webcam_frame_width = float_param("webcam_frame_width")
        self.webcam_robot_center_x = float_param("webcam_robot_center_x")
        self.webcam_robot_center_y = float_param("webcam_robot_center_y")
        self.webcam_fov_x_deg = float_param("webcam_fov_x_deg")
        self.realsense_timeout_sec = float_param("realsense_timeout_sec")
        self.webcam_timeout_sec = float_param("webcam_timeout_sec")
        self.hoop_timeout_sec = float_param("hoop_timeout_sec")
        self.publish_hz = float_param("publish_hz")
        self.print_every_n_frames = max(
            1, int(self.get_parameter("print_every_n_frames").value)
        )

        self.latest_realsense: Optional[Dict[str, Any]] = None
        self.latest_realsense_time = 0.0
        self.latest_webcam: Optional[Dict[str, Any]] = None
        self.latest_webcam_time = 0.0
        self.latest_hoop: Optional[Dict[str, Any]] = None
        self.latest_hoop_time = 0.0
        self.ball_in_hand = False
        self.frame_count = 0
        self.last_realsense_diagnostic_label: Optional[str] = None
        self.ball_detection_active = bool(
            self.get_parameter("active_on_start").value
        )
        self.managed_hoop_active: Optional[bool] = None

        activity_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub_active = self.create_subscription(
            Bool, self.active_topic, self.cb_ball_active, activity_qos
        )
        self.pub_ball_active = self.create_publisher(
            Bool, self.active_topic, activity_qos
        )
        self.pub_hoop_active = self.create_publisher(
            Bool, self.hoop_active_topic, activity_qos
        )
        self.sub_realsense_yolo_state = self.create_subscription(
            String,
            self.realsense_yolo_state_topic,
            self.cb_realsense_yolo_state,
            state_qos,
        )
        self.sub_webcam = self.create_subscription(
            String, self.webcam_state_topic, self.cb_webcam_state, 10
        )
        self.sub_hoop_state = self.create_subscription(
            String, self.hoop_state_topic, self.cb_hoop_state, state_qos
        )
        self.sub_raw_ball_in_hand = self.create_subscription(
            Bool,
            self.raw_ball_in_hand_topic,
            self.cb_raw_ball_in_hand,
            10,
        )
        self.ball_status_publisher = BallStatusPublisher(
            self, topic_name=self.ball_result_topic
        )
        self.pub_vision_state = self.create_publisher(
            String, self.vision_state_topic, 10
        )

        if self.manage_activity_from_ball_in_hand:
            self._set_vision_mode_from_ball_in_hand(False, force=True)
        self.timer = self.create_timer(
            1.0 / max(self.publish_hz, 1.0), self.publish_ball_features
        )
        self.get_logger().info(
            "BallVisionFusionNode started in YOLO-only mode."
        )

    def _clear_ball_detection_state(self) -> None:
        self.latest_realsense = None
        self.latest_realsense_time = 0.0
        self.latest_webcam = None
        self.latest_webcam_time = 0.0

    def cb_ball_active(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested == self.ball_detection_active:
            return
        self.ball_detection_active = requested
        self._clear_ball_detection_state()
        self.ball_status_publisher.set_detection_enabled(requested)
        self.get_logger().info(
            f"Ball YOLO state processing {'ON' if requested else 'OFF'}"
        )

    def _set_vision_mode_from_ball_in_hand(
        self, ball_in_hand: bool, *, force: bool = False
    ) -> bool:
        """노드를 재시작하지 않고 공/hoop 표시 및 판단 모드만 전환한다."""
        if not getattr(self, "manage_activity_from_ball_in_hand", True):
            return False
        hoop_active = bool(ball_in_hand)
        if self.managed_hoop_active == hoop_active and not force:
            return False

        if hoop_active:
            self.cb_ball_active(Bool(data=False))
            self.pub_ball_active.publish(Bool(data=False))
            self.pub_hoop_active.publish(Bool(data=True))
        else:
            self.pub_hoop_active.publish(Bool(data=False))
            self.cb_ball_active(Bool(data=True))
            self.pub_ball_active.publish(Bool(data=True))
        self.managed_hoop_active = hoop_active
        self.get_logger().info(
            "[VisionMode] "
            f"ball={'OFF' if hoop_active else 'ON'}, "
            f"hoop={'ON' if hoop_active else 'OFF'} (detectors kept alive)"
        )
        return True

    def cb_realsense_yolo_state(self, msg: String) -> None:
        if not getattr(self, "ball_detection_active", True):
            return
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warning("Invalid RealSense YOLO state JSON")
            return
        if not isinstance(payload, dict):
            return

        state = self._empty_realsense_state(True)
        state.update(payload)
        detected = bool(state.get("realsense_ball_detected", False))
        if detected:
            try:
                distance = float(state["realsense_ball_distance_cm"])
                angle = float(state["realsense_ball_angle_error"])
                detected = (
                    math.isfinite(distance)
                    and distance > 0.0
                    and math.isfinite(angle)
                )
            except (KeyError, TypeError, ValueError):
                detected = False
        if not detected:
            diagnostic = state.get("realsense_diagnostic")
            state = self._empty_realsense_state(True)
            if isinstance(diagnostic, dict):
                state["realsense_diagnostic"] = diagnostic
        self.latest_realsense = state
        self.latest_realsense_time = time.monotonic()

    @staticmethod
    def _empty_realsense_state(active: bool = True) -> Dict[str, Any]:
        return {
            "realsense_ball_detected": False,
            "realsense_ball_distance_cm": None,
            "realsense_ball_angle_error": None,
            "raw_x_m": None,
            "raw_y_m": None,
            "raw_z_m": None,
            "raw_ball_x": None,
            "raw_ball_y": None,
            "raw_ball_conf": 0.0,
            "raw_ball_bbox": [],
            "realsense_diagnostic": {
                "category": "waiting" if active else "inactive",
                "detail": (
                    "no_yolo_ball_detection" if active else "ball_mode_off"
                ),
            },
            "active": active,
            "source": "realsense_yolo",
            "process_ms": 0.0,
        }

    @staticmethod
    def _finite(payload: Dict[str, Any], key: str) -> Optional[float]:
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def cb_webcam_state(self, msg: String) -> None:
        if not getattr(self, "ball_detection_active", True):
            return
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warning("Invalid webcam YOLO state JSON")
            return
        if not isinstance(payload, dict):
            return
        if not bool(payload.get("ball_detected", False)):
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        ball_x = self._finite(payload, "ball_x")
        ball_y = self._finite(payload, "ball_y")
        if ball_x is None or ball_y is None or ball_x < 0.0 or ball_y < 0.0:
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        robot_x = self._finite(payload, "robot_center_x")
        robot_y = self._finite(payload, "robot_center_y")
        x_offset = self._finite(payload, "ball_x_offset_px")
        x_distance = self._finite(payload, "ball_x_distance_px")
        y_distance = self._finite(payload, "ball_y_distance_px")
        distance_px = self._finite(payload, "ball_distance_px")
        angle = self._finite(payload, "ball_angle_deg")
        if x_offset is None:
            x_offset = ball_x - (
                robot_x
                if robot_x is not None
                else self.webcam_robot_center_x
            )
        if x_distance is None:
            x_distance = x_offset
        elif x_offset == 0.0:
            x_distance = 0.0
        else:
            x_distance = math.copysign(abs(x_distance), x_offset)
        if y_distance is None:
            center_y = (
                robot_y
                if robot_y is not None
                else self.webcam_robot_center_y
            )
            y_distance = abs(center_y - ball_y)
        if distance_px is None:
            distance_px = math.hypot(x_distance, y_distance)
        if angle is None and 0.0 < self.webcam_fov_x_deg < 180.0:
            focal_px = self.webcam_frame_width / (
                2.0 * math.tan(math.radians(self.webcam_fov_x_deg) / 2.0)
            )
            angle = math.degrees(math.atan2(x_offset, focal_px))

        self.latest_webcam = {
            "webcam_ball_detected": True,
            "webcam_ball_x_distance": float(x_distance),
            "webcam_ball_y_distance": float(y_distance),
            "webcam_ball_angle_error": angle,
            "webcam_ball_distance_px": float(distance_px),
            "raw_ball_x": ball_x,
            "raw_ball_y": ball_y,
            "raw_ball_conf": self._finite(payload, "ball_conf") or 0.0,
            "raw_ball_bbox": payload.get("ball_bbox", []),
        }
        self.latest_webcam_time = now

    @staticmethod
    def _empty_webcam_state() -> Dict[str, Any]:
        return {
            "webcam_ball_detected": False,
            "webcam_ball_x_distance": None,
            "webcam_ball_y_distance": None,
            "webcam_ball_angle_error": None,
            "webcam_ball_distance_px": None,
            "raw_ball_x": None,
            "raw_ball_y": None,
            "raw_ball_conf": 0.0,
            "raw_ball_bbox": [],
        }

    def cb_hoop_state(self, msg: String) -> None:
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warning("Invalid hoop YOLO state JSON")
            return
        if not isinstance(payload, dict) or not bool(payload.get("detected")):
            self.latest_hoop = self._empty_hoop_state()
            self.latest_hoop_time = now
            return
        distance = self._finite(payload, "realsense_goal_distance_cm")
        angle = self._finite(payload, "realsense_goal_angle")
        if distance is None or distance <= 0.0 or angle is None:
            self.latest_hoop = self._empty_hoop_state()
        else:
            self.latest_hoop = {
                "hoop_detected": True,
                "realsense_goal_distance_cm": distance,
                "realsense_goal_angle": angle,
            }
        self.latest_hoop_time = now

    @staticmethod
    def _empty_hoop_state() -> Dict[str, Any]:
        return {
            "hoop_detected": False,
            "realsense_goal_distance_cm": None,
            "realsense_goal_angle": None,
        }

    def cb_raw_ball_in_hand(self, msg: Bool) -> None:
        self.ball_in_hand = bool(msg.data)

    def _published_realsense_diagnostic(
        self,
        realsense_valid: bool,
        realsense_age: Optional[float],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.ball_detection_active:
            return {"category": "inactive", "detail": "ball_mode_off"}
        if realsense_valid:
            distance = features.get("realsense_ball_distance_cm")
            limit = float(
                self.ball_status_publisher.ball_decision.ball_entry_distance_cm
            )
            if (
                not features["ball_in_hand"]
                and distance is not None
                and float(distance) > limit
            ):
                return {
                    "category": "distance",
                    "detail": f"ball_entry_gate_{float(distance):.1f}cm",
                    "distance_cm": float(distance),
                    "limit_cm": limit,
                }
        if (
            realsense_age is not None
            and realsense_age > self.realsense_timeout_sec
        ):
            return {
                "category": "timeout",
                "detail": f"last_frame_age_{realsense_age:.2f}s",
            }
        if self.latest_realsense is not None:
            value = self.latest_realsense.get("realsense_diagnostic")
            if isinstance(value, dict):
                return dict(value)
        return {"category": "waiting", "detail": "no_realsense_frame"}

    def _log_diagnostic(self, diagnostic: Dict[str, Any]) -> None:
        label = (
            f"{diagnostic.get('category', 'unknown')}:"
            f"{diagnostic.get('detail', 'unknown')}"
        )
        if label != self.last_realsense_diagnostic_label:
            self.last_realsense_diagnostic_label = label
            self.get_logger().info(f"[RealSenseBallDiagnostic] {label}")

    def publish_ball_features(self) -> None:
        now = time.monotonic()
        rs_age = self._age(
            now, self.latest_realsense, self.latest_realsense_time
        )
        webcam_age = self._age(
            now, self.latest_webcam, self.latest_webcam_time
        )
        hoop_age = self._age(now, self.latest_hoop, self.latest_hoop_time)
        rs_valid = bool(
            self.ball_detection_active
            and self.latest_realsense is not None
            and rs_age is not None
            and rs_age <= self.realsense_timeout_sec
            and self.latest_realsense.get("realsense_ball_detected")
        )
        webcam_valid = bool(
            self.ball_detection_active
            and self.latest_webcam is not None
            and webcam_age is not None
            and webcam_age <= self.webcam_timeout_sec
            and self.latest_webcam.get("webcam_ball_detected")
        )
        hoop_valid = bool(
            self.latest_hoop is not None
            and hoop_age is not None
            and hoop_age <= self.hoop_timeout_sec
            and self.latest_hoop.get("hoop_detected")
        )

        features: Dict[str, Any] = {
            "realsense_ball_detected": False,
            "realsense_ball_distance_cm": None,
            "realsense_ball_angle_error": None,
            "webcam_ball_detected": False,
            "webcam_ball_x_distance": None,
            "webcam_ball_y_distance": None,
            "webcam_ball_angle_error": None,
            "webcam_ball_distance_px": None,
            "ball_in_hand": bool(self.ball_in_hand),
            "realsense_goal_distance_cm": None,
            "realsense_goal_angle": None,
        }
        if rs_valid:
            for key in (
                "realsense_ball_detected",
                "realsense_ball_distance_cm",
                "realsense_ball_angle_error",
            ):
                features[key] = self.latest_realsense[key]
        if webcam_valid:
            for key in (
                "webcam_ball_detected",
                "webcam_ball_x_distance",
                "webcam_ball_y_distance",
                "webcam_ball_angle_error",
                "webcam_ball_distance_px",
            ):
                features[key] = self.latest_webcam[key]
        if hoop_valid:
            for key in ("realsense_goal_distance_cm", "realsense_goal_angle"):
                features[key] = self.latest_hoop[key]

        diagnostic = self._published_realsense_diagnostic(
            rs_valid, rs_age, features
        )
        self._log_diagnostic(diagnostic)
        status, angle = self.ball_status_publisher.publish_ball_status(
            **features
        )
        self._set_vision_mode_from_ball_in_hand(
            self.ball_status_publisher.ball_in_hand
        )
        source = (
            "webcam"
            if webcam_valid
            else "realsense"
            if rs_valid
            else "none"
        )
        output = dict(features)
        output.update(
            {
                "source_priority": source,
                "ball_detection_active": self.ball_detection_active,
                "realsense_detection_method": "yolo_depth",
                "realsense_age_sec": rs_age,
                "webcam_age_sec": webcam_age,
                "hoop_age_sec": hoop_age,
                "hoop_detected": hoop_valid,
                "ball_status": int(status),
                "ball_status_angle": float(angle),
                "realsense_diagnostic": diagnostic,
            }
        )
        if rs_valid:
            output["realsense_raw"] = {
                key: self.latest_realsense.get(key)
                for key in (
                    "raw_x_m",
                    "raw_y_m",
                    "raw_z_m",
                    "raw_ball_x",
                    "raw_ball_y",
                    "raw_ball_conf",
                    "raw_ball_bbox",
                    "process_ms",
                )
            }
        self.pub_vision_state.publish(
            String(data=json.dumps(output, ensure_ascii=False))
        )

        self.frame_count += 1
        if self.frame_count % self.print_every_n_frames == 0:
            self.get_logger().info(
                f"ball_vision src={source} "
                f"rs={features['realsense_ball_detected']} "
                f"webcam={features['webcam_ball_detected']} "
                f"hand={features['ball_in_hand']} "
                f"status={status} angle={angle:.2f}"
            )

    @staticmethod
    def _age(
        now: float, value: Optional[Dict[str, Any]], stamp: float
    ) -> Optional[float]:
        return now - stamp if value is not None else None


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
