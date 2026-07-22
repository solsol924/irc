#!/usr/bin/env python3
"""Webcam-only hurdle result publisher.

The webcam YOLO node publishes line and hurdle information from the same frame
on ``/line_tracker/state``.  This node validates that state, forwards only the
webcam hurdle detection to ``hurdle_result``, and preserves the signed angle
from the robot image center to the line/hurdle intersection.

RealSense hurdle detection is intentionally not used.  The no-op
``/hurdle/recalibrate`` service remains for compatibility with the motion
startup sequence; webcam YOLO needs no floor/depth calibration.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Optional

import rclpy
from msgs.msg import HurdleResult
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from hurdle_status_publisher import HurdleStatus


class WebcamHurdlePublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("webcam_hurdle_publisher")

        self.declare_parameter("webcam_state_topic", "/line_tracker/state")
        self.declare_parameter("vision_state_topic", "/hurdle/vision_state")
        self.declare_parameter("hurdle_result_topic", "hurdle_result")
        self.declare_parameter("recalibrate_service", "/hurdle/recalibrate")
        self.declare_parameter("webcam_timeout_sec", 0.6)
        self.declare_parameter("webcam_min_conf", 0.35)
        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("print_every_n_frames", 10)

        self.webcam_state_topic = str(
            self.get_parameter("webcam_state_topic").value
        )
        self.vision_state_topic = str(
            self.get_parameter("vision_state_topic").value
        )
        self.hurdle_result_topic = str(
            self.get_parameter("hurdle_result_topic").value
        )
        self.recalibrate_service_name = str(
            self.get_parameter("recalibrate_service").value
        )
        self.webcam_timeout_sec = max(
            0.05,
            float(self.get_parameter("webcam_timeout_sec").value),
        )
        self.webcam_min_conf = max(
            0.0,
            float(self.get_parameter("webcam_min_conf").value),
        )
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.print_every_n_frames = max(
            1,
            int(self.get_parameter("print_every_n_frames").value),
        )

        self.latest_webcam: Optional[Dict[str, Any]] = None
        self.latest_webcam_time = 0.0
        self.publish_count = 0

        self.create_subscription(
            String,
            self.webcam_state_topic,
            self.cb_webcam_state,
            10,
        )
        self.hurdle_result_publisher = self.create_publisher(
            HurdleResult,
            self.hurdle_result_topic,
            10,
        )
        self.pub_vision_state = self.create_publisher(
            String,
            self.vision_state_topic,
            10,
        )
        self.create_service(
            Trigger,
            self.recalibrate_service_name,
            self.cb_recalibrate,
        )
        self.create_timer(1.0 / self.publish_hz, self.publish_hurdle_features)

        self.get_logger().info(
            "Webcam-only hurdle publisher started: "
            f"input={self.webcam_state_topic}, output={self.hurdle_result_topic}"
        )
        self.get_logger().info(
            "RealSense hurdle OpenCV is disabled; signed angle convention is "
            "left(-), right(+)."
        )

    @staticmethod
    def _empty_webcam_state() -> Dict[str, Any]:
        return {
            "webcam_hurdle_detected": False,
            "webcam_hurdle_x": None,
            "webcam_hurdle_y": None,
            "webcam_hurdle_conf": 0.0,
            "webcam_hurdle_bbox": [],
            "intersection_valid": False,
            "intersection_x": None,
            "intersection_y": None,
            "signed_angle_deg": 0.0,
            "angle_sign": 0,
            "angle_direction": "unknown",
        }

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    def cb_webcam_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn("Failed to parse /line_tracker/state JSON")
            return
        if not isinstance(payload, dict):
            return

        detected = bool(payload.get("hurdle_detected", False))
        confidence = self._finite_float(payload.get("hurdle_conf"), 0.0)
        detected = bool(detected and confidence >= self.webcam_min_conf)
        intersection_valid = bool(
            detected and payload.get("hurdle_line_intersection_valid", False)
        )
        signed_angle = (
            self._finite_float(payload.get("hurdle_intersection_angle_deg"), 0.0)
            if intersection_valid
            else 0.0
        )
        sign = -1 if signed_angle < 0.0 else (1 if signed_angle > 0.0 else 0)

        self.latest_webcam = {
            "webcam_hurdle_detected": detected,
            "webcam_hurdle_x": self._finite_float(
                payload.get("hurdle_x"), -1.0
            ),
            "webcam_hurdle_y": self._finite_float(
                payload.get("hurdle_y"), -1.0
            ),
            "webcam_hurdle_conf": confidence,
            "webcam_hurdle_bbox": payload.get("hurdle_bbox", []),
            "intersection_valid": intersection_valid,
            "intersection_x": (
                self._finite_float(payload.get("hurdle_line_intersection_x"), -1.0)
                if intersection_valid
                else None
            ),
            "intersection_y": (
                self._finite_float(payload.get("hurdle_line_intersection_y"), -1.0)
                if intersection_valid
                else None
            ),
            "signed_angle_deg": signed_angle,
            "angle_sign": sign,
            "angle_direction": (
                "left" if sign < 0 else ("right" if sign > 0 else "center")
            ),
        }
        self.latest_webcam_time = time.monotonic()

    def cb_recalibrate(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        response.success = True
        response.message = (
            "Webcam-only hurdle vision: RealSense floor calibration is not required."
        )
        self.get_logger().info(response.message)
        return response

    def publish_hurdle_result(
        self,
        hurdle_detected: bool,
        signed_angle_deg: float,
    ) -> tuple[int, float]:
        status = (
            HurdleStatus.Hurdle_Detected
            if hurdle_detected
            else HurdleStatus.Hurdle_None
        )
        angle = self._finite_float(signed_angle_deg, 0.0) if hurdle_detected else 0.0

        msg = HurdleResult()
        msg.status = int(status)
        msg.angle = float(angle)
        self.hurdle_result_publisher.publish(msg)
        return status, angle

    def publish_hurdle_features(self) -> None:
        now = time.monotonic()
        webcam_age = (
            now - self.latest_webcam_time
            if self.latest_webcam is not None
            else None
        )
        webcam_fresh = bool(
            self.latest_webcam is not None
            and webcam_age is not None
            and webcam_age <= self.webcam_timeout_sec
        )
        webcam_detected = bool(
            webcam_fresh
            and self.latest_webcam is not None
            and self.latest_webcam.get("webcam_hurdle_detected", False)
        )
        state = (
            dict(self.latest_webcam)
            if webcam_fresh and self.latest_webcam is not None
            else self._empty_webcam_state()
        )
        intersection_valid = bool(
            webcam_detected and state.get("intersection_valid", False)
        )
        signed_angle = (
            self._finite_float(state.get("signed_angle_deg"), 0.0)
            if intersection_valid
            else 0.0
        )

        status, published_angle = self.publish_hurdle_result(
            hurdle_detected=webcam_detected,
            signed_angle_deg=signed_angle,
        )

        output: Dict[str, Any] = {
            "source": "webcam" if webcam_detected else "none",
            "webcam_valid": webcam_detected,
            "webcam_age_sec": webcam_age,
            "intersection_valid": intersection_valid,
            "signed_angle_deg": float(published_angle),
            "angle_sign": int(state.get("angle_sign", 0)) if intersection_valid else 0,
            "angle_direction": (
                state.get("angle_direction", "unknown")
                if intersection_valid
                else "unknown"
            ),
            "hurdle_status": int(status),
            "hurdle_status_angle": float(published_angle),
            "webcam": state,
        }
        self.pub_vision_state.publish(
            String(data=json.dumps(output, ensure_ascii=False))
        )

        self.publish_count += 1
        if self.publish_count % self.print_every_n_frames == 0:
            self.get_logger().info(
                "hurdle_webcam "
                f"detected={int(webcam_detected)} "
                f"cross={int(intersection_valid)} "
                f"angle={published_angle:+.1f}deg "
                f"status={status}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebcamHurdlePublisherNode()
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
