#!/usr/bin/env python3
"""웹캠 전용 허들 결과 발행기입니다.

웹캠 YOLO 노드는 동일한 프레임에서 얻은 선과 허들 정보를
``/line_tracker/state``로 발행합니다. 이 노드는 해당 상태를 검증하고
``hurdle_status_publisher``의 거리 및 각도 판정을 적용한 뒤
``hurdle_result``를 발행합니다.

RealSense 허들 검출은 의도적으로 사용하지 않습니다. 아무 작업도 하지 않는
``/hurdle/recalibrate`` 서비스는 모션 시작 절차와의 호환성을 위해 유지합니다.
웹캠 YOLO에는 바닥 및 깊이 보정이 필요하지 않습니다.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from hurdle_status_publisher import HurdleStatusPublisher


class WebcamHurdlePublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("webcam_hurdle_publisher")

        self.declare_parameter("webcam_state_topic", "/line_tracker/state")
        self.declare_parameter("vision_state_topic", "/hurdle/vision_state")
        self.declare_parameter("hurdle_result_topic", "hurdle_result")
        self.declare_parameter("recalibrate_service", "/hurdle/recalibrate")
        self.declare_parameter("webcam_timeout_sec", 0.6)
        self.declare_parameter("webcam_min_conf", 0.0)
        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("print_every_n_frames", 10)
        self.declare_parameter("post_crossing_cooldown_sec", 2.0)

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
        self.publish_hz = max(
            1.0,
            float(self.get_parameter("publish_hz").value),
        )
        self.print_every_n_frames = max(
            1,
            int(self.get_parameter("print_every_n_frames").value),
        )
        self.post_crossing_cooldown_sec = max(
            0.0,
            float(
                self.get_parameter(
                    "post_crossing_cooldown_sec"
                ).value
            ),
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
        self.hurdle_status_publisher = HurdleStatusPublisher(
            self,
            self.hurdle_result_topic,
            post_crossing_cooldown_sec=self.post_crossing_cooldown_sec,
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
            f"input={self.webcam_state_topic}, "
            f"output={self.hurdle_result_topic}"
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
            "line_point_count": 0,
            "line_follow_angle_deg": None,
            "line_second_point_distance_px": None,
            "hurdle_line_angle_deg": None,
        }

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _optional_finite_float(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _nonnegative_int(value: Any) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, number)

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
            "line_point_count": self._nonnegative_int(
                payload.get("point_count", 0)
            ),
            "line_follow_angle_deg": self._optional_finite_float(
                payload.get("follow_angle")
            ),
            "line_second_point_distance_px": self._optional_finite_float(
                payload.get("line_second_point_distance_px")
            ),
            "hurdle_line_angle_deg": self._optional_finite_float(
                payload.get("hurdle_line_angle_deg")
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
            "Webcam-only hurdle vision: "
            "RealSense floor calibration is not required."
        )
        self.get_logger().info(response.message)
        return response

    def publish_hurdle_features(self) -> None:
        now = time.monotonic()
        webcam_age = (
            now - self.latest_webcam_time
            if self.latest_webcam is not None
            else None
        )
        webcam_age_value = (
            webcam_age if webcam_age is not None else float("inf")
        )
        webcam_fresh = all(
            (
                self.latest_webcam is not None,
                webcam_age_value <= self.webcam_timeout_sec,
            )
        )
        latest_hurdle_detected = (
            bool(
                self.latest_webcam.get(
                    "webcam_hurdle_detected",
                    False,
                )
            )
            if self.latest_webcam is not None
            else False
        )
        webcam_detected = webcam_fresh and latest_hurdle_detected
        state = (
            dict(self.latest_webcam)
            if webcam_fresh and self.latest_webcam is not None
            else self._empty_webcam_state()
        )
        line_point_count = self._nonnegative_int(
            state.get("line_point_count", 0)
        )
        line_follow_angle = self._optional_finite_float(
            state.get("line_follow_angle_deg")
        )
        second_point_distance = self._finite_float(
            state.get("line_second_point_distance_px"),
            0.0,
        )
        line_angle = self._finite_float(
            state.get("hurdle_line_angle_deg"),
            0.0,
        )

        status, published_angle, hurdle_ready = (
            self.hurdle_status_publisher.publish_hurdle_status(
                hurdle_detected=webcam_detected,
                line_point_count=line_point_count,
                line_follow_angle_deg=line_follow_angle,
                line_second_point_distance_px=second_point_distance,
                line_angle_deg=line_angle,
            )
        )
        suppression_reason = (
            self.hurdle_status_publisher.suppression_reason()
        )
        cooldown_remaining_sec = (
            self.hurdle_status_publisher.cooldown_remaining_sec()
        )
        published_sign = (
            -1
            if published_angle < 0.0
            else (1 if published_angle > 0.0 else 0)
        )

        output: Dict[str, Any] = {
            "source": "webcam" if webcam_detected else "none",
            "webcam_valid": webcam_detected,
            "webcam_age_sec": webcam_age,
            "signed_angle_deg": float(published_angle),
            "angle_sign": published_sign,
            "angle_direction": (
                "left"
                if published_angle < 0.0
                else ("right" if published_angle > 0.0 else "center")
            ),
            "hurdle_status": int(status),
            "hurdle_status_angle": float(published_angle),
            "hurdle_ready": bool(hurdle_ready),
            "detection_suppressed": suppression_reason is not None,
            "suppression_reason": suppression_reason,
            "cooldown_remaining_sec": cooldown_remaining_sec,
            "line_point_count": line_point_count,
            "line_follow_angle_deg": line_follow_angle,
            "line_second_point_distance_px": second_point_distance,
            "hurdle_line_angle_deg": line_angle,
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
                f"line_points={line_point_count} "
                f"distance={second_point_distance} "
                f"angle={published_angle:+.1f}deg "
                f"status={status} "
                f"ready={int(hurdle_ready)} "
                f"suppressed={suppression_reason or 'no'} "
                f"cooldown={cooldown_remaining_sec:.1f}s"
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
