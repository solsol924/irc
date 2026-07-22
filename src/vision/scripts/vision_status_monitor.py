#!/usr/bin/env python3
"""라인·공·허들·후프 판단과 fusion 상태를 한 화면에 요약하는 ROS2 노드."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from msgs.msg import BallResult, HurdleResult, LineResult


class VisionStatusMonitor(Node):
    def __init__(self) -> None:
        super().__init__("vision_status_monitor")

        self.declare_parameter("print_hz", 1.0)
        self.declare_parameter("stale_timeout_sec", 2.0)

        self.print_hz = max(0.2, float(self.get_parameter("print_hz").value))
        self.stale_timeout_sec = max(
            0.1, float(self.get_parameter("stale_timeout_sec").value)
        )

        self.line_result: Optional[Dict[str, Any]] = None
        self.ball_result: Optional[Dict[str, Any]] = None
        self.hurdle_result: Optional[Dict[str, Any]] = None
        self.line_state: Optional[Dict[str, Any]] = None
        self.ball_state: Optional[Dict[str, Any]] = None
        self.hurdle_state: Optional[Dict[str, Any]] = None
        self.hoop_state: Optional[Dict[str, Any]] = None

        self.timestamps: Dict[str, float] = {}

        self.create_subscription(LineResult, "line_result", self.cb_line_result, 10)
        self.create_subscription(BallResult, "ball_result", self.cb_ball_result, 10)
        self.create_subscription(HurdleResult, "hurdle_result", self.cb_hurdle_result, 10)
        self.create_subscription(String, "/line_tracker/state", self.cb_line_state, 10)
        self.create_subscription(String, "/ball/vision_state", self.cb_ball_state, 10)
        self.create_subscription(String, "/hurdle/vision_state", self.cb_hurdle_state, 10)
        self.create_subscription(String, "/hoop/vision_state", self.cb_hoop_state, 10)

        self.create_timer(1.0 / self.print_hz, self.print_summary)
        self.get_logger().info(
            "VisionStatusMonitor started: line, ball, hurdle, hoop"
        )

    def _stamp(self, key: str) -> None:
        self.timestamps[key] = time.monotonic()

    def cb_line_result(self, msg: LineResult) -> None:
        self.line_result = {
            "status": int(msg.status),
            "angle": float(msg.angle),
            "follow_point": bool(getattr(msg, "follow_point", False)),
        }
        self._stamp("line_result")

    def cb_ball_result(self, msg: BallResult) -> None:
        self.ball_result = {
            "status": int(msg.status),
            "angle": float(msg.angle),
            "ball_in_hand": bool(getattr(msg, "ball_in_hand", False)),
        }
        self._stamp("ball_result")

    def cb_hurdle_result(self, msg: HurdleResult) -> None:
        self.hurdle_result = {
            "status": int(msg.status),
            "angle": float(msg.angle),
        }
        self._stamp("hurdle_result")

    @staticmethod
    def _parse_json(msg: String) -> Optional[Dict[str, Any]]:
        try:
            value = json.loads(msg.data)
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def cb_line_state(self, msg: String) -> None:
        parsed = self._parse_json(msg)
        if parsed is not None:
            self.line_state = parsed
            self._stamp("line_state")

    def cb_ball_state(self, msg: String) -> None:
        parsed = self._parse_json(msg)
        if parsed is not None:
            self.ball_state = parsed
            self._stamp("ball_state")

    def cb_hurdle_state(self, msg: String) -> None:
        parsed = self._parse_json(msg)
        if parsed is not None:
            self.hurdle_state = parsed
            self._stamp("hurdle_state")

    def cb_hoop_state(self, msg: String) -> None:
        parsed = self._parse_json(msg)
        if parsed is not None:
            self.hoop_state = parsed
            self._stamp("hoop_state")

    def _fresh(self, key: str) -> bool:
        stamp = self.timestamps.get(key)
        return stamp is not None and (time.monotonic() - stamp) <= self.stale_timeout_sec

    @staticmethod
    def _fmt_result(value: Optional[Dict[str, Any]], fresh: bool) -> str:
        if value is None:
            return "WAIT"
        if not fresh:
            return "STALE"
        return f"status={value.get('status')} angle={value.get('angle', 0.0):+.1f}"

    def print_summary(self) -> None:
        line_text = self._fmt_result(self.line_result, self._fresh("line_result"))
        ball_text = self._fmt_result(self.ball_result, self._fresh("ball_result"))
        hurdle_text = self._fmt_result(
            self.hurdle_result, self._fresh("hurdle_result")
        )

        line_extra = ""
        if self.line_state is not None and self._fresh("line_state"):
            line_extra = (
                f" pc={self.line_state.get('point_count')}"
                f" B={int(bool(self.line_state.get('ball_detected', False)))}"
                f" H={int(bool(self.line_state.get('hurdle_detected', False)))}"
            )

        ball_extra = ""
        if self.ball_state is not None and self._fresh("ball_state"):
            ball_extra = (
                f" src={self.ball_state.get('source_priority', 'none')}"
                f" rs={int(bool(self.ball_state.get('realsense_ball_detected', False)))}"
                f" cam={int(bool(self.ball_state.get('webcam_ball_detected', False)))}"
            )

        hurdle_extra = ""
        if self.hurdle_state is not None and self._fresh("hurdle_state"):
            hurdle_extra = (
                f" src={self.hurdle_state.get('source', 'none')}"
                f" cam={int(bool(self.hurdle_state.get('webcam_valid', False)))}"
                f" cross={int(bool(self.hurdle_state.get('intersection_valid', False)))}"
                f" signed={self.hurdle_state.get('signed_angle_deg', 0.0):+.1f}"
            )

        hoop_text = "WAIT"
        if self.hoop_state is not None:
            if not self._fresh("hoop_state"):
                hoop_text = "STALE"
            elif bool(self.hoop_state.get("detected", False)):
                distance = self.hoop_state.get("distance_cm")
                angle = self.hoop_state.get("center_angle_deg")
                yaw = self.hoop_state.get("yaw_deg")
                distance_text = (
                    f"{float(distance):.1f}cm" if distance is not None else "N/A"
                )
                angle_text = (
                    f"{float(angle):+.1f}" if angle is not None else "N/A"
                )
                yaw_text = f"{float(yaw):+.1f}" if yaw is not None else "N/A"
                hoop_text = (
                    f"det=1 dist={distance_text} "
                    f"angle={angle_text} yaw={yaw_text}"
                )
            else:
                hoop_text = "det=0"

        self.get_logger().info(
            "[VISION CHECK] "
            f"LINE({line_text}{line_extra}) | "
            f"BALL({ball_text}{ball_extra}) | "
            f"HURDLE({hurdle_text}{hurdle_extra}) | "
            f"HOOP({hoop_text})"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionStatusMonitor()
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
