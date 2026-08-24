#!/usr/bin/env python3

"""공/허들/후프 디버그 영상 중 현재 활성화된 화면 하나를 선택한다."""

import json
import time
from typing import Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class RealSenseDebugSelector(Node):
    def __init__(self) -> None:
        super().__init__("realsense_debug_selector")

        self.declare_parameter(
            "ball_debug_topic",
            "/ball/realsense_debug_image",
        )
        self.declare_parameter(
            "hurdle_debug_topic",
            "/hurdle/realsense_debug_image",
        )
        self.declare_parameter("hoop_debug_topic", "/hoop/debug_image")
        self.declare_parameter("raw_color_topic", "/camera/color/image_raw")
        self.declare_parameter("ball_state_topic", "/ball/vision_state")
        self.declare_parameter("hurdle_state_topic", "/hurdle/vision_state")
        self.declare_parameter("hoop_state_topic", "/hoop/vision_state")
        self.declare_parameter("ball_active_topic", "/vision/ball_active")
        self.declare_parameter("hoop_active_topic", "/vision/hoop_active")
        self.declare_parameter(
            "output_topic",
            "/vision/realsense_debug_image",
        )
        self.declare_parameter("state_timeout_sec", 0.5)
        self.declare_parameter("debug_timeout_sec", 0.25)
        self.declare_parameter("show_window", True)
        self.declare_parameter(
            "window_name",
            "RealSense Ball / Hurdle / Hoop Vision",
        )

        self.state_timeout_sec = float(
            self.get_parameter("state_timeout_sec").value
        )
        self.debug_timeout_sec = max(
            0.0,
            float(self.get_parameter("debug_timeout_sec").value),
        )
        self.show_window = bool(self.get_parameter("show_window").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.bridge = CvBridge()
        self.ball_detected = False
        self.hurdle_detected = False
        self.hoop_detected = False
        # None means that the decision node has not announced a mode yet.
        # Until then, the uninterrupted raw RealSense stream is shown.
        self.ball_enabled: Optional[bool] = None
        self.hoop_enabled: Optional[bool] = None
        self.selected_source = "default"
        self.ball_state_time = 0.0
        self.hurdle_state_time = 0.0
        self.hoop_state_time = 0.0
        self.latest_ball_image: Optional[Image] = None
        self.latest_hurdle_image: Optional[Image] = None
        self.latest_hoop_image: Optional[Image] = None
        self.latest_ball_image_time = 0.0
        self.latest_hurdle_image_time = 0.0
        self.latest_hoop_image_time = 0.0
        self.latest_raw_image: Optional[Image] = None
        self.last_debug_output_time = 0.0
        self.image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub_image = self.create_publisher(
            Image,
            str(self.get_parameter("output_topic").value),
            self.image_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("raw_color_topic").value),
            self.cb_raw_image,
            self.image_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("ball_debug_topic").value),
            self.cb_ball_image,
            self.image_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("hurdle_debug_topic").value),
            self.cb_hurdle_image,
            self.image_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("hoop_debug_topic").value),
            self.cb_hoop_image,
            self.image_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("ball_state_topic").value),
            self.cb_ball_state,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("hurdle_state_topic").value),
            self.cb_hurdle_state,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("hoop_state_topic").value),
            self.cb_hoop_state,
            10,
        )
        # ball_vision_fusion publishes these as transient-local values. Matching
        # that durability restores the current mode if this selector restarts.
        vision_mode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("ball_active_topic").value),
            self.cb_ball_active,
            vision_mode_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("hoop_active_topic").value),
            self.cb_hoop_active,
            vision_mode_qos,
        )

        self.get_logger().info(
            "RealSense debug selector started: "
            "/vision/realsense_debug_image, "
            f"show_window={self.show_window}"
        )

    def _publish_and_show(self, msg: Image) -> None:
        self.pub_image.publish(msg)
        if not self.show_window:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rclpy.shutdown()
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to show selected RealSense debug image: {exc}"
            )

    def cb_ball_state(self, msg: String) -> None:
        try:
            state = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        self.ball_detected = bool(
            state.get("realsense_ball_detected", False)
        )
        self.ball_state_time = time.monotonic()

    def cb_hurdle_state(self, msg: String) -> None:
        try:
            state = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        self.hurdle_detected = bool(
            state.get("realsense_valid", False)
            or state.get("fused_hurdle_detected", False)
        )
        self.hurdle_state_time = time.monotonic()

    def cb_hoop_state(self, msg: String) -> None:
        try:
            state = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        self.hoop_detected = bool(state.get("detected", False))
        self.hoop_state_time = time.monotonic()

    def cb_ball_active(self, msg: Bool) -> None:
        self.ball_enabled = bool(msg.data)
        if self.ball_enabled:
            self.selected_source = "ball"
            # A previous ball phase must never be reused after reactivation.
            self.latest_ball_image = None
            self.latest_ball_image_time = 0.0
        if not self.ball_enabled:
            # Do not let an image from the previous match phase block the
            # next detector's default stream.
            self.latest_ball_image = None
            self.latest_ball_image_time = 0.0
            if self.selected_source == "ball":
                self.selected_source = "default"

    def cb_hoop_active(self, msg: Bool) -> None:
        self.hoop_enabled = bool(msg.data)
        if self.hoop_enabled:
            self.selected_source = "hoop"
            # Wait for a frame produced after this activation, not a cached one.
            self.latest_hoop_image = None
            self.latest_hoop_image_time = 0.0
        if not self.hoop_enabled:
            self.latest_hoop_image = None
            self.latest_hoop_image_time = 0.0
            if self.selected_source == "hoop":
                self.selected_source = "default"

    def _active_source(self) -> str:
        # Select the detector that is running, even while it currently has no
        # detection. Detection state is not an activity signal: using it here
        # made the window retain the last ball frame after switching to hoop.
        selected_source = getattr(self, "selected_source", "default")
        if selected_source in {"ball", "hoop"}:
            return selected_source
        if self.hoop_enabled is True:
            return "hoop"
        if self.ball_enabled is True:
            return "ball"

        now = time.monotonic()
        hurdle_active = bool(
            self.hurdle_detected
            and now - self.hurdle_state_time <= self.state_timeout_sec
        )
        if hurdle_active:
            return "hurdle"
        return "default"

    def _fresh_debug_image(
        self,
        source: str,
        now: float,
    ) -> tuple[Optional[Image], float]:
        image = getattr(self, f"latest_{source}_image", None)
        received_at = float(
            getattr(self, f"latest_{source}_image_time", 0.0)
        )
        if image is None or received_at <= 0.0:
            return None, 0.0
        if now - received_at > self.debug_timeout_sec:
            return None, 0.0
        return image, received_at

    def cb_raw_image(self, msg: Image) -> None:
        """Drive the output from the uninterrupted RealSense color stream.

        Detector debug callbacks only cache their newest image.  Every raw color
        frame publishes either one newly arrived active-mode debug frame or the
        raw frame itself, so changing modes cannot leave the output topic silent
        while the next detector produces its first result.
        """
        self.latest_raw_image = msg
        now = time.monotonic()
        source = self._active_source()
        selected: Optional[Image] = None
        received_at = 0.0
        if source in {"ball", "hurdle", "hoop"}:
            selected, received_at = self._fresh_debug_image(source, now)

        if selected is not None and received_at > self.last_debug_output_time:
            self.last_debug_output_time = received_at
            self._publish_and_show(selected)
            return

        self._publish_and_show(msg)

    def cb_ball_image(self, msg: Image) -> None:
        if self.ball_enabled is not True:
            return
        self.latest_ball_image = msg
        self.latest_ball_image_time = time.monotonic()

    def cb_hurdle_image(self, msg: Image) -> None:
        self.latest_hurdle_image = msg
        self.latest_hurdle_image_time = time.monotonic()

    def cb_hoop_image(self, msg: Image) -> None:
        if self.hoop_enabled is not True:
            return
        self.latest_hoop_image = msg
        self.latest_hoop_image_time = time.monotonic()

    def destroy_node(self):
        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealSenseDebugSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
