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
        # None은 아직 비전 모드가 전달되지 않은 시작 상태를 뜻한다.
        # 이때는 끊김 없는 RealSense 원본 영상을 표시한다.
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
        # ball_vision_fusion이 transient-local로 발행하므로 같은 내구성
        # 정책을 사용하면 선택기만 재시작해도 현재 모드를 복구할 수 있다.
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
            # 재활성화할 때 이전 공 모드의 영상을 다시 사용하지 않는다.
            self.latest_ball_image = None
            self.latest_ball_image_time = 0.0
        if not self.ball_enabled:
            # 이전 경기 단계의 영상이 다음 검출기의 기본 영상을 막지 않게 한다.
            self.latest_ball_image = None
            self.latest_ball_image_time = 0.0
            if self.selected_source == "ball":
                self.selected_source = "default"

    def cb_hoop_active(self, msg: Bool) -> None:
        self.hoop_enabled = bool(msg.data)
        if self.hoop_enabled:
            self.selected_source = "hoop"
            # 저장된 옛 영상이 아니라 활성화 이후 생성된 새 영상을 기다린다.
            self.latest_hoop_image = None
            self.latest_hoop_image_time = 0.0
        if not self.hoop_enabled:
            self.latest_hoop_image = None
            self.latest_hoop_image_time = 0.0
            if self.selected_source == "hoop":
                self.selected_source = "default"

    def _active_source(self) -> str:
        # 현재 검출에 실패했더라도 실행 중인 검출기를 선택한다. 검출 성공 여부를
        # 활성 신호로 사용하면 hoop 전환 뒤에도 마지막 공 영상이 남을 수 있다.
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
        """끊김 없는 RealSense 컬러 영상을 기준으로 출력 주기를 유지한다.

        각 검출기의 디버그 콜백은 최신 영상 한 장만 저장한다. 원본 컬러
        프레임이 들어올 때 활성 모드의 새 디버그 영상이 있으면 그 영상을,
        없으면 원본을 발행해 모드 전환 중에도 출력 토픽이 멈추지 않게 한다.
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
