#!/usr/bin/env python3
"""
RealSense YOLO detector for IRC.

Inputs:
- /camera/color/image_raw
- /camera/aligned_depth_to_color/image_raw
- /camera/color/camera_info
- /vision/ball_active
- /vision/hoop_active

Outputs:
- /realsense_yolo/ball_state
- /hoop/vision_state
- /realsense_yolo/ball_detected
- /hoop/detected
- /vision/realsense_ball_image
- /vision/realsense_hoop_image
- /vision/realsense_combined_image
- /vision/realsense_debug_image (currently active mode)
- /ball/realsense_debug_image
- /hoop/debug_image

Expected model classes:
- goal
- backboard
- ball
"""

from __future__ import annotations

import configparser
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


YELLOW = (0, 235, 255)
CYAN = (245, 235, 180)
WHITE = (255, 255, 255)
PANEL_BG = (18, 18, 18)


def _display_number(
    value: object,
    suffix: str = "",
    digits: int = 1,
) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}{suffix}"


def draw_info_panel(
    frame: np.ndarray,
    lines: List[str],
    *,
    align: str = "left",
) -> None:
    """Draw the old OpenCV-style information panel with a yellow border."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    padding_x, padding_y = 8, 7
    line_gap = 6
    sizes = [
        cv2.getTextSize(line, font, scale, thickness)[0]
        for line in lines
    ]
    text_height = max((size[1] for size in sizes), default=10)
    width = max((size[0] for size in sizes), default=80) + padding_x * 2
    height = (
        padding_y * 2
        + len(lines) * text_height
        + max(0, len(lines) - 1) * line_gap
    )
    x1 = 6 if align == "left" else max(6, frame.shape[1] - width - 6)
    y1 = 6
    x2 = min(frame.shape[1] - 2, x1 + width)
    y2 = min(frame.shape[0] - 2, y1 + height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PANEL_BG, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)

    baseline = y1 + padding_y + text_height
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x1 + padding_x, baseline + index * (text_height + line_gap)),
            font,
            scale,
            WHITE,
            thickness,
            cv2.LINE_AA,
        )


@dataclass
class Detection:
    name: str
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


def load_config(ini_path: str) -> Dict[str, object]:
    defaults: Dict[str, object] = {
        "model": "/home/rnd/realsense_goal_backboard_ball_best.engine",
        "conf": 0.20,
        "goal_conf": 0.30,
        "backboard_conf": 0.30,
        "ball_conf": 0.25,
        "imgsz": 640,
        "device": "0",
        "max_fps": 10.0,
        "goal_class": "goal",
        "backboard_class": "backboard",
        "ball_class": "ball",
        "depth_scale": 0.001,
        "depth_min_m": 0.08,
        "depth_max_m": 3.0,
        "depth_inner_ratio_ball": 0.45,
        "depth_inner_ratio_hoop": 0.30,
        "min_valid_depth_pixels": 5,
        "publish_debug_image": True,
        "print_every_n_frames": 10,
    }

    p = Path(ini_path).expanduser()
    if not p.exists():
        print(f"[WARN] {p} not found -> using defaults")
        return defaults

    ini = configparser.ConfigParser()
    ini.read(p, encoding="utf-8")
    section = "realsense_yolo"
    if not ini.has_section(section):
        print(f"[WARN] [{section}] not found -> using defaults")
        return defaults

    cfg = dict(defaults)

    def gs(k: str) -> str:
        return ini.get(section, k, fallback=str(defaults[k]))

    def gf(k: str) -> float:
        return ini.getfloat(section, k, fallback=float(defaults[k]))

    def gi(k: str) -> int:
        return ini.getint(section, k, fallback=int(defaults[k]))

    def gb(k: str) -> bool:
        return ini.getboolean(section, k, fallback=bool(defaults[k]))

    cfg.update(
        {
            "model": gs("model"),
            "conf": gf("conf"),
            "goal_conf": gf("goal_conf"),
            "backboard_conf": gf("backboard_conf"),
            "ball_conf": gf("ball_conf"),
            "imgsz": gi("imgsz"),
            "device": gs("device"),
            "max_fps": gf("max_fps"),
            "goal_class": gs("goal_class"),
            "backboard_class": gs("backboard_class"),
            "ball_class": gs("ball_class"),
            "depth_scale": gf("depth_scale"),
            "depth_min_m": gf("depth_min_m"),
            "depth_max_m": gf("depth_max_m"),
            "depth_inner_ratio_ball": gf("depth_inner_ratio_ball"),
            "depth_inner_ratio_hoop": gf("depth_inner_ratio_hoop"),
            "min_valid_depth_pixels": gi("min_valid_depth_pixels"),
            "publish_debug_image": gb("publish_debug_image"),
            "print_every_n_frames": gi("print_every_n_frames"),
        }
    )
    return cfg


class RealSenseYoloDetector(Node):
    def __init__(self, ini_path: str) -> None:
        super().__init__("realsense_yolo_detector")

        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")

        self.cfg = load_config(ini_path)
        model_path = str(self.cfg["model"])
        self.get_logger().info(f"Loading RealSense YOLO: {model_path}")
        self.model = YOLO(model_path, task="detect")

        self.bridge = CvBridge()

        # Fallback intrinsics; replaced by CameraInfo as soon as available.
        self.fx = 607.0
        self.fy = 606.0
        self.cx_intr = 325.5
        self.cy_intr = 239.4
        self.camera_info_received = False
        self.centerline_x_px = 358.0

        # Safe standalone defaults. main_decision / ball fusion overwrite via
        # TRANSIENT_LOCAL activity topics when running the full stack.
        self.ball_active = True
        self.hoop_active = False
        # OFF -> ON 순서로 모드를 전환하는 짧은 구간에도 마지막 화면을
        # 유지한다. 화면 토픽 이름은 바뀌지 않고 패널 내용만 전환된다.
        self.display_mode = "ball"
        self.last_inference_time = 0.0
        self.frame_count = 0
        self.latest_ball_detection: Optional[Detection] = None
        self.latest_backboard_detection: Optional[Detection] = None
        self.latest_goal_detection: Optional[Detection] = None
        self.latest_ball_state = self._empty_ball_state(True)
        self.latest_hoop_state = self._empty_hoop_state(False)
        self.latest_process_ms = 0.0

        self.image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.activity_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # JSON 상태도 오래된 샘플을 쌓지 않는다. RELIABLE을 유지해 기존
        # ball_vision_fusion/decision 구독자와의 QoS 호환성은 보존한다.
        self.state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.ball_state_pub = self.create_publisher(
            String, "/realsense_yolo/ball_state", self.state_qos
        )
        self.hoop_state_pub = self.create_publisher(
            String, "/hoop/vision_state", self.state_qos
        )
        self.ball_detected_pub = self.create_publisher(
            Bool, "/realsense_yolo/ball_detected", 10
        )
        self.hoop_detected_pub = self.create_publisher(
            Bool, "/hoop/detected", 10
        )

        # 각 화면은 같은 YOLO 추론 결과로 만들며, 실제 구독자가 있을 때만
        # frame.copy/OpenCV drawing/image publish를 수행한다.
        self.ball_view_pub = self.create_publisher(
            Image, "/vision/realsense_ball_image", self.image_qos
        )
        self.hoop_view_pub = self.create_publisher(
            Image, "/vision/realsense_hoop_image", self.image_qos
        )
        self.combined_view_pub = self.create_publisher(
            Image, "/vision/realsense_combined_image", self.image_qos
        )
        self.selected_view_pub = self.create_publisher(
            Image, "/vision/realsense_debug_image", self.image_qos
        )

        # 기존 rqt 설정/외부 스크립트용 호환 토픽이다. 새 launch에서는
        # selector를 실행하지 않으므로 중복 영상 복사나 재발행이 없다.
        self.ball_debug_pub = self.create_publisher(
            Image, "/ball/realsense_debug_image", self.image_qos
        )
        self.hoop_debug_pub = self.create_publisher(
            Image, "/hoop/debug_image", self.image_qos
        )

        self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.cb_camera_info,
            10,
        )
        self.create_subscription(
            Bool,
            "/vision/ball_active",
            self.cb_ball_active,
            self.activity_qos,
        )
        self.create_subscription(
            Bool,
            "/vision/hoop_active",
            self.cb_hoop_active,
            self.activity_qos,
        )

        self.color_sub = Subscriber(
            self,
            Image,
            "/camera/color/image_raw",
            qos_profile=self.image_qos,
        )
        self.depth_sub = Subscriber(
            self,
            Image,
            "/camera/aligned_depth_to_color/image_raw",
            qos_profile=self.image_qos,
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=2,
            slop=0.05,
        )
        self.sync.registerCallback(self.cb_images)

        self.get_logger().info(
            "RealSenseYoloDetector started "
            f"max_fps={float(self.cfg['max_fps']):.1f}; "
            "rqt topics=ball, hoop, combined, active-mode"
        )

    def cb_camera_info(self, msg: CameraInfo) -> None:
        if len(msg.k) < 9:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        if fx <= 0.0 or fy <= 0.0:
            return

        self.fx = fx
        self.fy = fy
        self.cx_intr = float(msg.k[2])
        self.cy_intr = float(msg.k[5])

        if not self.camera_info_received:
            self.camera_info_received = True
            self.get_logger().info(
                f"CameraInfo fx={self.fx:.2f} fy={self.fy:.2f} "
                f"cx={self.cx_intr:.2f} cy={self.cy_intr:.2f}"
            )

    def cb_ball_active(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested:
            self.display_mode = "ball"
        if requested == self.ball_active:
            return
        self.ball_active = requested
        self.get_logger().info(
            f"RS YOLO ball {'ON' if requested else 'OFF'}"
        )
        if not requested:
            self._publish_ball_state(self._empty_ball_state(False))

    def cb_hoop_active(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested:
            self.display_mode = "hoop"
        if requested == self.hoop_active:
            return
        self.hoop_active = requested
        self.get_logger().info(
            f"RS YOLO hoop {'ON' if requested else 'OFF'}"
        )
        if not requested:
            self._publish_hoop_state(self._empty_hoop_state(False))

    @staticmethod
    def _result_name(result, cls_id: int) -> str:
        names = result.names
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        try:
            return str(names[cls_id])
        except Exception:
            return str(cls_id)

    def _run_yolo(self, frame: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            source=frame,
            imgsz=int(self.cfg["imgsz"]),
            conf=float(self.cfg["conf"]),
            device=str(self.cfg["device"]),
            verbose=False,
        )[0]

        if result.boxes is None:
            return []

        dets: List[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self._result_name(result, cls_id)
            x1, y1, x2, y2 = [
                float(v) for v in box.xyxy[0].tolist()
            ]

            threshold = float(self.cfg["conf"])
            if name == str(self.cfg["ball_class"]):
                threshold = float(self.cfg["ball_conf"])
            elif name == str(self.cfg["backboard_class"]):
                threshold = float(self.cfg["backboard_conf"])
            elif name == str(self.cfg["goal_class"]):
                threshold = float(self.cfg["goal_conf"])

            if conf < threshold:
                continue

            dets.append(
                Detection(name, cls_id, conf, x1, y1, x2, y2)
            )
        return dets

    @staticmethod
    def _best(dets: List[Detection], name: str) -> Optional[Detection]:
        candidates = [d for d in dets if d.name == name]
        return max(candidates, key=lambda d: d.conf) if candidates else None

    def _depth_m(
        self, depth_raw: np.ndarray, encoding: str
    ) -> np.ndarray:
        depth = np.asarray(depth_raw, dtype=np.float32)
        if str(encoding).upper() == "32FC1":
            return depth
        return depth * float(self.cfg["depth_scale"])

    def _sample_depth(
        self,
        depth_m: np.ndarray,
        det: Detection,
        inner_ratio: float,
    ) -> Optional[float]:
        h, w = depth_m.shape[:2]
        ratio = max(0.08, min(float(inner_ratio), 0.95))

        half_w = max(2, int(round(det.width * ratio * 0.5)))
        half_h = max(2, int(round(det.height * ratio * 0.5)))
        cx = int(round(det.cx))
        cy = int(round(det.cy))

        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w + 1)
        y1 = max(0, cy - half_h)
        y2 = min(h, cy + half_h + 1)

        patch = depth_m[y1:y2, x1:x2]
        valid = patch[
            np.isfinite(patch)
            & (patch >= float(self.cfg["depth_min_m"]))
            & (patch <= float(self.cfg["depth_max_m"]))
        ]

        minimum = max(1, int(self.cfg["min_valid_depth_pixels"]))
        if valid.size < minimum:
            radius = 5
            x1 = max(0, cx - radius)
            x2 = min(w, cx + radius + 1)
            y1 = max(0, cy - radius)
            y2 = min(h, cy + radius + 1)
            patch = depth_m[y1:y2, x1:x2]
            valid = patch[
                np.isfinite(patch)
                & (patch >= float(self.cfg["depth_min_m"]))
                & (patch <= float(self.cfg["depth_max_m"]))
            ]

        if valid.size < minimum:
            return None
        return float(np.median(valid))

    def _xyz(
        self, px: float, py: float, z_m: float
    ) -> Tuple[float, float, float]:
        x_m = (px - self.cx_intr) * z_m / max(self.fx, 1e-6)
        y_m = (py - self.cy_intr) * z_m / max(self.fy, 1e-6)
        return float(x_m), float(y_m), float(z_m)

    @staticmethod
    def _distance(x_m: float, y_m: float, z_m: float) -> float:
        return math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)

    def _make_ball_state(
        self,
        det: Optional[Detection],
        depth_m: np.ndarray,
        process_ms: float,
    ) -> Dict[str, object]:
        if det is None:
            state = self._empty_ball_state(self.ball_active)
            state["process_ms"] = process_ms
            return state

        z_m = self._sample_depth(
            depth_m,
            det,
            float(self.cfg["depth_inner_ratio_ball"]),
        )
        if z_m is None:
            state = self._empty_ball_state(self.ball_active)
            state["raw_ball_conf"] = det.conf
            state["raw_ball_bbox"] = [det.x1, det.y1, det.x2, det.y2]
            state["realsense_diagnostic"] = {
                "category": "depth",
                "detail": "yolo_ball_detected_but_depth_invalid",
            }
            state["process_ms"] = process_ms
            return state

        x_m, y_m, z_m = self._xyz(det.cx, det.cy, z_m)
        distance_cm = self._distance(x_m, y_m, z_m) * 100.0
        camera_angle_deg = math.degrees(math.atan2(x_m, z_m))
        centerline_angle_deg = math.degrees(
            math.atan2(self.centerline_x_px - self.cx_intr, self.fx)
        )
        angle_deg = camera_angle_deg - centerline_angle_deg

        return {
            "realsense_ball_detected": True,
            "realsense_ball_distance_cm": float(distance_cm),
            "realsense_ball_angle_error": float(angle_deg),
            "raw_x_m": x_m,
            "raw_y_m": y_m,
            "raw_z_m": z_m,
            "raw_ball_x": det.cx,
            "raw_ball_y": det.cy,
            "raw_ball_conf": det.conf,
            "raw_ball_bbox": [det.x1, det.y1, det.x2, det.y2],
            "held_previous_detection": False,
            "realsense_diagnostic": {
                "category": "accepted",
                "detail": "yolo_ball_with_valid_depth",
            },
            "active": self.ball_active,
            "source": "realsense_yolo",
            "process_ms": process_ms,
        }

    def _make_hoop_state(
        self,
        backboard: Optional[Detection],
        depth_m: np.ndarray,
        frame_h: int,
        process_ms: float,
    ) -> Dict[str, object]:
        det = backboard
        if det is None:
            state = self._empty_hoop_state(self.hoop_active)
            state["process_ms"] = process_ms
            return state

        z_m = self._sample_depth(
            depth_m,
            det,
            float(self.cfg["depth_inner_ratio_hoop"]),
        )
        if z_m is None:
            state = self._empty_hoop_state(self.hoop_active)
            state["confidence"] = det.conf
            state["target_class"] = det.name
            state["diagnostic"] = "yolo_hoop_detected_but_depth_invalid"
            state["process_ms"] = process_ms
            return state

        x_m, y_m, z_m = self._xyz(det.cx, det.cy, z_m)
        distance_cm = self._distance(x_m, y_m, z_m) * 100.0

        robot_x = self.centerline_x_px
        robot_y = float(max(0, frame_h - 1))
        camera_angle_deg = math.degrees(math.atan2(x_m, z_m))
        centerline_angle_deg = math.degrees(
            math.atan2(robot_x - self.cx_intr, self.fx)
        )
        angle_deg = camera_angle_deg - centerline_angle_deg

        return {
            "detected": True,
            "raw_detected": True,
            "held_previous_detection": False,
            "center_x": det.cx,
            "center_y": det.cy,
            "realsense_goal_distance_cm": float(distance_cm),
            "realsense_goal_angle": float(angle_deg),
            "center_depth_cm": float(z_m * 100.0),
            "robot_center_x": float(robot_x),
            "robot_bottom_y": float(robot_y),
            "confidence": det.conf,
            "target_class": det.name,
            "backboard_bbox": [det.x1, det.y1, det.x2, det.y2],
            "backboard_detected": True,
            "active": self.hoop_active,
            "camera_info_received": self.camera_info_received,
            "source": "realsense_yolo",
            "process_ms": process_ms,
            "stamp_sec": time.monotonic(),
        }

    @staticmethod
    def _empty_ball_state(active: bool) -> Dict[str, object]:
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
            "held_previous_detection": False,
            "realsense_diagnostic": {
                "category": "waiting" if active else "inactive",
                "detail": "no_yolo_ball_detection" if active else "ball_mode_off",
            },
            "active": active,
            "source": "realsense_yolo",
            "process_ms": 0.0,
        }

    @staticmethod
    def _empty_hoop_state(active: bool) -> Dict[str, object]:
        return {
            "detected": False,
            "raw_detected": False,
            "held_previous_detection": False,
            "center_x": None,
            "center_y": None,
            "realsense_goal_distance_cm": None,
            "realsense_goal_angle": None,
            "center_depth_cm": None,
            "robot_center_x": None,
            "robot_bottom_y": None,
            "confidence": 0.0,
            "target_class": None,
            "backboard_bbox": [],
            "backboard_detected": False,
            "active": active,
            "source": "realsense_yolo",
            "process_ms": 0.0,
            "stamp_sec": time.monotonic(),
        }

    def _publish_ball_state(self, state: Dict[str, object]) -> None:
        self.ball_state_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))
        self.ball_detected_pub.publish(
            Bool(data=bool(state.get("realsense_ball_detected", False)))
        )

    def _publish_hoop_state(self, state: Dict[str, object]) -> None:
        self.hoop_state_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))
        self.hoop_detected_pub.publish(Bool(data=bool(state.get("detected", False))))

    @staticmethod
    def _draw_detection(
        frame: np.ndarray,
        detection: Optional[Detection],
        label: str,
    ) -> None:
        if detection is None:
            return
        x1, y1, x2, y2 = map(
            int,
            [
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
            ],
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 3)
        cv2.putText(
            frame,
            f"{label} {detection.conf:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            YELLOW,
            2,
            cv2.LINE_AA,
        )

    def _draw_target_guide(
        self,
        frame: np.ndarray,
        detection: Optional[Detection],
    ) -> None:
        if detection is None:
            return
        centerline_x = int(round(self.centerline_x_px))
        robot = (centerline_x, frame.shape[0] - 1)
        target = (
            int(round(detection.cx)),
            int(round(detection.cy)),
        )
        cv2.line(frame, robot, target, CYAN, 2, cv2.LINE_AA)
        cv2.circle(frame, target, 5, YELLOW, -1)
        cv2.circle(frame, robot, 5, WHITE, -1)

    def _draw_debug_view(
        self,
        frame: np.ndarray,
        mode: str,
        ball: Optional[Detection],
        backboard: Optional[Detection],
        goal: Optional[Detection],
        ball_state: Dict[str, object],
        hoop_state: Dict[str, object],
        process_ms: float,
    ) -> np.ndarray:
        debug = frame.copy()
        cv2.rectangle(
            debug,
            (1, 1),
            (debug.shape[1] - 2, debug.shape[0] - 2),
            YELLOW,
            2,
        )
        centerline_x = int(round(self.centerline_x_px))
        cv2.line(
            debug,
            (centerline_x, 0),
            (centerline_x, debug.shape[0] - 1),
            (128, 128, 128),
            1,
        )

        show_ball = mode in {"ball", "combined"}
        show_hoop = mode in {"hoop", "combined"}
        if show_ball:
            self._draw_detection(debug, ball, "BALL")
            self._draw_target_guide(debug, ball)
            ball_depth_valid = bool(
                ball_state.get("realsense_ball_detected", False)
            )
            ball_mode = "ACTIVE" if self.ball_active else "PREVIEW"
            ball_lines = [
                f"REALSENSE YOLO / BALL {ball_mode}",
                f"detect:{'YES' if ball is not None else 'NO'} "
                f"conf:{_display_number(ball.conf if ball else None, digits=2)}",
                (
                    "distance:"
                    f"{_display_number(ball_state.get('realsense_ball_distance_cm'), 'cm')}"
                ),
                (
                    "angle:"
                    f"{_display_number(ball_state.get('realsense_ball_angle_error'), 'deg')}"
                ),
                (
                    "x:"
                    f"{_display_number(ball_state.get('raw_x_m'), 'm', 3)} "
                    "y:"
                    f"{_display_number(ball_state.get('raw_y_m'), 'm', 3)}"
                ),
                (
                    "z:"
                    f"{_display_number(ball_state.get('raw_z_m'), 'm', 3)} "
                    f"depth:{'OK' if ball_depth_valid else 'INVALID'}"
                ),
                f"inference:{process_ms:.1f}ms",
            ]
            draw_info_panel(debug, ball_lines, align="left")

        if show_hoop:
            # backboard가 거리/각도 기준이고 goal은 모델의 추가 검출 정보다.
            self._draw_detection(debug, backboard, "BACKBOARD")
            if goal is not None:
                self._draw_detection(debug, goal, "GOAL")
            self._draw_target_guide(debug, backboard)
            hoop_depth_valid = bool(hoop_state.get("detected", False))
            hoop_mode = "ACTIVE" if self.hoop_active else "PREVIEW"
            hoop_lines = [
                f"REALSENSE YOLO / HOOP {hoop_mode}",
                f"detect:{'YES' if backboard is not None else 'NO'} "
                f"conf:{_display_number(backboard.conf if backboard else None, digits=2)}",
                (
                    "distance:"
                    f"{_display_number(hoop_state.get('realsense_goal_distance_cm'), 'cm')}"
                ),
                (
                    "angle:"
                    f"{_display_number(hoop_state.get('realsense_goal_angle'), 'deg')}"
                ),
                (
                    "center:"
                    f"{_display_number(hoop_state.get('center_x'), 'px')} / "
                    f"{_display_number(hoop_state.get('center_y'), 'px')}"
                ),
                (
                    "depth:"
                    f"{_display_number(hoop_state.get('center_depth_cm'), 'cm')} "
                    f"{'OK' if hoop_depth_valid else 'INVALID'}"
                ),
                f"inference:{process_ms:.1f}ms",
            ]
            draw_info_panel(
                debug,
                hoop_lines,
                align="right" if mode == "combined" else "left",
            )
        return debug

    @staticmethod
    def _has_viewer(publisher) -> bool:
        return publisher.get_subscription_count() > 0

    def _debug_view_requested(self) -> bool:
        if not bool(self.cfg["publish_debug_image"]):
            return False
        return any(
            self._has_viewer(publisher)
            for publisher in (
                self.ball_view_pub,
                self.hoop_view_pub,
                self.combined_view_pub,
                self.selected_view_pub,
                self.ball_debug_pub,
                self.hoop_debug_pub,
            )
        )

    def _publish_debug_views(
        self,
        frame: np.ndarray,
        color_msg: Image,
        ball: Optional[Detection],
        backboard: Optional[Detection],
        goal: Optional[Detection],
        ball_state: Dict[str, object],
        hoop_state: Dict[str, object],
        process_ms: float,
    ) -> None:
        if not bool(self.cfg["publish_debug_image"]):
            return

        publishers = {"ball": [], "hoop": [], "combined": []}
        if self._has_viewer(self.ball_view_pub):
            publishers["ball"].append(self.ball_view_pub)
        if self._has_viewer(self.hoop_view_pub):
            publishers["hoop"].append(self.hoop_view_pub)
        if self._has_viewer(self.combined_view_pub):
            publishers["combined"].append(self.combined_view_pub)
        if self._has_viewer(self.ball_debug_pub):
            publishers["ball"].append(self.ball_debug_pub)
        if self._has_viewer(self.hoop_debug_pub):
            publishers["hoop"].append(self.hoop_debug_pub)
        if self._has_viewer(self.selected_view_pub):
            selected_mode = getattr(self, "display_mode", "ball")
            publishers[selected_mode].append(self.selected_view_pub)

        for mode, outputs in publishers.items():
            if not outputs:
                continue
            debug = self._draw_debug_view(
                frame,
                mode,
                ball,
                backboard,
                goal,
                ball_state,
                hoop_state,
                process_ms,
            )
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = color_msg.header
            for publisher in outputs:
                publisher.publish(debug_msg)

    def _publish_cached_debug(
        self,
        frame: np.ndarray,
        color_msg: Image,
    ) -> None:
        """Keep image topics moving while inference is throttled or recovering."""
        self._publish_debug_views(
            frame,
            color_msg,
            self.latest_ball_detection,
            self.latest_backboard_detection,
            self.latest_goal_detection,
            self.latest_ball_state,
            self.latest_hoop_state,
            self.latest_process_ms,
        )

    def cb_images(self, color_msg: Image, depth_msg: Image) -> None:
        debug_requested = self._debug_view_requested()
        if (
            not self.ball_active
            and not self.hoop_active
            and not debug_requested
        ):
            return

        now = time.monotonic()
        max_fps = max(0.1, float(self.cfg["max_fps"]))
        # 카메라와 max_fps가 모두 30일 때 타임스탬프 흔들림 때문에 매 두 번째
        # 프레임이 버려지지 않도록 10% 허용한다. 추론을 쉬는 프레임도 아래에서
        # 최신 결과를 새 원본 영상 위에 그려 화면 송출 자체는 계속한다.
        should_infer = (
            now - self.last_inference_time >= (1.0 / max_fps) * 0.90
        )
        if not should_infer and not debug_requested:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Color conversion failed: {exc}")
            return

        if not should_infer:
            self._publish_cached_debug(frame, color_msg)
            return

        self.last_inference_time = now
        start = time.perf_counter()
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            self.get_logger().warning(f"Depth conversion failed: {exc}")
            self._publish_cached_debug(frame, color_msg)
            return

        depth_m = self._depth_m(np.asarray(depth_raw), depth_msg.encoding)
        if (
            depth_m.ndim != 2
            or depth_m.shape[0] != frame.shape[0]
            or depth_m.shape[1] != frame.shape[1]
        ):
            self.get_logger().warning(
                "Color/depth mismatch; aligned_depth_to_color is required"
            )
            self._publish_cached_debug(frame, color_msg)
            return

        try:
            dets = self._run_yolo(frame)
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            self._publish_cached_debug(frame, color_msg)
            return

        ball = self._best(dets, str(self.cfg["ball_class"]))
        backboard = self._best(dets, str(self.cfg["backboard_class"]))
        goal = self._best(dets, str(self.cfg["goal_class"]))

        process_ms = (time.perf_counter() - start) * 1000.0

        # 추론은 한 번만 수행한다. 두 상태를 같은 결과에서 계산해 별도/통합
        # rqt 화면이 모드 전환 중에도 같은 프레임을 보여 주게 한다. 실제 판단용
        # 상태 토픽은 아래에서 활성 모드일 때만 발행한다.
        ball_state = self._make_ball_state(ball, depth_m, process_ms)
        hoop_state = self._make_hoop_state(
            backboard,
            depth_m,
            frame.shape[0],
            process_ms,
        )
        self.latest_ball_detection = ball
        self.latest_backboard_detection = backboard
        self.latest_goal_detection = goal
        self.latest_ball_state = ball_state
        self.latest_hoop_state = hoop_state
        self.latest_process_ms = process_ms

        if self.ball_active:
            self._publish_ball_state(ball_state)
        if self.hoop_active:
            self._publish_hoop_state(hoop_state)

        self._publish_debug_views(
            frame,
            color_msg,
            ball,
            backboard,
            goal,
            ball_state,
            hoop_state,
            process_ms,
        )

        self.frame_count += 1
        every = max(1, int(self.cfg["print_every_n_frames"]))
        if self.frame_count % every == 0:
            self.get_logger().info(
                f"RS-YOLO process={process_ms:.1f}ms "
                f"ball={int(bool(ball_state.get('realsense_ball_detected', False)))} "
                f"hoop={int(bool(hoop_state.get('detected', False)))}"
            )


def main() -> None:
    command_line = sys.argv[1:]
    ini_path = "settings.ini"
    if command_line and not command_line[0].startswith("--"):
        ini_path = command_line[0]
        command_line = command_line[1:]

    rclpy.init(args=command_line if command_line else None)
    node = RealSenseYoloDetector(ini_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
