#!/usr/bin/env python3

"""
YOLO Line Tracker - Vision Module
IRC 2026 Humanoid Robot Competition

YOLO detection 결과의 line bbox 중심점으로 라인 주행 값을 계산

publish 값:
- point_count
- line_angle
- curve_a
- tangent_angle
- line_distance
- target_x, target_y
- follow_distance
- line_second_point_distance_px, hurdle_line_angle_deg
- ball / hurdle detection 정보
- raw_ball_in_hand (/raw_ball_in_hand)

공/허들 fusion 관련 주의:
- 이 파일은 웹캠 YOLO의 ball/hurdle 검출 필드를 /line_tracker/state로 전달합니다.
- ball_result는 ball_vision_fusion.py가 담당합니다.
- hurdle_result는 hurdle_vision_fusion.py가 담당하도록 기본값에서 직접 발행을 끕니다.
  YOLO 단독 모드가 필요할 때만 settings.ini의 publish_hurdle_result=true를 사용하세요.
"""

import configparser
import cv2
import numpy as np
import time
import math
import json
import gc
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from collections import deque

from line_status_publisher import (
    LineDecision,
    LineFeatures,
    LineStatus,
    LineStatusPublisher,
)
from hurdle_status_publisher import HurdleStatusPublisher

try:
    from ultralytics import YOLO
except ImportError :
    YOLO = None


RAW_BALL_IN_HAND_DEFAULTS = {
    "raw_ball_in_hand_hold_seconds": 0.3,
    "raw_ball_in_hand_angle_min_deg": 56.0,
    "raw_ball_in_hand_angle_max_deg": 58.0,
    "raw_ball_in_hand_x_min_px": 282.0,
    "raw_ball_in_hand_x_max_px": 289.0,
    "raw_ball_in_hand_y_min_px": 182.0,
    "raw_ball_in_hand_y_max_px": 190.0,
}


# ═══════════════════════════════════════════════════════
#  데이터 클래스
# ═══════════════════════════════════════════════════════

@dataclass
class ObjectDetection:
    name: str
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


# 실제 motion 노드가 실행하는 모션 ID/이름입니다.
MOTION_NAME = {
    0: "Initial_Pose",
    1: "Forward_4step",
    2: "Left_Half_Forward",
    3: "Right_Half_Forward",
    4: "Left_Turn_Half",
    5: "Right_Turn_Half",
    6: "Left_Turn",
    7: "Right_Turn",
    8: "Forward_half",
    9: "Backward_half",
    10: "Left_Move",
    11: "Right_Move",
    12: "Pick",
    13: "Shoot",
    14: "Neck_Up",
    15: "Neck_Left",
    16: "Neck_Right",
    17: "Neck_Center",
    18: "Neck_Down",
    19: "Hurdle_Go",
    20: "Forward_3step",
    21: "Left_Half_Forward_3step",
    22: "Right_Half_Forward_3step",
    23: "Left_Turn_Mission",
    24: "Right_Turn_Mission",
    25: "Hurdle_1step",
    26: "Hurdle_Forward_20",
    27: "Back_To_Initial",
}


@dataclass
class MotionDisplayState:
    """Track commands that the motion node has actually started."""

    received_command: Optional[int] = None
    active_command: Optional[int] = None
    last_started_command: Optional[int] = None
    running: bool = False
    ready: bool = False

    def on_command(self, command: int) -> None:
        self.received_command = int(command)

        # motion_end=false가 서로 다른 topic에서 먼저 도착한 경우에만
        # 뒤늦게 도착한 command를 현재 실행 모션에 연결합니다.
        if self.running and self.active_command is None:
            self.active_command = self.received_command
            self.last_started_command = self.received_command

    def on_motion_state(self, motion_end: bool, motion_ready: bool) -> None:
        self.ready = bool(motion_ready)

        if not bool(motion_end):
            self.running = True
            self.active_command = self.received_command
            if self.active_command is not None:
                self.last_started_command = self.active_command
            return

        self.running = False
        self.active_command = None


def motion_overlay_lines(state: Optional[MotionDisplayState]) -> list[str]:
    """Return the actual-motion lines rendered on the webcam image."""
    if state is None:
        return ["motion:-- N/A", "run:N/A"]

    command = (
        state.active_command
        if state.running
        else state.last_started_command
    )
    run_text = "RUNNING" if state.running else "IDLE"

    if command is None:
        return [
            "motion:-- UNKNOWN",
            f"run:{run_text} ready:{int(state.ready)}",
        ]

    name = MOTION_NAME.get(command, "Unknown")
    return [
        f"motion:{command} {name}",
        f"run:{run_text} ready:{int(state.ready)}",
    ]


# ═══════════════════════════════════════════════════════
#  라인 모션 판단 코드
# ═══════════════════════════════════════════════════════

LINE_STATUS_NAME = {
    LineStatus.Forward_4step: "Forward_4step",
    LineStatus.Left_Half_Forward: "Left_Half_Forward",
    LineStatus.Right_Half_Forward: "Right_Half_Forward",
    LineStatus.Left_Turn_Half: "Left_Turn_Half",
    LineStatus.Right_Turn_Half: "Right_Turn_Half",
    LineStatus.Left_Turn: "Left_Turn",
    LineStatus.Right_Turn: "Right_Turn",
    LineStatus.Forward_half: "Forward_half",
    LineStatus.Backward_half: "Backward_half",
    LineStatus.Left_Move: "Left_Move",
    LineStatus.Right_Move: "Right_Move",
    LineStatus.Line_Lost: "Line_Lost",
}


def apply_line_status(
    payload: dict,
    frame_w: int,
    frame_h: int,
    publisher: Optional[LineStatusPublisher] = None,
) -> dict:
    """line_status_publisher의 판단 로직을 호출하고 ROS 모드에서는 즉시 발행한다."""
    values = {
        "point_count": int(payload.get("point_count", 0)),
        "line_angle": payload.get("line_angle"),
        "curve_a": payload.get("curve_a"),
        "tangent_angle": payload.get("tangent_angle"),
        "line_distance": payload.get("line_distance"),
        "target_x": payload.get("target_x"),
        "target_y": payload.get("target_y"),
        "robot_center_x": float(
            payload.get("robot_center_x", frame_w / 2.0)
        ),
        "robot_center_y": float(frame_h),
        "follow_angle": payload.get("follow_angle"),
        "follow_distance": payload.get("follow_distance"),
    }

    if publisher is not None:
        status, angle = publisher.publish_line_status(**values)
    else:
        # OpenCV 단독 실행 모드에서는 ROS publish 없이 같은 판단 로직만 사용한다.
        status, angle = LineDecision().decide(LineFeatures(**values))

    payload["status"] = int(status)
    payload["status_name"] = LINE_STATUS_NAME.get(int(status), "UNKNOWN")
    payload["angle"] = float(angle)
    return payload


class LinePayloadSmoother:
    def __init__(self, window=5, min_valid=3):
        self.window = window
        self.min_valid = min_valid
        self.buffer = deque(maxlen=window)

    def _median(self, values):
        vals = []
        for v in values:
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(fv):
                vals.append(fv)

        if not vals:
            return None

        return float(np.median(vals))

    def smooth(self, payload: dict, frame_w: int, frame_h: int) -> dict:
        self.buffer.append(dict(payload))

        valid = [
            p for p in self.buffer
            if int(p.get("point_count", 0)) >= 1
        ]

        # 최근 window 중 유효한 라인이 너무 적으면 LOST 유지
        if len(valid) < self.min_valid:
            payload["point_count"] = 0
            payload["smooth_valid_count"] = len(valid)
            return payload

        smoothed = dict(payload)

        for key in [
            "line_angle",
            "curve_a",
            "tangent_angle",
            "line_distance",
            "target_x",
            "target_y",
            "follow_angle",
            "follow_distance",
            "line_second_point_distance_px",
            "hurdle_line_angle_deg",
        ]:
            med = self._median([p.get(key) for p in valid])
            if med is not None:
                smoothed[key] = med

        # point_count는 최근 값 중 중앙값 느낌으로 안정화
        smoothed["point_count"] = int(round(self._median([p.get("point_count", 0) for p in valid]) or 0))
        smoothed["smooth_valid_count"] = len(valid)

        return smoothed
    
LINE_SMOOTHER = LinePayloadSmoother(window=5, min_valid=3)

# ═══════════════════════════════════════════════════════
#  settings.ini 로더
# ═══════════════════════════════════════════════════════

def load_config(ini_path: str = "settings.ini") -> dict:
    defaults = {
        # camera
        "cam_index": 0,
        "cam_width": 640,
        "cam_height": 480,
        "cam_fps": 30,
        "flip_vertical": False,
        "robot_center_offset_x_px": 25.0,

        # ROI: YOLO line 중심점 중 이 영역 안에 있는 것만 주행용으로 사용
        "roi_top_ratio": 0.00,
        "roi_bottom_ratio": 1.00,
        "roi_left_ratio": 0.00,
        "roi_right_ratio": 1.00,
        "min_points_for_poly": 3,

        # YOLO
        "yolo_model": "best.pt",
        "yolo_conf": 0.20,
        "line_conf": 0.35,
        "line_display_conf": 0.40,
        "ball_conf": 0.20,
        "hurdle_conf": 0.35,
        "yolo_imgsz": 640,
        "yolo_device": "0",
        "line_class": "line",
        "ball_class": "ball",
        "hurdle_class": "hurdle",

        # 화면의 BALL 패널에 표시되는 angle/x/y가 모두 범위 안일 때 True.
        **RAW_BALL_IN_HAND_DEFAULTS,

        # visibility filter
        # min_visible_ratio = 0.70 means: hide/reject objects if estimated visible area is below 70%.
        # Because YOLO boxes are usually clipped to the image, reject_edge_cut_objects is used
        # to remove ball/hurdle boxes touching the image border.
        "min_visible_ratio": 0.70,
        "reject_edge_cut_objects": False,
        "edge_margin": 3,
        "partial_filter_classes": "",

        # output
        "show_window": True,
        "save_video": "",
        "print_every_n_frames": 5,
        # Fusion 노드와 hurdle_result 중복 발행을 막기 위해 기본 False.
        "publish_hurdle_result": False,
    }

    p = Path(ini_path)
    if not p.exists():
        print(f"[WARN] {ini_path} not found -> using defaults")
        return defaults

    ini = configparser.ConfigParser()
    ini.read(p, encoding="utf-8")

    def gi(s, k, fb): return ini.getint(s, k, fallback=fb)
    def gf(s, k, fb): return ini.getfloat(s, k, fallback=fb)
    def gb(s, k, fb): return ini.getboolean(s, k, fallback=fb)
    def gs(s, k, fb): return ini.get(s, k, fallback=fb)

    cfg = dict(defaults)
    cfg.update({
        "cam_index": gi("camera", "index", defaults["cam_index"]),
        "cam_width": gi("camera", "width", defaults["cam_width"]),
        "cam_height": gi("camera", "height", defaults["cam_height"]),
        "cam_fps": gi("camera", "fps", defaults["cam_fps"]),
        "flip_vertical": gb("camera", "flip_vertical", defaults["flip_vertical"]),
        "robot_center_offset_x_px": gf(
            "camera",
            "robot_center_offset_x_px",
            defaults["robot_center_offset_x_px"],
        ),

        "roi_top_ratio": gf("detection", "roi_top_ratio", defaults["roi_top_ratio"]),
        "roi_bottom_ratio": gf("detection", "roi_bottom_ratio", defaults["roi_bottom_ratio"]),
        "roi_left_ratio": gf("detection", "roi_left_ratio", defaults["roi_left_ratio"]),
        "roi_right_ratio": gf("detection", "roi_right_ratio", defaults["roi_right_ratio"]),
        "min_points_for_poly": gi("curve", "min_points_for_poly", defaults["min_points_for_poly"]),

        "yolo_model": gs("yolo", "model", defaults["yolo_model"]),
        "yolo_conf": gf("yolo", "conf", defaults["yolo_conf"]),
        "line_conf": gf("yolo", "line_conf", defaults["line_conf"]),
        "line_display_conf": gf(
            "yolo", "line_display_conf", defaults["line_display_conf"]
        ),
        "ball_conf": gf("yolo", "ball_conf", defaults["ball_conf"]),
        "hurdle_conf": gf("yolo", "hurdle_conf", defaults["hurdle_conf"]),
        "yolo_imgsz": gi("yolo", "imgsz", defaults["yolo_imgsz"]),
        "yolo_device": gs("yolo", "device", defaults["yolo_device"]),
        "line_class": gs("yolo", "line_class", defaults["line_class"]),
        "ball_class": gs("yolo", "ball_class", defaults["ball_class"]),
        "hurdle_class": gs("yolo", "hurdle_class", defaults["hurdle_class"]),

        "raw_ball_in_hand_angle_min_deg": gf(
            "raw_ball_in_hand",
            "angle_min_deg",
            defaults["raw_ball_in_hand_angle_min_deg"],
        ),
        "raw_ball_in_hand_angle_max_deg": gf(
            "raw_ball_in_hand",
            "angle_max_deg",
            defaults["raw_ball_in_hand_angle_max_deg"],
        ),
        "raw_ball_in_hand_hold_seconds": gf(
            "raw_ball_in_hand",
            "hold_seconds",
            defaults["raw_ball_in_hand_hold_seconds"],
        ),
        "raw_ball_in_hand_x_min_px": gf(
            "raw_ball_in_hand", "x_min_px", defaults["raw_ball_in_hand_x_min_px"]
        ),
        "raw_ball_in_hand_x_max_px": gf(
            "raw_ball_in_hand", "x_max_px", defaults["raw_ball_in_hand_x_max_px"]
        ),
        "raw_ball_in_hand_y_min_px": gf(
            "raw_ball_in_hand", "y_min_px", defaults["raw_ball_in_hand_y_min_px"]
        ),
        "raw_ball_in_hand_y_max_px": gf(
            "raw_ball_in_hand", "y_max_px", defaults["raw_ball_in_hand_y_max_px"]
        ),

        "min_visible_ratio": gf("visibility", "min_visible_ratio", defaults["min_visible_ratio"]),
        "reject_edge_cut_objects": gb("visibility", "reject_edge_cut_objects", defaults["reject_edge_cut_objects"]),
        "edge_margin": gi("visibility", "edge_margin", defaults["edge_margin"]),
        "partial_filter_classes": gs("visibility", "partial_filter_classes", defaults["partial_filter_classes"]),

        "show_window": gb("output", "show_window", defaults["show_window"]),
        "save_video": gs("output", "save_video", defaults["save_video"]),
        "print_every_n_frames": gi("output", "print_every_n_frames", defaults["print_every_n_frames"]),
        "publish_hurdle_result": gb("output", "publish_hurdle_result", defaults["publish_hurdle_result"]),
    })
    return cfg


# ═══════════════════════════════════════════════════════
#  기존 detect_line.py의 2차 피팅 로직 유지
# ═══════════════════════════════════════════════════════

def fit_poly2(points):
    """x = a*y^2 + b*y + c 형태로 2차 피팅."""
    if len(points) < 3:
        return None
    pts = np.array(points, dtype=np.float64)
    ys, xs = pts[:, 1], pts[:, 0]
    y_mean = ys.mean()
    y_std = ys.std() if ys.std() > 1e-6 else 1.0
    yn = (ys - y_mean) / y_std
    try:
        a_n, b_n, c_n = np.polyfit(yn, xs, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    s = y_std
    a = a_n / s**2
    b = -2 * a_n * y_mean / s**2 + b_n / s
    c = a_n * y_mean**2 / s**2 - b_n * y_mean / s + c_n
    return np.array([a, b, c], dtype=np.float64)


def fit_line(points):
    """x = b*y + c 형태로 직선 피팅."""
    if len(points) < 2:
        return None
    pts = np.array(points, dtype=np.float64)
    ys, xs = pts[:, 1], pts[:, 0]
    y_mean = ys.mean()
    y_std = ys.std()
    if y_std <= 1e-6:
        return None
    yn = (ys - y_mean) / y_std
    try:
        b_n, c_n = np.polyfit(yn, xs, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    b = b_n / y_std
    c = c_n - b_n * y_mean / y_std
    return np.array([b, c], dtype=np.float64)


# ═══════════════════════════════════════════════════════
#  YOLO detection
# ═══════════════════════════════════════════════════════

def load_yolo_model(cfg: dict):
    if YOLO is None:
        raise RuntimeError("ultralytics가 설치되어 있지 않습니다. pip install -U ultralytics 후 실행하세요.")
    model_path = cfg["yolo_model"]
    print(f"[YOLO] loading model: {model_path}")
    return YOLO(model_path, task="detect")


def yolo_detect(model, frame: np.ndarray, cfg: dict) -> list[ObjectDetection]:
    results = model.predict(
        source=frame,
        imgsz=cfg["yolo_imgsz"],
        conf=cfg["yolo_conf"],
        device=cfg["yolo_device"],
        verbose=False,
    )

    dets: list[ObjectDetection] = []
    result = results[0]
    if result.boxes is None:
        return dets

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = str(model.names[cls_id])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        dets.append(ObjectDetection(name, cls_id, conf, x1, y1, x2, y2))

    return dets


def get_yolo_line_points(dets: list[ObjectDetection], frame_w: int, frame_h: int, cfg: dict):
    """YOLO가 찾은 line bbox 중심점을 주행용 line point로 변환."""
    roi_top = int(frame_h * cfg["roi_top_ratio"])
    roi_bottom = int(frame_h * cfg["roi_bottom_ratio"])
    roi_left = int(frame_w * cfg["roi_left_ratio"])
    roi_right = int(frame_w * cfg["roi_right_ratio"])

    pts = []
    for d in dets:
        if d.name != cfg["line_class"]:
            continue
        if d.conf < cfg["line_conf"]:
            continue
        if not (roi_left <= d.cx <= roi_right and roi_top <= d.cy <= roi_bottom):
            continue
        pts.append((float(d.cx), float(d.cy)))

    pts.sort(key=lambda p: -p[1])  # 화면 아래쪽, 즉 가까운 점부터
    return pts, (roi_top, roi_bottom, roi_left, roi_right)


# ═══════════════════════════════════════════════════════
#  알고리즘 쪽으로 보낼 값 계산
# ═══════════════════════════════════════════════════════

def robot_center_x(
    frame_w: int,
    robot_center_offset_x_px: float = 25.0,
) -> float:
    """Return the calibrated robot center X clamped to the image bounds."""
    return min(
        max((float(frame_w) / 2.0) + float(robot_center_offset_x_px), 0.0),
        max(0.0, float(frame_w - 1)),
    )


def make_line_payload(
    line_points: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
    robot_center_offset_x_px: float = 25.0,
) -> dict:
    """
    알고리즘 패키지로 넘길 line 값 계산.
    point_count <= 1이면 알고리즘 쪽에서는 LOST로 보면 됨.
    """
    point_count = len(line_points)

    payload = {
        "point_count": int(point_count),
        "line_angle": 0.0,
        "curve_a": 0.0,
        "tangent_angle": 0.0,
        "line_distance": 0.0,
        "target_x": -1.0,
        "target_y": -1.0,
        "follow_angle": 0.0,
        "follow_distance": -1.0,
        "line_second_point_distance_px": None,
        "hurdle_line_angle_deg": None,
    }

    if point_count == 0:
        return payload

    robot_x = robot_center_x(frame_w, robot_center_offset_x_px)
    robot_y = float(frame_h)

    # 가장 가까운 점 기준. 음수면 라인이 왼쪽, 양수면 오른쪽.
    nearest_x, nearest_y = line_points[0]
    payload["line_distance"] = float(nearest_x - robot_x)

    # target은 기본적으로 두 번째 점. 없으면 첫 번째 점.
    if point_count >= 2:
        target_x, target_y = line_points[1]
    else:
        target_x, target_y = line_points[0]

    payload["target_x"] = float(target_x)
    payload["target_y"] = float(target_y)
    target_dx = target_x - robot_x
    target_angle = float(
        math.degrees(math.atan2(target_dx, robot_y - target_y))
    )

    payload["follow_distance"] = float(
        math.hypot(target_dx, target_y - robot_y)
    )
    # 점 1개: 로봇 중심선과 '로봇 중심 -> 검출점' 선 사이의 부호 있는 각도.
    # 화면 위쪽을 0도로 보고, 검출점이 오른쪽이면 +, 왼쪽이면 -이다.
    payload["follow_angle"] = target_angle

    # 허들 거리 판단은 로봇에서 두 번째로 가까운 라인점을 기준으로 한다.
    # 기존 line_distance는 첫 번째 점 기준이므로 두 번째 점 거리를 별도로 보존한다.
    if point_count >= 2:
        payload["line_second_point_distance_px"] = float(target_dx)

        # 허들 각도는 로봇에서 라인점을 바라보는 각도가 아니라,
        # 검출된 라인점들을 직선으로 이은 선과 로봇 수직 중심선 사이의 각도다.
        hurdle_line_coeffs = fit_line(line_points)
        if hurdle_line_coeffs is not None:
            hurdle_b, _hurdle_c = hurdle_line_coeffs
            payload["hurdle_line_angle_deg"] = float(
                math.degrees(math.atan2(-hurdle_b, 1.0))
            )

    # 점 2~3개는 검출점 전체를 직선으로 피팅한다.
    # 점 2개에서는 LineDecision이 follow_angle을 사용하고, 점 3개에서는
    # line_angle을 사용하므로 동일한 직선 피팅각을 해당 필드에 넣는다.
    if point_count >= 2:
        if point_count <= 3:
            line_coeffs = fit_line(line_points)
            if line_coeffs is not None:
                b, _c = line_coeffs
                fitted_angle = float(math.degrees(math.atan2(-b, 1.0)))
                payload["tangent_angle"] = fitted_angle
                if point_count == 2:
                    payload["follow_angle"] = fitted_angle
                else:
                    payload["line_angle"] = fitted_angle

    # 점 4개 이상이면 검출점 전체로 2차함수를 피팅한다. 세 번째로 가까운
    # 점에서 구한 접선각을 line_angle에도 넣어 실제 주행 모드 판단에 사용한다.
    if point_count >= 4:
        coeffs = fit_poly2(line_points)
        if coeffs is not None:
            a, b, _c = coeffs
            payload["curve_a"] = float(a)

            tangent_y = line_points[2][1]
            slope_dx_dy_down = 2.0 * a * tangent_y + b
            # 이미지 y는 아래로 증가하므로, 로봇 진행 방향인 위쪽 기준으로 부호 반전
            payload["tangent_angle"] = float(math.degrees(math.atan2(-slope_dx_dy_down, 1.0)))
            payload["line_angle"] = payload["tangent_angle"]

    return payload



def _partial_filter_class_set(cfg: dict) -> set[str]:
    raw = str(cfg.get("partial_filter_classes", "ball,hurdle"))
    return {x.strip() for x in raw.split(",") if x.strip()}


def _box_intersection_ratio(d: ObjectDetection, left: float, top: float, right: float, bottom: float) -> float:
    """
    Returns how much of the detected bbox lies inside the given rectangle.
    Note: YOLO xyxy boxes are usually already clipped to the image, so for frame-boundary
    partial objects this often returns 1.0. The edge-touch filter below handles that case.
    """
    box_area = max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1)
    if box_area <= 1e-6:
        return 0.0
    ix1 = max(d.x1, left)
    iy1 = max(d.y1, top)
    ix2 = min(d.x2, right)
    iy2 = min(d.y2, bottom)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(inter / box_area)


def visible_enough(d: ObjectDetection, frame_w: int, frame_h: int, cfg: dict) -> bool:
    """
    Filters partially visible objects.
    - min_visible_ratio=0.70 means at least 70% of the bbox must be inside the frame.
    - reject_edge_cut_objects=True rejects selected classes if their bbox touches image edges.

    This is an approximation: with a normal bbox detector, we cannot know the true hidden
    area of an occluded/cut object. It works best for objects cut by the image border.
    """
    classes = _partial_filter_class_set(cfg)
    if d.name not in classes:
        return True

    min_ratio = float(cfg.get("min_visible_ratio", 0.70))
    ratio = _box_intersection_ratio(d, 0.0, 0.0, float(frame_w), float(frame_h))
    if ratio < min_ratio:
        return False

    if bool(cfg.get("reject_edge_cut_objects", True)):
        m = float(cfg.get("edge_margin", 3))
        touches_edge = (
            d.x1 <= m or d.y1 <= m or
            d.x2 >= frame_w - m or d.y2 >= frame_h - m
        )
        if touches_edge:
            return False

    return True

def best_object_detection(
    dets: list[ObjectDetection],
    cfg: dict,
    class_key: str,
    frame_w: int,
    frame_h: int,
) -> Optional[ObjectDetection]:
    """Return the highest-confidence usable detection for one object class."""
    class_name = cfg[f"{class_key}_class"]
    conf_thres = cfg[f"{class_key}_conf"]
    objs = [
        d
        for d in dets
        if (
            d.name == class_name
            and d.conf >= conf_thres
            and visible_enough(d, frame_w, frame_h, cfg)
        )
    ]

    return max(objs, key=lambda d: d.conf, default=None)


def best_object_payload(dets: list[ObjectDetection], cfg: dict, class_key: str, frame_w: int, frame_h: int) -> dict:
    """ball/hurdle 중 confidence가 가장 높은 객체 하나를 payload로 변환."""
    best = best_object_detection(dets, cfg, class_key, frame_w, frame_h)

    if best is None:
        return {
            f"{class_key}_detected": False,
            f"{class_key}_x": -1.0,
            f"{class_key}_y": -1.0,
            f"{class_key}_conf": 0.0,
            f"{class_key}_bbox": [],
        }

    return {
        f"{class_key}_detected": True,
        f"{class_key}_x": float(best.cx),
        f"{class_key}_y": float(best.cy),
        f"{class_key}_conf": float(best.conf),
        f"{class_key}_bbox": [float(best.x1), float(best.y1), float(best.x2), float(best.y2)],
    }


def add_ball_geometry(
    payload: dict,
    frame_w: int,
    frame_h: int,
    robot_center_offset_x_px: float = 25.0,
) -> dict:
    """공 중심과 로봇 하단 기준점 사이의 화면상 기하 정보를 추가한다."""
    robot_x = robot_center_x(frame_w, robot_center_offset_x_px)
    robot_y = max(0.0, float(frame_h - 1))

    payload["robot_center_x"] = robot_x
    payload["robot_center_y"] = robot_y
    payload["ball_x_distance_px"] = None
    payload["ball_y_distance_px"] = None
    payload["ball_distance_px"] = None
    payload["ball_angle_deg"] = None

    if not bool(payload.get("ball_detected", False)):
        return payload

    try:
        ball_x = float(payload["ball_x"])
        ball_y = float(payload["ball_y"])
    except (KeyError, TypeError, ValueError):
        return payload

    if not math.isfinite(ball_x) or not math.isfinite(ball_y):
        return payload

    # x_distance는 로봇 중심선을 기준으로 왼쪽이 음수, 오른쪽이 양수다.
    x_distance = ball_x - robot_x
    y_distance = abs(robot_y - ball_y)
    payload["ball_x_distance_px"] = float(x_distance)
    payload["ball_y_distance_px"] = float(y_distance)
    payload["ball_distance_px"] = float(math.hypot(x_distance, y_distance))
    payload["ball_angle_deg"] = float(
        math.degrees(math.atan2(x_distance, y_distance))
    )
    return payload


def is_raw_ball_in_hand(payload: dict, cfg: Optional[dict] = None) -> bool:
    """화면에 표시되는 공 angle/x/y가 설정 범위에 모두 들어오는지 판단한다."""
    if not bool(payload.get("ball_detected", False)):
        return False

    cfg = cfg or {}
    try:
        angle = float(payload["ball_angle_deg"])
        x_distance = float(payload["ball_x_distance_px"])
        y_distance = float(payload["ball_y_distance_px"])
    except (KeyError, TypeError, ValueError):
        return False

    if not all(math.isfinite(value) for value in (angle, x_distance, y_distance)):
        return False

    return (
        float(cfg.get("raw_ball_in_hand_angle_min_deg", 56.0))
        <= angle
        <= float(cfg.get("raw_ball_in_hand_angle_max_deg", 58.0))
        and float(cfg.get("raw_ball_in_hand_x_min_px", 282.0))
        <= x_distance
        <= float(cfg.get("raw_ball_in_hand_x_max_px", 289.0))
        and float(cfg.get("raw_ball_in_hand_y_min_px", 182.0))
        <= y_distance
        <= float(cfg.get("raw_ball_in_hand_y_max_px", 190.0))
    )


@dataclass
class ContinuousTrueGate:
    """True 조건이 지정된 시간 동안 연속 유지된 뒤에만 True를 반환한다."""

    hold_seconds: float
    started_at: Optional[float] = None

    def reset(self) -> None:
        self.started_at = None

    def update(self, condition: bool, now: Optional[float] = None) -> bool:
        if not condition:
            self.reset()
            return False

        current_time = time.monotonic() if now is None else float(now)
        if self.started_at is None:
            self.started_at = current_time
            return self.hold_seconds <= 0.0

        return current_time - self.started_at >= max(0.0, self.hold_seconds)


def make_vision_payload(dets: list[ObjectDetection], line_points: list[tuple[float, float]], frame_w: int, frame_h: int, cfg: dict) -> dict:
    center_offset_x = float(cfg.get("robot_center_offset_x_px", 25.0))
    payload = make_line_payload(
        line_points,
        frame_w,
        frame_h,
        center_offset_x,
    )
    payload.update(best_object_payload(dets, cfg, "ball", frame_w, frame_h))
    payload.update(best_object_payload(dets, cfg, "hurdle", frame_w, frame_h))
    payload = add_ball_geometry(
        payload,
        frame_w,
        frame_h,
        center_offset_x,
    )
    payload["raw_ball_in_hand"] = is_raw_ball_in_hand(payload, cfg)
    return payload


# ═══════════════════════════════════════════════════════
#  시각화
# ═══════════════════════════════════════════════════════

def visualize_yolo(
    frame: np.ndarray,
    dets: list[ObjectDetection],
    raw_line_points,
    payload: dict,
    roi_box,
    cfg: dict,
    motion_state: Optional[MotionDisplayState] = None,
):
    vis = frame.copy()
    h, w = vis.shape[:2]
    roi_top, roi_bottom, roi_left, roi_right = roi_box
    line_display_conf = float(cfg.get("line_display_conf", 0.40))
    display_line_points = [
        (float(d.cx), float(d.cy))
        for d in dets
        if (
            d.name == cfg["line_class"]
            and d.conf > line_display_conf
            and roi_left <= d.cx <= roi_right
            and roi_top <= d.cy <= roi_bottom
        )
    ]
    display_line_points.sort(key=lambda point: -point[1])

    robot_x = int(round(float(payload.get("robot_center_x", (w / 2.0) + 25.0))))
    robot_y = int(round(float(payload.get("robot_center_y", h - 1))))
    robot_x = min(max(robot_x, 0), max(0, w - 1))
    robot_y = min(max(robot_y, 0), max(0, h - 1))
    light_sky_blue = (250, 206, 135)  # OpenCV BGR
    best_ball = best_object_detection(dets, cfg, "ball", w, h)

    cv2.rectangle(vis, (roi_left, roi_top), (roi_right, roi_bottom), (80, 80, 80), 2)
    cv2.line(vis, (robot_x, robot_y), (robot_x, roi_top), (200, 200, 200), 1, cv2.LINE_AA)

    # YOLO boxes
    for d in dets:
        # 계산에 사용하는 최고 confidence 공 하나만 화면에 표시한다.
        if d.name == cfg["ball_class"] and d is not best_ball:
            continue
        # confidence 0.40 이하는 라인 판단에는 사용할 수 있지만 화면에는 숨긴다.
        if d.name == cfg["line_class"] and d.conf <= line_display_conf:
            continue
        # Do not draw partially visible ball/hurdle boxes.
        # line is not filtered by default because bottom line markers are often partially visible.
        if not visible_enough(d, w, h, cfg):
            continue

        if d.name == "line":
            color = (0, 255, 255)
        elif d.name == "ball":
            color = (0, 180, 255)
        elif d.name == "hurdle":
            # OpenCV는 BGR 순서. LightSkyBlue(RGB 135, 206, 250).
            color = (250, 206, 135)
        else:
            color = (180, 180, 180)

        x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{d.name} {d.conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # raw line bbox centers
    for cx, cy in display_line_points:
        cv2.circle(vis, (int(cx), int(cy)), 4, (255, 255, 255), -1)

    # target point
    # 빨간 십자/선은 라인을 놓쳤을 때만 표시.
    # 정상 주행/좌표 추종 상태에서는 화면을 가리지 않도록 표시하지 않음.
    show_target_marker = (
        int(payload.get("status", -1)) == LineStatus.Line_Lost
        or int(payload.get("point_count", 0)) <= 1
    )

    if show_target_marker and payload["target_x"] >= 0 and payload["target_y"] >= 0:
        tx, ty = int(payload["target_x"]), int(payload["target_y"])
        cv2.drawMarker(vis, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.line(vis, (robot_x, robot_y), (tx, ty), (0, 0, 255), 1, cv2.LINE_AA)

    # 공 중심과 로봇 하단 기준점을 연결하고 수직 기준선 대비 각도를 표시한다.
    if bool(payload.get("ball_detected", False)):
        ball_x = int(round(float(payload.get("ball_x", -1.0))))
        ball_y = int(round(float(payload.get("ball_y", -1.0))))
        ball_angle = payload.get("ball_angle_deg")
        if 0 <= ball_x < w and 0 <= ball_y < h and ball_angle is not None:
            cv2.line(
                vis,
                (robot_x, robot_y),
                (ball_x, ball_y),
                light_sky_blue,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(vis, (ball_x, ball_y), 5, light_sky_blue, -1)
            label_x = min(max((robot_x + ball_x) // 2 + 6, 0), max(0, w - 90))
            label_y = min(max((robot_y + ball_y) // 2, 15), max(15, h - 5))
            cv2.putText(
                vis,
                f"{float(ball_angle):+.1f}deg",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                light_sky_blue,
                1,
                cv2.LINE_AA,
            )

    # 2~3개는 직선, 4개 이상은 이차곡선으로 표시한다. 두 경우 모두
    # 실제 검출점의 y 범위에서만 그려 데이터가 없는 영역으로 외삽하지 않는다.
    if len(display_line_points) >= 2:
        y_min = max(roi_top, int(math.floor(min(y for _x, y in display_line_points))))
        y_max = min(roi_bottom - 1, int(math.ceil(max(y for _x, y in display_line_points))))
        draw_coeffs = (
            fit_line(display_line_points)
            if len(display_line_points) <= 3
            else fit_poly2(display_line_points)
        )
        if draw_coeffs is not None and y_max > y_min:
            pts_curve = []
            for y_px in range(y_min, y_max + 1, 2):
                x_px = float(np.polyval(draw_coeffs, y_px))
                if 0 <= x_px < w:
                    pts_curve.append((int(round(x_px)), y_px))
            if len(pts_curve) > 1:
                cv2.polylines(
                    vis,
                    [np.array(pts_curve, dtype=np.int32).reshape(-1, 1, 2)],
                    False,
                    (180, 140, 255),  # medium pastel pink (OpenCV BGR)
                    2,
                    cv2.LINE_AA,
                )

    def draw_text_panel(
        panel_lines,
        panel_y,
        font_scale,
        border_color,
        align="left",
    ):
        """화면 좌/우 상단에 내용 크기에 맞는 작은 정보 패널을 그린다."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        padding_x, padding_y = 5, 4
        line_gap = 3
        text_sizes = [
            cv2.getTextSize(text, font, font_scale, font_thickness)[0]
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
        panel_x = 4 if align == "left" else max(4, w - panel_width - 4)
        panel_right = min(w - 1, panel_x + panel_width)
        panel_bottom = min(h - 1, panel_y + panel_height)

        cv2.rectangle(
            vis,
            (panel_x, panel_y),
            (panel_right, panel_bottom),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            vis,
            (panel_x, panel_y),
            (panel_right, panel_bottom),
            border_color,
            1,
        )

        text_x = panel_x + padding_x
        first_baseline_y = panel_y + padding_y + text_height
        for index, text in enumerate(panel_lines):
            cv2.putText(
                vis,
                text,
                (text_x, first_baseline_y + index * line_height),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )
        return panel_bottom

    # 공이 검출되면 기존 허들 패널과 같은 크기의 공 정보 패널을 우측 상단에 표시한다.
    next_right_panel_y = 4
    ball_detected = bool(payload.get("ball_detected", False))
    if ball_detected:
        ball_angle = payload.get("ball_angle_deg")
        ball_dx = payload.get("ball_x_distance_px")
        ball_dy = payload.get("ball_y_distance_px")
        ball_lines = [
            f"BALL:YES conf:{float(payload.get('ball_conf', 0.0)):.2f}",
            (
                f"ang:{float(ball_angle):+.1f} x:{float(ball_dx):+.0f} y:{float(ball_dy):.0f}px"
                if ball_angle is not None and ball_dx is not None and ball_dy is not None
                else "ang:N/A x:N/A y:N/A"
            ),
            f"RAW_BALL_IN_HAND:{str(bool(payload.get('raw_ball_in_hand', False))).upper()}",
        ]
        panel_bottom = draw_text_panel(
            ball_lines,
            panel_y=next_right_panel_y,
            font_scale=0.32,
            border_color=light_sky_blue,
            align="right",
        )
        next_right_panel_y = panel_bottom + 4

    # 허들 정보는 실제 허들이 검출된 동안에만 표시한다.
    hurdle_detected = bool(payload.get("hurdle_detected", False))
    if hurdle_detected:
        hurdle_line_angle = payload.get("hurdle_line_angle_deg")
        hurdle_lines = [
            f"HURDLE:YES conf:{float(payload.get('hurdle_conf', 0.0)):.2f}",
            (
                f"line angle:{float(hurdle_line_angle):+.1f}deg"
                if hurdle_line_angle is not None
                else "line angle:N/A"
            ),
        ]
        draw_text_panel(
            hurdle_lines,
            panel_y=next_right_panel_y,
            font_scale=0.32,
            border_color=light_sky_blue,
            align="right",
        )

    # 왼쪽 상단에는 프레임 판단 status 대신 실제 motion 노드의 실행
    # 모션을 표시합니다. 나머지 라인 검출 진단값은 그대로 유지합니다.
    line_lines = [
        *motion_overlay_lines(motion_state),
        f"pc:{payload['point_count']} dist:{payload['line_distance']:+.0f}px",
        f"ang:{payload['line_angle']:+.1f} tan:{payload['tangent_angle']:+.1f}",
        f"a:{payload['curve_a']:+.1e} f_ang:{payload['follow_angle']:+.1f}",
        f"tar:({payload['target_x']:.0f},{payload['target_y']:.0f}) fd:{payload['follow_distance']:.0f}",
        f"B:{int(payload['ball_detected'])}",
    ]
    draw_text_panel(
        line_lines,
        panel_y=4,
        font_scale=0.36,
        border_color=(180, 140, 255),
        align="left",
    )

    return vis


def analyze_frame_yolo(
    frame: np.ndarray,
    model,
    cfg: dict,
    line_status_publisher: Optional[LineStatusPublisher] = None,
    motion_state: Optional[MotionDisplayState] = None,
    raw_ball_in_hand_gate: Optional[ContinuousTrueGate] = None,
) -> tuple[dict, np.ndarray]:
    h, w = frame.shape[:2]
    dets = yolo_detect(model, frame, cfg)
    raw_line_points, roi_box = get_yolo_line_points(dets, w, h, cfg)

    # 주행용 직선/이차 피팅에는 검출된 모든 라인 중심점을 사용한다.
    line_points = raw_line_points
    payload = make_vision_payload(dets, line_points, w, h, cfg)
    if raw_ball_in_hand_gate is not None:
        payload["raw_ball_in_hand"] = raw_ball_in_hand_gate.update(
            bool(payload.get("raw_ball_in_hand", False))
        )
    payload = LINE_SMOOTHER.smooth(payload, w, h)
    payload = apply_line_status(payload, w, h, line_status_publisher)

    # 디버깅용으로 raw 개수도 같이 넣어둠. 알고리즘 쪽에서 안 쓰면 무시해도 됨.
    payload["raw_point_count"] = int(len(raw_line_points))

    vis = visualize_yolo(
        frame,
        dets,
        raw_line_points,
        payload,
        roi_box,
        cfg,
        motion_state,
    )
    return payload, vis


# ═══════════════════════════════════════════════════════
#  ROS2 노드
# ═══════════════════════════════════════════════════════

def main_ros2(ini_path: str = "settings.ini"):
    import rclpy
    from msgs.msg import MotionCommand, MotionEnd
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String
    from cv_bridge import CvBridge

    cfg = load_config(ini_path)

    class YoloVisionNode(Node):
        def __init__(self):
            super().__init__("yolo_vision")
            self.cfg = cfg
            self.bridge = CvBridge()
            self.frame_count = 0
            self.inference_failures = 0
            self.last_model_reload = 0.0
            self.motion_display_state = MotionDisplayState()
            self.raw_ball_in_hand_gate = ContinuousTrueGate(
                hold_seconds=float(
                    self.cfg.get("raw_ball_in_hand_hold_seconds", 0.3)
                )
            )
            self.model = load_yolo_model(self.cfg)
            # YOLO is interested in the newest camera frame only.  A deep reliable
            # queue retains stale full-resolution images while TensorRT is busy and
            # can exhaust Jetson's shared CPU/GPU memory.
            from rclpy.qos import qos_profile_sensor_data
            self.sub = self.create_subscription(
                Image, "/camera/image_raw", self.cb_image, qos_profile_sensor_data
            )
            self.pub_state = self.create_publisher(String, "/line_tracker/state", 10)
            self.pub_raw_ball_in_hand = self.create_publisher(
                Bool,
                "/raw_ball_in_hand",
                10,
            )
            self.pub_debug = self.create_publisher(Image, "/line_tracker/debug_image", 10)
            self.line_status_publisher = LineStatusPublisher(self)
            self.motion_command_sub = self.create_subscription(
                MotionCommand,
                "/motion_command",
                self.cb_motion_command,
                10,
            )
            motion_state_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.motion_end_sub = self.create_subscription(
                MotionEnd,
                "/motion_end",
                self.cb_motion_end,
                motion_state_qos,
            )
            self.hurdle_status_publisher = (
                HurdleStatusPublisher(self)
                if self.cfg.get("publish_hurdle_result", False)
                else None
            )

            # 공의 최종 BallStatus 판단은 별도 ball_vision_fusion.py가 담당한다.
            # 이 노드는 /line_tracker/state에 공 검출값과 기준점 대비
            # angle/x/y 픽셀 거리를 담아 웹캠 Vision 결과를 전달한다.
            self.get_logger().info(f"YoloVisionNode started cfg={ini_path}")
            self.get_logger().info(
                "YOLO direct hurdle_result publish="
                f"{bool(self.hurdle_status_publisher is not None)}"
            )

        def cb_motion_command(self, msg: MotionCommand):
            self.motion_display_state.on_command(msg.command)

        def cb_motion_end(self, msg: MotionEnd):
            self.motion_display_state.on_motion_state(
                msg.motion_end,
                msg.motion_ready,
            )

        def publish_hurdle_status(self, payload: dict):
            """YOLO 단독 모드에서만 hurdle_result를 직접 발행한다."""
            if self.hurdle_status_publisher is None:
                # hurdle_detected/x/y/conf/bbox는 그대로 /line_tracker/state에 남는다.
                # 최종 hurdle_result는 hurdle_vision_fusion.py가 발행한다.
                payload["hurdle_result_publisher"] = "hurdle_vision_fusion"
                return

            hurdle_status, hurdle_angle, hurdle_ready = (
                self.hurdle_status_publisher.publish_hurdle_status(
                    hurdle_detected=bool(payload.get("hurdle_detected", False)),
                    line_point_count=int(payload.get("point_count", 0)),
                    line_follow_angle_deg=payload.get("follow_angle"),
                    line_second_point_distance_px=payload.get(
                        "line_second_point_distance_px"
                    )
                    or 0.0,
                    line_angle_deg=payload.get("hurdle_line_angle_deg")
                    or 0.0,
                )
            )
            payload["hurdle_status"] = int(hurdle_status)
            payload["hurdle_status_angle"] = float(hurdle_angle)
            payload["hurdle_ready"] = bool(hurdle_ready)
            payload["hurdle_result_publisher"] = "yolo_vision"

        def cb_image(self, msg: Image):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                if self.cfg["flip_vertical"]:
                    frame = cv2.flip(frame, 0)

                payload, vis = analyze_frame_yolo(
                    frame,
                    self.model,
                    self.cfg,
                    self.line_status_publisher,
                    self.motion_display_state,
                    self.raw_ball_in_hand_gate,
                )
                self.inference_failures = 0
            except Exception:
                self.inference_failures += 1
                self.raw_ball_in_hand_gate.reset()
                self.pub_raw_ball_in_hand.publish(Bool(data=False))
                self.get_logger().error(
                    "GPU inference failed; keeping YOLO node alive "
                    f"(consecutive={self.inference_failures})\n{traceback.format_exc()}"
                )

                # A transient TensorRT/CUDA allocation failure must not take down
                # the ROS process.  Release the wrapper and reload at most once
                # every five seconds; no CPU model fallback is used.
                now = time.monotonic()
                if now - self.last_model_reload >= 5.0:
                    self.last_model_reload = now
                    try:
                        self.model = None
                        gc.collect()
                        self.model = load_yolo_model(self.cfg)
                        self.get_logger().warn("TensorRT model reloaded after inference failure")
                    except Exception:
                        self.get_logger().error(
                            "TensorRT model reload failed; will retry on a later frame\n"
                            + traceback.format_exc()
                        )
                return
            self.publish_hurdle_status(payload)
            self.pub_raw_ball_in_hand.publish(
                Bool(data=bool(payload.get("raw_ball_in_hand", False)))
            )
            self.pub_state.publish(String(data=json.dumps(payload, ensure_ascii=False)))

            debug_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)

            # ROS 모드에서도 settings.ini의 show_window 설정을 적용한다.
            if self.cfg["show_window"]:
                cv2.imshow("YOLO Vision", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rclpy.shutdown()

            self.frame_count += 1
            if self.frame_count % self.cfg["print_every_n_frames"] == 0:
                self.get_logger().info(
                    f"[{self.frame_count}] status={payload['status']}({payload['status_name']}) "
                    f"pc={payload['point_count']} "
                    f"dist={payload['line_distance']:+.0f}px "
                    f"ang={payload['line_angle']:+.1f} "
                    f"a={payload['curve_a']:+.2e} "
                    f"ball={payload['ball_detected']} hurdle={payload['hurdle_detected']} "
                    f"h_dist={payload.get('line_second_point_distance_px')} "
                    f"h_line_ang={payload.get('hurdle_line_angle_deg')}"
                )

    rclpy.init()
    node = YoloVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    if rclpy.ok():
        rclpy.shutdown()


# ═══════════════════════════════════════════════════════
#  단독 실행: PC/Jetson에서 웹캠 테스트
# ═══════════════════════════════════════════════════════

def main_standalone(ini_path: str = "settings.ini"):
    cfg = load_config(ini_path)
    model = load_yolo_model(cfg)
    raw_ball_in_hand_gate = ContinuousTrueGate(
        hold_seconds=float(cfg.get("raw_ball_in_hand_hold_seconds", 0.3))
    )

    cap = cv2.VideoCapture(cfg["cam_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["cam_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["cam_height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["cam_fps"])

    writer = None
    if cfg["save_video"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cfg["save_video"], fourcc, cfg["cam_fps"],
                                 (cfg["cam_width"], cfg["cam_height"]))

    frame_count = 0
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cfg["flip_vertical"]:
            frame = cv2.flip(frame, 0)

        t0 = time.perf_counter()
        payload, vis = analyze_frame_yolo(
            frame,
            model,
            cfg,
            raw_ball_in_hand_gate=raw_ball_in_hand_gate,
        )
        process_ms = (time.perf_counter() - t0) * 1000.0
        payload["process_ms"] = float(process_ms)

        cv2.putText(vis, f"Time: {process_ms:.1f} ms", (vis.shape[1] - 185, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

        if cfg["show_window"]:
            cv2.imshow("YOLO Vision", vis)
        if writer:
            writer.write(vis)

        frame_count += 1
        if frame_count % cfg["print_every_n_frames"] == 0:
            print(json.dumps(payload, ensure_ascii=False))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    ini = next((a for a in sys.argv[1:] if a.endswith(".ini")), "settings.ini")
    if "--ros2" in sys.argv:
        main_ros2(ini)
    else:
        main_standalone(ini)
