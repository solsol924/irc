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
- ball / hurdle detection 정보

공 관련 주의:
- 이 파일은 웹캠 YOLO의 ball_detected/ball_x/ball_y만 /line_tracker/state로 전달합니다.
- RealSense와의 결합 및 ball_result 발행은 ball_vision_fusion.py가 담당합니다.
"""

import configparser
import cv2
import numpy as np
import time
import math
import json
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


LINE_REPRESENTATIVE_POINT_COUNT = 3  #라인 대표점 최대 개수 


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


# ═══════════════════════════════════════════════════════
#  라인 모션 판단 코드
# ═══════════════════════════════════════════════════════

LINE_STATUS_NAME = {
    LineStatus.Forward_4step: "Forward_4step",
    LineStatus.Left_Half_Forward: "Left_Half_Forward",
    LineStatus.Right_Half_Forward: "Right_Half_Forward",
    LineStatus.Left_Forward: "Left_Forward",
    LineStatus.Right_Forward: "Right_Forward",
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
        "robot_center_x": frame_w / 2.0,
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
        "ball_conf": 0.20,
        "hurdle_conf": 0.35,
        "yolo_imgsz": 640,
        "yolo_device": "0",
        "line_class": "line",
        "ball_class": "ball",
        "hurdle_class": "hurdle",

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

        "roi_top_ratio": gf("detection", "roi_top_ratio", defaults["roi_top_ratio"]),
        "roi_bottom_ratio": gf("detection", "roi_bottom_ratio", defaults["roi_bottom_ratio"]),
        "roi_left_ratio": gf("detection", "roi_left_ratio", defaults["roi_left_ratio"]),
        "roi_right_ratio": gf("detection", "roi_right_ratio", defaults["roi_right_ratio"]),
        "min_points_for_poly": gi("curve", "min_points_for_poly", defaults["min_points_for_poly"]),

        "yolo_model": gs("yolo", "model", defaults["yolo_model"]),
        "yolo_conf": gf("yolo", "conf", defaults["yolo_conf"]),
        "line_conf": gf("yolo", "line_conf", defaults["line_conf"]),
        "ball_conf": gf("yolo", "ball_conf", defaults["ball_conf"]),
        "hurdle_conf": gf("yolo", "hurdle_conf", defaults["hurdle_conf"]),
        "yolo_imgsz": gi("yolo", "imgsz", defaults["yolo_imgsz"]),
        "yolo_device": gs("yolo", "device", defaults["yolo_device"]),
        "line_class": gs("yolo", "line_class", defaults["line_class"]),
        "ball_class": gs("yolo", "ball_class", defaults["ball_class"]),
        "hurdle_class": gs("yolo", "hurdle_class", defaults["hurdle_class"]),

        "min_visible_ratio": gf("visibility", "min_visible_ratio", defaults["min_visible_ratio"]),
        "reject_edge_cut_objects": gb("visibility", "reject_edge_cut_objects", defaults["reject_edge_cut_objects"]),
        "edge_margin": gi("visibility", "edge_margin", defaults["edge_margin"]),
        "partial_filter_classes": gs("visibility", "partial_filter_classes", defaults["partial_filter_classes"]),

        "show_window": gb("output", "show_window", defaults["show_window"]),
        "save_video": gs("output", "save_video", defaults["save_video"]),
        "print_every_n_frames": gi("output", "print_every_n_frames", defaults["print_every_n_frames"]),
    })
    return cfg


# ═══════════════════════════════════════════════════════
#  기존 detect_line.py의 2차 피팅 로직 유지
# ═══════════════════════════════════════════════════════

def band_sample(centroids, roi_top, roi_bottom, n_bands):
    """
    검출된 line bbox 중심점으로 주행용 대표점을 만든다.

    점이 n_bands개 이하면 bbox 중심점을 그대로 사용한다. 따라서 라인이 2개
    검출된 경우 두 핑크 점은 각각의 bbox 중앙에 놓인다. 점이 n_bands개
    이상이면 화면 아래쪽부터 균등하게 n_bands그룹으로 나누므로, 고정 높이
    구간에 검출이 몰려도 항상 n_bands개의 대표점이 만들어진다.
    """
    if not centroids:
        return []
    if n_bands <= 0:
        return sorted(centroids, key=lambda p: -p[1])

    sorted_points = sorted(centroids, key=lambda p: -p[1])
    if len(sorted_points) <= n_bands:
        return [(float(cx), float(cy)) for cx, cy in sorted_points]

    result = []
    for group in np.array_split(np.asarray(sorted_points, dtype=np.float64), n_bands):
        result.append((float(np.median(group[:, 0])), float(np.mean(group[:, 1]))))
    return result


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


# ═══════════════════════════════════════════════════════
#  YOLO detection
# ═══════════════════════════════════════════════════════

def load_yolo_model(cfg: dict):
    if YOLO is None:
        raise RuntimeError("ultralytics가 설치되어 있지 않습니다. pip install -U ultralytics 후 실행하세요.")
    model_path = cfg["yolo_model"]
    print(f"[YOLO] loading model: {model_path}")
    return YOLO(model_path)


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

def make_line_payload(line_points: list[tuple[float, float]], frame_w: int, frame_h: int) -> dict:
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
    }

    if point_count == 0:
        return payload

    robot_x = frame_w / 2.0
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
    payload["follow_distance"] = float(math.hypot(target_x - robot_x, target_y - robot_y))
    # 점 1개: 로봇 중심선과 '로봇 중심 -> 검출점' 선 사이의 부호 있는 각도.
    # 화면 위쪽을 0도로 보고, 검출점이 오른쪽이면 +, 왼쪽이면 -이다.
    payload["follow_angle"] = float(math.degrees(math.atan2(target_x - robot_x, robot_y - target_y)))

    # 점 2개 이상이면 두 점을 이은 선과 로봇 중심선의 부호 있는 각도차 계산
    if point_count >= 2:
        p0 = line_points[0]  # 가까운 점
        p1 = line_points[1]  # 그다음 점
        dx = p1[0] - p0[0]
        dy_up = p0[1] - p1[1]  # 화면 위쪽 방향을 +로 보기 위함
        two_point_angle = float(math.degrees(math.atan2(dx, dy_up)))

        # 점 2개에서는 line_angle을 사용하지 않고 follow_angle로만 전달한다.
        if point_count == 2:
            payload["follow_angle"] = two_point_angle
        else:
            # 점 3개 이상에서만 line_angle을 사용한다.
            payload["line_angle"] = two_point_angle
            payload["tangent_angle"] = payload["line_angle"]

    # 점 3개 이상이면 2차함수 a값과 두 번째 점 접선 각도 계산
    if point_count >= 3:
        coeffs = fit_poly2(line_points)
        if coeffs is not None:
            a, b, _c = coeffs
            payload["curve_a"] = float(a)

            y2 = line_points[1][1]
            slope_dx_dy_down = 2.0 * a * y2 + b
            # 이미지 y는 아래로 증가하므로, 로봇 진행 방향인 위쪽 기준으로 부호 반전
            payload["tangent_angle"] = float(math.degrees(math.atan2(-slope_dx_dy_down, 1.0)))

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

def best_object_payload(dets: list[ObjectDetection], cfg: dict, class_key: str, frame_w: int, frame_h: int) -> dict:
    """ball/hurdle 중 confidence가 가장 높은 객체 하나를 payload로 변환."""
    class_name = cfg[f"{class_key}_class"]
    conf_thres = cfg[f"{class_key}_conf"]
    objs = [d for d in dets if d.name == class_name and d.conf >= conf_thres and visible_enough(d, frame_w, frame_h, cfg)]

    if not objs:
        return {
            f"{class_key}_detected": False,
            f"{class_key}_x": -1.0,
            f"{class_key}_y": -1.0,
            f"{class_key}_conf": 0.0,
            f"{class_key}_bbox": [],
        }

    best = max(objs, key=lambda d: d.conf)
    return {
        f"{class_key}_detected": True,
        f"{class_key}_x": float(best.cx),
        f"{class_key}_y": float(best.cy),
        f"{class_key}_conf": float(best.conf),
        f"{class_key}_bbox": [float(best.x1), float(best.y1), float(best.x2), float(best.y2)],
    }


def make_vision_payload(dets: list[ObjectDetection], line_points: list[tuple[float, float]], frame_w: int, frame_h: int, cfg: dict) -> dict:
    payload = make_line_payload(line_points, frame_w, frame_h)
    payload.update(best_object_payload(dets, cfg, "ball", frame_w, frame_h))
    payload.update(best_object_payload(dets, cfg, "hurdle", frame_w, frame_h))
    return payload


# ═══════════════════════════════════════════════════════
#  시각화
# ═══════════════════════════════════════════════════════

def visualize_yolo(frame: np.ndarray, dets: list[ObjectDetection], raw_line_points, line_points, payload: dict, roi_box, cfg: dict):
    vis = frame.copy()
    h, w = vis.shape[:2]
    roi_top, roi_bottom, roi_left, roi_right = roi_box

    cv2.rectangle(vis, (roi_left, roi_top), (roi_right, roi_bottom), (80, 80, 80), 2)
    cv2.line(vis, (w // 2, h), (w // 2, roi_top), (200, 200, 200), 1, cv2.LINE_AA)

    # YOLO boxes
    for d in dets:
        # Do not draw partially visible ball/hurdle boxes.
        # line is not filtered by default because bottom line markers are often partially visible.
        if not visible_enough(d, w, h, cfg):
            continue

        if d.name == "line":
            color = (0, 255, 255)
        elif d.name == "ball":
            color = (0, 180, 255)
        elif d.name == "hurdle":
            color = (255, 100, 0)
        else:
            color = (180, 180, 180)

        x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{d.name} {d.conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # raw line bbox centers
    for cx, cy in raw_line_points:
        cv2.circle(vis, (int(cx), int(cy)), 4, (255, 255, 255), -1)

    # band sampled / geometry points
    for i, (cx, cy) in enumerate(line_points):
        cv2.circle(vis, (int(cx), int(cy)), 8, (255, 0, 255), -1)
        cv2.putText(vis, str(i), (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

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
        cv2.line(vis, (w // 2, h), (tx, ty), (0, 0, 255), 1, cv2.LINE_AA)

    # polynomial curve
    if len(line_points) >= 3:
        coeffs = fit_poly2(line_points)
        if coeffs is not None:
            a, b, c = coeffs
            pts_curve = []
            for y_px in range(roi_top, roi_bottom, 4):
                x_px = int(a * y_px**2 + b * y_px + c)
                if 0 <= x_px < w:
                    pts_curve.append((x_px, y_px))
            if len(pts_curve) > 1:
                cv2.polylines(vis, [np.array(pts_curve, dtype=np.int32).reshape(-1, 1, 2)],
                              False, (255, 0, 255), 2, cv2.LINE_AA)

    # text panel - keep the font size, but fit the panel to its contents.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.36
    font_thickness = 1
    lines = [
        f"st:{payload['status']} {payload['status_name'][:12]}",
        f"pc:{payload['point_count']} dist:{payload['line_distance']:+.0f}px",
        f"ang:{payload['line_angle']:+.1f} tan:{payload['tangent_angle']:+.1f}",
        f"a:{payload['curve_a']:+.1e} f_ang:{payload['follow_angle']:+.1f}",
        f"tar:({payload['target_x']:.0f},{payload['target_y']:.0f}) fd:{payload['follow_distance']:.0f}",
        f"B:{int(payload['ball_detected'])} H:{int(payload['hurdle_detected'])}",
    ]

    panel_x, panel_y = 4, 4
    padding_x, padding_y = 6, 5
    line_gap = 4
    text_sizes = [
        cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        for text in lines
    ]
    text_height = max(size[1] for size in text_sizes)
    line_height = text_height + line_gap
    panel_width = max(size[0] for size in text_sizes) + 2 * padding_x
    panel_height = (
        2 * padding_y
        + len(lines) * text_height
        + (len(lines) - 1) * line_gap
    )
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
        (255, 0, 255),
        1,
    )

    text_x = panel_x + padding_x
    first_baseline_y = panel_y + padding_y + text_height
    for i, text in enumerate(lines):
        cv2.putText(
            vis,
            text,
            (text_x, first_baseline_y + i * line_height),
            font,
            font_scale,
            (230, 230, 230),
            font_thickness,
            cv2.LINE_AA,
        )

    return vis


def analyze_frame_yolo(
    frame: np.ndarray,
    model,
    cfg: dict,
    line_status_publisher: Optional[LineStatusPublisher] = None,
) -> tuple[dict, np.ndarray]:
    h, w = frame.shape[:2]
    dets = yolo_detect(model, frame, cfg)
    raw_line_points, roi_box = get_yolo_line_points(dets, w, h, cfg)

    roi_top, roi_bottom, _roi_left, _roi_right = roi_box
    band_points = band_sample(
        raw_line_points,
        roi_top,
        roi_bottom,
        LINE_REPRESENTATIVE_POINT_COUNT,
    )

    # band point가 있으면 그걸 알고리즘용 point로 사용. 없으면 raw point 사용.
    line_points = band_points if band_points else raw_line_points
    payload = make_vision_payload(dets, line_points, w, h, cfg)
    payload = LINE_SMOOTHER.smooth(payload, w, h)
    payload = apply_line_status(payload, w, h, line_status_publisher)

    # 디버깅용으로 raw 개수도 같이 넣어둠. 알고리즘 쪽에서 안 쓰면 무시해도 됨.
    payload["raw_point_count"] = int(len(raw_line_points))

    vis = visualize_yolo(frame, dets, raw_line_points, line_points, payload, roi_box, cfg)
    return payload, vis


# ═══════════════════════════════════════════════════════
#  ROS2 노드
# ═══════════════════════════════════════════════════════

def main_ros2(ini_path: str = "settings.ini"):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge

    cfg = load_config(ini_path)

    class YoloVisionNode(Node):
        def __init__(self):
            super().__init__("yolo_vision")
            self.cfg = cfg
            self.bridge = CvBridge()
            self.frame_count = 0
            self.model = load_yolo_model(self.cfg)
            self.sub = self.create_subscription(Image, "/camera/image_raw", self.cb_image, 10)
            self.pub_state = self.create_publisher(String, "/line_tracker/state", 10)
            self.pub_debug = self.create_publisher(Image, "/line_tracker/debug_image", 10)
            self.line_status_publisher = LineStatusPublisher(self)
            self.hurdle_status_publisher = HurdleStatusPublisher(self)

            # 공의 최종 BallStatus 판단은 별도 ball_vision_fusion.py가 담당한다.
            # 이 노드는 /line_tracker/state에 ball_detected, ball_x, ball_y,
            # ball_conf, ball_bbox만 담아 웹캠 Vision 결과를 전달한다.
            self.get_logger().info(f"YoloVisionNode started cfg={ini_path}")

        def publish_hurdle_status(self, payload: dict):
            """
            허들 상태만 이 노드에서 발행한다.

            공 상태는 RealSense와 웹캠을 동시에 사용해야 하므로
            별도 ball_vision_fusion.py에서 ball_result를 발행한다.
            """
            hurdle_status, hurdle_angle = (
                self.hurdle_status_publisher.publish_hurdle_status(
                    hurdle_detected=bool(payload.get("hurdle_detected", False)),
                )
            )

            payload["hurdle_status"] = int(hurdle_status)
            payload["hurdle_status_angle"] = float(hurdle_angle)

        def cb_image(self, msg: Image):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.cfg["flip_vertical"]:
                frame = cv2.flip(frame, 0)

            payload, vis = analyze_frame_yolo(
                frame,
                self.model,
                self.cfg,
                self.line_status_publisher,
            )
            self.publish_hurdle_status(payload)
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
                    f"ball={payload['ball_detected']} hurdle={payload['hurdle_detected']}"
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
        payload, vis = analyze_frame_yolo(frame, model, cfg)
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
