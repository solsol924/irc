#!/usr/bin/env python3

"""
공 비전 융합 노드.

역할
1. RealSense color + aligned depth 영상에서 OpenCV로 주황색 공을 직접 검출한다.
2. 검출된 공 중심 픽셀과 depth를 이용해 공의 3차원 위치, 거리, 좌우 각도를 계산한다.
3. 웹캠 YOLO가 /line_tracker/state로 보내는 공 중심 좌표를 구독한다.
4. 후프 상태에서 백보드 중심 거리와 각도를 받는다.
5. RealSense + 웹캠 + 후프 값을 BallStatusPublisher에 전달한다.
6. 디버깅용으로 /ball/vision_state와 /ball/realsense_debug_image를 발행한다.

입력
- /camera/color/image_raw
- /camera/aligned_depth_to_color/image_raw
- /camera/color/camera_info
- /line_tracker/state
- /hoop/vision_state
- /raw_ball_in_hand

출력
- ball_result
- /ball/vision_state
- /ball/realsense_debug_image
- /vision/ball_active
- /vision/hoop_active

주의
- OpenCV HSV 값은 경기장 조명과 공 색상에 맞게 반드시 조정해야 한다.
- aligned depth 토픽을 사용하므로 color 픽셀과 depth 픽셀을 같은 좌표로 사용한다.
"""

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from ball_detector_core import (
    BallDepthSample,
    BallMatchConfig,
    BallSupportConfig,
    ball_support_confidence,
    black_support_mask,
    expanded_tracking_roi,
    evaluate_ball_support,
    hsv_range_mask,
    same_ball_candidate,
    sample_ball_inner_depth,
)
from ball_status_publisher import BallStatusPublisher


class BallVisionFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("ball_vision_fusion")

        source_config = (
            Path(__file__).resolve().parent.parent / "config" / "ball_hsv.yaml"
        )
        default_hsv_config = (
            source_config
            if source_config.exists()
            else Path.home()
            / "irc"
            / "src"
            / "vision"
            / "config"
            / "ball_hsv.yaml"
        )
        self.declare_parameter("hsv_config_file", str(default_hsv_config))
        self.hsv_config_path = Path(
            str(self.get_parameter("hsv_config_file").value)
        ).expanduser()
        hsv_defaults = self._load_hsv_defaults(self.hsv_config_path)

        # =========================================================
        # ROS 토픽
        # =========================================================
        self.declare_parameter(
            "realsense_color_topic",
            "/camera/color/image_raw",
        )
        self.declare_parameter(
            "realsense_depth_topic",
            "/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "realsense_camera_info_topic",
            "/camera/color/camera_info",
        )
        self.declare_parameter("use_realsense_yolo", False)
        self.declare_parameter(
            "realsense_yolo_state_topic",
            "/realsense_yolo/ball_state",
        )
        self.declare_parameter("webcam_state_topic", "/line_tracker/state")
        self.declare_parameter("hoop_state_topic", "/hoop/vision_state")
        self.declare_parameter("active_topic", "/vision/ball_active")
        self.declare_parameter("hoop_active_topic", "/vision/hoop_active")
        self.declare_parameter("active_on_start", True)
        self.declare_parameter("manage_activity_from_ball_in_hand", True)
        self.declare_parameter(
            "raw_ball_in_hand_topic",
            "/raw_ball_in_hand",
        )
        self.declare_parameter("vision_state_topic", "/ball/vision_state")
        self.declare_parameter(
            "realsense_debug_image_topic",
            "/ball/realsense_debug_image",
        )
        self.declare_parameter("ball_result_topic", "ball_result")

        # =========================================================
        # RealSense OpenCV 공 검출 파라미터
        # =========================================================
        # RealSense 전체 카메라 프레임을 검출 ROI로 사용한다.
        self.declare_parameter("rs_roi_left_ratio", 0.0)
        self.declare_parameter("rs_roi_right_ratio", 1.0)
        self.declare_parameter("rs_roi_top_ratio", 0.0)
        self.declare_parameter("rs_roi_bottom_ratio", 1.0)

        # 시작할 때 vision/config/ball_hsv.yaml의 마지막 확정값을 기본값으로
        # 사용한다. 개별 ROS 파라미터 override가 있으면 그것이 우선한다.
        self.declare_parameter("h_low", hsv_defaults["h_low"])
        self.declare_parameter("h_high", hsv_defaults["h_high"])
        self.declare_parameter("s_low", hsv_defaults["s_low"])
        self.declare_parameter("s_high", hsv_defaults["s_high"])
        self.declare_parameter("v_low", hsv_defaults["v_low"])
        self.declare_parameter("v_high", hsv_defaults["v_high"])

        # 공은 경기 규칙상 지름 150 mm의 검은 받침대 위에 있다. 다만 받침대는
        # 카메라 각도/흔들림에 따라 타원 또는 공 아래쪽 반달처럼 보일 수 있다.
        # 아래 값들은 후보를 즉시 폐기하는 hard gate가 아니라 어느 후보가 더
        # 공다운지 고르는 보조 점수에 사용한다.
        self.declare_parameter(
            "support_diameter_m", hsv_defaults["support_diameter_m"]
        )
        self.declare_parameter("support_v_max", hsv_defaults["support_v_max"])
        self.declare_parameter(
            "support_black_ratio_min",
            hsv_defaults["support_black_ratio_min"],
        )
        self.declare_parameter(
            "edge_support_black_ratio_min",
            hsv_defaults["edge_support_black_ratio_min"],
        )
        self.declare_parameter(
            "support_ball_color_ratio_max",
            hsv_defaults["support_ball_color_ratio_max"],
        )
        self.declare_parameter(
            "support_floor_ratio_max",
            hsv_defaults["support_floor_ratio_max"],
        )
        self.declare_parameter(
            "support_sector_black_ratio_min",
            hsv_defaults["support_sector_black_ratio_min"],
        )
        self.declare_parameter(
            "support_min_sectors", hsv_defaults["support_min_sectors"]
        )
        self.declare_parameter(
            "edge_support_min_sectors",
            hsv_defaults["edge_support_min_sectors"],
        )
        self.declare_parameter(
            "support_min_visible_fraction",
            hsv_defaults["support_min_visible_fraction"],
        )
        self.declare_parameter(
            "edge_support_min_visible_fraction",
            hsv_defaults["edge_support_min_visible_fraction"],
        )
        self.declare_parameter("floor_h_low", hsv_defaults["floor_h_low"])
        self.declare_parameter("floor_h_high", hsv_defaults["floor_h_high"])
        self.declare_parameter("floor_s_low", hsv_defaults["floor_s_low"])
        self.declare_parameter("floor_s_high", hsv_defaults["floor_s_high"])
        self.declare_parameter("floor_v_low", hsv_defaults["floor_v_low"])
        self.declare_parameter("floor_v_high", hsv_defaults["floor_v_high"])

        self.declare_parameter("depth_threshold_m", 1.5)
        self.declare_parameter("depth_scale", 0.001)  # 16UC1 mm -> m
        # HSV 보정 프로파일의 min_area=120과 실제 런타임을 맞춘다. 이전 300은
        # 1~1.5 m에서 작아진 공 또는 일부가 어두워진 공을 너무 쉽게 버렸다.
        self.declare_parameter("min_contour_area", 120.0)
        # 검은 무늬·반사광으로 HSV 원 내부가 비어도 허용한다. 이제 depth hole은
        # HSV 윤곽에서 제거하지 않으므로 이 값은 순수 색상 마스크 형상만 다룬다.
        self.declare_parameter("max_circle_ratio_error", 0.65)
        # 화면 경계에 걸려 일부가 잘린 공은 면적과 원형도 조건을 완화한다.
        self.declare_parameter("edge_ball_margin_px", 3)
        self.declare_parameter("edge_min_contour_area_ratio", 0.65)
        self.declare_parameter("edge_max_circle_ratio_error", 0.75)
        # 실제 공 지름은 약 5~7 cm이다. Depth와 CameraInfo로 영상에서
        # 예상되는 픽셀 반지름을 계산해 지나치게 크거나 작은 색상 물체를 제거한다.
        # 마스크가 공 전체를 채우지 않을 수 있어 초기 허용 범위는 넉넉하게 둔다.
        self.declare_parameter("ball_diameter_min_m", 0.050)
        self.declare_parameter("ball_diameter_max_m", 0.070)
        self.declare_parameter("radius_size_min_ratio", 0.45)
        self.declare_parameter("radius_size_max_ratio", 1.70)
        self.declare_parameter("edge_radius_size_min_ratio", 0.60)
        self.declare_parameter("edge_radius_size_max_ratio", 1.50)
        # 기존 0.55는 흔들림/무늬 때문에 찌그러진 실제 공을 자주 거부했다.
        # 시작값 0.38은 hsv_profiles.yaml의 보정값(0.33)에 가깝게 완화하되,
        # 종횡비·실물 크기·깊이·시간 일치 검사를 뒤에서 계속 적용한다.
        self.declare_parameter("min_circularity", 0.38)
        self.declare_parameter("edge_min_circularity", 0.30)
        self.declare_parameter("min_aspect_ratio", 0.55)
        self.declare_parameter("max_aspect_ratio", 1.80)
        self.declare_parameter("edge_min_aspect_ratio", 0.45)
        self.declare_parameter("edge_max_aspect_ratio", 2.20)
        self.declare_parameter("morph_kernel_size", 5)
        # 깊이는 중심 3x3 대신 검출 반지름의 안쪽 50% 원에서 읽는다.
        # 0.50은 현장 조정 시작값이다. depth가 부족하면 0.60~0.70, 배경이
        # 섞이면 0.40~0.45로 바꾼다. 이 파라미터 이름을 검색하면 실제 사용
        # 위치와 상세 설명을 쉽게 찾을 수 있다.
        self.declare_parameter("depth_inner_radius_ratio", 0.50)
        self.declare_parameter("depth_min_valid_pixels", 8)
        self.declare_parameter("depth_min_valid_ratio", 0.15)
        # 받침대 confidence가 후보 선택에 미치는 정도. 0이면 받침대 점수를
        # 사용하지 않고, 값이 클수록 검은 받침대가 잘 보이는 후보를 선호한다.
        self.declare_parameter("support_score_weight", 1.0)

        # 원본(raw) 검출 3개 중 같은 공 2개가 있어야 SEARCH를 확정한다.
        # TRACK 중에는 매 프레임 이 투표를 다시 하지 않는다. 아래 위치/깊이
        # 허용값은 보행 흔들림을 흡수하도록 의도적으로 넓게 시작한다.
        self.declare_parameter("confirmation_window_frames", 3)
        self.declare_parameter("confirmation_required_hits", 2)
        self.declare_parameter("confirm_center_distance_px", 80.0)
        self.declare_parameter("confirm_center_radius_scale", 4.0)
        self.declare_parameter("confirm_depth_absolute_m", 0.30)
        self.declare_parameter("confirm_depth_relative", 0.30)
        self.declare_parameter("confirm_radius_ratio_min", 0.45)
        self.declare_parameter("confirm_radius_ratio_max", 2.20)

        # 확정 뒤에는 예측 ROI를 먼저 검사한다. ROI에서 못 찾거나 직전 공과
        # 일치하지 않으면 같은 프레임에서 전체 설정 ROI를 즉시 다시 검사하므로
        # ROI 확대가 로봇 정지시간이나 프레임 대기를 추가하지 않는다.
        self.declare_parameter("tracking_roi_enabled", True)
        self.declare_parameter("tracking_roi_radius_scale", 4.0)
        self.declare_parameter("tracking_roi_expansion_per_miss", 1.5)
        self.declare_parameter("tracking_roi_min_half_size_px", 48.0)
        self.declare_parameter("realsense_hold_frames", 3)

        # 15 FPS 프레임 간격은 약 0.067초이다. 0.1초 slop은 서로 인접한 다른
        # 프레임을 짝지을 수 있어 기본 0.03초로 줄이고, 짧은 지연은 queue가
        # 흡수하도록 한다.
        self.declare_parameter("realsense_sync_queue", 5)
        self.declare_parameter("realsense_sync_slop_sec", 0.03)

        # CameraInfo를 아직 받지 못했을 때 사용할 선배 코드의 기본 내부 파라미터
        self.declare_parameter("fallback_fx", 607.0)
        self.declare_parameter("fallback_fy", 606.0)
        self.declare_parameter("fallback_cx", 325.5)
        self.declare_parameter("fallback_cy", 239.4)

        # 디버그 영상 발행 여부. imshow는 headless 환경 문제 때문에 기본 False.
        self.declare_parameter("publish_realsense_debug_image", True)
        self.declare_parameter("show_realsense_window", False)
        self.declare_parameter("debug_mask_preview_width", 96)
        self.declare_parameter("debug_publish_every_n_frames", 2)

        # =========================================================
        # 웹캠 기하 파라미터
        # =========================================================
        self.declare_parameter("webcam_frame_width", 640.0)
        self.declare_parameter("webcam_robot_center_x", 320.0)
        self.declare_parameter("webcam_robot_center_y", 420.0)
        self.declare_parameter("webcam_fov_x_deg", 60.0)

        # =========================================================
        # 유효시간 및 출력 주기
        # =========================================================
        self.declare_parameter("realsense_timeout_sec", 0.5)
        self.declare_parameter("webcam_timeout_sec", 0.5)
        self.declare_parameter("hoop_timeout_sec", 0.5)
        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("print_every_n_frames", 10)
        self.declare_parameter("realsense_use_euclidean_distance", False)

        # =========================================================
        # 파라미터 로드
        # =========================================================
        self.realsense_color_topic = str(
            self.get_parameter("realsense_color_topic").value
        )
        self.realsense_depth_topic = str(
            self.get_parameter("realsense_depth_topic").value
        )
        self.realsense_camera_info_topic = str(
            self.get_parameter("realsense_camera_info_topic").value
        )
        self.use_realsense_yolo = bool(
            self.get_parameter("use_realsense_yolo").value
        )
        self.realsense_yolo_state_topic = str(
            self.get_parameter("realsense_yolo_state_topic").value
        )
        self.webcam_state_topic = str(
            self.get_parameter("webcam_state_topic").value
        )
        self.hoop_state_topic = str(
            self.get_parameter("hoop_state_topic").value
        )
        self.active_topic = str(self.get_parameter("active_topic").value)
        self.hoop_active_topic = str(
            self.get_parameter("hoop_active_topic").value
        )
        self.manage_activity_from_ball_in_hand = bool(
            self.get_parameter("manage_activity_from_ball_in_hand").value
        )
        self.raw_ball_in_hand_topic = str(
            self.get_parameter("raw_ball_in_hand_topic").value
        )
        self.vision_state_topic = str(
            self.get_parameter("vision_state_topic").value
        )
        self.realsense_debug_image_topic = str(
            self.get_parameter("realsense_debug_image_topic").value
        )
        self.ball_result_topic = str(
            self.get_parameter("ball_result_topic").value
        )

        self.rs_roi_left_ratio = float(
            self.get_parameter("rs_roi_left_ratio").value
        )
        self.rs_roi_right_ratio = float(
            self.get_parameter("rs_roi_right_ratio").value
        )
        self.rs_roi_top_ratio = float(
            self.get_parameter("rs_roi_top_ratio").value
        )
        self.rs_roi_bottom_ratio = float(
            self.get_parameter("rs_roi_bottom_ratio").value
        )

        self.h_low = int(self.get_parameter("h_low").value)
        self.h_high = int(self.get_parameter("h_high").value)
        self.s_low = int(self.get_parameter("s_low").value)
        self.s_high = int(self.get_parameter("s_high").value)
        self.v_low = int(self.get_parameter("v_low").value)
        self.v_high = int(self.get_parameter("v_high").value)

        self.support_diameter_m = float(
            self.get_parameter("support_diameter_m").value
        )
        self.support_v_max = int(
            self.get_parameter("support_v_max").value
        )
        self.support_black_ratio_min = float(
            self.get_parameter("support_black_ratio_min").value
        )
        self.edge_support_black_ratio_min = float(
            self.get_parameter("edge_support_black_ratio_min").value
        )
        self.support_ball_color_ratio_max = float(
            self.get_parameter("support_ball_color_ratio_max").value
        )
        self.support_floor_ratio_max = float(
            self.get_parameter("support_floor_ratio_max").value
        )
        self.support_sector_black_ratio_min = float(
            self.get_parameter("support_sector_black_ratio_min").value
        )
        self.support_min_sectors = int(
            self.get_parameter("support_min_sectors").value
        )
        self.edge_support_min_sectors = int(
            self.get_parameter("edge_support_min_sectors").value
        )
        self.support_min_visible_fraction = float(
            self.get_parameter("support_min_visible_fraction").value
        )
        self.edge_support_min_visible_fraction = float(
            self.get_parameter("edge_support_min_visible_fraction").value
        )
        self.floor_h_low = int(self.get_parameter("floor_h_low").value)
        self.floor_h_high = int(self.get_parameter("floor_h_high").value)
        self.floor_s_low = int(self.get_parameter("floor_s_low").value)
        self.floor_s_high = int(self.get_parameter("floor_s_high").value)
        self.floor_v_low = int(self.get_parameter("floor_v_low").value)
        self.floor_v_high = int(self.get_parameter("floor_v_high").value)

        self.depth_threshold_m = float(
            self.get_parameter("depth_threshold_m").value
        )
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_contour_area = float(
            self.get_parameter("min_contour_area").value
        )
        self.max_circle_ratio_error = float(
            self.get_parameter("max_circle_ratio_error").value
        )
        self.edge_ball_margin_px = max(
            0, int(self.get_parameter("edge_ball_margin_px").value)
        )
        self.edge_min_contour_area_ratio = float(
            self.get_parameter("edge_min_contour_area_ratio").value
        )
        self.edge_max_circle_ratio_error = float(
            self.get_parameter("edge_max_circle_ratio_error").value
        )
        self.ball_diameter_min_m = float(
            self.get_parameter("ball_diameter_min_m").value
        )
        self.ball_diameter_max_m = float(
            self.get_parameter("ball_diameter_max_m").value
        )
        self.radius_size_min_ratio = float(
            self.get_parameter("radius_size_min_ratio").value
        )
        self.radius_size_max_ratio = float(
            self.get_parameter("radius_size_max_ratio").value
        )
        self.edge_radius_size_min_ratio = float(
            self.get_parameter("edge_radius_size_min_ratio").value
        )
        self.edge_radius_size_max_ratio = float(
            self.get_parameter("edge_radius_size_max_ratio").value
        )
        self.min_circularity = float(
            self.get_parameter("min_circularity").value
        )
        self.edge_min_circularity = float(
            self.get_parameter("edge_min_circularity").value
        )
        self.min_aspect_ratio = float(
            self.get_parameter("min_aspect_ratio").value
        )
        self.max_aspect_ratio = float(
            self.get_parameter("max_aspect_ratio").value
        )
        self.edge_min_aspect_ratio = float(
            self.get_parameter("edge_min_aspect_ratio").value
        )
        self.edge_max_aspect_ratio = float(
            self.get_parameter("edge_max_aspect_ratio").value
        )
        self.depth_inner_radius_ratio = float(
            self.get_parameter("depth_inner_radius_ratio").value
        )
        self.depth_min_valid_pixels = max(
            1, int(self.get_parameter("depth_min_valid_pixels").value)
        )
        self.depth_min_valid_ratio = float(
            self.get_parameter("depth_min_valid_ratio").value
        )
        self.support_score_weight = max(
            0.0, float(self.get_parameter("support_score_weight").value)
        )
        self.confirmation_window_frames = max(
            1,
            int(self.get_parameter("confirmation_window_frames").value),
        )
        self.confirmation_required_hits = max(
            1,
            min(
                self.confirmation_window_frames,
                int(
                    self.get_parameter(
                        "confirmation_required_hits"
                    ).value
                ),
            ),
        )
        self.ball_match_config = BallMatchConfig(
            center_distance_px=float(
                self.get_parameter("confirm_center_distance_px").value
            ),
            center_radius_scale=float(
                self.get_parameter("confirm_center_radius_scale").value
            ),
            depth_absolute_m=float(
                self.get_parameter("confirm_depth_absolute_m").value
            ),
            depth_relative=float(
                self.get_parameter("confirm_depth_relative").value
            ),
            radius_ratio_min=float(
                self.get_parameter("confirm_radius_ratio_min").value
            ),
            radius_ratio_max=float(
                self.get_parameter("confirm_radius_ratio_max").value
            ),
        )
        self.tracking_roi_enabled = bool(
            self.get_parameter("tracking_roi_enabled").value
        )
        self.tracking_roi_radius_scale = float(
            self.get_parameter("tracking_roi_radius_scale").value
        )
        self.tracking_roi_expansion_per_miss = float(
            self.get_parameter(
                "tracking_roi_expansion_per_miss"
            ).value
        )
        self.tracking_roi_min_half_size_px = float(
            self.get_parameter("tracking_roi_min_half_size_px").value
        )
        self.realsense_hold_frames = max(
            0, int(self.get_parameter("realsense_hold_frames").value)
        )
        self.realsense_sync_queue = max(
            2, int(self.get_parameter("realsense_sync_queue").value)
        )
        self.realsense_sync_slop_sec = max(
            0.0,
            float(self.get_parameter("realsense_sync_slop_sec").value),
        )

        kernel_size = max(
            1, int(self.get_parameter("morph_kernel_size").value)
        )
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.morph_kernel_size = kernel_size
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        self.fx = float(self.get_parameter("fallback_fx").value)
        self.fy = float(self.get_parameter("fallback_fy").value)
        self.cx_intr = float(self.get_parameter("fallback_cx").value)
        self.cy_intr = float(self.get_parameter("fallback_cy").value)
        self.camera_info_received = False

        self.publish_realsense_debug_image = bool(
            self.get_parameter("publish_realsense_debug_image").value
        )
        self.show_realsense_window = bool(
            self.get_parameter("show_realsense_window").value
        )
        self.debug_mask_preview_width = max(
            0,
            int(self.get_parameter("debug_mask_preview_width").value),
        )
        self.debug_publish_every_n_frames = max(
            1,
            int(
                self.get_parameter(
                    "debug_publish_every_n_frames"
                ).value
            ),
        )

        self.webcam_frame_width = float(
            self.get_parameter("webcam_frame_width").value
        )
        self.webcam_robot_center_x = float(
            self.get_parameter("webcam_robot_center_x").value
        )
        self.webcam_robot_center_y = float(
            self.get_parameter("webcam_robot_center_y").value
        )
        self.webcam_fov_x_deg = float(
            self.get_parameter("webcam_fov_x_deg").value
        )

        self.realsense_timeout_sec = float(
            self.get_parameter("realsense_timeout_sec").value
        )
        self.webcam_timeout_sec = float(
            self.get_parameter("webcam_timeout_sec").value
        )
        self.hoop_timeout_sec = float(
            self.get_parameter("hoop_timeout_sec").value
        )
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.print_every_n_frames = max(
            1,
            int(self.get_parameter("print_every_n_frames").value),
        )
        self.realsense_use_euclidean_distance = bool(
            self.get_parameter("realsense_use_euclidean_distance").value
        )

        self._update_hsv_arrays()
        self.add_on_set_parameters_callback(self.parameter_callback)

        # =========================================================
        # 상태
        # =========================================================
        self.bridge = CvBridge()

        self.latest_realsense: Optional[Dict[str, Any]] = None
        self.latest_realsense_time = 0.0
        self.last_realsense_detection: Optional[Dict[str, Any]] = None
        self.realsense_lost_frames = self.realsense_hold_frames
        self.realsense_track_confirmed = False
        self.realsense_track_velocity_px = (0.0, 0.0)
        self.raw_detection_history = deque(
            maxlen=self.confirmation_window_frames
        )

        self.latest_webcam: Optional[Dict[str, Any]] = None
        self.latest_webcam_time = 0.0
        self.latest_hoop: Optional[Dict[str, Any]] = None
        self.latest_hoop_time = 0.0

        self.ball_in_hand = False
        self.frame_count = 0
        self.realsense_frame_count = 0
        self.last_realsense_diagnostic_label: Optional[str] = None
        self.ball_detection_active = bool(
            self.get_parameter("active_on_start").value
        )
        self.managed_hoop_active: Optional[bool] = None
        self.image_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # =========================================================
        # ROS I/O
        # =========================================================
        # RealSense 구독과 synchronizer는 프로세스 수명 동안 유지한다.
        # 모드 전환 때 DDS 구독을 삭제/재생성하면 첫 프레임까지 공백이
        # 생기거나 message_filters가 다시 채워지는 동안 화면이 멈출 수 있다.
        # 비활성 모드에서는 콜백 초입에서 즉시 반환해 OpenCV 연산만 쉰다.
        self.rs_color_sub = None
        self.rs_depth_sub = None
        self.rs_sync = None

        self.activity_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_active = self.create_subscription(
            Bool,
            self.active_topic,
            self.cb_ball_active,
            self.activity_qos,
        )
        self.pub_ball_active = self.create_publisher(
            Bool,
            self.active_topic,
            self.activity_qos,
        )
        self.pub_hoop_active = self.create_publisher(
            Bool,
            self.hoop_active_topic,
            self.activity_qos,
        )

        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            self.realsense_camera_info_topic,
            self.cb_camera_info,
            10,
        )
        self.sub_webcam = self.create_subscription(
            String,
            self.webcam_state_topic,
            self.cb_webcam_state,
            10,
        )
        self.sub_hoop_state = self.create_subscription(
            String,
            self.hoop_state_topic,
            self.cb_hoop_state,
            10,
        )
        self.sub_raw_ball_in_hand = self.create_subscription(
            Bool,
            self.raw_ball_in_hand_topic,
            self.cb_raw_ball_in_hand,
            10,
        )
        self.sub_realsense_yolo_state = None
        if self.use_realsense_yolo:
            self.sub_realsense_yolo_state = self.create_subscription(
                String,
                self.realsense_yolo_state_topic,
                self.cb_realsense_yolo_state,
                10,
            )
            self.get_logger().info(
                "RealSense ball source: YOLO "
                f"{self.realsense_yolo_state_topic}"
            )

        self.ball_status_publisher = BallStatusPublisher(
            self,
            topic_name=self.ball_result_topic,
        )

        if self.manage_activity_from_ball_in_hand:
            self._set_vision_mode_from_ball_in_hand(False, force=True)

        self.pub_vision_state = self.create_publisher(
            String,
            self.vision_state_topic,
            10,
        )
        self.pub_realsense_debug = self.create_publisher(
            Image,
            self.realsense_debug_image_topic,
            self.image_qos,
        )

        timer_period = 1.0 / max(self.publish_hz, 1.0)
        self.timer = self.create_timer(
            timer_period,
            self.publish_ball_features,
        )

        if self.use_realsense_yolo:
            self.get_logger().info(
                "Legacy RealSense HSV/OpenCV ball detector disabled."
            )
        else:
            self._start_ball_image_subscriptions()

        self.get_logger().info("BallVisionFusionNode started.")
        self.get_logger().info(
            f"RealSense color: {self.realsense_color_topic}"
        )
        self.get_logger().info(
            f"RealSense aligned depth: {self.realsense_depth_topic}"
        )
        self.get_logger().info(
            f"Webcam YOLO input: {self.webcam_state_topic}"
        )
        self.get_logger().info(
            f"Hoop input: {self.hoop_state_topic}"
        )
        self.get_logger().info(
            f"BallResult output: {self.ball_result_topic}"
        )
        self.get_logger().info(
            "Ball image detection: "
            f"{'ON' if self.ball_detection_active else 'OFF'}"
        )

    def _load_hsv_defaults(self, path: Path) -> Dict[str, Any]:
        """시작할 때 마지막으로 확정한 공·받침대·바닥 보정값을 읽는다."""
        fallback: Dict[str, Any] = {
            "h_low": 8,
            "h_high": 60,
            "s_low": 60,
            "s_high": 255,
            "v_low": 60,
            "v_high": 255,
            "support_diameter_m": 0.150,
            "support_v_max": 75,
            "support_black_ratio_min": 0.30,
            "edge_support_black_ratio_min": 0.25,
            "support_ball_color_ratio_max": 0.15,
            "support_floor_ratio_max": 0.35,
            "support_sector_black_ratio_min": 0.18,
            "support_min_sectors": 3,
            "edge_support_min_sectors": 2,
            "support_min_visible_fraction": 0.45,
            "edge_support_min_visible_fraction": 0.12,
            "floor_h_low": 0,
            "floor_h_high": 12,
            "floor_s_low": 50,
            "floor_s_high": 255,
            "floor_v_low": 25,
            "floor_v_high": 255,
        }
        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            raw = (
                data.get("ball_vision_fusion", {})
                .get("ros__parameters", {})
            )
            values = dict(fallback)
            integer_names = {
                "h_low", "h_high", "s_low", "s_high", "v_low", "v_high",
                "support_v_max", "support_min_sectors",
                "edge_support_min_sectors", "floor_h_low", "floor_h_high",
                "floor_s_low", "floor_s_high", "floor_v_low", "floor_v_high",
            }
            for name, default in fallback.items():
                raw_value = raw.get(name, default)
                values[name] = (
                    int(raw_value)
                    if name in integer_names
                    else float(raw_value)
                )
        except (
            AttributeError,
            OSError,
            TypeError,
            ValueError,
            yaml.YAMLError,
        ) as exc:
            self.get_logger().warning(
                f"Could not load ball detector calibration from {path}: {exc}; "
                "using built-in defaults"
            )
            return fallback

        valid = (
            0 <= values["h_low"] <= values["h_high"] <= 179
            and 0 <= values["s_low"] <= values["s_high"] <= 255
            and 0 <= values["v_low"] <= values["v_high"] <= 255
            and 0 <= values["support_v_max"] <= 255
            and 0 <= values["floor_h_low"] <= 179
            and 0 <= values["floor_h_high"] <= 179
            and 0 <= values["floor_s_low"] <= values["floor_s_high"] <= 255
            and 0 <= values["floor_v_low"] <= values["floor_v_high"] <= 255
        )
        if not valid:
            self.get_logger().warning(
                f"Invalid ball detector calibration in {path}; "
                "using built-in defaults"
            )
            return fallback

        self.get_logger().info(
            f"Loaded ball/support/floor calibration from {path}: "
            f"H={values['h_low']}..{values['h_high']} "
            f"S={values['s_low']}..{values['s_high']} "
            f"V={values['v_low']}..{values['v_high']} "
            f"support_V<={values['support_v_max']}"
        )
        return values

    # =============================================================
    # 파라미터
    # =============================================================
    def _update_hsv_arrays(self) -> None:
        self.lower_hsv = np.array(
            [self.h_low, self.s_low, self.v_low],
            dtype=np.uint8,
        )
        self.upper_hsv = np.array(
            [self.h_high, self.s_high, self.v_high],
            dtype=np.uint8,
        )

    def _support_config(self) -> BallSupportConfig:
        return BallSupportConfig(
            support_diameter_m=self.support_diameter_m,
            support_v_max=self.support_v_max,
            black_ratio_min=self.support_black_ratio_min,
            edge_black_ratio_min=self.edge_support_black_ratio_min,
            surrounding_ball_ratio_max=self.support_ball_color_ratio_max,
            floor_ratio_max=self.support_floor_ratio_max,
            sector_black_ratio_min=self.support_sector_black_ratio_min,
            min_sectors=self.support_min_sectors,
            edge_min_sectors=self.edge_support_min_sectors,
            min_visible_fraction=self.support_min_visible_fraction,
            edge_min_visible_fraction=self.edge_support_min_visible_fraction,
        )

    def parameter_callback(self, params) -> SetParametersResult:
        values = {
            "h_low": self.h_low,
            "h_high": self.h_high,
            "s_low": self.s_low,
            "s_high": self.s_high,
            "v_low": self.v_low,
            "v_high": self.v_high,
        }
        floor_values = {
            "floor_h_low": self.floor_h_low,
            "floor_h_high": self.floor_h_high,
            "floor_s_low": self.floor_s_low,
            "floor_s_high": self.floor_s_high,
            "floor_v_low": self.floor_v_low,
            "floor_v_high": self.floor_v_high,
        }

        for param in params:
            if param.name in values:
                try:
                    values[param.name] = int(param.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f"{param.name} must be an integer",
                    )
            elif param.name in floor_values:
                try:
                    floor_values[param.name] = int(param.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason=f"{param.name} must be an integer",
                    )

        if not (
            0 <= values["h_low"] <= values["h_high"] <= 179
            and 0 <= values["s_low"] <= values["s_high"] <= 255
            and 0 <= values["v_low"] <= values["v_high"] <= 255
        ):
            return SetParametersResult(
                successful=False,
                reason="Invalid HSV range",
            )
        if not (
            0 <= floor_values["floor_h_low"] <= 179
            and 0 <= floor_values["floor_h_high"] <= 179
            and 0
            <= floor_values["floor_s_low"]
            <= floor_values["floor_s_high"]
            <= 255
            and 0
            <= floor_values["floor_v_low"]
            <= floor_values["floor_v_high"]
            <= 255
        ):
            return SetParametersResult(
                successful=False,
                reason="Invalid floor HSV range",
            )

        self.h_low = values["h_low"]
        self.h_high = values["h_high"]
        self.s_low = values["s_low"]
        self.s_high = values["s_high"]
        self.v_low = values["v_low"]
        self.v_high = values["v_high"]
        for name, value in floor_values.items():
            setattr(self, name, value)
        self._update_hsv_arrays()

        for param in params:
            if param.name == "depth_threshold_m":
                self.depth_threshold_m = float(param.value)
            elif param.name == "min_contour_area":
                self.min_contour_area = float(param.value)
            elif param.name in {
                "max_circle_ratio_error", "edge_max_circle_ratio_error",
                "edge_min_contour_area_ratio",
                "ball_diameter_min_m", "ball_diameter_max_m",
                "radius_size_min_ratio", "radius_size_max_ratio",
                "edge_radius_size_min_ratio", "edge_radius_size_max_ratio",
                "min_circularity", "edge_min_circularity",
                "min_aspect_ratio", "max_aspect_ratio",
                "edge_min_aspect_ratio", "edge_max_aspect_ratio",
                "support_diameter_m", "support_black_ratio_min",
                "edge_support_black_ratio_min",
                "support_ball_color_ratio_max", "support_floor_ratio_max",
                "support_sector_black_ratio_min",
                "support_min_visible_fraction",
                "edge_support_min_visible_fraction",
                "depth_inner_radius_ratio", "depth_min_valid_ratio",
                "support_score_weight", "tracking_roi_radius_scale",
                "tracking_roi_expansion_per_miss",
                "tracking_roi_min_half_size_px",
            }:
                setattr(self, param.name, float(param.value))
            elif param.name in {
                "support_v_max", "support_min_sectors",
                "edge_support_min_sectors", "realsense_hold_frames",
                "depth_min_valid_pixels",
            }:
                setattr(self, param.name, int(param.value))
            elif param.name == "tracking_roi_enabled":
                self.tracking_roi_enabled = bool(param.value)
            elif param.name in {
                "confirmation_window_frames", "confirmation_required_hits",
            }:
                if param.name == "confirmation_window_frames":
                    new_window = max(1, int(param.value))
                    old_items = list(self.raw_detection_history)
                    self.confirmation_window_frames = new_window
                    self.raw_detection_history = deque(
                        old_items[-new_window:],
                        maxlen=new_window,
                    )
                else:
                    self.confirmation_required_hits = max(
                        1,
                        min(
                            self.confirmation_window_frames,
                            int(param.value),
                        ),
                    )
            elif param.name in {
                "confirm_center_distance_px",
                "confirm_center_radius_scale",
                "confirm_depth_absolute_m",
                "confirm_depth_relative",
                "confirm_radius_ratio_min",
                "confirm_radius_ratio_max",
            }:
                match_values = {
                    "center_distance_px":
                        self.ball_match_config.center_distance_px,
                    "center_radius_scale":
                        self.ball_match_config.center_radius_scale,
                    "depth_absolute_m":
                        self.ball_match_config.depth_absolute_m,
                    "depth_relative":
                        self.ball_match_config.depth_relative,
                    "radius_ratio_min":
                        self.ball_match_config.radius_ratio_min,
                    "radius_ratio_max":
                        self.ball_match_config.radius_ratio_max,
                }
                config_name = {
                    "confirm_center_distance_px": "center_distance_px",
                    "confirm_center_radius_scale": "center_radius_scale",
                    "confirm_depth_absolute_m": "depth_absolute_m",
                    "confirm_depth_relative": "depth_relative",
                    "confirm_radius_ratio_min": "radius_ratio_min",
                    "confirm_radius_ratio_max": "radius_ratio_max",
                }[param.name]
                match_values[config_name] = float(param.value)
                self.ball_match_config = BallMatchConfig(**match_values)
            elif param.name == "debug_publish_every_n_frames":
                self.debug_publish_every_n_frames = max(
                    1,
                    int(param.value),
                )

        return SetParametersResult(successful=True)

    # =============================================================
    # 프로세스 수명 동안 유지하는 공 영상 구독
    # =============================================================
    def _start_ball_image_subscriptions(self) -> None:
        if self.rs_color_sub is not None or self.rs_depth_sub is not None:
            return

        self.rs_color_sub = Subscriber(
            self,
            Image,
            self.realsense_color_topic,
            qos_profile=self.image_qos,
        )
        self.rs_depth_sub = Subscriber(
            self,
            Image,
            self.realsense_depth_topic,
            qos_profile=self.image_qos,
        )
        self.rs_sync = ApproximateTimeSynchronizer(
            [self.rs_color_sub, self.rs_depth_sub],
            queue_size=self.realsense_sync_queue,
            slop=self.realsense_sync_slop_sec,
        )
        self.rs_sync.registerCallback(self.cb_realsense_images)

    def _clear_ball_detection_state(self) -> None:
        self.latest_realsense = None
        self.latest_realsense_time = 0.0
        self.last_realsense_detection = None
        self.realsense_lost_frames = self.realsense_hold_frames
        self.realsense_track_confirmed = False
        self.realsense_track_velocity_px = (0.0, 0.0)
        self.raw_detection_history.clear()
        self.latest_webcam = None
        self.latest_webcam_time = 0.0

    def cb_ball_active(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested == self.ball_detection_active:
            return

        # 구독과 synchronizer는 그대로 두고 처리 플래그만 바꾼다.
        # 이미 큐에 들어온 콜백도 초입의 active 검사에서 즉시 반환한다.
        self.ball_detection_active = requested
        self._clear_ball_detection_state()
        self.ball_status_publisher.set_detection_enabled(requested)

        self.get_logger().info(
            "Ball image processing switched "
            f"{'ON' if requested else 'OFF'}"
        )

    def _set_vision_mode_from_ball_in_hand(
        self,
        ball_in_hand: bool,
        *,
        force: bool = False,
    ) -> bool:
        """
        Vision 내부에서 공과 hoop 처리 모드를 전환한다.

        RealSense 카메라와 두 동기화 구독은 계속 유지한다. 확정된 공 소유
        상태에 맞는 OpenCV 콜백만 활성화하며, 두 검출기가 전환 순간 함께
        실행되지 않도록 항상 OFF를 먼저 발행하고 ON을 나중에 발행한다.
        """
        if not getattr(self, "manage_activity_from_ball_in_hand", True):
            return False

        hoop_active = bool(ball_in_hand)
        previous = getattr(self, "managed_hoop_active", None)
        if previous == hoop_active and not force:
            return False

        ball_active = not hoop_active
        ball_pub = getattr(self, "pub_ball_active", None)
        hoop_pub = getattr(self, "pub_hoop_active", None)

        if hoop_active:
            # 공 검출 OFF → hoop 검출 ON
            self.cb_ball_active(Bool(data=False))
            if ball_pub is not None:
                ball_pub.publish(Bool(data=False))
            if hoop_pub is not None:
                hoop_pub.publish(Bool(data=True))
        else:
            # hoop 검출 OFF → 공 검출 ON
            if hoop_pub is not None:
                hoop_pub.publish(Bool(data=False))
            self.cb_ball_active(Bool(data=True))
            if ball_pub is not None:
                ball_pub.publish(Bool(data=True))

        self.managed_hoop_active = hoop_active
        self.get_logger().info(
            "[VisionMode] "
            f"ball={'ON' if ball_active else 'OFF'}, "
            f"hoop={'ON' if hoop_active else 'OFF'} "
            "(latched ball_in_hand)"
        )
        return True

    def cb_realsense_yolo_state(self, msg: String) -> None:
        """Feed the dedicated RealSense YOLO result into existing decisions."""
        if not getattr(self, "ball_detection_active", True):
            return

        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(
                "Failed to parse /realsense_yolo/ball_state JSON"
            )
            return
        if not isinstance(payload, dict):
            return

        state = self._empty_realsense_state()
        state.update(payload)
        detected = bool(state.get("realsense_ball_detected", False))
        if detected:
            try:
                distance_cm = float(state["realsense_ball_distance_cm"])
                angle_deg = float(state["realsense_ball_angle_error"])
            except (KeyError, TypeError, ValueError):
                detected = False
            else:
                detected = bool(
                    math.isfinite(distance_cm)
                    and distance_cm > 0.0
                    and math.isfinite(angle_deg)
                )

        if not detected:
            diagnostic = state.get("realsense_diagnostic")
            state = self._empty_realsense_state()
            if isinstance(diagnostic, dict):
                state["realsense_diagnostic"] = diagnostic

        self.latest_realsense = state
        self.latest_realsense_time = time.monotonic()

    # =============================================================
    # 카메라 내부 파라미터
    # =============================================================
    def cb_camera_info(self, msg: CameraInfo) -> None:
        if len(msg.k) < 9:
            return

        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])

        if fx <= 0.0 or fy <= 0.0:
            return

        self.fx = fx
        self.fy = fy
        self.cx_intr = cx
        self.cy_intr = cy

        if not self.camera_info_received:
            self.camera_info_received = True
            self.get_logger().info(
                "RealSense CameraInfo received: "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx_intr:.2f}, cy={self.cy_intr:.2f}"
            )

    # =============================================================
    # RealSense OpenCV 검출
    # =============================================================
    def cb_realsense_images(
        self,
        color_msg: Image,
        depth_msg: Image,
    ) -> None:
        if not getattr(self, "ball_detection_active", True):
            return

        now = time.monotonic()

        try:
            frame = self.bridge.imgmsg_to_cv2(
                color_msg,
                desired_encoding="bgr8",
            )
            depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except Exception as exc:
            self.get_logger().warn(
                f"RealSense image conversion failed: {exc}"
            )
            return

        if frame is None or depth_raw is None:
            return

        depth = np.asarray(depth_raw, dtype=np.float32)
        if depth.ndim != 2:
            self.get_logger().warn("Depth image must be single-channel")
            return

        frame_h, frame_w = frame.shape[:2]
        if depth.shape[0] != frame_h or depth.shape[1] != frame_w:
            self.get_logger().warn(
                "Color/depth size mismatch. "
                "Use aligned_depth_to_color topic."
            )
            return

        full_roi = self._configured_realsense_roi(frame_w, frame_h)
        search_roi = full_roi
        used_tracking_roi = False

        # TRACK에서는 직전 위치와 속도로 예측한 작은 ROI를 먼저 처리한다.
        # ROI가 실패하거나 다른 주황색 물체를 잡으면 아래에서 *같은 콜백,
        # 같은 프레임*의 전체 설정 ROI를 즉시 처리한다. Timer/sleep/정지 모션은
        # 전혀 없으므로 ROI 확대 때문에 로봇 대기시간이 길어지지 않는다.
        if (
            self.tracking_roi_enabled
            and self.realsense_track_confirmed
            and self.last_realsense_detection is not None
        ):
            predicted_roi = expanded_tracking_roi(
                frame_width=frame_w,
                frame_height=frame_h,
                detection=self.last_realsense_detection,
                velocity_px=self.realsense_track_velocity_px,
                missed_frames=self.realsense_lost_frames,
                radius_scale=self.tracking_roi_radius_scale,
                expansion_per_miss=(
                    self.tracking_roi_expansion_per_miss
                ),
                minimum_half_size_px=(
                    self.tracking_roi_min_half_size_px
                ),
            )
            search_roi = self._intersect_rois(predicted_roi, full_roi)
            used_tracking_roi = search_roi != full_roi

        (
            detection,
            rejection_diagnostic,
            ball_color_mask,
            mask,
        ) = self._detect_ball_in_roi(frame, depth, search_roi)

        roi_candidate_matches_track = bool(
            detection is not None
            and self.last_realsense_detection is not None
            and same_ball_candidate(
                self.last_realsense_detection,
                detection,
                self.ball_match_config,
            )
        )
        if used_tracking_roi and (
            detection is None or not roi_candidate_matches_track
        ):
            # ROI 밖으로 크게 튄 실제 공을 놓치지 않도록 같은 프레임에서 즉시
            # 전체 화면 fallback을 한다. 이 동작은 계산량만 조금 늘리고 검출
            # 결과를 기다리는 프레임 수나 정지시간은 늘리지 않는다.
            (
                full_detection,
                full_diagnostic,
                full_color_mask,
                full_mask,
            ) = self._detect_ball_in_roi(frame, depth, full_roi)
            search_roi = full_roi
            ball_color_mask = full_color_mask
            mask = full_mask
            if full_detection is not None:
                detection = full_detection
            rejection_diagnostic = full_diagnostic

        held_previous = self._update_realsense_tracking(
            detection=detection,
            rejection_diagnostic=rejection_diagnostic,
            now=now,
        )
        x_start, y_start, x_end, y_end = search_roi

        self.realsense_frame_count += 1
        publish_debug_now = (
            self.publish_realsense_debug_image
            and self.realsense_frame_count
            % self.debug_publish_every_n_frames
            == 0
        )
        if publish_debug_now or self.show_realsense_window:
            debug = self._draw_realsense_debug(
                frame=frame,
                raw_mask=ball_color_mask,
                mask=mask,
                detection=detection,
                held_previous=held_previous,
                roi=(x_start, y_start, x_end, y_end),
            )

            if publish_debug_now:
                debug_msg = self.bridge.cv2_to_imgmsg(
                    debug,
                    encoding="bgr8",
                )
                debug_msg.header = color_msg.header
                self.pub_realsense_debug.publish(debug_msg)

            if self.show_realsense_window:
                cv2.imshow("Ball RealSense OpenCV", debug)
                cv2.waitKey(1)

    def _configured_realsense_roi(
        self,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """설정 비율을 안전한 전체 SEARCH ROI 픽셀 좌표로 바꾼다."""
        x1 = int(frame_width * self.rs_roi_left_ratio)
        x2 = int(frame_width * self.rs_roi_right_ratio)
        y1 = int(frame_height * self.rs_roi_top_ratio)
        y2 = int(frame_height * self.rs_roi_bottom_ratio)
        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(x1 + 1, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(y1 + 1, min(y2, frame_height))
        return (x1, y1, x2, y2)

    @staticmethod
    def _intersect_rois(
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """TRACK ROI가 사용자가 설정한 SEARCH ROI를 벗어나지 않게 한다."""
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        if x2 <= x1 or y2 <= y1:
            return second
        return (x1, y1, x2, y2)

    def _detect_ball_in_roi(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Dict[str, Any],
        np.ndarray,
        np.ndarray,
    ]:
        """
        한 ROI에서 원본 공 후보 하나를 찾는다.

        형상용 ``mask``는 의도적으로 HSV만 사용한다. 이전 구현처럼 유효하지
        않은 depth 픽셀을 먼저 0으로 만들면 공 표면의 depth hole이 색상 윤곽을
        찢고, 그 결과 실제 공이 circularity/area에서 탈락한다. depth는 윤곽을
        찾은 뒤 각 후보의 안쪽 50% 영역에서 별도로 검증한다.
        """
        x1, y1, x2, y2 = roi
        roi_color = frame[y1:y2, x1:x2]
        roi_depth = depth[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        ball_color_mask = cv2.inRange(
            hsv,
            self.lower_hsv,
            self.upper_hsv,
        )
        floor_mask = hsv_range_mask(
            hsv,
            (
                self.floor_h_low,
                self.floor_s_low,
                self.floor_v_low,
            ),
            (
                self.floor_h_high,
                self.floor_s_high,
                self.floor_v_high,
            ),
        )

        # 이 depth-filtered 마스크는 진단 수치에만 사용한다. 아래 morphology의
        # 입력은 ball_color_mask이므로 depth 결손이 공 형상을 훼손하지 않는다.
        roi_depth_m = roi_depth * self.depth_scale
        valid_depth = (
            np.isfinite(roi_depth_m)
            & (roi_depth_m > 0.0)
            & (roi_depth_m <= self.depth_threshold_m)
        )
        depth_filtered_mask = np.zeros_like(ball_color_mask)
        depth_filtered_mask[
            (ball_color_mask > 0) & valid_depth
        ] = 255

        mask = cv2.morphologyEx(
            ball_color_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
        )

        rejection_counts: Dict[str, int] = {}
        detection = self._find_best_ball(
            mask=mask,
            hsv=hsv,
            ball_color_mask=ball_color_mask,
            floor_mask=floor_mask,
            depth=depth,
            roi_x_start=x1,
            roi_y_start=y1,
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
            rejection_counts=rejection_counts,
        )
        diagnostic = self._classify_realsense_rejection(
            ball_color_mask=ball_color_mask,
            roi_depth_m=roi_depth_m,
            depth_filtered_mask=depth_filtered_mask,
            rejection_counts=rejection_counts,
        )
        return detection, diagnostic, ball_color_mask, mask

    def _confirmation_hits(self, candidate: Dict[str, Any]) -> int:
        """최근 원본 창에서 현재 후보와 같은 공으로 보이는 개수를 센다."""
        return sum(
            1
            for previous in self.raw_detection_history
            if previous is not None
            and same_ball_candidate(
                previous,
                candidate,
                self.ball_match_config,
            )
        )

    def _accept_confirmed_detection(
        self,
        detection: Dict[str, Any],
        *,
        detail: str,
        now: float,
    ) -> None:
        """원본 검출만 TRACK 상태와 행동 입력으로 승격한다."""
        previous = self.last_realsense_detection
        if previous is not None:
            dx = float(detection["raw_ball_x"]) - float(
                previous["raw_ball_x"]
            )
            dy = float(detection["raw_ball_y"]) - float(
                previous["raw_ball_y"]
            )
            # 한 프레임의 튐을 그대로 예측에 쓰지 않도록 간단한 EMA를 적용한다.
            old_vx, old_vy = self.realsense_track_velocity_px
            self.realsense_track_velocity_px = (
                0.5 * old_vx + 0.5 * dx,
                0.5 * old_vy + 0.5 * dy,
            )
        else:
            self.realsense_track_velocity_px = (0.0, 0.0)

        accepted = dict(detection)
        accepted["held_previous_detection"] = False
        accepted["realsense_diagnostic"] = {
            "category": "accepted",
            "detail": detail,
        }
        self.realsense_track_confirmed = True
        self.realsense_lost_frames = 0
        self.last_realsense_detection = dict(accepted)
        self.latest_realsense = accepted
        self.latest_realsense_time = now

    def _update_realsense_tracking(
        self,
        *,
        detection: Optional[Dict[str, Any]],
        rejection_diagnostic: Dict[str, Any],
        now: float,
    ) -> bool:
        """
        SEARCH의 2/3 확정, TRACK 갱신, 표시 전용 HELD를 관리한다.

        반환값은 디버그 화면에 HOLD를 그릴지 여부이다. HELD 상태는 직전 좌표를
        화면과 다음 ROI 예측에만 보존하며 ``realsense_ball_detected=False``로
        강제한다. 따라서 복사 프레임이 신규 투표나 실제 행동 근거가 될 수 없다.
        """
        if detection is not None and self.realsense_track_confirmed:
            if (
                self.last_realsense_detection is not None
                and same_ball_candidate(
                    self.last_realsense_detection,
                    detection,
                    self.ball_match_config,
                )
            ):
                self.raw_detection_history.clear()
                self.raw_detection_history.append(dict(detection))
                self._accept_confirmed_detection(
                    detection,
                    detail="tracked_raw_detection",
                    now=now,
                )
                return False
            # TRACK ROI/전체 fallback이 다른 주황색 후보를 잡은 경우이다. 기존
            # 공을 즉시 바꿔치기하지 않고 한 번의 미검출처럼 처리한다.
            rejection_diagnostic = {
                "category": "pending",
                "detail": "raw_candidate_did_not_match_confirmed_track",
            }
            detection = None

        if detection is not None:
            # SEARCH 또는 완전 분실 후 재검색. deque에는 HELD가 아니라 실제
            # 카메라 콜백에서 새로 계산된 원본 후보/None만 들어간다.
            self.raw_detection_history.append(dict(detection))
            hits = self._confirmation_hits(detection)
            if hits >= self.confirmation_required_hits:
                self._accept_confirmed_detection(
                    detection,
                    detail=(
                        f"confirmed_{hits}_of_"
                        f"{self.confirmation_window_frames}_raw"
                    ),
                    now=now,
                )
            else:
                pending = self._empty_realsense_state()
                pending["realsense_diagnostic"] = {
                    "category": "pending",
                    "detail": (
                        f"raw_confirmation_{hits}_of_"
                        f"{self.confirmation_required_hits}"
                    ),
                }
                self.latest_realsense = pending
                self.latest_realsense_time = now
            return False

        if not self.realsense_track_confirmed:
            self.raw_detection_history.append(None)
            empty = self._empty_realsense_state()
            empty["realsense_diagnostic"] = rejection_diagnostic
            self.latest_realsense = empty
            self.latest_realsense_time = now
            return False

        self.realsense_lost_frames += 1
        if (
            self.last_realsense_detection is not None
            and self.realsense_lost_frames <= self.realsense_hold_frames
        ):
            held = dict(self.last_realsense_detection)
            # 가장 중요한 안전 조건: 표시/ROI용 복사본은 검출 true가 아니다.
            held["realsense_ball_detected"] = False
            held["held_previous_detection"] = True
            held["realsense_diagnostic"] = {
                "category": "held",
                "detail": (
                    f"display_only_after_{rejection_diagnostic['category']}:"
                    f"{rejection_diagnostic['detail']}"
                ),
            }
            self.latest_realsense = held
            self.latest_realsense_time = now
            return True

        # 설정 횟수보다 오래 원본을 못 찾았을 때만 TRACK을 해제한다. 다음
        # 프레임은 전체 SEARCH ROI에서 시작하며 다시 2/3 원본 확인을 거친다.
        self.realsense_track_confirmed = False
        self.realsense_track_velocity_px = (0.0, 0.0)
        self.last_realsense_detection = None
        self.raw_detection_history.clear()
        empty = self._empty_realsense_state()
        empty["realsense_diagnostic"] = rejection_diagnostic
        self.latest_realsense = empty
        self.latest_realsense_time = now
        return False

    def _classify_realsense_rejection(
        self,
        *,
        ball_color_mask: np.ndarray,
        roi_depth_m: np.ndarray,
        depth_filtered_mask: np.ndarray,
        rejection_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Return the furthest detector stage reached by this frame."""
        color_region = ball_color_mask > 0
        hsv_pixels = int(np.count_nonzero(color_region))
        depth_filtered_pixels = int(
            np.count_nonzero(depth_filtered_mask)
        )
        color_depth = roi_depth_m[color_region]
        finite_positive = color_depth[
            np.isfinite(color_depth) & (color_depth > 0.0)
        ]
        valid_depth_pixels = int(
            np.count_nonzero(
                finite_positive <= self.depth_threshold_m
            )
        )
        over_distance_pixels = int(
            np.count_nonzero(
                finite_positive > self.depth_threshold_m
            )
        )

        metrics = {
            "hsv_pixels": hsv_pixels,
            "valid_depth_pixels": valid_depth_pixels,
            "over_distance_pixels": over_distance_pixels,
            "depth_filtered_pixels": depth_filtered_pixels,
            "rejection_counts": dict(sorted(rejection_counts.items())),
        }

        def diagnostic(category: str, detail: str) -> Dict[str, Any]:
            return {
                "category": category,
                "detail": detail,
                **metrics,
            }

        if hsv_pixels == 0:
            return diagnostic("hsv", "no_pixels_in_hsv_range")
        if finite_positive.size == 0:
            return diagnostic("depth", "no_finite_positive_depth")
        if valid_depth_pixels == 0 and over_distance_pixels > 0:
            return diagnostic(
                "distance",
                f"all_hsv_pixels_over_{self.depth_threshold_m:.2f}m",
            )
        if depth_filtered_pixels == 0:
            return diagnostic("depth", "no_depth_valid_hsv_pixels")

        def most_common(prefix: str) -> Optional[str]:
            matches = [
                (reason, count)
                for reason, count in rejection_counts.items()
                if reason.startswith(prefix)
            ]
            if not matches:
                return None
            return max(matches, key=lambda item: item[1])[0]

        support_reason = most_common("support:")
        if support_reason is not None:
            return diagnostic("support", support_reason.split(":", 1)[1])
        if rejection_counts.get("depth_inner_region_invalid", 0) > 0:
            return diagnostic("depth", "candidate_inner_depth_invalid")
        if rejection_counts.get("physical_size", 0) > 0:
            return diagnostic("shape", "depth_size_mismatch")

        shape_reasons = (
            "circle_fill",
            "circularity",
            "aspect_ratio",
            "invalid_radius",
            "invalid_perimeter",
            "contour_area",
            "no_contour_after_morphology",
        )
        selected_shape = max(
            shape_reasons,
            key=lambda reason: rejection_counts.get(reason, 0),
        )
        if rejection_counts.get(selected_shape, 0) > 0:
            return diagnostic("shape", selected_shape)
        return diagnostic("shape", "no_accepted_candidate")

    def _find_best_ball(
        self,
        mask: np.ndarray,
        hsv: np.ndarray,
        ball_color_mask: np.ndarray,
        floor_mask: np.ndarray,
        depth: np.ndarray,
        roi_x_start: int,
        roi_y_start: int,
        frame_w: int,
        frame_h: int,
        rejection_counts: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """검은 받침대 위에 있는 물리적으로 타당한 주황색 공을 찾는다."""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best: Optional[Dict[str, Any]] = None

        def rejected(reason: str) -> None:
            if rejection_counts is None:
                return
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        if not contours:
            rejected("no_contour_after_morphology")

        support_config = self._support_config()
        support_black_mask = black_support_mask(
            hsv,
            support_config.support_v_max,
        )

        for contour in contours:
            area = float(cv2.contourArea(contour))
            bx, by, bw, bh = cv2.boundingRect(contour)
            mask_h, mask_w = mask.shape[:2]
            margin = self.edge_ball_margin_px
            touches_edge = bool(
                bx <= margin
                or by <= margin
                or bx + bw >= mask_w - margin
                or by + bh >= mask_h - margin
            )

            min_area = self.min_contour_area
            if touches_edge:
                min_area *= self.edge_min_contour_area_ratio
            if area < min_area:
                rejected("contour_area")
                continue

            aspect_ratio = float(bw) / max(float(bh), 1.0)
            min_aspect = (
                self.edge_min_aspect_ratio if touches_edge
                else self.min_aspect_ratio
            )
            max_aspect = (
                self.edge_max_aspect_ratio if touches_edge
                else self.max_aspect_ratio
            )
            if not (min_aspect <= aspect_ratio <= max_aspect):
                rejected("aspect_ratio")
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                rejected("invalid_perimeter")
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            required_circularity = (
                self.edge_min_circularity if touches_edge
                else self.min_circularity
            )
            if circularity < required_circularity:
                rejected("circularity")
                continue

            (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(contour)
            if radius <= 1e-6:
                rejected("invalid_radius")
                continue

            circle_area = math.pi * radius * radius
            ratio_error = abs((area / circle_area) - 1.0)

            max_ratio_error = (
                self.edge_max_circle_ratio_error
                if touches_edge
                else self.max_circle_ratio_error
            )
            if ratio_error > max_ratio_error:
                rejected("circle_fill")
                continue

            cx_img = int(round(cx_roi + roi_x_start))
            cy_img = int(round(cy_roi + roi_y_start))
            depth_sample = self._read_depth_m(
                depth=depth,
                cx=cx_img,
                cy=cy_img,
                detected_radius_px=float(radius),
            )
            z_m = depth_sample.depth_m
            if z_m is None or self.fx <= 0.0:
                rejected("depth_inner_region_invalid")
                continue

            expected_radius_min = (
                self.fx * (self.ball_diameter_min_m * 0.5) / z_m
            )
            expected_radius_max = (
                self.fx * (self.ball_diameter_max_m * 0.5) / z_m
            )
            size_min_ratio = (
                self.edge_radius_size_min_ratio if touches_edge
                else self.radius_size_min_ratio
            )
            size_max_ratio = (
                self.edge_radius_size_max_ratio if touches_edge
                else self.radius_size_max_ratio
            )
            allowed_radius_min = expected_radius_min * size_min_ratio
            allowed_radius_max = expected_radius_max * size_max_ratio
            if not (allowed_radius_min <= radius <= allowed_radius_max):
                rejected("physical_size")
                continue

            expected_radius_mid = 0.5 * (
                expected_radius_min + expected_radius_max
            )
            radius_ratio = radius / max(expected_radius_mid, 1e-6)
            size_error = abs(radius_ratio - 1.0)
            support = evaluate_ball_support(
                hsv=hsv,
                ball_color_mask=ball_color_mask,
                floor_mask=floor_mask,
                center=(float(cx_roi), float(cy_roi)),
                detected_radius_px=float(radius),
                expected_ball_radius_px=float(expected_radius_mid),
                depth_m=float(z_m),
                focal_x_px=float(self.fx),
                touches_edge=touches_edge,
                config=support_config,
                black_mask=support_black_mask,
            )
            support_confidence = ball_support_confidence(support)

            score = (
                ratio_error
                + (1.0 - min(circularity, 1.0))
                + size_error
                + self.support_score_weight * (1.0 - support_confidence)
            )

            candidate = {
                "score": float(score),
                "cx_roi": float(cx_roi),
                "cy_roi": float(cy_roi),
                "radius": float(radius),
                "area": area,
                "z_m": float(z_m),
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "radius_ratio": float(radius_ratio),
                "touches_edge": touches_edge,
                "support": support,
                "support_confidence": float(support_confidence),
                "depth_sample": depth_sample,
            }
            if best is None or score < float(best["score"]):
                best = candidate

        if best is None:
            return None

        cx_roi = float(best["cx_roi"])
        cy_roi = float(best["cy_roi"])
        radius = float(best["radius"])
        area = float(best["area"])
        z_m = float(best["z_m"])
        circularity = float(best["circularity"])
        aspect_ratio = float(best["aspect_ratio"])
        radius_ratio = float(best["radius_ratio"])
        support = best["support"]
        support_confidence = float(best["support_confidence"])
        depth_sample: BallDepthSample = best["depth_sample"]
        circle_area = math.pi * radius * radius
        ratio_error = abs((area / circle_area) - 1.0)
        cx_img = int(round(cx_roi + roi_x_start))
        cy_img = int(round(cy_roi + roi_y_start))

        # pinhole camera model을 이용한 3차원 좌표
        x_m = (cx_img - self.cx_intr) * z_m / self.fx
        y_m = (cy_img - self.cy_intr) * z_m / self.fy

        if self.realsense_use_euclidean_distance:
            distance_m = math.sqrt(
                x_m * x_m + y_m * y_m + z_m * z_m
            )
        else:
            distance_m = z_m

        angle_error_deg = math.degrees(math.atan2(x_m, z_m))

        return {
            "realsense_ball_detected": True,
            "realsense_ball_distance_cm": float(distance_m * 100.0),
            "realsense_ball_angle_error": float(angle_error_deg),
            "raw_x_m": float(x_m),
            "raw_y_m": float(y_m),
            "raw_z_m": float(z_m),
            "raw_ball_x": float(cx_img),
            "raw_ball_y": float(cy_img),
            "raw_radius": float(radius),
            "raw_contour_area": float(area),
            "raw_circle_ratio_error": float(ratio_error),
            "raw_circularity": float(circularity),
            "raw_aspect_ratio": float(aspect_ratio),
            "raw_radius_size_ratio": float(radius_ratio),
            "raw_touches_edge": bool(best["touches_edge"]),
            "raw_support_black_ratio": float(support["black_ratio"]),
            "raw_support_ball_color_ratio": float(
                support["surrounding_ball_ratio"]
            ),
            "raw_support_floor_ratio": float(support["floor_ratio"]),
            "raw_support_sectors": int(support["qualified_sectors"]),
            "raw_support_visible_fraction": float(
                support["visible_fraction"]
            ),
            "raw_support_outer_radius": float(support["outer_radius_px"]),
            "raw_support_passed": bool(support["accepted"]),
            "raw_support_reason": str(support["reason"]),
            "raw_support_confidence": support_confidence,
            "raw_depth_valid_pixels": int(depth_sample.valid_pixels),
            "raw_depth_sample_pixels": int(depth_sample.sample_pixels),
            "raw_depth_valid_ratio": float(depth_sample.valid_ratio),
            "raw_depth_mad_m": (
                float(depth_sample.median_absolute_deviation_m)
                if depth_sample.median_absolute_deviation_m is not None
                else None
            ),
            "held_previous_detection": False,
        }

    def _read_depth_m(
        self,
        depth: np.ndarray,
        cx: int,
        cy: int,
        detected_radius_px: float,
    ) -> BallDepthSample:
        """
        공 안쪽 영역의 깊이와 유효 픽셀 통계를 반환한다.

        ``depth_inner_radius_ratio`` 기본 0.50은 쉽게 현장 조정할 수 있도록
        파라미터로 노출되어 있다. 상세한 조정 기준과 수학적 영역 정의는
        ``ball_detector_core.sample_ball_inner_depth`` 주석을 참고한다.
        """
        return sample_ball_inner_depth(
            depth=depth,
            center=(float(cx), float(cy)),
            detected_radius_px=float(detected_radius_px),
            depth_scale=self.depth_scale,
            depth_max_m=self.depth_threshold_m,
            inner_radius_ratio=self.depth_inner_radius_ratio,
            min_valid_pixels=self.depth_min_valid_pixels,
            min_valid_ratio=self.depth_min_valid_ratio,
        )

    def _draw_realsense_debug(
        self,
        frame: np.ndarray,
        raw_mask: np.ndarray,
        mask: np.ndarray,
        detection: Optional[Dict[str, Any]],
        held_previous: bool,
        roi: Tuple[int, int, int, int],
    ) -> np.ndarray:
        debug = frame.copy()
        x_start, y_start, x_end, y_end = roi

        color = (0, 255, 0) if detection is not None else (0, 0, 255)
        if held_previous:
            color = (0, 255, 255)

        cv2.rectangle(
            debug,
            (x_start, y_start),
            (x_end, y_end),
            color,
            2,
        )

        # HOLD 중에는 ROI fallback에서 우연히 찾은 다른 주황색 후보가 아니라
        # 마지막으로 확정된 공 위치를 노란색으로 표시한다.
        draw_state = (
            self.last_realsense_detection
            if held_previous
            else detection
        )

        if draw_state is not None:
            cx = int(round(draw_state["raw_ball_x"]))
            cy = int(round(draw_state["raw_ball_y"]))
            radius = int(round(draw_state["raw_radius"]))
            support_radius = int(
                round(draw_state["raw_support_outer_radius"])
            )

            cv2.circle(debug, (cx, cy), radius, color, 2)
            cv2.circle(
                debug,
                (cx, cy),
                max(1, support_radius),
                (255, 180, 0),
                1,
            )
            cv2.circle(debug, (cx, cy), 4, (0, 0, 255), -1)

            state_text = "HOLD" if held_previous else "DETECTED"
            text = (
                f"{state_text} "
                f"{draw_state['realsense_ball_distance_cm']:.1f}cm "
                f"{draw_state['realsense_ball_angle_error']:+.1f}deg"
            )
            support_text = (
                f"support score={draw_state['raw_support_confidence']:.2f} "
                f"pass={int(draw_state['raw_support_passed'])} "
                f"black={draw_state['raw_support_black_ratio']:.2f} "
                f"floor={draw_state['raw_support_floor_ratio']:.2f} "
                f"sectors={draw_state['raw_support_sectors']} "
                f"depth_ok={draw_state['raw_depth_valid_ratio']:.2f}"
            )
        else:
            text = "BALL MISS"
            support_text = "support is an auxiliary candidate score"

        cv2.putText(
            debug,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        hsv_text = (
            f"HSV [{self.h_low},{self.s_low},{self.v_low}]"
            f"-[{self.h_high},{self.s_high},{self.v_high}]"
        )
        cv2.putText(
            debug,
            hsv_text,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            support_text,
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        # 마스크를 작은 미리보기로 합성
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview_w = min(
            self.debug_mask_preview_width,
            debug.shape[1] // 3,
        )
        if preview_w > 0 and mask_bgr.shape[1] > 0:
            scale = preview_w / mask_bgr.shape[1]
            preview_h = max(1, int(mask_bgr.shape[0] * scale))
            preview = cv2.resize(
                mask_bgr,
                (preview_w, preview_h),
                interpolation=cv2.INTER_NEAREST,
            )
            y0 = 5
            x0 = max(0, debug.shape[1] - preview_w - 5)
            y1 = min(debug.shape[0], y0 + preview_h)
            x1 = min(debug.shape[1], x0 + preview_w)
            debug[y0:y1, x0:x1] = preview[: y1 - y0, : x1 - x0]

        return debug

    def _empty_realsense_state(self) -> Dict[str, Any]:
        return {
            "realsense_ball_detected": False,
            "realsense_ball_distance_cm": None,
            "realsense_ball_angle_error": None,
            "raw_x_m": None,
            "raw_y_m": None,
            "raw_z_m": None,
            "raw_ball_x": None,
            "raw_ball_y": None,
            "raw_radius": None,
            "raw_contour_area": None,
            "raw_circle_ratio_error": None,
            "raw_circularity": None,
            "raw_aspect_ratio": None,
            "raw_radius_size_ratio": None,
            "raw_touches_edge": False,
            "raw_support_black_ratio": None,
            "raw_support_ball_color_ratio": None,
            "raw_support_floor_ratio": None,
            "raw_support_sectors": None,
            "raw_support_visible_fraction": None,
            "raw_support_outer_radius": None,
            "raw_support_passed": False,
            "raw_support_reason": None,
            "raw_support_confidence": None,
            "raw_depth_valid_pixels": 0,
            "raw_depth_sample_pixels": 0,
            "raw_depth_valid_ratio": 0.0,
            "raw_depth_mad_m": None,
            "held_previous_detection": False,
            "realsense_diagnostic": {
                "category": "waiting",
                "detail": "no_realsense_frame",
            },
        }

    # =============================================================
    # 웹캠 YOLO
    # =============================================================
    def cb_webcam_state(self, msg: String) -> None:
        if not getattr(self, "ball_detection_active", True):
            return

        now = time.monotonic()

        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(
                "Failed to parse /line_tracker/state JSON"
            )
            return

        ball_detected = bool(payload.get("ball_detected", False))

        if not ball_detected:
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        try:
            ball_x = float(payload.get("ball_x", -1.0))
            ball_y = float(payload.get("ball_y", -1.0))
            ball_conf = float(payload.get("ball_conf", 0.0))
        except (TypeError, ValueError):
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        ball_bbox = payload.get("ball_bbox", [])

        if (
            not math.isfinite(ball_x)
            or not math.isfinite(ball_y)
            or ball_x < 0.0
            or ball_y < 0.0
        ):
            self.latest_webcam = self._empty_webcam_state()
            self.latest_webcam_time = now
            return

        def finite_payload_float(key: str) -> Optional[float]:
            try:
                value = float(payload.get(key))
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        robot_x = finite_payload_float("robot_center_x")
        robot_y = finite_payload_float("robot_center_y")
        x_offset = finite_payload_float("ball_x_offset_px")
        x_distance = finite_payload_float("ball_x_distance_px")
        y_distance = finite_payload_float("ball_y_distance_px")
        distance_px = finite_payload_float("ball_distance_px")
        payload_angle_deg = finite_payload_float("ball_angle_deg")

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
            # 구 버전 payload가 절댓값을 보내더라도 로봇 중심선 기준 부호를 복원한다.
            x_distance = math.copysign(abs(x_distance), x_offset)
        if y_distance is None:
            y_distance = abs((
                robot_y
                if robot_y is not None
                else self.webcam_robot_center_y
            ) - ball_y)
        if distance_px is None:
            distance_px = math.hypot(x_distance, y_distance)

        angle_error_deg: Optional[float]
        if payload_angle_deg is not None:
            angle_error_deg = payload_angle_deg
        elif (
            0.0 < self.webcam_fov_x_deg < 180.0
            and self.webcam_frame_width > 0.0
        ):
            focal_px = self.webcam_frame_width / (
                2.0
                * math.tan(
                    math.radians(self.webcam_fov_x_deg) / 2.0
                )
            )
            angle_error_deg = float(
                math.degrees(math.atan2(x_offset, focal_px))
            )
        else:
            angle_error_deg = None

        self.latest_webcam = {
            "webcam_ball_detected": True,
            "webcam_ball_x_distance": float(x_distance),
            "webcam_ball_y_distance": float(y_distance),
            "webcam_ball_angle_error": angle_error_deg,
            "webcam_ball_distance_px": float(distance_px),
            "raw_ball_x": ball_x,
            "raw_ball_y": ball_y,
            "raw_ball_conf": ball_conf,
            "raw_ball_bbox": ball_bbox,
        }
        self.latest_webcam_time = now

    def _empty_webcam_state(self) -> Dict[str, Any]:
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
        """후프 JSON에서 BallResult로 전달할 거리와 각도를 보관한다."""
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn("Failed to parse /hoop/vision_state JSON")
            return

        if (
            not isinstance(payload, dict)
            or not bool(payload.get("detected", False))
        ):
            self.latest_hoop = self._empty_hoop_state()
            self.latest_hoop_time = now
            return

        try:
            distance_cm = float(payload.get("realsense_goal_distance_cm"))
            angle_deg = float(payload.get("realsense_goal_angle"))
        except (TypeError, ValueError):
            self.latest_hoop = self._empty_hoop_state()
            self.latest_hoop_time = now
            return

        if (
            not math.isfinite(distance_cm)
            or distance_cm <= 0.0
            or not math.isfinite(angle_deg)
        ):
            self.latest_hoop = self._empty_hoop_state()
            self.latest_hoop_time = now
            return

        self.latest_hoop = {
            "hoop_detected": True,
            "realsense_goal_distance_cm": distance_cm,
            "realsense_goal_angle": angle_deg,
        }
        self.latest_hoop_time = now

    @staticmethod
    def _empty_hoop_state() -> Dict[str, Any]:
        return {
            "hoop_detected": False,
            "realsense_goal_distance_cm": None,
            "realsense_goal_angle": None,
        }

    # =============================================================
    # raw_ball_in_hand
    # =============================================================
    def cb_raw_ball_in_hand(self, msg: Bool) -> None:
        self.ball_in_hand = bool(msg.data)

    # =============================================================
    # BallFeatures 생성 및 알고리즘 전달
    # =============================================================
    def _published_realsense_diagnostic(
        self,
        *,
        realsense_valid: bool,
        realsense_age: Optional[float],
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add the 120 cm mission gate to the detector-stage diagnosis."""
        if not self.ball_detection_active:
            return {
                "category": "inactive",
                "detail": "ball_processing_disabled",
            }

        if realsense_valid:
            distance_cm = features.get("realsense_ball_distance_cm")
            entry_cm = float(
                self.ball_status_publisher.ball_decision.ball_entry_distance_cm
            )
            if (
                not features.get("ball_in_hand", False)
                and distance_cm is not None
                and float(distance_cm) > entry_cm
            ):
                return {
                    "category": "distance",
                    "detail": (
                        f"ball_entry_gate_{float(distance_cm):.1f}cm"
                        f">{entry_cm:.1f}cm"
                    ),
                    "distance_cm": float(distance_cm),
                    "limit_cm": entry_cm,
                }

            if self.latest_realsense is not None:
                return dict(
                    self.latest_realsense.get(
                        "realsense_diagnostic",
                        {
                            "category": "accepted",
                            "detail": "detector_passed",
                        },
                    )
                )

        if (
            realsense_age is not None
            and realsense_age > self.realsense_timeout_sec
        ):
            return {
                "category": "timeout",
                "detail": f"last_frame_age_{realsense_age:.2f}s",
            }

        if self.latest_realsense is not None:
            return dict(
                self.latest_realsense.get(
                    "realsense_diagnostic",
                    {
                        "category": "waiting",
                        "detail": "no_diagnostic",
                    },
                )
            )
        return {
            "category": "waiting",
            "detail": "no_realsense_frame",
        }

    @staticmethod
    def _format_realsense_diagnostic(diagnostic: Dict[str, Any]) -> str:
        category = str(diagnostic.get("category", "unknown"))
        detail = str(diagnostic.get("detail", "unknown"))
        labels = {
            "accepted": "ACCEPTED",
            "held": "HELD",
            "hsv": "REJECT_HSV",
            "depth": "REJECT_DEPTH",
            "distance": "REJECT_DISTANCE",
            "shape": "REJECT_SHAPE",
            "support": "REJECT_SUPPORT",
            "timeout": "REJECT_TIMEOUT",
            "pending": "PENDING_RAW_CONFIRMATION",
            "inactive": "INACTIVE",
            "waiting": "WAITING",
        }
        return f"{labels.get(category, category.upper())}:{detail}"

    def _log_realsense_diagnostic_transition(
        self,
        diagnostic: Dict[str, Any],
    ) -> bool:
        """Log each detector rejection transition once without frame spam."""
        label = BallVisionFusionNode._format_realsense_diagnostic(
            diagnostic
        )
        if label == getattr(
            self,
            "last_realsense_diagnostic_label",
            None,
        ):
            return False

        self.last_realsense_diagnostic_label = label
        metric_parts = []
        metric_names = (
            "hsv_pixels",
            "valid_depth_pixels",
            "over_distance_pixels",
            "depth_filtered_pixels",
            "distance_cm",
            "limit_cm",
        )
        for name in metric_names:
            if name in diagnostic:
                metric_parts.append(f"{name}={diagnostic[name]}")
        rejection_counts = diagnostic.get("rejection_counts")
        if rejection_counts:
            metric_parts.append(f"filters={rejection_counts}")

        suffix = f" ({', '.join(metric_parts)})" if metric_parts else ""
        self.get_logger().info(
            f"[RealSenseBallDiagnostic] {label}{suffix}"
        )
        return True

    def publish_ball_features(self) -> None:
        now = time.monotonic()

        realsense_age = (
            now - self.latest_realsense_time
            if self.latest_realsense is not None
            else None
        )
        webcam_age = (
            now - self.latest_webcam_time
            if self.latest_webcam is not None
            else None
        )
        hoop_age = (
            now - self.latest_hoop_time
            if self.latest_hoop is not None
            else None
        )

        realsense_valid = bool(
            self.ball_detection_active
            and self.latest_realsense is not None
            and realsense_age is not None
            and realsense_age <= self.realsense_timeout_sec
            and self.latest_realsense.get(
                "realsense_ball_detected",
                False,
            )
        )

        webcam_valid = bool(
            self.ball_detection_active
            and self.latest_webcam is not None
            and webcam_age is not None
            and webcam_age <= self.webcam_timeout_sec
            and self.latest_webcam.get(
                "webcam_ball_detected",
                False,
            )
        )
        hoop_valid = bool(
            self.latest_hoop is not None
            and hoop_age is not None
            and hoop_age <= self.hoop_timeout_sec
            and self.latest_hoop.get("hoop_detected", False)
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

        if realsense_valid and self.latest_realsense is not None:
            features.update(
                {
                    "realsense_ball_detected": True,
                    "realsense_ball_distance_cm":
                        self.latest_realsense[
                            "realsense_ball_distance_cm"
                        ],
                    "realsense_ball_angle_error":
                        self.latest_realsense[
                            "realsense_ball_angle_error"
                        ],
                }
            )

        if webcam_valid and self.latest_webcam is not None:
            features.update(
                {
                    "webcam_ball_detected": True,
                    "webcam_ball_x_distance":
                        self.latest_webcam[
                            "webcam_ball_x_distance"
                        ],
                    "webcam_ball_y_distance":
                        self.latest_webcam[
                            "webcam_ball_y_distance"
                        ],
                    "webcam_ball_angle_error":
                        self.latest_webcam[
                            "webcam_ball_angle_error"
                        ],
                    "webcam_ball_distance_px":
                        self.latest_webcam[
                            "webcam_ball_distance_px"
                        ],
                }
            )

        if hoop_valid and self.latest_hoop is not None:
            features.update(
                {
                    "realsense_goal_distance_cm":
                        self.latest_hoop[
                            "realsense_goal_distance_cm"
                        ],
                    "realsense_goal_angle":
                        self.latest_hoop[
                            "realsense_goal_angle"
                        ],
                }
            )

        realsense_diagnostic = self._published_realsense_diagnostic(
            realsense_valid=realsense_valid,
            realsense_age=realsense_age,
            features=features,
        )
        self._log_realsense_diagnostic_transition(realsense_diagnostic)

        status, angle = (
            self.ball_status_publisher.publish_ball_status(
                **features
            )
        )
        self._set_vision_mode_from_ball_in_hand(
            self.ball_status_publisher.ball_in_hand
        )

        if webcam_valid:
            source_priority = "webcam"
        elif realsense_valid:
            source_priority = "realsense"
        else:
            source_priority = "none"

        output: Dict[str, Any] = dict(features)
        output.update(
            {
                "source_priority": source_priority,
                "ball_detection_active": self.ball_detection_active,
                "realsense_detection_method": "opencv_hsv_depth",
                "realsense_age_sec": realsense_age,
                "webcam_age_sec": webcam_age,
                "hoop_age_sec": hoop_age,
                "hoop_detected": hoop_valid,
                "ball_status": int(status),
                "ball_status_angle": float(angle),
                "camera_info_received": self.camera_info_received,
                "realsense_diagnostic": realsense_diagnostic,
            }
        )

        if realsense_valid and self.latest_realsense is not None:
            output["realsense_raw"] = {
                "x_m": self.latest_realsense["raw_x_m"],
                "y_m": self.latest_realsense["raw_y_m"],
                "z_m": self.latest_realsense["raw_z_m"],
                "ball_x": self.latest_realsense["raw_ball_x"],
                "ball_y": self.latest_realsense["raw_ball_y"],
                "radius": self.latest_realsense["raw_radius"],
                "contour_area":
                    self.latest_realsense["raw_contour_area"],
                "circle_ratio_error":
                    self.latest_realsense[
                        "raw_circle_ratio_error"
                    ],
                "circularity":
                    self.latest_realsense["raw_circularity"],
                "aspect_ratio":
                    self.latest_realsense["raw_aspect_ratio"],
                "radius_size_ratio":
                    self.latest_realsense["raw_radius_size_ratio"],
                "touches_edge":
                    self.latest_realsense["raw_touches_edge"],
                "support_black_ratio":
                    self.latest_realsense[
                        "raw_support_black_ratio"
                    ],
                "support_ball_color_ratio":
                    self.latest_realsense[
                        "raw_support_ball_color_ratio"
                    ],
                "support_floor_ratio":
                    self.latest_realsense[
                        "raw_support_floor_ratio"
                    ],
                "support_sectors":
                    self.latest_realsense["raw_support_sectors"],
                "support_visible_fraction":
                    self.latest_realsense[
                        "raw_support_visible_fraction"
                    ],
                "support_passed":
                    self.latest_realsense["raw_support_passed"],
                "support_reason":
                    self.latest_realsense["raw_support_reason"],
                "support_confidence":
                    self.latest_realsense["raw_support_confidence"],
                "depth_valid_pixels":
                    self.latest_realsense["raw_depth_valid_pixels"],
                "depth_sample_pixels":
                    self.latest_realsense["raw_depth_sample_pixels"],
                "depth_valid_ratio":
                    self.latest_realsense["raw_depth_valid_ratio"],
                "depth_mad_m":
                    self.latest_realsense["raw_depth_mad_m"],
                "held_previous_detection":
                    self.latest_realsense[
                        "held_previous_detection"
                    ],
            }

        if webcam_valid and self.latest_webcam is not None:
            output["webcam_raw"] = {
                "ball_x": self.latest_webcam["raw_ball_x"],
                "ball_y": self.latest_webcam["raw_ball_y"],
                "ball_conf":
                    self.latest_webcam["raw_ball_conf"],
                "ball_bbox":
                    self.latest_webcam["raw_ball_bbox"],
            }

        self.pub_vision_state.publish(
            String(data=json.dumps(output, ensure_ascii=False))
        )

        self.frame_count += 1
        if self.frame_count % self.print_every_n_frames == 0:
            self.get_logger().info(
                "ball_vision "
                f"src={source_priority} "
                f"rs={features['realsense_ball_detected']} "
                f"rs_dist={features['realsense_ball_distance_cm']} "
                f"rs_ang={features['realsense_ball_angle_error']} "
                "rs_diag="
                f"{self._format_realsense_diagnostic(realsense_diagnostic)} "
                f"webcam={features['webcam_ball_detected']} "
                f"webcam_x={features['webcam_ball_x_distance']} "
                f"webcam_y={features['webcam_ball_y_distance']} "
                f"webcam_dist={features['webcam_ball_distance_px']} "
                f"hand={features['ball_in_hand']} "
                f"status={status} angle={angle:.2f}"
            )

    def destroy_node(self):
        if self.show_realsense_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


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
