#!/usr/bin/env python3
"""IRC 전체 Vision 실행: webcam YOLO + RealSense + ball/hurdle fusion.

배치 위치:
  ~/irc/src/robot_bringup/launch/vision_stack.launch.py

기본 Vision 스크립트 위치:
  ~/irc/src/vision/scripts
"""

import glob
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _make_webcam_node(context):
    device = LaunchConfiguration("webcam_device").perform(context)
    if device == "auto":
        c920_devices = sorted(
            glob.glob("/dev/v4l/by-id/*C920*video-index0")
        )
        if not c920_devices:
            raise RuntimeError(
                "C920 webcam not found under /dev/v4l/by-id/. "
                "Connect the webcam or set webcam_device explicitly."
            )
        device = c920_devices[0]
    width = int(LaunchConfiguration("webcam_width").perform(context))
    height = int(LaunchConfiguration("webcam_height").perform(context))
    fps = int(LaunchConfiguration("webcam_fps").perform(context))

    return [
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="webcam",
            output="screen",
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration("start_webcam")),
            parameters=[
                {
                    "video_device": device,
                    "image_size": [width, height],
                    "time_per_frame": [1, fps],
                    # C920 영상은 픽셀 기반 검출에만 사용합니다. 유효한
                    # 미보정 CameraInfo를 지정해 드라이버의 기본 ~/.ros
                    # 캘리브레이션 파일 조회와 불필요한 오류를 막습니다.
                    "camera_info_url": (
                        "package://robot_bringup/config/"
                        "webcam_uncalibrated.yaml"
                    ),
                }
            ],
            remappings=[
                ("image_raw", "/camera/image_raw"),
                ("camera_info", "/camera/camera_info"),
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    scripts_dir = LaunchConfiguration("scripts_dir")
    settings_ini = LaunchConfiguration("settings_ini")
    hurdle_params = LaunchConfiguration("hurdle_params")

    start_realsense = LaunchConfiguration("start_realsense")
    start_webcam = LaunchConfiguration("start_webcam")
    start_yolo = LaunchConfiguration("start_yolo")
    start_ball = LaunchConfiguration("start_ball")
    start_hurdle = LaunchConfiguration("start_hurdle")
    start_monitor = LaunchConfiguration("start_monitor")
    start_selector = LaunchConfiguration("start_selector")

    yolo_script = PathJoinSubstitution([scripts_dir, "yolo_detector.py"])
    ball_script = PathJoinSubstitution([scripts_dir, "ball_vision_fusion.py"])
    hurdle_script = PathJoinSubstitution([scripts_dir, "hurdle_vision_fusion.py"])
    monitor_script = PathJoinSubstitution([scripts_dir, "vision_status_monitor.py"])
    selector_script = PathJoinSubstitution([scripts_dir, "realsense_debug_selector.py"])

    declarations = [
        DeclareLaunchArgument(
            "scripts_dir",
            default_value=PathJoinSubstitution(
                [EnvironmentVariable("HOME"), "irc", "src", "vision", "scripts"]
            ),
            description="vision Python scripts/settings/model directory",
        ),
        DeclareLaunchArgument(
  	    "settings_ini",
   	    default_value=PathJoinSubstitution(
       	        [
           	    EnvironmentVariable("HOME"),
           	    "irc",
                    "src",
                    "vision",
                    "config",
                    "settings.ini",
                ]
  	    ),
  	),
        DeclareLaunchArgument(
            "hurdle_params",
            default_value=PathJoinSubstitution([scripts_dir, "hurdle_vision_params.yaml"]),
        ),
        DeclareLaunchArgument("start_realsense", default_value="true"),
        DeclareLaunchArgument("start_webcam", default_value="true"),
        DeclareLaunchArgument("start_yolo", default_value="true"),
        DeclareLaunchArgument("start_ball", default_value="true"),
        DeclareLaunchArgument("start_hurdle", default_value="true"),
        DeclareLaunchArgument("start_monitor", default_value="true"),
        DeclareLaunchArgument("start_selector", default_value="true"),
        DeclareLaunchArgument(
            "webcam_device",
            default_value="auto",
            description=(
                "Webcam device path. 'auto' selects the C920 video-index0 "
                "device from /dev/v4l/by-id/."
            ),
        ),
        DeclareLaunchArgument("webcam_width", default_value="640"),
        DeclareLaunchArgument("webcam_height", default_value="480"),
        DeclareLaunchArgument("webcam_fps", default_value="30"),
    ]

    # 한 RealSense 드라이버가 color/depth를 한 번만 발행하고,
    # ball_vision_fusion과 hurdle_vision_fusion이 같은 토픽을 구독한다.
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="",
        name="camera",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(start_realsense),
        parameters=[
            {
                "camera_name": "camera",
                "enable_color": True,
                "enable_depth": True,
                "enable_infra": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_gyro": False,
                "enable_accel": False,
                "rgb_camera.color_profile": "640,480,15",
                "depth_module.depth_profile": "640,480,15",
                "enable_sync": True,
                "align_depth.enable": True,
                "pointcloud.enable": False,
                "initial_reset": False,
            }
        ],
    )

    webcam_node = OpaqueFunction(function=_make_webcam_node)

    yolo_process = ExecuteProcess(
        name="yolo_vision_process",
        cmd=[sys.executable, yolo_script, settings_ini, "--ros2"],
        cwd=scripts_dir,
        output="screen",
        emulate_tty=True,
        respawn=True,
        respawn_delay=2.0,
        condition=IfCondition(start_yolo),
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    ball_process = ExecuteProcess(
        name="ball_vision_fusion_process",
        cmd=[sys.executable, ball_script],
        cwd=scripts_dir,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(start_ball),
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    hurdle_process = ExecuteProcess(
        name="hurdle_vision_fusion_process",
        cmd=[
            sys.executable,
            hurdle_script,
            "--ros-args",
            "--params-file",
            hurdle_params,
        ],
        cwd=scripts_dir,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(start_hurdle),
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    monitor_process = ExecuteProcess(
        name="vision_status_monitor_process",
        cmd=[sys.executable, monitor_script],
        cwd=scripts_dir,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(start_monitor),
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    selector_process = ExecuteProcess(
        name="realsense_debug_selector_process",
        cmd=[sys.executable, selector_script],
        cwd=scripts_dir,
        output="screen",
        emulate_tty=True,
        condition=IfCondition(start_selector),
        additional_env={"PYTHONUNBUFFERED": "1"},
    )

    # YOLO는 모델/런타임 초기화에 시간이 걸리므로 카메라와 동시에 먼저
    # 시작한다. 구독 생성 전 모델을 로드하므로 카메라 토픽이 아직 없어도
    # 안전하며, 나머지 vision 프로세스는 기존처럼 2초 뒤 시작한다.
    delayed_vision = TimerAction(
        period=2.0,
        actions=[
            ball_process,
            hurdle_process,
            monitor_process,
            selector_process,
        ],
    )

    return LaunchDescription(
        declarations + [realsense_node, webcam_node, yolo_process, delayed_vision]
    )
