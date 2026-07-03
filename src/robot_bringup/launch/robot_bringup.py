import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 실행 여부를 터미널에서 켜고 끌 수 있는 옵션
    # 예: ros2 launch robot_bringup robot_bringup.py use_webcam:=false
    use_realsense = LaunchConfiguration('use_realsense')
    use_webcam = LaunchConfiguration('use_webcam')
    use_motion = LaunchConfiguration('use_motion')
    use_vision = LaunchConfiguration('use_vision')
    use_decision = LaunchConfiguration('use_decision')

    declare_use_realsense = DeclareLaunchArgument(
        'use_realsense',
        default_value='true',
        description='Start Intel RealSense camera launch',
    )
    declare_use_webcam = DeclareLaunchArgument(
        'use_webcam',
        default_value='true',
        description='Start USB webcam node',
    )
    declare_use_motion = DeclareLaunchArgument(
        'use_motion',
        default_value='true',
        description='Start Dynamixel motion node',
    )
    declare_use_vision = DeclareLaunchArgument(
        'use_vision',
        default_value='true',
        description='Start vision package nodes',
    )
    declare_use_decision = DeclareLaunchArgument(
        'use_decision',
        default_value='true',
        description='Start decision package node',
    )

    # 1. Dynamixel motion 노드
    motion_node = Node(
        condition=IfCondition(use_motion),
        package='motion',             # TODO: 실제 motion 패키지명
        executable='main_motion',      # TODO: 실제 Dynamixel/motion 실행파일명
        name='main_motion',            #노드 이름
        output='screen',               #로그를 터미널에 출력
        emulate_tty=True,
        remappings=[
            ('motion_command', '/motion_command'),
            ('motion_end', '/motion_end'),
            ('motion_prepare', '/motion_prepare'),
        ],
        arguments=['--ros-args', '--log-level', 'info'],
    )

    # 2. RealSense 카메라
    # TODO: 사용하는 RealSense launch 파일 이름에 맞게 rs_launch.py를 수정하세요.
    # 예: rs_launch.py, dual_realsense_launch.py 등
    realsense_launch = TimerAction(
        period=2.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory('realsense2_camera'),
                        'launch',
                        'rs_launch.py',  # TODO: 실제 RealSense launch 파일명
                    )
                ),
                condition=IfCondition(use_realsense),
            )
        ],
    )

    # 3. Webcam 카메라
    # TODO: usb_cam을 쓸 경우 보통 package='usb_cam', executable='usb_cam_node_exe' 입니다.
    # 다른 webcam 패키지를 쓰면 package/executable/parameters를 실제 이름으로 수정하세요.
    webcam_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                condition=IfCondition(use_webcam),
                package='usb_cam',              # TODO: 실제 webcam 패키지명
                executable='usb_cam_node_exe',   # TODO: 실제 webcam 실행파일명
                name='webcam',
                output='screen',
                emulate_tty=True,
                parameters=[
                    {
                        'video_device': '/dev/video0',  # TODO: webcam 장치 번호 확인 후 수정
                        'image_width': 640,
                        'image_height': 480,
                        'framerate': 30.0,
                        'pixel_format': 'yuyv',
                    }
                ],
                remappings=[
                    ('image_raw', '/webcam/image_raw'),
                    ('camera_info', '/webcam/camera_info'),
                ],
                arguments=['--ros-args', '--log-level', 'info'],
            )
        ],
    )

    # 4. Vision 패키지 노드들
    # TODO: vision 패키지 안에서 만든 실제 노드 이름으로 package/executable을 수정하세요.
    vision_nodes = TimerAction(
        period=4.0,
        actions=[
            Node(
                condition=IfCondition(use_vision),
                package='vision',            # TODO: 실제 vision 패키지명
                executable='line_vision',     # TODO: line result 발행 노드 실행파일명
                name='line_vision',
                output='screen',
                emulate_tty=True,
                remappings=[
                    ('image', '/camera/camera/color/image_raw'),  # TODO: RealSense/Webcam 입력 토픽에 맞게 수정
                    ('line_result', '/line_result'),
                ],
                arguments=['--ros-args', '--log-level', 'info'],
            ),
            Node(
                condition=IfCondition(use_vision),
                package='vision',            # TODO: 실제 vision 패키지명
                executable='ball_vision',     # TODO: ball result 발행 노드 실행파일명
                name='ball_vision',
                output='screen',
                emulate_tty=True,
                remappings=[
                    ('image', '/webcam/image_raw'),  # TODO: 공 인식에 쓰는 카메라 토픽으로 수정
                    ('ball_result', '/ball_result'),
                ],
                arguments=['--ros-args', '--log-level', 'info'],
            ),
            Node(
                condition=IfCondition(use_vision),
                package='vision',             # TODO: 실제 vision 패키지명
                executable='hurdle_vision',    # TODO: hurdle result 발행 노드 실행파일명
                name='hurdle_vision',
                output='screen',
                emulate_tty=True,
                remappings=[
                    ('image', '/camera/camera/color/image_raw'),  # TODO: 장애물 인식 입력 토픽으로 수정
                    ('hurdle_result', '/hurdle_result'),
                ],
                arguments=['--ros-args', '--log-level', 'info'],
            ),
        ],
    )

    # 5. Decision 패키지 노드
    # TODO: src/decision/setup.py 기준 실행파일 이름은 main_decision 입니다.
    decision_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                condition=IfCondition(use_decision),
                package='decision',            # TODO: 실제 decision 패키지명
                executable='main_decision',     # TODO: 실제 decision 실행파일명
                name='main_decision',
                output='screen',
                emulate_tty=True,
                remappings=[
                    ('line_result', '/line_result'),
                    ('ball_result', '/ball_result'),
                    ('hurdle_result', '/hurdle_result'),
                    ('motion_end', '/motion_end'),
                    ('motion_prepare', '/motion_prepare'),
                    ('motion_command', '/motion_command'),
                ],
                arguments=['--ros-args', '--log-level', 'info'],
            )
        ],
    )

    return LaunchDescription([
        declare_use_realsense,
        declare_use_webcam,
        declare_use_motion,
        declare_use_vision,
        declare_use_decision,
        motion_node,
        realsense_launch,
        webcam_node,
        vision_nodes,
        decision_node,
    ])
