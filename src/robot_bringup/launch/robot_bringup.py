import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_realsense = LaunchConfiguration('use_realsense')
    use_webcam = LaunchConfiguration('use_webcam')
    use_motion = LaunchConfiguration('use_motion')
    use_vision = LaunchConfiguration('use_vision')
    use_decision = LaunchConfiguration('use_decision')
    decision_test_mode = LaunchConfiguration('decision_test_mode')
    webcam_device = LaunchConfiguration('webcam_device')

    declarations = [
        DeclareLaunchArgument(
            'use_realsense',
            default_value='true',
            description='Start Intel RealSense camera',
        ),
        DeclareLaunchArgument(
            'use_webcam',
            default_value='true',
            description='Start USB webcam',
        ),
        DeclareLaunchArgument(
            'use_motion',
            default_value='true',
            description='Start Dynamixel motion node',
        ),
        DeclareLaunchArgument(
            'use_vision',
            default_value='true',
            description='Start all vision processing nodes',
        ),
        DeclareLaunchArgument(
            'use_decision',
            default_value='true',
            description='Start decision package node',
        ),
        DeclareLaunchArgument(
            'decision_test_mode',
            default_value='false',
            description='Keep motion_end true for decision-only tests',
        ),
        DeclareLaunchArgument(
            'webcam_device',
            default_value='auto',
            description=(
                "Webcam device path. 'auto' selects the C920 by its stable "
                "device ID."
            ),
        ),
    ]

    motion_node = Node(
        condition=IfCondition(use_motion),
        package='motion',
        executable='main',
        name='main_motion',
        output='screen',
        emulate_tty=True,
        remappings=[
            ('motion_command', '/motion_command'),
            ('motion_end', '/motion_end'),
            ('motion_prepare', '/motion_prepare'),
        ],
        arguments=['--ros-args', '--log-level', 'info'],
    )

    # Camera and vision nodes are owned by vision_stack.launch.py.
    # Do not place this include in a scoped GroupAction: its TimerAction evaluates
    # these launch configurations after the include action has returned.
    vision_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('robot_bringup'),
                'launch',
                'vision_stack.launch.py',
            )
        ),
        launch_arguments={
            'start_realsense': use_realsense,
            'start_webcam': use_webcam,
            'start_yolo': use_vision,
            'start_ball': use_vision,
            'start_hurdle': use_vision,
            'start_monitor': use_vision,
            'start_selector': use_vision,
            'webcam_device': webcam_device,
        }.items(),
    )

    decision_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                condition=IfCondition(use_decision),
                package='decision',
                executable='main_decision',
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
                parameters=[{'test_mode': decision_test_mode}],
                arguments=['--ros-args', '--log-level', 'info'],
            )
        ],
    )

    return LaunchDescription(
        declarations + [motion_node, vision_stack, decision_node]
    )
