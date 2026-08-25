import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.node import Node
from msgs.msg import BallResult, MotionCommand


class BallStatus:
    Forward_4step = 1
    Forward_3step = 20
    Left_Half_Forward = 2
    Right_Half_Forward = 3
    Left_Forward = 4
    Right_Forward = 5
    Left_Turn = 6
    Right_Turn = 7
    Forward_half = 8
    Backward_half = 9
    Left_Move = 10
    Right_Move = 11
    Pick_Ready = 12
    Shoot = 13
    Neck_Up = 14
    Neck_Down = 18
    Left_Half_Forward_3step = 21
    Right_Half_Forward_3step = 22
    Left_Turn_10 = 23
    Right_Turn_10 = 24
    Back_To_Initial = 27
    Left_Turn_5 = 28
    Right_Turn_5 = 29
    Left_Turn_Afterpick = 30
    Right_Turn_Afterpick = 31
    Shoot_Close = 32
    Ball_Lost = 45
    Ball_None = 99


@dataclass
class BallFeatures:
    realsense_ball_detected: bool = False
    realsense_ball_distance_cm: Optional[float] = None
    realsense_ball_angle_error: Optional[float] = None

    realsense_goal_distance_cm: Optional[float] = None
    realsense_goal_angle: Optional[float] = None

    webcam_ball_detected: bool = False
    # 로봇 중심선 기준 거리: 왼쪽 음수, 오른쪽 양수.
    webcam_ball_x_distance: Optional[float] = None
    webcam_ball_y_distance: Optional[float] = None
    webcam_ball_angle_error: Optional[float] = None
    webcam_ball_distance_px: Optional[float] = None

    ball_in_hand: bool = False


class BallDecision:
    def __init__(self):
        # 120cm 이하이면 공 모드
        self.ball_entry_distance_cm = 120.0

        # Realsense 직진, 회전 기준각 5도
        self.angle_center_tol = 5.0

        # 공을 잡은 뒤 Realsense 골대 기준
        self.goal_entry_distance_cm = 120.0
        self.goal_shoot_max_distance_cm = 70.0
        self.goal_normal_shoot_min_distance_cm = 60.0
        self.goal_too_close_distance_cm = 50.0

        self.goal_approach_center_tol = 5.0
        self.goal_approach_large_angle = 60.0
        self.goal_shoot_center_tol = 10.0
        self.goal_shoot_large_angle = 20.0

        # Webcam 접근 및 pick 기준
        self.webcam_angle_center_tol = 5.0
        self.webcam_pick_y_max_px = 90.0
        self.webcam_pick_x_min_px = -35.0
        self.webcam_pick_x_max_px = 30.0

    def decide(self, features: BallFeatures) -> Tuple[int, float]:
        # 공을 잡고 있고 골대가 120cm 이내이면 골대 기준으로 판단한다.
        if features.ball_in_hand:
            goal_distance = features.realsense_goal_distance_cm
            if (
                goal_distance is not None
                and goal_distance <= self.goal_entry_distance_cm
            ):
                return self._decide_from_goal(features)
            return BallStatus.Ball_None, 0.0

        if not self.Ball_mission_ready(features):
            return BallStatus.Ball_None, 0.0

        # webcam에서 공이 감지되면 webcam 기준으로 판단한다.
        if features.webcam_ball_detected:
            return self._decide_from_webcam(features)

        #realsense에서 공이 감지되면 realsense 기준으로 판단
        distance = features.realsense_ball_distance_cm
        if (
            features.realsense_ball_detected
            and distance is not None
            and distance <= self.ball_entry_distance_cm
        ):
            return self._decide_from_realsense(
                features.realsense_ball_angle_error
            )

        return BallStatus.Ball_None, 0.0

    #webcam에서 먼저 판단 후 없으면 realsense에서 판단
    def Ball_mission_ready(self, features: BallFeatures) -> bool:
        if features.webcam_ball_detected:
            return True

        if not features.realsense_ball_detected:
            return False

        distance = features.realsense_ball_distance_cm
        if distance is None:
            return False

        return distance <= self.ball_entry_distance_cm

    #Webcam 판단
    def _decide_from_webcam(self, features: BallFeatures) -> Tuple[int, float]:
        webcam_ball_x_distance = features.webcam_ball_x_distance
        webcam_ball_y_distance = features.webcam_ball_y_distance
        if (
            webcam_ball_x_distance is None
            or webcam_ball_y_distance is None
        ):
            return BallStatus.Ball_None, 0.0

        angle = self.webcam_angle(features.webcam_ball_angle_error)

        # y 거리를 먼저 판단한다. 입력 y 거리는 항상 양수로 들어온다.
        if webcam_ball_y_distance > self.webcam_pick_y_max_px:
            if angle < -10.0:
                return BallStatus.Left_Turn_10, angle
            if angle < -self.angle_center_tol:
                return BallStatus.Left_Turn_5, angle
            if angle <= self.angle_center_tol:
                return BallStatus.Forward_half, 0.0
            if angle <= 10.0:
                return BallStatus.Right_Turn_5, angle
            return BallStatus.Right_Turn_10, angle

        # pick 거리 안에서는 x 거리로 좌우 정렬 여부를 판단한다.
        if webcam_ball_x_distance < self.webcam_pick_x_min_px:
            return BallStatus.Left_Move, angle
        if webcam_ball_x_distance > self.webcam_pick_x_max_px:
            return BallStatus.Right_Move, angle
        return BallStatus.Pick_Ready, 0.0

    #realsense 기준으로 판단하는 각도
    def _decide_from_realsense(self, angle: Optional[float]) -> Tuple[int, float]:
        if angle is None:
            return BallStatus.Ball_None, 0.0

        if angle < -60.0:
            return BallStatus.Left_Turn, angle
        if angle < -self.angle_center_tol:
            return BallStatus.Left_Half_Forward, angle
        if angle <= self.angle_center_tol:
            return BallStatus.Forward_4step, 0.0
        if angle <= 60.0:
            return BallStatus.Right_Half_Forward, angle
        return BallStatus.Right_Turn, angle

    #골 넣을 때 거리에 따라 shoot 선택, 각도 판단
    def _decide_from_goal(self, features: BallFeatures) -> Tuple[int, float]:
        distance = features.realsense_goal_distance_cm
        angle = features.realsense_goal_angle

        if distance is None or angle is None:
            return BallStatus.Ball_None, 0.0

        if distance <= self.goal_too_close_distance_cm:
            return BallStatus.Backward_half, 0.0

        if distance <= self.goal_shoot_max_distance_cm:
            shoot_status = (
                BallStatus.Shoot
                if distance >= self.goal_normal_shoot_min_distance_cm
                else BallStatus.Shoot_Close
            )
            return self._shoot_status_from_goal_angle(angle, shoot_status)

        return self._goal_status_from_angle(angle)

    #골대에 접근하는 로직
    def _goal_status_from_angle(self, angle: float) -> Tuple[int, float]:
        if angle < -self.goal_approach_large_angle:
            return BallStatus.Left_Turn, angle
        if angle < -self.goal_approach_center_tol:
            return BallStatus.Left_Half_Forward, angle
        if angle <= self.goal_approach_center_tol:
            return BallStatus.Forward_4step, 0.0
        if angle <= self.goal_approach_large_angle:
            return BallStatus.Right_Half_Forward, angle
        return BallStatus.Right_Turn, angle

    #슛 직전 각도에 따라 좌우 회전 판단
    def _shoot_status_from_goal_angle(
        self,
        angle: float,
        shoot_status: int,
    ) -> Tuple[int, float]:
        if angle < -self.goal_shoot_large_angle:
            return BallStatus.Left_Turn_10, angle
        if angle < -self.goal_shoot_center_tol:
            return BallStatus.Left_Turn_5, angle
        if angle <= self.goal_shoot_center_tol:
            return shoot_status, 0.0
        if angle <= self.goal_shoot_large_angle:
            return BallStatus.Right_Turn_5, angle
        return BallStatus.Right_Turn_10, angle

    #webcam 각도 값이 없을 때 안전하게 처리, 값 있으면 그대로 반환
    def webcam_angle(self, angle: Optional[float]) -> float:
        if angle is None:
            return 0.0

        return angle

class BallStatusPublisher:
    def __init__(self, node: Node, topic_name: str = 'ball_result'):
        self.node = node
        self.ball_decision = BallDecision()
        # 웹캠 원본 검출 5개 중 3개 이상일 때 최초 공 검출을
        # 확정한다.
        self.webcam_detection_buffer = deque(maxlen=5)
        self.webcam_ball_confirmed = False
        self.detection_enabled = True
        # 27번이 실제 motion_command로 발행될 때까지 결과를 유지하고,
        # 실행 확인 후에는 Pick 결과 확인이 끝날 때까지 다시
        # 발행하지 않는다.
        self.back_to_initial_waiting = False
        self.back_to_initial_done = False
        self.pick_command_seen = False
        # Pick 이후 비전에서 true를 한 번 확인하면 Shoot 완료까지
        # 유지하는 공 소유 상태입니다.
        self.ball_in_hand = False
        # 슛 가능 거리 진입 시 기본자세는 한 번만 실행한다.
        self.shoot_initial_waiting = False
        self.shoot_initial_done = False
        self.shoot_command_seen = False
        self.ball_pub = self.node.create_publisher(BallResult, topic_name, 10)
        self.motion_command_sub = self.node.create_subscription(
            MotionCommand,
            'motion_command',
            self._motion_command_callback,
            10,
        )

    def _log_info(self, message: str) -> None:
        get_logger = getattr(self.node, 'get_logger', None)
        if callable(get_logger):
            get_logger().info(message)

    def _reset_webcam_detection_cycle(self) -> None:
        self.webcam_detection_buffer.clear()
        self.webcam_ball_confirmed = False
        self.back_to_initial_waiting = False
        self.back_to_initial_done = False

    def set_detection_enabled(self, enabled: bool) -> None:
        """Enable voting only for frames captured after the mode switch."""
        self.detection_enabled = bool(enabled)
        self._reset_webcam_detection_cycle()

    def _reset_shoot_cycle(self) -> None:
        self.shoot_initial_waiting = False
        self.shoot_initial_done = False
        self.shoot_command_seen = False

    def _motion_command_callback(self, msg: MotionCommand) -> None:
        command = int(msg.command)

        if (
            self.back_to_initial_waiting
            and command == BallStatus.Back_To_Initial
        ):
            self.back_to_initial_waiting = False
            self.back_to_initial_done = True
            self._log_info(
                "Ball Back_To_Initial execution confirmed; "
                "initial-pose trigger locked until Pick result check."
            )

        if (
            self.shoot_initial_waiting
            and command == BallStatus.Back_To_Initial
        ):
            self.shoot_initial_waiting = False
            self.shoot_initial_done = True
            self._log_info(
                "Shoot Back_To_Initial execution confirmed; "
                "shoot initial-pose trigger locked."
            )

        if command in (BallStatus.Shoot, BallStatus.Shoot_Close):
            self.shoot_command_seen = True

        if (
            self.shoot_command_seen
            and command == BallStatus.Neck_Down
        ):
            self.ball_in_hand = False
            self._reset_shoot_cycle()
            self._log_info(
                "Shoot completed; ball-in-hand and shoot locks released."
            )

        if command == BallStatus.Pick_Ready:
            self.pick_command_seen = True
            return

        # MainDecision은 Pick 모션이 끝난 뒤 CheckBall()을 실행한
        # 다음,
        # 성공이면 Neck_Up, 실패이면 Backward_half를 발행한다.
        # 두 경우 모두 이후 Backward_half를 거쳐 Afterpick 회전으로 진행한다.
        if self.pick_command_seen and command in (
            BallStatus.Neck_Up,
            BallStatus.Backward_half,
            BallStatus.Left_Turn_Afterpick,
            BallStatus.Right_Turn_Afterpick,
        ):
            self.pick_command_seen = False
            self._reset_webcam_detection_cycle()
            self._log_info(
                "Pick result check completed; webcam ball detection "
                "lock released."
            )

    def publish_ball_status(
        self,
        realsense_ball_detected: bool = False,
        realsense_ball_distance_cm: Optional[float] = None,
        realsense_ball_angle_error: Optional[float] = None,
        realsense_goal_distance_cm: Optional[float] = None,
        realsense_goal_angle: Optional[float] = None,
        webcam_ball_detected: bool = False,
        webcam_ball_x_distance: Optional[float] = None,
        webcam_ball_y_distance: Optional[float] = None,
        webcam_ball_angle_error: Optional[float] = None,
        webcam_ball_distance_px: Optional[float] = None,
        ball_in_hand: bool = False,
    ) -> Tuple[int, float]:
        # 함수 인자는 비전의 원본 값이고, self.ball_in_hand는
        # Shoot 완료까지 유지되는 확정값이다.
        ball_in_hand = bool(ball_in_hand)
        if self.pick_command_seen and ball_in_hand:
            if not self.ball_in_hand:
                self._log_info(
                    "Ball in hand confirmed; possession is locked until "
                    "Shoot completes."
                )
            self.ball_in_hand = True

        if not self.ball_in_hand:
            self._reset_shoot_cycle()

        if (
            self.detection_enabled
            and not self.back_to_initial_done
            and not self.webcam_ball_confirmed
        ):
            # 손에 든 공은 다음 공의 최초 웹캠 검출로 집계하지
            # 않는다.
            detected_for_vote = bool(
                webcam_ball_detected and not self.ball_in_hand
            )
            self.webcam_detection_buffer.append(detected_for_vote)
            detected_count = sum(self.webcam_detection_buffer)
            if (
                len(self.webcam_detection_buffer) == 5
                and detected_count >= 3
            ):
                self.webcam_ball_confirmed = True
                self._log_info(
                    "Webcam ball confirmed by 5-frame majority: "
                    f"true={detected_count}, false={5 - detected_count}."
                )

        # 최초 웹캠 검출이 확정되기 전에는 webcam 기반 접근 명령을
        # 막는다. 확정 후에는 27번 모션의 실제 발행을 확인할 때까지
        # 27을 유지한다.
        webcam_enabled = bool(
            self.detection_enabled
            and webcam_ball_detected
            and self.back_to_initial_done
        )
        features = BallFeatures(
            realsense_ball_detected=realsense_ball_detected,
            realsense_ball_distance_cm=realsense_ball_distance_cm,
            realsense_ball_angle_error=realsense_ball_angle_error,
            realsense_goal_distance_cm=realsense_goal_distance_cm,
            realsense_goal_angle=realsense_goal_angle,
            webcam_ball_detected=webcam_enabled,
            webcam_ball_x_distance=webcam_ball_x_distance,
            webcam_ball_y_distance=webcam_ball_y_distance,
            webcam_ball_angle_error=webcam_ball_angle_error,
            webcam_ball_distance_px=webcam_ball_distance_px,
            ball_in_hand=self.ball_in_hand,
        )

        status, angle = self.ball_decision.decide(features)

        if self.webcam_ball_confirmed and not self.back_to_initial_done:
            self.back_to_initial_waiting = True
            status = BallStatus.Back_To_Initial
            angle = 0.0

        goal_in_shoot_zone = bool(
            self.ball_in_hand
            and realsense_goal_distance_cm is not None
            and math.isfinite(realsense_goal_distance_cm)
            and self.ball_decision.goal_too_close_distance_cm
            < realsense_goal_distance_cm
            <= self.ball_decision.goal_shoot_max_distance_cm
        )
        if self.shoot_command_seen:
            status = BallStatus.Ball_None
            angle = 0.0
        elif self.shoot_initial_waiting or (
            goal_in_shoot_zone and not self.shoot_initial_done
        ):
            self.shoot_initial_waiting = True
            status = BallStatus.Back_To_Initial
            angle = 0.0

        msg = BallResult()
        msg.status = int(status)
        msg.angle = float(angle)
        if hasattr(msg, 'ball_in_hand'):
            msg.ball_in_hand = self.ball_in_hand
        if self.ball_in_hand and realsense_goal_angle is not None:
            measured_angle = realsense_goal_angle
        else:
            measured_angle = (
                webcam_ball_angle_error
                if webcam_ball_angle_error is not None
                else realsense_ball_angle_error
            )
        if hasattr(msg, 'detected_angle'):
            msg.detected_angle = float(measured_angle or 0.0)
        if hasattr(msg, 'x_distance_px'):
            msg.x_distance_px = float(webcam_ball_x_distance or 0.0)
        if hasattr(msg, 'y_distance_px'):
            msg.y_distance_px = float(webcam_ball_y_distance or 0.0)
        if hasattr(msg, 'goal_distance_cm'):
            msg.goal_distance_cm = float(
                realsense_goal_distance_cm or 0.0
            )

        self.ball_pub.publish(msg)

        return status, angle
