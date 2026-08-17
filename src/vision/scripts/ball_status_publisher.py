from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.node import Node
from msgs.msg import BallResult


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
    Left_Half_Forward_3step = 21
    Right_Half_Forward_3step = 22
    Left_Turn_10 = 23
    Right_Turn_10 = 24
    Left_Turn_5 = 28
    Right_Turn_5 = 29
    Ball_In_Hand = 50
    Ball_Lost = 45
    Ball_None = 99


@dataclass
class BallFeatures:
    realsense_ball_detected: bool = False
    realsense_ball_distance_cm: Optional[float] = None
    realsense_ball_angle_error: Optional[float] = None

    webcam_ball_detected: bool = False
    # x 오프셋과 x 거리는 모두 왼쪽 음수, 오른쪽 양수인 signed 값이다.
    webcam_ball_x_offset: Optional[float] = None
    webcam_ball_x_distance: Optional[float] = None
    webcam_ball_y_distance: Optional[float] = None
    webcam_ball_angle_error: Optional[float] = None
    webcam_ball_distance_px: Optional[float] = None

    ball_in_hand: bool = False


class BallDecision:
    def __init__(self):
        #100cm 이하이면 공 모드, 25cm 이하에서는 webcam에서 보이는 거리(임의)
        self.ball_entry_distance_cm = 100.0

        # Realsense 직진, 제자리회전 기준각 5도
        self.angle_center_tol = 5.0

        # Webcam 접근 및 pick 기준
        self.webcam_angle_center_tol = 4.0
        self.webcam_pick_y_max_px = 78.0
        self.webcam_pick_x_min_px = -40.0
        self.webcam_pick_x_max_px = 35.0

    def decide(self, features: BallFeatures) -> Tuple[int, float]:
        #공을 잡고 있으면 접근 명령을 보내지 않음
        if features.ball_in_hand:
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
            return self._status_from_angle(features.realsense_ball_angle_error)

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
        # x_offset은 좌우 방향을 포함한 signed 거리이다.
        webcam_ball_x_offset = features.webcam_ball_x_offset
        if webcam_ball_x_offset is None:
            # 구 호출부에서는 x_distance가 signed 오프셋이었다.
            webcam_ball_x_offset = features.webcam_ball_x_distance
        webcam_ball_y_distance = features.webcam_ball_y_distance
        if (
            webcam_ball_x_offset is None
            or webcam_ball_y_distance is None
        ):
            return BallStatus.Ball_None, 0.0

        angle = self.webcam_angle(features.webcam_ball_angle_error)

        # y 거리를 먼저 판단한다. 입력 y 거리는 항상 양수로 들어온다.
        if webcam_ball_y_distance > self.webcam_pick_y_max_px:
            if angle < -self.webcam_angle_center_tol:
                return BallStatus.Left_Turn_5, angle
            if angle > self.webcam_angle_center_tol:
                return BallStatus.Right_Turn_5, angle
            return BallStatus.Forward_half, 0.0

        # pick 거리 안에서는 signed x 거리로 좌우 정렬 여부를 판단한다.
        if webcam_ball_x_offset < self.webcam_pick_x_min_px:
            return BallStatus.Left_Move, angle
        if webcam_ball_x_offset > self.webcam_pick_x_max_px:
            return BallStatus.Right_Move, angle
        return BallStatus.Pick_Ready, 0.0

    #realsense 기준으로 판단하는 각도
    def _status_from_angle(self, angle: Optional[float]) -> Tuple[int, float]:
        if angle is None:
            return BallStatus.Ball_None, 0.0

        #5도 이하는 직진
        if -self.angle_center_tol <= angle <= self.angle_center_tol:
            return BallStatus.Forward_3step, 0.0

        #5도 이상은 제자리회전
        if abs(angle) > self.angle_center_tol:
            if angle < 0:
                return BallStatus.Left_Turn_10, angle
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
        self.ball_pub = self.node.create_publisher(BallResult, topic_name, 10)

    def publish_ball_status(
        self,
        realsense_ball_detected: bool = False,
        realsense_ball_distance_cm: Optional[float] = None,
        realsense_ball_angle_error: Optional[float] = None,
        webcam_ball_detected: bool = False,
        webcam_ball_x_offset: Optional[float] = None,
        webcam_ball_x_distance: Optional[float] = None,
        webcam_ball_y_distance: Optional[float] = None,
        webcam_ball_angle_error: Optional[float] = None,
        webcam_ball_distance_px: Optional[float] = None,
        ball_in_hand: bool = False,
    ) -> Tuple[int, float]:
        features = BallFeatures(
            realsense_ball_detected=realsense_ball_detected,
            realsense_ball_distance_cm=realsense_ball_distance_cm,
            realsense_ball_angle_error=realsense_ball_angle_error,
            webcam_ball_detected=webcam_ball_detected,
            webcam_ball_x_offset=webcam_ball_x_offset,
            webcam_ball_x_distance=webcam_ball_x_distance,
            webcam_ball_y_distance=webcam_ball_y_distance,
            webcam_ball_angle_error=webcam_ball_angle_error,
            webcam_ball_distance_px=webcam_ball_distance_px,
            ball_in_hand=ball_in_hand,
        )

        status, angle = self.ball_decision.decide(features)

        msg = BallResult()
        msg.status = int(status)
        msg.angle = float(angle)
        if hasattr(msg, 'ball_in_hand'):
            msg.ball_in_hand = bool(ball_in_hand)
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

        self.ball_pub.publish(msg)

        return status, angle
