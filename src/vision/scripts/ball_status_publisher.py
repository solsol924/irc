from dataclasses import dataclass
import math
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
    Recatch = 13
    Shoot = 14
    Left_Half_Forward_3step = 21
    Right_Half_Forward_3step = 22
    Left_Turn_Ball = 23
    Right_Turn_Ball = 24
    Ball_In_Hand = 50
    Ball_Lost = 45
    Ball_None = 99


@dataclass
class BallFeatures:
    realsense_ball_detected: bool = False
    realsense_ball_distance_cm: Optional[float] = None
    realsense_ball_angle_error: Optional[float] = None

    webcam_ball_detected: bool = False
    #로봇 중심선에서 떨어진 정도
    webcam_ball_x_distance: Optional[float] = None
    webcam_ball_angle_error: Optional[float] = None
    webcam_ball_distance_px: Optional[float] = None

    ball_in_hand: bool = False


class BallDecision:
    def __init__(self):
        #100cm 이하이면 공 모드, 25cm 이하에서는 webcam에서 보이는 거리(임의)
        self.ball_entry_distance_cm = 100.0

        # 직진, 제자리회전 기준각 5도
        self.angle_center_tol = 5.0

        # 웹캠의 x좌표 거리 기준
        ### 25px 이내면 중앙
        ### 60px 이내면 약한 방향 보정
        ### 가까운 상태에서 60px 초과면 좌우 이동
        self.x_center_tol_px = 25.0
        self.x_half_forward_tol_px = 60.0
        self.x_move_tol_px = 60.0

        #pick_ready 기준
        ### 공 중심까지 거리가 80px, 오차범위 ± 20px이내, 3프레임 연속 들어오면
        # 공 중심과 로봇 중심의 y축 거리가 260px 이하면 미세 접근 시작
        self.fine_adjust_start_distance_px = 260.0
        self.pick_distance_px = 80.0
        self.pick_distance_tol_px = 20.0


    def decide(self, features: BallFeatures) -> Tuple[int, float]:
        #공을 잡고 있으면 접근 명령을 보내지 않음
        if features.ball_in_hand:
            return BallStatus.Ball_None, 0.0

        if not self.Ball_mission_ready(features):
            return BallStatus.Ball_None, 0.0

        #webcam에서 공이 감지되면 webcam 기준으로 판단
        if features.webcam_ball_detected and features.webcam_ball_x_distance is not None:
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
        #webcam에서 감지된 공의 x좌표 거리, 각도, 픽셀 거리
        webcam_ball_x_distance = features.webcam_ball_x_distance
        angle = self.webcam_angle(features.webcam_ball_angle_error)
        webcam_ball_distance = features.webcam_ball_distance_px

        # 전체 픽셀 거리에서 x축 성분을 제외해 y축 거리의 절댓값을 구한다.
        if webcam_ball_distance is not None:
            webcam_ball_distance = math.sqrt(max(
                0.0,
                webcam_ball_distance ** 2 - webcam_ball_x_distance ** 2,
            ))

        #아직 멀리 있으면 방향보정하며 공에 접근
        if not self.Close_to_ball(webcam_ball_distance):
            return self.Move_to_Ball(webcam_ball_x_distance, angle)

        #공이 가까이 있으면 좌우 이동하며 중심 맞추기
        if abs(webcam_ball_x_distance) > self.x_move_tol_px:
            if webcam_ball_x_distance < 0:
                return BallStatus.Left_Move, angle
            return BallStatus.Right_Move, angle

        #층분히 가까우면서 x 좌표도 중앙에 있으면 pick 판단하기
        return self.PickReady(
            webcam_ball_distance,
            webcam_ball_x_distance,
            angle,
        )

    #공이 가까이 있는지 판단
    def Close_to_ball(
        self,
        webcam_ball_distance_px: Optional[float],
    ) -> bool:

        # 공 중심과 로봇 중심의 y축 픽셀 거리 기준
        if webcam_ball_distance_px is None:
            return False

        return webcam_ball_distance_px <= self.fine_adjust_start_distance_px

    #Pick ready 판단
    def PickReady(
        self,
        webcam_ball_distance_px: Optional[float],
        webcam_ball_x_distance: float,
        angle: float,
    ) -> Tuple[int, float]:

        #공이 중앙보다 20px이상 벗어나있으면 좌우 이동해서 정렬
        if abs(webcam_ball_x_distance) > self.x_center_tol_px:
            if webcam_ball_x_distance < 0:
                return BallStatus.Left_Move, angle
            return BallStatus.Right_Move, angle

        #거리값이 없으면 판단불가
        if webcam_ball_distance_px is None:
            return BallStatus.Ball_None, 0.0

        #목표 거리와의 오차를 계산, 허용가능 오차범위(20px)안에 들어오면 Pick_Ready
        distance_error = webcam_ball_distance_px - self.pick_distance_px
        if abs(distance_error) <= self.pick_distance_tol_px:
            return BallStatus.Pick_Ready, 0.0

        #거리오차가 양수이면 미세전진, 음수이면 미세후진
        if distance_error > 0:
            return BallStatus.Forward_half, 0.0

        return BallStatus.Backward_half, 0.0

    #아직 공과 멀리 있을 때 접근
    def Move_to_Ball(
        self,
        webcam_ball_x_distance: float,
        angle: float,
    ) -> Tuple[int, float]:
        abs_x_distance = abs(webcam_ball_x_distance)

        #거의 정면일때는 직진
        if abs_x_distance <= self.x_center_tol_px:
            return BallStatus.Forward_3step, 0.0

        #20~60px 오차에는 약한 방향보정하며 접근
        if abs_x_distance <= self.x_half_forward_tol_px:
            if webcam_ball_x_distance < 0:
                return BallStatus.Left_Half_Forward_3step, angle
            return BallStatus.Right_Half_Forward_3step, angle

        #60px 이상 오차에는 좌/우 회전하며 접근
        if webcam_ball_x_distance < 0:
            return BallStatus.Left_Forward, angle

        return BallStatus.Right_Forward, angle

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
                return BallStatus.Left_Turn_Ball, angle
            return BallStatus.Right_Turn_Ball, angle

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
        webcam_ball_x_distance: Optional[float] = None,
        webcam_ball_angle_error: Optional[float] = None,
        webcam_ball_distance_px: Optional[float] = None,
        ball_in_hand: bool = False,
    ) -> Tuple[int, float]:
        features = BallFeatures(
            realsense_ball_detected=realsense_ball_detected,
            realsense_ball_distance_cm=realsense_ball_distance_cm,
            realsense_ball_angle_error=realsense_ball_angle_error,
            webcam_ball_detected=webcam_ball_detected,
            webcam_ball_x_distance=webcam_ball_x_distance,
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

        self.ball_pub.publish(msg)

        return status, angle
