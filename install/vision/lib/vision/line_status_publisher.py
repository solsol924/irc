from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.node import Node
from msgs.msg import LineResult

class LineStatus:
    Forward_4step = 1
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
    Line_Lost = 99

@dataclass
class LineFeatures:
    # 검출된 라인 점 개수
    point_count: int

    # 점 2개일 때 또는 직선 판단일 때 사용하는 선 각도
    line_angle: Optional[float] = None

    # 점 3개 이상일 때 이차함수의 a값
    curve_a: Optional[float] = None

    # 곡선일 때 중앙점에서의 접선 각도
    tangent_angle: Optional[float] = None

    # 로봇 중심선과 라인 사이 거리
    # 왼쪽(-), 오른쪽(+)
    line_distance: Optional[float] = None

    # 좌표 따라 이동을 쓸 경우 사용할 목표 좌표
    target_x: Optional[float] = None
    target_y: Optional[float] = None

    #로봇 중심점 좌표
    robot_center_x: float = 320.0
    robot_center_y: float = 480.0

    # 제자리 회전 기준 각도와 목표점까지의 거리
    follow_angle: Optional[float] = None
    follow_distance: Optional[float] = None



class LineDecision:
    def __init__(self):
        #직진, 미세회전, 회전 각도 기준 설정
        self.forward_angle = 15.0
        self.turn_angle = 38.0

        # x = a*y^2 + b*y + c 픽셀 좌표 피팅 기준
        self.curve_a = 1e-4

        #거리기준 - 픽셀 단위로 맞춰서 수정하기
        self.move_distance = 130.0


    def decide(self, features: LineFeatures) -> Tuple[int, float]:
        if features.point_count <= 0:
            return LineStatus.Line_Lost, 0.0

        # 점 1~2개에서는 follow_angle의 부호만으로 제자리 회전 방향을 정한다.
        if features.point_count <= 2:
            return self._status_from_follow_angle(features.follow_angle)

        # 점 3개 이상에서 라인이 중심선으로부터 130px 이상 벗어나면
        # 각도보다 거리 보정을 우선하여 라인이 있는 방향으로 미세회전한다.
        distance = features.line_distance
        if (
            distance is not None
            and abs(distance) >= self.move_distance
        ):
            if distance < 0:
                return LineStatus.Left_Half_Forward, 0.0
            return LineStatus.Right_Half_Forward, 0.0

        # 중심선과 가까우면 line_angle을 15도/38도 기준으로 판단한다.
        return self._status_from_line_angle(features.line_angle)

    def _status_from_follow_angle(self, angle: Optional[float]) -> Tuple[int, float]:
        if angle is None:
            return LineStatus.Line_Lost, 0.0

        if angle < 0.0:
            return LineStatus.Left_Turn, abs(angle)
        if angle > 0.0:
            return LineStatus.Right_Turn, abs(angle)
        return LineStatus.Forward_4step, 0.0

    def _status_from_line_angle(self, angle: Optional[float]) -> Tuple[int, float]:
        if angle is None:
            return LineStatus.Line_Lost, 0.0

        abs_angle = abs(angle)

        # 15도 이하: 직진
        if abs_angle <= self.forward_angle:
            return LineStatus.Forward_4step, 0.0

        # 15~38도: 미세회전
        if abs_angle <= self.turn_angle:
            if angle < 0:
                return LineStatus.Left_Half_Forward, abs_angle
            else:
                return LineStatus.Right_Half_Forward, abs_angle

        # 38도 초과: 회전
        if angle < 0:
            return LineStatus.Left_Forward, abs_angle
        else:
            return LineStatus.Right_Forward, abs_angle


class LineStatusPublisher:
    def __init__(self, node: Node, topic_name: str = 'line_result'):
        self.node = node
        self.line_decision = LineDecision()
        self.line_pub = self.node.create_publisher(LineResult, topic_name, 10)

    #라인 상태를 판단하고 Publish하는 함수
    def publish_line_status(
        self,
        point_count: int,
        line_angle: Optional[float] = None,
        curve_a: Optional[float] = None,
        tangent_angle: Optional[float] = None,
        line_distance: Optional[float] = None,
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
        robot_center_x: float = 320.0,
        robot_center_y: float = 480.0,
        follow_angle: Optional[float] = None,
        follow_distance: Optional[float] = None,
    ) -> Tuple[int, float]:

        #LineFeatures 객체 생성
        features = LineFeatures(
            point_count=point_count,
            line_angle=line_angle,
            curve_a=curve_a,
            tangent_angle=tangent_angle,
            line_distance=line_distance,
            target_x=target_x,
            target_y=target_y,
            robot_center_x=robot_center_x,
            robot_center_y=robot_center_y,
            follow_angle=follow_angle,
            follow_distance=follow_distance,
        )

        #라인 상태를 판단
        status, angle = self.line_decision.decide(features)

        #라인 상태를 Publish
        msg = LineResult()
        msg.status = int(status)
        msg.angle = float(angle)
        msg.follow_point = False

        self.line_pub.publish(msg)

        return status, angle