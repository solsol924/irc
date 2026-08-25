import math
from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.node import Node
from msgs.msg import LineResult

class LineStatus:
    Forward_4step = 1
    Left_Half_Forward = 2
    Right_Half_Forward = 3
    Left_Turn_Half = 4
    Right_Turn_Half = 5
    Left_Turn = 6
    Right_Turn = 7
    Forward_half = 8
    Backward_half = 9
    Left_Move = 10
    Right_Move = 11
    Left_Turn_Curve = 21
    Right_Turn_Curve = 22
    Line_Lost = 99

@dataclass
class LineFeatures:
    # 검출된 라인 점 개수
    point_count: int

    # 점 3개 이하일 때 또는 직선 판단일 때 사용하는 선 각도
    line_angle: Optional[float] = None

    # 점 4개 이상일 때 이차함수의 a값
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
        # 직진, 미세회전, 중간회전, 회전 각도 기준
        self.forward_angle = 7.0
        self.fine_turn_angle = 22.5
        self.half_turn_angle = 30.0

        # x = a*y^2 + b*y + c 픽셀 좌표 피팅 기준
        self.curve_a = 1e-4

        #거리기준 - 픽셀 단위로 맞춰서 수정하기
        self.move_distance = 90.0
        self.steering_distance_max = 130.0
        self.curve_distance = 100.0

        # 직선에서 거리와 각도가 서로 반대 방향을 가리키는 특수
        # 상황에만 조향각을 사용한다. 평소에는 기존 거리 우선 로직을
        # 유지하고, 거리 보정은 방향 충돌을 완화하는 용도로 제한한다.
        self.steering_scale_px = 600.0
        self.steering_limit = 10.0


    def decide(self, features: LineFeatures) -> Tuple[int, float]:
        if features.point_count <= 0:
            return LineStatus.Line_Lost, 0.0

        # 점 1~2개에서는 follow_angle의 부호만으로 제자리 회전 방향을 정한다.
        if features.point_count <= 2:
            return self._status_from_follow_angle(features.follow_angle)

        # 점 3개는 일반 직선 상황이다.
        if features.point_count == 3:
            return self._status_from_straight_line(
                features.line_angle,
                features.line_distance,
            )

        # 점 4개 이상은 먼저 이차함수의 a값으로 직선과 곡선을 구분한다.
        curve_a = features.curve_a
        is_curve = curve_a is not None and abs(curve_a) > self.curve_a

        # 곡선 구간의 기존 거리 우선 및 접선 각도 판단은 유지한다.
        if is_curve:
            distance = features.line_distance
            if (
                distance is not None
                and abs(distance) >= self.curve_distance
            ):
                if distance < 0:
                    return LineStatus.Left_Half_Forward, 0.0
                return LineStatus.Right_Half_Forward, 0.0
            return self._status_from_curve_angle(features.tangent_angle)

        return self._status_from_straight_line(
            features.line_angle,
            features.line_distance,
        )

    def _status_from_straight_line(
        self,
        line_angle: Optional[float],
        line_distance: Optional[float],
    ) -> Tuple[int, float]:
        """Use steering only when straight-line distance and angle conflict."""
        distance_status = self._distance_half_status(line_distance)
        angle_status, angle_value = self._status_from_line_angle(line_angle)
        opposite_half = bool(
            (
                distance_status == LineStatus.Left_Half_Forward
                and angle_status == LineStatus.Right_Half_Forward
            )
            or (
                distance_status == LineStatus.Right_Half_Forward
                and angle_status == LineStatus.Left_Half_Forward
            )
        )
        use_steering = bool(
            line_angle is not None
            and line_distance is not None
            and self.move_distance
            <= abs(line_distance)
            < self.steering_distance_max
            and self.forward_angle
            < abs(line_angle)
            <= self.half_turn_angle
            and opposite_half
        )
        if use_steering:
            return self._status_from_conflicting_straight_errors(
                line_angle,
                line_distance,
            )

        # 기존 직선 로직: 중심에서 90px 이상 벗어나면 거리 방향의
        # 반보행을 우선하고, 그 안에서는 원래 라인 각도를 사용한다.
        if distance_status is not None:
            return distance_status, 0.0

        return angle_status, angle_value

    def _distance_half_status(
        self,
        line_distance: Optional[float],
    ) -> Optional[int]:
        if line_distance is None or abs(line_distance) < self.move_distance:
            return None
        if line_distance < 0.0:
            return LineStatus.Left_Half_Forward
        return LineStatus.Right_Half_Forward

    def _status_from_conflicting_straight_errors(
        self,
        line_angle: float,
        line_distance: float,
    ) -> Tuple[int, float]:
        """Combine only meaningful, opposite straight-line errors."""

        distance_angle = math.degrees(
            math.atan(
                line_distance
                / self.steering_scale_px
            )
        )
        distance_angle = max(
            -self.steering_limit,
            min(self.steering_limit, distance_angle),
        )
        steering_angle = line_angle + distance_angle
        status, angle = self._status_from_line_angle(steering_angle)

        # 거리 보정만으로 중간회전이 full turn으로 승격되지 않게 한다.
        if (
            status in (LineStatus.Left_Turn, LineStatus.Right_Turn)
            and abs(line_angle) <= self.half_turn_angle
        ):
            if steering_angle < 0.0:
                return LineStatus.Left_Turn_Half, abs(steering_angle)
            return LineStatus.Right_Turn_Half, abs(steering_angle)

        return status, angle

    def _status_from_follow_angle(self, angle: Optional[float]) -> Tuple[int, float]:
        if angle is None:
            return LineStatus.Line_Lost, 0.0

        if angle < 0.0:
            return LineStatus.Left_Turn, abs(angle)
        if angle > 0.0:
            return LineStatus.Right_Turn, abs(angle)

        return LineStatus.Forward_4step, 0.0

    #직선구간에서 판단기준
    def _status_from_line_angle(self, angle: Optional[float]) -> Tuple[int, float]:
        if angle is None:
            return LineStatus.Line_Lost, 0.0

        abs_angle = abs(angle)

        # 7도 이하: 직진
        if abs_angle <= self.forward_angle:
            return LineStatus.Forward_4step, 0.0

        # 7~22.5도: 전진하며 미세회전
        if abs_angle <= self.fine_turn_angle:
            if angle < 0:
                return LineStatus.Left_Half_Forward, abs_angle
            else:
                return LineStatus.Right_Half_Forward, abs_angle

        # 22.5도 초과, 30도 미만: 중간 제자리회전
        if abs_angle < self.half_turn_angle:
            if angle < 0:
                return LineStatus.Left_Turn_Half, abs_angle
            else:
                return LineStatus.Right_Turn_Half, abs_angle

        # 30도 이상: full turn
        if angle < 0:
            return LineStatus.Left_Turn, abs_angle
        else:
            return LineStatus.Right_Turn, abs_angle

    #곡선구간에서 판단 기준
    def _status_from_curve_angle(self, angle: Optional[float]) -> Tuple[int, float]:
            if angle is None:
                return LineStatus.Line_Lost, 0.0

            abs_angle = abs(angle)

            # 7도 이하: 직진
            if abs_angle <= self.forward_angle:
                return LineStatus.Forward_4step, 0.0

            # 7~22.5도: 미세회전
            if abs_angle <= self.fine_turn_angle:
                if angle < 0:
                    return LineStatus.Left_Half_Forward, abs_angle
                else:
                    return LineStatus.Right_Half_Forward, abs_angle

            # 22.5도 초과: curve 회전
            if angle < 0:
                return LineStatus.Left_Turn_Curve, abs_angle
            else:
                return LineStatus.Right_Turn_Curve, abs_angle


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
