from dataclasses import dataclass
from typing import Tuple

from rclpy.node import Node
from msgs.msg import HurdleResult

class HurdleStatus:
    Hurdle_Detected = 18
    Hurdle_None = 99

@dataclass
class HurdleFeatures:
    #webcam에서 허들 발견 여부
    hurdle_detected: bool = False

class HurdleDecision:
    def __init__(self):
        pass

    def decide(self, features: HurdleFeatures) -> Tuple[int, float]:
        #허들 발견되면 Deteted publish
        if features.hurdle_detected:
            return HurdleStatus.Hurdle_Detected, 0.0

        return HurdleStatus.Hurdle_None, 0.0

class HurdleStatusPublisher:
    def __init__(self, node: Node, topic_name: str = 'hurdle_result'):
        self.node = node
        self.hurdle_decision = HurdleDecision()
        self.hurdle_pub = self.node.create_publisher(HurdleResult, topic_name, 10)

    #publish 함수
    def publish_hurdle_status(
            self,
            hurdle_detected: bool = False,
    ) -> Tuple[int, float]:

        features = HurdleFeatures(
            hurdle_detected=hurdle_detected,
        )

        status, angle = self.hurdle_decision.decide(features)

        msg = HurdleResult()
        msg.status = int(status)
        msg.angle = float(angle)

        self.hurdle_pub.publish(msg)

        return status, angle
