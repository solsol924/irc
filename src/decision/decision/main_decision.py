import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from collections import deque, Counter
from msgs.msg import LineResult, MotionCommand, MotionEnd, BallResult, HurdleResult

# 커스텀메시지 가져오기

class Motion:
    Initial_Pose = 0
    Forward_4step = 1
    Left_Half_Forward = 2  #기본4스텝
    Right_Half_Forward = 3
    Left_Forward = 4
    Right_Forward = 5
    Left_Turn = 6
    Right_Turn = 7
    Forward_half = 8
    Backward_half = 9
    Left_Move = 10
    Right_Move = 11
    Pick = 12
    Recatch = 13
    Shoot = 14
    Neck_Left = 15
    Neck_Right = 16
    Neck_Center = 17
    Hurdle_Forward_20 = 18
    Hurdle_Go = 19
    Forward_3step = 20
    Left_Half_Forward_3step = 21
    Right_Half_Forward_3step = 22
    Left_Turn_Ball = 23
    Right_Turn_Ball = 24

    Data_None = 99
    
    # 모션 번호 나열하기


MOTION_NAME = {
    value: name
    for name, value in vars(Motion).items()
    if not name.startswith('_') and isinstance(value, int)
}

class Ball:
    Ball_None = 99
    Ball_Forward = Motion.Forward_3step
    Ball_Forward_1step = Motion.Forward_half
    Ball_Lost = 45
    Ball_Right = Motion.Right_Move
    Ball_Left = Motion.Left_Move
    Pick_Ready = Motion.Pick
    Ball_In_Hand = 50

    Shoot_Ready = Motion.Shoot

class Line:
    Line_None = 99

class Hurdle:
    Hurdle_Detected = Motion.Hurdle_Forward_20
    Hurdle_Go = 19
    Hurdle_None = 99


    
class MainDecision(Node):
    def __init__(self):
        super().__init__('main_decision')

        #test_mode 파라미터 선언 및 초기화
        self.declare_parameter('test_mode', False)
        test_mode_param = self.get_parameter('test_mode').value
        if isinstance(test_mode_param, str):
            self.test_mode = test_mode_param.lower() in ('true', '1', 'yes', 'on')
        else:
            self.test_mode = bool(test_mode_param)

        #초기값 설정
        self.status = 0
        self.current_mode = "WaitingMode"
        #test mode true/false에 따라 초기값 조정
        self.motion_end = self.test_mode
        self.motion_ready = self.test_mode
        self.ball_decision_delay_sec = 1.0
        self.motion_end_received_time_ns = None
        self.ball_delay_log_printed = False
        self.line_data = False
        self.ball_data = False
        self.hurdle_data = False

        if self.test_mode:
            self.get_logger().info(
                "test_mode enabled: motion_ready and motion_end will stay true"
            )
        else:
            self.get_logger().info("motion_ready=true 수신 전까지 판단을 대기합니다.")

        #ball
        self.has_ball = False
        self.pick_try_count = 0
        self.pick_done = False
        self.ball_in_hand = False
        #pick이후 회전
        self.ball_count = 0
        self.turn_after_pick = False
        self.turn_count = 0
        self.turn_pick = Motion.Right_Turn
        #lost
        self.lost_count = 0
        self.lost_step = 0
        self.lost_found_dir = 0
        self.lost_body_turn_count = 0
        #goal
        self.goal_count = 0
        self.turn_after_shoot = False
        self.turn_shoot = Motion.Right_Turn

        #hurdle
        self.hurdle_step = 0
        self.hurdle_done = False

        #최근 5개의 데이터를 저장하는 버퍼
        self.line_buffer = deque(maxlen=5)
        self.line_follow_point_buffer = deque(maxlen=5)
        self.ball_buffer = deque(maxlen=5)
        self.ball_in_hand_buffer = deque(maxlen=5)
        self.hurdle_buffer = deque(maxlen=5)
        
        # subscribe
        self.line_result_sub = self.create_subscription(LineResult, 'line_result', self.LineResultCallback, 10)
        self.ball_result_sub = self.create_subscription(BallResult, 'ball_result', self.BallResultCallback, 10)
        self.hurdle_result_sub = self.create_subscription(HurdleResult, 'hurdle_result', self.HurdleResultCallback, 10)
        #motion_ready 명령을 못 받는 상황 방지
        motion_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.motion_end_sub = self.create_subscription(
            MotionEnd, 'motion_end', self.MotionEndCallback, motion_state_qos
        )
        #publish
        self.motion_pub = self.create_publisher(MotionCommand, 'motion_command', 10)
        
    # 콜백함수에서 모션 종료 여부를 업데이트
    def MotionEndCallback(self, motion_end_msg:MotionEnd):
        if self.test_mode:
            self.motion_ready = True
            self.motion_end = True
            return
        #메시지 받기 이전 상태 저장
        was_ready = self.motion_ready
        was_motion_end = self.motion_end
        #최신 상태 갱신
        self.motion_ready = motion_end_msg.motion_ready
        self.motion_end = motion_end_msg.motion_end

        # 모션 종료 신호가 새로 들어온 시각 저장
        if self.motion_end and not was_motion_end:
            self.motion_end_received_time_ns = self.get_clock().now().nanoseconds
            self.ball_delay_log_printed = False

        self.get_logger().info(
            f"motion_ready: {self.motion_ready}, motion_end: {self.motion_end}"
        )
        if self.motion_ready and not was_ready:
            self.get_logger().info("초기자세 완료 확인: 판단을 시작합니다.")
        
        
    def LineResultCallback(self, line_msg:LineResult):
        if not self.motion_ready:
            return

        #최신 데이터 갱신
        self.line_buffer.append(line_msg.status)
        self.line_follow_point_buffer.append(line_msg.follow_point)
        if self.motion_end == True:
            if len (self.line_buffer) >= 3:
                # Counter를 사용해 가장 빈도수가 높은 값 추출
                counts = Counter(self.line_buffer)
                most_common_status = counts.most_common(1)[0][0]
                #다수결 따라 라인 status 결정
                self.line_status = most_common_status
                follow_counts = Counter(self.line_follow_point_buffer)
                self.line_follow_point = follow_counts.most_common(1)[0][0]
                self.angle = line_msg.angle
                #라인 데이터 ready
                self.line_data = True
                # line result 상태, 각도를 로그로 출력
                self.get_logger().info(
                    f"[LineResult] status: {line_msg.status}, "
                    f"angle: {line_msg.angle}, "
                    f"follow_point: {self.line_follow_point}"
                )
                self.Decision()
            else:
                self.get_logger().info(f"not enough data in line buffer")
        else:
            self.get_logger().info(f"line: motion not ended yet")
            
    def BallResultCallback(self, ball_msg:BallResult):
        if not self.motion_ready:
            return

        self.ball_buffer.append(ball_msg.status)
        self.ball_in_hand_buffer.append(bool(getattr(ball_msg, 'ball_in_hand', False)))
        
        if self.motion_end == True:
            if len (self.ball_buffer) >= 3:
                counts = Counter(self.ball_buffer)
                most_common_status = counts.most_common(1)[0][0]
                hand_counts = Counter(self.ball_in_hand_buffer)
                #다수결 따라 ball status 결정
                self.ball_status = most_common_status
                self.ball_angle = ball_msg.angle
                self.ball_in_hand = hand_counts.most_common(1)[0][0]
                #ball 데이터 ready
                self.ball_data = True
                self.get_logger().info(f"[BallResult] status: {ball_msg.status}, angle: {ball_msg.angle}, in_hand: {self.ball_in_hand}")
                self.Decision()
            else:
                self.get_logger().info(f"not enough data in ball buffer")
        else:
            self.get_logger().info(f"ball: motion not ended yet")
            
    def HurdleResultCallback(self, hurdle_msg:HurdleResult):
        if not self.motion_ready:
            return

        self.hurdle_buffer.append(hurdle_msg.status)
        if self.motion_end == True:
            if len (self.hurdle_buffer) >= 3:
                counts = Counter(self.hurdle_buffer)
                most_common_status = counts.most_common(1)[0][0]
                #다수결 따라 hurdle status 결정
                self.hurdle_status = most_common_status
                self.hurdle_angle = hurdle_msg.angle
                #hurdle 데이터 ready
                self.hurdle_data = True
                self.get_logger().info(f"[HurdleResult] status: {hurdle_msg.status}, angle: {hurdle_msg.angle}")
                self.Decision()
            else:
                self.get_logger().info(f"not enough data in hurdle buffer")
        else:
            self.get_logger().info(f"hurdle: motion not ended yet")


###### 판단 로직 시작 #######
    def Decision(self):
        if not self.motion_ready:
            return

        if not (self.line_data == True and self.ball_data == True and self.hurdle_data == True):   
            self.get_logger().info("아직 모든 데이터가 도착하지 않았습니다. 판단 대기중...")
            return
        
        #모든 데이터가 준비된 경우에만 의사결정 로직 실행
        self.get_logger().info("3가지 데이터 모두 도착 완료! 판단을 시작합니다.")
        
        # BallMode 내부에서 Pick 확인, Pick 이후 회전까지 처리
        #우선순위 1 : ball mode
        if (
            self.pick_done == True
            or self.turn_after_pick == True
            or self.turn_after_shoot == True
            or (self.ball_status != Ball.Ball_None)
        ):
            # BallMode만 motion_end 수신 후 1초가 지난 뒤 판단
            if (
                not self.test_mode
                and self.motion_end_received_time_ns is not None
            ):
                elapsed_sec = (
                    self.get_clock().now().nanoseconds
                    - self.motion_end_received_time_ns
                ) / 1_000_000_000

                if elapsed_sec < self.ball_decision_delay_sec:
                    if not self.ball_delay_log_printed:
                        self.get_logger().info(
                            "BallMode 판단 대기: motion_end 수신 후 1초 대기합니다."
                        )
                        self.ball_delay_log_printed = True
                    return

            self.BallMode()

        #우선순위 2 : hurdle mode
        elif self.hurdle_status != Hurdle.Hurdle_None:
            self.lost_count = 0
            self.lost_step = 0
            self.lost_found_dir = 0
            self.lost_body_turn_count = 0

            self.HurdleMode()

        #lostmode 진행중이면 계속 lostmode
        elif self.lost_step != 0:
            self.LostMode()

        #우선순위 3 : lost mode    
        elif self.line_status == Line.Line_None:    
            self.LostMode()

        #우선순위 4 : line tracking mode
        else:
            self.LineTracking()
                
      
    def CheckBall(self):
        self.pick_done = False
        if self.ball_in_hand == True:
            self.has_ball = True
            self.get_logger().info("pick success: ball is in hand")
        else:
            self.has_ball = False
            self.get_logger().info("pick failed: ball is not in hand")

        return self.has_ball

    def TurnAfterPick(self):
        #회전 시작 첫 호출 시에만 방향을 결정
        if self.turn_count == 0:
            if self.ball_count == 0:
                self.turn_pick = Motion.Right_Turn
            elif self.ball_count == 1:
                self.turn_pick = Motion.Left_Turn
            else:
                self.turn_after_pick = False
                self.LineTracking()
                return

            self.ball_count += 1

        # 최소 한 번 회전한 뒤, 라인이 보이면 회전 종료
        if self.turn_count > 0 and self.line_status != Line.Line_None:
            self.turn_after_pick = False
            self.turn_count = 0
            self.pick_try_count = 0
            self.LineTracking()
            return
        
        # 라인이 안 보이면 최대 5번까지만 회전
        if self.turn_count >= 5:
            self.turn_after_pick = False
            self.turn_count = 0
            self.pick_try_count = 0
            self.LostMode()
            return

        self.status = self.turn_pick
        self.turn_count += 1
        self.MotionCommand()

    def TurnAfterShoot(self):
        #회전 시작 첫 호출 시에만 방향을 결정
        if self.turn_count == 0:
            if self.goal_count == 0:
                self.turn_shoot = Motion.Right_Turn
            elif self.goal_count == 1:
                self.turn_shoot = Motion.Left_Turn
            else:
                self.turn_after_shoot = False
                self.LineTracking()
                return

            self.goal_count += 1

        # 최소 한 번 회전한 뒤, 라인이 보이면 회전 종료
        if self.turn_count > 0 and self.line_status != Line.Line_None:
            self.turn_after_shoot = False
            self.turn_count = 0
            self.LineTracking()
            return
        
        # 라인이 안 보이면 최대 5번까지만 회전
        if self.turn_count >= 5:
            self.turn_after_shoot = False
            self.turn_count = 0
            self.LostMode()
            return

        self.status = self.turn_shoot
        self.turn_count += 1
        self.MotionCommand()

    #Ball mission            
    def BallMode(self):
        self.current_mode = "BallMode"

        #Pick 이후 공 확인, 회전 처리
        if self.pick_done == True:
            self.CheckBall()
            self.turn_after_pick = True
            self.TurnAfterPick()
            return
        
        #pick 회전루프
        if self.turn_after_pick == True:
            self.TurnAfterPick()
            return
        
        #Shoot 회전루프
        if self.turn_after_shoot == True:
            self.TurnAfterShoot()
            return
        
        ##### 공이 있음, shoot Mode #####
        #goal이 보이고 공을 가지고 있으면 shoot 시도
        if self.has_ball == True:
            #shoot 직전 공확인
            self.CheckBall()

            #공 없으면 무시하고 라인트래킹
            if self.has_ball == False:
                self.LineTracking()
                return
            
            #shoot 준비완료
            if self.ball_status == Ball.Shoot_Ready:
                self.status = Motion.Shoot
                #shoot 이후 처리
                self.has_ball = False
                self.turn_after_shoot = True
                self.turn_count = 0
                self.MotionCommand()
                return

            else:
                self.status = self.ball_status
                self.MotionCommand()
            return
        
        ##### 공이 없으면 Pick Mode #####
        #공이 없는데 ShootReady이면 무시
        if self.ball_status == Ball.Shoot_Ready:
            self.LineTracking()
            return
        
        #Pick은 한번만 시도 -> 나중에 횟수 변경하기
        if self.pick_try_count >= 1:
            self.LineTracking()
            return
        
        #Pick 준비 완료되면 동작 실행
        if self.ball_status == Ball.Pick_Ready:
            self.pick_try_count += 1
            self.status = Motion.Pick
            self.pick_done = True
            self.MotionCommand()
            return
        
        #그 외에는 비전이 준 명령 실행
        else:
            self.status = self.ball_status

        self.MotionCommand()
        
    #Hurdle mission            
    def HurdleMode(self):
        self.current_mode = "HurdleMode"

        #step 0: 허들 감지 후 20번 종종걸음
        if self.hurdle_step == 0:
            self.hurdle_step = 1
            self.status = Motion.Hurdle_Forward_20
            self.MotionCommand()
            return
        
        #step 1: 허들 넘기 실행
        if self.hurdle_step == 1:
            self.hurdle_step = 0
            self.status = Motion.Hurdle_Go
            self.MotionCommand()
            return

    
    #Lost             
    def LostMode(self):
        self.current_mode = "LostMode"

        #step 0 
        if self.lost_step == 0:
            if self.line_status != Line.Line_None:
                self.LineTracking()
                return
            # 목 왼쪽 회전
            self.lost_step = 1
            self.status = Motion.Neck_Left
            self.MotionCommand()
            return
        
        #step 1 : 왼쪽에서 라인 확인 
        if self.lost_step == 1:
            #라인 발견하면 step 3 이동, 목 원점 복귀, 방향 저장
            if self.line_status != Line.Line_None:
                self.lost_found_dir = -1
                self.lost_step = 3
                self.status = Motion.Neck_Center
                self.MotionCommand()
                return
            #목 오른쪽 회전
            self.lost_step = 2
            self.status = Motion.Neck_Right
            self.MotionCommand()
            return

        #step 2 : 오른쪽에서 라인 확인
        if self.lost_step == 2:
            #라인 발견하면 step 3 이동, 목 원점 복귀, 방향 저장
            if self.line_status != Line.Line_None:
                self.lost_found_dir = 1
                self.lost_step = 3
                self.status = Motion.Neck_Center
                self.MotionCommand()
                return

            # 목 중앙 복귀 후 처음부터 탐색
            self.lost_count = 0
            self.lost_step = 0
            self.lost_found_dir = 0
            self.lost_body_turn_count = 0

            self.status = Motion.Neck_Center
            self.MotionCommand()
            return

        #step 3 : 몸통 회전 명령
        if self.lost_step == 3:
            #라인 발견하면 lost mode 종료, line tracking으로 이동
            if self.line_status != Line.Line_None:
                self.LineTracking()
                return
            
            #왼쪽 회전 기억
            if self.lost_found_dir == -1:
                self.lost_step = 4
                self.lost_body_turn_count = 1
                self.status = Motion.Left_Turn
                self.MotionCommand()
                return
            
            #오른쪽 회전 기억
            elif self.lost_found_dir == 1:
                self.lost_step = 4
                self.lost_body_turn_count = 1
                self.status = Motion.Right_Turn
                self.MotionCommand()
                return

            else:
                self.lost_count = 0
                self.lost_step = 0
                self.lost_found_dir = 0
                self.lost_body_turn_count = 0
                self.LostMode()
                return

        #step 4 : 몸통 회전 후 라인 보이는지 판단
        if self.lost_step == 4:
            if self.line_status != Line.Line_None:
                self.LineTracking()
                return
            
            if self.lost_body_turn_count < 5:
                self.lost_body_turn_count += 1

                if self.lost_found_dir == -1:
                    self.status = Motion.Left_Turn

                elif self.lost_found_dir == 1:
                    self.status = Motion.Right_Turn
                else:
                    self.lost_count = 0
                    self.lost_step = 0
                    self.lost_found_dir = 0
                    self.lost_body_turn_count = 0
                    self.LostMode()
                    return

                self.MotionCommand()
                return

            self.lost_step = 0
            self.lost_found_dir = 0
            self.lost_body_turn_count = 0
            self.status = Motion.Forward_3step
            self.MotionCommand()
            return


    #Line tracking 
    def LineTracking(self):  
        self.current_mode = "LineTrackingMode"

        #라인을 찾고 lost count 초기화
        self.lost_count = 0 
        self.lost_step = 0
        self.lost_found_dir = 0
        self.lost_body_turn_count = 0

        #vision에서 받은 명령 그대로 실행
        self.status = self.line_status
        self.MotionCommand()
                
    def MotionCommand(self):
        if not self.motion_ready:
            self.get_logger().info("motion_ready=false: 모션 명령을 보내지 않습니다.")
            return

        motion_msg = MotionCommand()
        
        if self.status == 0:
            motion_msg.command = Motion.Initial_Pose
            
        elif self.status == 1:
            motion_msg.command = Motion.Forward_4step
        
        elif self.status == 2:
            motion_msg.command = Motion.Left_Half_Forward
        
        elif self.status == 3:
            motion_msg.command = Motion.Right_Half_Forward
        
        elif self.status == 4:
            motion_msg.command = Motion.Left_Forward
        
        elif self.status == 5:
            motion_msg.command = Motion.Right_Forward   
        
        elif self.status == 6:
            motion_msg.command = Motion.Left_Turn
        
        elif self.status == 7:
            motion_msg.command = Motion.Right_Turn
        
        elif self.status == 8:
            motion_msg.command = Motion.Forward_half
        
        elif self.status == 9:
            motion_msg.command = Motion.Backward_half
        
        elif self.status == 10:
            motion_msg.command = Motion.Left_Move
        
        elif self.status == 11:
            motion_msg.command = Motion.Right_Move
        
        elif self.status == 12:
            motion_msg.command = Motion.Pick
        
        elif self.status == 13:
            motion_msg.command = Motion.Recatch

        elif self.status == 14:
            motion_msg.command = Motion.Shoot

        elif self.status == 15:
            motion_msg.command = Motion.Neck_Left

        elif self.status == 16:
            motion_msg.command = Motion.Neck_Right

        elif self.status == 17:
            motion_msg.command = Motion.Neck_Center

        elif self.status == 18:
            motion_msg.command = Motion.Hurdle_Forward_20
            
        elif self.status == 19:
            motion_msg.command = Motion.Hurdle_Go
            
        elif self.status == 20:
            motion_msg.command = Motion.Forward_3step
            
        elif self.status == 21:
            motion_msg.command = Motion.Left_Half_Forward_3step
        
        elif self.status == 22:
            motion_msg.command = Motion.Right_Half_Forward_3step
            
        elif self.status == 23:
            motion_msg.command = Motion.Left_Turn_Ball
        
        elif self.status == 24:
            motion_msg.command = Motion.Right_Turn_Ball
        
        self.motion_pub.publish(motion_msg)
        motion_name = MOTION_NAME.get(motion_msg.command, 'Unknown')
        self.get_logger().info(
            f"[MotionCommand] command={motion_msg.command}, "
            f"motion={motion_name}, mode={self.current_mode}"
        )
        
        self.line_data = False
        self.ball_data = False
        self.hurdle_data = False
        #test mode일 때는 true 로 유지
        self.motion_end = True if self.test_mode else False
        self.line_buffer.clear()
        self.line_follow_point_buffer.clear()
        self.ball_buffer.clear()
        self.ball_in_hand_buffer.clear()
        self.hurdle_buffer.clear()
        
        
def main(args=None):
    rclpy.init(args=args)
    node = MainDecision()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
