import rclpy
from rclpy.node import Node
from collections import deque, Counter
from msgs.msg import LineResult, MotionCommand, MotionEnd, BallResult, HurdleResult

# 커스텀메시지 가져오기

class Motion:
    Initial_Pose = 0
    Forward = 1
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
    Pick = 12
    Recatch = 13
    Shoot = 14
    Neck_Left = 15
    Neck_Right = 16
    Neck_Center = 17

    Data_None = 99
    
    # 모션 번호 나열하기

class Ball:
    Ball_None = 99
    Ball_Forward = Motion.Forward
    Ball_Forward_1step = Motion.Forward_half
    Ball_Lost = 45
    Ball_Right = Motion.Right_Move
    Ball_Left = Motion.Left_Move
    Pick_Ready = Motion.Pick
    Ball_In_Hand = 50

    Shoot_Ready = Motion.Shoot

class Line:
    Line_None = 99

    
class MainDecision(Node):
    def __init__(self):
        super().__init__('main_decision')

        #초기값 설정
        self.status = 0
        self.motion_end = False
        self.command_sent_for_current_motion = False
        self.line_data = False
        self.ball_data = False
        self.hurdle_data = False
        self.line_status = Line.Line_None
        self.ball_status = Ball.Ball_None
        self.hurdle_status = Line.Line_None
        self.angle = 0
        self.ball_angle = 0
        self.hurdle_angle = 0

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
        
        #최근 5개의 데이터를 저장하는 버퍼
        self.line_buffer = deque(maxlen=5)
        self.ball_buffer = deque(maxlen=5)
        self.ball_in_hand_buffer = deque(maxlen=5)
        self.hurdle_buffer = deque(maxlen=5)
        
        # subscribe
        self.line_result_sub = self.create_subscription(LineResult, 'line_result', self.LineResultCallback, 10)
        self.ball_result_sub = self.create_subscription(BallResult, 'ball_result', self.BallResultCallback, 10)
        self.hurdle_result_sub = self.create_subscription(HurdleResult, 'hurdle_result', self.HurdleResultCallback, 10)
        self.motion_end_sub = self.create_subscription(MotionEnd, 'motion_end', self.MotionEndCallback, 10)
        self.motion_prepare_sub = self.create_subscription(MotionEnd, 'motion_prepare', self.MotionPrepareCallback, 10)
        #publish
        self.motion_pub = self.create_publisher(MotionCommand, 'motion_command', 10)

        self.motion_name_by_command = {
            value: name
            for name, value in Motion.__dict__.items()
            if not name.startswith('_') and isinstance(value, int)
        }
        
    # 콜백함수에서 모션 종료 여부를 업데이트
    def MotionEndCallback(self, motion_end_msg:MotionEnd):
        self.motion_end = motion_end_msg.motion_end
        if motion_end_msg.motion_end == True:
            self.command_sent_for_current_motion = False
        self.get_logger().info(f"motion_end: {motion_end_msg.motion_end}")

    # motion에서 모션 종료 직전 prepare=True를 보내면 다음 명령을 미리 판단
    def MotionPrepareCallback(self, motion_prepare_msg:MotionEnd):
        if motion_prepare_msg.motion_end != True:
            return

        if self.command_sent_for_current_motion == True:
            self.get_logger().info("motion_prepare received, but command already sent")
            return

        self.get_logger().info("motion_prepare received: prepare next motion command")
        self.PrepareDecision()

    def PrepareDecision(self):
        if len(self.line_buffer) < 3:
            self.get_logger().info("not enough data in line buffer")
            return

        if len(self.ball_buffer) < 3:
            self.get_logger().info("not enough data in ball buffer")
            return

        if len(self.hurdle_buffer) < 3:
            self.get_logger().info("not enough data in hurdle buffer")
            return

        self.line_status = Counter(self.line_buffer).most_common(1)[0][0]
        self.ball_status = Counter(self.ball_buffer).most_common(1)[0][0]
        self.ball_in_hand = Counter(self.ball_in_hand_buffer).most_common(1)[0][0]
        self.hurdle_status = Counter(self.hurdle_buffer).most_common(1)[0][0]

        self.line_data = True
        self.ball_data = True
        self.hurdle_data = True

        self.get_logger().info(
            f"[PreparedData] line: {self.line_status}, ball: {self.ball_status}, "
            f"hurdle: {self.hurdle_status}, in_hand: {self.ball_in_hand}"
        )
        self.Decision()
        
        
    def LineResultCallback(self, line_msg:LineResult):
        #최신 데이터 갱신
        self.line_buffer.append(line_msg.status)
        self.angle = line_msg.angle
        self.get_logger().info(f"[LineResult] status: {line_msg.status}, angle: {line_msg.angle}")
            
    def BallResultCallback(self, ball_msg:BallResult):
        self.ball_buffer.append(ball_msg.status)
        self.ball_in_hand_buffer.append(bool(getattr(ball_msg, 'ball_in_hand', False)))
        self.ball_angle = ball_msg.angle
        self.get_logger().info(f"[BallResult] status: {ball_msg.status}, angle: {ball_msg.angle}, in_hand: {self.ball_in_hand}")
            
    def HurdleResultCallback(self, hurdle_msg:HurdleResult):
        self.hurdle_buffer.append(hurdle_msg.status)
        self.hurdle_angle = hurdle_msg.angle
        self.get_logger().info(f"[HurdleResult] status: {hurdle_msg.status}, angle: {hurdle_msg.angle}")


###### 판단 로직 시작 #######
    def Decision(self):
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
            self.BallMode()

        #우선순위 2 : hurdle mode
        elif self.hurdle_status == 11:
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
        pass
    
    #Lost             
    def LostMode(self):
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
            self.status = Motion.Forward
            self.MotionCommand()
            return


    #Line tracking 
    def LineTracking(self):  
        #라인을 찾고 lost count 초기화
        self.lost_count = 0 
        self.lost_step = 0
        self.lost_found_dir = 0
        self.lost_body_turn_count = 0

        #vision에서 받은 명령 그대로 실행
        self.status = self.line_status
        self.MotionCommand()
                
    def MotionCommand(self):
        motion_msg = MotionCommand()
        
        if self.status == 0:
            motion_msg.command = Motion.Initial_Pose
            
        elif self.status == 1:
            motion_msg.command = Motion.Forward
        
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
        
        self.motion_pub.publish(motion_msg)
        self.command_sent_for_current_motion = True
        motion_name = self.motion_name_by_command.get(motion_msg.command, "Unknown")
        self.get_logger().info(f"motion command: {motion_name} ({motion_msg.command})")
        
        self.line_data = False
        self.ball_data = False
        self.hurdle_data = False
        self.motion_end = False
        self.line_buffer.clear()
        self.ball_buffer.clear()
        self.ball_in_hand_buffer.clear()
        self.hurdle_buffer.clear()
        
        
def main(args=None):
    rclpy.init(args=args)
    node = MainDecision()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
