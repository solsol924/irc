#!/usr/bin/env python3
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from geometry_msgs.msg import PointStamped  # type: ignore
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer # type: ignore  동기화
from rcl_interfaces.msg import SetParametersResult

import rclpy  # type: ignore
import cv2
import numpy as np
import time

class BasketballDetectorNode(Node):
    def __init__(self):
        super().__init__('basketball_detector')
        
        # 공용 변수
        self.image_width = 640
        self.image_height = 480

        self.roi_x_start = int(self.image_width * 1 // 5)  # 초록 박스 관심 구역
        self.roi_x_end   = int(self.image_width * 4 // 5)
        self.roi_y_start = int(self.image_height * 1 // 12)
        self.roi_y_end   = int(self.image_height * 11 // 12)

        # zandi
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = 'AVG: --- ms | FPS: --'
        self.last_position_text = 'Dist: -- m | Pos: --, --' # 위치 출력

        # 추적
        self.last_cx_img = None
        self.last_cy_ball = None
        self.last_radius = None
        self.last_z = None
        self.lost = 0

        # 변수
        self.ball_color = (0, 255, 0)
        self.rect_color = (0, 255, 0)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.fx, self.fy = 607.0, 606.0   # ros2 topic echo /camera/color/camera_info - 카메라 고유값
        self.cx_intr, self.cy_intr = 325.5, 239.4

        self.lower_hsv = np.array([8, 60, 60])
        self.upper_hsv = np.array([60, 255, 255]) # 주황색 기준으로
        self.depth_thresh = 1500.0  # mm 기준 마스크 거리
        self.depth_scale = 0.001    # m 변환용
        
        self.bridge = CvBridge()

        # 컬러, 깊이 영상 동기화
        color_sub = Subscriber(self, Image, '/camera/color/image_raw') # 칼라
        depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw') # 깊이
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        # 파라미터 선언
        self.declare_parameter("h_low", 8) 
        self.declare_parameter("h_high", 60)  
        self.declare_parameter("s_low", 60)  
        self.declare_parameter("s_high", 255)  
        self.declare_parameter("v_low", 0)  
        self.declare_parameter("v_high", 255)  

        # 파라미터 적용
        self.h_low = self.get_parameter("h_low").value
        self.h_high = self.get_parameter("h_high").value
        self.s_low = self.get_parameter("s_low").value
        self.s_high = self.get_parameter("s_high").value
        self.v_low = self.get_parameter("v_low").value
        self.v_high = self.get_parameter("v_high").value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.lower_hsv = np.array([self.h_low, self.s_low, self.v_low ], dtype=np.uint8) # 색공간 미리 선언
        self.upper_hsv = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
        self.hsv = None      # hsv 미리 선언

        self.pub_ball = self.create_publisher(PointStamped, '/basketball/position', 10) # 나중에 다른 노드에 전송하기 위한 퍼블리셔

        # 화면 클릭
        cv2.namedWindow('Basketball Detection')
        cv2.setMouseCallback('Basketball Detection', self.click)   

    def parameter_callback(self, params):
        for param in params:
            if param.name == "h_low":
                if param.value >= 0 and param.value <= self.h_high:
                    self.h_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "h_high":
                if param.value >= self.h_low and param.value <= 255:
                    self.h_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "s_low":
                if param.value >= 0 and param.value <= self.s_high:
                    self.s_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "s_high":
                if param.value >= self.s_low and param.value <= 255:
                    self.s_high = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "v_low":
                if param.value >= 0 and param.value <= self.v_high:
                    self.v_low = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "v_high":
                if param.value >= self.v_low and param.value <= 255:
                    self.v_high = param.value
                else:
                    return SetParametersResult(successful=False)
            
            self.lower_hsv = np.array([self.h_low, self.s_low, self.v_low ], dtype=np.uint8) # 색공간 변하면 적용
            self.upper_hsv = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)
            
        return SetParametersResult(successful=True)
    
    def click(self, event, x, y, _, __): # 화면 클릭
        if event != cv2.EVENT_LBUTTONDOWN or self.hsv is None:
            return
        if (self.roi_x_start <= x <= self.roi_x_end and self.roi_y_start <= y <= self.roi_y_end):
            # LAB 및 BGR 읽기
            H, S, V = [int(v) for v in self.hsv[y - self.roi_y_start, x - self.roi_x_start]]
            self.get_logger().info(f"[Pos] x={x - self.zandi_x}, y={- (y - self.zandi_y)} | HSV=({H},{S},{V})")
        else:
            return

    def image_callback(self, color_msg: Image, depth_msg: Image):
        start_time = time.time()
        
        # 영상 받아오기
        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        roi_color = frame[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]
        roi_depth = depth[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        # HSV 색 조절
        self.hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(self.hsv, self.lower_hsv, self.upper_hsv) # 주황색 범위 색만
        raw_mask[roi_depth >= self.depth_thresh] = 0  

        mask = raw_mask.copy() # 출력용 raw_mask

        # 모폴로지 연산
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel) # 침식 - 팽창
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 팽창 - 침식

        # 컨투어
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어
        best_cnt = None
        best_ratio = 1
        x_best = y_best = radius = None
        for cnt in contours: # 가장 원형에 가까운 컨투어 찾기
            area = cv2.contourArea(cnt) # 1. 면적 > 1000
            if area > 1000:
                (x, y), circle_r = cv2.minEnclosingCircle(cnt)
                circle_area = circle_r * circle_r * 3.1416
                ratio = abs((area / circle_area) - 1)
                # 2. 컨투어 면적과 외접원 면적의 비율이 가장 작은 놈
                if ratio < best_ratio and ratio < 0.3:
                    best_ratio = ratio
                    best_cnt = cnt
                    x_best = int(x)
                    y_best = int(y)
                    radius = circle_r

        # 검출 결과 처리: 이전 위치 유지 로직
        if best_cnt is not None:
            # 원 탐지를 했다면
            self.lost = 0
            cx_ball = x_best + self.roi_x_start
            cy_ball = y_best + self.roi_y_start  # 전체화면에서 중심 좌표
            
            x1 = max(x_best - 1, 0)
            x2 = min(x_best + 2, self.image_width)
            y1 = max(y_best - 1, 0)
            y2 = min(y_best + 2, self.image_height)

            roi_patch = roi_depth[y1:y2, x1:x2]
            z = float(roi_patch.mean()) * self.depth_scale # 3x3 거리
            
            # 이전 위치 업데이트
            self.last_cx_img = cx_ball
            self.last_cy_ball = cy_ball
            self.last_radius = radius
            self.last_z = z

            self.rect_color = (0, 255, 0)
            self.ball_color = (255, 0, 0)

        elif self.lost < 10 and self.last_cx_img is not None:
            # 최근 10프레임 내에는 이전 위치 유지
            self.lost += 1
            cx_ball = self.last_cx_img
            cy_ball = self.last_cy_ball
            radius = self.last_radius
            z = self.last_z
            self.ball_color = (0, 255, 255)
        else:
            # Miss 상태
            self.lost = 10
            self.last_position_text = 'Miss'
            # 표시할 위치 없음
            cx_ball = cy_ball = radius = z = None
            self.rect_color = (0, 0, 255)

        # 위치 퍼블리시 및 화면 표시
        if cx_ball is not None:
            # 카메라까지 거리 보정 값
            X = (cx_ball - self.cx_intr) * z / self.fx
            Y = (cy_ball - self.cy_intr) * z / self.fy
            msg = PointStamped() # 메세지 만들고
            msg.header = color_msg.header # 칼라 화면 프레임 정보
            msg.point.x, msg.point.y, msg.point.z = X, Y, z # 공 위치 좌표 3개
            self.pub_ball.publish(msg) # 보내기

            # 원 그리기
            cv2.circle(frame, (cx_ball, cy_ball), int(radius), self.ball_color, 2)
            cv2.circle(frame, (cx_ball, cy_ball), 5, (0, 0, 255), -1)

            # 원점과의 거리 정보
            dx = cx_ball - self.zandi_x
            dy = cy_ball - self.zandi_y
            self.last_position_text = f'Dist: {z:.2f}m | Pos: {dx}, {-dy}'
 
        # ROI랑 속도 표시
        cv2.rectangle(frame, (self.roi_x_start, self.roi_y_start), (self.roi_x_end, self.roi_y_end), self.rect_color, 1)
        cv2.circle(frame, (self.zandi_x, self.zandi_y), 3, (255, 255, 255), -1)

        # 후처리 된 마스킹 영상
        final_mask = np.zeros_like(mask)
        if best_cnt is not None:
            cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)

        # 딜레이 측정
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        now = time.time()
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            fps = self.frame_count / (now - self.last_report_time)
            self.last_avg_text = f'AVG: {avg_time*1000:.2f} ms | FPS: {fps:.2f}'
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        cv2.putText(frame, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, self.last_position_text, (10, self.roi_y_end + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 30), 2)

        cv2.imshow('Raw', raw_mask) # 기준 거리 이내, 주황색
        cv2.imshow('Mask', mask) # 기준 거리 이내, 주황색, 보정 들어간 마스크
        cv2.imshow('Final Mask', final_mask) # 최종적으로 공이라 판단한 마스크
        cv2.imshow('Basketball Detection', frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = BasketballDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
