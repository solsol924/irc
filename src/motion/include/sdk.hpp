#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>

constexpr int NUMBER_OF_JOINTS = 22;

// 200Hz 제어
// 1 tick = 1 / 200 = 0.005초
constexpr double HZ = 200.0;
constexpr double PI = 3.14159265358979323846;

// 포즈 이음새 방식
enum class BlendType
{
    Stop,   // 해당 포즈에서 속도 0, 가속도 0으로 멈춤
    Smooth  // 해당 포즈를 멈추지 않고 평균속도로 지나감
};

// 모션 시퀀스 구조체
struct MotionSequence
{
    std::vector<std::vector<double>> poses; // poses[i] = i번째 포즈의 관절각 배열
    std::vector<double> durations; // durations[0] = current_pose -> poses[0], 이후 포즈 사이 이동 시간 (초)
    std::vector<BlendType> blends; // blends[i] = poses[i] -> poses[i+1] 이동 시, stop/smooth 선택
};

// 5차 다항식으로 만들어진 한 구간
struct TrajectorySegment
{
    std::vector<double> c0, c1, c2, c3, c4, c5; // 5차 다항식 계수
    int total_ticks = 0; // 이 구간을 몇 tick 동안 실행할지 (HZ 기준)
    double T = 0.0; // 이 구간의 총 시간 (초)
};

class SDK_Motion // SDK_Motion 클래스는 모션 라이브러리와 궤적 생성 기능을 제공합니다.
{
public:
    SDK_Motion();

    // 모션 궤적 생성
    //
    // active_trajectory_ 안에 앞으로 실행할 궤적을 미리 만들어두는 함수.
    //
    // current_pose부터 모션 내부 poses를 durations/blends 기준으로 실행함.
    bool Generate_Trajectory(int motion_id, double* current_pose);

    // 200Hz마다 호출되는 함수
    // 매 tick마다 다음 목표 관절각을 All_Theta에 써줌.
    // main.cpp나 callback.cpp에서 이 값을 실제 모터로 보내면 됨.
    bool Get_Next_Tick(double* All_Theta);

    // 현재 모션 실행 중인지 확인
    bool Is_Moving() const;

    // 모션 ID 존재 여부 확인
    bool Has_Motion(int motion_id) const;

private:
    void define_motions();

    // 시간을 0.005초 단위로 보정
    double Snap_Time(double seconds) const;

    // 한 구간 q0 -> qf를 5차 다항식 계수로 변환
    void calculate_coefficients(
        const std::vector<double>& q0,
        const std::vector<double>& qf,
        const std::vector<double>& v0,
        const std::vector<double>& vf,
        const std::vector<double>& a0,
        const std::vector<double>& af,
        double T,
        TrajectorySegment& segment
    );

private:
    std::map<int, MotionSequence> motion_library_;

    std::vector<TrajectorySegment> active_trajectory_;

    int current_seg_idx_ = 0;
    int current_tick_ = 0;
    int current_motion_id_ = -1;

    std::vector<double> current_v_;
    std::vector<double> current_a_;
};