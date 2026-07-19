#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

constexpr int NUMBER_OF_JOINTS = 22;
constexpr double HZ = 200.0;
constexpr double PI = 3.14159265358979323846;

enum class BlendType
{
    Stop,
    Smooth
};

// Streamlit 호환 버전에서는 pose를 Dynamixel raw count 단위로 저장합니다.
struct MotionSequence
{
    std::vector<std::vector<double>> poses;
    std::vector<double> durations;
    std::vector<BlendType> blends;
};

struct TrajectorySegment
{
    std::vector<double> c0, c1, c2, c3, c4, c5;
    int total_ticks = 0;
    double T = 0.0;
};

class SDK_Motion
{
public:
    SDK_Motion();

    // getpose.py처럼 current pose, waypoints, waypoint 속도/가속도까지만 준비합니다.
    bool Generate_Trajectory(int motion_id, const double* current_pose_raw);

    // Python outer for-loop처럼 각 segment 계수는 해당 segment 진입 직전에 계산합니다.
    bool Needs_Segment_Preparation() const;
    bool Prepare_Current_Segment();

    // Python inner for-loop 한 번과 동일합니다.
    // 목표 raw를 계산하고 통신 결과와 무관하게 내부 tick을 즉시 진행합니다.
    bool Get_Next_Tick(double* all_raw);

    bool Get_Final_Pose(double* final_raw) const;

    void Abort_Motion();

    bool Is_Moving() const;
    bool Has_Motion(int motion_id) const;

private:
    void define_motions();

    // Streamlit은 duration을 5ms 배수로 별도 반올림하지 않습니다.
    double Snap_Time(double seconds) const;

    void calculate_coefficients(
        const std::vector<double>& q0,
        const std::vector<double>& qf,
        const std::vector<double>& v0,
        const std::vector<double>& vf,
        const std::vector<double>& a0,
        const std::vector<double>& af,
        double T,
        TrajectorySegment& segment);

    void ClearActiveMotion();

private:
    std::map<int, MotionSequence> motion_library_;

    std::vector<std::vector<double>> active_waypoints_;
    std::vector<double> active_durations_;
    std::vector<std::vector<double>> active_velocities_;
    std::vector<std::vector<double>> active_accelerations_;

    TrajectorySegment current_segment_;
    bool segment_prepared_ = false;

    int current_seg_idx_ = 0;
    int current_tick_ = 0;
    int current_motion_id_ = -1;

    std::vector<double> current_v_;
    std::vector<double> current_a_;
    std::vector<double> final_pose_raw_;
};