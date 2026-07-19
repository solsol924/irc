#ifndef DYNAMIXEL_HPP
#define DYNAMIXEL_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include "dynamixel_sdk/dynamixel_sdk.h"

using namespace Eigen;

#define DEVICE_NAME          "/dev/ttyUSB0"
#define BAUDRATE             2000000
#define PROTOCOL_VERSION     2.0
#define NUMBER_OF_DYNAMIXELS 22

#define Position_Control_Mode 3

#define DxlReg_OperatingMode     11
#define DxlReg_TorqueEnable      64
#define DxlReg_LED               65
#define DxlReg_PositionDGain     80
#define DxlReg_PositionIGain     82
#define DxlReg_PositionPGain     84
#define DxlReg_GoalPosition      116
#define DxlReg_PresentVelocity   128
#define DxlReg_PresentPosition   132

#define RAD_TO_VALUE (4096.0 / (2.0 * M_PI))

class Dxl
{
public:
    using RawArray = std::array<int32_t, NUMBER_OF_DYNAMIXELS>;

    explicit Dxl(bool use_virtual = false);
    ~Dxl();

    Dxl(const Dxl&) = delete;
    Dxl& operator=(const Dxl&) = delete;

    bool IsReady() const;

    // getpose.py의 재생 버튼과 동일: 1~22번 Torque Enable=1을 순서대로 쓰고
    // 통신 결과 때문에 재생 흐름을 바꾸지 않습니다.
    void EnableTorqueAllStreamlitStyle();
    void DisableTorqueAll();

    // getpose.py의 q_start 구성과 동일합니다.
    // 각 ID를 한 번 읽고, COMM_SUCCESS가 아니면 이전 UI slider 역할의 fallback 값을 씁니다.
    bool ReadPresentRawStreamlitStyle(RawArray& raw);

    // getpose.py처럼 모션 시작 시 GroupSyncWrite를 한 번 만들고 전체 모션 동안 재사용합니다.
    bool BeginStreamWrite();
    int StreamWriteRaw(const RawArray& raw);
    void EndStreamWrite();

    // Streamlit이 재생 종료 후 마지막 pose를 slider에 반영하는 동작과 같습니다.
    void SetFallbackRaw(const RawArray& raw);

    // 기존 radian API 호환용
    bool syncReadTheta();
    bool GetThetaAct(VectorXd& theta);
    void syncReadThetaDot();
    VectorXd GetThetaDot();
    void CalculateEstimatedThetaDot(int dt_us);
    VectorXd GetThetaDotEstimated();
    int16_t GetPresentMode();
    VectorXd read_rad();

    bool SetThetaRef(const VectorXd& theta);
    bool syncWriteTheta();

    void SetPIDGain(VectorXd PID_Gain);
    int16_t SetPresentMode(int16_t Mode);

    void getParam(int32_t data, uint8_t* param);
    float convertValue2Radian(int32_t value);
    void Loop(bool RxTh, bool RxThDot);
    void initActuatorValues();
    void MoveToTargetQuintic(
        const VectorXd& theta_goal,
        int steps,
        int delay_ms);

    VectorXd th_;
    VectorXd th_last_;
    VectorXd th_dot_est_;
    VectorXd ref_th_;
    VectorXd ref_th_value_;

private:
    dynamixel::PortHandler* portHandler_{nullptr};
    dynamixel::PacketHandler* packetHandler_{nullptr};
    std::unique_ptr<dynamixel::GroupSyncWrite> motion_sync_write_;

    std::array<uint8_t, NUMBER_OF_DYNAMIXELS> dxl_id_{{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}};

    std::array<int32_t, NUMBER_OF_DYNAMIXELS> position_{};
    std::array<int32_t, NUMBER_OF_DYNAMIXELS> velocity_{};
    std::array<double, NUMBER_OF_DYNAMIXELS> zero_manual_offset_{};

    RawArray fallback_raw_{};
    RawArray last_goal_raw_{};

    int16_t mode_{Position_Control_Mode};
    bool virtual_mode_{false};
    bool ready_{false};
    bool port_open_{false};

    static constexpr double kPi = 3.14159265358979323846;
};

#endif  // DYNAMIXEL_HPP