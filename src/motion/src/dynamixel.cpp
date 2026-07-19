#include "dynamixel.hpp"

#include <chrono>
#include <iostream>
#include <thread>

Dxl::Dxl(bool use_virtual)
: virtual_mode_(use_virtual)
{
    th_.resize(NUMBER_OF_DYNAMIXELS);
    th_last_.resize(NUMBER_OF_DYNAMIXELS);
    th_dot_est_.resize(NUMBER_OF_DYNAMIXELS);
    ref_th_.resize(NUMBER_OF_DYNAMIXELS);
    ref_th_value_.resize(NUMBER_OF_DYNAMIXELS);

    th_.setZero();
    th_last_.setZero();
    th_dot_est_.setZero();
    ref_th_.setZero();
    ref_th_value_.setZero();

    fallback_raw_.fill(2048);
    last_goal_raw_.fill(2048);
    position_.fill(2048);
    velocity_.fill(0);
    zero_manual_offset_.fill(0.0);

    portHandler_ =
        dynamixel::PortHandler::getPortHandler(DEVICE_NAME);
    packetHandler_ =
        dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    if (virtual_mode_)
    {
        ready_ = true;
        std::cout
            << "[Info] Dxl initialized in VIRTUAL MODE"
            << std::endl;
        return;
    }

    if (!portHandler_->openPort())
    {
        std::cerr
            << "[Error] Failed to open the port: "
            << DEVICE_NAME << std::endl;
        return;
    }
    port_open_ = true;

    if (!portHandler_->setBaudRate(BAUDRATE))
    {
        std::cerr
            << "[Error] Failed to set baudrate: "
            << BAUDRATE << std::endl;
        portHandler_->closePort();
        port_open_ = false;
        return;
    }

    // getpose.py init_dxl()과 동일합니다.
    portHandler_->clearPort();

    // Streamlit UI의 최초 slider 초기화와 동일합니다.
    // 각 모터를 한 번 읽고 실패하면 2048을 fallback으로 둡니다.
    for (std::size_t i = 0; i < dxl_id_.size(); ++i)
    {
        uint32_t present_position = 0;
        uint8_t dxl_error = 0;
        const int comm_result = packetHandler_->read4ByteTxRx(
            portHandler_,
            dxl_id_[i],
            DxlReg_PresentPosition,
            &present_position,
            &dxl_error);

        if (comm_result == COMM_SUCCESS)
        {
            fallback_raw_[i] =
                static_cast<int32_t>(present_position);
        }

        position_[i] = fallback_raw_[i];
        last_goal_raw_[i] = fallback_raw_[i];
        th_[static_cast<Eigen::Index>(i)] =
            (static_cast<double>(fallback_raw_[i]) - 2048.0)
            * (2.0 * kPi / 4096.0);
        th_last_[static_cast<Eigen::Index>(i)] =
            th_[static_cast<Eigen::Index>(i)];
        ref_th_[static_cast<Eigen::Index>(i)] =
            th_[static_cast<Eigen::Index>(i)];
        ref_th_value_[static_cast<Eigen::Index>(i)] =
            static_cast<double>(fallback_raw_[i]);
    }

    ready_ = true;

    std::cout
        << "[Info] Dynamixel Streamlit-compatible initialization complete."
        << std::endl;
}


Dxl::~Dxl()
{
    EndStreamWrite();

    if (virtual_mode_)
    {
        return;
    }

    if (port_open_)
    {
        DisableTorqueAll();
        portHandler_->closePort();
        port_open_ = false;
    }
}


bool Dxl::IsReady() const
{
    return ready_;
}


void Dxl::EnableTorqueAllStreamlitStyle()
{
    if (virtual_mode_ || !ready_)
    {
        return;
    }

    // getpose.py의 재생 버튼과 동일하게 결과를 분기 조건으로 쓰지 않습니다.
    for (std::size_t i = 0; i < dxl_id_.size(); ++i)
    {
        uint8_t dxl_error = 0;
        packetHandler_->write1ByteTxRx(
            portHandler_,
            dxl_id_[i],
            DxlReg_TorqueEnable,
            1,
            &dxl_error);
    }
}


void Dxl::DisableTorqueAll()
{
    if (virtual_mode_ || !port_open_)
    {
        return;
    }

    for (std::size_t i = 0; i < dxl_id_.size(); ++i)
    {
        uint8_t dxl_error = 0;
        packetHandler_->write1ByteTxRx(
            portHandler_,
            dxl_id_[i],
            DxlReg_TorqueEnable,
            0,
            &dxl_error);
    }
}


bool Dxl::ReadPresentRawStreamlitStyle(RawArray& raw)
{
    if (virtual_mode_)
    {
        raw = last_goal_raw_;
        return true;
    }

    if (!ready_ || !port_open_)
    {
        return false;
    }

    // getpose.py move_sequence_smoothly()의 q_start 구성과 동일:
    // ID별 한 번 읽고 COMM_SUCCESS가 아니면 slider fallback 사용.
    for (std::size_t i = 0; i < dxl_id_.size(); ++i)
    {
        uint32_t present_position = 0;
        uint8_t dxl_error = 0;
        const int comm_result = packetHandler_->read4ByteTxRx(
            portHandler_,
            dxl_id_[i],
            DxlReg_PresentPosition,
            &present_position,
            &dxl_error);

        raw[i] =
            (comm_result == COMM_SUCCESS)
            ? static_cast<int32_t>(present_position)
            : fallback_raw_[i];
    }

    return true;
}


bool Dxl::BeginStreamWrite()
{
    if (virtual_mode_)
    {
        return true;
    }

    if (!ready_ || !port_open_)
    {
        return false;
    }

    // getpose.py는 모션 한 번당 GroupSyncWrite 객체를 정확히 한 번 만듭니다.
    motion_sync_write_ =
        std::make_unique<dynamixel::GroupSyncWrite>(
            portHandler_,
            packetHandler_,
            DxlReg_GoalPosition,
            4);

    return true;
}


int Dxl::StreamWriteRaw(const RawArray& raw)
{
    if (virtual_mode_)
    {
        last_goal_raw_ = raw;
        for (std::size_t i = 0; i < raw.size(); ++i)
        {
            th_[static_cast<Eigen::Index>(i)] =
                (static_cast<double>(raw[i]) - 2048.0)
                * (2.0 * kPi / 4096.0);
        }
        return COMM_SUCCESS;
    }

    if (!motion_sync_write_)
    {
        return COMM_TX_FAIL;
    }

    // Python은 addParam() 반환값을 확인하지 않습니다.
    for (std::size_t i = 0; i < dxl_id_.size(); ++i)
    {
        uint8_t parameter[4] = {0, 0, 0, 0};
        getParam(raw[i], parameter);
        motion_sync_write_->addParam(dxl_id_[i], parameter);
    }

    // Python은 txPacket() 결과와 무관하게 다음 tick으로 진행합니다.
    const int comm_result = motion_sync_write_->txPacket();
    motion_sync_write_->clearParam();

    last_goal_raw_ = raw;
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        ref_th_value_[static_cast<Eigen::Index>(i)] =
            static_cast<double>(raw[i]);
        ref_th_[static_cast<Eigen::Index>(i)] =
            (static_cast<double>(raw[i]) - 2048.0)
            * (2.0 * kPi / 4096.0);
    }

    return comm_result;
}


void Dxl::EndStreamWrite()
{
    if (motion_sync_write_)
    {
        motion_sync_write_->clearParam();
        motion_sync_write_.reset();
    }
}


void Dxl::SetFallbackRaw(const RawArray& raw)
{
    fallback_raw_ = raw;
}


bool Dxl::syncReadTheta()
{
    RawArray raw{};
    if (!ReadPresentRawStreamlitStyle(raw))
    {
        return false;
    }

    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        position_[i] = raw[i];
        th_[static_cast<Eigen::Index>(i)] =
            (static_cast<double>(raw[i]) - 2048.0)
            * (2.0 * kPi / 4096.0)
            - zero_manual_offset_[i];
    }

    return true;
}


bool Dxl::GetThetaAct(VectorXd& theta)
{
    if (!syncReadTheta())
    {
        return false;
    }

    theta = th_;
    return true;
}


void Dxl::syncReadThetaDot()
{
    if (virtual_mode_ || !ready_)
    {
        return;
    }

    dynamixel::GroupSyncRead group_sync_read(
        portHandler_,
        packetHandler_,
        DxlReg_PresentVelocity,
        4);

    for (std::size_t i = 0; i < dxl_id_.size(); ++i)
    {
        group_sync_read.addParam(dxl_id_[i]);
    }

    if (group_sync_read.txRxPacket() == COMM_SUCCESS)
    {
        for (std::size_t i = 0; i < dxl_id_.size(); ++i)
        {
            velocity_[i] = static_cast<int32_t>(
                group_sync_read.getData(
                    dxl_id_[i],
                    DxlReg_PresentVelocity,
                    4));
        }
    }

    group_sync_read.clearParam();
}


VectorXd Dxl::GetThetaDot()
{
    VectorXd result(NUMBER_OF_DYNAMIXELS);

    if (virtual_mode_)
    {
        result.setZero();
        return result;
    }

    for (std::size_t i = 0; i < velocity_.size(); ++i)
    {
        result[static_cast<Eigen::Index>(i)] =
            static_cast<double>(velocity_[i]) * 0.003816667;
    }

    return result;
}


void Dxl::CalculateEstimatedThetaDot(int dt_us)
{
    if (dt_us <= 0)
    {
        th_dot_est_.setZero();
        return;
    }

    const double dt_seconds =
        static_cast<double>(dt_us) * 1.0e-6;
    th_dot_est_ = (th_ - th_last_) / dt_seconds;
    th_last_ = th_;
}


VectorXd Dxl::GetThetaDotEstimated()
{
    return th_dot_est_;
}


int16_t Dxl::GetPresentMode()
{
    return mode_;
}


VectorXd Dxl::read_rad()
{
    VectorXd result(NUMBER_OF_DYNAMIXELS);
    RawArray raw{};

    if (!ReadPresentRawStreamlitStyle(raw))
    {
        result.setZero();
        return result;
    }

    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        result[static_cast<Eigen::Index>(i)] =
            (static_cast<double>(raw[i]) - 2048.0)
            * (2.0 * kPi / 4096.0);
    }

    return result;
}


bool Dxl::SetThetaRef(const VectorXd& theta)
{
    if (theta.size() != NUMBER_OF_DYNAMIXELS || !theta.allFinite())
    {
        return false;
    }

    ref_th_ = theta;
    return true;
}


bool Dxl::syncWriteTheta()
{
    RawArray raw{};

    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        const double raw_double =
            ref_th_[static_cast<Eigen::Index>(i)]
            * RAD_TO_VALUE
            + 2048.0;
        raw[i] = static_cast<int32_t>(raw_double);
    }

    const bool created_here = !motion_sync_write_;
    if (created_here && !BeginStreamWrite())
    {
        return false;
    }

    const int result = StreamWriteRaw(raw);

    if (created_here)
    {
        EndStreamWrite();
    }

    return result == COMM_SUCCESS;
}


void Dxl::SetPIDGain(VectorXd PID_Gain)
{
    if (virtual_mode_ || !ready_ || PID_Gain.size() < 3)
    {
        return;
    }

    const uint16_t p_gain = static_cast<uint16_t>(PID_Gain(0));
    const uint16_t i_gain = static_cast<uint16_t>(PID_Gain(1));
    const uint16_t d_gain = static_cast<uint16_t>(PID_Gain(2));

    for (std::size_t i = 0; i < dxl_id_.size(); ++i)
    {
        uint8_t dxl_error = 0;
        packetHandler_->write2ByteTxRx(
            portHandler_, dxl_id_[i], DxlReg_PositionPGain,
            p_gain, &dxl_error);
        packetHandler_->write2ByteTxRx(
            portHandler_, dxl_id_[i], DxlReg_PositionIGain,
            i_gain, &dxl_error);
        packetHandler_->write2ByteTxRx(
            portHandler_, dxl_id_[i], DxlReg_PositionDGain,
            d_gain, &dxl_error);
    }
}


int16_t Dxl::SetPresentMode(int16_t /*Mode*/)
{
    mode_ = Position_Control_Mode;
    return mode_;
}


void Dxl::getParam(int32_t data, uint8_t* param)
{
    param[0] = DXL_LOBYTE(DXL_LOWORD(data));
    param[1] = DXL_HIBYTE(DXL_LOWORD(data));
    param[2] = DXL_LOBYTE(DXL_HIWORD(data));
    param[3] = DXL_HIBYTE(DXL_HIWORD(data));
}


float Dxl::convertValue2Radian(int32_t value)
{
    return static_cast<float>(
        (static_cast<double>(value) - 2048.0)
        * (2.0 * kPi / 4096.0));
}


void Dxl::Loop(bool RxTh, bool RxThDot)
{
    if (RxTh)
    {
        syncReadTheta();
    }
    if (RxThDot)
    {
        syncReadThetaDot();
    }
}


void Dxl::initActuatorValues()
{
    zero_manual_offset_.fill(0.0);
}


void Dxl::MoveToTargetQuintic(
    const VectorXd& theta_goal,
    int steps,
    int delay_ms)
{
    if (steps <= 0)
    {
        return;
    }

    const VectorXd theta_now = read_rad();

    for (int step = 1; step <= steps; ++step)
    {
        const double t =
            static_cast<double>(step) / static_cast<double>(steps);
        const double rate =
            t * t * t * (10.0 - 15.0 * t + 6.0 * t * t);
        const VectorXd theta_interp =
            theta_now + (theta_goal - theta_now) * rate;

        SetThetaRef(theta_interp);
        syncWriteTheta();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(delay_ms));
    }

    SetThetaRef(theta_goal);
    syncWriteTheta();
}