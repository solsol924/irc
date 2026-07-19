#pragma once

#include <array>
#include <cstdint>

#include <rclcpp/rclcpp.hpp>

#include "sdk.hpp"

class Callback : public rclcpp::Node
{
public:
    Callback();

    // getpose.py의 q_start와 동일하게 Dynamixel raw count를 시작점으로 사용합니다.
    bool SetCurrentRaw(
        const std::array<int32_t, NUMBER_OF_JOINTS>& raw);

    bool SelectMotion(int motion_id);

    bool NeedsSegmentPreparation() const;
    bool PrepareCurrentSegment();

    // getpose.py inner-loop 한 tick과 동일합니다.
    // SDK tick은 이 함수 호출 시 즉시 진행되며 통신 성공 여부와 연결되지 않습니다.
    bool GetNextRaw(
        std::array<int32_t, NUMBER_OF_JOINTS>& raw_target);

    bool GetFinalRaw(
        std::array<int32_t, NUMBER_OF_JOINTS>& final_raw) const;

    bool IsMoving() const;
    bool HasMotion(int motion_id) const;
    void AbortMotion();

private:
    SDK_Motion sdk_motion_;
    double all_raw_[NUMBER_OF_JOINTS]{};
};