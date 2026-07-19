#include "callback.hpp"

#include <algorithm>

Callback::Callback()
: Node("callback_node")
{
    std::fill_n(all_raw_, NUMBER_OF_JOINTS, 2048.0);
    RCLCPP_INFO(
        this->get_logger(),
        "GOAT callback_node has been started (Streamlit-compatible raw mode).");
}


bool Callback::SetCurrentRaw(
    const std::array<int32_t, NUMBER_OF_JOINTS>& raw)
{
    for (int i = 0; i < NUMBER_OF_JOINTS; ++i)
    {
        all_raw_[i] = static_cast<double>(raw[static_cast<std::size_t>(i)]);
    }

    return true;
}


bool Callback::SelectMotion(int motion_id)
{
    if (sdk_motion_.Is_Moving())
    {
        RCLCPP_WARN(
            this->get_logger(),
            "[%d번] 명령 무시: 현재 모션 진행 중",
            motion_id);
        return false;
    }

    if (!sdk_motion_.Generate_Trajectory(motion_id, all_raw_))
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "[%d번] 모션을 찾을 수 없거나 궤적 생성에 실패했습니다.",
            motion_id);
        return false;
    }

    RCLCPP_INFO(
        this->get_logger(),
        "[%d번] Streamlit 방식 모션 실행 준비 완료",
        motion_id);
    return true;
}


bool Callback::NeedsSegmentPreparation() const
{
    return sdk_motion_.Needs_Segment_Preparation();
}


bool Callback::PrepareCurrentSegment()
{
    return sdk_motion_.Prepare_Current_Segment();
}


bool Callback::GetNextRaw(
    std::array<int32_t, NUMBER_OF_JOINTS>& raw_target)
{
    if (!sdk_motion_.Get_Next_Tick(all_raw_))
    {
        return false;
    }

    for (int i = 0; i < NUMBER_OF_JOINTS; ++i)
    {
        // Python의 int(current_q)와 동일하게 0 방향으로 소수점을 버립니다.
        raw_target[static_cast<std::size_t>(i)] =
            static_cast<int32_t>(all_raw_[i]);
    }

    return true;
}


bool Callback::GetFinalRaw(
    std::array<int32_t, NUMBER_OF_JOINTS>& final_raw) const
{
    double raw_double[NUMBER_OF_JOINTS]{};
    if (!sdk_motion_.Get_Final_Pose(raw_double))
    {
        return false;
    }

    for (int i = 0; i < NUMBER_OF_JOINTS; ++i)
    {
        final_raw[static_cast<std::size_t>(i)] =
            static_cast<int32_t>(raw_double[i]);
    }

    return true;
}


bool Callback::IsMoving() const
{
    return sdk_motion_.Is_Moving();
}


bool Callback::HasMotion(int motion_id) const
{
    return sdk_motion_.Has_Motion(motion_id);
}


void Callback::AbortMotion()
{
    sdk_motion_.Abort_Motion();
}