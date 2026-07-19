#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include "msgs/msg/motion_command.hpp"
#include "msgs/msg/motion_end.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "callback.hpp"
#include "dynamixel.hpp"

class MainNode : public rclcpp::Node
{
public:
    MainNode();
    ~MainNode() override;

private:
    struct MotionCompletion
    {
        int motion_id{-1};
        bool initial_motion{false};
        bool stopped{false};
        std::uint64_t tick_count{0};
        std::uint64_t tx_failure_count{0};
        double elapsed_seconds{0.0};
    };

    std::shared_ptr<Callback> callback_;
    std::shared_ptr<Dxl> dxl_port_;

    rclcpp::Subscription<msgs::msg::MotionCommand>::SharedPtr
        motion_command_sub_;
    rclcpp::Publisher<msgs::msg::MotionEnd>::SharedPtr
        motion_end_pub_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr
        hurdle_recalibrate_client_;

    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr completion_timer_;
    rclcpp::TimerBase::SharedPtr calibration_timer_;
    rclcpp::TimerBase::SharedPtr ready_timer_;

    std::thread motion_thread_;
    std::atomic<bool> motion_running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> current_motion_id_{-1};

    std::mutex completion_mutex_;
    std::optional<MotionCompletion> completion_;

    bool initial_pose_done_{false};
    bool initialization_ready_{false};
    bool hurdle_calibration_started_{false};

    void StartInitialPose();
    bool StartMotion(int motion_id, bool initial_motion);
    void RunMotionStreamlitStyle(
        int motion_id,
        bool initial_motion,
        Dxl::RawArray start_raw);
    void HandleMotionCompletion();
    void JoinFinishedMotionThread();

    void MotionCallback(
        const msgs::msg::MotionCommand::SharedPtr msg);

    void SchedulePostInitialPoseSequence();
    void RequestHurdleRecalibration();
    void FinishInitialization();

    void PublishMotionState(bool motion_end, bool motion_ready);
};