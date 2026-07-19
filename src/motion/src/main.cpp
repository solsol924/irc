#include "main.hpp"

#include <chrono>
#include <exception>
#include <functional>
#include <utility>

using namespace std::chrono_literals;

static_assert(
    NUMBER_OF_JOINTS == NUMBER_OF_DYNAMIXELS,
    "SDK 관절 수와 Dynamixel 개수가 다릅니다.");


MainNode::MainNode()
: Node("motion_main_node")
{
    RCLCPP_INFO(
        this->get_logger(),
        "Motion MainNode 시작");

    callback_ = std::make_shared<Callback>();
    dxl_port_ = std::make_shared<Dxl>(false);

    auto motion_state_qos =
        rclcpp::QoS(1).reliable().transient_local();
    motion_end_pub_ =
        this->create_publisher<msgs::msg::MotionEnd>(
            "motion_end",
            motion_state_qos);

    hurdle_recalibrate_client_ =
        this->create_client<std_srvs::srv::Trigger>(
            "/hurdle/recalibrate");

    motion_command_sub_ =
        this->create_subscription<msgs::msg::MotionCommand>(
            "motion_command",
            10,
            std::bind(
                &MainNode::MotionCallback,
                this,
                std::placeholders::_1));

    // 작업 스레드가 끝났는지만 확인하는 timer입니다.
    // 200Hz 모션 전송에는 관여하지 않습니다.
    completion_timer_ = this->create_wall_timer(
        1ms,
        std::bind(
            &MainNode::HandleMotionCompletion,
            this));

    PublishMotionState(false, false);

    if (!dxl_port_->IsReady())
    {
        RCLCPP_FATAL(
            this->get_logger(),
            "Dynamixel 포트 초기화 실패. 초기자세를 시작하지 않습니다.");
        return;
    }

    // 생성자 안에서 스레드를 바로 시작하지 않고 executor가 시작된 직후 실행합니다.
    // v3의 200ms 안정화 지연은 제거했습니다.
    startup_timer_ = this->create_wall_timer(
        1ms,
        [this]()
        {
            startup_timer_->cancel();
            StartInitialPose();
        });
}


MainNode::~MainNode()
{
    stop_requested_.store(true);

    if (startup_timer_)
    {
        startup_timer_->cancel();
    }
    if (completion_timer_)
    {
        completion_timer_->cancel();
    }
    if (calibration_timer_)
    {
        calibration_timer_->cancel();
    }
    if (ready_timer_)
    {
        ready_timer_->cancel();
    }

    if (motion_thread_.joinable())
    {
        motion_thread_.join();
    }
}


void MainNode::StartInitialPose()
{
    if (!StartMotion(0, true))
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "초기자세 모션을 시작하지 못했습니다.");
        PublishMotionState(false, false);
    }
}


bool MainNode::StartMotion(int motion_id, bool initial_motion)
{
    if (!dxl_port_->IsReady())
    {
        return false;
    }

    if (motion_running_.load())
    {
        return false;
    }

    if (!callback_->HasMotion(motion_id))
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "존재하지 않는 모션 ID: %d",
            motion_id);
        return false;
    }

    JoinFinishedMotionThread();

    // getpose.py 재생 버튼 순서와 동일합니다.
    // 1) 전체 Torque ON
    // 2) move_sequence_smoothly() 내부에서 Present Position 읽기
    dxl_port_->EnableTorqueAllStreamlitStyle();

    Dxl::RawArray current_raw{};
    if (!dxl_port_->ReadPresentRawStreamlitStyle(current_raw))
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "Dynamixel 현재 위치를 읽을 수 없습니다.");
        return false;
    }

    callback_->SetCurrentRaw(current_raw);

    if (!callback_->SelectMotion(motion_id) ||
        !callback_->IsMoving())
    {
        return false;
    }

    current_motion_id_.store(motion_id);
    motion_running_.store(true);

    if (initial_motion)
    {
        PublishMotionState(false, false);
    }
    else
    {
        PublishMotionState(false, true);
    }

    try
    {
        motion_thread_ = std::thread(
            &MainNode::RunMotionStreamlitStyle,
            this,
            motion_id,
            initial_motion,
            current_raw);
    }
    catch (const std::exception& error)
    {
        motion_running_.store(false);
        current_motion_id_.store(-1);
        callback_->AbortMotion();

        RCLCPP_ERROR(
            this->get_logger(),
            "모션 스레드 생성 실패: %s",
            error.what());
        return false;
    }

    RCLCPP_INFO(
        this->get_logger(),
        "모션 %d 실행 시작",
        motion_id);

    return true;
}


void MainNode::RunMotionStreamlitStyle(
    int motion_id,
    bool initial_motion,
    Dxl::RawArray start_raw)
{
    MotionCompletion result;
    result.motion_id = motion_id;
    result.initial_motion = initial_motion;

    Dxl::RawArray last_raw = start_raw;
    const auto motion_start = std::chrono::steady_clock::now();

    if (!dxl_port_->BeginStreamWrite())
    {
        result.stopped = true;
    }
    else
    {
        // getpose.py: dt = 1.0 / hz
        const std::chrono::duration<double> dt_seconds(1.0 / HZ);

        while (!stop_requested_.load() && callback_->IsMoving())
        {
            // getpose.py는 각 outer segment 진입 시 계수를 먼저 계산하고,
            // 그 다음 inner tick의 loop_start를 측정합니다.
            if (callback_->NeedsSegmentPreparation() &&
                !callback_->PrepareCurrentSegment())
            {
                result.stopped = true;
                break;
            }

            // getpose.py: loop_start = time.time()
            const auto loop_start = std::chrono::steady_clock::now();

            Dxl::RawArray target_raw{};
            if (!callback_->GetNextRaw(target_raw))
            {
                result.stopped = true;
                break;
            }

            // getpose.py와 동일하게 txPacket 결과 때문에 tick 진행을 바꾸지 않습니다.
            const int comm_result =
                dxl_port_->StreamWriteRaw(target_raw);
            if (comm_result != COMM_SUCCESS)
            {
                ++result.tx_failure_count;
            }

            last_raw = target_raw;
            ++result.tick_count;

            // getpose.py:
            // elapsed = time.time() - loop_start
            // time.sleep(max(0.0, dt - elapsed))
            const auto elapsed =
                std::chrono::steady_clock::now() - loop_start;
            const auto remaining = dt_seconds - elapsed;

            if (remaining.count() > 0.0)
            {
                std::this_thread::sleep_for(remaining);
            }
        }

        dxl_port_->EndStreamWrite();
    }

    if (stop_requested_.load())
    {
        result.stopped = true;
    }

    // Streamlit은 함수 반환 후 저장된 마지막 pose를 slider에 반영합니다.
    // 마지막 보간 tick의 부동소수점 오차값이 아니라 원본 저장 pose를 사용합니다.
    Dxl::RawArray final_pose_raw{};
    if (callback_->GetFinalRaw(final_pose_raw))
    {
        dxl_port_->SetFallbackRaw(final_pose_raw);
    }
    else
    {
        dxl_port_->SetFallbackRaw(last_raw);
    }

    result.elapsed_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - motion_start)
            .count();

    {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        completion_ = result;
    }
}


void MainNode::HandleMotionCompletion()
{
    std::optional<MotionCompletion> result;

    {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        if (!completion_.has_value())
        {
            return;
        }

        result = completion_;
        completion_.reset();
    }

    if (motion_thread_.joinable())
    {
        motion_thread_.join();
    }

    motion_running_.store(false);
    current_motion_id_.store(-1);

    if (!result.has_value() || result->stopped)
    {
        callback_->AbortMotion();
        PublishMotionState(false, false);
        RCLCPP_ERROR(
            this->get_logger(),
            "모션 %d 실행이 중단되었습니다.",
            result.has_value() ? result->motion_id : -1);
        return;
    }

    if (result->tx_failure_count > 0)
    {
        // 모션 중에는 출력하지 않아 Streamlit 루프 시간을 건드리지 않고,
        // 종료 후에만 진단 정보를 한 번 표시합니다.
        RCLCPP_WARN(
            this->get_logger(),
            "모션 %d 중 txPacket 실패 %llu회. "
            "Streamlit과 동일하게 해당 tick은 재시도하지 않고 진행했습니다.",
            result->motion_id,
            static_cast<unsigned long long>(result->tx_failure_count));
    }

    RCLCPP_INFO(
        this->get_logger(),
        "모션 %d 완료: ticks=%llu, 실제 실행시간=%.6f초",
        result->motion_id,
        static_cast<unsigned long long>(result->tick_count),
        result->elapsed_seconds);

    if (result->initial_motion && !initial_pose_done_)
    {
        initial_pose_done_ = true;
        PublishMotionState(false, false);

        RCLCPP_INFO(
            this->get_logger(),
            "초기자세 완료: 5초 뒤 hurdle 캘리브레이션, "
            "10초 뒤 motion_ready=true");

        SchedulePostInitialPoseSequence();
        return;
    }

    PublishMotionState(true, true);
}


void MainNode::JoinFinishedMotionThread()
{
    if (!motion_running_.load() && motion_thread_.joinable())
    {
        motion_thread_.join();
    }
}


void MainNode::MotionCallback(
    const msgs::msg::MotionCommand::SharedPtr msg)
{
    const int command = msg->command;

    RCLCPP_INFO(
        this->get_logger(),
        "MotionCommand 수신: %d",
        command);

    if (!initialization_ready_)
    {
        RCLCPP_WARN(
            this->get_logger(),
            "초기자세 후 대기/캘리브레이션 완료 전, 모션 명령 무시");
        return;
    }

    if (motion_running_.load())
    {
        RCLCPP_WARN(
            this->get_logger(),
            "모션 %d 실행 중, 새 모션 %d 명령 무시",
            current_motion_id_.load(),
            command);
        return;
    }

    if (!StartMotion(command, false))
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "모션 %d 시작 실패",
            command);
        PublishMotionState(true, true);
    }
}


void MainNode::SchedulePostInitialPoseSequence()
{
    calibration_timer_ = this->create_wall_timer(
        5s,
        std::bind(
            &MainNode::RequestHurdleRecalibration,
            this));

    ready_timer_ = this->create_wall_timer(
        10s,
        std::bind(
            &MainNode::FinishInitialization,
            this));
}


void MainNode::RequestHurdleRecalibration()
{
    calibration_timer_->cancel();

    if (!hurdle_recalibrate_client_->service_is_ready())
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "/hurdle/recalibrate 서비스를 찾지 못했습니다. "
            "캘리브레이션을 시작할 수 없습니다.");
        return;
    }

    auto request =
        std::make_shared<std_srvs::srv::Trigger::Request>();

    hurdle_recalibrate_client_->async_send_request(
        request,
        [this](
            rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future)
        {
            const auto response = future.get();

            if (response->success)
            {
                hurdle_calibration_started_ = true;
                RCLCPP_INFO(
                    this->get_logger(),
                    "Hurdle 캘리브레이션 시작: "
                    "약 3초 동안 초기자세를 유지합니다.");
            }
            else
            {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Hurdle 캘리브레이션 요청 실패: %s",
                    response->message.c_str());
            }
        });
}


void MainNode::FinishInitialization()
{
    ready_timer_->cancel();

    if (!hurdle_calibration_started_)
    {
        PublishMotionState(false, false);
        RCLCPP_ERROR(
            this->get_logger(),
            "Hurdle 캘리브레이션 시작이 확인되지 않아 "
            "motion_ready=false를 유지합니다.");
        return;
    }

    initialization_ready_ = true;
    PublishMotionState(true, true);

    RCLCPP_INFO(
        this->get_logger(),
        "초기자세 후 10초 대기 완료: "
        "motion_ready=true, motion_end=true");
}


void MainNode::PublishMotionState(
    bool motion_end,
    bool motion_ready)
{
    msgs::msg::MotionEnd msg;
    msg.motion_end = motion_end;
    msg.motion_ready = motion_ready;
    motion_end_pub_->publish(msg);
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MainNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}