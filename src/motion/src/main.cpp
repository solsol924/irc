#include "main.hpp"

#include <chrono>
#include <functional>
#include <memory>

using namespace std::chrono_literals;


// SDK 관절 수와 Dynamixel 개수가 다르면 컴파일 중 오류
static_assert(
    NUMBER_OF_JOINTS == NUMBER_OF_DYNAMIXELS,
    "SDK 관절 수와 Dynamixel 개수가 다릅니다."
);

// ==========================================================
// MainNode 생성자
// ==========================================================
MainNode::MainNode() 
: Node("motion_main_node")
{
    RCLCPP_INFO(
        this->get_logger(), "Motion MainNode 시작");
    
    // Callback 객체 생성
    callback_ = std::make_shared<Callback>();
    // Dxl 객체 생성
    dxl_port_ = std::make_shared<Dxl>(false);

    // motion_end와 motion_ready를 같이 보내는 publisher
    auto motion_state_qos = rclcpp::QoS(1).reliable().transient_local();
    motion_end_pub_ = this->create_publisher<msgs::msg::MotionEnd>(
        "motion_end", motion_state_qos);

    // command subscriber
    motion_command_sub_ = this->create_subscription<msgs::msg::MotionCommand>(
        "motion_command", 10, std::bind(
            &MainNode::MotionCallback,
            this,
            std::placeholders::_1));

    // 200Hz 제어루프 타이머 생성
    timer_ = this->create_wall_timer(
        5ms, std::bind(&MainNode::ControlLoop, this));
    // 초기에는 타이머를 멈춤, 초기 자세 완료 후 시작
    timer_->cancel(); 
    
    //초기자세 Publish 후 실행
    PublishMotionState(false, false);
    StartInitialPose();
}

void MainNode::StartInitialPose()
{
    //다이나믹셀 현재 괁절각 읽기
    RCLCPP_INFO(this->get_logger(), "다이나믹셀 현재 관절값 읽기");
    Eigen::VectorXd current_theta = dxl_port_->GetThetaAct();   

    //관절각 유효성 검사
    if (!IsValidTheta(current_theta)) {
        RCLCPP_ERROR(this->get_logger(), "다이나믹셀 현재 관절값이 유효하지 않습니다. 초기자세를 시작할 수 없습니다.");
        //초기자세를 시작하면 안되므로
        PublishMotionState(false, false);
        return;
    }

    //Callback 객체에 현재 관절각 설정
    callback_->SetCurrentTheta(current_theta);

    //초기자세까지 3초동안 이동하는 궤적 생성
    callback_->SelectMotion(0);

    if (!callback_->IsMoving()) {
        RCLCPP_ERROR(this->get_logger(), "초기자세 궤적 생성 실패");
        PublishMotionState(false, false);
        return;
    }
    current_motion_id_ = 0;
    //이동중이므로 아직 false
    PublishMotionState(false, false);
    //제어루프 타이머 시작
    timer_->reset(); 
    
    RCLCPP_INFO(this->get_logger(), "초기자세 이동 시작");
}

// ==========================================================
// Decision에서 모션 명령 수신
// ==========================================================
void MainNode::MotionCallback(
    const msgs::msg::MotionCommand::SharedPtr msg)
{
    //MotionCommand 메시지에서 command 번호를 꺼내기
    const int command = msg->command;

    RCLCPP_INFO(this->get_logger(), "MotionCommand 수신: %d", command);

    //초기 자세 완료 여부 확인
    if (!initial_pose_done_) {
        RCLCPP_WARN(this->get_logger(), "초기자세 완료 전, 모션 명령 무시");
        return;
    }

    //다른 모션이 실행중인지 확인
    if (callback_->IsMoving()) {
        RCLCPP_WARN(this->get_logger(), "다른 모션이 실행중, 모션 명령 무시");
        return;
    }

    //모션 시작 전 현재 관절각 읽기
    Eigen::VectorXd current_theta = dxl_port_->GetThetaAct();

    if (!IsValidTheta(current_theta)) {
        RCLCPP_ERROR(this->get_logger(), "다이나믹셀 현재 관절값이 유효하지 않습니다. 모션을 시작할 수 없습니다.");
        PublishMotionState(false, false);
        return;
    }

    //현재 자세를 새 궤적의 시작점으로 설정
    callback_->SetCurrentTheta(current_theta);

    //command -> selectmotion
    switch (command)
    {
        case 0:
            callback_->SelectMotion(0);
            break;

        case 1:
            callback_->SelectMotion(1);
            break;

        case 2: 
            callback_->SelectMotion(2);
            break;

        case 3:
            callback_->SelectMotion(3);
            break;

        case 4:
            callback_->SelectMotion(4);
            break;

        case 5:
            callback_->SelectMotion(5);
            break;

        case 6:
            callback_->SelectMotion(6);
            break;

        case 7:
            callback_->SelectMotion(7);
            break;

        case 8:
            callback_->SelectMotion(8);
            break;

        case 9:
            callback_->SelectMotion(9);
            break;

        case 10:
            callback_->SelectMotion(10);
            break;

        case 11:
            callback_->SelectMotion(11);
            break;

        case 12:
            callback_->SelectMotion(12);
            break;

        case 13:
            callback_->SelectMotion(13);
            break;

        case 14:
            callback_->SelectMotion(14);
            break;

        case 15:
            callback_->SelectMotion(15);
            break;

        case 16:
            callback_->SelectMotion(16);
            break;

        case 17:
            callback_->SelectMotion(17);
            break;

        case 18:
            callback_->SelectMotion(18);
            break;
        
        case 19:
            callback_->SelectMotion(19);
            break;

        case 20:
            callback_->SelectMotion(20);
            break;
        
        case 21:
            callback_->SelectMotion(21);
            break;

        case 22:
            callback_->SelectMotion(22);
            break;

        case 23:
            callback_->SelectMotion(23);
            break;

        case 24:
            callback_->SelectMotion(24);
            break;

        default:
            RCLCPP_WARN(this->get_logger(), "알 수 없는 모션 명령: %d", command);
            return;
    }
    current_motion_id_ = command;

    //motion 실행 중 상태 발행
    PublishMotionState(false, true);
    //제어루프 타이머 시작
    timer_->reset();

    RCLCPP_INFO(this->get_logger(), "모션 %d 실행 시작", current_motion_id_);

}

// ==========================================================
// 200Hz 제어 루프
// ==========================================================
void MainNode::ControlLoop()
{
    //실행중인 궤적이 없으면 타이머 중지
    if (!callback_->IsMoving()) {
        timer_->cancel();
        return;
    }

    //다음 5ms tick 목표각 계산
    // callback의 배열을 Eigen::VectorXd로 변환
    callback_->Write_All_Theta();
    Eigen::VectorXd target_theta =
        Eigen::Map<Eigen::VectorXd>(
            callback_->All_Theta,
             NUMBER_OF_JOINTS);

    //목표각 검사 
    // 잘못된 목표각은 중지, 리셋
    if (!IsValidTheta(target_theta))
    {
        RCLCPP_ERROR(this->get_logger(), "다음 tick 목표각이 유효하지 않습니다. 모션을 중지합니다.");
        timer_->cancel();
        current_motion_id_ = -1;
        PublishMotionState(false, false);
        return;
    }
    
    //목표 관절각 Dxl 객체에 저장
    dxl_port_->SetThetaRef(target_theta);

    //다이나믹셀에 목표각 전송
    dxl_port_->syncWriteTheta();

    // 모션이 끝났는지 확인
    if (callback_->IsMoving())
    {
        return;
    }
    //모션 완료 후 타이머 중지
    timer_->cancel();

    /////초기자세 완료 처리
    if (!initial_pose_done_ && current_motion_id_ == 0) {
        initial_pose_done_ = true;
        current_motion_id_ = -1;
        //초기자세 완료. motion ready
        PublishMotionState(true, true);
        RCLCPP_INFO(this->get_logger(), "초기자세 완료");
        return;
    }

    //일반 모션 완료 처리
    RCLCPP_INFO(this->get_logger(), "모션 %d 완료", current_motion_id_);
    current_motion_id_ = -1;
    //모션 완료. motion end
    PublishMotionState(true, true);

}


// ==========================================================
// 관절각 데이터 검사
// ==========================================================
bool MainNode::IsValidTheta(
    const Eigen::VectorXd& theta) const
{
    // 관절 개수가 SDK 설정과 같은지 검사
    if (theta.size() != NUMBER_OF_JOINTS)
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "관절각 개수 오류: 현재=%ld, 필요=%d",
            static_cast<long>(theta.size()),
            NUMBER_OF_JOINTS
        );

        return false;
    }


    // NaN 또는 Inf가 포함되어 있는지 검사
    if (!theta.allFinite())
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "관절각에 NaN 또는 Inf가 포함되어 있습니다."
        );

        return false;
    }


    return true;
}


void MainNode::PublishMotionState(bool motion_end, bool motion_ready)
{
    msgs::msg::MotionEnd msg;
    msg.motion_end = motion_end;
    msg.motion_ready = motion_ready;
    motion_end_pub_->publish(msg);
}


int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MainNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
