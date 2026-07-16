#include "callback.hpp"

Callback::Callback() : Node("callback_node")
// 생성자 이름은 클래스 이름과 동일해야 하며, 콜백 노드의 이름을 "callback_node"로 설정합니다.
{
    RCLCPP_INFO(this->get_logger(), "GOAT callback_node has been started.");
}
// main.cpp에서 아래와 같이 사용됩니다.
// rclcpp::spin(std::make_shared<Callback>());  콜백 노드 실행


void Callback::SetCurrentTheta(const Eigen::VectorXd& theta)
{
// 처음 시작할 때 현재 관절각을 설정하는 함수입니다. theta는 Eigen::VectorXd 타입으로, 각 관절의 현재 각도를 나타냅니다.
    for (int i = 0; i < NUMBER_OF_JOINTS; ++i) {
        All_Theta[i] = theta[i];
    }
}
// MAIN.cpp에서 아래와 같이 사용됩니다.
// VectorXd validated_theta = dxl_port.GetThetaAct();  다이나믹셀에서 현재각 읽기
// callback.SetCurrentTheta(validated_theta);  All_Theta에 현재각 넣기
// callback.SelectMotion(0);  현재각 -> 기본자세 SDK에 설정된 시간 동안 이동


void Callback::SelectMotion(int go)
// 모션 선택 및 실행 함수입니다.
// go는 실행할 모션 ID를 나타냅니다.
{
    if (sdk_motion.Is_Moving())
    // Is_Moving(): 현재 모션 진행 여부
    {
        RCLCPP_WARN(this->get_logger(), "[%d번] 명령 무시: 현재 모션 진행 중", go);
        return;
    }

    bool success = sdk_motion.Generate_Trajectory(go, All_Theta);
    // success: Generate_Trajectory() 함수의 반환값, 모션 궤적 생성 여부
    // false를 반환하게 되면 그 시점의 마지막 목표각을 유지하고, 새로운 모션 명령은 무시됩니다.
    if (success)
    {
        RCLCPP_INFO(this->get_logger(), "[%d번] 모션 실행", go);
    }
    else {
        RCLCPP_ERROR(this->get_logger(), "[%d번] 모션을 찾을 수 없습니다", go);
    }
}

void Callback::Write_All_Theta()
{
    sdk_motion.Get_Next_Tick(All_Theta); // Get_Next_Tick(): 다음 tick 목표각 계산
}

bool Callback::IsMoving()
{
    return sdk_motion.Is_Moving(); // 현재 모션 진행 여부 반환
}


// main.cpp에서 아래와 같이 사용 (예시)

// callback.SelectMotion(1);  모션 ID 1 실행, 현재각 -> 첫 포즈 SDK에 설정된 시간 동안 이동
// 궤적이 생성되어 active_trajectory_에 저장됨
// 1번 모션이 끝나기 전까지는 새로운 모션 명령이 들어와도 무시됨

// 반복 루프 200Hz
// callback.Write_All_Theta();  All_Theta에 다음 tick 목표각 계산
// dxl_port.SetThetaDes(callback.All_Theta);  All_Theta를 다이나믹셀 목표각 배열에 복사
// dxl_port.WriteThetaDes();  다이나믹셀에 목표각 전송
//
// ...
//
// if (!callback.IsMoving()) {callback.SelectMotion(2);}
// 모션 ID 1이 끝나면 모션 ID 2 실행, 현재각 -> 첫 포즈 SDK에 설정된 시간 동안 이동



// 동작 중 명령 무시 기능은 sdk에도 처리가 가능하지만 IsMoving 함수를 이용해 main.cpp에서 처리하는 것이 더 깔끔해보임

// Callback 명령 받을지 말지 판단
// SDK_Motion 받은 명령으로 궤적 계산
// Main 실제 모터 읽기/쓰기와 타이머 실행


// 목 부분 따로 제어 하는 방향성
//
// void Callback::Write_All_Theta() {
//     sdk_motion.Get_Next_Tick(All_Theta); // 1~21번 또는 몸 모션 채움
//
//     extra_joint.Update(All_Theta);       // 22, 23번만 따로 채움
// }
//
// 이런 느낌으로 별도 페이지 하나 더 만들고, callback에 Write_All_Theta()에서 합치기
//
// 1. Get_Next_Tick()으로 기본 모션 값 채움
// 2. extra 함수가 필요한 관절만 덮어씀
// 3. main.cpp가 All_Theta 전체를 모터로 보냄