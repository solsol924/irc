#pragma once
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include "sdk.hpp"

class Callback : public rclcpp::Node
// Callback 클래스는 ROS 2 노드로서, 모션 제어와 관련된 콜백 기능을 제공합니다.
{
private:
    SDK_Motion sdk_motion; // SDK_Motion를 사용하여 모션 라이브러리와 궤적 생성 기능을 제공합니다.

public:
    Callback(); // 생성자
    
    double All_Theta[NUMBER_OF_JOINTS]; 

    void SetCurrentTheta(const Eigen::VectorXd& theta);
    void SelectMotion(int go);
    void Write_All_Theta();
    bool IsMoving();
};