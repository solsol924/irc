#include "dynamixel_controller.hpp"

#include <limits>

// dynamixel_controller.cpp 파일을 사용하는 이유는 Dxl_Controller 클래스의 메서드 구현을 이 파일에 작성하기 위해서입니다.
// Dxl_Controller 클래스는 Dxl 클래스의 객체를 사용하여 다이나믹셀 모터의 상태를 읽고 제어하는 역할을 합니다. 
// 따라서 Dxl_Controller 클래스의 메서드 구현에서는 Dxl 클래스의 메서드를 호출하여 관절의 각도와 속도를 읽거나 목표 각도를 설정하는 등의 작업을 수행합니다.
// Dxl_Controller 클래스의 메서드 구현을 dynamixel_controller.cpp 파일에 작성함으로써 코드의 구조를 명확하게 하고, 유지보수성을 높일 수 있습니다.

Dxl_Controller::Dxl_Controller(Dxl *dxlPtr) : dxlPtr(dxlPtr)
{

}
// Dxl_Controller 클래스의 생성자에서는 Dxl 클래스의 객체 포인터를 받아서 멤버 변수로 저장합니다.
// 이 포인터를 통해 Dxl_Controller 클래스는 Dxl 클래스의 메서드에 접근하여 다이나믹셀 모터의 상태를 읽거나 제어할 수 있습니다.
// dslPtr는 Dxl 클래스의 객체를 가리키는 포인터입니다.
// Dxl 클래스는 다이나믹셀 모터와의 통신을 담당하는 클래스입니다.
// Dxl_Controller 클래스는 Dxl 클래스의 메서드를 호출하여 관절의 각도와 속도를 읽고, 목표 각도를 설정하는 등의 작업을 수행합니다.


// ************************************ GETTERS ***************************************** //

// Getter() : 관절각도[rad]
// dynamixel 클래스에 있는 GetThetaAct() 함수를 호출하여 관절 각도를 읽어와서 th_cont 벡터에 저장한 후 반환하는 형태입니다.
// GetThetaAct()를 사용하지않고 GetJointTheta()에서 직접 관절 각도를 읽어오는 방식으로 구현할 수도 있지만
// 이렇게 하면 코드의 재사용성이 떨어지고 유지보수가 어려워질 수 있습니다.

VectorXd Dxl_Controller::GetJointTheta() // 관절 각도 [rad] 반환
{
    VectorXd th_(NUMBER_OF_DYNAMIXELS);
    if (!dxlPtr->GetThetaAct(th_)) {
        th_.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
    for (uint8_t i=0; i<NUMBER_OF_DYNAMIXELS; i++)
    {
        th_cont[i] = th_[i]; // th_에 저장되어있던 각도 데이터를 th_cont 벡터에 복사합니다.
    }
    return th_cont;
}


//Getter() : 관절각속도[rad/s]
VectorXd Dxl_Controller::GetThetaDot()
{
    VectorXd th_dot_(NUMBER_OF_DYNAMIXELS);
    th_dot_ = dxlPtr->GetThetaDot();
    for(uint8_t i=0; i<NUMBER_OF_DYNAMIXELS; i++)    
    {
        th_dot_cont[i] = th_dot_[i];
    }
    return th_dot_cont;
}

//Getter() : 각도의 차이와 이동평균필터를 이용해 각속도 계산 
VectorXd Dxl_Controller::GetThetaDotMAF()
{
    VectorXd a_th_dot(NUMBER_OF_DYNAMIXELS);
    a_th_dot = dxlPtr->GetThetaDotEstimated();
    for (uint8_t i=0; i<NUMBER_OF_DYNAMIXELS; i++)
    {
        th_dot_cont[i] = a_th_dot[i];
    }
    MAF << MAF.block<Window_Size-1, NUMBER_OF_DYNAMIXELS>(1, 0), th_dot_cont[0], th_dot_cont[1], th_dot_cont[2], th_dot_cont[3], th_dot_cont[4], th_dot_cont[5], th_dot_cont[6];
    th_dot_MovAvgFilterd = MAF.colwise().mean();

    return th_dot_MovAvgFilterd;
}

// **************************** SETTERS ******************************** //


//Setter() : 목표 theta값 설정[rad]
void Dxl_Controller::SetPosition(VectorXd theta)
{
    for (uint8_t i=0; i<NUMBER_OF_DYNAMIXELS; i++)
    {
        th_cont[i] = theta[i];
    }
    dxlPtr->SetThetaRef(th_cont);
}
