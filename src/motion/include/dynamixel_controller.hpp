#ifndef DYNAMIXEL_CONTROLLER_HPP
#define DYNAMIXEL_CONTROLLER_HPP

#include <Eigen/Dense>
#include "dynamixel.hpp" // Dxl 클래스 정의가 포함된 헤더

// 필요한 상수 정의 (준혁 님의 프로젝트 설정에 맞게 수정하세요)
#ifndef NUMBER_OF_DYNAMIXELS
#define NUMBER_OF_DYNAMIXELS 23 // G.O.A.T 로봇의 전체 모터 수
#endif

#ifndef Window_Size
#define Window_Size 10 // 이동평균필터(MAF)의 윈도우 크기
#endif

using namespace Eigen;

class Dxl_Controller
{
private:
    Dxl *dxlPtr; // 하드웨어 제어 객체 포인터

    // 내부 계산용 벡터 및 행렬
    VectorXd th_cont;              // 현재 각도 저장
    VectorXd th_dot_cont;          // 현재 각속도 저장
    VectorXd th_dot_MovAvgFilterd; // 필터링된 각속도
    
    MatrixXd MAF; // 이동평균필터를 위한 데이터 버퍼 (Window_Size x NUMBER_OF_DYNAMIXELS)

public:
    // 생성자
    Dxl_Controller(Dxl *dxlPtr);

    // ************************************ GETTERS ***************************************** //
    VectorXd GetJointTheta();    // 관절 각도 [rad] 반환
    VectorXd GetThetaDot();     // 관절 각속도 [rad/s] 반환
    VectorXd GetThetaDotMAF();  // 이동평균필터가 적용된 각속도 반환

    // **************************** SETTERS ******************************** //
    void SetTorque(VectorXd tau);     // 목표 토크 설정
    void SetPosition(VectorXd theta);  // 목표 위치(각도) 설정
};

#endif // DYNAMIXEL_CONTROLLER_HPP