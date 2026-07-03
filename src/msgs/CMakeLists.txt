cmake_minimum_required(VERSION 3.8)
project(msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 필수 의존성 패키지들을 찾습니다.
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED) # <-- 이 줄이 빠져서 에러가 난 것입니다!
find_package(std_msgs REQUIRED)                # <-- 기본 메시지 타입을 위해 필요합니다.

# 2. 메시지 인터페이스 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LineResult.msg"
  "msg/BallResult.msg"
  "msg/HurdleResult.msg"
  "msg/MotionEnd.msg"
  "msg/MotionCommand.msg"
  DEPENDENCIES std_msgs                          # <-- 의존성 명시
)


# ------------------------------------------------------------------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()