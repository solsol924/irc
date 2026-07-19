cmake_minimum_required(VERSION 3.8)
project(motion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

add_executable(main
  src/main.cpp
  src/callback.cpp
  src/dynamixel.cpp
  src/sdk.cpp
)

target_include_directories(main PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(main PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
endif()

ament_target_dependencies(main
  rclcpp
  msgs
  std_srvs
  dynamixel_sdk
)

target_link_libraries(main
  Eigen3::Eigen
  Threads::Threads
)

install(TARGETS main
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
