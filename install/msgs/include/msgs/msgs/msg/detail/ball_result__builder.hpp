// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from msgs:msg/BallResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__BALL_RESULT__BUILDER_HPP_
#define MSGS__MSG__DETAIL__BALL_RESULT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "msgs/msg/detail/ball_result__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace msgs
{

namespace msg
{

namespace builder
{

class Init_BallResult_ball_in_hand
{
public:
  explicit Init_BallResult_ball_in_hand(::msgs::msg::BallResult & msg)
  : msg_(msg)
  {}
  ::msgs::msg::BallResult ball_in_hand(::msgs::msg::BallResult::_ball_in_hand_type arg)
  {
    msg_.ball_in_hand = std::move(arg);
    return std::move(msg_);
  }

private:
  ::msgs::msg::BallResult msg_;
};

class Init_BallResult_angle
{
public:
  explicit Init_BallResult_angle(::msgs::msg::BallResult & msg)
  : msg_(msg)
  {}
  Init_BallResult_ball_in_hand angle(::msgs::msg::BallResult::_angle_type arg)
  {
    msg_.angle = std::move(arg);
    return Init_BallResult_ball_in_hand(msg_);
  }

private:
  ::msgs::msg::BallResult msg_;
};

class Init_BallResult_status
{
public:
  Init_BallResult_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_BallResult_angle status(::msgs::msg::BallResult::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_BallResult_angle(msg_);
  }

private:
  ::msgs::msg::BallResult msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::msgs::msg::BallResult>()
{
  return msgs::msg::builder::Init_BallResult_status();
}

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__BALL_RESULT__BUILDER_HPP_
