// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from msgs:msg/HurdleResult.idl
// generated code does not contain a copyright notice

#ifndef MSGS__MSG__DETAIL__HURDLE_RESULT__BUILDER_HPP_
#define MSGS__MSG__DETAIL__HURDLE_RESULT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "msgs/msg/detail/hurdle_result__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace msgs
{

namespace msg
{

namespace builder
{

class Init_HurdleResult_angle
{
public:
  explicit Init_HurdleResult_angle(::msgs::msg::HurdleResult & msg)
  : msg_(msg)
  {}
  ::msgs::msg::HurdleResult angle(::msgs::msg::HurdleResult::_angle_type arg)
  {
    msg_.angle = std::move(arg);
    return std::move(msg_);
  }

private:
  ::msgs::msg::HurdleResult msg_;
};

class Init_HurdleResult_status
{
public:
  Init_HurdleResult_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_HurdleResult_angle status(::msgs::msg::HurdleResult::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_HurdleResult_angle(msg_);
  }

private:
  ::msgs::msg::HurdleResult msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::msgs::msg::HurdleResult>()
{
  return msgs::msg::builder::Init_HurdleResult_status();
}

}  // namespace msgs

#endif  // MSGS__MSG__DETAIL__HURDLE_RESULT__BUILDER_HPP_
